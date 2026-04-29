use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::time::sleep;

use crate::rag::{VectorDbConfig, load_vector_db_config};
use crate::tools::ollama::{OllamaConfig, ollama_embed};
use crate::tools::qdrant::{QdrantVectorRecord, qdrant_delete_vectors, qdrant_upsert_vectors};

const SYSTEM_MANIFEST_PATH: &str = "/etc/agentos/configs/rag_index_manifest.json";
const DEFAULT_COLLECTION: &str = "repos";
const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text:latest";
const MAX_FILE_BYTES: u64 = 2_000_000;

#[derive(Debug, Clone)]
pub struct Orchestrator {
    collection: String,
    embedding_model: String,
    batch_size: usize,
    chunk_tokens: usize,
    chunk_overlap: f32,
    manifest_path: PathBuf,
    vector_db: VectorDbConfig,
    ollama: OllamaConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationSummary {
    pub status: String,
    pub processed_chunks: usize,
    pub upserted_ids: Vec<String>,
    pub deleted_ids: Vec<String>,
    pub errors: Vec<String>,
}

impl OperationSummary {
    fn ok() -> Self {
        Self {
            status: "ok".to_string(),
            processed_chunks: 0,
            upserted_ids: Vec::new(),
            deleted_ids: Vec::new(),
            errors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileIngestRequest {
    pub path: String,
    pub content: String,
    pub commit: String,
    pub source_type: String,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffIngestRequest {
    pub repo: String,
    pub commit: String,
    pub changed_files: Vec<String>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebuildRequest {
    pub scope: String,
    pub collection: String,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupRequest {
    pub collection: String,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorUpsertRequest {
    pub collection: String,
    pub vectors: Vec<VectorRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDeleteRequest {
    pub collection: String,
    pub ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub embedding: Vec<f32>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexedChunk {
    id: String,
    source: String,
    path: String,
    chunk_index: usize,
    checksum: String,
    embedding_model: String,
    commit: String,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct IndexManifest {
    chunks: BTreeMap<String, IndexedChunk>,
}

#[derive(Debug, Clone)]
struct PreparedChunk {
    id: String,
    text: String,
    metadata: ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkMetadata {
    id: String,
    source: String,
    commit: String,
    lang: String,
    chunk_index: usize,
    created_at: DateTime<Utc>,
    embedding_model: String,
    checksum: String,
    tags: String,
    repo: String,
    path: String,
    text: String,
}

impl Orchestrator {
    pub fn from_env() -> Result<Self> {
        let mut vector_db = load_vector_db_config().unwrap_or_else(|_| VectorDbConfig {
            provider: "qdrant".to_string(),
            host: "http://127.0.0.1:6333".to_string(),
            collection: DEFAULT_COLLECTION.to_string(),
            embedding_dim: 768,
            distance: "Cosine".to_string(),
        });
        let collection = env_string("RAG_COLLECTION")
            .or_else(|| env_string("COLLECTION_NAME"))
            .unwrap_or_else(|| vector_db.collection.clone());
        vector_db.collection = collection.clone();

        let embedding_model =
            env_string("EMBEDDING_MODEL").unwrap_or_else(|| DEFAULT_EMBEDDING_MODEL.to_string());
        let ollama = OllamaConfig {
            endpoint: env_string("OLLAMA_ENDPOINT")
                .unwrap_or_else(|| "http://127.0.0.1:11434".to_string()),
            embedding_model: embedding_model.clone(),
        };

        Ok(Self {
            collection,
            embedding_model,
            batch_size: env_usize("BATCH_SIZE", 128).max(1),
            chunk_tokens: env_usize("CHUNK_TOKENS", 800).max(1),
            chunk_overlap: env_f32("CHUNK_OVERLAP", 0.25).clamp(0.0, 0.95),
            manifest_path: manifest_path_from_env(),
            vector_db,
            ollama,
        })
    }

    pub async fn ingest_file(&self, request: FileIngestRequest) -> Result<OperationSummary> {
        let started = Instant::now();
        emit_event("ingest_started", 0, 0, 0, None, started);
        let (repo, relative_path) = split_source_path(&request.path);
        let chunks = self.prepare_chunks(&repo, &relative_path, &request.content, &request.commit);
        let summary = if chunks.is_empty() {
            self.delete_source(&format!("{repo}:{relative_path}"), request.dry_run)
                .await?
        } else {
            self.upsert_prepared(chunks, request.dry_run).await?
        };
        emit_event(
            "ingest_completed",
            summary.processed_chunks,
            summary.upserted_ids.len(),
            summary.deleted_ids.len(),
            None,
            started,
        );
        Ok(summary)
    }

    pub async fn ingest_diff(&self, request: DiffIngestRequest) -> Result<OperationSummary> {
        let started = Instant::now();
        emit_event("ingest_started", 0, 0, 0, None, started);
        let repo_root = resolve_repo_root(&request.repo)?;
        let repo_name = repo_root
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| request.repo.clone());
        let mut combined = OperationSummary::ok();

        let mut manifest = self.load_manifest().await?;
        let mut manifest_changed = false;

        for changed in request.changed_files {
            let relative = match safe_relative_path(&changed) {
                Ok(relative) => relative,
                Err(error) => {
                    combined.errors.push(error.to_string());
                    continue;
                }
            };
            let path = repo_root.join(&relative);
            if path.exists() {
                let canonical = path
                    .canonicalize()
                    .with_context(|| format!("failed to canonicalize {}", path.display()))?;
                if !canonical.starts_with(&repo_root) {
                    combined
                        .errors
                        .push(format!("skipped path outside repo root: {changed}"));
                    continue;
                }
                if is_binary_or_large(&canonical).await? {
                    combined
                        .errors
                        .push(format!("skipped binary or large file: {relative}"));
                    continue;
                }
                let content = tokio::fs::read_to_string(&canonical)
                    .await
                    .with_context(|| format!("failed to read {}", canonical.display()))?;
                let chunks = self.prepare_chunks(&repo_name, &relative, &content, &request.commit);
                let partial = self
                    .upsert_prepared_with_manifest(&mut manifest, chunks, request.dry_run)
                    .await?;
                manifest_changed |=
                    !partial.upserted_ids.is_empty() || !partial.deleted_ids.is_empty();
                merge_summary(&mut combined, partial);
            } else {
                let deleted = self
                    .delete_source_with_manifest(
                        &mut manifest,
                        &format!("{repo_name}:{relative}"),
                        request.dry_run,
                    )
                    .await?;
                manifest_changed |= !deleted.deleted_ids.is_empty();
                merge_summary(&mut combined, deleted);
            }
        }

        if manifest_changed && !request.dry_run {
            self.save_manifest(&manifest).await?;
        }

        combined.status = if combined.errors.is_empty() {
            "ok".to_string()
        } else {
            "partial".to_string()
        };
        emit_event(
            "ingest_completed",
            combined.processed_chunks,
            combined.upserted_ids.len(),
            combined.deleted_ids.len(),
            None,
            started,
        );
        Ok(combined)
    }

    pub async fn rebuild(&self, request: RebuildRequest) -> Result<OperationSummary> {
        let started = Instant::now();
        emit_event("rebuild_started", 0, 0, 0, None, started);
        self.ensure_collection(&request.collection)?;
        if request.scope != "full" {
            bail!("only full rebuild is implemented in this pass");
        }

        let roots = repo_roots_from_env();
        let mut combined = OperationSummary::ok();
        let mut manifest = self.load_manifest().await?;
        let mut manifest_changed = false;
        for root in roots {
            if !root.is_dir() {
                combined
                    .errors
                    .push(format!("skipped missing repo root: {}", root.display()));
                continue;
            }
            let repo_name = root
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| "repo".to_string());
            for file in collect_indexable_files(&root)? {
                let relative = file
                    .strip_prefix(&root)?
                    .to_string_lossy()
                    .replace('\\', "/");
                if is_binary_or_large(&file).await? {
                    combined
                        .errors
                        .push(format!("skipped binary or large file: {relative}"));
                    continue;
                }
                let content = tokio::fs::read_to_string(&file)
                    .await
                    .with_context(|| format!("failed to read {}", file.display()))?;
                let chunks = self.prepare_chunks(&repo_name, &relative, &content, "rebuild");
                let partial = self
                    .upsert_prepared_with_manifest(&mut manifest, chunks, request.dry_run)
                    .await?;
                manifest_changed |=
                    !partial.upserted_ids.is_empty() || !partial.deleted_ids.is_empty();
                merge_summary(&mut combined, partial);
            }
        }

        if manifest_changed && !request.dry_run {
            self.save_manifest(&manifest).await?;
        }

        combined.status = if combined.errors.is_empty() {
            "ok".to_string()
        } else {
            "partial".to_string()
        };
        emit_event(
            "rebuild_completed",
            combined.processed_chunks,
            combined.upserted_ids.len(),
            combined.deleted_ids.len(),
            None,
            started,
        );
        Ok(combined)
    }

    pub async fn cleanup(&self, request: CleanupRequest) -> Result<OperationSummary> {
        self.ensure_collection(&request.collection)?;
        if !matches!(request.mode.as_str(), "stale" | "missing" | "all") {
            bail!("unsupported cleanup mode: {}", request.mode);
        }
        if matches!(request.mode.as_str(), "stale" | "missing") {
            bail!("cleanup mode '{}' is not implemented yet", request.mode);
        }
        let manifest = self.load_manifest().await?;
        let ids = manifest.chunks.keys().cloned().collect::<Vec<_>>();
        if request.mode == "all" {
            let summary = self
                .vectors_delete(VectorDeleteRequest {
                    collection: self.collection.clone(),
                    ids,
                })
                .await?;
            self.save_manifest(&IndexManifest::default()).await?;
            return Ok(summary);
        }
        Ok(OperationSummary::ok())
    }

    pub async fn vectors_upsert(&self, request: VectorUpsertRequest) -> Result<OperationSummary> {
        self.ensure_collection(&request.collection)?;
        let started = Instant::now();
        let records = request
            .vectors
            .iter()
            .map(|vector| QdrantVectorRecord {
                id: &vector.id,
                embedding: &vector.embedding,
                metadata: &vector.metadata,
            })
            .collect::<Vec<_>>();
        retry_async(|| qdrant_upsert_vectors(&self.vector_db, &records)).await?;
        let ids = request
            .vectors
            .into_iter()
            .map(|vector| vector.id)
            .collect::<Vec<_>>();
        emit_event("upsert_success", ids.len(), ids.len(), 0, None, started);
        Ok(OperationSummary {
            status: "ok".to_string(),
            processed_chunks: ids.len(),
            upserted_ids: ids,
            deleted_ids: Vec::new(),
            errors: Vec::new(),
        })
    }

    pub async fn vectors_delete(&self, request: VectorDeleteRequest) -> Result<OperationSummary> {
        self.ensure_collection(&request.collection)?;
        let started = Instant::now();
        retry_async(|| qdrant_delete_vectors(&self.vector_db, &request.ids)).await?;
        emit_event(
            "delete_success",
            request.ids.len(),
            0,
            request.ids.len(),
            None,
            started,
        );
        Ok(OperationSummary {
            status: "ok".to_string(),
            processed_chunks: 0,
            upserted_ids: Vec::new(),
            deleted_ids: request.ids,
            errors: Vec::new(),
        })
    }

    fn ensure_collection(&self, collection: &str) -> Result<()> {
        if collection != self.collection {
            bail!(
                "collection mismatch: request={} configured={}",
                collection,
                self.collection
            );
        }
        Ok(())
    }

    fn prepare_chunks(
        &self,
        repo: &str,
        relative_path: &str,
        content: &str,
        commit: &str,
    ) -> Vec<PreparedChunk> {
        let lang = detect_lang(relative_path);
        chunk_content(
            relative_path,
            content,
            self.chunk_tokens,
            self.chunk_overlap,
        )
        .into_iter()
        .enumerate()
        .map(|(chunk_index, text)| {
            let source_path = format!("{repo}:{relative_path}");
            let id = chunk_id(&source_path, chunk_index);
            let checksum = sha256_hex(text.as_bytes());
            let metadata = ChunkMetadata {
                id: id.clone(),
                source: source_path.clone(),
                commit: commit.to_string(),
                lang: lang.to_string(),
                chunk_index,
                created_at: Utc::now(),
                embedding_model: self.embedding_model.clone(),
                checksum,
                tags: format!("repo:{repo};lang:{lang}"),
                repo: repo.to_string(),
                path: relative_path.to_string(),
                text: text.clone(),
            };
            PreparedChunk { id, text, metadata }
        })
        .collect()
    }

    async fn upsert_prepared(
        &self,
        chunks: Vec<PreparedChunk>,
        dry_run: bool,
    ) -> Result<OperationSummary> {
        let mut manifest = self.load_manifest().await?;
        let summary = self
            .upsert_prepared_with_manifest(&mut manifest, chunks, dry_run)
            .await?;
        if !dry_run && (!summary.upserted_ids.is_empty() || !summary.deleted_ids.is_empty()) {
            self.save_manifest(&manifest).await?;
        }
        Ok(summary)
    }

    async fn upsert_prepared_with_manifest(
        &self,
        manifest: &mut IndexManifest,
        chunks: Vec<PreparedChunk>,
        dry_run: bool,
    ) -> Result<OperationSummary> {
        let mut summary = OperationSummary::ok();
        summary.processed_chunks = chunks.len();
        let source = chunks
            .first()
            .map(|chunk| chunk.metadata.source.clone())
            .unwrap_or_default();
        let new_ids = chunks
            .iter()
            .map(|chunk| chunk.id.clone())
            .collect::<BTreeSet<_>>();
        let mut to_embed = Vec::new();

        for chunk in chunks {
            let changed = manifest
                .chunks
                .get(&chunk.id)
                .map(|old| {
                    old.checksum != chunk.metadata.checksum
                        || old.embedding_model != self.embedding_model
                })
                .unwrap_or(true);
            if changed {
                to_embed.push(chunk);
            }
        }

        let removed = manifest
            .chunks
            .values()
            .filter(|chunk| chunk.source == source && !new_ids.contains(&chunk.id))
            .map(|chunk| chunk.id.clone())
            .collect::<Vec<_>>();

        if dry_run {
            summary.upserted_ids = to_embed.iter().map(|chunk| chunk.id.clone()).collect();
            summary.deleted_ids = removed;
            return Ok(summary);
        }

        for batch in to_embed.chunks(self.batch_size) {
            let started = Instant::now();
            let mut records = Vec::new();
            let mut indexed = Vec::new();
            for chunk in batch {
                match retry_async(|| ollama_embed(&chunk.text, &self.ollama)).await {
                    Ok(embedding) => {
                        records.push(VectorRecord {
                            id: chunk.id.clone(),
                            embedding,
                            metadata: serde_json::to_value(&chunk.metadata)?,
                        });
                        indexed.push(IndexedChunk {
                            id: chunk.id.clone(),
                            source: chunk.metadata.source.clone(),
                            path: chunk.metadata.path.clone(),
                            chunk_index: chunk.metadata.chunk_index,
                            checksum: chunk.metadata.checksum.clone(),
                            embedding_model: chunk.metadata.embedding_model.clone(),
                            commit: chunk.metadata.commit.clone(),
                            updated_at: Utc::now(),
                        });
                    }
                    Err(error) => {
                        emit_event(
                            "embed_batch_fail",
                            batch.len(),
                            0,
                            0,
                            Some(error.to_string()),
                            started,
                        );
                        summary.errors.push(error.to_string());
                    }
                }
            }
            if !records.is_empty() {
                let qdrant_records = records
                    .iter()
                    .map(|record| QdrantVectorRecord {
                        id: &record.id,
                        embedding: &record.embedding,
                        metadata: &record.metadata,
                    })
                    .collect::<Vec<_>>();
                retry_async(|| qdrant_upsert_vectors(&self.vector_db, &qdrant_records)).await?;
                emit_event(
                    "embed_batch_success",
                    records.len(),
                    records.len(),
                    0,
                    None,
                    started,
                );
                emit_event(
                    "upsert_success",
                    records.len(),
                    records.len(),
                    0,
                    None,
                    started,
                );
                for chunk in indexed {
                    summary.upserted_ids.push(chunk.id.clone());
                    manifest.chunks.insert(chunk.id.clone(), chunk);
                }
            }
        }

        if !removed.is_empty() {
            retry_async(|| qdrant_delete_vectors(&self.vector_db, &removed)).await?;
            for id in &removed {
                manifest.chunks.remove(id);
            }
            summary.deleted_ids.extend(removed);
        }

        if !summary.errors.is_empty() {
            summary.status = "partial".to_string();
        }
        Ok(summary)
    }

    async fn delete_source(&self, source: &str, dry_run: bool) -> Result<OperationSummary> {
        let mut manifest = self.load_manifest().await?;
        let summary = self
            .delete_source_with_manifest(&mut manifest, source, dry_run)
            .await?;
        if !dry_run && !summary.deleted_ids.is_empty() {
            self.save_manifest(&manifest).await?;
        }
        Ok(summary)
    }

    async fn delete_source_with_manifest(
        &self,
        manifest: &mut IndexManifest,
        source: &str,
        dry_run: bool,
    ) -> Result<OperationSummary> {
        let ids = manifest
            .chunks
            .values()
            .filter(|chunk| chunk.source == source)
            .map(|chunk| chunk.id.clone())
            .collect::<Vec<_>>();
        if !dry_run {
            retry_async(|| qdrant_delete_vectors(&self.vector_db, &ids)).await?;
            for id in &ids {
                manifest.chunks.remove(id);
            }
        }
        Ok(OperationSummary {
            status: "ok".to_string(),
            processed_chunks: 0,
            upserted_ids: Vec::new(),
            deleted_ids: ids,
            errors: Vec::new(),
        })
    }

    async fn load_manifest(&self) -> Result<IndexManifest> {
        if tokio::fs::metadata(&self.manifest_path).await.is_err() {
            return Ok(IndexManifest::default());
        }
        let raw = tokio::fs::read_to_string(&self.manifest_path)
            .await
            .with_context(|| format!("failed to read {}", self.manifest_path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse {}", self.manifest_path.display()))
    }

    async fn save_manifest(&self, manifest: &IndexManifest) -> Result<()> {
        if let Some(parent) = self.manifest_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let raw = serde_json::to_string_pretty(manifest)?;
        tokio::fs::write(&self.manifest_path, raw)
            .await
            .with_context(|| format!("failed to write {}", self.manifest_path.display()))
    }
}

fn merge_summary(target: &mut OperationSummary, other: OperationSummary) {
    target.processed_chunks += other.processed_chunks;
    target.upserted_ids.extend(other.upserted_ids);
    target.deleted_ids.extend(other.deleted_ids);
    target.errors.extend(other.errors);
}

async fn retry_async<F, Fut, T>(mut op: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut delay = Duration::from_millis(200);
    let mut last_error = None;
    for _ in 0..3 {
        match op().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                last_error = Some(error);
                sleep(delay).await;
                delay *= 2;
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("operation failed without error")))
}

fn emit_event(
    event: &'static str,
    processed_chunks: usize,
    upserted: usize,
    deleted: usize,
    error: Option<String>,
    started: Instant,
) {
    tracing::info!(
        event,
        processed_chunks,
        upserted,
        deleted,
        latency_ms = started.elapsed().as_millis() as u64,
        error = error.unwrap_or_default(),
        "rag_orchestrator_event"
    );
}

fn chunk_content(path: &str, content: &str, chunk_tokens: usize, overlap: f32) -> Vec<String> {
    if path.ends_with(".md") || path.ends_with(".markdown") {
        return chunk_markdown(content, chunk_tokens, overlap);
    }
    if is_code_path(path) {
        return chunk_code(content, chunk_tokens);
    }
    chunk_tokens_window(content, chunk_tokens, overlap)
}

fn chunk_markdown(content: &str, chunk_tokens: usize, overlap: f32) -> Vec<String> {
    let mut sections = Vec::new();
    let mut current = String::new();
    for line in content.lines() {
        if line.starts_with('#') && !current.trim().is_empty() {
            sections.extend(chunk_tokens_window(&current, chunk_tokens, overlap));
            current.clear();
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        sections.extend(chunk_tokens_window(&current, chunk_tokens, overlap));
    }
    sections
}

fn chunk_code(content: &str, chunk_tokens: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in content.lines() {
        let starts_item = is_code_boundary(line);
        if starts_item && !current.trim().is_empty() && approx_tokens(&current) >= chunk_tokens / 2
        {
            chunks.extend(chunk_tokens_window(&current, chunk_tokens, 0.0));
            current.clear();
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        chunks.extend(chunk_tokens_window(&current, chunk_tokens, 0.0));
    }
    chunks
}

fn chunk_tokens_window(content: &str, chunk_tokens: usize, overlap: f32) -> Vec<String> {
    let chunk_tokens = effective_chunk_tokens(chunk_tokens);
    let words = content.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return Vec::new();
    }
    let overlap_words = ((chunk_tokens as f32) * overlap).round() as usize;
    let step = chunk_tokens.saturating_sub(overlap_words).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < words.len() {
        let end = (start + chunk_tokens).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start += step;
    }
    chunks
}

fn is_code_boundary(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("fn ")
        || trimmed.starts_with("pub fn ")
        || trimmed.starts_with("async fn ")
        || trimmed.starts_with("pub async fn ")
        || trimmed.starts_with("class ")
        || trimmed.starts_with("def ")
        || trimmed.starts_with("function ")
        || trimmed.starts_with("export function ")
}

fn is_code_path(path: &str) -> bool {
    matches!(
        Path::new(path).extension().and_then(|ext| ext.to_str()),
        Some("rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "go" | "java" | "c" | "cc" | "cpp" | "h")
    )
}

fn approx_tokens(content: &str) -> usize {
    let words = content.split_whitespace().count();
    let punctuation = content
        .bytes()
        .filter(|byte| {
            matches!(
                byte,
                b'{' | b'}'
                    | b'('
                    | b')'
                    | b'['
                    | b']'
                    | b'.'
                    | b':'
                    | b';'
                    | b','
                    | b'_'
                    | b'-'
                    | b'/'
                    | b'\\'
            )
        })
        .count();
    words + (punctuation / 4)
}

fn effective_chunk_tokens(chunk_tokens: usize) -> usize {
    ((chunk_tokens as f32) * 0.75).floor().max(1.0) as usize
}

fn detect_lang(path: &str) -> &'static str {
    match Path::new(path).extension().and_then(|ext| ext.to_str()) {
        Some("rs") => "rust",
        Some("py") => "python",
        Some("ts" | "tsx") => "typescript",
        Some("js" | "jsx") => "javascript",
        Some("md" | "markdown") => "markdown",
        Some("toml") => "toml",
        Some("json" | "jsonc") => "json",
        Some("yaml" | "yml") => "yaml",
        _ => "text",
    }
}

fn split_source_path(path: &str) -> (String, String) {
    if let Some((repo, relative)) = path.split_once(':') {
        return (repo.to_string(), normalize_path_string(relative));
    }

    let path = PathBuf::from(path);
    if path.is_absolute() {
        if let Some((repo, relative)) = split_absolute_source_path(&path) {
            return (repo, relative);
        }
    }

    let normalized = normalize_path_string(&path.to_string_lossy());
    let mut parts = normalized.split('/').collect::<Vec<_>>();
    if parts.len() > 1 {
        let repo = parts.remove(0).to_string();
        return (repo, parts.join("/"));
    }
    ("repo".to_string(), normalized)
}

fn split_absolute_source_path(path: &Path) -> Option<(String, String)> {
    for root in repo_roots_from_env() {
        let Ok(root) = root.canonicalize() else {
            continue;
        };
        if path.starts_with(&root) {
            let repo = root.file_name()?.to_string_lossy().to_string();
            let relative = path.strip_prefix(&root).ok()?.to_string_lossy();
            return Some((repo, normalize_path_string(&relative)));
        }
    }
    None
}

fn normalize_path_string(path: &str) -> String {
    path.replace('\\', "/")
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect::<Vec<_>>()
        .join("/")
}

fn safe_relative_path(path: &str) -> Result<String> {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        bail!("skipped absolute path outside repo root: {path}");
    }
    if candidate.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::Prefix(_) | Component::RootDir
        )
    }) {
        bail!("skipped unsafe relative path: {path}");
    }
    Ok(normalize_path_string(path))
}

fn resolve_repo_root(repo: &str) -> Result<PathBuf> {
    let direct = PathBuf::from(repo);
    if direct.is_dir() {
        return direct
            .canonicalize()
            .with_context(|| format!("failed to canonicalize repo root: {repo}"));
    }
    for root in repo_roots_from_env() {
        if root
            .file_name()
            .map(|name| name.to_string_lossy() == repo)
            .unwrap_or(false)
        {
            return root
                .canonicalize()
                .with_context(|| format!("failed to canonicalize repo root: {}", root.display()));
        }
    }
    bail!("unable to resolve repo root: {repo}")
}

fn repo_roots_from_env() -> Vec<PathBuf> {
    std::env::var("RAG_REPO_ROOTS")
        .ok()
        .map(|value| {
            value
                .split(':')
                .filter(|part| !part.trim().is_empty())
                .map(PathBuf::from)
                .collect::<Vec<_>>()
        })
        .filter(|roots| !roots.is_empty())
        .unwrap_or_else(|| vec![std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))])
}

fn collect_indexable_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root) {
        let entry = entry.with_context(|| format!("failed to walk {}", root.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let relative = path.strip_prefix(root)?;
        let rel = relative.to_string_lossy().replace('\\', "/");
        if rel.starts_with(".git/")
            || rel.starts_with("target/")
            || rel.starts_with("node_modules/")
            || rel.starts_with(".idea/")
            || rel.starts_with(".codex/")
        {
            continue;
        }
        if is_indexable_path(path) {
            files.push(path.to_path_buf());
        }
    }
    Ok(files)
}

fn is_indexable_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some(
            "rs" | "py"
                | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "go"
                | "java"
                | "c"
                | "cc"
                | "cpp"
                | "h"
                | "md"
                | "markdown"
                | "toml"
                | "json"
                | "jsonc"
                | "yaml"
                | "yml"
                | "txt"
        )
    )
}

async fn is_binary_or_large(path: &Path) -> Result<bool> {
    let metadata = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("failed to stat {}", path.display()))?;
    if metadata.len() > MAX_FILE_BYTES {
        return Ok(true);
    }
    Ok(!is_indexable_path(path))
}

fn chunk_id(source_path: &str, chunk_index: usize) -> String {
    sha256_hex(format!("{source_path}:{chunk_index}").as_bytes())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn manifest_path_from_env() -> PathBuf {
    env_string("AGENTOS_RAG_INDEX_MANIFEST")
        .or_else(|| env_string("RAG_INDEX_MANIFEST"))
        .map(PathBuf::from)
        .unwrap_or_else(default_manifest_path)
}

fn default_manifest_path() -> PathBuf {
    let system_path = PathBuf::from(SYSTEM_MANIFEST_PATH);
    if system_path
        .parent()
        .map(|parent| parent.is_dir() && is_writable_dir(parent))
        .unwrap_or(false)
    {
        return system_path;
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .join("agentos")
        .join("rag_index_manifest.json")
}

fn is_writable_dir(path: &Path) -> bool {
    let test_path = path.join(format!(".agentos-write-test-{}", std::process::id()));
    match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&test_path)
    {
        Ok(_) => std::fs::remove_file(test_path).is_ok(),
        Err(_) => false,
    }
}

fn env_string(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_id_is_stable_sha256_hex() {
        let id = chunk_id("agentos-core:src/main.rs", 3);
        assert_eq!(id.len(), 64);
        assert_eq!(id, chunk_id("agentos-core:src/main.rs", 3));
        assert_ne!(id, chunk_id("agentos-core:src/main.rs", 4));
    }

    #[test]
    fn markdown_chunks_by_heading() {
        let chunks = chunk_content("README.md", "# A\none two\n# B\nthree four", 10, 0.25);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("one"));
        assert!(chunks[1].contains("three"));
    }

    #[test]
    fn code_chunks_on_function_boundaries() {
        let content = "fn a() {}\n\nfn b() {}\n";
        let chunks = chunk_content("src/lib.rs", content, 2, 0.0);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn unsafe_diff_paths_are_rejected() {
        assert!(safe_relative_path("../secret.txt").is_err());
        assert!(safe_relative_path("/etc/passwd").is_err());
        assert_eq!(safe_relative_path("src/main.rs").unwrap(), "src/main.rs");
    }

    #[test]
    fn split_source_path_preserves_deep_relative_paths() {
        let left = split_source_path("repo/src/main.rs");
        let right = split_source_path("other/src/main.rs");

        assert_eq!(left, ("repo".to_string(), "src/main.rs".to_string()));
        assert_eq!(right, ("other".to_string(), "src/main.rs".to_string()));
        assert_ne!(
            format!("{}:{}", left.0, left.1),
            format!("{}:{}", right.0, right.1)
        );
    }

    #[test]
    fn approx_tokens_counts_code_density() {
        assert!(approx_tokens("foo.bar_baz(qux);") > 1);
        assert_eq!(effective_chunk_tokens(800), 600);
    }
}
