use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::tools::ollama::{OllamaConfig, ollama_embed};
use crate::tools::qdrant::qdrant_upsert;

const RAG_CONFIG_PATH: &str = "/etc/agentos/configs/rag.json";
const VECTOR_DB_CONFIG_PATH: &str = "/etc/agentos/configs/vector_db.json";
const REPO_INDEX_CONFIG_PATH: &str = "/etc/agentos/configs/repo_index.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub embedding_provider: String,
    pub embedding_model: String,
    pub vector_db: String,
    pub rerank_provider: String,
    pub max_context_chunks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDbConfig {
    pub provider: String,
    pub host: String,
    pub collection: String,
    pub embedding_dim: usize,
    pub distance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoIndexConfig {
    pub roots: Vec<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Chunk {
    pub repo: String,
    pub path: String,
    pub chunk_index: usize,
    pub text: String,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexReposReport {
    pub indexed_roots: Vec<String>,
    pub skipped_roots: Vec<String>,
    pub chunks_indexed: usize,
}

pub fn load_rag_config() -> Result<RagConfig> {
    let raw = std::fs::read_to_string(RAG_CONFIG_PATH)
        .with_context(|| format!("failed to read {RAG_CONFIG_PATH}"))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {RAG_CONFIG_PATH}"))
}

pub fn load_vector_db_config() -> Result<VectorDbConfig> {
    let raw = std::fs::read_to_string(VECTOR_DB_CONFIG_PATH)
        .with_context(|| format!("failed to read {VECTOR_DB_CONFIG_PATH}"))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {VECTOR_DB_CONFIG_PATH}"))
}

pub fn load_repo_index_config() -> Result<RepoIndexConfig> {
    let raw = std::fs::read_to_string(REPO_INDEX_CONFIG_PATH)
        .with_context(|| format!("failed to read {REPO_INDEX_CONFIG_PATH}"))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {REPO_INDEX_CONFIG_PATH}"))
}

pub fn ollama_config_from_rag(cfg: &RagConfig) -> OllamaConfig {
    OllamaConfig {
        endpoint: "http://127.0.0.1:11434".to_string(),
        embedding_model: cfg.embedding_model.clone(),
    }
}

pub async fn search(query: &str, cfg: &RagConfig, db: &VectorDbConfig) -> Result<Vec<Chunk>> {
    if cfg.embedding_provider != "ollama" {
        bail!(
            "unsupported embedding provider for local RAG: {}",
            cfg.embedding_provider
        );
    }

    let embedding = ollama_embed(query, &ollama_config_from_rag(cfg)).await?;
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "vector": embedding,
        "limit": cfg.max_context_chunks
    });

    let resp = client
        .post(format!(
            "{}/collections/{}/points/search",
            db.host.trim_end_matches('/'),
            db.collection
        ))
        .json(&body)
        .send()
        .await
        .context("failed to call Qdrant search API")?
        .error_for_status()
        .context("Qdrant search API returned an error")?
        .json::<serde_json::Value>()
        .await
        .context("failed to parse Qdrant search response")?;

    let results = resp
        .get("result")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();

    let chunks = results
        .into_iter()
        .filter_map(|item| {
            let payload = item.get("payload")?;
            Some(Chunk {
                repo: payload.get("repo")?.as_str()?.to_string(),
                path: payload.get("path")?.as_str()?.to_string(),
                chunk_index: payload
                    .get("chunk_index")
                    .and_then(|value| value.as_u64())
                    .unwrap_or_default() as usize,
                text: payload.get("text")?.as_str()?.to_string(),
                score: item.get("score").and_then(|value| value.as_f64()),
            })
        })
        .collect();

    Ok(chunks)
}

pub async fn index_default_repos(cfg: &RagConfig, db: &VectorDbConfig) -> Result<IndexReposReport> {
    let mut indexed_roots = Vec::new();
    let mut skipped_roots = Vec::new();
    let mut chunks_indexed = 0usize;
    let ollama = ollama_config_from_rag(cfg);
    let repo_index = load_repo_index_config()?;

    for root in repo_index.roots {
        let root_path = PathBuf::from(&root);
        if !root_path.is_dir() {
            skipped_roots.push(root);
            continue;
        }

        indexed_roots.push(root);
        for chunk in repo_chunks(&root_path, cfg.chunk_size, cfg.chunk_overlap)? {
            let embedding = ollama_embed(&chunk.text, &ollama).await?;
            let id = stable_chunk_id(&chunk);
            let metadata = serde_json::json!({
                "repo": chunk.repo,
                "path": chunk.path,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text
            });
            qdrant_upsert(db, id, embedding, metadata).await?;
            chunks_indexed += 1;
        }
    }

    Ok(IndexReposReport {
        indexed_roots,
        skipped_roots,
        chunks_indexed,
    })
}

fn repo_chunks(root: &Path, chunk_size: usize, chunk_overlap: usize) -> Result<Vec<Chunk>> {
    let repo = root
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "repo".to_string());
    let mut chunks = Vec::new();

    for entry in WalkDir::new(root) {
        let entry = entry.with_context(|| format!("failed to walk {}", root.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let relative = path.strip_prefix(root)?;
        if !should_index_for_rag(relative) {
            continue;
        }

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let relative_path = relative.to_string_lossy().replace('\\', "/");
        for (chunk_index, text) in split_chunks(&content, chunk_size, chunk_overlap)
            .into_iter()
            .enumerate()
        {
            chunks.push(Chunk {
                repo: repo.clone(),
                path: relative_path.clone(),
                chunk_index,
                text,
                score: None,
            });
        }
    }

    Ok(chunks)
}

fn split_chunks(content: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let chars = content.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return Vec::new();
    }

    let chunk_size = chunk_size.max(1);
    let chunk_overlap = chunk_overlap.min(chunk_size.saturating_sub(1));
    let step = chunk_size.saturating_sub(chunk_overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let text = chars[start..end].iter().collect::<String>();
        if !text.trim().is_empty() {
            chunks.push(text);
        }
        if end == chars.len() {
            break;
        }
        start += step;
    }

    chunks
}

fn should_index_for_rag(relative: &Path) -> bool {
    let rel = relative.to_string_lossy().replace('\\', "/");
    if rel.contains("/.git/")
        || rel.starts_with(".git/")
        || rel.contains("/target/")
        || rel.starts_with("target/")
        || rel.contains("/node_modules/")
        || rel.starts_with("node_modules/")
    {
        return false;
    }

    matches!(
        relative.extension().and_then(|ext| ext.to_str()),
        Some("rs" | "toml" | "md" | "json" | "jsonc" | "yaml" | "yml" | "txt")
    )
}

fn stable_chunk_id(chunk: &Chunk) -> usize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    chunk.repo.hash(&mut hasher);
    chunk.path.hash(&mut hasher);
    chunk.chunk_index.hash(&mut hasher);
    hasher.finish() as usize
}
