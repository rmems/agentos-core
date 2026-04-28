use std::path::PathBuf;

use anyhow::Result;
use rmcp::{
    ErrorData, Json, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        AnnotateAble, ListResourceTemplatesResult, ListResourcesResult, RawResource,
        RawResourceTemplate, ReadResourceRequestParams, ReadResourceResult, ResourceContents,
        ServerCapabilities, ServerInfo,
    },
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use crate::config::ServerConfig;
use crate::rag::{
    Chunk, IndexReposReport, index_default_repos, load_rag_config, load_vector_db_config, search,
};
use crate::repo::{
    InvariantReport, ModuleGraph, PublicSymbol, RepoSummary, SearchHit, SearchRequest,
    TermResolution, TerminologyReport, build_module_graph, build_summary, list_public_api,
    module_graph_markdown, public_api_markdown, resolve_term, run_invariants, scan_repo,
    search_snapshot, terminology_markdown, validate_terminology,
};
use crate::session::{SessionSeed, SessionStore, SessionUpdate, SessionView};
use crate::tools::ollama::ollama_generate;

#[derive(Debug, Clone)]
pub struct DiscoveryServer {
    tool_router: ToolRouter<Self>,
    config: ServerConfig,
    sessions: SessionStore,
}

impl DiscoveryServer {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            tool_router: Self::tool_router(),
            config,
            sessions: SessionStore::default(),
        }
    }

    fn default_root(&self) -> PathBuf {
        self.config.resolved_default_repo_root()
    }

    fn snapshot_for(
        &self,
        session_id: Option<&str>,
    ) -> Result<crate::repo::RepoSnapshot, ErrorData> {
        match session_id {
            Some(id) => self.sessions.snapshot(id).map_err(internal_error),
            None => scan_repo(&self.default_root()).map_err(internal_error),
        }
    }

    fn effective_terms(&self, session_id: Option<&str>) -> Result<Vec<String>, ErrorData> {
        match session_id {
            Some(id) => self.sessions.effective_terms(id).map_err(internal_error),
            None => Ok(self.config.default_pinned_terms.clone()),
        }
    }

    fn policy_resource(&self) -> String {
        self.config.policy_markdown()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CreateSessionRequest {
    pub label: Option<String>,
    pub repo_root: Option<String>,
    pub active_stage: Option<String>,
    #[serde(default)]
    pub pinned_terms: Vec<String>,
    #[serde(default)]
    pub invariants: Vec<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionRefRequest {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UpdateSessionRequest {
    pub session_id: String,
    pub label: Option<String>,
    pub active_stage: Option<String>,
    pub pinned_terms: Option<Vec<String>>,
    pub invariants: Option<Vec<String>>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RepoScopedRequest {
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchCanonRequest {
    pub session_id: Option<String>,
    pub query: String,
    pub max_results: Option<usize>,
    pub path_substring: Option<String>,
    pub case_sensitive: Option<bool>,
    pub regex: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchCanonResponse {
    pub snapshot_id: String,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ResolveTermRequest {
    pub session_id: Option<String>,
    pub term: String,
    pub max_hits: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PublicApiRequest {
    pub session_id: Option<String>,
    pub module_prefix: Option<String>,
    pub max_items: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PublicApiResponse {
    pub snapshot_id: String,
    pub symbols: Vec<PublicSymbol>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TerminologyRequest {
    pub session_id: Option<String>,
    pub terms: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RunInvariantsRequest {
    pub session_id: Option<String>,
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionListResponse {
    pub sessions: Vec<SessionView>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CallOllamaRequest {
    pub model: String,
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CallOllamaResponse {
    pub model: String,
    pub response: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchReposRequest {
    pub query: String,
    pub max_context_chunks: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchReposResponse {
    pub chunks: Vec<Chunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GetFileRequest {
    pub repo_root: Option<String>,
    pub relative_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GetFileResponse {
    pub repo_root: String,
    pub relative_path: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SummarizeChunksRequest {
    pub query: String,
    pub max_context_chunks: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SummarizeChunksResponse {
    pub query: String,
    pub summary: String,
    pub chunks: Vec<Chunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CallOpencodeRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CallOpencodeResponse {
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

#[tool_router]
impl DiscoveryServer {
    #[tool(
        description = "Create an explicit research session handle pinned to a repo and canon scope."
    )]
    async fn create_session(
        &self,
        Parameters(request): Parameters<CreateSessionRequest>,
    ) -> Result<Json<SessionView>, ErrorData> {
        let repo_root = request
            .repo_root
            .map(PathBuf::from)
            .unwrap_or_else(|| self.default_root());
        let pinned_terms = if request.pinned_terms.is_empty() {
            self.config.default_pinned_terms.clone()
        } else {
            request.pinned_terms
        };
        let invariants = if request.invariants.is_empty() {
            self.config.default_invariants.clone()
        } else {
            request.invariants
        };

        self.sessions
            .create(SessionSeed {
                label: request.label,
                repo_root,
                active_stage: request.active_stage,
                pinned_terms,
                invariants,
                notes: request.notes,
            })
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "List all explicit session handles currently held by the server.")]
    async fn list_sessions(&self) -> Result<Json<SessionListResponse>, ErrorData> {
        self.sessions
            .list()
            .map(|sessions| Json(SessionListResponse { sessions }))
            .map_err(internal_error)
    }

    #[tool(description = "Inspect a single session handle and its pinned scope.")]
    async fn get_session(
        &self,
        Parameters(request): Parameters<SessionRefRequest>,
    ) -> Result<Json<SessionView>, ErrorData> {
        self.sessions
            .get(&request.session_id)
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(
        description = "Update the label, stage, pinned terms, or invariant scope of a session handle."
    )]
    async fn update_session(
        &self,
        Parameters(request): Parameters<UpdateSessionRequest>,
    ) -> Result<Json<SessionView>, ErrorData> {
        self.sessions
            .update(SessionUpdate {
                session_id: request.session_id,
                label: request.label,
                active_stage: request.active_stage,
                pinned_terms: request.pinned_terms,
                invariants: request.invariants,
                notes: request.notes,
            })
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "Rescan the repo and refresh the indexed canon for a session handle.")]
    async fn reset_session(
        &self,
        Parameters(request): Parameters<SessionRefRequest>,
    ) -> Result<Json<SessionView>, ErrorData> {
        self.sessions
            .reset(&request.session_id)
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "Drop a session handle and forget its explicit research state.")]
    async fn drop_session(
        &self,
        Parameters(request): Parameters<SessionRefRequest>,
    ) -> Result<Json<SessionView>, ErrorData> {
        self.sessions
            .drop_session(&request.session_id)
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(
        description = "Summarize the authoritative repo canon, architecture, and public API surface."
    )]
    async fn repo_summary(
        &self,
        Parameters(request): Parameters<RepoScopedRequest>,
    ) -> Result<Json<RepoSummary>, ErrorData> {
        let snapshot = self.snapshot_for(request.session_id.as_deref())?;
        Ok(Json(build_summary(&snapshot)))
    }

    #[tool(description = "Search the indexed canon with bounded text or regex matching.")]
    async fn search_canon(
        &self,
        Parameters(request): Parameters<SearchCanonRequest>,
    ) -> Result<Json<SearchCanonResponse>, ErrorData> {
        let snapshot = self.snapshot_for(request.session_id.as_deref())?;
        let hits = search_snapshot(
            &snapshot,
            &SearchRequest {
                query: request.query,
                max_results: request.max_results.unwrap_or(10).clamp(1, 50),
                path_substring: request.path_substring,
                case_sensitive: request.case_sensitive.unwrap_or(false),
                regex: request.regex.unwrap_or(false),
            },
        )
        .map_err(internal_error)?;

        Ok(Json(SearchCanonResponse {
            snapshot_id: snapshot.snapshot_id,
            hits,
        }))
    }

    #[tool(
        description = "Resolve a repo term against authoritative canon and surface ambiguity instead of normalizing it away."
    )]
    async fn resolve_term(
        &self,
        Parameters(request): Parameters<ResolveTermRequest>,
    ) -> Result<Json<TermResolution>, ErrorData> {
        let snapshot = self.snapshot_for(request.session_id.as_deref())?;
        resolve_term(
            &snapshot,
            &request.term,
            request.max_hits.unwrap_or(10).clamp(1, 25),
        )
        .map(Json)
        .map_err(internal_error)
    }

    #[tool(description = "List the public Rust API extracted from the authoritative repo.")]
    async fn list_public_api(
        &self,
        Parameters(request): Parameters<PublicApiRequest>,
    ) -> Result<Json<PublicApiResponse>, ErrorData> {
        let snapshot = self.snapshot_for(request.session_id.as_deref())?;
        let symbols = list_public_api(
            &snapshot,
            request.module_prefix.as_deref(),
            request.max_items.unwrap_or(100).clamp(1, 500),
        );
        Ok(Json(PublicApiResponse {
            snapshot_id: snapshot.snapshot_id,
            symbols,
        }))
    }

    #[tool(
        description = "Generate the repo module and import graph from the indexed Rust source tree."
    )]
    async fn module_graph(
        &self,
        Parameters(request): Parameters<RepoScopedRequest>,
    ) -> Result<Json<ModuleGraph>, ErrorData> {
        let snapshot = self.snapshot_for(request.session_id.as_deref())?;
        Ok(Json(build_module_graph(&snapshot)))
    }

    #[tool(description = "Check terminology consistency for pinned or requested terms.")]
    async fn validate_terminology(
        &self,
        Parameters(request): Parameters<TerminologyRequest>,
    ) -> Result<Json<TerminologyReport>, ErrorData> {
        let snapshot = self.snapshot_for(request.session_id.as_deref())?;
        let terms = request
            .terms
            .unwrap_or(self.effective_terms(request.session_id.as_deref())?);
        validate_terminology(&snapshot, &terms)
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(
        description = "Run bounded repo invariants, including cargo test and terminology checks."
    )]
    async fn run_invariants(
        &self,
        Parameters(request): Parameters<RunInvariantsRequest>,
    ) -> Result<Json<InvariantReport>, ErrorData> {
        let snapshot = self.snapshot_for(request.session_id.as_deref())?;
        let terms = self.effective_terms(request.session_id.as_deref())?;
        run_invariants(&snapshot, &terms, request.timeout_seconds.unwrap_or(60))
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "Call a local Ollama model for bounded text generation.")]
    async fn call_ollama(
        &self,
        Parameters(request): Parameters<CallOllamaRequest>,
    ) -> Result<Json<CallOllamaResponse>, ErrorData> {
        let response =
            ollama_generate(&request.model, &request.prompt, "http://127.0.0.1:11434")
                .await
                .map_err(internal_error)?;
        Ok(Json(CallOllamaResponse {
            model: request.model,
            response,
        }))
    }

    #[tool(description = "Index configured local repos into the local Qdrant vector database.")]
    async fn index_repos(&self) -> Result<Json<IndexReposReport>, ErrorData> {
        let rag = load_rag_config().map_err(internal_error)?;
        let db = load_vector_db_config().map_err(internal_error)?;
        index_default_repos(&rag, &db)
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "Search locally indexed repo chunks through Ollama embeddings and Qdrant.")]
    async fn search_repos(
        &self,
        Parameters(request): Parameters<SearchReposRequest>,
    ) -> Result<Json<SearchReposResponse>, ErrorData> {
        let mut rag = load_rag_config().map_err(internal_error)?;
        if let Some(limit) = request.max_context_chunks {
            rag.max_context_chunks = limit.clamp(1, 32);
        }
        let db = load_vector_db_config().map_err(internal_error)?;
        search(&request.query, &rag, &db)
            .await
            .map(|chunks| Json(SearchReposResponse { chunks }))
            .map_err(internal_error)
    }

    #[tool(description = "Read a file from the default canon root or an explicitly provided repo root.")]
    async fn get_file(
        &self,
        Parameters(request): Parameters<GetFileRequest>,
    ) -> Result<Json<GetFileResponse>, ErrorData> {
        let repo_root = request
            .repo_root
            .map(PathBuf::from)
            .unwrap_or_else(|| self.default_root());
        let relative = PathBuf::from(&request.relative_path);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(ErrorData::invalid_params(
                "relative_path must be a relative path without parent directory components",
                None,
            ));
        }
        let path = repo_root.join(&relative);
        let content = tokio::fs::read_to_string(&path).await.map_err(internal_error)?;
        Ok(Json(GetFileResponse {
            repo_root: repo_root.display().to_string(),
            relative_path: request.relative_path,
            content,
        }))
    }

    #[tool(description = "Search repo chunks and produce a compact extractive summary.")]
    async fn summarize_chunks(
        &self,
        Parameters(request): Parameters<SummarizeChunksRequest>,
    ) -> Result<Json<SummarizeChunksResponse>, ErrorData> {
        let mut rag = load_rag_config().map_err(internal_error)?;
        if let Some(limit) = request.max_context_chunks {
            rag.max_context_chunks = limit.clamp(1, 32);
        }
        let db = load_vector_db_config().map_err(internal_error)?;
        let chunks = search(&request.query, &rag, &db)
            .await
            .map_err(internal_error)?;
        let summary = chunks
            .iter()
            .map(|chunk| {
                let excerpt = chunk.text.lines().take(4).collect::<Vec<_>>().join(" ");
                format!(
                    "- {}:{} [{}]: {}",
                    chunk.repo,
                    chunk.path,
                    chunk
                        .score
                        .map(|score| format!("{score:.3}"))
                        .unwrap_or_else(|| "n/a".to_string()),
                    excerpt.trim()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(Json(SummarizeChunksResponse {
            query: request.query,
            summary,
            chunks,
        }))
    }

    #[tool(description = "Call OpenCode with a bounded prompt and optional model override.")]
    async fn call_opencode(
        &self,
        Parameters(request): Parameters<CallOpencodeRequest>,
    ) -> Result<Json<CallOpencodeResponse>, ErrorData> {
        let mut command = Command::new("opencode");
        command.arg("run");
        if let Some(model) = &request.model {
            command.arg("--model").arg(model);
        }
        command.arg(&request.prompt);

        let result = timeout(
            Duration::from_secs(request.timeout_seconds.unwrap_or(60).clamp(1, 300)),
            command.output(),
        )
        .await
        .map_err(|_| ErrorData::internal_error("opencode call timed out", None))?
        .map_err(internal_error)?;

        Ok(Json(CallOpencodeResponse {
            status: result.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&result.stdout).to_string(),
            stderr: String::from_utf8_lossy(&result.stderr).to_string(),
        }))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for DiscoveryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_instructions(
            "Authoritative local repo analysis server for corinth-canal. Use explicit session \
             handles for scoped work. Prefer repo canon over inference. Surface ambiguous terms \
             instead of normalizing them. Do not assume authority from unnamed sibling repos.",
        )
    }

    fn list_resources(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListResourcesResult, ErrorData>>
    + rmcp::service::MaybeSendFuture
    + '_ {
        async move {
            let snapshot = scan_repo(&self.default_root()).map_err(internal_error)?;
            Ok(ListResourcesResult {
                meta: None,
                resources: vec![
                    RawResource::new("canon://policy", "canon-policy")
                        .with_description("Pinned repo policy and canon boundaries.")
                        .with_mime_type("text/markdown")
                        .no_annotation(),
                    RawResource::new("canon://summary", "repo-summary")
                        .with_description("High-level summary of the authoritative repo.")
                        .with_mime_type("text/markdown")
                        .no_annotation(),
                    RawResource::new("canon://readme", "repo-readme")
                        .with_description("README.md from the authoritative repo.")
                        .with_mime_type("text/markdown")
                        .no_annotation(),
                    RawResource::new("canon://public-api", "public-api")
                        .with_description("Extracted public Rust API surface.")
                        .with_mime_type("text/markdown")
                        .no_annotation(),
                    RawResource::new("canon://module-graph", "module-graph")
                        .with_description("Module graph extracted from Rust source.")
                        .with_mime_type("text/markdown")
                        .no_annotation(),
                    RawResource::new("canon://terminology", "terminology")
                        .with_description("Terminology overview for pinned canon terms.")
                        .with_mime_type("text/markdown")
                        .no_annotation(),
                    RawResource::new(
                        format!(
                            "canon://file/{}",
                            snapshot
                                .files
                                .first()
                                .map(|f| f.relative_path.clone())
                                .unwrap_or_default()
                        ),
                        "repo-file-template-example",
                    )
                    .with_description(
                        "Example file resource URI. Use the file template for any repo path.",
                    )
                    .with_mime_type("text/plain")
                    .no_annotation(),
                ],
                next_cursor: None,
            })
        }
    }

    fn list_resource_templates(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListResourceTemplatesResult, ErrorData>>
    + rmcp::service::MaybeSendFuture
    + '_ {
        async move {
            Ok(ListResourceTemplatesResult {
                meta: None,
                resource_templates: vec![
                    RawResourceTemplate::new("canon://file/{relative_path}", "repo-file")
                        .with_description("Read any indexed file from the authoritative repo.")
                        .with_mime_type("text/plain")
                        .no_annotation(),
                ],
                next_cursor: None,
            })
        }
    }

    fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> impl std::future::Future<Output = Result<ReadResourceResult, ErrorData>>
    + rmcp::service::MaybeSendFuture
    + '_ {
        async move {
            let snapshot = scan_repo(&self.default_root()).map_err(internal_error)?;
            let content = match request.uri.as_str() {
                "canon://policy" => {
                    ResourceContents::text(self.policy_resource(), "canon://policy")
                        .with_mime_type("text/markdown")
                }
                "canon://summary" => ResourceContents::text(
                    serde_json::to_string_pretty(&build_summary(&snapshot))
                        .map_err(internal_error)?,
                    "canon://summary",
                )
                .with_mime_type("application/json"),
                "canon://readme" => {
                    let readme = snapshot
                        .files
                        .iter()
                        .find(|file| file.relative_path.eq_ignore_ascii_case("README.md"))
                        .ok_or_else(|| {
                            ErrorData::resource_not_found("README.md not found", None)
                        })?;
                    ResourceContents::text(readme.content.clone(), "canon://readme")
                        .with_mime_type("text/markdown")
                }
                "canon://public-api" => ResourceContents::text(
                    public_api_markdown(&snapshot.public_symbols),
                    "canon://public-api",
                )
                .with_mime_type("text/markdown"),
                "canon://module-graph" => ResourceContents::text(
                    module_graph_markdown(&build_module_graph(&snapshot)),
                    "canon://module-graph",
                )
                .with_mime_type("text/markdown"),
                "canon://terminology" => ResourceContents::text(
                    terminology_markdown(&snapshot, &self.config.default_pinned_terms)
                        .map_err(internal_error)?,
                    "canon://terminology",
                )
                .with_mime_type("text/markdown"),
                uri if uri.starts_with("canon://file/") => {
                    let relative = uri.trim_start_matches("canon://file/");
                    let file = snapshot
                        .files
                        .iter()
                        .find(|file| file.relative_path == relative)
                        .ok_or_else(|| {
                            ErrorData::resource_not_found(
                                format!("repo file not found: {relative}"),
                                None,
                            )
                        })?;
                    ResourceContents::text(file.content.clone(), request.uri.clone())
                        .with_mime_type(file_mime_type(relative))
                }
                _ => {
                    return Err(ErrorData::resource_not_found(
                        format!("unknown resource uri: {}", request.uri),
                        None,
                    ));
                }
            };

            Ok(ReadResourceResult::new(vec![content]))
        }
    }
}

fn file_mime_type(relative_path: &str) -> &'static str {
    if relative_path.ends_with(".md") {
        "text/markdown"
    } else if relative_path.ends_with(".rs") {
        "text/rust"
    } else if relative_path.ends_with(".toml") {
        "application/toml"
    } else {
        "text/plain"
    }
}

fn internal_error<E: std::fmt::Display>(error: E) -> ErrorData {
    ErrorData::internal_error(error.to_string(), None)
}
