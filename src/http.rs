use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};

use crate::orchestrator::{
    CleanupRequest, DiffIngestRequest, FileIngestRequest, OperationSummary, Orchestrator,
    RebuildRequest, VectorDeleteRequest, VectorUpsertRequest,
};

#[derive(Clone)]
struct HttpState {
    orchestrator: Arc<Orchestrator>,
    auth_token: Option<Arc<str>>,
}

pub async fn serve(bind: &str) -> Result<()> {
    let state = Arc::new(HttpState {
        orchestrator: Arc::new(Orchestrator::from_env()?),
        auth_token: std::env::var("AGENTOS_RAG_JWT")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .map(Arc::<str>::from),
    });
    let app = Router::new()
        .route("/ingest/file", post(ingest_file))
        .route("/ingest/diff", post(ingest_diff))
        .route("/rebuild", post(rebuild))
        .route("/cleanup", post(cleanup))
        .route("/vectors/upsert", post(vectors_upsert))
        .route("/vectors/delete", post(vectors_delete))
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state, auth));

    let addr = bind
        .parse::<SocketAddr>()
        .with_context(|| format!("invalid bind address: {bind}"))?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    tracing::info!(event = "rag_http_started", bind = %addr);
    axum::serve(listener, app)
        .await
        .context("RAG HTTP server failed")
}

async fn auth(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    request: Request<Body>,
    next: Next,
) -> Response {
    let Some(expected) = state.auth_token.as_ref() else {
        return next.run(request).await;
    };

    let authorized = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(|token| constant_time_eq(token.as_bytes(), expected.as_bytes()))
        .unwrap_or(false);

    if authorized {
        next.run(request).await
    } else {
        (StatusCode::UNAUTHORIZED, "unauthorized").into_response()
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .fold(0u8, |acc, (left, right)| acc | (left ^ right))
        == 0
}

async fn ingest_file(
    State(state): State<Arc<HttpState>>,
    Json(request): Json<FileIngestRequest>,
) -> Response {
    respond(state.orchestrator.ingest_file(request).await)
}

async fn ingest_diff(
    State(state): State<Arc<HttpState>>,
    Json(request): Json<DiffIngestRequest>,
) -> Response {
    respond(state.orchestrator.ingest_diff(request).await)
}

async fn rebuild(
    State(state): State<Arc<HttpState>>,
    Json(request): Json<RebuildRequest>,
) -> Response {
    respond(state.orchestrator.rebuild(request).await)
}

async fn cleanup(
    State(state): State<Arc<HttpState>>,
    Json(request): Json<CleanupRequest>,
) -> Response {
    respond(state.orchestrator.cleanup(request).await)
}

async fn vectors_upsert(
    State(state): State<Arc<HttpState>>,
    Json(request): Json<VectorUpsertRequest>,
) -> Response {
    respond(state.orchestrator.vectors_upsert(request).await)
}

async fn vectors_delete(
    State(state): State<Arc<HttpState>>,
    Json(request): Json<VectorDeleteRequest>,
) -> Response {
    respond(state.orchestrator.vectors_delete(request).await)
}

fn respond(result: Result<OperationSummary>) -> Response {
    match result {
        Ok(summary) => (StatusCode::OK, Json(summary)).into_response(),
        Err(error) => {
            tracing::error!(event = "rag_http_error", error = %error);
            let summary = OperationSummary {
                status: "error".to_string(),
                processed_chunks: 0,
                upserted_ids: Vec::new(),
                deleted_ids: Vec::new(),
                errors: vec![error.to_string()],
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(summary)).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches_equal_bytes() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secrex"));
        assert!(!constant_time_eq(b"secret", b"secret-longer"));
    }
}
