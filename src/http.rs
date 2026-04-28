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

pub async fn serve(bind: &str) -> Result<()> {
    let state = Arc::new(Orchestrator::from_env()?);
    let app = Router::new()
        .route("/ingest/file", post(ingest_file))
        .route("/ingest/diff", post(ingest_diff))
        .route("/rebuild", post(rebuild))
        .route("/cleanup", post(cleanup))
        .route("/vectors/upsert", post(vectors_upsert))
        .route("/vectors/delete", post(vectors_delete))
        .with_state(state)
        .layer(middleware::from_fn(auth));

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

async fn auth(headers: HeaderMap, request: Request<Body>, next: Next) -> Response {
    let expected = std::env::var("AGENTOS_RAG_JWT")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let Some(expected) = expected else {
        return next.run(request).await;
    };

    let authorized = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(|token| token == expected)
        .unwrap_or(false);

    if authorized {
        next.run(request).await
    } else {
        (StatusCode::UNAUTHORIZED, "unauthorized").into_response()
    }
}

async fn ingest_file(
    State(orchestrator): State<Arc<Orchestrator>>,
    Json(request): Json<FileIngestRequest>,
) -> Response {
    respond(orchestrator.ingest_file(request).await)
}

async fn ingest_diff(
    State(orchestrator): State<Arc<Orchestrator>>,
    Json(request): Json<DiffIngestRequest>,
) -> Response {
    respond(orchestrator.ingest_diff(request).await)
}

async fn rebuild(
    State(orchestrator): State<Arc<Orchestrator>>,
    Json(request): Json<RebuildRequest>,
) -> Response {
    respond(orchestrator.rebuild(request).await)
}

async fn cleanup(
    State(orchestrator): State<Arc<Orchestrator>>,
    Json(request): Json<CleanupRequest>,
) -> Response {
    respond(orchestrator.cleanup(request).await)
}

async fn vectors_upsert(
    State(orchestrator): State<Arc<Orchestrator>>,
    Json(request): Json<VectorUpsertRequest>,
) -> Response {
    respond(orchestrator.vectors_upsert(request).await)
}

async fn vectors_delete(
    State(orchestrator): State<Arc<Orchestrator>>,
    Json(request): Json<VectorDeleteRequest>,
) -> Response {
    respond(orchestrator.vectors_delete(request).await)
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
