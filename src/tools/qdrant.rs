use anyhow::{Context, Result};

use crate::rag::VectorDbConfig;

pub async fn qdrant_upsert(
    cfg: &VectorDbConfig,
    id: usize,
    embedding: Vec<f32>,
    metadata: serde_json::Value,
) -> Result<()> {
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "points": [{
            "id": id,
            "vector": embedding,
            "payload": metadata
        }]
    });

    client
        .put(format!(
            "{}/collections/{}/points?wait=true",
            cfg.host.trim_end_matches('/'),
            cfg.collection
        ))
        .json(&body)
        .send()
        .await
        .context("failed to call Qdrant upsert API")?
        .error_for_status()
        .context("Qdrant upsert API returned an error")?;

    Ok(())
}
