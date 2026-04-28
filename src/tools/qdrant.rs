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

pub async fn qdrant_upsert_vectors(
    cfg: &VectorDbConfig,
    vectors: &[crate::orchestrator::VectorRecord],
) -> Result<()> {
    if vectors.is_empty() {
        return Ok(());
    }

    let client = reqwest::Client::new();
    let points = vectors
        .iter()
        .map(|vector| {
            serde_json::json!({
                "id": qdrant_point_id(&vector.id),
                "vector": vector.embedding,
                "payload": vector.metadata
            })
        })
        .collect::<Vec<_>>();

    let body = serde_json::json!({ "points": points });
    client
        .put(format!(
            "{}/collections/{}/points?wait=true",
            cfg.host.trim_end_matches('/'),
            cfg.collection
        ))
        .json(&body)
        .send()
        .await
        .context("failed to call Qdrant vector upsert API")?
        .error_for_status()
        .context("Qdrant vector upsert API returned an error")?;

    Ok(())
}

pub async fn qdrant_delete_vectors(cfg: &VectorDbConfig, ids: &[String]) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }

    let client = reqwest::Client::new();
    let points = ids.iter().map(|id| qdrant_point_id(id)).collect::<Vec<_>>();
    let body = serde_json::json!({ "points": points });
    client
        .post(format!(
            "{}/collections/{}/points/delete?wait=true",
            cfg.host.trim_end_matches('/'),
            cfg.collection
        ))
        .json(&body)
        .send()
        .await
        .context("failed to call Qdrant vector delete API")?
        .error_for_status()
        .context("Qdrant vector delete API returned an error")?;

    Ok(())
}

fn qdrant_point_id(id: &str) -> String {
    if is_uuid(id) {
        return id.to_string();
    }
    let hex = if id.len() >= 32 && id.chars().take(32).all(|ch| ch.is_ascii_hexdigit()) {
        id[..32].to_ascii_lowercase()
    } else {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hasher);
        format!("{:016x}{:016x}", hasher.finish(), hasher.finish())
    };
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

fn is_uuid(id: &str) -> bool {
    let bytes = id.as_bytes();
    bytes.len() == 36
        && bytes[8] == b'-'
        && bytes[13] == b'-'
        && bytes[18] == b'-'
        && bytes[23] == b'-'
        && id
            .chars()
            .filter(|ch| *ch != '-')
            .all(|ch| ch.is_ascii_hexdigit())
}
