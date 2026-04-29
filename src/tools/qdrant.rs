use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use crate::rag::VectorDbConfig;

pub struct QdrantVectorRecord<'a> {
    pub id: &'a str,
    pub embedding: &'a [f32],
    pub metadata: &'a serde_json::Value,
}

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
    vectors: &[QdrantVectorRecord<'_>],
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

    let bytes = id.as_bytes();
    let uuid_bytes = if bytes.len() >= 32 && bytes[..32].iter().all(|byte| byte.is_ascii_hexdigit())
    {
        decode_hex_16(&bytes[..32])
    } else {
        let digest = Sha256::digest(bytes);
        let mut out = [0u8; 16];
        out.copy_from_slice(&digest[..16]);
        out
    };

    let hex = uuid_bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
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

fn decode_hex_16(hex: &[u8]) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    for (idx, pair) in hex.chunks_exact(2).enumerate() {
        bytes[idx] = (hex_value(pair[0]) << 4) | hex_value(pair[1]);
    }
    bytes
}

fn hex_value(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_point_id_is_stable_uuid_shape() {
        let first = qdrant_point_id("agentos-core:src/main.rs:0");
        let second = qdrant_point_id("agentos-core:src/main.rs:0");

        assert_eq!(first, second);
        assert!(is_uuid(&first));
    }

    #[test]
    fn fallback_point_id_handles_non_ascii() {
        let point_id = qdrant_point_id("repo:src/\u{fc}mlaut.rs:0");

        assert!(is_uuid(&point_id));
    }
}
