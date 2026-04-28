use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub endpoint: String,
    pub embedding_model: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:11434".to_string(),
            embedding_model: "nomic-embed-text:latest".to_string(),
        }
    }
}

pub async fn ollama_embed(text: &str, cfg: &OllamaConfig) -> Result<Vec<f32>> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": cfg.embedding_model,
        "input": text
    });

    let resp = client
        .post(format!("{}/api/embed", cfg.endpoint.trim_end_matches('/')))
        .json(&body)
        .send()
        .await
        .context("failed to call Ollama embed API")?
        .error_for_status()
        .context("Ollama embed API returned an error")?
        .json::<serde_json::Value>()
        .await
        .context("failed to parse Ollama embed response")?;

    let values = resp
        .get("embeddings")
        .and_then(|embeddings| embeddings.get(0))
        .or_else(|| resp.get("embedding"))
        .and_then(|embedding| embedding.as_array())
        .ok_or_else(|| anyhow::anyhow!("Ollama response did not contain an embedding"))?;

    values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| anyhow::anyhow!("embedding contained a non-number value"))
        })
        .collect()
}

pub async fn ollama_generate(model: &str, prompt: &str, endpoint: &str) -> Result<String> {
    if model.trim().is_empty() {
        bail!("model must not be empty");
    }
    if prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false
    });

    let resp = client
        .post(format!("{}/api/generate", endpoint.trim_end_matches('/')))
        .json(&body)
        .send()
        .await
        .context("failed to call Ollama generate API")?
        .error_for_status()
        .context("Ollama generate API returned an error")?
        .json::<serde_json::Value>()
        .await
        .context("failed to parse Ollama generate response")?;

    resp.get("response")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow::anyhow!("Ollama response did not contain text"))
}
