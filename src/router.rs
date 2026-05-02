use std::env;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ModelProvider {
    Ollama,
    OpenAI,
    Anthropic,
    Google,
    Azure,
}

impl ModelProvider {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::OpenAI => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
            Self::Azure => "azure",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "openai" => Some(Self::OpenAI),
            "anthropic" => Some(Self::Anthropic),
            "google" => Some(Self::Google),
            "azure" => Some(Self::Azure),
            _ => None,
        }
    }

    pub fn requires_api_key(&self) -> bool {
        !matches!(self, Self::Ollama)
    }
}

pub struct ModelRouter {
    pub default_provider: ModelProvider,
    pub ollama_endpoint: String,
    pub ollama_model: String,
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self {
            default_provider: ModelProvider::Ollama,
            ollama_endpoint: env::var("OLLAMA_ENDPOINT")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            ollama_model: env::var("OLLAMA_MODEL")
                .unwrap_or_else(|_| "llama3.2".to_string()),
        }
    }
}

impl ModelRouter {
    pub fn from_env() -> Result<Self> {
        let default_provider = env::var("MODEL_PROVIDER")
            .ok()
            .and_then(|p| ModelProvider::from_str(&p))
            .unwrap_or(ModelProvider::Ollama);

        // Validate API key is set for non-local providers
        if default_provider.requires_api_key() {
            let (key_var, provider_name) = match default_provider {
                ModelProvider::OpenAI => ("OPENAI_API_KEY", "openai"),
                ModelProvider::Anthropic => ("ANTHROPIC_API_KEY", "anthropic"),
                ModelProvider::Google => ("GOOGLE_API_KEY", "google"),
                ModelProvider::Azure => ("AZURE_OPENAI_API_KEY", "azure"),
                ModelProvider::Ollama => ("", "ollama"),
            };
            if !key_var.is_empty() && env::var(key_var).is_err() {
                bail!("{} is required for {} provider but not set", key_var, provider_name);
            }
        }

        Ok(Self {
            default_provider,
            ollama_endpoint: env::var("OLLAMA_ENDPOINT")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            ollama_model: env::var("OLLAMA_MODEL")
                .unwrap_or_else(|_| "llama3.2".to_string()),
        })
    }

    pub fn name(&self) -> &str {
        match self.default_provider {
            ModelProvider::Ollama => "ollama",
            ModelProvider::OpenAI => "openai",
            ModelProvider::Anthropic => "anthropic",
            ModelProvider::Google => "google",
            ModelProvider::Azure => "azure",
        }
    }
}