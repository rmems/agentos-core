use std::env;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ModelProvider {
    Ollama,
    OllamaCloud,
    OpenAI,
    Anthropic,
    GoogleAiStudio,
    Google,
    Azure,
    OpenRouter,
    NvidiaNim,
    Codex,
    OpenCode,
}

impl ModelProvider {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::OllamaCloud => "ollama-cloud",
            Self::OpenAI => "openai",
            Self::Anthropic => "anthropic",
            Self::GoogleAiStudio => "google-ai-studio",
            Self::Google => "google",
            Self::Azure => "azure",
            Self::OpenRouter => "openrouter",
            Self::NvidiaNim => "nvidia-nim",
            Self::Codex => "codex",
            Self::OpenCode => "opencode",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "ollama-cloud" | "ollamacloud" => Some(Self::OllamaCloud),
            "openai" => Some(Self::OpenAI),
            "anthropic" => Some(Self::Anthropic),
            "google-ai-studio" | "googleaistudio" => Some(Self::GoogleAiStudio),
            "google" => Some(Self::Google),
            "azure" => Some(Self::Azure),
            "openrouter" => Some(Self::OpenRouter),
            "nvidia-nim" | "nvidianim" => Some(Self::NvidiaNim),
            "codex" => Some(Self::Codex),
            "opencode" => Some(Self::OpenCode),
            _ => None,
        }
    }

    pub fn requires_api_key(&self) -> bool {
        !matches!(self, Self::Ollama)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ProviderRole {
    DefaultCoding,
    LongContext,
    LocalPrivate,
    RagEmbedding,
    RagEmbeddingCloud,
    CudaGpu,
    Fallback,
}

impl ProviderRole {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "default_coding" => Some(Self::DefaultCoding),
            "long_context" => Some(Self::LongContext),
            "local_private" => Some(Self::LocalPrivate),
            "rag_embedding" => Some(Self::RagEmbedding),
            "rag_embedding_cloud" => Some(Self::RagEmbeddingCloud),
            "cuda_gpu" => Some(Self::CudaGpu),
            "fallback" => Some(Self::Fallback),
            _ => None,
        }
    }

    pub fn to_env_key(&self) -> String {
        match self {
            Self::DefaultCoding => "AGENTOS_ROLE_DEFAULT_CODING".to_string(),
            Self::LongContext => "AGENTOS_ROLE_LONG_CONTEXT".to_string(),
            Self::LocalPrivate => "AGENTOS_ROLE_LOCAL_PRIVATE".to_string(),
            Self::RagEmbedding => "AGENTOS_ROLE_RAG_EMBEDDING".to_string(),
            Self::RagEmbeddingCloud => "AGENTOS_ROLE_RAG_EMBEDDING_CLOUD".to_string(),
            Self::CudaGpu => "AGENTOS_ROLE_CUDA_GPU".to_string(),
            Self::Fallback => "AGENTOS_ROLE_FALLBACK".to_string(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::DefaultCoding => "default_coding",
            Self::LongContext => "long_context",
            Self::LocalPrivate => "local_private",
            Self::RagEmbedding => "rag_embedding",
            Self::RagEmbeddingCloud => "rag_embedding_cloud",
            Self::CudaGpu => "cuda_gpu",
            Self::Fallback => "fallback",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![
            Self::DefaultCoding,
            Self::LongContext,
            Self::LocalPrivate,
            Self::RagEmbedding,
            Self::RagEmbeddingCloud,
            Self::CudaGpu,
            Self::Fallback,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleProvider {
    pub role: ProviderRole,
    pub provider: ModelProvider,
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelRouter {
    pub roles: Vec<RoleProvider>,
    pub ollama_endpoint: String,
    pub ollama_model: String,
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self {
            roles: vec![
                RoleProvider {
                    role: ProviderRole::DefaultCoding,
                    provider: ModelProvider::Ollama,
                    model: Some("llama3.2".to_string()),
                },
                RoleProvider {
                    role: ProviderRole::Fallback,
                    provider: ModelProvider::Ollama,
                    model: Some("llama3.2".to_string()),
                },
                RoleProvider {
                    role: ProviderRole::RagEmbedding,
                    provider: ModelProvider::Ollama,
                    model: Some("nomic-embed-text:latest".to_string()),
                },
            ],
            ollama_endpoint: env::var("OLLAMA_ENDPOINT")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            ollama_model: env::var("OLLAMA_MODEL")
                .unwrap_or_else(|_| "llama3.2".to_string()),
        }
    }
}

impl ModelRouter {
    pub fn from_env() -> Result<Self> {
        let mut roles: Vec<RoleProvider> = Vec::new();
        let all_roles = ProviderRole::all();

        for role in all_roles {
            let provider_str = env::var(role.to_env_key()).ok();
            let provider = provider_str
                .as_ref()
                .and_then(|p| ModelProvider::from_str(p));

            if let Some(p) = provider {
                if p.requires_api_key() {
                    let (key_var, provider_name) = match p {
                        ModelProvider::OpenAI => ("OPENAI_API_KEY", "openai"),
                        ModelProvider::Anthropic => ("ANTHROPIC_API_KEY", "anthropic"),
                        ModelProvider::GoogleAiStudio | ModelProvider::Google => {
                            ("GOOGLE_API_KEY", "google")
                        }
                        ModelProvider::Azure => ("AZURE_OPENAI_API_KEY", "azure"),
                        ModelProvider::OpenRouter => ("OPENROUTER_API_KEY", "openrouter"),
                        ModelProvider::NvidiaNim => ("NVIDIA_NIM_API_KEY", "nvidia-nim"),
                        ModelProvider::OllamaCloud => ("OLLAMA_API_KEY", "ollama-cloud"),
                        _ => ("", ""),
                    };
                    if !key_var.is_empty() && env::var(key_var).is_err() {
                        bail!(
                            "{} is required for {} provider in role {} but not set",
                            key_var,
                            provider_name,
                            role.name()
                        );
                    }
                }

                let model = env::var(format!("{}_MODEL", role.to_env_key())).ok();
                roles.push(RoleProvider {
                    role,
                    provider: p,
                    model,
                });
            }
        }

        if roles.is_empty() {
            let default_router = Self::default();
            return Ok(default_router);
        }

        Ok(Self {
            roles,
            ollama_endpoint: env::var("OLLAMA_ENDPOINT")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            ollama_model: env::var("OLLAMA_MODEL")
                .unwrap_or_else(|_| "llama3.2".to_string()),
        })
    }

    pub fn get_provider(&self, role: ProviderRole) -> Option<&RoleProvider> {
        self.roles.iter().find(|r| r.role == role)
    }

    pub fn get_fallback(&self) -> Option<&RoleProvider> {
        self.get_provider(ProviderRole::Fallback)
            .or_else(|| self.get_provider(ProviderRole::DefaultCoding))
    }

    pub fn get_embedding_provider(&self) -> Option<&RoleProvider> {
        self.get_provider(ProviderRole::RagEmbedding)
            .or_else(|| self.get_provider(ProviderRole::RagEmbeddingCloud))
            .or_else(|| self.get_provider(ProviderRole::Fallback))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_provider_from_str_handles_case_insensitive() {
        assert_eq!(ModelProvider::from_str("OPENAI"), Some(ModelProvider::OpenAI));
        assert_eq!(ModelProvider::from_str("openai"), Some(ModelProvider::OpenAI));
        assert_eq!(ModelProvider::from_str("OpenAI"), Some(ModelProvider::OpenAI));
    }

    #[test]
    fn model_provider_requires_api_key() {
        assert!(!ModelProvider::Ollama.requires_api_key());
        assert!(ModelProvider::OpenAI.requires_api_key());
        assert!(ModelProvider::Anthropic.requires_api_key());
        assert!(ModelProvider::OpenRouter.requires_api_key());
    }

    #[test]
    fn provider_role_all_includes_expected_roles() {
        let all = ProviderRole::all();
        assert!(all.contains(&ProviderRole::DefaultCoding));
        assert!(all.contains(&ProviderRole::RagEmbedding));
        assert!(all.contains(&ProviderRole::Fallback));
    }

    #[test]
    fn provider_role_env_keys_are_correct() {
        assert_eq!(
            ProviderRole::DefaultCoding.to_env_key(),
            "AGENTOS_ROLE_DEFAULT_CODING"
        );
        assert_eq!(
            ProviderRole::RagEmbedding.to_env_key(),
            "AGENTOS_ROLE_RAG_EMBEDDING"
        );
    }

    #[test]
    fn default_router_has_ollama_as_fallback() {
        let router = ModelRouter::default();
        let fallback = router.get_fallback();
        assert!(fallback.is_some());
        assert_eq!(fallback.unwrap().provider, ModelProvider::Ollama);
    }
}