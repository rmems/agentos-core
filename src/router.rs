use std::env;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[cfg(unix)]
const SYSTEM_ROUTER_CONFIG_PATH: &str = "/etc/agentos/configs/model_router.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelProvider {
    Ollama,
    OllamaCloud,
    Openai,
    Anthropic,
    #[serde(rename = "google-ai-studio")]
    GoogleAiStudio,
    Google,
    Azure,
    Openrouter,
    #[serde(rename = "nvidia-nim")]
    NvidiaNim,
    Codex,
    #[serde(rename = "opencode")]
    OpenCode,
}

impl ModelProvider {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::OllamaCloud => "ollama-cloud",
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::GoogleAiStudio => "google-ai-studio",
            Self::Google => "google",
            Self::Azure => "azure",
            Self::Openrouter => "openrouter",
            Self::NvidiaNim => "nvidia-nim",
            Self::Codex => "codex",
            Self::OpenCode => "opencode",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "ollama-cloud" | "ollamacloud" => Some(Self::OllamaCloud),
            "openai" => Some(Self::Openai),
            "anthropic" => Some(Self::Anthropic),
            "google-ai-studio" | "googleaistudio" => Some(Self::GoogleAiStudio),
            "google" => Some(Self::Google),
            "azure" => Some(Self::Azure),
            "openrouter" => Some(Self::Openrouter),
            "nvidia-nim" | "nvidianim" => Some(Self::NvidiaNim),
            "codex" => Some(Self::Codex),
            "opencode" => Some(Self::OpenCode),
            _ => None,
        }
    }

    pub fn api_key_env_var(&self) -> Option<&'static str> {
        match self {
            Self::Openai => Some("OPENAI_API_KEY"),
            Self::Anthropic => Some("ANTHROPIC_API_KEY"),
            Self::GoogleAiStudio | Self::Google => Some("GEMINI_API_KEY"),
            Self::Azure => Some("AZURE_OPENAI_API_KEY"),
            Self::Openrouter => Some("OPENROUTER_API_KEY"),
            Self::NvidiaNim => Some("NVIDIA_NIM_API_KEY"),
            Self::OllamaCloud => Some("OLLAMA_API_KEY"),
            Self::Ollama | Self::Codex | Self::OpenCode => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRouterConfig {
    pub roles: Vec<RoleProvider>,
}

impl Default for ModelRouter {
    fn default() -> Self {
        // Built-in defaults only. Config/env overlays are applied explicitly by callers.
        Self::built_in_defaults()
    }
}

impl ModelRouter {
    pub fn built_in_defaults() -> Self {
        Self {
            roles: vec![RoleProvider {
                role: ProviderRole::RagEmbedding,
                provider: ModelProvider::Ollama,
                model: Some("nomic-embed-text:latest".to_string()),
            }],
        }
    }

    pub fn load_system_config_if_present() -> Result<Option<Self>> {
        #[cfg(unix)]
        {
            let path = std::path::Path::new(SYSTEM_ROUTER_CONFIG_PATH);
            if !path.exists() {
                return Ok(None);
            }
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let config: ModelRouterConfig = serde_json::from_str(&raw)
                .with_context(|| format!("failed to parse {}", path.display()))?;
            Ok(Some(Self {
                roles: config.roles,
            }))
        }
        #[cfg(not(unix))]
        {
            Ok(None)
        }
    }

    pub fn from_env() -> Result<Self> {
        let mut roles: Vec<RoleProvider> = Vec::new();
        let all_roles = ProviderRole::all();

        for role in all_roles {
            let key = role.to_env_key();
            let Some(provider_str) = env::var(&key)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
            else {
                continue;
            };

            let Some(provider) = ModelProvider::from_str(&provider_str) else {
                bail!("unknown provider '{}' for {}", provider_str, key);
            };

            if let Some(key_var) = provider.api_key_env_var() {
                let has_key = env::var(key_var)
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .is_some();
                if !has_key {
                    bail!(
                        "{} is required for {} provider in role {} but not set",
                        key_var,
                        provider.name(),
                        role.name()
                    );
                }
            }

            let model = env::var(format!("{key}_MODEL"))
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty());
            roles.push(RoleProvider {
                role,
                provider,
                model,
            });
        }

        // Return only explicit overrides. Callers can layer these on top of defaults.
        Ok(Self { roles })
    }

    pub fn get_provider(&self, role: ProviderRole) -> Option<&RoleProvider> {
        self.roles.iter().find(|r| r.role == role)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn with_env_lock<T>(f: impl FnOnce() -> T) -> T {
        let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().expect("env lock poisoned");
        f()
    }

    fn clear_role_env() {
        for role in ProviderRole::all() {
            let key = role.to_env_key();
            unsafe {
                std::env::remove_var(&key);
                std::env::remove_var(format!("{key}_MODEL"));
            }
        }
        for key in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "NVIDIA_NIM_API_KEY",
            "OLLAMA_API_KEY",
        ] {
            unsafe {
                std::env::remove_var(key);
            }
        }
    }

    #[test]
    fn model_provider_from_str_handles_case_insensitive() {
        assert_eq!(
            ModelProvider::from_str("OPENAI"),
            Some(ModelProvider::Openai)
        );
        assert_eq!(
            ModelProvider::from_str("openai"),
            Some(ModelProvider::Openai)
        );
        assert_eq!(
            ModelProvider::from_str("OpenAI"),
            Some(ModelProvider::Openai)
        );
    }

    #[test]
    fn model_provider_api_key_env_vars_are_explicit() {
        assert_eq!(ModelProvider::Ollama.api_key_env_var(), None);
        assert_eq!(ModelProvider::Codex.api_key_env_var(), None);
        assert_eq!(ModelProvider::OpenCode.api_key_env_var(), None);
        assert_eq!(
            ModelProvider::Openai.api_key_env_var(),
            Some("OPENAI_API_KEY")
        );
        assert_eq!(
            ModelProvider::GoogleAiStudio.api_key_env_var(),
            Some("GEMINI_API_KEY")
        );
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
    fn default_router_has_rag_embedding_by_default() {
        let router = ModelRouter::default();
        let rag = router.get_provider(ProviderRole::RagEmbedding);
        assert!(rag.is_some());
        assert_eq!(rag.unwrap().provider, ModelProvider::Ollama);
    }

    #[test]
    fn env_parsing_unknown_provider_errors() {
        with_env_lock(|| {
            clear_role_env();
            unsafe {
                std::env::set_var("AGENTOS_ROLE_DEFAULT_CODING", "definitely-not-a-provider");
            }
            let err = ModelRouter::from_env().unwrap_err().to_string();
            assert!(err.contains("unknown provider"), "{err}");
        });
    }

    #[test]
    fn env_parsing_missing_key_errors_for_keyed_provider() {
        with_env_lock(|| {
            clear_role_env();
            unsafe {
                std::env::set_var("AGENTOS_ROLE_DEFAULT_CODING", "openai");
            }
            let err = ModelRouter::from_env().unwrap_err().to_string();
            assert!(err.contains("OPENAI_API_KEY"), "{err}");
        });
    }

    #[test]
    fn env_parsing_empty_key_is_treated_as_missing() {
        with_env_lock(|| {
            clear_role_env();
            unsafe {
                std::env::set_var("AGENTOS_ROLE_DEFAULT_CODING", "openai");
                std::env::set_var("OPENAI_API_KEY", "   ");
            }
            let err = ModelRouter::from_env().unwrap_err().to_string();
            assert!(err.contains("OPENAI_API_KEY"), "{err}");
        });
    }

    #[test]
    fn env_parsing_valid_provider_with_key_yields_override() {
        with_env_lock(|| {
            clear_role_env();
            unsafe {
                std::env::set_var("AGENTOS_ROLE_DEFAULT_CODING", "openai");
                std::env::set_var("OPENAI_API_KEY", "test");
            }
            let router = ModelRouter::from_env().unwrap();
            let rp = router.get_provider(ProviderRole::DefaultCoding).unwrap();
            assert_eq!(rp.provider, ModelProvider::Openai);
        });
    }

    #[test]
    fn env_parsing_no_role_vars_returns_empty_overrides() {
        with_env_lock(|| {
            clear_role_env();
            let router = ModelRouter::from_env().unwrap();
            assert!(router.roles.is_empty());
        });
    }
}
