use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub server_name: String,
    pub default_repo_root: PathBuf,
    pub default_pinned_terms: Vec<String>,
    pub default_invariants: Vec<String>,
}

impl ServerConfig {
    pub fn resolved_default_repo_root(&self) -> PathBuf {
        env::var("AGENTOS_CORE_CANON_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| self.default_repo_root.clone())
    }

    pub fn policy_markdown(&self) -> String {
        let bullets = self
            .default_invariants
            .iter()
            .map(|line| format!("- {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "# Canon Policy\n\nDefault canon root: `{}`\n\n{}\n",
            self.resolved_default_repo_root().display(),
            bullets
        )
    }
}

pub fn discover_repo_home() -> Result<PathBuf> {
    if let Ok(home) = env::var("AGENTOS_CORE_HOME") {
        let path = PathBuf::from(home);
        if path.exists() {
            return Ok(path);
        }
    }

    env::current_dir().context("failed to resolve current directory")
}

/// Split `RAG_REPO_ROOTS` using the platform path separator rules (`:` vs `;`).
pub fn parse_rag_repo_roots(value: &OsStr) -> Vec<PathBuf> {
    env::split_paths(value)
        .filter(|p| !p.as_os_str().is_empty())
        .collect()
}

pub fn load_server_config(repo_home: &Path) -> Result<ServerConfig> {
    let path = repo_home.join("config/server.toml");
    if !path.exists() {
        bail!(
            "missing {}: run from the repository root, or set AGENTOS_CORE_HOME to a checkout that contains config/server.toml (current repo_home: {})",
            path.display(),
            repo_home.display()
        );
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rag_repo_roots_uses_split_paths() {
        let dir1 = tempfile::tempdir().expect("tempdir 1");
        let dir2 = tempfile::tempdir().expect("tempdir 2");
        let joined = std::env::join_paths([dir1.path(), dir2.path()]).expect("join_paths");
        let parsed = parse_rag_repo_roots(&joined);
        assert_eq!(
            parsed,
            vec![dir1.path().to_path_buf(), dir2.path().to_path_buf()]
        );
    }
}
