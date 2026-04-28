use std::collections::{BTreeSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{Duration, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use regex::{Regex, RegexBuilder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio::time;
use walkdir::WalkDir;

static PUBLIC_ITEM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^\s*pub(?:\([^)]*\))?\s+(struct|enum|type|trait|fn|const)\s+([A-Za-z_][A-Za-z0-9_]*)",
    )
    .expect("valid public item regex")
});
static MOD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;").expect("valid mod regex")
});
static USE_CRATE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*use\s+crate::([A-Za-z0-9_:]+)").expect("valid crate use regex")
});

#[derive(Debug, Clone)]
pub struct RepoSnapshot {
    pub repo_name: String,
    pub root: PathBuf,
    pub snapshot_id: String,
    pub indexed_at: DateTime<Utc>,
    pub files: Vec<RepoFile>,
    pub public_symbols: Vec<PublicSymbol>,
    pub module_edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone)]
pub struct RepoFile {
    pub relative_path: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PublicSymbol {
    pub name: String,
    pub kind: String,
    pub module: String,
    pub relative_path: String,
    pub line: usize,
    pub declaration: String,
    pub docs: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchHit {
    pub relative_path: String,
    pub line: usize,
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RepoSummary {
    pub repo_name: String,
    pub repo_root: String,
    pub snapshot_id: String,
    pub title: Option<String>,
    pub overview: String,
    pub architecture_excerpt: Vec<String>,
    pub key_modules: Vec<String>,
    pub public_api_highlights: Vec<String>,
    pub doc_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModuleGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TermResolution {
    pub term: String,
    pub ambiguous: bool,
    pub ambiguity_reason: Option<String>,
    pub surface_forms: Vec<String>,
    pub public_symbol_matches: Vec<PublicSymbol>,
    pub hits: Vec<SearchHit>,
    pub related_symbols: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TerminologyCheck {
    pub term: String,
    pub found: bool,
    pub surface_forms: Vec<String>,
    pub warnings: Vec<String>,
    pub sample_locations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TerminologyReport {
    pub snapshot_id: String,
    pub checks: Vec<TerminologyCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InvariantCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InvariantReport {
    pub snapshot_id: String,
    pub checks: Vec<InvariantCheck>,
    pub terminology: TerminologyReport,
    pub public_api_count: usize,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SearchRequest {
    pub query: String,
    pub max_results: usize,
    pub path_substring: Option<String>,
    pub case_sensitive: bool,
    pub regex: bool,
}

pub fn scan_repo(root: &Path) -> Result<RepoSnapshot> {
    if !root.exists() {
        bail!("repo root does not exist: {}", root.display());
    }
    if !root.is_dir() {
        bail!("repo root is not a directory: {}", root.display());
    }

    let mut files = Vec::new();
    let mut public_symbols = Vec::new();
    let mut module_edges = Vec::new();
    let mut hasher = DefaultHasher::new();

    for entry in WalkDir::new(root) {
        let entry = entry.with_context(|| format!("failed to walk {}", root.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let relative = path
            .strip_prefix(root)
            .with_context(|| format!("failed to relativize {}", path.display()))?;
        if !should_index(relative) {
            continue;
        }

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let relative_path = relative.to_string_lossy().replace('\\', "/");

        relative_path.hash(&mut hasher);
        content.len().hash(&mut hasher);
        if let Ok(metadata) = entry.metadata() {
            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = modified.duration_since(UNIX_EPOCH) {
                    duration.as_secs().hash(&mut hasher);
                    duration.subsec_nanos().hash(&mut hasher);
                }
            }
        }

        if relative_path.ends_with(".rs") {
            public_symbols.extend(extract_public_symbols(&relative_path, &content));
            module_edges.extend(extract_module_edges(&relative_path, &content));
        }

        files.push(RepoFile {
            relative_path,
            content,
        });
    }

    let repo_name = root
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "repo".to_string());
    let snapshot_id = format!("{:016x}", hasher.finish());

    Ok(RepoSnapshot {
        repo_name,
        root: root.to_path_buf(),
        snapshot_id,
        indexed_at: Utc::now(),
        files,
        public_symbols,
        module_edges,
    })
}

pub fn build_summary(snapshot: &RepoSnapshot) -> RepoSummary {
    let readme = snapshot
        .files
        .iter()
        .find(|file| file.relative_path.eq_ignore_ascii_case("README.md"))
        .map(|file| file.content.as_str());

    let title = readme.and_then(extract_markdown_title);
    let overview = readme
        .and_then(|content| extract_markdown_section(content, "Overview"))
        .unwrap_or_else(|| {
            title
                .as_ref()
                .map(|heading| format!("{heading} repository indexed for canon discovery."))
                .unwrap_or_else(|| "Repository indexed for canon discovery.".to_string())
        });
    let architecture_excerpt = readme
        .and_then(|content| extract_markdown_section_lines(content, "Architecture", 8))
        .unwrap_or_default();

    let mut key_modules = snapshot
        .public_symbols
        .iter()
        .map(|symbol| symbol.module.clone())
        .filter(|module| module != "crate")
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if key_modules.is_empty() {
        key_modules.push("crate".to_string());
    }

    let public_api_highlights = snapshot
        .public_symbols
        .iter()
        .take(12)
        .map(|symbol| format!("{} {} ({})", symbol.kind, symbol.name, symbol.module))
        .collect::<Vec<_>>();

    let doc_paths = snapshot
        .files
        .iter()
        .filter(|file| file.relative_path.ends_with(".md"))
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();

    RepoSummary {
        repo_name: snapshot.repo_name.clone(),
        repo_root: snapshot.root.display().to_string(),
        snapshot_id: snapshot.snapshot_id.clone(),
        title,
        overview,
        architecture_excerpt,
        key_modules,
        public_api_highlights,
        doc_paths,
    }
}

pub fn build_module_graph(snapshot: &RepoSnapshot) -> ModuleGraph {
    let mut nodes = BTreeSet::new();
    for symbol in &snapshot.public_symbols {
        nodes.insert(symbol.module.clone());
    }
    for edge in &snapshot.module_edges {
        nodes.insert(edge.from.clone());
        nodes.insert(edge.to.clone());
    }

    ModuleGraph {
        nodes: nodes.into_iter().collect(),
        edges: snapshot.module_edges.clone(),
    }
}

pub fn search_snapshot(snapshot: &RepoSnapshot, request: &SearchRequest) -> Result<Vec<SearchHit>> {
    if request.query.trim().is_empty() {
        bail!("query must not be empty");
    }

    let matcher = if request.regex {
        Some(
            RegexBuilder::new(&request.query)
                .case_insensitive(!request.case_sensitive)
                .build()
                .with_context(|| format!("invalid regex: {}", request.query))?,
        )
    } else {
        None
    };

    let path_filter = request.path_substring.as_ref().map(|value| {
        if request.case_sensitive {
            value.clone()
        } else {
            value.to_lowercase()
        }
    });

    let query_cmp = if request.case_sensitive {
        request.query.clone()
    } else {
        request.query.to_lowercase()
    };

    let mut hits = Vec::new();
    for file in &snapshot.files {
        if let Some(filter) = &path_filter {
            let file_cmp = if request.case_sensitive {
                file.relative_path.clone()
            } else {
                file.relative_path.to_lowercase()
            };
            if !file_cmp.contains(filter) {
                continue;
            }
        }

        for (idx, line) in file.content.lines().enumerate() {
            let matched = if let Some(regex) = &matcher {
                regex.is_match(line)
            } else if request.case_sensitive {
                line.contains(&query_cmp)
            } else {
                line.to_lowercase().contains(&query_cmp)
            };

            if matched {
                hits.push(SearchHit {
                    relative_path: file.relative_path.clone(),
                    line: idx + 1,
                    excerpt: line.trim().to_string(),
                });
                if hits.len() >= request.max_results {
                    return Ok(hits);
                }
            }
        }
    }

    Ok(hits)
}

pub fn list_public_api(
    snapshot: &RepoSnapshot,
    module_prefix: Option<&str>,
    max_items: usize,
) -> Vec<PublicSymbol> {
    snapshot
        .public_symbols
        .iter()
        .filter(|symbol| {
            module_prefix
                .map(|prefix| symbol.module.starts_with(prefix))
                .unwrap_or(true)
        })
        .take(max_items)
        .cloned()
        .collect()
}

pub fn resolve_term(
    snapshot: &RepoSnapshot,
    term: &str,
    max_hits: usize,
) -> Result<TermResolution> {
    if term.trim().is_empty() {
        bail!("term must not be empty");
    }

    let escaped = regex::escape(term);
    let matcher = RegexBuilder::new(&format!(r"\b{escaped}\b"))
        .case_insensitive(true)
        .build()
        .context("failed to compile term matcher")?;

    let mut surface_forms = BTreeSet::new();
    let mut hits = Vec::new();
    for file in &snapshot.files {
        for (idx, line) in file.content.lines().enumerate() {
            let mut line_matched = false;
            for capture in matcher.find_iter(line) {
                surface_forms.insert(capture.as_str().to_string());
                line_matched = true;
            }
            if line_matched {
                hits.push(SearchHit {
                    relative_path: file.relative_path.clone(),
                    line: idx + 1,
                    excerpt: line.trim().to_string(),
                });
                if hits.len() >= max_hits {
                    break;
                }
            }
        }
        if hits.len() >= max_hits {
            break;
        }
    }

    let public_symbol_matches = snapshot
        .public_symbols
        .iter()
        .filter(|symbol| symbol.name.eq_ignore_ascii_case(term))
        .cloned()
        .collect::<Vec<_>>();

    let related_symbols = snapshot
        .public_symbols
        .iter()
        .filter(|symbol| {
            symbol.name.to_lowercase().contains(&term.to_lowercase())
                && !symbol.name.eq_ignore_ascii_case(term)
        })
        .map(|symbol| symbol.name.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let ambiguous_reason = if surface_forms.len() > 1 {
        Some("multiple surface forms found for the same normalized term".to_string())
    } else if public_symbol_matches.len() > 1 {
        Some("multiple public symbols matched the same term".to_string())
    } else {
        None
    };

    Ok(TermResolution {
        term: term.to_string(),
        ambiguous: ambiguous_reason.is_some(),
        ambiguity_reason: ambiguous_reason,
        surface_forms: surface_forms.into_iter().collect(),
        public_symbol_matches,
        hits,
        related_symbols,
    })
}

pub fn validate_terminology(
    snapshot: &RepoSnapshot,
    terms: &[String],
) -> Result<TerminologyReport> {
    let mut checks = Vec::new();
    for term in terms {
        let resolution = resolve_term(snapshot, term, 8)?;
        let mut warnings = Vec::new();
        if resolution.hits.is_empty() && resolution.public_symbol_matches.is_empty() {
            warnings.push("term was not found in the indexed canon".to_string());
        }
        if resolution.surface_forms.len() > 1 {
            warnings.push("term appears with multiple surface forms".to_string());
        }
        if resolution.public_symbol_matches.len() > 1 {
            warnings.push("term resolves to multiple public symbols".to_string());
        }

        checks.push(TerminologyCheck {
            term: term.clone(),
            found: !resolution.hits.is_empty() || !resolution.public_symbol_matches.is_empty(),
            surface_forms: resolution.surface_forms.clone(),
            warnings,
            sample_locations: resolution
                .hits
                .iter()
                .take(3)
                .map(|hit| format!("{}:{}", hit.relative_path, hit.line))
                .collect(),
        });
    }

    Ok(TerminologyReport {
        snapshot_id: snapshot.snapshot_id.clone(),
        checks,
    })
}

pub async fn run_invariants(
    snapshot: &RepoSnapshot,
    terms: &[String],
    timeout_seconds: u64,
) -> Result<InvariantReport> {
    let mut checks = Vec::new();
    let cargo_manifest = snapshot.root.join("Cargo.toml");
    let readme = snapshot.root.join("README.md");

    checks.push(InvariantCheck {
        name: "cargo-manifest-present".to_string(),
        passed: cargo_manifest.exists(),
        detail: cargo_manifest.display().to_string(),
    });
    checks.push(InvariantCheck {
        name: "readme-present".to_string(),
        passed: readme.exists(),
        detail: readme.display().to_string(),
    });

    if cargo_manifest.exists() {
        let output = time::timeout(
            Duration::from_secs(timeout_seconds.max(1)),
            Command::new("cargo")
                .arg("test")
                .arg("-q")
                .current_dir(&snapshot.root)
                .output(),
        )
        .await;

        match output {
            Ok(Ok(result)) => {
                let mut combined = String::new();
                if !result.stdout.is_empty() {
                    combined.push_str(&String::from_utf8_lossy(&result.stdout));
                }
                if !result.stderr.is_empty() {
                    if !combined.is_empty() {
                        combined.push('\n');
                    }
                    combined.push_str(&String::from_utf8_lossy(&result.stderr));
                }
                checks.push(InvariantCheck {
                    name: "cargo-test".to_string(),
                    passed: result.status.success(),
                    detail: trim_output(&combined),
                });
            }
            Ok(Err(error)) => checks.push(InvariantCheck {
                name: "cargo-test".to_string(),
                passed: false,
                detail: error.to_string(),
            }),
            Err(_) => checks.push(InvariantCheck {
                name: "cargo-test".to_string(),
                passed: false,
                detail: format!("timed out after {}s", timeout_seconds.max(1)),
            }),
        }
    }

    let terminology = validate_terminology(snapshot, terms)?;
    Ok(InvariantReport {
        snapshot_id: snapshot.snapshot_id.clone(),
        checks,
        terminology,
        public_api_count: snapshot.public_symbols.len(),
    })
}

pub fn terminology_markdown(snapshot: &RepoSnapshot, terms: &[String]) -> Result<String> {
    let report = validate_terminology(snapshot, terms)?;
    let mut lines = vec!["# Terminology".to_string(), String::new()];
    for check in report.checks {
        lines.push(format!("## {}", check.term));
        lines.push(format!("- found: {}", check.found));
        if !check.surface_forms.is_empty() {
            lines.push(format!(
                "- surface forms: {}",
                check.surface_forms.join(", ")
            ));
        }
        if !check.warnings.is_empty() {
            lines.push(format!("- warnings: {}", check.warnings.join("; ")));
        }
        if !check.sample_locations.is_empty() {
            lines.push(format!("- samples: {}", check.sample_locations.join(", ")));
        }
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

pub fn public_api_markdown(symbols: &[PublicSymbol]) -> String {
    let mut lines = vec!["# Public API".to_string(), String::new()];
    for symbol in symbols {
        lines.push(format!(
            "- `{}` `{}` in `{}` at {}:{}",
            symbol.kind, symbol.name, symbol.module, symbol.relative_path, symbol.line
        ));
        if let Some(docs) = &symbol.docs {
            lines.push(format!("  {}", docs));
        }
    }
    lines.join("\n")
}

pub fn module_graph_markdown(graph: &ModuleGraph) -> String {
    let mut lines = vec!["# Module Graph".to_string(), String::new()];
    lines.push("## Nodes".to_string());
    lines.extend(graph.nodes.iter().map(|node| format!("- `{node}`")));
    lines.push(String::new());
    lines.push("## Edges".to_string());
    lines.extend(
        graph
            .edges
            .iter()
            .map(|edge| format!("- `{}` --{}--> `{}`", edge.from, edge.kind, edge.to)),
    );
    lines.join("\n")
}

fn should_index(relative: &Path) -> bool {
    let rel = relative.to_string_lossy().replace('\\', "/");
    if rel == "README.md" || rel == "AGENTS.md" || rel == "Cargo.toml" {
        return true;
    }
    rel.starts_with("src/") && rel.ends_with(".rs")
        || rel.starts_with("docs/") && rel.ends_with(".md")
}

fn extract_public_symbols(relative_path: &str, content: &str) -> Vec<PublicSymbol> {
    let module = module_name_from_path(relative_path);
    let mut results = Vec::new();
    let mut pending_docs = Vec::new();

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("///") {
            pending_docs.push(trimmed.trim_start_matches("///").trim().to_string());
            continue;
        }
        if trimmed.starts_with("//!") {
            pending_docs.push(trimmed.trim_start_matches("//!").trim().to_string());
            continue;
        }

        if let Some(captures) = PUBLIC_ITEM_RE.captures(line) {
            let kind = captures
                .get(1)
                .map(|value| value.as_str())
                .unwrap_or_default();
            let name = captures
                .get(2)
                .map(|value| value.as_str())
                .unwrap_or_default();
            let docs = if pending_docs.is_empty() {
                None
            } else {
                Some(pending_docs.join(" "))
            };

            results.push(PublicSymbol {
                name: name.to_string(),
                kind: kind.to_string(),
                module: module.clone(),
                relative_path: relative_path.to_string(),
                line: idx + 1,
                declaration: trimmed.to_string(),
                docs,
            });
        }

        if !trimmed.is_empty() && !trimmed.starts_with("//") {
            pending_docs.clear();
        }
    }

    results
}

fn extract_module_edges(relative_path: &str, content: &str) -> Vec<GraphEdge> {
    let module = module_name_from_path(relative_path);
    let mut edges = Vec::new();
    for line in content.lines() {
        if let Some(captures) = MOD_RE.captures(line) {
            if let Some(child) = captures.get(1) {
                let to = if module == "crate" {
                    child.as_str().to_string()
                } else {
                    format!("{module}::{}", child.as_str())
                };
                edges.push(GraphEdge {
                    from: module.clone(),
                    to,
                    kind: "declares".to_string(),
                });
            }
        }

        if let Some(captures) = USE_CRATE_RE.captures(line) {
            if let Some(imported) = captures.get(1) {
                let to = imported.as_str().replace("::{", "::");
                edges.push(GraphEdge {
                    from: module.clone(),
                    to,
                    kind: "uses".to_string(),
                });
            }
        }
    }
    edges
}

fn module_name_from_path(relative_path: &str) -> String {
    if relative_path == "src/lib.rs" {
        return "crate".to_string();
    }
    if !relative_path.starts_with("src/") {
        return relative_path.replace('/', "::");
    }

    let module = relative_path
        .trim_start_matches("src/")
        .trim_end_matches(".rs")
        .trim_end_matches("/mod");
    if module.is_empty() {
        "crate".to_string()
    } else {
        module.replace('/', "::")
    }
}

fn extract_markdown_title(content: &str) -> Option<String> {
    content
        .lines()
        .find(|line| line.starts_with("# "))
        .map(|line| line.trim_start_matches("# ").trim().to_string())
}

fn extract_markdown_section(content: &str, heading: &str) -> Option<String> {
    let marker = format!("## {heading}");
    let mut in_section = false;
    let mut lines = Vec::new();

    for line in content.lines() {
        if line.trim() == marker {
            in_section = true;
            continue;
        }
        if in_section && line.starts_with("## ") {
            break;
        }
        if in_section {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                continue;
            }
            if !trimmed.is_empty() {
                lines.push(trimmed.to_string());
            } else if !lines.is_empty() {
                break;
            }
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(lines.join(" "))
    }
}

fn extract_markdown_section_lines(
    content: &str,
    heading: &str,
    max_lines: usize,
) -> Option<Vec<String>> {
    let marker = format!("## {heading}");
    let mut in_section = false;
    let mut lines = Vec::new();

    for line in content.lines() {
        if line.trim() == marker {
            in_section = true;
            continue;
        }
        if in_section && line.starts_with("## ") {
            break;
        }
        if in_section {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("```") {
                continue;
            }
            lines.push(trimmed.to_string());
            if lines.len() >= max_lines {
                break;
            }
        }
    }

    if lines.is_empty() { None } else { Some(lines) }
}

fn trim_output(output: &str) -> String {
    let lines = output
        .lines()
        .rev()
        .take(12)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let trimmed = lines.join("\n").trim().to_string();
    if trimmed.is_empty() {
        "no output".to_string()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn scan_repo_extracts_public_symbols_and_edges() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/hybrid")).unwrap();
        std::fs::write(
            dir.path().join("README.md"),
            "# sample\n\n## Overview\n\nA tiny repo.\n\n## Architecture\n\nOne layer.\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"sample\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "pub mod hybrid;\n/// hello\npub struct Root;\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("src/hybrid/mod.rs"),
            "use crate::Root;\npub fn run() {}\n",
        )
        .unwrap();

        let snapshot = scan_repo(dir.path()).unwrap();
        assert_eq!(snapshot.public_symbols.len(), 2);
        assert!(
            snapshot
                .module_edges
                .iter()
                .any(|edge| edge.kind == "declares")
        );
        assert!(snapshot.module_edges.iter().any(|edge| edge.kind == "uses"));

        let summary = build_summary(&snapshot);
        assert_eq!(summary.title.as_deref(), Some("sample"));
    }
}
