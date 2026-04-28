use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use dirs::home_dir;
use serde_json::{Map, Value, json};

const SERVER_KEY: &str = "saaq-discovery";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum ClientTarget {
    Codex,
    Cursor,
    Vscode,
    Windsurf,
    Trae,
    Antigravity,
    Jetbrains,
}

#[derive(Debug, Clone)]
pub struct InstallContext {
    pub repo_home: PathBuf,
}

impl InstallContext {
    pub fn launcher_path(&self) -> PathBuf {
        self.repo_home.join("scripts/launch.sh")
    }

    pub fn codex_add_args(&self) -> Vec<String> {
        vec![
            "mcp".to_string(),
            "add".to_string(),
            SERVER_KEY.to_string(),
            "--".to_string(),
            self.launcher_path().display().to_string(),
        ]
    }
}

pub fn all_targets() -> Vec<ClientTarget> {
    vec![
        ClientTarget::Codex,
        ClientTarget::Cursor,
        ClientTarget::Vscode,
        ClientTarget::Windsurf,
        ClientTarget::Trae,
        ClientTarget::Antigravity,
        ClientTarget::Jetbrains,
    ]
}

pub fn install(
    ctx: &InstallContext,
    targets: &[ClientTarget],
    dry_run: bool,
) -> Result<Vec<String>> {
    let selected = expand_targets(targets);
    let mut changes = Vec::new();
    for target in selected {
        match target {
            ClientTarget::Codex => {
                let args = ctx.codex_add_args();
                if dry_run {
                    changes.push(format!("codex {}", args.join(" ")));
                } else {
                    let status = Command::new("codex")
                        .args(&args)
                        .status()
                        .context("failed to run codex mcp add")?;
                    if !status.success() {
                        bail!("codex mcp add failed with status {status}");
                    }
                    changes.push("updated Codex via codex mcp add".to_string());
                }
            }
            ClientTarget::Cursor => {
                let path = home_path(".cursor/mcp.json")?;
                let entry = stdio_entry_json(&ctx.launcher_path());
                upsert_json_server(&path, "mcpServers", SERVER_KEY, entry, dry_run)?;
                changes.push(path.display().to_string());
            }
            ClientTarget::Vscode => {
                let path = home_path(".config/Code/User/mcp.json")?;
                let entry = vscode_stdio_entry_json(&ctx.launcher_path());
                upsert_json_server(&path, "servers", SERVER_KEY, entry, dry_run)?;
                changes.push(path.display().to_string());
            }
            ClientTarget::Windsurf => {
                let path = home_path(".codeium/windsurf/mcp_config.json")?;
                let entry = stdio_entry_json(&ctx.launcher_path());
                upsert_json_server(&path, "mcpServers", SERVER_KEY, entry, dry_run)?;
                changes.push(path.display().to_string());
            }
            ClientTarget::Trae => {
                let path = home_path(".config/Trae/User/mcp.json")?;
                let entry = stdio_entry_json(&ctx.launcher_path());
                upsert_json_server(&path, "mcpServers", SERVER_KEY, entry, dry_run)?;
                changes.push(path.display().to_string());
            }
            ClientTarget::Antigravity => {
                let path = home_path(".antigravity/settings.json")?;
                let entry = stdio_entry_json(&ctx.launcher_path());
                upsert_json_server(&path, "mcpServers", SERVER_KEY, entry, dry_run)?;
                changes.push(path.display().to_string());
            }
            ClientTarget::Jetbrains => {
                let junie_path = home_path(".junie/mcp/mcp.json")?;
                let snippet_path = home_path(".jetbrains/saaq-discovery.mcp.json")?;
                let entry = stdio_entry_json(&ctx.launcher_path());
                upsert_json_server(
                    &junie_path,
                    "mcpServers",
                    SERVER_KEY,
                    entry.clone(),
                    dry_run,
                )?;
                write_json_value(
                    &snippet_path,
                    json!({ "mcpServers": { SERVER_KEY: entry } }),
                    dry_run,
                )?;
                changes.push(junie_path.display().to_string());
                changes.push(snippet_path.display().to_string());
            }
        }
    }
    Ok(changes)
}

pub fn uninstall(
    ctx: &InstallContext,
    targets: &[ClientTarget],
    dry_run: bool,
) -> Result<Vec<String>> {
    let selected = expand_targets(targets);
    let mut changes = Vec::new();
    for target in selected {
        match target {
            ClientTarget::Codex => {
                let args = ["mcp", "remove", SERVER_KEY];
                if dry_run {
                    changes.push(format!("codex {}", args.join(" ")));
                } else {
                    let status = Command::new("codex")
                        .args(args)
                        .status()
                        .context("failed to run codex mcp remove")?;
                    if !status.success() {
                        bail!("codex mcp remove failed with status {status}");
                    }
                    changes.push("removed Codex config".to_string());
                }
            }
            ClientTarget::Cursor => {
                let path = home_path(".cursor/mcp.json")?;
                remove_json_server(&path, "mcpServers", SERVER_KEY, dry_run)?;
                changes.push(path.display().to_string());
            }
            ClientTarget::Vscode => {
                let path = home_path(".config/Code/User/mcp.json")?;
                remove_json_server(&path, "servers", SERVER_KEY, dry_run)?;
                changes.push(path.display().to_string());
            }
            ClientTarget::Windsurf => {
                let path = home_path(".codeium/windsurf/mcp_config.json")?;
                remove_json_server(&path, "mcpServers", SERVER_KEY, dry_run)?;
                changes.push(path.display().to_string());
            }
            ClientTarget::Trae => {
                let path = home_path(".config/Trae/User/mcp.json")?;
                remove_json_server(&path, "mcpServers", SERVER_KEY, dry_run)?;
                changes.push(path.display().to_string());
            }
            ClientTarget::Antigravity => {
                let path = home_path(".antigravity/settings.json")?;
                remove_json_server(&path, "mcpServers", SERVER_KEY, dry_run)?;
                changes.push(path.display().to_string());
            }
            ClientTarget::Jetbrains => {
                let junie_path = home_path(".junie/mcp/mcp.json")?;
                let snippet_path = home_path(".jetbrains/saaq-discovery.mcp.json")?;
                remove_json_server(&junie_path, "mcpServers", SERVER_KEY, dry_run)?;
                if !dry_run && snippet_path.exists() {
                    std::fs::remove_file(&snippet_path)
                        .with_context(|| format!("failed to remove {}", snippet_path.display()))?;
                }
                changes.push(junie_path.display().to_string());
                changes.push(snippet_path.display().to_string());
            }
        }
    }
    let _ = ctx;
    Ok(changes)
}

pub fn print_config(ctx: &InstallContext, target: ClientTarget) -> Result<String> {
    let launcher = ctx.launcher_path();
    Ok(match target {
        ClientTarget::Codex => format!(
            "codex {}\n\n[mcp_servers.{SERVER_KEY}]\ncommand = \"{}\"",
            ctx.codex_add_args().join(" "),
            launcher.display()
        ),
        ClientTarget::Cursor
        | ClientTarget::Windsurf
        | ClientTarget::Trae
        | ClientTarget::Antigravity
        | ClientTarget::Jetbrains => serde_json::to_string_pretty(&json!({
            "mcpServers": {
                SERVER_KEY: stdio_entry_json(&launcher)
            }
        }))?,
        ClientTarget::Vscode => serde_json::to_string_pretty(&json!({
            "servers": {
                SERVER_KEY: vscode_stdio_entry_json(&launcher)
            }
        }))?,
    })
}

pub fn doctor(ctx: &InstallContext) -> Result<String> {
    let mut lines = Vec::new();
    lines.push(format!("repo_home={}", ctx.repo_home.display()));
    lines.push(format!("launcher={}", ctx.launcher_path().display()));

    let checks = [
        ("cursor", home_path(".cursor/mcp.json")?),
        ("vscode", home_path(".config/Code/User/mcp.json")?),
        ("windsurf", home_path(".codeium/windsurf/mcp_config.json")?),
        ("trae", home_path(".config/Trae/User/mcp.json")?),
        ("antigravity", home_path(".antigravity/settings.json")?),
        ("jetbrains-junie", home_path(".junie/mcp/mcp.json")?),
        (
            "jetbrains-snippet",
            home_path(".jetbrains/saaq-discovery.mcp.json")?,
        ),
    ];

    for (name, path) in checks {
        lines.push(format!(
            "{name}: {} ({})",
            if path.exists() { "present" } else { "missing" },
            path.display()
        ));
    }

    lines.push(format!(
        "codex-installed={}",
        codex_has_server().unwrap_or(false)
    ));
    Ok(lines.join("\n"))
}

fn expand_targets(targets: &[ClientTarget]) -> Vec<ClientTarget> {
    if targets.is_empty() {
        return all_targets();
    }

    let mut unique = BTreeSet::new();
    unique.extend(targets.iter().copied());
    unique.into_iter().collect()
}

fn home_path(relative: &str) -> Result<PathBuf> {
    let home = home_dir().context("failed to resolve home directory")?;
    Ok(home.join(relative))
}

fn stdio_entry_json(launcher: &Path) -> Value {
    json!({
        "command": launcher.display().to_string(),
        "args": []
    })
}

fn vscode_stdio_entry_json(launcher: &Path) -> Value {
    json!({
        "type": "stdio",
        "command": launcher.display().to_string(),
        "args": []
    })
}

fn upsert_json_server(
    path: &Path,
    root_key: &str,
    server_name: &str,
    server_value: Value,
    dry_run: bool,
) -> Result<()> {
    let mut root = read_json_object(path)?;
    let mut servers = root
        .remove(root_key)
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    servers.insert(server_name.to_string(), server_value);
    root.insert(root_key.to_string(), Value::Object(servers));
    write_json_value(path, Value::Object(root), dry_run)
}

fn remove_json_server(path: &Path, root_key: &str, server_name: &str, dry_run: bool) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }

    let mut root = read_json_object(path)?;
    if let Some(value) = root.get_mut(root_key) {
        if let Some(servers) = value.as_object_mut() {
            servers.remove(server_name);
        }
    }
    write_json_value(path, Value::Object(root), dry_run)
}

fn read_json_object(path: &Path) -> Result<Map<String, Value>> {
    if !path.exists() {
        return Ok(Map::new());
    }

    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    if raw.trim().is_empty() {
        return Ok(Map::new());
    }
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(value.as_object().cloned().unwrap_or_default())
}

fn write_json_value(path: &Path, value: Value, dry_run: bool) -> Result<()> {
    if dry_run {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let rendered = serde_json::to_string_pretty(&value)?;
    std::fs::write(path, format!("{rendered}\n"))
        .with_context(|| format!("failed to write {}", path.display()))
}

fn codex_has_server() -> Result<bool> {
    let output = Command::new("codex")
        .args(["mcp", "list"])
        .output()
        .context("failed to run codex mcp list")?;
    Ok(String::from_utf8_lossy(&output.stdout).contains(SERVER_KEY))
}
