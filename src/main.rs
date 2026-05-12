mod config;
mod http;
mod install;
mod orchestrator;
mod rag;
mod repo;
mod router;
mod schema;
mod server;
mod session;
mod tools;

use anyhow::{Result, anyhow, bail};
use clap::{Parser, Subcommand};
use install::{ClientTarget, InstallContext};
use rmcp::ServiceExt;
use tracing_subscriber::FmtSubscriber;

use crate::config::{discover_repo_home, load_server_config};
use crate::router::{ModelProvider, ModelRouter, ProviderRole};
use crate::server::DiscoveryServer;

#[derive(Debug, Parser)]
#[command(
    name = "agentos-core",
    version,
    about = "Local MCP server for AgentOS context and bounded repo analysis"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Serve,
    ServeHttp {
        #[arg(long, default_value = "127.0.0.1:8765")]
        bind: String,
    },
    Doctor,
    PrintConfig {
        client: ClientTarget,
    },
    Route {
        #[arg(long, default_value_t = false)]
        env: bool,
        #[arg(long)]
        provider: Option<String>,
    },
    Install {
        #[arg(long, value_delimiter = ',')]
        clients: Vec<ClientTarget>,
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    Uninstall {
        #[arg(long, value_delimiter = ',')]
        clients: Vec<ClientTarget>,
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    Index,
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_target(false)
        .with_ansi(false)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);

    let cli = Cli::parse();
    match cli.command.unwrap_or(Commands::Serve) {
        Commands::ServeHttp { bind } => {
            http::serve(&bind).await?;
        }
        Commands::Route { env, provider } => {
            let built_in = ModelRouter::built_in_defaults();
            let config_router = ModelRouter::load_system_config_if_present()?;
            let has_role_env = ProviderRole::all().iter().any(|role| {
                std::env::var(role.to_env_key())
                    .ok()
                    .is_some_and(|value| !value.trim().is_empty())
            });

            let overrides = if env || has_role_env {
                ModelRouter::from_env()?
            } else {
                ModelRouter { roles: vec![] }
            };

            let provider_filter = match provider {
                Some(provider_name) => Some(
                    ModelProvider::from_str(&provider_name)
                        .ok_or_else(|| anyhow!("unknown provider '{provider_name}'"))?,
                ),
                None => None,
            };

            let mut matched_any = false;

            println!("# Role-based providers:");
            for role in ProviderRole::all() {
                let override_p = overrides.get_provider(role);
                let config_p = config_router.as_ref().and_then(|r| r.get_provider(role));
                let built_in_p = built_in.get_provider(role);
                let chosen = override_p.or(config_p).or(built_in_p);
                let Some(chosen) = chosen else {
                    continue;
                };

                if let Some(provider_filter) = provider_filter {
                    if chosen.provider != provider_filter {
                        continue;
                    }
                }

                matched_any = true;

                let status = if override_p.is_some() {
                    "(env)"
                } else if config_p.is_some() {
                    "(config)"
                } else if built_in_p.is_some() {
                    "(built-in)"
                } else {
                    "(not set)"
                };

                println!("{:?}: {} {}", role, chosen.provider.name(), status);
            }

            if !matched_any {
                match provider_filter {
                    Some(filter) => bail!("no routes found for provider '{}'", filter.name()),
                    None => println!("No roles are configured."),
                }
            }

            if !env {
                println!("\n# Environment variables:");
                for role in ProviderRole::all() {
                    println!("- {} (provider name)", role.to_env_key());
                    println!("- {}_MODEL (model name)", role.to_env_key());
                }
            }
        }
        Commands::Index => {
            let rag = crate::rag::load_rag_config()?;
            let db = crate::orchestrator::resolved_vector_db_config();
            println!("Indexing repositories into Qdrant...");
            let report = crate::rag::index_default_repos(&rag, &db).await?;
            println!("Indexing complete!");
            println!("Indexed roots: {:?}", report.indexed_roots);
            println!("Skipped roots: {:?}", report.skipped_roots);
            println!("Chunks indexed: {}", report.chunks_indexed);
        }
        cmd => {
            let repo_home = discover_repo_home()?;
            let config = load_server_config(&repo_home)?;
            let install_ctx = InstallContext {
                repo_home: repo_home.clone(),
            };

            match cmd {
                Commands::Serve => {
                    let server = DiscoveryServer::new(config);
                    server
                        .serve(rmcp::transport::stdio())
                        .await?
                        .waiting()
                        .await?;
                }
                Commands::Doctor => {
                    println!("{}", install::doctor(&install_ctx)?);
                }
                Commands::PrintConfig { client } => {
                    println!("{}", install::print_config(&install_ctx, client)?);
                }
                Commands::Install { clients, dry_run } => {
                    for line in install::install(&install_ctx, &clients, dry_run)? {
                        println!("{line}");
                    }
                }
                Commands::Uninstall { clients, dry_run } => {
                    for line in install::uninstall(&install_ctx, &clients, dry_run)? {
                        println!("{line}");
                    }
                }
                Commands::ServeHttp { .. } | Commands::Route { .. } | Commands::Index => {
                    unreachable!()
                }
            }
        }
    }

    Ok(())
}
