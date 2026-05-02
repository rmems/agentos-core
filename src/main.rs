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

use anyhow::Result;
use clap::{Parser, Subcommand};
use install::{ClientTarget, InstallContext};
use rmcp::ServiceExt;
use tracing_subscriber::FmtSubscriber;

use crate::config::{discover_repo_home, load_server_config};
use crate::router::ModelRouter;
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_target(false)
        .with_ansi(false)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);

    let cli = Cli::parse();
    let repo_home = discover_repo_home()?;
    let config = load_server_config(&repo_home)?;
    let install_ctx = InstallContext {
        repo_home: repo_home.clone(),
    };

    match cli.command.unwrap_or(Commands::Serve) {
        Commands::Serve => {
            let server = DiscoveryServer::new(config);
            server
                .serve(rmcp::transport::stdio())
                .await?
                .waiting()
                .await?;
        }
        Commands::ServeHttp { bind } => {
            http::serve(&bind).await?;
        }
        Commands::Doctor => {
            println!("{}", install::doctor(&install_ctx)?);
        }
        Commands::PrintConfig { client } => {
            println!("{}", install::print_config(&install_ctx, client)?);
        }
        Commands::Route { provider } => {
            if let Some(p) = provider {
                println!("provider={}", p);
                // Validate the provider selection
                if let Some(router) = ModelRouter::from_env().ok() {
                    println!("model={}", router.name());
                }
            } else {
                println!("provider=ollama (default)");
            }
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
    }

    Ok(())
}
