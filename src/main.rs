mod config;
mod install;
mod rag;
mod repo;
mod server;
mod session;
mod tools;

use anyhow::Result;
use clap::{Parser, Subcommand};
use install::{ClientTarget, InstallContext};
use rmcp::ServiceExt;
use tracing_subscriber::FmtSubscriber;

use crate::config::{discover_repo_home, load_server_config};
use crate::server::DiscoveryServer;

#[derive(Debug, Parser)]
#[command(
    name = "saaq-discovery",
    version,
    about = "Local MCP server for bounded canon discovery"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Serve,
    Doctor,
    PrintConfig {
        client: ClientTarget,
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
    }

    Ok(())
}
