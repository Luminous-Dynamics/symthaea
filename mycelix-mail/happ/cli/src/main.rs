// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::Result;
use clap::{Parser, Subcommand};

mod client;
mod commands;
mod config;
mod types;

use commands::*;

/// Mycelix Mail - Decentralized email with trust-based spam filtering
///
/// Built on Holochain DHT with DID addressing and MATL trust scores.
/// Zero fees, true privacy, censorship resistant.
#[derive(Parser, Debug)]
#[command(name = "mycelix-mail")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Holochain conductor URL
    #[arg(
        short,
        long,
        env = "HOLOCHAIN_URL",
        default_value = "ws://localhost:8888"
    )]
    conductor: String,

    /// DID registry URL
    #[arg(
        long,
        env = "DID_REGISTRY_URL",
        default_value = "http://localhost:5000"
    )]
    did_registry: String,

    /// MATL bridge URL
    #[arg(long, env = "MATL_BRIDGE_URL", default_value = "http://localhost:5001")]
    matl_bridge: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Initialize Mycelix Mail (setup keys and configuration)
    Init {
        /// Your email address (DID will be generated)
        #[arg(short, long)]
        email: Option<String>,

        /// Import existing keys
        #[arg(long)]
        import_keys: Option<String>,
    },

    /// Send an email message
    Send {
        /// Recipient DID or email address
        to: String,

        /// Email subject
        #[arg(short, long)]
        subject: String,

        /// Message body (or use stdin with -)
        #[arg(short, long)]
        body: Option<String>,

        /// Path to attachment file
        #[arg(short, long)]
        attach: Option<Vec<String>>,

        /// Reply to message ID
        #[arg(long)]
        reply_to: Option<String>,

        /// Epistemic tier (0-4)
        #[arg(long, default_value = "2")]
        tier: u8,
    },

    /// List inbox messages
    Inbox {
        /// Filter by sender DID
        #[arg(short, long)]
        from: Option<String>,

        /// Filter by minimum trust score
        #[arg(short, long)]
        trust_min: Option<f64>,

        /// Show only unread messages
        #[arg(short, long)]
        unread: bool,

        /// Number of messages to display
        #[arg(short, long, default_value = "20")]
        limit: usize,

        /// Output format (table, json, raw)
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Read a specific message
    Read {
        /// Message ID
        message_id: String,

        /// Mark as read
        #[arg(long, default_value = "true")]
        mark_read: bool,
    },

    /// Manage trust scores
    Trust {
        #[command(subcommand)]
        command: TrustCommands,
    },

    /// Manage DIDs and contacts
    Did {
        #[command(subcommand)]
        command: DidCommands,
    },

    /// Search messages
    Search {
        /// Search query
        query: String,

        /// Search in (subject, body, from, all)
        #[arg(short, long, default_value = "all")]
        in_field: String,

        /// Maximum results
        #[arg(short, long, default_value = "50")]
        limit: usize,

        /// Output format (table, json, raw)
        #[arg(short = 'f', long, default_value = "table")]
        format: String,
    },

    /// Export messages or data
    Export {
        /// Export format (json, mbox, csv)
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Output file
        #[arg(short, long)]
        output: String,

        /// Filter by date range
        #[arg(long)]
        since: Option<String>,
    },

    /// Show configuration and status
    Status {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Update local cache from DHT
    Sync {
        /// Force full sync
        #[arg(short, long)]
        force: bool,
    },
}

#[derive(Subcommand, Debug)]
enum TrustCommands {
    /// Get trust score for a DID
    Get {
        /// DID to query
        did: String,
    },

    /// Set trust score for a DID
    Set {
        /// DID to set score for
        did: String,

        /// Trust score (0.0 - 1.0)
        score: f64,
    },

    /// List all trust scores
    List {
        /// Minimum score to display
        #[arg(short, long)]
        min: Option<f64>,

        /// Sort by score
        #[arg(short, long)]
        sort: bool,
    },

    /// Sync trust scores from MATL
    Sync {
        /// Specific DID to sync
        did: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum DidCommands {
    /// Register a new DID
    Register {
        /// DID to register
        did: String,

        /// Agent public key
        #[arg(long)]
        agent_key: String,
    },

    /// Resolve a DID to agent key
    Resolve {
        /// DID to resolve
        did: String,
    },

    /// List all known DIDs
    List {
        /// Filter by pattern
        #[arg(short, long)]
        filter: Option<String>,
    },

    /// Show your current DID
    Whoami,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt().with_env_filter(log_level).init();

    // Handle Init command early (before loading config or creating client)
    if let Commands::Init { email, import_keys } = &cli.command {
        return init::handle_init(
            email.clone(),
            import_keys.clone(),
            &cli.conductor,
            &cli.did_registry,
            &cli.matl_bridge,
        )
        .await;
    }

    // Load configuration (required for all other commands)
    let config = config::Config::load_or_create()?;

    // Create client
    let client =
        client::MycellixClient::new(&cli.conductor, &cli.did_registry, &cli.matl_bridge, config)
            .await?;

    // Execute command
    match cli.command {
        Commands::Init { .. } => {
            // Already handled above
            unreachable!()
        }

        Commands::Send {
            to,
            subject,
            body,
            attach,
            reply_to,
            tier,
        } => {
            send::handle_send(&client, to, subject, body, attach, reply_to, tier).await?;
        }

        Commands::Inbox {
            from,
            trust_min,
            unread,
            limit,
            format,
        } => {
            inbox::handle_inbox(&client, from, trust_min, unread, limit, &format).await?;
        }

        Commands::Read {
            message_id,
            mark_read,
        } => {
            read::handle_read(&client, message_id, mark_read).await?;
        }

        Commands::Trust { command } => match command {
            TrustCommands::Get { did } => {
                trust::handle_get(&client, did).await?;
            }
            TrustCommands::Set { did, score } => {
                trust::handle_set(&client, did, score).await?;
            }
            TrustCommands::List { min, sort } => {
                trust::handle_list(&client, min, sort).await?;
            }
            TrustCommands::Sync { did } => {
                trust::handle_sync(&client, did).await?;
            }
        },

        Commands::Did { command } => match command {
            DidCommands::Register { did, agent_key } => {
                did::handle_register(&client, did, agent_key).await?;
            }
            DidCommands::Resolve { did } => {
                did::handle_resolve(&client, did).await?;
            }
            DidCommands::List { filter } => {
                did::handle_list(&client, filter).await?;
            }
            DidCommands::Whoami => {
                did::handle_whoami(&client).await?;
            }
        },

        Commands::Search {
            query,
            in_field,
            limit,
            format,
        } => {
            search::handle_search(&client, query, &in_field, limit, &format).await?;
        }

        Commands::Export {
            format,
            output,
            since,
        } => {
            export::handle_export(&client, &format, output, since).await?;
        }

        Commands::Status { detailed } => {
            status::handle_status(&client, detailed).await?;
        }

        Commands::Sync { force } => {
            sync::handle_sync(&client, force).await?;
        }
    }

    Ok(())
}
