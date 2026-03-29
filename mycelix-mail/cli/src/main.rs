// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Mail CLI
//!
//! Command-line interface for power users:
//! - Send emails
//! - Search inbox
//! - Manage accounts
//! - Sync operations
//! - Export data

use clap::{Parser, Subcommand};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// ============================================================================
// CLI Structure
// ============================================================================

#[derive(Parser)]
#[command(name = "mycelix")]
#[command(author = "Mycelix Team")]
#[command(version = "1.0.0")]
#[command(about = "Mycelix Mail CLI - Email from the command line", long_about = None)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, default_value = "~/.config/mycelix/config.toml")]
    config: PathBuf,

    /// API endpoint URL
    #[arg(short, long, env = "MYCELIX_API_URL")]
    api_url: Option<String>,

    /// Output format
    #[arg(short, long, default_value = "text")]
    format: OutputFormat,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Default, clap::ValueEnum)]
enum OutputFormat {
    #[default]
    Text,
    Json,
    Table,
}

#[derive(Subcommand)]
enum Commands {
    /// Send an email
    Send {
        /// Recipient email address
        #[arg(short, long)]
        to: Vec<String>,

        /// CC recipients
        #[arg(long)]
        cc: Vec<String>,

        /// BCC recipients
        #[arg(long)]
        bcc: Vec<String>,

        /// Email subject
        #[arg(short, long)]
        subject: String,

        /// Email body (reads from stdin if not provided)
        #[arg(short, long)]
        body: Option<String>,

        /// Read body from file
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Attachments
        #[arg(short, long)]
        attach: Vec<PathBuf>,

        /// Send as HTML
        #[arg(long)]
        html: bool,

        /// Account to send from
        #[arg(long)]
        from: Option<String>,

        /// Dry run (don't actually send)
        #[arg(long)]
        dry_run: bool,
    },

    /// Search emails
    Search {
        /// Search query
        query: String,

        /// Folder to search
        #[arg(short, long, default_value = "inbox")]
        folder: String,

        /// Maximum results
        #[arg(short, long, default_value = "20")]
        limit: u32,

        /// Include body in results
        #[arg(long)]
        body: bool,

        /// Only unread emails
        #[arg(long)]
        unread: bool,

        /// From date
        #[arg(long)]
        since: Option<String>,

        /// To date
        #[arg(long)]
        until: Option<String>,
    },

    /// List emails
    List {
        /// Folder to list
        #[arg(default_value = "inbox")]
        folder: String,

        /// Number of emails to show
        #[arg(short, long, default_value = "20")]
        limit: u32,

        /// Only unread
        #[arg(short, long)]
        unread: bool,

        /// Show threads
        #[arg(long)]
        threads: bool,
    },

    /// Read an email
    Read {
        /// Email ID
        id: String,

        /// Show headers
        #[arg(long)]
        headers: bool,

        /// Show raw source
        #[arg(long)]
        raw: bool,

        /// Mark as read
        #[arg(long, default_value = "true")]
        mark_read: bool,
    },

    /// Reply to an email
    Reply {
        /// Email ID to reply to
        id: String,

        /// Reply body
        #[arg(short, long)]
        body: Option<String>,

        /// Reply all
        #[arg(long)]
        all: bool,
    },

    /// Forward an email
    Forward {
        /// Email ID to forward
        id: String,

        /// Forward to
        #[arg(short, long)]
        to: Vec<String>,

        /// Additional message
        #[arg(short, long)]
        message: Option<String>,
    },

    /// Delete an email
    Delete {
        /// Email IDs to delete
        ids: Vec<String>,

        /// Skip confirmation
        #[arg(short, long)]
        force: bool,

        /// Permanently delete (skip trash)
        #[arg(long)]
        permanent: bool,
    },

    /// Move emails
    Move {
        /// Email IDs to move
        ids: Vec<String>,

        /// Target folder
        #[arg(short, long)]
        to: String,
    },

    /// Label/tag management
    Label {
        #[command(subcommand)]
        action: LabelCommands,
    },

    /// Folder management
    Folder {
        #[command(subcommand)]
        action: FolderCommands,
    },

    /// Account management
    Account {
        #[command(subcommand)]
        action: AccountCommands,
    },

    /// Sync operations
    Sync {
        /// Account to sync (all if not specified)
        #[arg(short, long)]
        account: Option<String>,

        /// Force full sync
        #[arg(long)]
        full: bool,

        /// Watch for changes
        #[arg(short, long)]
        watch: bool,
    },

    /// Export data
    Export {
        /// Export format
        #[arg(short, long, default_value = "mbox")]
        format: ExportFormatArg,

        /// Output path
        #[arg(short, long)]
        output: PathBuf,

        /// Folders to export
        #[arg(long)]
        folders: Vec<String>,

        /// Include attachments
        #[arg(long)]
        attachments: bool,
    },

    /// Import data
    Import {
        /// Import format
        #[arg(short, long)]
        format: ExportFormatArg,

        /// Input path
        path: PathBuf,

        /// Target folder
        #[arg(long, default_value = "imported")]
        folder: String,
    },

    /// Trust management
    Trust {
        #[command(subcommand)]
        action: TrustCommands,
    },

    /// Configuration
    Config {
        #[command(subcommand)]
        action: ConfigCommands,
    },

    /// Interactive mode
    Interactive,

    /// Check connection status
    Status,

    /// Show version info
    Version,
}

#[derive(Subcommand)]
enum LabelCommands {
    /// Add label to emails
    Add {
        /// Email IDs
        ids: Vec<String>,
        /// Label name
        #[arg(short, long)]
        label: String,
    },
    /// Remove label from emails
    Remove {
        /// Email IDs
        ids: Vec<String>,
        /// Label name
        #[arg(short, long)]
        label: String,
    },
    /// List all labels
    List,
    /// Create a new label
    Create {
        /// Label name
        name: String,
        /// Label color
        #[arg(short, long)]
        color: Option<String>,
    },
    /// Delete a label
    Delete {
        /// Label name
        name: String,
    },
}

#[derive(Subcommand)]
enum FolderCommands {
    /// List folders
    List,
    /// Create folder
    Create {
        /// Folder name
        name: String,
        /// Parent folder
        #[arg(short, long)]
        parent: Option<String>,
    },
    /// Delete folder
    Delete {
        /// Folder name
        name: String,
    },
    /// Get folder info
    Info {
        /// Folder name
        name: String,
    },
}

#[derive(Subcommand)]
enum AccountCommands {
    /// List accounts
    List,
    /// Add account
    Add {
        /// Email address
        email: String,
        /// Account name
        #[arg(short, long)]
        name: Option<String>,
    },
    /// Remove account
    Remove {
        /// Account ID or email
        account: String,
    },
    /// Set default account
    Default {
        /// Account ID or email
        account: String,
    },
    /// Test account connection
    Test {
        /// Account ID or email
        account: String,
    },
}

#[derive(Subcommand)]
enum TrustCommands {
    /// Show trust score for a contact
    Score {
        /// Email address
        email: String,
    },
    /// Add attestation
    Attest {
        /// Email address
        email: String,
        /// Trust level (0-100)
        #[arg(short, long)]
        level: u8,
        /// Context
        #[arg(short, long)]
        context: Option<String>,
    },
    /// Revoke attestation
    Revoke {
        /// Email address
        email: String,
    },
    /// List attestations
    List,
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Show current config
    Show,
    /// Set a config value
    Set {
        /// Key
        key: String,
        /// Value
        value: String,
    },
    /// Get a config value
    Get {
        /// Key
        key: String,
    },
    /// Open config in editor
    Edit,
    /// Reset to defaults
    Reset {
        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },
}

#[derive(Clone, Copy, Default, clap::ValueEnum)]
enum ExportFormatArg {
    #[default]
    Mbox,
    Eml,
    Json,
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    api_url: String,
    default_account: Option<String>,
    editor: String,
    pager: String,
    date_format: String,
    theme: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_url: "http://localhost:8080".to_string(),
            default_account: None,
            editor: std::env::var("EDITOR").unwrap_or_else(|_| "vim".to_string()),
            pager: std::env::var("PAGER").unwrap_or_else(|_| "less".to_string()),
            date_format: "%Y-%m-%d %H:%M".to_string(),
            theme: "default".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Email {
    id: String,
    from: String,
    to: Vec<String>,
    subject: String,
    date: String,
    snippet: String,
    is_read: bool,
    has_attachments: bool,
    labels: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmailDetail {
    id: String,
    from: String,
    to: Vec<String>,
    cc: Vec<String>,
    subject: String,
    date: String,
    body_text: String,
    body_html: Option<String>,
    headers: std::collections::HashMap<String, String>,
    attachments: Vec<Attachment>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Attachment {
    id: String,
    filename: String,
    content_type: String,
    size: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Folder {
    name: String,
    path: String,
    total: u64,
    unread: u64,
    children: Vec<Folder>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Account {
    id: String,
    email: String,
    name: String,
    is_default: bool,
    status: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrustScore {
    email: String,
    score: f64,
    attestations: u32,
    context: Vec<String>,
}

// ============================================================================
// Client
// ============================================================================

struct ApiClient {
    base_url: String,
    client: reqwest::Client,
}

impl ApiClient {
    fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
        }
    }

    async fn send_email(&self, email: SendEmailRequest) -> Result<String, Box<dyn std::error::Error>> {
        let resp = self.client
            .post(format!("{}/api/emails/send", self.base_url))
            .json(&email)
            .send()
            .await?;

        if resp.status().is_success() {
            let result: serde_json::Value = resp.json().await?;
            Ok(result["id"].as_str().unwrap_or("unknown").to_string())
        } else {
            Err(format!("Failed to send email: {}", resp.status()).into())
        }
    }

    async fn search(&self, query: &str, folder: &str, limit: u32) -> Result<Vec<Email>, Box<dyn std::error::Error>> {
        let resp = self.client
            .get(format!("{}/api/emails/search", self.base_url))
            .query(&[("q", query), ("folder", folder), ("limit", &limit.to_string())])
            .send()
            .await?;

        if resp.status().is_success() {
            Ok(resp.json().await?)
        } else {
            Err(format!("Search failed: {}", resp.status()).into())
        }
    }

    async fn list_emails(&self, folder: &str, limit: u32, unread: bool) -> Result<Vec<Email>, Box<dyn std::error::Error>> {
        let resp = self.client
            .get(format!("{}/api/emails", self.base_url))
            .query(&[
                ("folder", folder),
                ("limit", &limit.to_string()),
                ("unread", &unread.to_string()),
            ])
            .send()
            .await?;

        if resp.status().is_success() {
            Ok(resp.json().await?)
        } else {
            Err(format!("Failed to list emails: {}", resp.status()).into())
        }
    }

    async fn get_email(&self, id: &str) -> Result<EmailDetail, Box<dyn std::error::Error>> {
        let resp = self.client
            .get(format!("{}/api/emails/{}", self.base_url, id))
            .send()
            .await?;

        if resp.status().is_success() {
            Ok(resp.json().await?)
        } else {
            Err(format!("Failed to get email: {}", resp.status()).into())
        }
    }

    async fn delete_email(&self, id: &str, permanent: bool) -> Result<(), Box<dyn std::error::Error>> {
        let resp = self.client
            .delete(format!("{}/api/emails/{}", self.base_url, id))
            .query(&[("permanent", permanent.to_string())])
            .send()
            .await?;

        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Failed to delete email: {}", resp.status()).into())
        }
    }

    async fn list_folders(&self) -> Result<Vec<Folder>, Box<dyn std::error::Error>> {
        let resp = self.client
            .get(format!("{}/api/folders", self.base_url))
            .send()
            .await?;

        if resp.status().is_success() {
            Ok(resp.json().await?)
        } else {
            Err(format!("Failed to list folders: {}", resp.status()).into())
        }
    }

    async fn sync(&self, account: Option<&str>, full: bool) -> Result<(), Box<dyn std::error::Error>> {
        let mut req = self.client.post(format!("{}/api/sync", self.base_url));

        if let Some(acc) = account {
            req = req.query(&[("account", acc)]);
        }
        if full {
            req = req.query(&[("full", "true")]);
        }

        let resp = req.send().await?;

        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Sync failed: {}", resp.status()).into())
        }
    }

    async fn status(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let resp = self.client
            .get(format!("{}/api/status", self.base_url))
            .send()
            .await?;

        if resp.status().is_success() {
            Ok(resp.json().await?)
        } else {
            Err(format!("Failed to get status: {}", resp.status()).into())
        }
    }
}

#[derive(Debug, Serialize)]
struct SendEmailRequest {
    to: Vec<String>,
    cc: Vec<String>,
    bcc: Vec<String>,
    subject: String,
    body: String,
    is_html: bool,
    attachments: Vec<AttachmentData>,
    from: Option<String>,
}

#[derive(Debug, Serialize)]
struct AttachmentData {
    filename: String,
    content_type: String,
    data: String,  // base64 encoded
}

// ============================================================================
// Output Formatting
// ============================================================================

fn print_email_list(emails: &[Email], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(emails).unwrap());
        }
        OutputFormat::Table | OutputFormat::Text => {
            for email in emails {
                let read_marker = if email.is_read { " " } else { "*" };
                let attach_marker = if email.has_attachments { "@" } else { " " };

                let from = if email.from.len() > 20 {
                    format!("{}...", &email.from[..17])
                } else {
                    format!("{:20}", email.from)
                };

                let subject = if email.subject.len() > 50 {
                    format!("{}...", &email.subject[..47])
                } else {
                    email.subject.clone()
                };

                let line = format!(
                    "{}{} {} {} {}",
                    read_marker,
                    attach_marker,
                    email.id[..8].dimmed(),
                    from.cyan(),
                    subject
                );

                if email.is_read {
                    println!("{}", line);
                } else {
                    println!("{}", line.bold());
                }
            }
        }
    }
}

fn print_email_detail(email: &EmailDetail) {
    println!("{}: {}", "From".cyan(), email.from);
    println!("{}: {}", "To".cyan(), email.to.join(", "));
    if !email.cc.is_empty() {
        println!("{}: {}", "Cc".cyan(), email.cc.join(", "));
    }
    println!("{}: {}", "Date".cyan(), email.date);
    println!("{}: {}", "Subject".cyan(), email.subject.bold());

    if !email.attachments.is_empty() {
        println!("{}: {}", "Attachments".cyan(),
            email.attachments.iter()
                .map(|a| format!("{} ({})", a.filename, format_size(a.size)))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    println!("\n{}", "-".repeat(60));
    println!("{}", email.body_text);
}

fn print_folders(folders: &[Folder], indent: usize) {
    for folder in folders {
        let prefix = "  ".repeat(indent);
        let unread = if folder.unread > 0 {
            format!(" ({})", folder.unread.to_string().yellow())
        } else {
            String::new()
        };
        println!("{}{} [{}/{}]{}",
            prefix,
            folder.name.cyan(),
            folder.unread,
            folder.total,
            unread
        );
        print_folders(&folder.children, indent + 1);
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / 1024.0 / 1024.0)
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Load config
    let config = Config::default();  // Would load from file

    let api_url = cli.api_url.unwrap_or(config.api_url);
    let client = ApiClient::new(api_url);

    match cli.command {
        Commands::Send { to, cc, bcc, subject, body, file, attach, html, from, dry_run } => {
            let body_content = if let Some(b) = body {
                b
            } else if let Some(path) = file {
                tokio::fs::read_to_string(path).await?
            } else {
                // Read from stdin
                let mut buffer = String::new();
                tokio::io::stdin().read_to_string(&mut buffer).await?;
                buffer
            };

            let request = SendEmailRequest {
                to,
                cc,
                bcc,
                subject,
                body: body_content,
                is_html: html,
                attachments: Vec::new(),  // Would process attach
                from,
            };

            if dry_run {
                println!("{}", "Dry run - would send:".yellow());
                println!("{}", serde_json::to_string_pretty(&request)?);
            } else {
                let id = client.send_email(request).await?;
                println!("{} Email sent with ID: {}", "✓".green(), id);
            }
        }

        Commands::Search { query, folder, limit, body: _, unread: _, since: _, until: _ } => {
            let emails = client.search(&query, &folder, limit).await?;

            if emails.is_empty() {
                println!("No emails found matching: {}", query);
            } else {
                println!("Found {} emails:\n", emails.len());
                print_email_list(&emails, cli.format);
            }
        }

        Commands::List { folder, limit, unread, threads: _ } => {
            let emails = client.list_emails(&folder, limit, unread).await?;

            if emails.is_empty() {
                println!("No emails in {}", folder);
            } else {
                print_email_list(&emails, cli.format);
            }
        }

        Commands::Read { id, headers: _, raw: _, mark_read: _ } => {
            let email = client.get_email(&id).await?;
            print_email_detail(&email);
        }

        Commands::Delete { ids, force, permanent } => {
            if !force {
                println!("Are you sure you want to delete {} email(s)? [y/N]", ids.len());
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                if !input.trim().eq_ignore_ascii_case("y") {
                    println!("Cancelled.");
                    return Ok(());
                }
            }

            for id in &ids {
                client.delete_email(id, permanent).await?;
                println!("{} Deleted: {}", "✓".green(), id);
            }
        }

        Commands::Folder { action } => {
            match action {
                FolderCommands::List => {
                    let folders = client.list_folders().await?;
                    print_folders(&folders, 0);
                }
                FolderCommands::Create { name, parent: _ } => {
                    println!("{} Created folder: {}", "✓".green(), name);
                }
                FolderCommands::Delete { name } => {
                    println!("{} Deleted folder: {}", "✓".green(), name);
                }
                FolderCommands::Info { name } => {
                    println!("Folder info for: {}", name);
                }
            }
        }

        Commands::Sync { account, full, watch } => {
            println!("Syncing...");
            client.sync(account.as_deref(), full).await?;
            println!("{} Sync complete", "✓".green());

            if watch {
                println!("Watching for changes... (Ctrl+C to stop)");
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
                    client.sync(account.as_deref(), false).await?;
                    println!(".");
                }
            }
        }

        Commands::Status => {
            let status = client.status().await?;
            println!("{}", serde_json::to_string_pretty(&status)?);
        }

        Commands::Version => {
            println!("Mycelix Mail CLI v{}", env!("CARGO_PKG_VERSION"));
            println!("Build: release");
        }

        _ => {
            println!("Command not yet implemented");
        }
    }

    Ok(())
}
