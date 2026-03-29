// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # ask-nix CLI
//!
//! Natural language interface for NixOS.
//!
//! ## Usage
//!
//! ```bash
//! # Natural language queries
//! ask-nix "search firefox"
//! ask-nix "install vim"
//! ask-nix "find a markdown editor"
//!
//! # Direct commands
//! ask-nix search firefox
//! ask-nix install vim
//! ask-nix generations
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::time::Instant;

use luminous_nix::{
    paths, CommandExecutor, Formatter, Intent, IntentClassifier, LearningSystem,
};

/// Natural language interface for NixOS
#[derive(Parser)]
#[command(
    name = "ask-nix",
    version,
    author = "Luminous Dynamics",
    about = "Natural language interface for NixOS - powered by Symthaea HDC",
    long_about = "ask-nix translates natural language queries into NixOS commands.\n\n\
                  Examples:\n  \
                  ask-nix \"search firefox\"\n  \
                  ask-nix \"install vim\"\n  \
                  ask-nix \"find markdown editor\"\n  \
                  ask-nix \"list generations\"\n  \
                  ask-nix \"remove package\""
)]
struct Cli {
    /// Natural language query (or use subcommands)
    #[arg(trailing_var_arg = true)]
    query: Option<Vec<String>>,

    /// Show verbose output including intent classification details
    #[arg(short, long)]
    verbose: bool,

    /// Disable colored output
    #[arg(long)]
    no_color: bool,

    /// Dry run - show command without executing
    #[arg(short = 'n', long)]
    dry_run: bool,

    /// JSON output format
    #[arg(long)]
    json: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

/// Direct commands (bypass intent classification)
#[derive(Subcommand)]
enum Commands {
    /// Search for packages
    #[command(alias = "find")]
    Search {
        /// Search query
        query: String,
        /// Limit results
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Install a package
    #[command(alias = "add")]
    Install {
        /// Package name
        package: String,
        /// Use flakes (nix profile install)
        #[arg(short, long)]
        flake: bool,
    },

    /// Remove a package
    #[command(alias = "uninstall", alias = "rm")]
    Remove {
        /// Package name
        package: String,
    },

    /// List system generations
    #[command(alias = "gens")]
    Generations {
        /// Limit to N most recent generations
        #[arg(short, long)]
        limit: Option<usize>,
    },

    /// Show system information
    #[command(alias = "info")]
    Status,

    /// Garbage collect (cleanup old generations)
    #[command(alias = "clean")]
    Gc {
        /// Delete generations older than N days
        #[arg(short, long)]
        older_than: Option<u32>,
    },

    /// Rebuild the system
    Rebuild {
        /// Switch to new configuration
        #[arg(long, default_value = "true")]
        switch: bool,
    },

    /// Interactive mode
    #[command(alias = "i")]
    Interactive,

    /// Provide feedback on the last command
    Feedback {
        /// Was the last command successful?
        #[arg(short, long)]
        success: bool,
    },

    /// Show learning statistics
    Stats,

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let start = Instant::now();

    // Initialize systems
    let mut classifier = IntentClassifier::new();
    let executor = CommandExecutor::new();
    let formatter = Formatter::new(!cli.no_color, cli.json);
    let mut learning = LearningSystem::load().unwrap_or_default();

    // Bootstrap classifier with NixOS knowledge
    classifier.bootstrap();

    // Load persistent HDC memory (learned experiences)
    if let Some(memory) = load_hdc_memory() {
        classifier.import_memory(memory);
        if cli.verbose {
            eprintln!("[Loaded {} learned experiences from disk]", classifier.stats().total_experiences);
        }
    }

    let result = match cli.command {
        // Direct subcommand
        Some(cmd) => execute_command(&cmd, &executor, &formatter, cli.dry_run).await,

        // Natural language query
        None => {
            if let Some(words) = cli.query {
                if words.is_empty() {
                    // No input - show help
                    println!("Usage: ask-nix <query> or ask-nix <command>");
                    println!("Try: ask-nix --help");
                    return Ok(());
                }

                let query = words.join(" ");
                process_natural_language(
                    &query,
                    &mut classifier,
                    &executor,
                    &formatter,
                    &mut learning,
                    cli.verbose,
                    cli.dry_run,
                )
                .await
            } else {
                // No query, no command - show help
                println!("Usage: ask-nix <query> or ask-nix <command>");
                println!("Try: ask-nix --help");
                Ok(())
            }
        }
    };

    if cli.verbose {
        let elapsed = start.elapsed();
        eprintln!("\n[Completed in {:?}]", elapsed);
    }

    result
}

/// Execute a direct subcommand
async fn execute_command(
    cmd: &Commands,
    executor: &CommandExecutor,
    formatter: &Formatter,
    dry_run: bool,
) -> Result<()> {
    match cmd {
        Commands::Search { query, limit } => {
            let result = executor.search(query, *limit, dry_run).await?;
            formatter.format_search_result(&result);
        }
        Commands::Install { package, flake } => {
            let result = executor.install(package, *flake, dry_run).await?;
            formatter.format_execution_result(&result);
        }
        Commands::Remove { package } => {
            let result = executor.remove(package, dry_run).await?;
            formatter.format_execution_result(&result);
        }
        Commands::Generations { limit } => {
            let result = executor.list_generations(*limit, dry_run).await?;
            formatter.format_generations(&result);
        }
        Commands::Status => {
            let result = executor.system_status(dry_run).await?;
            formatter.format_status(&result);
        }
        Commands::Gc { older_than } => {
            let result = executor.garbage_collect(*older_than, dry_run).await?;
            formatter.format_execution_result(&result);
        }
        Commands::Rebuild { switch } => {
            let result = executor.rebuild(*switch, dry_run).await?;
            formatter.format_execution_result(&result);
        }
        Commands::Interactive => {
            run_interactive().await?;
        }
        Commands::Feedback { success } => {
            println!(
                "Feedback recorded: {}",
                if *success { "success" } else { "failure" }
            );
        }
        Commands::Stats => {
            let learning = LearningSystem::load().unwrap_or_default();
            formatter.format_learning_stats(&learning);
        }
        Commands::Completions { shell } => {
            use clap::CommandFactory;
            let mut cmd = Cli::command();
            clap_complete::generate(*shell, &mut cmd, "ask-nix", &mut std::io::stdout());
        }
    }
    Ok(())
}

/// Process a natural language query
async fn process_natural_language(
    query: &str,
    classifier: &mut IntentClassifier,
    executor: &CommandExecutor,
    formatter: &Formatter,
    learning: &mut LearningSystem,
    verbose: bool,
    dry_run: bool,
) -> Result<()> {
    // Classify intent using HDC
    let intent_result = classifier.classify(query);

    if verbose {
        eprintln!("[Intent: {:?} (confidence: {:.2})]", intent_result.intent, intent_result.confidence);
        if !intent_result.alternatives.is_empty() {
            eprintln!("[Alternatives: {:?}]", intent_result.alternatives);
        }
    }

    // Check confidence threshold
    if intent_result.confidence < 0.3 {
        println!("I'm not sure what you want to do. Could you rephrase?");
        println!("Try commands like:");
        println!("  - search <package>  - find packages");
        println!("  - install <package> - install a package");
        println!("  - generations       - list system generations");
        return Ok(());
    }

    // Execute based on intent
    let result = match &intent_result.intent {
        Intent::Search { query: search_query } => {
            let result = executor.search(search_query, 20, dry_run).await?;
            formatter.format_search_result(&result);
            !result.packages.is_empty() || result.dry_run
        }
        Intent::Install { package } => {
            let result = executor.install(package, false, dry_run).await?;
            formatter.format_execution_result(&result);
            result.success
        }
        Intent::Remove { package } => {
            let result = executor.remove(package, dry_run).await?;
            formatter.format_execution_result(&result);
            result.success
        }
        Intent::ListGenerations => {
            let result = executor.list_generations(None, dry_run).await?;
            formatter.format_generations(&result);
            true
        }
        Intent::Rebuild => {
            let result = executor.rebuild(true, dry_run).await?;
            formatter.format_execution_result(&result);
            result.success
        }
        Intent::GarbageCollect => {
            let result = executor.garbage_collect(None, dry_run).await?;
            formatter.format_execution_result(&result);
            result.success
        }
        Intent::Status => {
            let result = executor.system_status(dry_run).await?;
            formatter.format_status(&result);
            true
        }
        Intent::Help => {
            print_help();
            true
        }
        Intent::Unknown => {
            println!("I couldn't understand that query.");
            println!("Try: ask-nix --help");
            false
        }
    };

    // Record outcome for learning
    let outcome = if result { 1.0 } else { -0.5 };
    learning.record(query, &intent_result.intent, outcome);
    learning.save().ok();

    // Feed back to classifier and persist HDC memory
    classifier.learn_from_outcome(query, &intent_result.intent, outcome);
    save_hdc_memory(&classifier);

    Ok(())
}

/// Run interactive REPL mode
async fn run_interactive() -> Result<()> {
    use std::io::{self, BufRead, Write};

    println!("ask-nix interactive mode (type 'quit' to exit)");
    println!();

    let mut classifier = IntentClassifier::new();
    classifier.bootstrap();

    // Load persistent HDC memory
    if let Some(memory) = load_hdc_memory() {
        classifier.import_memory(memory);
        println!("[Loaded learned experiences from previous sessions]");
    }

    let executor = CommandExecutor::new();
    let formatter = Formatter::new(true, false);
    let mut learning = LearningSystem::load().unwrap_or_default();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("nix> ");
        stdout.flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }

        let query = line.trim();
        if query.is_empty() {
            continue;
        }
        if query == "quit" || query == "exit" || query == "q" {
            break;
        }

        if let Err(e) = process_natural_language(
            query,
            &mut classifier,
            &executor,
            &formatter,
            &mut learning,
            false,
            false,
        )
        .await
        {
            eprintln!("Error: {}", e);
        }
        println!();
    }

    // Save final HDC memory state
    save_hdc_memory(&classifier);
    println!("Goodbye!");
    Ok(())
}

/// Print help information
fn print_help() {
    println!("ask-nix - Natural language interface for NixOS");
    println!();
    println!("Examples:");
    println!("  ask-nix \"search firefox\"      - Find packages matching 'firefox'");
    println!("  ask-nix \"install vim\"         - Install vim");
    println!("  ask-nix \"find markdown editor\" - Search for markdown editors");
    println!("  ask-nix \"list generations\"    - Show system generations");
    println!("  ask-nix \"garbage collect\"     - Clean up old generations");
    println!("  ask-nix \"rebuild system\"      - Rebuild NixOS configuration");
    println!();
    println!("Direct commands:");
    println!("  ask-nix search <query>         - Search packages");
    println!("  ask-nix install <package>      - Install package");
    println!("  ask-nix remove <package>       - Remove package");
    println!("  ask-nix generations            - List generations");
    println!("  ask-nix status                 - System information");
    println!("  ask-nix gc                     - Garbage collect");
    println!("  ask-nix interactive            - Enter interactive mode");
}

// =============================================================================
// HDC Memory Persistence
// =============================================================================

use mycelix_core_types::wisdom_engine::MemorySnapshot;

/// Load HDC memory from disk
fn load_hdc_memory() -> Option<MemorySnapshot> {
    let data_dir = paths::data_dir()?;
    let path = data_dir.join("hdc_memory.json");
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Save HDC memory to disk
fn save_hdc_memory(classifier: &IntentClassifier) {
    if let Some(data_dir) = paths::data_dir() {
        let _ = std::fs::create_dir_all(&data_dir);
        let path = data_dir.join("hdc_memory.json");
        let snapshot = classifier.export_memory();
        if let Ok(content) = serde_json::to_string(&snapshot) {
            let _ = std::fs::write(path, content);
        }
    }
}
