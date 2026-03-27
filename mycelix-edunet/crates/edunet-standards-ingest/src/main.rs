// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CLI entry point for edunet-standards-ingest.

use clap::{Parser, Subcommand};
use edunet_standards_ingest::client::CspClient;
use edunet_standards_ingest::converter;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "edunet-standards-ingest",
    about = "Fetch K-12 standards from the Common Standards Project and output EduNet curriculum JSON",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available jurisdictions (states, organizations, schools)
    ListJurisdictions {
        /// Filter by type: state, organization, school
        #[arg(long, short = 't')]
        r#type: Option<String>,
    },

    /// List standard sets within a jurisdiction
    ListSets {
        /// Jurisdiction ID
        jurisdiction_id: String,
        /// Filter by subject (substring match)
        #[arg(long, short = 's')]
        subject: Option<String>,
        /// Filter by education level (e.g., "03" for Grade 3)
        #[arg(long, short = 'g')]
        grade: Option<String>,
    },

    /// Fetch a standard set and output curriculum JSON
    Fetch {
        /// Standard set ID (from list-sets output)
        standard_set_id: String,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
        /// Pretty-print JSON output
        #[arg(long, default_value_t = true)]
        pretty: bool,
    },

    /// Fetch all standard sets for a jurisdiction and output curriculum JSON files
    FetchAll {
        /// Jurisdiction ID
        jurisdiction_id: String,
        /// Output directory
        #[arg(long, short = 'o', default_value = ".")]
        output_dir: PathBuf,
        /// Filter by subject (substring match)
        #[arg(long, short = 's')]
        subject: Option<String>,
        /// Filter by education level (e.g., "03" for Grade 3)
        #[arg(long, short = 'g')]
        grade: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let client = CspClient::new()?;

    match cli.command {
        Commands::ListJurisdictions { r#type } => {
            let jurisdictions = client.list_jurisdictions().await?;

            let filtered: Vec<_> = if let Some(ref t) = r#type {
                jurisdictions
                    .iter()
                    .filter(|j| j.jurisdiction_type.eq_ignore_ascii_case(t))
                    .collect()
            } else {
                jurisdictions.iter().collect()
            };

            eprintln!("Found {} jurisdictions", filtered.len());
            println!("{:<40} {:<12} {}", "ID", "TYPE", "TITLE");
            println!("{}", "-".repeat(80));
            for j in &filtered {
                println!("{:<40} {:<12} {}", j.id, j.jurisdiction_type, j.title);
            }
        }

        Commands::ListSets {
            jurisdiction_id,
            subject,
            grade,
        } => {
            let detail = client.get_jurisdiction(&jurisdiction_id).await?;

            let filtered: Vec<_> = detail
                .standard_sets
                .iter()
                .filter(|s| {
                    if let Some(ref subj) = subject {
                        if !s.subject.to_lowercase().contains(&subj.to_lowercase()) {
                            return false;
                        }
                    }
                    if let Some(ref g) = grade {
                        if !s.education_levels.iter().any(|l| l == g) {
                            return false;
                        }
                    }
                    true
                })
                .collect();

            eprintln!(
                "Found {} standard sets for {}",
                filtered.len(),
                detail.title
            );
            println!("{:<60} {:<20} {}", "ID", "SUBJECT", "GRADES");
            println!("{}", "-".repeat(100));
            for s in &filtered {
                let grades = s.education_levels.join(",");
                println!("{:<60} {:<20} {}", s.id, s.subject, grades);
            }
        }

        Commands::Fetch {
            standard_set_id,
            output,
            pretty,
        } => {
            eprintln!("Fetching standard set {}...", standard_set_id);
            let set = client.get_standard_set(&standard_set_id).await?;
            let doc = converter::convert_standard_set(&set);

            let json = if pretty {
                serde_json::to_string_pretty(&doc)?
            } else {
                serde_json::to_string(&doc)?
            };

            if let Some(path) = output {
                std::fs::write(&path, &json)?;
                eprintln!(
                    "Wrote {} standards to {}",
                    doc.metadata.total_standards,
                    path.display()
                );
            } else {
                println!("{json}");
            }
        }

        Commands::FetchAll {
            jurisdiction_id,
            output_dir,
            subject,
            grade,
        } => {
            std::fs::create_dir_all(&output_dir)?;

            let detail = client.get_jurisdiction(&jurisdiction_id).await?;
            let sets: Vec<_> = detail
                .standard_sets
                .iter()
                .filter(|s| {
                    if let Some(ref subj) = subject {
                        if !s.subject.to_lowercase().contains(&subj.to_lowercase()) {
                            return false;
                        }
                    }
                    if let Some(ref g) = grade {
                        if !s.education_levels.iter().any(|l| l == g) {
                            return false;
                        }
                    }
                    true
                })
                .collect();

            eprintln!(
                "Fetching {} standard sets for {}...",
                sets.len(),
                detail.title
            );

            for (i, set_summary) in sets.iter().enumerate() {
                eprint!("[{}/{}] {}... ", i + 1, sets.len(), set_summary.title);

                match client.get_standard_set(&set_summary.id).await {
                    Ok(set) => {
                        let doc = converter::convert_standard_set(&set);
                        let filename = sanitize_filename(&set_summary.title);
                        let path = output_dir.join(format!("{filename}.json"));
                        let json = serde_json::to_string_pretty(&doc)?;
                        std::fs::write(&path, &json)?;
                        eprintln!("{} standards", doc.metadata.total_standards);
                    }
                    Err(e) => {
                        eprintln!("FAILED: {e}");
                    }
                }
            }
            eprintln!("Done.");
        }
    }

    Ok(())
}

/// Sanitize a string for use as a filename.
fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            ' ' => '_',
            _ => '_',
        })
        .collect::<String>()
        .to_lowercase()
}
