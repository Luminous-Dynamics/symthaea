// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CLI entry point for edunet-standards-ingest.

use clap::{Parser, Subcommand};
use edunet_standards_ingest::client::CspClient;
use edunet_standards_ingest::converter;
use edunet_standards_ingest::sources::CurriculumSource;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "edunet-standards-ingest",
    about = "Fetch K-12 through PhD standards and output EduNet curriculum JSON",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available jurisdictions (states, organizations, schools) — K-12
    ListJurisdictions {
        /// Filter by type: state, organization, school
        #[arg(long, short = 't')]
        r#type: Option<String>,
    },

    /// List standard sets within a jurisdiction — K-12
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

    /// Fetch a K-12 standard set and output curriculum JSON
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

    /// Fetch all K-12 standard sets for a jurisdiction
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

    // ---- Higher Education ----

    /// List CIP taxonomy families (university program classifications)
    ListCip,

    /// Fetch a CIP family as curriculum JSON (e.g., "11" for Computer Science)
    IngestCip {
        /// CIP family code (e.g., "11", "14", "27")
        family_code: String,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },

    /// List ACM CS2013 Knowledge Areas
    ListAcm,

    /// Fetch an ACM CS2013 Knowledge Area as curriculum JSON
    IngestAcm {
        /// Knowledge Area ID (e.g., "AL", "DS", "SE")
        ka_id: String,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },

    /// Fetch all ACM CS2013 Knowledge Areas
    IngestAcmAll {
        /// Output directory
        #[arg(long, short = 'o', default_value = ".")]
        output_dir: PathBuf,
    },

    /// List available PhD progression templates
    ListPhd,

    /// Fetch a PhD progression template as curriculum JSON
    IngestPhd {
        /// Template ID (e.g., "phd-cs", "phd-physics", "phd-math")
        template_id: String,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },

    /// Fetch all PhD templates
    IngestPhdAll {
        /// Output directory
        #[arg(long, short = 'o', default_value = ".")]
        output_dir: PathBuf,
    },

    // ---- Cross-Level Bridge ----

    /// Generate cross-level bridge edges between curriculum files
    ///
    /// Takes multiple curriculum JSON files (K-12, undergrad, grad, PhD) and
    /// generates LeadsTo edges that connect terminal nodes at each level to
    /// entry nodes at the next. This creates a continuous learning pathway
    /// from any grade through PhD.
    Bridge {
        /// Input curriculum JSON files (2 or more at different levels)
        #[arg(required = true, num_args = 2..)]
        files: Vec<PathBuf>,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },

    /// Merge multiple curriculum documents + bridge files into a unified graph
    ///
    /// Combines all nodes, edges, and bridge connections into a single
    /// CurriculumDocument — the Lifelong Epistemic Path.
    Merge {
        /// Curriculum JSON files to merge
        #[arg(required = true, num_args = 1..)]
        files: Vec<PathBuf>,
        /// Bridge JSON files to include (from the 'bridge' command)
        #[arg(long, short = 'b')]
        bridges: Vec<PathBuf>,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        // ============================================================
        // K-12 Commands (unchanged)
        // ============================================================
        Commands::ListJurisdictions { r#type } => {
            let client = CspClient::new()?;
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
            let client = CspClient::new()?;
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
            let client = CspClient::new()?;
            eprintln!("Fetching standard set {}...", standard_set_id);
            let set = client.get_standard_set(&standard_set_id).await?;
            let doc = converter::convert_standard_set(&set);
            write_document(&doc, output.as_deref(), pretty)?;
        }

        Commands::FetchAll {
            jurisdiction_id,
            output_dir,
            subject,
            grade,
        } => {
            let client = CspClient::new()?;
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

        // ============================================================
        // CIP Taxonomy Commands
        // ============================================================
        Commands::ListCip => {
            let source = edunet_standards_ingest::sources::cip::CipSource::new();
            let entries = source.list_available()?;
            eprintln!("Found {} CIP families", entries.len());
            println!("{:<8} {:<60} {}", "CODE", "TITLE", "LEVEL");
            println!("{}", "-".repeat(80));
            for e in &entries {
                println!("{:<8} {:<60} {}", e.id, e.title, e.level);
            }
        }

        Commands::IngestCip { family_code, output } => {
            let source = edunet_standards_ingest::sources::cip::CipSource::new();
            eprintln!("Generating curriculum for CIP family {}...", family_code);
            let doc = source.fetch(&family_code)?;
            write_document(&doc, output.as_deref(), true)?;
        }

        // ============================================================
        // ACM CS2013 Commands
        // ============================================================
        Commands::ListAcm => {
            let source = edunet_standards_ingest::sources::acm::AcmSource::new();
            let entries = source.list_available()?;
            eprintln!("Found {} ACM CS2013 Knowledge Areas", entries.len());
            println!("{:<8} {:<50} {}", "ID", "TITLE", "DETAILS");
            println!("{}", "-".repeat(80));
            for e in &entries {
                println!("{:<8} {:<50} {}", e.id, e.title, e.description);
            }
        }

        Commands::IngestAcm { ka_id, output } => {
            let source = edunet_standards_ingest::sources::acm::AcmSource::new();
            eprintln!("Generating curriculum for ACM KA {}...", ka_id);
            let doc = source.fetch(&ka_id)?;
            write_document(&doc, output.as_deref(), true)?;
        }

        Commands::IngestAcmAll { output_dir } => {
            std::fs::create_dir_all(&output_dir)?;
            let source = edunet_standards_ingest::sources::acm::AcmSource::new();
            let entries = source.list_available()?;
            eprintln!("Generating {} ACM Knowledge Area curricula...", entries.len());

            for (i, entry) in entries.iter().enumerate() {
                eprint!("[{}/{}] {}... ", i + 1, entries.len(), entry.title);
                let doc = source.fetch(&entry.id)?;
                let path = output_dir.join(format!("acm_cs2013_{}.json", entry.id.to_lowercase()));
                let json = serde_json::to_string_pretty(&doc)?;
                std::fs::write(&path, &json)?;
                eprintln!("{} nodes", doc.nodes.len());
            }
            eprintln!("Done.");
        }

        // ============================================================
        // PhD Template Commands
        // ============================================================
        Commands::ListPhd => {
            let source = edunet_standards_ingest::sources::phd::PhDSource::new();
            let entries = source.list_available()?;
            eprintln!("Found {} PhD templates", entries.len());
            println!("{:<25} {:<40} {}", "ID", "DISCIPLINE", "MILESTONES");
            println!("{}", "-".repeat(80));
            for e in &entries {
                println!("{:<25} {:<40} {}", e.id, e.title, e.description);
            }
        }

        Commands::IngestPhd {
            template_id,
            output,
        } => {
            let source = edunet_standards_ingest::sources::phd::PhDSource::new();
            eprintln!("Generating PhD progression for {}...", template_id);
            let doc = source.fetch(&template_id)?;
            write_document(&doc, output.as_deref(), true)?;
        }

        Commands::IngestPhdAll { output_dir } => {
            std::fs::create_dir_all(&output_dir)?;
            let source = edunet_standards_ingest::sources::phd::PhDSource::new();
            let entries = source.list_available()?;
            eprintln!("Generating {} PhD templates...", entries.len());

            for (i, entry) in entries.iter().enumerate() {
                eprint!("[{}/{}] {}... ", i + 1, entries.len(), entry.title);
                let doc = source.fetch(&entry.id)?;
                let path = output_dir.join(format!("{}.json", entry.id));
                let json = serde_json::to_string_pretty(&doc)?;
                std::fs::write(&path, &json)?;
                eprintln!("{} milestones", doc.nodes.len());
            }
            eprintln!("Done.");
        }

        // ============================================================
        // Cross-Level Bridge
        // ============================================================
        Commands::Bridge { files, output } => {
            eprintln!("Loading {} curriculum files...", files.len());
            let mut documents = Vec::new();

            for path in &files {
                let content = std::fs::read_to_string(path)?;
                let doc: converter::CurriculumDocument = serde_json::from_str(&content)?;
                eprintln!(
                    "  {} — {} ({}, {} nodes)",
                    path.display(),
                    doc.metadata.title,
                    doc.metadata.grade_level,
                    doc.nodes.len()
                );
                documents.push(doc);
            }

            let bridge = edunet_standards_ingest::bridge::generate_bridge(&documents);

            eprintln!(
                "\nGenerated {} bridge edges:",
                bridge.statistics.total_edges
            );
            eprintln!(
                "  K-12 → Undergraduate: {}",
                bridge.statistics.k12_to_undergrad
            );
            eprintln!(
                "  Undergraduate → Graduate: {}",
                bridge.statistics.undergrad_to_grad
            );
            eprintln!(
                "  Graduate → Doctoral: {}",
                bridge.statistics.grad_to_phd
            );

            let json = serde_json::to_string_pretty(&bridge)?;
            if let Some(path) = output {
                std::fs::write(&path, &json)?;
                eprintln!("\nWrote bridge to {}", path.display());
            } else {
                println!("{json}");
            }
        }

        // ============================================================
        // Merge (Unified Graph)
        // ============================================================
        Commands::Merge {
            files,
            bridges,
            output,
        } => {
            eprintln!("Loading {} curriculum files...", files.len());
            let mut documents = Vec::new();
            for path in &files {
                let content = std::fs::read_to_string(path)?;
                let doc: converter::CurriculumDocument = serde_json::from_str(&content)?;
                eprintln!(
                    "  {} — {} ({}, {} nodes)",
                    path.display(),
                    doc.metadata.title,
                    doc.metadata.grade_level,
                    doc.nodes.len()
                );
                documents.push(doc);
            }

            let mut bridge_docs = Vec::new();
            if !bridges.is_empty() {
                eprintln!("Loading {} bridge files...", bridges.len());
                for path in &bridges {
                    let content = std::fs::read_to_string(path)?;
                    let bridge: edunet_standards_ingest::bridge::BridgeDocument =
                        serde_json::from_str(&content)?;
                    eprintln!(
                        "  {} — {} edges",
                        path.display(),
                        bridge.edges.len()
                    );
                    bridge_docs.push(bridge);
                }
            }

            let (merged, stats) =
                edunet_standards_ingest::merge::merge_documents(&documents, &bridge_docs);

            eprintln!("\nMerged graph:");
            eprintln!("  Nodes: {}", stats.total_nodes);
            eprintln!("  Edges: {}", stats.total_edges);
            eprintln!("  Sources: {}", stats.sources_merged);
            eprintln!("  Bridge edges: {}", stats.bridge_edges_added);
            eprintln!("  Duplicates skipped: {}", stats.duplicate_nodes_skipped);
            eprintln!("  Levels: {:?}", stats.levels);
            eprintln!("  Subjects: {:?}", stats.subjects);

            let json = serde_json::to_string_pretty(&merged)?;
            if let Some(path) = output {
                std::fs::write(&path, &json)?;
                eprintln!("\nWrote unified graph to {}", path.display());
            } else {
                println!("{json}");
            }
        }
    }

    Ok(())
}

/// Write a curriculum document to a file or stdout.
fn write_document(
    doc: &converter::CurriculumDocument,
    output: Option<&std::path::Path>,
    pretty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = if pretty {
        serde_json::to_string_pretty(doc)?
    } else {
        serde_json::to_string(doc)?
    };

    if let Some(path) = output {
        std::fs::write(path, &json)?;
        eprintln!(
            "Wrote {} nodes to {}",
            doc.metadata.total_standards,
            path.display()
        );
    } else {
        println!("{json}");
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

