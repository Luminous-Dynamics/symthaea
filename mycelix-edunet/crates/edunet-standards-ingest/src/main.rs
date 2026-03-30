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

    /// Fetch a K-12 subject source (c3, iste, arts, pe, cefr)
    IngestK12 {
        /// Source ID: c3-k2, c3-35, c3-68, c3-912, iste-students, arts-elementary, arts-secondary, pe-elementary, pe-secondary, cefr
        source_id: String,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },

    /// Fetch South African CAPS curriculum (Matric Math & Physical Sciences)
    IngestCaps {
        /// Source ID: caps-math-10, caps-math-11, caps-math-12, caps-physics-10, caps-physics-11, caps-physics-12
        source_id: String,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },

    /// Fetch all CAPS curricula (Grade 10-12 Math + Physical Sciences)
    IngestCapsAll {
        /// Output directory
        #[arg(long, short = 'o', default_value = ".")]
        output_dir: PathBuf,
    },

    /// Fetch Luminous Dynamics curriculum (symthaea, mycelix, programming)
    IngestLuminous {
        /// Source ID: symthaea, mycelix, programming
        source_id: String,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },

    /// List MIT OCW departments
    ListOcw,

    /// Fetch MIT OCW courses for a department
    IngestOcw {
        /// Department ID (e.g., "6" for EECS, "18" for Math, "8" for Physics)
        dept_id: String,
        /// Maximum courses to fetch (default: 50)
        #[arg(long, default_value_t = 50)]
        limit: usize,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },

    /// Show career intelligence profiles (salary, growth, SDGs, automation risk)
    Careers {
        /// Filter by field name (substring match, or "all")
        #[arg(default_value = "all")]
        field: String,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show detailed career profile (activities, skills, tools, values, progression)
    CareerDetail {
        /// Field name (exact or substring match)
        field: String,
    },

    /// Compare two careers side-by-side
    CareerCompare {
        /// First career field name
        field_a: String,
        /// Second career field name
        field_b: String,
    },

    /// List ESCO career fields
    ListEsco,

    /// Fetch ESCO career pathway (occupations + skills)
    IngestEsco {
        /// Career field (e.g., "software", "data-science", "healthcare")
        field: String,
        /// Maximum occupations to fetch (default: 5)
        #[arg(long, default_value_t = 5)]
        limit: usize,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },

    /// Batch-generate lessons, assessments, and flashcards for curriculum nodes
    BatchGenerate {
        /// Curriculum JSON file (unified graph or single subject)
        file: PathBuf,
        /// Output directory for generated content
        #[arg(long, short = 'o', default_value = "generated_content")]
        output_dir: PathBuf,
        /// Minimum coherence threshold (0.0-1.0)
        #[arg(long, default_value_t = 0.6)]
        quality_threshold: f32,
        /// Filter by subject (substring match)
        #[arg(long, short = 's')]
        subject: Option<String>,
        /// Filter by grade level
        #[arg(long, short = 'g')]
        grade: Option<String>,
        /// Resume from previous progress
        #[arg(long)]
        resume: bool,
    },

    /// Find an optimal learning path through the curriculum graph
    Plan {
        /// Curriculum JSON file (unified graph)
        file: PathBuf,
        /// Starting node ID
        #[arg(long)]
        from: String,
        /// Goal: node ID, "career:Software", or "level:6:Mathematics"
        #[arg(long)]
        to: String,
        /// Maximum total hours budget
        #[arg(long)]
        max_hours: Option<u32>,
        /// Enforce strict Bloom level progression
        #[arg(long)]
        bloom_strict: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Analyze a curriculum JSON file and print statistics
    Stats {
        /// Curriculum JSON file to analyze
        file: PathBuf,
        /// Output as JSON instead of human-readable
        #[arg(long)]
        json: bool,
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
        // K-12 Subject Sources
        // ============================================================
        Commands::IngestK12 { source_id, output } => {
            use edunet_standards_ingest::sources::k12_subjects::*;

            let doc = match source_id.as_str() {
                s if s.starts_with("c3-") => C3Source::new().fetch(s)?,
                "iste-students" => IsteSource::new().fetch("iste-students")?,
                s if s.starts_with("arts-") => ArtsSource::new().fetch(s)?,
                s if s.starts_with("pe-") => PeSource::new().fetch(s)?,
                "cefr" => CefrSource::new().fetch("cefr")?,
                s if s.starts_with("cyber-") => edunet_standards_ingest::sources::cybersecurity::CybersecuritySource::new().fetch(s)?,
                s if s.starts_with("philosophy-") => edunet_standards_ingest::sources::philosophy::PhilosophySource::new().fetch(s)?,
                "financial-literacy" | "critical-thinking" | "digital-literacy" | "health-literacy" |
                "civic-literacy" | "systems-thinking" | "emotional-intelligence" | "communication" |
                "sustainability" | "learning-skills" | "economics-basics" | "statistics-data" =>
                    edunet_standards_ingest::sources::universal::UniversalSource::new().fetch(&source_id)?,
                _ => return Err(format!("Unknown source: {source_id}").into()),
            };
            eprintln!("Generated {} nodes for {}", doc.nodes.len(), doc.metadata.title);
            write_document(&doc, output.as_deref(), true)?;
        }

        // ============================================================
        // Luminous Dynamics Curriculum
        // ============================================================
        // ============================================================
        // South African CAPS Curriculum
        // ============================================================
        Commands::IngestCaps { source_id, output } => {
            let source = edunet_standards_ingest::sources::caps::CapsSource::new();
            eprintln!("Generating CAPS curriculum: {}...", source_id);
            let doc = source.fetch(&source_id)?;
            eprintln!("Generated {} nodes for {}", doc.nodes.len(), doc.metadata.title);
            write_document(&doc, output.as_deref(), true)?;
        }

        Commands::IngestCapsAll { output_dir } => {
            std::fs::create_dir_all(&output_dir)?;
            let source = edunet_standards_ingest::sources::caps::CapsSource::new();
            let entries = source.list_available()?;
            eprintln!("Generating {} CAPS curricula...", entries.len());

            for (i, entry) in entries.iter().enumerate() {
                eprint!("[{}/{}] {}... ", i + 1, entries.len(), entry.title);
                let doc = source.fetch(&entry.id)?;
                let path = output_dir.join(format!("{}.json", entry.id));
                let json = serde_json::to_string_pretty(&doc)?;
                std::fs::write(&path, &json)?;
                eprintln!("{} nodes", doc.nodes.len());
            }
            eprintln!("Done.");
        }

        // ============================================================
        // Luminous Dynamics Curriculum
        // ============================================================
        Commands::IngestLuminous { source_id, output } => {
            let source = edunet_standards_ingest::sources::luminous::LuminousSource::new();
            eprintln!("Generating Luminous Dynamics curriculum: {}...", source_id);
            let doc = source.fetch(&source_id)?;
            write_document(&doc, output.as_deref(), true)?;
        }

        // ============================================================
        // MIT OCW Commands
        // ============================================================
        Commands::ListOcw => {
            let source = edunet_standards_ingest::sources::ocw::OcwSource::new()?;
            let entries = source.list_available()?;
            eprintln!("Found {} MIT departments", entries.len());
            println!("{:<6} {:<55} {}", "DEPT", "NAME", "LEVEL");
            println!("{}", "-".repeat(75));
            for e in &entries {
                println!("{:<6} {:<55} {}", e.id, e.title, e.level);
            }
        }

        Commands::IngestOcw {
            dept_id,
            limit,
            output,
        } => {
            let source = edunet_standards_ingest::sources::ocw::OcwSource::new()?;
            eprintln!("Fetching up to {} OCW courses for department {}...", limit, dept_id);
            let (doc, count) = source.fetch_and_convert(&dept_id, limit).await?;
            eprintln!("Got {} courses from MIT Learn API", count);
            write_document(&doc, output.as_deref(), true)?;
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
        // Career Intelligence
        // ============================================================
        Commands::Careers { field, json } => {
            let profiles = edunet_standards_ingest::career_profile::all_career_profiles_enriched();
            let filtered: Vec<_> = if field == "all" {
                profiles.iter().collect()
            } else {
                profiles.iter().filter(|p| p.field.to_lowercase().contains(&field.to_lowercase())).collect()
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&filtered)?);
            } else {
                for p in &filtered {
                    eprintln!("{}", edunet_standards_ingest::career_profile::format_career_summary(p));
                }
                eprintln!("--- {} career profiles ---", filtered.len());
            }
        }

        Commands::CareerDetail { field } => {
            let profiles = edunet_standards_ingest::career_profile::all_career_profiles_enriched();
            match profiles.iter().find(|p| p.field.to_lowercase().contains(&field.to_lowercase())) {
                Some(p) => eprintln!("{}", edunet_standards_ingest::career_profile::format_career_detail(p)),
                None => eprintln!("No career profile found matching '{field}'. Try: careers all"),
            }
        }

        Commands::CareerCompare { field_a, field_b } => {
            let profiles = edunet_standards_ingest::career_profile::all_career_profiles_enriched();
            let a = profiles.iter().find(|p| p.field.to_lowercase().contains(&field_a.to_lowercase()));
            let b = profiles.iter().find(|p| p.field.to_lowercase().contains(&field_b.to_lowercase()));
            match (a, b) {
                (Some(a), Some(b)) => eprintln!("{}", edunet_standards_ingest::career_profile::format_career_comparison(a, b)),
                (None, _) => eprintln!("No career profile found matching '{field_a}'"),
                (_, None) => eprintln!("No career profile found matching '{field_b}'"),
            }
        }

        // ============================================================
        // ESCO Career Pathways
        // ============================================================
        Commands::ListEsco => {
            let source = edunet_standards_ingest::sources::esco::EscoSource::new()?;
            let entries = source.list_available()?;
            eprintln!("Available ESCO career fields:");
            println!("{:<20} {}", "ID", "FIELD");
            println!("{}", "-".repeat(50));
            for e in &entries {
                println!("{:<20} {}", e.id, e.subject);
            }
        }

        Commands::IngestEsco {
            field,
            limit,
            output,
        } => {
            let source = edunet_standards_ingest::sources::esco::EscoSource::new()?;
            eprintln!("Fetching ESCO career pathway for '{}'...", field);
            let doc = source.fetch_career_pathway(&field, limit).await?;
            eprintln!("Got {} nodes ({} occupations + skills)", doc.nodes.len(), limit);
            write_document(&doc, output.as_deref(), true)?;
        }

        // ============================================================
        // Batch Content Generation
        // ============================================================
        Commands::BatchGenerate {
            file,
            output_dir,
            quality_threshold,
            subject,
            grade,
            resume,
        } => {
            let content = std::fs::read_to_string(&file)?;
            let standards = edunet_content_gen::ingest::load_curriculum_json(&content)
                .map_err(|e| format!("Failed to load curriculum: {e}"))?;

            eprintln!(
                "Loaded {} standards from {}",
                standards.len(),
                file.display()
            );

            // Use CAPS-aware generator for SA curriculum, mock for everything else
            let is_caps = content.contains("CAPS (South Africa DBE)");

            let filter = if let Some(ref subj) = subject {
                edunet_content_gen::batch::NodeFilter::BySubject(subj.clone())
            } else if let Some(ref g) = grade {
                edunet_content_gen::batch::NodeFilter::ByGradeLevel(g.clone())
            } else {
                edunet_content_gen::batch::NodeFilter::All
            };

            let progress_file = if resume {
                output_dir.join("batch_progress.json")
            } else {
                let _ = std::fs::remove_file(output_dir.join("batch_progress.json"));
                output_dir.join("batch_progress.json")
            };

            let config = edunet_content_gen::batch::BatchConfig {
                output_dir: output_dir.clone(),
                progress_file,
                quality_threshold,
                filter,
                ..Default::default()
            };

            let report = if is_caps {
                let pipeline = edunet_content_gen::pipeline::ContentPipeline::new(
                    edunet_content_gen::caps_content::CapsContentGenerator,
                ).with_quality_threshold(quality_threshold);
                edunet_content_gen::batch::run_batch(&pipeline, &standards, &config)
            } else {
                let pipeline = edunet_content_gen::pipeline::ContentPipeline::new(
                    edunet_content_gen::mock::MockGenerator,
                ).with_quality_threshold(quality_threshold);
                edunet_content_gen::batch::run_batch(&pipeline, &standards, &config)
            };
            edunet_content_gen::batch::print_report(&report);
        }

        // ============================================================
        // Learning Path Planner
        // ============================================================
        Commands::Plan {
            file,
            from,
            to,
            max_hours,
            bloom_strict,
            json,
        } => {
            let content = std::fs::read_to_string(&file)?;
            let doc: converter::CurriculumDocument = serde_json::from_str(&content)?;

            let goal = if to.starts_with("career:") {
                edunet_standards_ingest::pathfind::PathGoal::Career(to[7..].to_string())
            } else if to.starts_with("level:") {
                let parts: Vec<&str> = to[6..].splitn(2, ':').collect();
                let level: u8 = parts[0].parse().unwrap_or(6);
                let subject = parts.get(1).unwrap_or(&"").to_string();
                edunet_standards_ingest::pathfind::PathGoal::Level {
                    isced_level: level,
                    subject,
                }
            } else {
                edunet_standards_ingest::pathfind::PathGoal::Node(to.clone())
            };

            let constraints = edunet_standards_ingest::pathfind::PathConstraints {
                max_hours,
                bloom_strict,
                ..Default::default()
            };

            eprintln!("Planning path from '{}' to '{}'...", from, to);
            match edunet_standards_ingest::pathfind::find_learning_path(
                &doc,
                &from,
                &goal,
                &std::collections::HashSet::new(),
                &constraints,
            ) {
                Some(plan) => {
                    if json {
                        println!("{}", serde_json::to_string_pretty(&plan)?);
                    } else {
                        edunet_standards_ingest::pathfind::print_plan(&plan);
                    }
                }
                None => {
                    eprintln!("No path found from '{}' to '{}'.", from, to);
                }
            }
        }

        // ============================================================
        // Statistics
        // ============================================================
        Commands::Stats { file, json } => {
            let content = std::fs::read_to_string(&file)?;
            let doc: converter::CurriculumDocument = serde_json::from_str(&content)?;
            let stats = edunet_standards_ingest::stats::analyze(&doc);

            if json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                edunet_standards_ingest::stats::print_stats(&stats);
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

