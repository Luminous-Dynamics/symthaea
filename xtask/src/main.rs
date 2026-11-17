use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod crate_status;
mod cognitive_loop_evidence;
mod cognitive_drift_score;
mod cognitive_drift_compare;
mod cognitive_adaptation_utility;
mod cognitive_adaptation_panel;
mod cognitive_drift_baselines;
mod cognitive_replay_receipt;
mod cognitive_replay_driver;
mod cognitive_adaptation_audit;
mod cognitive_adaptation_replay;
mod cognitive_missingness;
mod cognitive_missingness_schedule;
mod cognitive_missingness_audit;
mod duplicate_scan;
mod manifest;
mod rhn_sweep;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Discovery check: flag a new crate that re-implements an existing symthaea-core module.
    /// Fails only on an UNADJUDICATED collision; known ones live in docs/crate-status.toml.
    DuplicateScan {
        #[arg(long, default_value = "docs/crate-status.toml")]
        registry: PathBuf,
    },

    /// Crate truth registry: join `cargo metadata` with `docs/crate-status.toml`.
    /// A crate's existence does not imply endorsement.
    CrateStatus {
        /// Emit the generated markdown inventory instead of checking.
        #[arg(long)]
        report: bool,
        /// Fail if any workspace member is unclassified.
        #[arg(long)]
        require_classified: bool,
        /// Fail on evidence-gap findings, not just integrity errors.
        #[arg(long)]
        strict: bool,
        #[arg(long, default_value = "docs/crate-status.toml")]
        registry: PathBuf,
    },
    RhnSweep {
        #[arg(long, default_value = "1024")]
        dims: String,
        #[arg(long, default_value = "32")]
        objects: String,
        #[arg(long, default_value = "1")]
        seeds: String,
        #[arg(long, default_value = "8")]
        branching: String,
        #[arg(long, default_value = "100")]
        split_thresholds: String,
        #[arg(long, default_value = "2")]
        redundancy_ks: String,
        #[arg(long, default_value = "3")]
        fanouts: String,
        #[arg(long, default_value = "LeafOnly")]
        policies: String,
        #[arg(long, default_value = "reports/rhn_v011_sweep")]
        out: PathBuf,
    },
    RhnFinalize {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    GenerateManifest {
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Validate the cognitive-loop evidence benchmark manifest and hard-gate contracts.
    ValidateCognitiveLoopEvidence {
        #[arg(long, default_value = "benchmarks/cognitive-loop-evidence/scenarios.json")]
        manifest: PathBuf,
    },
    /// Score alarm timestamps against ground-truth drift episodes.
    ScoreCognitiveLoopDrift {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Compare paired detector results with a dataset-clustered bootstrap interval.
    CompareCognitiveLoopDrift {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Score task loss and intervention cost across counterfactual adaptation policies.
    ScoreCounterfactualAdaptationUtility {
        #[arg(long)]
        input: PathBuf,
        /// Raw replay receipt to validate and bind to the score. Required for confirmatory panels.
        #[arg(long)]
        receipt: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Generate deterministic policy/evaluator sidecars for delay and informative-missingness stress tests.
    GenerateCognitiveMissingnessSchedule {
        #[arg(long)]
        input: PathBuf,
        /// Output contains messages for one tick only; never use a whole future delivery queue as policy input.
        #[arg(long)]
        policy_tick_out: PathBuf,
        /// Evaluator-only artifact; must remain outside the policy process.
        #[arg(long)]
        evaluator_out: PathBuf,
        /// Logical decision/feedback tick to deliver to the policy.
        #[arg(long)]
        tick: u64,
    },
    /// Independently re-derive policy/evaluator missingness sidecars from pinned source input.
    AuditCognitiveMissingnessSchedule {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        policy_tick: PathBuf,
        #[arg(long)]
        evaluator: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Score bounded-loss identification bounds and MNAR sensitivity for missing outcomes.
    /// This is a statistical score, not a proof of measurement authenticity or causal identification.
    ScoreCognitiveMissingness {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run the fixed-threshold or EWMA control-chart baseline on a valid PE stream.
    RunCognitiveDriftBaseline {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// Validate causal timing, event parity, oracle isolation, and adaptation transactions in a replay receipt.
    ValidateCognitiveReplayReceipt {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// Aggregate five-arm utility records across the complete preregistered dataset × seed × scenario grid.
    CompareCognitiveAdaptationPanel {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Independently reopen raw receipts and recompute every score in a confirmatory adaptation panel.
    AuditCognitiveAdaptationPanel {
        /// JSON bundle manifest; all artifact paths are relative to its directory.
        #[arg(long)]
        input: PathBuf,
        /// SHA-256 published by an independent preregistration record before outcomes are examined.
        #[arg(long)]
        expected_bundle_sha256: String,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Generate a deterministic synthetic replay to smoke-test receipt and score plumbing.
    /// Output is always synthetic-not-measured and cannot count as benchmark evidence.
    RunCognitiveAdaptationSyntheticReplay {
        #[arg(long, default_value = "abrupt")]
        scenario: String,
        #[arg(long, default_value = "64")]
        ticks: usize,
        #[arg(long, default_value = "7")]
        seed: u64,
        #[arg(long, default_value = "reports/cognitive-adaptation-synthetic-replay")]
        out_dir: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::RhnSweep {
            dims,
            objects,
            seeds,
            branching,
            split_thresholds,
            redundancy_ks,
            fanouts,
            policies,
            out,
        } => {
            let dims = parse_list(&dims)?;
            let objects = parse_list(&objects)?;
            let seeds = parse_list(&seeds)?;
            let branching = parse_list(&branching)?;
            let split_thresholds = parse_list(&split_thresholds)?;
            let redundancy_ks = parse_list(&redundancy_ks)?;
            let fanouts = parse_list(&fanouts)?;
            let policies = policies.split(',').map(|s| s.to_string()).collect();

            rhn_sweep::run_sweep(
                dims,
                objects,
                seeds,
                branching,
                split_thresholds,
                redundancy_ks,
                fanouts,
                policies,
                out,
            )?;
        }
        Commands::RhnFinalize { input, out } => {
            rhn_sweep::run_finalize(input, out)?;
        }
        Commands::GenerateManifest { root } => {
            let files = vec!["Cargo.toml", "src/lib.rs", "symthaea-core/Cargo.toml"];
            manifest::generate_manifest(&root, &files)?;
            println!(
                "Manifest generated at {}",
                root.join("manifest.json").display()
            );
        }
        Commands::DuplicateScan { registry } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let registry_path = if registry.is_absolute() {
                registry
            } else {
                root.join(registry)
            };
            duplicate_scan::scan(&root, &registry_path)?;
        }
        Commands::CrateStatus {
            report,
            require_classified,
            strict,
            registry,
        } => {
            // The workspace root is this manifest's parent; xtask always runs from inside it.
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask always lives one level below the workspace root")
                .to_path_buf();
            let registry_path = if registry.is_absolute() {
                registry
            } else {
                root.join(registry)
            };
            if report {
                crate_status::report(&root, &registry_path)?;
            } else {
                crate_status::check(&root, &registry_path, require_classified, strict)?;
            }
        }
        Commands::ValidateCognitiveLoopEvidence { manifest } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask always lives one level below the workspace root")
                .to_path_buf();
            let manifest_path = if manifest.is_absolute() {
                manifest
            } else {
                root.join(manifest)
            };
            let summary = cognitive_loop_evidence::validate_manifest(&manifest_path)?;
            println!(
                "OK: cognitive-loop evidence manifest has {} baselines, {} suites, {} scenarios; hard-gate and provenance contracts are complete.",
                summary.baseline_count, summary.suite_count, summary.case_count
            );
        }
        Commands::ScoreCognitiveLoopDrift { input, out } => {
            let score = cognitive_drift_score::evaluate_file(&input)?;
            let rendered = serde_json::to_string_pretty(&score)?;
            if let Some(path) = out {
                std::fs::write(&path, rendered.as_bytes())?;
                println!("Wrote drift score to {}", path.display());
            } else {
                println!("{rendered}");
            }
        }
        Commands::CompareCognitiveLoopDrift { input, out } => {
            let result = cognitive_drift_compare::evaluate_file(&input)?;
            let rendered = serde_json::to_string_pretty(&result)?;
            if let Some(path) = out {
                std::fs::write(&path, rendered.as_bytes())?;
                println!("Wrote paired comparison to {}", path.display());
            } else {
                println!("{rendered}");
            }
        }
        Commands::ScoreCounterfactualAdaptationUtility { input, receipt, out } => {
            let result = if let Some(receipt_path) = receipt {
                cognitive_adaptation_utility::evaluate_file_with_receipt(&input, &receipt_path)?
            } else {
                cognitive_adaptation_utility::evaluate_file(&input)?
            };
            let rendered = serde_json::to_string_pretty(&result)?;
            if let Some(path) = out {
                std::fs::write(&path, rendered.as_bytes())?;
                println!("Wrote counterfactual adaptation utility score to {}", path.display());
            } else {
                println!("{rendered}");
            }
        }
        Commands::GenerateCognitiveMissingnessSchedule { input, policy_tick_out, evaluator_out, tick } => {
            let summary = cognitive_missingness_schedule::generate_file(
                &input,
                &policy_tick_out,
                &evaluator_out,
                tick,
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Commands::AuditCognitiveMissingnessSchedule { input, policy_tick, evaluator, out } => {
            let report = cognitive_missingness_audit::audit_files(&input, &policy_tick, &evaluator)?;
            let rendered = serde_json::to_string_pretty(&report)?;
            if let Some(path) = out {
                std::fs::write(&path, rendered.as_bytes())?;
                println!("Wrote independently re-derived missingness audit to {}", path.display());
            } else {
                println!("{rendered}");
            }
        }
        Commands::ScoreCognitiveMissingness { input, out } => {
            let result = cognitive_missingness::evaluate_file(&input)?;
            let rendered = serde_json::to_string_pretty(&result)?;
            if let Some(path) = out {
                std::fs::write(&path, rendered.as_bytes())?;
                println!("Wrote informative-missingness identification score to {}", path.display());
            } else {
                println!("{rendered}");
            }
        }
        Commands::RunCognitiveDriftBaseline { input, out } => {
            let result = cognitive_drift_baselines::evaluate_file(&input)?;
            let rendered = serde_json::to_string_pretty(&result)?;
            if let Some(path) = out {
                std::fs::write(&path, rendered.as_bytes())?;
                println!("Wrote scalar drift baseline result to {}", path.display());
            } else {
                println!("{rendered}");
            }
        }

        Commands::ValidateCognitiveReplayReceipt { input, out } => {
            let result = cognitive_replay_receipt::evaluate_file(&input)?;
            let rendered = serde_json::to_string_pretty(&result)?;
            if let Some(path) = out {
                std::fs::write(&path, rendered.as_bytes())?;
                println!("Wrote cognitive replay receipt validation to {}", path.display());
            } else {
                println!("{rendered}");
            }
        }

        Commands::CompareCognitiveAdaptationPanel { input, out } => {
            let result = cognitive_adaptation_panel::evaluate_file(&input)?;
            let rendered = serde_json::to_string_pretty(&result)?;
            if let Some(path) = out {
                std::fs::write(&path, rendered.as_bytes())?;
                println!("Wrote cognitive adaptation panel result to {}", path.display());
            } else {
                println!("{rendered}");
            }
        }
        Commands::AuditCognitiveAdaptationPanel { input, expected_bundle_sha256, out } => {
            let result = cognitive_adaptation_audit::evaluate_file(&input, &expected_bundle_sha256)?;
            let rendered = serde_json::to_string_pretty(&result)?;
            if let Some(path) = out {
                std::fs::write(&path, rendered.as_bytes())?;
                println!("Wrote independently audited adaptation panel to {}", path.display());
            } else {
                println!("{rendered}");
            }
        }
        Commands::RunCognitiveAdaptationSyntheticReplay { scenario, ticks, seed, out_dir } => {
            let summary = cognitive_adaptation_replay::run_synthetic(&scenario, ticks, seed, &out_dir)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
    }
    Ok(())
}

fn parse_list(s: &str) -> anyhow::Result<Vec<usize>> {
    s.split(',')
        .map(|item| item.parse::<usize>().map_err(anyhow::Error::from))
        .collect()
}
