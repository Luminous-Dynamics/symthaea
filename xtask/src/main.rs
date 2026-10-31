use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod basalt_state;
mod basalt_observer;
mod basalt_observer_result;
mod basalt_fixture;
mod basalt_cargo_fixture;
mod basalt_cargo_mtime_fixture;
mod basalt_coverage;
mod basalt_linux_probe;
mod basalt_validate;
mod crate_status;
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
    /// Audit currentness and re-verification obligations for a BASALT evidence timeline.
    BasaltStateAudit {
        /// Path to the event-sourced resource/evidence timeline JSON.
        #[arg(long)]
        input: PathBuf,
    },
    /// Audit the structure and cross-references of a BASALT observer challenge catalog.
    BasaltObserverCatalogAudit {
        /// Path to the observer-challenge catalog JSON.
        #[arg(long)]
        input: PathBuf,
    },
    /// Adjudicate untrusted observer receipts against a separate BASALT oracle ledger.
    BasaltObserverResultAudit {
        /// Path to the exact challenge catalog bytes.
        #[arg(long)]
        catalog: PathBuf,
        /// Path to the independently authored challenge-oracle ledger.
        #[arg(long)]
        oracle: PathBuf,
        /// Path to the raw observer-receipt ledger (treated as untrusted).
        #[arg(long)]
        receipts: PathBuf,
    },
    /// Execute controlled counterfactual fixtures and write a separate BASALT oracle ledger.
    BasaltObserverFixtureRun {
        /// Path to the challenge catalog.
        #[arg(long)]
        catalog: PathBuf,
        /// Output directory; a unique run subdirectory is created inside it.
        #[arg(long)]
        out_dir: PathBuf,
        /// Case ID to execute; reciprocal paired controls are added automatically.
        #[arg(long = "case-id", action = clap::ArgAction::Append, required = true)]
        case_ids: Vec<String>,
    },
    /// Run a real Cargo build-script challenge with an intentionally undeclared input.
    BasaltCargoBuildScriptFixtureRun {
        #[arg(long)]
        catalog: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Run a Cargo mtime-preserving build-script input challenge.
    BasaltCargoMtimeFixtureRun {
        #[arg(long)]
        catalog: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Independently verify a retained Cargo mtime fixture and replay its warm/fresh states.
    BasaltCargoMtimeEvidenceVerify {
        /// Path to cargo-mtime-fixture-evidence.v1.json.
        #[arg(long)]
        evidence: PathBuf,
        /// Path to the exact observer challenge catalog.
        #[arg(long)]
        catalog: PathBuf,
    },
    /// Run a Cargo build-script environment-variable invalidation challenge.
    BasaltCargoEnvironmentFixtureRun {
        #[arg(long)]
        catalog: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Audit an observer coverage envelope against a separately pinned candidate profile.
    BasaltCoverageEnvelopeAudit {
        /// Path to the exact versioned coverage profile.
        #[arg(long)]
        profile: PathBuf,
        /// Path to the run-level coverage envelope JSON.
        #[arg(long)]
        receipt: PathBuf,
    },
    /// Inventory read-only Linux interfaces for selecting a BASALT observer profile.
    BasaltLinuxCapabilityProbe {
        /// Create a new JSON inventory report at this path.
        #[arg(long)]
        out: PathBuf,
    },
    /// Validate a BASALT versioned run record and its cross-record evidence invariants.
    BasaltValidate {
        /// Path to the run-record JSON file.
        #[arg(long)]
        input: PathBuf,
        /// Path to the frozen suite manifest; its exact bytes must match source.manifest_digest.
        #[arg(long)]
        manifest: PathBuf,
        /// Directory of offline UTF-8 task specifications, named <sha256-hex>.txt.
        #[arg(long)]
        spec_root: PathBuf,
    },
    /// Discovery check: flag a new crate that re-implements an existing symthaea-core module.
    /// Fails only on an UNADJUDICATED collision; known ones live in docs/crate-status.toml.
    DuplicateScan {
        #[arg(long, default_value = "docs/crate-status.toml")]
        registry: PathBuf,
    },
    /// Crate truth registry: join cargo metadata with docs/crate-status.toml.
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
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::BasaltStateAudit { input } => {
            let report = basalt_state::audit_file(&input)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.structurally_valid {
                anyhow::bail!("BASALT evidence-state timeline is structurally invalid");
            }
        }
        Commands::BasaltObserverCatalogAudit { input } => {
            let report = basalt_observer::audit_file(&input)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.structurally_valid {
                anyhow::bail!("BASALT observer challenge catalog is structurally invalid");
            }
        }
        Commands::BasaltObserverResultAudit {
            catalog,
            oracle,
            receipts,
        } => {
            let report =
                basalt_observer_result::audit_files(&catalog, &oracle, &receipts)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.adjudication_passed {
                anyhow::bail!("BASALT observer result adjudication failed");
            }
        }
        Commands::BasaltObserverFixtureRun {
            catalog,
            out_dir,
            case_ids,
        } => {
            let report = basalt_fixture::run(&catalog, &out_dir, &case_ids)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::BasaltCargoBuildScriptFixtureRun { catalog, out_dir } => {
            let report = basalt_cargo_fixture::run(&catalog, &out_dir)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::BasaltCargoMtimeFixtureRun { catalog, out_dir } => {
            let report = basalt_cargo_mtime_fixture::run(&catalog, &out_dir)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::BasaltCargoMtimeEvidenceVerify { evidence, catalog } => {
            let report = basalt_cargo_mtime_fixture::verify(&evidence, &catalog)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::BasaltCargoEnvironmentFixtureRun { catalog, out_dir } => {
            let report = basalt_cargo_mtime_fixture::run_environment(&catalog, &out_dir)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::BasaltCoverageEnvelopeAudit { profile, receipt } => {
            let report = basalt_coverage::audit_files(&profile, &receipt)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.coverage_claim_consistent {
                anyhow::bail!("BASALT observer coverage envelope is malformed, unbound or overclaims completeness");
            }
        }
        Commands::BasaltLinuxCapabilityProbe { out } => {
            let report = basalt_linux_probe::run(&out)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::BasaltValidate { input, manifest, spec_root } => {
            let report = basalt_validate::validate_file(&input, &manifest, &spec_root)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.valid {
                anyhow::bail!("BASALT run-record validation failed");
            }
        }
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
    }
    Ok(())
}

fn parse_list(s: &str) -> anyhow::Result<Vec<usize>> {
    s.split(',')
        .map(|item| item.parse::<usize>().map_err(anyhow::Error::from))
        .collect()
}
