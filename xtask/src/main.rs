use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod crate_status;
mod duplicate_scan;
mod enzyme_bench;
mod enzyme_calibration;
mod enzyme_evidence_graph;
mod enzyme_model_falsification;
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
    /// Validate a frozen, provenance-complete enzyme benchmark before any model scoring.
    EnzymeBenchCheck {
        /// Manifest path, relative to the workspace root unless absolute.
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Reproduce and verify one declared zero-intercept EnzymeML calibration fit.
    EnzymeCalibrationCheck {
        /// Path to the local calibration JSON artifact.
        #[arg(long)]
        input: PathBuf,
        /// Expected SHA-256 of the exact input bytes.
        #[arg(long)]
        sha256: String,
    },
    /// Verify all five hash-pinned SLAC calibration files from a local research-data staging area.
    EnzymeCalibrationSuiteCheck {
        /// Metadata-only suite manifest, workspace-relative unless absolute.
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Independently verify evidence-graph references, edge constraints, hashes, and transformation closure.
    EnzymeEvidenceGraphCheck {
        /// Graph JSON path, relative to the workspace root unless absolute.
        #[arg(long)]
        input: PathBuf,
        /// External SHA-256 pin for the exact serialized graph bytes (required for status=frozen).
        #[arg(long)]
        sha256: Option<String>,
    },
    /// Run deterministic synthetic parameter-recovery and model-discrimination fixtures.
    EnzymeModelFalsification {
        /// Output path for the machine-readable synthetic-only report.
        #[arg(long, default_value = "reports/enzyme_model_falsification_v0.json")]
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
        Commands::EnzymeBenchCheck { manifest } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let manifest_path = if manifest.is_absolute() {
                manifest
            } else {
                root.join(manifest)
            };
            enzyme_bench::check(&root, &manifest_path)?;
        }
        Commands::EnzymeCalibrationCheck { input, sha256 } => {
            enzyme_calibration::check(&input, &sha256)?;
        }
        Commands::EnzymeCalibrationSuiteCheck { manifest } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let manifest_path = if manifest.is_absolute() { manifest } else { root.join(manifest) };
            enzyme_calibration::check_suite(&root, &manifest_path)?;
        }
        Commands::EnzymeEvidenceGraphCheck { input, sha256 } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            enzyme_evidence_graph::check(&root, &input, sha256.as_deref())?;
        }
        Commands::EnzymeModelFalsification { out } => {
            enzyme_model_falsification::run(&out)?;
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
