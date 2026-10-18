use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod crate_status;
mod duplicate_scan;
mod manifest;
mod rhn_sweep;
mod twowheeler_benchmark;
mod two_wheeler_wheel_demand;
mod two_wheeler_trace_energy;
mod powertrain_energy_ledger;
mod two_wheeler_validation_scorecard;

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
    /// Recompute decisions in the frozen synthetic two-wheeler benchmark fixture.
    /// A pass establishes fixture-rule conformance only, not physical qualification.
    TwoWheelerBenchmark {
        #[arg(long, default_value = "docs/research/data/two_wheeler_a0_benchmark_v0_1.json")]
        input: PathBuf,
        /// Optional path for a JSON report; otherwise report to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Verify the frozen synthetic powertrain energy-ledger controls.
    /// A pass establishes accounting-contract conformance only, not physical efficiency.
    PowertrainEnergyLedger {
        #[arg(long, default_value = "docs/research/data/two_wheeler_powertrain_energy_ledger_v0_1.json")]
        input: PathBuf,
        /// Optional path for a JSON report; otherwise report to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Recompute the frozen synthetic pointwise two-wheeler wheel-demand oracle.
    /// A pass is not a validated vehicle simulation or powertrain efficiency result.
    TwoWheelerWheelDemand {
        #[arg(long, default_value = "docs/research/data/two_wheeler_wheel_demand_v0_1.json")]
        input: PathBuf,
        /// Optional path for a JSON report; otherwise report to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Integrate signed wheel work over the synthetic trace fixture.
    /// Negative wheel work is not treated as regenerative energy.
    TwoWheelerTraceEnergy {
        #[arg(long, default_value = "docs/research/data/two_wheeler_trace_energy_v0_1.json")]
        input: PathBuf,
        /// Optional path for a JSON report; otherwise report to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Score matched, identity-checked two-wheeler energy pairs; synthetic conformance is not physical validation.
    TwoWheelerValidationScorecard {
        #[arg(long, default_value = "docs/research/data/two_wheeler_validation_scorecard_v0_1.json")]
        input: PathBuf,
        /// Optional path for a JSON report; otherwise report to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
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
        Commands::TwoWheelerBenchmark { input, out } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let input = if input.is_absolute() {
                input
            } else {
                root.join(input)
            };
            let out = out.map(|path| if path.is_absolute() { path } else { root.join(path) });
            twowheeler_benchmark::run(&input, out.as_deref())?;
        }
        Commands::PowertrainEnergyLedger { input, out } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let input = if input.is_absolute() { input } else { root.join(input) };
            let out = out.map(|path| if path.is_absolute() { path } else { root.join(path) });
            powertrain_energy_ledger::run(&input, out.as_deref())?;
        }
        Commands::TwoWheelerWheelDemand { input, out } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let input = if input.is_absolute() { input } else { root.join(input) };
            let out = out.map(|path| if path.is_absolute() { path } else { root.join(path) });
            two_wheeler_wheel_demand::run(&input, out.as_deref())?;
        }
        Commands::TwoWheelerTraceEnergy { input, out } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let input = if input.is_absolute() { input } else { root.join(input) };
            let out = out.map(|path| if path.is_absolute() { path } else { root.join(path) });
            two_wheeler_trace_energy::run(&input, out.as_deref())?;
        }
        Commands::TwoWheelerValidationScorecard { input, out } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let input = if input.is_absolute() { input } else { root.join(input) };
            let out = out.map(|path| if path.is_absolute() { path } else { root.join(path) });
            two_wheeler_validation_scorecard::run(&input, out.as_deref())?;
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
