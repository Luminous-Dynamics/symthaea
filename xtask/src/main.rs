use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod cargo_context;
mod cargo_graph;
mod cargo_impact;
mod cargo_invocation;
mod cargo_observation;
mod cargo_observation_verify;
mod crate_status;
mod duplicate_scan;
mod manifest;
mod rhn_sweep;
mod rust_diagnostics;

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
    /// Emit a canonical, content-addressed snapshot of the locked Cargo resolve graph.
    CargoGraph {
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare two Cargo graph snapshots for conservative per-package dependency impact.
    CargoImpact {
        /// Base snapshot emitted by `cargo xtask cargo-graph`.
        #[arg(long)]
        base: PathBuf,
        /// Head snapshot emitted by `cargo xtask cargo-graph`.
        #[arg(long)]
        head: PathBuf,
        /// Workspace package name, stable ID, or workspace-relative manifest path.
        #[arg(long = "package", required = true)]
        packages: Vec<String>,
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Canonicalize a declared Cargo build-context spec and emit stable context/invocation IDs.
    CargoContext {
        /// Strict JSON build-context specification.
        #[arg(long)]
        spec: PathBuf,
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Parse already-tokenized Cargo argv into a declared build context, failing closed on unknown semantics.
    CargoInvocation {
        /// Strict JSON containing exact raw argv plus non-argv toolchain/config fingerprints.
        #[arg(long)]
        spec: PathBuf,
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Bind stable Cargo JSON-message observations to an exact context and invocation.
    CargoObservation {
        /// Strict JSON containing context/invocation/toolchain/exit identities.
        #[arg(long)]
        spec: PathBuf,
        /// Exact stdout transcript from Cargo `--message-format=json`.
        #[arg(long)]
        messages: PathBuf,
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Re-derive a stored Cargo observation from its exact transcript and fail on any mismatch.
    CargoObservationVerify {
        /// Frozen Cargo observation receipt emitted by `cargo-observation`.
        #[arg(long)]
        observation: PathBuf,
        /// Exact Cargo JSONL transcript bound by the receipt.
        #[arg(long)]
        messages: PathBuf,
        /// Write verification JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Extract rustc diagnostic spans and compiler-authored repair suggestions from a bound Cargo observation.
    RustDiagnostics {
        /// Strict JSON binding this extraction to an observation ID + stdout digest.
        #[arg(long)]
        spec: PathBuf,
        /// Exact Cargo JSONL transcript previously bound by `cargo-observation`.
        #[arg(long)]
        messages: PathBuf,
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
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
        Commands::CargoGraph { output } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask always lives one level below the workspace root")
                .to_path_buf();
            cargo_graph::run(&root, output)?;
        }
        Commands::CargoImpact {
            base,
            head,
            packages,
            output,
        } => {
            cargo_impact::run(&base, &head, packages, output)?;
        }
        Commands::CargoContext { spec, output } => {
            cargo_context::run(&spec, output)?;
        }
        Commands::CargoInvocation { spec, output } => {
            cargo_invocation::run(&spec, output)?;
        }
        Commands::CargoObservation {
            spec,
            messages,
            output,
        } => {
            cargo_observation::run(&spec, &messages, output)?;
        }
        Commands::CargoObservationVerify {
            observation,
            messages,
            output,
        } => {
            cargo_observation_verify::run(&observation, &messages, output)?;
        }
        Commands::RustDiagnostics {
            spec,
            messages,
            output,
        } => {
            rust_diagnostics::run(&spec, &messages, output)?;
        }
    }
    Ok(())
}

fn parse_list(s: &str) -> anyhow::Result<Vec<usize>> {
    s.split(',')
        .map(|item| item.parse::<usize>().map_err(anyhow::Error::from))
        .collect()
}
