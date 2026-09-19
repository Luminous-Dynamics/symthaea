use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[cfg(test)]
mod cargo_adapter_commit;
#[cfg(test)]
mod cargo_adapter_postflight;
#[cfg(test)]
mod cargo_adapter_state;
mod cargo_execution_attempt;
mod cargo_execution_contract;
mod crate_status;
mod duplicate_scan;
mod manifest;
mod repository_effect_policy;
mod repository_snapshot;
mod repository_snapshot_diff;
mod repository_snapshot_receipt;
mod repository_snapshot_verify;
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
    /// Snapshot exact worktree source bytes for a declared repository scope.
    RepositorySnapshot {
        /// Explicit ignored file/symlink inputs to bind in addition to tracked and non-ignored untracked files.
        #[arg(long = "include-ignored")]
        include_ignored: Vec<PathBuf>,
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Recompute the declared source scope and fail if it drifted from a frozen snapshot receipt.
    RepositorySnapshotVerify {
        /// Frozen snapshot emitted by `repository-snapshot`.
        #[arg(long)]
        snapshot: PathBuf,
    },
    /// Compare two validated repository-source receipts and emit a typed deterministic transition.
    RepositorySnapshotDiff {
        /// Base repository-source snapshot receipt.
        #[arg(long)]
        base: PathBuf,
        /// Head repository-source snapshot receipt.
        #[arg(long)]
        head: PathBuf,
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Evaluate an exact source-transition policy over two validated repository snapshots.
    RepositoryEffectCheck {
        /// Base repository-source snapshot receipt.
        #[arg(long)]
        base: PathBuf,
        /// Head repository-source snapshot receipt.
        #[arg(long)]
        head: PathBuf,
        /// Strict repository effect-policy JSON.
        #[arg(long)]
        policy: PathBuf,
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Canonicalize and identify a Cargo execution intent for assurance admission.
    CargoExecutionIntent {
        /// Strict Cargo execution-intent input JSON spec.
        #[arg(long)]
        spec: PathBuf,
        /// Validated repository-source subject to bind before execution.
        #[arg(long = "pre-snapshot")]
        pre_snapshot: PathBuf,
        /// Exact repository effect policy to bind before execution.
        #[arg(long = "effect-policy")]
        effect_policy: PathBuf,
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Cross-bind a frozen Cargo intent to observed build/effect evidence.
    CargoExecutionResult {
        /// Frozen Cargo execution intent emitted by `cargo-execution-intent`.
        #[arg(long)]
        intent: PathBuf,
        /// Frozen repository-source subject verified before execution.
        #[arg(long = "pre-snapshot")]
        pre_snapshot: PathBuf,
        /// Frozen repository-source subject captured after execution.
        #[arg(long = "post-snapshot")]
        post_snapshot: PathBuf,
        /// Raw Cargo build observation emitted by the Cargo observation layer.
        #[arg(long)]
        observation: PathBuf,
        /// Exact repository effect policy admitted before execution.
        #[arg(long = "effect-policy")]
        effect_policy: PathBuf,
        /// Optional companion Git worktree-state subject after execution.
        #[arg(long = "git-worktree-state-after")]
        git_worktree_state_after: Option<String>,
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
        Commands::RepositorySnapshot {
            include_ignored,
            output,
        } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask always lives one level below the workspace root")
                .to_path_buf();
            repository_snapshot::run(&root, include_ignored, output)?;
        }
        Commands::RepositorySnapshotVerify { snapshot } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask always lives one level below the workspace root")
                .to_path_buf();
            repository_snapshot_verify::run(&root, &snapshot)?;
        }
        Commands::RepositorySnapshotDiff { base, head, output } => {
            repository_snapshot_diff::run(&base, &head, output)?;
        }
        Commands::RepositoryEffectCheck {
            base,
            head,
            policy,
            output,
        } => {
            repository_effect_policy::run(&base, &head, &policy, output)?;
        }
        Commands::CargoExecutionIntent {
            spec,
            pre_snapshot,
            effect_policy,
            output,
        } => {
            cargo_execution_contract::run_intent(&spec, &pre_snapshot, &effect_policy, output)?;
        }
        Commands::CargoExecutionResult {
            intent,
            pre_snapshot,
            post_snapshot,
            observation,
            effect_policy,
            git_worktree_state_after,
            output,
        } => {
            cargo_execution_contract::run_result(
                &intent,
                &pre_snapshot,
                &post_snapshot,
                &observation,
                &effect_policy,
                git_worktree_state_after,
                output,
            )?;
        }
    }
    Ok(())
}

fn parse_list(s: &str) -> anyhow::Result<Vec<usize>> {
    s.split(',')
        .map(|item| item.parse::<usize>().map_err(anyhow::Error::from))
        .collect()
}
