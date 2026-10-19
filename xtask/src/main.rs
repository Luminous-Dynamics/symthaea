use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod crate_status;
mod duplicate_scan;
mod manifest;
mod rhn_sweep;
mod tire_monitoring_intake;
mod mobility_telemetry_manifest;

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
    /// Validate the vendored CC BY 4.0 tire-monitoring CSV shards; no tire performance is qualified.
    TireMonitoringIntake {
        #[arg(long, default_value = "data/third_party/tire-monitoring-dataset")]
        input_dir: PathBuf,
        /// Optional path for a JSON report; otherwise report to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Validate cross-vehicle telemetry manifest structure, artifact digests and claim eligibility.
    MobilityTelemetryManifest {
        #[arg(long, default_value = "docs/research/data/mobility_telemetry_manifest_v0_1.json")]
        input: PathBuf,
        /// Optional path for a JSON report; otherwise report to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run the frozen mobility telemetry fault matrix; exits non-zero on any false accept/reject.
    MobilityTelemetryBenchmark {
        #[arg(long, default_value = "docs/research/data/mobility_telemetry_manifest_v0_1.json")]
        manifest: PathBuf,
        #[arg(long, default_value = "docs/research/data/mobility_telemetry_fault_matrix_v0_1.json")]
        matrix: PathBuf,
        /// Exact full Git commit ID for the code being benchmarked (40 or 64 hex characters).
        #[arg(long)]
        source_commit: String,
        /// Optional path for a JSON report; otherwise report to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Verify canonical digest and internal consistency of a saved mobility telemetry benchmark report.
    MobilityTelemetryVerifyReport {
        #[arg(long, default_value = "reports/mobility_telemetry_benchmark.json")]
        input: PathBuf,
    },
    /// Validate an append-only multi-rate raw telemetry CSV against the declared manifest channels.
    MobilityTelemetryTrace {
        #[arg(long, default_value = "docs/research/data/mobility_telemetry_manifest_v0_1.json")]
        manifest: PathBuf,
        #[arg(long, default_value = "docs/research/data/mobility_telemetry_trace_v0_1.csv")]
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
        Commands::TireMonitoringIntake { input_dir, out } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let input_dir = if input_dir.is_absolute() { input_dir } else { root.join(input_dir) };
            let out = out.map(|path| if path.is_absolute() { path } else { root.join(path) });
            tire_monitoring_intake::run(&input_dir, out.as_deref())?;
        }
        Commands::MobilityTelemetryManifest { input, out } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let input = if input.is_absolute() { input } else { root.join(input) };
            let out = out.map(|path| if path.is_absolute() { path } else { root.join(path) });
            mobility_telemetry_manifest::run(&input, out.as_deref())?;
        }
        Commands::MobilityTelemetryBenchmark { manifest, matrix, source_commit, out } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let out = out.map(|path| if path.is_absolute() { path } else { root.join(path) });
            let git_head = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(&root)
                .output()?;
            if !git_head.status.success() {
                anyhow::bail!("could not verify checkout HEAD via git rev-parse HEAD");
            }
            let actual_head = String::from_utf8(git_head.stdout)?.trim().to_owned();
            if actual_head != source_commit {
                anyhow::bail!(
                    "--source-commit {} does not match current checkout HEAD {}",
                    source_commit,
                    actual_head
                );
            }
            let tracked_diff = std::process::Command::new("git")
                .args(["diff", "--quiet", "HEAD", "--"])
                .current_dir(&root)
                .status()?;
            if !tracked_diff.success() {
                anyhow::bail!(
                    "refusing benchmark on a dirty tracked worktree/index; commit the source and rerun"
                );
            }
            let status = std::process::Command::new("git")
                .args(["status", "--porcelain", "--untracked-files=all"])
                .current_dir(&root)
                .output()?;
            if !status.status.success() {
                anyhow::bail!("could not verify the benchmark checkout status");
            }
            let status_text = String::from_utf8(status.stdout)?;
            for line in status_text.lines() {
                let changed_path = line.get(3..).unwrap_or_default().trim();
                let is_requested_report = out.as_ref().is_some_and(|output_path| {
                    output_path.strip_prefix(&root).is_ok_and(|relative| {
                        relative.to_string_lossy().replace('\\', "/") == changed_path
                    })
                });
                if !is_requested_report {
                    anyhow::bail!(
                        "refusing benchmark on a checkout with untracked or changed files: {}; commit source files first",
                        line
                    );
                }
            }

            // Capture the executing environment rather than trusting a caller-supplied
            // string that could falsely label the toolchain used for this report.
            let rustc = std::process::Command::new("rustc").arg("-Vv").output()?;
            if !rustc.status.success() {
                anyhow::bail!("rustc -Vv failed; cannot attest runtime toolchain");
            }
            let cargo = std::process::Command::new("cargo").arg("--version").output()?;
            if !cargo.status.success() {
                anyhow::bail!("cargo --version failed; cannot attest runtime toolchain");
            }
            let toolchain_id = format!(
                "{}\n{}",
                String::from_utf8(rustc.stdout)?.trim(),
                String::from_utf8(cargo.stdout)?.trim()
            );

            let manifest = if manifest.is_absolute() { manifest } else { root.join(manifest) };
            let matrix = if matrix.is_absolute() { matrix } else { root.join(matrix) };
            mobility_telemetry_manifest::run_benchmark(
                &manifest, &matrix, &source_commit, &toolchain_id, out.as_deref()
            )?;
        }
        Commands::MobilityTelemetryVerifyReport { input } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let input = if input.is_absolute() { input } else { root.join(input) };
            mobility_telemetry_manifest::run_verify_report(&input)?;
        }
        Commands::MobilityTelemetryTrace { manifest, input, out } => {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let manifest = if manifest.is_absolute() { manifest } else { root.join(manifest) };
            let input = if input.is_absolute() { input } else { root.join(input) };
            let out = out.map(|path| if path.is_absolute() { path } else { root.join(path) });
            mobility_telemetry_manifest::run_trace(&manifest, &input, out.as_deref())?;
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
