use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod crate_status;
mod duplicate_scan;
mod interoception_actions_archive;
mod interoception_capsule_archive;
mod interoception_github_live;
mod interoception_qualification;
mod manifest;
mod rhn_sweep;
mod symthaea_interoception;

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
    /// Capture one fixed Native Interoception v0.1 local qualification gate.
    InteroceptionCaptureLocal {
        #[arg(long)]
        subject_commit: String,
        #[arg(long)]
        gate: String,
        #[arg(long, default_value = ".")]
        repo_root: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Recompute and verify a captured local qualification gate package.
    InteroceptionVerifyLocal {
        #[arg(long)]
        evidence_dir: PathBuf,
        #[arg(long)]
        repo_root: Option<PathBuf>,
    },
    /// Build and immediately verify an immutable GitHub Actions archive manifest.
    InteroceptionBuildActionsArchive {
        #[arg(long)]
        archive_dir: PathBuf,
        #[arg(long)]
        gate: String,
        #[arg(long, default_value = "Luminous-Dynamics/symthaea")]
        repository: String,
        #[arg(long)]
        repo_root: Option<PathBuf>,
    },
    /// Verify a durable GitHub Actions archive for one required workspace gate.
    InteroceptionVerifyActions {
        #[arg(long)]
        archive_dir: PathBuf,
        #[arg(long)]
        repo_root: Option<PathBuf>,
    },
    /// Compare a durable Actions archive against the live exact GitHub run attempt.
    InteroceptionVerifyActionsLive {
        #[arg(long)]
        archive_dir: PathBuf,
        #[arg(long)]
        repo_root: Option<PathBuf>,
    },
    /// Build the external path map that makes every EvidenceCapsuleManifest digest resolvable.
    InteroceptionBuildCapsuleArchive {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        evidence_root: PathBuf,
        #[arg(long)]
        preregistration: String,
        #[arg(long)]
        experiment_config: String,
        #[arg(long)]
        input_sequence: String,
        #[arg(long)]
        evidence_plane: String,
        #[arg(long)]
        repo_root: Option<PathBuf>,
    },
    /// Verify every logical/raw evidence-capsule object against the bundle's declared digest.
    InteroceptionVerifyCapsuleArchive {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        evidence_root: PathBuf,
        #[arg(long)]
        repo_root: Option<PathBuf>,
    },
    /// Inspect only the structural v0.1 bundle state. This never authorizes promotion.
    InteroceptionInspectBundle {
        #[arg(long)]
        bundle: PathBuf,
    },
    /// Verify the frozen bundle, all local/capsule bytes, archived Actions evidence, and live exact attempts.
    InteroceptionAuthorizePromotion {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        repo_root: PathBuf,
        #[arg(long)]
        evidence_root: PathBuf,
        #[arg(long)]
        local_fmt: PathBuf,
        #[arg(long)]
        local_test: PathBuf,
        #[arg(long)]
        local_clippy: PathBuf,
        #[arg(long)]
        workspace_ci: PathBuf,
        #[arg(long)]
        showroom_integrity: PathBuf,
        #[arg(long)]
        out: PathBuf,
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
        Commands::InteroceptionCaptureLocal {
            subject_commit,
            gate,
            repo_root,
            out,
        } => {
            guard_external_empty_output_dir(&repo_root, &out)?;
            let verified = interoception_qualification::capture_local_gate(
                &repo_root,
                &subject_commit,
                &gate,
                &out,
            )?;
            println!("{}", serde_json::to_string_pretty(&verified)?);
        }
        Commands::InteroceptionVerifyLocal {
            evidence_dir,
            repo_root,
        } => {
            let verified = interoception_qualification::verify_local_gate(
                &interoception_qualification::local_manifest_path(&evidence_dir),
                &interoception_qualification::local_transcript_path(&evidence_dir),
                repo_root.as_deref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&verified)?);
        }
        Commands::InteroceptionBuildActionsArchive {
            archive_dir,
            gate,
            repository,
            repo_root,
        } => {
            if let Some(root) = repo_root.as_deref() {
                guard_external_existing_dir(root, &archive_dir)?;
            }
            let verified = interoception_actions_archive::build_actions_archive(
                &archive_dir,
                &gate,
                &repository,
                repo_root.as_deref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&verified)?);
        }
        Commands::InteroceptionVerifyActions {
            archive_dir,
            repo_root,
        } => {
            let verified = interoception_qualification::verify_actions_archive(
                &archive_dir,
                repo_root.as_deref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&verified)?);
        }
        Commands::InteroceptionVerifyActionsLive {
            archive_dir,
            repo_root,
        } => {
            let verified = interoception_github_live::verify_actions_live(
                &archive_dir,
                repo_root.as_deref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&verified)?);
        }
        Commands::InteroceptionBuildCapsuleArchive {
            bundle,
            evidence_root,
            preregistration,
            experiment_config,
            input_sequence,
            evidence_plane,
            repo_root,
        } => {
            if let Some(root) = repo_root.as_deref() {
                guard_external_existing_dir(root, &evidence_root)?;
            }
            let verified = interoception_capsule_archive::build_capsule_archive_manifest(
                &bundle,
                &evidence_root,
                interoception_capsule_archive::EvidenceCapsuleLogicalPaths {
                    preregistration,
                    experiment_config,
                    input_sequence,
                    evidence_plane,
                },
                repo_root.as_deref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&verified)?);
        }
        Commands::InteroceptionVerifyCapsuleArchive {
            bundle,
            evidence_root,
            repo_root,
        } => {
            let verified = interoception_capsule_archive::verify_capsule_archive(
                &bundle,
                &evidence_root,
                repo_root.as_deref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&verified)?);
        }
        Commands::InteroceptionInspectBundle { bundle } => {
            let report = interoception_qualification::inspect_structural_bundle(&bundle)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::InteroceptionAuthorizePromotion {
            bundle,
            repo_root,
            evidence_root,
            local_fmt,
            local_test,
            local_clippy,
            workspace_ci,
            showroom_integrity,
            out,
        } => {
            guard_external_existing_dir(&repo_root, &evidence_root)?;
            guard_external_new_file(&repo_root, &out)?;
            let envelope = interoception_github_live::authorize_promotion_live(
                &bundle,
                &repo_root,
                &evidence_root,
                &local_fmt,
                &local_test,
                &local_clippy,
                &workspace_ci,
                &showroom_integrity,
                &out,
            )?;
            println!("{}", serde_json::to_string_pretty(&envelope)?);
        }
    }
    Ok(())
}

fn guard_external_empty_output_dir(
    repo_root: &std::path::Path,
    out: &std::path::Path,
) -> anyhow::Result<()> {
    let canonical_repo = std::fs::canonicalize(repo_root)?;
    let created = !out.exists();
    if created {
        std::fs::create_dir_all(out)?;
    }
    if !out.is_dir() {
        anyhow::bail!("evidence output path is not a directory: {}", out.display());
    }
    if std::fs::read_dir(out)?.next().transpose()?.is_some() {
        anyhow::bail!(
            "evidence output directory must be empty to prevent artifact overwrite: {}",
            out.display()
        );
    }
    let canonical_out = std::fs::canonicalize(out)?;
    if canonical_out.starts_with(&canonical_repo) {
        if created {
            let _ = std::fs::remove_dir_all(out);
        }
        anyhow::bail!(
            "qualification evidence output must live outside the target source checkout: {}",
            out.display()
        );
    }
    Ok(())
}

fn guard_external_existing_dir(
    repo_root: &std::path::Path,
    dir: &std::path::Path,
) -> anyhow::Result<()> {
    let canonical_repo = std::fs::canonicalize(repo_root)?;
    let canonical_dir = std::fs::canonicalize(dir)?;
    if !canonical_dir.is_dir() {
        anyhow::bail!("expected directory: {}", dir.display());
    }
    if canonical_dir.starts_with(canonical_repo) {
        anyhow::bail!(
            "qualification evidence/archive directory must live outside the target source checkout: {}",
            dir.display()
        );
    }
    Ok(())
}

fn guard_external_new_file(
    repo_root: &std::path::Path,
    out: &std::path::Path,
) -> anyhow::Result<()> {
    if out.exists() {
        anyhow::bail!("refusing to overwrite existing output file: {}", out.display());
    }
    let canonical_repo = std::fs::canonicalize(repo_root)?;
    let parent = out.parent().unwrap_or_else(|| std::path::Path::new("."));
    let canonical_parent = std::fs::canonicalize(parent)?;
    if canonical_parent.starts_with(canonical_repo) {
        anyhow::bail!(
            "promotion authorization output must live outside the target source checkout: {}",
            out.display()
        );
    }
    Ok(())
}

fn parse_list(s: &str) -> anyhow::Result<Vec<usize>> {
    s.split(',')
        .map(|item| item.parse::<usize>().map_err(anyhow::Error::from))
        .collect()
}
