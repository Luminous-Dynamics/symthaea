use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod crate_status;
mod cropfoundry_csv_audit;
mod cropfoundry_nutrition_accounting;
mod cropfoundry_nutrition_portfolio;
mod cropfoundry_postharvest_audit;
mod cropfoundry_robust_policy;
mod cropfoundry_trial_power;
mod cropfoundry_validate;
mod cropfoundry_water_policy;
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
    /// Validate the CropFoundry research benchmark and data-source contracts without running crop models.
    CropfoundryValidate {
        /// Workspace root to validate.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Audit a locally supplied CropFoundry CSV without training a model or authorizing data use.
    CropfoundryAuditCsv {
        /// Path to the local CSV file. No data is downloaded by this command.
        #[arg(long)]
        input: PathBuf,
        /// Source-manifest identifier.
        #[arg(long, default_value = "DS-010")]
        source_id: String,
        /// Canonical dataset source URI.
        #[arg(long, default_value = "https://zenodo.org/records/16745911")]
        source_uri: String,
        /// Single-byte CSV delimiter.
        #[arg(long, default_value = ";")]
        delimiter: char,
        /// Optional expected data-record count from a pinned file-level source note.
        #[arg(long)]
        expected_data_records: Option<usize>,
        /// Optional SHA-256 from an independently trusted source listing or checksum file.
        #[arg(long)]
        expected_sha256: Option<String>,
        /// Optional path to write the JSON audit receipt.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run the deterministic synthetic irrigation-policy harness; no crop-performance claim.
    CropfoundryWaterPolicyDemo {
        /// Optional path to write the JSON scenario receipt.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Audit local lot/stage mass-balance data; does not validate food safety or authorize training.
    CropfoundryPostharvestAudit {
        /// Path to a local CSV; no network request is made.
        #[arg(long)]
        input: PathBuf,
        /// Explicit source/dataset identifier for provenance in the receipt.
        #[arg(long)]
        source_id: String,
        /// Canonical URL or URI for the specific source/artifact.
        #[arg(long)]
        source_uri: String,
        /// Single-byte CSV delimiter.
        #[arg(long, default_value = ";")]
        delimiter: char,
        /// Absolute mass-balance tolerance in kg.
        #[arg(long, default_value = "0.02")]
        absolute_tolerance_kg: f64,
        /// Relative mass-balance tolerance as a fraction of input mass.
        #[arg(long, default_value = "0.01")]
        relative_tolerance: f64,
        /// Optional expected SHA-256 from an independent trusted source.
        #[arg(long)]
        expected_sha256: Option<String>,
        /// Optional path for the JSON audit receipt.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Account for nutrient delivery from explicit, source-backed JSON; no nutrition adequacy claim.
    CropfoundryNutritionAccounting {
        /// Path to a JSON input matching cropfoundry-nutrition-accounting-input-v1 (v0 point-only inputs are accepted).
        #[arg(long)]
        input: PathBuf,
        /// Optional path for the JSON accounting receipt.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Compare complete nutrient delivery across options without double-counting batch resources.
    CropfoundryNutritionPortfolio {
        /// Path to the portfolio JSON envelope; records use the row-level nutrition accounting schema.
        #[arg(long)]
        input: PathBuf,
        /// Optional path for the JSON portfolio receipt.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run tail-risk, Pareto and preference-sensitivity diagnostics over the synthetic irrigation suite.
    CropfoundryRobustPolicyDemo {
        /// Optional path to write the JSON robustness receipt.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Estimate field-trial sample size from explicit pilot assumptions; this is a preflight, not a final analysis.
    CropfoundryTrialPowerPlan {
        /// Input JSON file matching cropfoundry-trial-power-input-v0.
        #[arg(long)]
        input: PathBuf,
        /// Optional path to write the SHA-256-bound planning receipt.
        #[arg(long)]
        out: Option<PathBuf>,
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
        Commands::CropfoundryValidate { root } => {
            let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("xtask lives one level below the workspace root")
                .to_path_buf();
            let root = if root.is_absolute() {
                root
            } else {
                workspace_root.join(root)
            };
            cropfoundry_validate::run(&root)?;
        }
        Commands::CropfoundryAuditCsv {
            input,
            source_id,
            source_uri,
            delimiter,
            expected_data_records,
            expected_sha256,
            out,
        } => {
            cropfoundry_csv_audit::run(
                input,
                source_id,
                source_uri,
                delimiter,
                expected_data_records,
                expected_sha256,
                out,
            )?;
        }
        Commands::CropfoundryWaterPolicyDemo { out } => {
            cropfoundry_water_policy::run(out)?;
        }
        Commands::CropfoundryPostharvestAudit {
            input,
            source_id,
            source_uri,
            delimiter,
            absolute_tolerance_kg,
            relative_tolerance,
            expected_sha256,
            out,
        } => {
            cropfoundry_postharvest_audit::run(
                input,
                source_id,
                source_uri,
                delimiter,
                absolute_tolerance_kg,
                relative_tolerance,
                expected_sha256,
                out,
            )?;
        }
        Commands::CropfoundryNutritionAccounting { input, out } => {
            cropfoundry_nutrition_accounting::run(input, out)?;
        }
        Commands::CropfoundryNutritionPortfolio { input, out } => {
            cropfoundry_nutrition_portfolio::run(input, out)?;
        }
        Commands::CropfoundryRobustPolicyDemo { out } => {
            cropfoundry_robust_policy::run(out)?;
        }
        Commands::CropfoundryTrialPowerPlan { input, out } => {
            cropfoundry_trial_power::run(input, out)?;
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
