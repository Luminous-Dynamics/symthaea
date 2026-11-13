use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod crate_status;
mod duplicate_scan;
mod manifest;
mod plasma_energy_audit;
mod plasma_surface_bondability;
mod plasma_virtual_metrology;
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
    /// Offline, source-bound audit of plasma energy corpora and thermal-store accounting fixtures.
    #[command(name = "plasma-energy-audit")]
    PlasmaEnergyAudit {
        #[arg(long, default_value = "data/research/plasma_energy_reference_cases_v0_1.json")]
        plasma_sources: PathBuf,
        #[arg(long, default_value = "data/research/hts_fusion_magnet_reference_cases_v0_1.json")]
        hts_sources: PathBuf,
        #[arg(long, default_value = "data/research/thermal_store_accounting_fixtures_v0_1.json")]
        thermal_fixtures: PathBuf,
        /// Emit stable structured JSON instead of the human-readable summary.
        #[arg(long)]
        json: bool,
    },

    /// Offline evaluator for synthetic plasma surface bondability and preservation gates.
    #[command(name = "plasma-surface-bondability-audit")]
    PlasmaSurfaceBondabilityAudit {
        #[arg(long, default_value = "data/research/plasma_surface_bondability_fixture_v0_1.json")]
        fixture: PathBuf,
        /// Emit stable structured JSON instead of the human-readable summary.
        #[arg(long)]
        json: bool,
    },

    /// Offline validator for synthetic plasma telemetry-to-product virtual metrology.
    #[command(name = "plasma-virtual-metrology-audit")]
    PlasmaVirtualMetrologyAudit {
        #[arg(long, default_value = "data/research/plasma_virtual_metrology_fixture_v0_1.json")]
        fixture: PathBuf,
        /// Emit stable structured JSON instead of the human-readable summary.
        #[arg(long)]
        json: bool,
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
        Commands::PlasmaEnergyAudit {
            plasma_sources,
            hts_sources,
            thermal_fixtures,
            json,
        } => {
            let report = plasma_energy_audit::run_audit(
                &plasma_sources,
                &hts_sources,
                &thermal_fixtures,
            )?;
            plasma_energy_audit::print_report(&report, json)?;
        }
        Commands::PlasmaSurfaceBondabilityAudit { fixture, json } => {
            let report = plasma_surface_bondability::run_audit(&fixture)?;
            plasma_surface_bondability::print_report(&report, json)?;
        }
        Commands::PlasmaVirtualMetrologyAudit { fixture, json } => {
            let report = plasma_virtual_metrology::run_audit(&fixture)?;
            plasma_virtual_metrology::print_report(&report, json)?;
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
