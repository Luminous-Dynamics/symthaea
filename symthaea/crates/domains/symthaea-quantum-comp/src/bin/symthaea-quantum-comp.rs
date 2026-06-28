//! Minimal dependency-free CLI for alpha research probes.
//!
//! The CLI intentionally avoids configuration parsing dependencies. It is a
//! lightweight convenience wrapper around named presets, schemas, replay plans,
//! and local gate summaries.

use std::env;
use std::process::ExitCode;

use symthaea_quantum_comp::{
    BindingProbeRunner, ClaimBoundary, ExperimentMatrixRunner, NoiseSweepRunner, ReplayPlan,
    ReplayScope, ReportTable, RunPreset, alpha9_to_alpha10_migration, audit_binding_probe,
    current_api_inventory, current_beta_readiness, current_release_manifest,
    current_validation_snapshot, current_verification_matrix, fixture_catalog, gate_local_artifact,
    known_schema_labels, named_fixture, preflight_binding_config, preflight_matrix_config,
    preflight_noise_sweep_config, supported_preset_names,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }

    let command = args.get(1).map(String::as_str).unwrap_or("help");
    let preset = args
        .get(2)
        .map(String::as_str)
        .and_then(RunPreset::from_name)
        .unwrap_or(RunPreset::Smoke);

    match command {
        "binding" => {
            let config = preset.binding_config();
            let preflight = preflight_binding_config(&config);
            println!("{}", preflight.to_text());
            if !preflight.can_run() {
                return Err("preflight failed".to_string());
            }
            let report = BindingProbeRunner::new(config)
                .map_err(|e| e.to_string())?
                .run()
                .map_err(|e| e.to_string())?;
            println!("{}", report.to_markdown());
        }
        "noise" => {
            let config = preset.noise_sweep_config();
            let preflight = preflight_noise_sweep_config(&config);
            println!("{}", preflight.to_text());
            if !preflight.can_run() {
                return Err("preflight failed".to_string());
            }
            let report = NoiseSweepRunner::new(config)
                .map_err(|e| e.to_string())?
                .run()
                .map_err(|e| e.to_string())?;
            println!("{}", report.to_markdown());
        }
        "matrix" => {
            let config = preset.matrix_config();
            let preflight = preflight_matrix_config(&config);
            println!("{}", preflight.to_text());
            if !preflight.can_run() {
                return Err("preflight failed".to_string());
            }
            let report = ExperimentMatrixRunner::new(config)
                .map_err(|e| e.to_string())?
                .run()
                .map_err(|e| e.to_string())?;
            println!("{}", report.to_markdown());
        }
        "presets" => {
            for name in supported_preset_names() {
                println!("{name}");
            }
        }
        "schemas" => {
            for label in known_schema_labels() {
                println!("{label}");
            }
        }
        "fixtures" => {
            for fixture in fixture_catalog() {
                println!("{}", fixture.to_text());
            }
        }
        "replay" => {
            let scope_name = args.get(2).map(String::as_str).unwrap_or("smoke");
            let scope = ReplayScope::from_name(scope_name)
                .ok_or_else(|| format!("unknown replay scope `{scope_name}`"))?;
            let plan = ReplayPlan::for_scope(scope);
            println!("{}", plan.to_markdown());
        }
        "gate" => {
            let fixture_name = args.get(2).map(String::as_str).unwrap_or("smoke-binding");
            let fixture = named_fixture(fixture_name)
                .ok_or_else(|| format!("unknown fixture `{fixture_name}`"))?;
            let preflight = preflight_binding_config(&fixture.config);
            if !preflight.can_run() {
                println!("{}", preflight.to_text());
                return Err("preflight failed".to_string());
            }
            let report = BindingProbeRunner::new(fixture.config)
                .map_err(|e| e.to_string())?
                .run()
                .map_err(|e| e.to_string())?;
            let audit = audit_binding_probe(&report, ClaimBoundary::LocalSimulation);
            let replay = ReplayPlan::for_scope(ReplayScope::Smoke);
            let gate = gate_local_artifact(&preflight, &audit, Some(&fixture), &replay);
            println!("{}", gate.to_text());
        }
        "inventory" => {
            let inventory = current_api_inventory();
            println!("{}", inventory.to_markdown());
        }
        "manifest" => {
            let manifest = current_release_manifest();
            println!("{}", manifest.to_markdown());
        }
        "verify-matrix" => {
            let matrix = current_verification_matrix();
            println!("{}", matrix.to_markdown());
        }
        "migration" => {
            let guide = alpha9_to_alpha10_migration();
            println!("{}", guide.to_markdown());
        }
        "beta" => {
            let report = current_beta_readiness();
            println!("{}", report.to_markdown());
        }
        "snapshot" => {
            let snapshot = current_validation_snapshot();
            println!("{}", snapshot.to_markdown());
        }
        "help" => print_help(),
        other => return Err(format!("unknown command `{other}`; run with --help")),
    }

    Ok(())
}

fn print_help() {
    println!(
        "symthaea-quantum-comp alpha CLI\n\nUsage:\n  cargo run --bin symthaea-quantum-comp -- <command> [preset|scope|fixture]\n\nCommands:\n  binding        run a binding probe\n  noise          run a noise sweep\n  matrix         run a dimension-by-noise matrix\n  presets        list supported run presets\n  schemas        list alpha.10 schema labels\n  fixtures       list named local fixtures\n  replay         print a replay plan for a scope\n  gate           run a local release gate for a fixture\n  inventory      print API inventory and alpha stability catalog\n  manifest       print alpha release manifest and blocked claims\n  verify-matrix  print the alpha.10 verification matrix\n  migration      print the alpha.9 to alpha.10 migration guide\n  beta           print conservative beta-readiness status\n  snapshot       print combined validation snapshot\n  help           print this help\n\nPresets/scopes:\n  smoke            tiny CI/smoke run\n  local-research   laptop-sized local research run\n  pilot-matrix     broader pilot matrix\n\nFixtures:\n  smoke-binding\n  demo-binding\n  pilot-binding\n\nClaim boundary:\n  This CLI runs local research probes only. It does not claim quantum consciousness, quantum advantage, physical backend execution, or Mycelix attestation."
    );
}
