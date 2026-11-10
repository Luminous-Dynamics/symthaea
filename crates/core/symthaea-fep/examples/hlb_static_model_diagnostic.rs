//! Run the exact bounded static-versus-rectangular HLB diagnostic fixture.
//!
//! This emits a source-bound JSON report. It is a synthetic policy-semantics
//! check, not a measured biological or deployment result.

use std::process::ExitCode;

use symthaea_fep::static_model_diagnostic::{
    StaticModelDiagnosticConfig, evaluate_static_model_diagnostic_problem,
    symmetric_calibration_fixture,
};

fn main() -> ExitCode {
    let source_sha = match std::env::var("SYMTHAEA_SOURCE_SHA") {
        Ok(value) => value,
        Err(_) => {
            eprintln!("SYMTHAEA_SOURCE_SHA must contain the exact 40-character source commit SHA");
            return ExitCode::FAILURE;
        }
    };

    let problem = symmetric_calibration_fixture();
    let evaluation = match evaluate_static_model_diagnostic_problem(
        &problem,
        StaticModelDiagnosticConfig::default(),
        &source_sha,
    ) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("HLB static-model diagnostic oracle failed: {error}");
            return ExitCode::FAILURE;
        }
    };

    match serde_json::to_string_pretty(&evaluation) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("failed to serialize static-model diagnostic receipt: {error}");
            ExitCode::FAILURE
        }
    }
}
