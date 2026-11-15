use std::{env, fs, path::PathBuf, process};

use serde_json::json;
use symthaea_marine_geometry::cad_exchange::{
    verify_step_translation_artifact, CadTranslationReceipt,
};
use symthaea_marine_geometry::step_header::StepHeaderProfile;

const USAGE: &str = concat!(
    "Usage: marine-cad-translation-verify <step-file> <receipt.json> <header-profile.json>\n",
    "Verifies Part 21 envelope, exact FILE_SCHEMA profile and receipt-to-file digest binding."
);

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let step_path = args.next().map(PathBuf::from);
    let receipt_path = args.next().map(PathBuf::from);
    let profile_path = args.next().map(PathBuf::from);
    if args.next().is_some()
        || step_path.is_none()
        || receipt_path.is_none()
        || profile_path.is_none()
    {
        return Err(USAGE.to_string());
    }

    let step_path = step_path.unwrap();
    let receipt_path = receipt_path.unwrap();
    let profile_path = profile_path.unwrap();

    let step_bytes = fs::read(&step_path)
        .map_err(|error| format!("cannot read STEP file {}: {error}", step_path.display()))?;
    let receipt_bytes = fs::read(&receipt_path)
        .map_err(|error| format!("cannot read receipt {}: {error}", receipt_path.display()))?;
    let profile_bytes = fs::read(&profile_path)
        .map_err(|error| format!("cannot read header profile {}: {error}", profile_path.display()))?;

    let receipt: CadTranslationReceipt = serde_json::from_slice(&receipt_bytes)
        .map_err(|error| format!("invalid CAD translation receipt: {error}"))?;
    let profile: StepHeaderProfile = serde_json::from_slice(&profile_bytes)
        .map_err(|error| format!("invalid STEP header profile: {error}"))?;

    match verify_step_translation_artifact(&step_bytes, &receipt, &profile) {
        Ok(qualification) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&qualification)
                    .map_err(|error| format!("cannot serialize qualification: {error}"))?
            );
            Ok(0)
        }
        Err(violations) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "not-qualified",
                    "violations": violations
                }))
                .map_err(|error| format!("cannot serialize violations: {error}"))?
            );
            Ok(1)
        }
    }
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            process::exit(2);
        }
    }
}
