use muse_observer_calibration::{
    run_calibration_v1, verify_calibration_bundle_v1, CalibrationContractV1,
    CalibrationDatasetV1, CalibrationBundleV1,
};
use std::{env, fs, process};

fn usage() -> &'static str {
    "muse-observer-calibration evaluate DATASET.json CONTRACT.json\n\
     muse-observer-calibration verify DATASET.json CONTRACT.json BUNDLE.json\n\
     Raw salience is calibrated only on the fit split; threshold selection and held-out groups remain separate."
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && matches!(args[1].as_str(), "-h" | "--help") {
        println!("{}", usage());
        return Ok(0);
    }
    match args.get(1).map(String::as_str) {
        Some("evaluate") if args.len() == 4 => {
            let dataset: CalibrationDatasetV1 = serde_json::from_slice(
                &fs::read(&args[2]).map_err(|error| format!("read dataset: {error}"))?
            ).map_err(|error| format!("dataset JSON: {error}"))?;
            let contract: CalibrationContractV1 = serde_json::from_slice(
                &fs::read(&args[3]).map_err(|error| format!("read contract: {error}"))?
            ).map_err(|error| format!("contract JSON: {error}"))?;
            let bundle = run_calibration_v1(&dataset, &contract)?;
            println!("{}", serde_json::to_string_pretty(&bundle).map_err(|error| error.to_string())?);
            Ok(0)
        }
        Some("verify") if args.len() == 5 => {
            let dataset: CalibrationDatasetV1 = serde_json::from_slice(
                &fs::read(&args[2]).map_err(|error| format!("read dataset: {error}"))?
            ).map_err(|error| format!("dataset JSON: {error}"))?;
            let contract: CalibrationContractV1 = serde_json::from_slice(
                &fs::read(&args[3]).map_err(|error| format!("read contract: {error}"))?
            ).map_err(|error| format!("contract JSON: {error}"))?;
            let bundle: CalibrationBundleV1 = serde_json::from_slice(
                &fs::read(&args[4]).map_err(|error| format!("read bundle: {error}"))?
            ).map_err(|error| format!("bundle JSON: {error}"))?;
            verify_calibration_bundle_v1(&bundle, &dataset, &contract)?;
            println!("calibration bundle replay verified");
            Ok(0)
        }
        _ => {
            println!("{}", usage());
            Ok(2)
        }
    }
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("muse-observer-calibration: {error}");
            process::exit(4);
        }
    }
}
