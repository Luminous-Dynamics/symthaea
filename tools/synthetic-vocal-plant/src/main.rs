use std::path::PathBuf;

use synthetic_vocal_plant::{default_output_dir, generate_suite};
use synthetic_vocal_plant::observer::generate_observer_bundle;

fn main() {
    let output_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_output_dir);

    match generate_suite(&output_dir) {
        Ok(manifest) => {
            println!("SYNTH-VOCAL-001 suite generated");
            println!("output: {}", output_dir.display());
            println!("scenarios: {}", manifest.scenarios.len());
            println!("renderer: {}", manifest.renderer_backend_id);
            println!("baseline PCM SHA-256: {}", manifest.baseline_pcm_sha256);
            println!(
                "claim boundary: {}",
                manifest.renderer_claim_boundary
            );
            match generate_observer_bundle(&output_dir) {
                Ok(observer) => {
                    println!("independent observations: {}", observer.observation_record_count);
                    println!("observer manifest: observer-manifest.json");
                    println!("calibration exact PCM repeatability: {}", observer.calibration.exact_pcm_repeatability);
                    println!("calibration/confirmatory split disjoint: {}", observer.calibration.split_identity_disjoint);
                }
                Err(error) => {
                    eprintln!("independent observer failed: {error}");
                    std::process::exit(1);
                }
            }
        }
        Err(error) => {
            eprintln!("synthetic vocal plant failed: {error}");
            std::process::exit(1);
        }
    }
}
