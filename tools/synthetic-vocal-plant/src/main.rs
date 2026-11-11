use std::path::PathBuf;

use synthetic_vocal_plant::{default_output_dir, generate_suite};

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
        }
        Err(error) => {
            eprintln!("synthetic vocal plant failed: {error}");
            std::process::exit(1);
        }
    }
}
