use std::path::PathBuf;

use za_click_gesture_bench::{generate_fixture_bundle, render_timeline};

fn main() {
    let output_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("za-click-gesture-output"));

    match generate_fixture_bundle(&output_dir) {
        Ok(manifest) => {
            println!("ZA-CLICK-001 synthetic gesture fixtures generated");
            println!("output: {}", output_dir.display());
            println!("gesture count: {}", manifest.gestures.len());
            println!("accepted symbolic fixtures: {}", manifest.validation.iter().filter(|result| result.accepted).count());
            println!("manifest SHA-256: {}", manifest.manifest_sha256);
            println!("claim boundary: {}", manifest.claim_boundary);
            if let Some(example) = manifest.gestures.first() {
                println!("{}", render_timeline(example));
            }
        }
        Err(error) => {
            eprintln!("ZA-CLICK-001 fixture generation failed closed: {error}");
            std::process::exit(1);
        }
    }
}
