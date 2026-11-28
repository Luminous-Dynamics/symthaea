use mel_hear_011_auditory_substrate::validate_fixture;
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let fixture_bytes = fs::read(root.join("fixtures/auditory-evidence-v0.1.json"))?;
    let source_asset = fs::read(root.join("fixtures/synthetic-tone-f32le-hex.txt"))?;
    let fixture: Value = serde_json::from_slice(&fixture_bytes)?;
    let report = validate_fixture(&fixture, &source_asset).map_err(std::io::Error::other)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
