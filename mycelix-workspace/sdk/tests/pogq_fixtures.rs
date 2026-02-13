use std::fs;
use std::path::Path;

use serde::Deserialize;

use mycelix_sdk::matl::ProofOfGradientQuality;

#[derive(Debug, Deserialize)]
struct PogqFixtureCase {
    name: String,
    quality: f64,
    consistency: f64,
    entropy: f64,
    reputation: f64,
    expected_composite: f64,
}

#[test]
fn pogq_fixtures_match_rust_implementation() {
    // Resolve fixture path relative to the sdk crate
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture_path =
        manifest_dir.join("../../mycelix-core/tests/shared-fixtures/pogq/simple_cases.json");

    // Skip test if fixture file doesn't exist (e.g., in CI without submodules)
    if !fixture_path.exists() {
        eprintln!(
            "Skipping pogq_fixtures test: fixture file not found at {:?}",
            fixture_path
        );
        return;
    }

    let data = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read fixture file {:?}: {}", fixture_path, e));

    let cases: Vec<PogqFixtureCase> = serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("Failed to parse fixture JSON: {}", e));

    for case in cases {
        let pogq = ProofOfGradientQuality::new(case.quality, case.consistency, case.entropy);
        let composite = pogq.composite_score(case.reputation);

        assert!(
            (composite - case.expected_composite).abs() < 1e-6,
            "Fixture '{}' expected {:.6}, got {:.6}",
            case.name,
            case.expected_composite,
            composite
        );
    }
}
