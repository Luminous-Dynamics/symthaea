use serde_json::json;
use symthaea_neo_observation_voi_verifier::verify_frozen_fixture;

fn main() {
    match verify_frozen_fixture() {
        Ok(report) => {
            let receipt = json!({
                "schema_version": "symthaea.neo_observation_voi.receipt.v0.1",
                "status": "PASS_SYNTHETIC_DECISION_ORACLE_ONLY",
                "fixture_sha256": report.sha256,
                "baseline_risk": report.baseline_risk,
                "nuisance_entropy_bits": report.nuisance_entropy_bits,
                "perfect_information_evsi": report.perfect_information_evsi,
                "single_reading": {
                    "expected_risk": report.single_reading.expected_risk,
                    "evsi": report.single_reading.evsi,
                    "information_gain_bits": report.single_reading.information_gain_bits
                },
                "same_source_pair": {
                    "expected_risk": report.same_source_pair.expected_risk,
                    "evsi": report.same_source_pair.evsi,
                    "information_gain_bits": report.same_source_pair.information_gain_bits
                },
                "independent_source_pair": {
                    "expected_risk": report.independent_source_pair.expected_risk,
                    "evsi": report.independent_source_pair.evsi,
                    "information_gain_bits": report.independent_source_pair.information_gain_bits
                },
                "unknown_dependence_control": {
                    "assessment": "unknown",
                    "decision": "ABSTAIN_DEPENDENCE_UNRESOLVED",
                    "numeric_evsi": null,
                    "independent_product_permitted": false
                },
                "claim_ceiling": "SYNTHETIC_ANALYTIC_ORACLE_ONLY"
            });
            println!("{}", serde_json::to_string_pretty(&receipt).expect("receipt serializes"));
        }
        Err(error) => {
            eprintln!("FAIL_SYNTHETIC_DECISION_ORACLE: {error}");
            std::process::exit(1);
        }
    }
}
