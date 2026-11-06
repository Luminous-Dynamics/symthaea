//! Support crate for fuzzing the actual REA formal provider-observation parser.

#[path = "../../../src/rea_compat.rs"]
pub mod rea_compat;

/// Build a deterministic valid v0 provider-observation envelope from the pinned
/// REA Evidence fixture. CI writes these exact bytes to the fuzzer seed corpus.
pub fn generate_valid_seed() -> Vec<u8> {
    use rea_compat::formal::export_formal_provider_observation;
    use rea_compat::{
        ClaimPredicate, ClaimRequest, ClaimScope, evaluate_claim, verify_evidence_json,
    };
    use serde_json::json;

    const EVIDENCE: &[u8] = include_bytes!(
        "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
    );
    let evidence = verify_evidence_json(EVIDENCE).expect("pinned fixture must verify");
    let request = ClaimRequest {
        schema: rea_compat::CLAIM_SCHEMA.to_owned(),
        claim_id: "formal-envelope-fuzz-seed-v1".to_owned(),
        predicate: ClaimPredicate::ProviderReportedObservation,
        scope: ClaimScope {
            subject_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .to_owned(),
            operation: "inspect_artifact".to_owned(),
            platform: None,
            platform_version: None,
            environment_id: None,
        },
        asserted_value: json!({
            "entries": [{"path": "classes.dex", "size": 42}],
            "status": "complete"
        }),
        source_predicate_type: Some("rea.analysis".to_owned()),
        evidence_ids: vec![evidence.evidence_id.clone()],
        assumptions: Vec::new(),
        falsification_condition:
            "A same-scope verified Evidence record has a different normalized_result".to_owned(),
    };
    let evaluation =
        evaluate_claim(&request, std::slice::from_ref(&evidence)).expect("fixture claim evaluates");
    let formal =
        export_formal_provider_observation(&evaluation).expect("fixture claim exports");
    serde_json::to_vec(&formal).expect("formal envelope serializes")
}

#[cfg(test)]
mod tests {
    use super::generate_valid_seed;
    use crate::rea_compat::formal::{
        FormalSemanticPolarity, parse_formal_provider_observation_json,
    };

    #[test]
    fn generated_seed_reaches_positive_formal_projection() {
        let bytes = generate_valid_seed();
        let parsed = parse_formal_provider_observation_json(&bytes).unwrap();
        assert_eq!(
            parsed.semantic_polarity,
            FormalSemanticPolarity::PositiveSupport
        );
    }

    #[test]
    fn generated_seed_is_stable_across_repeated_generation() {
        assert_eq!(generate_valid_seed(), generate_valid_seed());
    }
}
