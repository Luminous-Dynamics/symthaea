#[path = "../../../src/rea_compat.rs"]
mod rea_compat;

use rea_compat::claims::{
    ClaimDisposition, ClaimPredicate, ClaimRequest, ClaimScope, evaluate_claim,
};
use serde_json::json;

#[test]
fn root_adapter_exposes_typed_claim_evaluator() {
    let raw = include_str!(
        "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
    );
    let evidence = rea_compat::verify_evidence_json(raw.as_bytes()).unwrap();
    let claim = ClaimRequest {
        schema: rea_compat::CLAIM_SCHEMA.to_owned(),
        claim_id: "root-adapter-smoke".to_owned(),
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
        falsification_condition: "A same-scope record has a different normalized result".to_owned(),
    };

    let result = evaluate_claim(&claim, std::slice::from_ref(&evidence)).unwrap();
    assert_eq!(result.disposition, ClaimDisposition::SupportedWithinScope);
    assert!(evidence.evidence_id.starts_with("ev_"));
}
