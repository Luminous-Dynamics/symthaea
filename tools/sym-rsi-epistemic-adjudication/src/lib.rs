//! Focused compilation harness for the production web-research epistemic types
//! and meta-learning module. It deliberately imports the source files by path
//! so tests exercise the repository implementation rather than copied shims.

extern crate self as anyhow;

/// The production meta-learning module only uses anyhow's result alias.
/// Keep this standalone harness free from unrelated runtime dependencies.
pub type Result<T> = std::result::Result<T, String>;

#[path = "../../../src/web_research/types.rs"]
pub mod types;

#[path = "../../../src/web_research/meta_learning.rs"]
pub mod meta_learning;

#[path = "../../../src/web_research/selective_prediction.rs"]
pub mod selective_prediction;

#[path = "../../../src/web_research/selective_observation.rs"]
pub mod selective_observation;

#[cfg(test)]
mod compatibility_tests {
    use super::types::{ResearchSource, VerificationOutcome};

    #[test]
    fn legacy_boolean_correctness_deserializes_as_adjudicated() {
        let value = serde_json::json!({
            "claim": "A legacy claim",
            "source_domain": "legacy.example",
            "source_type": "Web",
            "was_correct": true,
            "predicted_confidence": 0.99,
            "domain": "science"
        });
        let outcome: VerificationOutcome =
            serde_json::from_value(value).expect("legacy boolean field remains compatible");
        assert_eq!(outcome.was_correct, Some(true));
    }

    #[test]
    fn absent_correctness_field_deserializes_as_unadjudicated() {
        let value = serde_json::json!({
            "claim": "A pending claim",
            "source_domain": "pending.example",
            "source_type": "Web",
            "predicted_confidence": 0.99,
            "domain": "science"
        });
        let outcome: VerificationOutcome =
            serde_json::from_value(value).expect("missing adjudication defaults to None");
        assert_eq!(outcome.was_correct, None);
        assert_eq!(outcome.source_type, ResearchSource::Web);
    }

    #[test]
    fn unadjudicated_serialization_preserves_none() {
        let outcome = VerificationOutcome::unadjudicated(
            "A pending claim",
            "pending.example",
            ResearchSource::Academic,
        )
        .with_confidence(0.91);
        let value = serde_json::to_value(outcome).expect("outcome serializes");
        assert!(value["was_correct"].is_null());
        let serialized_confidence = value["predicted_confidence"]
            .as_f64()
            .expect("confidence serializes as a JSON number");
        assert!((serialized_confidence - 0.91).abs() < 1e-6);
    }
}
