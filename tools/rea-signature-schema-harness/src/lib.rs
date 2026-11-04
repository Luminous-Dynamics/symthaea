//! Structural qualification for the versioned Android/iOS package-signature evidence schema.
//!
//! The harness deliberately tests separation of cryptographic validity, signer identity,
//! trust-policy acceptance, package authorization, installability and runtime observations.

use jsonschema::draft202012;
use serde_json::Value;

const PACKAGE_SIGNATURE_SCHEMA: &str =
    include_str!("../../../docs/research/rea-package-signature-evaluation-v0.schema.json");

pub fn load_schema() -> Result<Value, String> {
    serde_json::from_str(PACKAGE_SIGNATURE_SCHEMA).map_err(|error| error.to_string())
}

pub fn validate_evaluation(evaluation: &Value) -> Result<(), String> {
    let schema = load_schema()?;
    let validator = draft202012::options()
        .build(&schema)
        .map_err(|error| error.to_string())?;
    validator
        .validate(evaluation)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_valid_but_unexpected_signer() -> Value {
        json!({
            "schema_version": "0.1.0",
            "package": {
                "format": "apk",
                "whole_package_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "size_bytes": "18446744073709551615",
                "bundle_or_application_id": "org.example.application",
                "display_name": "Example App"
            },
            "platform": {
                "ecosystem": "android",
                "os_version": "16",
                "api_or_platform_build": "API 36",
                "architecture": "arm64-v8a",
                "evaluation_environment": "official_tool",
                "device_or_runtime_model": null
            },
            "scheme_evaluations": [
                {
                    "scheme": "apk_v3",
                    "presence": "present",
                    "cryptographic_result": "valid",
                    "coverage_scope": "APK Signing Block and protected APK content",
                    "protected_digest_algorithm": "SHA-256",
                    "signature_algorithm": "RSA-PSS-SHA256",
                    "signer_indices": [0],
                    "limitations": [],
                    "verifier": {
                        "implementation": "apksig reference adapter",
                        "version": "pinned test version",
                        "source_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "toolchain": "Rust 1.96.0",
                        "target": "x86_64-unknown-linux-gnu",
                        "dependency_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    }
                }
            ],
            "signers": [
                {
                    "index": 0,
                    "certificate_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                    "public_key_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                    "subject_display": "CN=Unexpected Publisher",
                    "issuer_display": "CN=Example Test CA",
                    "signature_result": "valid",
                    "identity_attribution": {
                        "result": "unverified",
                        "attribution_claim": "Unexpected Publisher",
                        "evidence_references": []
                    },
                    "rotation_lineage": null
                }
            ],
            "trust_policy": {
                "result": "rejected",
                "policy_id": "approved-publisher-policy",
                "policy_version": "2026-11",
                "evaluation_time": "2026-11-04T20:00:00Z",
                "trust_anchor_set_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                "revocation_policy": "required",
                "revocation_result": "good",
                "chain_results": [
                    {
                        "signer_index": 0,
                        "result": "valid",
                        "detail": "Certificate chain valid but signer not in approved publisher set"
                    }
                ]
            },
            "authorization": {
                "result": "denied",
                "policy_id": "managed-device-app-allowlist-v1",
                "application_or_bundle_id": "org.example.application",
                "team_or_signing_id": null,
                "entitlement_evaluation": {
                    "result": "not_evaluated",
                    "claimed_entitlements_sha256": null,
                    "allowed_entitlements_sha256": null,
                    "violations": []
                },
                "distribution_constraints": ["Signer is not approved for this managed-device fleet"]
            },
            "nested_code": {
                "coverage_result": "partial",
                "components": []
            },
            "installability": {
                "result": "not_tested",
                "platform_context": "No device installation attempted",
                "detail": null
            },
            "runtime": {
                "result": "not_tested",
                "test_context": null,
                "observations": []
            },
            "provenance": {
                "implementation": "REA package signature schema test",
                "version": "0.1.0",
                "source_commit": "ffffffffffffffffffffffffffffffffffffffff",
                "toolchain": "Rust 1.96.0",
                "target": "x86_64-unknown-linux-gnu",
                "dependency_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            },
            "claim_scope": {
                "asserted_claims": ["signature_cryptographic_validity"],
                "not_established": [
                    "publisher_benevolence",
                    "code_safety",
                    "absence_of_malicious_behavior",
                    "future_runtime_behavior",
                    "user_intent_alignment",
                    "all_platforms_compatibility",
                    "unobserved_signing_schemes",
                    "unobserved_nested_code",
                    "global_trustworthiness"
                ]
            },
            "attestation": {
                "status": "not_provided",
                "in_toto_dsse_reference": null
            }
        })
    }

    #[test]
    fn package_signature_schema_is_valid_draft_2020_12() {
        let schema = load_schema().unwrap();
        jsonschema::meta::validate(&schema).unwrap();
        draft202012::options().build(&schema).unwrap();
    }

    #[test]
    fn cryptographic_validity_can_be_valid_while_trust_and_authorization_are_denied() {
        let evaluation = sample_valid_but_unexpected_signer();
        validate_evaluation(&evaluation).unwrap();

        assert_eq!(
            evaluation["scheme_evaluations"][0]["cryptographic_result"],
            "valid"
        );
        assert_eq!(evaluation["signers"][0]["signature_result"], "valid");
        assert_eq!(evaluation["trust_policy"]["result"], "rejected");
        assert_eq!(evaluation["authorization"]["result"], "denied");
    }

    #[test]
    fn schema_does_not_define_a_generic_trusted_boolean() {
        let schema = load_schema().unwrap();
        let root_properties = schema["properties"].as_object().unwrap();
        assert!(!root_properties.contains_key("trusted"));
        assert!(!root_properties.contains_key("signature_valid"));
        assert!(!root_properties.contains_key("overall_trust"));
    }

    #[test]
    fn unknown_scheme_and_unobserved_runtime_are_not_coerced_to_validity() {
        let mut evaluation = sample_valid_but_unexpected_signer();
        evaluation["scheme_evaluations"][0]["cryptographic_result"] = json!("unknown");
        evaluation["scheme_evaluations"][0]["presence"] = json!("unknown");
        evaluation["runtime"]["result"] = json!("unknown");
        validate_evaluation(&evaluation).unwrap();
    }

    #[test]
    fn extra_generic_trust_field_is_rejected() {
        let mut evaluation = sample_valid_but_unexpected_signer();
        evaluation["trusted"] = json!(true);
        assert!(validate_evaluation(&evaluation).is_err());
    }

    #[test]
    fn large_package_size_remains_an_exact_decimal_string() {
        let evaluation = sample_valid_but_unexpected_signer();
        assert_eq!(
            evaluation["package"]["size_bytes"].as_str(),
            Some("18446744073709551615")
        );
        validate_evaluation(&evaluation).unwrap();
    }
}
