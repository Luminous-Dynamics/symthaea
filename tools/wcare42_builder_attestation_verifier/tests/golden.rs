mod verifier {
    #![allow(dead_code)]
    include!("../src/main.rs");

    #[test]
    fn cross_implementation_builder_vector_matches_and_verifies() {
        let envelope = json!({
            "protocol_version": "wcare41-authenticated-preregistration-v1",
            "wcare40_plan_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "wcare40_result_sha256": "-",
            "subject_kind": "BuilderProvenance",
            "subject_receipt_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "provenance_strength_claim": "ExternalVerified",
            "relation_evidence_strength_claim": "-",
            "issuer_key_id": "issuer.test",
            "issuer_public_key_ed25519_hex": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "issuer_policy_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "issued_at_utc": "2026-09-13T12:00:00Z",
            "expires_at_utc": "2026-09-14T12:00:00Z",
            "nonce_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
            "domain": "builder-evidence-attestation",
            "signature_ed25519_hex": "d9bc6dae63c3fc60c50b2c1f443a6a9d714006659c31380609b35d85a0efe4cdf15b5cb1f862ce618c540ae5617862f51b661c06acb1d83b174acd296ea56c06"
        });

        let message = canonical_message(&envelope).expect("canonicalize WCARE-42 golden vector");
        assert_eq!(
            sha256_hex(message.as_bytes()),
            "2054958f9c99f594131c7335f89e8ce5dc68b46ba92eb91a3201eafadc4f523f"
        );
        assert!(verify_signature(&envelope, message.as_bytes()).expect("verify golden signature"));
    }

    #[test]
    fn altered_signature_is_rejected() {
        let mut envelope = json!({
            "protocol_version": "wcare41-authenticated-preregistration-v1",
            "wcare40_plan_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "wcare40_result_sha256": "-",
            "subject_kind": "BuilderProvenance",
            "subject_receipt_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "provenance_strength_claim": "ExternalVerified",
            "relation_evidence_strength_claim": "-",
            "issuer_key_id": "issuer.test",
            "issuer_public_key_ed25519_hex": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "issuer_policy_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "issued_at_utc": "2026-09-13T12:00:00Z",
            "expires_at_utc": "2026-09-14T12:00:00Z",
            "nonce_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
            "domain": "builder-evidence-attestation",
            "signature_ed25519_hex": "d9bc6dae63c3fc60c50b2c1f443a6a9d714006659c31380609b35d85a0efe4cdf15b5cb1f862ce618c540ae5617862f51b661c06acb1d83b174acd296ea56c06"
        });
        let message = canonical_message(&envelope).expect("canonicalize");
        envelope["signature_ed25519_hex"] = Value::String(
            "00bc6dae63c3fc60c50b2c1f443a6a9d714006659c31380609b35d85a0efe4cdf15b5cb1f862ce618c540ae5617862f51b661c06acb1d83b174acd296ea56c06".into(),
        );
        assert!(!verify_signature(&envelope, message.as_bytes()).expect("verify altered signature"));
    }

    #[test]
    fn subject_kind_and_strength_fields_cannot_be_crossed() {
        let envelope = json!({
            "protocol_version": "wcare41-authenticated-preregistration-v1",
            "wcare40_plan_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "wcare40_result_sha256": "-",
            "subject_kind": "BuilderRelation",
            "subject_receipt_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "provenance_strength_claim": "ExternalVerified",
            "relation_evidence_strength_claim": "ExternalVerified",
            "issuer_key_id": "issuer.test",
            "issuer_public_key_ed25519_hex": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "issuer_policy_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "issued_at_utc": "2026-09-13T12:00:00Z",
            "expires_at_utc": "2026-09-14T12:00:00Z",
            "nonce_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
            "domain": "builder-evidence-attestation",
            "signature_ed25519_hex": "d9bc6dae63c3fc60c50b2c1f443a6a9d714006659c31380609b35d85a0efe4cdf15b5cb1f862ce618c540ae5617862f51b661c06acb1d83b174acd296ea56c06"
        });
        assert_eq!(
            canonical_message(&envelope).unwrap_err(),
            "invalid_builder_relation_scope_fields"
        );
    }

    #[test]
    fn expiry_must_be_strictly_after_issue_time() {
        let envelope = json!({
            "protocol_version": "wcare41-authenticated-preregistration-v1",
            "wcare40_plan_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "wcare40_result_sha256": "-",
            "subject_kind": "BuilderProvenance",
            "subject_receipt_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "provenance_strength_claim": "ExternalVerified",
            "relation_evidence_strength_claim": "-",
            "issuer_key_id": "issuer.test",
            "issuer_public_key_ed25519_hex": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "issuer_policy_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "issued_at_utc": "2026-09-13T12:00:00Z",
            "expires_at_utc": "2026-09-13T12:00:00Z",
            "nonce_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
            "domain": "builder-evidence-attestation",
            "signature_ed25519_hex": "d9bc6dae63c3fc60c50b2c1f443a6a9d714006659c31380609b35d85a0efe4cdf15b5cb1f862ce618c540ae5617862f51b661c06acb1d83b174acd296ea56c06"
        });
        assert_eq!(canonical_message(&envelope).unwrap_err(), "expiry_not_after_issue_time");
    }
}
