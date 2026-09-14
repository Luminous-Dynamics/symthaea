mod verifier {
    #![allow(dead_code)]
    include!("../src/main.rs");

    fn golden_envelope() -> Value {
        json!({
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
        })
    }

    fn artifact(value: Value) -> Artifact {
        let bytes = serde_json::to_vec(&value).expect("serialize test artifact");
        Artifact {
            json: value,
            sha256: sha256_hex(&bytes),
        }
    }

    fn subject_fixture() -> (Artifact, Artifact, Artifact, Artifact, Value) {
        let plan = artifact(json!({
            "protocol_version": WCARE40_PROTOCOL,
            "replica_slots": []
        }));
        let result = artifact(json!({
            "protocol_version": WCARE40_PROTOCOL,
            "plan_sha256": plan.sha256.clone(),
            "disposition": "REPLICATION_SUPPORTED"
        }));
        let subject = artifact(json!({
            "protocol_version": WCARE40_PROTOCOL,
            "plan_sha256": plan.sha256.clone(),
            "replica_id": "a",
            "provenance_strength": "ExternalVerified"
        }));
        let policy = artifact(json!({
            "protocol_version": PROTOCOL,
            "policy_id": "builder.policy.test",
            "policy_created_utc": "2026-09-13T11:00:00Z",
            "issuer_keys": [{
                "issuer_key_id": "issuer.test",
                "issuer_public_key_ed25519_hex": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
                "issuer_identity_commitment_sha256": "5555555555555555555555555555555555555555555555555555555555555555",
                "valid_from_utc": "2026-09-13T10:00:00Z",
                "valid_until_utc": null,
                "revocation_effective_utc": null,
                "allowed_subject_kinds": ["BuilderProvenance"],
                "allowed_provenance_strengths": ["ExternalVerified"],
                "allowed_relation_evidence_strengths": []
            }]
        }));
        let envelope = json!({
            "protocol_version": PROTOCOL,
            "wcare40_plan_sha256": plan.sha256.clone(),
            "wcare40_result_sha256": "-",
            "subject_kind": "BuilderProvenance",
            "subject_receipt_sha256": subject.sha256.clone(),
            "provenance_strength_claim": "ExternalVerified",
            "relation_evidence_strength_claim": "-",
            "issuer_key_id": "issuer.test",
            "issuer_public_key_ed25519_hex": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "issuer_policy_sha256": policy.sha256.clone(),
            "issued_at_utc": "2026-09-13T12:00:00Z",
            "expires_at_utc": "2026-09-14T12:00:00Z",
            "nonce_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
            "domain": DOMAIN,
            "signature_ed25519_hex": "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        });
        (plan, result, subject, policy, envelope)
    }

    #[test]
    fn cross_implementation_builder_vector_matches_and_verifies() {
        let envelope = golden_envelope();
        let message = canonical_message(&envelope).expect("canonicalize WCARE-42 golden vector");
        assert_eq!(
            sha256_hex(message.as_bytes()),
            "2054958f9c99f594131c7335f89e8ce5dc68b46ba92eb91a3201eafadc4f523f"
        );
        assert!(verify_signature(&envelope, message.as_bytes()).expect("verify golden signature"));
    }

    #[test]
    fn altered_signature_is_rejected() {
        let mut envelope = golden_envelope();
        let message = canonical_message(&envelope).expect("canonicalize");
        envelope["signature_ed25519_hex"] = Value::String(
            "00bc6dae63c3fc60c50b2c1f443a6a9d714006659c31380609b35d85a0efe4cdf15b5cb1f862ce618c540ae5617862f51b661c06acb1d83b174acd296ea56c06".into(),
        );
        assert!(!verify_signature(&envelope, message.as_bytes()).expect("verify altered signature"));
    }

    #[test]
    fn subject_kind_and_strength_fields_cannot_be_crossed() {
        let mut envelope = golden_envelope();
        envelope["subject_kind"] = Value::String("BuilderRelation".into());
        envelope["relation_evidence_strength_claim"] = Value::String("ExternalVerified".into());
        assert_eq!(
            canonical_message(&envelope).unwrap_err(),
            "invalid_builder_relation_scope_fields"
        );
    }

    #[test]
    fn expiry_must_be_strictly_after_issue_time() {
        let mut envelope = golden_envelope();
        envelope["expires_at_utc"] = Value::String("2026-09-13T12:00:00Z".into());
        assert_eq!(canonical_message(&envelope).unwrap_err(), "expiry_not_after_issue_time");
    }

    #[test]
    fn pre_result_attestation_does_not_bind_later_result() {
        let (plan, result, subject, policy, mut envelope) = subject_fixture();
        let (bound, result_bound) = subject_binding(&envelope, &policy, &plan, &result, &subject)
            .expect("evaluate pre-result subject binding");
        assert!(bound);
        assert!(!result_bound);

        envelope["wcare40_result_sha256"] = Value::String(result.sha256.clone());
        let (bound, result_bound) = subject_binding(&envelope, &policy, &plan, &result, &subject)
            .expect("evaluate result-bound subject binding");
        assert!(bound);
        assert!(result_bound);
    }

    #[test]
    fn altered_subject_or_policy_cannot_replay_attestation() {
        let (plan, result, subject, policy, envelope) = subject_fixture();
        let altered_subject = artifact(json!({
            "protocol_version": WCARE40_PROTOCOL,
            "plan_sha256": plan.sha256.clone(),
            "replica_id": "a",
            "provenance_strength": "OrganizerVerified"
        }));
        assert!(!subject_binding(&envelope, &policy, &plan, &result, &altered_subject)
            .expect("altered subject binding")
            .0);

        let altered_policy = artifact(json!({
            "protocol_version": PROTOCOL,
            "policy_id": "other.policy",
            "policy_created_utc": "2026-09-13T11:00:00Z",
            "issuer_keys": []
        }));
        assert!(!subject_binding(&envelope, &altered_policy, &plan, &result, &subject)
            .expect("altered policy binding")
            .0);
    }

    #[test]
    fn revoked_expired_or_unauthorized_issuer_is_not_trusted() {
        let (_plan, _result, _subject, policy, envelope) = subject_fixture();

        let mut checks = Checks::default();
        evaluate_trust(
            &envelope,
            &policy.json,
            parse_utc("2026-09-13T13:00:00Z").unwrap(),
            &mut checks,
        )
        .expect("trusted baseline");
        assert!(checks.issuer_trusted_for_claim);

        let mut revoked_policy = policy.json.clone();
        revoked_policy["issuer_keys"][0]["revocation_effective_utc"] =
            Value::String("2026-09-13T12:00:00Z".into());
        let mut revoked = Checks::default();
        evaluate_trust(
            &envelope,
            &revoked_policy,
            parse_utc("2026-09-13T13:00:00Z").unwrap(),
            &mut revoked,
        )
        .expect("revoked trust evaluation");
        assert!(!revoked.revocation_policy_satisfied);
        assert!(!revoked.issuer_trusted_for_claim);

        let mut expired = Checks::default();
        evaluate_trust(
            &envelope,
            &policy.json,
            parse_utc("2026-09-14T12:00:00Z").unwrap(),
            &mut expired,
        )
        .expect("expired trust evaluation");
        assert!(!expired.attestation_current_at_evaluation);
        assert!(!expired.issuer_trusted_for_claim);

        let mut unauthorized_policy = policy.json.clone();
        unauthorized_policy["issuer_keys"][0]["allowed_provenance_strengths"] =
            json!(["OrganizerVerified"]);
        let mut unauthorized = Checks::default();
        evaluate_trust(
            &envelope,
            &unauthorized_policy,
            parse_utc("2026-09-13T13:00:00Z").unwrap(),
            &mut unauthorized,
        )
        .expect("unauthorized strength evaluation");
        assert!(!unauthorized.claimed_strength_authorized);
        assert!(!unauthorized.issuer_trusted_for_claim);
    }
}
