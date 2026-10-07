    fn v2_consume_returns_self_contained_approval_provenance_capsule() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = LocalApprovalRequestStoreV1::new(&daemon);
        let request = request_for(&store, 7);
        let projection_digest = "ab".repeat(32);
        let request_id = request.request_id().unwrap();
        let request_created_at = request.created_at_unix_ms;
        let request_expires_at = request.expires_at_unix_ms;
        let submission = LocalApprovalSubmissionV2 {
            request_id: request_id.clone(),
            daemon_incarnation_id: request.daemon_incarnation_id.clone(),
            action_intent_digest: request.action_intent_digest.clone(),
            projection_digest: projection_digest.clone(),
            decision: LocalApprovalDecisionKindV1::Approved,
            decided_at_unix_ms: 1_200,
        };
        store
            .install_pending_with_projection(request, projection_digest.clone())
            .unwrap();

        let consumed = store
            .consume_verified_submission_v2(
                &submission,
                &peer(1000, 42),
                AuthoritativeEvaluationV1::from_unix_millis_for_test(ms(1_300)),
            )
            .unwrap();

        assert_eq!(consumed.request_id(), request_id);
        assert_eq!(consumed.decision_kind(), LocalApprovalDecisionKindV1::Approved);
        assert_eq!(consumed.projection_digest(), projection_digest);
        assert_eq!(consumed.required_approval_profile(), "same-uid-process-v1");
        assert_eq!(consumed.approver_evidence_ref().profile, super::super::approver_evidence::ApproverEvidenceProfileV1::LocalUnixPeerCredentialV1);
        assert_eq!(consumed.approver_evidence_ref().evidence_digest.len(), 64);
        assert_eq!(consumed.transport_instance_ref(), "unix-socket-instance:test-42");
        assert_eq!(consumed.peer_observed_at(), ms(1_100));
        assert_eq!(consumed.consumed_at(), ms(1_300));
        assert_eq!(consumed.request_created_at(), ms(request_created_at));
        assert_eq!(consumed.request_expires_at(), ms(request_expires_at));
        assert_ne!(consumed.request_created_at(), consumed.consumed_at());
        assert_eq!(store.pending_count().unwrap(), 0);
    }
