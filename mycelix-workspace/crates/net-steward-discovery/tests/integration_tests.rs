use net_steward_discovery::{
    evaluate_symthaea_safety_rules, generate_audit_trail_ledger, generate_blast_radius_preview,
    generate_demo_topology, generate_dry_run_rollback_plan, generate_nixos_drift_report,
    parse_linux_arp_table, parse_linux_routing_table,
};
use net_steward_schema::{
    ActionKind, BlastRadiusRiskTier, DidAgentBinding, DidBindingResolver, OperationIntent,
    OperationKind, ProofStatus, RiskLevel, SafetyVerdict,
};

#[test]
fn test_stewardship_end_to_end_flow() {
    // 1. Discover topology (with provenance fields)
    let topology = generate_demo_topology();
    assert_eq!(topology.nodes.len(), 5);
    assert_eq!(topology.edges.len(), 4);

    // Check provenance fields
    assert_eq!(
        topology.nodes[0].source_collector,
        "static_topology_manifest"
    );
    assert_eq!(topology.nodes[0].confidence, 1.0);
    assert_eq!(
        topology.edges[0].source_collector,
        "static_topology_manifest"
    );

    // 2. Perform drift verification
    let drift = generate_nixos_drift_report("luminous-router");
    assert!(drift.diff_closure.is_some());

    // 3. Evaluate safety rules via Symthaea Safety Advisor Loop (with ZK-STARK proof generation)
    let summary = evaluate_symthaea_safety_rules(&drift, &topology);
    assert_eq!(summary.safety_verdict, SafetyVerdict::Blocked);
    assert!(!summary.safety_violations.is_empty());
    assert!(summary.safety_violations[0].contains("Critical port block"));
    assert_eq!(summary.affected_users, vec!["Tristan"]);

    // Verify Zero-Knowledge correctness attributes and claim-disciplined verification status
    assert!(summary.safety_proof.is_some());
    assert!(summary.safety_commitment.is_some());
    assert_eq!(summary.proof_status, ProofStatus::SimulatedEnvelope);
    let proof = summary.safety_proof.as_ref().unwrap();
    assert_eq!(&proof[..8], b"STARK_PR"); // STARK Proof Signature Header verified!

    // 4. Generate & verify hash-chained audit ledger
    let receipts = generate_audit_trail_ledger("luminous-router");
    assert_eq!(receipts.len(), 2);

    let genesis = &receipts[0];
    let update = &receipts[1];

    assert_eq!(genesis.action_kind, ActionKind::ApplyConfig);
    assert_eq!(update.action_kind, ActionKind::ApplyConfig);

    // Validate hash-chaining link
    assert!(update.parent_hash.is_some());
    let expected_parent_hash = format!("sha256-hash-of-{:?}", genesis.receipt_id);
    assert_eq!(update.parent_hash.as_ref().unwrap(), &expected_parent_hash);

    // Validate signature presence
    assert!(genesis.cryptographic_signature.is_some());
    assert!(update.cryptographic_signature.is_some());

    // 5. Test active OS table parsers (should run without panic, returning edges on Linux hosts or empty vectors in sandboxes)
    let arp_edges = parse_linux_arp_table();
    let route_edges = parse_linux_routing_table();
    println!(
        "Active Discovery Table edges parsed: arp: {}, routes: {}",
        arp_edges.len(),
        route_edges.len()
    );

    // 6. Test OPNsense / OpenWrt adapters with mock input payloads
    let opnsense_arp_json = r#"{
        "rows": [
            { "ip": "192.168.1.100", "mac": "00:11:22:33:44:55", "hostname": "laptop", "intf": "lan" }
        ]
    }"#;
    let opn_arp_edges =
        net_steward_discovery::adapters::parse_opnsense_arp(opnsense_arp_json).unwrap();
    assert_eq!(opn_arp_edges.len(), 1);
    assert_eq!(opn_arp_edges[0].source_node_id, "mac-00:11:22:33:44:55");

    let opnsense_leases_json = r#"{
        "rows": [
            { "ip": "192.168.1.100", "mac": "00:11:22:33:44:55", "hostname": "laptop", "state": "active" }
        ]
    }"#;
    let opn_lease_edges =
        net_steward_discovery::adapters::parse_opnsense_leases(opnsense_leases_json).unwrap();
    assert_eq!(opn_lease_edges.len(), 1);
    assert_eq!(opn_lease_edges[0].target_node_id, "ip-192.168.1.100");

    let openwrt_ubus_json = r#"{
        "leases": [
            { "ip": "192.168.1.150", "mac": "aa:bb:cc:dd:ee:ff", "hostname": "phone" }
        ]
    }"#;
    let openwrt_lease_edges =
        net_steward_discovery::adapters::parse_openwrt_leases(openwrt_ubus_json).unwrap();
    assert_eq!(openwrt_lease_edges.len(), 1);
    assert_eq!(
        openwrt_lease_edges[0].source_node_id,
        "mac-aa:bb:cc:dd:ee:ff"
    );

    let openwrt_leases_file = "1724219412 11:22:33:44:55:66 10.0.0.50 my-device *\n";
    let openwrt_file_edges =
        net_steward_discovery::adapters::parse_openwrt_dhcp_leases_file(openwrt_leases_file);
    assert_eq!(openwrt_file_edges.len(), 1);
    assert_eq!(openwrt_file_edges[0].target_node_id, "ip-10.0.0.50");

    // 7. Validate RollbackPlan schema properties
    let plan = generate_dry_run_rollback_plan("luminous-router", "Generation 427");
    assert_eq!(plan.risk_level, RiskLevel::Low);
    assert!(plan.requires_approval);
    assert_eq!(plan.expected_changes.len(), 4);

    // 8. Assert API Golden Fixtures structures load correctly
    let topology_linux_raw = include_str!("fixtures/topology_linux.json");
    let top_linux: net_steward_schema::ObservedTopologySnapshot =
        serde_json::from_str(topology_linux_raw).unwrap();
    assert_eq!(top_linux.snapshot_id, "fixture-topology-linux");

    let topology_openwrt_raw = include_str!("fixtures/topology_openwrt.json");
    let top_openwrt: net_steward_schema::ObservedTopologySnapshot =
        serde_json::from_str(topology_openwrt_raw).unwrap();
    assert_eq!(top_openwrt.snapshot_id, "fixture-topology-openwrt");

    let topology_opnsense_raw = include_str!("fixtures/topology_opnsense.json");
    let top_opnsense: net_steward_schema::ObservedTopologySnapshot =
        serde_json::from_str(topology_opnsense_raw).unwrap();
    assert_eq!(top_opnsense.snapshot_id, "fixture-topology-opnsense");

    let drift_nixos_raw = include_str!("fixtures/drift_nixos.json");
    let drift_nix: net_steward_schema::ConfigDriftReport =
        serde_json::from_str(drift_nixos_raw).unwrap();
    assert_eq!(drift_nix.report_id, "drift-report-demo-0");

    let verdict_raw = include_str!("fixtures/verdict_commitment_only.json");
    let verdict: net_steward_schema::HumanReadableIncidentSummary =
        serde_json::from_str(verdict_raw).unwrap();
    assert_eq!(verdict.proof_status, ProofStatus::SimulatedEnvelope);

    // 9. API Contract validations
    let api_healthz_raw = include_str!("fixtures/api_healthz.json");
    let healthz_val: serde_json::Value = serde_json::from_str(api_healthz_raw).unwrap();
    assert_eq!(healthz_val["status"], "OK");

    let api_capabilities_raw = include_str!("fixtures/api_capabilities.json");
    let caps_val: serde_json::Value = serde_json::from_str(api_capabilities_raw).unwrap();
    assert_eq!(caps_val["read_only"], true);
    assert_eq!(caps_val["rollback_apply_enabled"], false);
    assert_eq!(caps_val["zk_verifier_enabled"], false);
    assert_eq!(caps_val["proof_mode"], "simulated_envelope");

    let api_topology_raw = include_str!("fixtures/api_topology.json");
    let top_val: net_steward_schema::ObservedTopologySnapshot =
        serde_json::from_str(api_topology_raw).unwrap();
    for node in &top_val.nodes {
        assert!(!node.source_collector.is_empty());
        assert!(node.confidence >= 0.0 && node.confidence <= 1.0);
        assert!(node.observed_at_unix_ms > 0);
        assert!(node.evidence_hash.is_some());
    }
    for edge in &top_val.edges {
        assert!(!edge.source_collector.is_empty());
        assert!(edge.confidence >= 0.0 && edge.confidence <= 1.0);
        assert!(edge.evidence_hash.is_some());
    }

    let api_verdict_raw = include_str!("fixtures/api_verdict.json");
    let api_verdict_val: net_steward_schema::HumanReadableIncidentSummary =
        serde_json::from_str(api_verdict_raw).unwrap();
    assert_ne!(
        api_verdict_val.proof_status,
        ProofStatus::Verified,
        "Verifier must not say Verified unless active verification paths are present"
    );

    let api_evidence_raw = include_str!("fixtures/api_evidence.json");
    let api_ev: Vec<net_steward_schema::InfrastructureReceipt> =
        serde_json::from_str(api_evidence_raw).unwrap();
    assert_eq!(
        api_ev[0].chronicle_status,
        net_steward_schema::ChronicleStatus::Committed
    );

    let api_rollback_raw = include_str!("fixtures/api_rollback_plan.json");
    let api_plan: net_steward_schema::RollbackPlan =
        serde_json::from_str(api_rollback_raw).unwrap();
    assert!(api_plan.requires_approval);

    let api_version_raw = include_str!("fixtures/api_version.json");
    let api_version_val: net_steward_schema::DaemonVersion =
        serde_json::from_str(api_version_raw).unwrap();
    assert_eq!(api_version_val.name, "net-steward-daemon");
    assert_eq!(api_version_val.version, "0.1.0-alpha.4");
    assert_eq!(api_version_val.mode, "read_only_witness");
    assert_eq!(api_version_val.mutation_enabled, false);

    // 10. Security Witness API Contract validations
    let api_sec_events_raw = include_str!("fixtures/api_security_events.json");
    let sec_events: Vec<net_steward_schema::SecurityEvent> =
        serde_json::from_str(api_sec_events_raw).unwrap();
    assert_eq!(sec_events.len(), 2);
    assert_eq!(sec_events[0].event_id, "sec-evt-101");
    assert_eq!(sec_events[0].severity, net_steward_schema::Severity::High);

    let api_sec_posture_raw = include_str!("fixtures/api_security_posture.json");
    let sec_posture: net_steward_schema::SecurityPosture =
        serde_json::from_str(api_sec_posture_raw).unwrap();
    assert_eq!(sec_posture.node_id, "forge-server");
    assert!(
        sec_posture
            .active_systemd_units
            .contains(&"backdoord.service".to_string())
    );

    let api_sec_baseline_raw = include_str!("fixtures/api_security_baseline.json");
    let sec_baseline: net_steward_schema::KnownGoodBaseline =
        serde_json::from_str(api_sec_baseline_raw).unwrap();
    assert_eq!(sec_baseline.node_id, "forge-server");
    assert_eq!(sec_baseline.desired_generation, "Generation 427");

    // 11. Federated Peer and Verifier API validations
    let api_peers_raw = include_str!("fixtures/api_security_peers.json");
    let report: net_steward_schema::PeerTelemetryReport =
        serde_json::from_str(api_peers_raw).unwrap();
    assert_eq!(report.peers.len(), 3);
    assert_eq!(report.peers[0].display_name, "Forge Build Server");
    assert_eq!(report.federation_mode, "local_fixture");
    assert!(report.transport_available);

    let api_verify_raw = include_str!("fixtures/api_incident_verify.json");
    let verify_res: net_steward_schema::IncidentVerificationResult =
        serde_json::from_str(api_verify_raw).unwrap();
    assert!(verify_res.result_passed);
    assert_eq!(
        verify_res.proof_status,
        net_steward_schema::ProofStatus::SimulatedEnvelope
    );
}

#[test]
fn test_release_manifest_matches_capabilities() {
    let manifest_data = std::fs::read_to_string("../../RELEASE_MANIFEST.json")
        .or_else(|_| std::fs::read_to_string("RELEASE_MANIFEST.json"))
        .or_else(|_| std::fs::read_to_string("../RELEASE_MANIFEST.json"))
        .expect("Failed to locate RELEASE_MANIFEST.json");

    let manifest_json: serde_json::Value = serde_json::from_str(&manifest_data).unwrap();

    // Permanent safety gates — must never flip true in lab mode.
    assert_eq!(manifest_json["mutation_enabled"].as_bool().unwrap(), false);
    assert_eq!(
        manifest_json["rollback_apply_enabled"].as_bool().unwrap(),
        false
    );
    assert_eq!(
        manifest_json["zk_verifier_enabled"].as_bool().unwrap(),
        false
    );

    // v0.3-alpha.4: blast-radius preview is enabled (read-only display only).
    assert_eq!(
        manifest_json["blast_radius_preview_enabled"]
            .as_bool()
            .unwrap(),
        true,
        "blast_radius_preview_enabled must be true in v0.3-alpha.4+"
    );

    // Alpha track must be at least alpha.4.
    let alpha_track = manifest_json["alpha_track"].as_str().unwrap_or("");
    assert!(
        alpha_track.contains("alpha.4")
            || alpha_track.contains("alpha.5")
            || alpha_track.contains("alpha.6"),
        "alpha_track should be alpha.4 or later, got: {}",
        alpha_track
    );
}

#[test]
fn test_federated_claims_transports() {
    use net_steward_discovery::{
        CapsuleFileTransport, ClaimFilter, FederationTransport, LocalFixtureTransport,
    };

    let local_tx = LocalFixtureTransport;
    let claims = local_tx
        .fetch_claims(ClaimFilter {
            issuer_did: None,
            subject_node_id: None,
        })
        .unwrap();
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].issuer_did, "did:mycelix:tristan-laptop");
    assert_eq!(
        claims[0].payload.posture_summary,
        net_steward_schema::PostureSummary::Healthy
    );

    let temp_file = std::env::temp_dir().join("test_peer_claim_envelope.json");
    let file_tx = CapsuleFileTransport {
        file_path: temp_file.clone(),
    };

    let claim_to_publish = claims[0].clone();
    file_tx.publish_claim(claim_to_publish).unwrap();

    let fetched_claims = file_tx
        .fetch_claims(ClaimFilter {
            issuer_did: None,
            subject_node_id: None,
        })
        .unwrap();
    assert_eq!(fetched_claims.len(), 1);
    assert_eq!(fetched_claims[0].envelope_id, "env-202");

    let _ = std::fs::remove_file(temp_file);
}

#[test]
fn test_posture_claim_reconciliation() {
    use net_steward_discovery::reconciliation::reconcile_claims;
    use net_steward_schema::{
        ClaimEnvelope, ClaimStatus, ClaimVerificationStatus, EncodingProfile, PeerPostureClaim,
        PeerTrustStatus, PostureSummary, SignatureScheme,
    };

    let now = 1719569000000;

    // 1. Unsigned newer claim vs signed older claim
    let claim_1 = ClaimEnvelope {
        schema_version: "claim_envelope_v0.1".to_string(),
        encoding_profile: EncodingProfile::NetStewardCanonicalJsonV1,
        envelope_id: "env-1".to_string(),
        payload: PeerPostureClaim {
            claim_id: "c-1".to_string(),
            issuer_did: "did:mycelix:trusted-1".to_string(),
            subject_node_id: "forge-server".to_string(),
            issued_at_unix_ms: now - 10000,
            expires_at_unix_ms: now + 60000,
            posture_summary: PostureSummary::Healthy,
            topology_refs: vec![],
            security_event_refs: vec![],
            evidence_refs: vec![],
            capsule_refs: vec![],
            claim_status: ClaimStatus::Valid,
        },
        payload_hash: "h-1".to_string(),
        issuer_did: "did:mycelix:trusted-1".to_string(),
        signature: "sig-1".to_string(),
        signature_scheme: SignatureScheme::SimulatedEnvelope,
        verification_status: ClaimVerificationStatus::VerifiedSignature,
        signatures: None,
    };

    // Conflicting claim from another peer
    let claim_2 = ClaimEnvelope {
        schema_version: "claim_envelope_v0.1".to_string(),
        encoding_profile: EncodingProfile::NetStewardCanonicalJsonV1,
        envelope_id: "env-2".to_string(),
        payload: PeerPostureClaim {
            claim_id: "c-2".to_string(),
            issuer_did: "did:mycelix:trusted-2".to_string(),
            subject_node_id: "forge-server".to_string(),
            issued_at_unix_ms: now - 5000,
            expires_at_unix_ms: now + 60000,
            posture_summary: PostureSummary::Degraded,
            topology_refs: vec![],
            security_event_refs: vec![],
            evidence_refs: vec![],
            capsule_refs: vec![],
            claim_status: ClaimStatus::Valid,
        },
        payload_hash: "h-2".to_string(),
        issuer_did: "did:mycelix:trusted-2".to_string(),
        signature: "sig-2".to_string(),
        signature_scheme: SignatureScheme::SimulatedEnvelope,
        verification_status: ClaimVerificationStatus::VerifiedSignature,
        signatures: None,
    };

    let claims = vec![claim_1, claim_2];
    let (peers, conflicts, accepted, rejected, stale) =
        reconcile_claims(claims, now, vec![], vec![]);

    assert_eq!(accepted, 2);
    assert_eq!(rejected, 0);
    assert_eq!(stale, 0);
    assert_eq!(peers.len(), 1);
    // Conflicting claims mapping
    assert_eq!(peers[0].trust_status, PeerTrustStatus::ConflictingClaims);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].subject_node_id, "forge-server");
}

#[test]
fn test_posture_claim_scope_validation() {
    use net_steward_discovery::reconciliation::reconcile_claims;
    use net_steward_schema::{
        CapabilityScope, ClaimEnvelope, ClaimStatus, ClaimVerificationStatus, DidAgentBinding,
        EncodingProfile, PeerPostureClaim, PostureSummary, SignatureScheme,
    };

    let now = 1719569000000;

    let claim = ClaimEnvelope {
        schema_version: "claim_envelope_v0.1".to_string(),
        encoding_profile: EncodingProfile::NetStewardCanonicalJsonV1,
        envelope_id: "env-scope-test".to_string(),
        payload: PeerPostureClaim {
            claim_id: "c-scope".to_string(),
            issuer_did: "did:mycelix:only-ports".to_string(),
            subject_node_id: "forge-server".to_string(),
            issued_at_unix_ms: now - 1000,
            expires_at_unix_ms: now + 60000,
            posture_summary: PostureSummary::Healthy,
            topology_refs: vec![],
            security_event_refs: vec![],
            evidence_refs: vec!["nixos_profile_collector".to_string()],
            capsule_refs: vec![],
            claim_status: ClaimStatus::Valid,
        },
        payload_hash: "hash-s".to_string(),
        issuer_did: "did:mycelix:only-ports".to_string(),
        signature: "sig-s".to_string(),
        signature_scheme: SignatureScheme::SimulatedEnvelope,
        verification_status: ClaimVerificationStatus::VerifiedSignature,
        signatures: None,
    };

    let binding = DidAgentBinding {
        issuer_did: "did:mycelix:only-ports".to_string(),
        agent_pubkey: "pk-ports".to_string(),
        device_id: None,
        public_key_multibase: "z6Mkm".to_string(),
        signature_scheme: SignatureScheme::SimulatedEnvelope,
        claim_scopes: vec![CapabilityScope::SecurityPostureObserve],
        valid_from_unix_ms: 1719513600000,
        expires_at_unix_ms: None,
        revoked: false,
        evidence_refs: vec![],
    };
    let claims = vec![claim];
    let (peers, _conflicts, accepted, rejected, _stale) =
        reconcile_claims(claims, now, vec![], vec![binding]);

    assert_eq!(accepted, 0);
    assert_eq!(rejected, 1);
    assert_eq!(peers.len(), 0);
}

#[test]
fn test_golden_claim_fixtures_verification() {
    use net_steward_schema::{
        CapabilityScope, ClaimEnvelope, ClaimSignatureVerifier, ClaimVerificationStatus,
        DidAgentBinding, Ed25519ClaimVerifier, PeerPostureClaim, SignatureScheme,
        SimulatedClaimVerifier,
    };

    let envelope_valid: ClaimEnvelope<PeerPostureClaim> =
        serde_json::from_str(include_str!("fixtures/claim_valid_ed25519.json")).unwrap();

    let binding = DidAgentBinding {
        issuer_did: "did:mycelix:trusted-1".to_string(),
        agent_pubkey: "pk-1".to_string(),
        device_id: None,
        public_key_multibase: "z6Mkm".to_string(),
        signature_scheme: SignatureScheme::SimulatedEnvelope,
        claim_scopes: vec![CapabilityScope::SecurityPostureObserve],
        valid_from_unix_ms: 1719513600000,
        expires_at_unix_ms: None,
        revoked: false,
        evidence_refs: vec![],
    };

    let verifier = SimulatedClaimVerifier;
    let status = verifier.verify_envelope_signature(&envelope_valid, Some(&binding));
    assert_eq!(status, ClaimVerificationStatus::VerifiedSignature);

    let ed_verifier = Ed25519ClaimVerifier;
    let ed_status = ed_verifier.verify_envelope_signature(&envelope_valid, Some(&binding));
    assert_eq!(ed_status, ClaimVerificationStatus::InvalidSignature);
}

#[test]
fn test_canonicalization_invariance_pretty_minified() {
    use net_steward_schema::{CanonicalClaimBytes, ClaimEnvelope, PeerPostureClaim};

    let envelope_a: ClaimEnvelope<PeerPostureClaim> =
        serde_json::from_str(include_str!("fixtures/claim_valid_pretty_a.json")).unwrap();
    let envelope_b: ClaimEnvelope<PeerPostureClaim> =
        serde_json::from_str(include_str!("fixtures/claim_valid_pretty_b_reordered.json")).unwrap();
    let envelope_min: ClaimEnvelope<PeerPostureClaim> =
        serde_json::from_str(include_str!("fixtures/claim_valid_minified.json")).unwrap();

    let bytes_a = envelope_a.payload.canonical_bytes().unwrap();
    let bytes_b = envelope_b.payload.canonical_bytes().unwrap();
    let bytes_min = envelope_min.payload.canonical_bytes().unwrap();

    // Verify canonical byte representations are identical regardless of format layout
    assert_eq!(bytes_a, bytes_b);
    assert_eq!(bytes_a, bytes_min);
}

#[test]
fn test_capsule_directory_bundle_serialization() {
    use net_steward_discovery::{
        create_incident_capsule, load_incident_capsule_from_directory,
        save_incident_capsule_to_directory,
    };

    let capsule = create_incident_capsule("sec-evt-101").unwrap();
    let temp_dir = std::env::temp_dir().join("incident_capsule_test_dir");

    // Save to directory bundle
    save_incident_capsule_to_directory(&capsule, &temp_dir).unwrap();

    // Load from directory bundle
    let loaded = load_incident_capsule_from_directory(&temp_dir).unwrap();

    assert_eq!(capsule.capsule_id, loaded.capsule_id);
    assert_eq!(capsule.target_event_id, loaded.target_event_id);
    assert_eq!(
        capsule.cryptographic_receipt_hash,
        loaded.cryptographic_receipt_hash
    );
    assert_eq!(capsule.security_events.len(), loaded.security_events.len());
    assert_eq!(capsule.evidence_ledger.len(), loaded.evidence_ledger.len());

    let _ = std::fs::remove_dir_all(temp_dir);
}

struct MockResolver {
    binding: DidAgentBinding,
}

impl DidBindingResolver for MockResolver {
    fn resolve_binding(&self, issuer_did: &str) -> Result<Option<DidAgentBinding>, String> {
        if self.binding.issuer_did == issuer_did {
            Ok(Some(self.binding.clone()))
        } else {
            Ok(None)
        }
    }
}

#[test]
fn test_reconciliation_with_dynamic_resolver() {
    use net_steward_discovery::reconciliation::reconcile_claims_with_resolver;
    use net_steward_schema::{
        CapabilityScope, ClaimEnvelope, ClaimStatus, ClaimVerificationStatus, DidAgentBinding,
        EncodingProfile, PeerPostureClaim, PostureSummary, SignatureScheme,
    };

    let now = 1719569000000;
    let claim = ClaimEnvelope {
        schema_version: "claim_envelope_v0.1".to_string(),
        encoding_profile: EncodingProfile::NetStewardCanonicalJsonV1,
        envelope_id: "env-scope-resolver-test".to_string(),
        payload: PeerPostureClaim {
            claim_id: "c-resolver".to_string(),
            issuer_did: "did:mycelix:dynamic-1".to_string(),
            subject_node_id: "forge-server".to_string(),
            issued_at_unix_ms: now - 1000,
            expires_at_unix_ms: now + 60000,
            posture_summary: PostureSummary::Healthy,
            topology_refs: vec![],
            security_event_refs: vec![],
            evidence_refs: vec!["nixos_profile_collector".to_string()],
            capsule_refs: vec![],
            claim_status: ClaimStatus::Valid,
        },
        payload_hash: "hash-r".to_string(),
        issuer_did: "did:mycelix:dynamic-1".to_string(),
        signature: "sig-r".to_string(),
        signature_scheme: SignatureScheme::SimulatedEnvelope,
        verification_status: ClaimVerificationStatus::VerifiedSignature,
        signatures: None,
    };

    let binding = DidAgentBinding {
        issuer_did: "did:mycelix:dynamic-1".to_string(),
        agent_pubkey: "pk-dyn".to_string(),
        device_id: None,
        public_key_multibase: "z6Mkm".to_string(),
        signature_scheme: SignatureScheme::SimulatedEnvelope,
        claim_scopes: vec![CapabilityScope::NixosGenerationObserve],
        valid_from_unix_ms: 1719513600000,
        expires_at_unix_ms: None,
        revoked: false,
        evidence_refs: vec![],
    };

    let resolver = MockResolver { binding };
    let (peers, _conflicts, accepted, rejected, _stale) =
        reconcile_claims_with_resolver(vec![claim], now, vec![], &resolver);

    assert_eq!(accepted, 1);
    assert_eq!(rejected, 0);
    assert_eq!(peers.len(), 1);
}

#[test]
fn test_federated_consensus_threshold_gating() {
    use net_steward_discovery::reconciliation::reconcile_claims;
    use net_steward_schema::{
        ClaimEnvelope, ClaimSignature, ClaimStatus, ClaimVerificationStatus, EncodingProfile,
        PeerPostureClaim, PeerTrustStatus, PostureSummary, SignatureScheme,
    };

    let now = 1719569000000;

    let co_signed_claim = ClaimEnvelope {
        schema_version: "claim_envelope_v0.1".to_string(),
        encoding_profile: EncodingProfile::NetStewardCanonicalJsonV1,
        envelope_id: "env-co-signed".to_string(),
        payload: PeerPostureClaim {
            claim_id: "c-co-signed".to_string(),
            issuer_did: "did:mycelix:trusted-1".to_string(),
            subject_node_id: "forge-server".to_string(),
            issued_at_unix_ms: now - 5000,
            expires_at_unix_ms: now + 60000,
            posture_summary: PostureSummary::Healthy,
            topology_refs: vec![],
            security_event_refs: vec![],
            evidence_refs: vec![],
            capsule_refs: vec![],
            claim_status: ClaimStatus::Valid,
        },
        payload_hash: "hash-co-signed".to_string(),
        issuer_did: "did:mycelix:trusted-1".to_string(),
        signature: "sig-1".to_string(),
        signature_scheme: SignatureScheme::Ed25519,
        verification_status: ClaimVerificationStatus::VerifiedSignature,
        signatures: Some(vec![ClaimSignature {
            issuer_did: "did:mycelix:trusted-2".to_string(),
            signature: "sig-2".to_string(),
            signature_scheme: SignatureScheme::Ed25519,
        }]),
    };

    let bindings = vec![
        net_steward_schema::DidAgentBinding {
            issuer_did: "did:mycelix:trusted-1".to_string(),
            agent_pubkey: "pk-1".to_string(),
            device_id: Some("dev-1".to_string()),
            public_key_multibase: "z6Mkm".to_string(),
            signature_scheme: SignatureScheme::Ed25519,
            claim_scopes: vec![net_steward_schema::CapabilityScope::SecurityPostureObserve],
            valid_from_unix_ms: now - 10000,
            expires_at_unix_ms: None,
            revoked: false,
            evidence_refs: vec![],
        },
        net_steward_schema::DidAgentBinding {
            issuer_did: "did:mycelix:trusted-2".to_string(),
            agent_pubkey: "pk-2".to_string(),
            device_id: Some("dev-2".to_string()),
            public_key_multibase: "z6Mkm".to_string(),
            signature_scheme: SignatureScheme::Ed25519,
            claim_scopes: vec![net_steward_schema::CapabilityScope::SecurityPostureObserve],
            valid_from_unix_ms: now - 10000,
            expires_at_unix_ms: None,
            revoked: false,
            evidence_refs: vec![],
        },
    ];

    let (peers, _conflicts, accepted, _rejected, _stale) =
        reconcile_claims(vec![co_signed_claim.clone()], now, vec![], bindings.clone());

    assert_eq!(accepted, 1);
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].trust_status, PeerTrustStatus::FederatedConsensus);
    // 1. Revoked DID check (revocation lists)
    let (peers_rev, _, _, _, _) = reconcile_claims(
        vec![co_signed_claim.clone()],
        now,
        vec!["did:mycelix:trusted-2".to_string()],
        bindings.clone(),
    );
    // Since one signer is revoked, active count is 1 < 2 -> should NOT be FederatedConsensus
    assert_ne!(
        peers_rev[0].trust_status,
        PeerTrustStatus::FederatedConsensus
    );

    // 2. Expired claim check
    let (peers_exp, _, _, _, _) = reconcile_claims(
        vec![co_signed_claim.clone()],
        now + 120000, // moves local time past expires_at_unix_ms (now + 60000)
        vec![],
        bindings.clone(),
    );
    // Expired claims cannot count -> unique signers == 0 -> should NOT be FederatedConsensus
    assert_ne!(
        peers_exp[0].trust_status,
        PeerTrustStatus::FederatedConsensus
    );

    // 3. Simulated envelope scheme check
    let mut simulated_claim = co_signed_claim.clone();
    simulated_claim.signature_scheme = SignatureScheme::SimulatedEnvelope;
    let (peers_sim, _, _, _, _) =
        reconcile_claims(vec![simulated_claim], now, vec![], bindings.clone());
    // SimulatedEnvelope does not count -> unique signers == 1 < 2 -> should NOT be FederatedConsensus
    assert_ne!(
        peers_sim[0].trust_status,
        PeerTrustStatus::FederatedConsensus
    );
}

#[test]
fn test_reconcile_adversarial_checks() {
    use net_steward_discovery::reconciliation::reconcile_claims;
    use net_steward_schema::{
        ClaimEnvelope, ClaimSignature, ClaimStatus, ClaimVerificationStatus, DidAgentBinding,
        EncodingProfile, PeerPostureClaim, PeerTrustStatus, PostureSummary, SignatureScheme,
    };

    let now = 1719569000000;

    let base_claim = ClaimEnvelope {
        schema_version: "claim_envelope_v0.1".to_string(),
        encoding_profile: EncodingProfile::NetStewardCanonicalJsonV1,
        envelope_id: "env-co-signed".to_string(),
        payload: PeerPostureClaim {
            claim_id: "c-co-signed".to_string(),
            issuer_did: "did:mycelix:trusted-1".to_string(),
            subject_node_id: "forge-server".to_string(),
            issued_at_unix_ms: now - 5000,
            expires_at_unix_ms: now + 60000,
            posture_summary: PostureSummary::Healthy,
            topology_refs: vec![],
            security_event_refs: vec![],
            evidence_refs: vec!["nixos_profile_collector".to_string()],
            capsule_refs: vec![],
            claim_status: ClaimStatus::Valid,
        },
        payload_hash: "hash-co-signed".to_string(),
        issuer_did: "did:mycelix:trusted-1".to_string(),
        signature: "sig-1".to_string(),
        signature_scheme: SignatureScheme::Ed25519,
        verification_status: ClaimVerificationStatus::VerifiedSignature,
        signatures: Some(vec![ClaimSignature {
            issuer_did: "did:mycelix:trusted-2".to_string(),
            signature: "sig-2".to_string(),
            signature_scheme: SignatureScheme::Ed25519,
        }]),
    };

    // 1. One real + one simulated signature does not reach consensus
    let mut real_simulated = base_claim.clone();
    if let Some(ref mut sigs) = real_simulated.signatures {
        sigs[0].signature_scheme = SignatureScheme::SimulatedEnvelope;
    }
    let (peers_real_sim, _, _, _, _) = reconcile_claims(vec![real_simulated], now, vec![], vec![]);
    assert_ne!(
        peers_real_sim[0].trust_status,
        PeerTrustStatus::FederatedConsensus
    );

    // 2. Duplicate signatures from same DID count once (co-signer is same as issuer)
    let mut dup_claim = base_claim.clone();
    if let Some(ref mut sigs) = dup_claim.signatures {
        sigs[0].issuer_did = "did:mycelix:trusted-1".to_string();
    }
    let (peers_dup, _, _, _, _) = reconcile_claims(vec![dup_claim], now, vec![], vec![]);
    assert_ne!(
        peers_dup[0].trust_status,
        PeerTrustStatus::FederatedConsensus
    );

    // 3. Out-of-scope signer cannot produce child/FederatedConsensus status
    let bindings = vec![DidAgentBinding {
        issuer_did: "did:mycelix:trusted-1".to_string(),
        agent_pubkey: "pk-1".to_string(),
        device_id: None,
        public_key_multibase: "z6Mkm".to_string(),
        signature_scheme: SignatureScheme::Ed25519,
        claim_scopes: vec![], // Nixos scope missing!
        valid_from_unix_ms: 1719513600000,
        expires_at_unix_ms: None,
        revoked: false,
        evidence_refs: vec![],
    }];
    let (peers_scope, _, _, rejected, _) =
        reconcile_claims(vec![base_claim.clone()], now, vec![], bindings);
    assert_eq!(rejected, 1);
    assert_eq!(peers_scope.len(), 0);
}

#[test]
fn test_identity_success_path_with_fixture_resolver() {
    use ed25519_dalek::Signer;
    use net_steward_discovery::reconciliation::reconcile_claims;
    use net_steward_schema::{
        CanonicalClaimBytes, CapabilityScope, ClaimEnvelope, ClaimSignature, ClaimStatus,
        ClaimVerificationStatus, DidAgentBinding, EncodingProfile, PeerPostureClaim,
        PeerTrustStatus, PostureSummary, SignatureScheme,
    };
    use rand::RngCore;
    use rand::rngs::OsRng;

    let now = 1719569000000;

    // Generate keys for two independent witnesses
    let mut csprng = OsRng;
    let mut secret1 = [0u8; 32];
    csprng.fill_bytes(&mut secret1);
    let key1 = ed25519_dalek::SigningKey::from_bytes(&secret1);
    let pk1_hex = hex::encode(key1.verifying_key().to_bytes());

    let mut secret2 = [0u8; 32];
    csprng.fill_bytes(&mut secret2);
    let key2 = ed25519_dalek::SigningKey::from_bytes(&secret2);
    let pk2_hex = hex::encode(key2.verifying_key().to_bytes());

    let payload = PeerPostureClaim {
        claim_id: "c-happy".to_string(),
        issuer_did: "did:mycelix:trusted-1".to_string(),
        subject_node_id: "forge-server".to_string(),
        issued_at_unix_ms: now - 5000,
        expires_at_unix_ms: now + 60000,
        posture_summary: PostureSummary::Healthy,
        topology_refs: vec![],
        security_event_refs: vec![],
        evidence_refs: vec!["nixos_profile_collector".to_string()],
        capsule_refs: vec![],
        claim_status: ClaimStatus::Valid,
    };

    let claim_bytes = payload.canonical_bytes().unwrap();
    let sig1 = key1.sign(&claim_bytes);
    let sig2 = key2.sign(&claim_bytes);

    let binding1 = DidAgentBinding {
        issuer_did: "did:mycelix:trusted-1".to_string(),
        agent_pubkey: pk1_hex.clone(),
        device_id: Some("dev-1".to_string()),
        public_key_multibase: "z6Mkm".to_string(),
        signature_scheme: SignatureScheme::Ed25519,
        claim_scopes: vec![
            CapabilityScope::SecurityPostureObserve,
            CapabilityScope::NixosGenerationObserve,
        ],
        valid_from_unix_ms: now - 10000,
        expires_at_unix_ms: None,
        revoked: false,
        evidence_refs: vec![],
    };

    let binding2 = DidAgentBinding {
        issuer_did: "did:mycelix:trusted-2".to_string(),
        agent_pubkey: pk2_hex.clone(),
        device_id: Some("dev-2".to_string()),
        public_key_multibase: "z6Mkm".to_string(),
        signature_scheme: SignatureScheme::Ed25519,
        claim_scopes: vec![
            CapabilityScope::SecurityPostureObserve,
            CapabilityScope::NixosGenerationObserve,
        ],
        valid_from_unix_ms: now - 10000,
        expires_at_unix_ms: None,
        revoked: false,
        evidence_refs: vec![],
    };

    // Assert binding details explicitly
    assert!(!binding1.revoked);
    assert_eq!(binding1.device_id, Some("dev-1".to_string()));
    assert!(
        binding1
            .claim_scopes
            .contains(&CapabilityScope::SecurityPostureObserve)
    );

    let single_signer_claim = ClaimEnvelope {
        schema_version: "claim_envelope_v0.1".to_string(),
        encoding_profile: EncodingProfile::NetStewardCanonicalJsonV1,
        envelope_id: "env-happy".to_string(),
        payload: payload.clone(),
        payload_hash: "hash-happy".to_string(),
        issuer_did: "did:mycelix:trusted-1".to_string(),
        signature: hex::encode(sig1.to_bytes()),
        signature_scheme: SignatureScheme::Ed25519,
        verification_status: ClaimVerificationStatus::VerifiedSignature,
        signatures: None,
    };

    // 1. Single verified bound claim yields VerifiedBoundFresh
    let (peers_single, _, accepted, _, _) = reconcile_claims(
        vec![single_signer_claim.clone()],
        now,
        vec![],
        vec![binding1.clone()],
    );
    assert_eq!(accepted, 1);
    assert_eq!(
        peers_single[0].trust_status,
        PeerTrustStatus::VerifiedBoundFresh
    );

    // 2. Two independent verified bound signers yield FederatedConsensus
    let co_signed_claim = ClaimEnvelope {
        schema_version: "claim_envelope_v0.1".to_string(),
        encoding_profile: EncodingProfile::NetStewardCanonicalJsonV1,
        envelope_id: "env-happy-2".to_string(),
        payload: payload.clone(),
        payload_hash: "hash-happy".to_string(),
        issuer_did: "did:mycelix:trusted-1".to_string(),
        signature: hex::encode(sig1.to_bytes()),
        signature_scheme: SignatureScheme::Ed25519,
        verification_status: ClaimVerificationStatus::VerifiedSignature,
        signatures: Some(vec![ClaimSignature {
            issuer_did: "did:mycelix:trusted-2".to_string(),
            signature: hex::encode(sig2.to_bytes()),
            signature_scheme: SignatureScheme::Ed25519,
        }]),
    };

    let (peers_fc, _, accepted_fc, _, _) = reconcile_claims(
        vec![co_signed_claim.clone()],
        now,
        vec![],
        vec![binding1.clone(), binding2.clone()],
    );
    assert_eq!(accepted_fc, 1);
    assert_eq!(
        peers_fc[0].trust_status,
        PeerTrustStatus::FederatedConsensus
    );

    // 3. Wrong key -> no VerifiedBoundFresh
    let mut wrong_key_claim = single_signer_claim.clone();
    wrong_key_claim.signature = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string();
    wrong_key_claim.verification_status = ClaimVerificationStatus::InvalidSignature;
    let (peers_wrong_key, _, _, _, _) =
        reconcile_claims(vec![wrong_key_claim], now, vec![], vec![binding1.clone()]);
    assert!(peers_wrong_key.is_empty());

    // 4. Revoked binding -> quarantined or rejected
    let mut revoked_binding = binding1.clone();
    revoked_binding.revoked = true;
    let (peers_revoked, _, _, rejected, _) = reconcile_claims(
        vec![single_signer_claim.clone()],
        now,
        vec![],
        vec![revoked_binding],
    );
    assert_eq!(rejected, 1);
    assert!(
        peers_revoked.is_empty() || peers_revoked[0].trust_status == PeerTrustStatus::Quarantined
    );

    // 5. Missing scope -> out of scope check
    let mut out_of_scope_binding = binding1.clone();
    out_of_scope_binding.claim_scopes = vec![];
    let (peers_out_of_scope, _, _, rejected_scope, _) = reconcile_claims(
        vec![single_signer_claim.clone()],
        now,
        vec![],
        vec![out_of_scope_binding],
    );
    assert_eq!(rejected_scope, 1);
    assert!(peers_out_of_scope.is_empty());

    // 6. Same device twice -> no consensus (remains VerifiedBoundFresh)
    let mut same_device_claim = co_signed_claim.clone();
    let mut binding2_same_dev = binding2.clone();
    binding2_same_dev.device_id = Some("dev-1".to_string());
    let (peers_same_device, _, _, _, _) = reconcile_claims(
        vec![same_device_claim],
        now,
        vec![],
        vec![binding1.clone(), binding2_same_dev],
    );
    assert_eq!(
        peers_same_device[0].trust_status,
        PeerTrustStatus::VerifiedBoundFresh
    );
}

#[cfg(feature = "holochain-conductor-tests")]
#[test]
fn test_identity_success_path_with_mycelix_holochain_resolver() {
    use net_steward_holochain::MycelixHolochainIdentityResolver;
    use net_steward_schema::DidBindingResolver;

    let resolver = MycelixHolochainIdentityResolver::new(
        "ws://127.0.0.1:8888",
        "net_steward",
        "agent_reputation",
    );
    let binding_opt = resolver
        .resolve_binding("did:mycelix:alice-test-key")
        .unwrap();
    assert!(binding_opt.is_some());
    let binding = binding_opt.unwrap();
    assert_eq!(binding.issuer_did, "did:mycelix:alice-test-key");
    assert!(!binding.revoked);
}

#[test]
fn test_operation_intent_serialization() {
    use net_steward_schema::{ApprovalEnvelope, OperationIntent, OperationKind, SignatureScheme};

    let intent = OperationIntent {
        intent_id: "intent-123".to_string(),
        actor_did: "did:mycelix:operator".to_string(),
        target_node_id: "forge-server".to_string(),
        operation_kind: OperationKind::GenerateRollbackPlan,
        reason: "System state drift audit".to_string(),
        evidence_refs: vec!["drift-ref".to_string()],
        rollback_plan_ref: Some("plan-ref".to_string()),
        expires_at_unix_ms: 1719569000000,
    };

    let serialized = serde_json::to_string(&intent).unwrap();
    let deserialized: OperationIntent = serde_json::from_str(&serialized).unwrap();
    assert_eq!(intent, deserialized);

    let approval = ApprovalEnvelope {
        intent_id: "intent-123".to_string(),
        approver_did: "did:mycelix:approver".to_string(),
        signature: "sig-abc".to_string(),
        signature_scheme: SignatureScheme::Ed25519,
        timestamp_unix_ms: 1719568000000,
    };

    let ser_app = serde_json::to_string(&approval).unwrap();
    let de_app: ApprovalEnvelope = serde_json::from_str(&ser_app).unwrap();
    assert_eq!(approval, de_app);
}

struct PolicyGateExecutor;

impl net_steward_schema::OperationExecutor for PolicyGateExecutor {
    fn dry_run(
        &self,
        intent: &net_steward_schema::OperationIntent,
    ) -> Result<net_steward_schema::ExecutionPlan, String> {
        Ok(net_steward_schema::ExecutionPlan {
            plan_id: "plan-123".to_string(),
            intent_id: intent.intent_id.clone(),
            execution_steps: vec![format!("Dry run step for {:?}", intent.operation_kind)],
            expected_target_state_hash: "hash-state".to_string(),
        })
    }

    fn apply(
        &self,
        _intent: &net_steward_schema::OperationIntent,
    ) -> Result<net_steward_schema::ExecutionResult, String> {
        Err("DisabledByPolicy".to_string())
    }
}

#[test]
fn test_executor_apply_disabled_by_policy() {
    use net_steward_schema::{OperationExecutor, OperationIntent, OperationKind};
    let intent = OperationIntent {
        intent_id: "intent-123".to_string(),
        actor_did: "did:mycelix:operator".to_string(),
        target_node_id: "forge-server".to_string(),
        operation_kind: OperationKind::GenerateRollbackPlan,
        reason: "System state drift audit".to_string(),
        evidence_refs: vec![],
        rollback_plan_ref: None,
        expires_at_unix_ms: 1719569000000,
    };

    let executor = PolicyGateExecutor;
    let plan = executor.dry_run(&intent).unwrap();
    assert_eq!(plan.execution_steps.len(), 1);

    let res = executor.apply(&intent);
    assert!(res.is_err());
    assert_eq!(res.err().unwrap(), "DisabledByPolicy");
}

// ─── v0.3-alpha.4: Blast-Radius Preview Tests ─────────────────────────────

/// Happy path: a GenerateRollbackPlan intent on a single isolated node
/// (no edges in the topology) should be Negligible risk — no interruption,
/// no witness approval required.
#[test]
fn test_blast_radius_negligible_for_readonly_isolated_node() {
    use net_steward_schema::{ManagementState, NodeKind, ObservedTopologySnapshot};

    let topology = ObservedTopologySnapshot {
        snapshot_id: "snap-isolated".to_string(),
        observed_at_unix_ms: 1719569000000,
        nodes: vec![net_steward_schema::InfrastructureNode {
            node_id: "vault-node".to_string(),
            hostname: Some("vault.local".to_string()),
            node_kind: NodeKind::Server,
            management_state: ManagementState::Managed,
            owner_did: None,
            site_id: None,
            observed_at_unix_ms: 1719569000000,
            source_collector: "test".to_string(),
            confidence: 1.0,
            staleness_ms: 0,
            evidence_hash: None,
        }],
        edges: vec![], // no edges — single isolated node
    };

    let intent = OperationIntent {
        intent_id: "intent-blast-1".to_string(),
        actor_did: "did:mycelix:operator".to_string(),
        target_node_id: "vault-node".to_string(),
        operation_kind: OperationKind::GenerateRollbackPlan,
        reason: "Audit drift".to_string(),
        evidence_refs: vec![],
        rollback_plan_ref: Some("plan-ref-0".to_string()),
        expires_at_unix_ms: 1719999999000,
    };

    let preview = generate_blast_radius_preview(&intent, &topology);

    assert_eq!(preview.intent_id, "intent-blast-1");
    assert_eq!(preview.risk_tier, BlastRadiusRiskTier::Negligible);
    assert!(!preview.requires_witness_approval);
    assert_eq!(preview.affected_services.len(), 1);
    assert_eq!(preview.transitive_node_ids, vec!["vault-node"]);
    assert!(preview.estimated_recovery_seconds.is_none());
    assert!(preview.rollback_plan_attached);
    assert!(preview.preview_commitment.is_some());
}

/// Topology with edges: the target node is connected to 2 additional nodes;
/// the preview should reflect all 3 in the transitive set and escalate to at
/// least Moderate risk.
#[test]
fn test_blast_radius_moderate_when_edge_connected_nodes_exist() {
    use net_steward_schema::{
        EdgeKind, ManagementState, NetworkEdge, NodeKind, ObservedTopologySnapshot,
    };

    let make_node = |id: &str, kind: NodeKind| net_steward_schema::InfrastructureNode {
        node_id: id.to_string(),
        hostname: None,
        node_kind: kind,
        management_state: ManagementState::Managed,
        owner_did: None,
        site_id: None,
        observed_at_unix_ms: 1719569000000,
        source_collector: "test".to_string(),
        confidence: 1.0,
        staleness_ms: 0,
        evidence_hash: None,
    };

    let topology = ObservedTopologySnapshot {
        snapshot_id: "snap-mesh".to_string(),
        observed_at_unix_ms: 1719569000000,
        nodes: vec![
            make_node("edge-router", NodeKind::Router),
            make_node("build-server", NodeKind::Server),
            make_node("ci-worker", NodeKind::Server),
        ],
        edges: vec![
            NetworkEdge {
                source_node_id: "edge-router".to_string(),
                target_node_id: "build-server".to_string(),
                edge_kind: EdgeKind::Route,
                confidence: 0.9,
                evidence_refs: vec![],
                source_collector: "test".to_string(),
                staleness_ms: 0,
                evidence_hash: None,
            },
            NetworkEdge {
                source_node_id: "edge-router".to_string(),
                target_node_id: "ci-worker".to_string(),
                edge_kind: EdgeKind::Route,
                confidence: 0.9,
                evidence_refs: vec![],
                source_collector: "test".to_string(),
                staleness_ms: 0,
                evidence_hash: None,
            },
        ],
    };

    let intent = OperationIntent {
        intent_id: "intent-blast-2".to_string(),
        actor_did: "did:mycelix:operator".to_string(),
        target_node_id: "edge-router".to_string(),
        operation_kind: OperationKind::ValidateRollbackPlan,
        reason: "Validate rollback pre-change".to_string(),
        evidence_refs: vec![],
        rollback_plan_ref: Some("plan-ref-1".to_string()),
        expires_at_unix_ms: 1719999999000,
    };

    let preview = generate_blast_radius_preview(&intent, &topology);

    // 3 nodes in transitive radius: the router + 2 connected servers
    assert_eq!(preview.transitive_node_ids.len(), 3);
    // Risk should be at least Moderate (gateway + 2 nodes)
    assert!(matches!(
        preview.risk_tier,
        BlastRadiusRiskTier::Moderate | BlastRadiusRiskTier::High
    ));
    assert!(preview.requires_witness_approval);
    assert!(preview.blast_radius_score >= 0.5);
}

/// Negative twin: PreviewFirewallChange with NO rollback plan attached must
/// always produce Critical risk tier, regardless of node count.
#[test]
fn test_blast_radius_critical_firewall_change_no_rollback() {
    use net_steward_schema::{ManagementState, NodeKind, ObservedTopologySnapshot};

    let topology = ObservedTopologySnapshot {
        snapshot_id: "snap-fw".to_string(),
        observed_at_unix_ms: 1719569000000,
        nodes: vec![net_steward_schema::InfrastructureNode {
            node_id: "border-fw".to_string(),
            hostname: Some("fw.luminous.local".to_string()),
            node_kind: NodeKind::Router,
            management_state: ManagementState::Managed,
            owner_did: None,
            site_id: None,
            observed_at_unix_ms: 1719569000000,
            source_collector: "test".to_string(),
            confidence: 1.0,
            staleness_ms: 0,
            evidence_hash: None,
        }],
        edges: vec![],
    };

    let intent = OperationIntent {
        intent_id: "intent-blast-3".to_string(),
        actor_did: "did:mycelix:operator".to_string(),
        target_node_id: "border-fw".to_string(),
        operation_kind: OperationKind::PreviewFirewallChange,
        reason: "Tighten egress rules".to_string(),
        evidence_refs: vec![],
        rollback_plan_ref: None, // ← no rollback plan
        expires_at_unix_ms: 1719999999000,
    };

    let preview = generate_blast_radius_preview(&intent, &topology);

    assert_eq!(
        preview.risk_tier,
        BlastRadiusRiskTier::Critical,
        "PreviewFirewallChange without a rollback plan must be Critical"
    );
    assert!(preview.requires_witness_approval);
    assert!(!preview.rollback_plan_attached);
    assert_eq!(preview.blast_radius_score, 1.0);
    assert_eq!(preview.estimated_recovery_seconds, Some(1800));
}

/// Commitment stability: two calls for the same intent and topology in the
/// same second should produce the same commitment (deterministic modulo the
/// timestamp component — we only verify the commitment is non-zero and
/// the preview is self-consistent).
#[test]
fn test_blast_radius_preview_commitment_is_nonzero_and_stable_within_call() {
    use net_steward_schema::{ManagementState, NodeKind, ObservedTopologySnapshot};

    let topology = ObservedTopologySnapshot {
        snapshot_id: "snap-commit".to_string(),
        observed_at_unix_ms: 1719569000000,
        nodes: vec![net_steward_schema::InfrastructureNode {
            node_id: "commit-node".to_string(),
            hostname: None,
            node_kind: NodeKind::Server,
            management_state: ManagementState::Managed,
            owner_did: None,
            site_id: None,
            observed_at_unix_ms: 1719569000000,
            source_collector: "test".to_string(),
            confidence: 1.0,
            staleness_ms: 0,
            evidence_hash: None,
        }],
        edges: vec![],
    };

    let intent = OperationIntent {
        intent_id: "intent-commit-test".to_string(),
        actor_did: "did:mycelix:operator".to_string(),
        target_node_id: "commit-node".to_string(),
        operation_kind: OperationKind::ExportIncidentCapsule,
        reason: "Audit export".to_string(),
        evidence_refs: vec![],
        rollback_plan_ref: None,
        expires_at_unix_ms: 1719999999000,
    };

    let preview = generate_blast_radius_preview(&intent, &topology);

    // Commitment must be present and non-zero.
    let commitment = preview
        .preview_commitment
        .expect("commitment should be present");
    assert_ne!(
        commitment, [0u8; 32],
        "commitment must not be the zero hash"
    );

    // preview_id must embed the intent_id.
    assert!(preview.preview_id.contains("intent-commit-test"));

    // Summary must mention the operation kind and node.
    assert!(preview.summary.contains("commit-node"));
}
