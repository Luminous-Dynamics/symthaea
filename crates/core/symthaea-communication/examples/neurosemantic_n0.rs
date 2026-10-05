use std::collections::{BTreeMap, BTreeSet};

use symthaea_communication::{
    AuthorizedNeurosemanticMessage, CognitiveChannel, CognitiveConsentLease,
    CognitiveSensitivity, CommunicationPurpose, ConceptKind, ConceptNode, GroundedConceptGraph,
    ChannelDirection, ExpressionDecision, ExpressionPolicy, ExpressionTarget,
    NeurosemanticPacket, NeurosemanticPayload, NeurosemanticReplayTracker, RepresentationFamily,
    NeurosemanticDataClass, NeurosemanticDataPolicy, NeurosemanticInferenceClass,
    NeurosemanticHandlingPolicy, NeurosemanticRetentionPolicy, ReplayDecision,
};

fn main() -> Result<(), String> {
    let graph = GroundedConceptGraph {
        nodes: vec![
            ConceptNode {
                id: "event-1".into(),
                kind: ConceptKind::Event,
                label: Some("approach".into()),
                grounded_by: vec!["obs-1".into()],
                confidence: 0.99,
            },
            ConceptNode {
                id: "agent-1".into(),
                kind: ConceptKind::Agent,
                label: Some("sender".into()),
                grounded_by: vec!["obs-1".into()],
                confidence: 0.99,
            },
            ConceptNode {
                id: "object-1".into(),
                kind: ConceptKind::Object,
                label: Some("object".into()),
                grounded_by: vec!["obs-1".into()],
                confidence: 0.94,
            },
        ],
        edges: vec![
            symthaea_communication::ConceptEdge {
                source: "agent-1".into(),
                relation: "initiates".into(),
                target: "event-1".into(),
                evidence_ids: vec!["obs-1".into()],
                confidence: 0.96,
            },
            symthaea_communication::ConceptEdge {
                source: "event-1".into(),
                relation: "targets".into(),
                target: "object-1".into(),
                evidence_ids: vec!["obs-1".into()],
                confidence: 0.95,
            },
        ],
    };

    let graph_bytes = serde_json::to_vec(&graph).map_err(|e| e.to_string())?;
    let decoded: GroundedConceptGraph =
        serde_json::from_slice(&graph_bytes).map_err(|e| e.to_string())?;
    let exact_roundtrip = decoded == graph;

    let lease = CognitiveConsentLease {
        lease_id: "n0-lease".into(),
        subject_id: "subject".into(),
        peer_id: "peer".into(),
        purpose: CommunicationPurpose::HumanCollaboration,
        read_scopes: BTreeSet::from([CognitiveChannel::Semantic]),
        write_scopes: BTreeSet::from([CognitiveChannel::Semantic]),
        max_read_sensitivity: CognitiveSensitivity::Private,
        max_write_sensitivity: CognitiveSensitivity::Private,
        read_data_classes: BTreeSet::from([NeurosemanticDataClass::SemanticRepresentation]),
        write_data_classes: BTreeSet::from([NeurosemanticDataClass::SemanticRepresentation]),
        read_inference_classes: BTreeSet::from([NeurosemanticInferenceClass::SemanticContent]),
        write_inference_classes: BTreeSet::from([NeurosemanticInferenceClass::SemanticContent]),
        issued_at_unix_s: 1_000,
        expires_at_unix_s: 2_000,
        consent_epoch: 4,
        revoked: false,
        revoked_at_unix_s: None,
    };

    let packet = NeurosemanticPacket::new_with_policy(
        1,
        "peer",
        "subject",
        CommunicationPurpose::HumanCollaboration,
        CognitiveChannel::Semantic,
        ChannelDirection::Write,
        RepresentationFamily::Custom("GroundedConceptGraph".into()),
        CognitiveSensitivity::Private,
        NeurosemanticDataPolicy {
            schema_version: symthaea_communication::NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            data_class: NeurosemanticDataClass::SemanticRepresentation,
            inference_classes: BTreeSet::from([NeurosemanticInferenceClass::SemanticContent]),
            permitted_purposes: BTreeSet::from([CommunicationPurpose::HumanCollaboration]),
            handling: NeurosemanticHandlingPolicy {
                schema_version: symthaea_communication::NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
                policy_provenance_ref: "synthetic-policy-record-1".into(),
                policy_provenance_hash: symthaea_communication::content_hash(b"synthetic-policy-record-1"),
                origin_jurisdiction: "ZA".into(),
                permitted_destination_jurisdictions: BTreeSet::from(["ZA".into()]),
                permitted_secondary_uses: BTreeSet::new(),
                retention: NeurosemanticRetentionPolicy::UntilUnixS(2_000),
            },
        },
        0.93,
        NeurosemanticPayload::SemanticGraph(graph_bytes.clone()),
    )?;

    let message = AuthorizedNeurosemanticMessage {
        packet,
        consent_epoch: lease.consent_epoch,
        lease_id: lease.lease_id.clone(),
    };
    message.validate(&lease, 1_500)?;
    let handling_policy_provenance_present = !message
        .packet
        .data_policy
        .handling
        .policy_provenance_ref
        .is_empty();
    let handling_policy_provenance_hash_valid = message
        .packet
        .data_policy
        .handling
        .verify_policy_provenance_bytes(b"synthetic-policy-record-1");
    let mut provenance_mismatch = message.clone();
    provenance_mismatch.packet.data_policy.handling.policy_provenance_hash =
        symthaea_communication::content_hash(b"synthetic-policy-record-2");
    provenance_mismatch.packet.refresh_hashes()?;
    let handling_policy_provenance_mismatch_blocked = !provenance_mismatch
        .packet
        .data_policy
        .handling
        .verify_policy_provenance_bytes(b"synthetic-policy-record-1");
    message.validate_for_handling(
        &lease,
        "ZA",
        symthaea_communication::NeurosemanticHandlingAction::Transmit,
        1_500,
    )?;

    let mut inference_escalation = message.clone();
    inference_escalation
        .packet
        .data_policy
        .handling
        .permitted_secondary_uses
        .insert(symthaea_communication::NeurosemanticSecondaryUse::AffectiveInference);
    inference_escalation.packet.refresh_hashes()?;
    let mut inference_aware_lease = lease.clone();
    inference_aware_lease
        .write_inference_classes
        .insert(NeurosemanticInferenceClass::AffectiveState);
    let inference_escalation_blocked = inference_escalation
        .validate_for_handling(
            &inference_aware_lease,
            "ZA",
            symthaea_communication::NeurosemanticHandlingAction::SecondaryUse(
                symthaea_communication::NeurosemanticSecondaryUse::AffectiveInference,
            ),
            1_500,
        )
        .is_err();

    let mut replay = NeurosemanticReplayTracker::default();
    let accepted = replay.observe_authorized(&message, &lease, 1_500)?;
    let duplicate = replay.observe_authorized(&message, &lease, 1_500)?;
    let expired_replay_state_reclaimed = replay.prune_expired(2_000) == 1;

    let mut scheduled_revocation = lease.clone();
    scheduled_revocation.revoked_at_unix_s = Some(1_750);
    let scheduled_revocation_honored = message.validate(&scheduled_revocation, 1_500).is_ok()
        && message.validate(&scheduled_revocation, 1_750).is_err();

    let mut tampered = message.clone();
    if let NeurosemanticPayload::SemanticGraph(bytes) = &mut tampered.packet.payload {
        bytes.push(0);
    }
    let tamper_detected = tampered.packet.validate_integrity().is_err();

    let mut unauthorized = message.clone();
    unauthorized.packet.channel = CognitiveChannel::Affective;
    unauthorized.packet.refresh_hashes()?;
    let unauthorized_blocked = unauthorized.validate(&lease, 1_500).is_err();

    let expired_blocked = message.validate(&lease, 2_000).is_err();
    let cross_jurisdiction_blocked = message
        .validate_for_handling(
            &lease,
            "GB",
            symthaea_communication::NeurosemanticHandlingAction::Transmit,
            1_500,
        )
        .is_err();
    let persistence_after_expiry_blocked = message
        .validate_for_handling(
            &lease,
            "ZA",
            symthaea_communication::NeurosemanticHandlingAction::Persist,
            2_000,
        )
        .is_err();
    let secondary_research_blocked = message
        .validate_for_handling(
            &lease,
            "ZA",
            symthaea_communication::NeurosemanticHandlingAction::SecondaryUse(
                symthaea_communication::NeurosemanticSecondaryUse::Research,
            ),
            1_500,
        )
        .is_err();

    let mut sensitivity_escalation = message.clone();
    sensitivity_escalation.packet.sensitivity = CognitiveSensitivity::HighlyPrivate;
    sensitivity_escalation.packet.refresh_hashes()?;
    let sensitivity_blocked = sensitivity_escalation.validate(&lease, 1_500).is_err();

    let expression_blocked = matches!(
        ExpressionPolicy.evaluate(
            &ExpressionTarget::Human,
            symthaea_communication::CapabilityLevel::Structure,
            &[],
            None,
        ),
        ExpressionDecision::Block(_)
    );

    let report = BTreeMap::from([
        ("exact_graph_roundtrip", exact_roundtrip),
        ("authorization_valid", true),
        ("handling_policy_provenance_present", handling_policy_provenance_present),
        ("handling_policy_provenance_hash_valid", handling_policy_provenance_hash_valid),
        ("handling_policy_provenance_mismatch_blocked", handling_policy_provenance_mismatch_blocked),
        ("inference_escalation_blocked", inference_escalation_blocked),
        ("first_packet_accepted", accepted == ReplayDecision::Accept),
        ("exact_replay_detected", duplicate == ReplayDecision::Duplicate),
        ("expired_replay_state_reclaimed", expired_replay_state_reclaimed),
        ("scheduled_revocation_honored", scheduled_revocation_honored),
        ("tamper_detected", tamper_detected),
        ("unauthorized_channel_blocked", unauthorized_blocked),
        ("expired_lease_blocked", expired_blocked),
        ("cross_jurisdiction_blocked", cross_jurisdiction_blocked),
        ("persistence_after_expiry_blocked", persistence_after_expiry_blocked),
        ("secondary_research_blocked", secondary_research_blocked),
        ("sensitivity_escalation_blocked", sensitivity_blocked),
        ("insufficient_capability_blocked", expression_blocked),
    ]);

    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    Ok(())
}
