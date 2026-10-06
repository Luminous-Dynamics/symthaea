use std::collections::BTreeSet;

use symthaea_communication::{
    AuthorizedNeurosemanticMessage, CognitiveChannel, CognitiveConsentLease,
    CognitiveSensitivity, CommunicationPurpose, ConceptKind, ConceptNode, GroundedConceptGraph,
    ChannelDirection, ExpressionDecision, ExpressionPolicy, ExpressionTarget,
    NeurosemanticPacket, NeurosemanticPayload, NeurosemanticReplayTracker, RepresentationFamily,
    NeurosemanticDataClass, NeurosemanticDataPolicy, NeurosemanticInferenceClass,
    NeurosemanticHandlingPolicy, NeurosemanticRetentionPolicy, ReplayDecision,
    NeurosemanticPolicyAuthorityAttestation, NeurosemanticPolicyProvenanceBinding,
    NeurosemanticAuthorityResolutionAttestation, NeurosemanticAuthorityStatus,
    NeurosemanticConsentBindingContext,
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
                policy_provenance_hash: symthaea_communication::compute_policy_provenance_hash(
                    "synthetic-policy-record-1",
                    b"synthetic-policy-record-1",
                ),
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
    use ed25519_dalek::Signer;
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]);
    let policy_fingerprint =
        message.packet.data_policy.handling.fingerprint_for_attestation()?;
    let authority_message = NeurosemanticPolicyAuthorityAttestation::message_bytes(
        "mycelix-policy-authority",
        "test-key-1",
        &policy_fingerprint,
        &message.packet.data_policy.handling.policy_provenance_hash,
        1_000,
        2_000,
    )?;
    let authority_attestation = NeurosemanticPolicyAuthorityAttestation {
        schema_version: symthaea_communication::NEUROSEMANTIC_POLICY_ATTESTATION_SCHEMA_VERSION,
        authority_ref: "mycelix-policy-authority".into(),
        key_ref: "test-key-1".into(),
        handling_policy_fingerprint: policy_fingerprint,
        policy_provenance_hash: message.packet.data_policy.handling.policy_provenance_hash.clone(),
        issued_at_unix_s: 1_000,
        expires_at_unix_s: 2_000,
        signature: signing_key.sign(&authority_message).to_bytes().to_vec(),
    };
    let resolution_context = NeurosemanticConsentBindingContext {
        subject_ref: lease.subject_id.clone(),
        peer_ref: lease.peer_id.clone(),
        lease_id: lease.lease_id.clone(),
        consent_epoch: lease.consent_epoch,
        consent_lease_fingerprint: lease.fingerprint_for_authorization()?,
        purpose: message.packet.purpose,
        channel: message.packet.channel,
        direction: message.packet.direction,
    };
    let resolver_signing_key = ed25519_dalek::SigningKey::from_bytes(&[9u8; 32]);
    let mut authority_resolution = NeurosemanticAuthorityResolutionAttestation {
        schema_version: symthaea_communication::NEUROSEMANTIC_AUTHORITY_RESOLUTION_SCHEMA_VERSION,
        resolver_ref: "mycelix-identity-policy-bridge".into(),
        resolver_key_ref: "resolver-key-1".into(),
        authority_ref: authority_attestation.authority_ref.clone(),
        authority_key_ref: authority_attestation.key_ref.clone(),
        authority_attestation_fingerprint: authority_attestation.fingerprint_for_attestation()?,
        handling_policy_fingerprint: policy_fingerprint.clone(),
        policy_provenance_ref: message.packet.data_policy.handling.policy_provenance_ref.clone(),
        policy_provenance_hash: message.packet.data_policy.handling.policy_provenance_hash.clone(),
        subject_ref: resolution_context.subject_ref.clone(),
        peer_ref: resolution_context.peer_ref.clone(),
        lease_id: resolution_context.lease_id.clone(),
        consent_epoch: resolution_context.consent_epoch,
        purpose: resolution_context.purpose,
        channel: resolution_context.channel,
        direction: resolution_context.direction,
        status: NeurosemanticAuthorityStatus::Active,
        status_source_ref: "mycelix-status:synthetic-1".into(),
        checked_at_unix_s: 1_200,
        expires_at_unix_s: 1_800,
        signature: Vec::new(),
    };
    let resolution_message = authority_resolution.message_bytes()?;
    authority_resolution.signature =
        ed25519_dalek::Signer::sign(&resolver_signing_key, &resolution_message)
            .to_bytes()
            .to_vec();

    let policy_provenance_binding: NeurosemanticPolicyProvenanceBinding = message
        .packet
        .data_policy
        .handling
        .bind_policy_provenance_with_attestation_and_resolution(
            b"synthetic-policy-record-1",
            &authority_attestation,
            &signing_key.verifying_key(),
            &authority_resolution,
            &resolver_signing_key.verifying_key(),
            &resolution_context,
            1_500,
        )?;
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
        .verify_policy_record_binding_bytes(b"synthetic-policy-record-1");

    let mut reference_mismatch = message.clone();
    reference_mismatch
        .packet
        .data_policy
        .handling
        .policy_provenance_ref = "synthetic-policy-record-2".into();
    reference_mismatch.packet.refresh_hashes()?;
    let handling_policy_provenance_reference_mismatch_blocked = !reference_mismatch
        .packet
        .data_policy
        .handling
        .verify_policy_record_binding_bytes(b"synthetic-policy-record-1");
    let mut provenance_mismatch = message.clone();
    provenance_mismatch.packet.data_policy.handling.policy_provenance_hash =
        symthaea_communication::compute_policy_provenance_hash(
            "synthetic-policy-record-1",
            b"synthetic-policy-record-2",
        );
    provenance_mismatch.packet.refresh_hashes()?;
    let handling_policy_provenance_mismatch_blocked = !provenance_mismatch
        .packet
        .data_policy
        .handling
        .verify_policy_record_binding_bytes(b"synthetic-policy-record-1");
    message.validate_for_handling(
        &lease,
        &policy_provenance_binding,
        "ZA",
        symthaea_communication::NeurosemanticHandlingAction::Transmit,
        1_500,
    )?;

    let mut stale_binding = message.clone();
    stale_binding
        .packet
        .data_policy
        .handling
        .permitted_secondary_uses
        .insert(symthaea_communication::NeurosemanticSecondaryUse::Research);
    stale_binding.packet.refresh_hashes()?;
    let stale_policy_binding_blocked = stale_binding
        .validate_for_handling(
            &lease,
            &policy_provenance_binding,
            "ZA",
            symthaea_communication::NeurosemanticHandlingAction::SecondaryUse(
                symthaea_communication::NeurosemanticSecondaryUse::Research,
            ),
            1_500,
        )
        .is_err();

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
            &policy_provenance_binding,
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
            &policy_provenance_binding,
            "GB",
            symthaea_communication::NeurosemanticHandlingAction::Transmit,
            1_500,
        )
        .is_err();
    let persistence_after_expiry_blocked = message
        .validate_for_handling(
            &lease,
            &policy_provenance_binding,
            "ZA",
            symthaea_communication::NeurosemanticHandlingAction::Persist,
            2_000,
        )
        .is_err();
    let secondary_research_blocked = message
        .validate_for_handling(
            &lease,
            &policy_provenance_binding,
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

    let execution_revision = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|revision| revision.trim().to_string())
        .filter(|revision| revision.len() == 40 || revision.len() == 64)
        .filter(|revision| revision.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .filter(|revision| revision.bytes().any(|byte| byte != b'0'))
        .ok_or_else(|| "unable to resolve exact execution revision".to_string())?;

    let substituted_authority_attestation = {
        let changed_message = NeurosemanticPolicyAuthorityAttestation::message_bytes(
            &authority_attestation.authority_ref,
            &authority_attestation.key_ref,
            &authority_attestation.handling_policy_fingerprint,
            &authority_attestation.policy_provenance_hash,
            1_100,
            1_900,
        )?;
        NeurosemanticPolicyAuthorityAttestation {
            issued_at_unix_s: 1_100,
            expires_at_unix_s: 1_900,
            signature: signing_key.sign(&changed_message).to_bytes().to_vec(),
            ..authority_attestation.clone()
        }
    };
    let authority_proof_substitution_blocked =
        substituted_authority_attestation.verify(
            &policy_fingerprint,
            &message.packet.data_policy.handling.policy_provenance_hash,
            &signing_key.verifying_key(),
            1_500,
        ).is_ok()
        && message
            .packet
            .data_policy
            .handling
            .bind_policy_provenance_with_attestation_and_resolution(
                b"synthetic-policy-record-1",
                &substituted_authority_attestation,
                &signing_key.verifying_key(),
                &authority_resolution,
                &resolver_signing_key.verifying_key(),
                &resolution_context,
                1_500,
            )
            .is_err();

    let mut forged_resolution = authority_resolution.clone();
    forged_resolution.status = NeurosemanticAuthorityStatus::Revoked;
    forged_resolution.signature =
        ed25519_dalek::Signer::sign(&resolver_signing_key, &forged_resolution.message_bytes()?)
            .to_bytes()
            .to_vec();
    let revoked_resolution_blocked = message
        .packet
        .data_policy
        .handling
        .bind_policy_provenance_with_attestation_and_resolution(
            b"synthetic-policy-record-1",
            &authority_attestation,
            &signing_key.verifying_key(),
            &forged_resolution,
            &resolver_signing_key.verifying_key(),
            &resolution_context,
            1_500,
        )
        .is_err();

    let mut wrong_context = resolution_context.clone();
    wrong_context.subject_ref = "other-subject".into();
    let wrong_context_resolution = {
        let mut candidate = authority_resolution.clone();
        candidate.subject_ref = wrong_context.subject_ref.clone();
        candidate.signature =
            ed25519_dalek::Signer::sign(&resolver_signing_key, &candidate.message_bytes()?)
                .to_bytes()
                .to_vec();
        candidate
    };
    let wrong_resolution_context_blocked = message
        .packet
        .data_policy
        .handling
        .bind_policy_provenance_with_attestation_and_resolution(
            b"synthetic-policy-record-1",
            &authority_attestation,
            &signing_key.verifying_key(),
            &wrong_context_resolution,
            &resolver_signing_key.verifying_key(),
            &resolution_context,
            1_500,
        )
        .is_err();

    let mut tampered_resolution = authority_resolution.clone();
    tampered_resolution.signature[0] ^= 1;
    let resolution_signature_blocked = message
        .packet
        .data_policy
        .handling
        .bind_policy_provenance_with_attestation_and_resolution(
            b"synthetic-policy-record-1",
            &authority_attestation,
            &signing_key.verifying_key(),
            &tampered_resolution,
            &resolver_signing_key.verifying_key(),
            &resolution_context,
            1_500,
        )
        .is_err();

    let mut pre_attestation_resolution = authority_resolution.clone();
    pre_attestation_resolution.checked_at_unix_s = 900;
    pre_attestation_resolution.signature =
        ed25519_dalek::Signer::sign(&resolver_signing_key, &pre_attestation_resolution.message_bytes()?)
            .to_bytes()
            .to_vec();
    let pre_attestation_resolution_blocked = message
        .packet
        .data_policy
        .handling
        .bind_policy_provenance_with_attestation_and_resolution(
            b"synthetic-policy-record-1",
            &authority_attestation,
            &signing_key.verifying_key(),
            &pre_attestation_resolution,
            &resolver_signing_key.verifying_key(),
            &resolution_context,
            1_500,
        )
        .is_err();

    let authority_resolution_fresh_until = authority_resolution.expires_at_unix_s;
    let authority_resolution_expiry_blocked = message
        .validate_for_handling(
            &lease,
            &policy_provenance_binding,
            "ZA",
            symthaea_communication::NeurosemanticHandlingAction::Transmit,
            authority_resolution_fresh_until,
        )
        .is_err();

    let mut mutated_lease = lease.clone();
    mutated_lease.max_write_sensitivity = CognitiveSensitivity::HighlyPrivate;
    let mutated_lease_blocked = message
        .validate_for_handling(
            &mutated_lease,
            &policy_provenance_binding,
            "ZA",
            symthaea_communication::NeurosemanticHandlingAction::Transmit,
            1_500,
        )
        .is_err();

    let report = serde_json::json!({
        "execution_revision": execution_revision,
        "exact_graph_roundtrip": exact_roundtrip,
        "authorization_valid": true,
        "handling_policy_provenance_present": handling_policy_provenance_present,
        "handling_policy_provenance_hash_valid": handling_policy_provenance_hash_valid,
        "handling_policy_provenance_reference_mismatch_blocked":
            handling_policy_provenance_reference_mismatch_blocked,
        "handling_policy_provenance_mismatch_blocked": handling_policy_provenance_mismatch_blocked,
        "stale_policy_provenance_binding_blocked": stale_policy_binding_blocked,
        "inference_escalation_blocked": inference_escalation_blocked,
        "first_packet_accepted": accepted == ReplayDecision::Accept,
        "exact_replay_detected": duplicate == ReplayDecision::Duplicate,
        "expired_replay_state_reclaimed": expired_replay_state_reclaimed,
        "scheduled_revocation_honored": scheduled_revocation_honored,
        "tamper_detected": tamper_detected,
        "unauthorized_channel_blocked": unauthorized_blocked,
        "expired_lease_blocked": expired_blocked,
        "cross_jurisdiction_blocked": cross_jurisdiction_blocked,
        "persistence_after_expiry_blocked": persistence_after_expiry_blocked,
        "secondary_research_blocked": secondary_research_blocked,
        "sensitivity_escalation_blocked": sensitivity_blocked,
        "insufficient_capability_blocked": expression_blocked,
        "revoked_authority_resolution_blocked": revoked_resolution_blocked,
        "wrong_authority_resolution_context_blocked": wrong_resolution_context_blocked,
        "authority_resolution_signature_blocked": resolution_signature_blocked,
        "authority_resolution_expiry_blocked": authority_resolution_expiry_blocked,
        "authority_resolution_predating_attestation_blocked": pre_attestation_resolution_blocked,
        "mutated_consent_lease_blocked": mutated_lease_blocked,
        "authority_proof_substitution_blocked": authority_proof_substitution_blocked
    });

    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    Ok(())
}
