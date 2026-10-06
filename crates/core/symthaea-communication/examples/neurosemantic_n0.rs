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
    NeurosemanticConsentBindingContext, NeurosemanticDerivationLineageRecord,
    NeurosemanticArtifactLifecycleReceipt, NeurosemanticArtifactLifecycleAction,
    NeurosemanticArtifactLifecycleState,
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

    let derivation_payload = NeurosemanticPayload::SemanticGraph(graph_bytes.clone());
    let derivation_output_artifact_hash = symthaea_communication::content_hash(
        &serde_json::to_vec(&derivation_payload).map_err(|e| e.to_string())?,
    );
    let derivation_lineage_record = NeurosemanticDerivationLineageRecord {
        schema_version: symthaea_communication::NEUROSEMANTIC_DERIVATION_LINEAGE_SCHEMA_VERSION,
        lineage_ref: "synthetic-derivation-record-1".into(),
        input_artifact_refs: vec!["synthetic-input-1".into(), "synthetic-input-2".into()],
        input_artifact_hashes: vec![
            symthaea_communication::content_hash(b"synthetic-input-artifact-1"),
            symthaea_communication::content_hash(b"synthetic-input-artifact-2"),
        ],
        activity_ref: "synthetic-semantic-graph-transform".into(),
        activity_revision: "transform-v1".into(),
        output_artifact_hash: derivation_output_artifact_hash.clone(),
        execution_revision: execution_revision.clone(),
        generated_at_unix_s: 1_200,
    };
    let derivation_record_bytes =
        serde_json::to_vec(&derivation_lineage_record).map_err(|e| e.to_string())?;

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
                derivation_provenance_ref: "synthetic-derivation-record-1".into(),
                derivation_provenance_hash: symthaea_communication::compute_derivation_provenance_hash(
                    "synthetic-derivation-record-1",
                    &derivation_record_bytes,
                ),
                derivation_output_artifact_hash,
                origin_jurisdiction: "ZA".into(),
                permitted_destination_jurisdictions: BTreeSet::from(["ZA".into()]),
                permitted_secondary_uses: BTreeSet::new(),
                retention: NeurosemanticRetentionPolicy::UntilUnixS(2_000),
                max_authority_resolution_age_s: 300,
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
        consent_lease_fingerprint: resolution_context.consent_lease_fingerprint.clone(),
        purpose: resolution_context.purpose,
        channel: resolution_context.channel,
        direction: resolution_context.direction,
        status: NeurosemanticAuthorityStatus::Active,
        status_source_ref: "mycelix-status:synthetic-1".into(),
        status_source_hash: symthaea_communication::compute_status_source_hash(
            "mycelix-status:synthetic-1",
            b"synthetic-status-record-1",
        ),
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
            &derivation_record_bytes,
            b"synthetic-status-record-1",
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

    let derivation_input_artifact_verified = derivation_lineage_record
        .verify_input_artifact_bytes(0, b"synthetic-input-artifact-1")
        .is_ok()
        && derivation_lineage_record
            .verify_input_artifact_bytes(1, b"synthetic-input-artifact-2")
            .is_ok();
    let derivation_input_artifact_mismatch_blocked = derivation_lineage_record
        .verify_input_artifact_bytes(0, b"synthetic-input-artifact-tampered")
        .is_err();
    let derivation_input_artifact_bounds_blocked = derivation_lineage_record
        .verify_input_artifact_bytes(2, b"synthetic-input-artifact-3")
        .is_err();

    let lifecycle_effect_evidence = b"synthetic-lifecycle-effect-v1";
    let lifecycle_verification_evidence = b"synthetic-independent-verification-v1";
    let lifecycle_verification_scope = b"synthetic-target-set-v1:derived-artifact-only";
    let lifecycle_verification_scope_ref = "synthetic-verification-scope-1";
    let derivation_ref = policy_provenance_binding.derivation_provenance_ref().to_string();
    let derivation_hash = policy_provenance_binding.derivation_provenance_hash().to_string();
    let lifecycle_scope_hash = symthaea_communication::compute_lifecycle_verification_scope_hash(
        lifecycle_verification_scope_ref,
        lifecycle_verification_scope,
    );
    let lifecycle_base = NeurosemanticArtifactLifecycleReceipt {
        schema_version: symthaea_communication::NEUROSEMANTIC_ARTIFACT_LIFECYCLE_RECEIPT_SCHEMA_VERSION,
        receipt_ref: "synthetic-lifecycle-receipt-1".into(),
        artifact_hash: message.packet.payload_hash.clone(),
        derivation_provenance_ref: derivation_ref.clone(),
        derivation_provenance_hash: derivation_hash.clone(),
        event_sequence: 0,
        previous_receipt_hash: None,
        action: NeurosemanticArtifactLifecycleAction::Erasure,
        state: NeurosemanticArtifactLifecycleState::Requested,
        effect_evidence_ref: None,
        effect_evidence_hash: None,
        verification_agent_ref: None,
        verification_evidence_ref: None,
        verification_evidence_hash: None,
        verification_target_effect_evidence_hash: None,
        verification_scope: None,
        verification_scope_ref: None,
        verification_scope_hash: None,
        execution_revision: execution_revision.clone(),
        observed_at_unix_s: 1_580,
        resulting_artifact_hash: None,
        resulting_derivation_provenance_ref: None,
        resulting_derivation_provenance_hash: None,
    };
    let lifecycle_accepted = NeurosemanticArtifactLifecycleReceipt {
        receipt_ref: "synthetic-lifecycle-receipt-2".into(),
        event_sequence: 1,
        previous_receipt_hash: Some(lifecycle_base.fingerprint()?),
        state: NeurosemanticArtifactLifecycleState::Accepted,
        observed_at_unix_s: 1_581,
        ..lifecycle_base.clone()
    };
    let lifecycle_processing = NeurosemanticArtifactLifecycleReceipt {
        receipt_ref: "synthetic-lifecycle-receipt-3".into(),
        event_sequence: 2,
        previous_receipt_hash: Some(lifecycle_accepted.fingerprint()?),
        state: NeurosemanticArtifactLifecycleState::Processing,
        observed_at_unix_s: 1_582,
        ..lifecycle_accepted.clone()
    };
    let lifecycle_applied = NeurosemanticArtifactLifecycleReceipt {
        receipt_ref: "synthetic-lifecycle-receipt-4".into(),
        event_sequence: 3,
        previous_receipt_hash: Some(lifecycle_processing.fingerprint()?),
        state: NeurosemanticArtifactLifecycleState::Applied,
        effect_evidence_ref: Some("synthetic-lifecycle-effect-1".into()),
        effect_evidence_hash: Some(symthaea_communication::content_hash(lifecycle_effect_evidence)),
        observed_at_unix_s: 1_583,
        ..lifecycle_processing.clone()
    };
    let lifecycle_receipt = NeurosemanticArtifactLifecycleReceipt {
        receipt_ref: "synthetic-lifecycle-receipt-5".into(),
        event_sequence: 4,
        previous_receipt_hash: Some(lifecycle_applied.fingerprint()?),
        state: NeurosemanticArtifactLifecycleState::IndependentlyVerified,
        verification_agent_ref: Some("synthetic-independent-verifier-1".into()),
        verification_evidence_ref: Some("synthetic-independent-verification-1".into()),
        verification_evidence_hash: Some(symthaea_communication::content_hash(lifecycle_verification_evidence)),
        verification_target_effect_evidence_hash: lifecycle_applied.effect_evidence_hash.clone(),
        verification_scope: Some(
            symthaea_communication::NeurosemanticArtifactLifecycleVerificationScope::EnumeratedTargetSet,
        ),
        verification_scope_ref: Some(lifecycle_verification_scope_ref.into()),
        verification_scope_hash: Some(lifecycle_scope_hash),
        observed_at_unix_s: 1_584,
        ..lifecycle_applied.clone()
    };
    let lifecycle_receipt_binding_verified = lifecycle_receipt
        .verify_binding(
            &message.packet.payload_hash,
            &derivation_ref,
            &derivation_hash,
            lifecycle_effect_evidence,
            1_700,
        )
        .is_ok();
    let lifecycle_independent_verification_verified = lifecycle_receipt
        .verify_independent_verification_bytes(lifecycle_verification_evidence, 1_700)
        .is_ok();
    let lifecycle_verification_scope_verified = lifecycle_receipt
        .verify_verification_scope_bytes(lifecycle_verification_scope)
        .is_ok();
    let lifecycle_chain_verified = lifecycle_accepted.verify_transition(&lifecycle_base).is_ok()
        && lifecycle_processing.verify_transition(&lifecycle_accepted).is_ok()
        && lifecycle_applied.verify_transition(&lifecycle_processing).is_ok()
        && lifecycle_receipt.verify_transition(&lifecycle_applied).is_ok();
    let lifecycle_sequence_gap_blocked = {
        let mut forged = lifecycle_receipt.clone();
        forged.event_sequence = 6;
        forged.verify_transition(&lifecycle_applied).is_err()
    };
    let lifecycle_state_regression_blocked = {
        let mut forged = lifecycle_receipt.clone();
        forged.state = NeurosemanticArtifactLifecycleState::Applied;
        forged.verify_transition(&lifecycle_applied).is_err()
    };
    let lifecycle_skipped_stage_blocked = {
        let mut forged = lifecycle_base.clone();
        forged.event_sequence = 1;
        forged.previous_receipt_hash = Some(lifecycle_base.fingerprint()?);
        forged.state = NeurosemanticArtifactLifecycleState::Applied;
        forged.effect_evidence_ref = Some("synthetic-lifecycle-effect-1".into());
        forged.effect_evidence_hash = Some(symthaea_communication::content_hash(lifecycle_effect_evidence));
        forged.verify_transition(&lifecycle_base).is_err()
    };
    let lifecycle_timestamp_regression_blocked = {
        let mut forged = lifecycle_receipt.clone();
        forged.observed_at_unix_s = 1_582;
        forged.verify_transition(&lifecycle_applied).is_err()
    };
    let lifecycle_effect_evidence_mismatch_blocked = !lifecycle_receipt
        .verify_effect_evidence_bytes(b"synthetic-lifecycle-effect-tampered");
    let lifecycle_independent_verification_mismatch_blocked = lifecycle_receipt
        .verify_independent_verification_bytes(b"synthetic-independent-verification-tampered", 1_700)
        .is_err();
    let lifecycle_verifier_target_mismatch_blocked = {
        let mut forged = lifecycle_receipt.clone();
        forged.verification_target_effect_evidence_hash = Some(symthaea_communication::content_hash(b"other-effect"));
        forged.validate().is_err()
    };
    let lifecycle_verifier_identity_required = {
        let mut forged = lifecycle_receipt.clone();
        forged.verification_agent_ref = None;
        forged.validate().is_err()
    };
    let lifecycle_verification_scope_mismatch_blocked = {
        let mut forged = lifecycle_receipt.clone();
        forged.verification_scope_hash = Some(symthaea_communication::content_hash(b"wrong-scope"));
        forged.validate().is_ok()
            && forged.verify_verification_scope_bytes(lifecycle_verification_scope).is_err()
    };
    let lifecycle_verification_scope_tamper_blocked = lifecycle_receipt
        .verify_verification_scope_bytes(b"synthetic-target-set-v1:TAMPERED")
        .is_err();
    let lifecycle_artifact_mismatch_blocked = lifecycle_receipt
        .verify_binding(
            &symthaea_communication::content_hash(b"wrong-artifact"),
            &derivation_ref,
            &derivation_hash,
            lifecycle_effect_evidence,
            1_700,
        )
        .is_err();
    let lifecycle_lineage_mismatch_blocked = lifecycle_receipt
        .verify_binding(
            &message.packet.payload_hash,
            "synthetic-derivation-record-2",
            &derivation_hash,
            lifecycle_effect_evidence,
            1_700,
        )
        .is_err();
    let lifecycle_future_timestamp_blocked = lifecycle_receipt
        .verify_binding(
            &message.packet.payload_hash,
            &derivation_ref,
            &derivation_hash,
            lifecycle_effect_evidence,
            1_583,
        )
        .is_err();
    let replacement_artifact_hash = symthaea_communication::content_hash(b"replacement-artifact");
    let replacement_lineage = NeurosemanticDerivationLineageRecord {
        lineage_ref: "synthetic-replacement-lineage".into(),
        output_artifact_hash: replacement_artifact_hash.clone(),
        ..derivation_lineage_record.clone()
    };
    let replacement_lineage_bytes =
        serde_json::to_vec(&replacement_lineage).map_err(|e| e.to_string())?;
    let mut rectification_receipt = lifecycle_receipt.clone();
    rectification_receipt.receipt_ref = "synthetic-lifecycle-receipt-rectification".into();
    rectification_receipt.action = NeurosemanticArtifactLifecycleAction::Rectification;
    rectification_receipt.resulting_artifact_hash = Some(replacement_artifact_hash);
    rectification_receipt.resulting_derivation_provenance_ref =
        Some(replacement_lineage.lineage_ref.clone());
    rectification_receipt.resulting_derivation_provenance_hash = Some(
        symthaea_communication::compute_derivation_provenance_hash(
            &replacement_lineage.lineage_ref,
            &replacement_lineage_bytes,
        ),
    );
    let lifecycle_replacement_lineage_verified = rectification_receipt
        .verify_resulting_lineage_binding_bytes(&replacement_lineage_bytes)
        .is_ok();
    let mut mismatched_replacement_lineage = replacement_lineage.clone();
    mismatched_replacement_lineage.output_artifact_hash =
        symthaea_communication::content_hash(b"other-artifact");
    let mismatched_replacement_lineage_bytes =
        serde_json::to_vec(&mismatched_replacement_lineage).map_err(|e| e.to_string())?;
    let lifecycle_replacement_lineage_mismatch_blocked = rectification_receipt
        .verify_resulting_lineage_binding_bytes(&mismatched_replacement_lineage_bytes)
        .is_err();
    let lifecycle_rectification_requires_replacement_artifact = {
        let mut invalid = lifecycle_receipt.clone();
        invalid.action = NeurosemanticArtifactLifecycleAction::Rectification;
        invalid.resulting_artifact_hash = None;
        invalid.resulting_derivation_provenance_ref = None;
        invalid.resulting_derivation_provenance_hash = None;
        invalid.validate().is_err()
    };
    let lifecycle_rectification_requires_replacement_lineage = {
        let mut invalid = rectification_receipt.clone();
        invalid.resulting_derivation_provenance_ref = None;
        invalid.resulting_derivation_provenance_hash = None;
        invalid.validate().is_err()
    };
    let derivation_provenance_present =
        !message.packet.data_policy.handling.derivation_provenance_ref.is_empty();
    let derivation_provenance_hash_valid = message
        .packet
        .data_policy
        .handling
        .verify_derivation_provenance_binding_bytes(&derivation_record_bytes);
    let mut mismatched_derivation_lineage_record = derivation_lineage_record.clone();
    mismatched_derivation_lineage_record.lineage_ref = "synthetic-derivation-record-2".into();
    let mismatched_derivation_record_bytes =
        serde_json::to_vec(&mismatched_derivation_lineage_record).map_err(|e| e.to_string())?;
    let derivation_provenance_mismatch_blocked = !message
        .packet
        .data_policy
        .handling
        .verify_derivation_provenance_binding_bytes(&mismatched_derivation_record_bytes);
    let derivation_lineage_structured = message
        .packet
        .data_policy
        .handling
        .verify_derivation_provenance_record_bytes(&derivation_record_bytes)
        .map(|record| record.output_artifact_hash == message.packet.payload_hash)
        .unwrap_or(false);
    let mut malformed_derivation_lineage = derivation_lineage_record.clone();
    malformed_derivation_lineage.execution_revision = "placeholder".into();
    let malformed_derivation_bytes =
        serde_json::to_vec(&malformed_derivation_lineage).map_err(|e| e.to_string())?;
    let malformed_derivation_lineage_blocked = message
        .packet
        .data_policy
        .handling
        .verify_derivation_provenance_record_bytes(&malformed_derivation_bytes)
        .is_err();

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
                &derivation_record_bytes,
                b"synthetic-status-record-1",
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
            &derivation_record_bytes,
            b"synthetic-status-record-1",
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
            &derivation_record_bytes,
            b"synthetic-status-record-1",
            &authority_attestation,
            &signing_key.verifying_key(),
            &wrong_context_resolution,
            &resolver_signing_key.verifying_key(),
            &resolution_context,
            1_500,
        )
        .is_err();

    let authority_status_source_mismatch_blocked = message
        .packet
        .data_policy
        .handling
        .bind_policy_provenance_with_attestation_and_resolution(
            b"synthetic-policy-record-1",
            &derivation_record_bytes,
            b"synthetic-status-record-2",
            &authority_attestation,
            &signing_key.verifying_key(),
            &authority_resolution,
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
            &derivation_record_bytes,
            b"synthetic-status-record-1",
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
            &derivation_record_bytes,
            b"synthetic-status-record-1",
            &authority_attestation,
            &signing_key.verifying_key(),
            &pre_attestation_resolution,
            &resolver_signing_key.verifying_key(),
            &resolution_context,
            1_500,
        )
        .is_err();

    let policy_freshness_expired_before_resolution_expiry = message
        .validate_for_handling(
            &lease,
            &policy_provenance_binding,
            "ZA",
            symthaea_communication::NeurosemanticHandlingAction::Transmit,
            1_501,
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
        "derivation_provenance_present": derivation_provenance_present,
        "derivation_provenance_hash_valid": derivation_provenance_hash_valid,
        "derivation_provenance_mismatch_blocked": derivation_provenance_mismatch_blocked,
        "derivation_lineage_structured": derivation_lineage_structured,
        "malformed_derivation_lineage_blocked": malformed_derivation_lineage_blocked,
        "derivation_input_artifact_verified": derivation_input_artifact_verified,
        "derivation_input_artifact_mismatch_blocked": derivation_input_artifact_mismatch_blocked,
        "derivation_input_artifact_bounds_blocked": derivation_input_artifact_bounds_blocked,
        "lifecycle_receipt_binding_verified": lifecycle_receipt_binding_verified,
        "lifecycle_independent_verification_verified": lifecycle_independent_verification_verified,
        "lifecycle_verification_scope_verified": lifecycle_verification_scope_verified,
        "lifecycle_chain_verified": lifecycle_chain_verified,
        "lifecycle_sequence_gap_blocked": lifecycle_sequence_gap_blocked,
        "lifecycle_skipped_stage_blocked": lifecycle_skipped_stage_blocked,
        "lifecycle_state_regression_blocked": lifecycle_state_regression_blocked,
        "lifecycle_timestamp_regression_blocked": lifecycle_timestamp_regression_blocked,
        "lifecycle_replacement_lineage_verified": lifecycle_replacement_lineage_verified,
        "lifecycle_replacement_lineage_mismatch_blocked": lifecycle_replacement_lineage_mismatch_blocked,
        "lifecycle_effect_evidence_mismatch_blocked": lifecycle_effect_evidence_mismatch_blocked,
        "lifecycle_independent_verification_mismatch_blocked": lifecycle_independent_verification_mismatch_blocked,
        "lifecycle_verifier_target_mismatch_blocked": lifecycle_verifier_target_mismatch_blocked,
        "lifecycle_verifier_identity_required": lifecycle_verifier_identity_required,
        "lifecycle_verification_scope_mismatch_blocked": lifecycle_verification_scope_mismatch_blocked,
        "lifecycle_verification_scope_tamper_blocked": lifecycle_verification_scope_tamper_blocked,
        "lifecycle_artifact_mismatch_blocked": lifecycle_artifact_mismatch_blocked,
        "lifecycle_lineage_mismatch_blocked": lifecycle_lineage_mismatch_blocked,
        "lifecycle_future_timestamp_blocked": lifecycle_future_timestamp_blocked,
        "lifecycle_rectification_requires_replacement_artifact": lifecycle_rectification_requires_replacement_artifact,
        "lifecycle_rectification_requires_replacement_lineage": lifecycle_rectification_requires_replacement_lineage,
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
        "policy_freshness_expired_before_resolution_expiry": policy_freshness_expired_before_resolution_expiry,
        "mutated_consent_lease_blocked": mutated_lease_blocked,
        "authority_proof_substitution_blocked": authority_proof_substitution_blocked,
        "authority_status_source_mismatch_blocked": authority_status_source_mismatch_blocked
    });

    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    Ok(())
}
