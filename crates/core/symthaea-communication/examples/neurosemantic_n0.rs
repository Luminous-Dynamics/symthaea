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
    NeurosemanticArtifactLifecycleState, NeurosemanticArtifactLifecycleVerificationTargetSet,
    NeurosemanticRemediationImpactArtifact, NeurosemanticRemediationImpactDisposition,
    NeurosemanticRemediationImpactEvidenceKind, NeurosemanticRemediationImpactLineageSide,
    NeurosemanticRemediationEvaluationSetKind, NeurosemanticRemediationEvaluationSetManifest,
    NeurosemanticRemediationEvaluationMethodKind, NeurosemanticRemediationEvaluationMethod,
    NeurosemanticRemediationEvaluationManifest,
    NeurosemanticRemediationEvaluationEnvironment,
    NeurosemanticRemediationMeasurement,
    NeurosemanticRemediationMeasurementArtifact,
    NeurosemanticRemediationMeasurementKind,
    NeurosemanticRemediationMetricDefinition,
    NeurosemanticRemediationMetricDirection,
    NeurosemanticRemediationUncertainty,
    NeurosemanticRemediationObservationRecord,
    NeurosemanticRemediationObservationSetArtifact,
    NeurosemanticRemediationMetricComputationArtifact,
    NeurosemanticRemediationUncertaintyComputationArtifact,
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
    let lifecycle_verification_scope_ref = "synthetic-verification-scope-1";
    let lifecycle_verification_target_set = NeurosemanticArtifactLifecycleVerificationTargetSet {
        schema_version: symthaea_communication::NEUROSEMANTIC_ARTIFACT_LIFECYCLE_VERIFICATION_SCOPE_SCHEMA_VERSION,
        scope_ref: lifecycle_verification_scope_ref.into(),
        root_artifact_hash: message.packet.payload_hash.clone(),
        target_artifact_hashes: vec![
            message.packet.payload_hash.clone(),
            symthaea_communication::content_hash(b"synthetic-descendant-artifact-1"),
        ],
    };
    let lifecycle_verification_scope =
        serde_json::to_vec(&lifecycle_verification_target_set).map_err(|e| e.to_string())?;
    let derivation_ref = policy_provenance_binding.derivation_provenance_ref().to_string();
    let derivation_hash = policy_provenance_binding.derivation_provenance_hash().to_string();
    let lifecycle_scope_hash = symthaea_communication::compute_lifecycle_verification_scope_hash(
        lifecycle_verification_scope_ref,
        &lifecycle_verification_scope,
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
        effect_agent_ref: Some("synthetic-effect-worker-1".into()),
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
        forged.effect_agent_ref = Some("synthetic-effect-worker-1".into());
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
    let lifecycle_verifier_independence_required = {
        let mut forged = lifecycle_receipt.clone();
        forged.verification_agent_ref = forged.effect_agent_ref.clone();
        forged.validate().is_err()
    };
    let lifecycle_effect_binding_mutation_blocked = {
        let mut forged = lifecycle_receipt.clone();
        forged.effect_evidence_hash = Some(symthaea_communication::content_hash(b"other-effect"));
        forged.verify_transition(&lifecycle_applied).is_err()
    };
    let lifecycle_verification_scope_mismatch_blocked = {
        let mut forged = lifecycle_receipt.clone();
        forged.verification_scope_hash = Some(symthaea_communication::content_hash(b"wrong-scope"));
        forged.validate().is_ok()
            && forged.verify_verification_scope_bytes(&lifecycle_verification_scope).is_err()
    };
    let lifecycle_verification_scope_root_mismatch_blocked = {
        let mut forged_set = lifecycle_verification_target_set.clone();
        forged_set.root_artifact_hash = symthaea_communication::content_hash(b"wrong-root");
        let forged_bytes = serde_json::to_vec(&forged_set).map_err(|e| e.to_string())?;
        lifecycle_receipt.verify_verification_scope_bytes(&forged_bytes).is_err()
    };
    let lifecycle_verification_scope_duplicate_blocked = {
        let mut forged_set = lifecycle_verification_target_set.clone();
        let duplicate = forged_set.target_artifact_hashes[0].clone();
        forged_set.target_artifact_hashes.push(duplicate);
        let forged_bytes = serde_json::to_vec(&forged_set).map_err(|e| e.to_string())?;
        lifecycle_receipt.verify_verification_scope_bytes(&forged_bytes).is_err()
    };
    let lifecycle_verification_scope_tamper_blocked = lifecycle_receipt
        .verify_verification_scope_bytes(b"{\\"schema_version\\":1,\\"scope_ref\\":\\"synthetic-verification-scope-1\\",\\"root_artifact_hash\\":\\"tampered\\",\\"target_artifact_hashes\\":[]}")
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
    let pre_model_bytes = b"synthetic-model-before-remediation-v1";
    let post_model_bytes = b"synthetic-model-after-remediation-v1";
    let pre_model_hash = symthaea_communication::content_hash(pre_model_bytes);
    let post_model_hash = symthaea_communication::content_hash(post_model_bytes);
    let pre_model_lineage = NeurosemanticDerivationLineageRecord {
        lineage_ref: "synthetic-model-lineage-pre".into(),
        output_artifact_hash: pre_model_hash.clone(),
        ..derivation_lineage_record.clone()
    };
    let post_model_lineage = NeurosemanticDerivationLineageRecord {
        lineage_ref: "synthetic-model-lineage-post".into(),
        output_artifact_hash: post_model_hash.clone(),
        ..replacement_lineage.clone()
    };
    let pre_model_lineage_bytes = serde_json::to_vec(&pre_model_lineage).map_err(|e| e.to_string())?;
    let post_model_lineage_bytes = serde_json::to_vec(&post_model_lineage).map_err(|e| e.to_string())?;
    let forget_evidence = b"synthetic-forgetfulness-evaluation-v1";
    let utility_evidence = b"synthetic-retained-utility-evaluation-v1";
    let fairness_evidence = b"synthetic-subgroup-impact-evaluation-v1";
    let residual_evidence = b"synthetic-residual-risk-evaluation-v1";
    let recovery_evidence = b"synthetic-recovery-attack-evaluation-v1";
    let representation_residual_evidence = b"synthetic-representation-residual-probe-v1";
    let study_protocol_bytes = b"synthetic-remediation-protocol-v1";
    let evaluation_split_manifest_bytes = b"synthetic-remediation-split-v1";
    let study_protocol_hash = symthaea_communication::content_hash(study_protocol_bytes);
    let evaluation_split_manifest_hash = symthaea_communication::content_hash(evaluation_split_manifest_bytes);
    let lifecycle_receipt_hash = lifecycle_receipt.fingerprint()?;
    let evaluation_environment = NeurosemanticRemediationEvaluationEnvironment {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_EVALUATION_ENVIRONMENT_SCHEMA_VERSION,
        environment_ref: "synthetic-evaluation-environment-v1".into(),
        platform_ref: "linux-x86_64".into(),
        runtime_ref: "rust-runtime".into(),
        toolchain_ref: "rust-1.96".into(),
        dependency_lock_hash: symthaea_communication::content_hash(b"synthetic-dependency-lock-v1"),
        configuration_hash: symthaea_communication::content_hash(b"synthetic-evaluation-config-v1"),
        execution_revision: execution_revision.clone(),
    };
    let evaluation_environment_bytes =
        serde_json::to_vec(&evaluation_environment).map_err(|e| e.to_string())?;
    let evaluation_environment_hash = evaluation_environment.fingerprint()?;
    let evaluation_verification_evidence =
        b"synthetic-independent-evaluation-verification-v1";
    let evaluation_verification_evidence_hash =
        symthaea_communication::content_hash(evaluation_verification_evidence);

    let source_dataset_manifest_hash = symthaea_communication::content_hash(b"synthetic-source-dataset-manifest-v1");
    let forget_set_manifest = NeurosemanticRemediationEvaluationSetManifest {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_EVALUATION_SET_SCHEMA_VERSION,
        set_ref: "synthetic-forget-set-v1".into(),
        set_kind: NeurosemanticRemediationEvaluationSetKind::Forget,
        source_dataset_manifest_hash: source_dataset_manifest_hash.clone(),
        member_artifact_hashes: vec![
            symthaea_communication::content_hash(b"synthetic-forget-member-1"),
            symthaea_communication::content_hash(b"synthetic-forget-member-2"),
        ],
    };
    let retain_set_manifest = NeurosemanticRemediationEvaluationSetManifest {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_EVALUATION_SET_SCHEMA_VERSION,
        set_ref: "synthetic-retain-set-v1".into(),
        set_kind: NeurosemanticRemediationEvaluationSetKind::Retain,
        source_dataset_manifest_hash,
        member_artifact_hashes: vec![
            symthaea_communication::content_hash(b"synthetic-retain-member-1"),
            symthaea_communication::content_hash(b"synthetic-retain-member-2"),
        ],
    };
    let forget_set_manifest_bytes = serde_json::to_vec(&forget_set_manifest).map_err(|e| e.to_string())?;
    let retain_set_manifest_bytes = serde_json::to_vec(&retain_set_manifest).map_err(|e| e.to_string())?;

    let recovery_method = NeurosemanticRemediationEvaluationMethod {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_EVALUATION_METHOD_SCHEMA_VERSION,
        method_ref: "synthetic-recovery-attack-v1".into(),
        kind: NeurosemanticRemediationEvaluationMethodKind::RecoveryAttack,
        protocol_hash: study_protocol_hash.clone(),
        implementation_revision: execution_revision.clone(),
    };
    let representation_probe_method = NeurosemanticRemediationEvaluationMethod {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_EVALUATION_METHOD_SCHEMA_VERSION,
        method_ref: "synthetic-representation-residual-probe-v1".into(),
        kind: NeurosemanticRemediationEvaluationMethodKind::RepresentationResidualProbe,
        protocol_hash: study_protocol_hash.clone(),
        implementation_revision: execution_revision.clone(),
    };
    let recovery_method_bytes = serde_json::to_vec(&recovery_method).map_err(|e| e.to_string())?;
    let representation_probe_method_bytes = serde_json::to_vec(&representation_probe_method).map_err(|e| e.to_string())?;
    let evaluation_manifest = NeurosemanticRemediationEvaluationManifest {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_EVALUATION_MANIFEST_SCHEMA_VERSION,
        evaluation_ref: "synthetic-remediation-evaluation-v1".into(),
        source_dataset_manifest_hash: forget_set_manifest.source_dataset_manifest_hash.clone(),
        forget_set_manifest_hash: forget_set_manifest.fingerprint()?,
        retain_set_manifest_hash: retain_set_manifest.fingerprint()?,
        study_protocol_hash: study_protocol_hash.clone(),
        evaluation_split_manifest_hash: evaluation_split_manifest_hash.clone(),
        recovery_method_hash: recovery_method.fingerprint()?,
        representation_probe_method_hash: representation_probe_method.fingerprint()?,
    };
    let evaluation_manifest_bytes =
        serde_json::to_vec(&evaluation_manifest).map_err(|e| e.to_string())?;

    let metric_forgetfulness = NeurosemanticRemediationMetricDefinition {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
        metric_ref: "metric-forgetfulness".into(),
        kind: NeurosemanticRemediationMeasurementKind::Forgetfulness,
        estimand_ref: "forgetfulness-on-forget-set".into(),
        scope_ref: "forget-set-v1".into(),
        unit_ref: "proportion".into(),
        aggregation_ref: "per-item-rate".into(),
        direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
    };
    let metric_utility = NeurosemanticRemediationMetricDefinition {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
        metric_ref: "metric-utility".into(),
        kind: NeurosemanticRemediationMeasurementKind::UtilityImpact,
        estimand_ref: "utility-on-retain-set".into(),
        scope_ref: "retain-set-v1".into(),
        unit_ref: "proportion".into(),
        aggregation_ref: "per-item-rate".into(),
        direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
    };
    let metric_fairness = NeurosemanticRemediationMetricDefinition {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
        metric_ref: "metric-fairness".into(),
        kind: NeurosemanticRemediationMeasurementKind::FairnessImpact,
        estimand_ref: "fairness-impact-on-declared-subgroups".into(),
        scope_ref: "fairness-split-v1".into(),
        unit_ref: "proportion".into(),
        aggregation_ref: "worst-subgroup-gap".into(),
        direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
    };
    let metric_recovery = NeurosemanticRemediationMetricDefinition {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
        metric_ref: "metric-recovery".into(),
        kind: NeurosemanticRemediationMeasurementKind::RecoveryRisk,
        estimand_ref: "recovery-risk-on-forget-set".into(),
        scope_ref: "forget-set-v1".into(),
        unit_ref: "proportion".into(),
        aggregation_ref: "attack-success-rate".into(),
        direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
    };
    let metric_representation = NeurosemanticRemediationMetricDefinition {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_METRIC_DEFINITION_SCHEMA_VERSION,
        metric_ref: "metric-representation".into(),
        kind: NeurosemanticRemediationMeasurementKind::RepresentationResidual,
        estimand_ref: "representation-residual-on-forget-set".into(),
        scope_ref: "forget-set-v1".into(),
        unit_ref: "proportion".into(),
        aggregation_ref: "probe-detection-rate".into(),
        direction: NeurosemanticRemediationMetricDirection::LowerIsBetter,
    };

    let forget_observation_set = NeurosemanticRemediationObservationSetArtifact {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_OBSERVATION_SET_SCHEMA_VERSION,
        observation_set_ref: "synthetic-forget-observations-v1".into(),
        metric_ref: metric_forgetfulness.metric_ref.clone(),
        kind: metric_forgetfulness.kind,
        scope_ref: metric_forgetfulness.scope_ref.clone(),
        population_manifest_hash: forget_set_manifest.fingerprint()?,
        eligible_subject_artifact_hashes: vec![
            symthaea_communication::content_hash(b"forget-1"),
            symthaea_communication::content_hash(b"forget-2"),
        ],
        observations: vec![
            NeurosemanticRemediationObservationRecord {
                observation_ref: "synthetic-forget-observation-1".into(),
                subject_artifact_hash: symthaea_communication::content_hash(b"forget-1"),
                failure_observed: false,
                group_ref: None,
            },
            NeurosemanticRemediationObservationRecord {
                observation_ref: "synthetic-forget-observation-2".into(),
                subject_artifact_hash: symthaea_communication::content_hash(b"forget-2"),
                failure_observed: false,
                group_ref: None,
            },
        ],
    };
    let retain_observation_set = NeurosemanticRemediationObservationSetArtifact {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_OBSERVATION_SET_SCHEMA_VERSION,
        observation_set_ref: "synthetic-retain-observations-v1".into(),
        metric_ref: metric_utility.metric_ref.clone(),
        kind: metric_utility.kind,
        scope_ref: metric_utility.scope_ref.clone(),
        population_manifest_hash: retain_set_manifest.fingerprint()?,
        eligible_subject_artifact_hashes: vec![
            symthaea_communication::content_hash(b"retain-1"),
            symthaea_communication::content_hash(b"retain-2"),
        ],
        observations: vec![
            NeurosemanticRemediationObservationRecord {
                observation_ref: "synthetic-retain-observation-1".into(),
                subject_artifact_hash: symthaea_communication::content_hash(b"retain-1"),
                failure_observed: false,
                group_ref: None,
            },
            NeurosemanticRemediationObservationRecord {
                observation_ref: "synthetic-retain-observation-2".into(),
                subject_artifact_hash: symthaea_communication::content_hash(b"retain-2"),
                failure_observed: false,
                group_ref: None,
            },
        ],
    };
    let fairness_observation_set = NeurosemanticRemediationObservationSetArtifact {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_OBSERVATION_SET_SCHEMA_VERSION,
        observation_set_ref: "synthetic-fairness-observations-v1".into(),
        metric_ref: metric_fairness.metric_ref.clone(),
        kind: metric_fairness.kind,
        scope_ref: metric_fairness.scope_ref.clone(),
        population_manifest_hash: evaluation_split_manifest_hash.clone(),
        eligible_subject_artifact_hashes: vec![
            symthaea_communication::content_hash(b"fairness-1"),
            symthaea_communication::content_hash(b"fairness-2"),
        ],
        observations: vec![
            NeurosemanticRemediationObservationRecord {
                observation_ref: "synthetic-fairness-observation-a".into(),
                subject_artifact_hash: symthaea_communication::content_hash(b"fairness-1"),
                failure_observed: false,
                group_ref: Some("group-a".into()),
            },
            NeurosemanticRemediationObservationRecord {
                observation_ref: "synthetic-fairness-observation-b".into(),
                subject_artifact_hash: symthaea_communication::content_hash(b"fairness-2"),
                failure_observed: false,
                group_ref: Some("group-b".into()),
            },
        ],
    };
    let recovery_observation_set = NeurosemanticRemediationObservationSetArtifact {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_OBSERVATION_SET_SCHEMA_VERSION,
        observation_set_ref: "synthetic-recovery-observations-v1".into(),
        metric_ref: metric_recovery.metric_ref.clone(),
        kind: metric_recovery.kind,
        scope_ref: metric_recovery.scope_ref.clone(),
        population_manifest_hash: forget_set_manifest.fingerprint()?,
        eligible_subject_artifact_hashes: vec![
            symthaea_communication::content_hash(b"forget-1"),
            symthaea_communication::content_hash(b"forget-2"),
        ],
        observations: vec![
            NeurosemanticRemediationObservationRecord {
                observation_ref: "synthetic-recovery-observation-1".into(),
                subject_artifact_hash: symthaea_communication::content_hash(b"forget-1"),
                failure_observed: false,
                group_ref: None,
            },
            NeurosemanticRemediationObservationRecord {
                observation_ref: "synthetic-recovery-observation-2".into(),
                subject_artifact_hash: symthaea_communication::content_hash(b"forget-2"),
                failure_observed: false,
                group_ref: None,
            },
        ],
    };
    let representation_observation_set = NeurosemanticRemediationObservationSetArtifact {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_OBSERVATION_SET_SCHEMA_VERSION,
        observation_set_ref: "synthetic-representation-observations-v1".into(),
        metric_ref: metric_representation.metric_ref.clone(),
        kind: metric_representation.kind,
        scope_ref: metric_representation.scope_ref.clone(),
        population_manifest_hash: forget_set_manifest.fingerprint()?,
        eligible_subject_artifact_hashes: vec![
            symthaea_communication::content_hash(b"forget-1"),
            symthaea_communication::content_hash(b"forget-2"),
        ],
        observations: vec![
            NeurosemanticRemediationObservationRecord {
                observation_ref: "synthetic-representation-observation-1".into(),
                subject_artifact_hash: symthaea_communication::content_hash(b"forget-1"),
                failure_observed: false,
                group_ref: None,
            },
            NeurosemanticRemediationObservationRecord {
                observation_ref: "synthetic-representation-observation-2".into(),
                subject_artifact_hash: symthaea_communication::content_hash(b"forget-2"),
                failure_observed: false,
                group_ref: None,
            },
        ],
    };

    let observation_sets = vec![
        forget_observation_set.clone(),
        retain_observation_set.clone(),
        fairness_observation_set.clone(),
        recovery_observation_set.clone(),
        representation_observation_set.clone(),
    ];
    let computation_for = |definition: &NeurosemanticRemediationMetricDefinition,
                           observation_set: &NeurosemanticRemediationObservationSetArtifact,
                           computation_ref: &str|
        NeurosemanticRemediationMetricComputationArtifact {
            schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_METRIC_COMPUTATION_SCHEMA_VERSION,
            computation_ref: computation_ref.into(),
            metric_ref: definition.metric_ref.clone(),
            kind: definition.kind,
            metric_definition_hash: definition.fingerprint().unwrap(),
            observation_set_hash: observation_set
                .fingerprint(&definition.aggregation_ref)
                .unwrap(),
            aggregation_ref: definition.aggregation_ref.clone(),
            execution_revision: execution_revision.clone(),
            estimate_numerator: 0,
            estimate_scale: 4,
            eligible_sample_count: observation_set.eligible_subject_artifact_hashes.len() as u64,
            observed_sample_count: observation_set.observations.len() as u64,
            failure_count: 0,
        };
    let computation_artifacts = vec![
        computation_for(&metric_forgetfulness, &observation_sets[0], "synthetic-computation-forgetfulness-v1"),
        computation_for(&metric_utility, &observation_sets[1], "synthetic-computation-utility-v1"),
        computation_for(&metric_fairness, &observation_sets[2], "synthetic-computation-fairness-v1"),
        computation_for(&metric_recovery, &observation_sets[3], "synthetic-computation-recovery-v1"),
        computation_for(&metric_representation, &observation_sets[4], "synthetic-computation-representation-v1"),
    ];
    let computation_bytes: Vec<Vec<u8>> = computation_artifacts
        .iter()
        .map(|artifact| serde_json::to_vec(artifact).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let computation_byte_refs: Vec<&[u8]> =
        computation_bytes.iter().map(Vec::as_slice).collect();
    let uncertainty_computation = NeurosemanticRemediationUncertaintyComputationArtifact {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_UNCERTAINTY_COMPUTATION_SCHEMA_VERSION,
        uncertainty_ref: "synthetic-uncertainty-computation-v1".into(),
        metric_ref: metric_forgetfulness.metric_ref.clone(),
        metric_definition_hash: metric_forgetfulness.fingerprint()?,
        observation_set_hash: observation_sets[0]
            .fingerprint(&metric_forgetfulness.aggregation_ref)
            .unwrap(),
        point_estimate_numerator: 0,
        point_estimate_scale: 4,
        lower_numerator: 0,
        upper_numerator: 6_577,
        scale: 4,
        confidence_level_bps: 9_500,
        method_ref: "wilson-score-95-v1".into(),
        assumptions_hash: symthaea_communication::content_hash(
            b"independent Bernoulli trials; fixed binary outcome; no clustering correction declared",
        ),
        assumptions_ref: "independent-bernoulli-trials-v1".into(),
        execution_revision: execution_revision.clone(),
    };
    let uncertainty_computation_bytes =
        serde_json::to_vec(&uncertainty_computation).map_err(|e| e.to_string())?;
    let uncertainty_computation_byte_refs: Vec<&[u8]> =
        vec![uncertainty_computation_bytes.as_slice()];
    let uncertainty_assumptions_bytes =
        b"independent Bernoulli trials; fixed binary outcome; no clustering correction declared";
    let uncertainty_assumption_byte_refs: Vec<&[u8]> =
        vec![uncertainty_assumptions_bytes.as_slice()];

    let observation_set_bytes: Vec<Vec<u8>> = observation_sets
        .iter()
        .map(|artifact| {
            serde_json::to_vec(artifact).map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    let observation_set_byte_refs: Vec<&[u8]> =
        observation_set_bytes.iter().map(Vec::as_slice).collect();
    let population_manifest_byte_refs: Vec<&[u8]> = vec![
        forget_set_manifest_bytes.as_slice(),
        retain_set_manifest_bytes.as_slice(),
        evaluation_split_manifest_bytes.as_slice(),
    ];

    let measurement = NeurosemanticRemediationMeasurementArtifact {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_MEASUREMENT_SCHEMA_VERSION,
        measurement_ref: "synthetic-remediation-measurement-v3".into(),
        metric_definitions: vec![
            metric_forgetfulness.clone(),
            metric_utility.clone(),
            metric_fairness.clone(),
            metric_recovery.clone(),
            metric_representation.clone(),
        ],
        metric_computation_artifact_hashes: computation_artifacts
            .iter()
            .map(|artifact| artifact.fingerprint().unwrap())
            .collect(),
        measurements: vec![
            NeurosemanticRemediationMeasurement {
                metric_ref: metric_forgetfulness.metric_ref,
                kind: metric_forgetfulness.kind,
                status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                estimate_numerator: 0,
                estimate_scale: 4,
                uncertainty: NeurosemanticRemediationUncertainty::Interval {
                    lower_numerator: 0,
                    upper_numerator: 6_577,
                    scale: 4,
                    confidence_level_bps: 9_500,
                    uncertainty_method_ref: "wilson-score-95-v1".into(),
                    uncertainty_computation_artifact_hash: uncertainty_computation.fingerprint()?,
                },
                eligible_sample_count: 2,
                observed_sample_count: 2,
                failure_count: 0,
            },
            NeurosemanticRemediationMeasurement {
                metric_ref: metric_utility.metric_ref,
                kind: metric_utility.kind,
                status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                estimate_numerator: 0,
                estimate_scale: 4,
                uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                eligible_sample_count: 2,
                observed_sample_count: 2,
                failure_count: 0,
            },
            NeurosemanticRemediationMeasurement {
                metric_ref: metric_fairness.metric_ref,
                kind: metric_fairness.kind,
                status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                estimate_numerator: 0,
                estimate_scale: 4,
                uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                eligible_sample_count: 2,
                observed_sample_count: 2,
                failure_count: 0,
            },
            NeurosemanticRemediationMeasurement {
                metric_ref: metric_recovery.metric_ref,
                kind: metric_recovery.kind,
                status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                estimate_numerator: 0,
                estimate_scale: 4,
                uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                eligible_sample_count: 2,
                observed_sample_count: 2,
                failure_count: 0,
            },
            NeurosemanticRemediationMeasurement {
                metric_ref: metric_representation.metric_ref,
                kind: metric_representation.kind,
                status: NeurosemanticRemediationImpactDisposition::Inconclusive,
                estimate_numerator: 0,
                estimate_scale: 4,
                uncertainty: NeurosemanticRemediationUncertainty::NotEstimated,
                eligible_sample_count: 2,
                observed_sample_count: 2,
                failure_count: 0,
            },
        ],
        worst_case_disposition: NeurosemanticRemediationImpactDisposition::Inconclusive,
    };
    let measurement_bytes = serde_json::to_vec(&measurement).map_err(|e| e.to_string())?;

    let remediation_impact = NeurosemanticRemediationImpactArtifact {
        schema_version: symthaea_communication::NEUROSEMANTIC_REMEDIATION_IMPACT_ARTIFACT_SCHEMA_VERSION,
        impact_ref: "synthetic-remediation-impact-1".into(),
        pre_remediation_model_hash: pre_model_hash.clone(),
        post_remediation_model_hash: post_model_hash.clone(),
        pre_remediation_lineage_ref: pre_model_lineage.lineage_ref.clone(),
        pre_remediation_lineage_hash: symthaea_communication::compute_derivation_provenance_hash(
            &pre_model_lineage.lineage_ref,
            &pre_model_lineage_bytes,
        ),
        post_remediation_lineage_ref: post_model_lineage.lineage_ref.clone(),
        post_remediation_lineage_hash: symthaea_communication::compute_derivation_provenance_hash(
            &post_model_lineage.lineage_ref,
            &post_model_lineage_bytes,
        ),
        lifecycle_receipt_hash,
        evaluation_manifest_hash: evaluation_manifest.fingerprint()?,
        measurement_artifact_hash: measurement.fingerprint()?,
        evaluation_agent_ref: "synthetic-evaluation-agent-1".into(),
        evaluation_verifier_ref: "synthetic-evaluation-verifier-1".into(),
        evaluation_verification_evidence_hash,
        evaluation_environment_hash,
        remediation_action: NeurosemanticArtifactLifecycleAction::Erasure,
        study_protocol_hash,
        evaluation_split_manifest_hash,
        forget_set_manifest_hash: forget_set_manifest.fingerprint()?,
        retain_set_manifest_hash: retain_set_manifest.fingerprint()?,
        recovery_method_hash: recovery_method.fingerprint()?,
        representation_probe_method_hash: representation_probe_method.fingerprint()?,
        forget_evidence_hash: symthaea_communication::content_hash(forget_evidence),
        utility_impact_evidence_hash: symthaea_communication::content_hash(utility_evidence),
        recovery_evidence_hash: symthaea_communication::content_hash(recovery_evidence),
        representation_residual_evidence_hash: symthaea_communication::content_hash(representation_residual_evidence),
        fairness_impact_evidence_hash: Some(symthaea_communication::content_hash(fairness_evidence)),
        residual_risk_evidence_hash: symthaea_communication::content_hash(residual_evidence),
        execution_revision: execution_revision.clone(),
        observed_at_unix_s: 1_590,
        disposition: NeurosemanticRemediationImpactDisposition::Inconclusive,
        dimensions: vec![
            "forgetfulness".into(),
            "utility-impact".into(),
            "fairness-impact".into(),
            "residual-risk".into(),
            "forget-set".into(),
            "retain-set".into(),
            "recovery-attack".into(),
            "representation-residual".into(),
        ],
    };
    let remediation_impact_bytes = serde_json::to_vec(&remediation_impact).map_err(|e| e.to_string())?;
    let remediation_impact_structured = NeurosemanticRemediationImpactArtifact::from_json_bytes(&remediation_impact_bytes).is_ok();
    let remediation_lifecycle_binding_verified = remediation_impact.verify_lifecycle_binding(&lifecycle_receipt).is_ok();
    let remediation_evaluation_verification_evidence_verified =
        remediation_impact
            .verify_evaluation_verification_evidence_bytes(
                "synthetic-evaluation-verifier-1",
                evaluation_verification_evidence,
            )
            .is_ok();
    let remediation_evaluation_environment_verified =
        remediation_impact
            .verify_evaluation_environment_bytes(evaluation_environment_bytes)
            .is_ok();
    let remediation_evaluation_manifest_verified =
        remediation_impact.verify_evaluation_manifest_bytes(&evaluation_manifest_bytes).is_ok();
    let remediation_measurement_verified =
        remediation_impact.verify_measurement_artifact_bytes(&measurement_bytes).is_ok();
    let remediation_measurement_computation_verified =
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &computation_byte_refs,
                &observation_set_byte_refs,
                &population_manifest_byte_refs,
                &uncertainty_computation_byte_refs,
                &uncertainty_assumption_byte_refs,
            )
            .is_ok();
    let remediation_wilson_uncertainty_numerically_recomputed =
        remediation_measurement_computation_verified;
    let remediation_metric_estimate_forgery_blocked = {
        let mut forged = computation_artifacts[0].clone();
        forged.estimate_numerator = 1;
        let mut forged_bytes = computation_bytes.clone();
        forged_bytes[0] = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        let forged_refs: Vec<&[u8]> = forged_bytes.iter().map(Vec::as_slice).collect();
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &forged_refs,
                &observation_set_byte_refs,
                &population_manifest_byte_refs,
                &uncertainty_computation_byte_refs,
                &uncertainty_assumption_byte_refs,
            )
            .is_err()
    };
    let remediation_observation_substitution_blocked = {
        let mut forged = observation_sets[0].clone();
        forged.observations[0].failure_observed = true;
        let forged_bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        let mut supplied = observation_set_bytes.clone();
        supplied[0] = forged_bytes;
        let supplied_refs: Vec<&[u8]> = supplied.iter().map(Vec::as_slice).collect();
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &computation_byte_refs,
                &supplied_refs,
                &population_manifest_byte_refs,
                &uncertainty_computation_byte_refs,
                &uncertainty_assumption_byte_refs,
            )
            .is_err()
    };
    let remediation_observation_omission_blocked = {
        let mut forged = observation_sets[0].clone();
        forged.observations.pop();
        let forged_bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        let mut supplied = observation_set_bytes.clone();
        supplied[0] = forged_bytes;
        let supplied_refs: Vec<&[u8]> = supplied.iter().map(Vec::as_slice).collect();
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &computation_byte_refs,
                &supplied_refs,
                &population_manifest_byte_refs,
                &uncertainty_computation_byte_refs,
                &uncertainty_assumption_byte_refs,
            )
            .is_err()
    };
    let remediation_observation_duplicate_blocked = {
        let mut forged = observation_sets[0].clone();
        forged.observations.push(forged.observations[0].clone());
        let forged_bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        let mut supplied = observation_set_bytes.clone();
        supplied[0] = forged_bytes;
        let supplied_refs: Vec<&[u8]> = supplied.iter().map(Vec::as_slice).collect();
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &computation_byte_refs,
                &supplied_refs,
                &population_manifest_byte_refs,
                &uncertainty_computation_byte_refs,
                &uncertainty_assumption_byte_refs,
            )
            .is_err()
    };
    let remediation_observation_scope_substitution_blocked = {
        let mut forged = observation_sets[0].clone();
        forged.scope_ref = "other-scope-v1".into();
        let forged_bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        let mut supplied = observation_set_bytes.clone();
        supplied[0] = forged_bytes;
        let supplied_refs: Vec<&[u8]> = supplied.iter().map(Vec::as_slice).collect();
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &computation_byte_refs,
                &supplied_refs,
                &population_manifest_byte_refs,
                &uncertainty_computation_byte_refs,
                &uncertainty_assumption_byte_refs,
            )
            .is_err()
    };
    let remediation_observation_population_substitution_blocked = {
        let mut forged = observation_sets[0].clone();
        forged.population_manifest_hash =
            symthaea_communication::content_hash(b"other-population-manifest");
        let forged_bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        let mut supplied = observation_set_bytes.clone();
        supplied[0] = forged_bytes;
        let supplied_refs: Vec<&[u8]> = supplied.iter().map(Vec::as_slice).collect();
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &computation_byte_refs,
                &supplied_refs,
                &population_manifest_byte_refs,
                &uncertainty_computation_byte_refs,
                &uncertainty_assumption_byte_refs,
            )
            .is_err()
    };
    let remediation_observation_membership_cherry_pick_blocked = {
        let mut forged = observation_sets[0].clone();
        forged.eligible_subject_artifact_hashes.pop();
        let forged_bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        let mut supplied = observation_set_bytes.clone();
        supplied[0] = forged_bytes;
        let supplied_refs: Vec<&[u8]> = supplied.iter().map(Vec::as_slice).collect();
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &computation_byte_refs,
                &supplied_refs,
                &population_manifest_byte_refs,
                &uncertainty_computation_byte_refs,
                &uncertainty_assumption_byte_refs,
            )
            .is_err()
    };
    let remediation_metric_direction_mismatch_blocked = {
        let mut forged = measurement.clone();
        forged.metric_definitions[0].direction =
            NeurosemanticRemediationMetricDirection::HigherIsBetter;
        forged.validate().is_err()
    };
    let remediation_uncertainty_substitution_blocked = {
        let mut forged = uncertainty_computation.clone();
        forged.lower_numerator = 0;
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &computation_byte_refs,
                &observation_set_byte_refs,
                &population_manifest_byte_refs,
                &[bytes.as_slice()],
            )
            .is_err()
    };
    let remediation_uncertainty_point_estimate_binding_blocked = {
        let mut forged = uncertainty_computation.clone();
        forged.point_estimate_numerator = 1;
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &computation_byte_refs,
                &observation_set_byte_refs,
                &population_manifest_byte_refs,
                &[bytes.as_slice()],
            )
            .is_err()
    };
    let remediation_uncertainty_assumptions_substitution_blocked = {
        let mut forged = uncertainty_assumptions_bytes.to_vec();
        forged.extend_from_slice(b"-tampered");
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &computation_byte_refs,
                &observation_set_byte_refs,
                &population_manifest_byte_refs,
                &uncertainty_computation_byte_refs,
                &[forged.as_slice()],
            )
            .is_err()
    };
    let remediation_computation_substitution_blocked = {
        let mut forged = computation_bytes.clone();
        forged.swap(0, 1);
        let forged_refs: Vec<&[u8]> = forged.iter().map(Vec::as_slice).collect();
        remediation_impact
            .verify_measurement_computation_bundle_bytes(
                &measurement_bytes,
                &forged_refs,
                &observation_set_byte_refs,
                &population_manifest_byte_refs,
                &uncertainty_computation_byte_refs,
                &uncertainty_assumption_byte_refs,
            )
            .is_err()
    };
    let remediation_measurement_worst_case_binding_blocked = {
        let mut forged = remediation_impact.clone();
        forged.disposition = NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds;
        forged
            .verify_measurement_artifact_bytes(&measurement_bytes)
            .is_err()
    };
    let remediation_measurement_missingness_fail_closed = {
        let mut forged = measurement.clone();
        forged.measurements[0].observed_sample_count = 1;
        forged.measurements[0].status =
            NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds;
        forged.validate().is_err()
    };
    let remediation_metric_definition_substitution_blocked = {
        let mut forged = measurement.clone();
        forged.metric_definitions[0].unit_ref = "other-unit".into();
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_measurement_artifact_bytes(&bytes).is_err()
    };
    let remediation_metric_definition_kind_mismatch_blocked = {
        let mut forged = measurement.clone();
        forged.metric_definitions[0].kind = NeurosemanticRemediationMeasurementKind::UtilityImpact;
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_measurement_artifact_bytes(&bytes).is_err()
    };
    let remediation_metric_uncertainty_substitution_blocked = {
        let mut forged = measurement.clone();
        forged.measurements[0].uncertainty =
            NeurosemanticRemediationUncertainty::Interval {
                lower_numerator: 2,
                upper_numerator: 1,
                scale: 4,
                confidence_level_bps: 9_500,
                uncertainty_method_ref: "synthetic-structural-interval-v1".into(),
            };
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_measurement_artifact_bytes(&bytes).is_err()
    };
    let remediation_pre_lineage_verified = remediation_impact
        .verify_lineage_bytes(NeurosemanticRemediationImpactLineageSide::PreRemediation, &pre_model_lineage_bytes)
        .is_ok();
    let remediation_post_lineage_verified = remediation_impact
        .verify_lineage_bytes(NeurosemanticRemediationImpactLineageSide::PostRemediation, &post_model_lineage_bytes)
        .is_ok();
    let remediation_set_pair_verified = remediation_impact
        .verify_evaluation_set_pair_bytes(&forget_set_manifest_bytes, &retain_set_manifest_bytes)
        .is_ok();
    let remediation_evaluation_bundle_verified = remediation_impact
        .verify_evaluation_bundle_identity(
            &evaluation_manifest_bytes,
            &forget_set_manifest_bytes,
            &retain_set_manifest_bytes,
        )
        .is_ok();
    let remediation_recovery_method_verified = remediation_impact
        .verify_evaluation_method_bytes(NeurosemanticRemediationEvaluationMethodKind::RecoveryAttack, &recovery_method_bytes)
        .is_ok();
    let remediation_representation_method_verified = remediation_impact
        .verify_evaluation_method_bytes(NeurosemanticRemediationEvaluationMethodKind::RepresentationResidualProbe, &representation_probe_method_bytes)
        .is_ok();
    let remediation_forget_evidence_verified = remediation_impact
        .verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::Forgetfulness, forget_evidence).is_ok();
    let remediation_utility_evidence_verified = remediation_impact
        .verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::UtilityImpact, utility_evidence).is_ok();
    let remediation_fairness_evidence_verified = remediation_impact
        .verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::FairnessImpact, fairness_evidence).is_ok();
    let remediation_residual_evidence_verified = remediation_impact
        .verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::ResidualRisk, residual_evidence).is_ok();
    let remediation_recovery_evidence_verified = remediation_impact
        .verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::RecoveryRisk, recovery_evidence).is_ok();
    let remediation_representation_residual_evidence_verified = remediation_impact
        .verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::RepresentationResidual, representation_residual_evidence).is_ok();
    let remediation_evaluation_verifier_substitution_blocked = remediation_impact
        .verify_evaluation_verification_evidence_bytes(
            "synthetic-other-verifier",
            evaluation_verification_evidence,
        )
        .is_err();
    let remediation_evaluation_self_verification_blocked = remediation_impact
        .verify_evaluation_verification_evidence_bytes(
            "synthetic-evaluation-agent-1",
            evaluation_verification_evidence,
        )
        .is_err();
    let remediation_evaluation_verification_evidence_substitution_blocked = remediation_impact
        .verify_evaluation_verification_evidence_bytes(
            "synthetic-evaluation-verifier-1",
            b"tampered-evaluation-verification",
        )
        .is_err();
    let remediation_evaluation_environment_substitution_blocked = remediation_impact
        .verify_evaluation_environment_bytes(b"other-evaluation-environment")
        .is_err();
    let remediation_evaluation_environment_revision_mismatch_blocked = {
        let mut forged = evaluation_environment.clone();
        forged.execution_revision = "4".repeat(40);
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_evaluation_environment_bytes(&bytes).is_err()
    };
    let remediation_measurement_substitution_blocked = {
        let mut forged = measurement.clone();
        forged.measurements[0].status =
            NeurosemanticRemediationImpactDisposition::WithinDeclaredBounds;
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_measurement_artifact_bytes(&bytes).is_err()
    };
    let remediation_measurement_dimension_drop_blocked = {
        let mut forged = measurement.clone();
        forged.measurements.retain(|item| {
            item.kind != NeurosemanticRemediationMeasurementKind::RepresentationResidual
        });
        forged.worst_case_disposition = forged.recomputed_worst_case_disposition()?;
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_measurement_artifact_bytes(&bytes).is_err()
    };
    let remediation_evaluation_manifest_substitution_blocked = {
        let mut forged = evaluation_manifest.clone();
        forged.forget_set_manifest_hash = symthaea_communication::content_hash(b"other-forget-set");
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_evaluation_manifest_bytes(&bytes).is_err()
    };
    let remediation_evaluation_source_dataset_mismatch_blocked = {
        let mut forged_manifest = retain_set_manifest.clone();
        forged_manifest.source_dataset_manifest_hash = symthaea_communication::content_hash(b"other-dataset");
        let bytes = serde_json::to_vec(&forged_manifest).map_err(|e| e.to_string())?;
        remediation_impact.verify_evaluation_bundle_identity(
            &evaluation_manifest_bytes,
            &forget_set_manifest_bytes,
            &bytes,
        )
        .is_err()
    };
    let remediation_set_role_substitution_blocked = {
        let mut forged = forget_set_manifest.clone();
        forged.set_kind = NeurosemanticRemediationEvaluationSetKind::Retain;
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_evaluation_set_manifest_bytes(NeurosemanticRemediationEvaluationSetKind::Forget, &bytes).is_err()
    };
    let remediation_set_overlap_blocked = {
        let mut forged = retain_set_manifest.clone();
        forged.member_artifact_hashes = vec![forget_set_manifest.member_artifact_hashes[0].clone()];
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_evaluation_set_pair_bytes(&forget_set_manifest_bytes, &bytes).is_err()
    };
    let remediation_recovery_method_substitution_blocked = {
        let mut forged = recovery_method.clone();
        forged.method_ref = "synthetic-other-recovery-method".into();
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_evaluation_method_bytes(NeurosemanticRemediationEvaluationMethodKind::RecoveryAttack, &bytes).is_err()
    };
    let remediation_representation_method_substitution_blocked = {
        let mut forged = representation_probe_method.clone();
        forged.protocol_hash = symthaea_communication::content_hash(b"other-protocol");
        let bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_evaluation_method_bytes(NeurosemanticRemediationEvaluationMethodKind::RepresentationResidualProbe, &bytes).is_err()
    };
    let remediation_recovery_evidence_substitution_blocked = remediation_impact
        .verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::RecoveryRisk, b"tampered-recovery")
        .is_err();
    let remediation_representation_evidence_substitution_blocked = remediation_impact
        .verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::RepresentationResidual, b"tampered-representation")
        .is_err();
    let remediation_effect_evidence_substitution_blocked = {
        let mut forged = remediation_impact.clone();
        forged.lifecycle_receipt_hash = symthaea_communication::content_hash(b"other-receipt");
        forged.verify_lifecycle_binding(&lifecycle_receipt).is_err()
    };
    let remediation_lifecycle_execution_revision_mismatch_blocked = {
        let mut forged = lifecycle_receipt.clone();
        forged.execution_revision = "4".repeat(40);
        forged.previous_receipt_hash = None;
        forged.event_sequence = 0;
        remediation_impact.verify_lifecycle_binding(&forged).is_err()
    };
    let remediation_model_substitution_blocked = {
        let mut forged = remediation_impact.clone();
        forged.post_remediation_model_hash = symthaea_communication::content_hash(b"wrong-model");
        forged.validate().is_ok() && forged.verify_lineage_bytes(
            NeurosemanticRemediationImpactLineageSide::PostRemediation,
            &post_model_lineage_bytes,
        ).is_err()
    };
    let remediation_lineage_substitution_blocked = {
        let mut forged = remediation_impact.clone();
        forged.post_remediation_lineage_ref = "synthetic-other-lineage".into();
        forged.verify_lineage_bytes(
            NeurosemanticRemediationImpactLineageSide::PostRemediation,
            &post_model_lineage_bytes,
        ).is_err()
    };
    let remediation_lineage_execution_revision_mismatch_blocked = {
        let mut forged = post_model_lineage.clone();
        forged.execution_revision = "4".repeat(40);
        let forged_bytes = serde_json::to_vec(&forged).map_err(|e| e.to_string())?;
        remediation_impact.verify_lineage_bytes(
            NeurosemanticRemediationImpactLineageSide::PostRemediation,
            &forged_bytes,
        ).is_err()
    };
    let remediation_protocol_verified = remediation_impact.verify_study_protocol_bytes(study_protocol_bytes).is_ok();
    let remediation_split_manifest_verified = remediation_impact.verify_evaluation_split_manifest_bytes(evaluation_split_manifest_bytes).is_ok();
    let remediation_protocol_substitution_blocked = {
        let mut forged = remediation_impact.clone();
        forged.study_protocol_hash = symthaea_communication::content_hash(b"other-protocol");
        forged.verify_study_protocol_bytes(study_protocol_bytes).is_err()
    };
    let remediation_split_substitution_blocked = {
        let mut forged = remediation_impact.clone();
        forged.evaluation_split_manifest_hash = symthaea_communication::content_hash(b"other-split");
        forged.verify_evaluation_split_manifest_bytes(evaluation_split_manifest_bytes).is_err()
    };
    let remediation_evidence_substitution_blocked = remediation_impact
        .verify_evidence_bytes(NeurosemanticRemediationImpactEvidenceKind::ResidualRisk, b"tampered-residual-evidence")
        .is_err();
    let remediation_dimension_claim_blocked = {
        let mut forged = remediation_impact.clone();
        forged.fairness_impact_evidence_hash = None;
        forged.validate().is_err()
    };
    let remediation_schema_migration_blocked = {
        let mut forged = remediation_impact.clone();
        forged.schema_version = 0;
        NeurosemanticRemediationImpactArtifact::from_json_bytes(&serde_json::to_vec(&forged).map_err(|e| e.to_string())?).is_err()
    };
    let remediation_bound_claim_is_narrow =
        remediation_impact.disposition == NeurosemanticRemediationImpactDisposition::Inconclusive;
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
        "lifecycle_verifier_independence_required": lifecycle_verifier_independence_required,
        "lifecycle_effect_binding_mutation_blocked": lifecycle_effect_binding_mutation_blocked,
        "lifecycle_verification_scope_mismatch_blocked": lifecycle_verification_scope_mismatch_blocked,
        "lifecycle_verification_scope_root_mismatch_blocked": lifecycle_verification_scope_root_mismatch_blocked,
        "lifecycle_verification_scope_duplicate_blocked": lifecycle_verification_scope_duplicate_blocked,
        "lifecycle_verification_scope_tamper_blocked": lifecycle_verification_scope_tamper_blocked,
        "lifecycle_artifact_mismatch_blocked": lifecycle_artifact_mismatch_blocked,
        "lifecycle_lineage_mismatch_blocked": lifecycle_lineage_mismatch_blocked,
        "lifecycle_future_timestamp_blocked": lifecycle_future_timestamp_blocked,
        "lifecycle_rectification_requires_replacement_artifact": lifecycle_rectification_requires_replacement_artifact,
        "lifecycle_rectification_requires_replacement_lineage": lifecycle_rectification_requires_replacement_lineage,
        "remediation_impact_structured": remediation_impact_structured,
        "remediation_evaluation_bundle_verified": remediation_evaluation_bundle_verified,
        "remediation_evaluation_source_dataset_mismatch_blocked": remediation_evaluation_source_dataset_mismatch_blocked,
        "remediation_evaluation_verifier_substitution_blocked": remediation_evaluation_verifier_substitution_blocked,
        "remediation_evaluation_self_verification_blocked": remediation_evaluation_self_verification_blocked,
        "remediation_evaluation_verification_evidence_substitution_blocked": remediation_evaluation_verification_evidence_substitution_blocked,
        "remediation_evaluation_environment_substitution_blocked": remediation_evaluation_environment_substitution_blocked,
        "remediation_evaluation_environment_revision_mismatch_blocked": remediation_evaluation_environment_revision_mismatch_blocked,
        "remediation_measurement_substitution_blocked": remediation_measurement_substitution_blocked,
        "remediation_measurement_dimension_drop_blocked": remediation_measurement_dimension_drop_blocked,
        "remediation_evaluation_manifest_substitution_blocked": remediation_evaluation_manifest_substitution_blocked,
        "remediation_set_pair_verified": remediation_set_pair_verified,
        "remediation_recovery_method_verified": remediation_recovery_method_verified,
        "remediation_representation_method_verified": remediation_representation_method_verified,
        "remediation_forget_evidence_verified": remediation_forget_evidence_verified,
        "remediation_utility_evidence_verified": remediation_utility_evidence_verified,
        "remediation_fairness_evidence_verified": remediation_fairness_evidence_verified,
        "remediation_residual_evidence_verified": remediation_residual_evidence_verified,
        "remediation_recovery_evidence_verified": remediation_recovery_evidence_verified,
        "remediation_representation_residual_evidence_verified": remediation_representation_residual_evidence_verified,
        "remediation_set_role_substitution_blocked": remediation_set_role_substitution_blocked,
        "remediation_set_overlap_blocked": remediation_set_overlap_blocked,
        "remediation_recovery_method_substitution_blocked": remediation_recovery_method_substitution_blocked,
        "remediation_representation_method_substitution_blocked": remediation_representation_method_substitution_blocked,
        "remediation_recovery_evidence_substitution_blocked": remediation_recovery_evidence_substitution_blocked,
        "remediation_representation_evidence_substitution_blocked": remediation_representation_evidence_substitution_blocked,
        "remediation_lifecycle_binding_verified": remediation_lifecycle_binding_verified,
        "remediation_evaluation_verification_evidence_verified": remediation_evaluation_verification_evidence_verified,
        "remediation_evaluation_environment_verified": remediation_evaluation_environment_verified,
        "remediation_evaluation_manifest_verified": remediation_evaluation_manifest_verified,
        "remediation_measurement_verified": remediation_measurement_verified,
        "remediation_measurement_computation_verified": remediation_measurement_computation_verified,
        "remediation_metric_estimate_forgery_blocked": remediation_metric_estimate_forgery_blocked,
        "remediation_observation_substitution_blocked": remediation_observation_substitution_blocked,
        "remediation_observation_omission_blocked": remediation_observation_omission_blocked,
        "remediation_observation_duplicate_blocked": remediation_observation_duplicate_blocked,
        "remediation_observation_scope_substitution_blocked": remediation_observation_scope_substitution_blocked,
        "remediation_computation_substitution_blocked": remediation_computation_substitution_blocked,
        "remediation_uncertainty_substitution_blocked": remediation_uncertainty_substitution_blocked,
        "remediation_metric_direction_mismatch_blocked": remediation_metric_direction_mismatch_blocked,
        "remediation_uncertainty_point_estimate_binding_blocked": remediation_uncertainty_point_estimate_binding_blocked,
        "remediation_uncertainty_assumptions_substitution_blocked": remediation_uncertainty_assumptions_substitution_blocked,
        "remediation_observation_population_substitution_blocked": remediation_observation_population_substitution_blocked,
        "remediation_observation_membership_cherry_pick_blocked": remediation_observation_membership_cherry_pick_blocked,
        "remediation_measurement_worst_case_binding_blocked": remediation_measurement_worst_case_binding_blocked,
        "remediation_measurement_missingness_fail_closed": remediation_measurement_missingness_fail_closed,
        "remediation_metric_definition_substitution_blocked": remediation_metric_definition_substitution_blocked,
        "remediation_metric_definition_kind_mismatch_blocked": remediation_metric_definition_kind_mismatch_blocked,
        "remediation_metric_uncertainty_substitution_blocked": remediation_metric_uncertainty_substitution_blocked,
        "remediation_pre_lineage_verified": remediation_pre_lineage_verified,
        "remediation_post_lineage_verified": remediation_post_lineage_verified,
        "remediation_forget_evidence_verified": remediation_forget_evidence_verified,
        "remediation_utility_evidence_verified": remediation_utility_evidence_verified,
        "remediation_fairness_evidence_verified": remediation_fairness_evidence_verified,
        "remediation_residual_evidence_verified": remediation_residual_evidence_verified,
        "remediation_effect_evidence_substitution_blocked": remediation_effect_evidence_substitution_blocked,
        "remediation_lifecycle_execution_revision_mismatch_blocked": remediation_lifecycle_execution_revision_mismatch_blocked,
        "remediation_model_substitution_blocked": remediation_model_substitution_blocked,
        "remediation_lineage_execution_revision_mismatch_blocked": remediation_lineage_execution_revision_mismatch_blocked,
        "remediation_lineage_substitution_blocked": remediation_lineage_substitution_blocked,
        "remediation_protocol_verified": remediation_protocol_verified,
        "remediation_split_manifest_verified": remediation_split_manifest_verified,
        "remediation_protocol_substitution_blocked": remediation_protocol_substitution_blocked,
        "remediation_split_substitution_blocked": remediation_split_substitution_blocked,
        "remediation_evidence_substitution_blocked": remediation_evidence_substitution_blocked,
        "remediation_dimension_claim_blocked": remediation_dimension_claim_blocked,
        "remediation_schema_migration_blocked": remediation_schema_migration_blocked,
        "remediation_bound_claim_is_narrow": remediation_bound_claim_is_narrow,
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
