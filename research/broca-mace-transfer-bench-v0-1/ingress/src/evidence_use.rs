// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Revalidate an admitted source immediately before a later decision uses it.
//! A JSON-exported receipt carries integrity data, not verifier custody.

use super::*;
use sha2::{Digest, Sha256};

const USE_RECEIPT_SCHEMA: &str = "mace_evidence_use_receipt_v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceUseContextV1 {
    /// Opaque non-personal slug; this is not an authorization token.
    pub purpose_code: String,
    pub required_scope: Scope,
    /// Digest of the frozen downstream policy that receives the evidence.
    pub consumer_policy_sha256: String,
    /// Unknown/partial ancestry remains visible unless a consumer explicitly vetoes it.
    pub require_complete_dependency_lineage: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceUseRejectV1 {
    InvalidPurposeCode,
    InvalidConsumerPolicyDigest,
    ScopeMismatch,
    KnownAtRegression,
    PriorReceiptMismatch,
    SourceIdentityChanged,
    SourceAssessmentChanged,
    TrustPolicyChanged,
    DependencyLineageChanged,
    SourceNotTrustedForScope,
    DependencyLineageIncomplete,
    RevalidationRejected(IngestReject),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceUseReceiptV1 {
    pub event_id: u32,
    pub payload_sha256: String,
    pub canonical_event_hash: String,
    pub prior_raw_event_receipt_digest: String,
    pub current_raw_event_receipt_digest: String,
    pub current_authority_receipt_digest: String,
    pub prior_lineage_receipt_digest: String,
    pub current_lineage_receipt_digest: String,
    pub prior_snapshot_sha256: String,
    pub current_snapshot_sha256: String,
    pub occurred_at_ms: i64,
    pub valid_from_ms: i64,
    pub valid_until_ms: Option<i64>,
    pub authority_received_at_ms: i64,
    pub prior_valid_at_ms: i64,
    pub prior_known_at_ms: i64,
    pub valid_at_ms: i64,
    pub known_at_ms: i64,
    pub action: Action,
    pub scope: Scope,
    pub source_family: u16,
    pub source_root_digest: SourceRootDigest,
    pub upstream_source_id: String,
    pub source_artifact_digest: String,
    pub transformation_digest: String,
    pub source_issuer_id: String,
    pub source_trust_status: SourceTrustStatus,
    pub source_trust_policy_sha256: String,
    pub lifecycle_status_as_known: SourceLifecycleStatus,
    pub dependency_lineage_status: DependencyLineageStatusV1,
    pub dependency_manifest_sha256: Option<String>,
    pub derived_from_event_ids: Vec<u32>,
    pub calibration_sha256: String,
    pub measurement_unit: String,
    pub measurement_error_bound_bits: u64,
    pub causal_assessment_sha256: String,
    pub semantic_estimand_sha256: String,
    pub outcome_sha256: String,
    pub purpose_code: String,
    pub consumer_policy_sha256: String,
    pub canonical_sha256: String,
    verified_in_process: bool,
}

fn put_bytes(h: &mut Sha256, bytes: &[u8]) {
    h.update((bytes.len() as u64).to_be_bytes());
    h.update(bytes);
}
fn put_text(h: &mut Sha256, value: &str) { put_bytes(h, value.as_bytes()); }
fn put_opt_text(h: &mut Sha256, value: Option<&str>) {
    match value { Some(v) => { h.update([1]); put_text(h, v); }, None => h.update([0]) }
}
fn put_opt_i64(h: &mut Sha256, value: Option<i64>) {
    match value { Some(v) => { h.update([1]); h.update(v.to_be_bytes()); }, None => h.update([0]) }
}
fn put_scope(h: &mut Sha256, scope: Scope) {
    h.update([
        match scope.regime { broca_mace_policy::Regime::Baseline => 1, broca_mace_policy::Regime::Shifted => 2 },
        match scope.objective { broca_mace_policy::Objective::Quality => 1, broca_mace_policy::Objective::Cost => 2, broca_mace_policy::Objective::Speed => 3 },
    ]);
}
fn put_lifecycle(h: &mut Sha256, v: SourceLifecycleStatus) {
    match v {
        SourceLifecycleStatus::Active => h.update([1]),
        SourceLifecycleStatus::Retracted => h.update([2]),
        SourceLifecycleStatus::Superseded { successor_event_id } => { h.update([3]); h.update(successor_event_id.to_be_bytes()); }
    }
}
fn put_dependency_status(h: &mut Sha256, v: DependencyLineageStatusV1) {
    h.update([match v { DependencyLineageStatusV1::Unknown => 0, DependencyLineageStatusV1::Complete => 1, DependencyLineageStatusV1::Partial => 2 }]);
}
fn lower_sha256(v: &str) -> bool {
    v.len() == 64 && v.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn purpose_code_valid(v: &str) -> bool {
    !v.is_empty() && v.len() <= 64 && v.as_bytes()[0].is_ascii_lowercase()
        && v.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-'))
}

fn snapshot_sha256(value: &IngestedExperience) -> String {
    let r = &value.receipts;
    let q = value.as_known;
    let mut h = Sha256::new();
    h.update(b"luminous:mace:ingress-snapshot:v1\0");
    h.update(value.experience.event_id.to_be_bytes());
    put_text(&mut h, &r.raw_event_digest);
    put_text(&mut h, &r.raw_event_receipt_digest);
    put_text(&mut h, &r.canonical_event_hash);
    put_text(&mut h, &r.authority_receipt_digest);
    h.update(r.event_valid_from_ms.to_be_bytes());
    put_opt_i64(&mut h, r.event_valid_until_ms);
    h.update(r.event_occurred_at_ms.to_be_bytes());
    h.update(r.authority_received_at_ms.to_be_bytes());
    h.update(q.valid_at_ms.to_be_bytes());
    h.update(q.known_at_ms.to_be_bytes());
    h.update(r.source_family.to_be_bytes());
    h.update(r.source_root_digest.as_bytes());
    put_text(&mut h, &r.upstream_source_id);
    put_text(&mut h, &r.source_artifact_digest);
    put_text(&mut h, &r.transformation_digest);
    put_text(&mut h, &r.lineage_digest);
    put_text(&mut h, &r.source_trust_policy_digest);
    put_dependency_status(&mut h, r.dependency_lineage_status);
    put_opt_text(&mut h, r.dependency_manifest_digest.as_deref());
    h.update((r.derived_from_event_ids.len() as u64).to_be_bytes());
    for id in &r.derived_from_event_ids { h.update(id.to_be_bytes()); }
    put_lifecycle(&mut h, r.lifecycle_status_as_known);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

impl EvidenceUseReceiptV1 {
    fn digest_without_commitment(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"luminous:mace:evidence-use-receipt:v1\0");
        h.update(self.event_id.to_be_bytes());
        put_text(&mut h, &self.payload_sha256);
        put_text(&mut h, &self.canonical_event_hash);
        put_text(&mut h, &self.prior_raw_event_receipt_digest);
        put_text(&mut h, &self.current_raw_event_receipt_digest);
        put_text(&mut h, &self.current_authority_receipt_digest);
        put_text(&mut h, &self.prior_lineage_receipt_digest);
        put_text(&mut h, &self.current_lineage_receipt_digest);
        put_text(&mut h, &self.prior_snapshot_sha256);
        put_text(&mut h, &self.current_snapshot_sha256);
        h.update(self.occurred_at_ms.to_be_bytes());
        h.update(self.valid_from_ms.to_be_bytes());
        put_opt_i64(&mut h, self.valid_until_ms);
        for t in [self.authority_received_at_ms, self.prior_valid_at_ms, self.prior_known_at_ms, self.valid_at_ms, self.known_at_ms] { h.update(t.to_be_bytes()); }
        h.update([match self.action { Action::A => 1, Action::B => 2 }]);
        put_scope(&mut h, self.scope);
        h.update(self.source_family.to_be_bytes());
        h.update(self.source_root_digest.as_bytes());
        for s in [&self.upstream_source_id, &self.source_artifact_digest, &self.transformation_digest, &self.source_issuer_id] { put_text(&mut h, s); }
        h.update([match self.source_trust_status { SourceTrustStatus::TrustedForScope => 1, SourceTrustStatus::Untrusted => 2, SourceTrustStatus::Unknown => 0 }]);
        put_text(&mut h, &self.source_trust_policy_sha256);
        put_lifecycle(&mut h, self.lifecycle_status_as_known);
        put_dependency_status(&mut h, self.dependency_lineage_status);
        put_opt_text(&mut h, self.dependency_manifest_sha256.as_deref());
        h.update((self.derived_from_event_ids.len() as u64).to_be_bytes());
        for id in &self.derived_from_event_ids { h.update(id.to_be_bytes()); }
        for s in [&self.calibration_sha256, &self.measurement_unit] { put_text(&mut h, s); }
        h.update(self.measurement_error_bound_bits.to_be_bytes());
        for s in [&self.causal_assessment_sha256, &self.semantic_estimand_sha256, &self.outcome_sha256, &self.purpose_code, &self.consumer_policy_sha256] { put_text(&mut h, s); }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn verify_integrity(&self) -> bool { self.canonical_sha256 == self.digest_without_commitment() }

    /// Only the original in-process result retains this verifier capability.
    pub fn has_verified_ingress_provenance(&self) -> bool { self.verified_in_process }

    /// JSON export is descriptive; its digest is not a signature or custody proof.
    pub fn to_json_value(&self) -> serde_json::Value {
        let lifecycle = match self.lifecycle_status_as_known {
            SourceLifecycleStatus::Active => serde_json::json!({"status":"active"}),
            SourceLifecycleStatus::Retracted => serde_json::json!({"status":"retracted"}),
            SourceLifecycleStatus::Superseded { successor_event_id } => serde_json::json!({"status":"superseded","successor_event_id":successor_event_id}),
        };
        let dependency_status = match self.dependency_lineage_status {
            DependencyLineageStatusV1::Complete => "complete",
            DependencyLineageStatusV1::Partial => "partial",
            DependencyLineageStatusV1::Unknown => "unknown",
        };
        serde_json::json!({
            "schema_version": USE_RECEIPT_SCHEMA,
            "event_id": self.event_id,
            "payload_sha256": self.payload_sha256,
            "canonical_event_hash": self.canonical_event_hash,
            "prior_raw_event_receipt_digest": self.prior_raw_event_receipt_digest,
            "current_raw_event_receipt_digest": self.current_raw_event_receipt_digest,
            "current_authority_receipt_digest": self.current_authority_receipt_digest,
            "prior_lineage_receipt_digest": self.prior_lineage_receipt_digest,
            "current_lineage_receipt_digest": self.current_lineage_receipt_digest,
            "prior_snapshot_sha256": self.prior_snapshot_sha256,
            "current_snapshot_sha256": self.current_snapshot_sha256,
            "event_time": {"occurred_at_ms":self.occurred_at_ms,"valid_from_ms":self.valid_from_ms,"valid_until_ms":self.valid_until_ms},
            "verification_time": {"authority_received_at_ms":self.authority_received_at_ms,"prior_valid_at_ms":self.prior_valid_at_ms,"prior_known_at_ms":self.prior_known_at_ms,"valid_at_ms":self.valid_at_ms,"known_at_ms":self.known_at_ms},
            "action": match self.action { Action::A => "A", Action::B => "B" },
            "scope": {
                "regime": match self.scope.regime { broca_mace_policy::Regime::Baseline => "baseline", broca_mace_policy::Regime::Shifted => "shifted" },
                "objective": match self.scope.objective { broca_mace_policy::Objective::Quality => "quality", broca_mace_policy::Objective::Cost => "cost", broca_mace_policy::Objective::Speed => "speed" }
            },
            "source": {
                "source_family":self.source_family,"source_root_digest":self.source_root_digest.to_hex(),
                "upstream_source_id":self.upstream_source_id,"source_artifact_digest":self.source_artifact_digest,
                "transformation_digest":self.transformation_digest,"issuer_id":self.source_issuer_id,
                "trust_status":match self.source_trust_status { SourceTrustStatus::TrustedForScope => "trusted_for_scope", SourceTrustStatus::Untrusted => "untrusted", SourceTrustStatus::Unknown => "unknown" },
                "trust_policy_sha256":self.source_trust_policy_sha256
            },
            "lifecycle":lifecycle,
            "dependency_lineage":{"status":dependency_status,"manifest_sha256":self.dependency_manifest_sha256,"derived_from_event_ids":self.derived_from_event_ids},
            "assessment":{"calibration_sha256":self.calibration_sha256,"measurement_unit":self.measurement_unit,
                "measurement_error_bound":f64::from_bits(self.measurement_error_bound_bits),
                "causal_assessment_sha256":self.causal_assessment_sha256,"semantic_estimand_sha256":self.semantic_estimand_sha256,"outcome_sha256":self.outcome_sha256},
            "use":{"purpose_code":self.purpose_code,"consumer_policy_sha256":self.consumer_policy_sha256},
            "canonical_sha256":self.canonical_sha256,
            "verification_custody":"originating_process_only"
        })
    }
}

/// Re-run the existing ingress verifier at the point of use. Changes to the
/// source identity, trust-policy revision, assessment or dependency declaration
/// force re-admission; a revoked or superseded event is rejected by ingress.
pub fn revalidate_for_use_v1(
    raw: &RawInteractionV1,
    prior: &IngestedExperience,
    current_query: AsKnownQuery,
    context: &EvidenceUseContextV1,
    verifier: &impl EvidenceVerifier,
) -> Result<EvidenceUseReceiptV1, EvidenceUseRejectV1> {
    if !purpose_code_valid(&context.purpose_code) { return Err(EvidenceUseRejectV1::InvalidPurposeCode); }
    if !lower_sha256(&context.consumer_policy_sha256) { return Err(EvidenceUseRejectV1::InvalidConsumerPolicyDigest); }
    if raw.scope != context.required_scope || prior.experience.scope != context.required_scope {
        return Err(EvidenceUseRejectV1::ScopeMismatch);
    }
    if current_query.known_at_ms < prior.as_known.known_at_ms {
        return Err(EvidenceUseRejectV1::KnownAtRegression);
    }
    if raw.event_id != prior.experience.event_id
        || raw.payload_sha256 != prior.receipts.raw_event_digest
        || raw.scope != prior.experience.scope
        || raw.action != prior.experience.action
        || raw.valid_from_ms != prior.receipts.event_valid_from_ms
        || raw.valid_until_ms != prior.receipts.event_valid_until_ms
        || raw.occurred_at_ms != prior.receipts.event_occurred_at_ms
    {
        return Err(EvidenceUseRejectV1::PriorReceiptMismatch);
    }

    let current = ingest_as_known(raw, current_query, verifier).map_err(EvidenceUseRejectV1::RevalidationRejected)?;
    let before = &prior.receipts;
    let after = &current.receipts;
    if after.raw_event_digest != before.raw_event_digest
        || after.canonical_event_hash != before.canonical_event_hash
        || after.event_valid_from_ms != before.event_valid_from_ms
        || after.event_valid_until_ms != before.event_valid_until_ms
        || after.event_occurred_at_ms != before.event_occurred_at_ms
        || after.source_family != before.source_family
        || after.source_root_digest != before.source_root_digest
        || after.upstream_source_id != before.upstream_source_id
        || after.source_artifact_digest != before.source_artifact_digest
        || after.transformation_digest != before.transformation_digest
        || after.source_acquisition_mode != before.source_acquisition_mode
        || after.source_issuer_id != before.source_issuer_id
    {
        return Err(EvidenceUseRejectV1::SourceIdentityChanged);
    }
    if after.source_trust_policy_digest != before.source_trust_policy_digest {
        return Err(EvidenceUseRejectV1::TrustPolicyChanged);
    }
    if after.calibration_digest != before.calibration_digest
        || after.calibration_revision != before.calibration_revision
        || after.measurement_unit != before.measurement_unit
        || after.measurement_instrument_id != before.measurement_instrument_id
        || after.calibration_method_id != before.calibration_method_id
        || current.experience.measurement_error_bound.to_bits() != prior.experience.measurement_error_bound.to_bits()
        || after.outcome_success != before.outcome_success
        || after.causal_assessment_digest != before.causal_assessment_digest
        || after.causal_identification != before.causal_identification
        || after.outcome_digest != before.outcome_digest
        || after.outcome_observed_utility_bits != before.outcome_observed_utility_bits
        || after.outcome_utility_unit != before.outcome_utility_unit
    {
        return Err(EvidenceUseRejectV1::SourceAssessmentChanged);
    }
    if after.dependency_lineage_status != before.dependency_lineage_status
        || after.dependency_manifest_digest != before.dependency_manifest_digest
        || after.derived_from_event_ids != before.derived_from_event_ids
    {
        return Err(EvidenceUseRejectV1::DependencyLineageChanged);
    }
    if after.source_trust_status != SourceTrustStatus::TrustedForScope {
        return Err(EvidenceUseRejectV1::SourceNotTrustedForScope);
    }
    if context.require_complete_dependency_lineage && after.dependency_lineage_status != DependencyLineageStatusV1::Complete {
        return Err(EvidenceUseRejectV1::DependencyLineageIncomplete);
    }
    if after.lifecycle_status_as_known != SourceLifecycleStatus::Active {
        return Err(EvidenceUseRejectV1::RevalidationRejected(IngestReject::RetractedAsKnown));
    }

    let mut receipt = EvidenceUseReceiptV1 {
        event_id: current.experience.event_id,
        payload_sha256: after.raw_event_digest.clone(),
        canonical_event_hash: after.canonical_event_hash.clone(),
        prior_raw_event_receipt_digest: before.raw_event_receipt_digest.clone(),
        current_raw_event_receipt_digest: after.raw_event_receipt_digest.clone(),
        current_authority_receipt_digest: after.authority_receipt_digest.clone(),
        prior_lineage_receipt_digest: before.lineage_digest.clone(),
        current_lineage_receipt_digest: after.lineage_digest.clone(),
        prior_snapshot_sha256: snapshot_sha256(prior),
        current_snapshot_sha256: snapshot_sha256(&current),
        occurred_at_ms: after.event_occurred_at_ms,
        valid_from_ms: after.event_valid_from_ms,
        valid_until_ms: after.event_valid_until_ms,
        authority_received_at_ms: after.authority_received_at_ms,
        prior_valid_at_ms: prior.as_known.valid_at_ms,
        prior_known_at_ms: prior.as_known.known_at_ms,
        valid_at_ms: current_query.valid_at_ms,
        known_at_ms: current_query.known_at_ms,
        action: current.experience.action,
        scope: current.experience.scope,
        source_family: after.source_family,
        source_root_digest: after.source_root_digest,
        upstream_source_id: after.upstream_source_id.clone(),
        source_artifact_digest: after.source_artifact_digest.clone(),
        transformation_digest: after.transformation_digest.clone(),
        source_issuer_id: after.source_issuer_id.clone(),
        source_trust_status: after.source_trust_status,
        source_trust_policy_sha256: after.source_trust_policy_digest.clone(),
        lifecycle_status_as_known: after.lifecycle_status_as_known,
        dependency_lineage_status: after.dependency_lineage_status,
        dependency_manifest_sha256: after.dependency_manifest_digest.clone(),
        derived_from_event_ids: after.derived_from_event_ids.clone(),
        calibration_sha256: after.calibration_digest.clone(),
        measurement_unit: after.measurement_unit.clone(),
        measurement_error_bound_bits: current.experience.measurement_error_bound.to_bits(),
        causal_assessment_sha256: after.causal_assessment_digest.clone(),
        semantic_estimand_sha256: after.causal_identification.estimand_digest.to_hex(),
        outcome_sha256: after.outcome_digest.clone(),
        purpose_code: context.purpose_code.clone(),
        consumer_policy_sha256: context.consumer_policy_sha256.clone(),
        canonical_sha256: String::new(),
        verified_in_process: true,
    };
    receipt.canonical_sha256 = receipt.digest_without_commitment();
    Ok(receipt)
}

/// Exact UTF-8 byte range of one source-derived representation in a frozen model context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextEvidenceSpanV1 {
    pub byte_start: usize,
    pub byte_end: usize,
    pub representation_sha256: String,
}

/// Context and candidate-action identities observed by the adapter. This is an exposure
/// measurement, not authorization to call a tool or proof that a model used the evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionEvidenceUseContextV1 {
    pub run_id: String,
    pub episode_id: String,
    pub action_sequence: u64,
    pub event_id: u32,
    pub retrieval_batch_sha256: String,
    pub model_context_sha256: String,
    pub candidate_action_sha256: String,
    pub candidate_action_type: String,
    pub expected_decision: Action,
    pub required_scope: Scope,
    pub purpose_code: String,
    pub consumer_policy_sha256: String,
    pub valid_at_ms: i64,
    pub known_at_ms: i64,
    pub context_exposed_at_ms: i64,
    pub action_observed_at_ms: i64,
    pub require_complete_dependency_lineage: bool,
    pub evidence_span: ContextEvidenceSpanV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionEvidenceUseLinkRejectV1 {
    InvalidRunId,
    InvalidEpisodeId,
    InvalidDigest,
    ReceiptIntegrityInvalid,
    MissingVerifierCustody,
    EventMismatch,
    ActionMismatch,
    ScopeMismatch,
    PurposeMismatch,
    ConsumerPolicyMismatch,
    SnapshotMismatch,
    SourceNotTrustedForScope,
    LifecycleNotActive,
    DependencyLineageIncomplete,
    InvalidValidityInterval,
    InvalidTimestampOrder,
    ModelContextDigestMismatch,
    EmptyEvidenceSpan,
    EvidenceSpanOutOfBounds,
    EvidenceSpanNotUtf8Boundary,
    EvidenceRepresentationDigestMismatch,
    CandidateActionDigestMismatch,
    InvalidCandidateActionRecord,
    CandidateActionMetadataMismatch,
    CandidateActionTypeMismatch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionEvidenceUseLinkV1 {
    pub schema_version: String,
    pub run_id: String,
    pub episode_id: String,
    pub action_sequence: u64,
    pub event_id: u32,
    pub retrieval_batch_sha256: String,
    pub evidence_use_receipt_sha256: String,
    pub model_context_sha256: String,
    pub evidence_representation_sha256: String,
    pub candidate_action_sha256: String,
    pub candidate_action_type: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub valid_at_ms: i64,
    pub known_at_ms: i64,
    pub context_exposed_at_ms: i64,
    pub action_observed_at_ms: i64,
    pub evaluated_decision: Action,
    pub scope: Scope,
    pub purpose_code: String,
    pub consumer_policy_sha256: String,
    pub dependency_lineage_status: DependencyLineageStatusV1,
    pub lifecycle_status_as_known: SourceLifecycleStatus,
    pub exposure_stage: String,
    pub verification_custody: String,
    pub canonical_sha256: String,
}

fn link_digest_without_commitment(link: &DecisionEvidenceUseLinkV1) -> String {
    let mut h = Sha256::new();
    h.update(b"luminous:mace:decision-evidence-use-link:v1\0");
    for value in [
        &link.schema_version, &link.run_id, &link.episode_id,
        &link.retrieval_batch_sha256, &link.evidence_use_receipt_sha256,
        &link.model_context_sha256, &link.evidence_representation_sha256,
        &link.candidate_action_sha256, &link.candidate_action_type, &link.purpose_code,
        &link.consumer_policy_sha256, &link.exposure_stage, &link.verification_custody,
    ] { put_text(&mut h, value); }
    h.update(link.action_sequence.to_be_bytes());
    h.update(link.event_id.to_be_bytes());
    h.update((link.byte_start as u64).to_be_bytes());
    h.update((link.byte_end as u64).to_be_bytes());
    for value in [link.valid_at_ms, link.known_at_ms, link.context_exposed_at_ms, link.action_observed_at_ms] {
        h.update(value.to_be_bytes());
    }
    h.update([match link.evaluated_decision { Action::A => 1, Action::B => 2 }]);
    put_scope(&mut h, link.scope);
    put_dependency_status(&mut h, link.dependency_lineage_status);
    put_lifecycle(&mut h, link.lifecycle_status_as_known);
    h.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

impl DecisionEvidenceUseLinkV1 {
    pub fn verify_integrity(&self) -> bool {
        self.canonical_sha256 == link_digest_without_commitment(self)
    }

    /// Integrity-only interchange; the digest is not a portable issuer signature.
    pub fn to_json_value(&self) -> serde_json::Value {
        let regime = match self.scope.regime {
            broca_mace_policy::Regime::Baseline => "baseline",
            broca_mace_policy::Regime::Shifted => "shifted",
        };
        let objective = match self.scope.objective {
            broca_mace_policy::Objective::Quality => "quality",
            broca_mace_policy::Objective::Cost => "cost",
            broca_mace_policy::Objective::Speed => "speed",
        };
        let lifecycle = match self.lifecycle_status_as_known {
            SourceLifecycleStatus::Active => serde_json::json!({"status":"active"}),
            SourceLifecycleStatus::Retracted => serde_json::json!({"status":"retracted"}),
            SourceLifecycleStatus::Superseded { successor_event_id } =>
                serde_json::json!({"status":"superseded","successor_event_id":successor_event_id}),
        };
        let dependency = match self.dependency_lineage_status {
            DependencyLineageStatusV1::Complete => "complete",
            DependencyLineageStatusV1::Partial => "partial",
            DependencyLineageStatusV1::Unknown => "unknown",
        };
        let evaluated_decision = match self.evaluated_decision { Action::A => "A", Action::B => "B" };
        serde_json::json!({
            "schema_version":self.schema_version,
            "run_id":self.run_id,
            "episode_id":self.episode_id,
            "action_sequence":self.action_sequence,
            "event_id":self.event_id,
            "retrieval_batch_sha256":self.retrieval_batch_sha256,
            "evidence_use_receipt_sha256":self.evidence_use_receipt_sha256,
            "model_context_sha256":self.model_context_sha256,
            "evidence_representation_sha256":self.evidence_representation_sha256,
            "candidate_action_sha256":self.candidate_action_sha256,
            "candidate_action_type":self.candidate_action_type,
            "evidence_span":{"byte_start":self.byte_start,"byte_end":self.byte_end},
            "time":{
                "valid_at_ms":self.valid_at_ms,
                "known_at_ms":self.known_at_ms,
                "context_exposed_at_ms":self.context_exposed_at_ms,
                "action_observed_at_ms":self.action_observed_at_ms
            },
            "evaluated_decision":evaluated_decision,
            "scope":{"regime":regime,"objective":objective},
            "use":{"purpose_code":self.purpose_code,"consumer_policy_sha256":self.consumer_policy_sha256},
            "dependency_lineage":{"status":dependency},
            "lifecycle":lifecycle,
            "exposure_stage":self.exposure_stage,
            "verification_custody":self.verification_custody,
            "canonical_sha256":self.canonical_sha256
        })
    }
}

/// Bind a verified use receipt to exact bytes in a frozen model context and a later
/// candidate-action digest. This proves byte exposure, not semantic faithfulness,
/// model attention, or causal influence; those require transformation provenance and ablation.
pub fn link_verified_evidence_to_decision_v1(
    receipt: &EvidenceUseReceiptV1,
    context: &DecisionEvidenceUseContextV1,
    model_context_bytes: &[u8],
    candidate_action_bytes: &[u8],
) -> Result<DecisionEvidenceUseLinkV1, DecisionEvidenceUseLinkRejectV1> {
    if context.run_id.trim().is_empty() || context.run_id.len() > 128 {
        return Err(DecisionEvidenceUseLinkRejectV1::InvalidRunId);
    }
    if context.episode_id.trim().is_empty() || context.episode_id.len() > 128 {
        return Err(DecisionEvidenceUseLinkRejectV1::InvalidEpisodeId);
    }
    for value in [
        context.retrieval_batch_sha256.as_str(),
        context.model_context_sha256.as_str(),
        context.candidate_action_sha256.as_str(),
        context.evidence_span.representation_sha256.as_str(),
    ] {
        if !lower_sha256(value) { return Err(DecisionEvidenceUseLinkRejectV1::InvalidDigest); }
    }
    if !receipt.verify_integrity() {
        return Err(DecisionEvidenceUseLinkRejectV1::ReceiptIntegrityInvalid);
    }
    if !receipt.has_verified_ingress_provenance() {
        return Err(DecisionEvidenceUseLinkRejectV1::MissingVerifierCustody);
    }
    if receipt.event_id != context.event_id {
        return Err(DecisionEvidenceUseLinkRejectV1::EventMismatch);
    }
    if receipt.action != context.expected_decision {
        return Err(DecisionEvidenceUseLinkRejectV1::ActionMismatch);
    }
    if receipt.scope != context.required_scope {
        return Err(DecisionEvidenceUseLinkRejectV1::ScopeMismatch);
    }
    if receipt.purpose_code != context.purpose_code {
        return Err(DecisionEvidenceUseLinkRejectV1::PurposeMismatch);
    }
    if receipt.consumer_policy_sha256 != context.consumer_policy_sha256 {
        return Err(DecisionEvidenceUseLinkRejectV1::ConsumerPolicyMismatch);
    }
    if receipt.valid_at_ms != context.valid_at_ms || receipt.known_at_ms != context.known_at_ms {
        return Err(DecisionEvidenceUseLinkRejectV1::SnapshotMismatch);
    }
    if receipt.source_trust_status != SourceTrustStatus::TrustedForScope {
        return Err(DecisionEvidenceUseLinkRejectV1::SourceNotTrustedForScope);
    }
    if receipt.lifecycle_status_as_known != SourceLifecycleStatus::Active {
        return Err(DecisionEvidenceUseLinkRejectV1::LifecycleNotActive);
    }
    if context.require_complete_dependency_lineage
        && receipt.dependency_lineage_status != DependencyLineageStatusV1::Complete {
        return Err(DecisionEvidenceUseLinkRejectV1::DependencyLineageIncomplete);
    }
    if receipt.valid_until_ms.is_some_and(|until| until <= receipt.valid_from_ms)
        || receipt.valid_at_ms < receipt.valid_from_ms
        || receipt.valid_until_ms.is_some_and(|until| receipt.valid_at_ms >= until) {
        return Err(DecisionEvidenceUseLinkRejectV1::InvalidValidityInterval);
    }
    if context.context_exposed_at_ms < receipt.known_at_ms
        || context.context_exposed_at_ms < receipt.authority_received_at_ms
        || context.action_observed_at_ms < context.context_exposed_at_ms {
        return Err(DecisionEvidenceUseLinkRejectV1::InvalidTimestampOrder);
    }

    let actual_context_sha = Sha256::digest(model_context_bytes)
        .iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    if actual_context_sha != context.model_context_sha256 {
        return Err(DecisionEvidenceUseLinkRejectV1::ModelContextDigestMismatch);
    }
    let actual_action_sha = Sha256::digest(candidate_action_bytes)
        .iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    if actual_action_sha != context.candidate_action_sha256 {
        return Err(DecisionEvidenceUseLinkRejectV1::CandidateActionDigestMismatch);
    }
    let candidate_record: serde_json::Value = serde_json::from_slice(candidate_action_bytes)
        .map_err(|_| DecisionEvidenceUseLinkRejectV1::InvalidCandidateActionRecord)?;
    let metadata_matches = candidate_record.get("schema_version").and_then(|v| v.as_str())
            == Some("mace_onpolicy_candidate_action_v1")
        && candidate_record.get("record_type").and_then(|v| v.as_str()) == Some("candidate_action")
        && candidate_record.get("run_id").and_then(|v| v.as_str()) == Some(context.run_id.as_str())
        && candidate_record.get("episode_id").and_then(|v| v.as_str()) == Some(context.episode_id.as_str())
        && candidate_record.get("action_sequence").and_then(|v| v.as_u64()) == Some(context.action_sequence);
    if !metadata_matches {
        return Err(DecisionEvidenceUseLinkRejectV1::CandidateActionMetadataMismatch);
    }
    let candidate_action_type = candidate_record.get("action_type").and_then(|v| v.as_str())
        .ok_or(DecisionEvidenceUseLinkRejectV1::InvalidCandidateActionRecord)?;
    if candidate_action_type != context.candidate_action_type
        || !matches!(candidate_action_type, "respond" | "ask_user" | "memory_mutation" | "tool_request" | "abstain")
    {
        return Err(DecisionEvidenceUseLinkRejectV1::CandidateActionTypeMismatch);
    }
    let span = &context.evidence_span;
    if span.byte_start >= span.byte_end {
        return Err(DecisionEvidenceUseLinkRejectV1::EmptyEvidenceSpan);
    }
    if span.byte_end > model_context_bytes.len() {
        return Err(DecisionEvidenceUseLinkRejectV1::EvidenceSpanOutOfBounds);
    }
    let model_context = std::str::from_utf8(model_context_bytes)
        .map_err(|_| DecisionEvidenceUseLinkRejectV1::EvidenceSpanNotUtf8Boundary)?;
    if !model_context.is_char_boundary(span.byte_start) || !model_context.is_char_boundary(span.byte_end) {
        return Err(DecisionEvidenceUseLinkRejectV1::EvidenceSpanNotUtf8Boundary);
    }
    let representation_sha = Sha256::digest(&model_context_bytes[span.byte_start..span.byte_end])
        .iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    if representation_sha != span.representation_sha256 {
        return Err(DecisionEvidenceUseLinkRejectV1::EvidenceRepresentationDigestMismatch);
    }

    let mut link = DecisionEvidenceUseLinkV1 {
        schema_version: "mace_decision_evidence_use_link_v1".into(),
        run_id: context.run_id.clone(),
        episode_id: context.episode_id.clone(),
        action_sequence: context.action_sequence,
        event_id: receipt.event_id,
        retrieval_batch_sha256: context.retrieval_batch_sha256.clone(),
        evidence_use_receipt_sha256: receipt.canonical_sha256.clone(),
        model_context_sha256: context.model_context_sha256.clone(),
        evidence_representation_sha256: representation_sha,
        candidate_action_sha256: context.candidate_action_sha256.clone(),
        candidate_action_type: context.candidate_action_type.clone(),
        byte_start: span.byte_start,
        byte_end: span.byte_end,
        valid_at_ms: context.valid_at_ms,
        known_at_ms: context.known_at_ms,
        context_exposed_at_ms: context.context_exposed_at_ms,
        action_observed_at_ms: context.action_observed_at_ms,
        evaluated_decision: receipt.action,
        scope: receipt.scope,
        purpose_code: receipt.purpose_code.clone(),
        consumer_policy_sha256: receipt.consumer_policy_sha256.clone(),
        dependency_lineage_status: receipt.dependency_lineage_status,
        lifecycle_status_as_known: receipt.lifecycle_status_as_known,
        exposure_stage: "model_context_exposure".into(),
        verification_custody: "originating_process_only".into(),
        canonical_sha256: String::new(),
    };
    link.canonical_sha256 = link_digest_without_commitment(&link);
    Ok(link)
}

#[cfg(test)]
mod decision_evidence_use_link_tests {
    use super::*;

    const ACTION_BYTES: &[u8] = br#"{"schema_version":"mace_onpolicy_candidate_action_v1","record_type":"candidate_action","run_id":"run-001","episode_id":"episode-002","action_sequence":3,"previous_event_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","action_type":"respond","content":"choose A","evidence_event_ids":[]}"#;

    fn digest(bytes: &[u8]) -> String {
        Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn fixture_receipt() -> EvidenceUseReceiptV1 {
        let mut value = EvidenceUseReceiptV1 {
            event_id: 42,
            payload_sha256: "a".repeat(64),
            canonical_event_hash: "b".repeat(64),
            prior_raw_event_receipt_digest: "c".repeat(64),
            current_raw_event_receipt_digest: "d".repeat(64),
            current_authority_receipt_digest: "e".repeat(64),
            prior_lineage_receipt_digest: "f".repeat(64),
            current_lineage_receipt_digest: "1".repeat(64),
            prior_snapshot_sha256: "2".repeat(64),
            current_snapshot_sha256: "3".repeat(64),
            occurred_at_ms: 110,
            valid_from_ms: 100,
            valid_until_ms: Some(500),
            authority_received_at_ms: 120,
            prior_valid_at_ms: 150,
            prior_known_at_ms: 180,
            valid_at_ms: 200,
            known_at_ms: 220,
            action: Action::A,
            scope: Scope {
                regime: broca_mace_policy::Regime::Baseline,
                objective: broca_mace_policy::Objective::Quality,
            },
            source_family: 7,
            source_root_digest: SourceRootDigest::from_bytes([7; 32]),
            upstream_source_id: "fixture-source".into(),
            source_artifact_digest: "4".repeat(64),
            transformation_digest: "5".repeat(64),
            source_issuer_id: "fixture-issuer".into(),
            source_trust_status: SourceTrustStatus::TrustedForScope,
            source_trust_policy_sha256: "6".repeat(64),
            lifecycle_status_as_known: SourceLifecycleStatus::Active,
            dependency_lineage_status: DependencyLineageStatusV1::Complete,
            dependency_manifest_sha256: Some("7".repeat(64)),
            derived_from_event_ids: vec![40, 41],
            calibration_sha256: "8".repeat(64),
            measurement_unit: "normalized_utility".into(),
            measurement_error_bound_bits: 0.02_f64.to_bits(),
            causal_assessment_sha256: "9".repeat(64),
            semantic_estimand_sha256: "a".repeat(64),
            outcome_sha256: "b".repeat(64),
            purpose_code: "mace.onpolicy.answer".into(),
            consumer_policy_sha256: "c".repeat(64),
            canonical_sha256: String::new(),
            verified_in_process: true,
        };
        value.canonical_sha256 = value.digest_without_commitment();
        value
    }

    fn context(bytes: &[u8]) -> DecisionEvidenceUseContextV1 {
        let marker = b"verified";
        let start = bytes.windows(marker.len()).position(|window| window == marker).unwrap_or(0);
        let end = if start > 0 || bytes.starts_with(marker) {
            (start + b"verified preference alpha".len()).min(bytes.len())
        } else {
            bytes.len()
        };
        DecisionEvidenceUseContextV1 {
            run_id: "run-001".into(),
            episode_id: "episode-002".into(),
            action_sequence: 3,
            event_id: 42,
            retrieval_batch_sha256: "d".repeat(64),
            model_context_sha256: digest(bytes),
            candidate_action_sha256: digest(ACTION_BYTES),
            candidate_action_type: "respond".into(),
            expected_decision: Action::A,
            required_scope: Scope {
                regime: broca_mace_policy::Regime::Baseline,
                objective: broca_mace_policy::Objective::Quality,
            },
            purpose_code: "mace.onpolicy.answer".into(),
            consumer_policy_sha256: "c".repeat(64),
            valid_at_ms: 200,
            known_at_ms: 220,
            context_exposed_at_ms: 225,
            action_observed_at_ms: 230,
            require_complete_dependency_lineage: true,
            evidence_span: ContextEvidenceSpanV1 {
                byte_start: start,
                byte_end: end,
                representation_sha256: digest(&bytes[start..end]),
            },
        }
    }

    #[test]
    fn links_in_process_receipt_to_exact_context_span_and_action() {
        let bytes = b"source excerpt: verified preference alpha";
        let receipt = fixture_receipt();
        let link = link_verified_evidence_to_decision_v1(&receipt, &context(bytes), bytes, ACTION_BYTES).unwrap();
        assert_eq!(link.event_id, 42);
        assert_eq!(link.exposure_stage, "model_context_exposure");
        assert_eq!(link.evidence_use_receipt_sha256, receipt.canonical_sha256);
        assert!(link.verify_integrity());
        assert_eq!(link.to_json_value()["verification_custody"], "originating_process_only");
    }

    #[test]
    fn rejects_serialized_receipt_without_private_verifier_custody() {
        let mut receipt = fixture_receipt();
        receipt.verified_in_process = false;
        let bytes = b"source excerpt: verified preference alpha";
        assert_eq!(
            link_verified_evidence_to_decision_v1(&receipt, &context(bytes), bytes, ACTION_BYTES),
            Err(DecisionEvidenceUseLinkRejectV1::MissingVerifierCustody)
        );
    }

    #[test]
    fn rejects_wrong_snapshot_event_policy_and_untrusted_source() {
        let bytes = b"source excerpt: verified preference alpha";
        let receipt = fixture_receipt();
        let mut wrong = context(bytes);
        wrong.valid_at_ms = 201;
        assert_eq!(link_verified_evidence_to_decision_v1(&receipt, &wrong, bytes, ACTION_BYTES), Err(DecisionEvidenceUseLinkRejectV1::SnapshotMismatch));
        let mut wrong = context(bytes);
        wrong.event_id = 99;
        assert_eq!(link_verified_evidence_to_decision_v1(&receipt, &wrong, bytes, ACTION_BYTES), Err(DecisionEvidenceUseLinkRejectV1::EventMismatch));
        let mut untrusted = fixture_receipt();
        untrusted.source_trust_status = SourceTrustStatus::Untrusted;
        untrusted.canonical_sha256 = untrusted.digest_without_commitment();
        assert_eq!(link_verified_evidence_to_decision_v1(&untrusted, &context(bytes), bytes, ACTION_BYTES), Err(DecisionEvidenceUseLinkRejectV1::SourceNotTrustedForScope));
    }

    #[test]
    fn rejects_expiry_incomplete_ancestry_and_time_reversal() {
        let bytes = b"source excerpt: verified preference alpha";
        let mut expired = fixture_receipt();
        expired.valid_until_ms = Some(200);
        expired.canonical_sha256 = expired.digest_without_commitment();
        assert_eq!(link_verified_evidence_to_decision_v1(&expired, &context(bytes), bytes, ACTION_BYTES), Err(DecisionEvidenceUseLinkRejectV1::InvalidValidityInterval));
        let mut incomplete = context(bytes);
        incomplete.require_complete_dependency_lineage = true;
        let mut partial = fixture_receipt();
        partial.dependency_lineage_status = DependencyLineageStatusV1::Partial;
        partial.canonical_sha256 = partial.digest_without_commitment();
        assert_eq!(link_verified_evidence_to_decision_v1(&partial, &incomplete, bytes, ACTION_BYTES), Err(DecisionEvidenceUseLinkRejectV1::DependencyLineageIncomplete));
        let mut reversed = context(bytes);
        reversed.action_observed_at_ms = 219;
        assert_eq!(link_verified_evidence_to_decision_v1(&fixture_receipt(), &reversed, bytes, ACTION_BYTES), Err(DecisionEvidenceUseLinkRejectV1::InvalidTimestampOrder));
    }

    #[test]
    fn rejects_context_digest_representation_and_utf8_boundary_changes() {
        let bytes = b"source excerpt: verified preference alpha";
        let receipt = fixture_receipt();
        let mut wrong_context = context(bytes);
        wrong_context.model_context_sha256 = "f".repeat(64);
        assert_eq!(link_verified_evidence_to_decision_v1(&receipt, &wrong_context, bytes, ACTION_BYTES), Err(DecisionEvidenceUseLinkRejectV1::ModelContextDigestMismatch));
        let mut wrong_span = context(bytes);
        wrong_span.evidence_span.representation_sha256 = "a".repeat(64);
        assert_eq!(link_verified_evidence_to_decision_v1(&receipt, &wrong_span, bytes, ACTION_BYTES), Err(DecisionEvidenceUseLinkRejectV1::EvidenceRepresentationDigestMismatch));
        let unicode = "évidence";
        let mut bad_boundary = context(unicode.as_bytes());
        bad_boundary.evidence_span.byte_start = 1;
        bad_boundary.evidence_span.byte_end = 4;
        bad_boundary.evidence_span.representation_sha256 = digest(&unicode.as_bytes()[1..4]);
        assert_eq!(link_verified_evidence_to_decision_v1(&receipt, &bad_boundary, unicode.as_bytes(), ACTION_BYTES), Err(DecisionEvidenceUseLinkRejectV1::EvidenceSpanNotUtf8Boundary));
    }

    #[test]
    fn rejects_candidate_action_bytes_not_matching_action_digest() {
        let bytes = b"source excerpt: verified preference alpha";
        let mut altered = context(bytes);
        altered.candidate_action_sha256 = digest(b"different candidate action");
        assert_eq!(
            link_verified_evidence_to_decision_v1(
                &fixture_receipt(), &altered, bytes, ACTION_BYTES
            ),
            Err(DecisionEvidenceUseLinkRejectV1::CandidateActionDigestMismatch)
        );
    }


    #[test]
    fn rejects_candidate_action_identity_and_type_mismatch() {
        let bytes = b"source excerpt: verified preference alpha";
        let receipt = fixture_receipt();
        let wrong_identity = ACTION_BYTES
            .windows(b"run-001".len())
            .position(|window| window == b"run-001")
            .map(|start| {
                let mut value = ACTION_BYTES.to_vec();
                value.splice(start..start + b"run-001".len(), b"run-999".iter().copied());
                value
            })
            .unwrap();
        let mut wrong_identity_context = context(bytes);
        wrong_identity_context.candidate_action_sha256 = digest(&wrong_identity);
        assert_eq!(
            link_verified_evidence_to_decision_v1(
                &receipt, &wrong_identity_context, bytes, &wrong_identity
            ),
            Err(DecisionEvidenceUseLinkRejectV1::CandidateActionMetadataMismatch)
        );

        let mut wrong_type_context = context(bytes);
        wrong_type_context.candidate_action_type = "tool_request".into();
        assert_eq!(
            link_verified_evidence_to_decision_v1(
                &receipt, &wrong_type_context, bytes, ACTION_BYTES
            ),
            Err(DecisionEvidenceUseLinkRejectV1::CandidateActionTypeMismatch)
        );
    }


    #[test]
    fn accepts_exposure_after_the_bitemporal_snapshot() {
        let bytes = b"source excerpt: verified preference alpha";
        let link = link_verified_evidence_to_decision_v1(
            &fixture_receipt(), &context(bytes), bytes, ACTION_BYTES
        ).unwrap();
        assert_eq!(link.known_at_ms, 220);
        assert_eq!(link.context_exposed_at_ms, 225);
        assert!(link.context_exposed_at_ms > link.known_at_ms);
    }

}


/// Test-only receipt fixture for sibling ingress modules. This helper deliberately
/// bypasses the real verifier and is compiled only into test builds.
#[cfg(test)]
pub(crate) fn synthetic_verified_receipt_for_test(
    event_id: u32,
    scope: Scope,
    valid_at_ms: i64,
    known_at_ms: i64,
    purpose_code: &str,
    consumer_policy_sha256: &str,
    source_trust_status: SourceTrustStatus,
    lifecycle_status_as_known: SourceLifecycleStatus,
    dependency_lineage_status: DependencyLineageStatusV1,
) -> EvidenceUseReceiptV1 {
    let mut receipt = EvidenceUseReceiptV1 {
        event_id,
        payload_sha256: "a".repeat(64),
        canonical_event_hash: "b".repeat(64),
        prior_raw_event_receipt_digest: "c".repeat(64),
        current_raw_event_receipt_digest: "d".repeat(64),
        current_authority_receipt_digest: "e".repeat(64),
        prior_lineage_receipt_digest: "f".repeat(64),
        current_lineage_receipt_digest: "1".repeat(64),
        prior_snapshot_sha256: "2".repeat(64),
        current_snapshot_sha256: "3".repeat(64),
        occurred_at_ms: 170,
        valid_from_ms: 100,
        valid_until_ms: Some(300),
        authority_received_at_ms: 220,
        prior_valid_at_ms: 180,
        prior_known_at_ms: 200,
        valid_at_ms,
        known_at_ms,
        action: Action::A,
        scope,
        source_family: 7,
        source_root_digest: SourceRootDigest::from_bytes([7; 32]),
        upstream_source_id: "source-ledger-7".into(),
        source_artifact_digest: "4".repeat(64),
        transformation_digest: "5".repeat(64),
        source_issuer_id: "did:key:fixture-issuer".into(),
        source_trust_status,
        source_trust_policy_sha256: "6".repeat(64),
        lifecycle_status_as_known,
        dependency_lineage_status,
        dependency_manifest_sha256: Some("8".repeat(64)),
        derived_from_event_ids: Vec::new(),
        calibration_sha256: "9".repeat(64),
        measurement_unit: "normalized_utility".into(),
        measurement_error_bound_bits: 0.01_f64.to_bits(),
        causal_assessment_sha256: "a".repeat(64),
        semantic_estimand_sha256: "b".repeat(64),
        outcome_sha256: "c".repeat(64),
        purpose_code: purpose_code.into(),
        consumer_policy_sha256: consumer_policy_sha256.into(),
        canonical_sha256: String::new(),
        verified_in_process: true,
    };
    receipt.canonical_sha256 = receipt.digest_without_commitment();
    receipt
}

