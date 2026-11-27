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
        || after.source_issuer_id != before.source_issuer_id
    {
        return Err(EvidenceUseRejectV1::SourceIdentityChanged);
    }
    if after.source_trust_policy_digest != before.source_trust_policy_digest {
        return Err(EvidenceUseRejectV1::TrustPolicyChanged);
    }
    if after.calibration_digest != before.calibration_digest
        || after.calibration_revision != before.calibration_revision
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
