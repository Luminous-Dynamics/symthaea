pub mod reobservation;
pub mod salience_learning;
pub mod salience;
pub mod memory_budget;

// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Proposal-only consumer for receipt-qualified MACE consolidation summaries.
//!
//! This crate may propose a reusable procedure abstraction. It does not write
//! durable memory, erase data, grant authority, or replace Mycelix lifecycle
//! and privacy governance. The thresholds are prototype policy constants.

use broca_mace_ingress::{
    ActionConsolidationEvidenceSummaryV1, AsKnownQuery, ConsolidationEvidencePackV1,
    ConsolidationPackReject, ConsolidationSourceBindingV1, SourceTrustStatus,
};
use broca_mace_policy::{
    Action, CausalIdentificationV1, EvidenceDigest256, Experience,
    IdentificationMethodV1, Scope, SemanticEstimandV1,
};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HoldReason {
    NoTargetScopeEvidence,
    ScopeMismatch,
    PreRegistrationMissing,
    InsufficientQualifiedEvents,
    InsufficientIndependentFamilies,
    InsufficientIndependentRoots,
    MeasurementBoundMissing,
    MeasurementBoundTooWide,
    InvalidPolicyConfiguration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuarantineReason {
    UntrustedSourcePresent,
    UnknownTrustStatePresent,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcedurePromotionProposalV1 {
    pub action: broca_mace_policy::Action,
    pub target_scope: broca_mace_policy::Scope,
    pub as_known: broca_mace_ingress::AsKnownQuery,
    pub evidence_event_ids: Vec<u32>,
    pub qualified_success_event_count: usize,
    pub independent_source_family_count: usize,
    pub independent_source_root_count: usize,
    pub maximum_measurement_error_bound: f64,
    pub minimum_qualified_success_events: usize,
    pub minimum_independent_source_families: usize,
    pub minimum_independent_source_roots: usize,
    pub policy_revision: &'static str,
}

/// A proposal is not a consolidation transaction or authority grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsolidationDispositionV1 {
    PromoteToProcedure,
    PromoteToRule,
    RetainRaw,
    CompressWithSourceLinks,
    Quarantine,
    Supersede,
    ForgetFromActiveIndex,
    EraseForPrivacy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReversibilityClassV1 {
    ReversibleWithSourceLinks,
    IrreversiblePrivacyErase,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConsolidationReceiptContextV1 {
    pub consolidation_run_id: String,
    pub principal_id: String,
    pub capability_receipt_digest: String,
    pub policy_digest: String,
    pub destination_abstraction_id: String,
    pub output_artifact_digest: String,
    pub retention_policy_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConsolidationReceiptProposalV1 {
    pub schema_version: u16,
    pub disposition: ConsolidationDispositionV1,
    pub action: Action,
    pub target_scope: Scope,
    pub as_known: AsKnownQuery,
    pub policy_revision: String,
    pub policy_digest: String,
    pub principal_id: String,
    pub capability_receipt_digest: String,
    pub consolidation_run_id: String,
    pub input_episode_set_digest: String,
    pub destination_abstraction_id: String,
    pub output_artifact_digest: String,
    pub retention_policy_id: String,
    pub minimum_qualified_success_events: usize,
    pub minimum_independent_source_families: usize,
    pub minimum_independent_source_roots: usize,
    pub maximum_measurement_error_bound_bits: u64,
    pub reversibility: ReversibilityClassV1,
    pub source_bindings: Vec<ConsolidationSourceBindingV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReceiptProposalReject {
    NotProcedurePromotion,
    KnowledgeSnapshotMismatch,
    InvalidPolicyDigest,
    InvalidCapabilityReceiptDigest,
    InvalidOutputArtifactDigest,
    MissingReceiptContext,
    EvidencePackRejected(ConsolidationPackReject),
    SourceActionOrScopeMismatch,
    InsufficientQualifiedSources,
    MeasurementBoundExceedsProposal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedConsolidationReceiptV1 {
    pub proposal_digest: String,
    pub verification_receipt_digest: String,
    pub verifier_id: String,
    pub verifier_role: String,
    pub verified_at_ms: i64,
    pub authority_scope: Scope,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReceiptUseReject {
    NotProcedurePromotion,
    ProposalDigestMismatch,
    MalformedVerificationReceipt,
    WrongAuthorityScope,
    ReceiptNotKnownAtTransfer,
    KnowledgeSnapshotMismatch,
    SourceBindingsChanged,
}

/// Implementations must verify an independently issued receipt over the exact
/// proposal digest, issuer, authority scope and revocation/validity status.
pub trait ConsolidationReceiptVerifierV1 {
    fn verify(
        &self,
        proposal: &ConsolidationReceiptProposalV1,
    ) -> Option<VerifiedConsolidationReceiptV1>;
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn scope_code(scope: Scope) -> &'static str {
    match (scope.regime, scope.objective) {
        (
            broca_mace_policy::Regime::Baseline,
            broca_mace_policy::Objective::Quality,
        ) => "baseline:quality",
        (
            broca_mace_policy::Regime::Baseline,
            broca_mace_policy::Objective::Cost,
        ) => "baseline:cost",
        (
            broca_mace_policy::Regime::Baseline,
            broca_mace_policy::Objective::Speed,
        ) => "baseline:speed",
        (
            broca_mace_policy::Regime::Shifted,
            broca_mace_policy::Objective::Quality,
        ) => "shifted:quality",
        (
            broca_mace_policy::Regime::Shifted,
            broca_mace_policy::Objective::Cost,
        ) => "shifted:cost",
        (
            broca_mace_policy::Regime::Shifted,
            broca_mace_policy::Objective::Speed,
        ) => "shifted:speed",
    }
}

fn source_trust_code(status: SourceTrustStatus) -> &'static str {
    match status {
        SourceTrustStatus::TrustedForScope => "trusted_for_scope",
        SourceTrustStatus::Untrusted => "untrusted",
        SourceTrustStatus::Unknown => "unknown",
    }
}

fn digest_fields(domain: &[u8], fields: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in fields {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn identification_method_code(method: IdentificationMethodV1) -> &'static str {
    match method {
        IdentificationMethodV1::RandomizedControlledTrial => "randomized_controlled_trial",
        IdentificationMethodV1::QuasiExperimental => "quasi_experimental",
        IdentificationMethodV1::Observational => "observational",
        IdentificationMethodV1::Unknown => "unknown",
    }
}

fn evidence_digest_code(digest: EvidenceDigest256) -> String {
    digest
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Canonical serialization of every causal-identification field. These values
/// are part of the source evidence binding, not optional policy annotations.
fn causal_identification_fields(identification: CausalIdentificationV1) -> Vec<String> {
    let mut fields = vec![
        identification_method_code(identification.method).to_owned(),
        evidence_digest_code(identification.design_digest),
        evidence_digest_code(identification.estimand_digest),
        identification.propensity_min_bps.to_string(),
        identification.propensity_max_bps.to_string(),
        identification.max_standardized_mean_difference_micros.to_string(),
        identification.interference_controlled.to_string(),
    ];
    fields.extend(identification.semantic_estimand.canonical_fields());
    fields
}

fn binding_fields(binding: &ConsolidationSourceBindingV1) -> Vec<String> {
    let receipt = &binding.receipts;
    let mut fields = vec![
        binding.event_id.to_string(),
        binding.action.name().to_owned(),
        scope_code(binding.scope).to_owned(),
        binding.measurement_error_bound_bits.to_string(),
        receipt.raw_event_digest.clone(),
        receipt.raw_event_receipt_digest.clone(),
        receipt.session_id.clone(),
        receipt.task_id.clone(),
        receipt.canonical_event_hash.clone(),
        receipt.authority_receipt_digest.clone(),
        receipt.event_occurred_at_ms.to_string(),
        receipt.authority_received_at_ms.to_string(),
        receipt.lineage_digest.clone(),
        receipt.source_family.to_string(),
        receipt.source_root_id.to_string(),
        receipt.source_root_digest.to_hex(),
        receipt.source_acquisition_mode.clone(),
        receipt.upstream_source_id.clone(),
        receipt.source_artifact_digest.clone(),
        receipt.transformation_digest.clone(),
        receipt.source_issuer_id.clone(),
        source_trust_code(receipt.source_trust_status).to_owned(),
        receipt.source_trust_policy_digest.clone(),
        receipt.calibration_digest.clone(),
        receipt.measurement_unit.clone(),
        receipt.measurement_instrument_id.clone(),
        receipt.calibration_method_id.clone(),
        receipt.calibration_revision.clone(),
        receipt.causal_assessment_digest.clone(),
    ];
    fields.extend(causal_identification_fields(receipt.causal_identification));
    fields.extend([
        receipt.outcome_success.to_string(),
        receipt.outcome_observed_utility_bits.to_string(),
        receipt.outcome_utility_unit.clone(),
        receipt.outcome_digest.clone(),
    ]);
    fields
}

impl ConsolidationReceiptProposalV1 {
    pub fn from_procedure_promotion(
        proposal: &ProcedurePromotionProposalV1,
        pack: &ConsolidationEvidencePackV1,
        context: ConsolidationReceiptContextV1,
    ) -> Result<Self, ReceiptProposalReject> {
        if pack.as_known() != proposal.as_known {
            return Err(ReceiptProposalReject::KnowledgeSnapshotMismatch);
        }
        if !is_lower_sha256(&context.policy_digest) {
            return Err(ReceiptProposalReject::InvalidPolicyDigest);
        }
        if !is_lower_sha256(&context.capability_receipt_digest) {
            return Err(ReceiptProposalReject::InvalidCapabilityReceiptDigest);
        }
        if !is_lower_sha256(&context.output_artifact_digest) {
            return Err(ReceiptProposalReject::InvalidOutputArtifactDigest);
        }
        if context.consolidation_run_id.is_empty()
            || context.principal_id.is_empty()
            || context.destination_abstraction_id.is_empty()
            || context.retention_policy_id.is_empty()
        {
            return Err(ReceiptProposalReject::MissingReceiptContext);
        }

        let source_bindings = pack
            .source_bindings_for_event_ids(&proposal.evidence_event_ids)
            .map_err(ReceiptProposalReject::EvidencePackRejected)?;
        if source_bindings.len() < proposal.minimum_qualified_success_events {
            return Err(ReceiptProposalReject::InsufficientQualifiedSources);
        }
        let mut families = std::collections::BTreeSet::new();
        let mut roots = std::collections::BTreeSet::new();
        for binding in &source_bindings {
            if binding.action != proposal.action || binding.scope != proposal.target_scope {
                return Err(ReceiptProposalReject::SourceActionOrScopeMismatch);
            }
            let error_bound = f64::from_bits(binding.measurement_error_bound_bits);
            if !error_bound.is_finite()
                || !(0.0..=proposal.maximum_measurement_error_bound)
                    .contains(&error_bound)
            {
                return Err(ReceiptProposalReject::MeasurementBoundExceedsProposal);
            }
            families.insert(binding.receipts.source_family);
            roots.insert(binding.receipts.source_root_digest);
        }
        if families.len() < proposal.minimum_independent_source_families
            || roots.len() < proposal.minimum_independent_source_roots
        {
            return Err(ReceiptProposalReject::InsufficientQualifiedSources);
        }

        let mut input_fields = Vec::new();
        for binding in &source_bindings {
            input_fields.extend(binding_fields(binding));
        }
        let input_episode_set_digest =
            digest_fields(b"mycelix:mace:episode-set:v1\0", &input_fields);

        Ok(Self {
            schema_version: 1,
            disposition: ConsolidationDispositionV1::PromoteToProcedure,
            action: proposal.action,
            target_scope: proposal.target_scope,
            as_known: proposal.as_known,
            policy_revision: proposal.policy_revision.to_owned(),
            policy_digest: context.policy_digest,
            principal_id: context.principal_id,
            capability_receipt_digest: context.capability_receipt_digest,
            consolidation_run_id: context.consolidation_run_id,
            input_episode_set_digest,
            destination_abstraction_id: context.destination_abstraction_id,
            output_artifact_digest: context.output_artifact_digest,
            retention_policy_id: context.retention_policy_id,
            minimum_qualified_success_events: proposal.minimum_qualified_success_events,
            minimum_independent_source_families: proposal.minimum_independent_source_families,
            minimum_independent_source_roots: proposal.minimum_independent_source_roots,
            maximum_measurement_error_bound_bits:
                proposal.maximum_measurement_error_bound.to_bits(),
            reversibility: ReversibilityClassV1::ReversibleWithSourceLinks,
            source_bindings,
        })
    }

    /// Canonical digest over decision context, output identity and every
    /// receipt-bound source revision. String fields are length-prefixed.
    pub fn canonical_digest(&self) -> String {
        let mut fields = vec![
            self.schema_version.to_string(),
            match self.disposition {
                ConsolidationDispositionV1::PromoteToProcedure => "promote_to_procedure",
                ConsolidationDispositionV1::PromoteToRule => "promote_to_rule",
                ConsolidationDispositionV1::RetainRaw => "retain_raw",
                ConsolidationDispositionV1::CompressWithSourceLinks => "compress_with_source_links",
                ConsolidationDispositionV1::Quarantine => "quarantine",
                ConsolidationDispositionV1::Supersede => "supersede",
                ConsolidationDispositionV1::ForgetFromActiveIndex => "forget_from_active_index",
                ConsolidationDispositionV1::EraseForPrivacy => "erase_for_privacy",
            }
            .to_owned(),
            self.action.name().to_owned(),
            scope_code(self.target_scope).to_owned(),
            self.as_known.valid_at_ms.to_string(),
            self.as_known.known_at_ms.to_string(),
            self.policy_revision.clone(),
            self.policy_digest.clone(),
            self.principal_id.clone(),
            self.capability_receipt_digest.clone(),
            self.consolidation_run_id.clone(),
            self.input_episode_set_digest.clone(),
            self.destination_abstraction_id.clone(),
            self.output_artifact_digest.clone(),
            self.retention_policy_id.clone(),
            self.minimum_qualified_success_events.to_string(),
            self.minimum_independent_source_families.to_string(),
            self.minimum_independent_source_roots.to_string(),
            self.maximum_measurement_error_bound_bits.to_string(),
            match self.reversibility {
                ReversibilityClassV1::ReversibleWithSourceLinks => "reversible_with_source_links",
                ReversibilityClassV1::IrreversiblePrivacyErase => "irreversible_privacy_erase",
            }
            .to_owned(),
            self.source_bindings.len().to_string(),
        ];
        for binding in &self.source_bindings {
            fields.extend(binding_fields(binding));
        }
        digest_fields(b"mycelix:mace:consolidation-decision:v1\0", &fields)
    }
}

pub fn verify_and_release_promoted_experiences(
    proposal: &ConsolidationReceiptProposalV1,
    pack: &ConsolidationEvidencePackV1,
    transfer_known_at_ms: i64,
    verifier: &impl ConsolidationReceiptVerifierV1,
) -> Result<(VerifiedConsolidationReceiptV1, Vec<Experience>), ReceiptUseReject> {
    if proposal.disposition != ConsolidationDispositionV1::PromoteToProcedure {
        return Err(ReceiptUseReject::NotProcedurePromotion);
    }
    if pack.as_known() != proposal.as_known
        || transfer_known_at_ms != proposal.as_known.known_at_ms
    {
        return Err(ReceiptUseReject::KnowledgeSnapshotMismatch);
    }

    let verified = verifier
        .verify(proposal)
        .ok_or(ReceiptUseReject::MalformedVerificationReceipt)?;
    if verified.proposal_digest != proposal.canonical_digest() {
        return Err(ReceiptUseReject::ProposalDigestMismatch);
    }
    if !is_lower_sha256(&verified.verification_receipt_digest)
        || verified.verifier_id.is_empty()
        || verified.verifier_role.is_empty()
    {
        return Err(ReceiptUseReject::MalformedVerificationReceipt);
    }
    if verified.authority_scope != proposal.target_scope {
        return Err(ReceiptUseReject::WrongAuthorityScope);
    }
    if verified.verified_at_ms < proposal.as_known.known_at_ms
        || verified.verified_at_ms > transfer_known_at_ms
    {
        return Err(ReceiptUseReject::ReceiptNotKnownAtTransfer);
    }

    let event_ids: Vec<u32> = proposal
        .source_bindings
        .iter()
        .map(|binding| binding.event_id)
        .collect();
    let current_bindings = pack
        .source_bindings_for_event_ids(&event_ids)
        .map_err(|_| ReceiptUseReject::SourceBindingsChanged)?;
    if current_bindings != proposal.source_bindings {
        return Err(ReceiptUseReject::SourceBindingsChanged);
    }

    let experiences = pack
        .events()
        .iter()
        .filter(|event| event_ids.contains(&event.event_id()))
        .map(|event| event.experience())
        .collect();
    if experiences.len() != event_ids.len() {
        return Err(ReceiptUseReject::SourceBindingsChanged);
    }
    Ok((verified, experiences))
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConsolidationProposalV1 {
    ProposeProcedure(ProcedurePromotionProposalV1),
    RetainProvisional(HoldReason),
    Quarantine(QuarantineReason),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsolidationPolicyV1 {
    pub minimum_qualified_success_events: usize,
    pub minimum_independent_source_families: usize,
    pub minimum_independent_source_roots: usize,
    pub maximum_measurement_error_bound: f64,
}

impl Default for ConsolidationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_qualified_success_events: 3,
            minimum_independent_source_families: 3,
            minimum_independent_source_roots: 3,
            maximum_measurement_error_bound: 0.05,
        }
    }
}

impl ConsolidationPolicyV1 {
    pub fn decide_procedure_promotion(
        &self,
        evidence: &ActionConsolidationEvidenceSummaryV1,
    ) -> ConsolidationProposalV1 {
        if self.minimum_qualified_success_events == 0
            || self.minimum_independent_source_families == 0
            || self.minimum_independent_source_roots == 0
            || !self.maximum_measurement_error_bound.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_measurement_error_bound)
        {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::InvalidPolicyConfiguration,
            );
        }
        if evidence.target_scope_event_count == 0 {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::NoTargetScopeEvidence,
            );
        }
        if !evidence.scope_matches {
            return ConsolidationProposalV1::RetainProvisional(HoldReason::ScopeMismatch);
        }
        if evidence.untrusted_event_count > 0 {
            return ConsolidationProposalV1::Quarantine(
                QuarantineReason::UntrustedSourcePresent,
            );
        }
        if evidence.unknown_trust_event_count > 0 {
            return ConsolidationProposalV1::Quarantine(
                QuarantineReason::UnknownTrustStatePresent,
            );
        }
        if !evidence.all_trusted_predictions_pre_registered {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::PreRegistrationMissing,
            );
        }
        if evidence.qualified_success_event_count
            < self.minimum_qualified_success_events
        {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::InsufficientQualifiedEvents,
            );
        }
        if evidence.qualified_success_source_family_count
            < self.minimum_independent_source_families
        {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::InsufficientIndependentFamilies,
            );
        }
        if evidence.qualified_success_source_root_count
            < self.minimum_independent_source_roots
        {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::InsufficientIndependentRoots,
            );
        }
        let Some(maximum_measurement_error_bound) =
            evidence.max_qualified_success_measurement_error_bound
        else {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::MeasurementBoundMissing,
            );
        };
        if !maximum_measurement_error_bound.is_finite()
            || !(0.0..=self.maximum_measurement_error_bound)
                .contains(&maximum_measurement_error_bound)
        {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::MeasurementBoundTooWide,
            );
        }

        ConsolidationProposalV1::ProposeProcedure(ProcedurePromotionProposalV1 {
            action: evidence.action,
            target_scope: evidence.target_scope,
            as_known: evidence.as_known,
            evidence_event_ids: evidence.qualified_success_event_ids.clone(),
            qualified_success_event_count: evidence.qualified_success_event_count,
            independent_source_family_count:
                evidence.qualified_success_source_family_count,
            independent_source_root_count:
                evidence.qualified_success_source_root_count,
            maximum_measurement_error_bound,
            minimum_qualified_success_events: self.minimum_qualified_success_events,
            minimum_independent_source_families: self.minimum_independent_source_families,
            minimum_independent_source_roots: self.minimum_independent_source_roots,
            policy_revision: "mace-consolidation-proposal-v1",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use broca_mace_ingress::{
        Action, AsKnownQuery, ConsolidationEvidenceSummaryV1, EvidenceReceiptBundle,
        SourceTrustStatus,
    };
    use broca_mace_policy::{Objective, Regime, Scope, SemanticEstimandV1};

    fn fixture_semantic_estimand() -> SemanticEstimandV1 {
        SemanticEstimandV1::new(
            1,
            EvidenceDigest256::from_bytes([21; 32]),
            EvidenceDigest256::from_bytes([22; 32]),
            EvidenceDigest256::from_bytes([23; 32]),
            EvidenceDigest256::from_bytes([24; 32]),
            EvidenceDigest256::from_bytes([25; 32]),
            30_000,
            EvidenceDigest256::from_bytes([26; 32]),
            EvidenceDigest256::from_bytes([27; 32]),
            EvidenceDigest256::from_bytes([28; 32]),
        )
    }

    fn summary() -> ActionConsolidationEvidenceSummaryV1 {
        ActionConsolidationEvidenceSummaryV1 {
            as_known: AsKnownQuery {
                valid_at_ms: 180,
                known_at_ms: 250,
            },
            target_scope: Scope {
                regime: Regime::Baseline,
                objective: Objective::Quality,
            },
            action: Action::A,
            action_episode_count: 3,
            target_scope_event_count: 3,
            trusted_event_count: 3,
            trusted_source_family_count: 3,
            trusted_source_root_count: 3,
            qualified_success_event_count: 3,
            qualified_success_source_family_count: 3,
            qualified_success_source_root_count: 3,
            untrusted_event_count: 0,
            unknown_trust_event_count: 0,
            qualified_success_event_ids: vec![100, 101, 102],
            max_qualified_success_measurement_error_bound: Some(0.02),
            scope_matches: true,
            all_trusted_predictions_pre_registered: true,
        }
    }

    fn source_binding_fixture(
        identification: CausalIdentificationV1,
    ) -> ConsolidationSourceBindingV1 {
        let digest = "a".repeat(64);
        ConsolidationSourceBindingV1 {
            event_id: 1,
            action: Action::A,
            scope: Scope {
                regime: Regime::Baseline,
                objective: Objective::Quality,
            },
            measurement_error_bound_bits: 0.02_f64.to_bits(),
            receipts: EvidenceReceiptBundle {
                raw_event_digest: digest.clone(),
                raw_event_receipt_digest: digest.clone(),
                session_id: "session-a".into(),
                task_id: "task-a".into(),
                canonical_event_hash: digest.clone(),
                authority_receipt_digest: digest.clone(),
                event_occurred_at_ms: 100,
                authority_received_at_ms: 200,
                lineage_digest: digest.clone(),
                source_family: 1,
                source_root_id: 1,
                source_root_digest: broca_mace_policy::SourceRootDigest::from_bytes([1; 32]),
                source_acquisition_mode: "fixture".into(),
                upstream_source_id: "upstream-1".into(),
                source_artifact_digest: digest.clone(),
                transformation_digest: digest.clone(),
                source_issuer_id: "issuer-1".into(),
                source_trust_status: SourceTrustStatus::TrustedForScope,
                source_trust_policy_digest: digest.clone(),
                calibration_digest: digest.clone(),
                measurement_unit: "normalized_utility".into(),
                measurement_instrument_id: "instrument-1".into(),
                calibration_method_id: "method-1".into(),
                calibration_revision: "cal-r1".into(),
                causal_assessment_digest: digest.clone(),
                causal_identification: identification,
                outcome_success: true,
                outcome_observed_utility_bits: 0.9_f64.to_bits(),
                outcome_utility_unit: "normalized_utility".into(),
                outcome_digest: digest,
            },
        }
    }

    #[test]
    fn canonical_source_binding_covers_every_semantic_estimand_dimension() {
        let base = fixture_semantic_estimand();
        let base_digest = digest_fields(
            b"mycelix:mace:semantic-estimand:v1\\0",
            &base.canonical_fields(),
        );
        let variants = [
            SemanticEstimandV1 { schema_version: 2, ..base },
            SemanticEstimandV1 {
                population_spec_digest: EvidenceDigest256::from_bytes([31; 32]),
                ..base
            },
            SemanticEstimandV1 {
                treatment_version_digest: EvidenceDigest256::from_bytes([32; 32]),
                ..base
            },
            SemanticEstimandV1 {
                comparator_version_digest: EvidenceDigest256::from_bytes([33; 32]),
                ..base
            },
            SemanticEstimandV1 {
                outcome_definition_digest: EvidenceDigest256::from_bytes([34; 32]),
                ..base
            },
            SemanticEstimandV1 {
                outcome_unit_digest: EvidenceDigest256::from_bytes([35; 32]),
                ..base
            },
            SemanticEstimandV1 {
                follow_up_horizon_ms: base.follow_up_horizon_ms + 1,
                ..base
            },
            SemanticEstimandV1 {
                censoring_strategy_digest: EvidenceDigest256::from_bytes([36; 32]),
                ..base
            },
            SemanticEstimandV1 {
                intercurrent_event_strategy_digest: EvidenceDigest256::from_bytes([37; 32]),
                ..base
            },
            SemanticEstimandV1 {
                target_weighting_digest: EvidenceDigest256::from_bytes([38; 32]),
                ..base
            },
        ];

        for variant in variants {
            assert_ne!(
                digest_fields(
                    b"mycelix:mace:semantic-estimand:v1\\0",
                    &variant.canonical_fields(),
                ),
                base_digest
            );
        }
    }

    #[test]
    fn canonical_source_binding_covers_every_causal_identification_field() {
        let mut bytes_a = [0_u8; 32];
        let mut bytes_b = [0_u8; 32];
        let mut bytes_c = [0_u8; 32];
        bytes_a[0] = 1;
        bytes_b[0] = 2;
        bytes_c[0] = 3;
        let base = CausalIdentificationV1::new(
            IdentificationMethodV1::RandomizedControlledTrial,
            EvidenceDigest256::from_bytes(bytes_a),
            EvidenceDigest256::from_bytes(bytes_b),
            5_000,
            5_000,
            50_000,
            true,
        );
        let base_binding = source_binding_fixture(base);
        let base_digest = digest_fields(
            b"mycelix:mace:source-binding-test:v1\\0",
            &binding_fields(&base_binding),
        );
        let variants = [
            CausalIdentificationV1::new(
                IdentificationMethodV1::QuasiExperimental,
                base.design_digest,
                base.estimand_digest,
                base.propensity_min_bps,
                base.propensity_max_bps,
                base.max_standardized_mean_difference_micros,
                base.interference_controlled,
            ),
            CausalIdentificationV1::new(
                base.method,
                EvidenceDigest256::from_bytes(bytes_c),
                base.estimand_digest,
                base.propensity_min_bps,
                base.propensity_max_bps,
                base.max_standardized_mean_difference_micros,
                base.interference_controlled,
            ),
            CausalIdentificationV1::new(
                base.method,
                base.design_digest,
                EvidenceDigest256::from_bytes(bytes_c),
                base.propensity_min_bps,
                base.propensity_max_bps,
                base.max_standardized_mean_difference_micros,
                base.interference_controlled,
            ),
            CausalIdentificationV1::new(
                base.method,
                base.design_digest,
                base.estimand_digest,
                4_000,
                base.propensity_max_bps,
                base.max_standardized_mean_difference_micros,
                base.interference_controlled,
            ),
            CausalIdentificationV1::new(
                base.method,
                base.design_digest,
                base.estimand_digest,
                base.propensity_min_bps,
                6_000,
                base.max_standardized_mean_difference_micros,
                base.interference_controlled,
            ),
            CausalIdentificationV1::new(
                base.method,
                base.design_digest,
                base.estimand_digest,
                base.propensity_min_bps,
                base.propensity_max_bps,
                60_000,
                base.interference_controlled,
            ),
            CausalIdentificationV1::new(
                base.method,
                base.design_digest,
                base.estimand_digest,
                base.propensity_min_bps,
                base.propensity_max_bps,
                base.max_standardized_mean_difference_micros,
                false,
            ),
        ];

        for variant in variants {
            let mut binding = base_binding.clone();
            binding.receipts.causal_identification = variant;
            let digest = digest_fields(
                b"mycelix:mace:source-binding-test:v1\\0",
                &binding_fields(&binding),
            );
            assert_ne!(digest, base_digest);
        }
        assert_eq!(causal_identification_fields(base).len(), 7);
    }

    #[test]
    fn three_independent_roots_produce_a_source_linked_proposal_only() {
        let decision = ConsolidationPolicyV1::default()
            .decide_procedure_promotion(&summary());
        let ConsolidationProposalV1::ProposeProcedure(proposal) = decision else {
            panic!("three independently supported successes should propose promotion");
        };
        assert_eq!(proposal.action, Action::A);
        assert_eq!(proposal.evidence_event_ids, vec![100, 101, 102]);
        assert_eq!(proposal.independent_source_root_count, 3);
        assert_eq!(proposal.independent_source_family_count, 3);
        assert_eq!(proposal.minimum_qualified_success_events, 3);
        assert_eq!(proposal.minimum_independent_source_families, 3);
        assert_eq!(proposal.minimum_independent_source_roots, 3);
        assert_eq!(proposal.maximum_measurement_error_bound, 0.05);
        assert_eq!(proposal.policy_revision, "mace-consolidation-proposal-v1");
    }

    #[test]
    fn three_family_labels_from_one_root_do_not_promote() {
        let mut evidence = summary();
        evidence.qualified_success_source_root_count = 1;
        let decision = ConsolidationPolicyV1::default()
            .decide_procedure_promotion(&evidence);
        assert_eq!(
            decision,
            ConsolidationProposalV1::RetainProvisional(
                HoldReason::InsufficientIndependentRoots
            )
        );
    }

    #[test]
    fn untrusted_or_unknown_sources_are_quarantined_before_promotion() {
        let mut untrusted = summary();
        untrusted.untrusted_event_count = 1;
        assert_eq!(
            ConsolidationPolicyV1::default().decide_procedure_promotion(&untrusted),
            ConsolidationProposalV1::Quarantine(
                QuarantineReason::UntrustedSourcePresent
            )
        );

        let mut unknown = summary();
        unknown.unknown_trust_event_count = 1;
        assert_eq!(
            ConsolidationPolicyV1::default().decide_procedure_promotion(&unknown),
            ConsolidationProposalV1::Quarantine(
                QuarantineReason::UnknownTrustStatePresent
            )
        );
    }

    #[test]
    fn scope_and_preregistration_are_required() {
        let mut wrong_scope = summary();
        wrong_scope.scope_matches = false;
        assert_eq!(
            ConsolidationPolicyV1::default().decide_procedure_promotion(&wrong_scope),
            ConsolidationProposalV1::RetainProvisional(HoldReason::ScopeMismatch)
        );

        let mut unregistered = summary();
        unregistered.all_trusted_predictions_pre_registered = false;
        assert_eq!(
            ConsolidationPolicyV1::default().decide_procedure_promotion(&unregistered),
            ConsolidationProposalV1::RetainProvisional(
                HoldReason::PreRegistrationMissing
            )
        );
    }

    #[test]
    fn error_bounds_must_be_present_finite_and_within_policy() {
        for bound in [None, Some(f64::NAN), Some(0.051)] {
            let mut evidence = summary();
            evidence.max_qualified_success_measurement_error_bound = bound;
            let expected = if bound.is_none() {
                HoldReason::MeasurementBoundMissing
            } else {
                HoldReason::MeasurementBoundTooWide
            };
            assert_eq!(
                ConsolidationPolicyV1::default().decide_procedure_promotion(&evidence),
                ConsolidationProposalV1::RetainProvisional(expected)
            );
        }
    }

    #[test]
    fn invalid_policy_configuration_cannot_lower_the_promotion_gate() {
        let mut policy = ConsolidationPolicyV1::default();
        policy.minimum_independent_source_roots = 0;
        assert_eq!(
            policy.decide_procedure_promotion(&summary()),
            ConsolidationProposalV1::RetainProvisional(
                HoldReason::InvalidPolicyConfiguration
            )
        );

        let mut invalid_bound = ConsolidationPolicyV1::default();
        invalid_bound.maximum_measurement_error_bound = f64::NAN;
        assert_eq!(
            invalid_bound.decide_procedure_promotion(&summary()),
            ConsolidationProposalV1::RetainProvisional(
                HoldReason::InvalidPolicyConfiguration
            )
        );
    }

    #[test]
    fn global_summary_type_remains_decision_neutral() {
        let _type_check: Option<ConsolidationEvidenceSummaryV1> = None;
        let _trust_class: SourceTrustStatus = SourceTrustStatus::TrustedForScope;
    }
}
