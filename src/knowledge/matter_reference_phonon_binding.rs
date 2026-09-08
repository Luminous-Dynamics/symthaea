// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bind explicit phonon post-processing execution to the frozen reference protocol.
//!
//! A provenance-preserving `pw -> ph -> q2r -> matdyn` chain still must not drift
//! from the benchmark's preregistered q-grid/shift/ASR policy. This boundary checks
//! that match while preserving the benchmark's execution-only authority and its
//! unconditional thermodynamic-phase withholding.
//!
//! The receipt is a local projection of an external AiiDA export. This module
//! validates canonical evidence roles/content identities and the protocol match,
//! but does not parse the external chain-manifest bytes itself. Therefore the
//! mapping remains declaration-level until a versioned wire parser owns it.

use std::collections::BTreeSet;
use std::fmt;

use symthaea_evidence_plane::external_receipt::{
    EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle, ExternalEvidenceError,
    Sha256Digest,
};

pub use super::matter_reference_crystal_benchmark::{
    AcousticSumRuleTreatment, BoundReferenceCrystalBenchmarkProtocolV1,
    PhononFinalizationPolicyV1, ReferenceBenchmarkAuthorityV1,
    ReferenceBenchmarkProtocolInterpretationV1, ThermodynamicPhaseEligibilityV1,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferencePhononChainReceiptV1 {
    pub chain_manifest_evidence_id: String,
    pub phonon_preflight_evidence_id: String,
    pub pw_execution_evidence_id: String,
    pub ph_execution_evidence_id: String,
    pub q2r_execution_evidence_id: String,
    pub matdyn_execution_evidence_id: String,
    pub ph_qpoints_evidence_id: String,
    pub matdyn_sampling_evidence_id: String,
    pub force_constants_evidence_id: String,
    pub phonon_bands_evidence_id: String,
    /// Exact q-grid reconstructed by the external adapter from `ph.x` KpointsData.
    pub ph_reciprocal_grid: [u32; 3],
    /// Exact f64 bits for the fractional q-grid shift.
    pub ph_reciprocal_shift_fractional_bits: [u64; 3],
    /// ASR treatment reconstructed from the explicit `matdyn.x` configuration.
    pub matdyn_acoustic_sum_rule: AcousticSumRuleTreatment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferencePhononFinalizationV1 {
    /// Explicit q2r + matdyn execution evidence is present and matches the
    /// benchmark's frozen q-grid/shift/ASR policy. This is not a stability claim.
    ExplicitPostprocessingChainBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferencePhononBindingInterpretationV1 {
    /// The supplied receipt is locally validated against canonical external
    /// references; the Rust side has not independently parsed the chain manifest.
    LocallyValidatedDeclaredChainMappingV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundReferencePhononChainV1 {
    pub benchmark: BoundReferenceCrystalBenchmarkProtocolV1,
    pub chain_manifest_evidence_id: String,
    pub chain_manifest_sha256: Sha256Digest,
    pub phonon_preflight_evidence_id: String,
    pub phonon_preflight_sha256: Sha256Digest,
    pub execution_evidence_ids: [String; 4],
    pub execution_sha256: [Sha256Digest; 4],
    pub ph_qpoints_evidence_id: String,
    pub ph_qpoints_sha256: Sha256Digest,
    pub matdyn_sampling_evidence_id: String,
    pub matdyn_sampling_sha256: Sha256Digest,
    pub force_constants_evidence_id: String,
    pub force_constants_sha256: Sha256Digest,
    pub phonon_bands_evidence_id: String,
    pub phonon_bands_sha256: Sha256Digest,
    pub reciprocal_grid: [u32; 3],
    pub reciprocal_shift_fractional_bits: [u64; 3],
    pub acoustic_sum_rule: AcousticSumRuleTreatment,
    pub finalization: ReferencePhononFinalizationV1,
    pub authority: ReferenceBenchmarkAuthorityV1,
    pub thermodynamic_phase_eligibility: ThermodynamicPhaseEligibilityV1,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub binding_interpretation: ReferencePhononBindingInterpretationV1,
}

pub fn bind_reference_phonon_chain_v1(
    benchmark: BoundReferenceCrystalBenchmarkProtocolV1,
    receipt: ReferencePhononChainReceiptV1,
    bundle: &ExternalEvidenceBundle,
) -> Result<BoundReferencePhononChainV1, ReferencePhononBindingError> {
    validate_receipt_shape(&receipt)?;
    validate_protocol_match(&benchmark, &receipt)?;

    let chain = bundle.require_role(
        &receipt.chain_manifest_evidence_id,
        EvidenceRole::ArtifactContent,
    )?;
    let preflight = bundle.require_role(
        &receipt.phonon_preflight_evidence_id,
        EvidenceRole::ImplementationSnapshot,
    )?;

    let execution_ids = [
        receipt.pw_execution_evidence_id.as_str(),
        receipt.ph_execution_evidence_id.as_str(),
        receipt.q2r_execution_evidence_id.as_str(),
        receipt.matdyn_execution_evidence_id.as_str(),
    ];
    let mut execution_refs = Vec::with_capacity(4);
    for id in execution_ids {
        execution_refs.push(bundle.require_role(id, EvidenceRole::SolverExecution)?);
    }
    require_unique_content(
        execution_refs
            .iter()
            .map(|reference| reference.content_sha256.as_str()),
        ReferencePhononBindingError::ExecutionContentAlias,
    )?;

    let ph_qpoints = bundle.require_role(
        &receipt.ph_qpoints_evidence_id,
        EvidenceRole::ArtifactContent,
    )?;
    let matdyn_sampling = bundle.require_role(
        &receipt.matdyn_sampling_evidence_id,
        EvidenceRole::ArtifactContent,
    )?;
    let force_constants = bundle.require_role(
        &receipt.force_constants_evidence_id,
        EvidenceRole::ArtifactContent,
    )?;
    let bands = bundle.require_role(
        &receipt.phonon_bands_evidence_id,
        EvidenceRole::ArtifactContent,
    )?;

    require_unique_content(
        [
            chain.content_sha256.as_str(),
            preflight.content_sha256.as_str(),
            ph_qpoints.content_sha256.as_str(),
            matdyn_sampling.content_sha256.as_str(),
            force_constants.content_sha256.as_str(),
            bands.content_sha256.as_str(),
        ],
        ReferencePhononBindingError::SemanticArtifactContentAlias,
    )?;

    Ok(BoundReferencePhononChainV1 {
        authority: benchmark.authority,
        thermodynamic_phase_eligibility: benchmark.thermodynamic_phase_eligibility,
        benchmark,
        chain_manifest_evidence_id: chain.id.as_str().to_string(),
        chain_manifest_sha256: chain.content_sha256.clone(),
        phonon_preflight_evidence_id: preflight.id.as_str().to_string(),
        phonon_preflight_sha256: preflight.content_sha256.clone(),
        execution_evidence_ids: [
            execution_refs[0].id.as_str().to_string(),
            execution_refs[1].id.as_str().to_string(),
            execution_refs[2].id.as_str().to_string(),
            execution_refs[3].id.as_str().to_string(),
        ],
        execution_sha256: [
            execution_refs[0].content_sha256.clone(),
            execution_refs[1].content_sha256.clone(),
            execution_refs[2].content_sha256.clone(),
            execution_refs[3].content_sha256.clone(),
        ],
        ph_qpoints_evidence_id: ph_qpoints.id.as_str().to_string(),
        ph_qpoints_sha256: ph_qpoints.content_sha256.clone(),
        matdyn_sampling_evidence_id: matdyn_sampling.id.as_str().to_string(),
        matdyn_sampling_sha256: matdyn_sampling.content_sha256.clone(),
        force_constants_evidence_id: force_constants.id.as_str().to_string(),
        force_constants_sha256: force_constants.content_sha256.clone(),
        phonon_bands_evidence_id: bands.id.as_str().to_string(),
        phonon_bands_sha256: bands.content_sha256.clone(),
        reciprocal_grid: receipt.ph_reciprocal_grid,
        reciprocal_shift_fractional_bits: receipt.ph_reciprocal_shift_fractional_bits,
        acoustic_sum_rule: receipt.matdyn_acoustic_sum_rule,
        finalization: ReferencePhononFinalizationV1::ExplicitPostprocessingChainBound,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        binding_interpretation:
            ReferencePhononBindingInterpretationV1::LocallyValidatedDeclaredChainMappingV1,
    })
}

fn validate_protocol_match(
    benchmark: &BoundReferenceCrystalBenchmarkProtocolV1,
    receipt: &ReferencePhononChainReceiptV1,
) -> Result<(), ReferencePhononBindingError> {
    if benchmark.phonon_finalization
        != PhononFinalizationPolicyV1::ExplicitPostprocessingExecutionRequired
    {
        return Err(ReferencePhononBindingError::UnsupportedBenchmarkFinalizationPolicy);
    }
    if receipt.ph_reciprocal_grid != benchmark.protocol.numerical.phonon_q_grid {
        return Err(ReferencePhononBindingError::ReciprocalGridMismatch);
    }
    if receipt.ph_reciprocal_shift_fractional_bits
        != benchmark.protocol.numerical.phonon_q_shift_fractional_bits
    {
        return Err(ReferencePhononBindingError::ReciprocalShiftMismatch);
    }
    if receipt.matdyn_acoustic_sum_rule != benchmark.protocol.numerical.acoustic_sum_rule {
        return Err(ReferencePhononBindingError::AcousticSumRuleMismatch);
    }
    Ok(())
}

fn validate_receipt_shape(
    receipt: &ReferencePhononChainReceiptV1,
) -> Result<(), ReferencePhononBindingError> {
    let ids = [
        receipt.chain_manifest_evidence_id.as_str(),
        receipt.phonon_preflight_evidence_id.as_str(),
        receipt.pw_execution_evidence_id.as_str(),
        receipt.ph_execution_evidence_id.as_str(),
        receipt.q2r_execution_evidence_id.as_str(),
        receipt.matdyn_execution_evidence_id.as_str(),
        receipt.ph_qpoints_evidence_id.as_str(),
        receipt.matdyn_sampling_evidence_id.as_str(),
        receipt.force_constants_evidence_id.as_str(),
        receipt.phonon_bands_evidence_id.as_str(),
    ];
    for id in ids {
        if id.trim().is_empty() {
            return Err(ReferencePhononBindingError::EmptyEvidenceId);
        }
        if id.chars().any(char::is_control) {
            return Err(ReferencePhononBindingError::ControlCharacterInEvidenceId);
        }
    }
    if ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err(ReferencePhononBindingError::DuplicateSemanticEvidenceId);
    }
    if receipt.ph_reciprocal_grid.iter().any(|&dimension| dimension == 0) {
        return Err(ReferencePhononBindingError::ZeroReciprocalGridDimension);
    }
    for bits in receipt.ph_reciprocal_shift_fractional_bits {
        let value = f64::from_bits(bits);
        if !value.is_finite() || !(0.0..1.0).contains(&value) {
            return Err(ReferencePhononBindingError::InvalidReciprocalShift);
        }
    }
    Ok(())
}

fn require_unique_content<'a, I>(
    values: I,
    error: ReferencePhononBindingError,
) -> Result<(), ReferencePhononBindingError>
where
    I: IntoIterator<Item = &'a str>,
{
    let values: Vec<_> = values.into_iter().collect();
    if values.iter().copied().collect::<BTreeSet<_>>().len() != values.len() {
        return Err(error);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferencePhononBindingError {
    ExternalEvidence(ExternalEvidenceError),
    EmptyEvidenceId,
    ControlCharacterInEvidenceId,
    DuplicateSemanticEvidenceId,
    ZeroReciprocalGridDimension,
    InvalidReciprocalShift,
    UnsupportedBenchmarkFinalizationPolicy,
    ReciprocalGridMismatch,
    ReciprocalShiftMismatch,
    AcousticSumRuleMismatch,
    ExecutionContentAlias,
    SemanticArtifactContentAlias,
}

impl From<ExternalEvidenceError> for ReferencePhononBindingError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for ReferencePhononBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "reference phonon chain rejected: {self:?}")
    }
}

impl std::error::Error for ReferencePhononBindingError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt() -> ReferencePhononChainReceiptV1 {
        ReferencePhononChainReceiptV1 {
            chain_manifest_evidence_id: "chain".into(),
            phonon_preflight_evidence_id: "preflight".into(),
            pw_execution_evidence_id: "pw".into(),
            ph_execution_evidence_id: "ph".into(),
            q2r_execution_evidence_id: "q2r".into(),
            matdyn_execution_evidence_id: "matdyn".into(),
            ph_qpoints_evidence_id: "phq".into(),
            matdyn_sampling_evidence_id: "matq".into(),
            force_constants_evidence_id: "fc".into(),
            phonon_bands_evidence_id: "bands".into(),
            ph_reciprocal_grid: [4, 4, 4],
            ph_reciprocal_shift_fractional_bits: [0.0_f64.to_bits(); 3],
            matdyn_acoustic_sum_rule: AcousticSumRuleTreatment::Translational,
        }
    }

    #[test]
    fn receipt_rejects_duplicate_semantic_ids() {
        let mut value = receipt();
        value.q2r_execution_evidence_id = value.ph_execution_evidence_id.clone();
        assert_eq!(
            validate_receipt_shape(&value).unwrap_err(),
            ReferencePhononBindingError::DuplicateSemanticEvidenceId
        );
    }

    #[test]
    fn receipt_rejects_nonfinite_shift() {
        let mut value = receipt();
        value.ph_reciprocal_shift_fractional_bits[1] = f64::NAN.to_bits();
        assert_eq!(
            validate_receipt_shape(&value).unwrap_err(),
            ReferencePhononBindingError::InvalidReciprocalShift
        );
    }
}
