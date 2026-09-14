// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Candidate-specific epistemic-grounding measurements for the V3 reasoning planner.
//!
//! This adapter is deliberately basis-driven rather than score-driven. Callers cannot submit an
//! arbitrary `epistemic_grounding = 0.82`; they must provide typed empirical, normative, and
//! materiality evidence bases bound to the exact active primitive encoding. The adapter derives
//! the E/N/M coordinate and normalized grounding score from those bases.
//!
//! Structural primitive facts (registry membership, base/derived status, derivation completeness,
//! activation strength, or global cognitive confidence) never upgrade these evidence bases by
//! themselves. Strong E/N/M variants are only admissible when a measurement names a producer
//! qualification and the corresponding evidence references. This module validates the contract;
//! it does not independently establish that an external producer deserves those strong variants.

use super::reasoning_active_primitive_evidence::{
    adapt_active_primitive_evidence, ActivePrimitiveEvidenceError, ActivePrimitiveEvidenceReport,
};
use super::reasoning_context_competition::{ContextCompetitionPolicy, ContextHypothesis};
use super::reasoning_evidence_seeking::{
    plan_with_evidence, EvidenceSeekingPlanReport, EvidenceSeekingPlannerError,
};
use super::reasoning_objective_evidence::{
    CandidateObjectiveEvidence, ObjectiveEvidence, ObjectiveEvidenceError, ObjectiveEvidenceStatus,
};
use crate::consciousness::epistemic_tiers::{
    EmpiricalTier, EpistemicCoordinate, MaterialityTier, NormativeTier,
};
use crate::consciousness::ActivePrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

pub const EPISTEMIC_GROUNDING_MEASUREMENT_VERSION: &str =
    "rq-006v-candidate-epistemic-grounding-v1";
const COMMITMENT_DOMAIN: &[u8] = b"symthaea/reasoning/epistemic-grounding-measurement/v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmpiricalGroundingBasis {
    /// A bounded assessment was performed and found no empirical evidence for this candidate.
    AssessedAbsent { assessment_ref: String },
    /// One concrete observation supports the candidate.
    SingleObservation { observation_ref: String },
    /// At least two distinct internal observations support the candidate.
    RepeatedInternal { observation_refs: Vec<String> },
    /// A proof artifact and a distinct verification artifact support the candidate.
    CryptographicVerification {
        proof_ref: String,
        verifier_ref: String,
    },
    /// Publicly inspectable data + code + at least one reproduction record support the candidate.
    PublicReproduction {
        data_ref: String,
        code_ref: String,
        reproduction_refs: Vec<String>,
    },
}

impl EmpiricalGroundingBasis {
    pub const fn tier(&self) -> EmpiricalTier {
        match self {
            Self::AssessedAbsent { .. } => EmpiricalTier::E0Null,
            Self::SingleObservation { .. } => EmpiricalTier::E1Testimonial,
            Self::RepeatedInternal { .. } => EmpiricalTier::E2PrivatelyVerifiable,
            Self::CryptographicVerification { .. } => EmpiricalTier::E3CryptographicallyProven,
            Self::PublicReproduction { .. } => EmpiricalTier::E4PubliclyReproducible,
        }
    }

    fn raw_refs(&self) -> Vec<&str> {
        match self {
            Self::AssessedAbsent { assessment_ref } => vec![assessment_ref],
            Self::SingleObservation { observation_ref } => vec![observation_ref],
            Self::RepeatedInternal { observation_refs } => {
                observation_refs.iter().map(String::as_str).collect()
            }
            Self::CryptographicVerification {
                proof_ref,
                verifier_ref,
            } => vec![proof_ref, verifier_ref],
            Self::PublicReproduction {
                data_ref,
                code_ref,
                reproduction_refs,
            } => {
                let mut refs = Vec::with_capacity(2 + reproduction_refs.len());
                refs.push(data_ref.as_str());
                refs.push(code_ref.as_str());
                refs.extend(reproduction_refs.iter().map(String::as_str));
                refs
            }
        }
    }

    fn validate(&self) -> Result<(), EpistemicGroundingMeasurementError> {
        match self {
            Self::AssessedAbsent { assessment_ref } => {
                require_nonempty("empirical.assessment_ref", assessment_ref)?;
            }
            Self::SingleObservation { observation_ref } => {
                require_nonempty("empirical.observation_ref", observation_ref)?;
            }
            Self::RepeatedInternal { observation_refs } => {
                if observation_refs.len() < 2 {
                    return Err(EpistemicGroundingMeasurementError::InsufficientEvidence {
                        basis: "empirical.repeated_internal",
                        minimum: 2,
                        found: observation_refs.len(),
                    });
                }
                validate_ref_set("empirical.repeated_internal", observation_refs.iter().map(String::as_str))?;
            }
            Self::CryptographicVerification {
                proof_ref,
                verifier_ref,
            } => {
                validate_ref_set(
                    "empirical.cryptographic_verification",
                    [proof_ref.as_str(), verifier_ref.as_str()],
                )?;
            }
            Self::PublicReproduction {
                data_ref,
                code_ref,
                reproduction_refs,
            } => {
                if reproduction_refs.is_empty() {
                    return Err(EpistemicGroundingMeasurementError::InsufficientEvidence {
                        basis: "empirical.public_reproduction.reproduction_refs",
                        minimum: 1,
                        found: 0,
                    });
                }
                let refs = std::iter::once(data_ref.as_str())
                    .chain(std::iter::once(code_ref.as_str()))
                    .chain(reproduction_refs.iter().map(String::as_str));
                validate_ref_set("empirical.public_reproduction", refs)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormativeGroundingBasis {
    /// This system/source alone classifies the candidate.
    PersonalAssessment { assessment_ref: String },
    /// At least two distinct corroboration records from a local/community scope.
    CommunalCorroboration { corroboration_refs: Vec<String> },
    /// At least three distinct corroboration records plus an explicit network-coverage record.
    NetworkCorroboration {
        corroboration_refs: Vec<String>,
        coverage_ref: String,
    },
    /// A formal basis and separate verification record establish axiomatic status.
    Axiomatic {
        formal_basis_ref: String,
        verifier_ref: String,
    },
}

impl NormativeGroundingBasis {
    pub const fn tier(&self) -> NormativeTier {
        match self {
            Self::PersonalAssessment { .. } => NormativeTier::N0Personal,
            Self::CommunalCorroboration { .. } => NormativeTier::N1Communal,
            Self::NetworkCorroboration { .. } => NormativeTier::N2Network,
            Self::Axiomatic { .. } => NormativeTier::N3Axiomatic,
        }
    }

    fn raw_refs(&self) -> Vec<&str> {
        match self {
            Self::PersonalAssessment { assessment_ref } => vec![assessment_ref],
            Self::CommunalCorroboration { corroboration_refs } => {
                corroboration_refs.iter().map(String::as_str).collect()
            }
            Self::NetworkCorroboration {
                corroboration_refs,
                coverage_ref,
            } => {
                let mut refs = Vec::with_capacity(corroboration_refs.len() + 1);
                refs.extend(corroboration_refs.iter().map(String::as_str));
                refs.push(coverage_ref.as_str());
                refs
            }
            Self::Axiomatic {
                formal_basis_ref,
                verifier_ref,
            } => vec![formal_basis_ref, verifier_ref],
        }
    }

    fn validate(&self) -> Result<(), EpistemicGroundingMeasurementError> {
        match self {
            Self::PersonalAssessment { assessment_ref } => {
                require_nonempty("normative.assessment_ref", assessment_ref)?;
            }
            Self::CommunalCorroboration { corroboration_refs } => {
                if corroboration_refs.len() < 2 {
                    return Err(EpistemicGroundingMeasurementError::InsufficientEvidence {
                        basis: "normative.communal_corroboration",
                        minimum: 2,
                        found: corroboration_refs.len(),
                    });
                }
                validate_ref_set(
                    "normative.communal_corroboration",
                    corroboration_refs.iter().map(String::as_str),
                )?;
            }
            Self::NetworkCorroboration {
                corroboration_refs,
                coverage_ref,
            } => {
                if corroboration_refs.len() < 3 {
                    return Err(EpistemicGroundingMeasurementError::InsufficientEvidence {
                        basis: "normative.network_corroboration",
                        minimum: 3,
                        found: corroboration_refs.len(),
                    });
                }
                let refs = corroboration_refs
                    .iter()
                    .map(String::as_str)
                    .chain(std::iter::once(coverage_ref.as_str()));
                validate_ref_set("normative.network_corroboration", refs)?;
            }
            Self::Axiomatic {
                formal_basis_ref,
                verifier_ref,
            } => {
                validate_ref_set(
                    "normative.axiomatic",
                    [formal_basis_ref.as_str(), verifier_ref.as_str()],
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialityGroundingBasis {
    /// Evidence is intentionally scoped only to the current reasoning session.
    Ephemeral { assessment_ref: String },
    /// Evidence is bound to one exact model/code/data version.
    VersionBound { version_ref: String },
    /// Evidence is stored in a persistent archive with a content digest.
    PersistentArchive {
        archive_ref: String,
        content_digest: String,
    },
    /// A formal foundational basis plus an explicit stability/governance record.
    Foundational {
        formal_basis_ref: String,
        stability_ref: String,
    },
}

impl MaterialityGroundingBasis {
    pub const fn tier(&self) -> MaterialityTier {
        match self {
            Self::Ephemeral { .. } => MaterialityTier::M0Ephemeral,
            Self::VersionBound { .. } => MaterialityTier::M1Temporal,
            Self::PersistentArchive { .. } => MaterialityTier::M2Persistent,
            Self::Foundational { .. } => MaterialityTier::M3Foundational,
        }
    }

    fn raw_refs(&self) -> Vec<&str> {
        match self {
            Self::Ephemeral { assessment_ref } => vec![assessment_ref],
            Self::VersionBound { version_ref } => vec![version_ref],
            Self::PersistentArchive { archive_ref, .. } => vec![archive_ref],
            Self::Foundational {
                formal_basis_ref,
                stability_ref,
            } => vec![formal_basis_ref, stability_ref],
        }
    }

    fn validate(&self) -> Result<(), EpistemicGroundingMeasurementError> {
        match self {
            Self::Ephemeral { assessment_ref } => {
                require_nonempty("materiality.assessment_ref", assessment_ref)?;
            }
            Self::VersionBound { version_ref } => {
                require_nonempty("materiality.version_ref", version_ref)?;
            }
            Self::PersistentArchive {
                archive_ref,
                content_digest,
            } => {
                require_nonempty("materiality.archive_ref", archive_ref)?;
                require_nonempty("materiality.content_digest", content_digest)?;
            }
            Self::Foundational {
                formal_basis_ref,
                stability_ref,
            } => {
                validate_ref_set(
                    "materiality.foundational",
                    [formal_basis_ref.as_str(), stability_ref.as_str()],
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateEpistemicGroundingMeasurement {
    pub candidate_id: String,
    /// Exact BLAKE3 digest of the 16,384-bit primitive encoding measured by this record.
    pub primitive_encoding_digest: String,
    /// Identity of the producer/adapter that emitted this measurement.
    pub measuring_source: String,
    /// Evidence proving this producer is qualified to emit the declared basis semantics.
    pub producer_qualification_ref: String,
    pub empirical_basis: EmpiricalGroundingBasis,
    pub normative_basis: NormativeGroundingBasis,
    pub materiality_basis: MaterialityGroundingBasis,
}

impl CandidateEpistemicGroundingMeasurement {
    pub fn coordinate(&self) -> EpistemicCoordinate {
        EpistemicCoordinate::new(
            self.empirical_basis.tier(),
            self.normative_basis.tier(),
            self.materiality_basis.tier(),
        )
    }

    pub fn score(&self) -> f64 {
        self.coordinate().quality_score()
    }

    pub fn commitment(&self) -> Result<String, EpistemicGroundingMeasurementError> {
        let serialized = serde_json::to_vec(self).map_err(|err| {
            EpistemicGroundingMeasurementError::Serialization(err.to_string())
        })?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(COMMITMENT_DOMAIN);
        hasher.update(&(serialized.len() as u64).to_le_bytes());
        hasher.update(&serialized);
        Ok(hasher.finalize().to_hex().to_string())
    }

    fn raw_basis_refs(&self) -> Vec<&str> {
        let empirical = self.empirical_basis.raw_refs();
        let normative = self.normative_basis.raw_refs();
        let materiality = self.materiality_basis.raw_refs();
        let mut refs = Vec::with_capacity(empirical.len() + normative.len() + materiality.len());
        refs.extend(empirical);
        refs.extend(normative);
        refs.extend(materiality);
        refs
    }

    fn validate(&self) -> Result<(), EpistemicGroundingMeasurementError> {
        require_nonempty("measurement.candidate_id", &self.candidate_id)?;
        require_nonempty(
            "measurement.primitive_encoding_digest",
            &self.primitive_encoding_digest,
        )?;
        require_nonempty("measurement.measuring_source", &self.measuring_source)?;
        require_nonempty(
            "measurement.producer_qualification_ref",
            &self.producer_qualification_ref,
        )?;
        self.empirical_basis.validate()?;
        self.normative_basis.validate()?;
        self.materiality_basis.validate()?;

        // One artifact may legitimately be relevant to several axes, but counting the same ref
        // three times would hide a common-mode dependency. Strong cross-axis reuse therefore fails
        // closed in this first adapter; later dependence-aware accounting may relax this explicitly.
        validate_ref_set("measurement.cross_axis_basis_refs", self.raw_basis_refs())?;
        Ok(())
    }

    fn objective_evidence(
        &self,
    ) -> Result<(String, ObjectiveEvidence), EpistemicGroundingMeasurementError> {
        self.validate()?;
        let commitment = self.commitment()?;
        let mut evidence_refs = Vec::new();
        evidence_refs.push(self.producer_qualification_ref.clone());
        evidence_refs.extend(self.raw_basis_refs().into_iter().map(str::to_owned));
        if let MaterialityGroundingBasis::PersistentArchive { content_digest, .. } =
            &self.materiality_basis
        {
            evidence_refs.push(format!("archive-content:{content_digest}"));
        }
        evidence_refs.push(format!(
            "candidate-encoding:{}",
            self.primitive_encoding_digest
        ));
        evidence_refs.push(format!("measurement:{commitment}"));
        validate_ref_set(
            "measurement.objective_evidence_refs",
            evidence_refs.iter().map(String::as_str),
        )?;

        let source = format!(
            "{}:{}",
            EPISTEMIC_GROUNDING_MEASUREMENT_VERSION, self.measuring_source
        );
        let evidence = ObjectiveEvidence::observed(source, evidence_refs, self.score())?;
        Ok((commitment, evidence))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppliedEpistemicGroundingMeasurement {
    pub candidate_id: String,
    pub primitive_encoding_digest: String,
    pub measuring_source: String,
    pub measurement_commitment: String,
    pub coordinate: EpistemicCoordinate,
    pub score: f64,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpistemicGroundingApplicationReport {
    pub adapter_version: String,
    /// Candidate evidence in the exact same order as the active-primitive evidence report.
    pub candidates: Vec<CandidateObjectiveEvidence>,
    pub applied_measurements: Vec<AppliedEpistemicGroundingMeasurement>,
    pub unmeasured_candidate_ids: Vec<String>,
    /// Raw basis refs reused across more than one candidate measurement. These are common-mode
    /// dependencies, not independent corroboration.
    pub shared_basis_evidence_refs: Vec<String>,
}

/// Apply candidate-specific epistemic measurements without touching any other objective axis.
pub fn apply_epistemic_grounding_measurements(
    active_report: &ActivePrimitiveEvidenceReport,
    measurements: &[CandidateEpistemicGroundingMeasurement],
) -> Result<EpistemicGroundingApplicationReport, EpistemicGroundingMeasurementError> {
    if active_report.profiles.is_empty() {
        return Err(EpistemicGroundingMeasurementError::EmptyActiveReport);
    }

    let mut profile_by_id = HashMap::with_capacity(active_report.profiles.len());
    for profile in &active_report.profiles {
        require_nonempty("active_profile.candidate_id", &profile.candidate_id)?;
        if profile_by_id
            .insert(profile.candidate_id.as_str(), profile)
            .is_some()
        {
            return Err(EpistemicGroundingMeasurementError::DuplicateActiveCandidate(
                profile.candidate_id.clone(),
            ));
        }
    }

    let mut measurement_by_id = HashMap::with_capacity(measurements.len());
    let mut basis_ref_counts: HashMap<String, usize> = HashMap::new();
    for measurement in measurements {
        measurement.validate()?;
        if measurement_by_id
            .insert(measurement.candidate_id.as_str(), measurement)
            .is_some()
        {
            return Err(EpistemicGroundingMeasurementError::DuplicateMeasurement(
                measurement.candidate_id.clone(),
            ));
        }
        if !profile_by_id.contains_key(measurement.candidate_id.as_str()) {
            return Err(EpistemicGroundingMeasurementError::UnknownCandidate(
                measurement.candidate_id.clone(),
            ));
        }
        for evidence_ref in measurement.raw_basis_refs() {
            *basis_ref_counts.entry(evidence_ref.to_owned()).or_default() += 1;
        }
    }

    let mut candidates = Vec::with_capacity(active_report.profiles.len());
    let mut applied_measurements = Vec::with_capacity(measurements.len());
    let mut unmeasured_candidate_ids = Vec::new();

    for profile in &active_report.profiles {
        let mut candidate = profile.objective_evidence.clone();
        candidate.validate()?;
        if let Some(measurement) = measurement_by_id.get(profile.candidate_id.as_str()) {
            if profile.observed_primitive.encoding_digest != measurement.primitive_encoding_digest {
                return Err(EpistemicGroundingMeasurementError::EncodingDigestMismatch {
                    candidate_id: profile.candidate_id.clone(),
                    expected: profile.observed_primitive.encoding_digest.clone(),
                    found: measurement.primitive_encoding_digest.clone(),
                });
            }
            if matches!(
                candidate.epistemic_grounding.status,
                ObjectiveEvidenceStatus::Observed { .. }
            ) {
                return Err(EpistemicGroundingMeasurementError::AxisAlreadyObserved(
                    profile.candidate_id.clone(),
                ));
            }

            let integration_before = candidate.integration_proxy.clone();
            let harmonic_before = candidate.harmonic_alignment.clone();
            let (commitment, objective_evidence) = measurement.objective_evidence()?;
            candidate.epistemic_grounding = objective_evidence.clone();
            candidate.validate()?;

            // This adapter is authority-limited to one axis. Treat accidental cross-axis mutation
            // as an internal contract violation rather than accepting it silently.
            if candidate.integration_proxy != integration_before
                || candidate.harmonic_alignment != harmonic_before
            {
                return Err(EpistemicGroundingMeasurementError::CrossAxisMutation(
                    profile.candidate_id.clone(),
                ));
            }

            applied_measurements.push(AppliedEpistemicGroundingMeasurement {
                candidate_id: profile.candidate_id.clone(),
                primitive_encoding_digest: measurement.primitive_encoding_digest.clone(),
                measuring_source: measurement.measuring_source.clone(),
                measurement_commitment: commitment,
                coordinate: measurement.coordinate(),
                score: measurement.score(),
                evidence_refs: objective_evidence.evidence_refs,
            });
        } else {
            unmeasured_candidate_ids.push(profile.candidate_id.clone());
        }
        candidates.push(candidate);
    }

    let mut shared_basis_evidence_refs = basis_ref_counts
        .into_iter()
        .filter_map(|(evidence_ref, count)| (count > 1).then_some(evidence_ref))
        .collect::<Vec<_>>();
    shared_basis_evidence_refs.sort();

    Ok(EpistemicGroundingApplicationReport {
        adapter_version: EPISTEMIC_GROUNDING_MEASUREMENT_VERSION.into(),
        candidates,
        applied_measurements,
        unmeasured_candidate_ids,
        shared_basis_evidence_refs,
    })
}

/// Compose actual live primitive identity + typed epistemic measurements + V3 replanning.
pub fn plan_active_primitive_with_epistemic_grounding(
    hypotheses: &[ContextHypothesis],
    context_policy: ContextCompetitionPolicy,
    active: &[ActivePrimitive],
    candidate_ids: &[String],
    measurements: &[CandidateEpistemicGroundingMeasurement],
) -> Result<
    (
        ActivePrimitiveEvidenceReport,
        EpistemicGroundingApplicationReport,
        EvidenceSeekingPlanReport,
    ),
    EpistemicGroundingMeasurementError,
> {
    let active_report = adapt_active_primitive_evidence(active, candidate_ids)?;
    let application = apply_epistemic_grounding_measurements(&active_report, measurements)?;
    let plan = plan_with_evidence(hypotheses, context_policy, &application.candidates)?;
    Ok((active_report, application, plan))
}

fn require_nonempty(
    field: &'static str,
    value: &str,
) -> Result<(), EpistemicGroundingMeasurementError> {
    if value.trim().is_empty() {
        Err(EpistemicGroundingMeasurementError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_ref_set<'a, I>(
    basis: &'static str,
    refs: I,
) -> Result<(), EpistemicGroundingMeasurementError>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut seen = HashSet::new();
    for evidence_ref in refs {
        if evidence_ref.trim().is_empty() {
            return Err(EpistemicGroundingMeasurementError::EmptyEvidenceRef(basis));
        }
        if !seen.insert(evidence_ref) {
            return Err(EpistemicGroundingMeasurementError::DuplicateEvidenceRef {
                basis,
                evidence_ref: evidence_ref.to_owned(),
            });
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum EpistemicGroundingMeasurementError {
    EmptyField(&'static str),
    EmptyEvidenceRef(&'static str),
    DuplicateEvidenceRef {
        basis: &'static str,
        evidence_ref: String,
    },
    InsufficientEvidence {
        basis: &'static str,
        minimum: usize,
        found: usize,
    },
    EmptyActiveReport,
    DuplicateActiveCandidate(String),
    DuplicateMeasurement(String),
    UnknownCandidate(String),
    EncodingDigestMismatch {
        candidate_id: String,
        expected: String,
        found: String,
    },
    AxisAlreadyObserved(String),
    CrossAxisMutation(String),
    Serialization(String),
    ActivePrimitive(ActivePrimitiveEvidenceError),
    Objective(ObjectiveEvidenceError),
    Planner(EvidenceSeekingPlannerError),
}

impl fmt::Display for EpistemicGroundingMeasurementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::EmptyEvidenceRef(basis) => {
                write!(f, "evidence basis `{basis}` contains an empty reference")
            }
            Self::DuplicateEvidenceRef {
                basis,
                evidence_ref,
            } => write!(
                f,
                "evidence basis `{basis}` reuses reference `{evidence_ref}` where distinct support is required"
            ),
            Self::InsufficientEvidence {
                basis,
                minimum,
                found,
            } => write!(
                f,
                "evidence basis `{basis}` requires at least {minimum} references, found {found}"
            ),
            Self::EmptyActiveReport => write!(f, "epistemic grounding requires active candidates"),
            Self::DuplicateActiveCandidate(id) => {
                write!(f, "active evidence report contains duplicate candidate `{id}`")
            }
            Self::DuplicateMeasurement(id) => {
                write!(f, "candidate `{id}` has more than one epistemic measurement")
            }
            Self::UnknownCandidate(id) => {
                write!(f, "epistemic measurement targets unknown candidate `{id}`")
            }
            Self::EncodingDigestMismatch {
                candidate_id,
                expected,
                found,
            } => write!(
                f,
                "candidate `{candidate_id}` epistemic measurement binds encoding `{found}`, expected `{expected}`"
            ),
            Self::AxisAlreadyObserved(id) => write!(
                f,
                "candidate `{id}` already has observed epistemic grounding; this adapter will not overwrite it"
            ),
            Self::CrossAxisMutation(id) => write!(
                f,
                "epistemic adapter attempted to mutate a non-epistemic objective for candidate `{id}`"
            ),
            Self::Serialization(err) => write!(f, "failed to serialize epistemic measurement: {err}"),
            Self::ActivePrimitive(err) => write!(f, "active primitive evidence error: {err}"),
            Self::Objective(err) => write!(f, "objective evidence error: {err}"),
            Self::Planner(err) => write!(f, "V3 planning error: {err}"),
        }
    }
}

impl std::error::Error for EpistemicGroundingMeasurementError {}

impl From<ActivePrimitiveEvidenceError> for EpistemicGroundingMeasurementError {
    fn from(value: ActivePrimitiveEvidenceError) -> Self {
        Self::ActivePrimitive(value)
    }
}

impl From<ObjectiveEvidenceError> for EpistemicGroundingMeasurementError {
    fn from(value: ObjectiveEvidenceError) -> Self {
        Self::Objective(value)
    }
}

impl From<EvidenceSeekingPlannerError> for EpistemicGroundingMeasurementError {
    fn from(value: EvidenceSeekingPlannerError) -> Self {
        Self::Planner(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reasoning_evidence_seeking::EvidenceSeekingOutcome;
    use crate::consciousness::context_aware_evolution::ReasoningContext;
    use crate::consciousness::{ActivationReason, ActivePrimitive};
    use symthaea_core::hdc::primitive_system::PrimitiveSystem;

    fn active(name: &str, activation: f64) -> ActivePrimitive {
        ActivePrimitive {
            primitive: PrimitiveSystem::global()
                .get(name)
                .unwrap_or_else(|| panic!("fixture primitive `{name}` must exist"))
                .clone(),
            activation,
            activation_reason: ActivationReason::BottomUp {
                input_similarity: activation,
            },
            duration: 2,
        }
    }

    fn hypothesis(context: ReasoningContext) -> ContextHypothesis {
        ContextHypothesis {
            context,
            support: 0.9,
            source: "fixture-context".into(),
            evidence_refs: vec!["query".into()],
        }
    }

    fn strong_measurement(
        candidate_id: &str,
        encoding_digest: &str,
    ) -> CandidateEpistemicGroundingMeasurement {
        CandidateEpistemicGroundingMeasurement {
            candidate_id: candidate_id.into(),
            primitive_encoding_digest: encoding_digest.into(),
            measuring_source: "fixture-qualified-producer".into(),
            producer_qualification_ref: format!("qualification:{candidate_id}"),
            empirical_basis: EmpiricalGroundingBasis::PublicReproduction {
                data_ref: format!("data:{candidate_id}"),
                code_ref: format!("code:{candidate_id}"),
                reproduction_refs: vec![format!("reproduction:{candidate_id}")],
            },
            normative_basis: NormativeGroundingBasis::Axiomatic {
                formal_basis_ref: format!("formal:{candidate_id}"),
                verifier_ref: format!("formal-verifier:{candidate_id}"),
            },
            materiality_basis: MaterialityGroundingBasis::Foundational {
                formal_basis_ref: format!("foundation:{candidate_id}"),
                stability_ref: format!("stability:{candidate_id}"),
            },
        }
    }

    fn weak_measurement(
        candidate_id: &str,
        encoding_digest: &str,
    ) -> CandidateEpistemicGroundingMeasurement {
        CandidateEpistemicGroundingMeasurement {
            candidate_id: candidate_id.into(),
            primitive_encoding_digest: encoding_digest.into(),
            measuring_source: "fixture-qualified-producer".into(),
            producer_qualification_ref: format!("qualification:{candidate_id}"),
            empirical_basis: EmpiricalGroundingBasis::AssessedAbsent {
                assessment_ref: format!("empirical-none:{candidate_id}"),
            },
            normative_basis: NormativeGroundingBasis::PersonalAssessment {
                assessment_ref: format!("personal:{candidate_id}"),
            },
            materiality_basis: MaterialityGroundingBasis::Ephemeral {
                assessment_ref: format!("ephemeral:{candidate_id}"),
            },
        }
    }

    #[test]
    fn structural_metadata_alone_does_not_become_epistemic_grounding() {
        let actives = [active("NSM_KNOW", 0.8)];
        let active_report =
            adapt_active_primitive_evidence(&actives, &["NSM_KNOW".into()]).unwrap();
        let report = apply_epistemic_grounding_measurements(&active_report, &[]).unwrap();
        assert_eq!(report.candidates[0].observed_axes(), 0);
        assert_eq!(report.unmeasured_candidate_ids, vec!["NSM_KNOW"]);
    }

    #[test]
    fn typed_bases_derive_coordinate_and_only_epistemic_axis() {
        let actives = [active("NSM_KNOW", 0.8)];
        let active_report =
            adapt_active_primitive_evidence(&actives, &["NSM_KNOW".into()]).unwrap();
        let digest = active_report.profiles[0]
            .observed_primitive
            .encoding_digest
            .clone();
        let measurement = CandidateEpistemicGroundingMeasurement {
            candidate_id: "NSM_KNOW".into(),
            primitive_encoding_digest: digest,
            measuring_source: "fixture-qualified-producer".into(),
            producer_qualification_ref: "qualification:fixture".into(),
            empirical_basis: EmpiricalGroundingBasis::RepeatedInternal {
                observation_refs: vec!["obs:1".into(), "obs:2".into()],
            },
            normative_basis: NormativeGroundingBasis::CommunalCorroboration {
                corroboration_refs: vec!["community:1".into(), "community:2".into()],
            },
            materiality_basis: MaterialityGroundingBasis::PersistentArchive {
                archive_ref: "archive:1".into(),
                content_digest: "abc123".into(),
            },
        };
        let expected = EpistemicCoordinate::new(
            EmpiricalTier::E2PrivatelyVerifiable,
            NormativeTier::N1Communal,
            MaterialityTier::M2Persistent,
        )
        .quality_score();
        let report =
            apply_epistemic_grounding_measurements(&active_report, &[measurement]).unwrap();
        assert_eq!(report.candidates[0].observed_axes(), 1);
        let ObjectiveEvidenceStatus::Observed { value } =
            report.candidates[0].epistemic_grounding.status
        else {
            panic!("expected observed epistemic grounding");
        };
        assert!((value - expected).abs() < 1.0e-12);
        assert!(!report.candidates[0].integration_proxy.is_observed());
        assert!(!report.candidates[0].harmonic_alignment.is_observed());
    }

    #[test]
    fn repeated_internal_requires_multiple_distinct_observations() {
        let measurement = CandidateEpistemicGroundingMeasurement {
            candidate_id: "candidate".into(),
            primitive_encoding_digest: "digest".into(),
            measuring_source: "producer".into(),
            producer_qualification_ref: "qualification".into(),
            empirical_basis: EmpiricalGroundingBasis::RepeatedInternal {
                observation_refs: vec!["obs:1".into()],
            },
            normative_basis: NormativeGroundingBasis::PersonalAssessment {
                assessment_ref: "personal".into(),
            },
            materiality_basis: MaterialityGroundingBasis::Ephemeral {
                assessment_ref: "ephemeral".into(),
            },
        };
        assert!(matches!(
            measurement.validate(),
            Err(EpistemicGroundingMeasurementError::InsufficientEvidence {
                basis: "empirical.repeated_internal",
                ..
            })
        ));
    }

    #[test]
    fn cross_axis_evidence_reuse_fails_closed() {
        let measurement = CandidateEpistemicGroundingMeasurement {
            candidate_id: "candidate".into(),
            primitive_encoding_digest: "digest".into(),
            measuring_source: "producer".into(),
            producer_qualification_ref: "qualification".into(),
            empirical_basis: EmpiricalGroundingBasis::SingleObservation {
                observation_ref: "same".into(),
            },
            normative_basis: NormativeGroundingBasis::PersonalAssessment {
                assessment_ref: "same".into(),
            },
            materiality_basis: MaterialityGroundingBasis::Ephemeral {
                assessment_ref: "other".into(),
            },
        };
        assert!(matches!(
            measurement.validate(),
            Err(EpistemicGroundingMeasurementError::DuplicateEvidenceRef {
                basis: "measurement.cross_axis_basis_refs",
                ..
            })
        ));
    }

    #[test]
    fn encoding_binding_prevents_measurement_replay_onto_changed_primitive() {
        let actives = [active("NSM_KNOW", 0.8)];
        let active_report =
            adapt_active_primitive_evidence(&actives, &["NSM_KNOW".into()]).unwrap();
        let measurement = strong_measurement("NSM_KNOW", "wrong-digest");
        assert!(matches!(
            apply_epistemic_grounding_measurements(&active_report, &[measurement]),
            Err(EpistemicGroundingMeasurementError::EncodingDigestMismatch { .. })
        ));
    }

    #[test]
    fn shared_candidate_evidence_is_reported_as_common_mode() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.6)];
        let active_report = adapt_active_primitive_evidence(
            &actives,
            &["NSM_KNOW".into(), "NSM_DO".into()],
        )
        .unwrap();
        let mut first = strong_measurement(
            "NSM_KNOW",
            &active_report.profiles[0].observed_primitive.encoding_digest,
        );
        let mut second = strong_measurement(
            "NSM_DO",
            &active_report.profiles[1].observed_primitive.encoding_digest,
        );
        first.empirical_basis = EmpiricalGroundingBasis::SingleObservation {
            observation_ref: "shared-benchmark".into(),
        };
        second.empirical_basis = EmpiricalGroundingBasis::SingleObservation {
            observation_ref: "shared-benchmark".into(),
        };
        let report =
            apply_epistemic_grounding_measurements(&active_report, &[first, second]).unwrap();
        assert_eq!(
            report.shared_basis_evidence_refs,
            vec!["shared-benchmark".to_string()]
        );
    }

    #[test]
    fn one_sided_measurement_stays_underidentified() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.6)];
        let ids = vec!["NSM_KNOW".into(), "NSM_DO".into()];
        let active_report = adapt_active_primitive_evidence(&actives, &ids).unwrap();
        let strong = strong_measurement(
            "NSM_KNOW",
            &active_report.profiles[0].observed_primitive.encoding_digest,
        );
        let (_, _, plan) = plan_active_primitive_with_epistemic_grounding(
            &[hypothesis(ReasoningContext::TechnicalImplementation)],
            ContextCompetitionPolicy::development_v1(),
            &actives,
            &ids,
            &[strong],
        )
        .unwrap();
        assert!(matches!(plan.outcome, EvidenceSeekingOutcome::NeedEvidence { .. }));
    }

    #[test]
    fn sufficiently_separated_epistemic_evidence_can_identify_technical_winner() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.6)];
        let ids = vec!["NSM_KNOW".into(), "NSM_DO".into()];
        let active_report = adapt_active_primitive_evidence(&actives, &ids).unwrap();
        let strong = strong_measurement(
            "NSM_KNOW",
            &active_report.profiles[0].observed_primitive.encoding_digest,
        );
        let weak = weak_measurement(
            "NSM_DO",
            &active_report.profiles[1].observed_primitive.encoding_digest,
        );
        let (_, application, plan) = plan_active_primitive_with_epistemic_grounding(
            &[hypothesis(ReasoningContext::TechnicalImplementation)],
            ContextCompetitionPolicy::development_v1(),
            &actives,
            &ids,
            &[strong, weak],
        )
        .unwrap();
        assert_eq!(application.applied_measurements[0].score, 1.0);
        assert_eq!(application.applied_measurements[1].score, 0.0);
        assert!(matches!(
            plan.outcome,
            EvidenceSeekingOutcome::Selected { ref candidate_id, .. } if candidate_id == "NSM_KNOW"
        ));
    }
}
