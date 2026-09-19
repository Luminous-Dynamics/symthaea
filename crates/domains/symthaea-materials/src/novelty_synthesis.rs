// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conservative prior-art, novelty-status, metastability, and synthesizability records.
//!
//! This module intentionally refuses to collapse distinct questions into a single
//! `new material` flag. Search absence is not novelty, thermodynamic plausibility is
//! not synthesizability, a proposed process is not a synthesis, and none of these
//! imply legal novelty or patentability.

use crate::conditioned_property::PropertyArtifactRef;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const SHA256_HEX_LEN: usize = 64;

/// Kind of prior-art source searched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriorArtSourceKind {
    /// Peer-reviewed/preprint literature index or corpus.
    Literature,
    /// Structured materials database.
    MaterialsDatabase,
    /// Patent database or patent-search service.
    PatentDatabase,
    /// Explicit other source class.
    Other(String),
}

/// How a prior-art hit relates to the target subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriorArtHitClass {
    /// Exact canonical subject or an explicitly equivalent representation.
    ExactSubject,
    /// Nearby composition under the bound neighborhood definition.
    CompositionNeighbor,
    /// Structurally related phase/material.
    StructuralNeighbor,
    /// Related processing/synthesis route.
    ProcessNeighbor,
}

/// Outcome of one bound prior-art source search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriorArtSearchOutcome {
    /// One or more relevant hits were found.
    HitFound {
        /// Stable hit identifiers supplied by the source/search artifact.
        hit_ids: Vec<String>,
        /// Relationship classes represented by the returned hits.
        hit_classes: Vec<PriorArtHitClass>,
    },
    /// Search completed with no matching result under the exact bound query/neighborhood.
    NoHit,
    /// Source could not be searched.
    Unavailable {
        /// Stable/human-readable reason.
        reason: String,
    },
    /// Search ran but coverage was incomplete.
    Incomplete {
        /// Stable/human-readable reason.
        reason: String,
    },
}

/// Reproducible neighborhood definition for related prior art.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriorArtNeighborhood {
    /// Stable neighborhood algorithm/version.
    pub method_id: String,
    /// SHA-256 of exact neighborhood/query-definition artifact.
    pub artifact_sha256: String,
    /// Optional maximum absolute composition difference per element in ppm.
    pub composition_tolerance_ppm: Option<u32>,
    /// Whether structural similarity is part of neighborhood matching.
    pub structure_sensitive: bool,
    /// Whether process/synthesis-route neighbors are included.
    pub process_sensitive: bool,
}

/// One source-bound prior-art search record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriorArtSearchRecord {
    /// Stable source/provider identifier.
    pub source_id: String,
    /// Source class.
    pub source_kind: PriorArtSourceKind,
    /// ISO `YYYY-MM-DD` search date.
    pub searched_on: String,
    /// Bound query manifest.
    pub query_manifest: PropertyArtifactRef,
    /// Bound raw/normalized search result artifact.
    pub result_artifact: PropertyArtifactRef,
    /// License/terms identifier for the searched/imported data.
    pub license_id: String,
    /// Search outcome.
    pub outcome: PriorArtSearchOutcome,
}

/// Conservative novelty-assessment state.
///
/// There is deliberately no automatic `Novel` or `Patentable` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoveltyAssessmentState {
    /// No prior-art search has been performed.
    Unchecked,
    /// Search has begun but coverage is not complete enough for a bounded conclusion.
    SearchInProgress,
    /// Relevant exact/neighbor prior art was found.
    PriorArtFound,
    /// No match was found in the exact set of bound searches; blind spots may remain.
    NoMatchInBoundSearch,
    /// Search evidence suggests distinction but does not establish legal novelty.
    PotentiallyDistinct,
    /// A named expert/legal/scientific review artifact exists.
    ExpertReviewed,
}

/// Complete novelty/prior-art assessment for one exact MAT-007 subject.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoveltyAssessment {
    /// Exact MAT-007 subject identity.
    pub subject_identity: String,
    /// Neighborhood definition used for related-art searches.
    pub neighborhood: PriorArtNeighborhood,
    /// Source searches performed.
    pub searches: Vec<PriorArtSearchRecord>,
    /// Known blind spots, inaccessible sources, language limits, or search limitations.
    pub blind_spots: Vec<String>,
    /// Conservative state.
    pub state: NoveltyAssessmentState,
    /// Optional expert-review artifact required by `ExpertReviewed`.
    pub expert_review: Option<PropertyArtifactRef>,
}

impl NoveltyAssessment {
    /// Validate search provenance and state consistency.
    pub fn validate(&self) -> Result<(), NoveltySynthesisError> {
        nonempty("subject_identity", &self.subject_identity)?;
        nonempty("neighborhood method_id", &self.neighborhood.method_id)?;
        sha256(&self.neighborhood.artifact_sha256)?;
        if self.neighborhood.composition_tolerance_ppm == Some(0) {
            return Err(NoveltySynthesisError::ZeroCompositionTolerance);
        }

        unique_nonempty(&self.blind_spots, "blind_spot")?;
        let mut search_keys = HashSet::new();
        for search in &self.searches {
            validate_search(search)?;
            let key = format!(
                "{}|{}|{}",
                search.source_id,
                search.query_manifest.artifact_sha256.to_ascii_lowercase(),
                search.result_artifact.artifact_sha256.to_ascii_lowercase()
            );
            if !search_keys.insert(key) {
                return Err(NoveltySynthesisError::DuplicatePriorArtSearch);
            }
        }
        if let Some(review) = &self.expert_review {
            artifact(review)?;
        }

        let any_hit = self
            .searches
            .iter()
            .any(|search| matches!(search.outcome, PriorArtSearchOutcome::HitFound { .. }));
        let all_no_hit = !self.searches.is_empty()
            && self
                .searches
                .iter()
                .all(|search| matches!(search.outcome, PriorArtSearchOutcome::NoHit));
        let any_incomplete = self.searches.iter().any(|search| {
            matches!(
                search.outcome,
                PriorArtSearchOutcome::Unavailable { .. } | PriorArtSearchOutcome::Incomplete { .. }
            )
        });

        match self.state {
            NoveltyAssessmentState::Unchecked if !self.searches.is_empty() => {
                Err(NoveltySynthesisError::UncheckedWithSearches)
            }
            NoveltyAssessmentState::Unchecked => Ok(()),
            NoveltyAssessmentState::SearchInProgress if self.searches.is_empty() || any_incomplete => {
                Ok(())
            }
            NoveltyAssessmentState::SearchInProgress => {
                Err(NoveltySynthesisError::SearchInProgressWithoutIncompleteCoverage)
            }
            NoveltyAssessmentState::PriorArtFound if any_hit => Ok(()),
            NoveltyAssessmentState::PriorArtFound => {
                Err(NoveltySynthesisError::PriorArtFoundWithoutHit)
            }
            NoveltyAssessmentState::NoMatchInBoundSearch if all_no_hit => Ok(()),
            NoveltyAssessmentState::NoMatchInBoundSearch => {
                Err(NoveltySynthesisError::NoMatchStateWithoutCompleteNoHitSearch)
            }
            NoveltyAssessmentState::PotentiallyDistinct if all_no_hit => Ok(()),
            NoveltyAssessmentState::PotentiallyDistinct => {
                Err(NoveltySynthesisError::PotentiallyDistinctWithoutCompleteNoHitSearch)
            }
            NoveltyAssessmentState::ExpertReviewed if self.expert_review.is_some() => Ok(()),
            NoveltyAssessmentState::ExpertReviewed => {
                Err(NoveltySynthesisError::ExpertReviewedWithoutArtifact)
            }
        }
    }

    /// Whether at least one exact/compositional/structural/process prior-art hit exists.
    pub fn has_prior_art_hit(&self) -> Result<bool, NoveltySynthesisError> {
        self.validate()?;
        Ok(self
            .searches
            .iter()
            .any(|search| matches!(search.outcome, PriorArtSearchOutcome::HitFound { .. })))
    }
}

/// One numeric/process condition defining a proposed metastable process window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessWindowVariable {
    /// Stable variable key such as `anneal_temperature`, `quench_rate`, or `pressure`.
    pub key: String,
    /// Inclusive lower bound.
    pub lower: f64,
    /// Inclusive upper bound.
    pub upper: f64,
    /// Explicit unit.
    pub unit: String,
}

/// Bound metastable/non-equilibrium processing hypothesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetastableProcessWindow {
    /// Stable route/window identifier.
    pub window_id: String,
    /// Process variables and ranges.
    pub variables: Vec<ProcessWindowVariable>,
    /// Exact calculation/literature/design artifact supporting the window.
    pub support_artifact: PropertyArtifactRef,
}

impl MetastableProcessWindow {
    /// Validate process-window bounds without claiming that synthesis will succeed.
    pub fn validate(&self) -> Result<(), NoveltySynthesisError> {
        nonempty("window_id", &self.window_id)?;
        artifact(&self.support_artifact)?;
        if self.variables.is_empty() {
            return Err(NoveltySynthesisError::EmptyProcessWindow);
        }
        let mut keys = HashSet::new();
        for variable in &self.variables {
            nonempty("process-window key", &variable.key)?;
            nonempty("process-window unit", &variable.unit)?;
            finite("process-window lower", variable.lower)?;
            finite("process-window upper", variable.upper)?;
            if variable.lower > variable.upper {
                return Err(NoveltySynthesisError::InvalidProcessWindowInterval(
                    variable.key.clone(),
                ));
            }
            if !keys.insert(variable.key.as_str()) {
                return Err(NoveltySynthesisError::DuplicateProcessWindowVariable(
                    variable.key.clone(),
                ));
            }
        }
        Ok(())
    }
}

/// Conservative state of synthesis/process-route evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthesizabilityState {
    /// No process route has been proposed.
    NoRouteProposed,
    /// A process route has been proposed but not supported by stronger evidence.
    RouteProposed,
    /// Computation supports a stated process/metastability window.
    ComputationallySupportedProcessWindow,
    /// Related literature demonstrates a comparable process route, not this exact subject.
    RelatedLiteraturePrecedent,
    /// A physical synthesis attempt occurred; target outcome not yet classified.
    AttemptedSynthesis,
    /// A physical synthesis attempt failed.
    SynthesisFailed,
    /// Material was produced but the target phase/architecture was not established.
    MaterialProducedTargetPhaseUnestablished,
    /// Experimental characterization established the intended phase/architecture.
    TargetPhaseEstablished,
}

/// Synthesis/process assessment kept separate from novelty and equilibrium stability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesisRouteAssessment {
    /// Exact MAT-007 subject identity.
    pub subject_identity: String,
    /// Stable route identifier.
    pub route_id: String,
    /// Current conservative synthesis/process state.
    pub state: SynthesizabilityState,
    /// Exact proposed/executed process protocol, when one exists.
    pub process_artifact: Option<PropertyArtifactRef>,
    /// Optional metastable/non-equilibrium process window.
    pub metastable_window: Option<MetastableProcessWindow>,
    /// Related literature/report IDs supporting route precedent.
    pub related_literature_ids: Vec<String>,
    /// Calculation/evidence IDs supporting process plausibility.
    pub computational_evidence_ids: Vec<String>,
    /// MAT-010/MAT-013 attempt/sample record ID for a real synthesis attempt.
    pub synthesis_attempt_id: Option<String>,
    /// Characterization evidence IDs establishing or failing to establish target phase.
    pub characterization_evidence_ids: Vec<String>,
}

impl SynthesisRouteAssessment {
    /// Validate route/state consistency without granting MAT-001 authority.
    pub fn validate(&self) -> Result<(), NoveltySynthesisError> {
        nonempty("subject_identity", &self.subject_identity)?;
        nonempty("route_id", &self.route_id)?;
        if let Some(process) = &self.process_artifact {
            artifact(process)?;
        }
        if let Some(window) = &self.metastable_window {
            window.validate()?;
        }
        unique_nonempty(&self.related_literature_ids, "related_literature_id")?;
        unique_nonempty(
            &self.computational_evidence_ids,
            "computational_evidence_id",
        )?;
        unique_nonempty(
            &self.characterization_evidence_ids,
            "characterization_evidence_id",
        )?;
        if let Some(attempt) = &self.synthesis_attempt_id {
            nonempty("synthesis_attempt_id", attempt)?;
        }

        let has_route = self.process_artifact.is_some();
        let has_attempt = self.synthesis_attempt_id.is_some();
        match self.state {
            SynthesizabilityState::NoRouteProposed
                if !has_route
                    && self.metastable_window.is_none()
                    && !has_attempt
                    && self.related_literature_ids.is_empty()
                    && self.computational_evidence_ids.is_empty()
                    && self.characterization_evidence_ids.is_empty() =>
            {
                Ok(())
            }
            SynthesizabilityState::NoRouteProposed => {
                Err(NoveltySynthesisError::NoRouteStateContainsRouteEvidence)
            }
            SynthesizabilityState::RouteProposed if has_route => Ok(()),
            SynthesizabilityState::RouteProposed => Err(NoveltySynthesisError::RouteStateMissingProtocol),
            SynthesizabilityState::ComputationallySupportedProcessWindow
                if has_route
                    && self.metastable_window.is_some()
                    && !self.computational_evidence_ids.is_empty() =>
            {
                Ok(())
            }
            SynthesizabilityState::ComputationallySupportedProcessWindow => {
                Err(NoveltySynthesisError::ComputationalRouteMissingEvidence)
            }
            SynthesizabilityState::RelatedLiteraturePrecedent
                if has_route && !self.related_literature_ids.is_empty() =>
            {
                Ok(())
            }
            SynthesizabilityState::RelatedLiteraturePrecedent => {
                Err(NoveltySynthesisError::LiteratureRouteMissingEvidence)
            }
            SynthesizabilityState::AttemptedSynthesis if has_route && has_attempt => Ok(()),
            SynthesizabilityState::AttemptedSynthesis => {
                Err(NoveltySynthesisError::AttemptStateMissingAttempt)
            }
            SynthesizabilityState::SynthesisFailed if has_route && has_attempt => Ok(()),
            SynthesizabilityState::SynthesisFailed => {
                Err(NoveltySynthesisError::AttemptStateMissingAttempt)
            }
            SynthesizabilityState::MaterialProducedTargetPhaseUnestablished
                if has_route && has_attempt && !self.characterization_evidence_ids.is_empty() =>
            {
                Ok(())
            }
            SynthesizabilityState::MaterialProducedTargetPhaseUnestablished => {
                Err(NoveltySynthesisError::ProducedStateMissingCharacterization)
            }
            SynthesizabilityState::TargetPhaseEstablished
                if has_route && has_attempt && !self.characterization_evidence_ids.is_empty() =>
            {
                Ok(())
            }
            SynthesizabilityState::TargetPhaseEstablished => {
                Err(NoveltySynthesisError::EstablishedStateMissingCharacterization)
            }
        }
    }
}

fn validate_search(search: &PriorArtSearchRecord) -> Result<(), NoveltySynthesisError> {
    nonempty("source_id", &search.source_id)?;
    validate_date(&search.searched_on)?;
    artifact(&search.query_manifest)?;
    artifact(&search.result_artifact)?;
    nonempty("license_id", &search.license_id)?;
    match &search.outcome {
        PriorArtSearchOutcome::HitFound {
            hit_ids,
            hit_classes,
        } => {
            unique_nonempty(hit_ids, "prior_art_hit_id")?;
            if hit_ids.is_empty() || hit_classes.is_empty() {
                return Err(NoveltySynthesisError::EmptyPriorArtHit);
            }
            let classes: HashSet<_> = hit_classes.iter().collect();
            if classes.len() != hit_classes.len() {
                return Err(NoveltySynthesisError::DuplicatePriorArtHitClass);
            }
            Ok(())
        }
        PriorArtSearchOutcome::NoHit => Ok(()),
        PriorArtSearchOutcome::Unavailable { reason }
        | PriorArtSearchOutcome::Incomplete { reason } => nonempty("search outcome reason", reason),
    }
}

fn artifact(value: &PropertyArtifactRef) -> Result<(), NoveltySynthesisError> {
    nonempty("artifact source_id", &value.source_id)?;
    sha256(&value.artifact_sha256)
}

fn sha256(value: &str) -> Result<(), NoveltySynthesisError> {
    if value.len() != SHA256_HEX_LEN || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(NoveltySynthesisError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn validate_date(value: &str) -> Result<(), NoveltySynthesisError> {
    let bytes = value.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
    {
        return Err(NoveltySynthesisError::InvalidDate(value.to_string()));
    }
    Ok(())
}

fn unique_nonempty(values: &[String], field: &'static str) -> Result<(), NoveltySynthesisError> {
    let mut seen = HashSet::new();
    for value in values {
        nonempty(field, value)?;
        if !seen.insert(value.as_str()) {
            return Err(NoveltySynthesisError::DuplicateStringValue {
                field,
                value: value.clone(),
            });
        }
    }
    Ok(())
}

fn nonempty(field: &'static str, value: &str) -> Result<(), NoveltySynthesisError> {
    if value.trim().is_empty() {
        Err(NoveltySynthesisError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), NoveltySynthesisError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(NoveltySynthesisError::NonFiniteValue { field, value })
    }
}

/// Novelty/prior-art/synthesizability validation failure.
#[derive(Debug, Clone, PartialEq)]
pub enum NoveltySynthesisError {
    /// Required text field was empty.
    EmptyField(&'static str),
    /// A SHA-256 binding was malformed.
    InvalidSha256,
    /// Search date was not canonical `YYYY-MM-DD`.
    InvalidDate(String),
    /// Numeric process-window value was NaN/infinite.
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Composition-neighborhood tolerance cannot be explicitly zero.
    ZeroCompositionTolerance,
    /// Same prior-art search binding was repeated.
    DuplicatePriorArtSearch,
    /// Hit outcome lacked hit IDs/classes.
    EmptyPriorArtHit,
    /// Same hit class was repeated.
    DuplicatePriorArtHitClass,
    /// Duplicate string value in a set-like field.
    DuplicateStringValue {
        /// Field name.
        field: &'static str,
        /// Repeated value.
        value: String,
    },
    /// `Unchecked` state carried search records.
    UncheckedWithSearches,
    /// `SearchInProgress` lacked incomplete/unavailable coverage state.
    SearchInProgressWithoutIncompleteCoverage,
    /// `PriorArtFound` lacked a hit.
    PriorArtFoundWithoutHit,
    /// `NoMatchInBoundSearch` did not have complete all-no-hit searches.
    NoMatchStateWithoutCompleteNoHitSearch,
    /// `PotentiallyDistinct` did not have complete all-no-hit searches.
    PotentiallyDistinctWithoutCompleteNoHitSearch,
    /// `ExpertReviewed` lacked a review artifact.
    ExpertReviewedWithoutArtifact,
    /// Metastable process window had no variables.
    EmptyProcessWindow,
    /// Process-window bounds were inverted.
    InvalidProcessWindowInterval(String),
    /// Process-window variable key repeated.
    DuplicateProcessWindowVariable(String),
    /// `NoRouteProposed` contained route/evidence state.
    NoRouteStateContainsRouteEvidence,
    /// Route-proposed state lacked a protocol artifact.
    RouteStateMissingProtocol,
    /// Computational route state lacked protocol/window/computational evidence.
    ComputationalRouteMissingEvidence,
    /// Literature-precedent state lacked route/literature evidence.
    LiteratureRouteMissingEvidence,
    /// Physical attempt/failure state lacked an attempt record.
    AttemptStateMissingAttempt,
    /// Produced-but-unestablished state lacked characterization.
    ProducedStateMissingCharacterization,
    /// Target-phase-established state lacked characterization.
    EstablishedStateMissingCharacterization,
}

#[cfg(test)]
mod tests {
    use super::*;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn artifact_ref(source: &str, hash: &str) -> PropertyArtifactRef {
        PropertyArtifactRef {
            source_id: source.to_string(),
            artifact_sha256: hash.to_string(),
        }
    }

    fn neighborhood() -> PriorArtNeighborhood {
        PriorArtNeighborhood {
            method_id: "composition-structure-neighborhood-v1".to_string(),
            artifact_sha256: A64.to_string(),
            composition_tolerance_ppm: Some(10_000),
            structure_sensitive: true,
            process_sensitive: true,
        }
    }

    fn no_hit(source: &str) -> PriorArtSearchRecord {
        PriorArtSearchRecord {
            source_id: source.to_string(),
            source_kind: PriorArtSourceKind::Literature,
            searched_on: "2026-09-19".to_string(),
            query_manifest: artifact_ref("query", B64),
            result_artifact: artifact_ref(&format!("result-{source}"), C64),
            license_id: "source-terms-v1".to_string(),
            outcome: PriorArtSearchOutcome::NoHit,
        }
    }

    #[test]
    fn no_match_in_bound_search_requires_complete_no_hit_searches() {
        let assessment = NoveltyAssessment {
            subject_identity: "material-subject:v1|fixture".to_string(),
            neighborhood: neighborhood(),
            searches: vec![no_hit("literature")],
            blind_spots: vec!["non-indexed theses".to_string()],
            state: NoveltyAssessmentState::NoMatchInBoundSearch,
            expert_review: None,
        };
        assert!(assessment.validate().is_ok());
        assert!(!assessment.has_prior_art_hit().unwrap());
    }

    #[test]
    fn composition_neighbor_counts_as_prior_art_even_when_exact_subject_is_absent() {
        let mut search = no_hit("materials-db");
        search.outcome = PriorArtSearchOutcome::HitFound {
            hit_ids: vec!["neighbor-record-17".to_string()],
            hit_classes: vec![PriorArtHitClass::CompositionNeighbor],
        };
        let assessment = NoveltyAssessment {
            subject_identity: "material-subject:v1|Ti-Zr-Nb-Ta-Er-fixture".to_string(),
            neighborhood: neighborhood(),
            searches: vec![search],
            blind_spots: vec![],
            state: NoveltyAssessmentState::PriorArtFound,
            expert_review: None,
        };
        assert!(assessment.validate().is_ok());
        assert!(assessment.has_prior_art_hit().unwrap());
    }

    #[test]
    fn no_match_state_is_rejected_if_any_prior_art_hit_exists() {
        let mut search = no_hit("patents");
        search.outcome = PriorArtSearchOutcome::HitFound {
            hit_ids: vec!["patent-neighbor".to_string()],
            hit_classes: vec![PriorArtHitClass::ProcessNeighbor],
        };
        let assessment = NoveltyAssessment {
            subject_identity: "material-subject:v1|fixture".to_string(),
            neighborhood: neighborhood(),
            searches: vec![search],
            blind_spots: vec![],
            state: NoveltyAssessmentState::NoMatchInBoundSearch,
            expert_review: None,
        };
        assert_eq!(
            assessment.validate(),
            Err(NoveltySynthesisError::NoMatchStateWithoutCompleteNoHitSearch)
        );
    }

    #[test]
    fn metastable_process_window_is_representable_without_equilibrium_stability_claim() {
        let window = MetastableProcessWindow {
            window_id: "rapid-quench-v1".to_string(),
            variables: vec![
                ProcessWindowVariable {
                    key: "anneal_temperature".to_string(),
                    lower: 900.0,
                    upper: 1100.0,
                    unit: "K".to_string(),
                },
                ProcessWindowVariable {
                    key: "quench_rate".to_string(),
                    lower: 1.0e3,
                    upper: 1.0e6,
                    unit: "K/s".to_string(),
                },
            ],
            support_artifact: artifact_ref("metastable-window", A64),
        };
        let route = SynthesisRouteAssessment {
            subject_identity: "material-subject:v1|metastable-fixture".to_string(),
            route_id: "rapid-quench-route".to_string(),
            state: SynthesizabilityState::ComputationallySupportedProcessWindow,
            process_artifact: Some(artifact_ref("route-protocol", B64)),
            metastable_window: Some(window),
            related_literature_ids: vec![],
            computational_evidence_ids: vec!["calphad-or-dft-evidence-1".to_string()],
            synthesis_attempt_id: None,
            characterization_evidence_ids: vec![],
        };
        assert!(route.validate().is_ok());
    }

    #[test]
    fn failed_synthesis_remains_failed_attempt_not_unsynthesizable_claim() {
        let route = SynthesisRouteAssessment {
            subject_identity: "material-subject:v1|fixture".to_string(),
            route_id: "arc-melt-v1".to_string(),
            state: SynthesizabilityState::SynthesisFailed,
            process_artifact: Some(artifact_ref("arc-melt-protocol", A64)),
            metastable_window: None,
            related_literature_ids: vec![],
            computational_evidence_ids: vec![],
            synthesis_attempt_id: Some("sample-attempt-42".to_string()),
            characterization_evidence_ids: vec![],
        };
        assert!(route.validate().is_ok());
        assert_eq!(route.state, SynthesizabilityState::SynthesisFailed);
    }

    #[test]
    fn target_phase_established_requires_characterization() {
        let route = SynthesisRouteAssessment {
            subject_identity: "material-subject:v1|fixture".to_string(),
            route_id: "route-v1".to_string(),
            state: SynthesizabilityState::TargetPhaseEstablished,
            process_artifact: Some(artifact_ref("protocol", A64)),
            metastable_window: None,
            related_literature_ids: vec![],
            computational_evidence_ids: vec![],
            synthesis_attempt_id: Some("attempt-1".to_string()),
            characterization_evidence_ids: vec![],
        };
        assert_eq!(
            route.validate(),
            Err(NoveltySynthesisError::EstablishedStateMissingCharacterization)
        );
    }
}
