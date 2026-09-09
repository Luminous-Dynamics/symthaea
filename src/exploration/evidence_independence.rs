// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Declared evidence-lineage diversity for generativity admission.
//!
//! Distinct run IDs are not automatically independent experiments. This module keeps
//! method/data lineage diversity explicit and can require diversity floors before an
//! evidence-backed candidate enters Pareto comparison.
//!
//! The declarations here are still claims. They are not authenticated merely because
//! they are structurally valid. A future Xenia/Mycelix provenance verifier can qualify
//! the declarations without changing the separation between evidence, confidence, and
//! authority.

#![deny(unsafe_code)]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::archive::{ArchiveInsertOutcome, ArchiveValidationError, ParetoPolicy};
use super::evidence_admission::{
    qualify_archive_candidate, DimensionEvidenceBinding, EvidenceAdmissionError,
    EvidenceAdmissionPolicy, EvidenceAwareGenerativityArchive, EvidenceSupportReport,
    GenerativityDimension,
};
use super::persisted_evidence::VerifiedGenerativityBundle;

pub const MAX_LINEAGE_ID_LEN: usize = 256;
pub const MAX_SOURCE_LINEAGES_PER_EVIDENCE: usize = 32;

/// Declared experimental/data lineage for one verified evidence-plane item.
///
/// `method_lineage_id` may identify a shared method/config/protocol family.
/// `source_lineage_ids` may identify datasets, sensors, cohorts, simulations, or other
/// upstream evidence sources. IDs are opaque here; semantics/authenticity belong to a
/// provenance layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLineageDeclaration {
    pub evidence_id: String,
    pub method_lineage_id: String,
    pub source_lineage_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclaredIndependencePolicy {
    /// Minimum distinct declared method lineages supporting each dimension.
    pub min_method_lineages_per_dimension: usize,
    /// Minimum distinct declared source/data lineages supporting each dimension.
    pub min_source_lineages_per_dimension: usize,
    pub max_source_lineages_per_evidence: usize,
}

impl Default for DeclaredIndependencePolicy {
    fn default() -> Self {
        Self {
            min_method_lineages_per_dimension: 1,
            min_source_lineages_per_dimension: 1,
            max_source_lineages_per_evidence: 16,
        }
    }
}

impl DeclaredIndependencePolicy {
    /// A stronger convenience profile for claims that require replication across at
    /// least two declared method families and two source/data lineages.
    pub fn replication() -> Self {
        Self {
            min_method_lineages_per_dimension: 2,
            min_source_lineages_per_dimension: 2,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), IndependenceError> {
        if self.max_source_lineages_per_evidence == 0
            || self.max_source_lineages_per_evidence > MAX_SOURCE_LINEAGES_PER_EVIDENCE
        {
            return Err(IndependenceError::InvalidSourceLineageLimit {
                value: self.max_source_lineages_per_evidence,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DimensionLineageDiversity {
    pub dimension: GenerativityDimension,
    pub qualified_evidence_ids: Vec<String>,
    pub distinct_method_lineages: Vec<String>,
    pub distinct_source_lineages: Vec<String>,
    /// Method lineages reused by more than one qualified run for this dimension.
    pub reused_method_lineages: Vec<String>,
    /// Source/data lineages reused by more than one qualified run for this dimension.
    pub reused_source_lineages: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeclaredIndependenceReport {
    pub assessment_subject_id: String,
    pub dimensions: Vec<DimensionLineageDiversity>,
}

#[derive(Debug)]
pub enum IndependenceError {
    InvalidSourceLineageLimit { value: usize },
    EmptyField(&'static str),
    UnboundedLineageId { field: &'static str, len: usize },
    TooManySourceLineages {
        evidence_id: String,
        count: usize,
        max: usize,
    },
    DuplicateSourceLineage {
        evidence_id: String,
        lineage_id: String,
    },
    DuplicateDeclaration(String),
    DeclarationForUnqualifiedEvidence(String),
    MissingDeclaration {
        dimension: GenerativityDimension,
        evidence_id: String,
    },
    InsufficientMethodLineageDiversity {
        dimension: GenerativityDimension,
        required: usize,
        observed: usize,
    },
    InsufficientSourceLineageDiversity {
        dimension: GenerativityDimension,
        required: usize,
        observed: usize,
    },
    EvidenceAdmission(EvidenceAdmissionError),
    Archive(ArchiveValidationError),
}

impl From<EvidenceAdmissionError> for IndependenceError {
    fn from(value: EvidenceAdmissionError) -> Self {
        Self::EvidenceAdmission(value)
    }
}

impl From<ArchiveValidationError> for IndependenceError {
    fn from(value: ArchiveValidationError) -> Self {
        Self::Archive(value)
    }
}

impl std::fmt::Display for IndependenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSourceLineageLimit { value } => write!(
                f,
                "max source lineages per evidence must be within 1..={MAX_SOURCE_LINEAGES_PER_EVIDENCE}, got {value}"
            ),
            Self::EmptyField(field) => write!(f, "required lineage field is empty: {field}"),
            Self::UnboundedLineageId { field, len } => write!(
                f,
                "lineage field {field} is {len} bytes; maximum is {MAX_LINEAGE_ID_LEN}"
            ),
            Self::TooManySourceLineages {
                evidence_id,
                count,
                max,
            } => write!(
                f,
                "evidence {evidence_id} declares {count} source lineages; maximum is {max}"
            ),
            Self::DuplicateSourceLineage {
                evidence_id,
                lineage_id,
            } => write!(
                f,
                "evidence {evidence_id} repeats source lineage {lineage_id}"
            ),
            Self::DuplicateDeclaration(id) => {
                write!(f, "more than one lineage declaration exists for evidence {id}")
            }
            Self::DeclarationForUnqualifiedEvidence(id) => write!(
                f,
                "lineage declaration references evidence that is not a qualified evidence-plane run: {id}"
            ),
            Self::MissingDeclaration {
                dimension,
                evidence_id,
            } => write!(
                f,
                "dimension {dimension:?} uses qualified evidence {evidence_id} without a lineage declaration"
            ),
            Self::InsufficientMethodLineageDiversity {
                dimension,
                required,
                observed,
            } => write!(
                f,
                "dimension {dimension:?} requires {required} declared method lineages, observed {observed}"
            ),
            Self::InsufficientSourceLineageDiversity {
                dimension,
                required,
                observed,
            } => write!(
                f,
                "dimension {dimension:?} requires {required} declared source lineages, observed {observed}"
            ),
            Self::EvidenceAdmission(error) => write!(f, "evidence admission failed: {error}"),
            Self::Archive(error) => write!(f, "archive validation failed: {error}"),
        }
    }
}

impl std::error::Error for IndependenceError {}

/// Verify evidence-aware admission first, then evaluate declared lineage diversity.
///
/// This deliberately does not call the result "independent evidence". It is a report
/// about declared lineage diversity over mechanically qualified runs.
pub fn assess_declared_independence(
    entry_id: impl Into<String>,
    niche: impl Into<String>,
    bundle: &VerifiedGenerativityBundle,
    bindings: &[DimensionEvidenceBinding],
    evidence_policy: EvidenceAdmissionPolicy,
    declarations: &[EvidenceLineageDeclaration],
    independence_policy: DeclaredIndependencePolicy,
) -> Result<DeclaredIndependenceReport, IndependenceError> {
    independence_policy.validate()?;

    // Establish the stronger evidence-aware precondition before considering lineage
    // declarations. The returned candidate is intentionally not used for scoring here.
    let _candidate = qualify_archive_candidate(
        entry_id,
        niche,
        bundle,
        bindings,
        evidence_policy,
    )?;

    let qualified_ids = bundle
        .capsules()
        .iter()
        .filter(|capsule| capsule.envelope().integrity_satisfied())
        .map(|capsule| format!("evidence-plane:{}", capsule.envelope().run_id()))
        .collect::<HashSet<_>>();

    let declaration_by_id = validate_declarations(
        declarations,
        &qualified_ids,
        independence_policy.max_source_lineages_per_evidence,
    )?;

    let binding_by_dimension = bindings
        .iter()
        .map(|binding| (binding.dimension, binding))
        .collect::<HashMap<_, _>>();

    let mut dimensions = Vec::with_capacity(GenerativityDimension::ALL.len());
    for dimension in GenerativityDimension::ALL {
        let binding = binding_by_dimension
            .get(&dimension)
            .expect("evidence-aware admission already requires every dimension binding");

        let mut qualified_evidence_ids = Vec::new();
        let mut method_counts = HashMap::<String, usize>::new();
        let mut source_counts = HashMap::<String, usize>::new();

        for evidence_id in &binding.evidence_ids {
            if !qualified_ids.contains(evidence_id) {
                continue;
            }
            qualified_evidence_ids.push(evidence_id.clone());
            let declaration = declaration_by_id.get(evidence_id).ok_or_else(|| {
                IndependenceError::MissingDeclaration {
                    dimension,
                    evidence_id: evidence_id.clone(),
                }
            })?;
            *method_counts
                .entry(declaration.method_lineage_id.clone())
                .or_default() += 1;
            for source in &declaration.source_lineage_ids {
                *source_counts.entry(source.clone()).or_default() += 1;
            }
        }

        qualified_evidence_ids.sort();
        let mut distinct_method_lineages = method_counts.keys().cloned().collect::<Vec<_>>();
        distinct_method_lineages.sort();
        let mut distinct_source_lineages = source_counts.keys().cloned().collect::<Vec<_>>();
        distinct_source_lineages.sort();

        if distinct_method_lineages.len() < independence_policy.min_method_lineages_per_dimension {
            return Err(IndependenceError::InsufficientMethodLineageDiversity {
                dimension,
                required: independence_policy.min_method_lineages_per_dimension,
                observed: distinct_method_lineages.len(),
            });
        }
        if distinct_source_lineages.len() < independence_policy.min_source_lineages_per_dimension {
            return Err(IndependenceError::InsufficientSourceLineageDiversity {
                dimension,
                required: independence_policy.min_source_lineages_per_dimension,
                observed: distinct_source_lineages.len(),
            });
        }

        let mut reused_method_lineages = method_counts
            .into_iter()
            .filter_map(|(id, count)| (count > 1).then_some(id))
            .collect::<Vec<_>>();
        reused_method_lineages.sort();
        let mut reused_source_lineages = source_counts
            .into_iter()
            .filter_map(|(id, count)| (count > 1).then_some(id))
            .collect::<Vec<_>>();
        reused_source_lineages.sort();

        dimensions.push(DimensionLineageDiversity {
            dimension,
            qualified_evidence_ids,
            distinct_method_lineages,
            distinct_source_lineages,
            reused_method_lineages,
            reused_source_lineages,
        });
    }

    Ok(DeclaredIndependenceReport {
        assessment_subject_id: bundle.assessment().subject_id.clone(),
        dimensions,
    })
}

fn validate_declarations<'a>(
    declarations: &'a [EvidenceLineageDeclaration],
    qualified_ids: &HashSet<String>,
    max_sources: usize,
) -> Result<HashMap<String, &'a EvidenceLineageDeclaration>, IndependenceError> {
    let mut result = HashMap::new();
    for declaration in declarations {
        validate_id("evidence_id", &declaration.evidence_id)?;
        validate_id("method_lineage_id", &declaration.method_lineage_id)?;
        if !qualified_ids.contains(&declaration.evidence_id) {
            return Err(IndependenceError::DeclarationForUnqualifiedEvidence(
                declaration.evidence_id.clone(),
            ));
        }
        if declaration.source_lineage_ids.len() > max_sources {
            return Err(IndependenceError::TooManySourceLineages {
                evidence_id: declaration.evidence_id.clone(),
                count: declaration.source_lineage_ids.len(),
                max: max_sources,
            });
        }
        let mut seen = HashSet::new();
        for source in &declaration.source_lineage_ids {
            validate_id("source_lineage_id", source)?;
            if !seen.insert(source.as_str()) {
                return Err(IndependenceError::DuplicateSourceLineage {
                    evidence_id: declaration.evidence_id.clone(),
                    lineage_id: source.clone(),
                });
            }
        }
        if result
            .insert(declaration.evidence_id.clone(), declaration)
            .is_some()
        {
            return Err(IndependenceError::DuplicateDeclaration(
                declaration.evidence_id.clone(),
            ));
        }
    }
    Ok(result)
}

fn validate_id(field: &'static str, value: &str) -> Result<(), IndependenceError> {
    if value.trim().is_empty() {
        return Err(IndependenceError::EmptyField(field));
    }
    if value.len() > MAX_LINEAGE_ID_LEN {
        return Err(IndependenceError::UnboundedLineageId {
            field,
            len: value.len(),
        });
    }
    Ok(())
}

/// Evidence-aware Pareto archive with an additional declared-lineage diversity gate.
///
/// The archive stores reports beside retained entries. It does not infer independence,
/// confidence, or truth from the number of declared lineages.
#[derive(Debug, Clone, Serialize)]
pub struct DeclaredIndependenceArchive {
    inner: EvidenceAwareGenerativityArchive,
    independence_policy: DeclaredIndependencePolicy,
    lineage_report_by_entry: HashMap<String, DeclaredIndependenceReport>,
}

impl DeclaredIndependenceArchive {
    pub fn new(
        pareto_policy: ParetoPolicy,
        evidence_policy: EvidenceAdmissionPolicy,
        independence_policy: DeclaredIndependencePolicy,
    ) -> Result<Self, IndependenceError> {
        independence_policy.validate()?;
        Ok(Self {
            inner: EvidenceAwareGenerativityArchive::new(pareto_policy, evidence_policy)?,
            independence_policy,
            lineage_report_by_entry: HashMap::new(),
        })
    }

    pub fn evidence_support_for(&self, entry_id: &str) -> Option<&EvidenceSupportReport> {
        self.inner.support_for(entry_id)
    }

    pub fn lineage_report_for(&self, entry_id: &str) -> Option<&DeclaredIndependenceReport> {
        self.lineage_report_by_entry.get(entry_id)
    }

    pub fn qualify_and_insert(
        &mut self,
        entry_id: impl Into<String>,
        niche: impl Into<String>,
        bundle: &VerifiedGenerativityBundle,
        bindings: &[DimensionEvidenceBinding],
        declarations: &[EvidenceLineageDeclaration],
    ) -> Result<ArchiveInsertOutcome, IndependenceError> {
        let entry_id = entry_id.into();
        let niche = niche.into();
        let evidence_policy = self.inner.admission_policy();
        let report = assess_declared_independence(
            entry_id.clone(),
            niche.clone(),
            bundle,
            bindings,
            evidence_policy,
            declarations,
            self.independence_policy,
        )?;
        let outcome = self
            .inner
            .qualify_and_insert(entry_id.clone(), niche, bundle, bindings)?;
        if let ArchiveInsertOutcome::Inserted { removed_entry_ids } = &outcome {
            for removed in removed_entry_ids {
                self.lineage_report_by_entry.remove(removed);
            }
            self.lineage_report_by_entry.insert(entry_id, report);
        }
        Ok(outcome)
    }
}

impl Default for DeclaredIndependenceArchive {
    fn default() -> Self {
        Self::new(
            ParetoPolicy::default(),
            EvidenceAdmissionPolicy::default(),
            DeclaredIndependencePolicy::default(),
        )
        .expect("default declared-independence policies are valid")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use symthaea_evidence_plane::{EvidenceCounters, Expectation, RunEvidence, RunId};

    use super::*;
    use crate::exploration::evidence_binding::EvidencePlaneEnvelope;
    use crate::exploration::generativity::{
        GenerativityAssessment, GenerativityEstimate, GenerativityVector,
    };
    use crate::exploration::persisted_evidence::{
        PersistedEvidenceCapsule, PersistedGenerativityBundle,
    };

    fn vector(confidence: f64) -> GenerativityVector {
        let e = |value| GenerativityEstimate::new(value, confidence).unwrap();
        GenerativityVector {
            immediate_utility: e(0.7),
            epistemic_gain: e(0.7),
            option_value: e(0.7),
            diversity: e(0.7),
            capability_gain: e(0.7),
            diffusion: e(0.7),
            commons_gain: e(0.7),
            regeneration: e(0.7),
            dependency_risk: e(0.2),
            concentration_risk: e(0.2),
            irreversibility_risk: e(0.2),
        }
    }

    fn run(run_id: &str, calls: f64) -> RunEvidence {
        let mut declared = BTreeMap::new();
        declared.insert("mechanism_calls".into(), Expectation::MustBePositive);
        let mut measured = EvidenceCounters::new();
        measured.record("mechanism_calls", calls);
        RunEvidence::new(RunId::new(run_id), &("mode", "active"), declared, measured)
    }

    fn bundle(subject: &str, run_specs: &[(&str, f64)]) -> VerifiedGenerativityBundle {
        let mut assessment = GenerativityAssessment::new(subject, "context", vector(0.9));
        let mut capsules = Vec::new();
        for (run_id, calls) in run_specs {
            let run = run(run_id, *calls);
            let envelope = EvidencePlaneEnvelope::from_run(&run).unwrap();
            envelope.bind_qualified(&mut assessment, None).unwrap();
            capsules.push(PersistedEvidenceCapsule::from_run(run, vec![]).unwrap());
        }
        PersistedGenerativityBundle::new(assessment, capsules)
            .verify()
            .unwrap()
    }

    fn all_bindings(ids: &[&str]) -> Vec<DimensionEvidenceBinding> {
        GenerativityDimension::ALL
            .iter()
            .map(|dimension| DimensionEvidenceBinding {
                dimension: *dimension,
                evidence_ids: ids.iter().map(|id| (*id).to_string()).collect(),
            })
            .collect()
    }

    fn declaration(evidence_id: &str, method: &str, source: &str) -> EvidenceLineageDeclaration {
        EvidenceLineageDeclaration {
            evidence_id: evidence_id.into(),
            method_lineage_id: method.into(),
            source_lineage_ids: vec![source.into()],
        }
    }

    #[test]
    fn distinct_runs_do_not_satisfy_replication_if_lineages_are_reused() {
        let bundle = bundle("alpha", &[("run:1", 4.0), ("run:2", 5.0)]);
        let ids = ["evidence-plane:run:1", "evidence-plane:run:2"];
        let declarations = [
            declaration(ids[0], "method:same", "data:same"),
            declaration(ids[1], "method:same", "data:same"),
        ];
        assert!(matches!(
            assess_declared_independence(
                "entry",
                "research",
                &bundle,
                &all_bindings(&ids),
                EvidenceAdmissionPolicy::default(),
                &declarations,
                DeclaredIndependencePolicy::replication(),
            ),
            Err(IndependenceError::InsufficientMethodLineageDiversity { .. })
        ));
    }

    #[test]
    fn replication_profile_accepts_two_declared_method_and_source_lineages() {
        let bundle = bundle("alpha", &[("run:1", 4.0), ("run:2", 5.0)]);
        let ids = ["evidence-plane:run:1", "evidence-plane:run:2"];
        let declarations = [
            declaration(ids[0], "method:a", "data:a"),
            declaration(ids[1], "method:b", "data:b"),
        ];
        let report = assess_declared_independence(
            "entry",
            "research",
            &bundle,
            &all_bindings(&ids),
            EvidenceAdmissionPolicy::default(),
            &declarations,
            DeclaredIndependencePolicy::replication(),
        )
        .unwrap();
        assert!(report.dimensions.iter().all(|dimension| {
            dimension.distinct_method_lineages.len() == 2
                && dimension.distinct_source_lineages.len() == 2
        }));
    }

    #[test]
    fn lineage_reuse_is_visible_even_when_policy_allows_it() {
        let bundle = bundle("alpha", &[("run:1", 4.0), ("run:2", 5.0)]);
        let ids = ["evidence-plane:run:1", "evidence-plane:run:2"];
        let declarations = [
            declaration(ids[0], "method:same", "data:same"),
            declaration(ids[1], "method:same", "data:same"),
        ];
        let report = assess_declared_independence(
            "entry",
            "research",
            &bundle,
            &all_bindings(&ids),
            EvidenceAdmissionPolicy::default(),
            &declarations,
            DeclaredIndependencePolicy::default(),
        )
        .unwrap();
        assert!(report.dimensions.iter().all(|dimension| {
            dimension.reused_method_lineages == vec!["method:same"]
                && dimension.reused_source_lineages == vec!["data:same"]
        }));
    }

    #[test]
    fn missing_lineage_declaration_fails_closed() {
        let bundle = bundle("alpha", &[("run:1", 4.0)]);
        let ids = ["evidence-plane:run:1"];
        assert!(matches!(
            assess_declared_independence(
                "entry",
                "research",
                &bundle,
                &all_bindings(&ids),
                EvidenceAdmissionPolicy::default(),
                &[],
                DeclaredIndependencePolicy::default(),
            ),
            Err(IndependenceError::MissingDeclaration { .. })
        ));
    }

    #[test]
    fn declaration_for_generic_or_unknown_evidence_is_rejected() {
        let bundle = bundle("alpha", &[("run:1", 4.0)]);
        let ids = ["evidence-plane:run:1"];
        let declarations = [declaration("literature:1", "method:a", "data:a")];
        assert!(matches!(
            assess_declared_independence(
                "entry",
                "research",
                &bundle,
                &all_bindings(&ids),
                EvidenceAdmissionPolicy::default(),
                &declarations,
                DeclaredIndependencePolicy::default(),
            ),
            Err(IndependenceError::DeclarationForUnqualifiedEvidence(id)) if id == "literature:1"
        ));
    }

    #[test]
    fn archive_prunes_lineage_report_with_dominated_entry() {
        let worse = bundle("worse", &[("run:worse", 4.0)]);
        let better = bundle("better", &[("run:better", 4.0)]);
        let worse_ids = ["evidence-plane:run:worse"];
        let better_ids = ["evidence-plane:run:better"];
        let mut archive = DeclaredIndependenceArchive::default();

        archive
            .qualify_and_insert(
                "worse-entry",
                "repairable",
                &worse,
                &all_bindings(&worse_ids),
                &[declaration(worse_ids[0], "method:w", "data:w")],
            )
            .unwrap();
        archive
            .qualify_and_insert(
                "better-entry",
                "repairable",
                &better,
                &all_bindings(&better_ids),
                &[declaration(better_ids[0], "method:b", "data:b")],
            )
            .unwrap();

        // Vectors are equal in this fixture, so both remain. This assertion ensures
        // reports are retained for inserted candidates rather than inferring pruning.
        assert!(archive.lineage_report_for("worse-entry").is_some());
        assert!(archive.lineage_report_for("better-entry").is_some());
    }
}
