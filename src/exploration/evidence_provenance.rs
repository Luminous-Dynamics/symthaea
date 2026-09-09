// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical provenance capsule for generativity evidence-lineage analysis.
//!
//! Xenia can authenticate arbitrary bytes, but Symthaea must define which bytes correspond
//! to the lineage semantics it actually uses. This module canonicalizes the qualified
//! evidence identities, per-dimension evidence assignment, immediate lineage declarations,
//! and reachable ancestry graph into one deterministic artifact.
//!
//! The capsule is still evidence metadata. A signature over it authenticates the signer and
//! exact semantic input; it does not make lineage declarations true or grant authority.

#![deny(unsafe_code)]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::evidence_admission::{DimensionEvidenceBinding, GenerativityDimension};
use super::evidence_ancestry::{LineageAncestryEdge, LineageDomain, MAX_ANCESTRY_EDGES};
use super::evidence_independence::{
    EvidenceLineageDeclaration, MAX_LINEAGE_ID_LEN, MAX_SOURCE_LINEAGES_PER_EVIDENCE,
};
use super::persisted_evidence::VerifiedGenerativityBundle;

/// Xenia `EvidenceArtifactBinding.artifact_domain` to use for this capsule.
pub const GENERATIVITY_PROVENANCE_ARTIFACT_DOMAIN: &str =
    "symthaea-generativity-provenance";
/// Domain-owned schema committed into the Xenia artifact binding.
pub const GENERATIVITY_PROVENANCE_CAPSULE_SCHEMA: &str =
    "symthaea-generativity-provenance-capsule-v1";
/// Digest algorithm used for local capsule identity.
pub const GENERATIVITY_PROVENANCE_DIGEST_ALGORITHM: &str = "blake3-256";
pub const MAX_PROVENANCE_SUBJECT_LEN: usize = 1_024;
pub const MAX_PROVENANCE_BINDING_IDS_PER_DIMENSION: usize = 64;
pub const MAX_PROVENANCE_DECLARATIONS: usize = 4_096;

/// Content identity for one mechanically qualified evidence-plane run used by lineage analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceEvidenceRef {
    pub evidence_id: String,
    /// Lowercase BLAKE3 hex from the verified evidence-plane envelope.
    pub envelope_digest: String,
}

/// Qualified evidence-plane items assigned to one generativity dimension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceDimensionBinding {
    pub dimension: GenerativityDimension,
    pub qualified_evidence_ids: Vec<String>,
}

/// Deterministic semantic artifact suitable for Xenia evidence-artifact attestation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerativityProvenanceCapsule {
    pub schema: String,
    pub assessment_subject_id: String,
    pub evidence: Vec<ProvenanceEvidenceRef>,
    pub dimension_bindings: Vec<ProvenanceDimensionBinding>,
    pub lineage_declarations: Vec<EvidenceLineageDeclaration>,
    /// Only ancestry reachable from the selected lineage declarations is committed.
    pub ancestry_edges: Vec<LineageAncestryEdge>,
}

impl GenerativityProvenanceCapsule {
    /// Build a normalized capsule from a verified bundle and the lineage-analysis inputs.
    ///
    /// Generic or violated evidence may remain in the assessment, but only mechanically
    /// qualified evidence-plane runs participate in lineage-independence provenance.
    pub fn from_analysis_inputs(
        bundle: &VerifiedGenerativityBundle,
        bindings: &[DimensionEvidenceBinding],
        declarations: &[EvidenceLineageDeclaration],
        ancestry_edges: &[LineageAncestryEdge],
    ) -> Result<Self, ProvenanceCapsuleError> {
        let subject = bundle.assessment().subject_id.trim();
        validate_label("assessment_subject_id", subject, MAX_PROVENANCE_SUBJECT_LEN)?;
        if declarations.len() > MAX_PROVENANCE_DECLARATIONS {
            return Err(ProvenanceCapsuleError::TooManyDeclarations {
                count: declarations.len(),
                max: MAX_PROVENANCE_DECLARATIONS,
            });
        }
        if ancestry_edges.len() > MAX_ANCESTRY_EDGES {
            return Err(ProvenanceCapsuleError::TooManyAncestryEdges {
                count: ancestry_edges.len(),
                max: MAX_ANCESTRY_EDGES,
            });
        }

        let assessment_evidence_ids = bundle
            .assessment()
            .evidence
            .iter()
            .map(|evidence| evidence.evidence_id.as_str())
            .collect::<HashSet<_>>();

        let qualified_by_id = bundle
            .capsules()
            .iter()
            .filter(|capsule| capsule.envelope().integrity_satisfied())
            .map(|capsule| {
                (
                    format!("evidence-plane:{}", capsule.envelope().run_id()),
                    capsule.envelope().envelope_digest().to_string(),
                )
            })
            .collect::<HashMap<_, _>>();

        let normalized_bindings = normalize_dimension_bindings(
            bindings,
            &assessment_evidence_ids,
            &qualified_by_id,
        )?;
        let used_qualified_ids = normalized_bindings
            .iter()
            .flat_map(|binding| binding.qualified_evidence_ids.iter().cloned())
            .collect::<HashSet<_>>();

        let normalized_declarations = normalize_used_declarations(
            declarations,
            &used_qualified_ids,
        )?;

        let mut evidence = used_qualified_ids
            .iter()
            .map(|evidence_id| ProvenanceEvidenceRef {
                evidence_id: evidence_id.clone(),
                envelope_digest: qualified_by_id
                    .get(evidence_id)
                    .expect("qualified binding IDs come from qualified_by_id")
                    .clone(),
            })
            .collect::<Vec<_>>();
        evidence.sort_by(|a, b| a.evidence_id.cmp(&b.evidence_id));

        let reachable_edges = normalize_reachable_ancestry(
            ancestry_edges,
            &normalized_declarations,
        )?;

        let capsule = Self {
            schema: GENERATIVITY_PROVENANCE_CAPSULE_SCHEMA.to_string(),
            assessment_subject_id: subject.to_string(),
            evidence,
            dimension_bindings: normalized_bindings,
            lineage_declarations: normalized_declarations,
            ancestry_edges: reachable_edges,
        };
        capsule.validate_canonical()?;
        Ok(capsule)
    }

    /// Validate that a persisted capsule is already in canonical normalized form.
    pub fn validate_canonical(&self) -> Result<(), ProvenanceCapsuleError> {
        if self.schema != GENERATIVITY_PROVENANCE_CAPSULE_SCHEMA {
            return Err(ProvenanceCapsuleError::UnsupportedSchema(self.schema.clone()));
        }
        validate_label(
            "assessment_subject_id",
            &self.assessment_subject_id,
            MAX_PROVENANCE_SUBJECT_LEN,
        )?;
        if self.evidence.is_empty() {
            return Err(ProvenanceCapsuleError::NoQualifiedEvidence);
        }

        let mut evidence_ids = HashSet::new();
        let mut previous_evidence_id: Option<&str> = None;
        for evidence in &self.evidence {
            validate_lineage_id("evidence_id", &evidence.evidence_id)?;
            validate_digest(&evidence.envelope_digest)?;
            if !evidence_ids.insert(evidence.evidence_id.as_str()) {
                return Err(ProvenanceCapsuleError::DuplicateEvidenceRef(
                    evidence.evidence_id.clone(),
                ));
            }
            if previous_evidence_id.is_some_and(|previous| previous >= evidence.evidence_id.as_str()) {
                return Err(ProvenanceCapsuleError::NonCanonicalOrdering("evidence"));
            }
            previous_evidence_id = Some(&evidence.evidence_id);
        }

        if self.dimension_bindings.len() != GenerativityDimension::ALL.len() {
            return Err(ProvenanceCapsuleError::IncompleteDimensionBindings {
                count: self.dimension_bindings.len(),
            });
        }
        let mut seen_dimensions = HashSet::new();
        let mut previous_dimension_tag = None;
        for binding in &self.dimension_bindings {
            if !seen_dimensions.insert(binding.dimension) {
                return Err(ProvenanceCapsuleError::DuplicateDimensionBinding(
                    binding.dimension,
                ));
            }
            let tag = dimension_tag(binding.dimension);
            if previous_dimension_tag.is_some_and(|previous| previous >= tag) {
                return Err(ProvenanceCapsuleError::NonCanonicalOrdering(
                    "dimension_bindings",
                ));
            }
            previous_dimension_tag = Some(tag);
            if binding.qualified_evidence_ids.len() > MAX_PROVENANCE_BINDING_IDS_PER_DIMENSION {
                return Err(ProvenanceCapsuleError::TooManyBindingEvidenceIds {
                    dimension: binding.dimension,
                    count: binding.qualified_evidence_ids.len(),
                    max: MAX_PROVENANCE_BINDING_IDS_PER_DIMENSION,
                });
            }
            let mut previous_id: Option<&str> = None;
            for id in &binding.qualified_evidence_ids {
                if !evidence_ids.contains(id.as_str()) {
                    return Err(ProvenanceCapsuleError::BindingReferencesUnknownQualifiedEvidence {
                        dimension: binding.dimension,
                        evidence_id: id.clone(),
                    });
                }
                if previous_id.is_some_and(|previous| previous >= id.as_str()) {
                    return Err(ProvenanceCapsuleError::NonCanonicalOrdering(
                        "qualified_evidence_ids",
                    ));
                }
                previous_id = Some(id);
            }
        }

        let declaration_ids = self
            .lineage_declarations
            .iter()
            .map(|declaration| declaration.evidence_id.as_str())
            .collect::<HashSet<_>>();
        if declaration_ids != evidence_ids {
            return Err(ProvenanceCapsuleError::DeclarationEvidenceSetMismatch);
        }
        let mut previous_declaration_id: Option<&str> = None;
        for declaration in &self.lineage_declarations {
            validate_declaration(declaration)?;
            if previous_declaration_id
                .is_some_and(|previous| previous >= declaration.evidence_id.as_str())
            {
                return Err(ProvenanceCapsuleError::NonCanonicalOrdering(
                    "lineage_declarations",
                ));
            }
            previous_declaration_id = Some(&declaration.evidence_id);
            let mut previous_source: Option<&str> = None;
            for source in &declaration.source_lineage_ids {
                if previous_source.is_some_and(|previous| previous >= source.as_str()) {
                    return Err(ProvenanceCapsuleError::NonCanonicalOrdering(
                        "source_lineage_ids",
                    ));
                }
                previous_source = Some(source);
            }
        }

        let mut previous_edge_key: Option<(u8, &str, &str)> = None;
        for edge in &self.ancestry_edges {
            validate_edge(edge)?;
            let key = (domain_tag(edge.domain), edge.child_id.as_str(), edge.parent_id.as_str());
            if previous_edge_key.is_some_and(|previous| previous >= key) {
                return Err(ProvenanceCapsuleError::NonCanonicalOrdering(
                    "ancestry_edges",
                ));
            }
            previous_edge_key = Some(key);
        }

        Ok(())
    }

    /// Exact deterministic bytes that should be supplied to Xenia's generic artifact attestation.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ProvenanceCapsuleError> {
        self.validate_canonical()?;
        let mut bytes = Vec::new();
        put_str(&mut bytes, GENERATIVITY_PROVENANCE_CAPSULE_SCHEMA);
        put_str(&mut bytes, &self.assessment_subject_id);

        put_u64(&mut bytes, self.evidence.len() as u64);
        for evidence in &self.evidence {
            put_str(&mut bytes, &evidence.evidence_id);
            put_str(&mut bytes, &evidence.envelope_digest);
        }

        put_u64(&mut bytes, self.dimension_bindings.len() as u64);
        for binding in &self.dimension_bindings {
            bytes.push(dimension_tag(binding.dimension));
            put_u64(&mut bytes, binding.qualified_evidence_ids.len() as u64);
            for evidence_id in &binding.qualified_evidence_ids {
                put_str(&mut bytes, evidence_id);
            }
        }

        put_u64(&mut bytes, self.lineage_declarations.len() as u64);
        for declaration in &self.lineage_declarations {
            put_str(&mut bytes, &declaration.evidence_id);
            put_str(&mut bytes, &declaration.method_lineage_id);
            put_u64(&mut bytes, declaration.source_lineage_ids.len() as u64);
            for source in &declaration.source_lineage_ids {
                put_str(&mut bytes, source);
            }
        }

        put_u64(&mut bytes, self.ancestry_edges.len() as u64);
        for edge in &self.ancestry_edges {
            bytes.push(domain_tag(edge.domain));
            put_str(&mut bytes, &edge.child_id);
            put_str(&mut bytes, &edge.parent_id);
        }
        Ok(bytes)
    }

    /// BLAKE3 identity over [`Self::canonical_bytes`].
    pub fn canonical_digest(&self) -> Result<String, ProvenanceCapsuleError> {
        Ok(blake3::hash(&self.canonical_bytes()?).to_hex().to_string())
    }
}

fn normalize_dimension_bindings(
    bindings: &[DimensionEvidenceBinding],
    assessment_evidence_ids: &HashSet<&str>,
    qualified_by_id: &HashMap<String, String>,
) -> Result<Vec<ProvenanceDimensionBinding>, ProvenanceCapsuleError> {
    let mut by_dimension = HashMap::new();
    for binding in bindings {
        if by_dimension.insert(binding.dimension, binding).is_some() {
            return Err(ProvenanceCapsuleError::DuplicateDimensionBinding(
                binding.dimension,
            ));
        }
    }
    if by_dimension.len() != GenerativityDimension::ALL.len() {
        return Err(ProvenanceCapsuleError::IncompleteDimensionBindings {
            count: by_dimension.len(),
        });
    }

    let mut normalized = Vec::with_capacity(GenerativityDimension::ALL.len());
    for dimension in GenerativityDimension::ALL {
        let binding = by_dimension
            .get(&dimension)
            .ok_or(ProvenanceCapsuleError::MissingDimensionBinding(dimension))?;
        if binding.evidence_ids.len() > MAX_PROVENANCE_BINDING_IDS_PER_DIMENSION {
            return Err(ProvenanceCapsuleError::TooManyBindingEvidenceIds {
                dimension,
                count: binding.evidence_ids.len(),
                max: MAX_PROVENANCE_BINDING_IDS_PER_DIMENSION,
            });
        }
        let mut local_seen = HashSet::new();
        let mut qualified = Vec::new();
        for evidence_id in &binding.evidence_ids {
            if !local_seen.insert(evidence_id.as_str()) {
                return Err(ProvenanceCapsuleError::DuplicateBindingEvidenceId {
                    dimension,
                    evidence_id: evidence_id.clone(),
                });
            }
            if !assessment_evidence_ids.contains(evidence_id.as_str()) {
                return Err(ProvenanceCapsuleError::UnknownAssessmentEvidenceId {
                    dimension,
                    evidence_id: evidence_id.clone(),
                });
            }
            if qualified_by_id.contains_key(evidence_id) {
                qualified.push(evidence_id.clone());
            }
        }
        qualified.sort();
        normalized.push(ProvenanceDimensionBinding {
            dimension,
            qualified_evidence_ids: qualified,
        });
    }
    normalized.sort_by_key(|binding| dimension_tag(binding.dimension));
    Ok(normalized)
}

fn normalize_used_declarations(
    declarations: &[EvidenceLineageDeclaration],
    used_ids: &HashSet<String>,
) -> Result<Vec<EvidenceLineageDeclaration>, ProvenanceCapsuleError> {
    if used_ids.is_empty() {
        return Err(ProvenanceCapsuleError::NoQualifiedEvidence);
    }
    let mut by_id = HashMap::new();
    for declaration in declarations {
        if !used_ids.contains(&declaration.evidence_id) {
            continue;
        }
        validate_declaration(declaration)?;
        let mut normalized = declaration.clone();
        normalized.source_lineage_ids.sort();
        if by_id
            .insert(normalized.evidence_id.clone(), normalized)
            .is_some()
        {
            return Err(ProvenanceCapsuleError::DuplicateDeclaration(
                declaration.evidence_id.clone(),
            ));
        }
    }

    for evidence_id in used_ids {
        if !by_id.contains_key(evidence_id) {
            return Err(ProvenanceCapsuleError::MissingDeclaration(
                evidence_id.clone(),
            ));
        }
    }
    let mut result = by_id.into_values().collect::<Vec<_>>();
    result.sort_by(|a, b| a.evidence_id.cmp(&b.evidence_id));
    Ok(result)
}

fn normalize_reachable_ancestry(
    edges: &[LineageAncestryEdge],
    declarations: &[EvidenceLineageDeclaration],
) -> Result<Vec<LineageAncestryEdge>, ProvenanceCapsuleError> {
    let mut edge_indexes: HashMap<LineageDomain, HashMap<&str, Vec<&LineageAncestryEdge>>> =
        HashMap::new();
    for edge in edges {
        edge_indexes
            .entry(edge.domain)
            .or_default()
            .entry(edge.child_id.as_str())
            .or_default()
            .push(edge);
    }

    let method_starts = declarations
        .iter()
        .map(|declaration| declaration.method_lineage_id.clone())
        .collect::<HashSet<_>>();
    let source_starts = declarations
        .iter()
        .flat_map(|declaration| declaration.source_lineage_ids.iter().cloned())
        .collect::<HashSet<_>>();

    let mut selected = Vec::new();
    collect_reachable_edges(
        LineageDomain::Method,
        &method_starts,
        &edge_indexes,
        &mut selected,
    )?;
    collect_reachable_edges(
        LineageDomain::Source,
        &source_starts,
        &edge_indexes,
        &mut selected,
    )?;

    selected.sort_by(|a, b| {
        (domain_tag(a.domain), a.child_id.as_str(), a.parent_id.as_str()).cmp(&(
            domain_tag(b.domain),
            b.child_id.as_str(),
            b.parent_id.as_str(),
        ))
    });
    Ok(selected)
}

fn collect_reachable_edges(
    domain: LineageDomain,
    starts: &HashSet<String>,
    indexes: &HashMap<LineageDomain, HashMap<&str, Vec<&LineageAncestryEdge>>>,
    selected: &mut Vec<LineageAncestryEdge>,
) -> Result<(), ProvenanceCapsuleError> {
    let mut stack = starts.iter().cloned().collect::<Vec<_>>();
    let mut visited_nodes = HashSet::new();
    let mut seen_edges = HashSet::new();

    while let Some(node) = stack.pop() {
        if !visited_nodes.insert(node.clone()) {
            continue;
        }
        for edge in indexes
            .get(&domain)
            .and_then(|index| index.get(node.as_str()))
            .into_iter()
            .flatten()
        {
            validate_edge(edge)?;
            let key = (edge.child_id.as_str(), edge.parent_id.as_str());
            if !seen_edges.insert(key) {
                return Err(ProvenanceCapsuleError::DuplicateAncestryEdge {
                    domain,
                    child: edge.child_id.clone(),
                    parent: edge.parent_id.clone(),
                });
            }
            selected.push((*edge).clone());
            stack.push(edge.parent_id.clone());
        }
    }
    Ok(())
}

fn validate_declaration(
    declaration: &EvidenceLineageDeclaration,
) -> Result<(), ProvenanceCapsuleError> {
    validate_lineage_id("evidence_id", &declaration.evidence_id)?;
    validate_lineage_id("method_lineage_id", &declaration.method_lineage_id)?;
    if declaration.source_lineage_ids.len() > MAX_SOURCE_LINEAGES_PER_EVIDENCE {
        return Err(ProvenanceCapsuleError::TooManySourceLineages {
            evidence_id: declaration.evidence_id.clone(),
            count: declaration.source_lineage_ids.len(),
            max: MAX_SOURCE_LINEAGES_PER_EVIDENCE,
        });
    }
    let mut seen = HashSet::new();
    for source in &declaration.source_lineage_ids {
        validate_lineage_id("source_lineage_id", source)?;
        if !seen.insert(source.as_str()) {
            return Err(ProvenanceCapsuleError::DuplicateSourceLineage {
                evidence_id: declaration.evidence_id.clone(),
                lineage_id: source.clone(),
            });
        }
    }
    Ok(())
}

fn validate_edge(edge: &LineageAncestryEdge) -> Result<(), ProvenanceCapsuleError> {
    validate_lineage_id("child_id", &edge.child_id)?;
    validate_lineage_id("parent_id", &edge.parent_id)?;
    if edge.child_id == edge.parent_id {
        return Err(ProvenanceCapsuleError::SelfParent {
            domain: edge.domain,
            id: edge.child_id.clone(),
        });
    }
    Ok(())
}

fn validate_label(
    field: &'static str,
    value: &str,
    max_len: usize,
) -> Result<(), ProvenanceCapsuleError> {
    if value.trim().is_empty() {
        return Err(ProvenanceCapsuleError::EmptyField(field));
    }
    if value.len() > max_len {
        return Err(ProvenanceCapsuleError::FieldTooLong {
            field,
            max: max_len,
            found: value.len(),
        });
    }
    Ok(())
}

fn validate_lineage_id(
    field: &'static str,
    value: &str,
) -> Result<(), ProvenanceCapsuleError> {
    validate_label(field, value, MAX_LINEAGE_ID_LEN)
}

fn validate_digest(digest: &str) -> Result<(), ProvenanceCapsuleError> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ProvenanceCapsuleError::InvalidEnvelopeDigest);
    }
    Ok(())
}

fn put_str(bytes: &mut Vec<u8>, value: &str) {
    put_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn dimension_tag(dimension: GenerativityDimension) -> u8 {
    match dimension {
        GenerativityDimension::ImmediateUtility => 0,
        GenerativityDimension::EpistemicGain => 1,
        GenerativityDimension::OptionValue => 2,
        GenerativityDimension::Diversity => 3,
        GenerativityDimension::CapabilityGain => 4,
        GenerativityDimension::Diffusion => 5,
        GenerativityDimension::CommonsGain => 6,
        GenerativityDimension::Regeneration => 7,
        GenerativityDimension::DependencyRisk => 8,
        GenerativityDimension::ConcentrationRisk => 9,
        GenerativityDimension::IrreversibilityRisk => 10,
    }
}

fn domain_tag(domain: LineageDomain) -> u8 {
    match domain {
        LineageDomain::Method => 0,
        LineageDomain::Source => 1,
    }
}

#[derive(Debug)]
pub enum ProvenanceCapsuleError {
    UnsupportedSchema(String),
    EmptyField(&'static str),
    FieldTooLong {
        field: &'static str,
        max: usize,
        found: usize,
    },
    InvalidEnvelopeDigest,
    TooManyDeclarations {
        count: usize,
        max: usize,
    },
    TooManyAncestryEdges {
        count: usize,
        max: usize,
    },
    TooManyBindingEvidenceIds {
        dimension: GenerativityDimension,
        count: usize,
        max: usize,
    },
    TooManySourceLineages {
        evidence_id: String,
        count: usize,
        max: usize,
    },
    IncompleteDimensionBindings {
        count: usize,
    },
    MissingDimensionBinding(GenerativityDimension),
    DuplicateDimensionBinding(GenerativityDimension),
    DuplicateBindingEvidenceId {
        dimension: GenerativityDimension,
        evidence_id: String,
    },
    UnknownAssessmentEvidenceId {
        dimension: GenerativityDimension,
        evidence_id: String,
    },
    BindingReferencesUnknownQualifiedEvidence {
        dimension: GenerativityDimension,
        evidence_id: String,
    },
    NoQualifiedEvidence,
    DuplicateEvidenceRef(String),
    DuplicateDeclaration(String),
    MissingDeclaration(String),
    DeclarationEvidenceSetMismatch,
    DuplicateSourceLineage {
        evidence_id: String,
        lineage_id: String,
    },
    SelfParent {
        domain: LineageDomain,
        id: String,
    },
    DuplicateAncestryEdge {
        domain: LineageDomain,
        child: String,
        parent: String,
    },
    NonCanonicalOrdering(&'static str),
}

impl std::fmt::Display for ProvenanceCapsuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ProvenanceCapsuleError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use symthaea_evidence_plane::{EvidenceCounters, Expectation, RunEvidence, RunId};

    use super::*;
    use crate::exploration::evidence_binding::EvidencePlaneEnvelope;
    use crate::exploration::generativity::{
        GenerativityAssessment, GenerativityEstimate, GenerativityEvidence, GenerativityVector,
    };
    use crate::exploration::persisted_evidence::{
        PersistedEvidenceCapsule, PersistedGenerativityBundle,
    };

    fn vector() -> GenerativityVector {
        let e = |value| GenerativityEstimate::new(value, 0.9).unwrap();
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

    fn verified_bundle(subject: &str) -> VerifiedGenerativityBundle {
        let mut assessment = GenerativityAssessment::new(subject, "context", vector());
        let mut capsules = Vec::new();
        for (id, calls) in [("run:1", 4.0), ("run:2", 5.0)] {
            let run = run(id, calls);
            let envelope = EvidencePlaneEnvelope::from_run(&run).unwrap();
            envelope.bind_qualified(&mut assessment, None).unwrap();
            capsules.push(PersistedEvidenceCapsule::from_run(run, vec![]).unwrap());
        }
        assessment.evidence.push(GenerativityEvidence {
            evidence_id: "literature:background".into(),
            kind: "literature".into(),
            reference: Some("doi:example".into()),
            note: None,
        });
        PersistedGenerativityBundle::new(assessment, capsules)
            .verify()
            .unwrap()
    }

    fn bindings(reverse: bool) -> Vec<DimensionEvidenceBinding> {
        let ids = if reverse {
            vec![
                "literature:background".into(),
                "evidence-plane:run:2".into(),
                "evidence-plane:run:1".into(),
            ]
        } else {
            vec![
                "evidence-plane:run:1".into(),
                "evidence-plane:run:2".into(),
                "literature:background".into(),
            ]
        };
        let mut result = GenerativityDimension::ALL
            .iter()
            .map(|dimension| DimensionEvidenceBinding {
                dimension: *dimension,
                evidence_ids: ids.clone(),
            })
            .collect::<Vec<_>>();
        if reverse {
            result.reverse();
        }
        result
    }

    fn declarations(reverse: bool) -> Vec<EvidenceLineageDeclaration> {
        let mut result = vec![
            EvidenceLineageDeclaration {
                evidence_id: "evidence-plane:run:1".into(),
                method_lineage_id: "method:a".into(),
                source_lineage_ids: vec!["data:a2".into(), "data:a1".into()],
            },
            EvidenceLineageDeclaration {
                evidence_id: "evidence-plane:run:2".into(),
                method_lineage_id: "method:b".into(),
                source_lineage_ids: vec!["data:b".into()],
            },
            // Unused declaration should not enter the authenticated semantic capsule.
            EvidenceLineageDeclaration {
                evidence_id: "unused:run".into(),
                method_lineage_id: "unused:method".into(),
                source_lineage_ids: vec!["unused:data".into()],
            },
        ];
        if reverse {
            result.reverse();
        }
        result
    }

    fn edges(reverse: bool) -> Vec<LineageAncestryEdge> {
        let mut result = vec![
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "method:a".into(),
                parent_id: "method:root-a".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "method:b".into(),
                parent_id: "method:root-b".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Source,
                child_id: "data:a1".into(),
                parent_id: "data:root-a".into(),
            },
            LineageAncestryEdge {
                domain: LineageDomain::Source,
                child_id: "data:b".into(),
                parent_id: "data:root-b".into(),
            },
            // Unreachable graph noise is intentionally omitted.
            LineageAncestryEdge {
                domain: LineageDomain::Method,
                child_id: "unused:method".into(),
                parent_id: "unused:root".into(),
            },
        ];
        if reverse {
            result.reverse();
        }
        result
    }

    #[test]
    fn canonical_bytes_ignore_input_order_but_preserve_semantics() {
        let bundle = verified_bundle("proposal:42");
        let a = GenerativityProvenanceCapsule::from_analysis_inputs(
            &bundle,
            &bindings(false),
            &declarations(false),
            &edges(false),
        )
        .unwrap();
        let b = GenerativityProvenanceCapsule::from_analysis_inputs(
            &bundle,
            &bindings(true),
            &declarations(true),
            &edges(true),
        )
        .unwrap();
        assert_eq!(a.canonical_bytes().unwrap(), b.canonical_bytes().unwrap());
        assert_eq!(a.canonical_digest().unwrap(), b.canonical_digest().unwrap());
    }

    #[test]
    fn logical_run_id_is_bound_to_verified_content_digest() {
        let capsule = GenerativityProvenanceCapsule::from_analysis_inputs(
            &verified_bundle("proposal:42"),
            &bindings(false),
            &declarations(false),
            &edges(false),
        )
        .unwrap();
        assert_eq!(capsule.evidence.len(), 2);
        assert!(capsule
            .evidence
            .iter()
            .all(|evidence| evidence.envelope_digest.len() == 64));
    }

    #[test]
    fn generic_evidence_does_not_masquerade_as_lineage_qualified_evidence() {
        let capsule = GenerativityProvenanceCapsule::from_analysis_inputs(
            &verified_bundle("proposal:42"),
            &bindings(false),
            &declarations(false),
            &edges(false),
        )
        .unwrap();
        assert!(capsule
            .dimension_bindings
            .iter()
            .all(|binding| !binding.qualified_evidence_ids.contains(&"literature:background".to_string())));
    }

    #[test]
    fn unrelated_declarations_and_edges_are_omitted() {
        let capsule = GenerativityProvenanceCapsule::from_analysis_inputs(
            &verified_bundle("proposal:42"),
            &bindings(false),
            &declarations(false),
            &edges(false),
        )
        .unwrap();
        assert_eq!(capsule.lineage_declarations.len(), 2);
        assert!(!capsule
            .ancestry_edges
            .iter()
            .any(|edge| edge.child_id.starts_with("unused:")));
    }

    #[test]
    fn reachable_ancestry_change_changes_capsule_digest() {
        let bundle = verified_bundle("proposal:42");
        let a = GenerativityProvenanceCapsule::from_analysis_inputs(
            &bundle,
            &bindings(false),
            &declarations(false),
            &edges(false),
        )
        .unwrap();
        let mut changed_edges = edges(false);
        changed_edges[0].parent_id = "method:other-root".into();
        let b = GenerativityProvenanceCapsule::from_analysis_inputs(
            &bundle,
            &bindings(false),
            &declarations(false),
            &changed_edges,
        )
        .unwrap();
        assert_ne!(a.canonical_digest().unwrap(), b.canonical_digest().unwrap());
    }

    #[test]
    fn subject_is_committed_to_capsule_identity() {
        let a = GenerativityProvenanceCapsule::from_analysis_inputs(
            &verified_bundle("proposal:42"),
            &bindings(false),
            &declarations(false),
            &edges(false),
        )
        .unwrap();
        let b = GenerativityProvenanceCapsule::from_analysis_inputs(
            &verified_bundle("proposal:43"),
            &bindings(false),
            &declarations(false),
            &edges(false),
        )
        .unwrap();
        assert_ne!(a.canonical_digest().unwrap(), b.canonical_digest().unwrap());
    }

    #[test]
    fn missing_used_declaration_fails_closed() {
        let mut declarations = declarations(false);
        declarations.retain(|declaration| declaration.evidence_id != "evidence-plane:run:2");
        assert!(matches!(
            GenerativityProvenanceCapsule::from_analysis_inputs(
                &verified_bundle("proposal:42"),
                &bindings(false),
                &declarations,
                &edges(false),
            ),
            Err(ProvenanceCapsuleError::MissingDeclaration(id)) if id == "evidence-plane:run:2"
        ));
    }

    #[test]
    fn artifact_metadata_matches_xenia_bridge_contract() {
        assert_eq!(
            GENERATIVITY_PROVENANCE_ARTIFACT_DOMAIN,
            "symthaea-generativity-provenance"
        );
        assert_eq!(
            GENERATIVITY_PROVENANCE_CAPSULE_SCHEMA,
            "symthaea-generativity-provenance-capsule-v1"
        );
        assert_eq!(GENERATIVITY_PROVENANCE_DIGEST_ALGORITHM, "blake3-256");
    }
}
