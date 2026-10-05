//! Identity-aware HDC ontology adapter.
//!
//! This is the explicit follow-on from the closed-world N0 interlingua.
//! It separates stable concept/predicate identity, grounding provenance, and
//! HDC retrieval coordinates.
//!
//! The ontology/identity manifest is an upstream assertion. HDC does not infer
//! that two labels, languages, modalities, or observations denote the same
//! concept. Cross-label/cross-language equivalence therefore requires a
//! separately validated mapping into the same declared scheme.
//!
//! Grounding references are deliberately not part of the HDC atom key. A
//! held-out observation of the same declared concept can therefore use new
//! grounding identifiers without rebuilding the codebook.
//!
//! OOV handling is fail-closed: encoding rejects identities absent from the
//! frozen codebook, while decoding requires explicit score/margin policy.

use crate::hdc_codec::{
    quantize_continuous, HdcBinaryFrame, HdcCodecDescriptor, HdcQuantizationMetrics,
};
use crate::interlingua::compare_graphs;
use crate::{ConceptEdge, ConceptKind, ConceptNode, GroundedConceptGraph};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION: u16 = 1;
pub const HDC_ONTOLOGY_ADAPTER_ID: &str =
    "symthaea.hdc.grounded-ontology-interlingua-v1";
pub const HDC_ONTOLOGY_CODEBOOK_ID: &str =
    "symthaea.hdc.grounded-ontology-codebook-v1";
pub const HDC_ONTOLOGY_CODEBOOK_ALGORITHM: &str =
    "blake3-seeded-random-stable-identities-v1";
pub const HDC_ONTOLOGY_ROLE_REVISION: &str =
    "source-relation-target-stable-identity-v1";
pub const HDC_ONTOLOGY_EDGE_TARGET_PERMUTATION: usize = 1;
/// N0 resource bounds for identity-aware graph reconstruction.
pub const HDC_ONTOLOGY_MAX_NODES: usize = 256;
pub const HDC_ONTOLOGY_MAX_EDGES: usize = 2048;
/// Maximum local identity entries carried by a manifest.
pub const HDC_ONTOLOGY_MAX_MANIFEST_CONCEPTS: usize = 4_096;
pub const HDC_ONTOLOGY_MAX_MANIFEST_RELATIONS: usize = 4_096;
/// Maximum frozen codebook candidates ranked by the receiver.
pub const HDC_ONTOLOGY_MAX_CODEBOOK_CONCEPTS: usize = 4_096;
pub const HDC_ONTOLOGY_MAX_CODEBOOK_RELATIONS: usize = 4_096;
/// Maximum calibration/inference candidate universe for the N1 primitive.
pub const HDC_ONTOLOGY_MAX_CONFORMAL_CANDIDATES: usize = 4_096;
/// Maximum number of training graphs consumed by the N0 codebook constructor.
pub const HDC_ONTOLOGY_MAX_TRAINING_GRAPHS: usize = 4_096;
/// Maximum UTF-8 byte length of manifest identity strings.
pub const HDC_ONTOLOGY_MAX_ID_BYTES: usize = 4_096;
/// Maximum provenance references attached to one concept binding.
pub const HDC_ONTOLOGY_MAX_GROUNDING_IDS_PER_CONCEPT: usize = 256;
/// Hard ceiling on receiver-side edge candidates before ranking allocation.
pub const HDC_ONTOLOGY_MAX_EDGE_CANDIDATES: usize = 1_000_000;
/// Versioned HDC nonconformity score used by the isolated N1 conformal primitive.
pub const HDC_ONTOLOGY_CONFORMAL_SCORE_REVISION: &str = "cosine-nonconformity-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcConceptIdentityBinding {
    /// Local node identity for this observation/graph.
    pub node_id: String,
    /// Stable opaque concept identity within scheme_id.
    pub concept_id: String,
    pub kind: ConceptKind,
    /// Observation/unit/context references are provenance, not identity.
    pub grounding_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcRelationIdentityBinding {
    /// Local/lexical relation representation in the current graph.
    pub local_relation: String,
    /// Stable opaque predicate identity within scheme_id.
    pub relation_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcOntologyManifest {
    pub schema_version: u16,
    /// Identity namespace / concept scheme selected by the upstream adapter.
    pub scheme_id: String,
    /// Content hash or other stable identifier for the externally validated
    /// identity-mapping authority. HDC never manufactures this mapping.
    pub mapping_provenance_hash: String,
    pub concepts: Vec<HdcConceptIdentityBinding>,
    pub relations: Vec<HdcRelationIdentityBinding>,
}

impl HdcOntologyManifest {
    pub fn validates(&self) -> bool {
        self.schema_version == HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION
            && !self.scheme_id.trim().is_empty()
            && self.scheme_id.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && !self.mapping_provenance_hash.trim().is_empty()
            && self.mapping_provenance_hash.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && !self.concepts.is_empty()
            && self.concepts.len() <= HDC_ONTOLOGY_MAX_MANIFEST_CONCEPTS
            && !self.relations.is_empty()
            && self.relations.len() <= HDC_ONTOLOGY_MAX_MANIFEST_RELATIONS
            && !self.concepts.iter().any(|binding| {
                binding.node_id.trim().is_empty()
                    || binding.concept_id.trim().is_empty()
                    || binding.node_id.len() > HDC_ONTOLOGY_MAX_ID_BYTES
                    || binding.concept_id.len() > HDC_ONTOLOGY_MAX_ID_BYTES
                    || binding.grounding_ids.is_empty()
                    || binding.grounding_ids.len() > HDC_ONTOLOGY_MAX_GROUNDING_IDS_PER_CONCEPT
                    || binding.grounding_ids.iter().any(|id| {
                        id.trim().is_empty() || id.len() > HDC_ONTOLOGY_MAX_ID_BYTES
                    })
            })
            && !self.relations.iter().any(|binding| {
                binding.local_relation.trim().is_empty()
                    || binding.relation_id.trim().is_empty()
                    || binding.local_relation.len() > HDC_ONTOLOGY_MAX_ID_BYTES
                    || binding.relation_id.len() > HDC_ONTOLOGY_MAX_ID_BYTES
            })
            && unique_concept_node_ids(&self.concepts)
            && unique_graph_local_relations(&self.relations)
            && concept_kinds_consistent(&self.concepts)
    }

    pub fn canonicalized(&self) -> Self {
        let mut canonical = self.clone();
        for binding in &mut canonical.concepts {
            binding.grounding_ids.sort();
            binding.grounding_ids.dedup();
        }
        canonical.concepts.sort_by(|a, b| {
            a.node_id
                .cmp(&b.node_id)
                .then_with(|| a.concept_id.cmp(&b.concept_id))
                .then_with(|| kind_tag(&a.kind).cmp(kind_tag(&b.kind)))
        });
        canonical.relations.sort_by(|a, b| {
            a.local_relation
                .cmp(&b.local_relation)
                .then_with(|| a.relation_id.cmp(&b.relation_id))
        });
        canonical
    }

    pub fn manifest_hash(&self) -> String {
        let bytes = serde_json::to_vec(&self.canonicalized())
            .expect("ontology manifest is serializable");
        crate::content_hash(&bytes)
    }

    fn node_binding(&self, node_id: &str) -> Option<&HdcConceptIdentityBinding> {
        self.concepts.iter().find(|binding| binding.node_id == node_id)
    }

    fn relation_id(&self, local_relation: &str) -> Option<&str> {
        self.relations
            .iter()
            .find(|binding| binding.local_relation == local_relation)
            .map(|binding| binding.relation_id.as_str())
    }

    fn local_relation_for_id(&self, relation_id: &str) -> Result<&str, String> {
        let matches = self
            .relations
            .iter()
            .filter(|binding| binding.relation_id == relation_id)
            .map(|binding| binding.local_relation.as_str())
            .collect::<Vec<_>>();

        match matches.as_slice() {
            [local_relation] => Ok(*local_relation),
            [] => Err(format!(
                "receiver identity manifest lacks stable relation id {relation_id}"
            )),
            _ => Err(format!(
                "receiver identity manifest ambiguously maps stable relation id {relation_id} to {} local relations",
                matches.len()
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcOntologyCodebookDescriptor {
    pub schema_version: u16,
    pub codebook_id: String,
    pub adapter_id: String,
    pub generation_algorithm: String,
    pub role_revision: String,
    pub seed: u64,
    pub dimension: usize,
    pub scheme_id: String,
    /// Versioned hash of the upstream identity-mapping authority.
    pub mapping_provenance_hash: String,
    /// Hash of stable concept IDs plus their declared kind, not grounding IDs.
    pub concept_manifest_hash: String,
    /// Hash of stable predicate IDs, not local/lexical relation labels.
    pub relation_manifest_hash: String,
    /// Hash of training graph structure expressed only in stable identities.
    pub training_manifest_hash: String,
    pub concept_count: usize,
    pub relation_count: usize,
}

impl HdcOntologyCodebookDescriptor {
    pub fn validates(&self) -> bool {
        self.schema_version == HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION
            && self.codebook_id == HDC_ONTOLOGY_CODEBOOK_ID
            && self.adapter_id == HDC_ONTOLOGY_ADAPTER_ID
            && self.generation_algorithm == HDC_ONTOLOGY_CODEBOOK_ALGORITHM
            && self.role_revision == HDC_ONTOLOGY_ROLE_REVISION
            && self.dimension == HDC_DIMENSION
            && !self.scheme_id.trim().is_empty()
            && self.scheme_id.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && !self.mapping_provenance_hash.trim().is_empty()
            && self.mapping_provenance_hash.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && !self.codebook_id.trim().is_empty()
            && self.codebook_id.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && !self.adapter_id.trim().is_empty()
            && self.adapter_id.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && !self.generation_algorithm.trim().is_empty()
            && self.generation_algorithm.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && !self.role_revision.trim().is_empty()
            && self.role_revision.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && !self.concept_manifest_hash.trim().is_empty()
            && self.concept_manifest_hash.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && !self.relation_manifest_hash.trim().is_empty()
            && self.relation_manifest_hash.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && !self.training_manifest_hash.trim().is_empty()
            && self.training_manifest_hash.len() <= HDC_ONTOLOGY_MAX_ID_BYTES
            && self.concept_count > 0
            && self.concept_count <= HDC_ONTOLOGY_MAX_CODEBOOK_CONCEPTS
            && self.relation_count > 0
            && self.relation_count <= HDC_ONTOLOGY_MAX_CODEBOOK_RELATIONS
    }

    pub fn codebook_hash(&self) -> String {
        let bytes =
            serde_json::to_vec(self).expect("ontology codebook descriptor is serializable");
        crate::content_hash(&bytes)
    }
}

#[derive(Clone, Debug)]
struct StableConceptAtom {
    kind: ConceptKind,
    vector: ContinuousHV,
}

#[derive(Clone, Debug)]
pub struct HdcOntologyCodebook {
    descriptor: HdcOntologyCodebookDescriptor,
    concepts: BTreeMap<String, StableConceptAtom>,
    relations: BTreeMap<String, ContinuousHV>,
    role_node: ContinuousHV,
    role_source: ContinuousHV,
    role_relation: ContinuousHV,
    role_target: ContinuousHV,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcOntologyRepresentation {
    pub schema_version: u16,
    pub adapter_id: String,
    pub codec: HdcCodecDescriptor,
    pub codebook: HdcOntologyCodebookDescriptor,
    /// Exact grounding/identity sidecar used by the sender. This is
    /// provenance and is intentionally not a decoder compatibility key.
    pub source_manifest_hash: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_quantization: HdcQuantizationMetrics,
    pub edge_quantization: HdcQuantizationMetrics,
    pub node_frame: HdcBinaryFrame,
    pub edge_frame: HdcBinaryFrame,
}

impl HdcOntologyRepresentation {
    pub fn validates(&self) -> bool {
        self.schema_version == HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION
            && self.adapter_id == HDC_ONTOLOGY_ADAPTER_ID
            && self.codebook.validates()
            && self.codec.validates()
            && self.codec.codebook_hash.as_deref()
                == Some(self.codebook.codebook_hash().as_str())
            && !self.source_manifest_hash.trim().is_empty()
            && self.node_count > 0
            && self.edge_count > 0
            && self.node_count <= HDC_ONTOLOGY_MAX_NODES
            && self.edge_count <= HDC_ONTOLOGY_MAX_EDGES
            && self.node_quantization.validates()
            && self.edge_quantization.validates()
            && self.node_quantization.dimension == HDC_DIMENSION
            && self.edge_quantization.dimension == HDC_DIMENSION
            && self.node_frame.to_binary().is_ok()
            && self.edge_frame.to_binary().is_ok()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcOntologyDecodePolicy {
    pub min_score: f64,
    pub min_margin: f64,
}

/// N1 building block: finite-sample split-conformal threshold over a
/// pre-registered nonconformity score. This primitive intentionally does not
/// claim coverage by itself; the exchangeability, calibration/test separation,
/// score definition, and deployment protocol must be established by the N1
/// experiment before the theorem applies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcOntologyConformalCalibration {
    pub schema_version: u16,
    pub alpha: f64,
    pub threshold: f64,
    pub calibration_case_count: usize,
    pub codebook_hash: String,
    /// Hash of the exact sorted multiset of calibration nonconformity scores.
    pub calibration_scores_hash: String,
    /// Canonical hash of the complete candidate universe used for calibration.
    pub candidate_universe_hash: String,
    pub candidate_universe_size: usize,
    /// Exact nonconformity definition used to produce `threshold`.
    pub score_revision: String,
}

impl HdcOntologyConformalCalibration {
    pub fn from_nonconformity_scores(
        codebook_hash: impl Into<String>,
        candidate_universe: &[String],
        calibration_scores: &[f64],
        alpha: f64,
    ) -> Result<Self, String> {
        let codebook_hash = codebook_hash.into();
        let candidate_universe_hash = candidate_universe_hash(candidate_universe)?;
        let calibration_scores_hash = calibration_scores_hash(calibration_scores)?;
        if codebook_hash.trim().is_empty() {
            return Err("conformal calibration requires a codebook hash".into());
        }
        if calibration_scores.is_empty() {
            return Err("conformal calibration requires at least one score".into());
        }
        if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
            return Err("conformal alpha must be finite and within (0, 1)".into());
        }
        if calibration_scores
            .iter()
            .any(|score| !score.is_finite() || !(0.0..=1.0).contains(score))
        {
            return Err("conformal nonconformity scores must be finite and within [0, 1]".into());
        }

        let mut sorted = calibration_scores.to_vec();
        sorted.sort_by(f64::total_cmp);

        // Split-conformal finite-sample rank:
        // k = ceil((n + 1) * (1 - alpha)).
        // The rank is clipped to n because the calibration set has only n
        // observed scores. Small calibration sets therefore conservatively
        // collapse to their maximum score rather than fabricating a tighter
        // threshold.
        let n = sorted.len();
        let rank = (((n + 1) as f64) * (1.0 - alpha)).ceil() as usize;
        let rank = rank.clamp(1, n);

        Ok(Self {
            schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
            alpha,
            threshold: sorted[rank - 1],
            calibration_case_count: n,
            codebook_hash,
            calibration_scores_hash,
            candidate_universe_hash,
            candidate_universe_size: candidate_universe.len(),
            score_revision: HDC_ONTOLOGY_CONFORMAL_SCORE_REVISION.into(),
        })
    }

    /// Convert bounded cosine similarity into a [0, 1] nonconformity score.
    /// Similarity +1 is perfect conformity and -1 is maximal nonconformity.
    pub fn nonconformity_from_cosine(similarity: f64) -> Result<f64, String> {
        if !similarity.is_finite() || !(-1.0..=1.0).contains(&similarity) {
            return Err("HDC cosine similarity must be finite and within [-1, 1]".into());
        }
        Ok((1.0 - similarity) / 2.0)
    }

    /// Return the conformal prediction set over a deterministic candidate pool.
    /// The caller must establish the calibration/test exchangeability assumptions
    /// and bind the calibration to the same frozen codebook.
    pub fn prediction_set(
        &self,
        codebook_hash: &str,
        candidates: &[HdcOntologyRetrievalCandidate],
    ) -> Result<Vec<String>, String> {
        if !self.validates() {
            return Err("invalid HDC ontology conformal calibration".into());
        }
        if codebook_hash != self.codebook_hash {
            return Err("conformal prediction codebook hash does not match calibration".into());
        }
        let candidate_ids = candidates
            .iter()
            .map(|candidate| candidate.stable_id.clone())
            .collect::<Vec<_>>();
        if candidate_universe_hash(&candidate_ids)? != self.candidate_universe_hash
            || candidate_ids.len() != self.candidate_universe_size
        {
            return Err("conformal candidate universe does not match calibration scope".into());
        }
        let mut ids = BTreeSet::new();
        let mut selected = Vec::new();
        for candidate in candidates {
            if candidate.stable_id.trim().is_empty()
                || !ids.insert(candidate.stable_id.clone())
            {
                return Err("conformal candidate stable IDs must be unique and non-empty".into());
            }
            let nonconformity = Self::nonconformity_from_cosine(candidate.score)?;
            if self.accepts(nonconformity) {
                selected.push(candidate.stable_id.clone());
            }
        }
        selected.sort();
        Ok(selected)
    }

    pub fn accepts(&self, nonconformity: f64) -> bool {
        nonconformity.is_finite()
            && (0.0..=1.0).contains(&nonconformity)
            && nonconformity <= self.threshold
    }

    pub fn minimum_cases_for_non_max_threshold(alpha: f64) -> Result<usize, String> {
        if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
            return Err("conformal alpha must be finite and within (0, 1)".into());
        }
        Ok(((2.0 / alpha) - 1.0).ceil() as usize)
    }

    pub fn validates(&self) -> bool {
        self.schema_version == HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION
            && self.alpha.is_finite()
            && (0.0..1.0).contains(&self.alpha)
            && self.threshold.is_finite()
            && (0.0..=1.0).contains(&self.threshold)
            && self.calibration_case_count > 0
            && !self.codebook_hash.trim().is_empty()
            && !self.calibration_scores_hash.trim().is_empty()
            && !self.candidate_universe_hash.trim().is_empty()
            && self.candidate_universe_size > 0
            && self.score_revision == HDC_ONTOLOGY_CONFORMAL_SCORE_REVISION
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcOntologyEmpiricalCalibration {
    pub baseline: HdcOntologyDecodePolicy,
    pub codebook_hash: String,
    pub quantile: f64,
    pub calibrated_min_score: f64,
    pub calibrated_min_margin: f64,
    pub calibration_case_count: usize,
}

impl HdcOntologyEmpiricalCalibration {
    /// Build a conservative empirical policy floor from successful calibration
    /// cases. This is descriptive calibration only; it does not provide a
    /// distribution-free or conformal coverage guarantee.
    pub fn from_clean_metrics(
        baseline: HdcOntologyDecodePolicy,
        calibration_metrics: &[HdcOntologyRoundtripMetrics],
        lower_quantile: f64,
    ) -> Result<Self, String> {
        if !baseline.validates() {
            return Err("invalid baseline ontology HDC decode policy".into());
        }
        if calibration_metrics.is_empty() {
            return Err("empirical calibration requires at least one clean case".into());
        }
        if !lower_quantile.is_finite() || !(0.0..=1.0).contains(&lower_quantile) {
            return Err("calibration quantile must be finite and within [0, 1]".into());
        }
        if calibration_metrics.iter().any(|metrics| {
            !metrics.validates()
                || !metrics.structural_equivalence
                || !metrics.concept_identity_exact
                || !metrics.relation_identity_exact
                || metrics.node_precision != 1.0
                || metrics.node_recall != 1.0
                || metrics.edge_precision != 1.0
                || metrics.edge_recall != 1.0
        }) {
            return Err("empirical calibration accepts only clean exact calibration metrics".into());
        }

        let codebook_hash = calibration_metrics[0].codebook_hash.clone();
        if codebook_hash.trim().is_empty()
            || calibration_metrics
                .iter()
                .any(|metrics| metrics.codebook_hash != codebook_hash)
        {
            return Err(
                "empirical calibration requires one consistent decoder codebook hash".into(),
            );
        }

        fn lower_empirical_quantile(values: &[f64], q: f64) -> f64 {
            let mut values = values.to_vec();
            values.sort_by(f64::total_cmp);
            let index = ((values.len() - 1) as f64 * q).floor() as usize;
            values[index]
        }

        let score_floor = lower_empirical_quantile(
            &calibration_metrics
                .iter()
                .flat_map(|metrics| [metrics.node_min_selected_score, metrics.edge_min_selected_score])
                .collect::<Vec<_>>(),
            lower_quantile,
        );
        let margin_floor = lower_empirical_quantile(
            &calibration_metrics
                .iter()
                .flat_map(|metrics| [metrics.node_selection_margin, metrics.edge_selection_margin])
                .collect::<Vec<_>>(),
            lower_quantile,
        );

        Ok(Self {
            baseline,
            codebook_hash,
            quantile: lower_quantile,
            calibrated_min_score: baseline.min_score.max(score_floor),
            calibrated_min_margin: baseline.min_margin.max(margin_floor),
            calibration_case_count: calibration_metrics.len(),
        })
    }

    pub fn policy(&self) -> HdcOntologyDecodePolicy {
        HdcOntologyDecodePolicy {
            min_score: self.calibrated_min_score,
            min_margin: self.calibrated_min_margin,
        }
    }

    pub fn validates(&self) -> bool {
        self.baseline.validates()
            && !self.codebook_hash.trim().is_empty()
            && self.quantile.is_finite()
            && (0.0..=1.0).contains(&self.quantile)
            && self.calibrated_min_score.is_finite()
            && (-1.0..=1.0).contains(&self.calibrated_min_score)
            && self.calibrated_min_score >= self.baseline.min_score
            && self.calibrated_min_margin.is_finite()
            && self.calibrated_min_margin >= self.baseline.min_margin
            && self.calibration_case_count > 0
    }
}

impl HdcOntologyDecodePolicy {
    pub fn conservative_default() -> Self {
        Self {
            min_score: 0.20,
            min_margin: 0.05,
        }
    }

    pub fn validates(&self) -> bool {
        self.min_score.is_finite()
            && (-1.0..=1.0).contains(&self.min_score)
            && self.min_margin.is_finite()
            && self.min_margin >= 0.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcOntologyRetrievalCandidate {
    pub stable_id: String,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcOntologyDecodedGraph {
    pub graph: GroundedConceptGraph,
    /// Stable concept identity remains separate from transport-local node IDs.
    pub concept_ids_by_node: BTreeMap<String, String>,
    /// Stable predicate identity remains separate from rendered local labels.
    pub relation_ids_by_edge: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcOntologyRoundtripMetrics {
    pub schema_version: u16,
    pub adapter_id: String,
    pub codebook_hash: String,
    pub source_manifest_hash: String,
    pub concept_identity_exact: bool,
    pub relation_identity_exact: bool,
    pub node_precision: f64,
    pub node_recall: f64,
    pub edge_precision: f64,
    pub edge_recall: f64,
    pub structural_equivalence: bool,
    pub confidence_mae: f64,
    pub node_min_selected_score: f64,
    pub node_selection_margin: f64,
    pub edge_min_selected_score: f64,
    pub edge_selection_margin: f64,
}

impl HdcOntologyRoundtripMetrics {
    pub fn validates(&self) -> bool {
        self.schema_version == HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION
            && self.adapter_id == HDC_ONTOLOGY_ADAPTER_ID
            && !self.codebook_hash.trim().is_empty()
            && !self.source_manifest_hash.trim().is_empty()
            && self.node_precision.is_finite()
            && self.node_recall.is_finite()
            && self.edge_precision.is_finite()
            && self.edge_recall.is_finite()
            && self.confidence_mae.is_finite()
            && (0.0..=1.0).contains(&self.node_precision)
            && (0.0..=1.0).contains(&self.node_recall)
            && (0.0..=1.0).contains(&self.edge_precision)
            && (0.0..=1.0).contains(&self.edge_recall)
            && self.confidence_mae >= 0.0
            && self.node_min_selected_score.is_finite()
            && (-1.0..=1.0).contains(&self.node_min_selected_score)
            && self.node_selection_margin.is_finite()
            && self.edge_min_selected_score.is_finite()
            && (-1.0..=1.0).contains(&self.edge_min_selected_score)
            && self.edge_selection_margin.is_finite()
    }
}

impl HdcOntologyCodebook {
    pub fn from_training_graphs(
        seed: u64,
        training_graphs: &[GroundedConceptGraph],
        training_manifest: &HdcOntologyManifest,
    ) -> Result<Self, String> {
        validate_manifest(training_manifest)?;
        if training_graphs.is_empty() {
            return Err("ontology HDC codebook requires training graphs".into());
        }
        if training_graphs.len() > HDC_ONTOLOGY_MAX_TRAINING_GRAPHS {
            return Err(format!(
                "ontology HDC training graph budget exceeded: {} > {}",
                training_graphs.len(),
                HDC_ONTOLOGY_MAX_TRAINING_GRAPHS
            ));
        }

        let mut concepts = BTreeMap::<String, ConceptKind>::new();
        let mut relations = BTreeSet::<String>::new();
        let mut training_signatures = Vec::with_capacity(training_graphs.len());

        for graph in training_graphs {
            let stable = stable_graph_signature(graph, training_manifest)?;
            training_signatures.push(stable.signature_hash);
            for (concept_id, kind) in stable.concepts {
                if let Some(existing) = concepts.get(&concept_id) {
                    if existing != &kind {
                        return Err(format!(
                            "stable concept id {concept_id} is bound to inconsistent kinds"
                        ));
                    }
                } else {
                    concepts.insert(concept_id, kind);
                }
            }
            relations.extend(stable.relation_ids);
        }

        if concepts.is_empty() || relations.is_empty() {
            return Err("ontology HDC codebook requires concepts and relations".into());
        }

        training_signatures.sort();
        let concept_manifest = concepts
            .iter()
            .map(|(concept_id, kind)| (concept_id.clone(), kind_tag(kind).to_string()))
            .collect::<Vec<_>>();
        let relation_manifest = relations.iter().cloned().collect::<Vec<_>>();

        let concept_manifest_hash = hash_json(&concept_manifest)?;
        let relation_manifest_hash = hash_json(&relation_manifest)?;
        let training_manifest_hash = hash_json(&training_signatures)?;

        let descriptor = HdcOntologyCodebookDescriptor {
            schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
            codebook_id: HDC_ONTOLOGY_CODEBOOK_ID.into(),
            adapter_id: HDC_ONTOLOGY_ADAPTER_ID.into(),
            generation_algorithm: HDC_ONTOLOGY_CODEBOOK_ALGORITHM.into(),
            role_revision: HDC_ONTOLOGY_ROLE_REVISION.into(),
            seed,
            dimension: HDC_DIMENSION,
            scheme_id: training_manifest.scheme_id.clone(),
            mapping_provenance_hash: training_manifest.mapping_provenance_hash.clone(),
            concept_manifest_hash,
            relation_manifest_hash,
            training_manifest_hash,
            concept_count: concepts.len(),
            relation_count: relations.len(),
        };

        let concept_atoms = concepts
            .into_iter()
            .map(|(concept_id, kind)| {
                (
                    concept_id.clone(),
                    StableConceptAtom {
                        kind,
                        vector: derive_atom_vector(seed, "concept", &concept_id),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();

        let relation_atoms = relations
            .into_iter()
            .map(|relation_id| {
                (
                    relation_id.clone(),
                    derive_atom_vector(seed, "relation", &relation_id),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let codebook = Self {
            descriptor,
            concepts: concept_atoms,
            relations: relation_atoms,
            role_node: derive_atom_vector(seed, "role", "node"),
            role_source: derive_atom_vector(seed, "role", "source"),
            role_relation: derive_atom_vector(seed, "role", "relation"),
            role_target: derive_atom_vector(seed, "role", "target"),
        };

        if !codebook.descriptor.validates() {
            return Err("constructed ontology codebook descriptor is invalid".into());
        }
        Ok(codebook)
    }

    pub fn descriptor(&self) -> &HdcOntologyCodebookDescriptor {
        &self.descriptor
    }

    pub fn codebook_hash(&self) -> String {
        self.descriptor.codebook_hash()
    }

    pub fn concept_ids(&self) -> Vec<String> {
        self.concepts.keys().cloned().collect()
    }

    pub fn relation_ids(&self) -> Vec<String> {
        self.relations.keys().cloned().collect()
    }

    pub fn encode_graph(
        &self,
        graph: &GroundedConceptGraph,
        manifest: &HdcOntologyManifest,
    ) -> Result<HdcOntologyRepresentation, String> {
        validate_manifest_compatibility(&self.descriptor, manifest)?;
        let stable = stable_graph_signature(graph, manifest)?;

        for concept_id in stable.concepts.iter().map(|(id, _)| id) {
            if !self.concepts.contains_key(concept_id) {
                return Err(format!(
                    "stable concept id is OOV for frozen codebook: {concept_id}"
                ));
            }
        }
        for relation_id in &stable.relation_ids {
            if !self.relations.contains_key(relation_id) {
                return Err(format!(
                    "stable relation id is OOV for frozen codebook: {relation_id}"
                ));
            }
        }

        let node_vectors = stable
            .concepts
            .iter()
            .map(|(concept_id, _)| self.node_vector(concept_id))
            .collect::<Result<Vec<_>, _>>()?;
        let edge_vectors = stable
            .edges
            .iter()
            .map(|(source, relation, target)| self.edge_vector(source, relation, target))
            .collect::<Result<Vec<_>, _>>()?;

        let node_refs = node_vectors.iter().collect::<Vec<_>>();
        let edge_refs = edge_vectors.iter().collect::<Vec<_>>();
        let node_bundle = ContinuousHV::bundle(&node_refs);
        let edge_bundle = ContinuousHV::bundle(&edge_refs);

        let (node_binary, node_quantization) = quantize_continuous(&node_bundle)?;
        let (edge_binary, edge_quantization) = quantize_continuous(&edge_bundle)?;

        let mut codec = HdcCodecDescriptor::v1();
        codec.encoder_revision = Some(HDC_ONTOLOGY_ADAPTER_ID.into());
        codec.codebook_hash = Some(self.codebook_hash());

        let representation = HdcOntologyRepresentation {
            schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
            adapter_id: HDC_ONTOLOGY_ADAPTER_ID.into(),
            codec,
            codebook: self.descriptor.clone(),
            source_manifest_hash: manifest.manifest_hash(),
            node_count: stable.concepts.len(),
            edge_count: stable.edges.len(),
            node_quantization,
            edge_quantization,
            node_frame: HdcBinaryFrame::from_binary(&node_binary),
            edge_frame: HdcBinaryFrame::from_binary(&edge_binary),
        };

        if !representation.validates() {
            return Err("ontology HDC representation failed validation".into());
        }
        Ok(representation)
    }

    pub fn decode_graph_with_policy(
        &self,
        representation: &HdcOntologyRepresentation,
        source_manifest: &HdcOntologyManifest,
        receiver_manifest: &HdcOntologyManifest,
        policy: HdcOntologyDecodePolicy,
    ) -> Result<HdcOntologyDecodedGraph, String> {
        if !policy.validates() {
            return Err("invalid ontology HDC decode policy".into());
        }
        self.validate_representation(representation)?;
        validate_manifest_compatibility(&self.descriptor, source_manifest)?;
        if representation.source_manifest_hash != source_manifest.manifest_hash() {
            return Err("source ontology manifest hash mismatch".into());
        }
        validate_manifest_compatibility(&self.descriptor, receiver_manifest)?;

        let node_bundle = representation.node_frame.to_binary()?.to_continuous();
        let edge_bundle = representation.edge_frame.to_binary()?.to_continuous();

        let mut node_candidates = self
            .concepts
            .keys()
            .map(|concept_id| {
                Ok(HdcOntologyRetrievalCandidate {
                    stable_id: concept_id.clone(),
                    score: full_cosine_similarity(
                        &node_bundle,
                        &self.node_vector(concept_id)?,
                    ),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        sort_candidates(&mut node_candidates);

        if representation.node_count > node_candidates.len() {
            return Err("representation requests more nodes than the codebook contains".into());
        }

        let selected_nodes =
            select_with_policy(&node_candidates, representation.node_count, policy)?;

        for concept_id in &selected_nodes {
            if !source_manifest
                .concepts
                .iter()
                .any(|binding| binding.concept_id == *concept_id)
            {
                return Err(format!(
                    "decoded stable concept is absent from source identity manifest: {concept_id}"
                ));
            }
            if !receiver_manifest
                .concepts
                .iter()
                .any(|binding| binding.concept_id == *concept_id)
            {
                return Err(format!(
                    "decoded stable concept is absent from receiver identity manifest: {concept_id}"
                ));
            }
        }

        let candidate_count = edge_candidate_count(selected_nodes.len(), self.relations.len())?;
        if candidate_count > HDC_ONTOLOGY_MAX_EDGE_CANDIDATES {
            return Err(format!(
                "ontology HDC edge candidate budget exceeded: {candidate_count} > {HDC_ONTOLOGY_MAX_EDGE_CANDIDATES}"
            ));
        }

        let mut edge_candidates = Vec::with_capacity(candidate_count);
        for source in &selected_nodes {
            for relation_id in self.relations.keys() {
                for target in &selected_nodes {
                    let candidate = self.edge_vector(source, relation_id, target)?;
                    edge_candidates.push(HdcOntologyRetrievalCandidate {
                        stable_id: serde_json::to_string(&(source, relation_id, target))
                            .map_err(|error| error.to_string())?,
                        score: full_cosine_similarity(&edge_bundle, &candidate),
                    });
                }
            }
        }
        sort_candidates(&mut edge_candidates);

        if representation.edge_count > edge_candidates.len() {
            return Err(
                "representation requests more edges than the decoded node set permits"
                    .into(),
            );
        }

        let selected_edges = select_with_policy(
            &edge_candidates,
            representation.edge_count,
            policy,
        )?
        .into_iter()
        .map(|key| parse_edge_candidate(&key))
        .collect::<Result<Vec<_>, _>>()?;

        let mut source_bindings_by_concept = BTreeMap::new();
        let mut receiver_bindings_by_concept = BTreeMap::new();
        for concept_id in &selected_nodes {
            let source_bindings = source_manifest
                .concepts
                .iter()
                .filter(|binding| binding.concept_id == *concept_id)
                .collect::<Vec<_>>();
            if source_bindings.len() != 1 {
                return Err(format!(
                    "source identity manifest must contain exactly one node binding for stable concept {concept_id}, found {}",
                    source_bindings.len()
                ));
            }

            let receiver_bindings = receiver_manifest
                .concepts
                .iter()
                .filter(|binding| binding.concept_id == *concept_id)
                .collect::<Vec<_>>();
            if receiver_bindings.len() != 1 {
                return Err(format!(
                    "receiver identity manifest must contain exactly one node binding for stable concept {concept_id}, found {}",
                    receiver_bindings.len()
                ));
            }

            source_bindings_by_concept.insert(concept_id.clone(), source_bindings[0]);
            receiver_bindings_by_concept.insert(concept_id.clone(), receiver_bindings[0]);
        }

        let mut nodes = Vec::with_capacity(selected_nodes.len());
        let mut concept_ids_by_node = BTreeMap::new();
        for concept_id in &selected_nodes {
            let source_binding = source_bindings_by_concept
                .get(concept_id)
                .copied()
                .ok_or_else(|| format!("missing source binding for concept {concept_id}"))?;
            let receiver_binding = receiver_bindings_by_concept
                .get(concept_id)
                .copied()
                .ok_or_else(|| format!("missing receiver binding for concept {concept_id}"))?;
            let atom = self
                .concepts
                .get(concept_id)
                .ok_or_else(|| format!("missing concept atom {concept_id}"))?;

            if atom.kind != source_binding.kind || atom.kind != receiver_binding.kind {
                return Err(format!(
                    "identity manifest kind mismatch for stable concept {concept_id}"
                ));
            }

            nodes.push(ConceptNode {
                id: receiver_binding.node_id.clone(),
                kind: receiver_binding.kind.clone(),
                label: None,
                // Preserve source provenance; receiver identity only determines
                // the local graph identifier used for reconstruction.
                grounded_by: source_binding.grounding_ids.clone(),
                confidence: 1.0,
            });
            concept_ids_by_node.insert(receiver_binding.node_id.clone(), concept_id.clone());
        }

        let node_ids_by_concept = receiver_bindings_by_concept
            .iter()
            .map(|(concept_id, binding)| (concept_id.clone(), binding.node_id.clone()))
            .collect::<BTreeMap<_, _>>();

        let mut edges = Vec::with_capacity(selected_edges.len());
        let mut relation_ids_by_edge = Vec::with_capacity(selected_edges.len());
        for (source, relation_id, target) in selected_edges {
            let source_id = node_ids_by_concept
                .get(&source)
                .cloned()
                .ok_or_else(|| format!("edge source concept not selected: {source}"))?;
            let target_id = node_ids_by_concept
                .get(&target)
                .cloned()
                .ok_or_else(|| format!("edge target concept not selected: {target}"))?;
            let local_relation = receiver_manifest.local_relation_for_id(&relation_id)?;

            edges.push(ConceptEdge {
                source: source_id,
                relation: local_relation.to_string(),
                target: target_id,
                evidence_ids: Vec::new(),
                confidence: 1.0,
            });
            relation_ids_by_edge.push(relation_id);
        }

        Ok(HdcOntologyDecodedGraph {
            graph: GroundedConceptGraph { nodes, edges },
            concept_ids_by_node,
            relation_ids_by_edge,
        })
    }

    pub fn measure_roundtrip(
        &self,
        expected: &GroundedConceptGraph,
        representation: &HdcOntologyRepresentation,
        source_manifest: &HdcOntologyManifest,
        receiver_manifest: &HdcOntologyManifest,
        expected_concept_ids: &BTreeMap<String, String>,
        expected_relation_ids: &[String],
        policy: HdcOntologyDecodePolicy,
    ) -> Result<HdcOntologyRoundtripMetrics, String> {
        let decoded = self.decode_graph_with_policy(
            representation,
            source_manifest,
            receiver_manifest,
            policy,
        )?;
        // The receiver is allowed to render a stable predicate with a different
        // local/lexical label. Structural metrics at this identity-aware layer
        // therefore compare the stable ontology identity + topology, not the
        // receiver's local spelling.
        let expected_identity_graph =
            identity_normalized_graph(expected, source_manifest)?;
        let observed_identity_graph =
            identity_normalized_graph(&decoded.graph, receiver_manifest)?;
        let interlingua = compare_graphs(&expected_identity_graph, &observed_identity_graph)?;
        let concept_identity_exact = decoded.concept_ids_by_node == *expected_concept_ids;
        let mut observed_relation_ids = decoded.relation_ids_by_edge.clone();
        observed_relation_ids.sort();
        let mut expected_relation_ids = expected_relation_ids.to_vec();
        expected_relation_ids.sort();
        let relation_identity_exact = observed_relation_ids == expected_relation_ids;

        let node_candidates = self.rank_nodes(representation)?;
        let edge_candidates = self.rank_edges(
            representation,
            &decoded.concept_ids_by_node,
        )?;

        let (node_min_selected_score, node_selection_margin) =
            selection_stats(&node_candidates, representation.node_count)?;
        let (edge_min_selected_score, edge_selection_margin) =
            selection_stats(&edge_candidates, representation.edge_count)?;

        let metrics = HdcOntologyRoundtripMetrics {
            schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
            adapter_id: HDC_ONTOLOGY_ADAPTER_ID.into(),
            codebook_hash: self.codebook_hash(),
            source_manifest_hash: representation.source_manifest_hash.clone(),
            concept_identity_exact,
            relation_identity_exact,
            node_precision: interlingua.node_precision,
            node_recall: interlingua.node_recall,
            edge_precision: interlingua.edge_precision,
            edge_recall: interlingua.edge_recall,
            structural_equivalence: interlingua.structural_equivalence,
            confidence_mae: interlingua.confidence_mae,
            node_min_selected_score,
            node_selection_margin,
            edge_min_selected_score,
            edge_selection_margin,
        };

        if !metrics.validates() {
            return Err("ontology roundtrip metrics failed validation".into());
        }
        Ok(metrics)
    }

    fn validate_representation(
        &self,
        representation: &HdcOntologyRepresentation,
    ) -> Result<(), String> {
        if !representation.validates() {
            return Err("invalid ontology HDC representation".into());
        }
        if representation.codebook != self.descriptor {
            return Err("ontology HDC codebook descriptor mismatch".into());
        }
        if representation.codec.codebook_hash.as_deref() != Some(self.codebook_hash().as_str()) {
            return Err("ontology HDC codebook hash mismatch".into());
        }
        Ok(())
    }

    fn rank_nodes(
        &self,
        representation: &HdcOntologyRepresentation,
    ) -> Result<Vec<HdcOntologyRetrievalCandidate>, String> {
        let bundle = representation.node_frame.to_binary()?.to_continuous();
        let mut candidates = self
            .concepts
            .keys()
            .map(|concept_id| {
                Ok(HdcOntologyRetrievalCandidate {
                    stable_id: concept_id.clone(),
                    score: full_cosine_similarity(&bundle, &self.node_vector(concept_id)?),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        sort_candidates(&mut candidates);
        Ok(candidates)
    }

    fn rank_edges(
        &self,
        representation: &HdcOntologyRepresentation,
        decoded_concepts: &BTreeMap<String, String>,
    ) -> Result<Vec<HdcOntologyRetrievalCandidate>, String> {
        let edge_bundle = representation.edge_frame.to_binary()?.to_continuous();
        let selected_nodes = decoded_concepts.values().cloned().collect::<Vec<_>>();
        let candidate_count = edge_candidate_count(selected_nodes.len(), self.relations.len())?;
        if candidate_count > HDC_ONTOLOGY_MAX_EDGE_CANDIDATES {
            return Err(format!(
                "ontology HDC edge candidate budget exceeded: {candidate_count} > {HDC_ONTOLOGY_MAX_EDGE_CANDIDATES}"
            ));
        }
        let mut candidates = Vec::with_capacity(candidate_count);
        for source in &selected_nodes {
            for relation_id in self.relations.keys() {
                for target in &selected_nodes {
                    candidates.push(HdcOntologyRetrievalCandidate {
                        stable_id: serde_json::to_string(&(source, relation_id, target))
                            .map_err(|error| error.to_string())?,
                        score: full_cosine_similarity(
                            &edge_bundle,
                            &self.edge_vector(source, relation_id, target)?,
                        ),
                    });
                }
            }
        }
        sort_candidates(&mut candidates);
        Ok(candidates)
    }

    fn node_vector(&self, concept_id: &str) -> Result<ContinuousHV, String> {
        self.concepts
            .get(concept_id)
            .map(|atom| self.role_node.bind(&atom.vector))
            .ok_or_else(|| format!("unknown stable concept id: {concept_id}"))
    }

    fn edge_vector(
        &self,
        source: &str,
        relation_id: &str,
        target: &str,
    ) -> Result<ContinuousHV, String> {
        let source_hv = self
            .concepts
            .get(source)
            .map(|atom| atom.vector.clone())
            .ok_or_else(|| format!("unknown stable source concept: {source}"))?;
        let relation_hv = self
            .relations
            .get(relation_id)
            .cloned()
            .ok_or_else(|| format!("unknown stable relation id: {relation_id}"))?;
        let target_hv = self
            .concepts
            .get(target)
            .map(|atom| atom.vector.clone())
            .ok_or_else(|| format!("unknown stable target concept: {target}"))?;

        let source_bound = self.role_source.bind(&source_hv);
        let relation_bound = self.role_relation.bind(&relation_hv);
        let target_bound = self
            .role_target
            .bind(&target_hv.permute(HDC_ONTOLOGY_EDGE_TARGET_PERMUTATION));

        Ok(source_bound
            .bind(&relation_bound)
            .bind(&target_bound))
    }
}

#[derive(Clone, Debug)]
struct StableGraphSignature {
    signature_hash: String,
    concepts: Vec<(String, ConceptKind)>,
    relation_ids: Vec<String>,
    edges: Vec<(String, String, String)>,
}

fn stable_graph_signature(
    graph: &GroundedConceptGraph,
    manifest: &HdcOntologyManifest,
) -> Result<StableGraphSignature, String> {
    validate_graph_shape(graph)?;
    validate_manifest(manifest)?;

    let mut concepts = Vec::with_capacity(graph.nodes.len());
    let mut node_to_concept = BTreeMap::new();
    let mut seen_concepts = BTreeSet::new();

    for node in &graph.nodes {
        let binding = manifest
            .node_binding(&node.id)
            .ok_or_else(|| format!("manifest lacks node identity binding: {}", node.id))?;
        if binding.kind != node.kind {
            return Err(format!(
                "manifest kind mismatch for node {}: expected {:?}, got {:?}",
                node.id, node.kind, binding.kind
            ));
        }
        if !seen_concepts.insert(binding.concept_id.clone()) {
            return Err(format!(
                "graph contains duplicate stable concept identity: {}",
                binding.concept_id
            ));
        }
        node_to_concept.insert(node.id.clone(), binding.concept_id.clone());
        concepts.push((binding.concept_id.clone(), binding.kind.clone()));
    }

    concepts.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| kind_tag(&a.1).cmp(kind_tag(&b.1)))
    });

    let mut edges = Vec::with_capacity(graph.edges.len());
    let mut relation_ids = Vec::with_capacity(graph.edges.len());

    for edge in &graph.edges {
        let source = node_to_concept
            .get(&edge.source)
            .cloned()
            .ok_or_else(|| format!("missing stable source node: {}", edge.source))?;
        let target = node_to_concept
            .get(&edge.target)
            .cloned()
            .ok_or_else(|| format!("missing stable target node: {}", edge.target))?;
        let relation_id = manifest
            .relation_id(&edge.relation)
            .ok_or_else(|| {
                format!(
                    "manifest lacks relation identity binding: {}",
                    edge.relation
                )
            })?
            .to_string();
        relation_ids.push(relation_id.clone());
        edges.push((source, relation_id, target));
    }
    edges.sort();
    relation_ids.sort();

    if edges.windows(2).any(|window| window[0] == window[1]) {
        return Err("graph contains duplicate stable relation edge".into());
    }

    let signature_hash = hash_json(&(&concepts, &edges))?;

    Ok(StableGraphSignature {
        signature_hash,
        concepts,
        relation_ids,
        edges,
    })
}

fn identity_normalized_graph(
    graph: &GroundedConceptGraph,
    manifest: &HdcOntologyManifest,
) -> Result<GroundedConceptGraph, String> {
    validate_manifest(manifest)?;
    let mut node_to_concept = BTreeMap::new();
    let mut nodes = Vec::with_capacity(graph.nodes.len());

    for node in &graph.nodes {
        let binding = manifest
            .node_binding(&node.id)
            .ok_or_else(|| format!("manifest lacks node identity binding: {}", node.id))?;
        if binding.kind != node.kind {
            return Err(format!(
                "manifest kind mismatch for node {}: expected {:?}, got {:?}",
                node.id, node.kind, binding.kind
            ));
        }
        node_to_concept.insert(node.id.clone(), binding.concept_id.clone());
        nodes.push(ConceptNode {
            id: binding.concept_id.clone(),
            kind: node.kind.clone(),
            label: None,
            grounded_by: node.grounded_by.clone(),
            confidence: node.confidence,
        });
    }

    let mut edges = Vec::with_capacity(graph.edges.len());
    for edge in &graph.edges {
        let source = node_to_concept
            .get(&edge.source)
            .cloned()
            .ok_or_else(|| format!("missing stable source node: {}", edge.source))?;
        let target = node_to_concept
            .get(&edge.target)
            .cloned()
            .ok_or_else(|| format!("missing stable target node: {}", edge.target))?;
        let relation = manifest
            .relation_id(&edge.relation)
            .ok_or_else(|| {
                format!(
                    "manifest lacks relation identity binding: {}",
                    edge.relation
                )
            })?
            .to_string();
        edges.push(ConceptEdge {
            source,
            relation,
            target,
            evidence_ids: edge.evidence_ids.clone(),
            confidence: edge.confidence,
        });
    }

    Ok(GroundedConceptGraph { nodes, edges })
}

fn validate_manifest(manifest: &HdcOntologyManifest) -> Result<(), String> {
    if !manifest.validates() {
        return Err("invalid ontology identity manifest".into());
    }
    Ok(())
}

fn validate_manifest_compatibility(
    descriptor: &HdcOntologyCodebookDescriptor,
    manifest: &HdcOntologyManifest,
) -> Result<(), String> {
    validate_manifest(manifest)?;
    if descriptor.scheme_id != manifest.scheme_id {
        return Err(format!(
            "identity scheme mismatch: codebook={}, manifest={}",
            descriptor.scheme_id, manifest.scheme_id
        ));
    }
    if descriptor.mapping_provenance_hash != manifest.mapping_provenance_hash {
        return Err("identity mapping provenance mismatch".into());
    }
    Ok(())
}

fn unique_concept_node_ids(concepts: &[HdcConceptIdentityBinding]) -> bool {
    let mut ids = BTreeSet::new();
    concepts
        .iter()
        .all(|binding| ids.insert(binding.node_id.clone()))
}

fn unique_graph_local_relations(relations: &[HdcRelationIdentityBinding]) -> bool {
    let mut ids = BTreeSet::new();
    relations
        .iter()
        .all(|binding| ids.insert(binding.local_relation.clone()))
}

fn concept_kinds_consistent(concepts: &[HdcConceptIdentityBinding]) -> bool {
    let mut kinds = BTreeMap::<&str, &ConceptKind>::new();
    for binding in concepts {
        if let Some(existing) = kinds.get(binding.concept_id.as_str()) {
            if *existing != &binding.kind {
                return false;
            }
        } else {
            kinds.insert(binding.concept_id.as_str(), &binding.kind);
        }
    }
    true
}

fn select_with_policy(
    candidates: &[HdcOntologyRetrievalCandidate],
    selected_count: usize,
    policy: HdcOntologyDecodePolicy,
) -> Result<Vec<String>, String> {
    if selected_count == 0 || selected_count > candidates.len() {
        return Err("invalid candidate selection count".into());
    }
    let (min_score, margin) = selection_stats(candidates, selected_count)?;
    if min_score < policy.min_score || margin < policy.min_margin {
        return Err("ontology HDC decoder abstained: insufficient retrieval evidence".into());
    }
    Ok(candidates[..selected_count]
        .iter()
        .map(|candidate| candidate.stable_id.clone())
        .collect())
}

fn selection_stats(
    candidates: &[HdcOntologyRetrievalCandidate],
    selected_count: usize,
) -> Result<(f64, f64), String> {
    if selected_count == 0 || selected_count > candidates.len() {
        return Err("invalid selected candidate count".into());
    }

    let min_selected = candidates[..selected_count]
        .iter()
        .map(|candidate| candidate.score)
        .fold(f64::INFINITY, f64::min);

    let max_unselected = candidates
        .get(selected_count)
        .map(|candidate| candidate.score)
        .unwrap_or(-1.0);

    Ok((min_selected, min_selected - max_unselected))
}

fn calibration_scores_hash(scores: &[f64]) -> Result<String, String> {
    if scores.is_empty() {
        return Err("conformal calibration scores cannot be empty".into());
    }
    if scores
        .iter()
        .any(|score| !score.is_finite() || !(0.0..=1.0).contains(score))
    {
        return Err("conformal calibration scores must be finite and within [0, 1]".into());
    }
    let mut canonical = scores.to_vec();
    canonical.sort_by(f64::total_cmp);
    hash_json(&canonical)
}

fn candidate_universe_hash(stable_ids: &[String]) -> Result<String, String> {
    if stable_ids.is_empty() {
        return Err("conformal candidate universe cannot be empty".into());
    }
    if stable_ids.len() > HDC_ONTOLOGY_MAX_CONFORMAL_CANDIDATES {
        return Err(format!(
            "conformal candidate universe exceeds {} candidates",
            HDC_ONTOLOGY_MAX_CONFORMAL_CANDIDATES
        ));
    }
    let mut ids = BTreeSet::new();
    for stable_id in stable_ids {
        if stable_id.trim().is_empty() || !ids.insert(stable_id.clone()) {
            return Err("conformal candidate universe IDs must be unique and non-empty".into());
        }
    }
    hash_json(&ids.into_iter().collect::<Vec<_>>())
}

fn edge_candidate_count(node_count: usize, relation_count: usize) -> Result<usize, String> {
    node_count
        .checked_mul(relation_count)
        .and_then(|count| count.checked_mul(node_count))
        .ok_or_else(|| "ontology HDC edge candidate count overflowed usize".into())
}

fn sort_candidates(candidates: &mut [HdcOntologyRetrievalCandidate]) {
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.stable_id.cmp(&b.stable_id))
    });
}

fn hash_json<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| crate::content_hash(&bytes))
        .map_err(|error| error.to_string())
}

fn parse_edge_candidate(value: &str) -> Result<(String, String, String), String> {
    serde_json::from_str(value)
        .map_err(|error| format!("invalid encoded ontology edge candidate: {value}: {error}"))
}

fn derive_atom_vector(seed: u64, domain: &str, key: &str) -> ContinuousHV {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(domain.as_bytes());
    hasher.update(&[0]);
    hasher.update(key.as_bytes());
    let digest = hasher.finalize();
    let derived_seed = u64::from_le_bytes(
        digest.as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 digest is at least eight bytes"),
    );
    ContinuousHV::random(HDC_DIMENSION, derived_seed)
}

fn full_cosine_similarity(a: &ContinuousHV, b: &ContinuousHV) -> f64 {
    let mut dot = 0.0_f64;
    let mut norm_a_sq = 0.0_f64;
    let mut norm_b_sq = 0.0_f64;

    for (x, y) in a.values.iter().zip(&b.values) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }

    let denominator = norm_a_sq.sqrt() * norm_b_sq.sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        (dot / denominator).clamp(-1.0, 1.0)
    }
}

fn validate_graph_shape(graph: &GroundedConceptGraph) -> Result<(), String> {
    if graph.nodes.is_empty() || graph.edges.is_empty() {
        return Err("ontology HDC interlingua requires at least one node and edge".into());
    }
    if graph.nodes.len() > HDC_ONTOLOGY_MAX_NODES {
        return Err(format!(
            "ontology HDC node budget exceeded: {} > {}",
            graph.nodes.len(),
            HDC_ONTOLOGY_MAX_NODES
        ));
    }
    if graph.edges.len() > HDC_ONTOLOGY_MAX_EDGES {
        return Err(format!(
            "ontology HDC edge budget exceeded: {} > {}",
            graph.edges.len(),
            HDC_ONTOLOGY_MAX_EDGES
        ));
    }

    let mut ids = BTreeSet::new();
    for node in &graph.nodes {
        if node.id.trim().is_empty() || !ids.insert(node.id.clone()) {
            return Err("graph node identifiers must be unique and non-empty".into());
        }
        if !crate::valid_confidence(node.confidence) {
            return Err("graph node confidence must be finite and within [0, 1]".into());
        }
    }

    for edge in &graph.edges {
        if !ids.contains(&edge.source) || !ids.contains(&edge.target) {
            return Err("graph edge references an unknown node identifier".into());
        }
        if edge.relation.trim().is_empty() {
            return Err("graph edge relations must be non-empty".into());
        }
        if !crate::valid_confidence(edge.confidence) {
            return Err("graph edge confidence must be finite and within [0, 1]".into());
        }
    }
    Ok(())
}

fn kind_tag(kind: &ConceptKind) -> &'static str {
    match kind {
        ConceptKind::Agent => "Agent",
        ConceptKind::Object => "Object",
        ConceptKind::Event => "Event",
        ConceptKind::Action => "Action",
        ConceptKind::State => "State",
        ConceptKind::Property => "Property",
        ConceptKind::Relation => "Relation",
        ConceptKind::Unknown => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

        fn graph_and_training_budgets_fail_closed_before_encoding() {
        let (training, training_manifest) = training_graph_and_manifest();
        let mut oversized_graph = training.clone();
        oversized_graph.nodes.extend(
            (0..=(HDC_ONTOLOGY_MAX_NODES - training.nodes.len()))
                .map(|index| ConceptNode {
                    id: format!("extra-{index}"),
                    kind: ConceptKind::Object,
                    label: None,
                    grounded_by: vec![format!("grounding-extra-{index}")],
                    confidence: 1.0,
                }),
        );
        assert!(validate_graph_shape(&oversized_graph).is_err());

        let codebook_error = HdcOntologyCodebook::from_training_graphs(
            77,
            &vec![training; HDC_ONTOLOGY_MAX_TRAINING_GRAPHS + 1],
            &training_manifest,
        );
        assert!(codebook_error.is_err());
    }

    #[test]
#[test]
    fn representation_declared_graph_size_is_bounded() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();
        let mut representation = codebook.encode_graph(&training, &training_manifest).unwrap();
        representation.node_count = HDC_ONTOLOGY_MAX_NODES + 1;
        assert!(!representation.validates());
        representation.node_count = training.nodes.len();
        representation.edge_count = HDC_ONTOLOGY_MAX_EDGES + 1;
        assert!(!representation.validates());
    }

    #[test]
    fn manifest_and_codebook_cardinality_are_bounded() {
        let (training, training_manifest) = training_graph_and_manifest();
        let mut oversized_manifest = training_manifest.clone();
        oversized_manifest.concepts = (0..=HDC_ONTOLOGY_MAX_MANIFEST_CONCEPTS)
            .map(|index| HdcConceptIdentityBinding {
                node_id: format!("node-{index}"),
                concept_id: format!("concept-{index}"),
                kind: ConceptKind::Object,
                grounding_ids: vec!["grounding".into()],
            })
            .collect();
        assert!(!oversized_manifest.validates());

        let mut oversized_grounding = training_manifest.clone();
        oversized_grounding.concepts[0].grounding_ids =
            vec!["g".repeat(HDC_ONTOLOGY_MAX_ID_BYTES); HDC_ONTOLOGY_MAX_GROUNDING_IDS_PER_CONCEPT + 1];
        assert!(!oversized_grounding.validates());

        let mut oversized_identifier = training_manifest.clone();
        oversized_identifier.concepts[0].concept_id =
            "x".repeat(HDC_ONTOLOGY_MAX_ID_BYTES + 1);
        assert!(!oversized_identifier.validates());

        let mut oversized_authority = training_manifest.clone();
        oversized_authority.scheme_id = "x".repeat(HDC_ONTOLOGY_MAX_ID_BYTES + 1);
        assert!(!oversized_authority.validates());

        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training], &training_manifest)
                .unwrap();
        let mut oversized_descriptor = codebook.descriptor().clone();
        oversized_descriptor.concept_count = HDC_ONTOLOGY_MAX_CODEBOOK_CONCEPTS + 1;
        assert!(!oversized_descriptor.validates());
        oversized_descriptor.concept_count = codebook.descriptor().concept_count;
        oversized_descriptor.relation_count = HDC_ONTOLOGY_MAX_CODEBOOK_RELATIONS + 1;
        assert!(!oversized_descriptor.validates());
    }

    #[test]
    fn edge_candidate_budget_is_fail_closed() {
        assert_eq!(edge_candidate_count(9, 4).unwrap(), 324);
        assert!(edge_candidate_count(usize::MAX, 2).is_err());
        let over = edge_candidate_count(512, 4_096).unwrap();
        assert!(over > HDC_ONTOLOGY_MAX_EDGE_CANDIDATES);
    }

    #[test]
    fn conformal_rank_is_finite_sample_conservative() {
        let calibration = HdcOntologyConformalCalibration::from_nonconformity_scores(
            "codebook",
            &["a".into(), "b".into(), "c".into()],
            &[0.10, 0.20, 0.30],
            0.10,
        )
        .unwrap();
        // ceil((3 + 1) * 0.90) = 4, clipped to n=3: the maximum score.
        assert_eq!(calibration.threshold, 0.30);
        assert!(calibration.accepts(0.30));
        assert!(!calibration.accepts(0.31));
        assert!(calibration.validates());
    }

    #[test]
    fn conformal_cosine_nonconformity_is_bounded_and_monotone() {
        assert_eq!(
            HdcOntologyConformalCalibration::nonconformity_from_cosine(1.0).unwrap(),
            0.0
        );
        assert_eq!(
            HdcOntologyConformalCalibration::nonconformity_from_cosine(-1.0).unwrap(),
            1.0
        );
        assert!(
            HdcOntologyConformalCalibration::nonconformity_from_cosine(0.8).unwrap()
                < HdcOntologyConformalCalibration::nonconformity_from_cosine(0.2).unwrap()
        );
        assert!(HdcOntologyConformalCalibration::nonconformity_from_cosine(f64::NAN).is_err());
        assert!(HdcOntologyConformalCalibration::nonconformity_from_cosine(1.01).is_err());
    }

    #[test]
    fn conformal_non_max_threshold_requires_enough_calibration_cases() {
        assert_eq!(
            HdcOntologyConformalCalibration::minimum_cases_for_non_max_threshold(0.10)
                .unwrap(),
            19
        );
        assert_eq!(
            HdcOntologyConformalCalibration::minimum_cases_for_non_max_threshold(0.05)
                .unwrap(),
            39
        );
    }

    #[test]
    fn conformal_prediction_set_is_thresholded_and_deterministic() {
        let calibration = HdcOntologyConformalCalibration::from_nonconformity_scores(
            "codebook",
            &["a".into(), "m".into(), "q".into(), "z".into()],
            &[0.0, 0.05, 0.1, 0.2],
            0.50,
        )
        .unwrap();
        assert_eq!(calibration.schema_version, HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION);
        assert_eq!(calibration.threshold, 0.1);
        assert_eq!(calibration.score_revision, HDC_ONTOLOGY_CONFORMAL_SCORE_REVISION);
        let candidates = vec![
            HdcOntologyRetrievalCandidate { stable_id: "z".into(), score: 0.8 },
            HdcOntologyRetrievalCandidate { stable_id: "a".into(), score: 1.0 },
            HdcOntologyRetrievalCandidate { stable_id: "m".into(), score: 0.6 },
            HdcOntologyRetrievalCandidate { stable_id: "q".into(), score: -1.0 },
        ];
        assert_eq!(calibration.prediction_set("codebook", &candidates).unwrap(), vec!["a", "z"]);
        assert!(calibration.prediction_set("different-codebook", &candidates).is_err());
        let reordered = vec![
            HdcOntologyRetrievalCandidate { stable_id: "q".into(), score: -1.0 },
            HdcOntologyRetrievalCandidate { stable_id: "z".into(), score: 0.8 },
            HdcOntologyRetrievalCandidate { stable_id: "a".into(), score: 1.0 },
            HdcOntologyRetrievalCandidate { stable_id: "m".into(), score: 0.6 },
        ];
        assert_eq!(calibration.prediction_set("codebook", &reordered).unwrap(), vec!["a", "z"]);
        let wrong_universe = vec![
            HdcOntologyRetrievalCandidate { stable_id: "a".into(), score: 1.0 },
            HdcOntologyRetrievalCandidate { stable_id: "m".into(), score: 0.6 },
            HdcOntologyRetrievalCandidate { stable_id: "x".into(), score: 0.8 },
            HdcOntologyRetrievalCandidate { stable_id: "z".into(), score: 0.8 },
        ];
        assert!(calibration.prediction_set("codebook", &wrong_universe).is_err());
        let oversized = (0..=HDC_ONTOLOGY_MAX_CONFORMAL_CANDIDATES)
            .map(|index| HdcOntologyRetrievalCandidate {
                stable_id: format!("candidate-{index}"),
                score: 1.0,
            })
            .collect::<Vec<_>>();
        assert!(calibration.prediction_set("codebook", &oversized).is_err());
    }

    #[test]
    fn conformal_prediction_set_rejects_ambiguous_candidates() {
        let calibration = HdcOntologyConformalCalibration::from_nonconformity_scores(
            "codebook",
            &["x".into(), "y".into(), "z".into()],
            &[0.1, 0.2, 0.3],
            0.10,
        )
        .unwrap();
        let duplicate = vec![
            HdcOntologyRetrievalCandidate { stable_id: "x".into(), score: 1.0 },
            HdcOntologyRetrievalCandidate { stable_id: "x".into(), score: 0.9 },
        ];
        assert!(calibration.prediction_set("codebook", &duplicate).is_err());
        let invalid = vec![HdcOntologyRetrievalCandidate {
            stable_id: "x".into(),
            score: f64::NAN,
        }];
        assert!(calibration.prediction_set("codebook", &invalid).is_err());
    }

    #[test]
    fn conformal_calibration_rejects_invalid_scores_and_alpha() {
        assert!(HdcOntologyConformalCalibration::from_nonconformity_scores(
            "codebook",
            &["a".into(), "b".into()],
            &[0.1, f64::NAN],
            0.10,
        )
        .is_err());
        assert!(HdcOntologyConformalCalibration::from_nonconformity_scores(
            "codebook",
            &["a".into(), "b".into()],
            &[0.1, 0.2],
            1.0,
        )
        .is_err());
    }

    #[test]
    fn empirical_calibration_never_weakens_baseline() {
        let baseline = HdcOntologyDecodePolicy::conservative_default();
        let metrics = HdcOntologyRoundtripMetrics {
            schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
            adapter_id: HDC_ONTOLOGY_ADAPTER_ID.into(),
            codebook_hash: "codebook".into(),
            source_manifest_hash: "manifest".into(),
            concept_identity_exact: true,
            relation_identity_exact: true,
            node_precision: 1.0,
            node_recall: 1.0,
            edge_precision: 1.0,
            edge_recall: 1.0,
            structural_equivalence: true,
            confidence_mae: 0.0,
            node_min_selected_score: 0.31,
            node_selection_margin: 0.09,
            edge_min_selected_score: 0.27,
            edge_selection_margin: 0.07,
        };
        let calibration =
            HdcOntologyEmpiricalCalibration::from_clean_metrics(baseline, &[metrics], 0.0)
                .unwrap();
        assert_eq!(calibration.calibrated_min_score, 0.31);
        assert_eq!(calibration.calibrated_min_margin, 0.09);
        assert!(calibration.validates());
        assert!(calibration.policy().min_score >= 0.20);
        assert!(calibration.policy().min_margin >= 0.05);
    }

    #[test]
    fn empirical_calibration_rejects_mixed_codebooks() {
        let baseline = HdcOntologyDecodePolicy::conservative_default();
        let mut first = HdcOntologyRoundtripMetrics {
            schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
            adapter_id: HDC_ONTOLOGY_ADAPTER_ID.into(),
            codebook_hash: "codebook-a".into(),
            source_manifest_hash: "manifest-a".into(),
            concept_identity_exact: true,
            relation_identity_exact: true,
            node_precision: 1.0,
            node_recall: 1.0,
            edge_precision: 1.0,
            edge_recall: 1.0,
            structural_equivalence: true,
            confidence_mae: 0.0,
            node_min_selected_score: 0.4,
            node_selection_margin: 0.1,
            edge_min_selected_score: 0.4,
            edge_selection_margin: 0.1,
        };
        let mut second = first.clone();
        second.codebook_hash = "codebook-b".into();
        first.source_manifest_hash = "manifest-a2".into();

        assert!(
            HdcOntologyEmpiricalCalibration::from_clean_metrics(
                baseline,
                &[first, second],
                0.1
            )
            .is_err()
        );
    }

    #[test]
    fn empirical_calibration_rejects_non_clean_metrics() {
        let baseline = HdcOntologyDecodePolicy::conservative_default();
        let mut metrics = HdcOntologyRoundtripMetrics {
            schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
            adapter_id: HDC_ONTOLOGY_ADAPTER_ID.into(),
            codebook_hash: "codebook".into(),
            source_manifest_hash: "manifest".into(),
            concept_identity_exact: true,
            relation_identity_exact: true,
            node_precision: 1.0,
            node_recall: 1.0,
            edge_precision: 1.0,
            edge_recall: 1.0,
            structural_equivalence: true,
            confidence_mae: 0.0,
            node_min_selected_score: 0.4,
            node_selection_margin: 0.1,
            edge_min_selected_score: 0.4,
            edge_selection_margin: 0.1,
        };
        metrics.concept_identity_exact = false;
        assert!(
            HdcOntologyEmpiricalCalibration::from_clean_metrics(baseline, &[metrics], 0.1)
                .is_err()
        );
    }

    use super::*;

    fn graph(
        nodes: &[(&str, ConceptKind, &str)],
        edges: &[(&str, &str, &str)],
    ) -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: nodes
                .iter()
                .map(|(id, kind, grounding)| ConceptNode {
                    id: (*id).into(),
                    kind: kind.clone(),
                    label: Some((*id).into()),
                    grounded_by: vec![(*grounding).into()],
                    confidence: 1.0,
                })
                .collect(),
            edges: edges
                .iter()
                .map(|(source, relation, target)| ConceptEdge {
                    source: (*source).into(),
                    relation: (*relation).into(),
                    target: (*target).into(),
                    evidence_ids: vec![],
                    confidence: 1.0,
                })
                .collect(),
        }
    }

    fn manifest(
        graph: &GroundedConceptGraph,
        concept_ids: &[(&str, &str)],
        relation_ids: &[(&str, &str)],
        scheme_id: &str,
    ) -> HdcOntologyManifest {
        let concepts = graph
            .nodes
            .iter()
            .map(|node| {
                let concept_id = concept_ids
                    .iter()
                    .find(|(node_id, _)| *node_id == node.id)
                    .map(|(_, concept_id)| *concept_id)
                    .unwrap();
                HdcConceptIdentityBinding {
                    node_id: node.id.clone(),
                    concept_id: concept_id.into(),
                    kind: node.kind.clone(),
                    grounding_ids: node.grounded_by.clone(),
                }
            })
            .collect();

        let relations = relation_ids
            .iter()
            .map(|(local_relation, relation_id)| HdcRelationIdentityBinding {
                local_relation: (*local_relation).into(),
                relation_id: (*relation_id).into(),
            })
            .collect();

        HdcOntologyManifest {
            schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
            scheme_id: scheme_id.into(),
            mapping_provenance_hash: crate::content_hash(b"validated-upstream-map"),
            concepts,
            relations,
        }
    }

    fn training_graph_and_manifest() -> (
        GroundedConceptGraph,
        HdcOntologyManifest,
    ) {
        let training = graph(
            &[
                ("alice", ConceptKind::Agent, "train-alice"),
                ("event", ConceptKind::Event, "train-event"),
                ("object", ConceptKind::Object, "train-object"),
            ],
            &[
                ("alice", "initiates", "event"),
                ("event", "targets", "object"),
            ],
        );
        let manifest = manifest(
            &training,
            &[
                ("alice", "concept:agent/alice"),
                ("event", "concept:event/approach"),
                ("object", "concept:object/target"),
            ],
            &[
                ("initiates", "relation:initiates"),
                ("targets", "relation:targets"),
            ],
            "scheme:example-v1",
        );
        (training, manifest)
    }

    fn grounding_map(graph: &GroundedConceptGraph) -> BTreeMap<String, Vec<String>> {
        graph
            .nodes
            .iter()
            .map(|node| (node.id.clone(), node.grounded_by.clone()))
            .collect()
    }

    #[test]
    fn manifest_identity_hash_is_independent_of_collection_order() {
        let (_training, manifest) = training_graph_and_manifest();
        let mut reordered = manifest.clone();
        reordered.concepts.reverse();
        reordered.relations.reverse();
        reordered.concepts[0].grounding_ids.reverse();

        assert!(manifest.validates());
        assert_eq!(manifest.manifest_hash(), reordered.manifest_hash());
    }

    #[test]
    fn same_concept_with_new_grounding_reuses_the_frozen_codebook() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();

        let held_out = graph(
            &[
                ("node-a", ConceptKind::Agent, "heldout-a"),
                ("node-b", ConceptKind::Event, "heldout-b"),
                ("node-c", ConceptKind::Object, "heldout-c"),
            ],
            &[
                ("node-a", "initiates", "node-b"),
                ("node-b", "targets", "node-c"),
            ],
        );
        let held_out_manifest = manifest(
            &held_out,
            &[
                ("node-a", "concept:agent/alice"),
                ("node-b", "concept:event/approach"),
                ("node-c", "concept:object/target"),
            ],
            &[
                ("initiates", "relation:initiates"),
                ("targets", "relation:targets"),
            ],
            "scheme:example-v1",
        );

        assert_ne!(
            training_manifest.manifest_hash(),
            held_out_manifest.manifest_hash()
        );

        let representation = codebook
            .encode_graph(&held_out, &held_out_manifest)
            .unwrap();
        let decoded = codebook
            .decode_graph_with_policy(
                &representation,
                &held_out_manifest,
                &held_out_manifest,
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .unwrap();

        assert_eq!(
            grounding_map(&decoded.graph),
            grounding_map(&held_out)
        );
    }

    #[test]
    fn stable_identity_is_preserved_separately_from_local_node_ids() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();
        let representation = codebook
            .encode_graph(&training, &training_manifest)
            .unwrap();
        let decoded = codebook
            .decode_graph_with_policy(
                &representation,
                &training_manifest,
                &training_manifest,
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .unwrap();

        let expected = BTreeMap::from([
            ("alice".into(), "concept:agent/alice".into()),
            ("event".into(), "concept:event/approach".into()),
            ("object".into(), "concept:object/target".into()),
        ]);
        assert_eq!(decoded.concept_ids_by_node, expected);
        assert_ne!(
            decoded.concept_ids_by_node.keys().collect::<Vec<_>>(),
            decoded.concept_ids_by_node.values().collect::<Vec<_>>()
        );
    }

    #[test]
    fn novel_stable_concept_is_rejected_as_oov() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();

        let held_out = graph(
            &[
                ("alice", ConceptKind::Agent, "new-a"),
                ("event", ConceptKind::Event, "new-e"),
                ("novel", ConceptKind::Object, "new-o"),
            ],
            &[
                ("alice", "initiates", "event"),
                ("event", "targets", "novel"),
            ],
        );
        let held_out_manifest = manifest(
            &held_out,
            &[
                ("alice", "concept:agent/alice"),
                ("event", "concept:event/approach"),
                ("novel", "concept:object/novel"),
            ],
            &[
                ("initiates", "relation:initiates"),
                ("targets", "relation:targets"),
            ],
            "scheme:example-v1",
        );

        assert!(codebook.encode_graph(&held_out, &held_out_manifest).is_err());
    }

    #[test]
    fn different_identity_scheme_is_fail_closed() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();
        let mut wrong = training_manifest.clone();
        wrong.scheme_id = "scheme:other".into();

        assert!(codebook.encode_graph(&training, &wrong).is_err());
    }

    #[test]
    fn source_provenance_is_preserved_when_receiver_grounding_differs() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();
        let representation = codebook.encode_graph(&training, &training_manifest).unwrap();

        let mut receiver = training_manifest.clone();
        for binding in &mut receiver.concepts {
            binding.node_id = format!("receiver-{}", binding.node_id);
            binding.grounding_ids = vec![format!("receiver-grounding-{}", binding.concept_id)];
        }

        let decoded = codebook
            .decode_graph_with_policy(
                &representation,
                &training_manifest,
                &receiver,
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .unwrap();

        assert!(decoded.graph.nodes.iter().all(|node| {
            !node.id.starts_with("alice")
                && !node.id.starts_with("event")
                && !node.id.starts_with("object")
        }));
        let mut observed_groundings = decoded.graph.nodes.iter()
            .map(|node| node.grounded_by.clone())
            .collect::<Vec<_>>();
        let mut expected_groundings = training.nodes.iter()
            .map(|node| node.grounded_by.clone())
            .collect::<Vec<_>>();
        observed_groundings.sort();
        expected_groundings.sort();
        assert_eq!(observed_groundings, expected_groundings);
    }

    #[test]
    fn ambiguous_receiver_relation_mapping_is_fail_closed() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();
        let representation = codebook
            .encode_graph(&training, &training_manifest)
            .unwrap();

        let mut ambiguous = training_manifest.clone();
        ambiguous.relations.push(HdcRelationIdentityBinding {
            local_relation: "starts".into(),
            relation_id: "relation:initiates".into(),
        });
        assert!(ambiguous.validates());

        let error = codebook
            .decode_graph_with_policy(
                &representation,
                &training_manifest,
                &ambiguous,
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .expect_err("ambiguous stable relation rendering must fail closed");
        assert!(error.contains("ambiguously maps stable relation id"));
    }

    #[test]
    fn source_manifest_hash_mismatch_is_fail_closed() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();
        let representation = codebook.encode_graph(&training, &training_manifest).unwrap();

        let mut tampered_source = training_manifest.clone();
        tampered_source.concepts[0].grounding_ids = vec!["tampered-grounding".into()];

        assert!(codebook
            .decode_graph_with_policy(
                &representation,
                &tampered_source,
                &training_manifest,
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .is_err());
    }

    #[test]
    fn mapping_provenance_mismatch_is_fail_closed() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();
        let mut wrong = training_manifest.clone();
        wrong.mapping_provenance_hash = crate::content_hash(b"different-authority-revision");

        assert!(codebook.encode_graph(&training, &wrong).is_err());
    }

    #[test]
    fn cross_label_mapping_uses_stable_relation_identity() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();

        let multilingual = graph(
            &[
                ("agent-fr", ConceptKind::Agent, "fr-a"),
                ("event-fr", ConceptKind::Event, "fr-e"),
                ("object-fr", ConceptKind::Object, "fr-o"),
            ],
            &[
                ("agent-fr", "commence", "event-fr"),
                ("event-fr", "cible", "object-fr"),
            ],
        );
        let multilingual_manifest = manifest(
            &multilingual,
            &[
                ("agent-fr", "concept:agent/alice"),
                ("event-fr", "concept:event/approach"),
                ("object-fr", "concept:object/target"),
            ],
            &[
                ("commence", "relation:initiates"),
                ("cible", "relation:targets"),
            ],
            "scheme:example-v1",
        );

        let representation = codebook
            .encode_graph(&multilingual, &multilingual_manifest)
            .unwrap();
        let decoded = codebook
            .decode_graph_with_policy(
                &representation,
                &multilingual_manifest,
                &multilingual_manifest,
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .unwrap();

        assert_eq!(
            decoded.relation_ids_by_edge,
            vec!["relation:initiates", "relation:targets"]
        );
        assert!(decoded
            .graph
            .edges
            .iter()
            .all(|edge| edge.relation == "commence" || edge.relation == "cible"));

        let mut receiver_manifest = multilingual_manifest.clone();
        receiver_manifest.relations[0].local_relation = "initiates-local".into();
        let expected_concepts = BTreeMap::from([
            ("agent-fr".into(), "concept:agent/alice".into()),
            ("event-fr".into(), "concept:event/approach".into()),
            ("object-fr".into(), "concept:object/target".into()),
        ]);
        let metrics = codebook
            .measure_roundtrip(
                &multilingual,
                &representation,
                &multilingual_manifest,
                &receiver_manifest,
                &expected_concepts,
                &["relation:initiates".into(), "relation:targets".into()],
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .unwrap();
        assert!(metrics.structural_equivalence);
        assert!(metrics.relation_identity_exact);
        assert_eq!(metrics.node_precision, 1.0);
        assert_eq!(metrics.edge_recall, 1.0);
    }

    #[test]
    fn representation_provenance_changes_but_codebook_compatibility_does_not() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();

        let held_out = graph(
            &[
                ("a2", ConceptKind::Agent, "obs-2a"),
                ("e2", ConceptKind::Event, "obs-2e"),
                ("o2", ConceptKind::Object, "obs-2o"),
            ],
            &[
                ("a2", "initiates", "e2"),
                ("e2", "targets", "o2"),
            ],
        );
        let held_out_manifest = manifest(
            &held_out,
            &[
                ("a2", "concept:agent/alice"),
                ("e2", "concept:event/approach"),
                ("o2", "concept:object/target"),
            ],
            &[
                ("initiates", "relation:initiates"),
                ("targets", "relation:targets"),
            ],
            "scheme:example-v1",
        );

        let a = codebook
            .encode_graph(&training, &training_manifest)
            .unwrap();
        let b = codebook
            .encode_graph(&held_out, &held_out_manifest)
            .unwrap();

        assert_ne!(a.source_manifest_hash, b.source_manifest_hash);
        assert_eq!(a.codebook, b.codebook);
        assert_eq!(a.node_frame, b.node_frame);
        assert_eq!(a.edge_frame, b.edge_frame);
    }

    #[test]
    fn receiver_manifest_ambiguity_is_fail_closed() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();
        let representation = codebook
            .encode_graph(&training, &training_manifest)
            .unwrap();

        let mut ambiguous = training_manifest.clone();
        ambiguous.concepts.push(HdcConceptIdentityBinding {
            node_id: "alias-for-alice".into(),
            concept_id: "concept:agent/alice".into(),
            kind: ConceptKind::Agent,
            grounding_ids: vec!["second-grounding".into()],
        });

        assert!(codebook
            .decode_graph_with_policy(
                &representation,
                &training_manifest,
                &ambiguous,
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .is_err());
    }

    #[test]
    fn relation_identity_exactness_is_order_independent() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();
        let representation = codebook
            .encode_graph(&training, &training_manifest)
            .unwrap();
        let decoded = codebook
            .decode_graph_with_policy(
                &representation,
                &training_manifest,
                &training_manifest,
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .unwrap();

        let expected_concepts = decoded.concept_ids_by_node.clone();
        let mut relations = decoded.relation_ids_by_edge.clone();
        relations.reverse();

        let metrics = codebook
            .measure_roundtrip(
                &training,
                &representation,
                &training_manifest,
                &training_manifest,
                &expected_concepts,
                &relations,
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .unwrap();

        assert!(metrics.relation_identity_exact);
    }

    #[test]
    fn decoder_abstains_on_unrelated_frames() {
        let (training, training_manifest) = training_graph_and_manifest();
        let codebook =
            HdcOntologyCodebook::from_training_graphs(77, &[training.clone()], &training_manifest)
                .unwrap();
        let mut representation = codebook
            .encode_graph(&training, &training_manifest)
            .unwrap();

        representation.node_frame = HdcBinaryFrame::from_binary(
            &symthaea_core::hdc::binary_hv::BinaryHV::random(0xBADC0DE),
        );
        representation.edge_frame = HdcBinaryFrame::from_binary(
            &symthaea_core::hdc::binary_hv::BinaryHV::random(0xD15EA5E),
        );

        assert!(codebook
            .decode_graph_with_policy(
                &representation,
                &training_manifest,
                &training_manifest,
                HdcOntologyDecodePolicy::conservative_default(),
            )
            .is_err());
    }
    #[test]
    fn identity_metrics_ignore_receiver_local_relation_label() {
        let source = GroundedConceptGraph {
            nodes: vec![
                ConceptNode { id: "src-a".into(), kind: ConceptKind::Agent, label: Some("sender".into()), grounded_by: vec!["g-a".into()], confidence: 1.0 },
                ConceptNode { id: "src-b".into(), kind: ConceptKind::Object, label: Some("target".into()), grounded_by: vec!["g-b".into()], confidence: 1.0 },
            ],
            edges: vec![ConceptEdge {
                source: "src-a".into(), relation: "commence".into(), target: "src-b".into(), evidence_ids: vec![], confidence: 1.0,
            }],
        };
        let receiver = GroundedConceptGraph {
            nodes: vec![
                ConceptNode { id: "rx-a".into(), kind: ConceptKind::Agent, label: Some("sender".into()), grounded_by: vec!["g-a".into()], confidence: 1.0 },
                ConceptNode { id: "rx-b".into(), kind: ConceptKind::Object, label: Some("target".into()), grounded_by: vec!["g-b".into()], confidence: 1.0 },
            ],
            edges: vec![ConceptEdge {
                source: "rx-a".into(), relation: "initiates".into(), target: "rx-b".into(), evidence_ids: vec![], confidence: 1.0,
            }],
        };
        let mut source_manifest = manifest_for_graph(&source, "commence", "relation:initiates");
        let mut receiver_manifest = manifest_for_graph(&receiver, "initiates", "relation:initiates");
        source_manifest.relations[0].local_relation = "commence".into();
        receiver_manifest.relations[0].local_relation = "initiates".into();
        let normalized_source = identity_normalized_graph(&source, &source_manifest).unwrap();
        let normalized_receiver = identity_normalized_graph(&receiver, &receiver_manifest).unwrap();
        let metrics = compare_graphs(&normalized_source, &normalized_receiver).unwrap();
        assert!(metrics.structural_equivalence);
        assert_eq!(metrics.node_precision, 1.0);
        assert_eq!(metrics.edge_recall, 1.0);
    }

}
