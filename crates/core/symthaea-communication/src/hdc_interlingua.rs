//! Deterministic HDC-backed grounded-concept interlingua.
//!
//! This module deliberately separates:
//! - graph structure from lexical labels and transport-local node identifiers;
//! - codebook construction from held-out graph evaluation;
//! - continuous composition from continuous->binary quantization;
//! - retrieval failure from any claim about semantic or neural decoding.
//!
//! The representation uses two independently framed binary channels:
//! 1. a bundle of role-marked node atoms;
//! 2. a bundle of compositional edge atoms.
//!
//! Keeping node and edge bundles separate avoids relying on even-cardinality
//! BinaryHV majority bundling across unrelated semantic roles.

use crate::hdc_codec::{quantize_continuous, HdcBinaryFrame, HdcCodecDescriptor, HdcQuantizationMetrics};
use crate::interlingua::{compare_graphs, graph_hash, structural_hash};
use crate::{ConceptEdge, ConceptKind, ConceptNode, GroundedConceptGraph};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION: u16 = 1;
pub const HDC_SEMANTIC_ADAPTER_ID: &str = "symthaea.hdc.semantic-interlingua-v1";
pub const HDC_SEMANTIC_CODEBOOK_ID: &str = "symthaea.hdc.semantic-codebook-v1";
pub const HDC_SEMANTIC_CODEBOOK_ALGORITHM: &str = "blake3-seeded-random-atoms-v1";
pub const HDC_SEMANTIC_ROLE_REVISION: &str = "source-relation-target-node-v2";
pub const HDC_EDGE_TARGET_PERMUTATION: usize = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcSemanticCodebookDescriptor {
    pub schema_version: u16,
    pub codebook_id: String,
    pub generation_algorithm: String,
    pub role_revision: String,
    pub seed: u64,
    pub dimension: usize,
    pub node_manifest_hash: String,
    pub relation_manifest_hash: String,
    pub training_manifest_hash: String,
    pub node_atom_count: usize,
    pub relation_atom_count: usize,
}

impl HdcSemanticCodebookDescriptor {
    pub fn validates(&self) -> bool {
        self.schema_version == HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION
            && self.codebook_id == HDC_SEMANTIC_CODEBOOK_ID
            && self.generation_algorithm == HDC_SEMANTIC_CODEBOOK_ALGORITHM
            && self.role_revision == HDC_SEMANTIC_ROLE_REVISION
            && self.dimension == HDC_DIMENSION
            && !self.node_manifest_hash.is_empty()
            && !self.relation_manifest_hash.is_empty()
            && !self.training_manifest_hash.is_empty()
            && self.node_atom_count > 0
            && self.relation_atom_count > 0
    }

    pub fn codebook_hash(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("codebook descriptor is serializable");
        crate::content_hash(&bytes)
    }
}

#[derive(Clone, Debug)]
struct NodeAtom {
    kind: ConceptKind,
    grounded_by: Vec<String>,
    vector: ContinuousHV,
}

#[derive(Clone, Debug)]
pub struct HdcSemanticCodebook {
    descriptor: HdcSemanticCodebookDescriptor,
    nodes: BTreeMap<String, NodeAtom>,
    relations: BTreeMap<String, ContinuousHV>,
    role_node: ContinuousHV,
    role_source: ContinuousHV,
    role_relation: ContinuousHV,
    role_target: ContinuousHV,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcSemanticRepresentation {
    pub schema_version: u16,
    pub adapter_id: String,
    pub codec: HdcCodecDescriptor,
    pub codebook: HdcSemanticCodebookDescriptor,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_quantization: HdcQuantizationMetrics,
    pub edge_quantization: HdcQuantizationMetrics,
    pub node_frame: HdcBinaryFrame,
    pub edge_frame: HdcBinaryFrame,
}

impl HdcSemanticRepresentation {
    pub fn validates(&self) -> bool {
        self.schema_version == HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION
            && self.adapter_id == HDC_SEMANTIC_ADAPTER_ID
            && self.codebook.validates()
            && self.codec.validates()
            && self.codec.codebook_hash.as_deref() == Some(self.codebook.codebook_hash().as_str())
            && self.node_count > 0
            && self.edge_count > 0
            && self.node_quantization.validates()
            && self.edge_quantization.validates()
            && self.node_quantization.dimension == HDC_DIMENSION
            && self.edge_quantization.dimension == HDC_DIMENSION
            && self.node_frame.to_binary().is_ok()
            && self.edge_frame.to_binary().is_ok()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcRetrievalCandidate {
    pub key: String,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcSemanticRoundtripMetrics {
    pub schema_version: u16,
    pub adapter_id: String,
    pub codec_id: String,
    pub codebook_hash: String,
    pub expected_graph_hash: String,
    pub observed_graph_hash: String,
    pub expected_structural_hash: String,
    pub observed_structural_hash: String,
    pub node_precision: f64,
    pub node_recall: f64,
    pub edge_precision: f64,
    pub edge_recall: f64,
    pub confidence_mae: f64,
    pub structural_equivalence: bool,
    pub node_min_selected_score: f64,
    pub node_selection_margin: f64,
    pub edge_min_selected_score: f64,
    pub edge_selection_margin: f64,
    pub expected_bytes: usize,
    pub representation_bytes: usize,
}

impl HdcSemanticRoundtripMetrics {
    pub fn validates(&self) -> bool {
        self.schema_version == HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION
            && self.adapter_id == HDC_SEMANTIC_ADAPTER_ID
            && !self.codec_id.is_empty()
            && !self.codebook_hash.is_empty()
            && !self.expected_graph_hash.is_empty()
            && !self.observed_graph_hash.is_empty()
            && !self.expected_structural_hash.is_empty()
            && !self.observed_structural_hash.is_empty()
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
            && self.expected_bytes > 0
            && self.representation_bytes > 0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcSemanticNegativeControls {
    pub schema_version: u16,
    pub unrelated_node_max_similarity: f64,
    pub unrelated_edge_max_similarity: f64,
    pub true_edge_similarity: f64,
    pub swapped_edge_similarity: f64,
}

impl HdcSemanticNegativeControls {
    pub fn validates(&self) -> bool {
        self.schema_version == HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION
            && self.unrelated_node_max_similarity.is_finite()
            && self.unrelated_edge_max_similarity.is_finite()
            && self.true_edge_similarity.is_finite()
            && self.swapped_edge_similarity.is_finite()
            && (-1.0..=1.0).contains(&self.unrelated_node_max_similarity)
            && (-1.0..=1.0).contains(&self.unrelated_edge_max_similarity)
            && (-1.0..=1.0).contains(&self.true_edge_similarity)
            && (-1.0..=1.0).contains(&self.swapped_edge_similarity)
    }
}

impl HdcSemanticCodebook {
    pub fn from_training_graphs(
        seed: u64,
        training_graphs: &[GroundedConceptGraph],
    ) -> Result<Self, String> {
        if training_graphs.is_empty() {
            return Err("HDC semantic codebook requires at least one training graph".into());
        }

        let mut nodes = BTreeMap::<String, (ConceptKind, Vec<String>)>::new();
        let mut relations = BTreeSet::<String>::new();
        let mut training_structural_hashes = Vec::with_capacity(training_graphs.len());

        for graph in training_graphs {
            validate_graph_shape(graph)?;
            training_structural_hashes.push(structural_hash(graph)?);

            let graph_nodes = canonical_node_atoms(graph)?;
            for (key, (kind, grounded_by)) in graph_nodes {
                nodes.entry(key).or_insert((kind, grounded_by));
            }

            for edge in &graph.edges {
                relations.insert(edge.relation.clone());
            }
        }

        if nodes.is_empty() || relations.is_empty() {
            return Err("HDC semantic codebook requires nodes and relations".into());
        }

        training_structural_hashes.sort();
        let node_manifest = nodes.keys().cloned().collect::<Vec<_>>();
        let relation_manifest = relations.iter().cloned().collect::<Vec<_>>();

        let node_manifest_hash = hash_manifest(&node_manifest)?;
        let relation_manifest_hash = hash_manifest(&relation_manifest)?;
        let training_manifest_hash = hash_manifest(&training_structural_hashes)?;

        let descriptor = HdcSemanticCodebookDescriptor {
            schema_version: HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION,
            codebook_id: HDC_SEMANTIC_CODEBOOK_ID.into(),
            generation_algorithm: HDC_SEMANTIC_CODEBOOK_ALGORITHM.into(),
            role_revision: HDC_SEMANTIC_ROLE_REVISION.into(),
            seed,
            dimension: HDC_DIMENSION,
            node_manifest_hash,
            relation_manifest_hash,
            training_manifest_hash,
            node_atom_count: node_manifest.len(),
            relation_atom_count: relation_manifest.len(),
        };

        let node_atoms = nodes
            .into_iter()
            .map(|(key, (kind, grounded_by))| {
                let vector = derive_atom_vector(seed, "node", &key);
                (
                    key,
                    NodeAtom {
                        kind,
                        grounded_by,
                        vector,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();

        let relation_atoms = relations
            .into_iter()
            .map(|relation| {
                (
                    relation.clone(),
                    derive_atom_vector(seed, "relation", &relation),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let role_node = derive_atom_vector(seed, "role", "node");
        let role_source = derive_atom_vector(seed, "role", "source");
        let role_relation = derive_atom_vector(seed, "role", "relation");
        let role_target = derive_atom_vector(seed, "role", "target");

        let codebook = Self {
            descriptor,
            nodes: node_atoms,
            relations: relation_atoms,
            role_node,
            role_source,
            role_relation,
            role_target,
        };

        if !codebook.descriptor.validates() {
            return Err("constructed HDC semantic codebook descriptor is invalid".into());
        }
        Ok(codebook)
    }

    pub fn descriptor(&self) -> &HdcSemanticCodebookDescriptor {
        &self.descriptor
    }

    pub fn codebook_hash(&self) -> String {
        self.descriptor.codebook_hash()
    }

    pub fn node_keys(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    pub fn relation_names(&self) -> Vec<String> {
        self.relations.keys().cloned().collect()
    }

    pub fn encode_graph(
        &self,
        graph: &GroundedConceptGraph,
    ) -> Result<HdcSemanticRepresentation, String> {
        validate_graph_shape(graph)?;

        let node_atoms = canonical_node_atoms(graph)?;
        if node_atoms
            .keys()
            .any(|key| !self.nodes.contains_key(key))
        {
            let missing = node_atoms
                .keys()
                .find(|key| !self.nodes.contains_key(*key))
                .cloned()
                .unwrap_or_default();
            return Err(format!(
                "graph contains node atom absent from the fixed training codebook: {missing}"
            ));
        }

        let canonical_edges = canonical_edge_atoms(graph)?;
        for (_, relation) in &canonical_edges {
            if !self.relations.contains_key(relation) {
                return Err(format!(
                    "graph contains relation absent from the fixed training codebook: {relation}"
                ));
            }
        }

        let node_vectors = node_atoms
            .keys()
            .map(|key| self.node_vector(key))
            .collect::<Result<Vec<_>, _>>()?;
        let edge_vectors = canonical_edges
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
        codec.encoder_revision = Some(HDC_SEMANTIC_ADAPTER_ID.into());
        codec.codebook_hash = Some(self.codebook_hash());

        let representation = HdcSemanticRepresentation {
            schema_version: HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION,
            adapter_id: HDC_SEMANTIC_ADAPTER_ID.into(),
            codec,
            codebook: self.descriptor.clone(),
            node_count: node_atoms.len(),
            edge_count: canonical_edges.len(),
            node_quantization,
            edge_quantization,
            node_frame: HdcBinaryFrame::from_binary(&node_binary),
            edge_frame: HdcBinaryFrame::from_binary(&edge_binary),
        };

        if !representation.validates() {
            return Err("HDC semantic representation failed validation".into());
        }
        Ok(representation)
    }

    pub fn corrupt_for_transport(
        &self,
        representation: &HdcSemanticRepresentation,
        flip_probability: f32,
        seed: u64,
    ) -> Result<HdcSemanticRepresentation, String> {
        if !representation.validates() {
            return Err("invalid HDC semantic representation".into());
        }
        if !flip_probability.is_finite() || !(0.0..=1.0).contains(&flip_probability) {
            return Err("flip probability must be finite and within [0, 1]".into());
        }

        let mut corrupted = representation.clone();
        let node = representation.node_frame.to_binary()?;
        let edge = representation.edge_frame.to_binary()?;
        corrupted.node_frame =
            HdcBinaryFrame::from_binary(&node.add_noise(flip_probability, seed));
        corrupted.edge_frame =
            HdcBinaryFrame::from_binary(&edge.add_noise(flip_probability, seed.wrapping_add(1)));
        Ok(corrupted)
    }

    pub fn decode_graph(
        &self,
        representation: &HdcSemanticRepresentation,
    ) -> Result<GroundedConceptGraph, String> {
        let decoded = self.decode_with_rankings(representation)?;
        Ok(decoded.graph)
    }

    pub fn measure_roundtrip(
        &self,
        expected: &GroundedConceptGraph,
        representation: &HdcSemanticRepresentation,
    ) -> Result<HdcSemanticRoundtripMetrics, String> {
        let decoded = self.decode_with_rankings(representation)?;
        let interlingua = compare_graphs(expected, &decoded.graph)?;
        let (node_min_selected_score, node_selection_margin) =
            selection_stats(&decoded.node_candidates, expected.nodes.len())?;
        let (edge_min_selected_score, edge_selection_margin) =
            selection_stats(&decoded.edge_candidates, expected.edges.len())?;

        let representation_bytes = serde_json::to_vec(representation)
            .map_err(|error| error.to_string())?
            .len();

        let metrics = HdcSemanticRoundtripMetrics {
            schema_version: HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION,
            adapter_id: HDC_SEMANTIC_ADAPTER_ID.into(),
            codec_id: representation.codec.codec_id.clone(),
            codebook_hash: self.codebook_hash(),
            expected_graph_hash: graph_hash(expected)?,
            observed_graph_hash: interlingua.observed_graph_hash,
            expected_structural_hash: interlingua.expected_structural_hash,
            observed_structural_hash: interlingua.observed_structural_hash,
            node_precision: interlingua.node_precision,
            node_recall: interlingua.node_recall,
            edge_precision: interlingua.edge_precision,
            edge_recall: interlingua.edge_recall,
            confidence_mae: interlingua.confidence_mae,
            structural_equivalence: interlingua.structural_equivalence,
            node_min_selected_score,
            node_selection_margin,
            edge_min_selected_score,
            edge_selection_margin,
            expected_bytes: interlingua.expected_bytes,
            representation_bytes,
        };

        if !metrics.validates() {
            return Err("HDC semantic roundtrip metrics failed validation".into());
        }
        Ok(metrics)
    }

    pub fn measure_negative_controls(
        &self,
        representation: &HdcSemanticRepresentation,
        expected: &GroundedConceptGraph,
        seed: u64,
    ) -> Result<HdcSemanticNegativeControls, String> {
        let edge_bundle = representation.edge_frame.to_binary()?.to_continuous();

        let unrelated = derive_atom_vector(seed, "negative", "unrelated");
        let unrelated_node_max_similarity = self
            .nodes
            .keys()
            .map(|key| full_cosine_similarity(&unrelated, &self.node_vector(key).unwrap()))
            .fold(f64::NEG_INFINITY, f64::max);

        let expected_node_keys = canonical_node_atoms(expected)?.keys().cloned().collect::<Vec<_>>();
        let unrelated_edge_max_similarity = expected_node_keys
            .iter()
            .flat_map(|source| {
                expected_node_keys.iter().flat_map(move |target| {
                    self.relations.keys().map(move |relation| {
                        self.edge_vector(source, relation, target)
                            .map(|candidate| full_cosine_similarity(&unrelated, &candidate))
                    })
                })
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max);

        let first_edge = expected
            .edges
            .first()
            .ok_or_else(|| "negative controls require at least one edge".to_string())?;
        let source_key = node_key_for_id(expected, &first_edge.source)?;
        let target_key = node_key_for_id(expected, &first_edge.target)?;
        let true_edge = self.edge_vector(&source_key, &first_edge.relation, &target_key)?;
        let swapped_edge = self.edge_vector(&target_key, &first_edge.relation, &source_key)?;

        let controls = HdcSemanticNegativeControls {
            schema_version: HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION,
            unrelated_node_max_similarity,
            unrelated_edge_max_similarity,
            true_edge_similarity: full_cosine_similarity(&edge_bundle, &true_edge),
            swapped_edge_similarity: full_cosine_similarity(&edge_bundle, &swapped_edge),
        };

        if !controls.validates() {
            return Err("HDC semantic negative-control metrics failed validation".into());
        }
        Ok(controls)
    }

    fn node_vector(&self, key: &str) -> Result<ContinuousHV, String> {
        self.nodes
            .get(key)
            .map(|node| self.role_node.bind(&node.vector))
            .ok_or_else(|| format!("unknown node atom: {key}"))
    }

    fn edge_vector(
        &self,
        source: &str,
        relation: &str,
        target: &str,
    ) -> Result<ContinuousHV, String> {
        let source_hv = self
            .nodes
            .get(source)
            .map(|node| node.vector.clone())
            .ok_or_else(|| format!("unknown source node atom: {source}"))?;
        let relation_hv = self
            .relations
            .get(relation)
            .cloned()
            .ok_or_else(|| format!("unknown relation atom: {relation}"))?;
        let target_hv = self
            .nodes
            .get(target)
            .map(|node| node.vector.clone())
            .ok_or_else(|| format!("unknown target node atom: {target}"))?;

        let source_bound = self.role_source.bind(&source_hv);
        let relation_bound = self.role_relation.bind(&relation_hv);
        // ContinuousHV::bind is commutative, so role labels alone cannot encode
        // source/target direction. A fixed permutation makes the target position
        // algebraically distinguishable from the source position.
        let target_bound = self
            .role_target
            .bind(&target_hv.permute(HDC_EDGE_TARGET_PERMUTATION));

        Ok(source_bound
            .bind(&relation_bound)
            .bind(&target_bound))
    }

    fn decode_with_rankings(
        &self,
        representation: &HdcSemanticRepresentation,
    ) -> Result<DecodedHdcGraph, String> {
        if !representation.validates() {
            return Err("invalid HDC semantic representation".into());
        }
        if representation.codebook != self.descriptor {
            return Err("HDC semantic codebook descriptor mismatch".into());
        }
        if representation
            .codec
            .codebook_hash
            .as_deref()
            != Some(self.codebook_hash().as_str())
        {
            return Err("HDC semantic codebook hash mismatch".into());
        }

        let node_bundle = representation.node_frame.to_binary()?.to_continuous();
        let edge_bundle = representation.edge_frame.to_binary()?.to_continuous();

        let mut node_candidates = self
            .nodes
            .keys()
            .map(|key| {
                Ok(HdcRetrievalCandidate {
                    key: key.clone(),
                    score: full_cosine_similarity(&node_bundle, &self.node_vector(key)?),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        sort_candidates(&mut node_candidates);

        if representation.node_count > node_candidates.len() {
            return Err("representation asks for more nodes than the codebook contains".into());
        }
        let selected_nodes = node_candidates
            .iter()
            .take(representation.node_count)
            .map(|candidate| candidate.key.clone())
            .collect::<Vec<_>>();

        let mut edge_candidates = Vec::<HdcRetrievalCandidate>::new();
        for source in &selected_nodes {
            for relation in self.relations.keys() {
                for target in &selected_nodes {
                    let candidate = self.edge_vector(source, relation, target)?;
                    edge_candidates.push(HdcRetrievalCandidate {
                        key: serde_json::to_string(&(source, relation, target))
                            .map_err(|error| error.to_string())?,
                        score: full_cosine_similarity(&edge_bundle, &candidate),
                    });
                }
            }
        }
        sort_candidates(&mut edge_candidates);

        if representation.edge_count > edge_candidates.len() {
            return Err("representation asks for more edges than the decoded node set permits".into());
        }

        let selected_edges = edge_candidates
            .iter()
            .take(representation.edge_count)
            .map(|candidate| parse_edge_candidate(&candidate.key))
            .collect::<Result<Vec<_>, String>>()?;

        let mut decoded_nodes = Vec::with_capacity(selected_nodes.len());
        let mut ids = BTreeMap::new();
        for (index, key) in selected_nodes.iter().enumerate() {
            let atom = self
                .nodes
                .get(key)
                .ok_or_else(|| format!("missing selected node atom: {key}"))?;
            let id = format!("decoded-node-{index}");
            ids.insert(key.clone(), id.clone());
            decoded_nodes.push(ConceptNode {
                id,
                kind: atom.kind.clone(),
                label: None,
                grounded_by: atom.grounded_by.clone(),
                confidence: 1.0,
            });
        }

        let decoded_edges = selected_edges
            .into_iter()
            .map(|(source, relation, target)| {
                Ok(ConceptEdge {
                    source: ids
                        .get(&source)
                        .cloned()
                        .ok_or_else(|| format!("decoded edge source is not selected: {source}"))?,
                    relation,
                    target: ids
                        .get(&target)
                        .cloned()
                        .ok_or_else(|| format!("decoded edge target is not selected: {target}"))?,
                    evidence_ids: Vec::new(),
                    confidence: 1.0,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(DecodedHdcGraph {
            graph: GroundedConceptGraph {
                nodes: decoded_nodes,
                edges: decoded_edges,
            },
            node_candidates,
            edge_candidates,
        })
    }
}

#[derive(Clone, Debug)]
struct DecodedHdcGraph {
    graph: GroundedConceptGraph,
    node_candidates: Vec<HdcRetrievalCandidate>,
    edge_candidates: Vec<HdcRetrievalCandidate>,
}

fn validate_graph_shape(graph: &GroundedConceptGraph) -> Result<(), String> {
    if graph.nodes.is_empty() || graph.edges.is_empty() {
        return Err("HDC semantic interlingua currently requires at least one node and edge".into());
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

fn node_key(node: &ConceptNode) -> String {
    let mut grounded_by = node.grounded_by.clone();
    grounded_by.sort();
    format!("{}|{}", kind_tag(&node.kind), grounded_by.join(","))
}

fn canonical_node_atoms(
    graph: &GroundedConceptGraph,
) -> Result<BTreeMap<String, (ConceptKind, Vec<String>)>, String> {
    let mut result = BTreeMap::new();
    for node in &graph.nodes {
        let key = node_key(node);
        let mut grounded_by = node.grounded_by.clone();
        grounded_by.sort();
        if result
            .insert(key.clone(), (node.kind.clone(), grounded_by))
            .is_some()
        {
            return Err(format!("graph contains duplicate canonical node atom: {key}"));
        }
    }
    Ok(result)
}

fn canonical_edge_atoms(
    graph: &GroundedConceptGraph,
) -> Result<Vec<(String, String, String)>, String> {
    let mut node_keys = BTreeMap::new();
    for node in &graph.nodes {
        let mut grounded_by = node.grounded_by.clone();
        grounded_by.sort();
        node_keys.insert(node.id.clone(), node_key(node));
    }

    let mut edges = Vec::with_capacity(graph.edges.len());
    for edge in &graph.edges {
        let source = node_keys
            .get(&edge.source)
            .cloned()
            .ok_or_else(|| format!("missing source node: {}", edge.source))?;
        let target = node_keys
            .get(&edge.target)
            .cloned()
            .ok_or_else(|| format!("missing target node: {}", edge.target))?;
        edges.push((source, edge.relation.clone(), target));
    }
    edges.sort();
    if edges.windows(2).any(|window| window[0] == window[1]) {
        return Err("graph contains duplicate canonical edge atom".into());
    }
    Ok(edges)
}

fn node_key_for_id(
    graph: &GroundedConceptGraph,
    id: &str,
) -> Result<String, String> {
    graph
        .nodes
        .iter()
        .find(|node| node.id == id)
        .map(node_key)
        .ok_or_else(|| format!("node identifier not found: {id}"))
}

fn parse_edge_candidate(value: &str) -> Result<(String, String, String), String> {
    serde_json::from_str(value)
        .map_err(|error| format!("invalid encoded edge candidate: {value}: {error}"))
}

fn selection_stats(
    candidates: &[HdcRetrievalCandidate],
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

fn sort_candidates(candidates: &mut [HdcRetrievalCandidate]) {
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.key.cmp(&b.key))
    });
}

fn hash_manifest(values: &[String]) -> Result<String, String> {
    let bytes = serde_json::to_vec(values).map_err(|error| error.to_string())?;
    Ok(crate::content_hash(&bytes))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interlingua::reorder_collections;

    fn graph_a() -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: vec![
                ConceptNode {
                    id: "agent-1".into(),
                    kind: ConceptKind::Agent,
                    label: Some("sender".into()),
                    grounded_by: vec!["obs-agent-1".into()],
                    confidence: 0.95,
                },
                ConceptNode {
                    id: "event-1".into(),
                    kind: ConceptKind::Event,
                    label: Some("approach".into()),
                    grounded_by: vec!["obs-event-1".into()],
                    confidence: 0.91,
                },
                ConceptNode {
                    id: "object-1".into(),
                    kind: ConceptKind::Object,
                    label: Some("target".into()),
                    grounded_by: vec!["obs-object-1".into()],
                    confidence: 0.89,
                },
            ],
            edges: vec![
                ConceptEdge {
                    source: "agent-1".into(),
                    relation: "initiates".into(),
                    target: "event-1".into(),
                    evidence_ids: vec!["obs-event-1".into()],
                    confidence: 0.91,
                },
                ConceptEdge {
                    source: "event-1".into(),
                    relation: "targets".into(),
                    target: "object-1".into(),
                    evidence_ids: vec!["obs-object-1".into()],
                    confidence: 0.89,
                },
            ],
        }
    }

    fn training_graphs() -> Vec<GroundedConceptGraph> {
        let base = graph_a();
        let reordered = reorder_collections(&base);
        vec![
            base,
            reordered,
        ]
    }

    #[test]
    fn codebook_is_deterministic_and_provenance_bound() {
        let training = training_graphs();
        let a = HdcSemanticCodebook::from_training_graphs(77, &training).unwrap();
        let b = HdcSemanticCodebook::from_training_graphs(77, &training).unwrap();
        assert_eq!(a.descriptor(), b.descriptor());
        assert_eq!(a.codebook_hash(), b.codebook_hash());

        let different_seed =
            HdcSemanticCodebook::from_training_graphs(78, &training).unwrap();
        assert_ne!(a.codebook_hash(), different_seed.codebook_hash());
    }

    #[test]
    fn representation_roundtrip_retrieves_structure() {
        let training = training_graphs();
        let codebook = HdcSemanticCodebook::from_training_graphs(77, &training).unwrap();
        let representation = codebook.encode_graph(&training[0]).unwrap();
        let decoded = codebook.decode_graph(&representation).unwrap();
        let metrics = codebook.measure_roundtrip(&training[0], &representation).unwrap();

        assert!(representation.validates());
        assert!(metrics.structural_equivalence, "{metrics:#?}");
        assert_eq!(metrics.node_precision, 1.0);
        assert_eq!(metrics.node_recall, 1.0);
        assert_eq!(metrics.edge_precision, 1.0);
        assert_eq!(metrics.edge_recall, 1.0);
        assert_eq!(decoded.nodes.len(), training[0].nodes.len());
        assert_eq!(decoded.edges.len(), training[0].edges.len());
    }

    #[test]
    fn collection_order_does_not_change_the_representation() {
        let training = training_graphs();
        let codebook = HdcSemanticCodebook::from_training_graphs(77, &training).unwrap();

        let a = codebook.encode_graph(&training[0]).unwrap();
        let b = codebook.encode_graph(&reorder_collections(&training[0])).unwrap();

        assert_eq!(a.node_frame, b.node_frame);
        assert_eq!(a.edge_frame, b.edge_frame);
        assert_eq!(a.codebook, b.codebook);
    }

    #[test]
    fn unknown_training_atom_is_rejected() {
        let training = training_graphs();
        let codebook = HdcSemanticCodebook::from_training_graphs(77, &training).unwrap();
        let mut held_out = training[0].clone();
        held_out.nodes[0].grounded_by = vec!["unseen-grounding".into()];

        assert!(codebook.encode_graph(&held_out).is_err());
    }

    #[test]
    fn wrong_codebook_is_rejected_before_interpretation() {
        let training = training_graphs();
        let codebook = HdcSemanticCodebook::from_training_graphs(77, &training).unwrap();
        let wrong = HdcSemanticCodebook::from_training_graphs(78, &training).unwrap();
        let representation = codebook.encode_graph(&training[0]).unwrap();

        assert!(wrong.decode_graph(&representation).is_err());
    }

    #[test]
    fn edge_candidate_key_is_collision_safe_for_delimiters() {
        let key = serde_json::to_string(&("a::source", "rel::type", "b::target")).unwrap();
        let parsed = parse_edge_candidate(&key).unwrap();
        assert_eq!(
            parsed,
            ("a::source".into(), "rel::type".into(), "b::target".into())
        );
    }

    #[test]
    fn deterministic_transport_corruption_is_reproducible() {
        let training = training_graphs();
        let codebook = HdcSemanticCodebook::from_training_graphs(77, &training).unwrap();
        let representation = codebook.encode_graph(&training[0]).unwrap();

        let a = codebook
            .corrupt_for_transport(&representation, 0.01, 1234)
            .unwrap();
        let b = codebook
            .corrupt_for_transport(&representation, 0.01, 1234)
            .unwrap();
        assert_eq!(a.node_frame, b.node_frame);
        assert_eq!(a.edge_frame, b.edge_frame);
    }

    #[test]
    fn swapped_edge_direction_is_a_negative_control() {
        let training = training_graphs();
        let codebook = HdcSemanticCodebook::from_training_graphs(77, &training).unwrap();
        let representation = codebook.encode_graph(&training[0]).unwrap();
        let controls = codebook
            .measure_negative_controls(&representation, &training[0], 1_000_003)
            .unwrap();

        let source_key = node_key_for_id(&training[0], &training[0].edges[0].source).unwrap();
        let target_key = node_key_for_id(&training[0], &training[0].edges[0].target).unwrap();
        let forward = codebook
            .edge_vector(&source_key, &training[0].edges[0].relation, &target_key)
            .unwrap();
        let reverse = codebook
            .edge_vector(&target_key, &training[0].edges[0].relation, &source_key)
            .unwrap();

        assert_ne!(forward, reverse);
        assert!(controls.true_edge_similarity > controls.swapped_edge_similarity);
        assert!(controls.validates());
    }
}
