// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Domain-neutral HDC representation for scientific world models.
//!
//! This module deliberately stops at representation and structural comparison.
//! HDC similarity is a computational relationship, not evidence that two
//! mechanisms are biologically equivalent or that either hypothesis is true.
//!
//! It uses the existing 16,384-bit BinaryHV substrate rather than introducing
//! a second vector implementation.

use crate::hdc::BinaryHV;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Stable role labels used to compose scientific relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScientificRole {
    /// The entity playing the causal-source role.
    Subject,
    /// The entity playing the causal-target role.
    Object,
    /// A relation label such as activates or requires.
    Relation,
    /// A contextual condition or boundary condition.
    Context,
    /// A mechanism identifier.
    Mechanism,
    /// A phenotype or measurable outcome.
    Outcome,
    /// A perturbation applied to a state.
    Perturbation,
}

/// A named scientific entity.
///
/// The id is an external identity, not a claim about biological validity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificEntity {
    /// Stable caller-supplied identity.
    pub id: String,
    /// Domain-neutral kind, e.g. protein, process, phenotype, condition.
    pub kind: String,
}

/// A typed relation between two scientific entities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificRelation {
    /// Subject entity identity.
    pub subject: String,
    /// Relation identity.
    pub relation: String,
    /// Object entity identity.
    pub object: String,
}

/// A compositional scientific state.
///
/// Facts are retained alongside the HDC encoding so the vector never becomes
/// the sole source of scientific meaning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificState {
    /// Explicit entities participating in the state.
    pub entities: Vec<ScientificEntity>,
    /// Explicit typed relations participating in the state.
    pub relations: Vec<ScientificRelation>,
    /// Optional named context.
    pub context: Option<String>,
    /// Deterministic HDC representation of the state.
    pub encoding: BinaryHV,
}

/// Cross-problem motif represented independently of any single biological
/// challenge. Motifs are intended for analogy/retrieval, not truth scoring.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificMotif {
    /// Stable motif identity.
    pub id: String,
    /// Human-readable structural description.
    pub description: String,
    /// Structural HDC fingerprint.
    pub encoding: BinaryHV,
}

/// Deterministic HDC encoder for domain-neutral scientific structures.
#[derive(Debug, Clone, Copy)]
pub struct ScientificHdcEncoder {
    seed: u64,
}

impl Default for ScientificHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ScientificHdcEncoder {
    /// Construct an encoder with a stable default codebook seed.
    pub const fn new() -> Self {
        Self { seed: 0x5343_4945_4E43_4557 }
    }

    /// Construct an encoder with an explicit reproducibility seed.
    pub const fn with_seed(seed: u64) -> Self {
        Self { seed }
    }

    /// Encode a stable symbol into the existing BinaryHV code space.
    ///
    /// This is a reproducible codebook operation, not a cryptographic hash.
    pub fn symbol(&self, namespace: &str, value: &str) -> BinaryHV {
        BinaryHV::random(self.symbol_seed(namespace, value))
    }

    /// Encode a typed binary relation with role binding.
    ///
    /// Role vectors prevent the same three symbols in different positions
    /// from collapsing to the same representation.
    pub fn relation(&self, relation: &ScientificRelation) -> BinaryHV {
        let subject = self.symbol("entity", &relation.subject);
        let predicate = self.symbol("relation", &relation.relation);
        let object = self.symbol("entity", &relation.object);

        let subject_role = self.symbol("role", "subject").bind(&subject);
        let relation_role = self.symbol("role", "relation").bind(&predicate);
        let object_role = self.symbol("role", "object").bind(&object);

        BinaryHV::bundle(&[subject_role, relation_role, object_role])
    }

    /// Encode an explicit state by superposing its typed relations and context.
    ///
    /// Empty states are rejected because there is no meaningful structural
    /// fingerprint to compare.
    pub fn state(
        &self,
        entities: &[ScientificEntity],
        relations: &[ScientificRelation],
        context: Option<&str>,
    ) -> Option<ScientificState> {
        if entities.is_empty() && relations.is_empty() && context.is_none() {
            return None;
        }

        let mut components = Vec::with_capacity(relations.len() + 1);
        for relation in relations {
            components.push(self.relation(relation));
        }

        if let Some(context) = context {
            let context_role = self.symbol("role", "context");
            components.push(context_role.bind(&self.symbol("context", context)));
        }

        if components.is_empty() {
            components.push(self.symbol("empty", "entity-set"));
        }

        Some(ScientificState {
            entities: entities.to_vec(),
            relations: relations.to_vec(),
            context: context.map(str::to_owned),
            encoding: BinaryHV::bundle(&components),
        })
    }

    /// Encode an abstract mechanism/outcome motif.
    pub fn motif(
        &self,
        id: impl Into<String>,
        description: impl Into<String>,
        mechanism: &str,
        outcome: &str,
        perturbation: &str,
    ) -> ScientificMotif {
        let mechanism_hv = self
            .symbol("role", "mechanism")
            .bind(&self.symbol("mechanism", mechanism));
        let outcome_hv = self
            .symbol("role", "outcome")
            .bind(&self.symbol("outcome", outcome));
        let perturbation_hv = self
            .symbol("role", "perturbation")
            .bind(&self.symbol("perturbation", perturbation));

        ScientificMotif {
            id: id.into(),
            description: description.into(),
            encoding: BinaryHV::bundle(&[mechanism_hv, outcome_hv, perturbation_hv]),
        }
    }

    /// Return structural similarity between two scientific states.
    ///
    /// This metric is descriptive only; it does not establish mechanistic
    /// equivalence, causal validity, or experimental evidence.
    pub fn similarity(&self, left: &ScientificState, right: &ScientificState) -> f32 {
        left.encoding.similarity(&right.encoding)
    }

    /// Rank candidates by HDC structural similarity, preserving deterministic
    /// input order for ties.
    pub fn rank_similar(
        &self,
        query: &ScientificState,
        candidates: &[ScientificState],
    ) -> Vec<(usize, f32)> {
        let mut ranked: Vec<_> = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| (index, self.similarity(query, candidate)))
            .collect();

        ranked.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_index.cmp(right_index))
        });
        ranked
    }

    /// Build a deterministic role-labelled representation map for audit output.
    pub fn role_bindings(
        &self,
        subject: &str,
        relation: &str,
        object: &str,
    ) -> BTreeMap<String, BinaryHV> {
        BTreeMap::from([
            (
                "object".to_owned(),
                self.symbol("role", "object")
                    .bind(&self.symbol("entity", object)),
            ),
            (
                "relation".to_owned(),
                self.symbol("role", "relation")
                    .bind(&self.symbol("relation", relation)),
            ),
            (
                "subject".to_owned(),
                self.symbol("role", "subject")
                    .bind(&self.symbol("entity", subject)),
            ),
        ])
    }

    fn symbol_seed(&self, namespace: &str, value: &str) -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed.to_le_bytes());
        hasher.update(namespace.as_bytes());
        hasher.update(&[0]);
        hasher.update(value.as_bytes());
        let digest = hasher.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest.as_bytes()[..8]);
        u64::from_le_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relation(subject: &str, predicate: &str, object: &str) -> ScientificRelation {
        ScientificRelation {
            subject: subject.to_owned(),
            relation: predicate.to_owned(),
            object: object.to_owned(),
        }
    }

    #[test]
    fn encoding_is_deterministic_across_encoder_instances() {
        let left = ScientificHdcEncoder::new();
        let right = ScientificHdcEncoder::new();
        assert_eq!(
            left.relation(&relation("rubisco", "catalyses", "carbon_fixation")),
            right.relation(&relation("rubisco", "catalyses", "carbon_fixation"))
        );
    }

    #[test]
    fn role_binding_preserves_argument_order() {
        let encoder = ScientificHdcEncoder::new();
        let forward = encoder.relation(&relation("a", "causes", "b"));
        let reversed = encoder.relation(&relation("b", "causes", "a"));
        assert_ne!(forward, reversed);
    }

    #[test]
    fn unrelated_symbols_are_not_identical() {
        let encoder = ScientificHdcEncoder::new();
        assert_ne!(encoder.symbol("entity", "a"), encoder.symbol("entity", "b"));
    }

    #[test]
    fn state_keeps_explicit_semantics_next_to_vector() {
        let encoder = ScientificHdcEncoder::new();
        let state = encoder
            .state(
                &[
                    ScientificEntity { id: "protease".into(), kind: "enzyme".into() },
                    ScientificEntity { id: "target".into(), kind: "protein".into() },
                ],
                &[relation("protease", "cleaves", "target")],
                Some("living_cell"),
            )
            .expect("non-empty state");
        assert_eq!(state.relations.len(), 1);
        assert_eq!(state.context.as_deref(), Some("living_cell"));
    }

    #[test]
    fn empty_state_is_rejected() {
        let encoder = ScientificHdcEncoder::new();
        assert!(encoder.state(&[], &[], None).is_none());
    }

    #[test]
    fn identical_structures_have_unit_similarity() {
        let encoder = ScientificHdcEncoder::new();
        let left = encoder.state(&[], &[relation("a", "activates", "b")], None).unwrap();
        let right = encoder.state(&[], &[relation("a", "activates", "b")], None).unwrap();
        assert!((encoder.similarity(&left, &right) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ranking_is_deterministic_and_returns_original_indices() {
        let encoder = ScientificHdcEncoder::new();
        let query = encoder.state(&[], &[relation("a", "activates", "b")], None).unwrap();
        let candidates = vec![
            encoder.state(&[], &[relation("x", "requires", "y")], None).unwrap(),
            query.clone(),
        ];
        let ranked = encoder.rank_similar(&query, &candidates);
        assert_eq!(ranked[0].0, 1);
        assert_eq!(ranked[0].1, 1.0);
    }

    #[test]
    fn motif_fingerprint_is_reproducible() {
        let encoder = ScientificHdcEncoder::with_seed(7);
        let a = encoder.motif("m", "description", "translation", "protein", "perturb");
        let b = encoder.motif("m", "description", "translation", "protein", "perturb");
        assert_eq!(a.encoding, b.encoding);
    }
}
