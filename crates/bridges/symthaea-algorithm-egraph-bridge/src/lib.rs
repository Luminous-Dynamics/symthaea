// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-preserving bridge from Symthaea's EML equality-saturation laboratory to the generic
//! algorithm registry.
//!
//! This bridge distinguishes:
//!
//! ```text
//! rules present in a saturation run
//! != rules newly applied somewhere in the e-graph
//! != one proof path from parent to extracted candidate
//! != semantic correctness for a downstream algorithm problem
//! ```
//!
//! `egg` explanations provide one concrete rewrite proof between the input expression and the
//! extracted expression. The resulting ordered, directional steps can be represented as
//! `AlgorithmLineage`, but that lineage is still derivation evidence rather than a domain-level
//! correctness theorem.

use egg::{AstSize, Extractor, FlatTerm, RecExpr, Runner};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::{
    AlgorithmLineage, ContentId, ImplementationId, RegistryError, TransformationRecord,
};
use symthaea_eml_egraph::{rewrite_rules, SymbolLang};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EgraphBridgeError {
    #[error("iteration limit must be positive")]
    ZeroIterationLimit,
    #[error("failed to parse EML/egg expression `{expression}`: {detail}")]
    Parse { expression: String, detail: String },
    #[error("explanation step contained no rewrite annotation")]
    MissingRewriteAnnotation,
    #[error("explanation step contained multiple rewrite annotations")]
    AmbiguousRewriteAnnotation,
    #[error("explanation referenced unknown rewrite rule `{0}`")]
    UnknownRewrite(String),
    #[error(transparent)]
    Registry(#[from] RegistryError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteDirection {
    Forward,
    Backward,
}

impl RewriteDirection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Backward => "backward",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteProofStep {
    pub rule_name: String,
    pub direction: RewriteDirection,
    pub transformation: TransformationRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IterationApplications {
    pub iteration: usize,
    pub rule_counts: BTreeMap<String, usize>,
}

/// One proof-producing equality-saturation result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EgraphRewriteProof {
    pub id: ContentId,
    pub rewrite_source_artifact_id: ContentId,
    pub input_expression: String,
    pub extracted_expression: String,
    pub iteration_limit: usize,
    pub stop_reason: String,
    pub proof_steps: Vec<RewriteProofStep>,
    pub iteration_applications: Vec<IterationApplications>,
}

impl EgraphRewriteProof {
    pub fn validate(&self) -> Result<(), EgraphBridgeError> {
        if self.iteration_limit == 0 {
            return Err(EgraphBridgeError::ZeroIterationLimit);
        }
        let known_rules = rewrite_rule_names();
        for step in &self.proof_steps {
            if !known_rules.contains(step.rule_name.as_str()) {
                return Err(EgraphBridgeError::UnknownRewrite(step.rule_name.clone()));
            }
            step.transformation.validate()?;
            let expected = directional_transformation(
                &step.rule_name,
                step.direction,
                &self.rewrite_source_artifact_id,
            )?;
            if expected != step.transformation {
                return Err(EgraphBridgeError::Registry(
                    RegistryError::IdentityMismatch {
                        kind: "transformation",
                    },
                ));
            }
        }
        let expected = derive_proof_id(
            &self.rewrite_source_artifact_id,
            &self.input_expression,
            &self.extracted_expression,
            self.iteration_limit,
            &self.stop_reason,
            &self.proof_steps,
            &self.iteration_applications,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(EgraphBridgeError::Registry(
                RegistryError::IdentityMismatch {
                    kind: "egraph rewrite proof",
                },
            ))
        }
    }
}

/// Stable set of rule names exported by the executing e-graph rule factory.
///
/// This intentionally records names only. Exact rule semantics are bound through the caller-
/// supplied `rewrite_source_artifact_id`, which should eventually come from the experiment
/// capsule collector rather than a hand-entered label.
pub fn rewrite_rule_names() -> BTreeSet<String> {
    rewrite_rules()
        .into_iter()
        .map(|rewrite| rewrite.name.to_string())
        .collect()
}

pub fn directional_transformation(
    rule_name: &str,
    direction: RewriteDirection,
    rewrite_source_artifact_id: &ContentId,
) -> Result<TransformationRecord, RegistryError> {
    TransformationRecord::new(
        format!("egg:{rule_name}:{}", direction.as_str()),
        format!(
            "Directional application of egg rewrite `{rule_name}` ({}) bound to exact rewrite-set source artifact {}.",
            direction.as_str(),
            rewrite_source_artifact_id
        ),
    )
}

/// Run equality saturation with explanations enabled and bind one concrete proof from the input
/// expression to the AstSize-minimal extracted expression.
pub fn extract_with_proof(
    expression: &str,
    rewrite_source_artifact_id: ContentId,
    iteration_limit: usize,
) -> Result<EgraphRewriteProof, EgraphBridgeError> {
    if iteration_limit == 0 {
        return Err(EgraphBridgeError::ZeroIterationLimit);
    }
    let input: RecExpr<SymbolLang> = expression
        .parse()
        .map_err(|error| EgraphBridgeError::Parse {
            expression: expression.to_string(),
            detail: format!("{error:?}"),
        })?;
    let rules = rewrite_rules();
    let known_rules: BTreeSet<String> = rules.iter().map(|rule| rule.name.to_string()).collect();

    let mut runner = Runner::<SymbolLang, ()>::default()
        .with_explanations_enabled()
        .with_iter_limit(iteration_limit)
        .with_expr(&input)
        .run(&rules);

    let root = runner.egraph.find(runner.roots[0]);
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_cost, extracted) = extractor.find_best(root);
    let extracted_expression = extracted.to_string();

    let iteration_applications = runner
        .iterations
        .iter()
        .enumerate()
        .map(|(iteration, data)| IterationApplications {
            iteration,
            rule_counts: data
                .applied
                .iter()
                .map(|(name, count)| (name.to_string(), *count))
                .collect(),
        })
        .collect::<Vec<_>>();

    let mut explanation = runner.explain_equivalence(&input, &extracted);
    let flat = explanation.make_flat_explanation();
    let mut proof_steps = Vec::with_capacity(flat.len().saturating_sub(1));
    for term in flat.iter().skip(1) {
        let mut annotations = Vec::new();
        collect_rewrite_annotations(term, &mut annotations);
        if annotations.is_empty() {
            return Err(EgraphBridgeError::MissingRewriteAnnotation);
        }
        if annotations.len() != 1 {
            return Err(EgraphBridgeError::AmbiguousRewriteAnnotation);
        }
        let (direction, rule_name) = annotations.remove(0);
        if !known_rules.contains(&rule_name) {
            return Err(EgraphBridgeError::UnknownRewrite(rule_name));
        }
        let transformation = directional_transformation(
            &rule_name,
            direction,
            &rewrite_source_artifact_id,
        )?;
        proof_steps.push(RewriteProofStep {
            rule_name,
            direction,
            transformation,
        });
    }

    let stop_reason = format!("{:?}", runner.stop_reason);
    let id = derive_proof_id(
        &rewrite_source_artifact_id,
        expression,
        &extracted_expression,
        iteration_limit,
        &stop_reason,
        &proof_steps,
        &iteration_applications,
    );
    let proof = EgraphRewriteProof {
        id,
        rewrite_source_artifact_id,
        input_expression: expression.to_string(),
        extracted_expression,
        iteration_limit,
        stop_reason,
        proof_steps,
        iteration_applications,
    };
    proof.validate()?;
    Ok(proof)
}

/// Convert one proof path into the generic ordered algorithm-lineage representation.
///
/// The parent is the exact implementation whose expression/artifact entered the e-graph; the
/// candidate is the exact extracted implementation. This function does not assert that either
/// implementation satisfies any domain-level `ProblemSpec`.
pub fn lineage_from_proof(
    candidate_id: ImplementationId,
    parent_id: ImplementationId,
    proof: &EgraphRewriteProof,
) -> Result<AlgorithmLineage, EgraphBridgeError> {
    proof.validate()?;
    let transformation_ids = proof
        .proof_steps
        .iter()
        .map(|step| step.transformation.id.clone())
        .collect();
    Ok(AlgorithmLineage::new(
        candidate_id,
        vec![parent_id],
        transformation_ids,
    )?)
}

fn collect_rewrite_annotations(
    term: &FlatTerm<SymbolLang>,
    annotations: &mut Vec<(RewriteDirection, String)>,
) {
    if let Some(rule) = term.forward_rule {
        annotations.push((RewriteDirection::Forward, rule.to_string()));
    }
    if let Some(rule) = term.backward_rule {
        annotations.push((RewriteDirection::Backward, rule.to_string()));
    }
    for child in &term.children {
        collect_rewrite_annotations(child, annotations);
    }
}

fn derive_proof_id(
    rewrite_source_artifact_id: &ContentId,
    input_expression: &str,
    extracted_expression: &str,
    iteration_limit: usize,
    stop_reason: &str,
    proof_steps: &[RewriteProofStep],
    iteration_applications: &[IterationApplications],
) -> ContentId {
    let iteration_limit = (iteration_limit as u64).to_be_bytes();
    let mut parts = vec![
        rewrite_source_artifact_id.as_str().as_bytes().to_vec(),
        input_expression.as_bytes().to_vec(),
        extracted_expression.as_bytes().to_vec(),
        iteration_limit.to_vec(),
        stop_reason.as_bytes().to_vec(),
    ];
    for step in proof_steps {
        parts.push(step.rule_name.as_bytes().to_vec());
        parts.push(step.direction.as_str().as_bytes().to_vec());
        parts.push(step.transformation.id.as_content_id().as_str().as_bytes().to_vec());
    }
    for iteration in iteration_applications {
        parts.push((iteration.iteration as u64).to_be_bytes().to_vec());
        for (rule_name, count) in &iteration.rule_counts {
            parts.push(rule_name.as_bytes().to_vec());
            parts.push((*count as u64).to_be_bytes().to_vec());
        }
    }
    ContentId::derive(
        "symthaea.algorithm-egraph-proof.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    #[test]
    fn extracts_smaller_expression_with_concrete_rewrite_proof() {
        let proof = extract_with_proof(
            "(mul x 1)",
            cid("rewrite-source", "eml-egraph-test"),
            12,
        )
        .unwrap();
        assert_eq!(proof.extracted_expression, "x");
        assert!(!proof.proof_steps.is_empty());
        assert!(proof
            .proof_steps
            .iter()
            .any(|step| step.rule_name == "mul-one-right"));
        assert!(proof.validate().is_ok());
    }

    #[test]
    fn proof_records_saturation_activity_separately_from_selected_path() {
        let proof = extract_with_proof(
            "(mul (mul x 1) 1)",
            cid("rewrite-source", "eml-egraph-test"),
            12,
        )
        .unwrap();
        let total_applications: usize = proof
            .iteration_applications
            .iter()
            .flat_map(|iteration| iteration.rule_counts.values())
            .sum();
        assert!(total_applications >= proof.proof_steps.len());
    }

    #[test]
    fn ordered_directional_proof_becomes_ordered_algorithm_lineage() {
        let proof = extract_with_proof(
            "(mul (mul x 1) 1)",
            cid("rewrite-source", "eml-egraph-test"),
            12,
        )
        .unwrap();
        let candidate = ImplementationId(cid("implementation", "candidate"));
        let parent = ImplementationId(cid("implementation", "parent"));
        let lineage = lineage_from_proof(candidate, parent, &proof).unwrap();
        assert_eq!(lineage.transformation_ids.len(), proof.proof_steps.len());
        for (lineage_id, proof_step) in lineage
            .transformation_ids
            .iter()
            .zip(proof.proof_steps.iter())
        {
            assert_eq!(lineage_id, &proof_step.transformation.id);
        }
    }

    #[test]
    fn rewrite_source_artifact_changes_transformation_identity() {
        let a = directional_transformation(
            "mul-one-right",
            RewriteDirection::Forward,
            &cid("source", "a"),
        )
        .unwrap();
        let b = directional_transformation(
            "mul-one-right",
            RewriteDirection::Forward,
            &cid("source", "b"),
        )
        .unwrap();
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn forward_and_backward_are_distinct_transformations() {
        let source = cid("source", "same");
        let forward = directional_transformation(
            "mul-one-right",
            RewriteDirection::Forward,
            &source,
        )
        .unwrap();
        let backward = directional_transformation(
            "mul-one-right",
            RewriteDirection::Backward,
            &source,
        )
        .unwrap();
        assert_ne!(forward.id, backward.id);
    }
}
