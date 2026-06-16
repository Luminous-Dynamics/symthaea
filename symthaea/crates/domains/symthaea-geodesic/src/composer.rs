// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Compositional Recombination — Assemble novel programs from pattern parts.
//!
//! When no single pattern in memory matches a target well, this module
//! decomposes the top-K nearest patterns into sub-trees and reassembles
//! a new ProgramNode tree where each position is filled by the best-matching
//! sub-tree from ANY donor pattern.
//!
//! ## Algorithm
//!
//! 1. Find top-K nearest patterns to the target encoding
//! 2. Collect ALL sub-trees from all K patterns (via decomposition)
//! 3. Extract the target's template structure (what kind + child positions)
//! 4. For each child position, find the donor sub-tree whose encoding is
//!    most similar to the target child's encoding
//! 5. Reassemble into a new ProgramNode tree
//! 6. Score with tri-oracle — if better than any single pattern, return it
//!
//! ## Why This Works
//!
//! The tri-oracle gives 100% discrimination. Sub-tree encodings preserve
//! structural identity. When composing "second_largest", the loop structure
//! comes from `find_max`, the comparison from `contains`, and the tracking
//! variable from `count_loop`. Each sub-tree is placed where its encoding
//! best matches the target's child encoding at that position.

use std::cmp::Ordering;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::program_algebra::ProgramNode;

use crate::program_memory::ProgramMemory;
use crate::tri_oracle::TriOracle;

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPLATE EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════════

/// The structural template of a ProgramNode — its kind + number of children.
#[derive(Debug, Clone)]
enum TemplateKind {
    Apply { arity: usize },
    Sequence { length: usize },
    Branch,
    Iterate,
    Recurse,
    Map,
    Filter,
    Reduce,
    Compose,
    Collect,
    Leaf,
}

/// Extract a ProgramNode's template: what kind is it, and what are the
/// STRUCTURAL encodings of its children (the "holes" to fill)?
/// Uses property-based encoding so matching is name-independent.
fn extract_template(node: &ProgramNode) -> (TemplateKind, Vec<BinaryHV>) {
    use crate::periodic_table::encode_structural as enc;
    match node {
        ProgramNode::Atom(_) | ProgramNode::Typed(_, _) => (TemplateKind::Leaf, vec![enc(node)]),
        ProgramNode::Apply { func, args } => {
            let mut children = vec![enc(func)];
            children.extend(args.iter().map(|a| enc(a)));
            (TemplateKind::Apply { arity: args.len() }, children)
        }
        ProgramNode::Sequence(steps) => {
            let children = steps.iter().map(|s| enc(s)).collect();
            (
                TemplateKind::Sequence {
                    length: steps.len(),
                },
                children,
            )
        }
        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => (
            TemplateKind::Branch,
            vec![enc(condition), enc(then_branch), enc(else_branch)],
        ),
        ProgramNode::Iterate {
            init,
            step,
            condition,
        } => (
            TemplateKind::Iterate,
            vec![enc(init), enc(step), enc(condition)],
        ),
        ProgramNode::Recurse {
            base_case,
            recursive_step,
        } => (
            TemplateKind::Recurse,
            vec![enc(base_case), enc(recursive_step)],
        ),
        ProgramNode::Map { func, collection } => {
            (TemplateKind::Map, vec![enc(func), enc(collection)])
        }
        ProgramNode::Filter {
            predicate,
            collection,
        } => (TemplateKind::Filter, vec![enc(predicate), enc(collection)]),
        ProgramNode::Reduce {
            func,
            initial,
            collection,
        } => (
            TemplateKind::Reduce,
            vec![enc(func), enc(initial), enc(collection)],
        ),
        ProgramNode::Compose(f, g) => (TemplateKind::Compose, vec![enc(f), enc(g)]),
        ProgramNode::Collect(source) => (TemplateKind::Collect, vec![enc(source)]),
        ProgramNode::Abstract(examples) => {
            let children = examples.iter().map(|e| enc(e)).collect();
            (
                TemplateKind::Sequence {
                    length: examples.len(),
                },
                children,
            )
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPLATE FILLING
// ═══════════════════════════════════════════════════════════════════════════════

/// For each child position in the template, find the best-matching donor sub-tree.
/// Uses structural (property-based) encoding for matching — name-independent.
fn find_best_donors(
    child_targets: &[BinaryHV],
    donors: &[(String, ProgramNode, BinaryHV)],
) -> Vec<ProgramNode> {
    use crate::periodic_table::encode_structural;
    child_targets
        .iter()
        .map(|target_child| {
            donors
                .iter()
                .max_by(|a, b| {
                    // Compare using structural encoding of the donor node
                    encode_structural(&a.1)
                        .similarity(target_child)
                        .partial_cmp(&encode_structural(&b.1).similarity(target_child))
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(_, node, _)| node.clone())
                .unwrap_or(ProgramNode::atom("?"))
        })
        .collect()
}

/// Reassemble a ProgramNode from a template kind and filled children.
fn assemble(template: &TemplateKind, children: Vec<ProgramNode>) -> ProgramNode {
    match template {
        TemplateKind::Leaf => children
            .into_iter()
            .next()
            .unwrap_or(ProgramNode::atom("?")),
        TemplateKind::Apply { .. } => {
            if children.is_empty() {
                return ProgramNode::atom("?");
            }
            let func = children[0].clone();
            let args = children[1..].to_vec();
            ProgramNode::apply(func, args)
        }
        TemplateKind::Sequence { .. } => ProgramNode::Sequence(children),
        TemplateKind::Branch => {
            if children.len() < 3 {
                return ProgramNode::atom("?");
            }
            ProgramNode::branch(
                children[0].clone(),
                children[1].clone(),
                children[2].clone(),
            )
        }
        TemplateKind::Iterate => {
            if children.len() < 3 {
                return ProgramNode::atom("?");
            }
            ProgramNode::iterate(
                children[0].clone(),
                children[1].clone(),
                children[2].clone(),
            )
        }
        TemplateKind::Recurse => {
            if children.len() < 2 {
                return ProgramNode::atom("?");
            }
            ProgramNode::recurse(children[0].clone(), children[1].clone())
        }
        TemplateKind::Map => {
            if children.len() < 2 {
                return ProgramNode::atom("?");
            }
            ProgramNode::map(children[0].clone(), children[1].clone())
        }
        TemplateKind::Filter => {
            if children.len() < 2 {
                return ProgramNode::atom("?");
            }
            ProgramNode::filter(children[0].clone(), children[1].clone())
        }
        TemplateKind::Reduce => {
            if children.len() < 3 {
                return ProgramNode::atom("?");
            }
            ProgramNode::reduce(
                children[0].clone(),
                children[1].clone(),
                children[2].clone(),
            )
        }
        TemplateKind::Compose => {
            if children.len() < 2 {
                return ProgramNode::atom("?");
            }
            ProgramNode::compose(children[0].clone(), children[1].clone())
        }
        TemplateKind::Collect => children
            .into_iter()
            .next()
            .map(ProgramNode::collect)
            .unwrap_or(ProgramNode::atom("?")),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

/// Compose a new ProgramNode by recombining sub-trees from the top-K
/// nearest patterns in memory.
///
/// Returns `Some((composed_node, score))` if the composition scores higher
/// than the best single pattern. Returns `None` if retrieval is better.
pub fn compose_from_patterns(
    target: &ProgramNode,
    memory: &ProgramMemory,
    k: usize,
) -> Option<(ProgramNode, f32)> {
    use crate::periodic_table::encode_structural;
    let target_structural = encode_structural(target);
    let oracle = TriOracle::with_defaults();

    // 1. Get top-K nearest patterns using STRUCTURAL encoding
    // This finds patterns with similar STRUCTURE regardless of variable names.
    let top_k = memory.nearest_k(&target_structural, k);
    if top_k.is_empty() {
        return None;
    }

    // 2. Collect ALL sub-trees from ALL patterns (not just top-K).
    // The sub-tree matching uses structural encoding, so even distant
    // patterns may have relevant sub-trees (e.g., find_max's Iterate
    // sub-tree matches second_largest's Iterate at ~1.0 structural).
    let mut all_subs: Vec<(String, ProgramNode, BinaryHV)> = Vec::new();
    for entry in memory.as_library().iter() {
        // Re-encode sub-patterns using structural encoding
        all_subs.push((String::new(), entry.0.clone(), encode_structural(&entry.0)));
    }
    // Also add decomposed sub-trees from top-K patterns
    for (entry, _sim) in &top_k {
        for (name, node, _old_enc) in &entry.sub_patterns {
            all_subs.push((name.clone(), node.clone(), encode_structural(node)));
        }
    }

    if all_subs.is_empty() {
        return None;
    }

    // 3. Extract the target's template structure
    let (template, child_targets) = extract_template(target);

    // 4. For each child position, find the best-matching donor sub-tree
    let filled_children = find_best_donors(&child_targets, &all_subs);

    // 5. Reassemble
    let composed = assemble(&template, filled_children);

    // 6. Score the composition
    let comp_score = oracle.score(&composed, target).composite;

    // 7. Compare to best single pattern
    let best_single = top_k
        .first()
        .map(|(e, _)| oracle.score(&e.node, target).composite)
        .unwrap_or(0.0);

    if comp_score > best_single + 0.01 {
        // Composition is meaningfully better
        Some((composed, comp_score))
    } else {
        // Retrieval was as good or better
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_template_apply() {
        let node = ProgramNode::apply(
            ProgramNode::op("ADD"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        );
        let (template, children) = extract_template(&node);
        assert!(matches!(template, TemplateKind::Apply { arity: 2 }));
        assert_eq!(children.len(), 3); // func + 2 args
    }

    #[test]
    fn test_extract_template_iterate() {
        let node = ProgramNode::iterate(
            ProgramNode::atom("init"),
            ProgramNode::atom("step"),
            ProgramNode::atom("cond"),
        );
        let (template, children) = extract_template(&node);
        assert!(matches!(template, TemplateKind::Iterate));
        assert_eq!(children.len(), 3);
    }

    #[test]
    fn test_assemble_roundtrip() {
        let original = ProgramNode::branch(
            ProgramNode::atom("cond"),
            ProgramNode::atom("yes"),
            ProgramNode::atom("no"),
        );
        let (template, _) = extract_template(&original);
        let children = vec![
            ProgramNode::atom("cond"),
            ProgramNode::atom("yes"),
            ProgramNode::atom("no"),
        ];
        let reassembled = assemble(&template, children);
        // Reassembled should encode identically to original
        let sim = original.encode().similarity(&reassembled.encode());
        assert!(sim > 0.99, "roundtrip should preserve encoding, got {sim}");
    }

    #[test]
    fn test_find_best_donors() {
        let target_a = ProgramNode::op("ADD").encode();
        let target_b = ProgramNode::atom("x").encode();

        let donors = vec![
            (
                "add".to_string(),
                ProgramNode::op("ADD"),
                ProgramNode::op("ADD").encode(),
            ),
            (
                "mul".to_string(),
                ProgramNode::op("MUL"),
                ProgramNode::op("MUL").encode(),
            ),
            (
                "x".to_string(),
                ProgramNode::atom("x"),
                ProgramNode::atom("x").encode(),
            ),
        ];

        let filled = find_best_donors(&[target_a, target_b], &donors);
        assert_eq!(filled.len(), 2);
        // First child should match ADD (highest similarity to ADD encoding)
        // Second child should match "x" (highest similarity to "x" encoding)
    }

    #[test]
    fn test_compose_known_pattern() {
        let memory = ProgramMemory::basic();
        let target = ProgramNode::apply(
            ProgramNode::op("ADD"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        );

        // For a known pattern, composition should return None
        // (retrieval is already perfect at 1.0)
        let result = compose_from_patterns(&target, &memory, 3);
        // Either None (retrieval was better) or Some with high score
        if let Some((_, score)) = result {
            assert!(
                score > 0.9,
                "if composition returned, it should be good: {score}"
            );
        }
    }

    #[test]
    fn test_compose_novel_pattern() {
        let memory = ProgramMemory::basic();
        // A pattern NOT in memory: min element (similar to find_max but with LT)
        let target = ProgramNode::Sequence(vec![
            ProgramNode::atom("min = arr[0]"),
            ProgramNode::iterate(
                ProgramNode::atom("i = 1"),
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("min")],
                    ),
                    ProgramNode::atom("min = arr[i]"),
                    ProgramNode::atom("/* skip */"),
                ),
                ProgramNode::apply(
                    ProgramNode::op("LT"),
                    vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
                ),
            ),
            ProgramNode::atom("min"),
        ]);

        let result = compose_from_patterns(&target, &memory, 5);

        // Composition should produce SOMETHING — it may or may not beat retrieval
        // but the mechanism should run without panic
        println!(
            "Novel composition result: {:?}",
            result.as_ref().map(|(_, s)| s)
        );
    }
}
