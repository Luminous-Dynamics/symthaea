// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Metamorphic checks for the dependency-aware truth-maintenance reference graph.
//!
//! The oracle below deliberately scans the canonical *forward* dependency sets
//! to a fixed point. It does not inspect or reuse the implementation's reverse
//! index or invalidation traversal. This provides a second algorithm for
//! checking source-retraction and derived-root invalidation over generated DAGs.
//!
//! These tests establish in-memory semantic properties only. They do not test
//! durable transactions, process-crash recovery, or production-worker behavior.

use std::collections::{BTreeMap, BTreeSet};

use symthaea_broca::memory_derivation::{
    DependencyRef, DerivationGraph, DerivedFactInput,
};

const EVENT_COUNT: usize = 8;
const FACT_COUNT: usize = 64;
const GENERATED_SEEDS: u64 = 32;

fn next_random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn event_id(index: usize) -> String {
    format!("E{index:02}")
}

fn fact_id(index: usize) -> String {
    format!("D{index:03}")
}

/// Build a logical DAG plan independently of the order in which nodes are
/// inserted into the production graph. Each node has at least one source leaf.
fn generated_plan(seed: u64) -> BTreeMap<String, DerivedFactInput> {
    let mut rng = seed;
    let mut plan = BTreeMap::new();

    for index in 0..FACT_COUNT {
        let mut dependencies = BTreeSet::new();
        let source_index = ((next_random(&mut rng) >> 32) as usize) % EVENT_COUNT;
        dependencies.insert(DependencyRef::SourceEvent(event_id(source_index)));

        for parent_index in 0..index {
            if (next_random(&mut rng) >> 32) % 23 == 0 {
                dependencies.insert(DependencyRef::DerivedFact(fact_id(parent_index)));
            }
        }

        let id = fact_id(index);
        plan.insert(
            id.clone(),
            DerivedFactInput {
                fact_id: id,
                subject: "generated-subject".into(),
                attribute: "derived-state".into(),
                value: "present".into(),
                valid_from: "2026-01-01".into(),
                valid_to: None,
                rule_id: "generated-rule".into(),
                rule_version: "v1".into(),
                dependencies,
            },
        );
    }

    plan
}

/// Produce a valid, seed-dependent topological insertion order. Choosing among
/// currently ready vertices permits multiple valid construction orders for the
/// same logical plan.
fn topological_order(
    plan: &BTreeMap<String, DerivedFactInput>,
    mut ordering_seed: u64,
) -> Vec<String> {
    let mut remaining: BTreeSet<String> = plan.keys().cloned().collect();
    let mut emitted = BTreeSet::new();
    let mut order = Vec::with_capacity(plan.len());

    while !remaining.is_empty() {
        let mut ready: Vec<String> = remaining
            .iter()
            .filter(|id| {
                plan.get(*id)
                    .expect("remaining ID must exist in the plan")
                    .dependencies
                    .iter()
                    .all(|dependency| match dependency {
                        DependencyRef::SourceEvent(_) => true,
                        DependencyRef::DerivedFact(parent_id) => emitted.contains(parent_id),
                    })
            })
            .cloned()
            .collect();

        ready.sort();
        assert!(!ready.is_empty(), "generated dependency plan must be acyclic");
        let selected = ((next_random(&mut ordering_seed) >> 32) as usize) % ready.len();
        let id = ready.remove(selected);
        remaining.remove(&id);
        emitted.insert(id.clone());
        order.push(id);
    }

    order
}

fn build_graph(
    plan: &BTreeMap<String, DerivedFactInput>,
    ordering_seed: u64,
    reverse_source_registration: bool,
) -> DerivationGraph {
    let mut graph = DerivationGraph::default();
    let mut event_indices: Vec<usize> = (0..EVENT_COUNT).collect();
    if reverse_source_registration {
        event_indices.reverse();
    }
    for index in event_indices {
        graph.register_source_event(event_id(index)).unwrap();
    }

    for id in topological_order(plan, ordering_seed) {
        graph.derive(plan.get(&id).unwrap().clone()).unwrap();
    }

    graph.verify_consistency().unwrap();
    graph
}

/// Independent oracle: repeatedly scan all forward dependency sets until no
/// more descendants can be reached from the mutated dependency. For a derived
/// root, the root itself is part of the expected invalidation set.
fn forward_scan_closure(
    plan: &BTreeMap<String, DerivedFactInput>,
    root: DependencyRef,
) -> Vec<String> {
    let mut affected = BTreeSet::new();
    if let DependencyRef::DerivedFact(root_id) = &root {
        assert!(plan.contains_key(root_id), "oracle root must exist");
        affected.insert(root_id.clone());
    }

    loop {
        let prior_len = affected.len();
        for (id, fact) in plan {
            if affected.contains(id) {
                continue;
            }

            let directly_depends_on_root = fact.dependencies.contains(&root);
            let depends_on_affected_parent = fact.dependencies.iter().any(|dependency| {
                matches!(
                    dependency,
                    DependencyRef::DerivedFact(parent_id) if affected.contains(parent_id)
                )
            });

            if directly_depends_on_root || depends_on_affected_parent {
                affected.insert(id.clone());
            }
        }

        if affected.len() == prior_len {
            break;
        }
    }

    affected.into_iter().collect()
}

fn assert_visible_state_matches(
    graph: &DerivationGraph,
    plan: &BTreeMap<String, DerivedFactInput>,
    expected_invalidated: &[String],
    context: &str,
) {
    let expected: BTreeSet<&str> = expected_invalidated.iter().map(String::as_str).collect();
    for id in plan.keys() {
        if expected.contains(id.as_str()) {
            assert!(
                graph.current_fact(id).is_none(),
                "{context}: invalidated fact {id} must not be current"
            );
        } else {
            assert!(
                graph.current_fact(id).is_some(),
                "{context}: unrelated fact {id} was incorrectly invalidated"
            );
        }
    }
    assert_eq!(
        graph.current_fact_count(),
        plan.len() - expected_invalidated.len(),
        "{context}: current-fact count disagrees with oracle"
    );
    assert_eq!(
        graph.invalidated_fact_count(),
        expected_invalidated.len(),
        "{context}: audit history must retain each invalidated fact exactly once"
    );
    graph.verify_consistency().unwrap();
}

#[test]
fn generated_dags_match_forward_scan_oracle_across_insertion_orders_and_rebuilds() {
    for seed in 1..=GENERATED_SEEDS {
        let plan = generated_plan(seed);
        let original_a = build_graph(&plan, seed.wrapping_mul(17).wrapping_add(1), false);
        let original_b = build_graph(&plan, seed.wrapping_mul(31).wrapping_add(7), true);

        let original_digest = original_a.state_sha256().unwrap();
        assert_eq!(
            original_digest,
            original_b.state_sha256().unwrap(),
            "seed {seed}: equivalent DAGs must have an insertion-order-independent digest"
        );

        for index in 0..EVENT_COUNT {
            let root_id = event_id(index);
            let root = DependencyRef::SourceEvent(root_id.clone());
            let expected = forward_scan_closure(&plan, root);

            let mut left = original_a.clone();
            let actual = left
                .invalidate_source_event(&root_id, "generated metamorphic retraction")
                .unwrap();
            assert_eq!(actual, expected, "seed {seed}, source {root_id}: closure mismatch");
            assert_visible_state_matches(
                &left,
                &plan,
                &expected,
                &format!("seed {seed}, source {root_id}"),
            );

            let digest_before_rebuild = left.state_sha256().unwrap();
            left.rebuild_reverse_index().unwrap();
            assert_eq!(
                left.state_sha256().unwrap(),
                digest_before_rebuild,
                "seed {seed}, source {root_id}: reverse-index rebuild changed logical state"
            );

            let mut right = original_b.clone();
            let right_actual = right
                .invalidate_source_event(&root_id, "generated metamorphic retraction")
                .unwrap();
            assert_eq!(right_actual, expected);
            assert_eq!(
                left.state_sha256().unwrap(),
                right.state_sha256().unwrap(),
                "seed {seed}, source {root_id}: mutation depends on insertion order"
            );
        }

        // Supersede selected rule-derived facts, not just source leaves. This
        // covers descendant closure rooted at an interior node of the DAG.
        for index in (0..FACT_COUNT).step_by(7) {
            let root_id = fact_id(index);
            let root = DependencyRef::DerivedFact(root_id.clone());
            let expected = forward_scan_closure(&plan, root);

            let mut left = original_a.clone();
            let actual = left
                .invalidate_derived_fact(&root_id, "generated rule-version supersession")
                .unwrap();
            assert_eq!(actual, expected, "seed {seed}, fact {root_id}: closure mismatch");
            assert_visible_state_matches(
                &left,
                &plan,
                &expected,
                &format!("seed {seed}, fact {root_id}"),
            );

            let digest_before_rebuild = left.state_sha256().unwrap();
            left.rebuild_reverse_index().unwrap();
            assert_eq!(
                left.state_sha256().unwrap(),
                digest_before_rebuild,
                "seed {seed}, fact {root_id}: reverse-index rebuild changed state"
            );

            let mut right = original_b.clone();
            let right_actual = right
                .invalidate_derived_fact(&root_id, "generated rule-version supersession")
                .unwrap();
            assert_eq!(right_actual, expected);
            assert_eq!(
                left.state_sha256().unwrap(),
                right.state_sha256().unwrap(),
                "seed {seed}, fact {root_id}: mutation depends on insertion order"
            );
        }
    }
}
