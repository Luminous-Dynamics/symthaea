// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Adversarial regressions for verified transition-history admission.
//!
//! These tests intentionally mutate otherwise canonical production chains. Every mutation must
//! reject the entire chain and leave the verified history with zero admitted evidence.

use super::reasoning_transition_history::VerifiedTransitionHistory;
use crate::consciousness::primitive_reasoning::{ReasoningChain, TaskType, TransformationType};
use symthaea_core::hdc::primitive_system::PrimitiveSystem;
use symthaea_core::hdc::BinaryHV;

fn canonical_chain(steps: usize) -> ReasoningChain {
    let primitive = PrimitiveSystem::global()
        .get("NSM_KNOW")
        .expect("NSM_KNOW fixture primitive must exist")
        .clone();
    let mut chain = ReasoningChain::new(BinaryHV::random(51_001));
    for _ in 0..steps {
        chain
            .execute_primitive(&primitive, TransformationType::Bundle)
            .expect("canonical production execution must succeed");
    }
    chain
}

fn assert_rejected_without_admission(chain: &ReasoningChain) {
    let mut history = VerifiedTransitionHistory::new();
    assert!(history.observe_chain(chain, TaskType::Generic).is_err());
    assert_eq!(history.admitted_chains(), 0);
    assert_eq!(history.admitted_executions(), 0);
    assert_eq!(history.rejected_chains(), 1);
}

#[test]
fn forged_local_contribution_is_rejected_transactionally() {
    let mut chain = canonical_chain(2);
    chain.executions[1].phi_contribution += 0.01;
    assert_rejected_without_admission(&chain);
}

#[test]
fn discontinuous_execution_input_is_rejected_transactionally() {
    let mut chain = canonical_chain(2);
    chain.executions[1].input = BinaryHV::random(51_002);
    assert_rejected_without_admission(&chain);
}

#[test]
fn forged_final_state_is_rejected_transactionally() {
    let mut chain = canonical_chain(2);
    chain.current_state = BinaryHV::random(51_003);
    assert_rejected_without_admission(&chain);
}

#[test]
fn forged_accumulated_total_is_rejected_transactionally() {
    let mut chain = canonical_chain(2);
    chain.total_phi += 0.01;
    assert_rejected_without_admission(&chain);
}

#[test]
fn empty_chain_is_not_admissible_evidence() {
    let chain = ReasoningChain::new(BinaryHV::random(51_004));
    assert_rejected_without_admission(&chain);
}
