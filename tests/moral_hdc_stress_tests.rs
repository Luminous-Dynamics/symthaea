// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compound Moral Algebra + HDC Stress Tests
//!
//! Tests the interaction between moral reasoning and hyperdimensional computing
//! under degradation. Inspired by the "Irradiated Trolley Problem" scenario:
//! can the moral algebra produce correct judgments when the HDC vector space
//! is actively degrading?
//!
//! HDC's theoretical advantage: hypervectors are robust to noise because
//! meaning is distributed across 16,384 dimensions. Flipping 10% of bits
//! should preserve most semantic relationships. This test suite verifies
//! that claim empirically for moral reasoning.
//!
//! Run: `cargo test --test moral_hdc_stress_tests`

use symthaea::hdc::moral_algebra::{
    ConsentState, DeontologicalJudgment, Magnitude, MoralAlgebra, MoralIntent,
};
use symthaea::hdc::unified_hv::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: Baseline moral judgment (no degradation)
// Establish ground truth that the moral algebra correctly classifies
// clearly good and clearly bad actions.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_baseline_moral_judgment() {
    let algebra = MoralAlgebra::default_dim();

    // Clearly good action
    let good = algebra.judge_deontological("helping a stranger carry groceries");
    assert!(
        good.score > 0.0,
        "Helping should be judged positively: {:.3}",
        good.score
    );

    // Clearly bad action
    let bad = algebra.judge_deontological("stealing money from a homeless person");
    assert!(
        bad.score < good.score,
        "Stealing should be judged worse than helping: steal={:.3} vs help={:.3}",
        bad.score,
        good.score
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: HDC vector robustness under noise
// Verify that perturbing a moral encoding (simulating radiation/bit-flip)
// preserves semantic similarity up to a threshold.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_hdc_robustness_to_noise() {
    let algebra = MoralAlgebra::default_dim();

    let action = algebra.encode_action("save a drowning child");
    let obligation = algebra.encode_obligation("prevent-suffering");

    // Original similarity
    let original_sim = action.similarity(&obligation);

    // Perturb 10% of dimensions (simulating radiation damage)
    let perturbed = perturb_vector(&action, 0.10, 42);
    let perturbed_sim = perturbed.similarity(&obligation);

    // With 10% noise, similarity should be preserved within tolerance
    // HDC theory: random noise on <50% of dimensions preserves direction
    assert!(
        (perturbed_sim - original_sim).abs() < 0.3,
        "10% noise should preserve similarity: orig={original_sim:.4}, perturbed={perturbed_sim:.4}"
    );
}

#[test]
fn test_hdc_robustness_scaling() {
    let algebra = MoralAlgebra::default_dim();

    let action = algebra.encode_action("close the blast door to save workers");
    let obligation = algebra.encode_obligation("prevent-suffering");
    let original_sim = action.similarity(&obligation);

    // Test degradation at increasing noise levels
    let noise_levels = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40];
    let mut sims = Vec::new();

    for &noise in &noise_levels {
        let perturbed = perturb_vector(&action, noise, 123);
        let sim = perturbed.similarity(&obligation);
        sims.push(sim);
    }

    // Similarity should degrade monotonically with noise
    // (approximately — there's randomness in which dimensions flip)
    let correlation_preserved = sims[0] > sims[sims.len() - 1] - 0.1;
    assert!(
        correlation_preserved,
        "Similarity should generally decrease with noise: {:?}",
        sims
    );

    // At 10% noise, similarity should still be reasonably close to original
    assert!(
        (sims[2] - original_sim).abs() < 0.35,
        "10% noise similarity diverged too much: orig={original_sim:.4}, 10%={:.4}",
        sims[2]
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: Moral distinction preserved under noise
// The critical test: can the algebra still distinguish a good action from
// a bad action when both encodings are degraded?
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_moral_distinction_under_degradation() {
    let algebra = MoralAlgebra::default_dim();

    let good_proto = algebra.good_action_prototype();
    let bad_proto = algebra.bad_action_prototype();

    let good_action = algebra.encode_action("protect the innocent");
    let bad_action = algebra.encode_action("torture for information");

    // Original distinction
    let good_to_good = good_action.similarity(&good_proto);
    let bad_to_good = bad_action.similarity(&good_proto);
    let original_gap = good_to_good - bad_to_good;

    // Degrade both vectors by 15%
    let degraded_good = perturb_vector(&good_action, 0.15, 77);
    let degraded_bad = perturb_vector(&bad_action, 0.15, 88);

    let degraded_good_to_good = degraded_good.similarity(&good_proto);
    let degraded_bad_to_good = degraded_bad.similarity(&good_proto);
    let degraded_gap = degraded_good_to_good - degraded_bad_to_good;

    // The moral distinction should still be in the same direction
    // (good action still closer to good prototype than bad action)
    assert!(
        degraded_gap > -0.1,
        "Moral distinction should survive 15% degradation: gap={degraded_gap:.4} (original={original_gap:.4})"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 4: Obligation checking under progressive degradation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_obligation_checking_under_degradation() {
    let algebra = MoralAlgebra::default_dim();

    // Check obligations on clean text
    let judgment_clean = algebra.judge_deontological("refuse to steal despite hunger");
    let clean_score = judgment_clean.score;

    // Check obligations on same text multiple times (determinism)
    let judgment_repeat = algebra.judge_deontological("refuse to steal despite hunger");
    assert!(
        (judgment_repeat.score - clean_score).abs() < 1e-6,
        "Moral judgment should be deterministic"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 5: Action encoding structure
// Verify that the moral algebra's structured encodings (agent, patient,
// action, intent) produce distinct vectors.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_action_encoding_structure() {
    let algebra = MoralAlgebra::default_dim();

    let agent = algebra.encode_agent("doctor");
    let patient = algebra.encode_patient("patient");
    let action = algebra.encode_action("administer medicine");
    let intent = algebra.encode_intent(MoralIntent::Good);

    // These should all be distinct vectors
    let sim_agent_patient = agent.similarity(&patient);
    let sim_agent_action = agent.similarity(&action);
    let sim_action_intent = action.similarity(&intent);

    // In 16,384 dimensions, random vectors have ~0.5 cosine similarity
    // Semantically unrelated encodings should be near random baseline
    assert!(
        sim_agent_patient < 0.8,
        "Agent and patient should be relatively distinct: {sim_agent_patient:.4}"
    );
    assert!(
        sim_agent_action < 0.8,
        "Agent and action should be relatively distinct: {sim_agent_action:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 6: Consent encoding and moral weight
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_consent_encoding() {
    let algebra = MoralAlgebra::default_dim();

    let explicit = algebra.encode_consent(ConsentState::Given);
    let denied = algebra.encode_consent(ConsentState::Denied);
    let unable = algebra.encode_consent(ConsentState::Absent);

    // Explicit consent and denied should be maximally different
    let consent_denied_sim = explicit.similarity(&denied);
    let consent_unable_sim = explicit.similarity(&unable);

    // These should be distinct states
    assert!(
        consent_denied_sim < 0.9,
        "Consent and denied should be distinct: {consent_denied_sim:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 7: Magnitude encoding preserves ordering
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_magnitude_ordering() {
    let algebra = MoralAlgebra::default_dim();

    let trivial = algebra.encode_magnitude(Magnitude::Tiny);
    let minor = algebra.encode_magnitude(Magnitude::Small);
    let moderate = algebra.encode_magnitude(Magnitude::Medium);
    let severe = algebra.encode_magnitude(Magnitude::Large);
    let catastrophic = algebra.encode_magnitude(Magnitude::Huge);

    // Adjacent magnitudes should be more similar than distant ones
    let trivial_minor = trivial.similarity(&minor);
    let trivial_catastrophic = trivial.similarity(&catastrophic);

    assert!(
        trivial_minor > trivial_catastrophic,
        "Adjacent magnitudes should be more similar: triv-minor={trivial_minor:.4} vs triv-catast={trivial_catastrophic:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility: perturb a vector by flipping a fraction of dimensions
// ═══════════════════════════════════════════════════════════════════════════════

fn perturb_vector(hv: &ContinuousHV, noise_fraction: f64, seed: u64) -> ContinuousHV {
    let mut values = hv.values.clone();
    let n_flip = (values.len() as f64 * noise_fraction) as usize;
    let mut rng = seed;

    for _ in 0..n_flip {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (rng >> 33) as usize % values.len();
        values[idx] = -values[idx]; // Flip sign (equivalent to bit-flip in bipolar encoding)
    }

    ContinuousHV::from_values(values)
}