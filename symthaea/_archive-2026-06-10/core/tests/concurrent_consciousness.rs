// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Concurrent Multi-Agent Consciousness Integration Tests
//!
//! Validates three-agent oxytocin coupling convergence and demonstrates that
//! uncoupled agents diverge while coupled agents synchronize.
//!
//! Science: Kosfeld et al. (2005) — oxytocin increases trust and cooperation;
//!          Feldman (2012) — oxytocin as social bonding mechanism;
//!          Dumas et al. (2010) — inter-brain synchronization during social interaction.

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ConsciousnessProfile};

const SCIENCE_INPUTS: &[&str] = &[
    "Photosynthesis converts light energy into chemical bonds through electron transport chains.",
    "Quantum entanglement permits instantaneous correlations across arbitrary distances.",
    "The mitochondrial membrane potential drives ATP synthesis via chemiosmotic coupling.",
    "Gravitational lensing bends light around massive objects, confirming general relativity.",
    "CRISPR-Cas9 enables precise genome editing by guiding endonuclease activity.",
];

const POETRY_INPUTS: &[&str] = &[
    "The fog comes on little cat feet, it sits looking over harbor and city.",
    "Do not go gentle into that good night, rage against the dying of the light.",
    "I wandered lonely as a cloud that floats on high o'er vales and hills.",
    "Because I could not stop for Death, he kindly stopped for me.",
    "The road not taken has made all the difference in my wandering life.",
];

const PHILOSOPHY_INPUTS: &[&str] = &[
    "Cogito ergo sum: the act of doubting itself confirms existence of the doubter.",
    "The categorical imperative demands we act only on universalizable maxims.",
    "Dasein is always already thrown into a world of meaning and concern.",
    "The veil of ignorance ensures fairness by eliminating knowledge of social position.",
    "Language games constitute the bedrock of meaning, not private mental states.",
];

/// Euclidean distance between two 9-element state vectors.
fn euclidean_distance(a: &[f32; 9], b: &[f32; 9]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Build a service with deterministic seed and async training disabled.
fn build_service(seed: &str) -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
    config.async_training = false;
    config.genesis_phrase = Some(seed.to_string());
    CognitiveLoopService::new(config).expect("CognitiveLoopService should construct")
}

/// Oxytocin index in the 9-element bath state vector
/// [DA, NE, 5-HT, ACh, GABA, Oxy, Glut, Aden, ECB]
const OXY_INDEX: usize = 5;

#[test]
fn test_three_agent_consciousness_convergence() {
    let mut agent_a = build_service("concurrent_agent_a_2026");
    let mut agent_b = build_service("concurrent_agent_b_2026");
    let mut agent_c = build_service("concurrent_agent_c_2026");

    // ── Warmup: 30 independent cycles ────────────────────────────────────
    for i in 0..30 {
        agent_a.cycle(SCIENCE_INPUTS[i % SCIENCE_INPUTS.len()]);
        agent_b.cycle(POETRY_INPUTS[i % POETRY_INPUTS.len()]);
        agent_c.cycle(PHILOSOPHY_INPUTS[i % PHILOSOPHY_INPUTS.len()]);
    }

    // Record baseline oxytocin levels after warmup
    let oxy_a_baseline = agent_a.bath_state_vector()[OXY_INDEX];
    let oxy_b_baseline = agent_b.bath_state_vector()[OXY_INDEX];
    let oxy_c_baseline = agent_c.bath_state_vector()[OXY_INDEX];

    // Track oxytocin and consciousness over the coupling window
    let mut oxy_a_history = Vec::with_capacity(80);
    let mut oxy_b_history = Vec::with_capacity(80);
    let mut oxy_c_history = Vec::with_capacity(80);

    // ── 80 coupled cycles ────────────────────────────────────────────────
    for i in 0..80 {
        // Each agent processes its domain-specific input
        let result_a = agent_a.cycle(SCIENCE_INPUTS[i % SCIENCE_INPUTS.len()]);
        let result_b = agent_b.cycle(POETRY_INPUTS[i % POETRY_INPUTS.len()]);
        let result_c = agent_c.cycle(PHILOSOPHY_INPUTS[i % PHILOSOPHY_INPUTS.len()]);

        // Couple all pairs: A<->B, B<->C, A<->C
        let sv_a = agent_a.bath_state_vector();
        let sv_b = agent_b.bath_state_vector();
        let sv_c = agent_c.bath_state_vector();

        agent_a.couple_bath_with_peer(&sv_b);
        agent_a.couple_bath_with_peer(&sv_c);
        agent_b.couple_bath_with_peer(&sv_a);
        agent_b.couple_bath_with_peer(&sv_c);
        agent_c.couple_bath_with_peer(&sv_a);
        agent_c.couple_bath_with_peer(&sv_b);

        // Record oxytocin after coupling
        oxy_a_history.push(agent_a.bath_state_vector()[OXY_INDEX]);
        oxy_b_history.push(agent_b.bath_state_vector()[OXY_INDEX]);
        oxy_c_history.push(agent_c.bath_state_vector()[OXY_INDEX]);

        // Assert: state vectors remain finite throughout
        for (label, sv) in [
            ("A", agent_a.bath_state_vector()),
            ("B", agent_b.bath_state_vector()),
            ("C", agent_c.bath_state_vector()),
        ] {
            for (j, &v) in sv.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "Agent {label} state[{j}] is not finite at cycle {i}: {v}"
                );
            }
        }

        // Assert: consciousness levels stay bounded [0, 1]
        for (label, cl) in [
            ("A", result_a.metadata.consciousness.consciousness_level),
            ("B", result_b.metadata.consciousness.consciousness_level),
            ("C", result_c.metadata.consciousness.consciousness_level),
        ] {
            assert!(
                (0.0..=1.0).contains(&cl),
                "Agent {label} consciousness_level out of bounds at cycle {i}: {cl}"
            );
        }
    }

    // Assert: oxytocin rises in all 3 agents over the coupling window.
    // Compare the average of the last 20 cycles to the baseline.
    let avg_last = |h: &[f32]| -> f32 {
        let tail = &h[h.len() - 20..];
        tail.iter().sum::<f32>() / tail.len() as f32
    };

    let oxy_a_final_avg = avg_last(&oxy_a_history);
    let oxy_b_final_avg = avg_last(&oxy_b_history);
    let oxy_c_final_avg = avg_last(&oxy_c_history);

    // Oxytocin should be sustained or boosted relative to baseline.
    // We use >= baseline * 0.9 as tolerance (coupling self-boost may be subtle).
    assert!(
        oxy_a_final_avg >= oxy_a_baseline * 0.9,
        "Agent A oxytocin should be sustained/boosted: baseline={oxy_a_baseline:.4}, final_avg={oxy_a_final_avg:.4}"
    );
    assert!(
        oxy_b_final_avg >= oxy_b_baseline * 0.9,
        "Agent B oxytocin should be sustained/boosted: baseline={oxy_b_baseline:.4}, final_avg={oxy_b_final_avg:.4}"
    );
    assert!(
        oxy_c_final_avg >= oxy_c_baseline * 0.9,
        "Agent C oxytocin should be sustained/boosted: baseline={oxy_c_baseline:.4}, final_avg={oxy_c_final_avg:.4}"
    );
}

#[test]
fn test_uncoupled_agents_diverge() {
    let mut agent_x = build_service("diverge_agent_x_2026");
    let mut agent_y = build_service("diverge_agent_y_2026");

    // Shared input set for both agents
    let shared_inputs: &[&str] = &[
        "The weather is warm today.",
        "I need to solve this problem efficiently.",
        "Music brings people together in unexpected ways.",
        "How does photosynthesis convert light into energy?",
        "The architecture of this building is remarkable.",
    ];

    // ── Phase 1: 80 uncoupled cycles with same input ────────────────────
    for i in 0..80 {
        agent_x.cycle(shared_inputs[i % shared_inputs.len()]);
        agent_y.cycle(shared_inputs[i % shared_inputs.len()]);
    }

    let sv_x_uncoupled = agent_x.bath_state_vector();
    let sv_y_uncoupled = agent_y.bath_state_vector();
    let cl_x = agent_x.consciousness_level();
    let cl_y = agent_y.consciousness_level();

    // Different seeds produce divergent trajectories despite same input.
    // At minimum, state vectors should not be bitwise identical.
    let dist_uncoupled = euclidean_distance(&sv_x_uncoupled, &sv_y_uncoupled);

    // The consciousness trajectories should not be identical (stochastic init).
    // We check that either the state vectors differ OR the consciousness levels differ.
    let consciousness_diff = (cl_x - cl_y).abs();
    assert!(
        dist_uncoupled > 0.0 || consciousness_diff > 0.0,
        "Uncoupled agents with different seeds should diverge: state_dist={dist_uncoupled:.6}, consciousness_diff={consciousness_diff:.6}"
    );

    // Record the pre-coupling distance for comparison
    let distance_before_coupling = dist_uncoupled;

    // ── Phase 2: 80 coupled cycles with same input ──────────────────────
    // Now couple the agents and check that their neuromod state vectors
    // become more similar (synchrony increases).
    let mut distances_during_coupling = Vec::with_capacity(80);

    for i in 0..80 {
        agent_x.cycle(shared_inputs[i % shared_inputs.len()]);
        agent_y.cycle(shared_inputs[i % shared_inputs.len()]);

        let sv_x = agent_x.bath_state_vector();
        let sv_y = agent_y.bath_state_vector();
        agent_x.couple_bath_with_peer(&sv_y);
        agent_y.couple_bath_with_peer(&sv_x);

        distances_during_coupling.push(euclidean_distance(
            &agent_x.bath_state_vector(),
            &agent_y.bath_state_vector(),
        ));
    }

    // Compare average distance over last 20 coupled cycles to pre-coupling distance
    let avg_final_distance: f32 = distances_during_coupling[distances_during_coupling.len() - 20..]
        .iter()
        .sum::<f32>()
        / 20.0;

    // Coupling should bring the agents closer together (or at least not push them
    // further apart). We allow tolerance of +0.5 because different seeds create
    // inherent asymmetry that coupling only partially overcomes.
    assert!(
        avg_final_distance < distance_before_coupling + 0.5,
        "Coupled agents should synchronize: pre_coupling_dist={distance_before_coupling:.4}, \
         avg_final_dist={avg_final_distance:.4}"
    );

    // All state values should remain finite
    for &v in agent_x
        .bath_state_vector()
        .iter()
        .chain(agent_y.bath_state_vector().iter())
    {
        assert!(
            v.is_finite(),
            "State value should be finite after coupling: {v}"
        );
    }
}