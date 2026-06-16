// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Moral Algebra HDC Invariants

Uses proptest to verify mathematical and semantic properties of the moral algebra
module's hyperdimensional encoding operations. These tests ensure that:

- Deterministic encodings are reproducible
- Different moral concepts produce distinguishable encodings
- Negation behaves as expected in HDC space
- Compositional operators preserve structural differences
- Proportionality judgments are consistent
- All encoded vectors contain finite values

Uses dim=512 for test speed (not full 4096) and 32 cases per property since
moral algebra construction is more expensive than raw HDC operations.
*/

use super::moral_algebra::{ConsentState, Magnitude, MoralAlgebra, MoralIntent};
use proptest::prelude::*;

const TEST_DIM: usize = 512;

fn name_strategy() -> impl Strategy<Value = &'static str> {
    prop::sample::select(vec!["alice", "bob", "charlie", "diana", "eve"])
}

fn action_strategy() -> impl Strategy<Value = &'static str> {
    prop::sample::select(vec!["help", "harm", "steal", "give", "discuss"])
}

fn intent_strategy() -> impl Strategy<Value = MoralIntent> {
    prop::sample::select(vec![
        MoralIntent::Good,
        MoralIntent::Bad,
        MoralIntent::Neutral,
        MoralIntent::Unknown,
    ])
}

fn magnitude_strategy() -> impl Strategy<Value = Magnitude> {
    prop::sample::select(vec![
        Magnitude::Tiny,
        Magnitude::Small,
        Magnitude::Medium,
        Magnitude::Large,
        Magnitude::Huge,
    ])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    // =========================================================================
    // Property 1: encode_agent is deterministic
    // Same name always produces the same encoding
    // =========================================================================
    #[test]
    fn prop_encode_agent_deterministic(name in name_strategy()) {
        let algebra = MoralAlgebra::new(TEST_DIM);
        let enc1 = algebra.encode_agent(name);
        let enc2 = algebra.encode_agent(name);

        let sim = enc1.similarity(&enc2);
        prop_assert!(
            (sim - 1.0).abs() < 1e-5,
            "encode_agent should be deterministic: sim({name}, {name}) = {sim}"
        );
    }

    // =========================================================================
    // Property 2: Different agents are distinguishable
    // encode_agent("alice") and encode_agent("bob") should have low similarity
    // =========================================================================
    #[test]
    fn prop_different_agents_distinguishable(
        name_a in name_strategy(),
        name_b in name_strategy(),
    ) {
        prop_assume!(name_a != name_b);
        let algebra = MoralAlgebra::new(TEST_DIM);
        let enc_a = algebra.encode_agent(name_a);
        let enc_b = algebra.encode_agent(name_b);

        let sim = enc_a.similarity(&enc_b).abs();
        prop_assert!(
            sim < 0.3,
            "Different agents should be distinguishable: sim({name_a}, {name_b}) = {sim}"
        );
    }

    // =========================================================================
    // Property 3: Different actions are distinguishable
    // encode_action("help") and encode_action("harm") should have low similarity
    // =========================================================================
    #[test]
    fn prop_different_actions_distinguishable(
        action_a in action_strategy(),
        action_b in action_strategy(),
    ) {
        prop_assume!(action_a != action_b);
        let algebra = MoralAlgebra::new(TEST_DIM);
        let enc_a = algebra.encode_action(action_a);
        let enc_b = algebra.encode_action(action_b);

        let sim = enc_a.similarity(&enc_b).abs();
        prop_assert!(
            sim < 0.3,
            "Different actions should be distinguishable: sim({action_a}, {action_b}) = {sim}"
        );
    }

    // =========================================================================
    // Property 4: All intents are mutually distinguishable
    // Pairwise similarity of all 4 intents should be < 0.3
    // =========================================================================
    #[test]
    fn prop_all_intents_distinguishable(_dummy in 0u8..1u8) {
        let algebra = MoralAlgebra::new(TEST_DIM);
        let intents = [
            MoralIntent::Good,
            MoralIntent::Bad,
            MoralIntent::Neutral,
            MoralIntent::Unknown,
        ];
        let encoded: Vec<_> = intents.iter().map(|i| algebra.encode_intent(*i)).collect();

        for i in 0..encoded.len() {
            for j in (i + 1)..encoded.len() {
                let sim = encoded[i].similarity(&encoded[j]).abs();
                prop_assert!(
                    sim < 0.3,
                    "Intents {:?} and {:?} too similar: {sim}",
                    intents[i], intents[j]
                );
            }
        }
    }

    // =========================================================================
    // Property 5: All consent states are mutually distinguishable
    // Pairwise similarity of all 4 consent states should be < 0.3
    // =========================================================================
    #[test]
    fn prop_all_consent_states_distinguishable(_dummy in 0u8..1u8) {
        let algebra = MoralAlgebra::new(TEST_DIM);
        let states = [
            ConsentState::Given,
            ConsentState::Denied,
            ConsentState::Absent,
            ConsentState::Implied,
        ];
        let encoded: Vec<_> = states.iter().map(|s| algebra.encode_consent(*s)).collect();

        for i in 0..encoded.len() {
            for j in (i + 1)..encoded.len() {
                let sim = encoded[i].similarity(&encoded[j]).abs();
                prop_assert!(
                    sim < 0.3,
                    "Consent states {:?} and {:?} too similar: {sim}",
                    states[i], states[j]
                );
            }
        }
    }

    // =========================================================================
    // Property 6: All magnitudes are mutually distinguishable
    // Pairwise similarity of all 5 magnitudes should be < 0.3
    // =========================================================================
    #[test]
    fn prop_all_magnitudes_distinguishable(_dummy in 0u8..1u8) {
        let algebra = MoralAlgebra::new(TEST_DIM);
        let mags = [
            Magnitude::Tiny,
            Magnitude::Small,
            Magnitude::Medium,
            Magnitude::Large,
            Magnitude::Huge,
        ];
        let encoded: Vec<_> = mags.iter().map(|m| algebra.encode_magnitude(*m)).collect();

        for i in 0..encoded.len() {
            for j in (i + 1)..encoded.len() {
                let sim = encoded[i].similarity(&encoded[j]).abs();
                prop_assert!(
                    sim < 0.3,
                    "Magnitudes {:?} and {:?} too similar: {sim}",
                    mags[i], mags[j]
                );
            }
        }
    }

    // =========================================================================
    // Property 7: Negation produces a different vector
    // sim(x, negate(x)) < 0.5
    // =========================================================================
    #[test]
    fn prop_negation_produces_different_vector(name in name_strategy()) {
        let algebra = MoralAlgebra::new(TEST_DIM);
        let original = algebra.encode_agent(name);
        let negated = algebra.negate(&original);

        let sim = original.similarity(&negated);
        prop_assert!(
            sim < 0.5,
            "Negation should produce a different vector: sim = {sim}"
        );
    }

    // =========================================================================
    // Property 8: Double negation is more similar to original than single
    // sim(x, negate(negate(x))) > sim(x, negate(x))
    // =========================================================================
    #[test]
    fn prop_double_negation_closer_to_original(name in name_strategy()) {
        let algebra = MoralAlgebra::new(TEST_DIM);
        let original = algebra.encode_agent(name);
        let single_neg = algebra.negate(&original);
        let double_neg = algebra.negate(&single_neg);

        let sim_single = original.similarity(&single_neg);
        let sim_double = original.similarity(&double_neg);
        prop_assert!(
            sim_double > sim_single,
            "Double negation should be more similar to original: \
             sim(x, neg(neg(x))) = {sim_double}, sim(x, neg(x)) = {sim_single}"
        );
    }

    // =========================================================================
    // Property 9: Causes produces a vector distinct from its inputs
    // causes(a,b) should not be identical to a or b (binding with the
    // CAUSES operator transforms the representation)
    // =========================================================================
    #[test]
    fn prop_causes_distinct_from_inputs(
        name_a in name_strategy(),
        name_b in name_strategy(),
    ) {
        prop_assume!(name_a != name_b);
        let algebra = MoralAlgebra::new(TEST_DIM);
        let hv_a = algebra.encode_action(name_a);
        let hv_b = algebra.encode_action(name_b);

        let caused = algebra.causes(&hv_a, &hv_b);

        let sim_to_a = caused.similarity(&hv_a).abs();
        let sim_to_b = caused.similarity(&hv_b).abs();
        prop_assert!(
            sim_to_a < 0.3,
            "causes(a,b) should differ from a: sim = {sim_to_a}"
        );
        prop_assert!(
            sim_to_b < 0.3,
            "causes(a,b) should differ from b: sim = {sim_to_b}"
        );
    }

    // =========================================================================
    // Property 10: Changing only the agent changes the action structure
    // encode_action_structure with different agents produces different encodings
    // =========================================================================
    #[test]
    fn prop_action_structure_differs_by_agent(
        agent_a in name_strategy(),
        agent_b in name_strategy(),
        action in action_strategy(),
        patient in name_strategy(),
        intent in intent_strategy(),
    ) {
        prop_assume!(agent_a != agent_b);
        let algebra = MoralAlgebra::new(TEST_DIM);
        let struct_a = algebra.encode_action_structure(agent_a, action, patient, intent);
        let struct_b = algebra.encode_action_structure(agent_b, action, patient, intent);

        let sim = struct_a.similarity(&struct_b);
        prop_assert!(
            sim < 0.99,
            "Different agents should yield different structures: \
             agents={agent_a}/{agent_b}, sim = {sim}"
        );
    }

    // =========================================================================
    // Property 11: Changing only the intent changes the action structure
    // encode_action_structure with different intents produces different encodings
    // =========================================================================
    #[test]
    fn prop_action_structure_differs_by_intent(
        agent in name_strategy(),
        action in action_strategy(),
        patient in name_strategy(),
        intent_a in intent_strategy(),
        intent_b in intent_strategy(),
    ) {
        prop_assume!(intent_a != intent_b);
        let algebra = MoralAlgebra::new(TEST_DIM);
        let struct_a = algebra.encode_action_structure(agent, action, patient, intent_a);
        let struct_b = algebra.encode_action_structure(agent, action, patient, intent_b);

        let sim = struct_a.similarity(&struct_b);
        prop_assert!(
            sim < 0.99,
            "Different intents should yield different structures: \
             intents={intent_a:?}/{intent_b:?}, sim = {sim}"
        );
    }

    // =========================================================================
    // Property 12: Equal magnitudes are proportional
    // encode_proportionality with same effort/reward magnitude -> is_proportional
    // =========================================================================
    #[test]
    fn prop_equal_magnitudes_proportional(mag in magnitude_strategy()) {
        let algebra = MoralAlgebra::new(TEST_DIM);
        let judgment = algebra.encode_proportionality("work", mag, "reward", mag);

        prop_assert!(
            judgment.is_proportional,
            "Same magnitude ({mag:?}) should be proportional"
        );
    }

    // =========================================================================
    // Property 13: Extreme magnitude difference is not proportional
    // Tiny effort + Huge reward -> is_proportional == false
    // =========================================================================
    #[test]
    fn prop_extreme_difference_not_proportional(_dummy in 0u8..1u8) {
        let algebra = MoralAlgebra::new(TEST_DIM);

        // Tiny effort, Huge reward
        let tiny_huge = algebra.encode_proportionality(
            "minimal work", Magnitude::Tiny,
            "massive reward", Magnitude::Huge,
        );
        prop_assert!(
            !tiny_huge.is_proportional,
            "Tiny effort + Huge reward should NOT be proportional"
        );

        // Huge effort, Tiny reward
        let huge_tiny = algebra.encode_proportionality(
            "massive work", Magnitude::Huge,
            "minimal reward", Magnitude::Tiny,
        );
        prop_assert!(
            !huge_tiny.is_proportional,
            "Huge effort + Tiny reward should NOT be proportional"
        );
    }

    // =========================================================================
    // Property 14: Primitives are approximately orthogonal
    // verify_orthogonality() < 0.15
    // =========================================================================
    #[test]
    fn prop_primitives_approximately_orthogonal(_dummy in 0u8..1u8) {
        let algebra = MoralAlgebra::new(TEST_DIM);
        let max_sim = algebra.primitives.verify_orthogonality();

        prop_assert!(
            max_sim < 0.15,
            "Primitives should be approximately orthogonal: max pairwise |sim| = {max_sim}"
        );
    }

    // =========================================================================
    // Property 15: All encoded vectors have finite values
    // Composition produces no NaN or Inf
    // =========================================================================
    #[test]
    fn prop_composition_produces_finite_values(
        agent in name_strategy(),
        action in action_strategy(),
        patient in name_strategy(),
        intent in intent_strategy(),
        magnitude in magnitude_strategy(),
    ) {
        let algebra = MoralAlgebra::new(TEST_DIM);

        // Encode various primitives
        let agent_hv = algebra.encode_agent(agent);
        let action_hv = algebra.encode_action(action);
        let patient_hv = algebra.encode_patient(patient);
        let intent_hv = algebra.encode_intent(intent);
        let mag_hv = algebra.encode_magnitude(magnitude);

        // Compose
        let structure = algebra.encode_action_structure(agent, action, patient, intent);
        let caused = algebra.causes(&action_hv, &patient_hv);
        let negated = algebra.negate(&action_hv);

        // Check finiteness via similarity (if any component is NaN/Inf,
        // similarity will return NaN)
        let sims = [
            agent_hv.similarity(&agent_hv),
            action_hv.similarity(&action_hv),
            patient_hv.similarity(&patient_hv),
            intent_hv.similarity(&intent_hv),
            mag_hv.similarity(&mag_hv),
            structure.similarity(&structure),
            caused.similarity(&caused),
            negated.similarity(&negated),
        ];

        for (i, sim) in sims.iter().enumerate() {
            prop_assert!(
                sim.is_finite(),
                "Encoded vector {i} has non-finite self-similarity: {sim}"
            );
        }
    }
}
