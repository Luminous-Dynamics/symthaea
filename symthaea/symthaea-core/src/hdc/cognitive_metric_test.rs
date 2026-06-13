// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[cfg(test)]
mod tests {
    use crate::hdc::cognitive_metric::CognitiveMetric;
    use crate::hdc::unified_hv::ContinuousHV;

    #[test]
    fn test_gravitational_warping() {
        let mut metric = CognitiveMetric::new();

        // 1. Create a high-mass 'Established Macro' vector
        let macro_v = ContinuousHV::random(1024, 42);
        metric.add_mass(macro_v.clone(), 10.0); // High mass

        // 2. Create two concept vectors that are slightly similar to the macro
        // but not to each other in Euclidean space.
        let mut concept_a = ContinuousHV::random(1024, 100);
        let mut concept_b = ContinuousHV::random(1024, 200);

        // Europan similarity
        let sim_euclidean = concept_a.similarity(&concept_b);

        // Warp them toward the macro
        concept_a = ContinuousHV::bundle(&[&concept_a, &macro_v]).normalize();
        concept_b = ContinuousHV::bundle(&[&concept_b, &macro_v]).normalize();

        let sim_warped = metric.warped_similarity(&concept_a, &concept_b);

        println!("Euclidean Sim: {:.4}", sim_euclidean);
        println!("Warped Sim: {:.4}", sim_warped);

        // The warped similarity should be significantly higher because they
        // both reside in the gravitational well of the established macro.
        assert!(sim_warped > sim_euclidean as f64);
    }
}
