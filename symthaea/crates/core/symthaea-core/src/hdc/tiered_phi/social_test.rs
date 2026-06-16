// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[cfg(test)]
mod tests {
    use crate::hdc::binary_hv::BinaryHV;
    use crate::hdc::tiered_phi::social::SocialPhiCalculator;

    #[test]
    fn test_social_phi_emergent_crystallization() {
        let mut calculator = SocialPhiCalculator::new();

        // Agent 1 components (random, low internal integration)
        let agent1 = vec![
            BinaryHV::random(1),
            BinaryHV::random(2),
            BinaryHV::random(3),
            BinaryHV::random(4),
        ];

        // Agent 2 components (same seed, identical thoughts → high shared integration?)
        // Actually, identical components across agents would create a global 'lock'
        let agent2 = vec![
            BinaryHV::random(1),
            BinaryHV::random(2),
            BinaryHV::random(3),
            BinaryHV::random(4),
        ];

        let res = calculator.compute_social_phi(&[agent1, agent2]);

        println!("Collective Phi: {:.4}", res.collective_phi);
        println!("Individual Avg Phi: {:.4}", res.individual_avg_phi);
        println!("Integration Ratio: {:.4}", res.integration_ratio);

        // In this case, identical agents should have high integration ratio
        // because the collective information is perfectly shared.
        assert!(res.integration_ratio >= 1.0);
    }
}
