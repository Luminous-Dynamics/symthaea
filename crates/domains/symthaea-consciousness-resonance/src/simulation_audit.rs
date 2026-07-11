#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{ResonanceAnalyzer, ResonanceConfig};
    use symthaea_core::hdc::BinaryHV;
    use symthaea_core::hdc::consciousness_integration::{ConsciousnessPipeline, IntegrationConfig};

    #[test]
    pub fn test_long_term_stability_simulation() {
        let mut pipeline = ConsciousnessPipeline::default();
        let mut analyzer = ResonanceAnalyzer::new(ResonanceConfig::default());

        let mut phi_history = Vec::new();
        let mut resonance_quality_history = Vec::new();

        for cycle in 0..20 {
            let input = vec![BinaryHV::random(cycle as u64)];
            let state = pipeline.process(input, &[1.0]);

            let components: Vec<BinaryHV> = (0..7)
                .map(|i| BinaryHV::random(i as u64 + cycle as u64))
                .collect();
            let resonance = analyzer.analyze_with_phi([0.5; 7], &components);

            phi_history.push(state.phi);
            resonance_quality_history.push(resonance.q_factor);
        }

        let mean_phi: f64 = phi_history.iter().sum::<f64>() / 20.0;
        let mean_q: f64 = resonance_quality_history.iter().sum::<f64>() / 20.0;

        println!("Mean Φ: {:.4}, Mean Resonance Q: {:.4}", mean_phi, mean_q);

        assert!(
            mean_phi > 0.0,
            "System should maintain positive integrated information."
        );
        assert!(
            mean_q > 0.1,
            "System should maintain stable resonance quality."
        );
    }
}
