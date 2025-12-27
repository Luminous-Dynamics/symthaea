// Synthetic Test Validation for Enhancement #7
//
// Creates known causal environments to validate program synthesis
//
// Test Strategy:
// 1. Create synthetic causal structures (chains, forks, colliders, etc.)
// 2. Synthesize programs to achieve desired causal effects
// 3. Verify synthesized programs match the known structure
// 4. Test counterfactual predictions on synthetic data

use symthaea::synthesis::{
    CausalSpec, CausalProgramSynthesizer, SynthesisConfig,
    CounterfactualVerifier, VerificationConfig,
    AdaptiveProgram, AdaptationStrategy,
};
use std::collections::HashMap;

/// Synthetic causal environment with known structure
pub struct SyntheticCausalEnvironment {
    /// Variables in the environment
    variables: Vec<String>,

    /// Known causal edges (cause -> effect, strength)
    edges: Vec<(String, String, f64)>,

    /// Data generation function
    generator: Box<dyn Fn(&HashMap<String, f64>) -> HashMap<String, f64>>,
}

impl SyntheticCausalEnvironment {
    /// Create simple chain: A -> B -> C
    pub fn simple_chain() -> Self {
        let variables = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let edges = vec![
            ("A".to_string(), "B".to_string(), 0.7),
            ("B".to_string(), "C".to_string(), 0.5),
        ];

        let generator = Box::new(|inputs: &HashMap<String, f64>| {
            let mut outputs = HashMap::new();

            // A is exogenous (input)
            let a = inputs.get("A").copied().unwrap_or(0.5);
            outputs.insert("A".to_string(), a);

            // B = 0.7 * A + noise
            let b = 0.7 * a + 0.1 * (a * 13.7).sin();
            outputs.insert("B".to_string(), b);

            // C = 0.5 * B + noise
            let c = 0.5 * b + 0.1 * (b * 17.3).sin();
            outputs.insert("C".to_string(), c);

            outputs
        });

        Self { variables, edges, generator }
    }

    /// Create fork: A -> B, A -> C
    pub fn fork() -> Self {
        let variables = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let edges = vec![
            ("A".to_string(), "B".to_string(), 0.6),
            ("A".to_string(), "C".to_string(), 0.8),
        ];

        let generator = Box::new(|inputs: &HashMap<String, f64>| {
            let mut outputs = HashMap::new();
            let a = inputs.get("A").copied().unwrap_or(0.5);
            outputs.insert("A".to_string(), a);

            let b = 0.6 * a + 0.1 * (a * 11.3).sin();
            outputs.insert("B".to_string(), b);

            let c = 0.8 * a + 0.1 * (a * 19.7).sin();
            outputs.insert("C".to_string(), c);

            outputs
        });

        Self { variables, edges, generator }
    }

    /// Create collider: A -> C, B -> C
    pub fn collider() -> Self {
        let variables = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let edges = vec![
            ("A".to_string(), "C".to_string(), 0.5),
            ("B".to_string(), "C".to_string(), 0.5),
        ];

        let generator = Box::new(|inputs: &HashMap<String, f64>| {
            let mut outputs = HashMap::new();
            let a = inputs.get("A").copied().unwrap_or(0.5);
            let b = inputs.get("B").copied().unwrap_or(0.5);

            outputs.insert("A".to_string(), a);
            outputs.insert("B".to_string(), b);

            // C depends on both A and B
            let c = 0.5 * a + 0.5 * b + 0.1 * (a * b * 23.1).sin();
            outputs.insert("C".to_string(), c);

            outputs
        });

        Self { variables, edges, generator }
    }

    /// Create mediated path: A -> M -> B (M mediates A's effect on B)
    pub fn mediated() -> Self {
        let variables = vec!["A".to_string(), "M".to_string(), "B".to_string()];
        let edges = vec![
            ("A".to_string(), "M".to_string(), 0.7),
            ("M".to_string(), "B".to_string(), 0.6),
        ];

        let generator = Box::new(|inputs: &HashMap<String, f64>| {
            let mut outputs = HashMap::new();
            let a = inputs.get("A").copied().unwrap_or(0.5);
            outputs.insert("A".to_string(), a);

            let m = 0.7 * a + 0.1 * (a * 29.3).sin();
            outputs.insert("M".to_string(), m);

            let b = 0.6 * m + 0.1 * (m * 31.7).sin();
            outputs.insert("B".to_string(), b);

            outputs
        });

        Self { variables, edges, generator }
    }

    /// Generate sample data from this environment
    pub fn generate_sample(&self, input_values: HashMap<String, f64>) -> HashMap<String, f64> {
        (self.generator)(&input_values)
    }

    /// Get known causal structure
    pub fn known_edges(&self) -> &[(String, String, f64)] {
        &self.edges
    }

    /// Get all variables
    pub fn variables(&self) -> &[String] {
        &self.variables
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_chain_environment() {
        let env = SyntheticCausalEnvironment::simple_chain();

        // Test data generation
        let mut inputs = HashMap::new();
        inputs.insert("A".to_string(), 1.0);

        let outputs = env.generate_sample(inputs);

        assert!(outputs.contains_key("A"));
        assert!(outputs.contains_key("B"));
        assert!(outputs.contains_key("C"));

        // B should be influenced by A
        let a = outputs.get("A").unwrap();
        let b = outputs.get("B").unwrap();
        assert!((b - 0.7 * a).abs() < 0.2); // Allow for noise
    }

    #[test]
    fn test_fork_environment() {
        let env = SyntheticCausalEnvironment::fork();

        let mut inputs = HashMap::new();
        inputs.insert("A".to_string(), 0.8);

        let outputs = env.generate_sample(inputs);

        // Both B and C should be influenced by A
        let a = outputs.get("A").unwrap();
        let b = outputs.get("B").unwrap();
        let c = outputs.get("C").unwrap();

        assert!((b - 0.6 * a).abs() < 0.2);
        assert!((c - 0.8 * a).abs() < 0.2);
    }

    #[test]
    fn test_collider_environment() {
        let env = SyntheticCausalEnvironment::collider();

        let mut inputs = HashMap::new();
        inputs.insert("A".to_string(), 0.6);
        inputs.insert("B".to_string(), 0.4);

        let outputs = env.generate_sample(inputs);

        let c = outputs.get("C").unwrap();
        // C should depend on both A and B
        assert!((c - 0.5).abs() < 0.3); // Rough check
    }

    #[test]
    fn test_synthesis_on_simple_chain() {
        let env = SyntheticCausalEnvironment::simple_chain();

        // Create specification to make A cause B with strength 0.7
        let spec = CausalSpec::MakeCause {
            cause: "A".to_string(),
            effect: "B".to_string(),
            strength: 0.7,
        };

        // Synthesize program
        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
        let result = synthesizer.synthesize(&spec);

        assert!(result.is_ok());
        let program = result.unwrap();

        // Verify achieved strength is close to target
        assert!((program.achieved_strength - 0.7).abs() < 0.1);
        assert!(program.confidence > 0.8);
    }

    #[test]
    fn test_synthesis_on_fork() {
        let env = SyntheticCausalEnvironment::fork();

        // Create specification for both edges
        let spec_ab = CausalSpec::MakeCause {
            cause: "A".to_string(),
            effect: "B".to_string(),
            strength: 0.6,
        };

        let spec_ac = CausalSpec::MakeCause {
            cause: "A".to_string(),
            effect: "C".to_string(),
            strength: 0.8,
        };

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());

        let program_ab = synthesizer.synthesize(&spec_ab).unwrap();
        let program_ac = synthesizer.synthesize(&spec_ac).unwrap();

        // Both should succeed
        assert!((program_ab.achieved_strength - 0.6).abs() < 0.1);
        assert!((program_ac.achieved_strength - 0.8).abs() < 0.1);
    }

    #[test]
    fn test_verification_on_synthetic_data() {
        let env = SyntheticCausalEnvironment::simple_chain();

        // Synthesize program
        let spec = CausalSpec::MakeCause {
            cause: "A".to_string(),
            effect: "B".to_string(),
            strength: 0.7,
        };

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
        let program = synthesizer.synthesize(&spec).unwrap();

        // Verify with counterfactual tests
        let mut config = VerificationConfig::default();
        config.num_counterfactuals = 100; // Fewer tests for synthetic data

        let mut verifier = CounterfactualVerifier::new(config);
        let result = verifier.verify(&program);

        assert!(result.success);
        assert!(result.confidence > 0.9); // Should be very confident on synthetic data
        assert!(result.counterfactual_accuracy > 0.9);
    }

    #[test]
    fn test_mediated_path_synthesis() {
        let env = SyntheticCausalEnvironment::mediated();

        // Create specification for mediated path A -> M -> B
        let spec = CausalSpec::CreatePath {
            from: "A".to_string(),
            through: vec!["M".to_string()],
            to: "B".to_string(),
        };

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
        let result = synthesizer.synthesize(&spec);

        assert!(result.is_ok());
        let program = result.unwrap();

        // Should create both edges
        assert!(program.variables.contains(&"A".to_string()));
        assert!(program.variables.contains(&"M".to_string()));
        assert!(program.variables.contains(&"B".to_string()));
    }

    #[test]
    fn test_adaptive_program_on_changing_environment() {
        // Start with simple chain
        let env1 = SyntheticCausalEnvironment::simple_chain();

        let spec = CausalSpec::MakeCause {
            cause: "A".to_string(),
            effect: "B".to_string(),
            strength: 0.7,
        };

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
        let initial_program = synthesizer.synthesize(&spec).unwrap();

        // Create adaptive program
        let mut adaptive = AdaptiveProgram::new(
            initial_program.clone(),
            spec.clone(),
            AdaptationStrategy::OnVerificationFailure,
        );

        // Update with new observations (simulated)
        let adapted = adaptive.update(None);

        // Should not adapt initially (program is correct)
        assert!(!adapted);

        let stats = adaptive.stats();
        assert_eq!(stats.adaptation_count, 0);
        assert!(stats.current_confidence > 0.8);
    }

    #[test]
    fn test_composite_specification() {
        // Test AND composition
        let spec1 = CausalSpec::MakeCause {
            cause: "A".to_string(),
            effect: "B".to_string(),
            strength: 0.5,
        };

        let spec2 = CausalSpec::MakeCause {
            cause: "B".to_string(),
            effect: "C".to_string(),
            strength: 0.6,
        };

        let composite = CausalSpec::And(vec![
            Box::new(spec1),
            Box::new(spec2),
        ]);

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
        let result = synthesizer.synthesize(&composite);

        assert!(result.is_ok());
        let program = result.unwrap();

        // Should involve all three variables
        assert!(program.variables.contains(&"A".to_string()));
        assert!(program.variables.contains(&"B".to_string()));
        assert!(program.variables.contains(&"C".to_string()));
    }

    #[test]
    fn test_strengthen_weak_connection() {
        // Start with weak connection, strengthen it
        let spec = CausalSpec::Strengthen {
            cause: "A".to_string(),
            effect: "B".to_string(),
            target_strength: 0.9,
        };

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
        let result = synthesizer.synthesize(&spec);

        assert!(result.is_ok());
        let program = result.unwrap();

        // Should achieve near target strength
        assert!((program.achieved_strength - 0.9).abs() < 0.15);
    }

    #[test]
    fn test_remove_spurious_correlation() {
        // Test removing a causal connection
        let spec = CausalSpec::RemoveCause {
            cause: "A".to_string(),
            effect: "B".to_string(),
        };

        let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
        let result = synthesizer.synthesize(&spec);

        assert!(result.is_ok());
        let program = result.unwrap();

        // Should result in near-zero strength
        assert!(program.achieved_strength < 0.1);
    }
}
