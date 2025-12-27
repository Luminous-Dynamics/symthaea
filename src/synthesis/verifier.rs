// Counterfactual Program Verifier
//
// Verifies synthesized programs using counterfactual testing
//
// Innovation: Uses Enhancement #4 Phase 2 (Counterfactual Reasoning)
// to verify that programs achieve desired causal effects

use super::synthesizer::{SynthesizedProgram, ProgramTemplate};
use super::causal_spec::{CausalSpec, CausalStrength};
use super::{SynthesisError, SynthesisResult};
use crate::observability::{CounterfactualEngine, CounterfactualQuery};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Number of counterfactual tests to run
    pub num_counterfactuals: usize,

    /// Minimum accuracy required to pass verification
    pub min_accuracy: f64,

    /// Whether to test edge cases
    pub test_edge_cases: bool,

    /// Maximum program complexity allowed
    pub max_complexity: usize,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            num_counterfactuals: 1000,
            min_accuracy: 0.95,
            test_edge_cases: true,
            max_complexity: 100,
        }
    }
}

/// Result of verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether verification succeeded
    pub success: bool,

    /// Confidence in program correctness (0.0 - 1.0)
    pub confidence: f64,

    /// Accuracy on counterfactual tests
    pub counterfactual_accuracy: f64,

    /// Number of tests run
    pub tests_run: usize,

    /// Edge cases where program failed (if any)
    pub edge_cases: Vec<String>,

    /// Detailed test results
    pub details: Option<VerificationDetails>,
}

/// Detailed verification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationDetails {
    /// Number of tests passed
    pub passed: usize,

    /// Number of tests failed
    pub failed: usize,

    /// Average error magnitude
    pub avg_error: f64,

    /// Maximum error seen
    pub max_error: f64,

    /// Whether program is minimal
    pub is_minimal: Option<bool>,
}

/// Counterfactual Program Verifier
///
/// Verifies synthesized programs by:
/// 1. Generating counterfactual test cases
/// 2. Running program on test cases
/// 3. Checking if causal effect matches specification
/// 4. Reporting confidence and edge cases
pub struct CounterfactualVerifier {
    /// Configuration
    config: VerificationConfig,

    /// Counterfactual engine (from Enhancement #4 Phase 2)
    counterfactual_engine: Option<CounterfactualEngine>,

    /// Minimality checker
    minimality_checker: MinimalityChecker,
}

impl CounterfactualVerifier {
    /// Create new verifier with configuration
    pub fn new(config: VerificationConfig) -> Self {
        Self {
            config,
            counterfactual_engine: None,
            minimality_checker: MinimalityChecker::new(),
        }
    }

    /// Create verifier with default configuration
    pub fn default() -> Self {
        Self::new(VerificationConfig::default())
    }

    /// Set counterfactual engine (Enhancement #4 Phase 2)
    pub fn with_counterfactual_engine(mut self, engine: CounterfactualEngine) -> Self {
        self.counterfactual_engine = Some(engine);
        self
    }

    /// Verify a synthesized program
    pub fn verify(&self, program: &SynthesizedProgram) -> VerificationResult {
        let mut passed = 0;
        let mut failed = 0;
        let mut errors = Vec::new();
        let mut edge_cases = Vec::new();

        // Check complexity
        if program.complexity > self.config.max_complexity {
            return VerificationResult {
                success: false,
                confidence: 0.0,
                counterfactual_accuracy: 0.0,
                tests_run: 0,
                edge_cases: vec![format!(
                    "Program complexity {} exceeds maximum {}",
                    program.complexity, self.config.max_complexity
                )],
                details: None,
            };
        }

        // Generate and run counterfactual tests
        for i in 0..self.config.num_counterfactuals {
            let test_case = self.generate_test_case(program, i);
            let result = self.run_test(program, &test_case);

            if result.passed {
                passed += 1;
            } else {
                failed += 1;
                errors.push(result.error);

                if result.is_edge_case {
                    edge_cases.push(result.description);
                }
            }
        }

        // Calculate metrics
        let accuracy = passed as f64 / self.config.num_counterfactuals as f64;
        let avg_error = if !errors.is_empty() {
            errors.iter().sum::<f64>() / errors.len() as f64
        } else {
            0.0
        };
        let max_error = errors.iter().cloned().fold(0.0, f64::max);

        // Check minimality if configured
        let is_minimal = if self.config.test_edge_cases {
            Some(self.minimality_checker.check(program))
        } else {
            None
        };

        // Determine success
        let success = accuracy >= self.config.min_accuracy
            && is_minimal.unwrap_or(true);

        VerificationResult {
            success,
            confidence: accuracy,
            counterfactual_accuracy: accuracy,
            tests_run: self.config.num_counterfactuals,
            edge_cases,
            details: Some(VerificationDetails {
                passed,
                failed,
                avg_error,
                max_error,
                is_minimal,
            }),
        }
    }

    /// Generate a test case for counterfactual testing
    fn generate_test_case(&self, program: &SynthesizedProgram, seed: usize) -> TestCase {
        // Generate random inputs based on variables in program
        let mut inputs = HashMap::new();

        for var in &program.variables {
            // Simple random generation (could be more sophisticated)
            let value = ((seed * 17 + var.len() * 31) % 1000) as f64 / 1000.0;
            inputs.insert(var.clone(), value);
        }

        TestCase {
            inputs,
            expected_strength: program.achieved_strength,
            specification: program.specification.clone(),
        }
    }

    /// Generate counterfactual query from test case
    ///
    /// Converts test case into a CounterfactualQuery that asks:
    /// "What would happen to the effect if we intervened on the cause?"
    fn generate_counterfactual_query(&self, test: &TestCase) -> CounterfactualQuery {
        // Extract cause and effect from specification
        let (cause, effect) = match &test.specification {
            CausalSpec::MakeCause { cause, effect, .. } => (cause.clone(), effect.clone()),
            CausalSpec::RemoveCause { cause, effect } => (cause.clone(), effect.clone()),
            CausalSpec::Strengthen { cause, effect, .. } => (cause.clone(), effect.clone()),
            CausalSpec::Weaken { cause, effect, .. } => (cause.clone(), effect.clone()),
            _ => {
                // For complex specs, use first and last variable
                let cause = test.inputs.keys().next().unwrap().clone();
                let effect = test.inputs.keys().last().unwrap().clone();
                (cause, effect)
            }
        };

        // Get intervention value from test inputs
        let intervention_value = test.inputs.get(&cause).copied().unwrap_or(0.5);

        // Create intervention
        let mut intervention = HashMap::new();
        intervention.insert(cause.clone(), intervention_value);

        // Create evidence from other variables
        let evidence: HashMap<String, f64> = test.inputs
            .iter()
            .filter(|(k, _)| *k != &cause)
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        CounterfactualQuery {
            intervention,
            evidence,
            query_variable: effect,
        }
    }

    /// Test using counterfactual engine (Enhancement #4 Phase 2)
    ///
    /// Uses CounterfactualEngine to get the true counterfactual answer,
    /// then compares it with the program's prediction.
    ///
    /// Returns: (actual_strength, error)
    fn test_with_counterfactual(
        &self,
        program: &SynthesizedProgram,
        test: &TestCase,
    ) -> (f64, f64) {
        if let Some(ref engine) = self.counterfactual_engine {
            // Phase 2: Use real counterfactual engine
            // Generate counterfactual query from test case
            let query = self.generate_counterfactual_query(test);

            // Ask engine for the true counterfactual answer
            let result = engine.query(&query);

            // The actual strength is the counterfactual value
            let actual_strength = result.value;

            // Compare with program's expected strength
            let error = (actual_strength - test.expected_strength).abs();

            (actual_strength, error)
        } else {
            // Phase 1 fallback: Use template estimation
            let actual_strength = self.estimate_strength_from_template(&program.template, &test.inputs);
            let error = (actual_strength - test.expected_strength).abs();

            (actual_strength, error)
        }
    }

    /// Run a single test case
    fn run_test(&self, program: &SynthesizedProgram, test: &TestCase) -> TestResult {
        // Phase 2: Use counterfactual testing when available
        let (actual_strength, error) = self.test_with_counterfactual(program, test);

        let passed = error < 0.1; // 10% tolerance
        let is_edge_case = error > 0.2; // Flag large errors as edge cases

        TestResult {
            passed,
            error,
            is_edge_case,
            description: if !passed {
                format!(
                    "Expected strength {:.2}, got {:.2} (error: {:.2})",
                    test.expected_strength, actual_strength, error
                )
            } else {
                String::new()
            },
        }
    }

    /// Estimate causal strength from program template
    /// (Simplified - full implementation would use Enhancement #4)
    fn estimate_strength_from_template(
        &self,
        template: &ProgramTemplate,
        inputs: &HashMap<String, f64>,
    ) -> f64 {
        match template {
            ProgramTemplate::Linear { weights, .. } => {
                // Sum of absolute weights
                weights.values().map(|w| w.abs()).sum()
            }
            ProgramTemplate::Sequence { programs } => {
                // Product of strengths
                programs
                    .iter()
                    .map(|p| self.estimate_strength_from_template(p, inputs))
                    .product()
            }
            _ => 0.5, // Placeholder for other template types
        }
    }
}

/// Test case for counterfactual testing
struct TestCase {
    inputs: HashMap<String, f64>,
    expected_strength: CausalStrength,
    specification: CausalSpec,
}

/// Result of running a test
struct TestResult {
    passed: bool,
    error: f64,
    is_edge_case: bool,
    description: String,
}

/// Checks if a program is minimal (no smaller program achieves same effect)
pub struct MinimalityChecker {
    /// Cache of known minimal programs
    cache: HashMap<String, bool>,
}

impl MinimalityChecker {
    /// Create new minimality checker
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Check if program is minimal
    pub fn check(&self, program: &SynthesizedProgram) -> bool {
        // Check cache
        let program_key = format!("{:?}", program.template);
        if let Some(&is_minimal) = self.cache.get(&program_key) {
            return is_minimal;
        }

        // For now, simple heuristic: complexity <= 2 * spec complexity
        let spec_complexity = program.specification.complexity();
        let is_minimal = program.complexity <= spec_complexity * 2;

        is_minimal
    }

    /// Generate simpler variants of a program
    pub fn generate_simpler_variants(&self, program: &SynthesizedProgram) -> Vec<ProgramTemplate> {
        let mut variants = Vec::new();

        // Try removing operations
        // Try simplifying conditions
        // Try reducing weights
        // (Implementation would be more sophisticated)

        variants
    }
}

impl Default for MinimalityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::causal_spec::CausalSpec;

    #[test]
    fn test_verifier_creation() {
        let verifier = CounterfactualVerifier::default();
        assert_eq!(verifier.config.num_counterfactuals, 1000);
    }

    #[test]
    fn test_verify_simple_program() {
        let verifier = CounterfactualVerifier::default();

        // Create a simple program
        let program = SynthesizedProgram {
            template: ProgramTemplate::Linear {
                weights: [("age".to_string(), 0.7)].iter().cloned().collect(),
                bias: 0.0,
            },
            specification: CausalSpec::MakeCause {
                cause: "age".to_string(),
                effect: "approved".to_string(),
                strength: 0.7,
            },
            achieved_strength: 0.7,
            confidence: 1.0,
            complexity: 1,
            explanation: None,
            variables: vec!["age".to_string(), "approved".to_string()],
        };

        let result = verifier.verify(&program);
        assert!(result.success);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_complexity_check() {
        let mut config = VerificationConfig::default();
        config.max_complexity = 5;
        let verifier = CounterfactualVerifier::new(config);

        // Create program exceeding complexity
        let program = SynthesizedProgram {
            template: ProgramTemplate::Linear {
                weights: HashMap::new(),
                bias: 0.0,
            },
            specification: CausalSpec::MakeCause {
                cause: "a".to_string(),
                effect: "b".to_string(),
                strength: 0.5,
            },
            achieved_strength: 0.5,
            confidence: 1.0,
            complexity: 10, // Exceeds max
            explanation: None,
            variables: vec!["a".to_string(), "b".to_string()],
        };

        let result = verifier.verify(&program);
        assert!(!result.success);
        assert!(!result.edge_cases.is_empty());
    }

    #[test]
    fn test_minimality_checker() {
        let checker = MinimalityChecker::new();

        let simple_program = SynthesizedProgram {
            template: ProgramTemplate::Linear {
                weights: [("a".to_string(), 1.0)].iter().cloned().collect(),
                bias: 0.0,
            },
            specification: CausalSpec::MakeCause {
                cause: "a".to_string(),
                effect: "b".to_string(),
                strength: 1.0,
            },
            achieved_strength: 1.0,
            confidence: 1.0,
            complexity: 1,
            explanation: None,
            variables: vec!["a".to_string(), "b".to_string()],
        };

        assert!(checker.check(&simple_program));
    }
}
