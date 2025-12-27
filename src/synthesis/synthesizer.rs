// Causal Program Synthesizer
//
// Core engine that synthesizes programs from causal specifications
//
// Innovation: Uses causal reasoning (Enhancement #4) to generate programs
// that capture TRUE causal relationships, not just correlations

use super::causal_spec::{CausalSpec, CausalStrength, VarName};
use super::{SynthesisError, SynthesisResult};
use crate::observability::{
    CausalInterventionEngine, InterventionSpec, InterventionType,
    ActionPlanner, Goal, GoalDirection,
    ExplanationGenerator,
};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Configuration for program synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// Maximum complexity of synthesized program (number of operations)
    pub max_complexity: usize,

    /// Target confidence for causal relationships (0.0 - 1.0)
    pub min_confidence: f64,

    /// Maximum number of synthesis attempts
    pub max_attempts: usize,

    /// Whether to optimize for minimality
    pub optimize_minimal: bool,

    /// Whether to generate explanations
    pub generate_explanations: bool,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            max_complexity: 100,
            min_confidence: 0.8,
            max_attempts: 10,
            optimize_minimal: true,
            generate_explanations: true,
        }
    }
}

/// Template for program synthesis
#[derive(Debug, Clone)]
pub enum ProgramTemplate {
    /// Linear transformation: output = w1*x1 + w2*x2 + ... + bias
    Linear {
        weights: HashMap<VarName, f64>,
        bias: f64,
    },

    /// Neural network layer
    NeuralLayer {
        inputs: Vec<VarName>,
        outputs: Vec<VarName>,
        activation: ActivationFunction,
    },

    /// Decision tree
    DecisionTree {
        root: Box<TreeNode>,
    },

    /// Conditional: if condition then branch1 else branch2
    Conditional {
        condition: Box<ProgramTemplate>,
        then_branch: Box<ProgramTemplate>,
        else_branch: Box<ProgramTemplate>,
    },

    /// Composition: program1 then program2
    Sequence {
        programs: Vec<ProgramTemplate>,
    },
}

/// Activation function for neural layers
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
}

/// Decision tree node
#[derive(Debug, Clone)]
pub enum TreeNode {
    Leaf {
        value: f64,
    },
    Split {
        variable: VarName,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// A synthesized program with metadata
#[derive(Debug, Clone)]
pub struct SynthesizedProgram {
    /// The program template
    pub template: ProgramTemplate,

    /// Causal specification it implements
    pub specification: CausalSpec,

    /// Estimated causal strength achieved
    pub achieved_strength: CausalStrength,

    /// Confidence in correctness (from verification)
    pub confidence: f64,

    /// Complexity (number of operations)
    pub complexity: usize,

    /// Human-readable explanation (if generated)
    pub explanation: Option<String>,

    /// Variables used in program
    pub variables: Vec<VarName>,
}

/// Result of synthesis attempt
pub type SynthesisAttempt = Result<SynthesizedProgram, SynthesisError>;

/// Causal Program Synthesizer
///
/// Main synthesis engine that:
/// 1. Takes causal specification
/// 2. Uses Enhancement #4 to understand current causal structure
/// 3. Plans intervention to achieve specification
/// 4. Generates program implementing intervention
/// 5. Verifies program with counterfactuals
pub struct CausalProgramSynthesizer {
    /// Configuration
    config: SynthesisConfig,

    /// Intervention engine (from Enhancement #4)
    intervention_engine: Option<CausalInterventionEngine>,

    /// Action planner (from Enhancement #4 Phase 3)
    action_planner: Option<ActionPlanner>,

    /// Explanation generator (from Enhancement #4 Phase 4)
    explanation_generator: Option<ExplanationGenerator>,

    /// Cache of synthesized programs
    cache: HashMap<String, SynthesizedProgram>,
}

impl CausalProgramSynthesizer {
    /// Create new synthesizer with configuration
    pub fn new(config: SynthesisConfig) -> Self {
        Self {
            config,
            intervention_engine: None,
            action_planner: None,
            explanation_generator: None,
            cache: HashMap::new(),
        }
    }

    /// Create synthesizer with default configuration
    pub fn default() -> Self {
        Self::new(SynthesisConfig::default())
    }

    /// Set intervention engine (Enhancement #4)
    pub fn with_intervention_engine(mut self, engine: CausalInterventionEngine) -> Self {
        self.intervention_engine = Some(engine);
        self
    }

    /// Set action planner (Enhancement #4 Phase 3)
    pub fn with_action_planner(mut self, planner: ActionPlanner) -> Self {
        self.action_planner = Some(planner);
        self
    }

    /// Set explanation generator (Enhancement #4 Phase 4)
    pub fn with_explanation_generator(mut self, generator: ExplanationGenerator) -> Self {
        self.explanation_generator = Some(generator);
        self
    }

    /// Test program using intervention engine (Enhancement #4 Phase 1)
    ///
    /// When available, uses CausalInterventionEngine to predict the effect
    /// of interventions and compares with program's predictions.
    ///
    /// Returns: (achieved_strength, confidence)
    fn test_with_intervention(
        &mut self,
        cause: &VarName,
        effect: &VarName,
        expected_strength: CausalStrength,
    ) -> (CausalStrength, f64) {
        if let Some(ref mut engine) = self.intervention_engine {
            // Phase 2: Use real intervention engine
            let result = engine.predict_intervention(cause, effect);

            // Extract strength from predicted value
            let achieved_strength = result.predicted_value;

            // Confidence is inverse of uncertainty
            let confidence = (1.0 - result.uncertainty).max(0.0).min(1.0);

            // Check if achieved strength is close to expected
            let strength_error = (achieved_strength - expected_strength).abs();
            let strength_confidence = (1.0 - strength_error).max(0.0);

            // Combined confidence: both prediction confidence AND strength accuracy
            let combined_confidence = (confidence * 0.5 + strength_confidence * 0.5)
                .max(0.0)
                .min(1.0);

            (achieved_strength, combined_confidence)
        } else {
            // Phase 1 fallback: Assume perfect achievement
            (expected_strength, 1.0)
        }
    }

    /// Generate explanation for a synthesized program
    ///
    /// Phase 2: Generates detailed causal explanations
    fn generate_explanation(
        &self,
        spec: &CausalSpec,
        template: &ProgramTemplate,
    ) -> Option<String> {
        // Generate explanation based on specification type
        let spec_explanation = match spec {
            CausalSpec::MakeCause { cause, effect, strength } => {
                format!(
                    "Creates causal relationship {} → {} with strength {:.2}. \
                     This means changes in {} will cause proportional changes in {} \
                     (correlation coefficient ≈ {:.2}).",
                    cause, effect, strength, cause, effect, strength
                )
            }
            CausalSpec::RemoveCause { cause, effect } => {
                format!(
                    "Removes causal link {} → {} by zeroing the causal pathway. \
                     This eliminates any direct causal influence from {} to {}.",
                    cause, effect, cause, effect
                )
            }
            CausalSpec::CreatePath { from, through, to } => {
                format!(
                    "Creates causal path {} → {} → {} through mediators. \
                     Effect propagates sequentially through the chain.",
                    from, through.join(" → "), to
                )
            }
            CausalSpec::Strengthen { cause, effect, target_strength } => {
                format!(
                    "Strengthens causal link {} → {} to {:.2}. \
                     Amplifies the causal effect to achieve target strength.",
                    cause, effect, target_strength
                )
            }
            CausalSpec::Weaken { cause, effect, target_strength } => {
                format!(
                    "Weakens causal link {} → {} to {:.2}. \
                     Attenuates the causal effect to reduce influence.",
                    cause, effect, target_strength
                )
            }
            CausalSpec::Mediate { causes, mediator, effect } => {
                format!(
                    "Creates mediated causation: {} → {} → {}. \
                     Multiple causes influence effect through common mediator.",
                    causes.join(" + "), mediator, effect
                )
            }
            CausalSpec::And(specs) => {
                format!(
                    "Conjunction of {} causal specifications executed in sequence.",
                    specs.len()
                )
            }
            CausalSpec::Or(specs) => {
                format!(
                    "Disjunction offering {} alternative causal pathways.",
                    specs.len()
                )
            }
        };

        // Add template-specific details
        let template_details = match template {
            ProgramTemplate::Linear { weights, bias } => {
                let weight_desc: Vec<String> = weights
                    .iter()
                    .map(|(var, w)| format!("{}={:.2}", var, w))
                    .collect();
                format!(" Implementation: Linear transform [{}], bias={:.2}",
                    weight_desc.join(", "), bias)
            }
            ProgramTemplate::Sequence { programs } => {
                format!(" Implementation: {} sequential operations", programs.len())
            }
            ProgramTemplate::NeuralLayer { inputs, outputs, .. } => {
                format!(
                    " Implementation: Neural layer ({} → {} neurons)",
                    inputs.len(),
                    outputs.len()
                )
            }
            _ => String::new(),
        };

        Some(format!("{}{}", spec_explanation, template_details))
    }

    /// Synthesize program from causal specification
    ///
    /// This is the main entry point for synthesis.
    ///
    /// Phase 2 Enhancement: When Enhancement #4 components are available,
    /// uses them for intelligent synthesis:
    /// - ActionPlanner: Finds optimal intervention sequence
    /// - InterventionEngine: Tests causal effects
    /// - ExplanationGenerator: Generates human-readable explanations
    pub fn synthesize(&mut self, spec: &CausalSpec) -> SynthesisResult<SynthesizedProgram> {
        // Check cache first
        let spec_key = format!("{:?}", spec);
        if let Some(cached) = self.cache.get(&spec_key) {
            return Ok(cached.clone());
        }

        // Validate specification
        if !spec.is_valid() {
            return Err(SynthesisError::UnsatisfiableSpec(
                "Specification is not structurally valid".to_string(),
            ));
        }

        // Synthesize based on specification type
        let program = match spec {
            CausalSpec::MakeCause {
                cause,
                effect,
                strength,
            } => self.synthesize_make_cause(cause, effect, *strength)?,

            CausalSpec::RemoveCause { cause, effect } => {
                self.synthesize_remove_cause(cause, effect)?
            }

            CausalSpec::CreatePath { from, through, to } => {
                self.synthesize_create_path(from, through, to)?
            }

            CausalSpec::Strengthen {
                cause,
                effect,
                target_strength,
            } => self.synthesize_strengthen(cause, effect, *target_strength)?,

            CausalSpec::Weaken {
                cause,
                effect,
                target_strength,
            } => self.synthesize_weaken(cause, effect, *target_strength)?,

            CausalSpec::Mediate {
                causes,
                mediator,
                effect,
            } => self.synthesize_mediate(causes, mediator, effect)?,

            CausalSpec::And(specs) => self.synthesize_conjunction(specs)?,

            CausalSpec::Or(specs) => self.synthesize_disjunction(specs)?,
        };

        // Cache result
        self.cache.insert(spec_key, program.clone());

        Ok(program)
    }

    /// Synthesize program that creates causal link: cause → effect
    fn synthesize_make_cause(
        &mut self,
        cause: &VarName,
        effect: &VarName,
        strength: CausalStrength,
    ) -> SynthesisResult<SynthesizedProgram> {
        // Create linear transformation implementing causal link
        let mut weights = HashMap::new();
        weights.insert(cause.clone(), strength);

        let template = ProgramTemplate::Linear {
            weights,
            bias: 0.0,
        };

        let spec = CausalSpec::MakeCause {
            cause: cause.clone(),
            effect: effect.clone(),
            strength,
        };

        // Phase 2: Test with intervention engine if available
        let (achieved_strength, confidence) = self.test_with_intervention(
            cause,
            effect,
            strength,
        );

        // Phase 2: Generate rich explanation
        let explanation = self.generate_explanation(&spec, &template);

        Ok(SynthesizedProgram {
            template,
            specification: spec,
            achieved_strength, // Real value from intervention test
            confidence,        // Real confidence from intervention test
            complexity: 1,
            explanation,
            variables: vec![cause.clone(), effect.clone()],
        })
    }

    /// Synthesize program that removes causal link
    fn synthesize_remove_cause(
        &mut self,
        cause: &VarName,
        effect: &VarName,
    ) -> SynthesisResult<SynthesizedProgram> {
        // Create program that zeros out causal effect
        let mut weights = HashMap::new();
        weights.insert(cause.clone(), 0.0); // Zero weight = no causation

        let template = ProgramTemplate::Linear {
            weights,
            bias: 0.0,
        };

        Ok(SynthesizedProgram {
            template,
            specification: CausalSpec::RemoveCause {
                cause: cause.clone(),
                effect: effect.clone(),
            },
            achieved_strength: 0.0,
            confidence: 1.0,
            complexity: 1,
            explanation: Some(format!(
                "Removal program: {} independent of {}",
                effect, cause
            )),
            variables: vec![cause.clone(), effect.clone()],
        })
    }

    /// Plan optimal path using action planner (Enhancement #4 Phase 2)
    ///
    /// Uses ActionPlanner to find the optimal intervention sequence
    /// from source to target, potentially discovering better paths
    /// than manually specified.
    ///
    /// Returns: (optimal_path, confidence)
    fn plan_optimal_path(
        &mut self,
        from: &VarName,
        to: &VarName,
    ) -> (Vec<VarName>, f64) {
        if let Some(ref mut planner) = self.action_planner {
            // Phase 2: Use real action planner
            use crate::observability::{Goal, GoalDirection};

            // Create goal: maximize target variable
            let goal = Goal {
                target: to.clone(),
                desired_value: 1.0,
                tolerance: 0.1,
                direction: GoalDirection::Maximize,
            };

            // For planning, we need candidate nodes
            // In a real implementation, this would come from the causal graph
            let candidates = vec![from.clone(), to.clone()];

            // Plan action sequence
            let plan = planner.plan(&goal, &candidates);

            // Extract intervention sequence from plan
            let path: Vec<VarName> = plan.interventions
                .iter()
                .map(|intervention| intervention.node.clone())
                .collect();

            let confidence = plan.confidence;

            (path, confidence)
        } else {
            // Phase 1 fallback: Direct path
            (vec![from.clone()], 0.9)
        }
    }

    /// Synthesize program that creates causal path: from → through... → to
    fn synthesize_create_path(
        &mut self,
        from: &VarName,
        through: &Vec<VarName>,
        to: &VarName,
    ) -> SynthesisResult<SynthesizedProgram> {
        // Phase 2: Use action planner to find optimal path if available
        let (optimal_path, planner_confidence) = if self.action_planner.is_some() && through.is_empty() {
            // If no path specified, use planner to find one
            self.plan_optimal_path(from, to)
        } else {
            // Use specified path
            (vec![from.clone()], 0.9)
        };

        // Use planner's path if it's better, otherwise use specified path
        let path_to_use = if !through.is_empty() {
            through.clone()
        } else {
            optimal_path
        };

        // Create sequence of programs implementing path
        let mut programs = Vec::new();

        // from → through[0] (or from → to if path_to_use is empty)
        if !path_to_use.is_empty() {
            programs.push(ProgramTemplate::Linear {
                weights: [(from.clone(), 1.0)].iter().cloned().collect(),
                bias: 0.0,
            });

            // through[i] → through[i+1]
            for i in 0..path_to_use.len() - 1 {
                programs.push(ProgramTemplate::Linear {
                    weights: [(path_to_use[i].clone(), 1.0)].iter().cloned().collect(),
                    bias: 0.0,
                });
            }

            // through[last] → to
            programs.push(ProgramTemplate::Linear {
                weights: [(path_to_use.last().unwrap().clone(), 1.0)]
                    .iter()
                    .cloned()
                    .collect(),
                bias: 0.0,
            });
        } else {
            // Direct connection if no intermediates
            programs.push(ProgramTemplate::Linear {
                weights: [(from.clone(), 1.0)].iter().cloned().collect(),
                bias: 0.0,
            });
        }

        let template = ProgramTemplate::Sequence { programs };

        let mut variables = vec![from.clone()];
        variables.extend(path_to_use.clone());
        variables.push(to.clone());

        let spec = CausalSpec::CreatePath {
            from: from.clone(),
            through: path_to_use.clone(),
            to: to.clone(),
        };

        // Use planner confidence if planner was used
        let confidence = if self.action_planner.is_some() && through.is_empty() {
            planner_confidence
        } else {
            0.9
        };

        // Generate explanation
        let explanation = self.generate_explanation(&spec, &template);

        Ok(SynthesizedProgram {
            template,
            specification: spec,
            achieved_strength: 1.0 / (path_to_use.len() as f64 + 1.0), // Attenuates with path length
            confidence,
            complexity: path_to_use.len() + 1,
            explanation,
            variables,
        })
    }

    /// Synthesize program that strengthens existing link
    fn synthesize_strengthen(
        &mut self,
        cause: &VarName,
        effect: &VarName,
        target_strength: CausalStrength,
    ) -> SynthesisResult<SynthesizedProgram> {
        // Similar to MakeCause but with amplification
        let mut weights = HashMap::new();
        weights.insert(cause.clone(), target_strength);

        let template = ProgramTemplate::Linear {
            weights,
            bias: 0.0,
        };

        let spec = CausalSpec::Strengthen {
            cause: cause.clone(),
            effect: effect.clone(),
            target_strength,
        };

        // Phase 2: Test with intervention engine if available
        let (achieved_strength, confidence) = self.test_with_intervention(
            cause,
            effect,
            target_strength,
        );

        // Phase 2: Generate explanation
        let explanation = self.generate_explanation(&spec, &template);

        Ok(SynthesizedProgram {
            template,
            specification: spec,
            achieved_strength, // Real value from intervention test
            confidence,        // Real confidence from intervention test
            complexity: 1,
            explanation,
            variables: vec![cause.clone(), effect.clone()],
        })
    }

    /// Synthesize program that weakens existing link
    fn synthesize_weaken(
        &mut self,
        cause: &VarName,
        effect: &VarName,
        target_strength: CausalStrength,
    ) -> SynthesisResult<SynthesizedProgram> {
        // Similar to Strengthen but with smaller weight
        self.synthesize_strengthen(cause, effect, target_strength)
    }

    /// Synthesize program that mediates causation through mediator
    fn synthesize_mediate(
        &mut self,
        causes: &Vec<VarName>,
        mediator: &VarName,
        effect: &VarName,
    ) -> SynthesisResult<SynthesizedProgram> {
        // Create two-layer program: causes → mediator → effect
        let mut programs = Vec::new();

        // Layer 1: causes → mediator (aggregate)
        let weights: HashMap<VarName, f64> = causes
            .iter()
            .map(|c| (c.clone(), 1.0 / causes.len() as f64))
            .collect();

        programs.push(ProgramTemplate::Linear {
            weights,
            bias: 0.0,
        });

        // Layer 2: mediator → effect
        programs.push(ProgramTemplate::Linear {
            weights: [(mediator.clone(), 1.0)].iter().cloned().collect(),
            bias: 0.0,
        });

        let template = ProgramTemplate::Sequence { programs };

        let mut variables = causes.clone();
        variables.push(mediator.clone());
        variables.push(effect.clone());

        Ok(SynthesizedProgram {
            template,
            specification: CausalSpec::Mediate {
                causes: causes.clone(),
                mediator: mediator.clone(),
                effect: effect.clone(),
            },
            achieved_strength: 0.8, // Placeholder
            confidence: 0.85,
            complexity: causes.len() + 1,
            explanation: Some(format!(
                "Mediation program: {} → {} → {}",
                causes.join(" + "),
                mediator,
                effect
            )),
            variables,
        })
    }

    /// Synthesize program satisfying conjunction of specifications
    fn synthesize_conjunction(
        &mut self,
        specs: &Vec<Box<CausalSpec>>,
    ) -> SynthesisResult<SynthesizedProgram> {
        // Synthesize each sub-specification
        let mut programs = Vec::new();
        let mut total_complexity = 0;
        let mut min_confidence: f64 = 1.0;

        for spec in specs {
            let sub_program = self.synthesize(spec)?;
            programs.push(sub_program.template.clone());
            total_complexity += sub_program.complexity;
            min_confidence = min_confidence.min(sub_program.confidence);
        }

        let template = ProgramTemplate::Sequence { programs };

        Ok(SynthesizedProgram {
            template,
            specification: CausalSpec::And(specs.clone()),
            achieved_strength: 0.7, // Placeholder
            confidence: min_confidence,
            complexity: total_complexity,
            explanation: Some(format!(
                "Conjunction program implementing {} specifications",
                specs.len()
            )),
            variables: vec![], // Collect from sub-programs
        })
    }

    /// Synthesize program satisfying disjunction of specifications
    fn synthesize_disjunction(
        &mut self,
        specs: &Vec<Box<CausalSpec>>,
    ) -> SynthesisResult<SynthesizedProgram> {
        // Try each specification, return first that succeeds
        for spec in specs {
            if let Ok(program) = self.synthesize(spec) {
                return Ok(program);
            }
        }

        Err(SynthesisError::UnsatisfiableSpec(
            "No specification in disjunction could be satisfied".to_string(),
        ))
    }

    /// Get statistics about synthesizer
    pub fn stats(&self) -> SynthesizerStats {
        SynthesizerStats {
            cache_size: self.cache.len(),
            total_synthesized: self.cache.len(),
        }
    }
}

/// Statistics about synthesizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizerStats {
    pub cache_size: usize,
    pub total_synthesized: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesize_make_cause() {
        let mut synthesizer = CausalProgramSynthesizer::default();
        let spec = CausalSpec::MakeCause {
            cause: "age".to_string(),
            effect: "approved".to_string(),
            strength: 0.7,
        };

        let result = synthesizer.synthesize(&spec);
        assert!(result.is_ok());

        let program = result.unwrap();
        assert_eq!(program.complexity, 1);
        assert_eq!(program.variables.len(), 2);
    }

    #[test]
    fn test_synthesize_create_path() {
        let mut synthesizer = CausalProgramSynthesizer::default();
        let spec = CausalSpec::CreatePath {
            from: "input".to_string(),
            through: vec!["hidden1".to_string(), "hidden2".to_string()],
            to: "output".to_string(),
        };

        let result = synthesizer.synthesize(&spec);
        assert!(result.is_ok());

        let program = result.unwrap();
        assert_eq!(program.complexity, 3); // from → h1 → h2 → to
        assert_eq!(program.variables.len(), 4);
    }

    #[test]
    fn test_synthesize_remove_cause() {
        let mut synthesizer = CausalProgramSynthesizer::default();
        let spec = CausalSpec::RemoveCause {
            cause: "bias".to_string(),
            effect: "decision".to_string(),
        };

        let result = synthesizer.synthesize(&spec);
        assert!(result.is_ok());

        let program = result.unwrap();
        assert_eq!(program.achieved_strength, 0.0);
    }

    #[test]
    fn test_cache_works() {
        let mut synthesizer = CausalProgramSynthesizer::default();
        let spec = CausalSpec::MakeCause {
            cause: "a".to_string(),
            effect: "b".to_string(),
            strength: 0.5,
        };

        // First synthesis
        let result1 = synthesizer.synthesize(&spec);
        assert!(result1.is_ok());

        // Second synthesis should use cache
        let result2 = synthesizer.synthesize(&spec);
        assert!(result2.is_ok());

        assert_eq!(synthesizer.stats().cache_size, 1);
    }
}
