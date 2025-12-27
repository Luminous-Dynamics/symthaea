// Causal Specification Language
//
// A domain-specific language for expressing desired causal relationships
// that programs should implement.
//
// Key Innovation: Specify WHAT causal effect you want, not HOW to implement it

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// Strength of a causal relationship (0.0 = no causation, 1.0 = perfect causation)
pub type CausalStrength = f64;

/// Variable name in causal specification
pub type VarName = String;

/// Causal specification - defines desired causal effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalSpec {
    /// Create a direct causal link: cause → effect
    ///
    /// Example: Make age cause approval decision with strength 0.7
    /// ```
    /// CausalSpec::MakeCause {
    ///     cause: "age".to_string(),
    ///     effect: "approved".to_string(),
    ///     strength: 0.7,
    /// }
    /// ```
    MakeCause {
        cause: VarName,
        effect: VarName,
        strength: CausalStrength,
    },

    /// Remove a causal link (eliminate unwanted causation)
    ///
    /// Example: Remove bias from decision-making
    /// ```
    /// CausalSpec::RemoveCause {
    ///     cause: "race".to_string(),
    ///     effect: "decision".to_string(),
    /// }
    /// ```
    RemoveCause {
        cause: VarName,
        effect: VarName,
    },

    /// Create an indirect causal path: from → through... → to
    ///
    /// Example: Create path: input → hidden1 → hidden2 → output
    /// ```
    /// CausalSpec::CreatePath {
    ///     from: "input".to_string(),
    ///     through: vec!["hidden1".to_string(), "hidden2".to_string()],
    ///     to: "output".to_string(),
    /// }
    /// ```
    CreatePath {
        from: VarName,
        through: Vec<VarName>,
        to: VarName,
    },

    /// Strengthen existing causal link
    ///
    /// Example: Make feature more influential
    /// ```
    /// CausalSpec::Strengthen {
    ///     cause: "important_feature".to_string(),
    ///     effect: "decision".to_string(),
    ///     target_strength: 0.9,
    /// }
    /// ```
    Strengthen {
        cause: VarName,
        effect: VarName,
        target_strength: CausalStrength,
    },

    /// Weaken existing causal link
    ///
    /// Example: Reduce influence of noisy feature
    /// ```
    /// CausalSpec::Weaken {
    ///     cause: "noisy_feature".to_string(),
    ///     effect: "decision".to_string(),
    ///     target_strength: 0.1,
    /// }
    /// ```
    Weaken {
        cause: VarName,
        effect: VarName,
        target_strength: CausalStrength,
    },

    /// Mediate: Make all causation flow through mediator
    ///
    /// Example: All features → mediator → output
    /// ```
    /// CausalSpec::Mediate {
    ///     causes: vec!["feat1".to_string(), "feat2".to_string()],
    ///     mediator: "attention".to_string(),
    ///     effect: "output".to_string(),
    /// }
    /// ```
    Mediate {
        causes: Vec<VarName>,
        mediator: VarName,
        effect: VarName,
    },

    /// Conjunction: Multiple specifications must all be satisfied
    ///
    /// Example: Create link AND remove bias
    /// ```
    /// CausalSpec::And(vec![
    ///     Box::new(CausalSpec::MakeCause { ... }),
    ///     Box::new(CausalSpec::RemoveCause { ... }),
    /// ])
    /// ```
    And(Vec<Box<CausalSpec>>),

    /// Disjunction: At least one specification must be satisfied
    ///
    /// Example: Either strengthen OR create new path
    /// ```
    /// CausalSpec::Or(vec![
    ///     Box::new(CausalSpec::Strengthen { ... }),
    ///     Box::new(CausalSpec::CreatePath { ... }),
    /// ])
    /// ```
    Or(Vec<Box<CausalSpec>>),
}

impl CausalSpec {
    /// Get all variables mentioned in this specification
    pub fn variables(&self) -> HashSet<VarName> {
        let mut vars = HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut HashSet<VarName>) {
        match self {
            CausalSpec::MakeCause { cause, effect, .. }
            | CausalSpec::RemoveCause { cause, effect }
            | CausalSpec::Strengthen { cause, effect, .. }
            | CausalSpec::Weaken { cause, effect, .. } => {
                vars.insert(cause.clone());
                vars.insert(effect.clone());
            }
            CausalSpec::CreatePath { from, through, to } => {
                vars.insert(from.clone());
                vars.insert(to.clone());
                for var in through {
                    vars.insert(var.clone());
                }
            }
            CausalSpec::Mediate {
                causes,
                mediator,
                effect,
            } => {
                for cause in causes {
                    vars.insert(cause.clone());
                }
                vars.insert(mediator.clone());
                vars.insert(effect.clone());
            }
            CausalSpec::And(specs) | CausalSpec::Or(specs) => {
                for spec in specs {
                    spec.collect_variables(vars);
                }
            }
        }
    }

    /// Check if specification is structurally valid
    pub fn is_valid(&self) -> bool {
        match self {
            CausalSpec::MakeCause { strength, .. }
            | CausalSpec::Strengthen {
                target_strength: strength,
                ..
            }
            | CausalSpec::Weaken {
                target_strength: strength,
                ..
            } => {
                // Strength must be between 0 and 1
                *strength >= 0.0 && *strength <= 1.0
            }
            CausalSpec::CreatePath { through, .. } => {
                // Path must have at least one intermediate step
                !through.is_empty()
            }
            CausalSpec::Mediate { causes, .. } => {
                // Must have at least one cause
                !causes.is_empty()
            }
            CausalSpec::And(specs) | CausalSpec::Or(specs) => {
                // All sub-specs must be valid
                !specs.is_empty() && specs.iter().all(|s| s.is_valid())
            }
            _ => true,
        }
    }

    /// Estimate complexity of implementing this specification
    pub fn complexity(&self) -> usize {
        match self {
            CausalSpec::MakeCause { .. }
            | CausalSpec::RemoveCause { .. }
            | CausalSpec::Strengthen { .. }
            | CausalSpec::Weaken { .. } => 1,

            CausalSpec::CreatePath { through, .. } => through.len() + 1,

            CausalSpec::Mediate { causes, .. } => causes.len() + 1,

            CausalSpec::And(specs) => specs.iter().map(|s| s.complexity()).sum(),

            CausalSpec::Or(specs) => {
                // OR complexity is minimum of branches
                specs.iter().map(|s| s.complexity()).min().unwrap_or(0)
            }
        }
    }
}

/// Path in a causal graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPath {
    pub nodes: Vec<VarName>,
    pub strength: CausalStrength,
}

impl CausalPath {
    /// Create new causal path
    pub fn new(nodes: Vec<VarName>, strength: CausalStrength) -> Self {
        Self { nodes, strength }
    }

    /// Get length of path (number of edges)
    pub fn len(&self) -> usize {
        if self.nodes.len() > 0 {
            self.nodes.len() - 1
        } else {
            0
        }
    }

    /// Check if path is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Verifies that a specification is satisfiable
pub struct SpecVerifier {
    /// Known impossible specifications (learned from experience)
    impossible: HashSet<String>,
}

impl SpecVerifier {
    /// Create new specification verifier
    pub fn new() -> Self {
        Self {
            impossible: HashSet::new(),
        }
    }

    /// Check if specification is satisfiable
    pub fn verify(&self, spec: &CausalSpec) -> Result<(), String> {
        // Check structural validity
        if !spec.is_valid() {
            return Err("Specification is not structurally valid".to_string());
        }

        // Check if we know this is impossible
        let spec_key = format!("{:?}", spec);
        if self.impossible.contains(&spec_key) {
            return Err("Specification is known to be impossible".to_string());
        }

        // Check for contradictions in AND specifications
        if let CausalSpec::And(specs) = spec {
            for i in 0..specs.len() {
                for j in (i + 1)..specs.len() {
                    if self.are_contradictory(&specs[i], &specs[j]) {
                        return Err("Contradictory specifications in AND".to_string());
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if two specifications contradict each other
    fn are_contradictory(&self, spec1: &CausalSpec, spec2: &CausalSpec) -> bool {
        match (spec1, spec2) {
            // Creating and removing same link is contradiction
            (
                CausalSpec::MakeCause { cause: c1, effect: e1, .. },
                CausalSpec::RemoveCause { cause: c2, effect: e2 },
            ) | (
                CausalSpec::RemoveCause { cause: c2, effect: e2 },
                CausalSpec::MakeCause { cause: c1, effect: e1, .. },
            ) => c1 == c2 && e1 == e2,

            _ => false,
        }
    }

    /// Mark a specification as impossible (learning from failure)
    pub fn mark_impossible(&mut self, spec: &CausalSpec) {
        let spec_key = format!("{:?}", spec);
        self.impossible.insert(spec_key);
    }
}

impl Default for SpecVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_cause_valid() {
        let spec = CausalSpec::MakeCause {
            cause: "age".to_string(),
            effect: "approved".to_string(),
            strength: 0.7,
        };
        assert!(spec.is_valid());
    }

    #[test]
    fn test_invalid_strength() {
        let spec = CausalSpec::MakeCause {
            cause: "age".to_string(),
            effect: "approved".to_string(),
            strength: 1.5, // Invalid: > 1.0
        };
        assert!(!spec.is_valid());
    }

    #[test]
    fn test_create_path_valid() {
        let spec = CausalSpec::CreatePath {
            from: "input".to_string(),
            through: vec!["hidden".to_string()],
            to: "output".to_string(),
        };
        assert!(spec.is_valid());
    }

    #[test]
    fn test_create_path_invalid() {
        let spec = CausalSpec::CreatePath {
            from: "input".to_string(),
            through: vec![], // Invalid: no intermediate steps
            to: "output".to_string(),
        };
        assert!(!spec.is_valid());
    }

    #[test]
    fn test_variables_collection() {
        let spec = CausalSpec::CreatePath {
            from: "a".to_string(),
            through: vec!["b".to_string(), "c".to_string()],
            to: "d".to_string(),
        };

        let vars = spec.variables();
        assert_eq!(vars.len(), 4);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
        assert!(vars.contains("c"));
        assert!(vars.contains("d"));
    }

    #[test]
    fn test_complexity_simple() {
        let spec = CausalSpec::MakeCause {
            cause: "a".to_string(),
            effect: "b".to_string(),
            strength: 0.5,
        };
        assert_eq!(spec.complexity(), 1);
    }

    #[test]
    fn test_complexity_path() {
        let spec = CausalSpec::CreatePath {
            from: "a".to_string(),
            through: vec!["b".to_string(), "c".to_string()],
            to: "d".to_string(),
        };
        assert_eq!(spec.complexity(), 3);
    }

    #[test]
    fn test_spec_verifier() {
        let verifier = SpecVerifier::new();
        let spec = CausalSpec::MakeCause {
            cause: "age".to_string(),
            effect: "approved".to_string(),
            strength: 0.7,
        };
        assert!(verifier.verify(&spec).is_ok());
    }

    #[test]
    fn test_contradictory_specs() {
        let verifier = SpecVerifier::new();
        let spec = CausalSpec::And(vec![
            Box::new(CausalSpec::MakeCause {
                cause: "a".to_string(),
                effect: "b".to_string(),
                strength: 0.7,
            }),
            Box::new(CausalSpec::RemoveCause {
                cause: "a".to_string(),
                effect: "b".to_string(),
            }),
        ]);
        assert!(verifier.verify(&spec).is_err());
    }
}
