//! NixOS Configuration Verifier
//!
//! Uses Z3 (when available) or SMT-LIB2 output for verification.

use super::constraints::{Constraint, ConstraintSet, ConstraintSeverity};
use super::encoder::{SmtEncoder, SmtFormula, NixConfig};
use crate::hdc::binary_hv::HV16;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// VERIFICATION RESULTS
// ═══════════════════════════════════════════════════════════════════════════

/// Result of verifying a single constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Constraint is satisfied
    Satisfied,

    /// Constraint is violated
    Violated {
        /// Why it's violated
        reason: String,

        /// Counterexample (values that cause violation)
        counterexample: HashMap<String, String>,

        /// Suggested fix
        suggested_fix: Option<String>,
    },

    /// Could not verify (timeout, unsupported, etc.)
    Unknown {
        reason: String,
    },
}

impl VerificationResult {
    /// Is this a violation?
    pub fn is_violated(&self) -> bool {
        matches!(self, VerificationResult::Violated { .. })
    }

    /// Is this satisfied?
    pub fn is_satisfied(&self) -> bool {
        matches!(self, VerificationResult::Satisfied)
    }
}

/// Complete verification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Configuration that was verified
    pub config_hash: String,

    /// Time taken for verification
    pub verification_time_ms: f64,

    /// Results per constraint
    pub results: HashMap<String, VerificationResult>,

    /// Summary statistics
    pub summary: VerificationSummary,

    /// Recommended actions
    pub recommendations: Vec<Recommendation>,
}

/// Summary of verification results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationSummary {
    /// Total constraints checked
    pub total_constraints: usize,

    /// Constraints satisfied
    pub satisfied: usize,

    /// Constraints violated
    pub violated: usize,

    /// Constraints unknown
    pub unknown: usize,

    /// Critical violations
    pub critical_violations: usize,

    /// Error violations
    pub error_violations: usize,

    /// Warning violations
    pub warning_violations: usize,
}

impl VerificationSummary {
    /// Is the configuration safe to apply?
    pub fn is_safe(&self) -> bool {
        self.critical_violations == 0 && self.error_violations == 0
    }

    /// Overall health score (0.0 - 1.0)
    pub fn health_score(&self) -> f64 {
        if self.total_constraints == 0 {
            return 1.0;
        }

        let weighted_violations =
            (self.critical_violations * 10) +
            (self.error_violations * 5) +
            (self.warning_violations * 1);

        let max_weight = self.total_constraints * 10;
        1.0 - (weighted_violations as f64 / max_weight as f64).min(1.0)
    }
}

/// Recommendation for fixing issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Priority (lower = more urgent)
    pub priority: u8,

    /// Constraint that triggered this
    pub constraint_id: String,

    /// What to do
    pub action: String,

    /// Why this helps
    pub rationale: String,

    /// Semantic embedding for similarity search
    #[serde(skip)]
    pub semantic: Option<HV16>,
}

// ═══════════════════════════════════════════════════════════════════════════
// VERIFIER IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierConfig {
    /// Timeout for Z3 solver (ms)
    pub timeout_ms: u64,

    /// Whether to generate counterexamples
    pub generate_counterexamples: bool,

    /// Whether to generate fix suggestions
    pub suggest_fixes: bool,

    /// Minimum severity to report
    pub min_severity: ConstraintSeverity,

    /// Use Z3 if available
    pub use_z3: bool,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 5000,
            generate_counterexamples: true,
            suggest_fixes: true,
            min_severity: ConstraintSeverity::Warning,
            use_z3: true,
        }
    }
}

/// NixOS Configuration Verifier
pub struct Verifier {
    /// Constraint set
    constraints: ConstraintSet,

    /// SMT encoder
    encoder: SmtEncoder,

    /// Configuration
    config: VerifierConfig,

    /// Statistics
    stats: VerifierStats,
}

/// Verifier statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerifierStats {
    /// Total verifications performed
    pub total_verifications: u64,

    /// Total constraints checked
    pub total_constraints_checked: u64,

    /// Total violations found
    pub total_violations: u64,

    /// Average verification time (ms)
    pub avg_verification_time_ms: f64,
}

impl Verifier {
    /// Create a new verifier with default constraints
    pub fn new() -> Self {
        Self::with_constraints(ConstraintSet::nixos_standard())
    }

    /// Create with custom constraints
    pub fn with_constraints(constraints: ConstraintSet) -> Self {
        Self {
            constraints,
            encoder: SmtEncoder::new(),
            config: VerifierConfig::default(),
            stats: VerifierStats::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(constraints: ConstraintSet, config: VerifierConfig) -> Self {
        Self {
            constraints,
            encoder: SmtEncoder::new(),
            config,
            stats: VerifierStats::default(),
        }
    }

    /// Verify a NixOS configuration
    pub fn verify(&mut self, nix_config: &NixConfig) -> Result<VerificationReport> {
        let start = Instant::now();
        self.stats.total_verifications += 1;

        let mut results = HashMap::new();
        let mut summary = VerificationSummary::default();
        let mut recommendations = Vec::new();

        // Filter and clone constraints by severity (to avoid borrow issues)
        let constraints: Vec<Constraint> = self.constraints
            .by_severity(self.config.min_severity)
            .into_iter()
            .cloned()
            .collect();

        summary.total_constraints = constraints.len();
        self.stats.total_constraints_checked += constraints.len() as u64;

        // Verify each constraint
        for constraint in &constraints {
            let result = self.verify_constraint(constraint, nix_config)?;

            // Update summary
            match &result {
                VerificationResult::Satisfied => summary.satisfied += 1,
                VerificationResult::Violated { .. } => {
                    summary.violated += 1;
                    self.stats.total_violations += 1;

                    match constraint.severity {
                        ConstraintSeverity::Critical => summary.critical_violations += 1,
                        ConstraintSeverity::Error => summary.error_violations += 1,
                        ConstraintSeverity::Warning => summary.warning_violations += 1,
                        ConstraintSeverity::Info => {}
                    }

                    // Generate recommendation
                    if self.config.suggest_fixes {
                        if let Some(rec) = self.generate_recommendation(constraint, &result) {
                            recommendations.push(rec);
                        }
                    }
                }
                VerificationResult::Unknown { .. } => summary.unknown += 1,
            }

            results.insert(constraint.id.clone(), result);
        }

        // Sort recommendations by priority
        recommendations.sort_by_key(|r| r.priority);

        let elapsed = start.elapsed().as_millis() as f64;
        self.stats.avg_verification_time_ms =
            (self.stats.avg_verification_time_ms * 0.9) + (elapsed * 0.1);

        Ok(VerificationReport {
            config_hash: self.hash_config(nix_config),
            verification_time_ms: elapsed,
            results,
            summary,
            recommendations,
        })
    }

    /// Verify a single constraint
    fn verify_constraint(
        &mut self,
        constraint: &Constraint,
        nix_config: &NixConfig,
    ) -> Result<VerificationResult> {
        // Encode to SMT (used when smt feature enabled)
        let _formula = self.encoder.encode(constraint, nix_config)?;

        // Use Z3 if available, otherwise use heuristic verification
        #[cfg(feature = "smt")]
        if self.config.use_z3 {
            return self.verify_with_z3(constraint, &_formula);
        }

        // Fallback: heuristic verification
        self.verify_heuristic(constraint, nix_config)
    }

    /// Verify using Z3 solver (when feature enabled)
    #[cfg(feature = "smt")]
    fn verify_with_z3(
        &self,
        constraint: &Constraint,
        formula: &SmtFormula,
    ) -> Result<VerificationResult> {
        use z3::{Config, Context, Solver, SatResult};

        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let solver = Solver::new(&ctx);

        // Set timeout
        let params = z3::Params::new(&ctx);
        params.set_u32("timeout", self.config.timeout_ms as u32);
        solver.set_params(&params);

        // Parse and add assertions
        // Note: This is simplified - real implementation would need proper parsing
        for assertion in &formula.assertions {
            // Z3 Rust bindings require building expressions programmatically
            // For now, we'll use heuristic verification
        }

        // Check satisfiability
        match solver.check() {
            SatResult::Sat => {
                // If satisfiable, the constraints CANNOT all be met
                // (we encode violations as satisfiable)
                if let Some(model) = solver.get_model() {
                    let mut counterexample = HashMap::new();
                    // Extract counterexample from model
                    Ok(VerificationResult::Violated {
                        reason: formula.explanation.clone(),
                        counterexample,
                        suggested_fix: None,
                    })
                } else {
                    Ok(VerificationResult::Violated {
                        reason: formula.explanation.clone(),
                        counterexample: HashMap::new(),
                        suggested_fix: None,
                    })
                }
            }
            SatResult::Unsat => {
                // Unsatisfiable means no violation possible
                Ok(VerificationResult::Satisfied)
            }
            SatResult::Unknown => {
                Ok(VerificationResult::Unknown {
                    reason: "Z3 returned unknown".to_string(),
                })
            }
        }
    }

    /// Fallback heuristic verification (when Z3 not available)
    fn verify_heuristic(
        &self,
        constraint: &Constraint,
        nix_config: &NixConfig,
    ) -> Result<VerificationResult> {
        use super::constraints::{ConstraintKind, ServiceConstraint, SecurityConstraint};

        match &constraint.kind {
            ConstraintKind::Service(ServiceConstraint::PortConflict {
                service_a, service_b, port, protocol
            }) => {
                let a_enabled = nix_config.is_service_enabled(service_a);
                let b_enabled = nix_config.is_service_enabled(service_b);

                if a_enabled && b_enabled {
                    Ok(VerificationResult::Violated {
                        reason: format!(
                            "Both {} and {} are enabled and would conflict on port {}/{}",
                            service_a, service_b, port, protocol
                        ),
                        counterexample: HashMap::from([
                            (format!("services.{}.enable", service_a), "true".to_string()),
                            (format!("services.{}.enable", service_b), "true".to_string()),
                        ]),
                        suggested_fix: Some(format!(
                            "Disable one of the services or change port for one of them"
                        )),
                    })
                } else {
                    Ok(VerificationResult::Satisfied)
                }
            }

            ConstraintKind::Service(ServiceConstraint::RequiresService {
                dependent, dependency
            }) => {
                let dep_enabled = nix_config.is_service_enabled(dependent);
                let req_enabled = nix_config.is_service_enabled(dependency);

                if dep_enabled && !req_enabled {
                    Ok(VerificationResult::Violated {
                        reason: format!(
                            "Service {} is enabled but required service {} is not",
                            dependent, dependency
                        ),
                        counterexample: HashMap::from([
                            (format!("services.{}.enable", dependent), "true".to_string()),
                            (format!("services.{}.enable", dependency), "false".to_string()),
                        ]),
                        suggested_fix: Some(format!(
                            "Enable services.{}.enable = true;",
                            dependency
                        )),
                    })
                } else {
                    Ok(VerificationResult::Satisfied)
                }
            }

            ConstraintKind::Security(SecurityConstraint::InsecureOption {
                option_path, reason, secure_alternative
            }) => {
                // Check if the insecure option is enabled
                let is_insecure = match nix_config.get(option_path) {
                    Some(super::encoder::ConfigValue::Bool(true)) => true,
                    Some(super::encoder::ConfigValue::String(s))
                        if s == "yes" || s == "true" || s == "prohibit-password" => false,
                    Some(super::encoder::ConfigValue::String(s))
                        if s == "yes" => true,
                    _ => false,
                };

                if is_insecure {
                    Ok(VerificationResult::Violated {
                        reason: format!("{}: {}", option_path, reason),
                        counterexample: HashMap::from([
                            (option_path.clone(), "true".to_string()),
                        ]),
                        suggested_fix: secure_alternative.clone(),
                    })
                } else {
                    Ok(VerificationResult::Satisfied)
                }
            }

            _ => {
                // For unhandled constraints, return unknown
                Ok(VerificationResult::Unknown {
                    reason: "Heuristic verification not implemented for this constraint type".to_string(),
                })
            }
        }
    }

    /// Generate a recommendation from a violation
    fn generate_recommendation(
        &self,
        constraint: &Constraint,
        result: &VerificationResult,
    ) -> Option<Recommendation> {
        if let VerificationResult::Violated { suggested_fix, .. } = result {
            let priority = match constraint.severity {
                ConstraintSeverity::Critical => 1,
                ConstraintSeverity::Error => 2,
                ConstraintSeverity::Warning => 3,
                ConstraintSeverity::Info => 4,
            };

            Some(Recommendation {
                priority,
                constraint_id: constraint.id.clone(),
                action: suggested_fix.clone().unwrap_or_else(|| {
                    format!("Review constraint: {}", constraint.description)
                }),
                rationale: constraint.description.clone(),
                semantic: constraint.semantic.clone(),
            })
        } else {
            None
        }
    }

    /// Hash config for tracking
    fn hash_config(&self, config: &NixConfig) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        format!("{:?}", config.enabled_services).hash(&mut hasher);
        format!("{:?}", config.installed_packages).hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get verification statistics
    pub fn stats(&self) -> &VerifierStats {
        &self.stats
    }

    /// Get constraint set
    pub fn constraints(&self) -> &ConstraintSet {
        &self.constraints
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.add(constraint);
    }

    /// Generate SMT program for external verification
    pub fn generate_smt_program(&mut self, nix_config: &NixConfig) -> Result<String> {
        let constraints: Vec<&Constraint> = self.constraints
            .by_severity(self.config.min_severity)
            .into_iter()
            .collect();

        let mut formulas = Vec::new();
        for constraint in constraints {
            formulas.push(self.encoder.encode(constraint, nix_config)?);
        }

        Ok(self.encoder.generate_smt_program(&formulas))
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// QUICK VERIFICATION API
// ═══════════════════════════════════════════════════════════════════════════

/// Quick check if a configuration is safe to apply
pub fn is_config_safe(config: &NixConfig) -> bool {
    let mut verifier = Verifier::new();
    match verifier.verify(config) {
        Ok(report) => report.summary.is_safe(),
        Err(_) => false,
    }
}

/// Get violations from a configuration
pub fn get_violations(config: &NixConfig) -> Vec<(String, String)> {
    let mut verifier = Verifier::new();
    match verifier.verify(config) {
        Ok(report) => {
            report.results
                .into_iter()
                .filter_map(|(id, result)| {
                    if let VerificationResult::Violated { reason, .. } = result {
                        Some((id, reason))
                    } else {
                        None
                    }
                })
                .collect()
        }
        Err(_) => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verifier_creation() {
        let verifier = Verifier::new();
        assert!(!verifier.constraints().is_empty());
    }

    #[test]
    fn test_empty_config_safe() {
        let config = NixConfig::default();
        assert!(is_config_safe(&config));
    }

    #[test]
    fn test_port_conflict_detection() {
        let mut config = NixConfig::default();
        config.enabled_services.push("nginx".to_string());
        config.enabled_services.push("apache".to_string());

        let mut verifier = Verifier::new();
        let report = verifier.verify(&config).unwrap();

        // Should detect port conflict
        assert!(report.summary.violated > 0);
    }

    #[test]
    fn test_service_dependency() {
        use super::super::constraints::{Constraint, ConstraintKind, ServiceConstraint, ConstraintSeverity};

        let mut verifier = Verifier::new();

        // Add dependency constraint
        verifier.add_constraint(Constraint {
            id: "test-dep".to_string(),
            description: "Test requires PostgreSQL".to_string(),
            kind: ConstraintKind::Service(ServiceConstraint::RequiresService {
                dependent: "nextcloud".to_string(),
                dependency: "postgresql".to_string(),
            }),
            severity: ConstraintSeverity::Error,
            semantic: None,
            source: "test".to_string(),
        });

        // Config with nextcloud but not postgresql
        let mut config = NixConfig::default();
        config.enabled_services.push("nextcloud".to_string());

        let report = verifier.verify(&config).unwrap();
        assert!(report.results.get("test-dep").map_or(false, |r| r.is_violated()));
    }

    #[test]
    fn test_verification_summary() {
        let summary = VerificationSummary {
            total_constraints: 10,
            satisfied: 8,
            violated: 2,
            unknown: 0,
            critical_violations: 0,
            error_violations: 1,
            warning_violations: 1,
        };

        assert!(!summary.is_safe());  // Has error violation
        assert!(summary.health_score() > 0.0);
        assert!(summary.health_score() < 1.0);
    }

    #[test]
    fn test_generate_smt_program() {
        let mut verifier = Verifier::new();
        let config = NixConfig::default();

        let program = verifier.generate_smt_program(&config).unwrap();

        assert!(program.contains("set-logic"));
        assert!(program.contains("check-sat"));
    }
}
