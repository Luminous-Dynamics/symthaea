//! SMT Encoder for NixOS Constraints
//!
//! Translates NixOS constraints into SMT-LIB2 formulas for Z3 verification.

use super::constraints::{
    Constraint, ConstraintKind, ServiceConstraint, PackageConstraint,
    SecurityConstraint, ResourceConstraint, ConstraintSeverity,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration value from NixOS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    Bool(bool),
    Int(i64),
    String(String),
    List(Vec<ConfigValue>),
    AttrSet(HashMap<String, ConfigValue>),
    Null,
}

impl ConfigValue {
    /// Get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ConfigValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            ConfigValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Extracted NixOS configuration for verification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NixConfig {
    /// Configuration values by path
    pub values: HashMap<String, ConfigValue>,

    /// Enabled services
    pub enabled_services: Vec<String>,

    /// Installed packages
    pub installed_packages: Vec<String>,

    /// Open ports (from firewall config)
    pub open_ports: Vec<(u16, String)>,  // (port, protocol)
}

impl NixConfig {
    /// Get a configuration value by path
    pub fn get(&self, path: &str) -> Option<&ConfigValue> {
        self.values.get(path)
    }

    /// Check if a service is enabled
    pub fn is_service_enabled(&self, service: &str) -> bool {
        self.enabled_services.contains(&service.to_string())
    }

    /// Check if a package is installed
    pub fn is_package_installed(&self, package: &str) -> bool {
        self.installed_packages.contains(&package.to_string())
    }

    /// Check if a port is open
    pub fn is_port_open(&self, port: u16, protocol: &str) -> bool {
        self.open_ports.contains(&(port, protocol.to_string()))
    }
}

/// SMT formula representation
#[derive(Debug, Clone)]
pub struct SmtFormula {
    /// SMT-LIB2 assertions
    pub assertions: Vec<String>,

    /// Variable declarations
    pub declarations: Vec<String>,

    /// Constraint ID this formula represents
    pub constraint_id: String,

    /// Human-readable explanation
    pub explanation: String,
}

/// SMT Encoder for NixOS constraints
pub struct SmtEncoder {
    /// Unique variable counter
    var_counter: u64,

    /// Variable mappings for config paths
    path_vars: HashMap<String, String>,
}

impl SmtEncoder {
    /// Create a new encoder
    pub fn new() -> Self {
        Self {
            var_counter: 0,
            path_vars: HashMap::new(),
        }
    }

    /// Generate a unique variable name
    fn fresh_var(&mut self, prefix: &str) -> String {
        self.var_counter += 1;
        format!("{}_{}", prefix, self.var_counter)
    }

    /// Get or create variable for config path
    fn var_for_path(&mut self, path: &str) -> String {
        if let Some(var) = self.path_vars.get(path) {
            var.clone()
        } else {
            let var = self.fresh_var("cfg");
            self.path_vars.insert(path.to_string(), var.clone());
            var
        }
    }

    /// Encode a constraint to SMT formula
    pub fn encode(&mut self, constraint: &Constraint, config: &NixConfig) -> Result<SmtFormula> {
        match &constraint.kind {
            ConstraintKind::Service(sc) => self.encode_service(constraint, sc, config),
            ConstraintKind::Package(pc) => self.encode_package(constraint, pc, config),
            ConstraintKind::Security(sec) => self.encode_security(constraint, sec, config),
            ConstraintKind::Resource(rc) => self.encode_resource(constraint, rc, config),
            ConstraintKind::Custom(cc) => self.encode_custom(constraint, cc, config),
        }
    }

    /// Encode service constraint
    fn encode_service(
        &mut self,
        constraint: &Constraint,
        sc: &ServiceConstraint,
        config: &NixConfig,
    ) -> Result<SmtFormula> {
        let mut decls = Vec::new();
        let mut asserts = Vec::new();

        match sc {
            ServiceConstraint::PortConflict { service_a, service_b, port, protocol } => {
                // Encode: NOT (service_a.enabled AND service_b.enabled AND same_port)
                let var_a = self.fresh_var("svc");
                let var_b = self.fresh_var("svc");

                let enabled_a = config.is_service_enabled(service_a);
                let enabled_b = config.is_service_enabled(service_b);

                decls.push(format!("(declare-const {} Bool)", var_a));
                decls.push(format!("(declare-const {} Bool)", var_b));
                asserts.push(format!("(assert (= {} {}))", var_a, enabled_a));
                asserts.push(format!("(assert (= {} {}))", var_b, enabled_b));

                // The constraint: NOT (both enabled)
                asserts.push(format!("(assert (not (and {} {})))", var_a, var_b));

                Ok(SmtFormula {
                    declarations: decls,
                    assertions: asserts,
                    constraint_id: constraint.id.clone(),
                    explanation: format!(
                        "Services {} and {} cannot both use port {}/{}",
                        service_a, service_b, port, protocol
                    ),
                })
            }

            ServiceConstraint::RequiresService { dependent, dependency } => {
                let var_dep = self.fresh_var("svc");
                let var_req = self.fresh_var("svc");

                let dep_enabled = config.is_service_enabled(dependent);
                let req_enabled = config.is_service_enabled(dependency);

                decls.push(format!("(declare-const {} Bool)", var_dep));
                decls.push(format!("(declare-const {} Bool)", var_req));
                asserts.push(format!("(assert (= {} {}))", var_dep, dep_enabled));
                asserts.push(format!("(assert (= {} {}))", var_req, req_enabled));

                // The constraint: dependent => dependency
                asserts.push(format!("(assert (=> {} {}))", var_dep, var_req));

                Ok(SmtFormula {
                    declarations: decls,
                    assertions: asserts,
                    constraint_id: constraint.id.clone(),
                    explanation: format!(
                        "Service {} requires service {} to be enabled",
                        dependent, dependency
                    ),
                })
            }

            ServiceConstraint::ConflictsWith { service_a, service_b, reason } => {
                let var_a = self.fresh_var("svc");
                let var_b = self.fresh_var("svc");

                let enabled_a = config.is_service_enabled(service_a);
                let enabled_b = config.is_service_enabled(service_b);

                decls.push(format!("(declare-const {} Bool)", var_a));
                decls.push(format!("(declare-const {} Bool)", var_b));
                asserts.push(format!("(assert (= {} {}))", var_a, enabled_a));
                asserts.push(format!("(assert (= {} {}))", var_b, enabled_b));

                // NOT (both enabled)
                asserts.push(format!("(assert (not (and {} {})))", var_a, var_b));

                Ok(SmtFormula {
                    declarations: decls,
                    assertions: asserts,
                    constraint_id: constraint.id.clone(),
                    explanation: format!(
                        "Services {} and {} conflict: {}",
                        service_a, service_b, reason
                    ),
                })
            }

            _ => {
                // Placeholder for other service constraints
                Ok(SmtFormula {
                    declarations: vec![],
                    assertions: vec!["(assert true)".to_string()],
                    constraint_id: constraint.id.clone(),
                    explanation: format!("Service constraint: {:?}", sc),
                })
            }
        }
    }

    /// Encode package constraint
    fn encode_package(
        &mut self,
        constraint: &Constraint,
        pc: &PackageConstraint,
        config: &NixConfig,
    ) -> Result<SmtFormula> {
        let mut decls = Vec::new();
        let mut asserts = Vec::new();

        match pc {
            PackageConstraint::Conflict { package_a, package_b, reason } => {
                let var_a = self.fresh_var("pkg");
                let var_b = self.fresh_var("pkg");

                let installed_a = config.is_package_installed(package_a);
                let installed_b = config.is_package_installed(package_b);

                decls.push(format!("(declare-const {} Bool)", var_a));
                decls.push(format!("(declare-const {} Bool)", var_b));
                asserts.push(format!("(assert (= {} {}))", var_a, installed_a));
                asserts.push(format!("(assert (= {} {}))", var_b, installed_b));

                // NOT (both installed)
                asserts.push(format!("(assert (not (and {} {})))", var_a, var_b));

                Ok(SmtFormula {
                    declarations: decls,
                    assertions: asserts,
                    constraint_id: constraint.id.clone(),
                    explanation: format!(
                        "Packages {} and {} conflict: {}",
                        package_a, package_b, reason
                    ),
                })
            }

            PackageConstraint::Dependency { dependent, dependency, .. } => {
                let var_dep = self.fresh_var("pkg");
                let var_req = self.fresh_var("pkg");

                let dep_installed = config.is_package_installed(dependent);
                let req_installed = config.is_package_installed(dependency);

                decls.push(format!("(declare-const {} Bool)", var_dep));
                decls.push(format!("(declare-const {} Bool)", var_req));
                asserts.push(format!("(assert (= {} {}))", var_dep, dep_installed));
                asserts.push(format!("(assert (= {} {}))", var_req, req_installed));

                // dependent => dependency
                asserts.push(format!("(assert (=> {} {}))", var_dep, var_req));

                Ok(SmtFormula {
                    declarations: decls,
                    assertions: asserts,
                    constraint_id: constraint.id.clone(),
                    explanation: format!(
                        "Package {} requires package {}",
                        dependent, dependency
                    ),
                })
            }

            _ => {
                Ok(SmtFormula {
                    declarations: vec![],
                    assertions: vec!["(assert true)".to_string()],
                    constraint_id: constraint.id.clone(),
                    explanation: format!("Package constraint: {:?}", pc),
                })
            }
        }
    }

    /// Encode security constraint
    fn encode_security(
        &mut self,
        constraint: &Constraint,
        sec: &SecurityConstraint,
        config: &NixConfig,
    ) -> Result<SmtFormula> {
        let mut decls = Vec::new();
        let mut asserts = Vec::new();

        match sec {
            SecurityConstraint::InsecureOption { option_path, reason, .. } => {
                let var = self.var_for_path(option_path);

                // Get the actual value from config
                let is_insecure = match config.get(option_path) {
                    Some(ConfigValue::Bool(true)) => true,  // True often means insecure
                    Some(ConfigValue::String(s)) if s == "yes" || s == "true" => true,
                    _ => false,
                };

                decls.push(format!("(declare-const {} Bool)", var));
                asserts.push(format!("(assert (= {} {}))", var, is_insecure));

                // The constraint: should NOT be insecure
                asserts.push(format!("(assert (not {}))", var));

                Ok(SmtFormula {
                    declarations: decls,
                    assertions: asserts,
                    constraint_id: constraint.id.clone(),
                    explanation: format!(
                        "Security: {} should not be enabled ({})",
                        option_path, reason
                    ),
                })
            }

            SecurityConstraint::PortFirewall { port, protocol, must_be } => {
                let var = self.fresh_var("port");
                let is_open = config.is_port_open(*port, protocol);

                decls.push(format!("(declare-const {} Bool)", var));
                asserts.push(format!("(assert (= {} {}))", var, is_open));

                use super::constraints::FirewallPolicy;
                let constraint_formula = match must_be {
                    FirewallPolicy::Allow => format!("(assert {})", var),
                    FirewallPolicy::Block => format!("(assert (not {}))", var),
                    FirewallPolicy::LocalOnly | FirewallPolicy::Restricted => {
                        // More complex - would need additional modeling
                        "(assert true)".to_string()
                    }
                };

                asserts.push(constraint_formula);

                Ok(SmtFormula {
                    declarations: decls,
                    assertions: asserts,
                    constraint_id: constraint.id.clone(),
                    explanation: format!(
                        "Port {}/{} must be {:?}",
                        port, protocol, must_be
                    ),
                })
            }

            _ => {
                Ok(SmtFormula {
                    declarations: vec![],
                    assertions: vec!["(assert true)".to_string()],
                    constraint_id: constraint.id.clone(),
                    explanation: format!("Security constraint: {:?}", sec),
                })
            }
        }
    }

    /// Encode resource constraint
    fn encode_resource(
        &mut self,
        constraint: &Constraint,
        rc: &ResourceConstraint,
        _config: &NixConfig,
    ) -> Result<SmtFormula> {
        // Resource constraints often need integer arithmetic
        let mut decls = Vec::new();
        let mut asserts = Vec::new();

        match rc {
            ResourceConstraint::MemoryLimit { service, max_memory_mb, .. } => {
                let var = self.fresh_var("mem");

                decls.push(format!("(declare-const {} Int)", var));
                asserts.push(format!("(assert (<= {} {}))", var, max_memory_mb));

                Ok(SmtFormula {
                    declarations: decls,
                    assertions: asserts,
                    constraint_id: constraint.id.clone(),
                    explanation: format!(
                        "Service {} memory limit: {} MB",
                        service, max_memory_mb
                    ),
                })
            }

            _ => {
                Ok(SmtFormula {
                    declarations: vec![],
                    assertions: vec!["(assert true)".to_string()],
                    constraint_id: constraint.id.clone(),
                    explanation: format!("Resource constraint: {:?}", rc),
                })
            }
        }
    }

    /// Encode custom constraint
    fn encode_custom(
        &mut self,
        constraint: &Constraint,
        cc: &super::constraints::CustomConstraint,
        _config: &NixConfig,
    ) -> Result<SmtFormula> {
        // Custom constraints provide their own formula
        Ok(SmtFormula {
            declarations: vec![],
            assertions: vec![format!("(assert {})", cc.formula)],
            constraint_id: constraint.id.clone(),
            explanation: "Custom SMT constraint".to_string(),
        })
    }

    /// Generate complete SMT-LIB2 program for verification
    pub fn generate_smt_program(&mut self, formulas: &[SmtFormula]) -> String {
        let mut program = String::new();

        // Header
        program.push_str("; NixOS Configuration Verification\n");
        program.push_str("; Generated by Symthaea\n\n");
        program.push_str("(set-logic QF_LIA)\n\n");  // Quantifier-free Linear Integer Arithmetic

        // Collect all declarations
        program.push_str("; Variable declarations\n");
        for formula in formulas {
            for decl in &formula.declarations {
                program.push_str(decl);
                program.push('\n');
            }
        }

        program.push_str("\n; Constraint assertions\n");
        for formula in formulas {
            program.push_str(&format!("; Constraint: {}\n", formula.constraint_id));
            program.push_str(&format!("; {}\n", formula.explanation));
            for assertion in &formula.assertions {
                program.push_str(assertion);
                program.push('\n');
            }
            program.push('\n');
        }

        // Check satisfiability
        program.push_str("; Check if all constraints can be satisfied\n");
        program.push_str("(check-sat)\n");
        program.push_str("(get-model)\n");

        program
    }
}

impl Default for SmtEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = SmtEncoder::new();
        assert_eq!(encoder.var_counter, 0);
    }

    #[test]
    fn test_fresh_var() {
        let mut encoder = SmtEncoder::new();
        let v1 = encoder.fresh_var("test");
        let v2 = encoder.fresh_var("test");
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_config_value() {
        let val = ConfigValue::Bool(true);
        assert_eq!(val.as_bool(), Some(true));

        let val = ConfigValue::Int(42);
        assert_eq!(val.as_int(), Some(42));
    }

    #[test]
    fn test_nix_config() {
        let mut config = NixConfig::default();
        config.enabled_services.push("nginx".to_string());
        config.installed_packages.push("vim".to_string());
        config.open_ports.push((80, "tcp".to_string()));

        assert!(config.is_service_enabled("nginx"));
        assert!(!config.is_service_enabled("apache"));
        assert!(config.is_package_installed("vim"));
        assert!(config.is_port_open(80, "tcp"));
    }
}
