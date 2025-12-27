//! NixOS Configuration Constraints
//!
//! Defines the types of constraints that can be verified.

use crate::hdc::binary_hv::HV16;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A constraint that must hold in a valid NixOS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Unique identifier
    pub id: String,

    /// Human-readable description
    pub description: String,

    /// The kind of constraint
    pub kind: ConstraintKind,

    /// Severity if violated
    pub severity: ConstraintSeverity,

    /// Semantic embedding for HDC-based similarity search
    #[serde(skip)]
    pub semantic: Option<HV16>,

    /// Source of this constraint (e.g., "services.nginx", "security.firewall")
    pub source: String,
}

/// The specific type of constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Service-level constraints
    Service(ServiceConstraint),

    /// Package-level constraints
    Package(PackageConstraint),

    /// Security constraints
    Security(SecurityConstraint),

    /// Resource constraints
    Resource(ResourceConstraint),

    /// Custom constraint with SMT formula
    Custom(CustomConstraint),
}

/// Constraint severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConstraintSeverity {
    /// Information only
    Info,

    /// Warning - may cause issues
    Warning,

    /// Error - will cause failure
    Error,

    /// Critical - security or data loss risk
    Critical,
}

// ═══════════════════════════════════════════════════════════════════════════
// SERVICE CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

/// Constraints on systemd services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceConstraint {
    /// Two services cannot bind the same port
    PortConflict {
        service_a: String,
        service_b: String,
        port: u16,
        protocol: String,  // "tcp" or "udp"
    },

    /// Service A must start before Service B
    OrderingDependency {
        before: String,
        after: String,
    },

    /// Service requires another service to be enabled
    RequiresService {
        dependent: String,
        dependency: String,
    },

    /// Service conflicts with another (cannot both be enabled)
    ConflictsWith {
        service_a: String,
        service_b: String,
        reason: String,
    },

    /// Service requires specific user/group
    RequiresUser {
        service: String,
        user: String,
        group: Option<String>,
    },

    /// Service requires specific capabilities
    RequiresCapability {
        service: String,
        capabilities: Vec<String>,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// PACKAGE CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

/// Constraints on packages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PackageConstraint {
    /// Package A conflicts with Package B
    Conflict {
        package_a: String,
        package_b: String,
        reason: String,
    },

    /// Package requires another package
    Dependency {
        dependent: String,
        dependency: String,
        version_constraint: Option<String>,
    },

    /// Package provides a virtual capability
    Provides {
        package: String,
        capability: String,
    },

    /// Package requires runtime dependency (not just build-time)
    RuntimeDependency {
        package: String,
        runtime_dep: String,
    },

    /// Package is deprecated
    Deprecated {
        package: String,
        replacement: Option<String>,
        reason: String,
    },

    /// Package has known vulnerability
    Vulnerability {
        package: String,
        cve: String,
        severity: String,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// SECURITY CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

/// Security-related constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityConstraint {
    /// Port must be protected by firewall
    PortFirewall {
        port: u16,
        protocol: String,
        must_be: FirewallPolicy,
    },

    /// Service must run as non-root
    NonRootService {
        service: String,
    },

    /// Option is insecure and should be avoided
    InsecureOption {
        option_path: String,
        reason: String,
        secure_alternative: Option<String>,
    },

    /// File/directory permission constraint
    Permission {
        path: String,
        max_mode: u32,  // e.g., 0o600
        reason: String,
    },

    /// Secret must not be in plain text in config
    NoPlainTextSecret {
        option_path: String,
        secret_type: String,  // e.g., "password", "api_key"
    },

    /// SSL/TLS must be enabled for service
    RequireTLS {
        service: String,
        min_version: Option<String>,  // e.g., "TLSv1.2"
    },
}

/// Firewall policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FirewallPolicy {
    /// Port must be allowed
    Allow,

    /// Port must be blocked
    Block,

    /// Port should only be accessible locally
    LocalOnly,

    /// Port should only be accessible from specific IPs
    Restricted,
}

// ═══════════════════════════════════════════════════════════════════════════
// RESOURCE CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

/// Resource allocation constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceConstraint {
    /// File path conflict
    PathConflict {
        path: String,
        owner_a: String,
        owner_b: String,
    },

    /// Memory limit
    MemoryLimit {
        service: String,
        max_memory_mb: u64,
        reason: String,
    },

    /// Disk space requirement
    DiskSpace {
        path: String,
        min_space_mb: u64,
    },

    /// CPU limit
    CpuLimit {
        service: String,
        max_cpu_percent: u32,
    },

    /// Network interface requirement
    NetworkInterface {
        service: String,
        interface: String,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// CUSTOM CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

/// Custom constraint with SMT formula
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomConstraint {
    /// SMT formula as string (SMT-LIB2 format)
    pub formula: String,

    /// Variables used in the formula
    pub variables: HashMap<String, VariableType>,

    /// How to interpret values from config
    pub bindings: HashMap<String, ConfigBinding>,
}

/// Variable types for SMT encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableType {
    Bool,
    Int,
    String,
    BitVec(u32),  // Bit vector of given width
}

/// Binding from NixOS config to SMT variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigBinding {
    /// Path in NixOS config (e.g., "services.nginx.enable")
    pub config_path: String,

    /// How to extract value
    pub extractor: ValueExtractor,
}

/// How to extract a value from config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueExtractor {
    /// Direct boolean value
    BoolValue,

    /// Integer value
    IntValue,

    /// String value
    StringValue,

    /// Check if value matches pattern
    Matches(String),

    /// Check if list contains value
    Contains(String),

    /// Length of list
    ListLength,
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTRAINT SET
// ═══════════════════════════════════════════════════════════════════════════

/// A collection of constraints for verification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintSet {
    /// All constraints
    constraints: Vec<Constraint>,

    /// Constraint categories for filtering
    categories: HashMap<String, HashSet<String>>,

    /// Quick lookup by ID
    by_id: HashMap<String, usize>,
}

impl ConstraintSet {
    /// Create an empty constraint set
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a constraint
    pub fn add(&mut self, constraint: Constraint) {
        let id = constraint.id.clone();
        let idx = self.constraints.len();

        // Add to category index
        let category = Self::extract_category(&constraint);
        self.categories
            .entry(category)
            .or_default()
            .insert(id.clone());

        // Add to ID index
        self.by_id.insert(id, idx);

        // Add constraint
        self.constraints.push(constraint);
    }

    /// Get constraint by ID
    pub fn get(&self, id: &str) -> Option<&Constraint> {
        self.by_id.get(id).map(|&idx| &self.constraints[idx])
    }

    /// Get all constraints
    pub fn all(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Get constraints by category
    pub fn by_category(&self, category: &str) -> Vec<&Constraint> {
        self.categories
            .get(category)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Filter constraints by severity
    pub fn by_severity(&self, min_severity: ConstraintSeverity) -> Vec<&Constraint> {
        self.constraints
            .iter()
            .filter(|c| c.severity >= min_severity)
            .collect()
    }

    /// Get number of constraints
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Extract category from constraint
    fn extract_category(constraint: &Constraint) -> String {
        match &constraint.kind {
            ConstraintKind::Service(_) => "service".to_string(),
            ConstraintKind::Package(_) => "package".to_string(),
            ConstraintKind::Security(_) => "security".to_string(),
            ConstraintKind::Resource(_) => "resource".to_string(),
            ConstraintKind::Custom(_) => "custom".to_string(),
        }
    }

    /// Create a standard NixOS constraint set with common constraints
    pub fn nixos_standard() -> Self {
        let mut set = Self::new();

        // Port conflict constraints for common services
        set.add(Constraint {
            id: "port-80-conflict".to_string(),
            description: "Only one service can bind port 80 (HTTP)".to_string(),
            kind: ConstraintKind::Service(ServiceConstraint::PortConflict {
                service_a: "nginx".to_string(),
                service_b: "apache".to_string(),
                port: 80,
                protocol: "tcp".to_string(),
            }),
            severity: ConstraintSeverity::Error,
            semantic: None,
            source: "services.nginx OR services.httpd".to_string(),
        });

        set.add(Constraint {
            id: "port-443-conflict".to_string(),
            description: "Only one service can bind port 443 (HTTPS)".to_string(),
            kind: ConstraintKind::Service(ServiceConstraint::PortConflict {
                service_a: "nginx".to_string(),
                service_b: "apache".to_string(),
                port: 443,
                protocol: "tcp".to_string(),
            }),
            severity: ConstraintSeverity::Error,
            semantic: None,
            source: "services.nginx OR services.httpd".to_string(),
        });

        // Security constraints
        set.add(Constraint {
            id: "ssh-permit-root-login".to_string(),
            description: "SSH should not permit root login".to_string(),
            kind: ConstraintKind::Security(SecurityConstraint::InsecureOption {
                option_path: "services.openssh.settings.PermitRootLogin".to_string(),
                reason: "Allowing root login via SSH is a security risk".to_string(),
                secure_alternative: Some("Use a regular user with sudo".to_string()),
            }),
            severity: ConstraintSeverity::Warning,
            semantic: None,
            source: "services.openssh".to_string(),
        });

        set.add(Constraint {
            id: "ssh-password-auth".to_string(),
            description: "SSH should use key-based authentication".to_string(),
            kind: ConstraintKind::Security(SecurityConstraint::InsecureOption {
                option_path: "services.openssh.settings.PasswordAuthentication".to_string(),
                reason: "Password authentication is vulnerable to brute force".to_string(),
                secure_alternative: Some("Use SSH keys".to_string()),
            }),
            severity: ConstraintSeverity::Warning,
            semantic: None,
            source: "services.openssh".to_string(),
        });

        set.add(Constraint {
            id: "firewall-enabled".to_string(),
            description: "Firewall should be enabled".to_string(),
            kind: ConstraintKind::Security(SecurityConstraint::InsecureOption {
                option_path: "networking.firewall.enable".to_string(),
                reason: "Disabling firewall exposes all services".to_string(),
                secure_alternative: None,
            }),
            severity: ConstraintSeverity::Critical,
            semantic: None,
            source: "networking.firewall".to_string(),
        });

        // Service dependencies
        set.add(Constraint {
            id: "postgresql-required-by-services".to_string(),
            description: "PostgreSQL must be enabled for database services".to_string(),
            kind: ConstraintKind::Service(ServiceConstraint::RequiresService {
                dependent: "services.nextcloud".to_string(),
                dependency: "services.postgresql".to_string(),
            }),
            severity: ConstraintSeverity::Error,
            semantic: None,
            source: "services.nextcloud".to_string(),
        });

        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_set_creation() {
        let set = ConstraintSet::new();
        assert!(set.is_empty());
    }

    #[test]
    fn test_add_constraint() {
        let mut set = ConstraintSet::new();
        set.add(Constraint {
            id: "test-1".to_string(),
            description: "Test constraint".to_string(),
            kind: ConstraintKind::Security(SecurityConstraint::NonRootService {
                service: "test".to_string(),
            }),
            severity: ConstraintSeverity::Warning,
            semantic: None,
            source: "test".to_string(),
        });

        assert_eq!(set.len(), 1);
        assert!(set.get("test-1").is_some());
    }

    #[test]
    fn test_nixos_standard() {
        let set = ConstraintSet::nixos_standard();
        assert!(!set.is_empty());

        // Should have security constraints
        let security = set.by_category("security");
        assert!(!security.is_empty());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ConstraintSeverity::Critical > ConstraintSeverity::Error);
        assert!(ConstraintSeverity::Error > ConstraintSeverity::Warning);
        assert!(ConstraintSeverity::Warning > ConstraintSeverity::Info);
    }
}
