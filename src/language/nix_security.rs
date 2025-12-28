//! # NixOS Security Kernel - Secret Detection & Redaction
//!
//! **Purpose**: Protect sensitive data in NixOS operations by:
//! 1. Detecting secrets in commands and output
//! 2. Redacting sensitive values before display
//! 3. Providing safe alternatives for dangerous operations
//! 4. Audit logging for security-relevant actions
//!
//! ## Secret Detection Patterns
//!
//! - API keys, tokens, passwords
//! - SSH keys and certificates
//! - AWS/GCP/Azure credentials
//! - GitHub/GitLab tokens
//! - JWT tokens and bearer tokens
//! - Database connection strings
//! - Private key material
//!
//! ## Usage
//!
//! ```rust
//! use symthaea::language::nix_security::{SecurityKernel, SecurityConfig};
//!
//! let kernel = SecurityKernel::new(SecurityConfig::default());
//!
//! // Redact secrets from output
//! let safe_output = kernel.redact_secrets("API_KEY=sk_live_abc123xyz");
//! assert_eq!(safe_output, "API_KEY=[REDACTED:api_key]");
//!
//! // Check if command contains secrets
//! let result = kernel.analyze_command("curl -H 'Authorization: Bearer token123'");
//! assert!(result.contains_secrets);
//! ```

use std::time::Instant;
use regex::Regex;
use serde::{Serialize, Deserialize};

// =============================================================================
// SECRET PATTERNS
// =============================================================================

/// Categories of secrets we detect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecretCategory {
    /// API keys (generic)
    ApiKey,
    /// AWS credentials
    AwsCredential,
    /// GCP credentials
    GcpCredential,
    /// Azure credentials
    AzureCredential,
    /// GitHub/GitLab tokens
    GitToken,
    /// SSH private keys
    SshKey,
    /// SSL/TLS certificates
    Certificate,
    /// JWT/Bearer tokens
    BearerToken,
    /// Database connection strings
    DatabaseUrl,
    /// Generic passwords
    Password,
    /// Private key material (generic)
    PrivateKey,
    /// Nix-specific secrets (age, sops)
    NixSecret,
    /// Bitwarden secrets
    BitwardenSecret,
    /// Environment variable secrets
    EnvSecret,
}

impl SecretCategory {
    /// Get display name for this category
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::ApiKey => "API Key",
            Self::AwsCredential => "AWS Credential",
            Self::GcpCredential => "GCP Credential",
            Self::AzureCredential => "Azure Credential",
            Self::GitToken => "Git Token",
            Self::SshKey => "SSH Key",
            Self::Certificate => "Certificate",
            Self::BearerToken => "Bearer Token",
            Self::DatabaseUrl => "Database URL",
            Self::Password => "Password",
            Self::PrivateKey => "Private Key",
            Self::NixSecret => "Nix Secret",
            Self::BitwardenSecret => "Bitwarden Secret",
            Self::EnvSecret => "Environment Secret",
        }
    }

    /// Get redaction placeholder for this category
    pub fn redaction_placeholder(&self) -> &'static str {
        match self {
            Self::ApiKey => "[REDACTED:api_key]",
            Self::AwsCredential => "[REDACTED:aws_cred]",
            Self::GcpCredential => "[REDACTED:gcp_cred]",
            Self::AzureCredential => "[REDACTED:azure_cred]",
            Self::GitToken => "[REDACTED:git_token]",
            Self::SshKey => "[REDACTED:ssh_key]",
            Self::Certificate => "[REDACTED:certificate]",
            Self::BearerToken => "[REDACTED:bearer_token]",
            Self::DatabaseUrl => "[REDACTED:database_url]",
            Self::Password => "[REDACTED:password]",
            Self::PrivateKey => "[REDACTED:private_key]",
            Self::NixSecret => "[REDACTED:nix_secret]",
            Self::BitwardenSecret => "[REDACTED:bws_secret]",
            Self::EnvSecret => "[REDACTED:env_secret]",
        }
    }
}

/// A detected secret pattern
#[derive(Debug, Clone)]
pub struct SecretPattern {
    /// Category of secret
    pub category: SecretCategory,
    /// Regex pattern to match
    pattern: Regex,
    /// Description of what this detects
    pub description: &'static str,
    /// Risk level (1-10)
    pub risk_level: u8,
}

impl SecretPattern {
    fn new(category: SecretCategory, pattern: &str, description: &'static str, risk_level: u8) -> Self {
        Self {
            category,
            pattern: Regex::new(pattern).expect("Invalid secret pattern regex"),
            description,
            risk_level,
        }
    }
}

// =============================================================================
// SECURITY CONFIG
// =============================================================================

/// Configuration for the security kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable secret redaction in output
    pub redact_secrets: bool,
    /// Enable audit logging
    pub audit_logging: bool,
    /// Minimum risk level to log (1-10)
    pub min_audit_risk: u8,
    /// Additional patterns to detect (regex strings)
    pub custom_patterns: Vec<String>,
    /// Whitelisted values (won't be redacted)
    pub whitelist: Vec<String>,
    /// Log file path for audit trail
    pub audit_log_path: Option<String>,
    /// Maximum length before truncating secrets in logs
    pub max_secret_preview: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            redact_secrets: true,
            audit_logging: true,
            min_audit_risk: 5,
            custom_patterns: Vec::new(),
            whitelist: Vec::new(),
            audit_log_path: None,
            max_secret_preview: 8,
        }
    }
}

// =============================================================================
// DETECTED SECRET
// =============================================================================

/// A secret detected in input
#[derive(Debug, Clone)]
pub struct DetectedSecret {
    /// Category of secret
    pub category: SecretCategory,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
    /// Preview (first few chars, redacted)
    pub preview: String,
    /// Risk level
    pub risk_level: u8,
    /// Pattern description
    pub description: &'static str,
}

// =============================================================================
// ANALYSIS RESULT
// =============================================================================

/// Result of analyzing text for secrets
#[derive(Debug, Clone)]
pub struct SecurityAnalysis {
    /// Whether secrets were found
    pub contains_secrets: bool,
    /// Detected secrets
    pub secrets: Vec<DetectedSecret>,
    /// Overall risk level (max of all detected)
    pub risk_level: u8,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Analysis timestamp
    pub timestamp: Instant,
}

impl SecurityAnalysis {
    /// Create empty analysis (no secrets found)
    pub fn clean() -> Self {
        Self {
            contains_secrets: false,
            secrets: Vec::new(),
            risk_level: 0,
            recommendations: Vec::new(),
            timestamp: Instant::now(),
        }
    }
}

// =============================================================================
// AUDIT EVENT
// =============================================================================

/// An audit log event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Event type
    pub event_type: AuditEventType,
    /// Description
    pub description: String,
    /// Risk level
    pub risk_level: u8,
    /// Categories involved
    pub categories: Vec<SecretCategory>,
    /// Timestamp (Unix epoch)
    pub timestamp_secs: u64,
    /// Context (command, file, etc.)
    pub context: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AuditEventType {
    SecretDetected,
    SecretRedacted,
    DangerousCommand,
    SafeAlternative,
    WhitelistOverride,
    CustomPatternMatch,
}

// =============================================================================
// SECURITY KERNEL
// =============================================================================

/// The main security kernel for NixOS operations
pub struct SecurityKernel {
    /// Configuration
    config: SecurityConfig,
    /// Built-in secret patterns
    patterns: Vec<SecretPattern>,
    /// Custom patterns
    custom_patterns: Vec<SecretPattern>,
    /// Audit log (in-memory, optionally persisted)
    audit_log: Vec<AuditEvent>,
    /// Statistics
    stats: SecurityStats,
}

#[derive(Debug, Clone, Default)]
pub struct SecurityStats {
    pub secrets_detected: u64,
    pub secrets_redacted: u64,
    pub commands_analyzed: u64,
    pub dangerous_commands_blocked: u64,
    pub audit_events: u64,
}

impl SecurityKernel {
    /// Create a new security kernel with the given config
    pub fn new(config: SecurityConfig) -> Self {
        let patterns = Self::build_patterns();
        let custom_patterns = config.custom_patterns.iter()
            .filter_map(|p| {
                Regex::new(p).ok().map(|_| {
                    SecretPattern::new(SecretCategory::EnvSecret, p, "Custom pattern", 5)
                })
            })
            .collect();

        Self {
            config,
            patterns,
            custom_patterns,
            audit_log: Vec::new(),
            stats: SecurityStats::default(),
        }
    }

    /// Build the default secret detection patterns
    fn build_patterns() -> Vec<SecretPattern> {
        vec![
            // API Keys (generic)
            SecretPattern::new(
                SecretCategory::ApiKey,
                r#"(?i)(api[_-]?key|apikey)\s*[=:]\s*['"]?([a-zA-Z0-9_-]{20,})['"]?"#,
                "Generic API key pattern",
                7,
            ),

            // AWS Credentials
            SecretPattern::new(
                SecretCategory::AwsCredential,
                r#"(?i)(aws[_-]?access[_-]?key[_-]?id|aws[_-]?secret[_-]?access[_-]?key)\s*[=:]\s*['"]?([A-Za-z0-9/+=]{20,})['"]?"#,
                "AWS access key or secret",
                9,
            ),
            SecretPattern::new(
                SecretCategory::AwsCredential,
                r"AKIA[0-9A-Z]{16}",
                "AWS Access Key ID",
                9,
            ),

            // GCP Credentials
            SecretPattern::new(
                SecretCategory::GcpCredential,
                r#"(?i)(google[_-]?api[_-]?key|gcp[_-]?key|gcloud[_-]?key)\s*[=:]\s*['"]?([a-zA-Z0-9_-]{30,})['"]?"#,
                "Google Cloud API key",
                9,
            ),

            // Azure Credentials
            SecretPattern::new(
                SecretCategory::AzureCredential,
                r#"(?i)(azure[_-]?storage[_-]?key|azure[_-]?connection[_-]?string)\s*[=:]\s*['"]?([a-zA-Z0-9+/=]{40,})['"]?"#,
                "Azure credential",
                9,
            ),

            // GitHub/GitLab Tokens
            SecretPattern::new(
                SecretCategory::GitToken,
                r"gh[pousr]_[A-Za-z0-9_]{36,}",
                "GitHub personal access token",
                8,
            ),
            SecretPattern::new(
                SecretCategory::GitToken,
                r"glpat-[A-Za-z0-9-_]{20,}",
                "GitLab personal access token",
                8,
            ),

            // SSH Keys
            SecretPattern::new(
                SecretCategory::SshKey,
                r"-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----",
                "SSH private key header",
                10,
            ),
            SecretPattern::new(
                SecretCategory::SshKey,
                r"ssh-(rsa|ed25519|ecdsa)\s+[A-Za-z0-9+/]+={0,3}",
                "SSH public key (may indicate private key nearby)",
                4,
            ),

            // Bearer Tokens
            SecretPattern::new(
                SecretCategory::BearerToken,
                r#"(?i)(bearer|authorization)\s*[=:]\s*['"]?bearer\s+([a-zA-Z0-9_.-]+)['"]?"#,
                "Bearer token in header",
                8,
            ),
            SecretPattern::new(
                SecretCategory::BearerToken,
                r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
                "JWT token",
                7,
            ),

            // Database URLs
            SecretPattern::new(
                SecretCategory::DatabaseUrl,
                r"(?i)(postgres|postgresql|mysql|mongodb|redis)://[^:]+:[^@]+@[^/]+",
                "Database connection string with credentials",
                9,
            ),

            // Passwords
            SecretPattern::new(
                SecretCategory::Password,
                r#"(?i)(password|passwd|pwd)\s*[=:]\s*['"]?([^\s'"]{6,})['"]?"#,
                "Password in plaintext",
                8,
            ),

            // Private Keys
            SecretPattern::new(
                SecretCategory::PrivateKey,
                r"-----BEGIN (ENCRYPTED )?PRIVATE KEY-----",
                "PEM private key",
                10,
            ),
            SecretPattern::new(
                SecretCategory::PrivateKey,
                r"-----BEGIN PGP PRIVATE KEY BLOCK-----",
                "PGP private key",
                10,
            ),

            // Nix-specific secrets
            SecretPattern::new(
                SecretCategory::NixSecret,
                r"age1[a-z0-9]{58}",
                "Age encryption key",
                8,
            ),
            SecretPattern::new(
                SecretCategory::NixSecret,
                r#"(?i)sops_age_key\s*[=:]\s*['"]?([^\s'"]+)['"]?"#,
                "SOPS age key",
                9,
            ),

            // Bitwarden
            SecretPattern::new(
                SecretCategory::BitwardenSecret,
                r#"(?i)(bws|bitwarden)[_-]?(secret|token|key)\s*[=:]\s*['"]?([a-zA-Z0-9_-]{20,})['"]?"#,
                "Bitwarden secret",
                8,
            ),

            // Environment variables with sensitive names
            SecretPattern::new(
                SecretCategory::EnvSecret,
                r#"(?i)(secret|token|credential|private|auth)[_-]?key\s*[=:]\s*['"]?([^\s'"]{10,})['"]?"#,
                "Environment variable with sensitive name",
                6,
            ),
        ]
    }

    /// Analyze text for secrets
    pub fn analyze(&mut self, text: &str) -> SecurityAnalysis {
        self.stats.commands_analyzed += 1;

        let mut secrets = Vec::new();
        let mut max_risk = 0u8;

        // Check built-in patterns
        for pattern in &self.patterns {
            for cap in pattern.pattern.find_iter(text) {
                // Skip whitelisted values
                if self.config.whitelist.iter().any(|w| cap.as_str().contains(w)) {
                    continue;
                }

                let preview = self.create_preview(cap.as_str());
                secrets.push(DetectedSecret {
                    category: pattern.category,
                    start: cap.start(),
                    end: cap.end(),
                    preview,
                    risk_level: pattern.risk_level,
                    description: pattern.description,
                });
                max_risk = max_risk.max(pattern.risk_level);
            }
        }

        // Check custom patterns
        for pattern in &self.custom_patterns {
            for cap in pattern.pattern.find_iter(text) {
                if self.config.whitelist.iter().any(|w| cap.as_str().contains(w)) {
                    continue;
                }

                let preview = self.create_preview(cap.as_str());
                secrets.push(DetectedSecret {
                    category: pattern.category,
                    start: cap.start(),
                    end: cap.end(),
                    preview,
                    risk_level: pattern.risk_level,
                    description: pattern.description,
                });
                max_risk = max_risk.max(pattern.risk_level);
            }
        }

        self.stats.secrets_detected += secrets.len() as u64;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&secrets);

        // Log if audit enabled
        if self.config.audit_logging && !secrets.is_empty() && max_risk >= self.config.min_audit_risk {
            self.log_detection(&secrets);
        }

        SecurityAnalysis {
            contains_secrets: !secrets.is_empty(),
            secrets,
            risk_level: max_risk,
            recommendations,
            timestamp: Instant::now(),
        }
    }

    /// Redact secrets from text
    pub fn redact_secrets(&mut self, text: &str) -> String {
        if !self.config.redact_secrets {
            return text.to_string();
        }

        let mut result = text.to_string();

        // Collect all matches first to avoid overlapping replacements
        let mut replacements: Vec<(usize, usize, SecretCategory)> = Vec::new();

        for pattern in &self.patterns {
            for cap in pattern.pattern.find_iter(text) {
                if self.config.whitelist.iter().any(|w| cap.as_str().contains(w)) {
                    continue;
                }
                replacements.push((cap.start(), cap.end(), pattern.category));
            }
        }

        // Sort by position (descending) to replace from end to start
        replacements.sort_by(|a, b| b.0.cmp(&a.0));

        for (start, end, category) in replacements {
            let placeholder = category.redaction_placeholder();
            result.replace_range(start..end, placeholder);
            self.stats.secrets_redacted += 1;
        }

        result
    }

    /// Check if a command is dangerous
    pub fn is_dangerous_command(&self, command: &str) -> Option<&'static str> {
        let cmd_lower = command.to_lowercase();

        // Simple patterns (substring matches)
        let simple_patterns = [
            ("rm -rf /", "This would delete your entire system!"),
            ("rm -rf ~", "This would delete your home directory!"),
            ("chmod 777", "Setting world-writable permissions is insecure"),
            ("chmod -R 777", "Recursively setting world-writable permissions is very insecure"),
            ("> /etc/", "Overwriting system files directly is dangerous"),
            ("nixos-rebuild switch --impure", "Impure builds may not be reproducible"),
            ("nix-env -i", "nix-env is deprecated, use nix profile or flakes"),
        ];

        for (pattern, reason) in simple_patterns {
            if cmd_lower.contains(&pattern.to_lowercase()) {
                return Some(reason);
            }
        }

        // Compound patterns (multiple conditions)
        // curl/wget piped to shell
        if (cmd_lower.contains("curl") || cmd_lower.contains("wget"))
            && (cmd_lower.contains("| sh") || cmd_lower.contains("| bash") || cmd_lower.contains("|sh") || cmd_lower.contains("|bash"))
        {
            return Some("Piping untrusted download to shell is dangerous");
        }

        None
    }

    /// Suggest a safe alternative for a dangerous command
    pub fn suggest_safe_alternative(&self, command: &str) -> Option<String> {
        let cmd_lower = command.to_lowercase();

        // nix-env alternatives
        if cmd_lower.contains("nix-env -i") {
            return Some("Use 'nix profile install' or add to configuration.nix/home.nix instead".to_string());
        }

        // Impure build alternatives
        if cmd_lower.contains("--impure") {
            return Some("Consider using flakes with locked inputs for reproducibility".to_string());
        }

        // Curl piping alternatives
        if cmd_lower.contains("curl") && (cmd_lower.contains("| sh") || cmd_lower.contains("| bash")) {
            return Some("Download the script first, review it, then execute".to_string());
        }

        None
    }

    /// Create a preview of a secret (first few chars + redaction)
    fn create_preview(&self, secret: &str) -> String {
        let preview_len = self.config.max_secret_preview.min(secret.len());
        let preview: String = secret.chars().take(preview_len).collect();
        format!("{}...", preview)
    }

    /// Generate recommendations based on detected secrets
    fn generate_recommendations(&self, secrets: &[DetectedSecret]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for secret in secrets {
            match secret.category {
                SecretCategory::Password => {
                    recommendations.push("Use a secret manager like sops-nix or agenix for passwords".to_string());
                }
                SecretCategory::AwsCredential | SecretCategory::GcpCredential | SecretCategory::AzureCredential => {
                    recommendations.push("Store cloud credentials in environment variables or secret manager, not in code".to_string());
                }
                SecretCategory::SshKey | SecretCategory::PrivateKey => {
                    recommendations.push("Never commit private keys. Use ssh-agent or a secret manager".to_string());
                }
                SecretCategory::GitToken => {
                    recommendations.push("Use SSH keys or credential helpers instead of hardcoded tokens".to_string());
                }
                SecretCategory::DatabaseUrl => {
                    recommendations.push("Store database credentials separately and inject at runtime".to_string());
                }
                SecretCategory::NixSecret => {
                    recommendations.push("Good: using age/sops for secrets. Ensure keys are properly managed".to_string());
                }
                _ => {}
            }
        }

        // Deduplicate
        recommendations.sort();
        recommendations.dedup();
        recommendations
    }

    /// Log a secret detection event
    fn log_detection(&mut self, secrets: &[DetectedSecret]) {
        let categories: Vec<SecretCategory> = secrets.iter()
            .map(|s| s.category)
            .collect();

        let max_risk = secrets.iter().map(|s| s.risk_level).max().unwrap_or(0);

        let event = AuditEvent {
            event_type: AuditEventType::SecretDetected,
            description: format!("Detected {} secret(s)", secrets.len()),
            risk_level: max_risk,
            categories,
            timestamp_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            context: None,
        };

        self.audit_log.push(event);
        self.stats.audit_events += 1;
    }

    /// Get statistics
    pub fn stats(&self) -> &SecurityStats {
        &self.stats
    }

    /// Get audit log
    pub fn audit_log(&self) -> &[AuditEvent] {
        &self.audit_log
    }

    /// Clear audit log
    pub fn clear_audit_log(&mut self) {
        self.audit_log.clear();
    }
}

impl Default for SecurityKernel {
    fn default() -> Self {
        Self::new(SecurityConfig::default())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = SecurityKernel::default();
        assert!(!kernel.patterns.is_empty());
        assert_eq!(kernel.stats.commands_analyzed, 0);
    }

    #[test]
    fn test_detect_api_key() {
        let mut kernel = SecurityKernel::default();
        let analysis = kernel.analyze("API_KEY=sk_live_abc123xyz9876543210000");

        assert!(analysis.contains_secrets);
        assert!(!analysis.secrets.is_empty());
        assert_eq!(analysis.secrets[0].category, SecretCategory::ApiKey);
    }

    #[test]
    fn test_detect_aws_key() {
        let mut kernel = SecurityKernel::default();
        let analysis = kernel.analyze("Found key: AKIAIOSFODNN7EXAMPLE");

        assert!(analysis.contains_secrets);
        assert_eq!(analysis.secrets[0].category, SecretCategory::AwsCredential);
    }

    #[test]
    fn test_detect_github_token() {
        let mut kernel = SecurityKernel::default();
        let analysis = kernel.analyze("GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");

        assert!(analysis.contains_secrets);
        assert_eq!(analysis.secrets[0].category, SecretCategory::GitToken);
    }

    #[test]
    fn test_detect_jwt() {
        let mut kernel = SecurityKernel::default();
        let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
        let analysis = kernel.analyze(jwt);

        assert!(analysis.contains_secrets);
        assert_eq!(analysis.secrets[0].category, SecretCategory::BearerToken);
    }

    #[test]
    fn test_detect_ssh_key() {
        let mut kernel = SecurityKernel::default();
        let analysis = kernel.analyze("-----BEGIN RSA PRIVATE KEY-----");

        assert!(analysis.contains_secrets);
        assert_eq!(analysis.secrets[0].category, SecretCategory::SshKey);
        assert_eq!(analysis.risk_level, 10);
    }

    #[test]
    fn test_detect_password() {
        let mut kernel = SecurityKernel::default();
        let analysis = kernel.analyze("password=supersecret123");

        assert!(analysis.contains_secrets);
        assert_eq!(analysis.secrets[0].category, SecretCategory::Password);
    }

    #[test]
    fn test_detect_database_url() {
        let mut kernel = SecurityKernel::default();
        let analysis = kernel.analyze("DATABASE_URL=postgres://user:pass@localhost/db");

        assert!(analysis.contains_secrets);
        assert_eq!(analysis.secrets[0].category, SecretCategory::DatabaseUrl);
    }

    #[test]
    fn test_redact_secrets() {
        let mut kernel = SecurityKernel::default();
        let result = kernel.redact_secrets("API_KEY=sk_live_abc123xyz9876543210000");

        assert!(result.contains("[REDACTED:api_key]"));
        assert!(!result.contains("sk_live"));
    }

    #[test]
    fn test_no_false_positives() {
        let mut kernel = SecurityKernel::default();
        let safe_text = "This is a normal sentence about NixOS configuration.";
        let analysis = kernel.analyze(safe_text);

        assert!(!analysis.contains_secrets);
        assert!(analysis.secrets.is_empty());
    }

    #[test]
    fn test_whitelist() {
        let config = SecurityConfig {
            whitelist: vec!["test_key".to_string()],
            ..Default::default()
        };
        let mut kernel = SecurityKernel::new(config);
        let analysis = kernel.analyze("API_KEY=test_key_12345678901234567890");

        // Should be whitelisted
        assert!(!analysis.contains_secrets);
    }

    #[test]
    fn test_dangerous_command_detection() {
        let kernel = SecurityKernel::default();

        assert!(kernel.is_dangerous_command("rm -rf /").is_some());
        assert!(kernel.is_dangerous_command("curl https://example.com/script.sh | bash").is_some());
        assert!(kernel.is_dangerous_command("chmod 777 /etc").is_some());

        // Safe commands
        assert!(kernel.is_dangerous_command("nixos-rebuild switch").is_none());
        assert!(kernel.is_dangerous_command("nix build").is_none());
    }

    #[test]
    fn test_safe_alternatives() {
        let kernel = SecurityKernel::default();

        let alt = kernel.suggest_safe_alternative("nix-env -iA nixpkgs.firefox");
        assert!(alt.is_some());
        assert!(alt.unwrap().contains("nix profile"));
    }

    #[test]
    fn test_audit_logging() {
        let config = SecurityConfig {
            audit_logging: true,
            min_audit_risk: 1,
            ..Default::default()
        };
        let mut kernel = SecurityKernel::new(config);

        kernel.analyze("password=secret123");

        assert!(!kernel.audit_log.is_empty());
        assert_eq!(kernel.stats.audit_events, 1);
    }

    #[test]
    fn test_age_key_detection() {
        let mut kernel = SecurityKernel::default();
        // Valid age key format
        let analysis = kernel.analyze("AGE_KEY=age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");

        assert!(analysis.contains_secrets);
        assert_eq!(analysis.secrets[0].category, SecretCategory::NixSecret);
    }

    #[test]
    fn test_recommendations() {
        let mut kernel = SecurityKernel::default();
        let analysis = kernel.analyze("password=mysecret123");

        assert!(!analysis.recommendations.is_empty());
        assert!(analysis.recommendations[0].contains("secret manager"));
    }
}
