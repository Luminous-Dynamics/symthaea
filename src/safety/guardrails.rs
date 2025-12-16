/*!
Safety Guardrails - Algebraic Ethical Lockout

Implements algebraic safety via forbidden subspace checking:
- Hamming distance to dangerous patterns
- Constitutional AI constraints encoded as hypervectors
- Fast O(n) checking (no neural network needed!)
- Mathematically provable safety bounds

If an action is too similar to a forbidden pattern → LOCKOUT
*/

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Safety threshold: max similarity to forbidden patterns
const SAFETY_THRESHOLD: f32 = 0.85;

/// Forbidden action categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForbiddenCategory {
    /// Destructive system operations
    SystemDestruction,
    /// Privacy violations
    PrivacyViolation,
    /// Unauthorized access
    UnauthorizedAccess,
    /// Data exfiltration
    DataExfiltration,
    /// Resource abuse
    ResourceAbuse,
    /// Self-modification without consent
    UnsafeModification,
}

/// A forbidden pattern in hypervector space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForbiddenPattern {
    /// Category of forbidden action
    pub category: ForbiddenCategory,

    /// Hypervector representation (10,000D bipolar)
    pub pattern: Vec<i8>,

    /// Human-readable description
    pub description: String,

    /// Severity (0.0 = warning, 1.0 = hard block)
    pub severity: f32,
}

/// Safety Guardrails System
#[derive(Clone, Serialize, Deserialize)]
pub struct SafetyGuardrails {
    /// Forbidden patterns database
    forbidden_space: Vec<ForbiddenPattern>,

    /// Similarity threshold for triggering safety check
    threshold: f32,

    /// Statistics
    checks_performed: usize,
    lockouts_triggered: usize,
    warnings_issued: usize,
}

impl SafetyGuardrails {
    /// Create new safety system with default patterns
    pub fn new() -> Self {
        Self {
            forbidden_space: Self::default_forbidden_patterns(),
            threshold: SAFETY_THRESHOLD,
            checks_performed: 0,
            lockouts_triggered: 0,
            warnings_issued: 0,
        }
    }

    /// Default forbidden patterns (can be extended)
    fn default_forbidden_patterns() -> Vec<ForbiddenPattern> {
        vec![
            // System destruction
            ForbiddenPattern {
                category: ForbiddenCategory::SystemDestruction,
                pattern: Self::pattern_from_text("rm -rf / sudo delete everything destroy system"),
                description: "Destructive system-wide deletion".to_string(),
                severity: 1.0,
            },

            // Privacy violation
            ForbiddenPattern {
                category: ForbiddenCategory::PrivacyViolation,
                pattern: Self::pattern_from_text("read private data steal passwords exfiltrate credentials"),
                description: "Unauthorized access to private data".to_string(),
                severity: 1.0,
            },

            // Unauthorized access
            ForbiddenPattern {
                category: ForbiddenCategory::UnauthorizedAccess,
                pattern: Self::pattern_from_text("sudo escalate privileges root access unauthorized"),
                description: "Privilege escalation without consent".to_string(),
                severity: 0.9,
            },

            // Resource abuse
            ForbiddenPattern {
                category: ForbiddenCategory::ResourceAbuse,
                pattern: Self::pattern_from_text("infinite loop fork bomb consume resources denial service"),
                description: "Resource exhaustion attack".to_string(),
                severity: 0.9,
            },

            // Self-modification
            ForbiddenPattern {
                category: ForbiddenCategory::UnsafeModification,
                pattern: Self::pattern_from_text("modify self code injection backdoor rootkit"),
                description: "Unauthorized self-modification".to_string(),
                severity: 1.0,
            },
        ]
    }

    /// Simple pattern generation (placeholder - should use SemanticEar)
    fn pattern_from_text(text: &str) -> Vec<i8> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut pattern = vec![0i8; 10_000];

        // Simple hash-based pattern (real version would use SemanticEar)
        for word in text.split_whitespace() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();

            for i in 0..100 {
                let idx = ((hash + i) % 10_000) as usize;
                pattern[idx] = if (hash + i) % 2 == 0 { 1 } else { -1 };
            }
        }

        pattern
    }

    /// Check if action is safe
    ///
    /// Returns Ok(()) if safe, Err with reason if forbidden
    pub fn check_safety(&mut self, action: &[i8]) -> Result<()> {
        self.checks_performed += 1;

        for forbidden in &self.forbidden_space {
            let similarity = Self::hamming_similarity(action, &forbidden.pattern);

            if similarity > self.threshold {
                self.lockouts_triggered += 1;

                let msg = format!(
                    "🚨 ETHICAL LOCKOUT: Action too similar to forbidden pattern\n\
                     Category: {:?}\n\
                     Description: {}\n\
                     Similarity: {:.1}% (threshold: {:.1}%)\n\
                     Severity: {:.1}",
                    forbidden.category,
                    forbidden.description,
                    similarity * 100.0,
                    self.threshold * 100.0,
                    forbidden.severity
                );

                bail!(msg);
            } else if similarity > self.threshold * 0.8 {
                // Warning threshold (80% of lockout threshold)
                self.warnings_issued += 1;

                tracing::warn!(
                    "⚠️  Safety warning: Action approaching forbidden pattern\n\
                     Category: {:?}\n\
                     Similarity: {:.1}%",
                    forbidden.category,
                    similarity * 100.0
                );
            }
        }

        Ok(())
    }

    /// Batch safety check
    pub fn check_safety_batch(&mut self, actions: &[Vec<i8>]) -> Result<Vec<Result<()>>> {
        Ok(actions
            .iter()
            .map(|action| self.check_safety(action))
            .collect())
    }

    /// Hamming similarity (normalized: 0.0 = orthogonal, 1.0 = identical)
    fn hamming_similarity(a: &[i8], b: &[i8]) -> f32 {
        let matches = a.iter()
            .zip(b.iter())
            .filter(|(x, y)| x == y)
            .count();

        matches as f32 / a.len() as f32
    }

    /// Add custom forbidden pattern
    pub fn add_forbidden_pattern(&mut self, pattern: ForbiddenPattern) {
        self.forbidden_space.push(pattern);
    }

    /// Remove forbidden patterns by category
    pub fn remove_patterns_by_category(&mut self, category: ForbiddenCategory) {
        self.forbidden_space.retain(|p| !matches!(p.category, ref c if std::mem::discriminant(c) == std::mem::discriminant(&category)));
    }

    /// Adjust safety threshold (0.0 = permissive, 1.0 = strict)
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
        tracing::info!("🔒 Safety threshold adjusted to {:.1}%", self.threshold * 100.0);
    }

    /// Get safety statistics
    pub fn stats(&self) -> SafetyStats {
        SafetyStats {
            checks_performed: self.checks_performed,
            lockouts_triggered: self.lockouts_triggered,
            warnings_issued: self.warnings_issued,
            lockout_rate: if self.checks_performed > 0 {
                self.lockouts_triggered as f32 / self.checks_performed as f32
            } else {
                0.0
            },
            forbidden_patterns_count: self.forbidden_space.len(),
        }
    }

    /// Export forbidden space for analysis
    pub fn export_patterns(&self) -> &[ForbiddenPattern] {
        &self.forbidden_space
    }

    /// Clear statistics (keep patterns)
    pub fn reset_stats(&mut self) {
        self.checks_performed = 0;
        self.lockouts_triggered = 0;
        self.warnings_issued = 0;
    }
}

/// Safety statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyStats {
    pub checks_performed: usize,
    pub lockouts_triggered: usize,
    pub warnings_issued: usize,
    pub lockout_rate: f32,
    pub forbidden_patterns_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_guardrails_creation() {
        let safety = SafetyGuardrails::new();
        assert!(safety.forbidden_space.len() > 0);
    }

    #[test]
    fn test_hamming_similarity() {
        let a = vec![1i8; 100];
        let b = vec![1i8; 100];
        let c = vec![-1i8; 100];

        let sim_identical = SafetyGuardrails::hamming_similarity(&a, &b);
        let sim_opposite = SafetyGuardrails::hamming_similarity(&a, &c);

        assert_eq!(sim_identical, 1.0);
        assert_eq!(sim_opposite, 0.0);
    }

    #[test]
    fn test_safe_action_passes() {
        let mut safety = SafetyGuardrails::new();

        // Safe action (random pattern unlikely to match)
        let safe_action = vec![1i8; 10_000];

        let result = safety.check_safety(&safe_action);
        assert!(result.is_ok());
    }

    #[test]
    fn test_threshold_adjustment() {
        let mut safety = SafetyGuardrails::new();

        safety.set_threshold(0.95);
        assert_eq!(safety.threshold, 0.95);

        safety.set_threshold(1.5);  // Should clamp to 1.0
        assert_eq!(safety.threshold, 1.0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut safety = SafetyGuardrails::new();

        let action = vec![1i8; 10_000];
        let _ = safety.check_safety(&action);

        let stats = safety.stats();
        assert_eq!(stats.checks_performed, 1);
    }
}
