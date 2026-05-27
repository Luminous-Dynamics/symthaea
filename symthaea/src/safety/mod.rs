// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safety module - safety constraints and gateways
//!
//! Provides safety checking capabilities including:
//! - Fast regex-based pre-cognitive veto (AmygdalaActor)
//! - Hypervector-based forbidden subspace checking (SafetyGuardrails)
//! - Unified safety gateway interface

pub mod gateway;
pub mod level;

// Genesis Mission Challenge 26: Safety Agents
#[cfg(feature = "safety-agents")]
pub mod agent;
#[cfg(feature = "safety-agents")]
pub mod audit;
#[cfg(feature = "safety-agents")]
pub mod gate;

// Cross-domain bridges: domain outputs → Safety Agent
#[cfg(all(feature = "safety-agents", feature = "accelerator"))]
pub mod accelerator_bridge;
#[cfg(all(feature = "safety-agents", feature = "datacenter"))]
pub mod datacenter_bridge;
#[cfg(all(feature = "safety-agents", feature = "experiment-planner"))]
pub mod experiment_bridge;
#[cfg(all(feature = "safety-agents", feature = "fission-reactor"))]
pub mod fission_bridge;
#[cfg(all(feature = "safety-agents", feature = "fusion-twin"))]
pub mod fusion_bridge;
#[cfg(all(feature = "safety-agents", feature = "grid-scaling"))]
pub mod grid_bridge;
#[cfg(all(feature = "safety-agents", feature = "materials"))]
pub mod materials_bridge;
#[cfg(all(feature = "safety-agents", feature = "critical-minerals"))]
pub mod mining_bridge;
#[cfg(all(feature = "safety-agents", feature = "nuclear-forensics"))]
pub mod nuclear_bridge;
#[cfg(all(feature = "safety-agents", feature = "proliferation-safeguards"))]
pub mod safeguards_bridge;
#[cfg(all(feature = "safety-agents", feature = "strategic-materials"))]
pub mod strategic_materials_bridge;
#[cfg(all(feature = "safety-agents", feature = "threat-assessment"))]
pub mod threat_bridge;
#[cfg(all(feature = "safety-agents", feature = "water-prediction"))]
pub mod water_bridge;

// Re-export key types — SafetyLevel is always available (not feature-gated)
pub use gateway::{SafetyCheck, SafetyDecision, SafetyGateway};
pub use level::SafetyLevel;

#[cfg(feature = "safety-agents")]
pub use agent::{
    SafetyAgent, SafetyAgentConfig, SafetyAssessment, SafetyMetrics, SafetyOverrideEntry,
};
#[cfg(feature = "safety-agents")]
pub use audit::SafetyAuditReport;
#[cfg(feature = "safety-agents")]
pub use gate::{SafetyGateResult, consciousness_gate, safety_gate};

#[cfg(all(feature = "safety-agents", feature = "accelerator"))]
pub use accelerator_bridge::AcceleratorSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "datacenter"))]
pub use datacenter_bridge::DatacenterSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "experiment-planner"))]
pub use experiment_bridge::ExperimentSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "fission-reactor"))]
pub use fission_bridge::FissionSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "fusion-twin"))]
pub use fusion_bridge::FusionSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "grid-scaling"))]
pub use grid_bridge::GridSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "materials"))]
pub use materials_bridge::MaterialSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "critical-minerals"))]
pub use mining_bridge::MiningSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "nuclear-forensics"))]
pub use nuclear_bridge::NuclearSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "proliferation-safeguards"))]
pub use safeguards_bridge::SafeguardsSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "strategic-materials"))]
pub use strategic_materials_bridge::StrategicMaterialsSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "threat-assessment"))]
pub use threat_bridge::ThreatSafetyAdapter;
#[cfg(all(feature = "safety-agents", feature = "water-prediction"))]
pub use water_bridge::WaterSafetyAdapter;

/// Categories of forbidden content/actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForbiddenCategory {
    /// Dangerous system commands
    DangerousCommand,
    /// Harmful content
    HarmfulContent,
    /// Privacy violation
    PrivacyViolation,
    /// Security risk
    SecurityRisk,
    /// Unethical request
    UnethicalRequest,
}

impl ForbiddenCategory {
    /// Get all categories for prototype initialization
    fn all() -> &'static [ForbiddenCategory] {
        &[
            ForbiddenCategory::DangerousCommand,
            ForbiddenCategory::HarmfulContent,
            ForbiddenCategory::PrivacyViolation,
            ForbiddenCategory::SecurityRisk,
            ForbiddenCategory::UnethicalRequest,
        ]
    }

    /// Get a deterministic seed for this category's prototype vector
    fn seed(&self) -> u64 {
        match self {
            ForbiddenCategory::DangerousCommand => 0xDEAD_0001,
            ForbiddenCategory::HarmfulContent => 0xDEAD_0002,
            ForbiddenCategory::PrivacyViolation => 0xDEAD_0003,
            ForbiddenCategory::SecurityRisk => 0xDEAD_0004,
            ForbiddenCategory::UnethicalRequest => 0xDEAD_0005,
        }
    }
}

/// Fast regex-based pre-cognitive safety veto
///
/// Acts like the amygdala - fast, pattern-matching safety check
/// before deeper processing occurs.
#[derive(Debug)]
pub struct AmygdalaActor {
    /// Dangerous command patterns
    dangerous_patterns: Vec<regex::Regex>,
    /// Number of patterns that failed to compile
    compile_failures: usize,
}

impl AmygdalaActor {
    /// Create a new AmygdalaActor with default dangerous patterns
    pub fn new() -> Self {
        let patterns = vec![
            // Destructive commands — hardened to catch flag-interleaved variants
            // Each pattern handles arbitrary flags (e.g. --no-preserve-root) between command and target
            r"rm\s+(-\S+\s+)*/",                // rm with any flags before path /
            r"dd\s+(\S+=\S+\s+)*if=.*of=/dev/", // dd with any key=value params
            r"mkfs\.",                          // mkfs.* (always dangerous)
            r":\(\)\{\s*:\|:&\s*\};:",          // Fork bomb
            r"chmod\s+(-\S+\s+)*777\s+/",       // chmod 777 with any flags
            r">\s*/dev/sd",                     // redirect to raw device
        ];

        let mut compile_failures = 0;
        let dangerous_patterns = patterns
            .into_iter()
            .filter_map(|p| match regex::Regex::new(p) {
                Ok(re) => Some(re),
                Err(e) => {
                    eprintln!("[safety] Failed to compile regex pattern '{p}': {e}");
                    compile_failures += 1;
                    None
                }
            })
            .collect();

        Self {
            dangerous_patterns,
            compile_failures,
        }
    }

    /// Scan text for dangerous patterns
    /// Returns Some(message) if dangerous, None if safe
    pub fn scan(&self, text: &str) -> Option<String> {
        for pattern in &self.dangerous_patterns {
            if pattern.is_match(text) {
                return Some(format!(
                    "Blocked: Dangerous pattern detected matching '{}'",
                    pattern.as_str()
                ));
            }
        }
        None
    }

    /// Number of regex patterns that failed to compile
    pub fn compile_failures(&self) -> usize {
        self.compile_failures
    }

    /// Number of active dangerous patterns
    pub fn pattern_count(&self) -> usize {
        self.dangerous_patterns.len()
    }
}

impl Default for AmygdalaActor {
    fn default() -> Self {
        Self::new()
    }
}

/// A forbidden-category prototype vector for HDC similarity checking
#[derive(Debug, Clone)]
struct ForbiddenPrototype {
    category: ForbiddenCategory,
    /// Prototype vector (f32 values)
    vector: Vec<f32>,
}

/// Hypervector-based safety guardrails
///
/// Uses hyperdimensional computing to check if content falls
/// within forbidden semantic subspaces. Each forbidden category
/// has a prototype vector; input vectors with high cosine similarity
/// to any prototype are flagged.
#[derive(Debug)]
pub struct SafetyGuardrails {
    /// Dimension of hypervectors
    dimension: usize,
    /// Whether guardrails are active
    active: bool,
    /// Prototype vectors for each forbidden category
    prototypes: Vec<ForbiddenPrototype>,
    /// Similarity threshold for flagging (0.0-1.0)
    threshold: f32,
}

impl SafetyGuardrails {
    /// Create new safety guardrails with default forbidden-subspace prototypes
    pub fn new() -> Self {
        Self::with_dimension(512)
    }

    /// Create guardrails with a specific dimension
    pub fn with_dimension(dimension: usize) -> Self {
        let prototypes = ForbiddenCategory::all()
            .iter()
            .map(|&cat| ForbiddenPrototype {
                category: cat,
                vector: Self::generate_prototype(dimension, cat.seed()),
            })
            .collect();

        Self {
            dimension,
            active: true,
            prototypes,
            threshold: 0.85,
        }
    }

    /// Generate a deterministic prototype vector from a seed
    fn generate_prototype(dim: usize, seed: u64) -> Vec<f32> {
        let mut values = Vec::with_capacity(dim);
        let mut state = seed;

        for _ in 0..dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let normalized = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
            values.push(normalized);
        }

        values
    }

    /// Cosine similarity between two f32 vectors (delegates to shared implementation)
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        symthaea_core::math::cosine_similarity_f32(a, b).clamp(-1.0, 1.0)
    }

    /// Check if a hypervector falls within forbidden subspace
    ///
    /// Compares input HV against each forbidden category prototype using
    /// cosine similarity. Returns the first category exceeding the threshold.
    pub fn check(&self, hv: &[f32]) -> Option<ForbiddenCategory> {
        if !self.active {
            return None;
        }

        if hv.len() != self.dimension {
            return None;
        }

        for proto in &self.prototypes {
            let sim = Self::cosine_similarity(hv, &proto.vector);
            if sim > self.threshold {
                return Some(proto.category);
            }
        }

        None
    }

    /// Check with detailed result including similarity scores
    pub fn check_detailed(&self, hv: &[f32]) -> Vec<(ForbiddenCategory, f32)> {
        if !self.active || hv.len() != self.dimension {
            return Vec::new();
        }

        self.prototypes
            .iter()
            .map(|proto| {
                let sim = Self::cosine_similarity(hv, &proto.vector);
                (proto.category, sim)
            })
            .collect()
    }

    /// Get the prototype vector for a specific category (for training/testing)
    pub fn prototype(&self, category: ForbiddenCategory) -> Option<&[f32]> {
        self.prototypes
            .iter()
            .find(|p| p.category == category)
            .map(|p| p.vector.as_slice())
    }

    /// Set the similarity threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Enable or disable guardrails
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl Default for SafetyGuardrails {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amygdala_blocks_rm_rf() {
        let amygdala = AmygdalaActor::new();
        assert!(amygdala.scan("rm -rf /").is_some());
    }

    #[test]
    fn test_amygdala_allows_safe_text() {
        let amygdala = AmygdalaActor::new();
        assert!(amygdala.scan("hello world").is_none());
    }

    #[test]
    fn test_amygdala_reports_compile_failures() {
        let amygdala = AmygdalaActor::new();
        // All default patterns should compile successfully
        assert_eq!(amygdala.compile_failures(), 0);
    }

    #[test]
    fn test_guardrails_detects_prototype_match() {
        let guardrails = SafetyGuardrails::new();
        // A vector identical to a prototype should be flagged
        let proto = guardrails
            .prototype(ForbiddenCategory::DangerousCommand)
            .unwrap()
            .to_vec();
        assert_eq!(
            guardrails.check(&proto),
            Some(ForbiddenCategory::DangerousCommand)
        );
    }

    #[test]
    fn test_guardrails_allows_random_vector() {
        let guardrails = SafetyGuardrails::new();
        // A random vector should NOT match any prototype (in high dimensions)
        let random_vec: Vec<f32> = (0..512)
            .map(|i| {
                let mut state = 0xBEEF_CAFE_u64.wrapping_add(i as u64);
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        assert_eq!(guardrails.check(&random_vec), None);
    }

    #[test]
    fn test_guardrails_inactive_allows_all() {
        let mut guardrails = SafetyGuardrails::new();
        guardrails.set_active(false);
        let proto = guardrails
            .prototype(ForbiddenCategory::DangerousCommand)
            .unwrap()
            .to_vec();
        assert_eq!(guardrails.check(&proto), None);
    }

    #[test]
    fn test_guardrails_wrong_dimension_allowed() {
        let guardrails = SafetyGuardrails::new();
        // Wrong dimension should not panic, just return None
        let wrong_dim = vec![1.0; 100];
        assert_eq!(guardrails.check(&wrong_dim), None);
    }

    #[test]
    fn test_guardrails_detailed_check() {
        let guardrails = SafetyGuardrails::new();
        let proto = guardrails
            .prototype(ForbiddenCategory::SecurityRisk)
            .unwrap()
            .to_vec();
        let results = guardrails.check_detailed(&proto);

        // SecurityRisk prototype should have high similarity with itself
        let security_sim = results
            .iter()
            .find(|(cat, _)| *cat == ForbiddenCategory::SecurityRisk)
            .map(|(_, sim)| *sim)
            .unwrap();
        assert!(
            security_sim > 0.99,
            "Self-similarity should be ~1.0, got {}",
            security_sim
        );

        // Other categories should have low similarity
        for (cat, sim) in &results {
            if *cat != ForbiddenCategory::SecurityRisk {
                assert!(
                    *sim < 0.5,
                    "Cross-category similarity should be low, got {} for {:?}",
                    sim,
                    cat
                );
            }
        }
    }

    #[test]
    fn test_forbidden_categories_have_distinct_prototypes() {
        let guardrails = SafetyGuardrails::new();
        let categories = ForbiddenCategory::all();

        for i in 0..categories.len() {
            for j in (i + 1)..categories.len() {
                let a = guardrails.prototype(categories[i]).unwrap();
                let b = guardrails.prototype(categories[j]).unwrap();
                let sim = SafetyGuardrails::cosine_similarity(a, b);
                assert!(
                    sim < 0.5,
                    "Categories {:?} and {:?} have too-similar prototypes (sim={})",
                    categories[i],
                    categories[j],
                    sim
                );
            }
        }
    }

    #[test]
    fn test_amygdala_rm_with_flags() {
        let amygdala = AmygdalaActor::new();
        assert!(
            amygdala.scan("rm -rf --no-preserve-root /").is_some(),
            "Should block rm with interleaved flags"
        );
    }

    #[test]
    fn test_amygdala_rm_flags_interleaved() {
        let amygdala = AmygdalaActor::new();
        assert!(
            amygdala.scan("rm --force -r /").is_some(),
            "Should block rm with long flags before -r"
        );
    }

    #[test]
    fn test_amygdala_case_sensitivity() {
        let amygdala = AmygdalaActor::new();
        // Unix commands are case-sensitive — uppercase variants are not real commands
        assert!(
            amygdala.scan("RM -RF /").is_none(),
            "Uppercase RM is not a real command on Unix"
        );
    }

    #[test]
    fn test_amygdala_partial_match_safe() {
        let amygdala = AmygdalaActor::new();
        // Substring match: "rm -rf /" appears inside the larger command
        assert!(
            amygdala.scan("cargo rm -rf /tmp/test").is_some(),
            "Should still block when rm -rf / appears as substring"
        );
    }

    #[test]
    fn test_amygdala_dd_with_flags() {
        let amygdala = AmygdalaActor::new();
        assert!(
            amygdala.scan("dd if=/dev/zero of=/dev/sda bs=4M").is_some(),
            "Should block dd writing to raw device"
        );
    }

    #[test]
    fn test_guardrails_large_dimension() {
        let guardrails = SafetyGuardrails::with_dimension(2048);
        assert_eq!(guardrails.dimension(), 2048);
        // Prototype self-match should still work at higher dimensions
        let proto = guardrails
            .prototype(ForbiddenCategory::HarmfulContent)
            .unwrap()
            .to_vec();
        assert_eq!(
            guardrails.check(&proto),
            Some(ForbiddenCategory::HarmfulContent)
        );
    }
}
