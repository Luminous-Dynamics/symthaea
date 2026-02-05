//! Safety module - safety constraints and gateways
//!
//! Provides safety checking capabilities including:
//! - Fast regex-based pre-cognitive veto (AmygdalaActor)
//! - Hypervector-based forbidden subspace checking (SafetyGuardrails)
//! - Unified safety gateway interface

pub mod gateway;

// Re-export key types
pub use gateway::{SafetyGateway, SafetyDecision, SafetyCheck};

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
            ForbiddenCategory::HarmfulContent   => 0xDEAD_0002,
            ForbiddenCategory::PrivacyViolation  => 0xDEAD_0003,
            ForbiddenCategory::SecurityRisk      => 0xDEAD_0004,
            ForbiddenCategory::UnethicalRequest  => 0xDEAD_0005,
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
            // Destructive commands
            r"rm\s+-rf\s+/",
            r"dd\s+if=.*of=/dev/",
            r"mkfs\.",
            r":\(\)\{\s*:\|:&\s*\};:",  // Fork bomb (escaped for regex)
            r"chmod\s+-R\s+777\s+/",
            r">\s*/dev/sd",
        ];

        let mut compile_failures = 0;
        let dangerous_patterns = patterns
            .into_iter()
            .filter_map(|p| {
                match regex::Regex::new(p) {
                    Ok(re) => Some(re),
                    Err(e) => {
                        eprintln!("[safety] Failed to compile regex pattern '{}': {}", p, e);
                        compile_failures += 1;
                        None
                    }
                }
            })
            .collect();

        Self { dangerous_patterns, compile_failures }
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
            .map(|&cat| {
                ForbiddenPrototype {
                    category: cat,
                    vector: Self::generate_prototype(dimension, cat.seed()),
                }
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

    /// Cosine similarity between two f32 vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
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
        let proto = guardrails.prototype(ForbiddenCategory::DangerousCommand).unwrap().to_vec();
        assert_eq!(guardrails.check(&proto), Some(ForbiddenCategory::DangerousCommand));
    }

    #[test]
    fn test_guardrails_allows_random_vector() {
        let guardrails = SafetyGuardrails::new();
        // A random vector should NOT match any prototype (in high dimensions)
        let random_vec: Vec<f32> = (0..512).map(|i| {
            let mut state = 0xBEEF_CAFE_u64.wrapping_add(i as u64);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        }).collect();
        assert_eq!(guardrails.check(&random_vec), None);
    }

    #[test]
    fn test_guardrails_inactive_allows_all() {
        let mut guardrails = SafetyGuardrails::new();
        guardrails.set_active(false);
        let proto = guardrails.prototype(ForbiddenCategory::DangerousCommand).unwrap().to_vec();
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
        let proto = guardrails.prototype(ForbiddenCategory::SecurityRisk).unwrap().to_vec();
        let results = guardrails.check_detailed(&proto);

        // SecurityRisk prototype should have high similarity with itself
        let security_sim = results.iter()
            .find(|(cat, _)| *cat == ForbiddenCategory::SecurityRisk)
            .map(|(_, sim)| *sim)
            .unwrap();
        assert!(security_sim > 0.99, "Self-similarity should be ~1.0, got {}", security_sim);

        // Other categories should have low similarity
        for (cat, sim) in &results {
            if *cat != ForbiddenCategory::SecurityRisk {
                assert!(*sim < 0.5, "Cross-category similarity should be low, got {} for {:?}", sim, cat);
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
                    categories[i], categories[j], sim
                );
            }
        }
    }
}
