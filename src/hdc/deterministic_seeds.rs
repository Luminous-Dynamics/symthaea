//! Deterministic Seed Generation for Hyperdimensional Computing
//!
//! # Why This Matters
//!
//! In HDC, we need deterministic seeds for:
//! 1. **Reproducibility**: Same concept → same HV16 across runs
//! 2. **Debugging**: Known seeds enable bisection-style debugging
//! 3. **Testing**: Deterministic tests don't flake
//! 4. **Composition**: Semantic relationships can be algebraically derived
//!
//! # Key Insight: BLAKE3 Makes Seed Value Irrelevant
//!
//! The HV16::random(seed) function uses BLAKE3, which has perfect avalanche properties.
//! This means:
//! - Prime numbers don't help (BLAKE3 doesn't use modular arithmetic)
//! - Sequential integers work fine (each hash is independent)
//! - What matters is UNIQUENESS and DETERMINISM, not the specific value
//!
//! # Revolutionary Feature: Gödel Semantic Primes
//!
//! While primes don't help for randomness, they enable **semantic algebra**:
//! - Each primitive concept gets a unique prime number
//! - Composite concepts = product of primes
//! - Factor decomposition reveals semantic structure
//! - GCD/LCM enable semantic similarity computation
//!
//! ## Example
//! ```text
//! INSTALL = 2 (prime)
//! PACKAGE = 3 (prime)
//! INSTALL_PACKAGE = 2 × 3 = 6 (composite)
//!
//! SEARCH = 5 (prime)
//! SEARCH_PACKAGE = 5 × 3 = 15 (composite)
//!
//! GCD(INSTALL_PACKAGE, SEARCH_PACKAGE) = 3 = PACKAGE
//! → They share the PACKAGE concept!
//! ```
//!
//! This is inspired by Gödel numbering from mathematical logic.

use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

/// Generate a deterministic seed from a string name.
///
/// This is the PRIMARY approach for named concepts.
/// Uses Rust's DefaultHasher for consistent u64 generation.
///
/// # Examples
/// ```
/// # use symthaea::hdc::deterministic_seeds::seed_from_name;
/// let install_seed = seed_from_name("INSTALL");
/// let search_seed = seed_from_name("SEARCH");
///
/// // Same name always produces same seed
/// assert_eq!(install_seed, seed_from_name("INSTALL"));
///
/// // Different names produce different seeds (with overwhelming probability)
/// assert_ne!(install_seed, search_seed);
/// ```
#[inline]
pub fn seed_from_name(name: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

/// Generate a namespaced seed for domain separation.
///
/// This prevents accidental collisions between different subsystems.
///
/// # Examples
/// ```
/// # use symthaea::hdc::deterministic_seeds::seed_namespaced;
/// let visual_seed = seed_namespaced("visual", 42);
/// let audio_seed = seed_namespaced("audio", 42);
///
/// // Same index, different namespace → different seeds
/// assert_ne!(visual_seed, audio_seed);
/// ```
#[inline]
pub fn seed_namespaced(namespace: &str, index: u64) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    namespace.hash(&mut hasher);
    index.hash(&mut hasher);
    hasher.finish()
}

/// Domain-specific seed generator with consistent namespacing.
#[derive(Clone, Debug)]
pub struct SeedDomain {
    namespace: String,
    next_index: u64,
}

impl SeedDomain {
    /// Create a new seed domain with given namespace.
    pub fn new(namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            next_index: 0,
        }
    }

    /// Get a seed for a named concept within this domain.
    pub fn named(&self, name: &str) -> u64 {
        let full_name = format!("{}::{}", self.namespace, name);
        seed_from_name(&full_name)
    }

    /// Get the next sequential seed in this domain.
    pub fn next(&mut self) -> u64 {
        let seed = seed_namespaced(&self.namespace, self.next_index);
        self.next_index += 1;
        seed
    }

    /// Get a seed for a specific index in this domain.
    pub fn at(&self, index: u64) -> u64 {
        seed_namespaced(&self.namespace, index)
    }
}

// ============================================================================
// GÖDEL SEMANTIC PRIMES: Revolutionary Semantic Algebra
// ============================================================================

/// First 100 prime numbers for Gödel encoding.
///
/// Each primitive concept is assigned a unique prime.
/// Composite concepts become products of primes.
pub static PRIMES: LazyLock<[u64; 100]> = LazyLock::new(|| {
    [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
        233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
        353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
        419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
        467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
    ]
});

/// NixOS-specific semantic primitives as Gödel primes.
///
/// This enables algebraic reasoning over NixOS concepts:
/// - What concepts does an operation involve?
/// - What's common between two operations?
/// - Can one operation be decomposed into others?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NixPrimeConcept {
    // Actions (primes 0-9)
    Install = 0,
    Search = 1,
    Remove = 2,
    Update = 3,
    Build = 4,
    Switch = 5,
    Rollback = 6,
    Query = 7,
    Enable = 8,
    Disable = 9,

    // Targets (primes 10-19)
    Package = 10,
    Service = 11,
    Option = 12,
    Flake = 13,
    Generation = 14,
    Profile = 15,
    Module = 16,
    Overlay = 17,
    Configuration = 18,
    Derivation = 19,

    // Modifiers (primes 20-29)
    System = 20,
    User = 21,
    Global = 22,
    Local = 23,
    Temporary = 24,
    Permanent = 25,
    Dry = 26,
    Force = 27,
    Verbose = 28,
    Quiet = 29,

    // Sources (primes 30-39)
    Nixpkgs = 30,
    Flakes = 31,
    Channel = 32,
    Git = 33,
    Path = 34,
    Store = 35,
    Cache = 36,
    Binary = 37,
    Source = 38,
    Remote = 39,
}

impl NixPrimeConcept {
    /// Get the prime number for this concept.
    pub fn prime(&self) -> u64 {
        PRIMES[*self as usize]
    }

    /// Get the seed for this concept (for HV16 generation).
    /// Uses the prime as base, offset to avoid collision with test seeds.
    pub fn seed(&self) -> u64 {
        // Use prime + large offset to separate from arbitrary test seeds
        self.prime() + 1_000_000_000
    }

    /// Get all concepts from a Gödel number (product of primes).
    pub fn factor(godel: u64) -> Vec<NixPrimeConcept> {
        let mut result = Vec::new();
        let mut remaining = godel;

        for i in 0..40 {
            let prime = PRIMES[i];
            while remaining % prime == 0 {
                remaining /= prime;
                // Safe because we only iterate over valid indices
                result.push(unsafe { std::mem::transmute::<u8, NixPrimeConcept>(i as u8) });
            }
            if remaining == 1 {
                break;
            }
        }

        result
    }
}

/// A Gödel number representing a composite NixOS concept.
///
/// This is the product of prime factors for each component concept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GodelNumber(pub u64);

impl GodelNumber {
    /// Create a Gödel number from a single concept.
    pub fn from_concept(concept: NixPrimeConcept) -> Self {
        Self(concept.prime())
    }

    /// Create a Gödel number from multiple concepts.
    pub fn from_concepts(concepts: &[NixPrimeConcept]) -> Self {
        let product = concepts.iter()
            .map(|c| c.prime())
            .product();
        Self(product)
    }

    /// Compose two Gödel numbers (multiply).
    pub fn compose(&self, other: &Self) -> Self {
        Self(self.0 * other.0)
    }

    /// Check if this number contains a concept.
    pub fn contains(&self, concept: NixPrimeConcept) -> bool {
        self.0 % concept.prime() == 0
    }

    /// Get the shared concepts between two Gödel numbers (GCD-based).
    pub fn shared(&self, other: &Self) -> Self {
        Self(gcd(self.0, other.0))
    }

    /// Get all concepts in this Gödel number.
    pub fn concepts(&self) -> Vec<NixPrimeConcept> {
        NixPrimeConcept::factor(self.0)
    }

    /// Generate a seed for HV16 from this Gödel number.
    pub fn seed(&self) -> u64 {
        // Hash the Gödel number to get a good seed distribution
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

/// Greatest Common Divisor (Euclidean algorithm).
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

// ============================================================================
// RECOMMENDED SEED PATTERNS
// ============================================================================

/// Standard domains for test seeds.
pub mod test_domains {
    use super::SeedDomain;

    /// Visual processing tests (seeds in 100-199 range effectively).
    pub fn visual() -> SeedDomain {
        SeedDomain::new("test::visual")
    }

    /// Auditory processing tests.
    pub fn auditory() -> SeedDomain {
        SeedDomain::new("test::auditory")
    }

    /// Linguistic processing tests.
    pub fn linguistic() -> SeedDomain {
        SeedDomain::new("test::linguistic")
    }

    /// Cross-modal binding tests.
    pub fn cross_modal() -> SeedDomain {
        SeedDomain::new("test::cross_modal")
    }

    /// NixOS-specific tests.
    pub fn nixos() -> SeedDomain {
        SeedDomain::new("test::nixos")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_from_name_deterministic() {
        let seed1 = seed_from_name("INSTALL");
        let seed2 = seed_from_name("INSTALL");
        assert_eq!(seed1, seed2);
    }

    #[test]
    fn test_seed_from_name_unique() {
        let install = seed_from_name("INSTALL");
        let search = seed_from_name("SEARCH");
        let update = seed_from_name("UPDATE");

        assert_ne!(install, search);
        assert_ne!(install, update);
        assert_ne!(search, update);
    }

    #[test]
    fn test_namespaced_seeds() {
        let visual_42 = seed_namespaced("visual", 42);
        let audio_42 = seed_namespaced("audio", 42);
        let visual_43 = seed_namespaced("visual", 43);

        // Same index, different namespace
        assert_ne!(visual_42, audio_42);
        // Same namespace, different index
        assert_ne!(visual_42, visual_43);
    }

    #[test]
    fn test_seed_domain() {
        let mut domain = SeedDomain::new("consciousness");

        let s1 = domain.next();
        let s2 = domain.next();
        let s3 = domain.next();

        // Sequential seeds are unique
        assert_ne!(s1, s2);
        assert_ne!(s2, s3);

        // Named seeds are deterministic
        let named1 = domain.named("AWARENESS");
        let named2 = domain.named("AWARENESS");
        assert_eq!(named1, named2);

        // Named seeds are unique
        let attention = domain.named("ATTENTION");
        assert_ne!(named1, attention);
    }

    #[test]
    fn test_godel_single_concept() {
        let install = GodelNumber::from_concept(NixPrimeConcept::Install);
        let search = GodelNumber::from_concept(NixPrimeConcept::Search);

        // INSTALL = 2 (first prime)
        assert_eq!(install.0, 2);
        // SEARCH = 3 (second prime)
        assert_eq!(search.0, 3);
    }

    #[test]
    fn test_godel_composition() {
        let install = GodelNumber::from_concept(NixPrimeConcept::Install);
        let package = GodelNumber::from_concept(NixPrimeConcept::Package);

        let install_package = install.compose(&package);

        // INSTALL (2) × PACKAGE (31) = 62
        assert_eq!(install_package.0, 2 * 31);

        // Contains both concepts
        assert!(install_package.contains(NixPrimeConcept::Install));
        assert!(install_package.contains(NixPrimeConcept::Package));
        assert!(!install_package.contains(NixPrimeConcept::Search));
    }

    #[test]
    fn test_godel_shared_concepts() {
        let install_package = GodelNumber::from_concepts(&[
            NixPrimeConcept::Install,
            NixPrimeConcept::Package,
        ]);
        let search_package = GodelNumber::from_concepts(&[
            NixPrimeConcept::Search,
            NixPrimeConcept::Package,
        ]);

        let shared = install_package.shared(&search_package);

        // They share PACKAGE
        assert!(shared.contains(NixPrimeConcept::Package));
        // But not the action
        assert!(!shared.contains(NixPrimeConcept::Install));
        assert!(!shared.contains(NixPrimeConcept::Search));
    }

    #[test]
    fn test_godel_factorization() {
        let composite = GodelNumber::from_concepts(&[
            NixPrimeConcept::Install,
            NixPrimeConcept::Package,
            NixPrimeConcept::System,
        ]);

        let concepts = composite.concepts();

        assert_eq!(concepts.len(), 3);
        assert!(concepts.contains(&NixPrimeConcept::Install));
        assert!(concepts.contains(&NixPrimeConcept::Package));
        assert!(concepts.contains(&NixPrimeConcept::System));
    }

    #[test]
    fn test_godel_seed_generation() {
        let install_pkg = GodelNumber::from_concepts(&[
            NixPrimeConcept::Install,
            NixPrimeConcept::Package,
        ]);
        let search_pkg = GodelNumber::from_concepts(&[
            NixPrimeConcept::Search,
            NixPrimeConcept::Package,
        ]);

        // Different Gödel numbers produce different seeds
        assert_ne!(install_pkg.seed(), search_pkg.seed());

        // Same Gödel number produces same seed
        let install_pkg2 = GodelNumber::from_concepts(&[
            NixPrimeConcept::Package,  // Order doesn't matter!
            NixPrimeConcept::Install,
        ]);
        assert_eq!(install_pkg.0, install_pkg2.0);  // Same product
        assert_eq!(install_pkg.seed(), install_pkg2.seed());
    }

    #[test]
    fn test_nix_concept_primes() {
        // Verify first few primes are correct
        assert_eq!(NixPrimeConcept::Install.prime(), 2);
        assert_eq!(NixPrimeConcept::Search.prime(), 3);
        assert_eq!(NixPrimeConcept::Package.prime(), 31);  // 10th prime
    }

    #[test]
    fn test_test_domains() {
        let mut visual = test_domains::visual();
        let mut audio = test_domains::auditory();

        // Different domains, same index → different seeds
        assert_ne!(visual.next(), audio.next());

        // Named items in domain are deterministic
        let v_cat1 = visual.named("cat");
        let v_cat2 = visual.named("cat");
        assert_eq!(v_cat1, v_cat2);
    }
}
