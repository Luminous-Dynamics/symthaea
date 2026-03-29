// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Learning Objectives - HDC-encoded concepts for the School system
//!
//! A `LearningObjective` represents a concept that Symthaea can learn.
//! Each objective is encoded as a hyperdimensional vector (HDC) that
//! captures its semantic meaning.

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// ═══════════════════════════════════════════════════════════════════════════════
// DIFFICULTY LEVELS
// ═══════════════════════════════════════════════════════════════════════════════

/// Difficulty level of a learning objective
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum Difficulty {
    /// Very easy, foundational concepts (0.0 - 0.2)
    Beginner,

    /// Basic concepts requiring some foundation (0.2 - 0.4)
    Elementary,

    /// Intermediate concepts (0.4 - 0.6)
    #[default]
    Intermediate,

    /// Advanced concepts requiring significant background (0.6 - 0.8)
    Advanced,

    /// Expert-level concepts (0.8 - 1.0)
    Expert,
}

impl Difficulty {
    /// Convert to numeric value (0.0 - 1.0)
    pub fn as_f32(&self) -> f32 {
        match self {
            Difficulty::Beginner => 0.1,
            Difficulty::Elementary => 0.3,
            Difficulty::Intermediate => 0.5,
            Difficulty::Advanced => 0.7,
            Difficulty::Expert => 0.9,
        }
    }

    /// Create from numeric value
    pub fn from_f32(v: f32) -> Self {
        match v {
            x if x < 0.2 => Difficulty::Beginner,
            x if x < 0.4 => Difficulty::Elementary,
            x if x < 0.6 => Difficulty::Intermediate,
            x if x < 0.8 => Difficulty::Advanced,
            _ => Difficulty::Expert,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOMAIN
// ═══════════════════════════════════════════════════════════════════════════════

/// Knowledge domain for categorizing objectives
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Domain {
    /// NixOS system administration
    #[default]
    NixOS,

    /// Nix expression language
    NixLang,

    /// Nix Flakes
    Flakes,

    /// Home Manager configuration
    HomeManager,

    /// General system administration
    SystemAdmin,

    /// Programming and development
    Development,

    /// Rust programming language
    Rust,

    /// Holochain distributed applications
    Holochain,

    /// Consciousness and philosophy
    Consciousness,

    /// Mathematics and logic
    Mathematics,

    /// Hyperdimensional computing
    HDC,

    /// Custom domain
    Custom(String),
}

impl Domain {
    /// Get the domain seed for HDC encoding
    pub fn seed(&self) -> u64 {
        match self {
            Domain::NixOS => 1000,
            Domain::NixLang => 2000,
            Domain::Flakes => 3000,
            Domain::HomeManager => 4000,
            Domain::SystemAdmin => 5000,
            Domain::Development => 6000,
            Domain::Rust => 6100,
            Domain::Holochain => 6200,
            Domain::Consciousness => 7000,
            Domain::Mathematics => 8000,
            Domain::HDC => 8100,
            Domain::Custom(s) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                s.hash(&mut hasher);
                hasher.finish()
            }
        }
    }

    /// Get domain name as string
    pub fn name(&self) -> &str {
        match self {
            Domain::NixOS => "NixOS",
            Domain::NixLang => "Nix Language",
            Domain::Flakes => "Flakes",
            Domain::HomeManager => "Home Manager",
            Domain::SystemAdmin => "System Administration",
            Domain::Development => "Development",
            Domain::Rust => "Rust",
            Domain::Holochain => "Holochain",
            Domain::Consciousness => "Consciousness",
            Domain::Mathematics => "Mathematics",
            Domain::HDC => "Hyperdimensional Computing",
            Domain::Custom(s) => s,
        }
    }
}

impl From<&str> for Domain {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "nixos" | "nix-os" => Domain::NixOS,
            "nix" | "nixlang" | "nix-lang" => Domain::NixLang,
            "flakes" | "flake" => Domain::Flakes,
            "home-manager" | "homemanager" | "hm" => Domain::HomeManager,
            "sysadmin" | "systemadmin" | "admin" => Domain::SystemAdmin,
            "dev" | "development" | "programming" => Domain::Development,
            "rust" => Domain::Rust,
            "holochain" | "holo" | "happ" => Domain::Holochain,
            "consciousness" | "philosophy" => Domain::Consciousness,
            "math" | "mathematics" | "logic" => Domain::Mathematics,
            "hdc" | "hyperdimensional" => Domain::HDC,
            other => Domain::Custom(other.to_string()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING OBJECTIVE
// ═══════════════════════════════════════════════════════════════════════════════

/// A learning objective represents a concept that Symthaea can learn.
///
/// Each objective is HDC-encoded, allowing for:
/// - Semantic similarity comparisons
/// - Efficient CfC-based lookahead predictions
/// - Integration with the consciousness graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningObjective {
    /// Unique identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Detailed description
    pub description: String,

    /// Knowledge domain
    pub domain: Domain,

    /// Difficulty level
    pub difficulty: Difficulty,

    /// HDC encoding of the objective's semantics
    pub encoding: ContinuousHV,

    /// IDs of prerequisite objectives
    pub prerequisites: Vec<String>,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Estimated learning time in minutes
    pub estimated_minutes: u32,
}

impl LearningObjective {
    /// Create a new learning objective with an ID and name
    pub fn new(id: &str, name: &str) -> ObjectiveBuilder {
        ObjectiveBuilder::new(id, name)
    }

    /// Create a placeholder objective (for testing or "complete" states)
    pub fn placeholder(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            domain: Domain::default(),
            difficulty: Difficulty::Beginner,
            encoding: ContinuousHV::zero(HDC_DIMENSION),
            prerequisites: Vec::new(),
            tags: Vec::new(),
            estimated_minutes: 0,
        }
    }

    /// Convert to CfC input array
    pub fn to_cfc_input(&self, size: usize) -> Array1<f32> {
        let enc_slice = self.encoding.values.as_slice();
        let mut input = Array1::zeros(size);

        // Copy encoding values (truncated if necessary)
        for i in 0..size.min(enc_slice.len()) {
            input[i] = enc_slice[i];
        }

        // Embed difficulty in the last position
        if size > 0 {
            input[size - 1] = self.difficulty.as_f32();
        }

        input
    }

    /// Get semantic similarity to another objective
    pub fn similarity(&self, other: &LearningObjective) -> f32 {
        self.encoding.similarity(&other.encoding)
    }

    /// Check if this objective is in the same domain as another
    pub fn same_domain(&self, other: &LearningObjective) -> bool {
        self.domain == other.domain
    }

    /// Check if prerequisites include a specific objective ID
    pub fn requires(&self, objective_id: &str) -> bool {
        self.prerequisites.iter().any(|p| p == objective_id)
    }
}

impl PartialEq for LearningObjective {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for LearningObjective {}

impl Hash for LearningObjective {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OBJECTIVE BUILDER
// ═══════════════════════════════════════════════════════════════════════════════

/// Builder for creating LearningObjectives
#[derive(Debug, Clone)]
pub struct ObjectiveBuilder {
    id: String,
    name: String,
    description: String,
    domain: Domain,
    difficulty: Difficulty,
    dimension: usize,
    seed: Option<u64>,
    prerequisites: Vec<String>,
    tags: Vec<String>,
    estimated_minutes: u32,
}

impl ObjectiveBuilder {
    /// Create a new builder
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            domain: Domain::default(),
            difficulty: Difficulty::default(),
            dimension: HDC_DIMENSION,
            seed: None,
            prerequisites: Vec::new(),
            tags: Vec::new(),
            estimated_minutes: 15,
        }
    }

    /// Set dimension
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Set description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    /// Set domain
    pub fn with_domain(mut self, domain: Domain) -> Self {
        self.domain = domain;
        self
    }

    /// Set difficulty
    pub fn with_difficulty(mut self, difficulty: Difficulty) -> Self {
        self.difficulty = difficulty;
        self
    }

    /// Set random seed for encoding
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Add a prerequisite
    pub fn with_prerequisite(mut self, prereq_id: &str) -> Self {
        self.prerequisites.push(prereq_id.to_string());
        self
    }

    /// Add multiple prerequisites
    pub fn with_prerequisites(mut self, prereqs: &[&str]) -> Self {
        self.prerequisites
            .extend(prereqs.iter().map(|s| s.to_string()));
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Add multiple tags
    pub fn with_tags(mut self, tags: &[&str]) -> Self {
        self.tags.extend(tags.iter().map(|s| s.to_string()));
        self
    }

    /// Set estimated learning time
    pub fn with_estimated_minutes(mut self, minutes: u32) -> Self {
        self.estimated_minutes = minutes;
        self
    }

    /// Build the LearningObjective
    pub fn build(self) -> LearningObjective {
        // Generate seed from id if not provided
        let seed = self.seed.unwrap_or_else(|| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.id.hash(&mut hasher);
            hasher.finish()
        });

        // Generate HDC encoding
        let base = ContinuousHV::random(self.dimension, seed);
        let domain_hv = ContinuousHV::random(self.dimension, self.domain.seed());

        // Bind base with domain, scale by inverse difficulty
        // (easier concepts have stronger/cleaner encodings)
        let encoding = base
            .bind(&domain_hv)
            .scale(1.0 - self.difficulty.as_f32() * 0.3);

        LearningObjective {
            id: self.id,
            name: self.name,
            description: self.description,
            domain: self.domain,
            difficulty: self.difficulty,
            encoding,
            prerequisites: self.prerequisites,
            tags: self.tags,
            estimated_minutes: self.estimated_minutes,
        }
    }
}

// Allow ObjectiveBuilder to be used directly as LearningObjective in curriculum
impl From<ObjectiveBuilder> for LearningObjective {
    fn from(builder: ObjectiveBuilder) -> Self {
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_objective_creation() {
        let obj = LearningObjective::new("test-id", "Test Objective")
            .with_domain(Domain::NixOS)
            .with_difficulty(Difficulty::Intermediate)
            .with_description("A test objective")
            .build();

        assert_eq!(obj.id, "test-id");
        assert_eq!(obj.name, "Test Objective");
        assert_eq!(obj.domain, Domain::NixOS);
        assert_eq!(obj.difficulty, Difficulty::Intermediate);
        assert_eq!(obj.encoding.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_prerequisites() {
        let obj = LearningObjective::new("advanced", "Advanced Topic")
            .with_prerequisites(&["basic1", "basic2"])
            .build();

        assert!(obj.requires("basic1"));
        assert!(obj.requires("basic2"));
        assert!(!obj.requires("basic3"));
    }

    #[test]
    fn test_cfc_input() {
        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Expert) // 0.9
            .build();

        let input = obj.to_cfc_input(256);
        assert_eq!(input.len(), 256);
        assert!((input[255] - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_similarity() {
        let obj1 = LearningObjective::new("nix1", "Nix Basics")
            .with_domain(Domain::NixOS)
            .with_seed(100)
            .build();

        let obj2 = LearningObjective::new("nix2", "Nix Advanced")
            .with_domain(Domain::NixOS)
            .with_seed(101)
            .build();

        let obj3 = LearningObjective::new("rust1", "Rust Basics")
            .with_domain(Domain::Development)
            .with_seed(200)
            .build();

        // Same domain objectives should be more similar than different domains
        let sim_same = obj1.similarity(&obj2);
        let sim_diff = obj1.similarity(&obj3);

        // Both should be in [-1, 1]
        assert!((-1.0..=1.0).contains(&sim_same));
        assert!((-1.0..=1.0).contains(&sim_diff));
    }

    #[test]
    fn test_difficulty_conversion() {
        assert!((Difficulty::Beginner.as_f32() - 0.1).abs() < 0.01);
        assert!((Difficulty::Expert.as_f32() - 0.9).abs() < 0.01);

        assert_eq!(Difficulty::from_f32(0.15), Difficulty::Beginner);
        assert_eq!(Difficulty::from_f32(0.85), Difficulty::Expert);
    }

    #[test]
    fn test_domain_from_str() {
        assert_eq!(Domain::from("nixos"), Domain::NixOS);
        assert_eq!(Domain::from("flakes"), Domain::Flakes);
        assert_eq!(
            Domain::from("custom-thing"),
            Domain::Custom("custom-thing".to_string())
        );
    }
}
