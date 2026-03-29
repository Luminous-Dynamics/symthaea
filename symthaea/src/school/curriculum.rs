// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Curriculum management for structured learning paths
//!
//! A `Curriculum` is a structured set of learning objectives with
//! prerequisite relationships, enabling coherent learning progressions.

use super::objective::{Difficulty, Domain, LearningObjective};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ═══════════════════════════════════════════════════════════════════════════════
// SCHEMA FOR LLM GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Lightweight schema for learning objectives, optimized for LLM generation.
/// Does not contain the large HDC vector; the vector is generated deterministically
/// from the ID and Domain upon conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveSchema {
    pub id: String,
    pub name: String,
    pub description: String,
    #[serde(default = "default_domain")]
    pub domain: String, // String to allow flexible parsing into Domain enum
    #[serde(default = "default_difficulty")]
    pub difficulty: f32, // 0.0 - 1.0
    #[serde(default)]
    pub prerequisites: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_minutes")]
    pub estimated_minutes: u32,
}

fn default_domain() -> String {
    "Custom".to_string()
}
fn default_difficulty() -> f32 {
    0.5
}
fn default_minutes() -> u32 {
    30
}

/// Schema for a full curriculum extension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumSchema {
    pub name: String,
    pub description: String,
    pub objectives: Vec<ObjectiveSchema>,
}

#[derive(Debug)]
pub enum CurriculumSchemaError {
    EmptyName,
    EmptyDescription,
    EmptyObjectives,
    EmptyObjectiveField { field: &'static str, index: usize },
    DuplicateObjectiveId { id: String },
    InvalidDifficulty { id: String, difficulty: f32 },
}

impl std::fmt::Display for CurriculumSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CurriculumSchemaError::EmptyName => write!(f, "Curriculum name cannot be empty"),
            CurriculumSchemaError::EmptyDescription => {
                write!(f, "Curriculum description cannot be empty")
            }
            CurriculumSchemaError::EmptyObjectives => {
                write!(f, "Curriculum must contain at least one objective")
            }
            CurriculumSchemaError::EmptyObjectiveField { field, index } => write!(
                f,
                "Objective at index {} has empty field '{}'",
                index, field
            ),
            CurriculumSchemaError::DuplicateObjectiveId { id } => {
                write!(f, "Duplicate objective id '{}'", id)
            }
            CurriculumSchemaError::InvalidDifficulty { id, difficulty } => write!(
                f,
                "Objective '{}' has invalid difficulty {} (expected 0.0..=1.0)",
                id, difficulty
            ),
        }
    }
}

impl std::error::Error for CurriculumSchemaError {}

impl CurriculumSchema {
    pub fn validate_basic(&self) -> Result<(), CurriculumSchemaError> {
        if self.name.trim().is_empty() {
            return Err(CurriculumSchemaError::EmptyName);
        }
        if self.description.trim().is_empty() {
            return Err(CurriculumSchemaError::EmptyDescription);
        }
        if self.objectives.is_empty() {
            return Err(CurriculumSchemaError::EmptyObjectives);
        }

        let mut ids = HashSet::new();
        for (index, obj) in self.objectives.iter().enumerate() {
            if obj.id.trim().is_empty() {
                return Err(CurriculumSchemaError::EmptyObjectiveField { field: "id", index });
            }
            if obj.name.trim().is_empty() {
                return Err(CurriculumSchemaError::EmptyObjectiveField {
                    field: "name",
                    index,
                });
            }
            if obj.description.trim().is_empty() {
                return Err(CurriculumSchemaError::EmptyObjectiveField {
                    field: "description",
                    index,
                });
            }
            if obj.domain.trim().is_empty() {
                return Err(CurriculumSchemaError::EmptyObjectiveField {
                    field: "domain",
                    index,
                });
            }
            if !(0.0..=1.0).contains(&obj.difficulty) {
                return Err(CurriculumSchemaError::InvalidDifficulty {
                    id: obj.id.clone(),
                    difficulty: obj.difficulty,
                });
            }
            if !ids.insert(obj.id.clone()) {
                return Err(CurriculumSchemaError::DuplicateObjectiveId { id: obj.id.clone() });
            }
        }

        Ok(())
    }
}

impl From<ObjectiveSchema> for LearningObjective {
    fn from(schema: ObjectiveSchema) -> Self {
        super::objective::ObjectiveBuilder::new(&schema.id, &schema.name)
            .with_description(&schema.description)
            .with_domain(Domain::from(schema.domain.as_str()))
            .with_difficulty(Difficulty::from_f32(schema.difficulty))
            .with_prerequisites(
                &schema
                    .prerequisites
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            )
            .with_tags(&schema.tags.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .with_estimated_minutes(schema.estimated_minutes)
            .build()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CURRICULUM TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Built-in curriculum types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CurriculumType {
    /// NixOS fundamentals and system administration
    NixOS,

    /// Nix Flakes ecosystem
    Flakes,

    /// Home Manager for user configuration
    HomeManager,

    /// Full NixOS mastery (combines NixOS + Flakes + HomeManager)
    NixOSMastery,

    /// Consciousness and philosophy fundamentals
    Consciousness,

    /// Advanced consciousness: IIT, phenomenology, neuroscience
    ConsciousnessAdvanced,

    /// Hyperdimensional computing concepts
    HDC,

    /// Advanced HDC: sparse codes, sequence learning, applications
    HDCAdvanced,

    /// Rust programming language fundamentals
    Rust,

    /// Advanced Rust: async, macros, unsafe, systems programming
    RustAdvanced,

    /// Holochain distributed application development
    Holochain,

    /// Code generation fundamentals
    CodeGeneration,

    /// Advanced code generation: composition, property tests, LLM integration
    CodeGenerationAdvanced,

    /// Mathematics: arithmetic through linear algebra, statistics, formal logic
    #[cfg(feature = "mathematics")]
    Mathematics,

    /// Advanced mathematics: Fourier analysis, optimization, formal verification
    #[cfg(feature = "mathematics")]
    MathematicsAdvanced,
}

impl CurriculumType {
    /// Get human-readable name
    pub fn name(&self) -> &str {
        match self {
            CurriculumType::NixOS => "NixOS Fundamentals",
            CurriculumType::Flakes => "Nix Flakes",
            CurriculumType::HomeManager => "Home Manager",
            CurriculumType::NixOSMastery => "NixOS Mastery",
            CurriculumType::Consciousness => "Consciousness Studies",
            CurriculumType::ConsciousnessAdvanced => "Advanced Consciousness",
            CurriculumType::HDC => "Hyperdimensional Computing",
            CurriculumType::HDCAdvanced => "Advanced HDC",
            CurriculumType::Rust => "Rust Programming",
            CurriculumType::RustAdvanced => "Advanced Rust",
            CurriculumType::Holochain => "Holochain Development",
            CurriculumType::CodeGeneration => "Code Generation",
            CurriculumType::CodeGenerationAdvanced => "Advanced Code Generation",
            #[cfg(feature = "mathematics")]
            CurriculumType::Mathematics => "Mathematics",
            #[cfg(feature = "mathematics")]
            CurriculumType::MathematicsAdvanced => "Advanced Mathematics",
        }
    }

    /// Get description
    pub fn description(&self) -> &str {
        match self {
            CurriculumType::NixOS => "Learn NixOS system administration from basics to advanced",
            CurriculumType::Flakes => {
                "Master the Nix Flakes ecosystem for reproducible development"
            }
            CurriculumType::HomeManager => "Configure your user environment with Home Manager",
            CurriculumType::NixOSMastery => "Complete NixOS mastery combining all core skills",
            CurriculumType::Consciousness => "Explore consciousness, IIT, and philosophy of mind",
            CurriculumType::ConsciousnessAdvanced => {
                "Deep dive into IIT 4.0, phenomenology, and computational consciousness"
            }
            CurriculumType::HDC => "Learn hyperdimensional computing for AI and cognition",
            CurriculumType::HDCAdvanced => {
                "Master sparse codes, sequence learning, and HDC applications"
            }
            CurriculumType::Rust => "Learn Rust programming from basics to proficiency",
            CurriculumType::RustAdvanced => {
                "Master async Rust, macros, unsafe, and systems programming"
            }
            CurriculumType::Holochain => "Build distributed applications on Holochain",
            CurriculumType::CodeGeneration => {
                "Learn consciousness-aware code generation with HDC + CfC"
            }
            CurriculumType::CodeGenerationAdvanced => {
                "Master pattern composition, property testing, and LLM integration"
            }
            #[cfg(feature = "mathematics")]
            CurriculumType::Mathematics => {
                "Progressive mathematics from arithmetic through linear algebra, statistics, and logic"
            }
            #[cfg(feature = "mathematics")]
            CurriculumType::MathematicsAdvanced => {
                "Expert mathematics: Fourier analysis, optimization, formal verification"
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CURRICULUM
// ═══════════════════════════════════════════════════════════════════════════════

/// A structured curriculum containing learning objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Curriculum {
    /// Unique identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Description
    pub description: String,

    /// Learning objectives in this curriculum
    pub objectives: Vec<LearningObjective>,

    /// Prerequisite curriculum IDs
    pub prerequisite_curricula: Vec<String>,

    /// Tags for categorization
    pub tags: Vec<String>,
}

impl Curriculum {
    /// Start building a new curriculum
    pub fn new(id: &str, name: &str) -> CurriculumBuilder {
        CurriculumBuilder::new(id, name)
    }

    /// Create a built-in curriculum
    pub fn builtin(curriculum_type: CurriculumType) -> Self {
        match curriculum_type {
            CurriculumType::NixOS => Self::builtin_nixos(),
            CurriculumType::Flakes => Self::builtin_flakes(),
            CurriculumType::HomeManager => Self::builtin_home_manager(),
            CurriculumType::NixOSMastery => Self::builtin_nixos_mastery(),
            CurriculumType::Consciousness => Self::builtin_consciousness(),
            CurriculumType::ConsciousnessAdvanced => Self::builtin_consciousness_advanced(),
            CurriculumType::HDC => Self::builtin_hdc(),
            CurriculumType::HDCAdvanced => Self::builtin_hdc_advanced(),
            CurriculumType::Rust => Self::builtin_rust(),
            CurriculumType::RustAdvanced => Self::builtin_rust_advanced(),
            CurriculumType::Holochain => Self::builtin_holochain(),
            CurriculumType::CodeGeneration => super::code_curriculum::code_generation_curriculum(),
            CurriculumType::CodeGenerationAdvanced => {
                super::code_curriculum::code_generation_advanced_curriculum()
            }
            #[cfg(feature = "mathematics")]
            CurriculumType::Mathematics => super::math_curriculum::math_curriculum(),
            #[cfg(feature = "mathematics")]
            CurriculumType::MathematicsAdvanced => {
                super::math_curriculum::math_curriculum_advanced()
            }
        }
    }

    /// Get objectives that have no prerequisites
    pub fn entry_points(&self) -> Vec<&LearningObjective> {
        self.objectives
            .iter()
            .filter(|obj| obj.prerequisites.is_empty())
            .collect()
    }

    /// Get objectives that depend on a given objective
    pub fn dependents(&self, objective_id: &str) -> Vec<&LearningObjective> {
        self.objectives
            .iter()
            .filter(|obj| obj.requires(objective_id))
            .collect()
    }

    /// Get an objective by ID
    pub fn get(&self, id: &str) -> Option<&LearningObjective> {
        self.objectives.iter().find(|obj| obj.id == id)
    }

    /// Extend the curriculum with new objectives from a JSON schema
    ///
    /// This is the "Neural Bridge" entry point: LLMs generate the JSON,
    /// and Symthaea integrates it into its knowledge graph.
    pub fn extend_from_json(&mut self, json: &str, dimension: usize) -> Result<()> {
        let schema: CurriculumSchema = serde_json::from_str(json)?;
        schema.validate_basic()?;

        let mut updated = self.clone();
        updated.apply_schema(schema, dimension);
        updated.validate()?;
        *self = updated;

        Ok(())
    }

    fn apply_schema(&mut self, schema: CurriculumSchema, dimension: usize) {
        // Convert schema objectives to full LearningObjectives (generating HDC vectors)
        let new_objectives: Vec<LearningObjective> = schema
            .objectives
            .into_iter()
            .map(|obj_schema| {
                super::objective::ObjectiveBuilder::new(&obj_schema.id, &obj_schema.name)
                    .with_description(&obj_schema.description)
                    .with_domain(Domain::from(obj_schema.domain.as_str()))
                    .with_difficulty(Difficulty::from_f32(obj_schema.difficulty))
                    .with_dimension(dimension)
                    .with_prerequisites(
                        &obj_schema
                            .prerequisites
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                    )
                    .with_tags(
                        &obj_schema
                            .tags
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                    )
                    .with_estimated_minutes(obj_schema.estimated_minutes)
                    .build()
            })
            .collect();

        // Add to curriculum, avoiding duplicates by ID
        for obj in new_objectives {
            if self.get(&obj.id).is_none() {
                self.objectives.push(obj);
            }
        }

        // Auto-heal missing prerequisites: create stubs for anything not found
        let mut missing_prereqs = Vec::new();
        let existing_ids: HashSet<String> = self.objectives.iter().map(|o| o.id.clone()).collect();

        for obj in &self.objectives {
            for prereq in &obj.prerequisites {
                if !existing_ids.contains(prereq) {
                    missing_prereqs.push(prereq.clone());
                }
            }
        }

        for missing_id in missing_prereqs {
            if self.get(&missing_id).is_none() {
                let stub =
                    LearningObjective::new(&missing_id, &format!("Implicit: {}", missing_id))
                        .with_description("Automatically created prerequisite stub.")
                        .with_difficulty(Difficulty::Beginner)
                        .with_dimension(dimension)
                        .build();
                self.objectives.push(stub);
            }
        }
    }

    /// Validate the curriculum (check for cycles, missing prerequisites)
    pub fn validate(&self) -> Result<(), CurriculumError> {
        let ids: HashSet<_> = self.objectives.iter().map(|o| &o.id).collect();

        // Check for missing prerequisites
        for obj in &self.objectives {
            for prereq in &obj.prerequisites {
                if !ids.contains(prereq) {
                    return Err(CurriculumError::MissingPrerequisite {
                        objective: obj.id.clone(),
                        prerequisite: prereq.clone(),
                    });
                }
            }
        }

        // Check for cycles using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for obj in &self.objectives {
            if self.has_cycle(&obj.id, &mut visited, &mut rec_stack) {
                return Err(CurriculumError::CyclicDependency {
                    objective: obj.id.clone(),
                });
            }
        }

        Ok(())
    }

    fn has_cycle(
        &self,
        id: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        if rec_stack.contains(id) {
            return true;
        }
        if visited.contains(id) {
            return false;
        }

        visited.insert(id.to_string());
        rec_stack.insert(id.to_string());

        if let Some(obj) = self.get(id) {
            for prereq in &obj.prerequisites {
                if self.has_cycle(prereq, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(id);
        false
    }

    /// Get topological order of objectives (prerequisites first)
    pub fn topological_order(&self) -> Vec<&LearningObjective> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();

        for obj in &self.objectives {
            self.topo_visit(&obj.id, &mut visited, &mut result);
        }

        result
    }

    fn topo_visit<'a>(
        &'a self,
        id: &str,
        visited: &mut HashSet<String>,
        result: &mut Vec<&'a LearningObjective>,
    ) {
        if visited.contains(id) {
            return;
        }

        if let Some(obj) = self.get(id) {
            for prereq in &obj.prerequisites {
                self.topo_visit(prereq, visited, result);
            }

            visited.insert(id.to_string());
            result.push(obj);
        }
    }

    // ───────────────────────────────────────────────────────────────────────────
    // Built-in Curricula
    // ───────────────────────────────────────────────────────────────────────────

    fn builtin_nixos() -> Self {
        Curriculum::new("nixos", "NixOS Fundamentals")
            .with_description("Learn NixOS system administration from basics to advanced")
            .with_objective(
                LearningObjective::new("nix-basics", "Nix Expression Language Basics")
                    .with_domain(Domain::NixLang)
                    .with_difficulty(Difficulty::Beginner)
                    .with_description("Learn the fundamentals of the Nix expression language")
                    .with_tags(&["nix", "language", "basics"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("derivations", "Understanding Derivations")
                    .with_domain(Domain::NixLang)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Learn how Nix derivations work")
                    .with_prerequisite("nix-basics")
                    .with_tags(&["nix", "derivations", "build"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("nixos-config", "NixOS Configuration Structure")
                    .with_domain(Domain::NixOS)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Understand the structure of NixOS configuration")
                    .with_prerequisite("nix-basics")
                    .with_tags(&["nixos", "configuration"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("nixos-modules", "NixOS Module System")
                    .with_domain(Domain::NixOS)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Create and use NixOS modules")
                    .with_prerequisites(&["nixos-config", "derivations"])
                    .with_tags(&["nixos", "modules", "advanced"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("nixos-services", "Systemd Services in NixOS")
                    .with_domain(Domain::NixOS)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Configure and manage systemd services")
                    .with_prerequisite("nixos-config")
                    .with_tags(&["nixos", "systemd", "services"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("nixos-networking", "NixOS Networking")
                    .with_domain(Domain::NixOS)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Configure networking in NixOS")
                    .with_prerequisite("nixos-config")
                    .with_tags(&["nixos", "networking"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("nixos-security", "NixOS Security Hardening")
                    .with_domain(Domain::NixOS)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Secure and harden your NixOS installation")
                    .with_prerequisites(&["nixos-services", "nixos-networking"])
                    .with_tags(&["nixos", "security", "hardening"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("overlays", "Creating Nix Overlays")
                    .with_domain(Domain::NixLang)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Create overlays to customize packages")
                    .with_prerequisite("derivations")
                    .with_tags(&["nix", "overlays", "customization"])
                    .with_estimated_minutes(45),
            )
            .build()
    }

    fn builtin_flakes() -> Self {
        Curriculum::new("flakes", "Nix Flakes")
            .with_description("Master the Nix Flakes ecosystem for reproducible development")
            .with_objective(
                LearningObjective::new("flakes-intro", "Introduction to Flakes")
                    .with_domain(Domain::Flakes)
                    .with_difficulty(Difficulty::Beginner)
                    .with_description("Understand what Flakes are and why they matter")
                    .with_tags(&["flakes", "intro"])
                    .with_estimated_minutes(20),
            )
            .with_objective(
                LearningObjective::new("flakes-structure", "Flake Structure")
                    .with_domain(Domain::Flakes)
                    .with_difficulty(Difficulty::Elementary)
                    .with_description("Learn the structure of a flake.nix file")
                    .with_prerequisite("flakes-intro")
                    .with_tags(&["flakes", "structure"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("flakes-inputs", "Flake Inputs and Outputs")
                    .with_domain(Domain::Flakes)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Master flake inputs, outputs, and dependencies")
                    .with_prerequisite("flakes-structure")
                    .with_tags(&["flakes", "inputs", "outputs"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("flakes-devshells", "Development Shells with Flakes")
                    .with_domain(Domain::Flakes)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Create reproducible development environments")
                    .with_prerequisite("flakes-inputs")
                    .with_tags(&["flakes", "devshells", "development"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("flakes-templates", "Flake Templates")
                    .with_domain(Domain::Flakes)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Create and use flake templates")
                    .with_prerequisite("flakes-inputs")
                    .with_tags(&["flakes", "templates"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("flakes-nixos", "NixOS with Flakes")
                    .with_domain(Domain::Flakes)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Configure NixOS using flakes")
                    .with_prerequisite("flakes-inputs")
                    .with_tags(&["flakes", "nixos"])
                    .with_estimated_minutes(60),
            )
            .build()
    }

    fn builtin_home_manager() -> Self {
        Curriculum::new("home-manager", "Home Manager")
            .with_description("Configure your user environment with Home Manager")
            .with_objective(
                LearningObjective::new("hm-intro", "Introduction to Home Manager")
                    .with_domain(Domain::HomeManager)
                    .with_difficulty(Difficulty::Beginner)
                    .with_description("Understand Home Manager and its purpose")
                    .with_tags(&["home-manager", "intro"])
                    .with_estimated_minutes(20),
            )
            .with_objective(
                LearningObjective::new("hm-installation", "Installing Home Manager")
                    .with_domain(Domain::HomeManager)
                    .with_difficulty(Difficulty::Beginner)
                    .with_description("Install Home Manager standalone or with NixOS")
                    .with_prerequisite("hm-intro")
                    .with_tags(&["home-manager", "installation"])
                    .with_estimated_minutes(15),
            )
            .with_objective(
                LearningObjective::new("hm-config", "Home Manager Configuration")
                    .with_domain(Domain::HomeManager)
                    .with_difficulty(Difficulty::Elementary)
                    .with_description("Write your first home.nix configuration")
                    .with_prerequisite("hm-installation")
                    .with_tags(&["home-manager", "configuration"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("hm-programs", "Managing Programs with Home Manager")
                    .with_domain(Domain::HomeManager)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Configure programs like git, neovim, zsh")
                    .with_prerequisite("hm-config")
                    .with_tags(&["home-manager", "programs"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("hm-dotfiles", "Dotfile Management")
                    .with_domain(Domain::HomeManager)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Manage dotfiles declaratively")
                    .with_prerequisite("hm-config")
                    .with_tags(&["home-manager", "dotfiles"])
                    .with_estimated_minutes(30),
            )
            .build()
    }

    fn builtin_nixos_mastery() -> Self {
        // Combines objectives from multiple curricula
        let nixos = Self::builtin_nixos();
        let flakes = Self::builtin_flakes();
        let hm = Self::builtin_home_manager();

        let mut objectives: Vec<LearningObjective> = Vec::new();
        objectives.extend(nixos.objectives);
        objectives.extend(flakes.objectives);
        objectives.extend(hm.objectives);

        // Add mastery capstone objectives
        objectives.push(
            LearningObjective::new("mastery-integration", "Integrated NixOS System")
                .with_domain(Domain::NixOS)
                .with_difficulty(Difficulty::Expert)
                .with_description(
                    "Create a fully integrated NixOS system with flakes and Home Manager",
                )
                .with_prerequisites(&["nixos-modules", "flakes-nixos", "hm-programs"])
                .with_tags(&["mastery", "integration"])
                .with_estimated_minutes(120)
                .build(),
        );

        Curriculum {
            id: "nixos-mastery".to_string(),
            name: "NixOS Mastery".to_string(),
            description: "Complete NixOS mastery combining all core skills".to_string(),
            objectives,
            prerequisite_curricula: Vec::new(),
            tags: vec![
                "nixos".to_string(),
                "mastery".to_string(),
                "complete".to_string(),
            ],
        }
    }

    fn builtin_consciousness() -> Self {
        Curriculum::new("consciousness", "Consciousness Studies")
            .with_description("Explore consciousness, IIT, and philosophy of mind")
            .with_objective(
                LearningObjective::new("consciousness-intro", "What is Consciousness?")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Beginner)
                    .with_description("Introduction to the study of consciousness")
                    .with_tags(&["consciousness", "philosophy", "intro"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("iit-basics", "Integrated Information Theory Basics")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Learn the fundamentals of IIT and Φ")
                    .with_prerequisite("consciousness-intro")
                    .with_tags(&["iit", "phi", "theory"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("phi-computation", "Computing Φ")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Understand how Φ is computed and approximated")
                    .with_prerequisite("iit-basics")
                    .with_tags(&["phi", "computation", "algorithms"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("gwt", "Global Workspace Theory")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Understand Baars' Global Workspace Theory")
                    .with_prerequisite("consciousness-intro")
                    .with_tags(&["gwt", "attention", "workspace"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("autopoiesis", "Autopoiesis and Self-Organization")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Learn about self-creating, self-maintaining systems")
                    .with_prerequisite("consciousness-intro")
                    .with_tags(&["autopoiesis", "self-organization", "emergence"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("power-telemetry", "Power Telemetry and INA219 Monitoring")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Beginner)
                    .with_description(
                        "Track interoceptive power draw using INA219 readings and SSM smoothing",
                    )
                    .with_prerequisite("consciousness-intro")
                    .with_tags(&["ina219", "power", "telemetry", "interoception", "ssm"])
                    .with_estimated_minutes(20),
            )
            .build()
    }

    fn builtin_hdc() -> Self {
        Curriculum::new("hdc", "Hyperdimensional Computing")
            .with_description("Learn hyperdimensional computing for AI and cognition")
            .with_objective(
                LearningObjective::new("hdc-intro", "Introduction to HDC")
                    .with_domain(Domain::HDC)
                    .with_difficulty(Difficulty::Beginner)
                    .with_description("Understand the principles of hyperdimensional computing")
                    .with_tags(&["hdc", "intro", "vectors"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("hdc-operations", "HDC Operations")
                    .with_domain(Domain::HDC)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Learn bind, bundle, and similarity operations")
                    .with_prerequisite("hdc-intro")
                    .with_tags(&["hdc", "operations", "algebra"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("hdc-memory", "Associative Memory with HDC")
                    .with_domain(Domain::HDC)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Build associative memory systems")
                    .with_prerequisite("hdc-operations")
                    .with_tags(&["hdc", "memory", "associative"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("hdc-learning", "Learning with HDC")
                    .with_domain(Domain::HDC)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Implement learning algorithms in hyperdimensional space")
                    .with_prerequisite("hdc-memory")
                    .with_tags(&["hdc", "learning", "classification"])
                    .with_estimated_minutes(60),
            )
            .build()
    }

    fn builtin_consciousness_advanced() -> Self {
        Curriculum::new("consciousness-advanced", "Advanced Consciousness Studies")
            .with_description(
                "Deep dive into IIT 4.0, phenomenology, and computational consciousness",
            )
            .with_prerequisite_curriculum("consciousness")
            .with_objective(
                LearningObjective::new("iit-4", "Integrated Information Theory 4.0")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Master the latest IIT formulation with unfolded TPM")
                    .with_tags(&["iit", "phi", "tpm", "advanced"])
                    .with_estimated_minutes(90),
            )
            .with_objective(
                LearningObjective::new("phi-mechanisms", "Φ Mechanisms and Complexes")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Expert)
                    .with_description("Understand mechanism-level Φ and system complexes")
                    .with_prerequisite("iit-4")
                    .with_tags(&["phi", "mechanisms", "complexes"])
                    .with_estimated_minutes(120),
            )
            .with_objective(
                LearningObjective::new("phenomenology", "Computational Phenomenology")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description(
                        "Bridge phenomenological experience with information structures",
                    )
                    .with_prerequisite("iit-4")
                    .with_tags(&["phenomenology", "qualia", "experience"])
                    .with_estimated_minutes(75),
            )
            .with_objective(
                LearningObjective::new("fep", "Free Energy Principle")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Expert)
                    .with_description(
                        "Understand Friston's Free Energy Principle and active inference",
                    )
                    .with_tags(&["fep", "active-inference", "prediction"])
                    .with_estimated_minutes(90),
            )
            .with_objective(
                LearningObjective::new("consciousness-topologies", "Consciousness Topologies")
                    .with_domain(Domain::Consciousness)
                    .with_difficulty(Difficulty::Expert)
                    .with_description("Study how network topology affects Φ and consciousness")
                    .with_prerequisites(&["phi-mechanisms", "phenomenology"])
                    .with_tags(&["topology", "networks", "emergence"])
                    .with_estimated_minutes(90),
            )
            .build()
    }

    fn builtin_hdc_advanced() -> Self {
        Curriculum::new("hdc-advanced", "Advanced Hyperdimensional Computing")
            .with_description("Master sparse codes, sequence learning, and HDC applications")
            .with_prerequisite_curriculum("hdc")
            .with_objective(
                LearningObjective::new("sparse-hdc", "Sparse Hypervectors")
                    .with_domain(Domain::HDC)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Implement and utilize sparse binary hypervectors")
                    .with_tags(&["hdc", "sparse", "efficiency"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("sequence-learning", "Sequence Learning in HDC")
                    .with_domain(Domain::HDC)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Encode and reason about temporal sequences")
                    .with_prerequisite("sparse-hdc")
                    .with_tags(&["hdc", "sequences", "temporal"])
                    .with_estimated_minutes(75),
            )
            .with_objective(
                LearningObjective::new("hdc-graphs", "Graph Reasoning with HDC")
                    .with_domain(Domain::HDC)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Represent and query graph structures in hyperspace")
                    .with_prerequisite("sparse-hdc")
                    .with_tags(&["hdc", "graphs", "reasoning"])
                    .with_estimated_minutes(75),
            )
            .with_objective(
                LearningObjective::new("hdc-neuromorphic", "Neuromorphic HDC")
                    .with_domain(Domain::HDC)
                    .with_difficulty(Difficulty::Expert)
                    .with_description("Implement HDC on neuromorphic hardware")
                    .with_prerequisites(&["sequence-learning", "hdc-graphs"])
                    .with_tags(&["hdc", "neuromorphic", "hardware"])
                    .with_estimated_minutes(90),
            )
            .with_objective(
                LearningObjective::new("hdc-consciousness", "HDC for Consciousness Models")
                    .with_domain(Domain::HDC)
                    .with_difficulty(Difficulty::Expert)
                    .with_description("Apply HDC to build computational consciousness models")
                    .with_prerequisite("hdc-neuromorphic")
                    .with_tags(&["hdc", "consciousness", "phi"])
                    .with_estimated_minutes(120),
            )
            .build()
    }

    fn builtin_rust() -> Self {
        Curriculum::new("rust", "Rust Programming")
            .with_description("Learn Rust programming from basics to proficiency")
            .with_objective(
                LearningObjective::new("rust-intro", "Introduction to Rust")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Beginner)
                    .with_description("Understand Rust's goals, toolchain, and ecosystem")
                    .with_tags(&["rust", "intro", "cargo"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("rust-ownership", "Ownership and Borrowing")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Master Rust's ownership model and borrow checker")
                    .with_prerequisite("rust-intro")
                    .with_tags(&["rust", "ownership", "borrowing"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("rust-types", "Structs, Enums, and Pattern Matching")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Elementary)
                    .with_description("Define custom types and use pattern matching")
                    .with_prerequisite("rust-intro")
                    .with_tags(&["rust", "types", "patterns"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("rust-traits", "Traits and Generics")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Write polymorphic code with traits and generics")
                    .with_prerequisites(&["rust-ownership", "rust-types"])
                    .with_tags(&["rust", "traits", "generics"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("rust-errors", "Error Handling")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Handle errors idiomatically with Result and Option")
                    .with_prerequisite("rust-types")
                    .with_tags(&["rust", "errors", "result"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("rust-iterators", "Iterators and Closures")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Use functional programming patterns in Rust")
                    .with_prerequisite("rust-traits")
                    .with_tags(&["rust", "iterators", "closures"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("rust-lifetimes", "Lifetimes")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Annotate and understand lifetime parameters")
                    .with_prerequisite("rust-ownership")
                    .with_tags(&["rust", "lifetimes", "references"])
                    .with_estimated_minutes(60),
            )
            .build()
    }

    fn builtin_rust_advanced() -> Self {
        Curriculum::new("rust-advanced", "Advanced Rust")
            .with_description("Master async Rust, macros, unsafe, and systems programming")
            .with_prerequisite_curriculum("rust")
            .with_objective(
                LearningObjective::new("rust-async", "Async Rust")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Write asynchronous code with async/await and tokio")
                    .with_tags(&["rust", "async", "tokio"])
                    .with_estimated_minutes(90),
            )
            .with_objective(
                LearningObjective::new("rust-macros", "Declarative Macros")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Write macro_rules! macros for code generation")
                    .with_tags(&["rust", "macros", "metaprogramming"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("rust-proc-macros", "Procedural Macros")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Expert)
                    .with_description("Write derive macros and attribute macros")
                    .with_prerequisite("rust-macros")
                    .with_tags(&["rust", "proc-macros", "derive"])
                    .with_estimated_minutes(90),
            )
            .with_objective(
                LearningObjective::new("rust-unsafe", "Unsafe Rust")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Expert)
                    .with_description("Use unsafe code safely and correctly")
                    .with_tags(&["rust", "unsafe", "ffi"])
                    .with_estimated_minutes(75),
            )
            .with_objective(
                LearningObjective::new("rust-ffi", "FFI and Interop")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Expert)
                    .with_description("Interface Rust with C, Python, and other languages")
                    .with_prerequisite("rust-unsafe")
                    .with_tags(&["rust", "ffi", "c", "python"])
                    .with_estimated_minutes(75),
            )
            .with_objective(
                LearningObjective::new("rust-perf", "Performance Optimization")
                    .with_domain(Domain::Rust)
                    .with_difficulty(Difficulty::Expert)
                    .with_description("Profile and optimize Rust programs for maximum performance")
                    .with_prerequisites(&["rust-async", "rust-unsafe"])
                    .with_tags(&["rust", "performance", "simd"])
                    .with_estimated_minutes(90),
            )
            .build()
    }

    fn builtin_holochain() -> Self {
        Curriculum::new("holochain", "Holochain Development")
            .with_description("Build distributed applications on Holochain")
            .with_objective(
                LearningObjective::new("holo-intro", "Introduction to Holochain")
                    .with_domain(Domain::Holochain)
                    .with_difficulty(Difficulty::Beginner)
                    .with_description("Understand Holochain's agent-centric architecture")
                    .with_tags(&["holochain", "intro", "distributed"])
                    .with_estimated_minutes(30),
            )
            .with_objective(
                LearningObjective::new("holo-dht", "Holochain DHT")
                    .with_domain(Domain::Holochain)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Understand the distributed hash table and gossip protocol")
                    .with_prerequisite("holo-intro")
                    .with_tags(&["holochain", "dht", "gossip"])
                    .with_estimated_minutes(45),
            )
            .with_objective(
                LearningObjective::new("hdk-basics", "HDK Basics")
                    .with_domain(Domain::Holochain)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Write Holochain zomes with the HDK")
                    .with_prerequisite("holo-dht")
                    .with_tags(&["holochain", "hdk", "rust"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("holo-entries", "Entries and Links")
                    .with_domain(Domain::Holochain)
                    .with_difficulty(Difficulty::Intermediate)
                    .with_description("Create, validate, and link entries in Holochain")
                    .with_prerequisite("hdk-basics")
                    .with_tags(&["holochain", "entries", "links"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("holo-validation", "Validation Rules")
                    .with_domain(Domain::Holochain)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Write validation rules for data integrity")
                    .with_prerequisite("holo-entries")
                    .with_tags(&["holochain", "validation", "integrity"])
                    .with_estimated_minutes(75),
            )
            .with_objective(
                LearningObjective::new("holo-signals", "Signals and Remote Calls")
                    .with_domain(Domain::Holochain)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Implement real-time communication and cross-zome calls")
                    .with_prerequisite("holo-entries")
                    .with_tags(&["holochain", "signals", "remote"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("happ-deployment", "hApp Deployment")
                    .with_domain(Domain::Holochain)
                    .with_difficulty(Difficulty::Advanced)
                    .with_description("Package and deploy a complete hApp")
                    .with_prerequisites(&["holo-validation", "holo-signals"])
                    .with_tags(&["holochain", "happ", "deployment"])
                    .with_estimated_minutes(60),
            )
            .with_objective(
                LearningObjective::new("holo-capabilities", "Capability-Based Security")
                    .with_domain(Domain::Holochain)
                    .with_difficulty(Difficulty::Expert)
                    .with_description("Implement fine-grained access control with capabilities")
                    .with_prerequisite("happ-deployment")
                    .with_tags(&["holochain", "capabilities", "security"])
                    .with_estimated_minutes(75),
            )
            .build()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CURRICULUM BUILDER
// ═══════════════════════════════════════════════════════════════════════════════

/// Builder for creating curricula
#[derive(Debug, Clone)]
pub struct CurriculumBuilder {
    id: String,
    name: String,
    description: String,
    objectives: Vec<LearningObjective>,
    prerequisite_curricula: Vec<String>,
    tags: Vec<String>,
}

impl CurriculumBuilder {
    /// Create a new builder
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            objectives: Vec::new(),
            prerequisite_curricula: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    /// Add an objective
    pub fn with_objective<O: Into<LearningObjective>>(mut self, objective: O) -> Self {
        self.objectives.push(objective.into());
        self
    }

    /// Add a prerequisite curriculum
    pub fn with_prerequisite_curriculum(mut self, curriculum_id: &str) -> Self {
        self.prerequisite_curricula.push(curriculum_id.to_string());
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

    /// Build the curriculum
    pub fn build(self) -> Curriculum {
        Curriculum {
            id: self.id,
            name: self.name,
            description: self.description,
            objectives: self.objectives,
            prerequisite_curricula: self.prerequisite_curricula,
            tags: self.tags,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ERRORS
// ═══════════════════════════════════════════════════════════════════════════════

/// Errors that can occur in curriculum operations
#[derive(Debug, Clone)]
pub enum CurriculumError {
    /// A prerequisite objective is missing
    MissingPrerequisite {
        objective: String,
        prerequisite: String,
    },

    /// The curriculum has a cyclic dependency
    CyclicDependency { objective: String },
}

impl std::fmt::Display for CurriculumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CurriculumError::MissingPrerequisite {
                objective,
                prerequisite,
            } => {
                write!(
                    f,
                    "Objective '{objective}' requires missing prerequisite '{prerequisite}'"
                )
            }
            CurriculumError::CyclicDependency { objective } => {
                write!(f, "Cyclic dependency detected at objective '{objective}'")
            }
        }
    }
}

impl std::error::Error for CurriculumError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_nixos() {
        let curriculum = Curriculum::builtin(CurriculumType::NixOS);
        assert!(!curriculum.objectives.is_empty());
        assert!(curriculum.validate().is_ok());
    }

    #[test]
    fn test_builtin_flakes() {
        let curriculum = Curriculum::builtin(CurriculumType::Flakes);
        assert!(!curriculum.objectives.is_empty());
        assert!(curriculum.validate().is_ok());
    }

    #[test]
    fn test_entry_points() {
        let curriculum = Curriculum::builtin(CurriculumType::NixOS);
        let entries = curriculum.entry_points();

        // Entry points should have no prerequisites
        for entry in entries {
            assert!(entry.prerequisites.is_empty());
        }
    }

    #[test]
    fn test_topological_order() {
        let curriculum = Curriculum::builtin(CurriculumType::NixOS);
        let order = curriculum.topological_order();

        // Prerequisites should come before dependents
        let mut seen = std::collections::HashSet::new();
        for obj in order {
            for prereq in &obj.prerequisites {
                assert!(
                    seen.contains(prereq),
                    "Prerequisite {} not seen before {}",
                    prereq,
                    obj.id
                );
            }
            seen.insert(obj.id.clone());
        }
    }

    #[test]
    fn test_custom_curriculum() {
        let curriculum = Curriculum::new("custom", "Custom Curriculum")
            .with_description("A test curriculum")
            .with_objective(
                LearningObjective::new("step1", "Step 1").with_difficulty(Difficulty::Beginner),
            )
            .with_objective(
                LearningObjective::new("step2", "Step 2")
                    .with_prerequisite("step1")
                    .with_difficulty(Difficulty::Intermediate),
            )
            .build();

        assert_eq!(curriculum.objectives.len(), 2);
        assert!(curriculum.validate().is_ok());
    }

    #[test]
    fn test_extend_from_json_dimension_and_auto_heal() {
        let mut curriculum = Curriculum::new("test", "Test Curriculum").build();
        let json = r#"{
            "name": "Meta Study",
            "description": "Test curriculum extension",
            "objectives": [
                {
                    "id": "ssm-basics",
                    "name": "SSM Basics",
                    "description": "Intro to state space models",
                    "domain": "Math",
                    "difficulty": 0.3,
                    "prerequisites": ["linear-algebra"],
                    "tags": ["ssm", "math"],
                    "estimated_minutes": 45
                }
            ]
        }"#;

        let dimension = 512;
        curriculum.extend_from_json(json, dimension).unwrap();

        let ssm = curriculum.get("ssm-basics").expect("ssm-basics missing");
        assert_eq!(ssm.encoding.values.len(), dimension);

        let prereq = curriculum
            .get("linear-algebra")
            .expect("auto-healed prerequisite missing");
        assert_eq!(prereq.encoding.values.len(), dimension);
    }

    #[test]
    fn test_invalid_prerequisite() {
        let curriculum = Curriculum::new("invalid", "Invalid")
            .with_objective(
                LearningObjective::new("obj1", "Obj 1").with_prerequisite("nonexistent"),
            )
            .build();

        assert!(curriculum.validate().is_err());
    }
}
