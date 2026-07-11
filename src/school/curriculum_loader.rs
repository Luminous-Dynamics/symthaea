// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Curriculum Loader - Load curricula from YAML/JSON files
//!
//! This module enables loading custom curricula from external files,
//! allowing users to define their own learning paths without modifying code.
//!
//! ## Supported Formats
//!
//! - **JSON**: Standard JSON format
//! - **YAML**: Human-readable YAML format (preferred for hand-authored curricula)
//!
//! ## Example YAML Curriculum
//!
//! ```yaml
//! id: my-custom-curriculum
//! name: My Custom Learning Path
//! description: A personalized curriculum
//! tags:
//!   - custom
//!   - learning
//! objectives:
//!   - id: step-1
//!     name: First Concept
//!     domain: NixOS
//!     difficulty: Beginner
//!     description: Learn the basics
//!     estimated_minutes: 30
//!   - id: step-2
//!     name: Second Concept
//!     domain: NixOS
//!     difficulty: Intermediate
//!     prerequisites:
//!       - step-1
//!     description: Build on the basics
//!     estimated_minutes: 45
//! ```

use super::curriculum::{Curriculum, CurriculumError};
use super::objective::{Difficulty, Domain, LearningObjective};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use symthaea_core::hdc::HDC_DIMENSION;

// ═══════════════════════════════════════════════════════════════════════════════
// SERIALIZABLE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Serializable representation of a learning objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    /// Unique identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Detailed description
    #[serde(default)]
    pub description: String,

    /// Knowledge domain (NixOS, Flakes, Rust, etc.)
    #[serde(default = "default_domain")]
    pub domain: String,

    /// Difficulty level (Beginner, Elementary, Intermediate, Advanced, Expert)
    #[serde(default = "default_difficulty")]
    pub difficulty: String,

    /// IDs of prerequisite objectives
    #[serde(default)]
    pub prerequisites: Vec<String>,

    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,

    /// Estimated learning time in minutes
    #[serde(default = "default_minutes")]
    pub estimated_minutes: u32,

    /// Optional seed for HDC encoding reproducibility
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_domain() -> String {
    "NixOS".to_string()
}

fn default_difficulty() -> String {
    "Intermediate".to_string()
}

fn default_minutes() -> u32 {
    15
}

/// Serializable representation of a curriculum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumSpec {
    /// Unique identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Description
    #[serde(default)]
    pub description: String,

    /// Learning objectives
    pub objectives: Vec<ObjectiveSpec>,

    /// Prerequisite curriculum IDs
    #[serde(default)]
    pub prerequisite_curricula: Vec<String>,

    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERSISTENCE METADATA
// ═══════════════════════════════════════════════════════════════════════════════

/// Metadata for persisted curricula
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumMeta {
    /// Schema version for the stored payload
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Last research topic applied to the curriculum
    #[serde(default)]
    pub last_research_topic: Option<String>,
    /// Timestamp of last research ingestion (RFC3339)
    #[serde(default)]
    pub last_research_at: Option<String>,
    /// Timestamp of last save (RFC3339)
    #[serde(default)]
    pub last_saved_at: Option<String>,
    /// Count of objectives added in the last research
    #[serde(default)]
    pub last_objectives_added: Option<usize>,
    /// Dimension used to encode objectives
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    /// Total objectives stored at last save
    #[serde(default)]
    pub total_objectives: usize,
}

impl CurriculumMeta {
    pub fn new(dimension: usize) -> Self {
        Self {
            schema_version: default_schema_version(),
            last_research_topic: None,
            last_research_at: None,
            last_saved_at: None,
            last_objectives_added: None,
            dimension,
            total_objectives: 0,
        }
    }
}

fn default_schema_version() -> u32 {
    1
}

fn default_dimension() -> usize {
    HDC_DIMENSION
}

/// Combined store for curriculum + metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumStore {
    pub meta: CurriculumMeta,
    pub curriculum: CurriculumSpec,
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOADER ERRORS
// ═══════════════════════════════════════════════════════════════════════════════

/// Errors that can occur when loading curricula
#[derive(Debug)]
pub enum LoadError {
    /// File not found
    FileNotFound(String),

    /// IO error reading file
    IoError(std::io::Error),

    /// JSON parsing error
    JsonError(serde_json::Error),

    /// YAML parsing error (if serde_yaml available)
    YamlError(String),

    /// Unsupported file format
    UnsupportedFormat(String),

    /// Validation error in curriculum structure
    ValidationError(CurriculumError),

    /// Invalid difficulty level
    InvalidDifficulty(String),

    /// Invalid domain
    InvalidDomain(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::FileNotFound(path) => write!(f, "File not found: {path}"),
            LoadError::IoError(e) => write!(f, "IO error: {e}"),
            LoadError::JsonError(e) => write!(f, "JSON parse error: {e}"),
            LoadError::YamlError(msg) => write!(f, "YAML parse error: {msg}"),
            LoadError::UnsupportedFormat(ext) => write!(f, "Unsupported file format: {ext}"),
            LoadError::ValidationError(e) => write!(f, "Validation error: {e}"),
            LoadError::InvalidDifficulty(s) => write!(f, "Invalid difficulty: {s}"),
            LoadError::InvalidDomain(s) => write!(f, "Invalid domain: {s}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::IoError(e)
    }
}

impl From<serde_json::Error> for LoadError {
    fn from(e: serde_json::Error) -> Self {
        LoadError::JsonError(e)
    }
}

impl From<CurriculumError> for LoadError {
    fn from(e: CurriculumError) -> Self {
        LoadError::ValidationError(e)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CURRICULUM LOADER
// ═══════════════════════════════════════════════════════════════════════════════

/// Curriculum loader for external files
pub struct CurriculumLoader;

impl CurriculumLoader {
    /// Load a curriculum from a file path
    ///
    /// Automatically detects format based on file extension:
    /// - `.json` → JSON format
    /// - `.yaml` or `.yml` → YAML format
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Curriculum, LoadError> {
        Self::load_from_file_with_dimension(path, HDC_DIMENSION)
    }

    /// Load a curriculum from a file path with a target HDC dimension.
    pub fn load_from_file_with_dimension<P: AsRef<Path>>(
        path: P,
        dimension: usize,
    ) -> Result<Curriculum, LoadError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(LoadError::FileNotFound(path.display().to_string()));
        }

        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let content = fs::read_to_string(path)?;

        match extension.to_lowercase().as_str() {
            "json" => Self::load_from_json_with_dimension(&content, dimension),
            "yaml" | "yml" => Self::load_from_yaml_with_dimension(&content, dimension),
            other => Err(LoadError::UnsupportedFormat(other.to_string())),
        }
    }

    /// Load a curriculum from a JSON string
    pub fn load_from_json(json: &str) -> Result<Curriculum, LoadError> {
        Self::load_from_json_with_dimension(json, HDC_DIMENSION)
    }

    /// Load a curriculum from a YAML string
    ///
    /// This uses a simple YAML parser that handles common cases.
    /// For complex YAML, consider using the serde_yaml crate.
    pub fn load_from_yaml(yaml: &str) -> Result<Curriculum, LoadError> {
        Self::load_from_yaml_with_dimension(yaml, HDC_DIMENSION)
    }

    /// Load a curriculum from a JSON string with a target HDC dimension
    pub fn load_from_json_with_dimension(
        json: &str,
        dimension: usize,
    ) -> Result<Curriculum, LoadError> {
        let spec: CurriculumSpec = serde_json::from_str(json)?;
        Self::build_curriculum_with_dimension(spec, dimension)
    }

    /// Load a curriculum from a YAML string with a target HDC dimension
    ///
    /// This uses a simple YAML parser that handles common cases.
    /// For complex YAML, consider using the serde_yaml crate.
    pub fn load_from_yaml_with_dimension(
        yaml: &str,
        dimension: usize,
    ) -> Result<Curriculum, LoadError> {
        // Simple YAML to JSON conversion for basic cases
        // This handles the subset of YAML we need for curricula
        let json = yaml_to_json(yaml).map_err(LoadError::YamlError)?;
        Self::load_from_json_with_dimension(&json, dimension)
    }

    /// Load a persisted curriculum store from a file path with a target dimension.
    pub fn load_store_from_file_with_dimension<P: AsRef<Path>>(
        path: P,
        dimension: usize,
    ) -> Result<(Curriculum, CurriculumMeta), LoadError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(LoadError::FileNotFound(path.display().to_string()));
        }

        let content = fs::read_to_string(path)?;
        Self::load_store_from_json_with_dimension(&content, dimension)
    }

    /// Load a persisted curriculum store from a JSON string with a target dimension.
    ///
    /// Falls back to legacy CurriculumSpec format when metadata is absent.
    pub fn load_store_from_json_with_dimension(
        json: &str,
        dimension: usize,
    ) -> Result<(Curriculum, CurriculumMeta), LoadError> {
        if let Ok(store) = serde_json::from_str::<CurriculumStore>(json) {
            let curriculum = Self::build_curriculum_with_dimension(store.curriculum, dimension)?;
            let mut meta = store.meta;
            if meta.dimension != dimension {
                meta.dimension = dimension;
            }
            meta.total_objectives = curriculum.objectives.len();
            return Ok((curriculum, meta));
        }

        let spec: CurriculumSpec = serde_json::from_str(json)?;
        let curriculum = Self::build_curriculum_with_dimension(spec, dimension)?;
        let mut meta = CurriculumMeta::new(dimension);
        meta.total_objectives = curriculum.objectives.len();
        Ok((curriculum, meta))
    }

    /// Build a Curriculum from a CurriculumSpec with a target HDC dimension
    fn build_curriculum_with_dimension(
        spec: CurriculumSpec,
        dimension: usize,
    ) -> Result<Curriculum, LoadError> {
        let mut objectives = Vec::new();

        for obj_spec in spec.objectives {
            let objective = Self::build_objective_with_dimension(obj_spec, dimension)?;
            objectives.push(objective);
        }

        let curriculum = Curriculum {
            id: spec.id,
            name: spec.name,
            description: spec.description,
            objectives,
            prerequisite_curricula: spec.prerequisite_curricula,
            tags: spec.tags,
        };

        // Validate the curriculum
        curriculum.validate()?;

        Ok(curriculum)
    }

    /// Build a LearningObjective from an ObjectiveSpec with a target HDC dimension
    fn build_objective_with_dimension(
        spec: ObjectiveSpec,
        dimension: usize,
    ) -> Result<LearningObjective, LoadError> {
        let difficulty = parse_difficulty(&spec.difficulty)
            .ok_or_else(|| LoadError::InvalidDifficulty(spec.difficulty.clone()))?;

        let domain = Domain::from(spec.domain.as_str());

        let mut builder = LearningObjective::new(&spec.id, &spec.name)
            .with_description(&spec.description)
            .with_domain(domain)
            .with_difficulty(difficulty)
            .with_dimension(dimension)
            .with_estimated_minutes(spec.estimated_minutes);

        for prereq in &spec.prerequisites {
            builder = builder.with_prerequisite(prereq);
        }

        for tag in &spec.tags {
            builder = builder.with_tag(tag);
        }

        if let Some(seed) = spec.seed {
            builder = builder.with_seed(seed);
        }

        Ok(builder.build())
    }

    /// Save a curriculum to a JSON file (atomic: temp file + rename)
    pub fn save_to_json<P: AsRef<Path>>(curriculum: &Curriculum, path: P) -> Result<(), LoadError> {
        let spec = Self::curriculum_to_spec(curriculum);
        let json = serde_json::to_string_pretty(&spec)?;
        write_atomic(path.as_ref(), &json)?;
        Ok(())
    }

    /// Save a curriculum store (metadata + curriculum) to a JSON file
    /// (atomic: temp file + rename)
    pub fn save_store_to_json<P: AsRef<Path>>(
        curriculum: &Curriculum,
        meta: &CurriculumMeta,
        path: P,
    ) -> Result<(), LoadError> {
        let store = CurriculumStore {
            meta: meta.clone(),
            curriculum: Self::curriculum_to_spec(curriculum),
        };
        let json = serde_json::to_string_pretty(&store)?;
        write_atomic(path.as_ref(), &json)?;
        Ok(())
    }

    /// Convert a Curriculum to CurriculumSpec for serialization
    fn curriculum_to_spec(curriculum: &Curriculum) -> CurriculumSpec {
        let objectives: Vec<ObjectiveSpec> = curriculum
            .objectives
            .iter()
            .map(|obj| ObjectiveSpec {
                id: obj.id.clone(),
                name: obj.name.clone(),
                description: obj.description.clone(),
                domain: obj.domain.name().to_string(),
                difficulty: difficulty_to_string(obj.difficulty),
                prerequisites: obj.prerequisites.clone(),
                tags: obj.tags.clone(),
                estimated_minutes: obj.estimated_minutes,
                seed: None,
            })
            .collect();

        CurriculumSpec {
            id: curriculum.id.clone(),
            name: curriculum.name.clone(),
            description: curriculum.description.clone(),
            objectives,
            prerequisite_curricula: curriculum.prerequisite_curricula.clone(),
            tags: curriculum.tags.clone(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Write `contents` to `path` atomically: write a temp file in the same
/// directory, then rename over the target. This guarantees readers (e.g. the
/// Broca curriculum-sync pipeline consuming `SYMTHAEA_CURRICULUM_PATH`) never
/// observe a partially-written curriculum. The temp filename embeds the
/// process ID so concurrent writers do not clobber each other's temp files.
fn write_atomic(path: &Path, contents: &str) -> Result<(), LoadError> {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("curriculum.json");
    let tmp_path = path.with_file_name(format!("{}.tmp.{}", file_name, std::process::id()));
    fs::write(&tmp_path, contents)?;
    if let Err(e) = fs::rename(&tmp_path, path) {
        // Best-effort cleanup of the orphaned temp file before surfacing the error.
        let _ = fs::remove_file(&tmp_path);
        return Err(LoadError::IoError(e));
    }
    Ok(())
}

/// Parse a difficulty string to Difficulty enum
fn parse_difficulty(s: &str) -> Option<Difficulty> {
    match s.to_lowercase().as_str() {
        "beginner" | "very easy" | "easy" => Some(Difficulty::Beginner),
        "elementary" | "basic" => Some(Difficulty::Elementary),
        "intermediate" | "medium" | "normal" => Some(Difficulty::Intermediate),
        "advanced" | "hard" | "difficult" => Some(Difficulty::Advanced),
        "expert" | "very hard" | "master" => Some(Difficulty::Expert),
        _ => None,
    }
}

/// Convert Difficulty to string for serialization
fn difficulty_to_string(d: Difficulty) -> String {
    match d {
        Difficulty::Beginner => "Beginner".to_string(),
        Difficulty::Elementary => "Elementary".to_string(),
        Difficulty::Intermediate => "Intermediate".to_string(),
        Difficulty::Advanced => "Advanced".to_string(),
        Difficulty::Expert => "Expert".to_string(),
    }
}

/// Simple YAML to JSON converter for curriculum files
///
/// This handles the subset of YAML needed for curriculum definitions:
/// - Key-value pairs
/// - Lists (with `-` prefix)
/// - Nested objects via indentation
///
/// For full YAML support, use the `serde_yaml` crate.
fn yaml_to_json(yaml: &str) -> Result<String, String> {
    // This is a simplified YAML parser that handles our curriculum format
    // For production, consider using serde_yaml

    let mut result = String::new();
    let mut indent_stack: Vec<usize> = vec![0];
    let mut in_list = false;
    let mut first_item = true;

    result.push('{');

    for line in yaml.lines() {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let indent = line.len() - line.trim_start().len();
        let is_list_item = trimmed.starts_with('-');

        // Handle list items
        if is_list_item {
            if !in_list {
                in_list = true;
                first_item = true;
            }

            let content = trimmed.strip_prefix('-').unwrap_or(trimmed).trim();

            if !first_item {
                result.push(',');
            }
            first_item = false;

            // Simple value list item
            if !content.contains(':') {
                result.push_str(&format!("\"{}\"", escape_json(content)));
            } else {
                // Object list item (like objectives)
                result.push('{');
                let parts: Vec<&str> = content.splitn(2, ':').collect();
                if parts.len() == 2 {
                    result.push_str(&format!(
                        "\"{}\":{}",
                        parts[0].trim(),
                        json_value(parts[1].trim())
                    ));
                }
            }
        } else if trimmed.contains(':') {
            // Key-value pair
            let parts: Vec<&str> = trimmed.splitn(2, ':').collect();
            if parts.len() == 2 {
                let key = parts[0].trim();
                let value = parts[1].trim();

                if !first_item && !in_list {
                    result.push(',');
                }

                if in_list && indent <= *indent_stack.last().unwrap_or(&0) {
                    result.push(']');
                    in_list = false;
                }

                first_item = false;

                if value.is_empty() {
                    // Start of a nested structure or list
                    result.push_str(&format!("\"{key}\":"));
                    // Check next line to determine if list or object
                    result.push('['); // Assume list for now
                    in_list = true;
                    first_item = true;
                } else {
                    result.push_str(&format!("\"{}\":{}", key, json_value(value)));
                }

                indent_stack.push(indent);
            }
        }
    }

    // Close any open lists
    if in_list {
        result.push(']');
    }

    result.push('}');

    // Try to parse to validate, if it fails, return a more helpful error
    match serde_json::from_str::<serde_json::Value>(&result) {
        Ok(_) => Ok(result),
        Err(e) => {
            // For complex YAML, suggest using JSON directly
            Err(format!(
                "YAML parsing failed: {e}. For complex curricula, consider using JSON format directly."
            ))
        }
    }
}

/// Escape a string for JSON
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Convert a YAML value to JSON value
fn json_value(s: &str) -> String {
    // Remove quotes if present
    let s = s.trim_matches(|c| c == '\'' || c == '"');

    // Check for booleans
    if s.eq_ignore_ascii_case("true") {
        return "true".to_string();
    }
    if s.eq_ignore_ascii_case("false") {
        return "false".to_string();
    }

    // Check for null
    if s.eq_ignore_ascii_case("null") || s == "~" {
        return "null".to_string();
    }

    // Check for numbers
    if s.parse::<i64>().is_ok() || s.parse::<f64>().is_ok() {
        return s.to_string();
    }

    // Otherwise, quote as string
    format!("\"{}\"", escape_json(s))
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_from_json() {
        let json = r#"
        {
            "id": "test-curriculum",
            "name": "Test Curriculum",
            "description": "A test curriculum",
            "objectives": [
                {
                    "id": "step1",
                    "name": "First Step",
                    "domain": "NixOS",
                    "difficulty": "Beginner",
                    "description": "Learn the basics",
                    "estimated_minutes": 30
                },
                {
                    "id": "step2",
                    "name": "Second Step",
                    "domain": "NixOS",
                    "difficulty": "Intermediate",
                    "prerequisites": ["step1"],
                    "description": "Build on basics",
                    "estimated_minutes": 45
                }
            ],
            "tags": ["test", "example"]
        }
        "#;

        let curriculum = CurriculumLoader::load_from_json(json).unwrap();

        assert_eq!(curriculum.id, "test-curriculum");
        assert_eq!(curriculum.objectives.len(), 2);
        assert_eq!(curriculum.objectives[0].id, "step1");
        assert!(curriculum.objectives[1].requires("step1"));
        assert!(curriculum.validate().is_ok());
    }

    #[test]
    fn test_parse_difficulty() {
        assert_eq!(parse_difficulty("Beginner"), Some(Difficulty::Beginner));
        assert_eq!(
            parse_difficulty("INTERMEDIATE"),
            Some(Difficulty::Intermediate)
        );
        assert_eq!(parse_difficulty("expert"), Some(Difficulty::Expert));
        assert_eq!(parse_difficulty("hard"), Some(Difficulty::Advanced));
        assert_eq!(parse_difficulty("invalid"), None);
    }

    #[test]
    fn test_difficulty_to_string() {
        assert_eq!(difficulty_to_string(Difficulty::Beginner), "Beginner");
        assert_eq!(difficulty_to_string(Difficulty::Expert), "Expert");
    }

    #[test]
    fn test_objective_spec_defaults() {
        let json = r#"{"id": "test", "name": "Test"}"#;
        let spec: ObjectiveSpec = serde_json::from_str(json).unwrap();

        assert_eq!(spec.domain, "NixOS");
        assert_eq!(spec.difficulty, "Intermediate");
        assert_eq!(spec.estimated_minutes, 15);
        assert!(spec.prerequisites.is_empty());
        assert!(spec.tags.is_empty());
    }

    #[test]
    fn test_invalid_prerequisite() {
        let json = r#"
        {
            "id": "broken",
            "name": "Broken",
            "objectives": [
                {
                    "id": "step1",
                    "name": "Step 1",
                    "prerequisites": ["nonexistent"]
                }
            ]
        }
        "#;

        let result = CurriculumLoader::load_from_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_and_load() {
        use super::super::curriculum::{Curriculum as C, CurriculumType};

        let original = C::builtin(CurriculumType::NixOS);
        let spec = CurriculumLoader::curriculum_to_spec(&original);

        // Serialize to JSON
        let json = serde_json::to_string(&spec).unwrap();

        // Load it back
        let loaded = CurriculumLoader::load_from_json(&json).unwrap();

        assert_eq!(loaded.id, original.id);
        assert_eq!(loaded.objectives.len(), original.objectives.len());
    }

    #[test]
    fn test_save_store_to_json_atomic_roundtrip() {
        use super::super::curriculum::{Curriculum as C, CurriculumType};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("curriculum.json");

        let original = C::builtin(CurriculumType::NixOS);
        let meta = CurriculumMeta::new(64);
        CurriculumLoader::save_store_to_json(&original, &meta, &path).unwrap();

        // Atomic write must not leave temp files behind.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(leftovers.is_empty(), "temp file left behind: {leftovers:?}");

        let (loaded, loaded_meta) =
            CurriculumLoader::load_store_from_file_with_dimension(&path, 64).unwrap();
        assert_eq!(loaded.id, original.id);
        assert_eq!(loaded.objectives.len(), original.objectives.len());
        assert_eq!(loaded_meta.dimension, 64);
    }

    #[test]
    fn test_load_store_with_dimension() {
        let json = r#"
        {
            "meta": {
                "schema_version": 1,
                "dimension": 2048,
                "last_research_topic": "test"
            },
            "curriculum": {
                "id": "store-test",
                "name": "Store Test",
                "description": "Store test curriculum",
                "objectives": [
                    {
                        "id": "step1",
                        "name": "Step 1",
                        "domain": "NixOS",
                        "difficulty": "Beginner",
                        "description": "Learn the basics",
                        "estimated_minutes": 30
                    }
                ]
            }
        }
        "#;

        let (curriculum, meta) =
            CurriculumLoader::load_store_from_json_with_dimension(json, 1024).unwrap();

        assert_eq!(curriculum.id, "store-test");
        assert_eq!(curriculum.objectives.len(), 1);
        assert_eq!(meta.dimension, 1024);
        assert_eq!(meta.last_research_topic.as_deref(), Some("test"));
    }

    #[test]
    fn test_json_value() {
        assert_eq!(json_value("42"), "42");
        assert_eq!(json_value("3.14"), "3.14");
        assert_eq!(json_value("true"), "true");
        assert_eq!(json_value("false"), "false");
        assert_eq!(json_value("null"), "null");
        assert_eq!(json_value("hello"), "\"hello\"");
        assert_eq!(json_value("'quoted'"), "\"quoted\"");
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("hello\"world"), "hello\\\"world");
        assert_eq!(escape_json("line1\nline2"), "line1\\nline2");
    }
}
