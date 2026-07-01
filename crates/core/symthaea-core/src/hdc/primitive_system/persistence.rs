// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::{CompositionAlgebra, CompositionExport, PrimitiveGraph, PrimitiveSystem};
use serde::{Deserialize, Serialize};

/// Persistence manager for saving/loading primitive compositions and session data.
///
/// Supports:
/// - Composition algebra definitions (named expressions)
/// - Session history (composition operations performed)
/// - Custom primitive definitions (experimental)
///
/// # Example
/// ```ignore
/// let system = PrimitiveSystem::global();
/// let mut algebra = CompositionAlgebra::new();
/// algebra.define("MY_COMP", "CAUSE ⊗ EFFECT", system)?;
///
/// // Save session
/// let persistence = PrimitivePersistence::new();
/// persistence.save_session("session.json", &algebra, &history)?;
///
/// // Later, restore
/// let (loaded_algebra, loaded_history) = persistence.load_session("session.json", system)?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct PrimitivePersistence;

/// Serializable session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    /// Version for forward compatibility
    pub version: u32,
    /// Timestamp (Unix seconds)
    pub timestamp: u64,
    /// Named compositions
    pub compositions: Vec<CompositionExport>,
    /// Operation history
    pub history: Vec<HistoryEntry>,
    /// Optional notes
    pub notes: Option<String>,
}

/// A single history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// The operation performed
    pub operation: String,
    /// The best-matching primitive result
    pub result_match: String,
    /// Similarity to the match
    pub similarity: f32,
}

impl PrimitivePersistence {
    /// Create a new persistence manager.
    pub fn new() -> Self {
        Self
    }

    /// Save session data to a JSON file.
    pub fn save_session(
        &self,
        path: &str,
        algebra: &CompositionAlgebra,
        history: &[HistoryEntry],
        notes: Option<&str>,
    ) -> Result<(), PersistenceError> {
        use std::fs::File;
        use std::io::Write;
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let session = SessionData {
            version: 1,
            timestamp,
            compositions: algebra.export(),
            history: history.to_vec(),
            notes: notes.map(|s| s.to_string()),
        };

        let json = serde_json::to_string_pretty(&session)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;

        let mut file = File::create(path).map_err(|e| PersistenceError::IoError(e.to_string()))?;

        file.write_all(json.as_bytes())
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load session data from a JSON file.
    pub fn load_session(
        &self,
        path: &str,
        system: &PrimitiveSystem,
    ) -> Result<(CompositionAlgebra, Vec<HistoryEntry>), PersistenceError> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path).map_err(|e| PersistenceError::IoError(e.to_string()))?;

        let mut json = String::new();
        file.read_to_string(&mut json)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        let session: SessionData = serde_json::from_str(&json)
            .map_err(|e| PersistenceError::DeserializationError(e.to_string()))?;

        // Rebuild algebra
        let mut algebra = CompositionAlgebra::new();
        algebra
            .import(&session.compositions, system)
            .map_err(|e| PersistenceError::CompositionError(e.to_string()))?;

        Ok((algebra, session.history))
    }

    /// Save just compositions (without history).
    pub fn save_compositions(
        &self,
        path: &str,
        algebra: &CompositionAlgebra,
    ) -> Result<(), PersistenceError> {
        use std::fs::File;
        use std::io::Write;

        let exports = algebra.export();
        let json = serde_json::to_string_pretty(&exports)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;

        let mut file = File::create(path).map_err(|e| PersistenceError::IoError(e.to_string()))?;

        file.write_all(json.as_bytes())
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load just compositions (without history).
    pub fn load_compositions(
        &self,
        path: &str,
        system: &PrimitiveSystem,
    ) -> Result<CompositionAlgebra, PersistenceError> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path).map_err(|e| PersistenceError::IoError(e.to_string()))?;

        let mut json = String::new();
        file.read_to_string(&mut json)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        let exports: Vec<CompositionExport> = serde_json::from_str(&json)
            .map_err(|e| PersistenceError::DeserializationError(e.to_string()))?;

        let mut algebra = CompositionAlgebra::new();
        algebra
            .import(&exports, system)
            .map_err(|e| PersistenceError::CompositionError(e.to_string()))?;

        Ok(algebra)
    }

    /// Export a graph to DOT format file.
    pub fn export_graph_dot(
        &self,
        path: &str,
        graph: &PrimitiveGraph,
    ) -> Result<(), PersistenceError> {
        use std::fs::File;
        use std::io::Write;

        let dot = graph.to_dot();

        let mut file = File::create(path).map_err(|e| PersistenceError::IoError(e.to_string()))?;

        file.write_all(dot.as_bytes())
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Export similarity matrix to CSV format.
    pub fn export_similarity_csv(
        &self,
        path: &str,
        system: &PrimitiveSystem,
        names: &[&str],
    ) -> Result<(), PersistenceError> {
        use std::fs::File;
        use std::io::Write;

        let matrix = system.similarity_matrix(names);

        let mut file = File::create(path).map_err(|e| PersistenceError::IoError(e.to_string()))?;

        // Header
        let mut header = String::from(",");
        header.push_str(&names.join(","));
        writeln!(file, "{header}").map_err(|e| PersistenceError::IoError(e.to_string()))?;

        // Rows
        for (i, row) in matrix.iter().enumerate() {
            let row_str: Vec<String> = row.iter().map(|v| format!("{v:.4}")).collect();
            writeln!(file, "{},{}", names[i], row_str.join(","))
                .map_err(|e| PersistenceError::IoError(e.to_string()))?;
        }

        Ok(())
    }
}

/// Errors from persistence operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistenceError {
    /// IO error (file not found, permission denied, etc.)
    IoError(String),
    /// Serialization error
    SerializationError(String),
    /// Deserialization error
    DeserializationError(String),
    /// Error rebuilding composition
    CompositionError(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceError::IoError(msg) => write!(f, "IO error: {msg}"),
            PersistenceError::SerializationError(msg) => write!(f, "serialization error: {msg}"),
            PersistenceError::DeserializationError(msg) => {
                write!(f, "deserialization error: {msg}")
            }
            PersistenceError::CompositionError(msg) => write!(f, "composition error: {msg}"),
        }
    }
}

impl std::error::Error for PersistenceError {}
