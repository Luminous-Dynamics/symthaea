// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified School types for the Symthaea runtime.

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum Difficulty {
    Beginner,
    Elementary,
    #[default]
    Intermediate,
    Advanced,
    Expert,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Domain {
    #[default]
    Engineering,
    Aerospace,
    Civil,
    Electrical,
    Custom(String),
}

impl From<&str> for Domain {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "engineering" => Domain::Engineering,
            "aerospace" => Domain::Aerospace,
            "civil" => Domain::Civil,
            "electrical" => Domain::Electrical,
            other => Domain::Custom(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningObjective {
    pub id: String,
    pub name: String,
    pub description: String,
    pub domain: Domain,
    pub difficulty: Difficulty,
    pub encoding: ContinuousHV,
    pub tags: Vec<String>,
}

impl LearningObjective {
    pub fn new(id: &str, name: &str) -> LearningObjective {
        LearningObjective {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            domain: Domain::default(),
            difficulty: Difficulty::Intermediate,
            encoding: ContinuousHV::zero(16384),
            tags: Vec::new(),
        }
    }
}

/// A serializable sample for batch ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndustrialSampleSpec {
    pub id: String,
    pub name: String,
    pub domain: String,
    pub ground_truth: HashMap<String, f64>,
    pub source: String,
}

/// Load a batch of industrial benchmarks from a JSON fixture.
pub fn load_industrial_benchmarks<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<Vec<LearningObjective>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let specs: Vec<IndustrialSampleSpec> = serde_json::from_str(&content)?;

    let mut objectives = Vec::new();
    for spec in specs {
        let mut obj = LearningObjective::new(&spec.id, &spec.name);
        obj.description = format!("Industrial validation against {} reference.", spec.source);
        obj.domain = Domain::from(spec.domain.as_str());
        obj.tags.push(spec.source);
        objectives.push(obj);
    }

    Ok(objectives)
}
