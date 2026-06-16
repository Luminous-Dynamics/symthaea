// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evolutionary Scaffolder — Synthetic Data 2.0
//!
//! Generates novel, verified training pairs by mutating existing high-quality code.
//! Uses the StructuralScorer to ensure mutated code remains syntactically and
//! structurally sound.

use crate::structural_scorer::NixStructuralScorer;
use crate::training::TrainingPair;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of an evolutionary mutation cycle.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvolutionResult {
    pub id: u64,
    pub success_score: f32,
    pub mutation_description: String,
    pub changed_files: Vec<String>,
    pub before_code: String,
    pub after_code: String,
    pub metrics: HashMap<String, f32>,
}

pub struct EvolutionaryScaffolder {
    scorer: NixStructuralScorer,
}

impl EvolutionaryScaffolder {
    pub fn new() -> Self {
        Self {
            scorer: NixStructuralScorer::new(),
        }
    }

    /// Generate a mutated version of a training pair.
    pub fn scaffold_pair(&self, pair: &TrainingPair) -> Option<TrainingPair> {
        let mut rng = rand::thread_rng();

        // Only mutate if it looks like Nix (for now)
        if !pair.target_text.contains("{") {
            return None;
        }

        let mutated_code = self.mutate_nix(&pair.target_text, &mut rng);

        // Verify: must still parse and have similar structure
        let verdict = self.scorer.score(&mutated_code, &pair.target_text);
        if verdict.parse_error.is_none() && verdict.missing_required.is_empty() {
            Some(TrainingPair {
                channels: pair.channels.clone(),
                target_text: mutated_code,
                target_ids: Vec::new(), // will be re-tokenized
                valence: pair.valence,
                arousal: pair.arousal,
            })
        } else {
            None
        }
    }

    fn mutate_nix(&self, code: &str, rng: &mut impl Rng) -> String {
        let mut lines: Vec<String> = code.lines().map(|s| s.to_string()).collect();

        // Mutation strategies:
        // 1. Swap boolean values
        // 2. Change numeric literals
        // 3. (Future) Add/Remove optional attributes from NixKG knowledge

        for line in lines.iter_mut() {
            if rng.gen_bool(0.2) {
                if line.contains("true") {
                    *line = line.replace("true", "false");
                } else if line.contains("false") {
                    *line = line.replace("false", "true");
                }
            }

            if rng.gen_bool(0.1) {
                // Try to find a number and increment/decrement it
                if let Some(pos) = line.find(|c: char| c.is_ascii_digit()) {
                    let end = line[pos..]
                        .find(|c: char| !c.is_ascii_digit())
                        .map_or(line.len(), |e| pos + e);
                    if let Ok(val) = line[pos..end].parse::<i32>() {
                        let new_val = val + if rng.gen_bool(0.5) { 1 } else { -1 };
                        *line = format!("{}{}{}", &line[..pos], new_val, &line[end..]);
                    }
                }
            }
        }

        lines.join("\n")
    }
}

impl Default for EvolutionaryScaffolder {
    fn default() -> Self {
        Self::new()
    }
}
