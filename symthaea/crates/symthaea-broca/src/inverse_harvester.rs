// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inverse Harvester — Phase 2 M7 data engine
//!
//! Crawls real-world .nix files, extracts high-signal fragments,
//! and uses NixKG to synthesize natural-language prompts.

use crate::nix_kg::NixKg;
use crate::structural_scorer::NixStructuralScorer;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize)]
pub struct InverseHarvestPair {
    pub prompt: String,
    pub code: String,
    pub intent: String,
    pub channels: Vec<f32>, // 17D Nix intent vector
    pub iterations: usize,
    pub repair_steps: usize,
    pub holdout: bool,
    pub valence: f32,
    pub arousal: f32,
    pub source_file: String,
}

pub struct InverseHarvester {
    nix_kg: NixKg,
    scorer: NixStructuralScorer,
    max_pairs: usize,
    min_diversity_threshold: f32,
}

impl InverseHarvester {
    pub fn new() -> Self {
        Self {
            nix_kg: NixKg::default(),
            scorer: NixStructuralScorer::new(),
            max_pairs: 2000,
            min_diversity_threshold: 0.75,
        }
    }

    pub fn with_max_pairs(mut self, max: usize) -> Self {
        self.max_pairs = max;
        self
    }

    pub fn with_min_diversity_threshold(mut self, thresh: f32) -> Self {
        self.min_diversity_threshold = thresh.clamp(0.5, 0.95);
        self
    }

    pub fn harvest_directory(&self, root: &Path) -> Vec<InverseHarvestPair> {
        let mut pairs = Vec::new();

        println!(" Harvesting from {}...", root.display());

        for entry in WalkDir::new(root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "nix"))
        {
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                if let Some(fragment) = self.extract_high_signal_fragment(&content) {
                    if let Some(prompt) = self.nix_kg.reverse_prompt(&fragment) {
                        let verdict = self.scorer.score(&fragment, &fragment);
                        if verdict.parse_error.is_none() {
                            pairs.push(InverseHarvestPair {
                                prompt,
                                code: fragment,
                                intent: "service".to_string(),
                                channels: vec![0.0; 17], // placeholder
                                iterations: 0,
                                repair_steps: 0,
                                holdout: false,
                                valence: 0.0,
                                arousal: 0.0,
                                source_file: entry.path().to_string_lossy().to_string(),
                            });
                        }
                    }
                }
            }
        }

        let deduped = self.dedup_and_diversify(pairs);
        deduped.into_iter().take(self.max_pairs).collect()
    }

    /// Deduplication + diversity filter pass
    fn dedup_and_diversify(&self, pairs: Vec<InverseHarvestPair>) -> Vec<InverseHarvestPair> {
        if pairs.is_empty() {
            return pairs;
        }

        // 1. Exact dedup via composite key (prompt + normalized code)
        let mut seen_exact: HashSet<String> = HashSet::new();
        let mut unique: Vec<InverseHarvestPair> = Vec::new();
        for p in pairs {
            let norm_code = p.code.trim().replace(|c: char| !c.is_alphanumeric(), " ");
            let key = format!("{}|{}", p.prompt.trim(), norm_code);
            if seen_exact.insert(key) {
                unique.push(p);
            }
        }

        // 2. Diversity filter: Jaccard on prompt tokens
        let mut diverse: Vec<InverseHarvestPair> = Vec::new();
        let mut prompt_tokens: Vec<HashSet<String>> = Vec::new();

        for p in unique {
            let tokens: HashSet<String> = p
                .prompt
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();

            let mut is_diverse = true;
            for existing in &prompt_tokens {
                let inter = tokens.intersection(existing).count() as f32;
                let union = tokens.union(existing).count() as f32;
                if union > 0.0 {
                    let jaccard = inter / union;
                    if jaccard > self.min_diversity_threshold {
                        is_diverse = false;
                        break;
                    }
                }
            }

            if is_diverse {
                diverse.push(p);
                prompt_tokens.push(tokens);
            }
        }

        diverse
    }

    fn extract_high_signal_fragment(&self, source: &str) -> Option<String> {
        let lines: Vec<&str> = source.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let lower = line.to_lowercase();
            // Match options.services.X or services.X.enable
            if (lower.contains("services.")
                || lower.contains("hardware.")
                || lower.contains("options.services."))
                && (lower.contains(".enable") || lower.contains(" = {"))
            {
                let start = i.saturating_sub(2);
                let end = (i + 15).min(lines.len());
                let fragment = lines[start..end].join("\n");

                if fragment.contains("=") || fragment.contains("{") {
                    return Some(fragment);
                }
            }
        }
        None
    }

    pub fn write_jsonl(&self, pairs: &[InverseHarvestPair], path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        for p in pairs {
            writeln!(writer, "{}", serde_json::to_string(p)?)?;
        }
        Ok(())
    }
}

impl Default for InverseHarvester {
    fn default() -> Self {
        Self::new()
    }
}
