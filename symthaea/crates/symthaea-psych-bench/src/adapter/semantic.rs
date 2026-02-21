//! Semantic encoding adapter for creativity benchmarks.
//!
//! Unlike the generic `ScenarioAdapter`, this adapter plants explicit semantic
//! links between RAT cue words and their solution, ensuring that the bundled
//! cue representation is more similar to the solution than to distractors.
//! This models real semantic associations rather than relying on random HDC geometry.

use super::StimulusAdapter;
use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

/// A word stimulus for semantic encoding.
pub struct Word(pub String);

/// Semantic adapter that creates shared "link" hypervectors between
/// related words (cues and their RAT solution), making associative
/// retrieval possible via HDC similarity.
pub struct SemanticScenarioAdapter {
    word_hvs: HashMap<String, ContinuousHV>,
    dim: usize,
    base_seed: u64,
}

/// RAT triad data passed to the adapter constructor.
pub struct RatTriadData {
    pub cues: [String; 3],
    pub solution: String,
    pub distractors: Vec<String>,
}

impl SemanticScenarioAdapter {
    /// Build an adapter pre-loaded with semantic links for RAT triads.
    ///
    /// For each triad:
    /// - 3 "link" HVs are created (shared between cue[i] and solution)
    /// - Solution HV = weighted_bundle([base, link0, link1, link2], [0.4, 0.2, 0.2, 0.2])
    /// - Cue[i] HV = weighted_bundle([base_cue_i, link_i], [0.6, 0.4])
    /// - Distractor HVs = pure random (no shared links)
    pub fn for_rat(triads: &[RatTriadData], dim: usize, seed: u64) -> Self {
        let mut word_hvs = HashMap::new();
        let mut next_seed = seed ^ 0x9E3779B97F4A7C15;

        let mut advance = || -> u64 {
            next_seed ^= next_seed << 13;
            next_seed ^= next_seed >> 7;
            next_seed ^= next_seed << 17;
            next_seed
        };

        for triad in triads {
            // Generate 3 semantic link HVs (shared between cue and solution)
            let links: Vec<ContinuousHV> = (0..3)
                .map(|_| ContinuousHV::random(dim, advance()))
                .collect();

            // Solution HV: base + all 3 links
            let base_sol = ContinuousHV::random(dim, advance());
            let sol_refs: Vec<&ContinuousHV> = std::iter::once(&base_sol)
                .chain(links.iter())
                .collect();
            let sol_hv = ContinuousHV::weighted_bundle(
                &sol_refs,
                &[0.4, 0.2, 0.2, 0.2],
            );
            word_hvs.insert(triad.solution.clone(), sol_hv);

            // Cue HVs: each shares one link with solution
            for (i, cue) in triad.cues.iter().enumerate() {
                let base_cue = ContinuousHV::random(dim, advance());
                let cue_refs: Vec<&ContinuousHV> = vec![&base_cue, &links[i]];
                let cue_hv = ContinuousHV::weighted_bundle(
                    &cue_refs,
                    &[0.6, 0.4],
                );
                word_hvs.insert(cue.clone(), cue_hv);
            }

            // Distractor HVs: pure random (no shared links)
            for d in &triad.distractors {
                if !word_hvs.contains_key(d) {
                    word_hvs.insert(d.clone(), ContinuousHV::random(dim, advance()));
                }
            }
        }

        Self {
            word_hvs,
            dim,
            base_seed: seed,
        }
    }
}

impl StimulusAdapter for SemanticScenarioAdapter {
    type Stimulus = Word;

    fn encode(&self, stimulus: &Word, dim: usize) -> ContinuousHV {
        if let Some(hv) = self.word_hvs.get(&stimulus.0) {
            hv.clone()
        } else {
            // Fallback: hash-based encoding for unknown words
            let hash = stimulus.0.bytes().fold(
                self.base_seed ^ 0x517CC1B727220A95,
                |acc, b| acc.wrapping_mul(0x100000001B3).wrapping_add(b as u64),
            );
            ContinuousHV::random(dim.max(self.dim), hash)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_triads() -> Vec<RatTriadData> {
        vec![RatTriadData {
            cues: [
                "cottage".to_string(),
                "swiss".to_string(),
                "cake".to_string(),
            ],
            solution: "cheese".to_string(),
            distractors: vec![
                "bread".to_string(),
                "milk".to_string(),
                "house".to_string(),
            ],
        }]
    }

    #[test]
    fn test_solution_ranks_higher() {
        let triads = test_triads();
        let adapter = SemanticScenarioAdapter::for_rat(&triads, 512, 42);

        let cue_hvs: Vec<ContinuousHV> = triads[0]
            .cues
            .iter()
            .map(|c| adapter.encode(&Word(c.clone()), 512))
            .collect();
        let bundle = ContinuousHV::bundle_owned(&cue_hvs);

        let sol_hv = adapter.encode(&Word(triads[0].solution.clone()), 512);
        let sol_sim = bundle.similarity(&sol_hv);

        for d in &triads[0].distractors {
            let d_hv = adapter.encode(&Word(d.clone()), 512);
            let d_sim = bundle.similarity(&d_hv);
            assert!(
                sol_sim > d_sim,
                "Solution should rank higher: sol_sim={}, dist_sim={} for '{}'",
                sol_sim, d_sim, d
            );
        }
    }

    #[test]
    fn test_deterministic() {
        let triads = test_triads();
        let a1 = SemanticScenarioAdapter::for_rat(&triads, 256, 99);
        let a2 = SemanticScenarioAdapter::for_rat(&triads, 256, 99);

        let h1 = a1.encode(&Word("cheese".to_string()), 256);
        let h2 = a2.encode(&Word("cheese".to_string()), 256);
        assert!(
            (h1.similarity(&h2) - 1.0).abs() < 1e-6,
            "Same seed should produce identical HVs"
        );
    }

    #[test]
    fn test_unknown_word_fallback() {
        let triads = test_triads();
        let adapter = SemanticScenarioAdapter::for_rat(&triads, 256, 42);
        let hv = adapter.encode(&Word("xylophone".to_string()), 256);
        // Should produce a valid HV, not panic
        assert_eq!(hv.values.len(), 256);
        let norm: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "Fallback HV should be non-zero");
    }
}
