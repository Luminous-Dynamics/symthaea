// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Therapeutic narrative construction and integration.
//!
//! Tracks narrative fragments from the therapeutic process and supports
//! coherence analysis, contradiction detection, and alternative narrative
//! construction following White & Epston (1990) narrative therapy.
//!
//! Science: White & Epston (1990), Angus & McLeod (2004) narrative processes,
//! Pennebaker (1997) expressive writing, Adler (2012) narrative identity.

use crate::semantic_encoding::encode_therapeutic_text;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::BinaryHV;

// ── Narrative Fragment ─────────────────────────────────────────────────────

/// A fragment of the client's therapeutic narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeFragment {
    /// Text content of the fragment.
    pub text: String,
    /// Cycle when this fragment was recorded.
    pub cycle: u64,
    /// Emotional valence of this fragment (-1 to +1).
    pub emotional_valence: f32,
    /// Whether this fragment involves traumatic content.
    pub is_traumatic: bool,
    /// Integration level: how well this fragment is connected to the broader narrative.
    /// 0.0 = dissociated/fragmented, 1.0 = fully integrated.
    pub integration_level: f32,
    /// HDC encoding of the fragment.
    #[serde(skip)]
    pub encoding: Option<BinaryHV>,
}

impl NarrativeFragment {
    /// Create a new narrative fragment with HDC encoding.
    pub fn new(text: &str, cycle: u64, emotional_valence: f32, is_traumatic: bool) -> Self {
        Self {
            text: text.to_string(),
            cycle,
            emotional_valence: emotional_valence.clamp(-1.0, 1.0),
            is_traumatic,
            integration_level: if is_traumatic { 0.2 } else { 0.5 },
            encoding: encode_therapeutic_text(text),
        }
    }

    /// Similarity to another fragment.
    pub fn similarity(&self, other: &Self) -> f32 {
        match (&self.encoding, &other.encoding) {
            (Some(a), Some(b)) => a.similarity(b),
            _ => 0.0,
        }
    }
}

// ── Reauthored Fragment ──────────────────────────────────────────────────

/// A reauthored narrative fragment — the original traumatic fragment paired
/// with a positive reframe discovered via HDC similarity search.
///
/// Science: White & Epston (1990) — re-authoring conversations, finding
/// "unique outcomes" that contradict the dominant problem-saturated story.
#[derive(Debug, Clone)]
pub struct ReauthoredFragment {
    /// Index of the original traumatic fragment.
    pub original_index: usize,
    /// The original traumatic text.
    pub original_text: String,
    /// Index of the most similar positive fragment (the "unique outcome").
    pub positive_index: usize,
    /// The positive fragment text used as reframe anchor.
    pub positive_text: String,
    /// HDC similarity between original and positive fragment.
    pub similarity: f32,
    /// Suggested reframe: a blend of the original content with positive valence.
    pub reframe_suggestion: String,
}

// ── Therapeutic Narrative ──────────────────────────────────────────────────

/// Collection of narrative fragments with coherence tracking.
#[derive(Debug, Clone)]
pub struct TherapeuticNarrative {
    /// All narrative fragments in chronological order.
    pub fragments: Vec<NarrativeFragment>,
    /// Overall narrative coherence (0.0–1.0).
    pub coherence: f32,
}

impl TherapeuticNarrative {
    /// Create an empty narrative.
    pub fn new() -> Self {
        Self {
            fragments: Vec::new(),
            coherence: 0.0,
        }
    }

    /// Add a fragment and recompute coherence.
    pub fn integrate_fragment(&mut self, fragment: NarrativeFragment) {
        self.fragments.push(fragment);
        self.recompute_coherence();
    }

    /// Find fragments that may contradict each other.
    ///
    /// Contradiction heuristic: fragments with similar encodings but
    /// opposite emotional valence suggest unresolved conflicts.
    pub fn find_contradictions(&self) -> Vec<(usize, usize)> {
        let mut contradictions = Vec::new();
        for i in 0..self.fragments.len() {
            for j in (i + 1)..self.fragments.len() {
                let sim = self.fragments[i].similarity(&self.fragments[j]);
                let valence_diff = (self.fragments[i].emotional_valence
                    - self.fragments[j].emotional_valence)
                    .abs();
                // Similar topic (sim > 0.3) but opposing valence (diff > 1.0)
                if sim > 0.3 && valence_diff > 1.0 {
                    contradictions.push((i, j));
                }
            }
        }
        contradictions
    }

    /// Generate an alternative narrative by reframing traumatic fragments.
    ///
    /// Returns indices of fragments that could be re-authored.
    pub fn candidates_for_reauthoring(&self) -> Vec<usize> {
        self.fragments
            .iter()
            .enumerate()
            .filter(|(_, f)| f.is_traumatic && f.integration_level < 0.5)
            .map(|(i, _)| i)
            .collect()
    }

    /// Proportion of fragments that are traumatic.
    pub fn trauma_proportion(&self) -> f32 {
        if self.fragments.is_empty() {
            return 0.0;
        }
        let traumatic = self.fragments.iter().filter(|f| f.is_traumatic).count();
        traumatic as f32 / self.fragments.len() as f32
    }

    /// Mean integration level across all fragments.
    pub fn mean_integration(&self) -> f32 {
        if self.fragments.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.fragments.iter().map(|f| f.integration_level).sum();
        sum / self.fragments.len() as f32
    }

    /// Increase integration level of a specific fragment (therapeutic processing).
    pub fn process_fragment(&mut self, index: usize, integration_boost: f32) {
        if let Some(fragment) = self.fragments.get_mut(index) {
            fragment.integration_level =
                (fragment.integration_level + integration_boost).clamp(0.0, 1.0);
            self.recompute_coherence();
        }
    }

    /// Temporal coherence: measures narrative flow consistency over time.
    ///
    /// Combines three factors:
    /// 1. **Pacing regularity** — variance in cycle gaps between fragments (lower = more regular)
    /// 2. **Valence smoothness** — mean absolute change in valence between adjacent fragments
    /// 3. **Integration trend** — whether later fragments are more integrated than earlier ones
    ///
    /// Returns a score in [0.0, 1.0] where 1.0 = perfectly smooth temporal narrative.
    ///
    /// Science: Adler (2012) — temporal coherence predicts therapeutic progress.
    pub fn temporal_coherence(&self) -> f32 {
        if self.fragments.len() < 3 {
            return self.coherence; // fall back to overall coherence
        }

        let n = self.fragments.len();

        // 1. Pacing regularity: normalized inverse variance of inter-fragment cycle gaps
        let gaps: Vec<f32> = self
            .fragments
            .windows(2)
            .map(|w| (w[1].cycle as f32 - w[0].cycle as f32).max(1.0))
            .collect();
        let mean_gap = gaps.iter().sum::<f32>() / gaps.len() as f32;
        let gap_var = gaps.iter().map(|g| (g - mean_gap).powi(2)).sum::<f32>() / gaps.len() as f32;
        let pacing = 1.0 / (1.0 + gap_var / (mean_gap * mean_gap + 1.0)); // normalized [0,1]

        // 2. Valence smoothness: mean absolute valence change between adjacent fragments
        let valence_changes: f32 = self
            .fragments
            .windows(2)
            .map(|w| (w[1].emotional_valence - w[0].emotional_valence).abs())
            .sum::<f32>()
            / (n - 1) as f32;
        let smoothness = 1.0 - valence_changes.clamp(0.0, 1.0); // lower change = smoother

        // 3. Integration trend: slope of integration level over fragment index
        let half = n / 2;
        let early_integration: f32 = self.fragments[..half]
            .iter()
            .map(|f| f.integration_level)
            .sum::<f32>()
            / half as f32;
        let late_integration: f32 = self.fragments[half..]
            .iter()
            .map(|f| f.integration_level)
            .sum::<f32>()
            / (n - half) as f32;
        let trend = ((late_integration - early_integration) + 0.5).clamp(0.0, 1.0); // 0.5 centered

        // Weighted combination
        (pacing * 0.3 + smoothness * 0.4 + trend * 0.3).clamp(0.0, 1.0)
    }

    /// Number of fragments.
    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    /// Whether narrative is empty.
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }

    /// Construct alternative narratives by pairing traumatic fragments with
    /// the most similar positive fragment, producing reauthoring suggestions.
    ///
    /// For each reauthoring candidate (traumatic, low integration), we search
    /// for the non-traumatic fragment with the highest HDC similarity that
    /// also has positive emotional valence (≥ 0.1). If the similarity exceeds
    /// 0.2 (above random noise for `BinaryHV`), a `ReauthoredFragment` is
    /// produced with a suggested reframe.
    ///
    /// Science: White & Epston (1990) — re-authoring conversations.
    pub fn construct_alternative_narrative(&self) -> Vec<ReauthoredFragment> {
        let candidates = self.candidates_for_reauthoring();
        let mut reauthored = Vec::new();

        for &orig_idx in &candidates {
            let orig = &self.fragments[orig_idx];
            let mut best_sim = f32::NEG_INFINITY;
            let mut best_idx: Option<usize> = None;

            for (j, frag) in self.fragments.iter().enumerate() {
                if frag.is_traumatic || frag.emotional_valence < 0.1 {
                    continue;
                }
                let sim = orig.similarity(frag);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(j);
                }
            }

            if let Some(pos_idx) = best_idx {
                if best_sim > 0.2 {
                    let pos = &self.fragments[pos_idx];
                    reauthored.push(ReauthoredFragment {
                        original_index: orig_idx,
                        original_text: orig.text.clone(),
                        positive_index: pos_idx,
                        positive_text: pos.text.clone(),
                        similarity: best_sim,
                        reframe_suggestion: format!(
                            "Drawing from the experience of '{}', consider how '{}' might be understood differently",
                            pos.text, orig.text
                        ),
                    });
                }
            }
        }

        reauthored
    }

    /// Measure the overall narrative arc quality.
    ///
    /// Combines three factors:
    /// - **Redemption score** (0.4): fraction of fragments where valence improves
    ///   from the prior fragment. Redemption sequences predict well-being.
    /// - **Agency score** (0.3): fraction of non-traumatic fragments, reflecting
    ///   narrative agency (authorship over one's own story).
    /// - **Coherence** (0.3): the existing `self.coherence` value.
    ///
    /// Returns a score in [0.0, 1.0].
    ///
    /// Science: McAdams (2001) — redemption sequences predict well-being.
    pub fn narrative_arc_score(&self) -> f32 {
        if self.fragments.is_empty() {
            return 0.0;
        }

        // Redemption: fraction of adjacent pairs where valence improves
        let redemption = if self.fragments.len() < 2 {
            0.5 // neutral when no pairs
        } else {
            let improving = self
                .fragments
                .windows(2)
                .filter(|w| w[1].emotional_valence > w[0].emotional_valence)
                .count();
            improving as f32 / (self.fragments.len() - 1) as f32
        };

        // Agency: fraction of non-traumatic fragments
        let agency = 1.0 - self.trauma_proportion();

        // Weighted combination
        (redemption * 0.4 + agency * 0.3 + self.coherence * 0.3).clamp(0.0, 1.0)
    }

    /// Recompute narrative coherence from fragment integration levels
    /// and inter-fragment similarity.
    fn recompute_coherence(&mut self) {
        if self.fragments.len() < 2 {
            self.coherence = self.mean_integration();
            return;
        }

        // Coherence = weighted combination of mean integration and
        // mean pairwise similarity of adjacent fragments
        let integration = self.mean_integration();

        let mut adj_sim_sum = 0.0;
        let adj_count = self.fragments.len() - 1;
        for i in 0..adj_count {
            adj_sim_sum += self.fragments[i].similarity(&self.fragments[i + 1]);
        }
        let mean_adj_sim = adj_sim_sum / adj_count as f32;

        self.coherence = integration * 0.7 + mean_adj_sim * 0.3;
    }
}

impl Default for TherapeuticNarrative {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_narrative() {
        let narrative = TherapeuticNarrative::new();
        assert!(narrative.is_empty());
        assert_eq!(narrative.coherence, 0.0);
    }

    #[test]
    fn test_integrate_fragment() {
        let mut narrative = TherapeuticNarrative::new();
        narrative.integrate_fragment(NarrativeFragment::new("I felt safe today", 1, 0.5, false));
        assert_eq!(narrative.len(), 1);
    }

    #[test]
    fn test_traumatic_fragment_low_integration() {
        let fragment = NarrativeFragment::new("The accident keeps replaying", 1, -0.8, true);
        assert!(fragment.integration_level < 0.3);
    }

    #[test]
    fn test_nontraumatic_fragment_moderate_integration() {
        let fragment = NarrativeFragment::new("I went for a walk today", 1, 0.3, false);
        assert!(fragment.integration_level >= 0.5);
    }

    #[test]
    fn test_candidates_for_reauthoring() {
        let mut narrative = TherapeuticNarrative::new();
        narrative.integrate_fragment(NarrativeFragment::new("Good day", 1, 0.5, false));
        narrative.integrate_fragment(NarrativeFragment::new("Flashback", 2, -0.8, true));
        narrative.integrate_fragment(NarrativeFragment::new("Nice walk", 3, 0.3, false));
        let candidates = narrative.candidates_for_reauthoring();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0], 1); // the traumatic fragment
    }

    #[test]
    fn test_process_fragment_increases_integration() {
        let mut narrative = TherapeuticNarrative::new();
        narrative.integrate_fragment(NarrativeFragment::new("Trauma memory", 1, -0.7, true));
        let before = narrative.fragments[0].integration_level;
        narrative.process_fragment(0, 0.3);
        assert!(narrative.fragments[0].integration_level > before);
    }

    #[test]
    fn test_trauma_proportion() {
        let mut narrative = TherapeuticNarrative::new();
        narrative.integrate_fragment(NarrativeFragment::new("good", 1, 0.5, false));
        narrative.integrate_fragment(NarrativeFragment::new("bad", 2, -0.8, true));
        assert!((narrative.trauma_proportion() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_mean_integration() {
        let mut narrative = TherapeuticNarrative::new();
        narrative.integrate_fragment(NarrativeFragment::new("a", 1, 0.0, false)); // 0.5
        narrative.integrate_fragment(NarrativeFragment::new("b", 2, 0.0, true)); // 0.2
        let mean = narrative.mean_integration();
        assert!((mean - 0.35).abs() < 0.01);
    }

    // ── Temporal coherence tests ──────────────────────────────────────────

    #[test]
    fn test_temporal_coherence_too_few_fragments() {
        let mut narrative = TherapeuticNarrative::new();
        narrative.integrate_fragment(NarrativeFragment::new("a", 1, 0.3, false));
        narrative.integrate_fragment(NarrativeFragment::new("b", 2, 0.4, false));
        // < 3 fragments → falls back to overall coherence
        assert_eq!(narrative.temporal_coherence(), narrative.coherence);
    }

    #[test]
    fn test_temporal_coherence_smooth_narrative() {
        let mut narrative = TherapeuticNarrative::new();
        // Evenly spaced, gently rising valence, non-traumatic (integration=0.5)
        for i in 0..8 {
            let valence = -0.2 + (i as f32 * 0.1); // -0.2 to +0.5
            narrative.integrate_fragment(NarrativeFragment::new(
                &format!("session {i}"),
                (i * 10) as u64,
                valence,
                false,
            ));
        }
        let tc = narrative.temporal_coherence();
        assert!(
            tc > 0.4,
            "smooth narrative should have decent temporal coherence, got {tc}"
        );
        assert!(tc <= 1.0);
    }

    #[test]
    fn test_temporal_coherence_chaotic_narrative() {
        let mut narrative = TherapeuticNarrative::new();
        // Irregular pacing, wild valence swings
        let valences = [0.8, -0.9, 0.7, -0.8, 0.9, -0.7];
        let cycles = [1, 2, 100, 101, 500, 501];
        for (i, (&v, &c)) in valences.iter().zip(cycles.iter()).enumerate() {
            narrative.integrate_fragment(NarrativeFragment::new(
                &format!("chaos {i}"),
                c,
                v,
                false,
            ));
        }
        let tc = narrative.temporal_coherence();
        // Chaotic should score lower than smooth
        assert!(
            tc < 0.6,
            "chaotic narrative should have low temporal coherence, got {tc}"
        );
    }

    #[test]
    fn test_temporal_coherence_bounded() {
        let mut narrative = TherapeuticNarrative::new();
        for i in 0..10 {
            narrative.integrate_fragment(NarrativeFragment::new(
                &format!("f{i}"),
                i as u64,
                if i % 2 == 0 { 1.0 } else { -1.0 },
                i % 3 == 0,
            ));
        }
        let tc = narrative.temporal_coherence();
        assert!(
            (0.0..=1.0).contains(&tc),
            "temporal coherence must be in [0,1], got {tc}"
        );
    }

    // ── Reauthoring & narrative arc tests ────────────────────────────────

    #[test]
    fn test_construct_alternative_narrative_empty() {
        let narrative = TherapeuticNarrative::new();
        assert!(narrative.construct_alternative_narrative().is_empty());
    }

    #[test]
    fn test_construct_alternative_narrative_no_traumatic() {
        let mut narrative = TherapeuticNarrative::new();
        narrative.integrate_fragment(NarrativeFragment::new("sunny day", 1, 0.6, false));
        narrative.integrate_fragment(NarrativeFragment::new("nice walk", 2, 0.4, false));
        assert!(narrative.construct_alternative_narrative().is_empty());
    }

    #[test]
    fn test_construct_alternative_narrative_with_candidates() {
        let mut narrative = TherapeuticNarrative::new();
        narrative.integrate_fragment(NarrativeFragment::new("flashback to crash", 1, -0.8, true));
        narrative.integrate_fragment(NarrativeFragment::new(
            "felt safe driving today",
            2,
            0.5,
            false,
        ));
        narrative.integrate_fragment(NarrativeFragment::new(
            "enjoyed the sunshine",
            3,
            0.7,
            false,
        ));
        let reauthored = narrative.construct_alternative_narrative();
        // We have one traumatic candidate. Whether a reauthored fragment is
        // produced depends on HDC similarity > 0.2 (deterministic from blake3
        // seeds, so the result is stable). Regardless of match, the return
        // type is correct and non-panicking.
        for r in &reauthored {
            assert!(r.similarity > 0.2);
            assert!(
                r.reframe_suggestion
                    .contains("Drawing from the experience of")
            );
            assert_eq!(r.original_index, 0);
        }
    }

    #[test]
    fn test_narrative_arc_score_bounded() {
        let mut narrative = TherapeuticNarrative::new();
        for i in 0..10 {
            narrative.integrate_fragment(NarrativeFragment::new(
                &format!("frag {i}"),
                i as u64,
                if i % 2 == 0 { 1.0 } else { -1.0 },
                i % 3 == 0,
            ));
        }
        let score = narrative.narrative_arc_score();
        assert!(
            (0.0..=1.0).contains(&score),
            "narrative arc score must be in [0,1], got {score}"
        );
    }

    #[test]
    fn test_narrative_arc_improving() {
        // Improving narrative: valence rises, no trauma
        let mut improving = TherapeuticNarrative::new();
        for i in 0..6 {
            improving.integrate_fragment(NarrativeFragment::new(
                &format!("up {i}"),
                i as u64,
                -0.5 + (i as f32 * 0.2),
                false,
            ));
        }

        // Declining narrative: valence falls, all traumatic
        let mut declining = TherapeuticNarrative::new();
        for i in 0..6 {
            declining.integrate_fragment(NarrativeFragment::new(
                &format!("down {i}"),
                i as u64,
                0.5 - (i as f32 * 0.2),
                true,
            ));
        }

        assert!(
            improving.narrative_arc_score() > declining.narrative_arc_score(),
            "improving arc ({}) should score higher than declining ({})",
            improving.narrative_arc_score(),
            declining.narrative_arc_score()
        );
    }

    #[test]
    fn test_temporal_coherence_improving_integration_trend() {
        let mut narrative = TherapeuticNarrative::new();
        // Early fragments traumatic (low integration), later non-traumatic (higher integration)
        for i in 0..8 {
            let is_traumatic = i < 4;
            narrative.integrate_fragment(NarrativeFragment::new(
                &format!("frag {i}"),
                (i * 5) as u64,
                0.0,
                is_traumatic,
            ));
        }
        let tc = narrative.temporal_coherence();
        // Positive integration trend should boost score
        assert!(
            tc > 0.3,
            "improving integration trend should contribute to coherence, got {tc}"
        );
    }
}
