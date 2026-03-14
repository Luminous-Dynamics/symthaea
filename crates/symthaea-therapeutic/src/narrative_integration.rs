//! Therapeutic narrative construction and integration.
//!
//! Tracks narrative fragments from the therapeutic process and supports
//! coherence analysis, contradiction detection, and alternative narrative
//! construction following White & Epston (1990) narrative therapy.
//!
//! Science: White & Epston (1990), Angus & McLeod (2004) narrative processes,
//! Pennebaker (1997) expressive writing, Adler (2012) narrative identity.

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
        let hash = blake3::hash(text.as_bytes());
        let seed = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        Self {
            text: text.to_string(),
            cycle,
            emotional_valence: emotional_valence.clamp(-1.0, 1.0),
            is_traumatic,
            integration_level: if is_traumatic { 0.2 } else { 0.5 },
            encoding: Some(BinaryHV::random(seed)),
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
                let valence_diff =
                    (self.fragments[i].emotional_valence - self.fragments[j].emotional_valence)
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

    /// Number of fragments.
    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    /// Whether narrative is empty.
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
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
}
