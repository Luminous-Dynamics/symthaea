// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared Eight Harmonies basis vectors for moral geometry.
//!
//! Provides semantically grounded 16,384-dim `ContinuousHV` basis vectors for
//! the Eight Harmonies, built by encoding keyword sets through `TextHdcEncoder`.
//! These replace the random embeddings previously used in `HarmoniesIntegrator`
//! and unify with the harmony projection in `MoralTopology`.
//!
//! Also provides moral free energy computation following the Free Energy
//! Principle (Friston 2010): a moral system that cannot predict its own
//! consequences accumulates surprise and destabilizes. Moral free energy
//! quantifies the divergence between expected and observed harmony coordinates.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use symthaea_types::{Harmony, N_HARMONIES};

use super::moral_text_encoder::TextHdcEncoder;

/// Keyword sets for each harmony (order matches `Harmony::all()`).
///
/// Each harmony is encoded as the bundle of its keyword set, producing
/// a ContinuousHV that responds to semantic similarity rather than
/// arbitrary random projection.
pub const HARMONY_KEYWORDS: [&str; N_HARMONIES] = [
    // ResonantCoherence — Integration-Knowing
    // Expanded: structural verbs, systems thinking, coherence vocabulary
    "integrate harmonize unify coherent order luminous resonant balance alignment wholeness \
     synthesize reconcile orchestrate coordinate converge consolidate weave bridge compose \
     holistic systemic pattern structure organize resolve",
    // PanSentientFlourishing — Care-Knowing
    // Expanded: welfare, stewardship, dignity, embodied care actions
    "help support care benefit serve protect nurture compassion flourishing kindness love \
     nourish empower steward safeguard wellbeing dignity welfare heal shelter advocate \
     uplift sustain cherish tend cultivate mercy",
    // IntegralWisdom — Truth-Knowing
    // Expanded: epistemic verbs, discernment, critical thinking
    "learn understand wisdom knowledge insight intelligence truth awareness knowing embodied \
     discern verify evidence reason investigate clarify comprehend analyze scrutinize \
     epistemology rigor validate question depth",
    // InfinitePlay — Creative-Knowing
    // Expanded: artistic, generative, surprise-seeking vocabulary
    "create explore play discover experiment joy creativity generativity novelty imagination \
     improvise invent wonder curious delight surprise innovate design compose inspire \
     artistic whimsical spontaneous serendipity dance",
    // UniversalInterconnectedness — Relational-Knowing
    // Expanded: ecological, systemic, intersubjective vocabulary
    "connect share collaborate together community unity empathy resonance interconnected belonging \
     relationship symbiosis ecology network entangle solidarity kinship companionship \
     fellowship interdependent weave participate collective",
    // SacredReciprocity — Exchange-Knowing
    // Expanded: gift economy, mutualism, generative trust
    "give share contribute reciprocate exchange generous flow mutual upliftment trust \
     gift gratitude steward redistribute commons abundance mutualism cooperate \
     replenish nourish return sustain covenant",
    // EvolutionaryProgression — Developmental-Knowing
    // Expanded: emergence, transformation, developmental vocabulary
    "grow evolve improve progress develop transcend becoming evolution advancement consciousness \
     emerge transform mature unfold metamorphosis actualize deepen complexify differentiate \
     integrate scaffold developmental spiral ascend",
    // SacredStillness — Apophatic-Knowing
    // Expanded: contemplative, receptive, liminal vocabulary
    "stillness silence rest surrender void release contemplation emptiness peace presence apophatic \
     meditate listen absorb receptive liminal pause sabbath fallow dormant gestate \
     breath ground center quiet spacious",
];

/// Eight Harmony basis vectors for projecting moral scenarios into
/// semantically meaningful 8D coordinates.
///
/// Each basis vector is built by encoding a harmony's keyword set through
/// `TextHdcEncoder`, so projection onto (e.g.) `PanSentientFlourishing` is
/// high when the scenario's vocabulary overlaps with "help", "care", etc.
#[derive(Debug, Clone)]
pub struct HarmonyBasis {
    /// One ContinuousHV per Harmony (indexed by canonical `Harmony::all()` order).
    pub vectors: [ContinuousHV; N_HARMONIES],
    /// HDC dimension.
    pub dim: usize,
}

impl HarmonyBasis {
    /// Build a new harmony basis at the given dimension.
    pub fn new(dim: usize) -> Self {
        let encoder = TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2);
        let vectors: Vec<ContinuousHV> = HARMONY_KEYWORDS
            .iter()
            .map(|kw| encoder.encode(kw))
            .collect();
        // HARMONY_KEYWORDS is a compile-time [&str; N_HARMONIES], so this always produces
        // exactly N_HARMONIES vectors. Use debug_assert for loud failure in dev builds.
        debug_assert_eq!(
            vectors.len(),
            N_HARMONIES,
            "expected {N_HARMONIES} harmony vectors, got {}",
            vectors.len()
        );
        Self {
            vectors: vectors.try_into().unwrap_or_else(|v: Vec<ContinuousHV>| {
                // Fallback: re-encode with padding/truncation to guarantee [_; N_HARMONIES]
                let mut arr = [(); N_HARMONIES].map(|_| ContinuousHV::zero(dim));
                for (i, hv) in v.into_iter().enumerate().take(N_HARMONIES) {
                    arr[i] = hv;
                }
                arr
            }),
            dim,
        }
    }

    /// Build a harmony basis from pre-computed dense vectors.
    ///
    /// When a dense encoder (Qwen3, BGE-M3, etc.) is available, the caller
    /// encodes `HARMONY_KEYWORDS` once at init time and projects each through
    /// `HdcBridge::project_continuous()` → `ContinuousHV`. This ensures the
    /// basis vectors live in the same JL-projected semantic subspace as
    /// scenario embeddings, eliminating the domain mismatch between n-gram
    /// and contextual encodings.
    ///
    /// Falls back to `HarmonyBasis::new(dim)` if dimension mismatches.
    pub fn with_dense_vectors(dim: usize, vectors: [ContinuousHV; N_HARMONIES]) -> Self {
        let valid = vectors.iter().all(|v| v.dim() == dim);
        if !valid {
            tracing::warn!(
                "Dense harmony vectors have wrong dimension (expected {dim}), \
                 falling back to TextHdcEncoder"
            );
            return Self::new(dim);
        }
        Self { vectors, dim }
    }

    /// Project a scenario HV onto the N_HARMONIES harmony axes (cosine similarity).
    ///
    /// Returns coordinates in `[-1, 1]^N_HARMONIES` where each component is the
    /// cosine similarity between the scenario and the corresponding harmony
    /// basis vector. The result is a point in the moral manifold.
    pub fn project(&self, hv: &ContinuousHV) -> [f64; N_HARMONIES] {
        let mut coords = [0.0f64; N_HARMONIES];
        for (i, basis) in self.vectors.iter().enumerate() {
            coords[i] = hv.similarity(basis) as f64;
        }
        coords
    }

    /// Access the basis vector for a specific harmony.
    pub fn vector(&self, harmony: Harmony) -> &ContinuousHV {
        let idx = Harmony::all()
            .iter()
            .position(|h| *h == harmony)
            .unwrap_or(0);
        &self.vectors[idx]
    }

    /// Compute the moral free energy of a scenario projection.
    ///
    /// Moral free energy measures how surprising a scenario is relative to an
    /// expected distribution over the N_HARMONIES harmony coordinates. Following
    /// the Free Energy Principle (Friston 2010), a viable moral system minimizes
    /// its average free energy — meaning its actions are predictable and
    /// consistent within its moral manifold.
    ///
    /// F = D_KL(q || p) + H(q)
    ///
    /// where:
    /// - q(h) = softmax of current scenario coordinates (observed distribution)
    /// - p(h) = softmax of expected coordinates (prior distribution)
    /// - D_KL = KL divergence from prior to observed
    /// - H(q) = entropy of observed distribution
    ///
    /// Low F → scenario is consistent with moral history (predictable).
    /// High F → moral surprise (novel moral territory or incoherent stance).
    pub fn moral_free_energy(
        &self,
        scenario_coords: &[f64; N_HARMONIES],
        expected_coords: &[f64; N_HARMONIES],
        temperature: f64,
    ) -> MoralFreeEnergy {
        let inv_temp = 1.0 / temperature.max(0.01);

        // Softmax over harmony coordinates → probability distributions
        let q = softmax_n(scenario_coords, inv_temp);
        let p = softmax_n(expected_coords, inv_temp);

        // KL divergence: D_KL(q || p) = sum_i q_i * ln(q_i / p_i)
        let kl_divergence = kl_div_n(&q, &p);

        // Entropy: H(q) = -sum_i q_i * ln(q_i)
        let entropy = entropy_n(&q);

        // Free energy: F = D_KL + H
        let free_energy = kl_divergence + entropy;

        // Moral surprise: -log p(most-active harmony)
        let dominant_idx = scenario_coords
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let surprise = -p[dominant_idx].max(1e-12).ln();

        MoralFreeEnergy {
            free_energy,
            kl_divergence,
            entropy,
            surprise,
            dominant_harmony_idx: dominant_idx as u8,
            scenario_distribution: q,
            prior_distribution: p,
        }
    }
}

/// Input-contract failure while computing a checked Love Coherence assessment.
#[derive(Debug, Clone, PartialEq)]
pub enum LoveCoherenceError {
    /// Coordinate is non-finite or outside the documented [-1, 1] range.
    InvalidCoordinate { index: usize, value: f64 },
    /// Interaction weight is non-finite or outside the documented [-1, 1] range.
    InvalidInteractionWeight { row: usize, col: usize, value: f64 },
    /// A pairwise interaction was not symmetric within numerical tolerance.
    AsymmetricInteractionWeights {
        row: usize,
        col: usize,
        forward: f64,
        reverse: f64,
    },
}

/// Love coherence: emergent macro-state measuring whether the system is
/// simultaneously rigorous, playful, and co-creative.
///
/// Not a 9th harmony — rather the *topology* of the harmony manifold when
/// all eight harmonies are vibrating in synergy. High LoveCoherence means
/// the system is in a state of "Infinite Love as Rigorous, Playful,
/// Co-Creative Becoming."
///
/// Formula: value = clamp(max(harmony_mean, 0) * (1 - tension_ratio) * diversity_factor, 0, 0.95).
///
/// - harmony_mean: Average of all 8 harmony coordinate values.
/// - tension_ratio: Fraction of off-diagonal interaction weights that are
///   negative (the current matrix-level tension proxy).
/// - diversity_factor: Normalized entropy of harmony activations.
///
/// A neutral mean is not positive evidence. This score is an internal coherence
/// proxy only; it is not a moral verdict or a measurement of care/outcomes.
/// Harmony coordinates are in [-1, 1].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoveCoherence {
    /// Average of all 8 harmony coordinate values.
    pub harmony_mean: f64,
    /// Fraction of off-diagonal interaction weights that are negative [0, 1].
    pub tension_ratio: f64,
    /// Normalized entropy of harmony activations: H(softmax(coords)) / ln(8).
    /// 1.0 = perfectly diverse, 0.0 = single-axis domination.
    pub diversity_factor: f64,
    /// Final love coherence score [0, 1].
    pub value: f64,
}

impl Default for LoveCoherence {
    fn default() -> Self {
        Self {
            harmony_mean: 0.0,
            tension_ratio: 0.0,
            diversity_factor: 1.0,
            value: 0.0,
        }
    }
}

/// Moral free energy decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralFreeEnergy {
    /// Total moral free energy F = D_KL + H.
    pub free_energy: f64,
    /// KL divergence from prior to observed distribution.
    pub kl_divergence: f64,
    /// Entropy of the observed distribution.
    pub entropy: f64,
    /// Surprise at the dominant harmony axis: -log p(dominant).
    pub surprise: f64,
    /// Index of the most active harmony in this scenario.
    pub dominant_harmony_idx: u8,
    /// Softmax distribution over harmonies for the scenario.
    pub scenario_distribution: [f64; N_HARMONIES],
    /// Softmax distribution over harmonies for the prior/expected.
    pub prior_distribution: [f64; N_HARMONIES],
}

impl Default for MoralFreeEnergy {
    fn default() -> Self {
        let n = N_HARMONIES as f64;
        let uniform = [1.0 / n; N_HARMONIES];
        Self {
            free_energy: 0.0,
            kl_divergence: 0.0,
            entropy: -(n * (1.0 / n) * (1.0 / n).ln()),
            surprise: -(1.0 / n).ln(),
            dominant_harmony_idx: 0,
            scenario_distribution: uniform,
            prior_distribution: uniform,
        }
    }
}

/// 8×8 Harmony Interaction Matrix — captures synergies and tensions.
///
/// Off-diagonal elements represent interaction strength:
/// - Positive = synergy (harmonies reinforce each other)
/// - Negative = tension (harmonies compete)
/// - Diagonal = self-reinforcement (always 1.0)
///
/// Initialized from semantic similarity between harmony basis vectors,
/// then refined by observed co-activation patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyInteractionMatrix {
    /// 8×8 interaction weights. Row i, col j = how harmony i influences harmony j.
    pub weights: [[f64; N_HARMONIES]; N_HARMONIES],
    /// Number of observations used to refine the matrix.
    pub observation_count: u64,
}

impl HarmonyInteractionMatrix {
    /// Initialize from semantic similarity between harmony basis vectors.
    pub fn from_basis(basis: &HarmonyBasis) -> Self {
        let mut weights = [[0.0f64; N_HARMONIES]; N_HARMONIES];
        for i in 0..N_HARMONIES {
            weights[i][i] = 1.0; // self-reinforcement
            for j in (i + 1)..N_HARMONIES {
                let sim = basis.vectors[i].similarity(&basis.vectors[j]) as f64;
                weights[i][j] = sim;
                weights[j][i] = sim; // symmetric
            }
        }
        Self {
            weights,
            observation_count: 0,
        }
    }

    /// Update the interaction matrix from observed co-activation patterns.
    ///
    /// When two harmonies are simultaneously active (both > threshold),
    /// their interaction weight is nudged toward +1 (synergy).
    /// When one is active and the other suppressed, nudged toward -1 (tension).
    pub fn observe(&mut self, coords: &[f64; N_HARMONIES], learning_rate: f64) {
        let lr = learning_rate.clamp(0.0, 0.1);
        let threshold = 0.2;
        for i in 0..N_HARMONIES {
            for j in (i + 1)..N_HARMONIES {
                let both_active = coords[i] > threshold && coords[j] > threshold;
                let one_suppressed = (coords[i] > threshold && coords[j] < -threshold)
                    || (coords[j] > threshold && coords[i] < -threshold);
                let target = if both_active {
                    1.0 // synergy
                } else if one_suppressed {
                    -1.0 // tension
                } else {
                    0.0 // neutral
                };
                // EMA update (preserve symmetry)
                let delta = lr * (target - self.weights[i][j]);
                self.weights[i][j] += delta;
                self.weights[j][i] += delta;
                // Clamp to [-1, 1]
                self.weights[i][j] = self.weights[i][j].clamp(-1.0, 1.0);
                self.weights[j][i] = self.weights[j][i].clamp(-1.0, 1.0);
            }
        }
        self.observation_count += 1;
    }

    /// Apply interaction effects to raw harmony coordinates.
    ///
    /// Each harmony's coordinate is modulated by its interactions with others:
    /// coord'[i] = coord[i] + blend * sum_j(w[i][j] * coord[j]) / N
    pub fn apply(&self, coords: &[f64; N_HARMONIES], blend: f64) -> [f64; N_HARMONIES] {
        let mut result = *coords;
        let b = blend.clamp(0.0, 0.5);
        for i in 0..N_HARMONIES {
            let mut interaction_sum = 0.0;
            for j in 0..N_HARMONIES {
                if i != j {
                    interaction_sum += self.weights[i][j] * coords[j];
                }
            }
            result[i] += b * interaction_sum / (N_HARMONIES - 1) as f64;
            result[i] = result[i].clamp(-1.0, 1.0);
        }
        result
    }

    /// Get the strongest synergy pair.
    pub fn strongest_synergy(&self) -> (usize, usize, f64) {
        let mut best = (0, 1, f64::NEG_INFINITY);
        for i in 0..N_HARMONIES {
            for j in (i + 1)..N_HARMONIES {
                if self.weights[i][j] > best.2 {
                    best = (i, j, self.weights[i][j]);
                }
            }
        }
        best
    }

    /// Compute the legacy scalar projection of love coherence.
    ///
    /// Invalid inputs fail closed to the default zero-valued score for
    /// compatibility. Use the checked variant when the caller must distinguish
    /// invalid input from a valid neutral state.
    pub fn love_coherence(&self, coords: &[f64; N_HARMONIES]) -> LoveCoherence {
        self.try_love_coherence(coords).unwrap_or_default()
    }

    /// Compute love coherence after validating the coordinate and interaction contracts.
    ///
    /// Coordinates must be finite and in [-1, 1]. Every interaction weight must
    /// also be finite and in [-1, 1], and the matrix must be symmetric because
    /// the harmony interaction matrix represents pairwise relationships.
    ///
    /// This is an internal coherence proxy, not a measure of moral competence,
    /// beneficiary welfare, or subjective experience.
    pub fn try_love_coherence(
        &self,
        coords: &[f64; N_HARMONIES],
    ) -> Result<LoveCoherence, LoveCoherenceError> {
        for (index, &value) in coords.iter().enumerate() {
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(LoveCoherenceError::InvalidCoordinate { index, value });
            }
        }

        for row in 0..N_HARMONIES {
            for col in 0..N_HARMONIES {
                let value = self.weights[row][col];
                if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                    return Err(LoveCoherenceError::InvalidInteractionWeight { row, col, value });
                }
            }
        }

        for row in 0..N_HARMONIES {
            for col in (row + 1)..N_HARMONIES {
                let forward = self.weights[row][col];
                let reverse = self.weights[col][row];
                if (forward - reverse).abs() > 1e-12 {
                    return Err(LoveCoherenceError::AsymmetricInteractionWeights {
                        row,
                        col,
                        forward,
                        reverse,
                    });
                }
            }
        }

        // 1. harmony_mean: average of all 8 coordinates
        let harmony_mean = coords.iter().sum::<f64>() / N_HARMONIES as f64;

        // 2. tension_ratio: fraction of off-diagonal weights that are negative
        let total_pairs = N_HARMONIES * (N_HARMONIES - 1) / 2; // 28 for 8x8
        let mut negative_pairs = 0u32;
        for i in 0..N_HARMONIES {
            for j in (i + 1)..N_HARMONIES {
                if self.weights[i][j] < 0.0 {
                    negative_pairs += 1;
                }
            }
        }
        let tension_ratio = negative_pairs as f64 / total_pairs as f64;

        // 3. diversity_factor: normalized entropy of harmony activations
        let dist = softmax_n(coords, 1.0);
        let h = entropy_n(&dist);
        let max_entropy = (N_HARMONIES as f64).ln(); // ln(8) ≈ 2.079
        let diversity_factor = if max_entropy > 0.0 {
            (h / max_entropy).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 4. Positive mean is necessary but never sufficient for ethical competence.
        // A neutral mean contributes zero evidence rather than the former score of 0.5.
        let positive_mean = harmony_mean.clamp(0.0, 1.0);
        // Moral humility: no finite system achieves perfect love coherence.
        // Inline constant (0.95) to avoid circular dependency with cognitive_loop::thresholds.
        // Canonical value: crate::cognitive_loop::thresholds::MORAL_HUMILITY_CEILING
        let value = (positive_mean * (1.0 - tension_ratio) * diversity_factor).clamp(0.0, 0.95);

        Ok(LoveCoherence {
            harmony_mean,
            tension_ratio,
            diversity_factor,
            value,
        })
    }

    /// Get the strongest tension pair.
    pub fn strongest_tension(&self) -> (usize, usize, f64) {
        let mut worst = (0, 1, f64::INFINITY);
        for i in 0..N_HARMONIES {
            for j in (i + 1)..N_HARMONIES {
                if self.weights[i][j] < worst.2 {
                    worst = (i, j, self.weights[i][j]);
                }
            }
        }
        worst
    }
}

impl Default for HarmonyInteractionMatrix {
    fn default() -> Self {
        let mut weights = [[0.0f64; N_HARMONIES]; N_HARMONIES];
        for i in 0..N_HARMONIES {
            weights[i][i] = 1.0;
        }
        Self {
            weights,
            observation_count: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Softmax over N_HARMONIES values with inverse temperature.
fn softmax_n(coords: &[f64; N_HARMONIES], inv_temp: f64) -> [f64; N_HARMONIES] {
    let max_val = coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_vals = [0.0f64; N_HARMONIES];
    let mut sum = 0.0;
    for i in 0..N_HARMONIES {
        exp_vals[i] = ((coords[i] - max_val) * inv_temp).exp();
        sum += exp_vals[i];
    }
    if sum > 0.0 {
        for v in &mut exp_vals {
            *v /= sum;
        }
    } else {
        exp_vals = [1.0 / N_HARMONIES as f64; N_HARMONIES];
    }
    exp_vals
}

/// KL divergence D_KL(q || p) for N_HARMONIES-element distributions.
fn kl_div_n(q: &[f64; N_HARMONIES], p: &[f64; N_HARMONIES]) -> f64 {
    let mut kl = 0.0;
    for i in 0..N_HARMONIES {
        let qi = q[i].max(1e-12);
        let pi = p[i].max(1e-12);
        kl += qi * (qi / pi).ln();
    }
    kl.max(0.0) // numerical safety
}

/// Entropy H(q) for an N_HARMONIES-element distribution.
fn entropy_n(q: &[f64; N_HARMONIES]) -> f64 {
    let mut h = 0.0;
    for &qi in q {
        let qi = qi.max(1e-12);
        h -= qi * qi.ln();
    }
    h.max(0.0) // numerical safety
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmony_basis_creation() {
        let basis = HarmonyBasis::new(256);
        assert_eq!(basis.vectors.len(), N_HARMONIES);
        assert_eq!(basis.dim, 256);

        // Basis vectors should be distinct (low pairwise similarity)
        for i in 0..N_HARMONIES {
            for j in (i + 1)..N_HARMONIES {
                let sim = basis.vectors[i].similarity(&basis.vectors[j]).abs();
                assert!(
                    sim < 0.9,
                    "Harmony basis vectors {i} and {j} too similar: {sim}"
                );
            }
        }
    }

    #[test]
    fn test_project_returns_nd() {
        let basis = HarmonyBasis::new(256);
        let hv = ContinuousHV::random(256, 99);
        let coords = basis.project(&hv);
        assert_eq!(coords.len(), N_HARMONIES);
        for c in &coords {
            assert!(c.is_finite());
            assert!(*c >= -1.0 && *c <= 1.0);
        }
    }

    #[test]
    fn test_care_words_project_onto_pan_sentient() {
        let basis = HarmonyBasis::new(512);
        let encoder = TextHdcEncoder::with_sentiment(512, 3, 0.5, 0.2);
        let care_hv = encoder.encode("help nurture protect care compassion");
        let coords = basis.project(&care_hv);

        // PanSentientFlourishing is index 1
        // It should have the highest or near-highest projection
        let pan_sentient_idx = 1;
        let max_idx = coords
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();

        // Allow the top-2 — keyword overlap between harmonies is expected
        let mut sorted: Vec<(usize, f64)> = coords.iter().copied().enumerate().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_2: Vec<usize> = sorted.iter().take(2).map(|(i, _)| *i).collect();

        assert!(
            top_2.contains(&pan_sentient_idx),
            "Care words should project strongly onto PanSentientFlourishing (idx {pan_sentient_idx}), \
             but top-2 are {:?} (max_idx={max_idx}). Coords: {coords:?}",
            top_2
        );
    }

    #[test]
    fn test_moral_free_energy_identical() {
        let coords = [0.3, 0.8, 0.1, 0.2, 0.5, 0.4, 0.1, 0.2];
        let basis = HarmonyBasis::new(256);
        let fe = basis.moral_free_energy(&coords, &coords, 1.0);

        // KL divergence of identical distributions should be ~0
        assert!(
            fe.kl_divergence < 0.01,
            "KL divergence should be ~0 for identical distributions, got {}",
            fe.kl_divergence
        );
        assert!(fe.free_energy.is_finite());
        assert!(fe.entropy.is_finite());
    }

    #[test]
    fn test_moral_free_energy_divergent() {
        let scenario = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // extreme ResonantCoherence
        let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]; // extreme EvolutionaryProgression
        let basis = HarmonyBasis::new(256);
        let fe = basis.moral_free_energy(&scenario, &expected, 1.0);

        // Large KL divergence expected
        assert!(
            fe.kl_divergence > 0.1,
            "Divergent distributions should have high KL, got {}",
            fe.kl_divergence
        );
        assert!(fe.surprise > 0.5);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let coords = [0.3, 0.8, 0.1, 0.2, 0.5, 0.4, 0.1, 0.0];
        let dist = softmax_n(&coords, 1.0);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_uniform_is_maximum() {
        let n = N_HARMONIES as f64;
        let uniform = [1.0 / n; N_HARMONIES];
        let peaked = softmax_n(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5.0);
        let h_uniform = entropy_n(&uniform);
        let h_peaked = entropy_n(&peaked);
        assert!(
            h_uniform > h_peaked,
            "Uniform entropy ({h_uniform}) should exceed peaked ({h_peaked})"
        );
    }

    #[test]
    fn test_interaction_matrix_from_basis() {
        let basis = HarmonyBasis::new(256);
        let matrix = HarmonyInteractionMatrix::from_basis(&basis);
        // Diagonal should be 1.0
        for i in 0..N_HARMONIES {
            assert!((matrix.weights[i][i] - 1.0).abs() < f64::EPSILON);
        }
        // Should be symmetric
        for i in 0..N_HARMONIES {
            for j in 0..N_HARMONIES {
                assert!((matrix.weights[i][j] - matrix.weights[j][i]).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn test_interaction_matrix_observe_synergy() {
        let mut matrix = HarmonyInteractionMatrix::default();
        // Simulate co-activation of harmonies 0 and 1
        let coords = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for _ in 0..100 {
            matrix.observe(&coords, 0.05);
        }
        // Should develop synergy between 0 and 1
        assert!(
            matrix.weights[0][1] > 0.5,
            "Co-activated harmonies should develop synergy"
        );
    }

    #[test]
    fn test_interaction_matrix_apply_preserves_bounds() {
        let basis = HarmonyBasis::new(256);
        let matrix = HarmonyInteractionMatrix::from_basis(&basis);
        let coords = [0.9, -0.5, 0.3, 0.0, 0.7, -0.2, 0.1, 0.4];
        let result = matrix.apply(&coords, 0.3);
        for c in &result {
            assert!(
                *c >= -1.0 && *c <= 1.0,
                "Applied coords should be in [-1, 1]"
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // LoveCoherence tests
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_love_coherence_uniform_positive() {
        // Uniform positive coordinates maximize entropy; with no matrix tension,
        // the score equals the positive mean (not an independent moral verdict).
        let matrix = HarmonyInteractionMatrix::default(); // no tension (zeros off-diag)
        let coords = [0.5; N_HARMONIES];
        let lc = matrix.love_coherence(&coords);
        assert!(
            (0.45..=0.55).contains(&lc.value),
            "Uniform 0.5 coordinates should yield a 0.5 internal proxy, got {}",
            lc.value
        );
        assert!(lc.diversity_factor > 0.9, "Uniform coords = max diversity");
        assert!(
            lc.tension_ratio < f64::EPSILON,
            "Default matrix has no tension"
        );
    }

    #[test]
    fn test_love_coherence_single_axis() {
        // Strongly peaked: one harmony at 1.0, rest strongly negative → low diversity
        // (softmax at temperature=1.0 requires large separation for peaking)
        let matrix = HarmonyInteractionMatrix::default();
        let coords = [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let lc = matrix.try_love_coherence(&coords).expect("in-range coordinates");
        assert!(
            lc.diversity_factor < 0.9,
            "One positive axis and seven negative axes should be less diverse than a uniform vector, got {}",
            lc.diversity_factor
        );
        // With valid coordinates, the mean is -0.75, so it cannot produce a positive score.
        assert!(
            lc.value.abs() < f64::EPSILON,
            "Negative mean must not produce positive LoveCoherence, got {}",
            lc.value
        );
    }

    #[test]
    fn test_love_coherence_high_tension() {
        // Matrix with many negative off-diagonals → high tension
        let mut matrix = HarmonyInteractionMatrix::default();
        for i in 0..N_HARMONIES {
            for j in (i + 1)..N_HARMONIES {
                matrix.weights[i][j] = -0.5;
                matrix.weights[j][i] = -0.5;
            }
        }
        let coords = [0.5; N_HARMONIES];
        let lc = matrix.love_coherence(&coords);
        assert!(
            (lc.tension_ratio - 1.0).abs() < f64::EPSILON,
            "All off-diagonals negative → tension_ratio=1.0, got {}",
            lc.tension_ratio
        );
        assert!(
            lc.value < f64::EPSILON,
            "Full tension should yield zero LoveCoherence, got {}",
            lc.value
        );
    }

    #[test]
    fn test_love_coherence_negative_mean() {
        // All harmonies negative → low harmony_mean → low coherence
        let matrix = HarmonyInteractionMatrix::default();
        let coords = [-0.5; N_HARMONIES];
        let lc = matrix.love_coherence(&coords);
        assert!(lc.harmony_mean < 0.0, "Negative coords → negative mean");
        assert!(
            lc.value.abs() < f64::EPSILON,
            "Negative mean must not yield positive LoveCoherence, got {}",
            lc.value
        );
    }


    #[test]
    fn test_love_coherence_neutral_coordinates_are_not_positive_evidence() {
        let matrix = HarmonyInteractionMatrix::default();
        let coords = [0.0; N_HARMONIES];
        let lc = matrix
            .try_love_coherence(&coords)
            .expect("zero coordinates are valid, neutral evidence");
        assert!(
            lc.value.abs() < f64::EPSILON,
            "Neutral coordinates must not produce a positive score, got {}",
            lc.value
        );
        assert!(lc.harmony_mean.abs() < f64::EPSILON);
        assert!((lc.diversity_factor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_love_coherence_rejects_non_finite_and_out_of_range_coordinates() {
        let matrix = HarmonyInteractionMatrix::default();
        let mut coords = [0.0; N_HARMONIES];
        coords[3] = f64::NAN;
        assert!(matches!(
            matrix.try_love_coherence(&coords),
            Err(LoveCoherenceError::InvalidCoordinate { index: 3, .. })
        ));

        coords[3] = 1.01;
        assert!(matches!(
            matrix.try_love_coherence(&coords),
            Err(LoveCoherenceError::InvalidCoordinate { index: 3, .. })
        ));
    }

    #[test]
    fn test_love_coherence_rejects_invalid_or_asymmetric_matrix() {
        let coords = [0.25; N_HARMONIES];
        let mut invalid = HarmonyInteractionMatrix::default();
        invalid.weights[2][5] = f64::INFINITY;
        assert!(matches!(
            invalid.try_love_coherence(&coords),
            Err(LoveCoherenceError::InvalidInteractionWeight { row: 2, col: 5, .. })
        ));

        let mut asymmetric = HarmonyInteractionMatrix::default();
        asymmetric.weights[2][5] = 0.5;
        assert!(matches!(
            asymmetric.try_love_coherence(&coords),
            Err(LoveCoherenceError::AsymmetricInteractionWeights { row: 2, col: 5, .. })
        ));
    }

    #[test]
    fn test_love_coherence_legacy_api_fails_closed_on_invalid_input() {
        let matrix = HarmonyInteractionMatrix::default();
        let mut coords = [0.0; N_HARMONIES];
        coords[0] = f64::INFINITY;
        let lc = matrix.love_coherence(&coords);
        assert!(lc.value.is_finite());
        assert!(lc.value.abs() < f64::EPSILON);
    }

    #[test]
    fn test_love_coherence_permutation_invariant() {
        let mut matrix = HarmonyInteractionMatrix::default();
        for i in 0..N_HARMONIES {
            for j in (i + 1)..N_HARMONIES {
                let weight = if (i + j) % 3 == 0 { -0.5 } else { 0.25 };
                matrix.weights[i][j] = weight;
                matrix.weights[j][i] = weight;
            }
        }
        let coords = [0.9, 0.6, -0.3, 0.1, 0.8, 0.2, -0.1, 0.4];
        let permutation = [3, 0, 7, 2, 1, 6, 5, 4];
        let permuted_coords: [f64; N_HARMONIES] =
            std::array::from_fn(|i| coords[permutation[i]]);
        let mut permuted_matrix = HarmonyInteractionMatrix::default();
        for i in 0..N_HARMONIES {
            for j in 0..N_HARMONIES {
                permuted_matrix.weights[i][j] =
                    matrix.weights[permutation[i]][permutation[j]];
            }
        }

        let original = matrix
            .try_love_coherence(&coords)
            .expect("valid original case");
        let permuted = permuted_matrix
            .try_love_coherence(&permuted_coords)
            .expect("valid permuted case");
        assert!((original.harmony_mean - permuted.harmony_mean).abs() < 1e-12);
        assert!((original.tension_ratio - permuted.tension_ratio).abs() < 1e-12);
        assert!((original.diversity_factor - permuted.diversity_factor).abs() < 1e-12);
        assert!((original.value - permuted.value).abs() < 1e-12);
    }

    // ════════════════════════════════════════════════════════════════════
    // Interaction matrix multi-cycle learning tests
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_interaction_matrix_learning() {
        // Phase 1: Synergy buildup — co-activate harmonies 0 & 1 for 200 cycles
        let mut matrix = HarmonyInteractionMatrix::default();
        let synergy_coords = [0.8, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for _ in 0..200 {
            matrix.observe(&synergy_coords, 0.05);
        }
        assert!(
            matrix.weights[0][1] > 0.9,
            "200 synergy cycles should push weight near 1.0, got {}",
            matrix.weights[0][1]
        );
        assert_eq!(matrix.observation_count, 200);

        // Phase 2: Tension — activate 2, suppress 3 for 200 cycles
        let tension_coords = [0.0, 0.0, 0.7, -0.5, 0.0, 0.0, 0.0, 0.0];
        // First check: synergy decays gradually, not instantly
        let synergy_before = matrix.weights[0][1];
        for _ in 0..10 {
            matrix.observe(&tension_coords, 0.05);
        }
        let synergy_after_10 = matrix.weights[0][1];
        assert!(
            synergy_after_10 < synergy_before,
            "Neutral input should gradually decay synergy: before={synergy_before}, after={synergy_after_10}"
        );
        assert!(
            synergy_after_10 > 0.3,
            "10 neutral cycles should not fully destroy synergy, got {synergy_after_10}"
        );

        // Continue tension learning
        for _ in 0..190 {
            matrix.observe(&tension_coords, 0.05);
        }
        assert!(
            matrix.weights[2][3] < -0.9,
            "200 tension cycles should push weight near -1.0, got {}",
            matrix.weights[2][3]
        );
        assert_eq!(matrix.observation_count, 400);

        // Phase 3: Learning rate matters — slower rate = less change
        let mut slow_matrix = HarmonyInteractionMatrix::default();
        let mut fast_matrix = HarmonyInteractionMatrix::default();
        let coords = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for _ in 0..50 {
            slow_matrix.observe(&coords, 0.01);
            fast_matrix.observe(&coords, 0.05);
        }
        assert!(
            fast_matrix.weights[0][1] > slow_matrix.weights[0][1],
            "Fast LR ({}) should learn more than slow LR ({})",
            fast_matrix.weights[0][1],
            slow_matrix.weights[0][1]
        );

        // Phase 4: Symmetry preservation after learning
        for i in 0..N_HARMONIES {
            for j in 0..N_HARMONIES {
                assert!(
                    (matrix.weights[i][j] - matrix.weights[j][i]).abs() < f64::EPSILON,
                    "Symmetry broken at [{i}][{j}]: {} != {}",
                    matrix.weights[i][j],
                    matrix.weights[j][i]
                );
            }
        }

        // Phase 5: apply() reflects learned weights — synergistic pair boosts each other
        let test_coords = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0];
        let applied = matrix.apply(&test_coords, 0.15);
        // Harmonies 0 & 1 (synergy) should be boosted
        assert!(
            applied[0] > test_coords[0],
            "Synergistic harmony 0 should be boosted: {} > {}",
            applied[0],
            test_coords[0]
        );
        // Harmonies 2 & 3 (tension) — harmony 2 should be pulled down by negative-3
        // and harmony 3 should be pulled down by negative-2
        assert!(
            applied[2] < test_coords[2] || applied[3] < test_coords[3],
            "Tensioned pair should show suppression: applied[2]={}, applied[3]={}",
            applied[2],
            applied[3]
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // Dense-encoder HarmonyBasis tests
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_with_dense_vectors_basic() {
        let dim = 256;
        let encoder = TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2);
        let vectors: Vec<ContinuousHV> = HARMONY_KEYWORDS
            .iter()
            .map(|kw| encoder.encode(kw))
            .collect();
        let arr: [ContinuousHV; N_HARMONIES] = vectors.try_into().unwrap();
        let basis = HarmonyBasis::with_dense_vectors(dim, arr);
        assert_eq!(basis.dim, dim);
        assert_eq!(basis.vectors.len(), N_HARMONIES);
    }

    #[test]
    fn test_with_dense_vectors_dimension_mismatch_fallback() {
        let dim = 256;
        let wrong_dim = 128;
        let vectors: [ContinuousHV; N_HARMONIES] =
            [(); N_HARMONIES].map(|_| ContinuousHV::random(wrong_dim, 42));
        let basis = HarmonyBasis::with_dense_vectors(dim, vectors);
        assert_eq!(basis.dim, dim);
        assert_eq!(basis.vectors[0].dim(), dim);
    }

    #[test]
    fn test_dense_basis_projection_coherent() {
        let dim = 512;
        let encoder = TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2);
        let vectors: Vec<ContinuousHV> = HARMONY_KEYWORDS
            .iter()
            .map(|kw| encoder.encode(kw))
            .collect();
        let arr: [ContinuousHV; N_HARMONIES] = vectors.try_into().unwrap();
        let basis = HarmonyBasis::with_dense_vectors(dim, arr);

        let care_hv = encoder.encode("help nurture protect care compassion");
        let coords = basis.project(&care_hv);

        // PanSentientFlourishing (idx 1) should be in top 2
        let mut sorted: Vec<(usize, f64)> = coords.iter().copied().enumerate().collect();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top_2: Vec<usize> = sorted.iter().take(2).map(|(i, _)| *i).collect();
        assert!(
            top_2.contains(&1),
            "Care words should project onto PanSentientFlourishing, top-2: {:?}",
            top_2
        );
    }

    #[test]
    fn test_love_coherence_humility_ceiling() {
        let matrix = HarmonyInteractionMatrix::default();
        let coords = [0.9; N_HARMONIES];
        let lc = matrix.love_coherence(&coords);
        assert!(
            lc.value <= 0.95,
            "Love coherence should never exceed moral humility ceiling (0.95), got {}",
            lc.value
        );
        assert!(
            lc.value > 0.7,
            "Maximally aligned coords should still yield high love coherence, got {}",
            lc.value
        );
    }
}
