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
    "integrate harmonize unify coherent order luminous resonant balance alignment wholeness",
    // PanSentientFlourishing — Care-Knowing
    "help support care benefit serve protect nurture compassion flourishing kindness love",
    // IntegralWisdom — Truth-Knowing
    "learn understand wisdom knowledge insight intelligence truth awareness knowing embodied",
    // InfinitePlay — Creative-Knowing
    "create explore play discover experiment joy creativity generativity novelty imagination",
    // UniversalInterconnectedness — Relational-Knowing
    "connect share collaborate together community unity empathy resonance interconnected belonging",
    // SacredReciprocity — Exchange-Knowing
    "give share contribute reciprocate exchange generous flow mutual upliftment trust",
    // EvolutionaryProgression — Developmental-Knowing
    "grow evolve improve progress develop transcend becoming evolution advancement consciousness",
    // SacredStillness — Apophatic-Knowing
    "stillness silence rest surrender void release contemplation emptiness peace presence apophatic",
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
}
