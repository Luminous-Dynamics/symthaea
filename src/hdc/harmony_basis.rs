//! Shared Seven Harmonies basis vectors for moral geometry.
//!
//! Provides semantically grounded 16,384-dim `ContinuousHV` basis vectors for
//! the Seven Harmonies, built by encoding keyword sets through `TextHdcEncoder`.
//! These replace the random embeddings previously used in `HarmoniesIntegrator`
//! and unify with the harmony projection in `MoralTopology`.
//!
//! Also provides moral free energy computation following the Free Energy
//! Principle (Friston 2010): a moral system that cannot predict its own
//! consequences accumulates surprise and destabilizes. Moral free energy
//! quantifies the divergence between expected and observed harmony coordinates.

use symthaea_core::hdc::ContinuousHV;
use symthaea_types::Harmony;

use super::moral_text_encoder::TextHdcEncoder;

/// Keyword sets for each harmony (order matches `Harmony::all()`).
///
/// Each harmony is encoded as the bundle of its keyword set, producing
/// a ContinuousHV that responds to semantic similarity rather than
/// arbitrary random projection.
pub const HARMONY_KEYWORDS: [&str; 7] = [
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
];

/// Seven Harmony basis vectors for projecting moral scenarios into
/// semantically meaningful 7D coordinates.
///
/// Each basis vector is built by encoding a harmony's keyword set through
/// `TextHdcEncoder`, so projection onto (e.g.) `PanSentientFlourishing` is
/// high when the scenario's vocabulary overlaps with "help", "care", etc.
#[derive(Debug, Clone)]
pub struct HarmonyBasis {
    /// One ContinuousHV per Harmony (indexed by canonical `Harmony::all()` order).
    pub vectors: [ContinuousHV; 7],
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
        Self {
            vectors: vectors.try_into().unwrap_or_else(|v: Vec<ContinuousHV>| {
                panic!("expected 7 harmony vectors, got {}", v.len())
            }),
            dim,
        }
    }

    /// Project a scenario HV onto the 7 harmony axes (cosine similarity).
    ///
    /// Returns coordinates in `[-1, 1]^7` where each component is the
    /// cosine similarity between the scenario and the corresponding harmony
    /// basis vector. The result is a point in the moral manifold.
    pub fn project(&self, hv: &ContinuousHV) -> [f64; 7] {
        let mut coords = [0.0f64; 7];
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
    /// expected distribution over the 7 harmony coordinates. Following the
    /// Free Energy Principle (Friston 2010), a viable moral system minimizes
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
        scenario_coords: &[f64; 7],
        expected_coords: &[f64; 7],
        temperature: f64,
    ) -> MoralFreeEnergy {
        let inv_temp = 1.0 / temperature.max(0.01);

        // Softmax over harmony coordinates → probability distributions
        let q = softmax_7(scenario_coords, inv_temp);
        let p = softmax_7(expected_coords, inv_temp);

        // KL divergence: D_KL(q || p) = sum_i q_i * ln(q_i / p_i)
        let kl_divergence = kl_div_7(&q, &p);

        // Entropy: H(q) = -sum_i q_i * ln(q_i)
        let entropy = entropy_7(&q);

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
#[derive(Debug, Clone)]
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
    pub scenario_distribution: [f64; 7],
    /// Softmax distribution over harmonies for the prior/expected.
    pub prior_distribution: [f64; 7],
}

impl Default for MoralFreeEnergy {
    fn default() -> Self {
        let uniform = [1.0 / 7.0; 7];
        Self {
            free_energy: 0.0,
            kl_divergence: 0.0,
            entropy: -(7.0_f64 * (1.0 / 7.0) * (1.0 / 7.0_f64).ln()),
            surprise: -(1.0 / 7.0_f64).ln(),
            dominant_harmony_idx: 0,
            scenario_distribution: uniform,
            prior_distribution: uniform,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Softmax over 7 values with inverse temperature.
fn softmax_7(coords: &[f64; 7], inv_temp: f64) -> [f64; 7] {
    let max_val = coords
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut exp_vals = [0.0f64; 7];
    let mut sum = 0.0;
    for i in 0..7 {
        exp_vals[i] = ((coords[i] - max_val) * inv_temp).exp();
        sum += exp_vals[i];
    }
    if sum > 0.0 {
        for v in &mut exp_vals {
            *v /= sum;
        }
    } else {
        exp_vals = [1.0 / 7.0; 7];
    }
    exp_vals
}

/// KL divergence D_KL(q || p) for 7-element distributions.
fn kl_div_7(q: &[f64; 7], p: &[f64; 7]) -> f64 {
    let mut kl = 0.0;
    for i in 0..7 {
        let qi = q[i].max(1e-12);
        let pi = p[i].max(1e-12);
        kl += qi * (qi / pi).ln();
    }
    kl.max(0.0) // numerical safety
}

/// Entropy H(q) for a 7-element distribution.
fn entropy_7(q: &[f64; 7]) -> f64 {
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
        assert_eq!(basis.vectors.len(), 7);
        assert_eq!(basis.dim, 256);

        // Basis vectors should be distinct (low pairwise similarity)
        for i in 0..7 {
            for j in (i + 1)..7 {
                let sim = basis.vectors[i].similarity(&basis.vectors[j]).abs();
                assert!(
                    sim < 0.9,
                    "Harmony basis vectors {i} and {j} too similar: {sim}"
                );
            }
        }
    }

    #[test]
    fn test_project_returns_7d() {
        let basis = HarmonyBasis::new(256);
        let hv = ContinuousHV::random(256, 99);
        let coords = basis.project(&hv);
        assert_eq!(coords.len(), 7);
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
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Allow the top-2 — keyword overlap between harmonies is expected
        let mut sorted: Vec<(usize, f64)> = coords.iter().copied().enumerate().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
        let coords = [0.3, 0.8, 0.1, 0.2, 0.5, 0.4, 0.1];
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
        let scenario = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // extreme ResonantCoherence
        let expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]; // extreme EvolutionaryProgression
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
        let coords = [0.3, 0.8, 0.1, 0.2, 0.5, 0.4, 0.1];
        let dist = softmax_7(&coords, 1.0);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_uniform_is_maximum() {
        let uniform = [1.0 / 7.0; 7];
        let peaked = softmax_7(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5.0);
        let h_uniform = entropy_7(&uniform);
        let h_peaked = entropy_7(&peaked);
        assert!(
            h_uniform > h_peaked,
            "Uniform entropy ({h_uniform}) should exceed peaked ({h_peaked})"
        );
    }
}
