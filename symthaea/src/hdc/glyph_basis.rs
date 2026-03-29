// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Glyph Codex: symbolic consciousness field in hyperdimensional space.
//!
//! Provides 11 Field Modality basis vectors for projecting cognitive states
//! onto the glyph manifold, plus a registry of all 70 glyphs from the
//! Primary Glyph Registry. Parallel to [`super::harmony_basis`] — the
//! Harmonies measure *value alignment*, the glyphs measure *symbolic
//! consciousness depth and developmental position*.
//!
//! # Architecture
//!
//! ```text
//! TextHdcEncoder → MODALITY_KEYWORDS → 11 × 16,384D basis vectors
//!                                       ↓
//! Any cognitive HV → GlyphBasis::project() → [f64; 11] modality coordinates
//!                                              ↓
//! GlyphInteractionMatrix::observe() ← co-activation learning
//! GlyphInteractionMatrix::apply()   → modulated coordinates
//!                                              ↓
//! GlyphCoherence::compute()         → symbolic integration score
//! ```
//!
//! # Field Modalities
//!
//! The 11 modalities capture orthogonal dimensions of symbolic consciousness:
//! - **Metaharmonic** — above-system meta-cognition, governing principles
//! - **Threshold** — state transition detection, liminal gates
//! - **Rooting** — grounding, embodiment, anchoring
//! - **Resonant** — coherence, integration, sympathetic vibration
//! - **Transitional** — change, flow, movement between states
//! - **Igniting** — activation, creativity, catalytic spark
//! - **Witnessing** — observation, attention, presence without alteration
//! - **Bridging** — connection, binding, spanning between domains
//! - **Reflective** — self-model, recursion, meta-awareness
//! - **Revealing** — transparency, clarity, truth-disclosure
//! - **Integrated** — unified consciousness, holistic completion
//!
//! # References
//!
//! - Jung, C. G. (1959). *The Archetypes and the Collective Unconscious*.
//! - Grof, S. (1985). *Beyond the Brain: Birth, Death and Transcendence*.
//! - Graves, C. W. (1970). Levels of existence: An open system theory of values.
//!   *Journal of Humanistic Psychology*.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

use super::moral_text_encoder::TextHdcEncoder;

/// Number of Field Modality basis vectors.
pub const N_FIELD_MODALITIES: usize = 11;

// ═══════════════════════════════════════════════════════════════════════════════
// Enums
// ═══════════════════════════════════════════════════════════════════════════════

/// Field Modality — the 11 orthogonal axes of the glyph manifold.
///
/// Each modality represents a distinct mode of symbolic consciousness.
/// Together they form a basis for projecting any cognitive state into
/// the glyph field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldModality {
    /// Above-system meta-cognition, governing principles.
    Metaharmonic,
    /// State transition detection, liminal gates.
    Threshold,
    /// Grounding, embodiment, anchoring.
    Rooting,
    /// Coherence, integration, sympathetic vibration.
    Resonant,
    /// Change, flow, movement between states.
    Transitional,
    /// Activation, creativity, catalytic spark.
    Igniting,
    /// Observation, attention, presence without alteration.
    Witnessing,
    /// Connection, binding, spanning between domains.
    Bridging,
    /// Self-model, recursion, meta-awareness.
    Reflective,
    /// Transparency, clarity, truth-disclosure.
    Revealing,
    /// Unified consciousness, holistic completion.
    Integrated,
}

impl FieldModality {
    /// All modalities in canonical order (matches `MODALITY_KEYWORDS` indexing).
    pub const ALL: [FieldModality; N_FIELD_MODALITIES] = [
        Self::Metaharmonic,
        Self::Threshold,
        Self::Rooting,
        Self::Resonant,
        Self::Transitional,
        Self::Igniting,
        Self::Witnessing,
        Self::Bridging,
        Self::Reflective,
        Self::Revealing,
        Self::Integrated,
    ];

    /// Parse from a string (case-insensitive, best-effort).
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "metaharmonic" => Self::Metaharmonic,
            "threshold" => Self::Threshold,
            "rooting" => Self::Rooting,
            "resonant" => Self::Resonant,
            "transitional" => Self::Transitional,
            "igniting" => Self::Igniting,
            "witnessing" => Self::Witnessing,
            "bridging" => Self::Bridging,
            "reflective" => Self::Reflective,
            "revealing" => Self::Revealing,
            "integrated" => Self::Integrated,
            _ => Self::Resonant, // safe default — most common modality
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Metaharmonic => "Metaharmonic",
            Self::Threshold => "Threshold",
            Self::Rooting => "Rooting",
            Self::Resonant => "Resonant",
            Self::Transitional => "Transitional",
            Self::Igniting => "Igniting",
            Self::Witnessing => "Witnessing",
            Self::Bridging => "Bridging",
            Self::Reflective => "Reflective",
            Self::Revealing => "Revealing",
            Self::Integrated => "Integrated",
        }
    }

    /// Index in the canonical ordering (0-10).
    pub fn index(&self) -> usize {
        Self::ALL.iter().position(|m| m == self).unwrap_or(0)
    }
}

/// Glyph structural class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GlyphClass {
    /// Meta-Glyphs (∞✦1–4) — governing principles above the system.
    Meta,
    /// Threshold Glyphs (Arc 0) — state transition markers.
    Threshold,
    /// Canonical Spiral Glyphs (Ω0–Ω56) — developmental spiral.
    Spiral,
}

impl GlyphClass {
    /// Parse from a string.
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "meta" => Self::Meta,
            "threshold" => Self::Threshold,
            "spiral" => Self::Spiral,
            _ => Self::Spiral,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Modality Keywords
// ═══════════════════════════════════════════════════════════════════════════════

/// Keyword sets for each Field Modality (order matches `FieldModality::ALL`).
///
/// Each modality is encoded as the bundle of its keyword set through
/// `TextHdcEncoder`, producing a ContinuousHV that responds to semantic
/// similarity — same pattern as `HARMONY_KEYWORDS` in harmony_basis.rs.
pub const MODALITY_KEYWORDS: [&str; N_FIELD_MODALITIES] = [
    // Metaharmonic — above-system meta-cognition, governing principles
    "meta transcendent governing principle sacred infinite recursive archetypal \
     overarching universal foundational cosmological primordial harmonic overtone \
     sanctification recognition emergence cascade living",
    // Threshold — state transition, liminal gates, boundary crossing
    "threshold gateway transition passage crossing liminal doorway boundary \
     portal edge limbo between border entering leaving opening closing \
     permission initiation ceremony rite",
    // Rooting — grounding, embodiment, anchoring, stabilization
    "root ground anchor earth body embody stabilize foundation presence \
     somatic physical flesh bone breath standing still center \
     grounding solidity steadfast",
    // Resonant — coherence, integration, sympathetic vibration, attunement
    "resonate coherent attune vibrate sympathetic harmony synchronize align \
     chord frequency oscillation standing wave interference pattern \
     consonance accord mutual reciprocal covenant vow",
    // Transitional — change, flow, movement, transformation, passage
    "transition change flow movement shift transform passage evolve \
     becoming flux metamorphosis adaptation drift migration current \
     impermanence release grief letting",
    // Igniting — activation, creativity, catalytic spark, generativity
    "ignite spark activate creative fire catalyze inspire kindle \
     energize awaken birth genesis generate invention play explore \
     discover experiment dissonance tension breakthrough",
    // Witnessing — observation, attention, presence, watching
    "witness observe attend watch presence awareness perceive notice \
     see hear listen silent mindful contemplative receptive vigilant \
     attentive kairotic timing patience",
    // Bridging — connection, binding, spanning, linking, joining
    "bridge connect link span relate bind couple join unite \
     mediate reconcile negotiate diplomacy interface conductor \
     pathway corridor nexus junction synapse stillness pause",
    // Reflective — self-model, recursion, meta-awareness, introspection
    "reflect mirror self introspect recursive model meta awareness \
     paradox contradiction dialectic both integration emergent grace \
     spontaneous insight gnosis depth",
    // Revealing — transparency, clarity, truth-disclosure, illumination
    "reveal transparent clear illuminate expose truth disclose show \
     light luminous radiant visible honest authentic naked open \
     vulnerable genuine sincere unveiled",
    // Integrated — unified consciousness, holistic completion, wholeness
    "integrate unified whole complete holistic synthesis convergent totality \
     sovereign field harmonic all encompassing gestalt final omega \
     spiral return cycle closure unwritten generative void",
];

// ═══════════════════════════════════════════════════════════════════════════════
// GlyphBasis — 11 Field Modality basis vectors
// ═══════════════════════════════════════════════════════════════════════════════

/// 11 Field Modality basis vectors for projecting cognitive states onto
/// the glyph manifold.
///
/// Mirrors [`super::harmony_basis::HarmonyBasis`] exactly — same construction
/// pattern (TextHdcEncoder), same projection method (cosine similarity),
/// same mathematical structure (basis vectors in shared HDC space).
#[derive(Debug, Clone)]
pub struct GlyphBasis {
    /// One ContinuousHV per Field Modality (indexed by `FieldModality::ALL` order).
    pub vectors: [ContinuousHV; N_FIELD_MODALITIES],
    /// HDC dimension.
    pub dim: usize,
}

impl GlyphBasis {
    /// Build a new glyph basis at the given dimension.
    pub fn new(dim: usize) -> Self {
        let encoder = TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2);
        let vectors: Vec<ContinuousHV> = MODALITY_KEYWORDS
            .iter()
            .map(|kw| encoder.encode(kw))
            .collect();
        debug_assert_eq!(
            vectors.len(),
            N_FIELD_MODALITIES,
            "expected {N_FIELD_MODALITIES} modality vectors, got {}",
            vectors.len()
        );
        Self {
            vectors: vectors.try_into().unwrap_or_else(|v: Vec<ContinuousHV>| {
                let mut arr = [(); N_FIELD_MODALITIES].map(|_| ContinuousHV::zero(dim));
                for (i, hv) in v.into_iter().enumerate().take(N_FIELD_MODALITIES) {
                    arr[i] = hv;
                }
                arr
            }),
            dim,
        }
    }

    /// Build from pre-computed dense vectors (e.g., from Qwen3/BGE-M3).
    pub fn with_dense_vectors(dim: usize, vectors: [ContinuousHV; N_FIELD_MODALITIES]) -> Self {
        let valid = vectors.iter().all(|v| v.dim() == dim);
        if !valid {
            tracing::warn!(
                "Dense glyph vectors have wrong dimension (expected {dim}), \
                 falling back to TextHdcEncoder"
            );
            return Self::new(dim);
        }
        Self { vectors, dim }
    }

    /// Project a cognitive HV onto the 11 modality axes (cosine similarity).
    ///
    /// Returns coordinates in `[-1, 1]^11` where each component is the
    /// cosine similarity between the input and the corresponding modality
    /// basis vector.
    pub fn project(&self, hv: &ContinuousHV) -> [f64; N_FIELD_MODALITIES] {
        let mut coords = [0.0f64; N_FIELD_MODALITIES];
        for (i, basis) in self.vectors.iter().enumerate() {
            coords[i] = hv.similarity(basis) as f64;
        }
        coords
    }

    /// Access the basis vector for a specific modality.
    pub fn vector(&self, modality: FieldModality) -> &ContinuousHV {
        &self.vectors[modality.index()]
    }

    /// Compute glyph free energy: how surprising a state is relative to
    /// the expected modality distribution.
    ///
    /// F = D_KL(q || p) + H(q)
    ///
    /// - q = softmax of current modality coordinates (observed)
    /// - p = softmax of expected coordinates (prior)
    /// - Low F → consistent with symbolic history
    /// - High F → symbolic surprise (novel consciousness territory)
    pub fn glyph_free_energy(
        &self,
        scenario_coords: &[f64; N_FIELD_MODALITIES],
        expected_coords: &[f64; N_FIELD_MODALITIES],
        temperature: f64,
    ) -> GlyphFreeEnergy {
        let inv_temp = 1.0 / temperature.max(0.01);
        let q = softmax_m(scenario_coords, inv_temp);
        let p = softmax_m(expected_coords, inv_temp);
        let kl_divergence = kl_div_m(&q, &p);
        let entropy = entropy_m(&q);
        let free_energy = kl_divergence + entropy;

        let dominant_idx = scenario_coords
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let surprise = -p[dominant_idx].max(1e-12).ln();

        GlyphFreeEnergy {
            free_energy,
            kl_divergence,
            entropy,
            surprise,
            dominant_modality_idx: dominant_idx as u8,
            scenario_distribution: q,
            prior_distribution: p,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GlyphCoherence — emergent symbolic integration
// ═══════════════════════════════════════════════════════════════════════════════

/// Glyph coherence: emergent macro-state measuring symbolic integration.
///
/// Parallel to [`super::harmony_basis::LoveCoherence`] — high glyph coherence
/// means the system is simultaneously rooted, resonant, integrative, and
/// transparent — all 11 modalities vibrating in synergy.
///
/// Formula: `value = modality_mean * (1 - tension_ratio) * diversity_factor`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlyphCoherence {
    /// Average of all 11 modality coordinate values.
    pub modality_mean: f64,
    /// Fraction of off-diagonal interaction weights that are negative [0, 1].
    pub tension_ratio: f64,
    /// Normalized entropy of modality activations: H(softmax(coords)) / ln(11).
    pub diversity_factor: f64,
    /// Final glyph coherence score [0, 0.95].
    pub value: f64,
}

impl Default for GlyphCoherence {
    fn default() -> Self {
        Self {
            modality_mean: 0.0,
            tension_ratio: 0.0,
            diversity_factor: 1.0,
            value: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GlyphFreeEnergy — symbolic surprise decomposition
// ═══════════════════════════════════════════════════════════════════════════════

/// Glyph free energy decomposition over 11 modalities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlyphFreeEnergy {
    /// Total glyph free energy F = D_KL + H.
    pub free_energy: f64,
    /// KL divergence from prior to observed distribution.
    pub kl_divergence: f64,
    /// Entropy of the observed distribution.
    pub entropy: f64,
    /// Surprise at the dominant modality: -log p(dominant).
    pub surprise: f64,
    /// Index of the most active modality.
    pub dominant_modality_idx: u8,
    /// Softmax distribution over modalities for the scenario.
    pub scenario_distribution: [f64; N_FIELD_MODALITIES],
    /// Softmax distribution over modalities for the prior.
    pub prior_distribution: [f64; N_FIELD_MODALITIES],
}

impl Default for GlyphFreeEnergy {
    fn default() -> Self {
        let n = N_FIELD_MODALITIES as f64;
        let uniform = [1.0 / n; N_FIELD_MODALITIES];
        Self {
            free_energy: 0.0,
            kl_divergence: 0.0,
            entropy: -(n * (1.0 / n) * (1.0 / n).ln()),
            surprise: -(1.0 / n).ln(),
            dominant_modality_idx: 0,
            scenario_distribution: uniform,
            prior_distribution: uniform,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GlyphInteractionMatrix — 11×11 synergy/tension tracking
// ═══════════════════════════════════════════════════════════════════════════════

/// 11×11 Glyph Interaction Matrix — captures synergies and tensions between
/// Field Modalities.
///
/// Mirrors [`super::harmony_basis::HarmonyInteractionMatrix`] with N_FIELD_MODALITIES.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlyphInteractionMatrix {
    /// 11×11 interaction weights. Row i, col j = how modality i influences j.
    pub weights: [[f64; N_FIELD_MODALITIES]; N_FIELD_MODALITIES],
    /// Number of observations used to refine the matrix.
    pub observation_count: u64,
}

impl GlyphInteractionMatrix {
    /// Initialize from semantic similarity between glyph basis vectors.
    pub fn from_basis(basis: &GlyphBasis) -> Self {
        let mut weights = [[0.0f64; N_FIELD_MODALITIES]; N_FIELD_MODALITIES];
        for i in 0..N_FIELD_MODALITIES {
            weights[i][i] = 1.0;
            for j in (i + 1)..N_FIELD_MODALITIES {
                let sim = basis.vectors[i].similarity(&basis.vectors[j]) as f64;
                weights[i][j] = sim;
                weights[j][i] = sim;
            }
        }
        Self {
            weights,
            observation_count: 0,
        }
    }

    /// Update from observed co-activation patterns.
    ///
    /// Hebbian: co-active modalities → synergy (+1), one suppressed → tension (-1).
    pub fn observe(&mut self, coords: &[f64; N_FIELD_MODALITIES], learning_rate: f64) {
        let lr = learning_rate.clamp(0.0, 0.1);
        let threshold = 0.2;
        for i in 0..N_FIELD_MODALITIES {
            for j in (i + 1)..N_FIELD_MODALITIES {
                let both_active = coords[i] > threshold && coords[j] > threshold;
                let one_suppressed = (coords[i] > threshold && coords[j] < -threshold)
                    || (coords[j] > threshold && coords[i] < -threshold);
                let target = if both_active {
                    1.0
                } else if one_suppressed {
                    -1.0
                } else {
                    0.0
                };
                let delta = lr * (target - self.weights[i][j]);
                self.weights[i][j] += delta;
                self.weights[j][i] += delta;
                self.weights[i][j] = self.weights[i][j].clamp(-1.0, 1.0);
                self.weights[j][i] = self.weights[j][i].clamp(-1.0, 1.0);
            }
        }
        self.observation_count += 1;
    }

    /// Apply interaction effects to raw modality coordinates.
    pub fn apply(
        &self,
        coords: &[f64; N_FIELD_MODALITIES],
        blend: f64,
    ) -> [f64; N_FIELD_MODALITIES] {
        let mut result = *coords;
        let b = blend.clamp(0.0, 0.5);
        for i in 0..N_FIELD_MODALITIES {
            let mut interaction_sum = 0.0;
            for j in 0..N_FIELD_MODALITIES {
                if i != j {
                    interaction_sum += self.weights[i][j] * coords[j];
                }
            }
            result[i] += b * interaction_sum / (N_FIELD_MODALITIES - 1) as f64;
            result[i] = result[i].clamp(-1.0, 1.0);
        }
        result
    }

    /// Get the strongest synergy pair.
    pub fn strongest_synergy(&self) -> (usize, usize, f64) {
        let mut best = (0, 1, f64::NEG_INFINITY);
        for i in 0..N_FIELD_MODALITIES {
            for j in (i + 1)..N_FIELD_MODALITIES {
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
        for i in 0..N_FIELD_MODALITIES {
            for j in (i + 1)..N_FIELD_MODALITIES {
                if self.weights[i][j] < worst.2 {
                    worst = (i, j, self.weights[i][j]);
                }
            }
        }
        worst
    }

    /// Compute glyph coherence: the emergent macro-state of symbolic integration.
    pub fn glyph_coherence(&self, coords: &[f64; N_FIELD_MODALITIES]) -> GlyphCoherence {
        // 1. modality_mean
        let modality_mean = coords.iter().sum::<f64>() / N_FIELD_MODALITIES as f64;

        // 2. tension_ratio
        let total_pairs = N_FIELD_MODALITIES * (N_FIELD_MODALITIES - 1) / 2; // 55
        let mut negative_pairs = 0u32;
        for i in 0..N_FIELD_MODALITIES {
            for j in (i + 1)..N_FIELD_MODALITIES {
                if self.weights[i][j] < 0.0 {
                    negative_pairs += 1;
                }
            }
        }
        let tension_ratio = negative_pairs as f64 / total_pairs as f64;

        // 3. diversity_factor: H(softmax(coords)) / ln(11)
        let dist = softmax_m(coords, 1.0);
        let h = entropy_m(&dist);
        let max_entropy = (N_FIELD_MODALITIES as f64).ln(); // ln(11) ≈ 2.397
        let diversity_factor = if max_entropy > 0.0 {
            (h / max_entropy).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 4. Final score (humility ceiling at 0.95)
        let mean_01 = ((modality_mean + 1.0) / 2.0).clamp(0.0, 1.0);
        let value = (mean_01 * (1.0 - tension_ratio) * diversity_factor).clamp(0.0, 0.95);

        GlyphCoherence {
            modality_mean,
            tension_ratio,
            diversity_factor,
            value,
        }
    }
}

impl Default for GlyphInteractionMatrix {
    fn default() -> Self {
        let mut weights = [[0.0f64; N_FIELD_MODALITIES]; N_FIELD_MODALITIES];
        for i in 0..N_FIELD_MODALITIES {
            weights[i][i] = 1.0;
        }
        Self {
            weights,
            observation_count: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GlyphRegistry — all 70 glyphs encoded into HDC space
// ═══════════════════════════════════════════════════════════════════════════════

/// A single glyph entry with its HDC encoding.
#[derive(Debug, Clone)]
pub struct GlyphEntry {
    /// Registry identifier (e.g., "INF1", "TH_LI", "O23").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Structural class (Meta, Threshold, Spiral).
    pub class: GlyphClass,
    /// Core function description.
    pub core_function: String,
    /// Echo/activation phrase.
    pub echo_phrase: String,
    /// Field modality assignment.
    pub field_modality: FieldModality,
    /// Relational archetype name.
    pub relational_archetype: String,
    /// Somatic/embodied practice.
    pub somatic_practice: String,
    /// Sonic signature description.
    pub sonic_signature: String,
    /// Spiral index (0-56) for Canonical Spiral Glyphs, None otherwise.
    pub spiral_index: Option<u8>,
    /// HDC encoding: composite HV from name + core_function + echo_phrase.
    pub encoding: ContinuousHV,
}

/// Registry of all 70 glyphs with lookup indices.
pub struct GlyphRegistry {
    /// All glyph entries in order.
    pub entries: Vec<GlyphEntry>,
    /// ID → index lookup.
    by_id: HashMap<String, usize>,
    /// Modality → indices lookup.
    by_modality: HashMap<FieldModality, Vec<usize>>,
    /// Class → indices lookup.
    by_class: HashMap<GlyphClass, Vec<usize>>,
}

impl GlyphRegistry {
    /// Build the registry, encoding all 70 glyphs via TextHdcEncoder.
    pub fn new(dim: usize) -> Self {
        use super::glyph_registry_data::GLYPH_DATA;

        let encoder = TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2);
        let mut entries = Vec::with_capacity(GLYPH_DATA.len());
        let mut by_id = HashMap::new();
        let mut by_modality: HashMap<FieldModality, Vec<usize>> = HashMap::new();
        let mut by_class: HashMap<GlyphClass, Vec<usize>> = HashMap::new();

        for (i, raw) in GLYPH_DATA.iter().enumerate() {
            // Encode: name + core_function + echo_phrase → composite HV
            let text = format!("{} {} {}", raw.name, raw.core_function, raw.echo_phrase);
            let encoding = encoder.encode(&text);

            let modality = FieldModality::from_str_lossy(raw.modality);
            let class = GlyphClass::from_str_lossy(raw.class);

            entries.push(GlyphEntry {
                id: raw.id.to_string(),
                name: raw.name.to_string(),
                class,
                core_function: raw.core_function.to_string(),
                echo_phrase: raw.echo_phrase.to_string(),
                field_modality: modality,
                relational_archetype: raw.archetype.to_string(),
                somatic_practice: raw.somatic.to_string(),
                sonic_signature: raw.sonic.to_string(),
                spiral_index: raw.spiral_index,
                encoding,
            });

            by_id.insert(raw.id.to_string(), i);
            by_modality.entry(modality).or_default().push(i);
            by_class.entry(class).or_default().push(i);
        }

        Self {
            entries,
            by_id,
            by_modality,
            by_class,
        }
    }

    /// Look up a glyph by its registry ID.
    pub fn get(&self, id: &str) -> Option<&GlyphEntry> {
        self.by_id.get(id).map(|&idx| &self.entries[idx])
    }

    /// Get all glyphs with a given modality.
    pub fn by_modality(&self, modality: FieldModality) -> Vec<&GlyphEntry> {
        self.by_modality
            .get(&modality)
            .map(|indices| indices.iter().map(|&i| &self.entries[i]).collect())
            .unwrap_or_default()
    }

    /// Get all glyphs of a given class.
    pub fn by_class(&self, class: GlyphClass) -> Vec<&GlyphEntry> {
        self.by_class
            .get(&class)
            .map(|indices| indices.iter().map(|&i| &self.entries[i]).collect())
            .unwrap_or_default()
    }

    /// Find the nearest glyph to a given HV by cosine similarity.
    pub fn nearest(&self, hv: &ContinuousHV) -> Option<(&GlyphEntry, f32)> {
        self.entries
            .iter()
            .map(|entry| {
                let sim = hv.similarity(&entry.encoding);
                (entry, sim)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Total number of glyphs in the registry.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal helpers — softmax, KL, entropy for N_FIELD_MODALITIES
// ═══════════════════════════════════════════════════════════════════════════════

fn softmax_m(coords: &[f64; N_FIELD_MODALITIES], inv_temp: f64) -> [f64; N_FIELD_MODALITIES] {
    let max_val = coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_vals = [0.0f64; N_FIELD_MODALITIES];
    let mut sum = 0.0;
    for i in 0..N_FIELD_MODALITIES {
        exp_vals[i] = ((coords[i] - max_val) * inv_temp).exp();
        sum += exp_vals[i];
    }
    if sum > 0.0 {
        for v in &mut exp_vals {
            *v /= sum;
        }
    } else {
        exp_vals = [1.0 / N_FIELD_MODALITIES as f64; N_FIELD_MODALITIES];
    }
    exp_vals
}

fn kl_div_m(q: &[f64; N_FIELD_MODALITIES], p: &[f64; N_FIELD_MODALITIES]) -> f64 {
    let mut kl = 0.0;
    for i in 0..N_FIELD_MODALITIES {
        let qi = q[i].max(1e-12);
        let pi = p[i].max(1e-12);
        kl += qi * (qi / pi).ln();
    }
    kl.max(0.0)
}

fn entropy_m(q: &[f64; N_FIELD_MODALITIES]) -> f64 {
    let mut h = 0.0;
    for &qi in q {
        let qi = qi.max(1e-12);
        h -= qi * qi.ln();
    }
    h.max(0.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glyph_basis_creation() {
        let basis = GlyphBasis::new(256);
        assert_eq!(basis.vectors.len(), N_FIELD_MODALITIES);
        assert_eq!(basis.dim, 256);

        // Basis vectors should be distinct
        for i in 0..N_FIELD_MODALITIES {
            for j in (i + 1)..N_FIELD_MODALITIES {
                let sim = basis.vectors[i].similarity(&basis.vectors[j]).abs();
                assert!(sim < 0.9, "Modality vectors {i} and {j} too similar: {sim}");
            }
        }
    }

    #[test]
    fn test_glyph_basis_project_bounds() {
        let basis = GlyphBasis::new(256);
        let hv = ContinuousHV::random(256, 99);
        let coords = basis.project(&hv);
        assert_eq!(coords.len(), N_FIELD_MODALITIES);
        for c in &coords {
            assert!(c.is_finite());
            assert!(*c >= -1.0 && *c <= 1.0);
        }
    }

    #[test]
    fn test_modality_semantic_validity() {
        let basis = GlyphBasis::new(512);
        let encoder = TextHdcEncoder::with_sentiment(512, 3, 0.5, 0.2);

        // "root ground earth body" should project strongly onto Rooting (idx 2)
        let root_hv = encoder.encode("root ground earth body stabilize");
        let coords = basis.project(&root_hv);

        let rooting_idx = FieldModality::Rooting.index(); // 2
        let mut sorted: Vec<(usize, f64)> = coords.iter().copied().enumerate().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_2: Vec<usize> = sorted.iter().take(2).map(|(i, _)| *i).collect();

        assert!(
            top_2.contains(&rooting_idx),
            "Root words should project onto Rooting (idx {rooting_idx}), \
             but top-2 are {:?}. Coords: {coords:?}",
            top_2
        );
    }

    #[test]
    fn test_glyph_registry_completeness() {
        let registry = GlyphRegistry::new(256);
        assert_eq!(registry.len(), 70, "Registry should contain all 70 glyphs");

        let meta = registry.by_class(GlyphClass::Meta);
        let threshold = registry.by_class(GlyphClass::Threshold);
        let spiral = registry.by_class(GlyphClass::Spiral);

        assert_eq!(meta.len(), 4, "Expected 4 Meta-Glyphs");
        assert_eq!(threshold.len(), 9, "Expected 9 Threshold Glyphs");
        assert_eq!(spiral.len(), 57, "Expected 57 Spiral Glyphs");
        assert_eq!(
            meta.len() + threshold.len() + spiral.len(),
            70,
            "Class counts must sum to 70"
        );
    }

    #[test]
    fn test_glyph_interaction_matrix_symmetry() {
        let basis = GlyphBasis::new(256);
        let matrix = GlyphInteractionMatrix::from_basis(&basis);

        for i in 0..N_FIELD_MODALITIES {
            assert!(
                (matrix.weights[i][i] - 1.0).abs() < f64::EPSILON,
                "Diagonal must be 1.0"
            );
            for j in 0..N_FIELD_MODALITIES {
                assert!(
                    (matrix.weights[i][j] - matrix.weights[j][i]).abs() < f64::EPSILON,
                    "Matrix must be symmetric at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn test_glyph_free_energy_identical() {
        let coords = [0.3, 0.8, 0.1, 0.2, 0.5, 0.4, 0.1, 0.2, 0.6, 0.3, 0.7];
        let basis = GlyphBasis::new(256);
        let fe = basis.glyph_free_energy(&coords, &coords, 1.0);

        assert!(
            fe.kl_divergence < 0.01,
            "KL divergence should be ~0 for identical distributions, got {}",
            fe.kl_divergence
        );
        assert!(fe.free_energy.is_finite());
        assert!(fe.entropy.is_finite());
    }

    #[test]
    fn test_glyph_coherence_ceiling() {
        let matrix = GlyphInteractionMatrix::default();
        let coords = [0.9; N_FIELD_MODALITIES];
        let gc = matrix.glyph_coherence(&coords);
        assert!(
            gc.value <= 0.95,
            "Glyph coherence should never exceed 0.95, got {}",
            gc.value
        );
        assert!(
            gc.value > 0.7,
            "Maximally aligned coords should yield high coherence, got {}",
            gc.value
        );
    }

    #[test]
    fn test_glyph_class_count_invariants() {
        // Verify the GLYPH_DATA array has correct class distribution
        use super::super::glyph_registry_data::GLYPH_DATA;
        let meta_count = GLYPH_DATA.iter().filter(|g| g.class == "Meta").count();
        let threshold_count = GLYPH_DATA.iter().filter(|g| g.class == "Threshold").count();
        let spiral_count = GLYPH_DATA.iter().filter(|g| g.class == "Spiral").count();

        assert_eq!(meta_count, 4, "Expected 4 Meta entries in GLYPH_DATA");
        assert_eq!(
            threshold_count, 9,
            "Expected 9 Threshold entries in GLYPH_DATA"
        );
        assert_eq!(spiral_count, 57, "Expected 57 Spiral entries in GLYPH_DATA");
        assert_eq!(
            GLYPH_DATA.len(),
            70,
            "GLYPH_DATA must have exactly 70 entries"
        );
    }

    #[test]
    fn test_glyph_registry_nearest() {
        let registry = GlyphRegistry::new(256);
        let encoder = TextHdcEncoder::with_sentiment(256, 3, 0.5, 0.2);

        // Encoding similar words to "Ethical Emergence" should find it or a close match
        let hv = encoder.encode("ethical emergence responsibility coherence birth");
        let (entry, sim) = registry.nearest(&hv).expect("Registry should find a match");
        assert!(
            sim > 0.0,
            "Nearest similarity should be positive, got {sim}"
        );
        // The match should be from the Igniting or Integrated class
        assert!(!entry.name.is_empty(), "Nearest glyph should have a name");
    }

    #[test]
    fn test_glyph_interaction_observe_synergy() {
        let mut matrix = GlyphInteractionMatrix::default();
        // Co-activate Rooting (2) and Resonant (3)
        let mut coords = [0.0; N_FIELD_MODALITIES];
        coords[2] = 0.8; // Rooting
        coords[3] = 0.9; // Resonant
        for _ in 0..100 {
            matrix.observe(&coords, 0.05);
        }
        assert!(
            matrix.weights[2][3] > 0.5,
            "Co-activated modalities should develop synergy, got {}",
            matrix.weights[2][3]
        );
    }

    #[test]
    fn test_field_modality_all_unique() {
        let all = FieldModality::ALL;
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "All modalities must be unique");
            }
        }
        assert_eq!(all.len(), N_FIELD_MODALITIES);
    }

    #[test]
    fn test_field_modality_roundtrip() {
        for modality in &FieldModality::ALL {
            let name = modality.name();
            let parsed = FieldModality::from_str_lossy(name);
            assert_eq!(*modality, parsed, "Roundtrip failed for {name}");
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let coords = [0.3, 0.8, 0.1, 0.2, 0.5, 0.4, 0.1, 0.0, 0.6, 0.3, 0.7];
        let dist = softmax_m(&coords, 1.0);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_glyph_coherence_high_tension() {
        let mut matrix = GlyphInteractionMatrix::default();
        for i in 0..N_FIELD_MODALITIES {
            for j in (i + 1)..N_FIELD_MODALITIES {
                matrix.weights[i][j] = -0.5;
                matrix.weights[j][i] = -0.5;
            }
        }
        let coords = [0.5; N_FIELD_MODALITIES];
        let gc = matrix.glyph_coherence(&coords);
        assert!(
            (gc.tension_ratio - 1.0).abs() < f64::EPSILON,
            "All negative off-diags → tension_ratio=1.0, got {}",
            gc.tension_ratio
        );
        assert!(
            gc.value < f64::EPSILON,
            "Full tension → zero coherence, got {}",
            gc.value
        );
    }
}
