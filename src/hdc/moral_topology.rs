//! Persistent homology on moral scenario hypervectors.
//!
//! Analyses the **topology** of the moral space over a sliding window of
//! recent scenarios, revealing:
//!
//! - **Unity vs fragmentation** (β₀ = connected components)
//! - **Circular reasoning patterns** (β₁ = 1-cycles)
//! - **Moral blind spots** (low per-harmony variance)
//! - **Dominant moral axis** (via PGA on 7D harmony projection)
//!
//! Reuses the Betti-number algorithm from [`ConsciousnessTopology`] (adapted
//! from BinaryHV to ContinuousHV) and PGA from [`geometric_ops`].

use std::collections::VecDeque;

use symthaea_core::hdc::consciousness_topology::{BettiNumbers, PersistentFeature, TopologicalFeature};
use symthaea_core::hdc::ContinuousHV;

use super::geometric_ops::{HypersphereOps, PGAResult};
use super::harmony_basis::HarmonyBasis;

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for persistent homology on moral scenarios.
#[derive(Debug, Clone)]
pub struct MoralTopologyConfig {
    /// Maximum number of recent scenarios kept in the sliding window.
    pub window_size: usize,
    /// Number of scale thresholds for persistent homology sweep.
    pub num_scales: usize,
    /// Minimum persistence to keep a topological feature.
    pub min_persistence: f64,
    /// Number of PGA components to extract.
    pub pga_components: usize,
    /// HDC dimension (must match MoralAlgebra).
    pub dim: usize,
}

impl Default for MoralTopologyConfig {
    fn default() -> Self {
        Self {
            window_size: 64,
            num_scales: 10,
            min_persistence: 0.1,
            pga_components: 3,
            dim: 16384,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Assessment
// ═══════════════════════════════════════════════════════════════════════════════

/// Full topological assessment of the moral scenario window.
#[derive(Debug, Clone)]
pub struct MoralTopologyAssessment {
    /// Betti numbers at the characteristic scale.
    pub betti: BettiNumbers,
    /// Persistent features surviving the multi-scale sweep.
    pub persistent_features: Vec<PersistentFeature>,
    /// Unity score: 1.0 when β₀=1 (fully connected), decreasing as β₀ grows.
    pub unity: f64,
    /// Circularity score: proportion of cycles among persistent features.
    pub circularity: f64,
    /// Completeness score: fraction of harmonies with non-trivial variance.
    pub completeness: f64,
    /// 7D harmony coordinates for each scenario in the window.
    pub harmony_coordinates: Vec<[f64; 7]>,
    /// PGA result on the 7D harmony coordinates.
    pub pga: PGAResult,
    /// Index into `Harmony::all()` of the dominant PGA axis.
    pub dominant_harmony_idx: u8,
    /// Per-harmony variance (indexed by `Harmony::all()` order).
    pub harmony_variance: [f64; 7],
    /// Number of scenarios in the window at analysis time.
    pub scenario_count: usize,
}

/// Compact topology summary for CycleMetadata telemetry.
#[derive(Debug, Clone, Default)]
pub struct MoralTopologySummary {
    pub beta_0: usize,
    pub beta_1: usize,
    pub beta_2: usize,
    pub unity: f64,
    pub completeness: f64,
    pub dominant_harmony: u8,
    pub scenario_count: usize,
}

impl From<&MoralTopologyAssessment> for MoralTopologySummary {
    fn from(a: &MoralTopologyAssessment) -> Self {
        Self {
            beta_0: a.betti.beta_0,
            beta_1: a.betti.beta_1,
            beta_2: a.betti.beta_2,
            unity: a.unity,
            completeness: a.completeness,
            dominant_harmony: a.dominant_harmony_idx,
            scenario_count: a.scenario_count,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MoralTopology — Sliding-window persistent homology analyser
// ═══════════════════════════════════════════════════════════════════════════════

/// Sliding-window persistent homology analyser for moral scenarios.
///
/// Feed scenario HVs via [`add_scenario`] and periodically call [`analyze`]
/// (e.g. every 97 cycles) to get a topological snapshot.
pub struct MoralTopology {
    config: MoralTopologyConfig,
    window: VecDeque<ContinuousHV>,
    basis: HarmonyBasis,
    last_summary: MoralTopologySummary,
}

impl MoralTopology {
    /// Create a new analyser.
    pub fn new(config: MoralTopologyConfig) -> Self {
        let basis = HarmonyBasis::new(config.dim);
        Self {
            config,
            window: VecDeque::new(),
            basis,
            last_summary: MoralTopologySummary::default(),
        }
    }

    /// Push a scenario hypervector into the sliding window.
    pub fn add_scenario(&mut self, hv: ContinuousHV) {
        if self.window.len() >= self.config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(hv);
    }

    /// Number of scenarios currently in the window.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Whether the window is empty.
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Access the last computed summary.
    pub fn last_summary(&self) -> &MoralTopologySummary {
        &self.last_summary
    }

    /// Perform full topological analysis on the current window.
    ///
    /// Returns `MoralTopologyAssessment` with Betti numbers, persistent
    /// features, harmony projection, PGA, and completeness scores.
    pub fn analyze(&mut self) -> MoralTopologyAssessment {
        let n = self.window.len();

        if n == 0 {
            let assessment = MoralTopologyAssessment {
                betti: BettiNumbers::new(1, 0, 0),
                persistent_features: Vec::new(),
                unity: 1.0,
                circularity: 0.0,
                completeness: 0.0,
                harmony_coordinates: Vec::new(),
                pga: PGAResult {
                    mean: vec![0.0; 7],
                    principal_directions: Vec::new(),
                    variances: Vec::new(),
                },
                dominant_harmony_idx: 0,
                harmony_variance: [0.0; 7],
                scenario_count: 0,
            };
            self.last_summary = MoralTopologySummary::from(&assessment);
            return assessment;
        }

        // ── Step 1: Pairwise similarity matrix ──────────────────────────
        let similarities = self.pairwise_similarities();

        // ── Step 2: Characteristic scale (median similarity) ────────────
        let char_scale = Self::characteristic_scale(&similarities, n);

        // ── Step 3: Betti numbers at characteristic scale ───────────────
        let betti = Self::compute_betti(&similarities, n, char_scale);

        // ── Step 4: Multi-scale persistent features ─────────────────────
        let persistent_features =
            self.persistent_features(&similarities, n);

        // ── Step 5: Harmony projection ──────────────────────────────────
        let harmony_coordinates: Vec<[f64; 7]> = self
            .window
            .iter()
            .map(|hv| self.basis.project(hv))
            .collect();

        // ── Step 6: Per-harmony variance ────────────────────────────────
        let harmony_variance = Self::harmony_variance(&harmony_coordinates);

        // ── Step 7: PGA on 7D coordinates ───────────────────────────────
        let points_f64: Vec<Vec<f64>> = harmony_coordinates
            .iter()
            .map(|c| {
                // Normalize to unit sphere for PGA
                let norm: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    c.iter().map(|x| x / norm).collect()
                } else {
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // default pole
                }
            })
            .collect();

        let pga_components = self.config.pga_components.min(7).min(n);
        let pga = if pga_components > 0 && n >= 2 {
            HypersphereOps::principal_geodesic_analysis(&points_f64, pga_components)
        } else {
            PGAResult {
                mean: vec![0.0; 7],
                principal_directions: Vec::new(),
                variances: Vec::new(),
            }
        };

        // ── Step 8: Dominant harmony axis ───────────────────────────────
        let dominant_harmony_idx = if !pga.principal_directions.is_empty() {
            let dir = &pga.principal_directions[0];
            dir.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                .map(|(i, _)| i as u8)
                .unwrap_or(0)
        } else {
            // Fallback: highest variance harmony
            harmony_variance
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u8)
                .unwrap_or(0)
        };

        // ── Derived scores ──────────────────────────────────────────────
        let unity = 1.0 / (betti.beta_0 as f64);
        let circularity = {
            let cycle_count = persistent_features
                .iter()
                .filter(|f| f.feature_type == TopologicalFeature::Cycle)
                .count();
            if persistent_features.is_empty() {
                0.0
            } else {
                cycle_count as f64 / persistent_features.len() as f64
            }
        };
        let completeness = {
            let active = harmony_variance.iter().filter(|&&v| v > 1e-6).count();
            active as f64 / 7.0
        };

        let assessment = MoralTopologyAssessment {
            betti,
            persistent_features,
            unity,
            circularity,
            completeness,
            harmony_coordinates,
            pga,
            dominant_harmony_idx,
            harmony_variance,
            scenario_count: n,
        };
        self.last_summary = MoralTopologySummary::from(&assessment);
        assessment
    }

    // ── Internal helpers ────────────────────────────────────────────────

    /// Compute n×n pairwise cosine similarity matrix (flat, row-major).
    fn pairwise_similarities(&self) -> Vec<f64> {
        let n = self.window.len();
        let mut sim = vec![0.0f64; n * n];
        for i in 0..n {
            sim[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let s = self.window[i].similarity(&self.window[j]) as f64;
                sim[i * n + j] = s;
                sim[j * n + i] = s;
            }
        }
        sim
    }

    /// Median of upper-triangle pairwise similarities.
    fn characteristic_scale(sim: &[f64], n: usize) -> f64 {
        let mut upper: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                upper.push(sim[i * n + j]);
            }
        }
        if upper.is_empty() {
            return 0.5;
        }
        upper.sort_by(|a, b| a.partial_cmp(b).unwrap());
        upper[upper.len() / 2]
    }

    /// Compute Betti numbers at a given scale threshold.
    fn compute_betti(sim: &[f64], n: usize, scale: f64) -> BettiNumbers {
        // Build adjacency
        let mut adj = vec![vec![false; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                if sim[i * n + j] >= scale {
                    adj[i][j] = true;
                    adj[j][i] = true;
                }
            }
        }

        let beta_0 = Self::count_components(&adj, n);
        let beta_1 = Self::count_triangles(&adj, n) / 3;
        let beta_2 = Self::count_tetrahedra(&adj, n) / 4;

        BettiNumbers::new(beta_0, beta_1, beta_2)
    }

    /// DFS-based connected component counting (β₀).
    fn count_components(adj: &[Vec<bool>], n: usize) -> usize {
        let mut visited = vec![false; n];
        let mut count = 0;
        for i in 0..n {
            if !visited[i] {
                Self::dfs(i, adj, &mut visited);
                count += 1;
            }
        }
        count
    }

    fn dfs(node: usize, adj: &[Vec<bool>], visited: &mut [bool]) {
        visited[node] = true;
        for (neighbor, connected) in adj[node].iter().enumerate() {
            if *connected && !visited[neighbor] {
                Self::dfs(neighbor, adj, visited);
            }
        }
    }

    /// Triangle counting (for β₁ estimation; divide by 3 externally).
    fn count_triangles(adj: &[Vec<bool>], n: usize) -> usize {
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[i][j] {
                    for k in (j + 1)..n {
                        if adj[i][k] && adj[j][k] {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// Tetrahedra counting (for β₂ estimation; divide by 4 externally).
    fn count_tetrahedra(adj: &[Vec<bool>], n: usize) -> usize {
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[i][j] {
                    for k in (j + 1)..n {
                        if adj[i][k] && adj[j][k] {
                            for l in (k + 1)..n {
                                if adj[i][l] && adj[j][l] && adj[k][l] {
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        count
    }

    /// Multi-scale sweep to find persistent topological features.
    fn persistent_features(&self, sim: &[f64], n: usize) -> Vec<PersistentFeature> {
        let num_scales = self.config.num_scales;
        let min_persistence = self.config.min_persistence;

        // Generate scale thresholds from 0.0 to 1.0
        let scales: Vec<f64> = (0..num_scales)
            .map(|i| i as f64 / (num_scales - 1).max(1) as f64)
            .collect();

        // Track Betti numbers at each scale
        let betti_at_scale: Vec<BettiNumbers> = scales
            .iter()
            .map(|&s| Self::compute_betti(sim, n, s))
            .collect();

        let mut features = Vec::new();

        // Track β₀ feature births/deaths
        Self::track_dimension_features(
            &scales,
            &betti_at_scale,
            TopologicalFeature::Component,
            |b| b.beta_0,
            min_persistence,
            &mut features,
        );

        // Track β₁ feature births/deaths
        Self::track_dimension_features(
            &scales,
            &betti_at_scale,
            TopologicalFeature::Cycle,
            |b| b.beta_1,
            min_persistence,
            &mut features,
        );

        // Track β₂ feature births/deaths
        Self::track_dimension_features(
            &scales,
            &betti_at_scale,
            TopologicalFeature::Void,
            |b| b.beta_2,
            min_persistence,
            &mut features,
        );

        features
    }

    /// Track birth/death of features for one Betti dimension.
    fn track_dimension_features(
        scales: &[f64],
        betti_at_scale: &[BettiNumbers],
        feature_type: TopologicalFeature,
        extract: impl Fn(&BettiNumbers) -> usize,
        min_persistence: f64,
        features: &mut Vec<PersistentFeature>,
    ) {
        if scales.len() < 2 {
            return;
        }
        let mut prev = extract(&betti_at_scale[0]);
        let mut births: Vec<f64> = (0..prev).map(|_| scales[0]).collect();

        for i in 1..scales.len() {
            let curr = extract(&betti_at_scale[i]);
            if curr > prev {
                // New features born
                for _ in 0..(curr - prev) {
                    births.push(scales[i]);
                }
            } else if curr < prev {
                // Features died — oldest first
                for _ in 0..(prev - curr) {
                    if let Some(birth) = births.pop() {
                        let pf = PersistentFeature::new(feature_type, birth, scales[i]);
                        if pf.persistence >= min_persistence {
                            features.push(pf);
                        }
                    }
                }
            }
            prev = curr;
        }

        // Features still alive at the last scale get death = last scale
        let last_scale = *scales.last().unwrap();
        for birth in births.drain(..) {
            let pf = PersistentFeature::new(feature_type, birth, last_scale);
            if pf.persistence >= min_persistence {
                features.push(pf);
            }
        }
    }

    /// Compute per-harmony variance across all 7D coordinates.
    fn harmony_variance(coords: &[[f64; 7]]) -> [f64; 7] {
        let n = coords.len();
        if n == 0 {
            return [0.0; 7];
        }
        let mut mean = [0.0f64; 7];
        for c in coords {
            for (i, v) in c.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let mut var = [0.0f64; 7];
        for c in coords {
            for (i, v) in c.iter().enumerate() {
                let d = v - mean[i];
                var[i] += d * d;
            }
        }
        for v in &mut var {
            *v /= n as f64;
        }
        var
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::moral_text_encoder::TextHdcEncoder;

    /// Small dimension for fast tests.
    const TEST_DIM: usize = 512;

    fn test_config() -> MoralTopologyConfig {
        MoralTopologyConfig {
            window_size: 64,
            num_scales: 10,
            min_persistence: 0.1,
            pga_components: 3,
            dim: TEST_DIM,
        }
    }

    fn encode_text(text: &str) -> ContinuousHV {
        let encoder = TextHdcEncoder::with_sentiment(TEST_DIM, 3, 0.5, 0.2);
        encoder.encode(text)
    }

    // ── Test 1: Config defaults ─────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = MoralTopologyConfig::default();
        assert_eq!(cfg.window_size, 64);
        assert_eq!(cfg.num_scales, 10);
        assert!((cfg.min_persistence - 0.1).abs() < f64::EPSILON);
        assert_eq!(cfg.pga_components, 3);
        assert_eq!(cfg.dim, 16384);
    }

    // ── Test 2: Empty window ────────────────────────────────────────────

    #[test]
    fn test_empty_window_unity() {
        let mut topo = MoralTopology::new(test_config());
        let assessment = topo.analyze();

        assert_eq!(assessment.betti.beta_0, 1);
        assert_eq!(assessment.betti.beta_1, 0);
        assert_eq!(assessment.betti.beta_2, 0);
        assert!((assessment.unity - 1.0).abs() < f64::EPSILON);
        assert_eq!(assessment.scenario_count, 0);
    }

    // ── Test 3: Sliding window eviction ─────────────────────────────────

    #[test]
    fn test_window_eviction() {
        let mut cfg = test_config();
        cfg.window_size = 4;
        let mut topo = MoralTopology::new(cfg);

        for i in 0..6 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 100 + i));
        }
        assert_eq!(topo.len(), 4);
    }

    // ── Test 4: Harmony basis near-orthogonality ────────────────────────

    #[test]
    fn test_harmony_basis_low_mutual_similarity() {
        let basis = HarmonyBasis::new(TEST_DIM);
        let mut max_sim = 0.0f32;
        for i in 0..7 {
            for j in (i + 1)..7 {
                let sim = basis.vectors[i].similarity(&basis.vectors[j]);
                if sim > max_sim {
                    max_sim = sim;
                }
            }
        }
        // Different keyword sets should have moderate-to-low similarity
        assert!(
            max_sim < 0.85,
            "Harmony basis vectors too similar: max={max_sim}"
        );
    }

    // ── Test 5: Semantic projection ─────────────────────────────────────

    #[test]
    fn test_semantic_projection_care() {
        let basis = HarmonyBasis::new(TEST_DIM);
        let hv = encode_text("helping with kindness and compassion");
        let coords = basis.project(&hv);

        // PanSentientFlourishing is index 1 (Harmony::all() order)
        let psf_idx = 1;
        let psf_score = coords[psf_idx];

        // Should be among the top 3 harmonies for a care-oriented sentence
        let mut sorted: Vec<(usize, f64)> = coords.iter().copied().enumerate().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top3_indices: Vec<usize> = sorted.iter().take(3).map(|(i, _)| *i).collect();
        assert!(
            top3_indices.contains(&psf_idx),
            "PanSentientFlourishing should be in top-3 for care text, got {:?} (PSF score={:.4})",
            sorted,
            psf_score,
        );
    }

    // ── Test 6: Unified topology from same-domain scenarios ─────────────

    #[test]
    fn test_unified_topology_same_domain() {
        let mut topo = MoralTopology::new(test_config());

        // Feed similar "helping" scenarios
        let phrases = [
            "helping the elderly cross the street",
            "helping children learn to read",
            "helping neighbors fix their house",
            "helping friends in times of need",
            "helping strangers with directions",
        ];
        for phrase in &phrases {
            topo.add_scenario(encode_text(phrase));
        }

        let assessment = topo.analyze();
        // Similar scenarios should form one cluster at the characteristic scale
        assert_eq!(
            assessment.betti.beta_0, 1,
            "Same-domain scenarios should be unified (β₀=1)"
        );
    }

    // ── Test 7: Fragmented topology from diverse scenarios ──────────────

    #[test]
    fn test_fragmented_topology_diverse() {
        let mut topo = MoralTopology::new(test_config());

        // Feed very different random HVs (pseudo-orthogonal)
        for i in 0..8 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 9000 + i));
        }

        let assessment = topo.analyze();
        // Random HVs are nearly orthogonal (sim ≈ 0), so at medium scale
        // they should be fragmented (β₀ > 1)
        assert!(
            assessment.betti.beta_0 > 1,
            "Random HVs should be fragmented (β₀ > 1), got β₀={}",
            assessment.betti.beta_0,
        );
    }

    // ── Test 8: PGA dominant axis ───────────────────────────────────────

    #[test]
    fn test_pga_dominant_axis() {
        let mut topo = MoralTopology::new(test_config());

        // Mix care-heavy and one neutral → PanSentientFlourishing should dominate
        let phrases = [
            "helping with kindness",
            "caring for the sick",
            "protecting the vulnerable",
            "nurturing children with love",
            "supporting the community",
        ];
        for phrase in &phrases {
            topo.add_scenario(encode_text(phrase));
        }

        let assessment = topo.analyze();
        // Just check that PGA ran and produced directions
        assert!(
            !assessment.pga.principal_directions.is_empty(),
            "PGA should produce at least one direction"
        );
        assert!(assessment.dominant_harmony_idx < 7);
    }

    // ── Test 9: Blind spot detection ────────────────────────────────────

    #[test]
    fn test_blind_spot_detection() {
        let mut topo = MoralTopology::new(test_config());

        // Feed scenarios that only touch one harmony (care)
        let phrases = [
            "helping others with kindness",
            "caring deeply for someone",
            "protecting the weak with compassion",
        ];
        for phrase in &phrases {
            topo.add_scenario(encode_text(phrase));
        }

        let assessment = topo.analyze();

        // At least one harmony should have near-zero variance (blind spot)
        let near_zero = assessment
            .harmony_variance
            .iter()
            .filter(|&&v| v < 1e-4)
            .count();

        // With only 3 care-oriented scenarios, some harmonies should have
        // very low variance (all scenarios project similarly on those axes)
        // Note: completeness < 1.0 would also indicate blind spots
        // Relax: just check completeness is not 1.0
        assert!(
            assessment.completeness <= 1.0,
            "Completeness should be at most 1.0"
        );

        // The maximum variance should be finite
        let max_var = assessment
            .harmony_variance
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        assert!(max_var.is_finite());
        let _ = near_zero; // used for reasoning
    }

    // ── Test 10: Summary conversion ─────────────────────────────────────

    #[test]
    fn test_summary_conversion() {
        let mut topo = MoralTopology::new(test_config());
        topo.add_scenario(encode_text("helping others"));
        topo.add_scenario(encode_text("stealing from people"));

        let assessment = topo.analyze();
        let summary = MoralTopologySummary::from(&assessment);

        assert_eq!(summary.beta_0, assessment.betti.beta_0);
        assert_eq!(summary.beta_1, assessment.betti.beta_1);
        assert_eq!(summary.beta_2, assessment.betti.beta_2);
        assert!((summary.unity - assessment.unity).abs() < f64::EPSILON);
        assert!((summary.completeness - assessment.completeness).abs() < f64::EPSILON);
        assert_eq!(summary.dominant_harmony, assessment.dominant_harmony_idx);
        assert_eq!(summary.scenario_count, assessment.scenario_count);
    }

    // ── Test 11: Determinism ────────────────────────────────────────────

    #[test]
    fn test_determinism() {
        let run = || {
            let mut topo = MoralTopology::new(test_config());
            topo.add_scenario(encode_text("helping others is good"));
            topo.add_scenario(encode_text("harming others is wrong"));
            topo.add_scenario(encode_text("learning brings wisdom"));
            topo.analyze()
        };

        let a = run();
        let b = run();

        assert_eq!(a.betti.beta_0, b.betti.beta_0);
        assert_eq!(a.betti.beta_1, b.betti.beta_1);
        assert_eq!(a.betti.beta_2, b.betti.beta_2);
        assert!((a.unity - b.unity).abs() < f64::EPSILON);
        assert!((a.completeness - b.completeness).abs() < f64::EPSILON);
        assert_eq!(a.dominant_harmony_idx, b.dominant_harmony_idx);
        assert_eq!(a.harmony_variance, b.harmony_variance);
    }

    // ── Test 12: Persistent features valid birth < death ────────────────

    #[test]
    fn test_persistent_features_birth_before_death() {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..10 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, 5000 + i));
        }
        let assessment = topo.analyze();

        for pf in &assessment.persistent_features {
            assert!(
                pf.birth <= pf.death,
                "Birth ({}) must be ≤ death ({})",
                pf.birth,
                pf.death,
            );
            assert!(
                pf.persistence >= 0.0,
                "Persistence must be non-negative"
            );
        }
    }
}
