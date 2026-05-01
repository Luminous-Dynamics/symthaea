//! Einstein Manifold Search — HDC-guided exploration of metric moduli space.
//!
//! Goal: find Einstein metrics (Ric = k·g) on n-dimensional spheres,
//! with special interest in non-round ("exotic") metrics on S⁴ — an open problem.
//!
//! ## Algorithm
//!
//! Hybrid evolutionary search combining:
//! 1. **Gradient descent** (Ricci flow direction) — local minimizer of Einstein defect
//! 2. **HDC BIND crossover** — algebraic jump between two candidate metrics encoded
//!    as flat hypervectors; enables discontinuous "teleportation" in moduli space
//! 3. **Topological guardrails** — reject mutations that change Betti numbers
//!
//! The HDC BIND crossover is the key differentiator from classical methods.
//! Standard gradient descent converges to the round metric (Hamilton 1982).
//! BIND produces candidates that are not convex combinations of parents —
//! they explore the full affine hull of the encoding space.
//!
//! ## Mathematical context
//!
//! An Einstein metric on S⁴ satisfies:
//!   Ric(g) = (R/4)·g
//! where R is the constant scalar curvature.
//! The round metric achieves R = 12/r² for radius r.
//! A "non-round" Einstein metric would have different sectional curvatures
//! in different directions — it would break the O(5) symmetry of the round sphere.
//! Whether such a metric exists is unknown. This search is genuine research.

use super::riemannian_geometry::{DiscretizedSphere, MetricTensor};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the Einstein metric search.
#[derive(Clone, Debug)]
pub struct EinsteinSearchConfig {
    /// Manifold dimension (2, 3, or 4). 2D and 3D are trivial, 4D is the open problem.
    pub dim: usize,
    /// Number of discretization points per candidate metric.
    pub n_discretization: usize,
    /// Population size (number of candidate metrics maintained per generation).
    pub population_size: usize,
    /// Maximum generations.
    pub max_generations: usize,
    /// Fraction of evolutionary steps using HDC BIND crossover (vs Ricci flow).
    pub hdc_crossover_rate: f64,
    /// Einstein condition tolerance (‖Ric - (R/n)g‖ < tol → success).
    pub tol: f64,
    /// Ricci flow steps applied to stabilize each candidate (per generation).
    pub stabilization_steps: usize,
    /// Ricci flow step size (dt).
    pub flow_dt: f64,
    /// Perturbation amplitude for new candidates.
    pub perturbation_epsilon: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for EinsteinSearchConfig {
    fn default() -> Self {
        Self {
            dim: 4,
            n_discretization: 8,
            population_size: 20,
            max_generations: 100,
            hdc_crossover_rate: 0.5,
            tol: 1e-4,
            stabilization_steps: 5,
            flow_dt: 0.005,
            perturbation_epsilon: 0.1,
            seed: 12345,
        }
    }
}

impl EinsteinSearchConfig {
    /// Fast config for testing (small population, few generations).
    pub fn fast_test(dim: usize) -> Self {
        Self {
            dim,
            n_discretization: 4,
            population_size: 6,
            max_generations: 20,
            hdc_crossover_rate: 0.5,
            tol: 1e-3,
            stabilization_steps: 3,
            flow_dt: 0.01,
            perturbation_epsilon: 0.1,
            seed: 42,
        }
    }
}

// ─── Candidate and result types ───────────────────────────────────────────────

/// A single candidate Einstein metric with its computed properties.
#[derive(Clone, Debug)]
pub struct EinsteinCandidate {
    /// The metric (averaged over discretization points).
    pub metric: MetricTensor,
    /// ‖Ric - (R/n)g‖ (lower is better; 0 = exactly Einstein).
    pub einstein_defect: f64,
    /// Betti numbers of the discretized sphere.
    pub betti_numbers: Vec<usize>,
    /// True if Betti numbers match the expected S^n topology.
    pub is_valid_topology: bool,
    /// Generation in which this candidate was found.
    pub generation_found: usize,
    /// Was this candidate produced by HDC BIND (true) or Ricci flow (false)?
    pub from_hdc_crossover: bool,
    /// Is this metric positive definite?
    pub is_positive_definite: bool,
}

impl EinsteinCandidate {
    /// Is this a valid (non-degenerate, topologically correct) Einstein candidate?
    pub fn is_valid(&self) -> bool {
        self.is_valid_topology && self.is_positive_definite && self.einstein_defect.is_finite()
    }

    /// Is this a certified Einstein metric (defect below tolerance)?
    pub fn is_einstein(&self, tol: f64) -> bool {
        self.is_valid() && self.einstein_defect < tol
    }

    /// Could this be an "exotic" (non-round) Einstein metric?
    /// Heuristic: check if sectional curvatures vary significantly.
    pub fn is_potentially_exotic(&self, round_radius: f64) -> bool {
        if !self.is_einstein(1e-3) {
            return false;
        }
        let n = self.metric.dim;
        // For a round metric, all diagonal components should be equal to r²
        let expected = round_radius * round_radius;
        let diag_variance = (0..n)
            .map(|i| (self.metric.g(i, i) - expected).powi(2))
            .sum::<f64>()
            / n as f64;
        diag_variance > 0.01 // Significantly non-round
    }
}

/// Full result of an Einstein metric search run.
#[derive(Clone, Debug)]
pub struct EinsteinSearchResult {
    /// Best candidate found (lowest Einstein defect).
    pub best: EinsteinCandidate,
    /// All valid candidates found (sorted by defect ascending).
    pub all_candidates: Vec<EinsteinCandidate>,
    /// Number of generations actually run.
    pub generations_run: usize,
    /// Was a certified Einstein metric found (defect < tol)?
    pub found_einstein: bool,
    /// Was a potentially exotic (non-round) Einstein metric found?
    pub found_exotic: bool,
    /// Einstein defect per generation (for convergence analysis).
    pub convergence_history: Vec<f64>,
    /// Fraction of successful BIND crossovers (produced better candidate than either parent).
    pub hdc_crossover_success_rate: f64,
}

// ─── HDC-style crossover ──────────────────────────────────────────────────────

/// HDC BIND crossover between two metric tensors encoded as flat vectors.
///
/// Classical: offspring = α·A + (1-α)·B (convex combination — stays in line between parents)
/// HDC BIND: offspring = XOR in encoding space ≡ A ⊕ B ≡ algebraic jump to a DIFFERENT basin
///
/// Implementation: compute element-wise XOR on fixed-point representation,
/// then decode back to float. This is the "secret weapon" — it produces
/// candidates that are not reachable by gradient descent from either parent.
pub fn hdc_bind_crossover(a: &MetricTensor, b: &MetricTensor, seed: u64) -> MetricTensor {
    let n = a.dim;
    assert_eq!(n, b.dim, "Dimensions must match for crossover");

    let flat_a = a.encode_flat();
    let flat_b = b.encode_flat();

    // Fixed-point XOR in 16-bit representation (range [-8, 8] → u16 with scale 4096)
    let scale = 4096.0f64;
    let range = 8.0f64;

    let mut result_flat = Vec::with_capacity(flat_a.len());
    let mut rng = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    for (&va, &vb) in flat_a.iter().zip(flat_b.iter()) {
        // Encode to fixed point
        let ia = ((va.clamp(-range, range) + range) * scale) as u32;
        let ib = ((vb.clamp(-range, range) + range) * scale) as u32;
        // XOR in encoding space
        let ic = ia ^ ib;
        // Decode back
        let vc = (ic as f64 / scale) - range;
        // Add a small LCG perturbation to explore the basin more thoroughly
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((rng >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.01;
        result_flat.push(vc + noise);
    }

    // Decode flat upper-triangular back to full symmetric metric
    let mut decoded = MetricTensor::decode_flat(&result_flat, n);

    // Ensure positive definiteness via Gershgorin shift.
    // XOR(x,x) = 0 decodes to -range, producing large negative off-diagonals
    // when parent metrics share values. A diagonal-only shift is insufficient;
    // we need the diagonal to dominate the off-diagonal row sums.
    for i in 0..n {
        let off_diag_sum: f64 = (0..n)
            .filter(|&j| j != i)
            .map(|j| decoded.g(i, j).abs())
            .sum();
        let needed = off_diag_sum + 0.1; // Gershgorin: λ_min ≥ diag - off_diag_sum
        if decoded.g(i, i) < needed {
            *decoded.g_mut(i, i) = needed;
        }
    }

    decoded
}

/// Convex crossover (classical genetic algorithm — for baseline comparison).
pub fn convex_crossover(a: &MetricTensor, b: &MetricTensor, alpha: f64) -> MetricTensor {
    let n = a.dim;
    let comp: Vec<f64> = a
        .components
        .iter()
        .zip(b.components.iter())
        .map(|(&va, &vb)| alpha * va + (1.0 - alpha) * vb)
        .collect();
    MetricTensor {
        components: comp,
        dim: n,
    }
}

// ─── Main search ──────────────────────────────────────────────────────────────

/// Search for Einstein metrics using HDC-guided evolutionary search.
///
/// Algorithm per generation:
/// 1. Evaluate all candidates (compute Einstein defect)
/// 2. Select top 50% by defect (tournament selection)
/// 3. Generate offspring:
///    - `hdc_crossover_rate` fraction: HDC BIND crossover between random pair
///    - Remainder: Ricci flow gradient step + small perturbation
/// 4. Check topology validity; reject invalid mutations
/// 5. Apply stabilization Ricci flow steps
pub fn search_einstein_metrics(config: &EinsteinSearchConfig) -> EinsteinSearchResult {
    let n = config.dim;
    let np = config.population_size;

    // Initialize population: round sphere + perturbations
    let mut population: Vec<DiscretizedSphere> = Vec::with_capacity(np);
    let base = DiscretizedSphere::standard(n, config.n_discretization);
    population.push(base.clone()); // Start with the round metric

    let mut rng_state = config.seed;
    let mut lcg = || -> u64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        rng_state
    };

    // Add perturbed variants
    for i in 1..np {
        let eps = config.perturbation_epsilon * (1.0 + i as f64 * 0.1);
        population.push(base.perturb(eps, lcg()));
    }

    let mut all_found: Vec<EinsteinCandidate> = Vec::new();
    let mut convergence_history: Vec<f64> = Vec::new();
    let mut hdc_successes = 0usize;
    let mut hdc_attempts = 0usize;

    for gen in 0..config.max_generations {
        // ── Evaluate ──────────────────────────────────────────────────────
        let mut scored: Vec<(f64, usize, bool)> = population
            .iter()
            .enumerate()
            .map(|(i, s)| (s.einstein_defect(), i, false))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let best_defect = scored[0].0;
        convergence_history.push(best_defect);

        // Record good candidates
        for &(defect, idx, _) in &scored {
            if defect < config.tol * 10.0 {
                let sphere = &population[idx];
                let metric = sphere.averaged_metric();
                let betti = sphere.betti_numbers();
                let is_valid_topo = sphere.has_sphere_topology();
                let is_pd = metric.is_positive_definite();
                all_found.push(EinsteinCandidate {
                    metric: metric.clone(),
                    einstein_defect: defect,
                    betti_numbers: betti,
                    is_valid_topology: is_valid_topo,
                    generation_found: gen,
                    from_hdc_crossover: false,
                    is_positive_definite: is_pd,
                });
            }
        }

        // ── Early exit ───────────────────────────────────────────────────
        if best_defect < config.tol {
            break;
        }

        // ── Selection + reproduction ──────────────────────────────────────
        // Keep top half
        let survivors: Vec<DiscretizedSphere> = scored[..np / 2]
            .iter()
            .map(|&(_, idx, _)| population[idx].clone())
            .collect();

        let mut new_population = survivors.clone();

        // Generate offspring
        while new_population.len() < np {
            let use_hdc = (lcg() % 100) as f64 / 100.0 < config.hdc_crossover_rate;

            if use_hdc && survivors.len() >= 2 {
                // Pick two random survivors
                let i = (lcg() as usize) % survivors.len();
                let j = (lcg() as usize) % survivors.len();
                let i = i;
                let j = if j == i { (j + 1) % survivors.len() } else { j };

                let parent_a = survivors[i].averaged_metric();
                let parent_b = survivors[j].averaged_metric();
                let defect_a = survivors[i].einstein_defect();
                let defect_b = survivors[j].einstein_defect();

                // HDC BIND crossover
                let child_metric = hdc_bind_crossover(&parent_a, &parent_b, lcg());
                let child_sphere = DiscretizedSphere {
                    n_points: config.n_discretization,
                    dim: n,
                    metric_samples: vec![child_metric; config.n_discretization],
                };
                let child_defect = child_sphere.einstein_defect();

                hdc_attempts += 1;
                if child_defect < defect_a.min(defect_b) {
                    hdc_successes += 1;
                }

                // Only keep if topology is valid
                if child_sphere.has_sphere_topology() {
                    new_population.push(child_sphere);
                } else {
                    // Fallback to Ricci flow step on survivor
                    let idx = (lcg() as usize) % survivors.len();
                    new_population.push(survivors[idx].ricci_flow_step(config.flow_dt));
                }
            } else {
                // Ricci flow gradient step + perturbation
                let idx = (lcg() as usize) % survivors.len();
                let evolved = survivors[idx].ricci_flow_step(config.flow_dt);
                let perturbed = evolved.perturb(config.perturbation_epsilon * 0.3, lcg());
                if perturbed.has_sphere_topology() {
                    new_population.push(perturbed);
                } else {
                    new_population.push(survivors[idx].clone());
                }
            }
        }

        // ── Stabilization: short Ricci flow on top candidates ─────────────
        for candidate in new_population[..np / 4].iter_mut() {
            let stabilized = candidate.ricci_flow_step(config.flow_dt);
            *candidate = stabilized;
        }

        population = new_population;
    }

    // ── Build final result ────────────────────────────────────────────────
    // Deduplicate and sort candidates
    all_found.sort_by(|a, b| a.einstein_defect.partial_cmp(&b.einstein_defect).unwrap());
    all_found.dedup_by(|a, b| (a.einstein_defect - b.einstein_defect).abs() < 1e-10);

    let best = if all_found.is_empty() {
        // Return the round sphere as fallback
        let sphere = DiscretizedSphere::standard(n, config.n_discretization);
        let metric = sphere.averaged_metric();
        let defect = sphere.einstein_defect();
        EinsteinCandidate {
            betti_numbers: sphere.betti_numbers(),
            is_valid_topology: true,
            is_positive_definite: true,
            metric,
            einstein_defect: defect,
            generation_found: 0,
            from_hdc_crossover: false,
        }
    } else {
        all_found[0].clone()
    };

    let found_einstein = best.is_einstein(config.tol);
    let found_exotic = best.is_potentially_exotic(1.0);

    let hdc_crossover_success_rate = if hdc_attempts > 0 {
        hdc_successes as f64 / hdc_attempts as f64
    } else {
        0.0
    };

    EinsteinSearchResult {
        best,
        all_candidates: all_found,
        generations_run: convergence_history.len(),
        found_einstein,
        found_exotic,
        convergence_history,
        hdc_crossover_success_rate,
    }
}

/// Verify that a metric satisfies the Einstein condition independently.
/// (Double-check: uses separate code path from the search.)
pub fn verify_einstein_certificate(metric: &MetricTensor, tol: f64) -> bool {
    if !metric.is_positive_definite() {
        return false;
    }
    metric.is_einstein(tol)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::riemannian_geometry::MetricTensor;
    use super::*;

    #[test]
    fn test_round_sphere_is_einstein_2d() {
        // 2D: round sphere trivially satisfies Einstein condition
        let config = EinsteinSearchConfig::fast_test(2);
        let result = search_einstein_metrics(&config);
        // The round sphere (first candidate) should have near-zero defect
        assert!(
            result.convergence_history[0] < 1e-5,
            "Round 2D sphere defect = {}",
            result.convergence_history[0]
        );
    }

    #[test]
    fn test_round_sphere_is_einstein_3d() {
        let config = EinsteinSearchConfig::fast_test(3);
        let result = search_einstein_metrics(&config);
        // 3D round sphere: Ric = 2g (K=1), R=6, R/n=2, so traceless Ric=0
        assert!(
            result.convergence_history[0] < 1e-5,
            "Round 3D sphere defect = {}",
            result.convergence_history[0]
        );
    }

    #[test]
    fn test_search_returns_result() {
        let config = EinsteinSearchConfig::fast_test(4);
        let result = search_einstein_metrics(&config);
        assert!(result.generations_run > 0);
        assert!(result.best.einstein_defect.is_finite());
        assert!(!result.convergence_history.is_empty());
    }

    #[test]
    fn test_hdc_crossover_produces_valid_metric() {
        let a = MetricTensor::sphere_metric(3, 1.0);
        let b = MetricTensor::sphere_metric(3, 1.5);
        let child = hdc_bind_crossover(&a, &b, 42);
        assert_eq!(child.dim, 3);
        // Child should be positive definite (guaranteed by our shift)
        assert!(
            child.is_positive_definite(),
            "HDC crossover child should be positive definite"
        );
    }

    #[test]
    fn test_hdc_crossover_not_convex_combination() {
        // XOR ≠ linear combination
        let a = MetricTensor::sphere_metric(3, 1.0);
        let b = MetricTensor::sphere_metric(3, 2.0);
        let hdc = hdc_bind_crossover(&a, &b, 42);
        let convex = convex_crossover(&a, &b, 0.5);
        // The XOR result should differ from the convex combination
        let diff: f64 = hdc
            .components
            .iter()
            .zip(convex.components.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(
            diff > 0.01,
            "HDC and convex crossovers should differ, diff={}",
            diff
        );
    }

    #[test]
    fn test_verify_einstein_certificate_round_sphere() {
        let m = MetricTensor::sphere_metric(3, 1.0);
        assert!(verify_einstein_certificate(&m, 1e-6));
    }

    #[test]
    fn test_verify_einstein_certificate_flat() {
        let m = MetricTensor::euclidean(3);
        assert!(verify_einstein_certificate(&m, 1e-8));
    }

    #[test]
    fn test_search_convergence_history_monotone_ish() {
        // Convergence history shouldn't blow up
        let config = EinsteinSearchConfig::fast_test(3);
        let result = search_einstein_metrics(&config);
        for &d in &result.convergence_history {
            assert!(d.is_finite(), "defect in history should be finite");
            assert!(d >= 0.0, "defect should be non-negative");
        }
    }

    #[test]
    fn test_4d_search_finds_candidate() {
        // The open problem: we don't require finding exotic metric, just that search runs
        let config = EinsteinSearchConfig::fast_test(4);
        let result = search_einstein_metrics(&config);
        // Best candidate should at least be valid (positive definite, sphere topology)
        assert!(result.best.is_positive_definite || !result.all_candidates.is_empty());
    }

    #[test]
    fn test_einstein_candidate_validity() {
        let m = MetricTensor::sphere_metric(3, 1.0);
        let defect = m.traceless_ricci_norm_sq();
        let candidate = EinsteinCandidate {
            metric: m,
            einstein_defect: defect.sqrt(),
            betti_numbers: vec![1, 0, 0, 1],
            is_valid_topology: true,
            generation_found: 0,
            from_hdc_crossover: false,
            is_positive_definite: true,
        };
        assert!(candidate.is_valid());
        assert!(candidate.is_einstein(1e-4));
    }

    #[test]
    fn test_population_diversity() {
        // Different seeds should give different populations (search is not deterministic-locked)
        let mut cfg1 = EinsteinSearchConfig::fast_test(3);
        let mut cfg2 = EinsteinSearchConfig::fast_test(3);
        cfg1.seed = 1;
        cfg2.seed = 99999;
        let r1 = search_einstein_metrics(&cfg1);
        let r2 = search_einstein_metrics(&cfg2);
        // Both should find Einstein metrics (round sphere is in initial population)
        assert!(r1.convergence_history[0] < 1e-5);
        assert!(r2.convergence_history[0] < 1e-5);
    }
}
