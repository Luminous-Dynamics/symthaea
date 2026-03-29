// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spark Engine ↔ Physics Bridge integration.
//!
//! Connects fusion reactor design space exploration with the physics
//! equation catalog. When parameter sweeps plateau (all nearby points
//! have similar feasibility), queries the physics bridge for structurally
//! analogous equations that suggest different parameter regimes.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::physics::{
    DesignPoint, DesignSpaceMapper, FusionReaction, MultiObjectiveDesign, ParametricSweep,
    ParetoFrontier, ParetoOptimizer,
};
use symthaea_physics_bridge::{PhysicsBridge, PhysicsDomain};

/// Minimum number of consecutive points to consider a plateau.
const MIN_PLATEAU_LEN: usize = 3;

/// Maximum metric variation (fraction) within a plateau.
const PLATEAU_METRIC_TOLERANCE: f64 = 0.05;

/// Which sweep variant to run.
///
/// Matches the three implemented `DesignSpaceMapper` sweep methods.
/// (Does NOT reuse `SweepType` which has a 4th `DutyCycleVsLifetime`
/// variant with no corresponding mapper method.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepKind {
    /// Power vs shielding thickness.
    PowerVsShielding,
    /// Power vs system mass.
    PowerVsMass,
    /// Power vs cost per kW.
    PowerVsCost,
}

/// A region of consecutive sweep points with similar metric values.
#[derive(Debug, Clone)]
pub struct PlateauRegion {
    /// Index of the first point in the plateau.
    pub start_idx: usize,
    /// Index of the last point (inclusive).
    pub end_idx: usize,
    /// Average metric value across the plateau.
    pub avg_metric: f64,
    /// Whether the plateau is in the feasible region.
    pub feasible: bool,
}

/// A physics analogy discovered for a design point.
#[derive(Debug, Clone)]
pub struct PhysicsAnalogy {
    /// Index of the design point in the sweep.
    pub design_point_idx: usize,
    /// Name of the matched equation.
    pub equation_name: String,
    /// Physics domain of the match.
    pub domain: PhysicsDomain,
    /// Cosine similarity score (0.0–1.0).
    pub similarity: f32,
    /// Human-readable insight.
    pub insight: String,
}

/// Enriched sweep result with plateau analysis and physics analogies.
#[derive(Debug, Clone)]
pub struct GuidedSweepResult {
    /// The underlying parametric sweep.
    pub sweep: ParametricSweep,
    /// Detected plateau regions.
    pub plateau_regions: Vec<PlateauRegion>,
    /// Physics analogies for representative plateau points.
    pub analogies: Vec<PhysicsAnalogy>,
}

/// Refinement suggestion: where to add resolution in a sweep.
#[derive(Debug, Clone)]
pub struct RefinementSuggestion {
    /// Index of the design point near the feasibility boundary.
    pub point_idx: usize,
    /// Similarity to the closest physics equation.
    pub physics_similarity: f32,
    /// Name of the matched equation.
    pub equation_name: String,
    /// Suggested reason for adding resolution here.
    pub reason: String,
}

/// Physics analogy annotating a Pareto frontier point.
#[derive(Debug, Clone)]
pub struct ParetoAnalogy {
    /// Index into the frontier vector.
    pub frontier_idx: usize,
    /// Power level of this frontier design.
    pub power_kw: f64,
    /// Name of the matched equation.
    pub equation_name: String,
    /// Physics domain of the match.
    pub domain: PhysicsDomain,
    /// Cosine similarity score (0.0–1.0).
    pub similarity: f32,
    /// Human-readable insight.
    pub insight: String,
}

/// Pareto frontier with physics analogies for each frontier point.
#[derive(Debug, Clone)]
pub struct GuidedParetoResult {
    /// The computed Pareto frontier.
    pub frontier: ParetoFrontier,
    /// Physics analogies for each frontier design.
    pub frontier_analogies: Vec<ParetoAnalogy>,
}

/// Cached basis vectors for encoding `DesignPoint` into HDC space.
///
/// Follows the `DimensionalEncoder` pattern: basis vectors are created once
/// and reused, avoiding 3×16,384-D allocations (~195 KB) per call.
pub struct DesignPointEncoder {
    power_basis: ContinuousHV,
    param_basis: ContinuousHV,
    metric_basis: ContinuousHV,
    gradient_basis: ContinuousHV,
}

impl DesignPointEncoder {
    /// Create with deterministic seeds.
    pub fn new() -> Self {
        let dim = 16_384;
        Self {
            power_basis: ContinuousHV::random(dim, 0xDE51_6001),
            param_basis: ContinuousHV::random(dim, 0xDE51_6002),
            metric_basis: ContinuousHV::random(dim, 0xDE51_6003),
            gradient_basis: ContinuousHV::random(dim, 0xDE51_6004),
        }
    }

    /// Encode a design point (power, param, metric) as a normalized HV.
    pub fn encode(&self, point: &DesignPoint) -> ContinuousHV {
        let power_scaled = self.power_basis.scale(point.power_kw as f32);
        let param_scaled = self.param_basis.scale(point.param_value as f32);
        let metric_scaled = self.metric_basis.scale(point.metric_value as f32);
        let bundled = ContinuousHV::bundle(&[&power_scaled, &param_scaled, &metric_scaled]);
        bundled.normalize()
    }

    /// Encode with an additional gradient dimension for feasibility boundaries.
    ///
    /// Boundary points get ±1.0 gradient (sign indicates direction of transition);
    /// interior/exterior points get ~0.
    pub fn encode_with_gradient(&self, point: &DesignPoint, gradient: f32) -> ContinuousHV {
        let power_scaled = self.power_basis.scale(point.power_kw as f32);
        let param_scaled = self.param_basis.scale(point.param_value as f32);
        let metric_scaled = self.metric_basis.scale(point.metric_value as f32);
        let gradient_scaled = self.gradient_basis.scale(gradient);
        let bundled = ContinuousHV::bundle(&[
            &power_scaled,
            &param_scaled,
            &metric_scaled,
            &gradient_scaled,
        ]);
        bundled.normalize()
    }
}

/// A plateau region that appears in multiple sweep dimensions.
#[derive(Debug, Clone)]
pub struct CrossCorrelatedPlateau {
    /// Which sweep kinds contributed.
    pub sweep_kinds: Vec<SweepKind>,
    /// Overlapping power range across the contributing sweeps.
    pub power_range: (f64, f64),
    /// Average metric per sweep kind.
    pub avg_metrics: Vec<(SweepKind, f64)>,
    /// Whether all contributing plateaus were feasible.
    pub all_feasible: bool,
}

/// Combined result from running shielding, mass, and cost sweeps.
#[derive(Debug, Clone)]
pub struct MultiSweepResult {
    pub shielding: GuidedSweepResult,
    pub mass: GuidedSweepResult,
    pub cost: GuidedSweepResult,
    pub cross_plateaus: Vec<CrossCorrelatedPlateau>,
}

/// Bridges `DesignSpaceMapper` with `PhysicsBridge` for physics-guided
/// design space exploration.
pub struct GuidedDesignExplorer {
    mapper: DesignSpaceMapper,
    bridge: PhysicsBridge,
    pareto: ParetoOptimizer,
    encoder: DesignPointEncoder,
}

impl GuidedDesignExplorer {
    /// Create from a genesis seed (deterministic).
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            mapper: DesignSpaceMapper::from_genesis(genesis),
            bridge: PhysicsBridge::new(),
            pareto: ParetoOptimizer::from_genesis(genesis),
            encoder: DesignPointEncoder::new(),
        }
    }

    /// Run a guided power-vs-shielding sweep (convenience wrapper).
    pub fn guided_sweep(
        &mut self,
        reaction: FusionReaction,
        power_range: (f64, f64),
        n_points: usize,
    ) -> GuidedSweepResult {
        self.guided_sweep_with_kind(SweepKind::PowerVsShielding, reaction, power_range, n_points)
    }

    /// Run a guided sweep of the specified kind with plateau detection and
    /// physics analogy discovery.
    pub fn guided_sweep_with_kind(
        &mut self,
        kind: SweepKind,
        reaction: FusionReaction,
        power_range: (f64, f64),
        n_points: usize,
    ) -> GuidedSweepResult {
        let sweep = match kind {
            SweepKind::PowerVsShielding => {
                self.mapper
                    .sweep_power_vs_shielding(reaction, power_range, n_points)
            }
            SweepKind::PowerVsMass => {
                self.mapper
                    .sweep_power_vs_mass(reaction, power_range, n_points)
            }
            SweepKind::PowerVsCost => {
                self.mapper
                    .sweep_power_vs_cost(reaction, power_range, n_points)
            }
        };
        let plateau_regions = detect_plateaus(&sweep.points);
        let analogies = self.find_analogies_for_plateaus(&sweep, &plateau_regions);
        GuidedSweepResult {
            sweep,
            plateau_regions,
            analogies,
        }
    }

    /// Compute a Pareto frontier and annotate each frontier point with a
    /// physics analogy from the HDC equation catalog.
    pub fn guided_pareto(
        &mut self,
        reaction: FusionReaction,
        power_range: (f64, f64),
        n_points: usize,
    ) -> GuidedParetoResult {
        let frontier = self
            .pareto
            .compute_pareto_frontier(reaction, power_range, n_points);
        let frontier_analogies = self.annotate_pareto_frontier(&frontier);
        GuidedParetoResult {
            frontier,
            frontier_analogies,
        }
    }

    /// Format a Pareto report with physics analogies appended.
    pub fn format_pareto_report(&self, result: &GuidedParetoResult) -> String {
        let mut report = self.pareto.format_report(&result.frontier);
        if !result.frontier_analogies.is_empty() {
            report.push_str("\n\nPhysics Analogies for Pareto Frontier:\n");
            for analogy in &result.frontier_analogies {
                report.push_str(&format!(
                    "  [{:>2}] {:.1} kW — {} ({:?}), sim {:.1}%\n       {}\n",
                    analogy.frontier_idx,
                    analogy.power_kw,
                    analogy.equation_name,
                    analogy.domain,
                    analogy.similarity * 100.0,
                    analogy.insight,
                ));
            }
        }
        report
    }

    /// Identify points near feasibility boundaries where physics similarity
    /// suggests interesting transitions, recommending additional resolution.
    pub fn suggest_refinement(&mut self, sweep: &ParametricSweep) -> Vec<RefinementSuggestion> {
        let mut suggestions = Vec::new();
        let points = &sweep.points;
        if points.len() < 2 {
            return suggestions;
        }

        for i in 1..points.len() {
            if points[i].feasible != points[i - 1].feasible {
                // Feasibility boundary — encode with gradient
                let gradient = if points[i].feasible { 1.0 } else { -1.0 };
                let hv = self.encoder.encode_with_gradient(&points[i], gradient);
                let result_hv = self.bridge.query_hv(&hv, 3);
                let _ = result_hv; // consume to update last_results
                let results = self.bridge.last_results();
                if let Some(top) = results.first() {
                    suggestions.push(RefinementSuggestion {
                        point_idx: i,
                        physics_similarity: top.score,
                        equation_name: top.name.clone(),
                        reason: format!(
                            "Feasibility boundary ({}→{}) near {:.1} kW; {:.0}% match to {}",
                            if points[i - 1].feasible {
                                "feasible"
                            } else {
                                "infeasible"
                            },
                            if points[i].feasible {
                                "feasible"
                            } else {
                                "infeasible"
                            },
                            points[i].power_kw,
                            top.score * 100.0,
                            top.name,
                        ),
                    });
                }
            }
        }
        suggestions
    }

    /// Run all three sweep kinds and cross-correlate their plateau regions.
    pub fn multi_sweep(
        &mut self,
        reaction: FusionReaction,
        power_range: (f64, f64),
        n_points: usize,
    ) -> MultiSweepResult {
        let shielding = self.guided_sweep_with_kind(
            SweepKind::PowerVsShielding,
            reaction,
            power_range,
            n_points,
        );
        let mass =
            self.guided_sweep_with_kind(SweepKind::PowerVsMass, reaction, power_range, n_points);
        let cost =
            self.guided_sweep_with_kind(SweepKind::PowerVsCost, reaction, power_range, n_points);

        let cross_plateaus = cross_correlate_plateaus(&[
            (SweepKind::PowerVsShielding, &shielding),
            (SweepKind::PowerVsMass, &mass),
            (SweepKind::PowerVsCost, &cost),
        ]);

        MultiSweepResult {
            shielding,
            mass,
            cost,
            cross_plateaus,
        }
    }

    /// Find the minimum feasible power for a reaction (delegates to mapper).
    pub fn find_minimum_feasible_power(&self, reaction: FusionReaction) -> Option<f64> {
        self.mapper.find_minimum_feasible_power(reaction)
    }

    /// Get an ASCII chart of a sweep (delegates to mapper).
    pub fn ascii_chart(&self, sweep: &ParametricSweep, width: usize, height: usize) -> String {
        self.mapper.ascii_chart(sweep, width, height)
    }

    // ── Internal ──

    fn find_analogies_for_plateaus(
        &mut self,
        sweep: &ParametricSweep,
        plateaus: &[PlateauRegion],
    ) -> Vec<PhysicsAnalogy> {
        let mut analogies = Vec::new();
        for plateau in plateaus {
            // Pick the midpoint of the plateau as representative
            let mid = (plateau.start_idx + plateau.end_idx) / 2;
            let point = &sweep.points[mid];
            let hv = self.encoder.encode(point);
            let result_hv = self.bridge.query_hv(&hv, 3);
            let _ = result_hv; // consume to update last_results
            let results = self.bridge.last_results();
            if let Some(top) = results.first() {
                let insight = format!(
                    "Plateau at {:.1}–{:.1} kW ({}, metric≈{:.2}) resembles {} ({:?})",
                    sweep.points[plateau.start_idx].power_kw,
                    sweep.points[plateau.end_idx].power_kw,
                    if plateau.feasible {
                        "feasible"
                    } else {
                        "infeasible"
                    },
                    plateau.avg_metric,
                    top.name,
                    top.domain,
                );
                analogies.push(PhysicsAnalogy {
                    design_point_idx: mid,
                    equation_name: top.name.clone(),
                    domain: top.domain.clone(),
                    similarity: top.score,
                    insight,
                });
            }
        }
        analogies
    }

    fn annotate_pareto_frontier(&mut self, frontier: &ParetoFrontier) -> Vec<ParetoAnalogy> {
        let mut analogies = Vec::new();
        for (idx, design) in frontier.frontier.iter().enumerate() {
            let point = multi_objective_to_design_point(design);
            let hv = self.encoder.encode(&point);
            let result_hv = self.bridge.query_hv(&hv, 3);
            let _ = result_hv;
            let results = self.bridge.last_results();
            if let Some(top) = results.first() {
                analogies.push(ParetoAnalogy {
                    frontier_idx: idx,
                    power_kw: design.power_kw,
                    equation_name: top.name.clone(),
                    domain: top.domain.clone(),
                    similarity: top.score,
                    insight: format!(
                        "Pareto point {:.1} kW (mass={:.0} kg, cost=${:.0}/kW) → {} ({:?})",
                        design.power_kw, design.mass_kg, design.cost_per_kw, top.name, top.domain,
                    ),
                });
            }
        }
        analogies
    }
}

/// Detect plateau regions in a sweep's design points.
///
/// A plateau is a run of ≥ `MIN_PLATEAU_LEN` consecutive points with the same
/// feasibility status and metric values within `PLATEAU_METRIC_TOLERANCE` of
/// each other (relative to the plateau's average).
fn detect_plateaus(points: &[DesignPoint]) -> Vec<PlateauRegion> {
    let mut plateaus = Vec::new();
    if points.is_empty() {
        return plateaus;
    }

    let mut start = 0;
    while start < points.len() {
        let feasible = points[start].feasible;
        let mut end = start;

        // Extend the run while feasibility matches
        while end + 1 < points.len() && points[end + 1].feasible == feasible {
            end += 1;
        }

        if end - start + 1 >= MIN_PLATEAU_LEN {
            // Check metric variation
            let metrics: Vec<f64> = points[start..=end].iter().map(|p| p.metric_value).collect();
            let avg = metrics.iter().sum::<f64>() / metrics.len() as f64;
            let max_dev = metrics
                .iter()
                .map(|m| (m - avg).abs())
                .fold(0.0f64, f64::max);
            let relative_dev = if avg.abs() > 1e-12 {
                max_dev / avg.abs()
            } else {
                max_dev
            };

            if relative_dev <= PLATEAU_METRIC_TOLERANCE {
                plateaus.push(PlateauRegion {
                    start_idx: start,
                    end_idx: end,
                    avg_metric: avg,
                    feasible,
                });
            }
        }

        start = end + 1;
    }

    plateaus
}

/// Convert a `MultiObjectiveDesign` into a `DesignPoint` for encoding.
fn multi_objective_to_design_point(design: &MultiObjectiveDesign) -> DesignPoint {
    DesignPoint {
        power_kw: design.power_kw,
        param_value: design.mass_kg,
        feasible: design.feasible,
        metric_value: design.cost_per_kw,
        notes: String::new(),
    }
}

/// Cross-correlate plateau regions from multiple sweeps.
///
/// Two plateaus overlap if their power ranges (derived from sweep points) intersect.
fn cross_correlate_plateaus(
    sweeps: &[(SweepKind, &GuidedSweepResult)],
) -> Vec<CrossCorrelatedPlateau> {
    // Collect all (kind, plateau, power_lo, power_hi) tuples.
    struct TaggedPlateau<'a> {
        kind: SweepKind,
        plateau: &'a PlateauRegion,
        power_lo: f64,
        power_hi: f64,
    }

    let mut tagged: Vec<TaggedPlateau> = Vec::new();
    for &(kind, ref result) in sweeps {
        for plateau in &result.plateau_regions {
            let lo = result.sweep.points[plateau.start_idx].power_kw;
            let hi = result.sweep.points[plateau.end_idx].power_kw;
            let (power_lo, power_hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            tagged.push(TaggedPlateau {
                kind,
                plateau,
                power_lo,
                power_hi,
            });
        }
    }

    // Pairwise overlap check
    let mut results = Vec::new();
    let n = tagged.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if tagged[i].kind == tagged[j].kind {
                continue; // only cross-sweep correlations
            }
            // Check power range overlap
            let overlap_lo = tagged[i].power_lo.max(tagged[j].power_lo);
            let overlap_hi = tagged[i].power_hi.min(tagged[j].power_hi);
            if overlap_lo <= overlap_hi {
                results.push(CrossCorrelatedPlateau {
                    sweep_kinds: vec![tagged[i].kind, tagged[j].kind],
                    power_range: (overlap_lo, overlap_hi),
                    avg_metrics: vec![
                        (tagged[i].kind, tagged[i].plateau.avg_metric),
                        (tagged[j].kind, tagged[j].plateau.avg_metric),
                    ],
                    all_feasible: tagged[i].plateau.feasible && tagged[j].plateau.feasible,
                });
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::physics::SweepType;

    fn genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("spark_physics_bridge_test")
    }

    #[test]
    fn test_guided_explorer_creates() {
        let g = genesis();
        let _explorer = GuidedDesignExplorer::from_genesis(&g);
    }

    #[test]
    fn test_detect_plateaus_empty() {
        assert!(detect_plateaus(&[]).is_empty());
    }

    #[test]
    fn test_detect_plateaus_uniform() {
        let points: Vec<DesignPoint> = (0..5)
            .map(|i| DesignPoint {
                power_kw: 10.0 + i as f64,
                param_value: 50.0,
                feasible: true,
                metric_value: 1.0,
                notes: String::new(),
            })
            .collect();
        let plateaus = detect_plateaus(&points);
        assert_eq!(plateaus.len(), 1);
        assert_eq!(plateaus[0].start_idx, 0);
        assert_eq!(plateaus[0].end_idx, 4);
        assert!(plateaus[0].feasible);
    }

    #[test]
    fn test_detect_plateaus_split_by_feasibility() {
        let mut points: Vec<DesignPoint> = (0..6)
            .map(|i| DesignPoint {
                power_kw: 10.0 + i as f64,
                param_value: 50.0,
                feasible: i < 3,
                metric_value: 1.0,
                notes: String::new(),
            })
            .collect();
        // Both halves are exactly 3 points — minimum for a plateau
        let plateaus = detect_plateaus(&points);
        assert_eq!(plateaus.len(), 2);
        assert!(plateaus[0].feasible);
        assert!(!plateaus[1].feasible);

        // Make the second half only 2 points (below threshold)
        points.pop();
        let plateaus2 = detect_plateaus(&points);
        assert_eq!(plateaus2.len(), 1);
    }

    #[test]
    fn test_encode_design_point_finite() {
        let encoder = DesignPointEncoder::new();
        let point = DesignPoint {
            power_kw: 25.0,
            param_value: 100.0,
            feasible: true,
            metric_value: 0.5,
            notes: String::new(),
        };
        let hv = encoder.encode(&point);
        assert_eq!(hv.values.len(), 16_384);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_encoder_deterministic() {
        let enc_a = DesignPointEncoder::new();
        let enc_b = DesignPointEncoder::new();
        let point = DesignPoint {
            power_kw: 10.0,
            param_value: 50.0,
            feasible: true,
            metric_value: 1.0,
            notes: String::new(),
        };
        let hv_a = enc_a.encode(&point);
        let hv_b = enc_b.encode(&point);
        assert_eq!(hv_a.values, hv_b.values, "Encoder should be deterministic");
    }

    #[test]
    fn test_encode_with_gradient_differs() {
        let encoder = DesignPointEncoder::new();
        let point = DesignPoint {
            power_kw: 25.0,
            param_value: 100.0,
            feasible: true,
            metric_value: 0.5,
            notes: String::new(),
        };
        let hv_no_grad = encoder.encode(&point);
        let hv_pos_grad = encoder.encode_with_gradient(&point, 1.0);
        let hv_neg_grad = encoder.encode_with_gradient(&point, -1.0);
        // All should be 16384-D and finite
        assert_eq!(hv_pos_grad.values.len(), 16_384);
        assert!(hv_pos_grad.values.iter().all(|v| v.is_finite()));
        // With gradient should differ from without
        let diff_pos: f32 = hv_no_grad
            .values
            .iter()
            .zip(hv_pos_grad.values.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff_pos > 0.01, "Gradient should change the encoding");
        // Positive and negative gradient should differ
        let diff_sign: f32 = hv_pos_grad
            .values
            .iter()
            .zip(hv_neg_grad.values.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff_sign > 0.01, "Opposite gradients should differ");
    }

    #[test]
    fn test_guided_sweep_returns_results() {
        let g = genesis();
        let mut explorer = GuidedDesignExplorer::from_genesis(&g);
        let result = explorer.guided_sweep(FusionReaction::DD, (1.0, 50.0), 20);
        assert!(!result.sweep.points.is_empty());
        // Plateaus may or may not exist depending on the sweep shape
    }

    #[test]
    fn test_suggest_refinement_at_boundaries() {
        let g = genesis();
        let mut explorer = GuidedDesignExplorer::from_genesis(&g);
        let sweep = explorer
            .mapper
            .sweep_power_vs_shielding(FusionReaction::DD, (1.0, 50.0), 20);
        let suggestions = explorer.suggest_refinement(&sweep);
        // If there's a feasibility boundary, we should get suggestions
        let has_boundary = sweep
            .points
            .windows(2)
            .any(|w| w[0].feasible != w[1].feasible);
        if has_boundary {
            assert!(
                !suggestions.is_empty(),
                "Should suggest refinement at feasibility boundaries"
            );
        }
    }

    #[test]
    fn test_plateau_metric_tolerance() {
        // Points with high metric variation should not form a plateau
        let points: Vec<DesignPoint> = (0..5)
            .map(|i| DesignPoint {
                power_kw: 10.0 + i as f64,
                param_value: 50.0,
                feasible: true,
                metric_value: 1.0 + (i as f64) * 0.5, // 1.0, 1.5, 2.0, 2.5, 3.0
                notes: String::new(),
            })
            .collect();
        let plateaus = detect_plateaus(&points);
        assert!(
            plateaus.is_empty(),
            "High metric variation should prevent plateau detection"
        );
    }

    // ── Fix 2: SweepKind tests ──

    #[test]
    fn test_guided_sweep_mass() {
        let g = genesis();
        let mut explorer = GuidedDesignExplorer::from_genesis(&g);
        let result = explorer.guided_sweep_with_kind(
            SweepKind::PowerVsMass,
            FusionReaction::DD,
            (1.0, 50.0),
            10,
        );
        assert!(!result.sweep.points.is_empty());
        // Mass sweep should produce param_values representing mass
        assert!(
            result.sweep.points.iter().any(|p| p.param_value > 0.0),
            "Mass sweep should have positive param values"
        );
    }

    #[test]
    fn test_guided_sweep_cost() {
        let g = genesis();
        let mut explorer = GuidedDesignExplorer::from_genesis(&g);
        let result = explorer.guided_sweep_with_kind(
            SweepKind::PowerVsCost,
            FusionReaction::DD,
            (1.0, 50.0),
            10,
        );
        assert!(!result.sweep.points.is_empty());
        assert!(
            result.sweep.points.iter().any(|p| p.param_value > 0.0),
            "Cost sweep should have positive param values"
        );
    }

    // ── Fix 3: Pareto tests ──

    #[test]
    fn test_guided_pareto_returns_frontier() {
        let g = genesis();
        let mut explorer = GuidedDesignExplorer::from_genesis(&g);
        let result = explorer.guided_pareto(FusionReaction::DD, (1.0, 50.0), 15);
        assert!(
            !result.frontier.all_designs.is_empty(),
            "Pareto should evaluate designs"
        );
        // Frontier should be non-empty (at least 1 rank-0 design)
        assert!(
            !result.frontier.frontier.is_empty(),
            "Pareto frontier should have rank-0 designs"
        );
    }

    #[test]
    fn test_guided_pareto_analogies_indexed_correctly() {
        let g = genesis();
        let mut explorer = GuidedDesignExplorer::from_genesis(&g);
        let result = explorer.guided_pareto(FusionReaction::DD, (1.0, 50.0), 15);
        for analogy in &result.frontier_analogies {
            assert!(
                analogy.frontier_idx < result.frontier.frontier.len(),
                "Analogy frontier_idx {} out of bounds (frontier len {})",
                analogy.frontier_idx,
                result.frontier.frontier.len(),
            );
            // Power should match the frontier design it references
            let design = &result.frontier.frontier[analogy.frontier_idx];
            assert!(
                (analogy.power_kw - design.power_kw).abs() < 1e-6,
                "Analogy power_kw should match frontier design"
            );
        }
    }

    // ── Multi-sweep tests ──

    #[test]
    fn test_multi_sweep_all_non_empty() {
        let g = genesis();
        let mut explorer = GuidedDesignExplorer::from_genesis(&g);
        let result = explorer.multi_sweep(FusionReaction::DD, (1.0, 50.0), 15);
        assert!(
            !result.shielding.sweep.points.is_empty(),
            "Shielding sweep should be non-empty"
        );
        assert!(
            !result.mass.sweep.points.is_empty(),
            "Mass sweep should be non-empty"
        );
        assert!(
            !result.cost.sweep.points.is_empty(),
            "Cost sweep should be non-empty"
        );
    }

    #[test]
    fn test_cross_correlate_synthetic_overlapping() {
        // Create two sweeps with overlapping plateau power ranges
        let make_points = |feasible: bool, metric: f64| -> Vec<DesignPoint> {
            (0..5)
                .map(|i| DesignPoint {
                    power_kw: 10.0 + i as f64, // 10-14 kW
                    param_value: 50.0,
                    feasible,
                    metric_value: metric,
                    notes: String::new(),
                })
                .collect()
        };
        let sweep_a = GuidedSweepResult {
            sweep: ParametricSweep {
                sweep_type: SweepType::PowerVsShielding,
                reaction: FusionReaction::DT,
                x_values: (0..5).map(|i| 10.0 + i as f64).collect(),
                x_label: "Power (kW)".into(),
                y_label: "Shielding (cm)".into(),
                points: make_points(true, 1.0),
                feasibility_boundary: vec![],
            },
            plateau_regions: vec![PlateauRegion {
                start_idx: 0,
                end_idx: 4,
                avg_metric: 1.0,
                feasible: true,
            }],
            analogies: vec![],
        };
        let sweep_b = GuidedSweepResult {
            sweep: ParametricSweep {
                sweep_type: SweepType::PowerVsMass,
                reaction: FusionReaction::DT,
                x_values: (0..5).map(|i| 10.0 + i as f64).collect(),
                x_label: "Power (kW)".into(),
                y_label: "Mass (kg)".into(),
                points: make_points(true, 2.0),
                feasibility_boundary: vec![],
            },
            plateau_regions: vec![PlateauRegion {
                start_idx: 0,
                end_idx: 4,
                avg_metric: 2.0,
                feasible: true,
            }],
            analogies: vec![],
        };

        let results = cross_correlate_plateaus(&[
            (SweepKind::PowerVsShielding, &sweep_a),
            (SweepKind::PowerVsMass, &sweep_b),
        ]);
        assert_eq!(results.len(), 1, "Should find 1 cross-correlated plateau");
        assert!(results[0].all_feasible);
        assert_eq!(results[0].sweep_kinds.len(), 2);
        assert!((results[0].power_range.0 - 10.0).abs() < 1e-6);
        assert!((results[0].power_range.1 - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_correlate_non_overlapping_empty() {
        let make_points = |base: f64| -> Vec<DesignPoint> {
            (0..5)
                .map(|i| DesignPoint {
                    power_kw: base + i as f64,
                    param_value: 50.0,
                    feasible: true,
                    metric_value: 1.0,
                    notes: String::new(),
                })
                .collect()
        };
        // Sweep A: 10-14 kW
        let sweep_a = GuidedSweepResult {
            sweep: ParametricSweep {
                sweep_type: SweepType::PowerVsShielding,
                reaction: FusionReaction::DT,
                x_values: (0..5).map(|i| 10.0 + i as f64).collect(),
                x_label: "Power (kW)".into(),
                y_label: "Shielding (cm)".into(),
                points: make_points(10.0),
                feasibility_boundary: vec![],
            },
            plateau_regions: vec![PlateauRegion {
                start_idx: 0,
                end_idx: 4,
                avg_metric: 1.0,
                feasible: true,
            }],
            analogies: vec![],
        };
        // Sweep B: 20-24 kW (no overlap)
        let sweep_b = GuidedSweepResult {
            sweep: ParametricSweep {
                sweep_type: SweepType::PowerVsMass,
                reaction: FusionReaction::DT,
                x_values: (0..5).map(|i| 20.0 + i as f64).collect(),
                x_label: "Power (kW)".into(),
                y_label: "Mass (kg)".into(),
                points: make_points(20.0),
                feasibility_boundary: vec![],
            },
            plateau_regions: vec![PlateauRegion {
                start_idx: 0,
                end_idx: 4,
                avg_metric: 2.0,
                feasible: true,
            }],
            analogies: vec![],
        };

        let results = cross_correlate_plateaus(&[
            (SweepKind::PowerVsShielding, &sweep_a),
            (SweepKind::PowerVsMass, &sweep_b),
        ]);
        assert!(
            results.is_empty(),
            "Non-overlapping plateaus should produce no cross-correlations"
        );
    }
}
