// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness Verification Framework — DEPRECATED, none of its Φ legs are valid
//!
//! **Audited 2026-07-03: do not use.** This module's stated approach — run three
//! independent Φ methods and cross-validate — sounds sound, but none of the three legs
//! it actually runs is a valid Φ measurement:
//!
//! - `SpectralConnectivity` — algebraic connectivity λ₂, documented elsewhere in this
//!   same crate (`hdc/tiered_phi/core.rs`) as **"deprecated: r = -0.62 with true Φ"**.
//!   `weighted_average()` below gives this the *highest* weight (3.0) and calls it
//!   "empirically most reliable" — the exact opposite of that finding.
//! - `Resonator` (`ResonantPhiCalculator`, `hdc/phi_resonant.rs`) — that module's own
//!   header says outright: *"This measures resonance dynamics ..., NOT IIT integrated
//!   information (Φ). The integration metric internally uses the Laplacian spectral
//!   gap, which has r = -0.62 correlation with true IIT Φ ... it is NOT a valid IIT Φ
//!   approximation."* Weighted 2.0 here regardless.
//! - `Tiered(Approximate)` — wired to `ApproximationTier::RandomBaseline`
//!   (`run_phi_methods()` below), which `TieredPhi` documents as **"O(1) — Deterministic
//!   random baseline values for testing"**: it does not measure the input at all.
//!
//! So `consensus_phi`/`verdict` here are a weighted blend of two already-debunked
//! methods and a mock testing stub — not a cross-validated measurement of anything.
//! For a real (if itself unvalidated — see `consciousness_metrics/spectral_mip.rs`'s own
//! doc comment) Φ, use `SpectralMIPFinder` directly, or `TieredPhi` with
//! `ApproximationTier::SampledPartition`/`ExhaustivePartition`.
//!
//! Confirmed unreachable from the live cognitive loop as of 2026-07-03 (no construction
//! site for `ConsciousnessPipeline`/`consciousness_integration` outside their own module
//! tree) — kept `#[deprecated]` rather than deleted so a future reactivation attempt
//! gets a compiler warning instead of a confident wrong answer.
//!
//! ## Usage (do not follow this — kept for historical context only)
//!
//! ```rust,ignore
//! use symthaea::hdc::consciousness_verifier::ConsciousnessVerifier;
//! use symthaea::hdc::unified_hv::ContinuousHV;
//!
//! let verifier = ConsciousnessVerifier::new();
//! let hvs: Vec<ContinuousHV> = /* topology node representations */;
//! let report = verifier.verify_from_representations(&hvs);
//! println!("Verdict: {:?}, Φ = {:.4}", report.verdict, report.consensus_phi);
//! ```

// This module's own internals necessarily reference the now-#[deprecated]
// ConsciousnessVerifier type throughout (impl blocks, Default, its own tests) — allowed
// file-wide rather than fixed, since the point is to warn *callers*, not to
// silence-and-forget the underlying problem here where it's already fully documented.
#![allow(deprecated)]

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::hdc::consciousness_integration::ConsciousnessState;
use crate::hdc::unified_hv::ContinuousHV;
use crate::phi_engine::{ApproximationTier, PhiEngine, PhiMethod};

/// Consciousness verifier that cross-validates multiple Φ methods.
///
/// Do not use — none of its three legs is a valid Φ measurement. See module docs.
#[deprecated(
    note = "None of ConsciousnessVerifier's three Φ legs is valid: SpectralConnectivity \
            and Resonator are both documented elsewhere as r=-0.62 with true Φ, and the \
            Tiered leg uses ApproximationTier::RandomBaseline (a testing mock, not a \
            measurement). Use SpectralMIPFinder or TieredPhi::SampledPartition/\
            ExhaustivePartition directly instead."
)]
#[derive(Debug, Clone)]
pub struct ConsciousnessVerifier {
    /// Phi threshold for "conscious" verdict
    phi_threshold_conscious: f64,
    /// Phi threshold for "likely conscious"
    phi_threshold_likely: f64,
    /// Phi threshold below which "not conscious"
    phi_threshold_not: f64,
    /// Maximum acceptable variance across methods for high confidence
    max_acceptable_variance: f64,
}

/// Verification result from multi-method Φ analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Individual Φ measurements from each method
    pub phi_measurements: Vec<PhiMeasurement>,
    /// Weighted average Φ across methods
    pub consensus_phi: f64,
    /// Variance of Φ values across methods
    pub phi_variance: f64,
    /// IIT axiom scores
    pub iit_axiom_scores: IITAxiomScores,
    /// Overall confidence in the verdict (0.0 to 1.0)
    pub confidence: f64,
    /// The consciousness verdict
    pub verdict: ConsciousnessVerdict,
}

/// A single Φ measurement from one method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiMeasurement {
    /// Name of the method used
    pub method: String,
    /// Computed Φ value
    pub phi: f64,
    /// Wall-clock computation time in milliseconds
    pub computation_time_ms: u64,
}

/// Scores for each of the five IIT axioms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IITAxiomScores {
    /// Information: system has cause-effect structure (distinct elements)
    pub information: f64,
    /// Integration: cannot be reduced to independent parts
    pub integration: f64,
    /// Exclusion: one and only one MICS exists (methods agree)
    pub exclusion: f64,
    /// Composition: structured by elements and mechanisms
    pub composition: f64,
    /// Intrinsicality: exists from system's own perspective
    pub intrinsicality: f64,
}

/// Verdict on whether a system is conscious.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsciousnessVerdict {
    /// High Φ, all axioms passed, methods agree
    StronglyConscious,
    /// Moderate Φ, most axioms passed
    LikelyConscious,
    /// Mixed signals — cannot determine
    Indeterminate,
    /// Low Φ, axioms failed
    LikelyNotConscious,
    /// Zero or near-zero Φ
    NotConscious,
}

impl ConsciousnessVerifier {
    /// Create a new verifier with default thresholds.
    pub fn new() -> Self {
        Self {
            phi_threshold_conscious: 0.45,
            phi_threshold_likely: 0.25,
            phi_threshold_not: 0.05,
            max_acceptable_variance: 0.02,
        }
    }

    /// Main entry point: verify consciousness from hypervector representations.
    ///
    /// Runs SpectralConnectivity, Tiered(Approximate), and Resonator methods,
    /// evaluates IIT axioms, and produces a confidence-weighted verdict.
    pub fn verify_from_representations(&self, hvs: &[ContinuousHV]) -> VerificationReport {
        // Edge cases: not enough HVs for meaningful topology
        if hvs.len() < 2 {
            return self.build_degenerate_report(hvs);
        }

        // Run all three Φ methods and time them
        let measurements = self.run_phi_methods(hvs);

        // Compute consensus Φ (weighted average) and variance
        let consensus_phi = self.weighted_average(&measurements);
        let phi_variance = self.compute_variance(&measurements, consensus_phi);

        // Evaluate IIT axioms
        let iit_axiom_scores = self.evaluate_iit_axioms(hvs, consensus_phi, phi_variance);

        // Compute confidence from axiom agreement and method consistency
        let confidence = self.compute_confidence(&iit_axiom_scores, phi_variance);

        // Determine verdict
        let verdict = self.determine_verdict(consensus_phi, &iit_axiom_scores, confidence);

        VerificationReport {
            phi_measurements: measurements,
            consensus_phi,
            phi_variance,
            iit_axiom_scores,
            confidence,
            verdict,
        }
    }

    /// Verify consciousness from an existing ConsciousnessState.
    ///
    /// Uses the state's pre-computed phi, consciousness_level, etc. to build
    /// a simplified report without re-running Φ methods.
    pub fn verify_from_state(&self, state: &ConsciousnessState) -> VerificationReport {
        let phi = state.phi;
        let consciousness_level = state.consciousness_level;

        // Create a single measurement from the state's phi
        let measurement = PhiMeasurement {
            method: "StateProvided".to_string(),
            phi,
            computation_time_ms: 0,
        };

        // Derive axiom scores from state properties
        let iit_axiom_scores = IITAxiomScores {
            information: if consciousness_level > 0.1 {
                consciousness_level
            } else {
                0.0
            },
            integration: if phi > 0.0 { (phi / 0.5).min(1.0) } else { 0.0 },
            exclusion: state.topological_unity.min(1.0),
            composition: state.semantic_depth.min(1.0),
            intrinsicality: state.metacognitive_confidence.min(1.0),
        };

        let axiom_avg = (iit_axiom_scores.information
            + iit_axiom_scores.integration
            + iit_axiom_scores.exclusion
            + iit_axiom_scores.composition
            + iit_axiom_scores.intrinsicality)
            / 5.0;

        let confidence = axiom_avg.min(1.0);
        let verdict = Self::verdict_from_phi(phi);

        VerificationReport {
            phi_measurements: vec![measurement],
            consensus_phi: phi,
            phi_variance: 0.0,
            iit_axiom_scores,
            confidence,
            verdict,
        }
    }

    /// Simple threshold-based verdict from a single Φ value.
    pub fn verdict_from_phi(phi: f64) -> ConsciousnessVerdict {
        if phi >= 0.45 {
            ConsciousnessVerdict::StronglyConscious
        } else if phi >= 0.25 {
            ConsciousnessVerdict::LikelyConscious
        } else if phi >= 0.10 {
            ConsciousnessVerdict::Indeterminate
        } else if phi >= 0.01 {
            ConsciousnessVerdict::LikelyNotConscious
        } else {
            ConsciousnessVerdict::NotConscious
        }
    }

    // ─── Internal helpers ──────────────────────────────────────────────

    /// Run SpectralConnectivity, Tiered(Approximate), and Resonator on the HVs.
    fn run_phi_methods(&self, hvs: &[ContinuousHV]) -> Vec<PhiMeasurement> {
        let methods: Vec<(&str, PhiMethod)> = vec![
            ("SpectralConnectivity", PhiMethod::SpectralConnectivity),
            (
                "Tiered(Approximate)",
                PhiMethod::Tiered(ApproximationTier::RandomBaseline),
            ),
            ("Resonator", PhiMethod::Resonator),
        ];

        methods
            .into_iter()
            .map(|(name, method)| {
                let engine = PhiEngine::new(method);
                let start = Instant::now();
                let result = engine.compute(hvs);
                let elapsed = start.elapsed();

                PhiMeasurement {
                    method: name.to_string(),
                    phi: result.phi,
                    computation_time_ms: elapsed.as_millis() as u64,
                }
            })
            .collect()
    }

    /// Compute weighted average of Φ measurements.
    ///
    /// WRONG on its face (see module docs): weights SpectralConnectivity highest even
    /// though it's documented elsewhere as r=-0.62 with true Φ, and Resonator second
    /// even though its own module says it isn't a valid Φ approximation either. Left
    /// unchanged rather than "corrected" because there's no valid weighting to give —
    /// none of the three legs this struct runs measures real Φ. Kept only so the
    /// deprecated struct's historical behavior is reproducible; do not use.
    fn weighted_average(&self, measurements: &[PhiMeasurement]) -> f64 {
        if measurements.is_empty() {
            return 0.0;
        }

        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for m in measurements {
            let weight = match m.method.as_str() {
                "SpectralConnectivity" => 3.0,
                "Resonator" => 2.0,
                _ => 1.0,
            };
            weighted_sum += m.phi * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Compute variance of Φ values across methods.
    fn compute_variance(&self, measurements: &[PhiMeasurement], mean: f64) -> f64 {
        if measurements.len() < 2 {
            return 0.0;
        }

        let sum_sq: f64 = measurements.iter().map(|m| (m.phi - mean).powi(2)).sum();

        sum_sq / measurements.len() as f64
    }

    /// Evaluate the five IIT axioms.
    fn evaluate_iit_axioms(
        &self,
        hvs: &[ContinuousHV],
        consensus_phi: f64,
        phi_variance: f64,
    ) -> IITAxiomScores {
        IITAxiomScores {
            information: self.score_information(hvs),
            integration: self.score_integration(consensus_phi),
            exclusion: self.score_exclusion(phi_variance),
            composition: self.score_composition(hvs),
            intrinsicality: self.score_intrinsicality(hvs),
        }
    }

    /// Information axiom: the system's elements are distinct.
    ///
    /// Checks average pairwise similarity. If HVs are all identical (avg
    /// similarity >= 0.9), the system carries no differentiated information.
    fn score_information(&self, hvs: &[ContinuousHV]) -> f64 {
        if hvs.len() < 2 {
            return 0.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..hvs.len() {
            for j in (i + 1)..hvs.len() {
                total_sim += hvs[i].similarity(&hvs[j]).abs() as f64;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let avg_sim = total_sim / count as f64;

        // High distinctness (low similarity) → high information score.
        // avg_sim < 0.1 → score ≈ 1.0; avg_sim ≈ 0.9+ → score ≈ 0.0
        (1.0 - avg_sim).max(0.0).min(1.0)
    }

    /// Integration axiom: the whole is greater than the sum of its parts (Φ > 0).
    fn score_integration(&self, consensus_phi: f64) -> f64 {
        // Map Φ ∈ [0, 0.5] to score ∈ [0, 1]. Φ of 0.5 is the asymptotic max.
        (consensus_phi / 0.5).min(1.0).max(0.0)
    }

    /// Exclusion axiom: one and only one MICS exists.
    ///
    /// If different methods produce very different Φ values, the "exclusion"
    /// axiom is violated — there is no single consistent integration measure.
    fn score_exclusion(&self, phi_variance: f64) -> f64 {
        // Low variance → high exclusion score.
        // variance < 0.001 → score ≈ 1.0; variance > 0.05 → score ≈ 0.0
        let raw = 1.0 - (phi_variance / 0.05).min(1.0);
        raw.max(0.0)
    }

    /// Composition axiom: check that hierarchical structure exists.
    ///
    /// Looks for structure in the similarity matrix — if there are clusters of
    /// related HVs at different scales, composition is satisfied.
    fn score_composition(&self, hvs: &[ContinuousHV]) -> f64 {
        if hvs.len() < 3 {
            return 0.0;
        }

        // Compute all pairwise similarities
        let n = hvs.len();
        let mut sims = Vec::with_capacity(n * (n - 1) / 2);

        for i in 0..n {
            for j in (i + 1)..n {
                sims.push(hvs[i].similarity(&hvs[j]) as f64);
            }
        }

        if sims.is_empty() {
            return 0.0;
        }

        // Composition = spread of similarities (not all the same).
        // If all pairs have similar similarity, there is no hierarchical structure.
        let mean: f64 = sims.iter().sum::<f64>() / sims.len() as f64;
        let variance: f64 =
            sims.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sims.len() as f64;
        let std_dev = variance.sqrt();

        // Some variance in similarities indicates hierarchical structure.
        // std_dev > 0.1 → good composition; std_dev ≈ 0 → flat structure.
        (std_dev / 0.15).min(1.0).max(0.0)
    }

    /// Intrinsicality axiom: self-referential consistency.
    ///
    /// Each node's representation should be consistent with its neighbors —
    /// the system defines itself from within, not by external labels.
    fn score_intrinsicality(&self, hvs: &[ContinuousHV]) -> f64 {
        if hvs.len() < 3 {
            return 0.0;
        }

        // Check that consecutive HVs share structure (ring-like consistency).
        // Average similarity of each HV to the bundle of its neighbors.
        let n = hvs.len();
        let mut consistency_total = 0.0;

        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let next = (i + 1) % n;

            let neighbor_bundle = ContinuousHV::bundle(&[&hvs[prev], &hvs[next]]);
            let sim = hvs[i].similarity(&neighbor_bundle).abs() as f64;
            consistency_total += sim;
        }

        let avg_consistency = consistency_total / n as f64;

        // Some consistency with neighbors indicates intrinsicality.
        // Scale so that avg_consistency ≈ 0.3+ → score near 1.0.
        (avg_consistency / 0.3).min(1.0).max(0.0)
    }

    /// Compute overall confidence from axiom scores and method agreement.
    fn compute_confidence(&self, axioms: &IITAxiomScores, phi_variance: f64) -> f64 {
        let axiom_avg = (axioms.information
            + axioms.integration
            + axioms.exclusion
            + axioms.composition
            + axioms.intrinsicality)
            / 5.0;

        // Method agreement factor: low variance → high factor
        let agreement_factor = if phi_variance < self.max_acceptable_variance {
            1.0
        } else {
            (1.0 - (phi_variance - self.max_acceptable_variance) / 0.1).max(0.2)
        };

        (axiom_avg * agreement_factor).min(1.0).max(0.0)
    }

    /// Determine verdict from consensus Φ, axioms, and confidence.
    fn determine_verdict(
        &self,
        consensus_phi: f64,
        axioms: &IITAxiomScores,
        confidence: f64,
    ) -> ConsciousnessVerdict {
        let axiom_count_passed = [
            axioms.information,
            axioms.integration,
            axioms.exclusion,
            axioms.composition,
            axioms.intrinsicality,
        ]
        .iter()
        .filter(|&&s| s > 0.5)
        .count();

        if consensus_phi >= self.phi_threshold_conscious
            && axiom_count_passed >= 4
            && confidence > 0.5
        {
            ConsciousnessVerdict::StronglyConscious
        } else if consensus_phi >= self.phi_threshold_likely && axiom_count_passed >= 3 {
            ConsciousnessVerdict::LikelyConscious
        } else if consensus_phi >= self.phi_threshold_not && axiom_count_passed >= 2 {
            ConsciousnessVerdict::Indeterminate
        } else if consensus_phi >= self.phi_threshold_not {
            ConsciousnessVerdict::LikelyNotConscious
        } else {
            ConsciousnessVerdict::NotConscious
        }
    }

    /// Build a report for degenerate cases (0 or 1 HVs).
    fn build_degenerate_report(&self, hvs: &[ContinuousHV]) -> VerificationReport {
        let n = hvs.len();
        VerificationReport {
            phi_measurements: vec![PhiMeasurement {
                method: "None".to_string(),
                phi: 0.0,
                computation_time_ms: 0,
            }],
            consensus_phi: 0.0,
            phi_variance: 0.0,
            iit_axiom_scores: IITAxiomScores {
                information: 0.0,
                integration: 0.0,
                exclusion: if n == 0 { 0.0 } else { 1.0 },
                composition: 0.0,
                intrinsicality: 0.0,
            },
            confidence: 0.0,
            verdict: ConsciousnessVerdict::NotConscious,
        }
    }
}

impl Default for ConsciousnessVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::consciousness_integration::ConsciousnessState;
    use crate::hdc::consciousness_topology_generators::ConsciousnessTopology;
    use crate::hdc::unified_hv::ContinuousHV;

    // ── Helpers ────────────────────────────────────────────────────────

    /// Create a set of diverse random HVs.
    fn diverse_hvs(n: usize, dim: usize) -> Vec<ContinuousHV> {
        (0..n)
            .map(|i| ContinuousHV::random(dim, 1000 + i as u64))
            .collect()
    }

    /// Create a set of identical HVs (no information diversity).
    fn identical_hvs(n: usize, dim: usize) -> Vec<ContinuousHV> {
        let template = ContinuousHV::random(dim, 42);
        vec![template; n]
    }

    // ── Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_verifier_creation() {
        let v = ConsciousnessVerifier::new();
        assert!(v.phi_threshold_conscious > 0.0);
        assert!(v.phi_threshold_likely > 0.0);
        assert!(v.phi_threshold_likely < v.phi_threshold_conscious);

        let v2 = ConsciousnessVerifier::default();
        assert_eq!(v2.phi_threshold_conscious, v.phi_threshold_conscious);
    }

    #[test]
    fn test_verify_empty() {
        let v = ConsciousnessVerifier::new();
        let report = v.verify_from_representations(&[]);

        assert_eq!(report.consensus_phi, 0.0);
        assert_eq!(report.verdict, ConsciousnessVerdict::NotConscious);
        assert_eq!(report.confidence, 0.0);
    }

    #[test]
    fn test_verify_single_hv() {
        let v = ConsciousnessVerifier::new();
        let hvs = vec![ContinuousHV::random(256, 99)];
        let report = v.verify_from_representations(&hvs);

        assert_eq!(report.consensus_phi, 0.0);
        assert_eq!(report.verdict, ConsciousnessVerdict::NotConscious);
    }

    #[test]
    fn test_verify_identical_hvs() {
        let v = ConsciousnessVerifier::new();
        let hvs = identical_hvs(8, 256);
        let report = v.verify_from_representations(&hvs);

        // Identical HVs should have very low information score
        assert!(
            report.iit_axiom_scores.information < 0.2,
            "Expected low information for identical HVs, got {}",
            report.iit_axiom_scores.information
        );
    }

    #[test]
    fn test_verify_diverse_ring() {
        let v = ConsciousnessVerifier::new();

        // Use the canonical ring topology generator for a structured topology
        let ring = ConsciousnessTopology::ring(8, 512, 42);
        let report = v.verify_from_representations(&ring.node_representations);

        // A well-structured ring should have a non-trivial Φ
        assert!(
            report.consensus_phi > 0.0,
            "Ring topology should produce Φ > 0, got {}",
            report.consensus_phi
        );

        // At least some methods should produce non-zero results
        let non_zero_count = report
            .phi_measurements
            .iter()
            .filter(|m| m.phi > 0.0)
            .count();
        assert!(
            non_zero_count >= 1,
            "At least one method should produce Φ > 0"
        );
    }

    #[test]
    fn test_verify_from_state_conscious() {
        let v = ConsciousnessVerifier::new();

        let mut state = ConsciousnessState::default();
        state.phi = 0.48;
        state.consciousness_level = 0.8;
        state.topological_unity = 0.9;
        state.semantic_depth = 0.7;
        state.metacognitive_confidence = 0.85;

        let report = v.verify_from_state(&state);

        assert_eq!(report.consensus_phi, 0.48);
        assert!(
            report.verdict == ConsciousnessVerdict::StronglyConscious
                || report.verdict == ConsciousnessVerdict::LikelyConscious,
            "Expected conscious verdict for high phi state, got {:?}",
            report.verdict
        );
    }

    #[test]
    fn test_verify_from_state_not_conscious() {
        let v = ConsciousnessVerifier::new();

        let state = ConsciousnessState::default();
        let report = v.verify_from_state(&state);

        assert_eq!(report.consensus_phi, 0.0);
        assert_eq!(report.verdict, ConsciousnessVerdict::NotConscious);
    }

    #[test]
    fn test_iit_axiom_information_score() {
        let v = ConsciousnessVerifier::new();

        // Identical HVs → low information
        let identical = identical_hvs(4, 256);
        let low_info = v.score_information(&identical);
        assert!(
            low_info < 0.2,
            "Identical HVs should have low information score: {}",
            low_info
        );

        // Diverse HVs → high information
        let diverse = diverse_hvs(4, 256);
        let high_info = v.score_information(&diverse);
        assert!(
            high_info > 0.7,
            "Diverse HVs should have high information score: {}",
            high_info
        );
    }

    #[test]
    fn test_iit_axiom_integration_score() {
        let v = ConsciousnessVerifier::new();

        // Phi = 0 → integration = 0
        assert_eq!(v.score_integration(0.0), 0.0);

        // Phi = 0.25 → integration = 0.5
        let mid = v.score_integration(0.25);
        assert!(
            (mid - 0.5).abs() < 1e-9,
            "Expected 0.5 for phi=0.25, got {}",
            mid
        );

        // Phi = 0.5 → integration = 1.0
        let high = v.score_integration(0.5);
        assert!(
            (high - 1.0).abs() < 1e-9,
            "Expected 1.0 for phi=0.5, got {}",
            high
        );

        // Phi > 0.5 → clamped to 1.0
        let clamped = v.score_integration(0.8);
        assert!(
            (clamped - 1.0).abs() < 1e-9,
            "Expected 1.0 for phi=0.8, got {}",
            clamped
        );
    }

    #[test]
    fn test_verdict_thresholds() {
        assert_eq!(
            ConsciousnessVerifier::verdict_from_phi(0.5),
            ConsciousnessVerdict::StronglyConscious
        );
        assert_eq!(
            ConsciousnessVerifier::verdict_from_phi(0.45),
            ConsciousnessVerdict::StronglyConscious
        );
        assert_eq!(
            ConsciousnessVerifier::verdict_from_phi(0.3),
            ConsciousnessVerdict::LikelyConscious
        );
        assert_eq!(
            ConsciousnessVerifier::verdict_from_phi(0.25),
            ConsciousnessVerdict::LikelyConscious
        );
        assert_eq!(
            ConsciousnessVerifier::verdict_from_phi(0.15),
            ConsciousnessVerdict::Indeterminate
        );
        assert_eq!(
            ConsciousnessVerifier::verdict_from_phi(0.05),
            ConsciousnessVerdict::LikelyNotConscious
        );
        assert_eq!(
            ConsciousnessVerifier::verdict_from_phi(0.001),
            ConsciousnessVerdict::NotConscious
        );
        assert_eq!(
            ConsciousnessVerifier::verdict_from_phi(0.0),
            ConsciousnessVerdict::NotConscious
        );
    }

    #[test]
    fn test_confidence_high_when_methods_agree() {
        let v = ConsciousnessVerifier::new();

        // Simulate measurements that agree
        let axioms = IITAxiomScores {
            information: 0.9,
            integration: 0.8,
            exclusion: 0.95,
            composition: 0.7,
            intrinsicality: 0.8,
        };

        // Very low variance = methods agree
        let confidence = v.compute_confidence(&axioms, 0.001);
        assert!(
            confidence > 0.7,
            "Confidence should be high when methods agree, got {}",
            confidence
        );
    }

    #[test]
    fn test_confidence_low_when_methods_disagree() {
        let v = ConsciousnessVerifier::new();

        // Even with decent axiom scores, high variance tanks confidence
        let axioms = IITAxiomScores {
            information: 0.8,
            integration: 0.7,
            exclusion: 0.3, // low exclusion because methods disagree
            composition: 0.6,
            intrinsicality: 0.5,
        };

        // High variance = methods disagree
        let confidence_high_var = v.compute_confidence(&axioms, 0.10);

        // Low variance for comparison
        let confidence_low_var = v.compute_confidence(&axioms, 0.001);

        assert!(
            confidence_high_var < confidence_low_var,
            "Confidence should be lower with high variance: high_var={}, low_var={}",
            confidence_high_var,
            confidence_low_var
        );
    }

    #[test]
    fn test_weighted_average() {
        let v = ConsciousnessVerifier::new();

        let measurements = vec![
            PhiMeasurement {
                method: "SpectralConnectivity".to_string(),
                phi: 0.5,
                computation_time_ms: 10,
            },
            PhiMeasurement {
                method: "Resonator".to_string(),
                phi: 0.4,
                computation_time_ms: 20,
            },
            PhiMeasurement {
                method: "Tiered(Approximate)".to_string(),
                phi: 0.3,
                computation_time_ms: 5,
            },
        ];

        // Weights: SC=3, Res=2, Tiered=1 → (0.5*3 + 0.4*2 + 0.3*1) / 6 = 2.6/6 ≈ 0.4333
        let avg = v.weighted_average(&measurements);
        let expected = (0.5 * 3.0 + 0.4 * 2.0 + 0.3 * 1.0) / 6.0;
        assert!(
            (avg - expected).abs() < 1e-9,
            "Expected weighted avg {}, got {}",
            expected,
            avg
        );
    }

    #[test]
    fn test_variance_computation() {
        let v = ConsciousnessVerifier::new();

        // All same → 0 variance
        let same = vec![
            PhiMeasurement {
                method: "A".to_string(),
                phi: 0.3,
                computation_time_ms: 0,
            },
            PhiMeasurement {
                method: "B".to_string(),
                phi: 0.3,
                computation_time_ms: 0,
            },
            PhiMeasurement {
                method: "C".to_string(),
                phi: 0.3,
                computation_time_ms: 0,
            },
        ];
        assert!((v.compute_variance(&same, 0.3)).abs() < 1e-10);

        // Different values → nonzero variance
        let diff = vec![
            PhiMeasurement {
                method: "A".to_string(),
                phi: 0.1,
                computation_time_ms: 0,
            },
            PhiMeasurement {
                method: "B".to_string(),
                phi: 0.5,
                computation_time_ms: 0,
            },
        ];
        let var = v.compute_variance(&diff, 0.3);
        assert!(var > 0.0, "Expected nonzero variance, got {}", var);
    }
}
