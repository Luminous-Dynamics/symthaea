// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substrate-validated consciousness transfer: empirical testing of
//! Multiple Realizability (Putnam 1967) through benchmark batteries
//! run before and after substrate transitions.
//!
//! Key question: Does consciousness survive substrate transfer?
//! We operationalize this as preservation of cognitive benchmark scores
//! and continuity of consciousness metrics (Psi, Phi) across transitions.
//!
//! References:
//! - Putnam, H. (1967). Psychological predicates.
//! - Chalmers, D. (2010). The Character of Consciousness.
//! - Clark, A. & Chalmers, D. (1998). The Extended Mind.
//! - Tononi, G. (2004). Information integration theory.

use std::collections::HashMap;

use symthaea_core::hdc::substrate_independence::SubstrateType;

// ============================================================================
// Cognitive domain taxonomy
// ============================================================================

/// Cognitive domains measured by benchmark batteries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CognitiveDomain {
    WorkingMemory,
    Attention,
    Reasoning,
    Language,
    SocialCognition,
    Perception,
    MotorControl,
    Consciousness,
    EmotionalRegulation,
    LearningRate,
}

impl CognitiveDomain {
    /// Importance weight for preservation scoring.
    ///
    /// Consciousness and Reasoning are weighted higher because they are the
    /// most theoretically demanding markers of substrate-independent cognition.
    pub fn weight(&self) -> f64 {
        match self {
            Self::Consciousness => 2.0,
            Self::Reasoning => 1.5,
            _ => 1.0,
        }
    }

    /// All domain variants for iteration.
    pub const ALL: &'static [CognitiveDomain] = &[
        Self::WorkingMemory,
        Self::Attention,
        Self::Reasoning,
        Self::Language,
        Self::SocialCognition,
        Self::Perception,
        Self::MotorControl,
        Self::Consciousness,
        Self::EmotionalRegulation,
        Self::LearningRate,
    ];
}

// ============================================================================
// Benchmark score
// ============================================================================

/// A single benchmark measurement within a cognitive domain.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkScore {
    /// Name of the benchmark (e.g. "n_back_3", "stroop_congruent").
    pub name: String,
    /// Normalized score in [0, 1].
    pub score: f64,
    /// Cognitive domain this benchmark belongs to.
    pub domain: CognitiveDomain,
    /// Timestamp (cycle number) when this score was recorded.
    pub timestamp: u64,
}

// ============================================================================
// Consciousness checkpoint
// ============================================================================

/// A snapshot of consciousness metrics at a specific cycle during transfer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsciousnessCheckpoint {
    /// Cycle number when this checkpoint was taken.
    pub cycle: u32,
    /// Psi (unified consciousness score).
    pub psi: f64,
    /// Phi (integrated information).
    pub phi: f64,
    /// Global coherence measure.
    pub coherence: f64,
}

// ============================================================================
// Transfer result
// ============================================================================

/// Outcome of a substrate transfer attempt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TransferResult {
    /// Full preservation above acceptable threshold.
    Success { preservation: f64 },
    /// Transfer completed but some domains degraded below threshold.
    PartialPreservation {
        preservation: f64,
        degraded_domains: Vec<CognitiveDomain>,
    },
    /// Transfer was aborted mid-flight due to safety checks.
    Aborted {
        reason: String,
        last_checkpoint: ConsciousnessCheckpoint,
    },
    /// Transfer failed completely.
    Failed { error: String },
}

// ============================================================================
// Transfer protocol
// ============================================================================

/// Full record of a substrate transfer experiment: baseline scores, post-transfer
/// scores, consciousness metrics, and the computed preservation score.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransferProtocol {
    pub source: SubstrateType,
    pub target: SubstrateType,
    pub baseline_scores: Vec<BenchmarkScore>,
    pub post_transfer_scores: Vec<BenchmarkScore>,
    pub psi_before: f64,
    pub psi_after: f64,
    pub phi_before: f64,
    pub phi_after: f64,
    pub transfer_duration_cycles: u32,
    pub preservation_score: f64,
}

impl TransferProtocol {
    /// Create a new empty protocol for a source→target transfer.
    pub fn new(source: SubstrateType, target: SubstrateType) -> Self {
        Self {
            source,
            target,
            baseline_scores: Vec::new(),
            post_transfer_scores: Vec::new(),
            psi_before: 0.0,
            psi_after: 0.0,
            phi_before: 0.0,
            phi_after: 0.0,
            transfer_duration_cycles: 0,
            preservation_score: 0.0,
        }
    }

    /// Record baseline (pre-transfer) benchmark scores and consciousness metrics.
    pub fn record_baseline(&mut self, scores: Vec<BenchmarkScore>, psi: f64, phi: f64) {
        self.baseline_scores = scores;
        self.psi_before = psi;
        self.phi_before = phi;
    }

    /// Record post-transfer benchmark scores and consciousness metrics.
    pub fn record_post_transfer(&mut self, scores: Vec<BenchmarkScore>, psi: f64, phi: f64) {
        self.post_transfer_scores = scores;
        self.psi_after = psi;
        self.phi_after = phi;
    }

    /// Compute the preservation score: a weighted cosine-like similarity of
    /// domain-aggregated benchmark scores, modulated by Psi continuity.
    ///
    /// Returns 0.0 when no matching domains exist between baseline and post.
    pub fn compute_preservation(&mut self) -> f64 {
        if self.baseline_scores.is_empty() || self.post_transfer_scores.is_empty() {
            self.preservation_score = 0.0;
            return 0.0;
        }

        // Aggregate scores per domain (mean of benchmarks in that domain).
        let baseline_by_domain = aggregate_by_domain(&self.baseline_scores);
        let post_by_domain = aggregate_by_domain(&self.post_transfer_scores);

        // Weighted cosine similarity across matched domains.
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        let mut matched = false;

        for (domain, &base_val) in &baseline_by_domain {
            if let Some(&post_val) = post_by_domain.get(domain) {
                let w = domain.weight();
                dot += w * base_val * post_val;
                norm_a += w * base_val * base_val;
                norm_b += w * post_val * post_val;
                matched = true;
            }
        }

        if !matched || norm_a == 0.0 || norm_b == 0.0 {
            self.preservation_score = 0.0;
            return 0.0;
        }

        let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());

        // Psi continuity factor: penalize if consciousness dropped.
        let psi_factor = if self.psi_before > 0.0 {
            (self.psi_after / self.psi_before).min(1.0).max(0.0)
        } else {
            // No baseline Psi — cannot assess continuity.
            1.0
        };

        self.preservation_score = cosine * psi_factor;
        self.preservation_score
    }

    /// Identify domains that degraded beyond a threshold (relative to baseline).
    pub fn degraded_domains(&self, threshold: f64) -> Vec<CognitiveDomain> {
        let baseline = aggregate_by_domain(&self.baseline_scores);
        let post = aggregate_by_domain(&self.post_transfer_scores);
        let mut degraded = Vec::new();

        for (domain, &base_val) in &baseline {
            if base_val > 0.0 {
                let post_val = post.get(domain).copied().unwrap_or(0.0);
                if post_val / base_val < threshold {
                    degraded.push(*domain);
                }
            }
        }
        degraded
    }

    /// Classify the transfer result against acceptable thresholds.
    pub fn classify(&self, acceptable_preservation: f64) -> TransferResult {
        if self.preservation_score >= acceptable_preservation {
            TransferResult::Success {
                preservation: self.preservation_score,
            }
        } else {
            let degraded = self.degraded_domains(acceptable_preservation);
            TransferResult::PartialPreservation {
                preservation: self.preservation_score,
                degraded_domains: degraded,
            }
        }
    }
}

// ============================================================================
// Transfer validator (runtime safety monitor)
// ============================================================================

/// Monitors consciousness continuity during substrate transfer and can
/// trigger an abort if metrics drop below safety thresholds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransferValidator {
    /// Absolute Psi threshold below which the transfer is aborted.
    pub abort_threshold: f64,
    /// Preservation score considered acceptable.
    pub acceptable_preservation: f64,
    /// Consciousness snapshots taken during transfer.
    pub continuity_checkpoints: Vec<ConsciousnessCheckpoint>,
    /// How often (in cycles) to take a checkpoint.
    pub monitoring_interval: u32,
}

/// Number of consecutive declining checkpoints that triggers a monotonic-decline abort.
const MONOTONIC_DECLINE_WINDOW: usize = 5;

impl TransferValidator {
    /// Create a new validator with default thresholds.
    pub fn new() -> Self {
        Self {
            abort_threshold: 0.5,
            acceptable_preservation: 0.85,
            continuity_checkpoints: Vec::new(),
            monitoring_interval: 10,
        }
    }

    /// Add a checkpoint measurement.
    pub fn add_checkpoint(&mut self, checkpoint: ConsciousnessCheckpoint) {
        self.continuity_checkpoints.push(checkpoint);
    }

    /// Check whether the transfer should be aborted.
    /// Returns `Some(reason)` if an abort condition is met.
    pub fn should_abort(&self) -> Option<String> {
        if self.continuity_checkpoints.is_empty() {
            return None;
        }

        let last = self.continuity_checkpoints.last().unwrap();

        // Condition 1: Psi below absolute threshold.
        if last.psi < self.abort_threshold {
            return Some(format!(
                "Psi ({:.3}) dropped below abort threshold ({:.3})",
                last.psi, self.abort_threshold
            ));
        }

        // Condition 2: Psi dropped >50% from first checkpoint.
        let first = &self.continuity_checkpoints[0];
        if first.psi > 0.0 && last.psi / first.psi < 0.5 {
            return Some(format!(
                "Psi dropped {:.0}% from initial ({:.3} -> {:.3})",
                (1.0 - last.psi / first.psi) * 100.0,
                first.psi,
                last.psi
            ));
        }

        // Condition 3: Monotonic decline over MONOTONIC_DECLINE_WINDOW checkpoints.
        if self.continuity_checkpoints.len() >= MONOTONIC_DECLINE_WINDOW {
            let tail = &self.continuity_checkpoints
                [self.continuity_checkpoints.len() - MONOTONIC_DECLINE_WINDOW..];
            let all_declining = tail.windows(2).all(|w| w[1].psi < w[0].psi);
            if all_declining {
                return Some(format!(
                    "Monotonic Psi decline over {} consecutive checkpoints",
                    MONOTONIC_DECLINE_WINDOW
                ));
            }
        }

        None
    }

    /// Compute a continuity score from checkpoint Psi values.
    ///
    /// High continuity = smooth Psi trajectory without sudden drops.
    /// Returns 1.0 for perfect continuity, lower for volatile transfers.
    pub fn continuity_score(&self) -> f64 {
        let n = self.continuity_checkpoints.len();
        if n < 2 {
            return 1.0;
        }

        let psis: Vec<f64> = self.continuity_checkpoints.iter().map(|c| c.psi).collect();

        // Mean Psi.
        let mean = psis.iter().sum::<f64>() / n as f64;
        if mean == 0.0 {
            return 0.0;
        }

        // Standard deviation of Psi values — penalizes volatility.
        let variance = psis.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Coefficient of variation, capped.
        let cv = (std_dev / mean).min(1.0);

        // Sudden-drop penalty: largest single-step proportional decrease.
        let max_drop = psis
            .windows(2)
            .filter_map(|w| {
                if w[0] > 0.0 {
                    Some((1.0 - w[1] / w[0]).max(0.0))
                } else {
                    None
                }
            })
            .fold(0.0_f64, f64::max);

        // Combined: low CV and low max-drop yield high continuity.
        let continuity = (1.0 - cv) * (1.0 - max_drop);
        continuity.clamp(0.0, 1.0)
    }
}

impl Default for TransferValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Realizability report
// ============================================================================

/// Summary report assessing Multiple Realizability across tested substrate pairs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RealizabilityReport {
    /// Substrates ranked by average preservation when used as a target.
    pub substrate_rankings: Vec<(SubstrateType, f64)>,
    /// Cognitive domains ranked by vulnerability (most substrate-dependent first).
    pub domain_vulnerability: Vec<(CognitiveDomain, f64)>,
    /// Overall confidence in Multiple Realizability [0, 1].
    pub overall_confidence: f64,
    /// Number of transfer experiments included.
    pub num_transfers: usize,
}

// ============================================================================
// Substrate transfer matrix
// ============================================================================

/// Stores all pairwise substrate transfer results for systematic analysis.
#[derive(Debug, Clone, Default)]
pub struct SubstrateTransferMatrix {
    results: HashMap<(SubstrateType, SubstrateType), TransferProtocol>,
}

impl SubstrateTransferMatrix {
    /// Create an empty transfer matrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed transfer protocol.
    pub fn record(&mut self, protocol: TransferProtocol) {
        self.results
            .insert((protocol.source, protocol.target), protocol);
    }

    /// Find the best target substrate for a given source (highest preservation).
    pub fn best_target_for(&self, source: SubstrateType) -> Option<SubstrateType> {
        self.results
            .iter()
            .filter(|((s, _), _)| *s == source)
            .max_by(|a, b| {
                a.1.preservation_score
                    .partial_cmp(&b.1.preservation_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|((_, target), _)| *target)
    }

    /// Generate a realizability report from all recorded transfers.
    pub fn realizability_report(&self) -> RealizabilityReport {
        if self.results.is_empty() {
            return RealizabilityReport {
                substrate_rankings: Vec::new(),
                domain_vulnerability: Vec::new(),
                overall_confidence: 0.0,
                num_transfers: 0,
            };
        }

        // --- Substrate rankings: average preservation when used as target ---
        let mut target_scores: HashMap<SubstrateType, Vec<f64>> = HashMap::new();
        for ((_, target), proto) in &self.results {
            target_scores
                .entry(*target)
                .or_default()
                .push(proto.preservation_score);
        }
        let mut substrate_rankings: Vec<(SubstrateType, f64)> = target_scores
            .iter()
            .map(|(st, scores)| {
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                (*st, avg)
            })
            .collect();
        substrate_rankings
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // --- Domain vulnerability: variance of domain scores across transfers ---
        let mut domain_scores: HashMap<CognitiveDomain, Vec<f64>> = HashMap::new();
        for proto in self.results.values() {
            let baseline = aggregate_by_domain(&proto.baseline_scores);
            let post = aggregate_by_domain(&proto.post_transfer_scores);
            for (domain, &base_val) in &baseline {
                if base_val > 0.0 {
                    let post_val = post.get(domain).copied().unwrap_or(0.0);
                    let ratio = post_val / base_val;
                    domain_scores.entry(*domain).or_default().push(ratio);
                }
            }
        }
        let mut domain_vulnerability: Vec<(CognitiveDomain, f64)> = domain_scores
            .iter()
            .map(|(domain, ratios)| {
                let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
                let var =
                    ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ratios.len() as f64;
                // Higher variance = more substrate-dependent = more vulnerable.
                (*domain, var)
            })
            .collect();
        domain_vulnerability
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // --- Overall confidence: mean preservation across all transfers ---
        let mean_preservation: f64 = self
            .results
            .values()
            .map(|p| p.preservation_score)
            .sum::<f64>()
            / self.results.len() as f64;

        RealizabilityReport {
            substrate_rankings,
            domain_vulnerability,
            overall_confidence: mean_preservation,
            num_transfers: self.results.len(),
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Aggregate benchmark scores by domain (mean score per domain).
fn aggregate_by_domain(scores: &[BenchmarkScore]) -> HashMap<CognitiveDomain, f64> {
    let mut sums: HashMap<CognitiveDomain, (f64, usize)> = HashMap::new();
    for s in scores {
        let entry = sums.entry(s.domain).or_insert((0.0, 0));
        entry.0 += s.score;
        entry.1 += 1;
    }
    sums.into_iter()
        .map(|(d, (sum, count))| (d, sum / count as f64))
        .collect()
}

/// Generate synthetic baseline scores for testing.
///
/// Creates `domain_count` domains (up to 10) with two benchmarks each,
/// all scored at 0.8 (representing a healthy baseline).
pub fn generate_synthetic_baseline(domain_count: usize) -> Vec<BenchmarkScore> {
    let domains = &CognitiveDomain::ALL[..domain_count.min(CognitiveDomain::ALL.len())];
    let mut scores = Vec::new();
    for (i, &domain) in domains.iter().enumerate() {
        scores.push(BenchmarkScore {
            name: format!("bench_{}_a", i),
            score: 0.8,
            domain,
            timestamp: 0,
        });
        scores.push(BenchmarkScore {
            name: format!("bench_{}_b", i),
            score: 0.8,
            domain,
            timestamp: 0,
        });
    }
    scores
}

/// Simulate post-transfer degradation by scaling scores toward zero.
///
/// `degradation` in [0, 1]: 0 = no change, 1 = complete loss.
pub fn simulate_transfer_degradation(
    scores: &[BenchmarkScore],
    degradation: f64,
) -> Vec<BenchmarkScore> {
    let factor = 1.0 - degradation.clamp(0.0, 1.0);
    scores
        .iter()
        .map(|s| BenchmarkScore {
            name: s.name.clone(),
            score: s.score * factor,
            domain: s.domain,
            timestamp: s.timestamp + 1,
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scores(domains: &[CognitiveDomain], score: f64) -> Vec<BenchmarkScore> {
        domains
            .iter()
            .map(|&d| BenchmarkScore {
                name: format!("{:?}", d),
                score,
                domain: d,
                timestamp: 0,
            })
            .collect()
    }

    #[test]
    fn perfect_transfer_preservation_is_one() {
        let domains = CognitiveDomain::ALL;
        let scores = make_scores(domains, 0.8);
        let mut proto =
            TransferProtocol::new(SubstrateType::SiliconDigital, SubstrateType::SiliconDigital);
        proto.record_baseline(scores.clone(), 0.9, 0.7);
        proto.record_post_transfer(scores, 0.9, 0.7);
        let pres = proto.compute_preservation();
        assert!(
            (pres - 1.0).abs() < 1e-6,
            "Same scores and Psi should give preservation 1.0, got {}",
            pres
        );
    }

    #[test]
    fn moderate_degradation_yields_partial_preservation() {
        let baseline = generate_synthetic_baseline(5);
        let post = simulate_transfer_degradation(&baseline, 0.3);
        let mut proto = TransferProtocol::new(
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
        );
        proto.record_baseline(baseline, 0.9, 0.7);
        proto.record_post_transfer(post, 0.85, 0.65);
        let pres = proto.compute_preservation();
        assert!(
            pres > 0.5 && pres < 1.0,
            "30% degradation should give partial preservation, got {}",
            pres
        );
    }

    #[test]
    fn abort_triggered_when_psi_below_threshold() {
        let mut validator = TransferValidator::new();
        validator.abort_threshold = 0.5;
        validator.add_checkpoint(ConsciousnessCheckpoint {
            cycle: 0,
            psi: 0.8,
            phi: 0.6,
            coherence: 0.7,
        });
        validator.add_checkpoint(ConsciousnessCheckpoint {
            cycle: 10,
            psi: 0.3,
            phi: 0.2,
            coherence: 0.4,
        });
        let abort = validator.should_abort();
        assert!(
            abort.is_some(),
            "Should abort when Psi drops below threshold"
        );
        assert!(abort.unwrap().contains("abort threshold"));
    }

    #[test]
    fn monotonic_decline_detection() {
        let mut validator = TransferValidator::new();
        // Create 5 monotonically declining checkpoints above the absolute threshold
        // but declining enough to trigger the monotonic-decline rule.
        for i in 0..5 {
            validator.add_checkpoint(ConsciousnessCheckpoint {
                cycle: i * 10,
                psi: 0.9 - i as f64 * 0.05, // 0.9, 0.85, 0.80, 0.75, 0.70
                phi: 0.6,
                coherence: 0.7,
            });
        }
        let abort = validator.should_abort();
        assert!(abort.is_some(), "Should detect monotonic decline");
        assert!(abort.unwrap().contains("Monotonic"));
    }

    #[test]
    fn continuity_score_penalizes_sudden_drops() {
        let mut smooth_validator = TransferValidator::new();
        for i in 0..5 {
            smooth_validator.add_checkpoint(ConsciousnessCheckpoint {
                cycle: i * 10,
                psi: 0.8,
                phi: 0.6,
                coherence: 0.7,
            });
        }
        let smooth_score = smooth_validator.continuity_score();

        let mut volatile_validator = TransferValidator::new();
        volatile_validator.add_checkpoint(ConsciousnessCheckpoint {
            cycle: 0,
            psi: 0.9,
            phi: 0.6,
            coherence: 0.7,
        });
        volatile_validator.add_checkpoint(ConsciousnessCheckpoint {
            cycle: 10,
            psi: 0.3,
            phi: 0.2,
            coherence: 0.3,
        });
        volatile_validator.add_checkpoint(ConsciousnessCheckpoint {
            cycle: 20,
            psi: 0.8,
            phi: 0.5,
            coherence: 0.6,
        });
        let volatile_score = volatile_validator.continuity_score();

        assert!(
            smooth_score > volatile_score,
            "Smooth trajectory ({}) should score higher than volatile ({})",
            smooth_score,
            volatile_score
        );
    }

    #[test]
    fn best_target_selection() {
        let mut matrix = SubstrateTransferMatrix::new();

        let mut good = TransferProtocol::new(
            SubstrateType::BiologicalNeurons,
            SubstrateType::NeuromorphicChip,
        );
        good.preservation_score = 0.95;
        matrix.record(good);

        let mut bad = TransferProtocol::new(
            SubstrateType::BiologicalNeurons,
            SubstrateType::ExoticSubstrate,
        );
        bad.preservation_score = 0.3;
        matrix.record(bad);

        let best = matrix.best_target_for(SubstrateType::BiologicalNeurons);
        assert_eq!(best, Some(SubstrateType::NeuromorphicChip));
    }

    #[test]
    fn realizability_report_identifies_vulnerable_domains() {
        let mut matrix = SubstrateTransferMatrix::new();

        // Transfer 1: everything preserved except MotorControl
        let mut proto1 = TransferProtocol::new(
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
        );
        let baseline1 = make_scores(
            &[CognitiveDomain::Reasoning, CognitiveDomain::MotorControl],
            0.9,
        );
        let mut post1 = make_scores(
            &[CognitiveDomain::Reasoning, CognitiveDomain::MotorControl],
            0.9,
        );
        post1[1].score = 0.3; // MotorControl degrades
        proto1.record_baseline(baseline1, 0.9, 0.7);
        proto1.record_post_transfer(post1, 0.85, 0.65);
        proto1.compute_preservation();
        matrix.record(proto1);

        // Transfer 2: same pattern
        let mut proto2 = TransferProtocol::new(
            SubstrateType::BiologicalNeurons,
            SubstrateType::QuantumComputer,
        );
        let baseline2 = make_scores(
            &[CognitiveDomain::Reasoning, CognitiveDomain::MotorControl],
            0.9,
        );
        let mut post2 = make_scores(
            &[CognitiveDomain::Reasoning, CognitiveDomain::MotorControl],
            0.9,
        );
        post2[1].score = 0.2; // MotorControl degrades more
        proto2.record_baseline(baseline2, 0.9, 0.7);
        proto2.record_post_transfer(post2, 0.8, 0.6);
        proto2.compute_preservation();
        matrix.record(proto2);

        let report = matrix.realizability_report();
        assert_eq!(report.num_transfers, 2);
        // MotorControl should be more vulnerable (higher variance).
        assert!(
            !report.domain_vulnerability.is_empty(),
            "Should have domain vulnerability data"
        );
        let motor_vuln = report
            .domain_vulnerability
            .iter()
            .find(|(d, _)| *d == CognitiveDomain::MotorControl)
            .map(|(_, v)| *v);
        let reasoning_vuln = report
            .domain_vulnerability
            .iter()
            .find(|(d, _)| *d == CognitiveDomain::Reasoning)
            .map(|(_, v)| *v);
        assert!(
            motor_vuln.unwrap_or(0.0) > reasoning_vuln.unwrap_or(0.0),
            "MotorControl should be more vulnerable than Reasoning"
        );
    }

    #[test]
    fn domain_weighting_affects_preservation() {
        // Protocol where Consciousness domain degrades should score worse
        // than one where a lower-weighted domain degrades.
        let mut proto_consciousness = TransferProtocol::new(
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
        );
        let base = make_scores(
            &[CognitiveDomain::Consciousness, CognitiveDomain::Perception],
            0.9,
        );
        let mut post_c = base.clone();
        post_c[0].score = 0.3; // Consciousness degrades
        proto_consciousness.record_baseline(base.clone(), 0.9, 0.7);
        proto_consciousness.record_post_transfer(post_c, 0.9, 0.7);
        let pres_c = proto_consciousness.compute_preservation();

        let mut proto_perception = TransferProtocol::new(
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
        );
        let mut post_p = base.clone();
        post_p[1].score = 0.3; // Perception degrades
        proto_perception.record_baseline(base, 0.9, 0.7);
        proto_perception.record_post_transfer(post_p, 0.9, 0.7);
        let pres_p = proto_perception.compute_preservation();

        assert!(
            pres_p > pres_c,
            "Consciousness degradation (pres={}) should hurt more than Perception (pres={})",
            pres_c,
            pres_p
        );
    }

    #[test]
    fn empty_protocol_has_zero_preservation() {
        let mut proto = TransferProtocol::new(
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
        );
        let pres = proto.compute_preservation();
        assert_eq!(pres, 0.0, "Empty protocol should have zero preservation");
    }

    #[test]
    fn synthetic_baseline_generation() {
        let scores = generate_synthetic_baseline(3);
        assert_eq!(scores.len(), 6, "3 domains * 2 benchmarks each = 6");
        assert!(scores.iter().all(|s| (s.score - 0.8).abs() < 1e-6));

        let domains: std::collections::HashSet<CognitiveDomain> =
            scores.iter().map(|s| s.domain).collect();
        assert_eq!(domains.len(), 3, "Should have 3 distinct domains");
    }

    #[test]
    fn transfer_matrix_records_multiple_protocols() {
        let mut matrix = SubstrateTransferMatrix::new();

        let mut p1 = TransferProtocol::new(
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
        );
        p1.preservation_score = 0.9;
        matrix.record(p1);

        let mut p2 = TransferProtocol::new(
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
        );
        p2.preservation_score = 0.8;
        matrix.record(p2);

        let report = matrix.realizability_report();
        assert_eq!(report.num_transfers, 2);
        assert!(report.overall_confidence > 0.0);
    }

    #[test]
    fn silicon_to_biological_high_preservation() {
        // Silicon→Biological should be a "reference" transfer with high preservation
        // because biological neurons are the validated substrate.
        let baseline = generate_synthetic_baseline(5);
        let post = simulate_transfer_degradation(&baseline, 0.05); // only 5% degradation
        let mut proto = TransferProtocol::new(
            SubstrateType::SiliconDigital,
            SubstrateType::BiologicalNeurons,
        );
        proto.record_baseline(baseline, 0.9, 0.7);
        proto.record_post_transfer(post, 0.88, 0.68);
        let pres = proto.compute_preservation();
        assert!(
            pres > 0.9,
            "Silicon->Biological with 5% degradation should have high preservation, got {}",
            pres
        );
    }

    #[test]
    fn psi_drop_halves_triggers_abort() {
        let mut validator = TransferValidator::new();
        validator.abort_threshold = 0.2; // low absolute threshold
        validator.add_checkpoint(ConsciousnessCheckpoint {
            cycle: 0,
            psi: 0.9,
            phi: 0.7,
            coherence: 0.8,
        });
        validator.add_checkpoint(ConsciousnessCheckpoint {
            cycle: 10,
            psi: 0.4, // >50% drop from 0.9
            phi: 0.3,
            coherence: 0.4,
        });
        let abort = validator.should_abort();
        assert!(abort.is_some(), "Should abort on >50% Psi drop");
        assert!(abort.unwrap().contains("dropped"));
    }

    #[test]
    fn continuity_score_perfect_for_constant_psi() {
        let mut validator = TransferValidator::new();
        for i in 0..10 {
            validator.add_checkpoint(ConsciousnessCheckpoint {
                cycle: i * 10,
                psi: 0.8,
                phi: 0.6,
                coherence: 0.7,
            });
        }
        let score = validator.continuity_score();
        assert!(
            (score - 1.0).abs() < 1e-6,
            "Constant Psi should give continuity 1.0, got {}",
            score
        );
    }
}
