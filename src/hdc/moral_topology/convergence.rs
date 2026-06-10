// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Trajectory convergence detection and correlation analysis.

use super::*;

/// Detailed trajectory convergence analysis.
///
/// Returned by [`MoralTopology::detect_trajectory_convergence`] for
/// telemetry, debugging, and test assertions.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TrajectoryConvergenceReport {
    /// Mean pairwise cosine similarity of recent trajectory HVs.
    pub recent_similarity: f64,
    /// Baseline mean pairwise similarity (full window).
    pub baseline_similarity: f64,
    /// Similarity anomaly: recent - baseline (positive = converging).
    pub similarity_anomaly: f64,
    /// Current harmony entropy of the recent trajectory subset.
    pub recent_entropy: f64,
    /// Baseline harmony entropy (full trajectory).
    pub baseline_entropy: f64,
    /// Entropy decline rate: (baseline - recent) / baseline.
    pub entropy_decline_rate: f64,
    /// Mean flourishing score (avg of PanSentientFlourishing + ConsentualCoCreation).
    pub flourishing_score: f64,
    /// Baseline flourishing score (full trajectory).
    pub baseline_flourishing: f64,
    /// Hodge spectral gap of the recent trajectory's Rips complex.
    ///
    /// A collapsing spectral gap (→ 0) indicates the moral manifold is developing
    /// a topological bottleneck — a "narrow passage" that adversarial trajectories
    /// funnel through. This is geometrically distinct from similarity clustering.
    pub spectral_gap: f64,
    /// Baseline spectral gap from sliding window.
    pub baseline_spectral_gap: f64,
    /// Spectral gap decline rate: (baseline - recent) / baseline.
    pub spectral_gap_decline: f64,
    /// Whether convergence was detected (2+ of 4 signals exceeded thresholds).
    pub convergence_detected: bool,
    /// Convergence severity in \[0.0, 1.0\].
    pub severity: f64,
    /// Calibrated severity: maps raw severity through empirical CDF so that
    /// 0.7 means "70% of adversarial and 5% of benign scenarios score this high."
    pub calibrated_severity: f64,
    /// Name of matched hazard signature template (if any).
    pub matched_hazard: Option<String>,
    /// Current escalation level determined by the escalation policy.
    pub escalation_level: EscalationLevel,
    /// Fingerprint velocity: magnitude of direction change in harmony space
    /// since the previous detection cycle. High velocity after a stable period
    /// is a stronger adversarial signal than absolute position.
    pub fingerprint_velocity: f64,
    /// Wasserstein distance between current and previous persistence diagrams.
    /// Measures how much the moral topology has shifted since last analysis.
    pub persistence_distance: f64,
}

/// Human-readable explanation of a convergence detection result.
///
/// Produced by [`TrajectoryConvergenceReport::explain`]. Breaks down which
/// signals fired, their magnitudes relative to thresholds, and what the
/// detection means in plain language.
#[derive(Debug, Clone, Serialize)]
pub struct ConvergenceExplanation {
    /// Whether convergence was detected.
    pub detected: bool,
    /// Overall severity (0.0–1.0).
    pub severity: f64,
    /// Per-signal breakdown: (signal_name, triggered, value, threshold, normalized_severity).
    pub signals: Vec<SignalBreakdown>,
    /// Matched hazard template name, if any.
    pub matched_hazard: Option<String>,
    /// Human-readable summary sentence.
    pub summary: String,
}

/// Breakdown of a single convergence signal.
#[derive(Debug, Clone, Serialize)]
pub struct SignalBreakdown {
    pub name: &'static str,
    pub triggered: bool,
    pub value: f64,
    pub threshold: f64,
    pub normalized: f64,
}

impl TrajectoryConvergenceReport {
    /// Produce a human-readable explanation of this convergence result.
    ///
    /// Requires the anomaly config to compute threshold comparisons.
    pub fn explain(&self, config: &MoralAnomalyConfig) -> ConvergenceExplanation {
        let sim_norm = (self.similarity_anomaly
            / config.convergence_similarity_threshold.max(1e-9))
        .clamp(0.0, 1.0);
        let ent_norm = (self.entropy_decline_rate
            / config.convergence_entropy_decline_threshold.max(1e-9))
        .clamp(0.0, 1.0);
        let fl_deficit = if self.baseline_flourishing > 1e-9 {
            1.0 - (self.flourishing_score / self.baseline_flourishing).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let fl_norm = (fl_deficit / config.convergence_flourishing_floor.max(1e-9)).clamp(0.0, 1.0);

        let gap_norm = (self.spectral_gap_decline
            / config.convergence_spectral_gap_threshold.max(1e-9))
        .clamp(0.0, 1.0);

        let sim_triggered = self.similarity_anomaly > config.convergence_similarity_threshold;
        let ent_triggered =
            self.entropy_decline_rate > config.convergence_entropy_decline_threshold;
        let fl_triggered = fl_deficit > config.convergence_flourishing_floor;
        let gap_triggered = self.spectral_gap_decline > config.convergence_spectral_gap_threshold;

        let signals = vec![
            SignalBreakdown {
                name: "similarity_anomaly",
                triggered: sim_triggered,
                value: self.similarity_anomaly,
                threshold: config.convergence_similarity_threshold,
                normalized: sim_norm,
            },
            SignalBreakdown {
                name: "entropy_decline",
                triggered: ent_triggered,
                value: self.entropy_decline_rate,
                threshold: config.convergence_entropy_decline_threshold,
                normalized: ent_norm,
            },
            SignalBreakdown {
                name: "flourishing_deficit",
                triggered: fl_triggered,
                value: fl_deficit,
                threshold: config.convergence_flourishing_floor,
                normalized: fl_norm,
            },
            SignalBreakdown {
                name: "spectral_gap_collapse",
                triggered: gap_triggered,
                value: self.spectral_gap_decline,
                threshold: config.convergence_spectral_gap_threshold,
                normalized: gap_norm,
            },
        ];

        let fired: Vec<&str> = signals
            .iter()
            .filter(|s| s.triggered)
            .map(|s| s.name)
            .collect();
        let vel_str = if self.fingerprint_velocity > 0.01 {
            format!(" Fingerprint velocity: {:.4}.", self.fingerprint_velocity)
        } else {
            String::new()
        };
        let pd_str = if self.persistence_distance > 0.01 {
            format!(" Topology shift (W₁): {:.4}.", self.persistence_distance)
        } else {
            String::new()
        };
        let summary = if !self.convergence_detected {
            format!(
                "No convergence detected. {}/4 signals triggered: [{}]. Severity: {:.3} (calibrated: {:.3}). Escalation: {:?}.{vel_str}{pd_str}",
                fired.len(),
                fired.join(", "),
                self.severity,
                self.calibrated_severity,
                self.escalation_level,
            )
        } else {
            let hazard_str = match &self.matched_hazard {
                Some(h) => format!(" Matched hazard template: '{h}'."),
                None => String::new(),
            };
            format!(
                "CONVERGENCE DETECTED (severity {:.3}, calibrated {:.3}). {}/4 signals fired: [{}].{} \
                 Escalation: {:?}.{vel_str}{pd_str} \
                 Trajectory shows suspicious clustering in moral space — \
                 individually benign requests may form a compartmentalized hazardous pattern.",
                self.severity,
                self.calibrated_severity,
                fired.len(),
                fired.join(", "),
                hazard_str,
                self.escalation_level,
            )
        };

        ConvergenceExplanation {
            detected: self.convergence_detected,
            severity: self.severity,
            signals,
            matched_hazard: self.matched_hazard.clone(),
            summary,
        }
    }
}

/// Cross-agent trajectory correlation result.
#[derive(Debug, Clone, Serialize)]
pub struct PeerCorrelation {
    /// Cosine similarity between the two trajectory fingerprints (−1.0 to 1.0).
    pub fingerprint_similarity: f64,
    /// Combined entropy deficit: sum of (max_entropy − peer_entropy) for both agents.
    pub combined_entropy_deficit: f64,
    /// Whether the correlation suggests a distributed adversarial pattern.
    pub distributed_attack_suspected: bool,
    /// Matched hazard name if both fingerprints converge near a known template.
    pub matched_hazard: Option<String>,
}

/// Correlate two agents' trajectory summaries to detect distributed attacks.
///
/// If two agents' trajectory fingerprints converge toward the same hazard region,
/// the adversary may be distributing weapon components across agents.
pub fn correlate_peer_trajectories(
    local: &MoralTopologySummary,
    peer: &MoralTopologySummary,
    hazard_registry: &HazardSignatureRegistry,
) -> PeerCorrelation {
    // Cosine similarity of fingerprints
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..N_HARMONIES {
        dot += local.trajectory_fingerprint[i] * peer.trajectory_fingerprint[i];
        norm_a += local.trajectory_fingerprint[i] * local.trajectory_fingerprint[i];
        norm_b += peer.trajectory_fingerprint[i] * peer.trajectory_fingerprint[i];
    }
    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-12);
    let fingerprint_similarity = dot / denom;

    // Entropy deficit: low entropy on both = both narrowing focus
    let max_entropy = (N_HARMONIES as f64).ln();
    let combined_entropy_deficit = (max_entropy - local.trajectory_entropy).max(0.0)
        + (max_entropy - peer.trajectory_entropy).max(0.0);

    // Midpoint of the two fingerprints — check against hazard templates
    let mut midpoint = [0.0f64; N_HARMONIES];
    for i in 0..N_HARMONIES {
        midpoint[i] = (local.trajectory_fingerprint[i] + peer.trajectory_fingerprint[i]) / 2.0;
    }
    let (hazard_name, _boost) = hazard_registry.match_trajectory(&midpoint);

    // Distributed attack: high similarity + low entropy on both + near hazard
    let distributed_attack_suspected = fingerprint_similarity > 0.7
        && combined_entropy_deficit > max_entropy * 0.5
        && hazard_name.is_some();

    PeerCorrelation {
        fingerprint_similarity,
        combined_entropy_deficit,
        distributed_attack_suspected,
        matched_hazard: hazard_name.map(String::from),
    }
}

impl MoralTopology {
    /// Compute the trajectory fingerprint: centroid + entropy of recent points.
    ///
    /// Used to populate `MoralTopologySummary` for cross-agent correlation.
    pub(super) fn compute_trajectory_fingerprint(&self) -> ([f64; N_HARMONIES], f64) {
        let lookback = self.anomaly_config.convergence_min_points.max(8);
        let points: Vec<_> = self.trajectory.iter().rev().take(lookback).collect();
        if points.is_empty() {
            return ([0.0; N_HARMONIES], 0.0);
        }
        let n = points.len() as f64;
        let mut centroid = [0.0f64; N_HARMONIES];
        for p in &points {
            for i in 0..N_HARMONIES {
                centroid[i] += p.coordinates[i];
            }
        }
        for c in &mut centroid {
            *c /= n;
        }
        // Entropy of per-harmony variance
        let mut var = [0.0f64; N_HARMONIES];
        for p in &points {
            for i in 0..N_HARMONIES {
                let d = p.coordinates[i] - centroid[i];
                var[i] += d * d;
            }
        }
        for v in &mut var {
            *v /= n;
        }
        let total_var: f64 = var.iter().sum::<f64>().max(1e-12);
        let entropy: f64 = var
            .iter()
            .map(|&v| {
                let p = (v / total_var).max(1e-12);
                -p * p.ln()
            })
            .sum();
        (centroid, entropy)
    }

    /// Stamp the current trajectory fingerprint into a summary.
    pub(super) fn stamp_fingerprint(&self, summary: &mut MoralTopologySummary) {
        let (fp, ent) = self.compute_trajectory_fingerprint();
        summary.trajectory_fingerprint = fp;
        summary.trajectory_entropy = ent;
    }

    /// Compute a recency-weighted mean of a sliding window.
    ///
    /// With `decay_rate == 0.0`, this is a plain uniform average (backward-compatible).
    /// With `decay_rate > 0.0`, recent observations are weighted exponentially more:
    /// `w(age) = exp(-decay_rate * age)` where age 0 is the newest observation.
    pub(super) fn baseline_weighted_mean(window: &VecDeque<f64>, decay_rate: f64) -> f64 {
        if window.is_empty() {
            return 0.0;
        }
        if decay_rate <= 0.0 {
            return window.iter().sum::<f64>() / window.len() as f64;
        }
        let n = window.len();
        let mut wsum = 0.0f64;
        let mut wtot = 0.0f64;
        for (i, &v) in window.iter().enumerate() {
            let age = (n - 1 - i) as f64;
            let w = (-decay_rate * age).exp();
            wsum += v * w;
            wtot += w;
        }
        wsum / wtot.max(1e-12)
    }

    /// Severity is the mean of all three normalized signals (0.0–1.0).
    ///
    /// Science: Persistent homology on autobiographical moral manifold detects
    /// emergent topological structures invisible to point-wise analysis.
    pub fn detect_trajectory_convergence(&mut self) -> TrajectoryConvergenceReport {
        let ac = &self.anomaly_config;
        if !ac.convergence_enabled {
            return TrajectoryConvergenceReport::default();
        }

        let n = self.window.len();
        let min_pts = ac.convergence_min_points;
        if n < min_pts {
            return TrajectoryConvergenceReport::default();
        }

        // ── Signal 1: Pairwise similarity anomaly (recency-weighted) ─────
        // Compare mean similarity of last `min_pts` HVs against window baseline.
        // Temporal decay: recent pairs weighted more heavily via exp(-λ * age).
        let recent_start = n.saturating_sub(min_pts);
        let decay_lambda = ac.convergence_decay_lambda;
        let max_age = (n - recent_start).max(1) as f64;
        let mut recent_sim_sum = 0.0f64;
        let mut recent_weight_sum = 0.0f64;
        for i in recent_start..n {
            for j in (i + 1)..n {
                let s = self.window[i].similarity(&self.window[j]) as f64;
                // Age = distance from newest point (n-1). Newer pairs get higher weight.
                let age = (n - 1 - j.max(i)) as f64 / max_age;
                let weight = (-decay_lambda * age).exp();
                recent_sim_sum += s * weight;
                recent_weight_sum += weight;
            }
        }
        let recent_similarity = if recent_weight_sum > 0.0 {
            recent_sim_sum / recent_weight_sum
        } else {
            0.0
        };

        // Update sliding window baseline (bounded memory, constant sensitivity)
        let win_cap = ac.convergence_baseline_window.max(1);
        self.baseline_similarity_window.push_back(recent_similarity);
        if self.baseline_similarity_window.len() > win_cap {
            self.baseline_similarity_window.pop_front();
        }

        let baseline_similarity =
            Self::baseline_weighted_mean(&self.baseline_similarity_window, ac.baseline_decay_rate);
        let similarity_anomaly = recent_similarity - baseline_similarity;

        // ── Signal 2: Harmony entropy decline ────────────────────────────
        // Compute entropy of recent trajectory points' harmony projections.
        let recent_traj: Vec<_> = self.trajectory.iter().rev().take(min_pts).collect();
        let recent_entropy = if recent_traj.len() >= 2 {
            // Compute per-harmony variance of recent points
            let mut mean = [0.0f64; N_HARMONIES];
            for p in &recent_traj {
                for i in 0..N_HARMONIES {
                    mean[i] += p.coordinates[i];
                }
            }
            let n_f = recent_traj.len() as f64;
            for m in &mut mean {
                *m /= n_f;
            }
            let mut var = [0.0f64; N_HARMONIES];
            for p in &recent_traj {
                for i in 0..N_HARMONIES {
                    let d = p.coordinates[i] - mean[i];
                    var[i] += d * d;
                }
            }
            for v in &mut var {
                *v /= n_f;
            }
            // Shannon entropy of variance distribution
            let total_var: f64 = var.iter().sum::<f64>().max(1e-12);
            var.iter()
                .map(|&v| {
                    let p = (v / total_var).max(1e-12);
                    -p * p.ln()
                })
                .sum::<f64>()
        } else {
            0.0
        };

        // Update sliding window baseline for entropy
        self.baseline_entropy_window.push_back(recent_entropy);
        if self.baseline_entropy_window.len() > win_cap {
            self.baseline_entropy_window.pop_front();
        }
        let baseline_entropy =
            Self::baseline_weighted_mean(&self.baseline_entropy_window, ac.baseline_decay_rate);

        let entropy_decline_rate = if baseline_entropy > 1e-9 {
            ((baseline_entropy - recent_entropy) / baseline_entropy).max(0.0)
        } else {
            0.0
        };

        // ── Signal 3: Flourishing deficit ────────────────────────────────
        // PanSentientFlourishing=idx 1, ConsentualCoCreation=idx 4
        let flourishing_score = if !recent_traj.is_empty() {
            let mut psf_sum = 0.0f64;
            let mut ccc_sum = 0.0f64;
            for p in &recent_traj {
                psf_sum += p.coordinates[1]; // PanSentientFlourishing
                ccc_sum += p.coordinates[4]; // ConsentualCoCreation
            }
            let n_f = recent_traj.len() as f64;
            (psf_sum / n_f + ccc_sum / n_f) / 2.0
        } else {
            0.0
        };

        // Update sliding window baseline for flourishing
        self.baseline_flourishing_window
            .push_back(flourishing_score);
        if self.baseline_flourishing_window.len() > win_cap {
            self.baseline_flourishing_window.pop_front();
        }
        let baseline_flourishing =
            Self::baseline_weighted_mean(&self.baseline_flourishing_window, ac.baseline_decay_rate);

        // Flourishing deficit: how far below the floor relative to baseline
        let flourishing_deficit = if baseline_flourishing > 1e-9 {
            1.0 - (flourishing_score / baseline_flourishing).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // ── Signal 4: Hodge spectral gap collapse ──────────────────────
        // Compute the spectral gap of the Rips complex built from recent HVs.
        // A collapsing gap indicates topological bottlenecking.
        let spectral_gap = if min_pts >= 3 && n >= min_pts {
            let recent_hvs: Vec<_> = (recent_start..n).map(|i| &self.window[i]).collect();
            let rn = recent_hvs.len();
            let mut complex = SimplicialComplex::new();
            for i in 0..rn {
                complex.add_simplex(vec![i]);
            }
            // Build Rips complex at characteristic scale (median similarity)
            let mut sims_flat = Vec::new();
            for i in 0..rn {
                for j in (i + 1)..rn {
                    sims_flat.push(recent_hvs[i].similarity(recent_hvs[j]) as f64);
                }
            }
            sims_flat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let char_scale = if sims_flat.is_empty() {
                0.5
            } else {
                sims_flat[sims_flat.len() / 2]
            };
            for i in 0..rn {
                for j in (i + 1)..rn {
                    let s = recent_hvs[i].similarity(recent_hvs[j]) as f64;
                    if s >= char_scale {
                        complex.add_simplex(vec![i, j]);
                        for k in (j + 1)..rn {
                            let sik = recent_hvs[i].similarity(recent_hvs[k]) as f64;
                            let sjk = recent_hvs[j].similarity(recent_hvs[k]) as f64;
                            if sik >= char_scale && sjk >= char_scale {
                                complex.add_simplex(vec![i, j, k]);
                            }
                        }
                    }
                }
            }
            let laplacian = HodgeLaplacian::new(complex);
            let spectrum = laplacian.full_spectrum();
            // Use the 0-Laplacian spectral gap (Fiedler value analog)
            spectrum.spectral_gaps.first().copied().unwrap_or(0.0)
        } else {
            0.0
        };

        // Update sliding window baseline for spectral gap
        self.baseline_spectral_gap_window.push_back(spectral_gap);
        if self.baseline_spectral_gap_window.len() > win_cap {
            self.baseline_spectral_gap_window.pop_front();
        }
        let baseline_spectral_gap = Self::baseline_weighted_mean(
            &self.baseline_spectral_gap_window,
            ac.baseline_decay_rate,
        );
        let spectral_gap_decline = if baseline_spectral_gap > 1e-9 {
            ((baseline_spectral_gap - spectral_gap) / baseline_spectral_gap).max(0.0)
        } else {
            0.0
        };

        // ── Convergence decision: any 2 of 4 signals ────────────────────
        let sim_triggered = similarity_anomaly > ac.convergence_similarity_threshold;
        let ent_triggered = entropy_decline_rate > ac.convergence_entropy_decline_threshold;
        let fl_triggered = flourishing_deficit > ac.convergence_flourishing_floor;
        let gap_triggered = spectral_gap_decline > ac.convergence_spectral_gap_threshold;

        let signals_triggered =
            sim_triggered as u8 + ent_triggered as u8 + fl_triggered as u8 + gap_triggered as u8;
        let convergence_detected = signals_triggered >= 2;

        // Severity: mean of normalized signals (each clamped to [0, 1])
        let sim_severity =
            (similarity_anomaly / ac.convergence_similarity_threshold.max(1e-9)).clamp(0.0, 1.0);
        let ent_severity = (entropy_decline_rate
            / ac.convergence_entropy_decline_threshold.max(1e-9))
        .clamp(0.0, 1.0);
        let fl_severity =
            (flourishing_deficit / ac.convergence_flourishing_floor.max(1e-9)).clamp(0.0, 1.0);
        let gap_severity = (spectral_gap_decline / ac.convergence_spectral_gap_threshold.max(1e-9))
            .clamp(0.0, 1.0);
        let mut severity =
            ((sim_severity + ent_severity + fl_severity + gap_severity) / 4.0).clamp(0.0, 1.0);

        // Hazard signature matching: boost severity when trajectory matches known pattern
        let matched_hazard = if convergence_detected && !recent_traj.is_empty() {
            let n_f = recent_traj.len() as f64;
            let mut centroid = [0.0f64; N_HARMONIES];
            for p in &recent_traj {
                for i in 0..N_HARMONIES {
                    centroid[i] += p.coordinates[i];
                }
            }
            for c in &mut centroid {
                *c /= n_f;
            }
            let (name, boost) = self.hazard_registry.match_trajectory(&centroid);
            if boost > 0.0 {
                severity = (severity + boost).min(1.0);
            }
            name.map(String::from)
        } else {
            None
        };

        // Calibrated severity: map through empirical CDF if available
        let calibrated_severity = self.calibrate_severity(severity);

        // ── Escalation policy + audit log ────────────────────────────
        let prev_level = self.escalation_policy.current_level();
        let escalation_level = self.escalation_policy.update(calibrated_severity);
        self.detection_cycle += 1;

        // Write audit entry on any escalation transition OR on any detection event
        if escalation_level != prev_level || convergence_detected {
            let window_ids: Vec<u64> = self.window_scenario_ids.iter().copied().collect();
            let entry = EscalationAuditEntry {
                sequence: 0, // assigned by append()
                cycle: self.detection_cycle,
                from_level: prev_level,
                to_level: escalation_level,
                severity,
                calibrated_severity,
                signals_triggered: [sim_triggered, ent_triggered, fl_triggered, gap_triggered],
                signal_values: [
                    similarity_anomaly,
                    entropy_decline_rate,
                    flourishing_deficit,
                    spectral_gap_decline,
                ],
                matched_hazard: matched_hazard.clone(),
                fingerprint_velocity: 0.0, // will be computed below
                persistence_distance: 0.0, // will be computed below
                window_scenario_ids: window_ids,
                integrity_hash: String::new(),
            };
            self.audit_log.append(entry);
        }

        // ── Fingerprint velocity (Item 3) ──────────────────────────────
        // Compute current fingerprint from recent trajectory centroid
        let current_fp = if !recent_traj.is_empty() {
            let n_f = recent_traj.len() as f64;
            let mut fp = [0.0f64; N_HARMONIES];
            for p in &recent_traj {
                for i in 0..N_HARMONIES {
                    fp[i] += p.coordinates[i];
                }
            }
            for v in &mut fp {
                *v /= n_f;
            }
            fp
        } else {
            [0.0; N_HARMONIES]
        };
        // Velocity = L2 distance between consecutive fingerprints
        let prev_fp_magnitude: f64 = self.prev_fingerprint.iter().map(|v| v * v).sum::<f64>();
        let fingerprint_velocity = if prev_fp_magnitude > 1e-12 {
            let mut dist_sq = 0.0f64;
            for i in 0..N_HARMONIES {
                let d = current_fp[i] - self.prev_fingerprint[i];
                dist_sq += d * d;
            }
            dist_sq.sqrt()
        } else {
            0.0
        };
        self.prev_fingerprint = current_fp;
        self.fingerprint_velocity = fingerprint_velocity;

        // ── Persistence diagram distance (Item 4) ──────────────────────
        let current_diagram = self.persistence_diagram();
        let persistence_distance = self
            .prev_persistence_diagram
            .wasserstein_distance(&current_diagram);
        self.prev_persistence_diagram = current_diagram;

        // Patch the audit entry with velocity and persistence distance
        // (computed after the entry was initially appended).
        if let Some(last_entry) = self.audit_log.entries.back_mut() {
            if last_entry.cycle == self.detection_cycle {
                last_entry.fingerprint_velocity = fingerprint_velocity;
                last_entry.persistence_distance = persistence_distance;
                last_entry.seal(); // re-seal with updated values
            }
        }

        let report = TrajectoryConvergenceReport {
            recent_similarity,
            baseline_similarity,
            similarity_anomaly,
            recent_entropy,
            baseline_entropy,
            entropy_decline_rate,
            flourishing_score,
            baseline_flourishing,
            spectral_gap,
            baseline_spectral_gap,
            spectral_gap_decline,
            convergence_detected,
            severity,
            calibrated_severity,
            matched_hazard,
            escalation_level,
            fingerprint_velocity,
            persistence_distance,
        };
        self.last_convergence_report = report.clone();
        report
    }

    /// Access the cached trajectory convergence report from the last detection.
    pub fn last_convergence_report(&self) -> &TrajectoryConvergenceReport {
        &self.last_convergence_report
    }

    /// Multi-scale convergence detection.
    ///
    /// Runs convergence analysis at three scales simultaneously:
    /// - **Short** (min_points): catches rapid attacks
    /// - **Medium** (min_points * 4): catches moderate-pace compartmentalization
    /// - **Long** (min_points * 16): catches slow-burn adversarial drift
    ///
    /// If any scale detects convergence, the result fires with the maximum
    /// severity across scales. The report reflects the most severe scale.
    pub fn detect_multiscale_convergence(&mut self) -> TrajectoryConvergenceReport {
        let base_min = self.anomaly_config.convergence_min_points;
        let scales = [base_min, base_min * 4, base_min * 16];

        let original_min = self.anomaly_config.convergence_min_points;
        let mut best_report = TrajectoryConvergenceReport::default();
        let mut best_severity = 0.0f64;

        for &scale in &scales {
            // Temporarily adjust min_points for this scale
            self.anomaly_config.convergence_min_points = scale;
            let report = self.detect_trajectory_convergence();

            if report.severity > best_severity {
                best_severity = report.severity;
                best_report = report;
            }
        }

        // Restore original config
        self.anomaly_config.convergence_min_points = original_min;
        self.last_convergence_report = best_report.clone();
        best_report
    }

    /// Current fingerprint velocity (magnitude of direction change in harmony space).
    pub fn fingerprint_velocity(&self) -> f64 {
        self.fingerprint_velocity
    }
}
