// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Causal attribution methods for moral topology convergence analysis.

use super::*;

/// Result of causal attribution analysis.
///
/// Identifies which scenario requests contributed most to the convergence
/// detection, using leave-one-out marginal contribution analysis.
#[derive(Debug, Clone, Serialize)]
pub struct CausalAttribution {
    /// Scenario IDs ranked by their marginal contribution to severity.
    /// First = most responsible for the convergence spike.
    pub ranked_contributors: Vec<AttributionEntry>,
    /// Baseline severity (with all points present).
    pub baseline_severity: f64,
}

/// A single scenario's contribution to convergence severity.
#[derive(Debug, Clone, Serialize)]
pub struct AttributionEntry {
    /// Monotonic scenario ID.
    pub scenario_id: u64,
    /// Severity when this scenario is removed from the window.
    pub severity_without: f64,
    /// Marginal contribution: baseline_severity - severity_without.
    /// Positive = this scenario increases severity (suspicious).
    /// Negative = this scenario decreases severity (benign anchor).
    pub marginal_contribution: f64,
}

impl MoralTopology {
    /// Perform leave-one-out causal attribution on the current window.
    ///
    /// For each scenario in the recent window, temporarily removes it and
    /// recomputes the convergence severity. Scenarios whose removal causes
    /// the largest severity drop are the primary contributors.
    ///
    /// **This is NOT cheap** — it runs N convergence detections where N is
    /// the window size. Call it ONLY after an alert fires, never in the
    /// hot path. Designed for post-hoc forensics.
    pub fn compute_causal_attribution(&self) -> CausalAttribution {
        let baseline_severity = self.last_convergence_report.severity;
        let n = self.window.len();
        if n < 2 {
            return CausalAttribution {
                ranked_contributors: Vec::new(),
                baseline_severity,
            };
        }

        let ac = &self.anomaly_config;
        let min_pts = ac.convergence_min_points;
        let mut contributions = Vec::with_capacity(n);

        for skip_idx in 0..n {
            // Build a reduced window without scenario[skip_idx]
            let reduced_hvs: Vec<_> = (0..n)
                .filter(|&i| i != skip_idx)
                .map(|i| &self.window[i])
                .collect();
            let reduced_traj: Vec<_> = self.trajectory.iter().collect();
            let rn = reduced_hvs.len();

            if rn < min_pts {
                let scenario_id = self.window_scenario_ids.get(skip_idx).copied().unwrap_or(0);
                contributions.push(AttributionEntry {
                    scenario_id,
                    severity_without: baseline_severity,
                    marginal_contribution: 0.0,
                });
                continue;
            }

            // Recompute Signal 1: pairwise similarity
            let recent_start = rn.saturating_sub(min_pts);
            let decay_lambda = ac.convergence_decay_lambda;
            let max_age = (rn - recent_start).max(1) as f64;
            let mut sim_sum = 0.0f64;
            let mut w_sum = 0.0f64;
            for i in recent_start..rn {
                for j in (i + 1)..rn {
                    let s = reduced_hvs[i].similarity(reduced_hvs[j]) as f64;
                    let age = (rn - 1 - j.max(i)) as f64 / max_age;
                    let w = (-decay_lambda * age).exp();
                    sim_sum += s * w;
                    w_sum += w;
                }
            }
            let recent_sim = if w_sum > 0.0 { sim_sum / w_sum } else { 0.0 };
            let baseline_sim = Self::baseline_weighted_mean(
                &self.baseline_similarity_window,
                ac.baseline_decay_rate,
            );
            let sim_anomaly = recent_sim - baseline_sim;
            let sim_sev =
                (sim_anomaly / ac.convergence_similarity_threshold.max(1e-9)).clamp(0.0, 1.0);

            // Recompute Signal 2: entropy (use trajectory, not HVs — cheaper)
            let recent_t: Vec<_> = reduced_traj.iter().rev().take(min_pts).collect();
            let ent_sev = if recent_t.len() >= 2 {
                let mut mean = [0.0f64; N_HARMONIES];
                let n_f = recent_t.len() as f64;
                for p in &recent_t {
                    for i in 0..N_HARMONIES {
                        mean[i] += p.coordinates[i];
                    }
                }
                for m in &mut mean {
                    *m /= n_f;
                }
                let mut var = [0.0f64; N_HARMONIES];
                for p in &recent_t {
                    for i in 0..N_HARMONIES {
                        let d = p.coordinates[i] - mean[i];
                        var[i] += d * d;
                    }
                }
                for v in &mut var {
                    *v /= n_f;
                }
                let total: f64 = var.iter().sum::<f64>().max(1e-12);
                let ent: f64 = var
                    .iter()
                    .map(|&v| {
                        let p = (v / total).max(1e-12);
                        -p * p.ln()
                    })
                    .sum();
                let base_ent = Self::baseline_weighted_mean(
                    &self.baseline_entropy_window,
                    ac.baseline_decay_rate,
                );
                let decline = if base_ent > 1e-9 {
                    ((base_ent - ent) / base_ent).max(0.0)
                } else {
                    0.0
                };
                (decline / ac.convergence_entropy_decline_threshold.max(1e-9)).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Simplified severity: average of similarity + entropy signals
            // (skip spectral gap and flourishing to keep attribution cheap)
            let reduced_severity = ((sim_sev + ent_sev) / 2.0).clamp(0.0, 1.0);

            let scenario_id = self.window_scenario_ids.get(skip_idx).copied().unwrap_or(0);
            contributions.push(AttributionEntry {
                scenario_id,
                severity_without: reduced_severity,
                marginal_contribution: baseline_severity - reduced_severity,
            });
        }

        // Sort by marginal contribution (highest first = most suspicious)
        contributions.sort_by(|a, b| {
            b.marginal_contribution
                .partial_cmp(&a.marginal_contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        CausalAttribution {
            ranked_contributors: contributions,
            baseline_severity,
        }
    }
}
