// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fabrication & manufacturing constants — Cincinnati quality monitoring,
//! ManufacturingTwin integration, and PoGF consciousness coupling.

// ═══════════════════════════════════════════════════════════════════════════════
// FABRICATION MANAGER — MANUFACTURING CONSCIOUSNESS COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Co-prime interval for fabrication manager (47 is co-prime with 7,11,13,19,23,29,37,41,43,53,67).
pub const FABRICATION_MANAGER_INTERVAL: u32 = 47;

/// Minimum dose magnitude to queue a neuromod effect (prevents spurious micro-nudges).
pub const FAB_NEUROMOD_FLOOR: f32 = 0.01;

/// NE phasic burst dose on Cincinnati anomaly detection (severity > 0.5).
/// Basis: Aston-Jones & Cohen (2005) — LC-NE phasic response to salient events
/// requiring attentional reorientation.
pub const FAB_ANOMALY_NE_DOSE: f32 = 0.06;

/// NE phasic burst half-life (cycles) on anomaly detection.
pub const FAB_ANOMALY_NE_HALFLIFE: u32 = 15;

/// Anomaly severity threshold for neuromod engagement.
/// Below this, anomalies are logged but don't trigger NE bursts.
pub const FAB_ANOMALY_SEVERITY_THRESHOLD: f32 = 0.5;

/// Anomaly severity threshold for ANOMALY_DETECTED flag.
pub const FAB_ANOMALY_FLAG_THRESHOLD: f32 = 0.7;

/// DA phasic burst dose on successful print completion.
/// Basis: Schultz (1997) — phasic DA encodes positive reward prediction error.
pub const FAB_PRINT_SUCCESS_DA_DOSE: f32 = 0.08;

/// DA phasic burst half-life (cycles) on print completion.
pub const FAB_PRINT_SUCCESS_DA_HALFLIFE: u32 = 25;

/// NE+5-HT stress dose on emergency halt (ManufacturingSafetyLevel::Red).
/// Basis: Sapolsky (2004) — acute stressor → concurrent NE surge + 5-HT dip.
pub const FAB_EMERGENCY_NE_DOSE: f32 = 0.10;

/// 5-HT baseline dip on emergency halt.
/// Basis: Sapolsky (2004) — stress → serotonin suppression.
pub const FAB_EMERGENCY_SHT_NUDGE: f32 = -0.05;

/// NE emergency half-life (cycles).
pub const FAB_EMERGENCY_NE_HALFLIFE: u32 = 30;

/// 5-HT baseline rise on sustained quality improvement (positive quality trend).
/// Basis: Crockett (2009) — positive social/environmental feedback → serotonin
/// supports well-being and continued prosocial behavior.
pub const FAB_QUALITY_TREND_SHT_NUDGE: f32 = 0.03;

/// Oxytocin boost on high PoGF score (prosocial reward for quality manufacturing).
/// Basis: Zak (2012) — prosocial outcomes → oxytocin release for trust/bonding.
pub const FAB_HIGH_POG_OXY_DOSE: f32 = 0.03;

/// Oxytocin half-life (cycles) for PoGF reward.
pub const FAB_HIGH_POG_OXY_HALFLIFE: u32 = 40;

/// PoGF threshold for oxytocin boost (above this = high quality).
pub const FAB_POG_HIGH_THRESHOLD: f32 = 0.7;

/// LR dampening factor on poor quality signals (quality < 0.5).
/// Basis: Friston (2010) — low-precision signals should reduce learning rate
/// to prevent encoding noisy patterns.
pub const FAB_LOW_QUALITY_LR_DAMPEN: f64 = 0.8;

/// Quality threshold below which LR is dampened.
pub const FAB_LOW_QUALITY_THRESHOLD: f32 = 0.5;

/// Exploration boost on very poor quality (quality < 0.3).
/// Basis: Berlyne (1960) — poor outcomes increase arousal and drive exploration
/// of alternative strategies.
pub const FAB_POOR_QUALITY_EXPLORE_DELTA: f64 = 0.1;

/// Quality threshold for exploration boost.
pub const FAB_POOR_QUALITY_THRESHOLD: f32 = 0.3;

/// EMA alpha for anomaly severity tracking.
pub const FAB_ANOMALY_EMA_ALPHA: f32 = 0.15;

/// EMA alpha for PoGF score tracking.
pub const FAB_POG_EMA_ALPHA: f32 = 0.1;

/// EMA decay for manufacturing reward tracking (alpha = 1 - decay).
/// Basis: Sutton & Barto (2018) — exponential recency-weighted average.
pub const FAB_REWARD_EMA_DECAY: f64 = 0.9;

/// Arousal delta on ManufacturingSafetyLevel::Red (emergency).
pub const FAB_EMERGENCY_AROUSAL: f32 = 0.15;

/// Arousal delta on ManufacturingSafetyLevel::Orange (elevated concern).
pub const FAB_ORANGE_AROUSAL: f32 = 0.05;

/// Confidence boost on successful print completion.
pub const FAB_PRINT_SUCCESS_CONFIDENCE: f64 = 0.02;

/// Confidence penalty on failed print / emergency halt.
pub const FAB_PRINT_FAILURE_CONFIDENCE: f64 = -0.04;

/// Maximum stored fabrication outcomes.
pub const FAB_MAX_OUTCOMES: usize = 64;

/// Maximum events drained per process() call.
pub const FAB_MAX_EVENTS_PER_CYCLE: usize = 32;

// ═══════════════════════════════════════════════════════════════════════════════
// ORDERING VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fabrication_threshold_ordering() {
        // Anomaly thresholds are ordered
        assert!(FAB_ANOMALY_SEVERITY_THRESHOLD < FAB_ANOMALY_FLAG_THRESHOLD);
        // Quality thresholds: poor < low
        assert!(FAB_POOR_QUALITY_THRESHOLD < FAB_LOW_QUALITY_THRESHOLD);
        // Emergency arousal > orange arousal
        assert!(FAB_EMERGENCY_AROUSAL > FAB_ORANGE_AROUSAL);
        // Reward EMA decay in (0, 1)
        assert!(FAB_REWARD_EMA_DECAY > 0.0 && FAB_REWARD_EMA_DECAY < 1.0);
        // PoGF threshold in (0, 1)
        assert!(FAB_POG_HIGH_THRESHOLD > 0.0 && FAB_POG_HIGH_THRESHOLD < 1.0);
    }

    #[test]
    fn fabrication_interval_coprime() {
        let interval = FABRICATION_MANAGER_INTERVAL;
        for other in [7u32, 11, 13, 19, 23, 29, 37, 41, 43, 53, 67] {
            assert_ne!(
                interval % other,
                0,
                "{} should be co-prime with {}",
                interval,
                other
            );
        }
    }
}
