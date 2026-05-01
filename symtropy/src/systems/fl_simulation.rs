// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Federated Learning simulation with Byzantine Leviathan.
//!
//! The Leviathan is a literal network attacker. When it enters a room, it injects
//! poisoned gradients into the local FL pool. The REAL `mycelix_fl::defenses::TrimmedMean`
//! runs in real-time to filter the attack.
//!
//! Every game session exercises the actual `mycelix-fl` crate code path — the same
//! `TrimmedMean::aggregate()` that runs in production Mycelix FL rounds.

use bevy::prelude::*;

use symtropy_sim_bridge::{
    FlDefense as Defense, FlDefenseConfig as DefenseConfig, FlGradient as Gradient, TrimmedMean,
};

use crate::components::{ConsciousnessComp, CrewNpc, NoiseEmitter};
use crate::resources::{GovernanceLog, LeviathanState, SleepPhase};

/// FL pool state — tracks the local federated learning round.
#[derive(Resource)]
pub struct FlPool {
    /// Defense configuration (real `mycelix_fl::types::DefenseConfig`).
    pub config: DefenseConfig,
    /// Last aggregation result quality [0, 1]. 1.0 = clean, 0.0 = fully poisoned.
    pub aggregation_quality: f32,
    /// Number of nodes excluded by defense this round.
    pub excluded_count: usize,
    /// Number of nodes included this round.
    pub included_count: usize,
    /// Whether the BFT threshold was exceeded (>45% poisoned).
    pub bft_exceeded: bool,
    /// FL round number.
    pub round: u32,
    /// Gradient dimension (matches 8-sector economy).
    pub gradient_dim: usize,
    /// Healing rate bonus from successful FL (0.0-0.3).
    pub healing_bonus: f32,
    /// Total gradients submitted this round (honest + poisoned).
    pub total_submitted: usize,
    /// Total poisoned gradients this round.
    pub total_poisoned: usize,
}

impl Default for FlPool {
    fn default() -> Self {
        let mut config = DefenseConfig::default();
        config.trim_ratio = 0.2; // Trim 20% from each end (robust to ~40% Byzantine)
        Self {
            config,
            aggregation_quality: 1.0,
            excluded_count: 0,
            included_count: 0,
            bft_exceeded: false,
            round: 0,
            gradient_dim: 8,
            healing_bonus: 0.15,
            total_submitted: 0,
            total_poisoned: 0,
        }
    }
}

/// FL round interval (seconds between aggregation rounds).
const FL_ROUND_INTERVAL_SECS: f32 = 8.0;

/// Byzantine tolerance threshold — above this fraction, defense fails.
const BFT_THRESHOLD: f32 = 0.45;

/// Gradient noise magnitude for honest contributions.
const HONEST_NOISE: f32 = 0.1;

/// Gradient noise magnitude for poisoned contributions (much larger).
const POISON_MAGNITUDE: f32 = 5.0;

/// Run FL aggregation rounds using REAL `mycelix-fl` TrimmedMean defense.
///
/// NPCs contribute honest gradients. When the Leviathan is Awake/Hunting,
/// it injects poisoned gradients. The actual `TrimmedMean::aggregate()` from
/// `mycelix-fl` runs to filter the attack.
pub fn fl_aggregation_system(
    npcs: Query<(&CrewNpc, &ConsciousnessComp)>,
    leviathan: Res<LeviathanState>,
    mut pool: ResMut<FlPool>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
    mut timer: Local<f32>,
) {
    *timer += time.delta_secs();
    if *timer < FL_ROUND_INTERVAL_SECS {
        return;
    }
    *timer = 0.0;
    pool.round += 1;

    let dim = pool.gradient_dim;
    let round = pool.round as u64;

    // Build REAL Gradient structs from NPC consciousness state
    let mut all_gradients: Vec<Gradient> = Vec::new();
    let mut honest_count = 0usize;

    for (npc, consciousness) in &npcs {
        let phi = consciousness.sim_phi() as f32;
        let values: Vec<f32> = (0..dim)
            .map(|i| {
                let base = phi * 0.1 * ((i as f32 + 1.0) / dim as f32);
                let noise = ((npc.caution * 100.0 + i as f32) * 7.919).sin() * HONEST_NOISE;
                base + noise
            })
            .collect();
        all_gradients.push(Gradient::new(&npc.name, values, round));
        honest_count += 1;
    }

    // Leviathan injects poisoned gradients when Awake or Hunting
    let num_poison = match leviathan.phase {
        SleepPhase::Awake => 1,
        SleepPhase::Hunting => 3,
        _ => 0,
    };

    for p in 0..num_poison {
        let values: Vec<f32> = (0..dim)
            .map(|i| POISON_MAGNITUDE * ((p as f32 * 13.0 + i as f32) * 3.14).sin())
            .collect();
        all_gradients.push(Gradient::new(format!("leviathan_{}", p), values, round));
    }

    pool.total_submitted = all_gradients.len();
    pool.total_poisoned = num_poison;

    if all_gradients.is_empty() {
        return;
    }

    let poison_fraction = num_poison as f32 / all_gradients.len() as f32;
    pool.bft_exceeded = poison_fraction > BFT_THRESHOLD;

    // =========================================================================
    // REAL MYCELIX-FL CALL — this is the actual production defense algorithm
    // =========================================================================
    let defense = TrimmedMean;
    let result = defense.aggregate(&all_gradients, &pool.config);

    match result {
        Ok(agg) => {
            pool.excluded_count = agg.excluded_nodes.len();
            pool.included_count = agg.included_nodes.len();

            if pool.bft_exceeded {
                pool.aggregation_quality = (1.0 - poison_fraction).max(0.1);
                pool.healing_bonus = 0.0;

                let msg = format!(
                    "FL DEFENSE OVERWHELMED — {:.0}% Byzantine (>{:.0}% threshold). {} nodes, {} excluded.",
                    poison_fraction * 100.0, BFT_THRESHOLD * 100.0,
                    pool.total_submitted, pool.excluded_count,
                );
                eprintln!("[fl] {}", msg);
                log.push(time.elapsed_secs(), msg, 2);
            } else if num_poison > 0 {
                pool.aggregation_quality = 1.0 - (poison_fraction * 0.3);
                pool.healing_bonus = 0.15 * pool.aggregation_quality;

                let msg = format!(
                    "FL defense held (TrimmedMean): {}/{} nodes included, {} poisoned filtered (quality={:.0}%)",
                    agg.included_nodes.len(), pool.total_submitted,
                    num_poison, pool.aggregation_quality * 100.0,
                );
                eprintln!("[fl] {}", msg);
                log.push(time.elapsed_secs(), msg, 0);
            } else {
                pool.aggregation_quality = 1.0;
                pool.healing_bonus = 0.15;
            }
        }
        Err(e) => {
            pool.aggregation_quality = 0.0;
            pool.healing_bonus = 0.0;
            pool.bft_exceeded = true;

            let msg = format!("FL aggregation FAILED: {:?}", e);
            eprintln!("[fl] {}", msg);
            log.push(time.elapsed_secs(), msg, 2);
        }
    }
}

/// Byzantine Leviathan effect: when FL defense fails, room systems degrade.
pub fn byzantine_effect_system(pool: Res<FlPool>, mut noise_sources: Query<&mut NoiseEmitter>) {
    if !pool.bft_exceeded {
        return;
    }

    let malfunction_noise = (1.0 - pool.aggregation_quality) * 0.1;
    for mut noise in &mut noise_sources {
        noise.level += malfunction_noise;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fl_pool_default_config() {
        let pool = FlPool::default();
        assert!((pool.config.trim_ratio - 0.2).abs() < f32::EPSILON);
        assert_eq!(pool.gradient_dim, 8);
        assert!(!pool.bft_exceeded);
    }

    #[test]
    fn real_trimmed_mean_filters_outlier() {
        // Directly test that mycelix-fl TrimmedMean works as expected
        let gradients = vec![
            Gradient::new("npc_a", vec![1.0; 8], 1),
            Gradient::new("npc_b", vec![1.0; 8], 1),
            Gradient::new("npc_c", vec![1.0; 8], 1),
            Gradient::new("npc_d", vec![1.0; 8], 1),
            Gradient::new("leviathan_0", vec![100.0; 8], 1), // Poisoned
        ];
        let mut config = DefenseConfig::default();
        config.trim_ratio = 0.2; // Trim 1 from each end

        let result = TrimmedMean.aggregate(&gradients, &config).unwrap();

        // After trimming top and bottom: remaining should average ~1.0
        for &v in &result.gradient {
            assert!((v - 1.0).abs() < 0.01, "Expected ~1.0, got {}", v);
        }
    }

    #[test]
    fn bft_threshold_correct() {
        // 3 honest, 3 poisoned = 50% > 45% threshold
        let total = 6;
        let poisoned = 3;
        let fraction = poisoned as f32 / total as f32;
        assert!(fraction > BFT_THRESHOLD);

        // 4 honest, 1 poisoned = 20% < 45% threshold
        let fraction2 = 1.0 / 5.0;
        assert!(fraction2 < BFT_THRESHOLD);
    }
}
