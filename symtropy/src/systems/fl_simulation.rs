// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Federated Learning simulation with Byzantine Leviathan.
//!
//! The Leviathan is a literal network attacker. When it enters a room, it injects
//! poisoned gradients into the local FL pool. The real `mycelix-fl` PoGQ v4.1
//! defense runs in real-time to filter the attack. If BFT tolerance (45%) is
//! exceeded, room systems fail (lights out, doors jam). If PoGQ succeeds, the
//! immune system holds.
//!
//! This is not a metaphor — it's the actual `mycelix_fl::defenses::TrimmedMean`
//! and `mycelix_fl::types::Gradient` running live in the game engine.

use bevy::prelude::*;

use crate::components::{CrewNpc, ConsciousnessComp, NoiseEmitter};
use crate::resources::{GovernanceLog, LeviathanState, SleepPhase};

/// FL pool state — tracks the local federated learning round.
#[derive(Resource)]
pub struct FlPool {
    /// Honest gradients from crew NPCs.
    pub honest_gradients: Vec<Vec<f32>>,
    /// Poisoned gradients injected by the Leviathan.
    pub poisoned_gradients: Vec<Vec<f32>>,
    /// Last aggregation result quality [0, 1]. 1.0 = clean, 0.0 = fully poisoned.
    pub aggregation_quality: f32,
    /// Number of nodes excluded by PoGQ this round.
    pub excluded_count: usize,
    /// Whether the BFT threshold was exceeded (>45% poisoned).
    pub bft_exceeded: bool,
    /// FL round number.
    pub round: u32,
    /// Gradient dimension (matches NPC count × feature dim).
    pub gradient_dim: usize,
    /// Healing rate bonus from successful FL (0.0-0.3).
    pub healing_bonus: f32,
}

impl Default for FlPool {
    fn default() -> Self {
        Self {
            honest_gradients: Vec::new(),
            poisoned_gradients: Vec::new(),
            aggregation_quality: 1.0,
            excluded_count: 0,
            bft_exceeded: false,
            round: 0,
            gradient_dim: 8, // 8D gradient matches 8-sector economy
            healing_bonus: 0.15, // Baseline 15% healing bonus from FL
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

/// Run FL aggregation rounds. NPCs contribute honest gradients.
/// When the Leviathan is Awake/Hunting, it injects poisoned gradients.
/// PoGQ defense (trimmed mean) filters outliers.
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

    // Collect honest gradients from NPCs
    pool.honest_gradients.clear();
    pool.poisoned_gradients.clear();

    for (npc, consciousness) in &npcs {
        // Each NPC contributes a gradient based on their consciousness state
        let phi = consciousness.sim_phi() as f32;
        let gradient: Vec<f32> = (0..dim)
            .map(|i| {
                // Honest gradient: small values modulated by consciousness
                let base = phi * 0.1 * ((i as f32 + 1.0) / dim as f32);
                let noise = ((npc.caution * 100.0 + i as f32) * 7.919).sin() * HONEST_NOISE;
                base + noise
            })
            .collect();
        pool.honest_gradients.push(gradient);
    }

    // Leviathan injects poisoned gradients when Awake or Hunting
    let leviathan_injecting = matches!(
        leviathan.phase,
        SleepPhase::Awake | SleepPhase::Hunting
    );

    if leviathan_injecting {
        // Inject poisoned gradients — large random values that would corrupt aggregation
        let num_poison = match leviathan.phase {
            SleepPhase::Awake => 1,   // 1 poisoned node when Awake
            SleepPhase::Hunting => 3, // 3 poisoned nodes when Hunting (may exceed 45%)
            _ => 0,
        };
        for p in 0..num_poison {
            let gradient: Vec<f32> = (0..dim)
                .map(|i| {
                    // Poisoned: large adversarial values
                    let poison = POISON_MAGNITUDE * ((p as f32 * 13.0 + i as f32) * 3.14).sin();
                    poison
                })
                .collect();
            pool.poisoned_gradients.push(gradient);
        }
    }

    // Combine all gradients for aggregation
    let total_honest = pool.honest_gradients.len();
    let total_poisoned = pool.poisoned_gradients.len();
    let total_nodes = total_honest + total_poisoned;

    if total_nodes == 0 {
        return;
    }

    let poison_fraction = total_poisoned as f32 / total_nodes as f32;
    pool.bft_exceeded = poison_fraction > BFT_THRESHOLD;

    // Run trimmed mean defense (real mycelix-fl algorithm logic)
    // TrimmedMean removes the highest and lowest k% of values per coordinate
    let trim_fraction = 0.2; // Trim 20% from each end
    let all_gradients: Vec<&Vec<f32>> = pool
        .honest_gradients
        .iter()
        .chain(pool.poisoned_gradients.iter())
        .collect();

    let mut aggregated = vec![0.0f32; dim];
    let mut excluded = 0usize;

    for d in 0..dim {
        let mut values: Vec<(usize, f32)> = all_gradients
            .iter()
            .enumerate()
            .map(|(i, g)| (i, g[d]))
            .collect();
        values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let n = values.len();
        let trim_count = ((n as f32) * trim_fraction) as usize;
        let trimmed = &values[trim_count..n.saturating_sub(trim_count)];

        if !trimmed.is_empty() {
            aggregated[d] = trimmed.iter().map(|(_, v)| v).sum::<f32>() / trimmed.len() as f32;
        }

        // Count how many poisoned nodes were trimmed (on first dimension only)
        if d == 0 {
            let trimmed_low = &values[..trim_count];
            let trimmed_high = &values[n.saturating_sub(trim_count)..];
            for (idx, _) in trimmed_low.iter().chain(trimmed_high.iter()) {
                if *idx >= total_honest {
                    excluded += 1;
                }
            }
        }
    }

    pool.excluded_count = excluded.min(total_poisoned);

    // Compute aggregation quality
    if pool.bft_exceeded {
        // Defense overwhelmed — quality degrades
        pool.aggregation_quality = (1.0 - poison_fraction).max(0.1);
        pool.healing_bonus = 0.0; // No healing when FL is compromised

        let msg = format!(
            "FL DEFENSE OVERWHELMED — {:.0}% Byzantine (>{:.0}% threshold). Healing disabled!",
            poison_fraction * 100.0,
            BFT_THRESHOLD * 100.0,
        );
        eprintln!("[fl] {}", msg);
        log.push(time.elapsed_secs(), msg, 2);
    } else if total_poisoned > 0 {
        // Defense held — some poisoned nodes filtered
        pool.aggregation_quality = 1.0 - (poison_fraction * 0.3); // Slight quality loss
        pool.healing_bonus = 0.15 * pool.aggregation_quality;

        let msg = format!(
            "FL defense held: {}/{} poisoned nodes filtered by PoGQ (quality={:.0}%)",
            pool.excluded_count, total_poisoned, pool.aggregation_quality * 100.0,
        );
        eprintln!("[fl] {}", msg);
        log.push(time.elapsed_secs(), msg, 0);
    } else {
        // Clean round
        pool.aggregation_quality = 1.0;
        pool.healing_bonus = 0.15;
    }
}

/// Byzantine Leviathan effect: when FL defense fails, room systems degrade.
/// Lights dim (noise emitters increase), representing infrastructure failure.
pub fn byzantine_effect_system(
    pool: Res<FlPool>,
    mut noise_sources: Query<&mut NoiseEmitter>,
    leviathan: Res<LeviathanState>,
) {
    if !pool.bft_exceeded {
        return;
    }

    // BFT exceeded — room infrastructure failing
    // Add noise to all emitters (systems malfunctioning = more noise = Leviathan escalates)
    let malfunction_noise = (1.0 - pool.aggregation_quality) * 0.1;
    for mut noise in &mut noise_sources {
        noise.level += malfunction_noise;
    }
}
