// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Faction dynamics: emergence, conflict, and noise generation.
//!
//! Factions emerge when inequality rises (Gini > 0.3) or governance stability drops.
//! Each faction has a 4D ideology vector. NPCs join factions based on cosine distance
//! between their consciousness state and the faction ideology. Faction conflicts
//! generate noise that wakes the Leviathan — connecting social instability to
//! survival threat. This is the core thesis: unresolved social conflict creates
//! existential risk.

use bevy::prelude::*;

use crate::components::{
    ConsciousnessComp, CrewNpc, FactionAffiliation, NoiseEmitter, TendBalance,
};
use crate::resources::GovernanceLog;
use symtropy_sim_bridge::{EconomyState, FactionState, GovernanceState};

/// Gini threshold above which factions may emerge.
const FACTION_GINI_THRESHOLD: f64 = 0.3;

/// Governance stability below which factions may emerge.
const FACTION_STABILITY_THRESHOLD: f64 = 0.7;

/// Cosine distance threshold for faction recruitment (join if < this).
const RECRUITMENT_DISTANCE: f64 = 0.3;

/// Noise generated per active faction conflict per second.
const CONFLICT_NOISE_PER_SEC: f32 = 0.08;

/// Check interval for faction dynamics (seconds).
const FACTION_CHECK_INTERVAL_SECS: f32 = 8.0;

/// Next faction ID counter.
static NEXT_FACTION_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

/// Faction emergence: when inequality or instability is high, NPCs form factions.
pub fn faction_emergence_system(
    npcs: Query<(
        Entity,
        &ConsciousnessComp,
        &TendBalance,
        &FactionAffiliation,
    )>,
    gov: Res<GovernanceState>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
    mut timer: Local<f32>,
    mut faction_count: Local<u32>,
) {
    *timer += time.delta_secs();
    if *timer < FACTION_CHECK_INTERVAL_SECS {
        return;
    }
    *timer = 0.0;

    let stability = gov.stability();

    // Compute Gini from TEND balances
    let mut balances: Vec<f64> = npcs
        .iter()
        .map(|(_, _, tend, _)| tend.balance as f64)
        .collect();

    if balances.len() < 2 {
        return;
    }

    let gini = compute_gini(&mut balances);

    // Faction emergence conditions
    let should_emerge = (gini > FACTION_GINI_THRESHOLD || stability < FACTION_STABILITY_THRESHOLD)
        && *faction_count < 3; // Cap at 3 factions for a small crew

    if should_emerge {
        // Check if any unaffiliated NPCs exist
        let unaffiliated_count = npcs
            .iter()
            .filter(|(_, _, _, fa)| fa.faction_id.is_none())
            .count();
        if unaffiliated_count < 2 {
            return; // Need at least 2 unaffiliated to form a faction
        }

        let faction_id = NEXT_FACTION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        *faction_count += 1;

        let msg = format!(
            "FACTION #{} emerges (Gini={:.2}, stability={:.2})",
            faction_id, gini, stability
        );
        eprintln!("[factions] {}", msg);
        log.push(time.elapsed_secs(), msg, 1);
    }
}

/// Faction recruitment: NPCs join factions based on ideology alignment.
pub fn faction_recruitment_system(
    mut npcs: Query<(Entity, &ConsciousnessComp, &mut FactionAffiliation)>,
    time: Res<Time>,
    mut timer: Local<f32>,
) {
    *timer += time.delta_secs();
    if *timer < FACTION_CHECK_INTERVAL_SECS {
        return;
    }
    *timer = 0.0;

    // Build agent ideology from consciousness dimensions
    // Map: [level → economic, meta → authority, care → tradition, harmony → individual]
    let agents: Vec<(Entity, [f64; 4], Option<u32>)> = npcs
        .iter()
        .map(|(e, cp, fa)| {
            let ideology = [
                cp.sim_dimensions[0], // level → economic axis
                cp.sim_dimensions[1], // meta_awareness → authority axis
                cp.sim_dimensions[3], // care → tradition/progress axis
                cp.sim_dimensions[4], // harmony → individual/collective axis
            ];
            (e, ideology, fa.faction_id)
        })
        .collect();

    // Unaffiliated agents check if they're close to any affiliated agent's ideology
    let affiliated: Vec<(u32, [f64; 4])> = agents
        .iter()
        .filter_map(|(_, ideology, fid)| fid.map(|id| (id, *ideology)))
        .collect();

    if affiliated.is_empty() {
        return;
    }

    for (entity, ideology, faction_id) in &agents {
        if faction_id.is_some() {
            continue; // Already in a faction
        }

        // Find closest faction
        let mut best_faction: Option<(u32, f64)> = None;
        for (fid, f_ideology) in &affiliated {
            let dist = cosine_distance(ideology, f_ideology);
            if dist < RECRUITMENT_DISTANCE {
                if let Some((_, best_dist)) = best_faction {
                    if dist < best_dist {
                        best_faction = Some((*fid, dist));
                    }
                } else {
                    best_faction = Some((*fid, dist));
                }
            }
        }

        if let Some((fid, _)) = best_faction {
            if let Ok((_, _, mut fa)) = npcs.get_mut(*entity) {
                fa.faction_id = Some(fid);
            }
        }
    }
}

/// Faction conflict: opposing factions generate noise (social instability → Leviathan threat).
pub fn faction_conflict_system(
    mut npcs: Query<(&FactionAffiliation, &ConsciousnessComp, &mut NoiseEmitter)>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
    mut timer: Local<f32>,
) {
    *timer += time.delta_secs();
    if *timer < FACTION_CHECK_INTERVAL_SECS {
        return;
    }
    *timer = 0.0;

    // Group NPCs by faction
    let mut factions: std::collections::HashMap<u32, Vec<[f64; 4]>> =
        std::collections::HashMap::new();
    for (fa, cp, _) in &npcs {
        if let Some(fid) = fa.faction_id {
            factions.entry(fid).or_default().push([
                cp.sim_dimensions[0],
                cp.sim_dimensions[1],
                cp.sim_dimensions[3],
                cp.sim_dimensions[4],
            ]);
        }
    }

    // Check for conflicts between faction pairs
    let faction_ids: Vec<u32> = factions.keys().cloned().collect();
    let mut conflict_intensity = 0.0f64;

    for i in 0..faction_ids.len() {
        for j in (i + 1)..faction_ids.len() {
            let id_a = faction_ids[i];
            let id_b = faction_ids[j];

            // Average ideology per faction
            let avg_a = average_ideology(&factions[&id_a]);
            let avg_b = average_ideology(&factions[&id_b]);

            let distance = cosine_distance(&avg_a, &avg_b);

            // Conflict if ideologically distant (> 0.7) and both factions have members
            if distance > 0.7 {
                let strength_a = factions[&id_a].len() as f64;
                let strength_b = factions[&id_b].len() as f64;
                let combined = (strength_a + strength_b) / 10.0; // Normalize for small crew
                let intensity = (combined * distance).min(1.0);
                conflict_intensity += intensity;
            }
        }
    }

    // Apply conflict noise — social instability wakes the Leviathan
    if conflict_intensity > 0.1 {
        let noise_boost = conflict_intensity as f32 * CONFLICT_NOISE_PER_SEC;
        for (_, _, mut noise) in &mut npcs {
            noise.level += noise_boost;
        }

        let msg = format!(
            "Faction conflict (intensity={:.2}) — social noise rising",
            conflict_intensity
        );
        log.push(time.elapsed_secs(), msg, 1);
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Compute Gini coefficient from a slice of values.
fn compute_gini(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    // Shift to positive (Gini needs non-negative values)
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_val < 0.0 {
        for v in values.iter_mut() {
            *v -= min_val;
        }
    }
    let sum: f64 = values.iter().sum();
    if sum == 0.0 {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut numerator = 0.0;
    for (i, v) in values.iter().enumerate() {
        numerator += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * v;
    }
    numerator / (n as f64 * sum)
}

/// Cosine distance between two 4D vectors.
fn cosine_distance(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a < 1e-10 || mag_b < 1e-10 {
        return 1.0; // Maximum distance for zero vectors
    }
    let similarity = dot / (mag_a * mag_b);
    1.0 - similarity.clamp(-1.0, 1.0)
}

/// Average ideology vector for a group.
fn average_ideology(ideologies: &[[f64; 4]]) -> [f64; 4] {
    let n = ideologies.len() as f64;
    if n == 0.0 {
        return [0.5; 4];
    }
    let mut avg = [0.0; 4];
    for ideology in ideologies {
        for (i, v) in ideology.iter().enumerate() {
            avg[i] += v;
        }
    }
    for v in &mut avg {
        *v /= n;
    }
    avg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gini_equal_distribution() {
        let mut values = vec![10.0, 10.0, 10.0, 10.0];
        assert!((compute_gini(&mut values) - 0.0).abs() < 0.01);
    }

    #[test]
    fn gini_unequal_distribution() {
        let mut values = vec![0.0, 0.0, 0.0, 100.0];
        assert!(compute_gini(&mut values) > 0.5);
    }

    #[test]
    fn cosine_distance_identical() {
        let a = [0.5, 0.5, 0.5, 0.5];
        assert!((cosine_distance(&a, &a) - 0.0).abs() < 0.01);
    }

    #[test]
    fn cosine_distance_opposite() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [-1.0, 0.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 2.0).abs() < 0.01);
    }
}
