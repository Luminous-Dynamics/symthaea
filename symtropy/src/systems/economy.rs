// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TEND economy systems: mutual credit exchange and demurrage.
//!
//! TEND (Time Exchange for Networked Dignity) is a zero-sum mutual credit currency
//! where 1 hour = 1 hour regardless of service type. Demurrage (2% per game-minute)
//! prevents hoarding and encourages circulation. NPCs exchange TEND for services
//! like healing, scouting, and extraction help.

use bevy::prelude::*;

use crate::components::{ConsciousnessComp, CrewNpc, NoiseEmitter, NpcTrust, Player, TendBalance};
use crate::resources::GovernanceLog;

/// Demurrage rate: 2% per game-minute (applied every tick).
const DEMURRAGE_RATE_PER_SEC: f64 = 0.02 / 60.0;

/// Demurrage timer: apply every 5 seconds to avoid per-frame float drift.
const DEMURRAGE_INTERVAL_SECS: f32 = 5.0;

/// TEND amount for basic service exchange.
const SERVICE_TEND_AMOUNT: i64 = 1;

/// Exchange cooldown per NPC (seconds).
const EXCHANGE_COOLDOWN_SECS: f32 = 10.0;

/// TEND exchange system: NPCs offer services to each other and the player.
///
/// Services generate TEND credits: provider gains, receiver loses (zero-sum).
/// Trust grows through successful exchanges. This is the restorative justice path —
/// after coercion (epistemic decay), the only way to recover NPC trust is TEND exchanges.
pub fn tend_exchange_system(
    mut npcs: Query<(
        Entity,
        &CrewNpc,
        &ConsciousnessComp,
        &mut TendBalance,
        &mut NpcTrust,
    )>,
    mut player_tend: Query<&mut TendBalance, (With<Player>, Without<CrewNpc>)>,
    time: Res<Time>,
    mut cooldowns: Local<Vec<(Entity, f32)>>,
    mut log: ResMut<GovernanceLog>,
) {
    let elapsed = time.elapsed_secs();

    // Tick cooldowns
    cooldowns.retain_mut(|(_, cd)| {
        *cd -= time.delta_secs();
        *cd > 0.0
    });

    // Collect NPC data for exchange decisions (avoid borrow conflicts)
    let npc_data: Vec<(Entity, String, f64, f64, i64)> = npcs
        .iter()
        .map(|(e, npc, cp, tend, trust)| {
            (
                e,
                npc.name.clone(),
                cp.combined_score(),
                trust.trust,
                tend.balance,
            )
        })
        .collect();

    if npc_data.len() < 2 {
        return;
    }

    // NPCs with high care_activation offer services to those with lower phi
    for i in 0..npc_data.len() {
        let (entity_i, ref name_i, phi_i, trust_i, balance_i) = npc_data[i];

        // Check cooldown
        if cooldowns.iter().any(|(e, _)| *e == entity_i) {
            continue;
        }

        // Only high-care NPCs initiate exchanges
        let care_i = npcs
            .get(entity_i)
            .map(|(_, _, cp, _, _)| cp.sim_dimensions[3])
            .unwrap_or(0.0);
        if care_i < 0.5 {
            continue;
        }

        // Find an NPC partner who could use help (lower phi, not self)
        for j in 0..npc_data.len() {
            if i == j {
                continue;
            }
            let (entity_j, ref name_j, phi_j, _, balance_j) = npc_data[j];

            // Help those with lower consciousness or lower TEND balance
            if phi_j >= phi_i && balance_j >= balance_i {
                continue;
            }

            // Check provider can afford it
            if let Ok((_, _, _, tend_i, _)) = npcs.get(entity_i) {
                if !tend_i.can_spend(SERVICE_TEND_AMOUNT) {
                    continue;
                }
            }

            // Execute exchange: provider loses TEND, receiver gains
            // This is counter-intuitive: the HELPER pays because they're investing
            // in community. The receiver owes back (mutual credit).
            // Actually: receiver pays for service received, provider earns.
            if let Ok((_, _, _, ref mut tend_j, _)) = npcs.get_mut(entity_j) {
                if !tend_j.can_spend(SERVICE_TEND_AMOUNT) {
                    continue;
                }
                tend_j.balance -= SERVICE_TEND_AMOUNT;
            }
            if let Ok((_, _, _, ref mut tend_i, _)) = npcs.get_mut(entity_i) {
                tend_i.balance += SERVICE_TEND_AMOUNT;
            }

            // Trust grows through exchange
            if let Ok((_, _, _, _, ref mut trust_j)) = npcs.get_mut(entity_j) {
                trust_j.trust = (trust_j.trust + 0.05).min(1.0);
            }

            cooldowns.push((entity_i, EXCHANGE_COOLDOWN_SECS));

            let msg = format!(
                "{} provides service to {} (+{} TEND)",
                name_i, name_j, SERVICE_TEND_AMOUNT
            );
            log.push(elapsed, msg, 0);

            break; // One exchange per NPC per cycle
        }
    }
}

/// Demurrage system: TEND balances decay over time, discouraging hoarding.
/// Positive balances shrink. Negative balances also shrink (debt forgiveness).
/// The system is zero-sum: total TEND always sums to zero.
pub fn demurrage_system(
    mut all_balances: Query<&mut TendBalance>,
    time: Res<Time>,
    mut timer: Local<f32>,
) {
    *timer += time.delta_secs();
    if *timer < DEMURRAGE_INTERVAL_SECS {
        return;
    }
    *timer = 0.0;

    let decay = DEMURRAGE_RATE_PER_SEC * DEMURRAGE_INTERVAL_SECS as f64;

    for mut tend in &mut all_balances {
        if tend.balance > 0 {
            // Positive balances decay toward zero (discourage hoarding)
            let loss = ((tend.balance as f64) * decay).max(1.0) as i64;
            tend.balance = (tend.balance - loss).max(0);
        } else if tend.balance < 0 {
            // Negative balances also decay toward zero (gradual debt forgiveness)
            let gain = (((-tend.balance) as f64) * decay).max(1.0) as i64;
            tend.balance = (tend.balance + gain).min(0);
        }
    }
}

/// Player TEND interaction: press T near an NPC to exchange services.
pub fn player_tend_interaction_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<(&Transform, &mut TendBalance), With<Player>>,
    mut npcs: Query<
        (
            &Transform,
            &CrewNpc,
            &mut TendBalance,
            &mut NpcTrust,
            &mut NoiseEmitter,
        ),
        Without<Player>,
    >,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
) {
    if !keyboard.just_pressed(KeyCode::KeyT) {
        return;
    }

    let Ok((player_tf, mut player_tend)) = player_query.single_mut() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();

    // Interact with the first NPC in range
    for (npc_tf, npc, mut npc_tend, mut npc_trust, _) in &mut npcs {
        let dist = player_pos.distance(npc_tf.translation.truncate());
        if dist >= 60.0 {
            continue;
        }

        if !player_tend.can_spend(SERVICE_TEND_AMOUNT) {
            eprintln!(
                "[economy] Player can't afford service (balance={})",
                player_tend.balance
            );
            break;
        }

        // Player pays NPC for service (healing, info, etc.)
        player_tend.balance -= SERVICE_TEND_AMOUNT;
        npc_tend.balance += SERVICE_TEND_AMOUNT;

        // Trust grows — this is the restorative justice path
        npc_trust.trust = (npc_trust.trust + 0.1).min(1.0);

        let msg = format!(
            "Player exchanges TEND with {} (trust now {:.0}%)",
            npc.name,
            npc_trust.trust * 100.0,
        );
        eprintln!("[economy] {}", msg);
        log.push(time.elapsed_secs(), msg, 0);

        break; // One exchange per keypress
    }
}
