// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governance systems: proposals, Phi-weighted voting, veto/override, oppression detection.
//!
//! NPCs generate governance proposals when their FEP free-energy is high (surprise).
//! The player and NPCs vote with Phi-weighted ballots. Guardians can veto but face
//! an 80% override threshold — creating a Nash equilibrium where rational Guardians
//! self-limit. Oppression detection monitors the tier distribution and escalates to
//! constitutional crisis if a majority lacks governance access.

use bevy::prelude::*;

use crate::components::{ConsciousnessComp, CrewNpc, NoiseEmitter, Player};
use crate::resources::{GamePhase, GovernanceLog};
use symtropy_sim_bridge::{ActiveProposal, GovernanceState};

/// Proposal descriptions that NPCs can generate, mapped to game effects.
const PROPOSALS: &[(&str, u8)] = &[
    // (description, tier: 0=Basic, 1=Major, 2=Constitutional)
    ("Lower Leviathan noise threshold by 10%", 0),
    ("Redistribute scavenged resources equally", 0),
    ("Open sealed corridor for faster extraction", 0),
    ("Double TEND credit limits during emergency", 1),
    ("Reduce extraction noise penalty by half", 1),
    ("Grant emergency powers to highest-Phi agent", 1),
    ("Change voting weights to quadratic", 2),
    ("Remove Guardian veto power", 2),
];

/// How often (in seconds) NPCs consider proposing. Prevents spam.
const PROPOSAL_COOLDOWN_SECS: f32 = 15.0;

/// Oppression index threshold that triggers visual/audio warning.
const OPPRESSION_WARNING: f64 = 0.3;

/// Oppression index threshold that triggers constitutional crisis.
const OPPRESSION_CRISIS: f64 = 0.5;

/// NPC proposal generation: high FEP surprise → governance proposals.
pub fn governance_proposal_system(
    npcs: Query<(&CrewNpc, &ConsciousnessComp)>,
    proposal: Res<ActiveProposal>,
    gov: Res<GovernanceState>,
    time: Res<Time>,
    mut cooldown: Local<f32>,
) {
    // Don't propose if there's already an active proposal
    if proposal.active {
        return;
    }

    *cooldown -= time.delta_secs();
    if *cooldown > 0.0 {
        return;
    }

    // Find the NPC with highest free-energy (most surprised / most need for change)
    let mut best_proposer: Option<(&str, f64, f64)> = None; // (name, surprise, phi)
    for (npc, consciousness) in &npcs {
        let surprise = npc.fep.current_free_energy();
        if let Some((_, best_surprise, _)) = best_proposer {
            if surprise > best_surprise {
                best_proposer = Some((&npc.name, surprise, consciousness.combined_score()));
            }
        } else {
            best_proposer = Some((&npc.name, surprise, consciousness.combined_score()));
        }
    }

    let Some((name, surprise, phi)) = best_proposer else {
        return;
    };

    // Only propose if surprise is high enough (FEP-driven governance need)
    if surprise < 0.4 {
        return;
    }

    // Choose proposal tier based on surprise level and oppression
    let tier = if gov.oppression_index() > OPPRESSION_CRISIS {
        2 // Constitutional crisis → constitutional proposal
    } else if surprise > 0.7 {
        1 // High surprise → major proposal
    } else {
        0 // Moderate surprise → basic proposal
    };

    // Check if this NPC's phi meets the tier threshold
    let min_phi = match tier {
        0 => 0.3,
        1 => 0.4,
        _ => 0.6,
    };
    if phi < min_phi {
        return;
    }

    // Select a proposal from the tier
    let candidates: Vec<_> = PROPOSALS.iter().filter(|(_, t)| *t == tier).collect();
    if candidates.is_empty() {
        return;
    }
    // Use sim tick as pseudo-random selector
    let idx = gov.sim_tick as usize % candidates.len();
    let (desc, _) = candidates[idx];

    eprintln!(
        "[governance] {} proposes (tier {}): \"{}\" (surprise={:.2}, phi={:.2})",
        name, tier, desc, surprise, phi
    );

    *cooldown = PROPOSAL_COOLDOWN_SECS;
}

/// NPC voting system: each NPC casts Phi-weighted votes on active proposals.
pub fn governance_voting_system(
    npcs: Query<(&CrewNpc, &ConsciousnessComp)>,
    player_consciousness: Query<&ConsciousnessComp, With<Player>>,
    mut proposal: ResMut<ActiveProposal>,
    gov: Res<GovernanceState>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
) {
    if !proposal.active {
        return;
    }

    // NPCs vote based on their consciousness state
    for (npc, consciousness) in &npcs {
        // Skip if already voted
        if proposal.votes.iter().any(|(n, _, _)| n == &npc.name) {
            continue;
        }

        // Vote decision: NPCs with high care_activation approve pro-social proposals
        let care = consciousness.sim_dimensions[3]; // care_activation
        let coherence = consciousness.sim_dimensions[2]; // coherence
                                                         // High-care NPCs approve; low-coherence NPCs abstain (don't vote)
        if coherence < 0.3 {
            continue; // Too confused to vote
        }

        let approve = care > 0.4; // Care-driven voting
        proposal.cast_vote(&npc.name, consciousness.combined_score(), approve);

        eprintln!(
            "[governance] {} votes {} (phi={:.2}, care={:.2})",
            npc.name,
            if approve { "YES" } else { "NO" },
            consciousness.combined_score(),
            care,
        );
    }

    // Tick proposal countdown
    let expired = proposal.tick();
    if expired {
        let passed = proposal.passed();
        let msg = if passed {
            format!(
                "Proposal PASSED: \"{}\" (approval={:.0}%)",
                proposal.description,
                proposal.approval_ratio() * 100.0
            )
        } else {
            format!(
                "Proposal FAILED: \"{}\" (approval={:.0}%)",
                proposal.description,
                proposal.approval_ratio() * 100.0
            )
        };
        eprintln!("[governance] {}", msg);
        log.push(time.elapsed_secs(), msg, if passed { 0 } else { 1 });
        proposal.clear();
    }
}

/// Veto/override system: Guardian NPCs can veto, community can override at 80%.
pub fn veto_override_system(
    npcs: Query<(&CrewNpc, &ConsciousnessComp)>,
    mut proposal: ResMut<ActiveProposal>,
    mut gov: ResMut<GovernanceState>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
) {
    if !proposal.active || proposal.vetoed {
        // If already vetoed, check for override votes
        if proposal.vetoed && proposal.active {
            for (npc, consciousness) in &npcs {
                if proposal
                    .override_votes
                    .iter()
                    .any(|(n, _, _)| n == &npc.name)
                {
                    continue;
                }
                // Most NPCs support override (democracy)
                let approve = consciousness.sim_dimensions[3] > 0.3; // care threshold
                proposal.cast_override_vote(&npc.name, consciousness.combined_score(), approve);
            }

            if proposal.override_succeeded() {
                let msg = "VETO OVERRIDDEN by 80%+ supermajority!".to_string();
                eprintln!("[governance] {}", msg);
                log.push(time.elapsed_secs(), msg, 2);
                gov.governance.veto_override_count += 1;
            }
        }
        return;
    }

    // Check if any Guardian NPC wants to veto
    for (npc, consciousness) in &npcs {
        if consciousness.tier() < 4 {
            continue; // Only Guardians (tier 4) can veto
        }

        // Guardian vetoes if they strongly disagree (low care + high caution)
        let care = consciousness.sim_dimensions[3];
        let harmony = consciousness.sim_dimensions[4];
        if care < 0.3 && harmony < 0.4 {
            if proposal.veto(&npc.name, consciousness.combined_score()) {
                let msg = format!(
                    "{} VETOES proposal (Guardian, phi={:.2})",
                    npc.name,
                    consciousness.combined_score()
                );
                eprintln!("[governance] {}", msg);
                log.push(time.elapsed_secs(), msg, 2);
                gov.governance.veto_count += 1;
            }
        }
    }
}

/// Oppression detection: monitors tier distribution and flags crises.
pub fn oppression_detection_system(
    npcs: Query<&ConsciousnessComp, With<CrewNpc>>,
    player_consciousness: Query<&ConsciousnessComp, With<Player>>,
    mut gov: ResMut<GovernanceState>,
    mut log: ResMut<GovernanceLog>,
    mut noise_sources: Query<&mut NoiseEmitter>,
    time: Res<Time>,
    mut consecutive_ticks: Local<u32>,
    mut warned: Local<bool>,
) {
    // Count agents per tier
    let mut tier_counts = [0u32; 5];
    let mut total = 0u32;

    for cp in &npcs {
        tier_counts[cp.tier() as usize] += 1;
        total += 1;
    }
    if let Ok(player_cp) = player_consciousness.single() {
        tier_counts[player_cp.tier() as usize] += 1;
        total += 1;
    }

    if total == 0 {
        return;
    }

    // Compute oppression index (same formula as multiworld-sim)
    let observer_frac = tier_counts[0] as f64 / total as f64;
    let guardian_frac = tier_counts[4] as f64 / total as f64;
    let oppression = if guardian_frac >= 0.05 {
        0.0
    } else {
        observer_frac * (1.0 - guardian_frac)
    };

    gov.governance.oppression_index = oppression;

    // Track consecutive high-oppression ticks
    if oppression > OPPRESSION_CRISIS {
        *consecutive_ticks += 1;
    } else {
        *consecutive_ticks = 0;
        *warned = false;
    }

    // Warning at moderate oppression
    if oppression > OPPRESSION_WARNING && !*warned {
        let msg = format!(
            "WARNING: Oppression index {:.2} — consciousness tiers are imbalanced",
            oppression
        );
        eprintln!("[governance] {}", msg);
        log.push(time.elapsed_secs(), msg, 1);
        *warned = true;
    }

    // Constitutional crisis at sustained high oppression (12 ticks = 60 seconds)
    if *consecutive_ticks >= 12 {
        let msg =
            "CONSTITUTIONAL CRISIS: Sustained oppression detected! Gating thresholds lowered."
                .to_string();
        eprintln!("[governance] {}", msg);
        log.push(time.elapsed_secs(), msg, 2);

        // Add noise — social instability wakes the Leviathan
        for mut noise in &mut noise_sources {
            noise.level += 0.15;
        }

        *consecutive_ticks = 0; // Reset after crisis action
    }
}

/// Update consciousness profiles based on gameplay state.
/// Consciousness evolves through gameplay — care increases through cooperation,
/// meta-awareness through governance participation, coherence through successful actions.
pub fn consciousness_evolution_system(
    mut npcs: Query<(&CrewNpc, &mut ConsciousnessComp)>,
    mut player_consciousness: Query<&mut ConsciousnessComp, With<Player>>,
    gov: Res<GovernanceState>,
    proposal: Res<ActiveProposal>,
    time: Res<Time>,
) {
    let dt = time.delta_secs() as f64;

    // NPCs evolve consciousness based on gameplay context
    for (npc, mut cp) in &mut npcs {
        let surprise = npc.fep.current_free_energy();

        // Meta-awareness grows when participating in governance
        if proposal.active {
            cp.sim_dimensions[1] += 0.01 * dt; // meta_awareness
        }

        // Care activation grows when not under extreme threat
        if surprise < 0.5 {
            cp.sim_dimensions[3] += 0.005 * dt; // care_activation
        } else {
            cp.sim_dimensions[3] -= 0.01 * dt; // threat suppresses care
        }

        // Harmonic alignment tracks governance stability
        cp.sim_dimensions[4] = cp.sim_dimensions[4] * 0.99 + gov.stability() * 0.01;

        // Epistemic confidence grows slowly through survival
        cp.sim_dimensions[5] += 0.002 * dt;

        // Clamp all dimensions
        for d in &mut cp.sim_dimensions {
            *d = d.clamp(0.0, 1.0);
        }
        cp.sync_engagement_from_sim();
    }

    // Player consciousness evolves too
    if let Ok(mut cp) = player_consciousness.single_mut() {
        // Player governance participation boosts meta-awareness
        if proposal.active {
            cp.sim_dimensions[1] += 0.01 * dt;
        }
        // Stability affects harmonic alignment
        cp.sim_dimensions[4] = cp.sim_dimensions[4] * 0.99 + gov.stability() * 0.01;

        for d in &mut cp.sim_dimensions {
            *d = d.clamp(0.0, 1.0);
        }
        cp.sync_engagement_from_sim();
    }
}
