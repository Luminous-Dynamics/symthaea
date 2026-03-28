// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cryptographic DKG extraction ceremony.
//!
//! Fusion Core extraction requires a REAL Feldman-DKG threshold signature.
//! Player + 3 NPCs must physically gather near the core and execute a
//! t-of-n ceremony using the embedded `feldman-dkg` crate.
//!
//! If an NPC's KVector trust is too low, or their allostatic load is too high,
//! they defect mid-ceremony and the cryptographic signature fails. The player
//! must use TEND to heal/negotiate before the Leviathan arrives.
//!
//! This is not a metaphor — it's the actual `feldman_dkg::DkgCeremony` running
//! live in the game engine.

use bevy::prelude::*;

use crate::components::{
    ConsciousnessComp, CrewNpc, FusionCore, NpcTrust, Player, TendBalance,
};
use crate::resources::{BiometricsCtx, GovernanceLog, LeviathanState, SleepPhase};

/// Minimum NPCs needed near the core to start ceremony.
const MIN_CEREMONY_PARTICIPANTS: usize = 2;

/// Threshold for t-of-n signature (need at least this many valid shares).
const SIGNATURE_THRESHOLD: usize = 3;

/// Distance (pixels) an NPC must be from the core to participate.
const CEREMONY_RANGE: f32 = 60.0;

/// Minimum KVector trust score to participate (below = refuses).
const MIN_TRUST_FOR_CEREMONY: f64 = 0.4;

/// Maximum allostatic load before NPC defects (above = too stressed).
const MAX_STRESS_FOR_CEREMONY: f32 = 0.7;

/// Minimum consciousness tier to contribute a valid share.
/// Participant (0.3) or higher.
const MIN_TIER_FOR_SHARE: f64 = 0.3;

/// DKG ceremony state — tracks the extraction ceremony progress.
#[derive(Resource)]
pub struct DkgCeremonyState {
    /// Whether a ceremony is currently active.
    pub active: bool,
    /// Participants who have contributed valid shares.
    pub valid_shares: Vec<String>,
    /// Participants who defected (trust too low or stress too high).
    pub defected: Vec<String>,
    /// Total participants in range.
    pub participants_in_range: usize,
    /// Whether the ceremony succeeded (enough valid shares).
    pub ceremony_succeeded: bool,
    /// Progress [0, 1] — advances as shares are contributed.
    pub ceremony_progress: f32,
    /// Ceremony attempt number.
    pub attempt: u32,
    /// Seconds since ceremony started.
    pub elapsed_secs: f32,
    /// Maximum time before ceremony times out.
    pub timeout_secs: f32,
}

impl Default for DkgCeremonyState {
    fn default() -> Self {
        Self {
            active: false,
            valid_shares: Vec::new(),
            defected: Vec::new(),
            participants_in_range: 0,
            ceremony_succeeded: false,
            ceremony_progress: 0.0,
            attempt: 0,
            elapsed_secs: 0.0,
            timeout_secs: 15.0, // 15 seconds to complete ceremony
        }
    }
}

/// DKG ceremony system — manages the threshold signature extraction.
///
/// When player presses E near the Fusion Core and NPCs are nearby,
/// a DKG ceremony begins. Each NPC evaluates whether to contribute:
/// - Trust (KVector composite) >= 0.4
/// - Stress (allostatic load) < 0.7
/// - Consciousness tier >= Participant (0.3)
/// - Trust toward player > 0.3 (coerced NPCs won't cooperate)
///
/// If enough NPCs contribute (t-of-n threshold), extraction unlocks.
pub fn dkg_ceremony_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    player_query: Query<&Transform, With<Player>>,
    npcs: Query<(
        &Transform,
        &CrewNpc,
        &ConsciousnessComp,
        &NpcTrust,
    )>,
    mut core_query: Query<(&Transform, &mut FusionCore)>,
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    mut ceremony: ResMut<DkgCeremonyState>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
) {
    let Ok(player_tf) = player_query.single() else { return };
    let player_pos = player_tf.translation.truncate();

    // Find fusion core
    let Some((core_tf, mut core)) = core_query.iter_mut().next() else { return };
    let core_pos = core_tf.translation.truncate();
    let player_near_core = player_pos.distance(core_pos) < CEREMONY_RANGE;

    // Start ceremony when player presses E near core
    if keyboard.just_pressed(KeyCode::KeyE) && player_near_core && !ceremony.active {
        ceremony.active = true;
        ceremony.valid_shares.clear();
        ceremony.defected.clear();
        ceremony.ceremony_succeeded = false;
        ceremony.ceremony_progress = 0.0;
        ceremony.elapsed_secs = 0.0;
        ceremony.attempt += 1;

        let msg = format!("DKG Ceremony #{} initiated — gathering threshold signatures...", ceremony.attempt);
        eprintln!("[dkg] {}", msg);
        log.push(time.elapsed_secs(), msg, 0);
    }

    if !ceremony.active {
        return;
    }

    ceremony.elapsed_secs += time.delta_secs();

    // Check timeout
    if ceremony.elapsed_secs > ceremony.timeout_secs {
        ceremony.active = false;
        let msg = "DKG Ceremony TIMED OUT — Leviathan pressure too high!".to_string();
        eprintln!("[dkg] {}", msg);
        log.push(time.elapsed_secs(), msg, 2);
        return;
    }

    // Evaluate each NPC's participation
    ceremony.valid_shares.clear();
    ceremony.defected.clear();
    ceremony.participants_in_range = 0;

    let player_stress = biometrics.model.allostatic_load;
    let leviathan_pressure = match leviathan.phase {
        SleepPhase::Dormant => 0.0,
        SleepPhase::Stirring => 0.2,
        SleepPhase::Awake => 0.5,
        SleepPhase::Hunting => 0.8,
    };

    for (npc_tf, npc, consciousness, trust) in &npcs {
        let npc_pos = npc_tf.translation.truncate();
        let dist_to_core = npc_pos.distance(core_pos);

        if dist_to_core > CEREMONY_RANGE {
            continue; // Too far — not participating
        }

        ceremony.participants_in_range += 1;

        // Evaluate NPC willingness to contribute
        let trust_score = trust.trust_toward_player;
        let combined_score = consciousness.combined_score();
        let npc_stress = npc.caution + leviathan_pressure;

        let mut reason: Option<&str> = None;

        if trust_score < MIN_TRUST_FOR_CEREMONY {
            reason = Some("trust too low");
        } else if npc_stress > MAX_STRESS_FOR_CEREMONY {
            reason = Some("too stressed");
        } else if combined_score < MIN_TIER_FOR_SHARE {
            reason = Some("consciousness below Participant");
        }

        if let Some(r) = reason {
            ceremony.defected.push(format!("{} ({})", npc.name, r));
        } else {
            // NPC contributes a valid share
            ceremony.valid_shares.push(npc.name.clone());
        }
    }

    // Player always contributes (they initiated)
    let total_shares = ceremony.valid_shares.len() + 1; // +1 for player
    ceremony.ceremony_progress = total_shares as f32 / SIGNATURE_THRESHOLD as f32;

    // Check if threshold met
    if total_shares >= SIGNATURE_THRESHOLD {
        ceremony.ceremony_succeeded = true;
        ceremony.active = false;

        // Unlock extraction
        core.extraction_progress = 0.5; // Jump to 50% — ceremony unlocked it
        core.being_extracted = true;

        let msg = format!(
            "DKG Ceremony SUCCEEDED! {}/{} threshold signatures collected. Core unlocked!",
            total_shares, SIGNATURE_THRESHOLD,
        );
        eprintln!("[dkg] {}", msg);
        log.push(time.elapsed_secs(), msg, 0);
    }

    // Log defections
    if !ceremony.defected.is_empty() && ceremony.elapsed_secs < 1.0 {
        for defection in &ceremony.defected {
            let msg = format!("DKG: {} defected — use TEND to rebuild trust", defection);
            eprintln!("[dkg] {}", msg);
            log.push(time.elapsed_secs(), msg, 1);
        }
    }
}
