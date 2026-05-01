// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cryptographic DKG extraction ceremony.
//!
//! Fusion Core extraction requires a REAL Feldman-DKG threshold signature.
//! Player + NPCs must physically gather near the core. The actual
//! `feldman_dkg::DkgCeremony` runs live — participant registration,
//! threshold validation, and phase transitions are the real crypto code.
//!
//! If an NPC's trust is too low or stress too high, they don't register
//! as a participant, and the ceremony can't reach threshold.

use bevy::prelude::*;

use symtropy_sim_bridge::{CeremonyPhase, DkgCeremony, DkgConfig, ParticipantId};

use crate::components::{ConsciousnessComp, CrewNpc, FusionCore, NpcTrust, Player};
use crate::resources::{GovernanceLog, LeviathanState, SleepPhase};

/// Distance (pixels) an entity must be from the core to participate.
const CEREMONY_RANGE: f32 = 60.0;

/// Minimum trust score to participate (below = refuses).
const MIN_TRUST_FOR_CEREMONY: f64 = 0.4;

/// Maximum caution (stress proxy) before NPC defects.
const MAX_STRESS_FOR_CEREMONY: f32 = 0.7;

/// Minimum consciousness score to contribute a valid share.
const MIN_CONSCIOUSNESS_FOR_SHARE: f64 = 0.3;

/// DKG ceremony state — wraps the REAL `feldman_dkg::DkgCeremony`.
#[derive(Resource)]
pub struct DkgCeremonyState {
    /// The real DKG ceremony (None when no ceremony is active).
    pub ceremony: Option<DkgCeremony>,
    /// Participants who contributed valid registrations (names for display).
    pub registered_names: Vec<String>,
    /// Participants who defected (names + reason for display).
    pub defected: Vec<(String, String)>,
    /// Total entities in range (including player).
    pub participants_in_range: usize,
    /// Whether the ceremony succeeded (enough valid participants).
    pub ceremony_succeeded: bool,
    /// Ceremony attempt number.
    pub attempt: u32,
    /// Seconds since ceremony started.
    pub elapsed_secs: f32,
    /// Maximum time before ceremony times out.
    pub timeout_secs: f32,
    /// The DKG threshold (t in t-of-n).
    pub threshold: usize,
    /// Total expected participants (n in t-of-n).
    pub total_expected: usize,
    /// Current ceremony phase (for display).
    pub phase_display: String,
}

impl Default for DkgCeremonyState {
    fn default() -> Self {
        Self {
            ceremony: None,
            registered_names: Vec::new(),
            defected: Vec::new(),
            participants_in_range: 0,
            ceremony_succeeded: false,
            attempt: 0,
            elapsed_secs: 0.0,
            timeout_secs: 15.0,
            threshold: 3,
            total_expected: 4, // Player + 3 NPCs
            phase_display: "inactive".to_string(),
        }
    }
}

/// DKG ceremony system — manages the REAL threshold signature extraction.
///
/// When player presses E near the Fusion Core:
/// 1. Creates a real `DkgConfig` with t=3, n=4
/// 2. Creates a real `DkgCeremony`
/// 3. Registers participants (player + qualifying NPCs) via `add_participant()`
/// 4. If enough register → ceremony auto-transitions to Dealing phase → success
/// 5. If not enough → ceremony stays in Registration → player sees who defected
pub fn dkg_ceremony_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    player_query: Query<&Transform, With<Player>>,
    npcs: Query<(&Transform, &CrewNpc, &ConsciousnessComp, &NpcTrust)>,
    mut core_query: Query<(&Transform, &mut FusionCore)>,
    leviathan: Res<LeviathanState>,
    mut state: ResMut<DkgCeremonyState>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();

    let Some((core_tf, mut core)) = core_query.iter_mut().next() else {
        return;
    };
    let core_pos = core_tf.translation.truncate();
    let player_near_core = player_pos.distance(core_pos) < CEREMONY_RANGE;

    // Already succeeded — nothing to do
    if state.ceremony_succeeded {
        return;
    }

    // Start ceremony when player presses E near core and no ceremony active
    if keyboard.just_pressed(KeyCode::KeyE) && player_near_core && state.ceremony.is_none() {
        state.attempt += 1;
        state.registered_names.clear();
        state.defected.clear();
        state.elapsed_secs = 0.0;
        state.ceremony_succeeded = false;

        // Count qualifying NPCs to determine n
        let mut qualifying = 1usize; // Player always qualifies
        let leviathan_pressure = match leviathan.phase {
            SleepPhase::Dormant => 0.0,
            SleepPhase::Stirring => 0.2,
            SleepPhase::Awake => 0.5,
            SleepPhase::Hunting => 0.8,
        };

        for (npc_tf, npc, consciousness, trust) in &npcs {
            let dist = npc_tf.translation.truncate().distance(core_pos);
            if dist < CEREMONY_RANGE {
                let stress = npc.caution + leviathan_pressure;
                if trust.trust >= MIN_TRUST_FOR_CEREMONY
                    && stress < MAX_STRESS_FOR_CEREMONY
                    && consciousness.combined_score() >= MIN_CONSCIOUSNESS_FOR_SHARE
                {
                    qualifying += 1;
                }
            }
        }

        // Create REAL DkgConfig — t=3, n=max(qualifying, threshold)
        let n = qualifying.max(state.threshold);
        let config = match DkgConfig::new(state.threshold, n) {
            Ok(c) => c,
            Err(e) => {
                let msg = format!(
                    "DKG config failed: {:?} (need {} participants, have {})",
                    e, state.threshold, qualifying
                );
                eprintln!("[dkg] {}", msg);
                log.push(time.elapsed_secs(), msg, 1);
                return;
            }
        };

        // Create REAL DkgCeremony
        let now_secs = time.elapsed_secs() as u64;
        let mut ceremony = DkgCeremony::new(config, now_secs);

        // Register player as participant 1
        if let Err(e) = ceremony.add_participant(ParticipantId(1), now_secs) {
            let msg = format!("DKG: Player registration failed: {:?}", e);
            eprintln!("[dkg] {}", msg);
            log.push(time.elapsed_secs(), msg, 2);
            return;
        }
        state.registered_names.push("Player".to_string());

        // Register qualifying NPCs
        let mut next_id = 2u32;
        for (npc_tf, npc, consciousness, trust) in &npcs {
            let dist = npc_tf.translation.truncate().distance(core_pos);
            if dist >= CEREMONY_RANGE {
                continue;
            }

            state.participants_in_range += 1;
            let stress = npc.caution + leviathan_pressure;

            if trust.trust < MIN_TRUST_FOR_CEREMONY {
                state
                    .defected
                    .push((npc.name.clone(), "trust too low".into()));
            } else if stress >= MAX_STRESS_FOR_CEREMONY {
                state
                    .defected
                    .push((npc.name.clone(), "too stressed".into()));
            } else if consciousness.combined_score() < MIN_CONSCIOUSNESS_FOR_SHARE {
                state
                    .defected
                    .push((npc.name.clone(), "below Participant tier".into()));
            } else if next_id <= n as u32 {
                // Register this NPC with the REAL ceremony
                match ceremony.add_participant(ParticipantId(next_id), now_secs) {
                    Ok(()) => {
                        state.registered_names.push(npc.name.clone());
                        next_id += 1;
                    }
                    Err(e) => {
                        state
                            .defected
                            .push((npc.name.clone(), format!("DKG error: {:?}", e)));
                    }
                }
            }
        }

        // Check ceremony phase — if all participants registered, it auto-transitions
        let phase = ceremony.phase();
        state.phase_display = format!("{:?}", phase);

        let registered = state.registered_names.len();
        let msg = format!(
            "DKG Ceremony #{}: {}/{} registered, phase={:?}",
            state.attempt, registered, n, phase
        );
        eprintln!("[dkg] {}", msg);
        log.push(time.elapsed_secs(), msg, 0);

        // If ceremony auto-transitioned past Registration → threshold met
        if phase != CeremonyPhase::Registration {
            state.ceremony_succeeded = true;
            core.extraction_progress = 0.5;
            core.being_extracted = true;

            let msg = format!(
                "DKG SUCCEEDED! {}-of-{} threshold met. Ceremony advanced to {:?}. Core unlocked!",
                state.threshold, n, phase,
            );
            eprintln!("[dkg] {}", msg);
            log.push(time.elapsed_secs(), msg, 0);
        } else {
            // Not enough participants — ceremony stays in Registration
            let msg = format!(
                "DKG FAILED: Only {}/{} participants. {} defected. Use TEND to rebuild trust!",
                registered,
                state.threshold,
                state.defected.len(),
            );
            eprintln!("[dkg] {}", msg);
            log.push(time.elapsed_secs(), msg, 2);

            for (name, reason) in &state.defected {
                eprintln!("[dkg]   {} defected: {}", name, reason);
            }
        }

        state.ceremony = Some(ceremony);
    }

    // Tick timeout for active ceremonies
    if state.ceremony.is_some() && !state.ceremony_succeeded {
        state.elapsed_secs += time.delta_secs();
        if state.elapsed_secs > state.timeout_secs {
            state.ceremony = None;
            let msg = "DKG Ceremony TIMED OUT".to_string();
            log.push(time.elapsed_secs(), msg, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_dkg_config_validates_threshold() {
        // t=3, n=4 should be valid
        assert!(DkgConfig::new(3, 4).is_ok());

        // t=0 should fail
        assert!(DkgConfig::new(0, 4).is_err());

        // t > n should fail
        assert!(DkgConfig::new(5, 4).is_err());

        // n=1 should fail (need at least 2)
        assert!(DkgConfig::new(1, 1).is_err());
    }

    #[test]
    fn real_dkg_ceremony_registration() {
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        // Should be in Registration phase
        assert_eq!(ceremony.phase(), CeremonyPhase::Registration);

        // Register 2 of 3 participants
        ceremony.add_participant(ParticipantId(1), 0).unwrap();
        assert_eq!(ceremony.phase(), CeremonyPhase::Registration);

        ceremony.add_participant(ParticipantId(2), 0).unwrap();
        assert_eq!(ceremony.phase(), CeremonyPhase::Registration);

        // Register 3rd → should auto-transition to Dealing
        ceremony.add_participant(ParticipantId(3), 0).unwrap();
        assert_eq!(ceremony.phase(), CeremonyPhase::Dealing);
    }

    #[test]
    fn dkg_rejects_duplicate_participant() {
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        ceremony.add_participant(ParticipantId(1), 0).unwrap();
        assert!(ceremony.add_participant(ParticipantId(1), 0).is_err());
    }
}
