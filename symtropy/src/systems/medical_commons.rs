// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Medical Commons — data pooling, consent, and healing.
//!
//! The Medical Bay is where crew members share biometric data via Federated Learning.
//! Three scenarios emerge through gameplay:
//!
//! 1. **Hoarded data** (no sharing): 60% healing rate — each NPC treats independently
//! 2. **Voluntary pooling** (consent + FL): 85% healing + TEND dividends — collective intelligence
//! 3. **Coerced sharing** (forced access): FL works but trust decays, NPCs defect from DKG
//!
//! Players discover through gameplay that voluntary pooling is the only stable equilibrium.

use bevy::prelude::*;

use super::fl_simulation::FlPool;
use crate::components::{ConsciousnessComp, CrewNpc, NpcTrust, Player, TendBalance};
use crate::resources::GovernanceLog;

/// Healing rates based on data sharing model.
const HEALING_RATE_HOARDED: f32 = 0.60;
const HEALING_RATE_POOLED: f32 = 0.85;
const HEALING_RATE_COERCED: f32 = 0.75; // FL works but degraded

/// TEND dividend per FL round for data contributors.
const TEND_DIVIDEND_PER_ROUND: i64 = 2;

/// Medical Bay state — tracks data sharing and healing.
#[derive(Resource)]
pub struct MedicalBayState {
    /// How many NPCs have voluntarily consented to share data.
    pub consenting_npcs: usize,
    /// Total crew size.
    pub total_crew: usize,
    /// Current effective healing rate [0, 1].
    pub healing_rate: f32,
    /// Whether any coercion has occurred (player accessed data without consent).
    pub coercion_detected: bool,
    /// Data sharing model (for display).
    pub sharing_model: SharingModel,
    /// Total TEND dividends distributed.
    pub total_dividends: i64,
    /// Attribution chain: (NPC name, contribution type, TEND earned).
    pub attribution_chain: Vec<(String, String, i64)>,
}

/// Current data sharing model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharingModel {
    /// No one shares — individual treatment only.
    Hoarded,
    /// Voluntary consent-based FL pooling.
    Voluntary,
    /// Player forced data access (trust degrading).
    Coerced,
}

impl Default for MedicalBayState {
    fn default() -> Self {
        Self {
            consenting_npcs: 0,
            total_crew: 0,
            healing_rate: HEALING_RATE_HOARDED,
            coercion_detected: false,
            sharing_model: SharingModel::Hoarded,
            total_dividends: 0,
            attribution_chain: Vec::new(),
        }
    }
}

/// Medical commons system — compute healing rate from data sharing state.
///
/// Runs every few seconds. Counts consenting NPCs (trust > 0.5 implies consent),
/// determines sharing model, and adjusts healing rate accordingly.
pub fn medical_commons_system(
    npcs: Query<(&CrewNpc, &NpcTrust, &ConsciousnessComp)>,
    fl_pool: Res<FlPool>,
    mut medical: ResMut<MedicalBayState>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
    mut timer: Local<f32>,
) {
    *timer += time.delta_secs();
    if *timer < 5.0 {
        return;
    }
    *timer = 0.0;

    let total = npcs.iter().count();
    let consenting = npcs
        .iter()
        .filter(|(_, trust, _)| trust.trust >= 0.5) // Consent = trust threshold
        .count();

    medical.total_crew = total;
    medical.consenting_npcs = consenting;

    let prev_model = medical.sharing_model;

    // Determine sharing model
    if medical.coercion_detected {
        medical.sharing_model = SharingModel::Coerced;
        medical.healing_rate = HEALING_RATE_COERCED * fl_pool.aggregation_quality;
    } else if consenting > 0 && fl_pool.round > 0 {
        medical.sharing_model = SharingModel::Voluntary;
        let consent_ratio = consenting as f32 / total.max(1) as f32;
        // Healing scales with consent ratio and FL quality
        medical.healing_rate = HEALING_RATE_HOARDED
            + (HEALING_RATE_POOLED - HEALING_RATE_HOARDED)
                * consent_ratio
                * fl_pool.aggregation_quality;
    } else {
        medical.sharing_model = SharingModel::Hoarded;
        medical.healing_rate = HEALING_RATE_HOARDED;
    }

    // Log model transitions
    if medical.sharing_model != prev_model {
        let msg = match medical.sharing_model {
            SharingModel::Hoarded => format!(
                "Medical Commons: HOARDED — {:.0}% healing. No data shared.",
                medical.healing_rate * 100.0,
            ),
            SharingModel::Voluntary => format!(
                "Medical Commons: VOLUNTARY — {:.0}% healing! {}/{} crew consent. FL quality={:.0}%",
                medical.healing_rate * 100.0, consenting, total,
                fl_pool.aggregation_quality * 100.0,
            ),
            SharingModel::Coerced => format!(
                "Medical Commons: COERCED — {:.0}% healing. Trust degrading. NPCs may defect!",
                medical.healing_rate * 100.0,
            ),
        };
        eprintln!("[medical] {}", msg);
        log.push(
            time.elapsed_secs(),
            msg,
            if medical.sharing_model == SharingModel::Coerced {
                2
            } else {
                0
            },
        );
    }
}

/// TEND dividend distribution for data contributors.
/// NPCs who consented receive TEND dividends when FL rounds complete successfully.
pub fn data_dividend_system(
    mut npcs: Query<(&CrewNpc, &NpcTrust, &mut TendBalance)>,
    fl_pool: Res<FlPool>,
    mut medical: ResMut<MedicalBayState>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
    mut last_round: Local<u32>,
) {
    // Only distribute on new FL rounds
    if fl_pool.round == *last_round || fl_pool.round == 0 {
        return;
    }
    *last_round = fl_pool.round;

    // Only distribute if FL quality is good and sharing is voluntary
    if medical.sharing_model != SharingModel::Voluntary || fl_pool.aggregation_quality < 0.5 {
        return;
    }

    let mut distributed = 0;
    for (npc, trust, mut tend) in &mut npcs {
        if trust.trust >= 0.5 {
            // NPC consented — receives dividend
            tend.balance += TEND_DIVIDEND_PER_ROUND;
            medical.total_dividends += TEND_DIVIDEND_PER_ROUND;
            medical.attribution_chain.push((
                npc.name.clone(),
                "FL data contribution".into(),
                TEND_DIVIDEND_PER_ROUND,
            ));
            distributed += 1;
        }
    }

    if distributed > 0 {
        let msg = format!(
            "Data dividends: {} TEND distributed to {} contributors (round {})",
            distributed * TEND_DIVIDEND_PER_ROUND,
            distributed,
            fl_pool.round,
        );
        log.push(time.elapsed_secs(), msg, 0);
    }

    // Trim attribution chain to last 50 entries
    if medical.attribution_chain.len() > 50 {
        let keep_from = medical.attribution_chain.len() - 50;
        medical.attribution_chain.drain(..keep_from);
    }
}

/// Coercion detection — if player accesses NPC data when trust < 0.3,
/// mark as coercion. This degrades the medical commons model.
pub fn coercion_detection_system(
    npcs: Query<(&CrewNpc, &NpcTrust)>,
    mut medical: ResMut<MedicalBayState>,
) {
    // Coercion is detected when any NPC's trust drops very low
    // (indicating the player forced data access via the economy system)
    for (_, trust) in &npcs {
        if trust.trust < 0.2 {
            if !medical.coercion_detected {
                medical.coercion_detected = true;
                eprintln!("[medical] COERCION DETECTED — NPC trust collapsed. Healing degraded.");
            }
            return;
        }
    }

    // Recovery: if all NPCs have trust > 0.4, coercion penalty lifts
    let all_trusting = npcs.iter().all(|(_, trust)| trust.trust >= 0.4);
    if all_trusting && medical.coercion_detected {
        medical.coercion_detected = false;
        eprintln!("[medical] Trust restored — coercion penalty lifted.");
    }
}
