// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Simulated family life: the house feels alive even in mock mode.
//!
//! Timer-driven events make the mock family do things on their own:
//! - Presence changes (Ember comes home, Sage goes to sleep)
//! - Gratitude expressions between members
//! - Bond decay over time
//! - Care task completions
//!
//! Each event emits a toast notification so you feel the shared space.

use leptos::prelude::*;
use gloo_timers::callback::Interval;

use crate::hearth_context::use_hearth;
use mycelix_leptos_core::{use_toasts, ToastKind};
use crate::hearth_context::{member_name, mock_now};
use hearth_leptos_types::*;

/// Phrases the simulated family members say as gratitude.
const GRATITUDE_PHRASES: &[(&str, &str, &str)] = &[
    ("agent_sage", "agent_river", "thank you for the quiet morning together"),
    ("agent_wren", "agent_sage", "you make the best stories at bedtime"),
    ("agent_river", "agent_rowan", "the garden is thriving because of you"),
    ("agent_ember", "agent_wren", "you always make me laugh"),
    ("agent_rowan", "agent_river", "your patience holds us all together"),
    ("agent_sage", "agent_ember", "watching you grow is a gift"),
    ("agent_wren", "agent_river", "thank you for always being here"),
    ("agent_river", "agent_sage", "your wisdom keeps the hearth steady"),
];

/// Presence change events.
const PRESENCE_EVENTS: &[(&str, &str)] = &[
    ("agent_ember", "Home"),
    ("agent_sage", "Home"),
    ("agent_ember", "Away"),
    ("agent_sage", "Working"),
    ("agent_rowan", "Away"),
    ("agent_rowan", "Home"),
    ("agent_wren", "Home"),
];

/// Start the simulated life system. Only runs in mock mode.
pub fn start_simulated_life() {
    let hearth = use_hearth();
    let toasts = use_toasts();

    // Gratitude: every 25-35 seconds, a family member expresses gratitude
    let gratitude_idx = std::cell::Cell::new(0usize);
    let toasts_g = toasts.clone();
    let _gratitude_timer = Interval::new(28_000, move || {
        let toasts = toasts_g.clone();
        let idx = gratitude_idx.get() % GRATITUDE_PHRASES.len();
        gratitude_idx.set(idx + 1);
        let (from, to, msg) = GRATITUDE_PHRASES[idx];

        let members = hearth.members.get_untracked();
        let from_name = member_name(&members, from);
        let to_name = member_name(&members, to);

        hearth.gratitude.update(|g| {
            g.insert(0, GratitudeExpressionView {
                hash: format!("sim_grat_{}", idx),
                from_agent: from.into(),
                to_agent: to.into(),
                message: msg.into(),
                gratitude_type: GratitudeType::Appreciation,
                visibility: HearthVisibility::AllMembers,
                created_at: mock_now(),
            });
            // Keep list manageable
            if g.len() > 20 { g.pop(); }
        });

        toasts.push(
            format!("{from_name} → {to_name}: “{msg}”"),
            ToastKind::Custom("gratitude".into()),
        );
    });
    std::mem::forget(_gratitude_timer);

    // Presence: every 20 seconds, someone changes status
    let presence_idx = std::cell::Cell::new(0usize);
    let toasts_p = toasts.clone();
    let _presence_timer = Interval::new(20_000, move || {
        let toasts = toasts_p.clone();
        let idx = presence_idx.get() % PRESENCE_EVENTS.len();
        presence_idx.set(idx + 1);
        let (agent, status_str) = PRESENCE_EVENTS[idx];

        let new_status = match status_str {
            "Home" => PresenceStatusType::Home,
            "Away" => PresenceStatusType::Away,
            "Working" => PresenceStatusType::Working,
            "Sleeping" => PresenceStatusType::Sleeping,
            _ => PresenceStatusType::Away,
        };

        let members = hearth.members.get_untracked();
        let name = member_name(&members, agent);

        hearth.presence.update(|plist| {
            if let Some(p) = plist.iter_mut().find(|p| p.agent == agent) {
                p.status = new_status;
                p.updated_at = mock_now();
            }
        });

        let verb = match new_status {
            PresenceStatusType::Home => "arrived home",
            PresenceStatusType::Away => "stepped out",
            PresenceStatusType::Working => "started working",
            PresenceStatusType::Sleeping => "fell asleep",
            PresenceStatusType::DoNotDisturb => "needs quiet",
        };

        toasts.push(format!("{name} {verb}"), ToastKind::Custom("presence".into()));
    });
    std::mem::forget(_presence_timer);

    // Bond decay: every 45 seconds, one bond loses a little strength
    let decay_idx = std::cell::Cell::new(0usize);
    let toasts_d = toasts.clone();
    let _decay_timer = Interval::new(45_000, move || {
        let toasts = toasts_d.clone();
        let bonds = hearth.bonds.get_untracked();
        if bonds.is_empty() { return; }
        let idx = decay_idx.get() % bonds.len();
        decay_idx.set(idx + 1);

        let bond_hash = bonds[idx].hash.clone();
        let old_strength = bonds[idx].strength_bp;

        // Decay by 200bp (simulating time passing)
        if old_strength > BOND_MIN + 200 {
            let members = hearth.members.get_untracked();
            let name_a = member_name(&members, &bonds[idx].member_a);
            let name_b = member_name(&members, &bonds[idx].member_b);

            hearth.bonds.update(|bs| {
                if let Some(b) = bs.iter_mut().find(|b| b.hash == bond_hash) {
                    b.strength_bp = b.strength_bp.saturating_sub(200).max(BOND_MIN);
                }
            });

            let new_strength = old_strength - 200;
            if new_strength < 4000 && old_strength >= 4000 {
                // Crossed into neglected territory
                toasts.push(
                    format!("the bond between {name_a} and {name_b} needs tending"),
                    ToastKind::Custom("bond".into()),
                );
            }
        }
    });
    std::mem::forget(_decay_timer);
}
