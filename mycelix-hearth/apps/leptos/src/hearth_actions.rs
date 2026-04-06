// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hearth action dispatch: the bridge between UI interactions and data.
//!
//! When connected to a conductor, dispatches real zome calls.
//! When in mock mode, mutates local RwSignal state directly.
//! The UI doesn't know the difference — it reads from the same signals.

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use crate::hearth_context::{use_hearth, member_name, mock_now};
use mycelix_leptos_core::holochain_provider::use_holochain;
use mycelix_leptos_core::{use_toasts, ToastKind};
use hearth_leptos_types::*;

/// Shared action dispatcher.
#[derive(Clone)]
#[allow(dead_code)]
pub struct HearthActions {
    is_mock: bool,
}

pub fn provide_hearth_actions() -> HearthActions {
    let hc = use_holochain();
    let actions = HearthActions {
        is_mock: hc.is_mock(),
    };
    provide_context(actions.clone());
    actions
}

pub fn use_hearth_actions() -> HearthActions {
    expect_context::<HearthActions>()
}

// ============================================================================
// Bond Tending
// ============================================================================

/// Tend a bond: increase strength, update last_tended.
pub fn tend_bond(bond_hash: String) {
    let hc = use_holochain();
    let hearth = use_hearth();
    let toasts = use_toasts();

    if hc.is_mock() {
        // Mock: mutate local state
        let members = hearth.members.get_untracked();
        hearth.bonds.update(|bonds| {
            if let Some(bond) = bonds.iter_mut().find(|b| b.hash == bond_hash) {
                bond.strength_bp = (bond.strength_bp + 500).min(BOND_MAX);
                bond.last_tended = mock_now();
                let name_a = member_name(&members, &bond.member_a);
                let name_b = member_name(&members, &bond.member_b);
                toasts.push(
                    format!("you tended the bond between {name_a} and {name_b}"),
                    ToastKind::Custom("bond".into()),
                );
            }
        });
    } else {
        // Real: call zome, then update signal on success
        let hc = hc.clone();
        spawn_local(async move {
            // NOTE: bond_hash is a String in our types but ActionHash on the wire.
            // The Holochain conductor accepts base64-encoded ActionHash bytes
            // via MessagePack. This will work if the hash string was originally
            // obtained from a conductor response (which it will be in Phase 6).
            #[derive(serde::Serialize)]
            struct TendBondInput {
                bond_hash: String,
                description: String,
                quality_bp: u32,
            }

            match hc.call_zome_default::<TendBondInput, ()>(
                "hearth_kinship",
                "tend_bond",
                &TendBondInput {
                    bond_hash: bond_hash.clone(),
                    description: "tended with care".into(),
                    quality_bp: 500,
                },
            ).await {
                Ok(_) => {
                    // Refresh bonds from conductor
                    if let Ok(bonds) = hc.call_zome_default::<(), Vec<BondView>>(
                        "hearth_kinship", "get_kinship_graph", &()
                    ).await {
                        hearth.bonds.set(bonds);
                    }
                    toasts.push("bond tended", ToastKind::Custom("bond".into()));
                }
                Err(e) => {
                    web_sys::console::log_1(&format!("tend_bond failed: {e}").into());
                    toasts.push("couldn’t tend that bond right now", ToastKind::Custom("bond".into()));
                }
            }
        });
    }
}

// ============================================================================
// Express Gratitude
// ============================================================================

pub fn express_gratitude(to_agent: String, message: String) {
    let hc = use_holochain();
    let hearth = use_hearth();
    let toasts = use_toasts();

    if hc.is_mock() {
        let from = hearth.my_agent.get_untracked();
        let members = hearth.members.get_untracked();
        let from_name = member_name(&members, &from);
        let to_name = member_name(&members, &to_agent);
        let msg_preview = message.clone();

        hearth.gratitude.update(|g| {
            g.insert(0, GratitudeExpressionView {
                hash: format!("grat_user_{}", g.len()),
                from_agent: from,
                to_agent: to_agent.clone(),
                message,
                gratitude_type: GratitudeType::Appreciation,
                visibility: HearthVisibility::AllMembers,
                created_at: mock_now(),
            });
        });

        toasts.push(
            format!("{from_name} → {to_name}: “{msg_preview}”"),
            ToastKind::Custom("gratitude".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct ExpressGratitudeInput {
                hearth_hash: String,
                to_agent: String,
                message: String,
                gratitude_type: GratitudeType,
                visibility: HearthVisibility,
            }

            let hearth_hash = hearth.current_hearth.get_untracked()
                .map(|h| h.hash)
                .unwrap_or_default();

            match hc.call_zome_default::<ExpressGratitudeInput, ()>(
                "hearth_gratitude",
                "express_gratitude",
                &ExpressGratitudeInput {
                    hearth_hash,
                    to_agent,
                    message,
                    gratitude_type: GratitudeType::Appreciation,
                    visibility: HearthVisibility::AllMembers,
                },
            ).await {
                Ok(_) => toasts.push("gratitude expressed", ToastKind::Custom("gratitude".into())),
                Err(e) => {
                    web_sys::console::log_1(&format!("express_gratitude failed: {e}").into());
                }
            }
        });
    }
}

// ============================================================================
// Cast Vote
// ============================================================================

pub fn cast_vote(decision_hash: String, choice: u32, reasoning: Option<String>, weight_bp: u32) {
    let hc = use_holochain();
    let hearth = use_hearth();
    let toasts = use_toasts();

    if hc.is_mock() {
        let my_agent = hearth.my_agent.get_untracked();
        hearth.votes.update(|votes| {
            votes.retain(|v| !(v.decision_hash == decision_hash && v.voter == my_agent));
            votes.push(VoteView {
                decision_hash: decision_hash.clone(),
                voter: my_agent,
                choice,
                weight_bp,
                reasoning,
                created_at: mock_now(),
            });
        });
        toasts.push("vote cast", ToastKind::Custom("decision".into()));
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct CastVoteInput {
                decision_hash: String,
                choice: u32,
                reasoning: Option<String>,
            }

            match hc.call_zome_default::<CastVoteInput, ()>(
                "hearth_decisions",
                "cast_vote",
                &CastVoteInput { decision_hash, choice, reasoning },
            ).await {
                Ok(_) => toasts.push("vote cast", ToastKind::Custom("decision".into())),
                Err(e) => {
                    web_sys::console::log_1(&format!("cast_vote failed: {e}").into());
                }
            }
        });
    }
}

// ============================================================================
// Change Presence
// ============================================================================

pub fn change_presence(new_status: PresenceStatusType) {
    let hc = use_holochain();
    let hearth = use_hearth();

    if hc.is_mock() {
        let my_agent = hearth.my_agent.get_untracked();
        hearth.presence.update(|plist| {
            if let Some(p) = plist.iter_mut().find(|p| p.agent == my_agent) {
                p.status = new_status;
                p.updated_at = mock_now();
            }
        });
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct SetPresenceInput {
                hearth_hash: String,
                status: PresenceStatusType,
                expected_return: Option<i64>,
            }

            let hearth_hash = hearth.current_hearth.get_untracked()
                .map(|h| h.hash)
                .unwrap_or_default();

            let _ = hc.call_zome_default::<SetPresenceInput, ()>(
                "hearth_rhythms",
                "set_presence",
                &SetPresenceInput {
                    hearth_hash,
                    status: new_status,
                    expected_return: None,
                },
            ).await;
        });
    }
}

// ============================================================================
// Complete Care Task
// ============================================================================

pub fn complete_care_task(task_hash: String) {
    let hc = use_holochain();
    let hearth = use_hearth();
    let toasts = use_toasts();

    if hc.is_mock() {
        hearth.care_schedules.update(|schedules| {
            if let Some(task) = schedules.iter_mut().find(|s| s.hash == task_hash) {
                task.status = CareScheduleStatus::Completed;
                task.completed_at = Some(mock_now());
            }
        });
        // Update homeostasis
        if let Some(set_care) = use_context::<WriteSignal<u32>>() {
            let active = hearth.care_schedules.get_untracked().iter()
                .filter(|c| c.status == CareScheduleStatus::Active)
                .count() as u32;
            set_care.set(active);
        }
        toasts.push("task completed", ToastKind::Custom("care".into()));
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            // Conductor expects ActionHash for schedule_hash
            #[derive(serde::Serialize)]
            struct CompleteTaskInput { schedule_hash: String }

            let _ = hc.call_zome_default::<CompleteTaskInput, ()>(
                "hearth_care",
                "complete_task",
                &CompleteTaskInput { schedule_hash: task_hash },
            ).await;
            toasts.push("task completed", ToastKind::Custom("care".into()));
        });
    }
}

// ============================================================================
// Invite Member
// ============================================================================

pub fn invite_member(display_name: String, role: MemberRole) {
    let hc = use_holochain();
    let hearth = use_hearth();
    let toasts = use_toasts();

    if hc.is_mock() {
        let agent_key = format!("agent_{}", display_name.to_lowercase().replace(' ', "_"));
        hearth.members.update(|members| {
            members.push(MemberView {
                agent: agent_key,
                display_name: display_name.clone(),
                role,
                status: MembershipStatus::Invited,
                joined_at: mock_now(),
            });
        });
        toasts.push(format!("invited {display_name} to the hearth"), ToastKind::Custom("presence".into()));
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            // Real invite requires AgentPubKey of invitee (not just a name).
            // In production, the inviter would select from a DID/agent lookup.
            // For now, we send the display_name as a placeholder.
            #[derive(serde::Serialize)]
            struct InviteInput {
                hearth_hash: String,
                invitee_agent: String, // AgentPubKey on the wire
                proposed_role: MemberRole,
                message: String,
                expires_at: i64,
            }

            let hearth_hash = hearth.current_hearth.get_untracked()
                .map(|h| h.hash)
                .unwrap_or_default();

            let _ = hc.call_zome_default::<InviteInput, ()>(
                "hearth_kinship",
                "invite_member",
                &InviteInput {
                    hearth_hash,
                    invitee_agent: display_name, // placeholder — needs real AgentPubKey
                    proposed_role: role,
                    message: "you are invited to join the hearth".into(),
                    expires_at: mock_now() + 7 * 86400, // 7 days
                },
            ).await;
        });
    }
}
