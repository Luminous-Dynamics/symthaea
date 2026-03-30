// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hearth context: current hearth, member list, user's role, bonds, and all domain data.
//!
//! Uses `RwSignal` so pages can both read and mutate state (mock mode).
//! In Phase 6, mutations will dispatch zome calls instead.

use leptos::prelude::*;
use hearth_leptos_types::*;
use crate::mock_data;

/// The active hearth context shared across pages.
#[derive(Clone)]
pub struct HearthCtx {
    pub current_hearth: RwSignal<Option<HearthView>>,
    pub members: RwSignal<Vec<MemberView>>,
    pub my_role: RwSignal<Option<MemberRole>>,
    pub bonds: RwSignal<Vec<BondView>>,
    pub care_schedules: RwSignal<Vec<CareScheduleView>>,
    pub decisions: RwSignal<Vec<DecisionView>>,
    pub votes: RwSignal<Vec<VoteView>>,
    pub gratitude: RwSignal<Vec<GratitudeExpressionView>>,
    pub stories: RwSignal<Vec<StoryView>>,
    pub rhythms: RwSignal<Vec<RhythmView>>,
    pub presence: RwSignal<Vec<PresenceView>>,
    pub emergency_alerts: RwSignal<Vec<EmergencyAlertView>>,
    pub resources: RwSignal<Vec<ResourceView>>,
    pub milestones: RwSignal<Vec<MilestoneView>>,
    pub autonomy_profiles: RwSignal<Vec<AutonomyProfileView>>,
    /// The current user's agent key (mock).
    pub my_agent: RwSignal<String>,
}

/// Initialize hearth context with mock data.
pub fn provide_hearth_context() -> HearthCtx {
    let ctx = HearthCtx {
        current_hearth: RwSignal::new(Some(mock_data::mock_hearth())),
        members: RwSignal::new(mock_data::mock_members()),
        my_role: RwSignal::new(Some(MemberRole::Adult)),
        bonds: RwSignal::new(mock_data::mock_bonds()),
        care_schedules: RwSignal::new(mock_data::mock_care_schedules()),
        decisions: RwSignal::new(mock_data::mock_decisions()),
        votes: RwSignal::new(mock_data::mock_votes()),
        gratitude: RwSignal::new(mock_data::mock_gratitude()),
        stories: RwSignal::new(mock_data::mock_stories()),
        rhythms: RwSignal::new(mock_data::mock_rhythms()),
        presence: RwSignal::new(mock_data::mock_presence()),
        emergency_alerts: RwSignal::new(mock_data::mock_emergency_alerts()),
        resources: RwSignal::new(mock_data::mock_resources()),
        milestones: RwSignal::new(mock_data::mock_milestones()),
        autonomy_profiles: RwSignal::new(mock_data::mock_autonomy_profiles()),
        my_agent: RwSignal::new("agent_rowan".into()),
    };

    provide_context(ctx.clone());

    // Feed homeostasis detection
    if let Some(set_care) = use_context::<WriteSignal<u32>>() {
        let care_count = ctx.care_schedules.get_untracked()
            .iter()
            .filter(|c| c.status == CareScheduleStatus::Active)
            .count() as u32;
        set_care.set(care_count);
    }

    ctx
}

pub fn use_hearth() -> HearthCtx {
    expect_context::<HearthCtx>()
}

/// Look up a member's display name by agent key.
pub fn member_name(members: &[MemberView], agent: &str) -> String {
    members.iter()
        .find(|m| m.agent == agent)
        .map(|m| m.display_name.clone())
        .unwrap_or_else(|| agent[..8.min(agent.len())].to_string())
}

/// Simple mock timestamp (seconds since epoch, approximate).
pub fn mock_now() -> i64 {
    1_774_934_400 // ~2026-03-30
}
