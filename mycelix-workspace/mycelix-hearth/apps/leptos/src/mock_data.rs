// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Realistic mock data for a chosen family of 5.
//!
//! The Hearthstone: River (Founder), Sage (Elder), Rowan (Adult),
//! Ember (Youth, 16), and Wren (Child, 7).

use hearth_leptos_types::*;
use personal_leptos_types::*;

// ============================================================================
// Members
// ============================================================================

pub fn mock_members() -> Vec<MemberView> {
    vec![
        MemberView {
            agent: "agent_river".into(),
            display_name: "River".into(),
            role: MemberRole::Founder,
            status: MembershipStatus::Active,
            joined_at: 1_711_900_000,
        },
        MemberView {
            agent: "agent_sage".into(),
            display_name: "Sage".into(),
            role: MemberRole::Elder,
            status: MembershipStatus::Active,
            joined_at: 1_711_900_100,
        },
        MemberView {
            agent: "agent_rowan".into(),
            display_name: "Rowan".into(),
            role: MemberRole::Adult,
            status: MembershipStatus::Active,
            joined_at: 1_711_900_200,
        },
        MemberView {
            agent: "agent_ember".into(),
            display_name: "Ember".into(),
            role: MemberRole::Youth,
            status: MembershipStatus::Active,
            joined_at: 1_711_900_300,
        },
        MemberView {
            agent: "agent_wren".into(),
            display_name: "Wren".into(),
            role: MemberRole::Child,
            status: MembershipStatus::Active,
            joined_at: 1_712_000_000,
        },
    ]
}

pub fn mock_hearth() -> HearthView {
    HearthView {
        hash: "hearth_001".into(),
        name: "the hearthstone".into(),
        description: "a warm, intentional family space rooted in care and trust".into(),
        hearth_type: HearthType::Chosen,
        created_by: "agent_river".into(),
        created_at: 1_711_900_000,
        max_members: 12,
    }
}

// ============================================================================
// Bonds (with varying health)
// ============================================================================

pub fn mock_bonds() -> Vec<BondView> {
    vec![
        // River-Sage: partners, well-tended (9200 bp)
        BondView {
            hash: "bond_001".into(),
            member_a: "agent_river".into(),
            member_b: "agent_sage".into(),
            bond_type: BondType::Partner,
            strength_bp: 9200,
            last_tended: now_minus_days(2),
            created_at: 1_711_900_000,
        },
        // River-Rowan: chosen family, good health (7800 bp)
        BondView {
            hash: "bond_002".into(),
            member_a: "agent_river".into(),
            member_b: "agent_rowan".into(),
            bond_type: BondType::ChosenFamily,
            strength_bp: 7800,
            last_tended: now_minus_days(5),
            created_at: 1_711_900_200,
        },
        // Sage-Ember: guardian, decaying (4500 bp — needs attention)
        BondView {
            hash: "bond_003".into(),
            member_a: "agent_sage".into(),
            member_b: "agent_ember".into(),
            bond_type: BondType::Guardian,
            strength_bp: 4500,
            last_tended: now_minus_days(25),
            created_at: 1_711_900_300,
        },
        // River-Wren: parent, healthy (8500 bp)
        BondView {
            hash: "bond_004".into(),
            member_a: "agent_river".into(),
            member_b: "agent_wren".into(),
            bond_type: BondType::Parent,
            strength_bp: 8500,
            last_tended: now_minus_days(1),
            created_at: 1_712_000_000,
        },
        // Ember-Wren: sibling, moderate (6200 bp)
        BondView {
            hash: "bond_005".into(),
            member_a: "agent_ember".into(),
            member_b: "agent_wren".into(),
            bond_type: BondType::Sibling,
            strength_bp: 6200,
            last_tended: now_minus_days(10),
            created_at: 1_712_000_100,
        },
        // Rowan-Sage: chosen family, neglected (2800 bp — near warning)
        BondView {
            hash: "bond_006".into(),
            member_a: "agent_rowan".into(),
            member_b: "agent_sage".into(),
            bond_type: BondType::ChosenFamily,
            strength_bp: 2800,
            last_tended: now_minus_days(45),
            created_at: 1_711_900_400,
        },
    ]
}

// ============================================================================
// Care Schedules
// ============================================================================

pub fn mock_care_schedules() -> Vec<CareScheduleView> {
    vec![
        CareScheduleView {
            hash: "care_001".into(),
            hearth_hash: "hearth_001".into(),
            care_type: CareType::MealPrep,
            title: "Dinner prep".into(),
            description: "Prepare tonight's meal for the family".into(),
            assigned_to: "agent_rowan".into(),
            recurrence: Recurrence::Daily,
            status: CareScheduleStatus::Active,
            completed_at: None,
        },
        CareScheduleView {
            hash: "care_002".into(),
            hearth_hash: "hearth_001".into(),
            care_type: CareType::Childcare,
            title: "School pickup".into(),
            description: "Pick Wren up from school at 3pm".into(),
            assigned_to: "agent_sage".into(),
            recurrence: Recurrence::Weekly,
            status: CareScheduleStatus::Active,
            completed_at: None,
        },
        CareScheduleView {
            hash: "care_003".into(),
            hearth_hash: "hearth_001".into(),
            care_type: CareType::Chore,
            title: "Garden maintenance".into(),
            description: "Water the vegetables and check compost".into(),
            assigned_to: "agent_ember".into(),
            recurrence: Recurrence::Weekly,
            status: CareScheduleStatus::Active,
            completed_at: None,
        },
    ]
}

// ============================================================================
// Decisions
// ============================================================================

pub fn mock_decisions() -> Vec<DecisionView> {
    vec![
        DecisionView {
            hash: "decision_001".into(),
            hearth_hash: "hearth_001".into(),
            title: "Summer camping trip destination".into(),
            description: "Where should we go camping this August? Three options proposed.".into(),
            decision_type: DecisionType::MajorityVote,
            eligible_roles: vec![MemberRole::Founder, MemberRole::Elder, MemberRole::Adult, MemberRole::Youth],
            options: vec!["Mountain lake".into(), "Ocean beach".into(), "Forest cabin".into()],
            deadline: now_plus_days(5),
            quorum_bp: Some(6000),
            status: DecisionStatus::Open,
            created_by: "agent_river".into(),
            created_at: now_minus_days(2),
        },
        DecisionView {
            hash: "decision_002".into(),
            hearth_hash: "hearth_001".into(),
            title: "New household rhythm: evening circle".into(),
            description: "Should we start a nightly check-in circle? 15 min before bedtime.".into(),
            decision_type: DecisionType::Consensus,
            eligible_roles: vec![MemberRole::Founder, MemberRole::Elder, MemberRole::Adult],
            options: vec!["Yes, every night".into(), "Yes, weeknights only".into(), "Not yet".into()],
            deadline: now_plus_days(3),
            quorum_bp: None,
            status: DecisionStatus::Open,
            created_by: "agent_sage".into(),
            created_at: now_minus_days(1),
        },
    ]
}

pub fn mock_votes() -> Vec<VoteView> {
    vec![
        VoteView {
            decision_hash: "decision_001".into(),
            voter: "agent_river".into(),
            choice: 0, // Mountain lake
            weight_bp: 10000,
            reasoning: Some("The kids would love the swimming".into()),
            created_at: now_minus_days(1),
        },
        VoteView {
            decision_hash: "decision_001".into(),
            voter: "agent_ember".into(),
            choice: 2, // Forest cabin
            weight_bp: 5000, // Youth weight
            reasoning: Some("I want to see the stars without light pollution".into()),
            created_at: now_minus_days(1),
        },
    ]
}

// ============================================================================
// Gratitude
// ============================================================================

pub fn mock_gratitude() -> Vec<GratitudeExpressionView> {
    vec![
        GratitudeExpressionView {
            hash: "grat_001".into(),
            from_agent: "agent_ember".into(),
            to_agent: "agent_river".into(),
            message: "Thank you for staying up late to help me with my project".into(),
            gratitude_type: GratitudeType::Appreciation,
            visibility: HearthVisibility::AllMembers,
            created_at: now_minus_days(0),
        },
        GratitudeExpressionView {
            hash: "grat_002".into(),
            from_agent: "agent_river".into(),
            to_agent: "agent_sage".into(),
            message: "Your patience with Wren this morning was beautiful".into(),
            gratitude_type: GratitudeType::Acknowledgment,
            visibility: HearthVisibility::AllMembers,
            created_at: now_minus_days(1),
        },
        GratitudeExpressionView {
            hash: "grat_003".into(),
            from_agent: "agent_rowan".into(),
            to_agent: "agent_ember".into(),
            message: "The garden looks amazing — your hard work shows".into(),
            gratitude_type: GratitudeType::Celebration,
            visibility: HearthVisibility::AllMembers,
            created_at: now_minus_days(2),
        },
        GratitudeExpressionView {
            hash: "grat_004".into(),
            from_agent: "agent_wren".into(),
            to_agent: "agent_rowan".into(),
            message: "Best pancakes ever!".into(),
            gratitude_type: GratitudeType::Appreciation,
            visibility: HearthVisibility::AllMembers,
            created_at: now_minus_days(3),
        },
    ]
}

// ============================================================================
// Stories & Traditions
// ============================================================================

pub fn mock_stories() -> Vec<StoryView> {
    vec![
        StoryView {
            hash: "story_001".into(),
            hearth_hash: "hearth_001".into(),
            title: "The Night We Found Each Other".into(),
            content: "It was raining the evening we first gathered around a table and decided to become a family. Sage brought soup. River brought candles. Rowan brought music. We stayed up until dawn, and by morning, the Hearthstone had a name.".into(),
            storyteller: "agent_river".into(),
            story_type: StoryType::Origin,
            tags: vec!["origin".into(), "founding".into(), "rain".into()],
            visibility: HearthVisibility::AllMembers,
            created_at: 1_711_900_500,
        },
        StoryView {
            hash: "story_002".into(),
            hearth_hash: "hearth_001".into(),
            title: "Sage's Grandmother's Bread Recipe".into(),
            content: "Three cups flour, one cup warm water, a tablespoon of honey, and patience. The secret is letting the dough rest for two hours, not one.".into(),
            storyteller: "agent_sage".into(),
            story_type: StoryType::Recipe,
            tags: vec!["recipe".into(), "bread".into(), "grandmother".into()],
            visibility: HearthVisibility::AllMembers,
            created_at: 1_712_100_000,
        },
    ]
}

// ============================================================================
// Rhythms & Presence
// ============================================================================

pub fn mock_rhythms() -> Vec<RhythmView> {
    vec![
        RhythmView {
            hash: "rhythm_001".into(),
            hearth_hash: "hearth_001".into(),
            name: "Morning circle".into(),
            rhythm_type: RhythmType::Morning,
            description: "Brief check-in: how are you feeling? what do you need today?".into(),
            participants: vec!["agent_river".into(), "agent_sage".into(), "agent_rowan".into(), "agent_ember".into(), "agent_wren".into()],
        },
        RhythmView {
            hash: "rhythm_002".into(),
            hearth_hash: "hearth_001".into(),
            name: "Sunday meal together".into(),
            rhythm_type: RhythmType::Weekly,
            description: "Everyone cooks one dish. We eat together, no screens.".into(),
            participants: vec!["agent_river".into(), "agent_sage".into(), "agent_rowan".into(), "agent_ember".into(), "agent_wren".into()],
        },
    ]
}

pub fn mock_presence() -> Vec<PresenceView> {
    vec![
        PresenceView { agent: "agent_river".into(), status: PresenceStatusType::Home, expected_return: None, updated_at: now_minus_days(0) },
        PresenceView { agent: "agent_sage".into(), status: PresenceStatusType::Working, expected_return: Some(now_plus_hours(3)), updated_at: now_minus_days(0) },
        PresenceView { agent: "agent_rowan".into(), status: PresenceStatusType::Home, expected_return: None, updated_at: now_minus_days(0) },
        PresenceView { agent: "agent_ember".into(), status: PresenceStatusType::Away, expected_return: Some(now_plus_hours(1)), updated_at: now_minus_days(0) },
        PresenceView { agent: "agent_wren".into(), status: PresenceStatusType::Sleeping, expected_return: None, updated_at: now_minus_days(0) },
    ]
}

// ============================================================================
// Emergency
// ============================================================================

pub fn mock_emergency_alerts() -> Vec<EmergencyAlertView> {
    // No active alerts — peaceful household
    vec![]
}

// ============================================================================
// Resources
// ============================================================================

pub fn mock_resources() -> Vec<ResourceView> {
    vec![
        ResourceView {
            hash: "res_001".into(),
            hearth_hash: "hearth_001".into(),
            name: "Family car".into(),
            description: "2019 Subaru Outback".into(),
            resource_type: ResourceType::Vehicle,
            current_holder: Some("agent_sage".into()),
            condition: "Good".into(),
            location: "Driveway".into(),
        },
        ResourceView {
            hash: "res_002".into(),
            hearth_hash: "hearth_001".into(),
            name: "Power drill".into(),
            description: "DeWalt 20V cordless".into(),
            resource_type: ResourceType::Tool,
            current_holder: None,
            condition: "Good".into(),
            location: "Garage shelf".into(),
        },
    ]
}

// ============================================================================
// Milestones
// ============================================================================

pub fn mock_milestones() -> Vec<MilestoneView> {
    vec![
        MilestoneView {
            hash: "mile_001".into(),
            hearth_hash: "hearth_001".into(),
            member: "agent_wren".into(),
            milestone_type: MilestoneType::Birthday,
            date: now_minus_days(30),
            description: "Wren turned 7! Butterfly cake and backyard party.".into(),
        },
        MilestoneView {
            hash: "mile_002".into(),
            hearth_hash: "hearth_001".into(),
            member: "agent_ember".into(),
            milestone_type: MilestoneType::Custom("First solo trip".into()),
            date: now_minus_days(60),
            description: "Ember's first overnight trip without guardians — wilderness camp.".into(),
        },
    ]
}

// ============================================================================
// Autonomy
// ============================================================================

pub fn mock_autonomy_profiles() -> Vec<AutonomyProfileView> {
    vec![
        AutonomyProfileView {
            hash: "auto_001".into(),
            hearth_hash: "hearth_001".into(),
            member: "agent_ember".into(),
            current_tier: AutonomyTier::Guided,
            capabilities: vec!["solo_transit".into(), "meal_prep".into(), "garden_management".into()],
            restrictions: vec!["overnight_travel".into(), "financial_decisions".into()],
        },
        AutonomyProfileView {
            hash: "auto_002".into(),
            hearth_hash: "hearth_001".into(),
            member: "agent_wren".into(),
            current_tier: AutonomyTier::Supervised,
            capabilities: vec!["play_outside".into()],
            restrictions: vec!["solo_transit".into(), "screen_time".into(), "cooking_unsupervised".into()],
        },
    ]
}

// ============================================================================
// Personal Vault (mock)
// ============================================================================

pub fn mock_profile() -> ProfileView {
    ProfileView {
        display_name: "Rowan".into(),
        avatar: None,
        bio: Some("Gardener, cook, and maker of things. Part of The Hearthstone since day one.".into()),
        metadata: [
            ("pronouns".into(), "they/them".into()),
            ("timezone".into(), "America/Chicago".into()),
        ].into(),
        updated_at: now_minus_days(7),
    }
}

pub fn mock_credentials() -> Vec<StoredCredentialView> {
    vec![
        StoredCredentialView {
            hash: "cred_001".into(),
            credential_type: CredentialType::Identity,
            issuer: "did:mycelix:identity_cluster".into(),
            issued_at: now_minus_days(90),
            expires_at: Some(now_plus_days(275)),
            revoked: false,
        },
        StoredCredentialView {
            hash: "cred_002".into(),
            credential_type: CredentialType::Governance,
            issuer: "did:mycelix:governance_cluster".into(),
            issued_at: now_minus_days(30),
            expires_at: None,
            revoked: false,
        },
    ]
}

pub fn mock_trust_credential() -> TrustCredentialView {
    TrustCredentialView {
        id: "trust_001".into(),
        subject_did: "did:mycelix:agent_rowan".into(),
        issuer_did: "did:mycelix:governance_cluster".into(),
        trust_tier: TrustTier::Standard,
        trust_score_range: TrustScoreRange { lower: 0.42, upper: 0.58 },
        issued_at: now_minus_days(14),
        expires_at: Some(now_plus_days(350)),
        revoked: false,
    }
}

// ============================================================================
// Time helpers (relative to "now" = March 30, 2026 epoch seconds)
// ============================================================================

const MOCK_NOW: i64 = 1_774_934_400; // ~2026-03-30

fn now_minus_days(days: i64) -> i64 {
    MOCK_NOW - (days * 86400)
}

fn now_plus_days(days: i64) -> i64 {
    MOCK_NOW + (days * 86400)
}

fn now_plus_hours(hours: i64) -> i64 {
    MOCK_NOW + (hours * 3600)
}
