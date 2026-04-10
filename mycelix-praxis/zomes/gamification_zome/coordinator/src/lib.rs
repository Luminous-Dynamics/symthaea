// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Gamification Coordinator Zome
//!
//! Implements the business logic for the gamification system including:
//! - XP earning and spending
//! - Level progression
//! - Badge earning and tracking
//! - Streak management
//! - Leaderboard updates

use hdk::prelude::*;
use gamification_integrity::*;

// ============== Helper Functions ==============

/// Convert a Holochain Timestamp to i64 (microseconds)
fn timestamp_to_i64(ts: Timestamp) -> i64 {
    ts.as_micros()
}

/// Get current time as i64
fn current_time() -> ExternResult<i64> {
    Ok(timestamp_to_i64(sys_time()?))
}

/// Get current date as YYYYMMDD
fn current_date() -> ExternResult<u32> {
    let now = current_time()?;
    let days_since_epoch = (now / (24 * 60 * 60 * 1_000_000)) as u32;
    Ok(19700101 + days_since_epoch)
}

/// Get learner anchor hash
fn learner_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("gamification.learner.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToXp)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

// ============== XP Management ==============

/// Get or create XP record for current learner
#[hdk_extern]
pub fn get_or_create_xp(_: ()) -> ExternResult<Record> {
    let learner = agent_info()?.agent_initial_pubkey;
    let anchor_hash = learner_anchor()?;

    // Check for existing XP record
    let links = get_links(
        LinkQuery::try_new(anchor_hash.clone(), LinkTypes::LearnerToXp)?,
        GetStrategy::Local,
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                return Ok(record);
            }
        }
    }

    // Create new XP record
    let now = current_time()?;
    let xp = LearnerXp {
        learner,
        total_xp: 0,
        level: 1,
        daily_xp: 0,
        weekly_xp: 0,
        monthly_xp: 0,
        last_activity_at: now,
        created_at: now,
        modified_at: now,
    };

    let action_hash = create_entry(EntryTypes::LearnerXp(xp))?;

    create_link(
        anchor_hash,
        action_hash.clone(),
        LinkTypes::LearnerToXp,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created XP record".into())))
}

/// Award XP to the current learner
#[hdk_extern]
pub fn award_xp(input: AwardXpInput) -> ExternResult<Record> {
    let now = current_time()?;
    let learner = agent_info()?.agent_initial_pubkey;

    // Get current XP record
    let xp_record = get_or_create_xp(())?;
    let xp_hash = xp_record.action_hashed().hash.clone();

    let mut xp: LearnerXp = xp_record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid XP entry".into())))?;

    // Get current streak for bonus multiplier
    let streak_bonus = get_streak_bonus()?;

    // Apply streak bonus to base XP
    let bonus_xp_u64 = input.base_xp as u64 * streak_bonus as u64 / 1000; let bonus_xp = bonus_xp_u64.min(u32::MAX as u64) as u32;

    // Update XP
    xp.total_xp = xp.total_xp.saturating_add(bonus_xp as u64);
    xp.daily_xp = xp.daily_xp.saturating_add(bonus_xp);
    xp.weekly_xp = xp.weekly_xp.saturating_add(bonus_xp);
    xp.monthly_xp = xp.monthly_xp.saturating_add(bonus_xp);
    xp.level = calculate_level(xp.total_xp);
    xp.last_activity_at = now;
    xp.modified_at = now;

    // Create transaction record
    let tx = XpTransaction {
        learner: learner.clone(),
        amount: bonus_xp as i64,
        activity_type: input.activity_type,
        reference_hash: input.reference_hash,
        occurred_at: now,
        description: input.description,
    };

    let tx_hash = create_entry(EntryTypes::XpTransaction(tx))?;

    // Link transaction to learner
    let anchor_hash = learner_anchor()?;
    create_link(
        anchor_hash,
        tx_hash,
        LinkTypes::LearnerToXpTransactions,
        (),
    )?;

    // Update XP record
    let new_hash = update_entry(xp_hash, xp)?;

    // Update streak (activity counts)
    update_daily_activity(bonus_xp)?;

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated XP record".into())))
}

/// Input for awarding XP
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AwardXpInput {
    pub base_xp: u32,
    pub activity_type: XpActivityType,
    pub reference_hash: Option<ActionHash>,
    pub description: String,
}

/// Get XP transactions for current learner
#[hdk_extern]
pub fn get_xp_transactions(limit: u32) -> ExternResult<Vec<Record>> {
    let anchor_hash = learner_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash, LinkTypes::LearnerToXpTransactions)?,
        GetStrategy::Local,
    )?;

    let mut transactions: Vec<(Record, i64)> = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(tx) = record.entry().to_app_option::<XpTransaction>().ok().flatten() {
                    transactions.push((record, tx.occurred_at));
                }
            }
        }
    }

    // Sort by time (newest first)
    transactions.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(transactions.into_iter().take(limit as usize).map(|(r, _)| r).collect())
}

// ============== Streak Management ==============

/// Get or create streak record for current learner
#[hdk_extern]
pub fn get_or_create_streak(_: ()) -> ExternResult<Record> {
    let learner = agent_info()?.agent_initial_pubkey;
    let anchor_hash = learner_anchor()?;

    // Check for existing streak record
    let links = get_links(
        LinkQuery::try_new(anchor_hash.clone(), LinkTypes::LearnerToStreak)?,
        GetStrategy::Local,
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                return Ok(record);
            }
        }
    }

    // Create new streak record
    let now = current_time()?;
    let today = current_date()?;
    let streak = LearnerStreak {
        learner,
        current_streak: 0,
        longest_streak: 0,
        total_active_days: 0,
        last_active_date: 0,
        is_frozen: false,
        freezes_remaining: 3,
        streak_start_date: today,
        streak_bonus_permille: 1000,
        created_at: now,
        modified_at: now,
    };

    let action_hash = create_entry(EntryTypes::LearnerStreak(streak))?;

    create_link(
        anchor_hash,
        action_hash.clone(),
        LinkTypes::LearnerToStreak,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created streak record".into())))
}

/// Get current streak bonus multiplier (as permille)
fn get_streak_bonus() -> ExternResult<u16> {
    let streak_record = get_or_create_streak(())?;
    let streak: LearnerStreak = streak_record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid streak entry".into())))?;

    Ok(streak_bonus_permille(streak.current_streak))
}

/// Update daily activity and streak
fn update_daily_activity(xp_earned: u32) -> ExternResult<()> {
    let now = current_time()?;
    let today = current_date()?;
    let learner = agent_info()?.agent_initial_pubkey;
    let anchor_hash = learner_anchor()?;

    // Get or create daily activity
    let links = get_links(
        LinkQuery::try_new(anchor_hash.clone(), LinkTypes::LearnerToDailyActivity)?,
        GetStrategy::Local,
    )?;

    let mut found_today = false;
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(mut activity) = record.entry().to_app_option::<DailyActivity>().ok().flatten() {
                    if activity.date == today {
                        // Update existing activity
                        activity.xp_earned += xp_earned;
                        activity.cards_reviewed += 1; // Simplified
                        activity.counts_for_streak = activity.xp_earned >= 10; // Min 10 XP for streak

                        update_entry(action_hash, activity)?;
                        found_today = true;
                        break;
                    }
                }
            }
        }
    }

    if !found_today {
        // Create new daily activity
        let activity = DailyActivity {
            learner: learner.clone(),
            date: today,
            lessons_completed: 0,
            cards_reviewed: 1,
            courses_progressed: 0,
            xp_earned,
            time_spent_seconds: 0,
            counts_for_streak: xp_earned >= 10,
            activities: vec!["activity".to_string()],
        };

        let action_hash = create_entry(EntryTypes::DailyActivity(activity))?;

        create_link(
            anchor_hash,
            action_hash,
            LinkTypes::LearnerToDailyActivity,
            (),
        )?;

        // Update streak
        update_streak(today)?;
    }

    Ok(())
}

/// Update streak based on today's activity
fn update_streak(today: u32) -> ExternResult<()> {
    let now = current_time()?;

    let streak_record = get_or_create_streak(())?;
    let streak_hash = streak_record.action_hashed().hash.clone();

    let mut streak: LearnerStreak = streak_record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid streak entry".into())))?;

    // Check if this is a new day of activity
    if streak.last_active_date == today {
        return Ok(()); // Already counted today
    }

    let yesterday = today - 1;

    if streak.last_active_date == yesterday || streak.last_active_date == 0 {
        // Continuing streak
        streak.current_streak += 1;
        if streak.current_streak > streak.longest_streak {
            streak.longest_streak = streak.current_streak;
        }
    } else if streak.is_frozen && streak.freezes_remaining > 0 {
        // Use a freeze
        streak.freezes_remaining -= 1;
        streak.is_frozen = false;
        streak.current_streak += 1;
    } else {
        // Streak broken - start new streak
        streak.current_streak = 1;
        streak.streak_start_date = today;
    }

    streak.last_active_date = today;
    streak.total_active_days += 1;
    streak.streak_bonus_permille = streak_bonus_permille(streak.current_streak);
    streak.modified_at = now;

    update_entry(streak_hash, streak)?;

    Ok(())
}

/// Freeze current streak (use a freeze)
#[hdk_extern]
pub fn freeze_streak(_: ()) -> ExternResult<Record> {
    let now = current_time()?;

    let streak_record = get_or_create_streak(())?;
    let streak_hash = streak_record.action_hashed().hash.clone();

    let mut streak: LearnerStreak = streak_record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid streak entry".into())))?;

    if streak.freezes_remaining == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest("No streak freezes remaining".into())));
    }

    streak.is_frozen = true;
    streak.modified_at = now;

    let new_hash = update_entry(streak_hash, streak)?;

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated streak".into())))
}

// ============== Badge Management ==============

/// Create a new badge definition
#[hdk_extern]
pub fn create_badge_definition(input: CreateBadgeInput) -> ExternResult<Record> {
    // Trust tier gate: requires Steward tier to create badge definitions
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_constitutional(),
        "create_badge_definition",
    )?;

    let now = current_time()?;

    let badge = BadgeDefinition {
        badge_id: input.badge_id,
        name: input.name,
        description: input.description,
        icon: input.icon,
        rarity: input.rarity,
        category: input.category,
        xp_reward: input.xp_reward,
        criteria_json: input.criteria_json,
        is_active: true,
        is_secret: input.is_secret,
        created_at: now,
    };

    let action_hash = create_entry(EntryTypes::BadgeDefinition(badge))?;

    // Link to all badges
    let all_badges_path = Path::from("gamification.all_badges");
    let typed_path = all_badges_path.typed(LinkTypes::AllBadges)?;
    typed_path.ensure()?;

    create_link(
        typed_path.path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllBadges,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created badge".into())))
}

/// Input for creating a badge
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateBadgeInput {
    pub badge_id: String,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub rarity: BadgeRarity,
    pub category: BadgeCategory,
    pub xp_reward: u32,
    pub criteria_json: String,
    pub is_secret: bool,
}

/// Award a badge to the current learner
#[hdk_extern]
pub fn award_badge(input: AwardBadgeInput) -> ExternResult<Record> {
    let now = current_time()?;
    let learner = agent_info()?.agent_initial_pubkey;

    // Create earned badge
    let earned = EarnedBadge {
        learner: learner.clone(),
        badge_definition_hash: input.badge_definition_hash.clone(),
        earned_at: now,
        progress_at_earn: input.progress_at_earn,
        context_hash: input.context_hash,
        is_displayed: true,
    };

    let action_hash = create_entry(EntryTypes::EarnedBadge(earned))?;

    // Link to learner
    let anchor_hash = learner_anchor()?;
    create_link(
        anchor_hash,
        action_hash.clone(),
        LinkTypes::LearnerToBadges,
        (),
    )?;

    // Link from badge definition
    create_link(
        input.badge_definition_hash.clone(),
        action_hash.clone(),
        LinkTypes::BadgeToEarned,
        (),
    )?;

    // Award XP for the badge
    if let Some(record) = get(input.badge_definition_hash, GetOptions::default())? {
        if let Some(badge) = record.entry().to_app_option::<BadgeDefinition>().ok().flatten() {
            award_xp(AwardXpInput {
                base_xp: badge.xp_reward,
                activity_type: XpActivityType::BadgeEarned,
                reference_hash: Some(action_hash.clone()),
                description: format!("Earned badge: {}", badge.name),
            })?;
        }
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get earned badge".into())))
}

/// Input for awarding a badge
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AwardBadgeInput {
    pub badge_definition_hash: ActionHash,
    pub progress_at_earn: u32,
    pub context_hash: Option<ActionHash>,
}

/// Get all badges earned by current learner
#[hdk_extern]
pub fn get_my_badges(_: ()) -> ExternResult<Vec<Record>> {
    let anchor_hash = learner_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash, LinkTypes::LearnerToBadges)?,
        GetStrategy::Local,
    )?;

    let mut badges = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                badges.push(record);
            }
        }
    }

    Ok(badges)
}

/// Get all available badge definitions
#[hdk_extern]
pub fn get_all_badge_definitions(_: ()) -> ExternResult<Vec<Record>> {
    let all_badges_path = Path::from("gamification.all_badges");
    let typed_path = all_badges_path.typed(LinkTypes::AllBadges)?;
    typed_path.ensure()?;
    let path_hash = typed_path.path.path_entry_hash()?;

    let links = get_links(
        LinkQuery::try_new(path_hash, LinkTypes::AllBadges)?,
        GetStrategy::Local,
    )?;

    let mut badges = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                badges.push(record);
            }
        }
    }

    Ok(badges)
}

// ============== Leaderboard Management ==============

/// Create a new leaderboard
#[hdk_extern]
pub fn create_leaderboard(input: CreateLeaderboardInput) -> ExternResult<Record> {
    let now = current_time()?;

    let leaderboard = Leaderboard {
        leaderboard_id: input.leaderboard_id,
        name: input.name,
        description: input.description,
        ranking_type: input.ranking_type,
        time_period: input.time_period,
        max_entries: input.max_entries,
        is_active: true,
        scope: input.scope,
        scope_hash: input.scope_hash,
        created_at: now,
    };

    let action_hash = create_entry(EntryTypes::Leaderboard(leaderboard))?;

    // Link to all leaderboards
    let all_lb_path = Path::from("gamification.all_leaderboards");
    let typed_path = all_lb_path.typed(LinkTypes::AllLeaderboards)?;
    typed_path.ensure()?;

    create_link(
        typed_path.path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllLeaderboards,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created leaderboard".into())))
}

/// Input for creating a leaderboard
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateLeaderboardInput {
    pub leaderboard_id: String,
    pub name: String,
    pub description: String,
    pub ranking_type: LeaderboardType,
    pub time_period: LeaderboardPeriod,
    pub max_entries: u32,
    pub scope: LeaderboardScope,
    pub scope_hash: Option<ActionHash>,
}

/// Get all leaderboards
#[hdk_extern]
pub fn get_all_leaderboards(_: ()) -> ExternResult<Vec<Record>> {
    let all_lb_path = Path::from("gamification.all_leaderboards");
    let typed_path = all_lb_path.typed(LinkTypes::AllLeaderboards)?;
    typed_path.ensure()?;
    let path_hash = typed_path.path.path_entry_hash()?;

    let links = get_links(
        LinkQuery::try_new(path_hash, LinkTypes::AllLeaderboards)?,
        GetStrategy::Local,
    )?;

    let mut leaderboards = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                leaderboards.push(record);
            }
        }
    }

    Ok(leaderboards)
}

/// Get leaderboard entries
#[hdk_extern]
pub fn get_leaderboard_entries(leaderboard_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(leaderboard_hash, LinkTypes::LeaderboardToEntries)?,
        GetStrategy::Local,
    )?;

    let mut entries: Vec<(Record, u32)> = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(entry) = record.entry().to_app_option::<LeaderboardEntry>().ok().flatten() {
                    entries.push((record, entry.rank));
                }
            }
        }
    }

    // Sort by rank
    entries.sort_by_key(|(_, rank)| *rank);

    Ok(entries.into_iter().map(|(r, _)| r).collect())
}

// ============== Rewards ==============

/// Create a new reward
#[hdk_extern]
pub fn create_reward(input: CreateRewardInput) -> ExternResult<Record> {
    let now = current_time()?;

    let reward = Reward {
        reward_id: input.reward_id,
        name: input.name,
        description: input.description,
        reward_type: input.reward_type,
        cost_xp: input.cost_xp,
        level_required: input.level_required,
        badges_required: input.badges_required,
        is_available: true,
        quantity_limit: input.quantity_limit,
        quantity_claimed: 0,
        created_at: now,
    };

    let action_hash = create_entry(EntryTypes::Reward(reward))?;

    // Link to all rewards
    let all_rewards_path = Path::from("gamification.all_rewards");
    let typed_path = all_rewards_path.typed(LinkTypes::AllRewards)?;
    typed_path.ensure()?;

    create_link(
        typed_path.path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllRewards,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created reward".into())))
}

/// Input for creating a reward
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateRewardInput {
    pub reward_id: String,
    pub name: String,
    pub description: String,
    pub reward_type: RewardType,
    pub cost_xp: u32,
    pub level_required: u32,
    pub badges_required: Vec<ActionHash>,
    pub quantity_limit: u32,
}

/// Claim a reward
#[hdk_extern]
pub fn claim_reward(reward_hash: ActionHash) -> ExternResult<Record> {
    let now = current_time()?;
    let learner = agent_info()?.agent_initial_pubkey;

    // Get reward and check eligibility
    let reward_record = get(reward_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Reward not found".into())))?;

    let reward: Reward = reward_record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid reward entry".into())))?;

    // Check if still available
    if !reward.is_available {
        return Err(wasm_error!(WasmErrorInner::Guest("Reward is no longer available".into())));
    }

    // Check quantity
    if reward.quantity_limit > 0 && reward.quantity_claimed >= reward.quantity_limit {
        return Err(wasm_error!(WasmErrorInner::Guest("Reward is sold out".into())));
    }

    // Check XP
    let xp_record = get_or_create_xp(())?;
    let xp: LearnerXp = xp_record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid XP entry".into())))?;

    if xp.total_xp < reward.cost_xp as u64 {
        return Err(wasm_error!(WasmErrorInner::Guest("Not enough XP".into())));
    }

    // Check level
    if xp.level < reward.level_required {
        return Err(wasm_error!(WasmErrorInner::Guest("Level too low".into())));
    }

    // Create claimed reward
    let claimed = ClaimedReward {
        learner: learner.clone(),
        reward_hash: reward_hash.clone(),
        xp_spent: reward.cost_xp,
        claimed_at: now,
        is_active: true,
        expires_at: None,
    };

    let action_hash = create_entry(EntryTypes::ClaimedReward(claimed))?;

    // Link to learner
    let anchor_hash = learner_anchor()?;
    create_link(
        anchor_hash,
        action_hash.clone(),
        LinkTypes::LearnerToRewards,
        (),
    )?;

    // Deduct XP (negative transaction)
    let deduct_tx = XpTransaction {
        learner,
        amount: -(reward.cost_xp as i64),
        activity_type: XpActivityType::Custom,
        reference_hash: Some(action_hash.clone()),
        occurred_at: now,
        description: format!("Claimed reward: {}", reward.name),
    };

    create_entry(EntryTypes::XpTransaction(deduct_tx))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get claimed reward".into())))
}

/// Get my claimed rewards
#[hdk_extern]
pub fn get_my_rewards(_: ()) -> ExternResult<Vec<Record>> {
    let anchor_hash = learner_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash, LinkTypes::LearnerToRewards)?,
        GetStrategy::Local,
    )?;

    let mut rewards = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                rewards.push(record);
            }
        }
    }

    Ok(rewards)
}

/// Get all available rewards
#[hdk_extern]
pub fn get_all_rewards(_: ()) -> ExternResult<Vec<Record>> {
    let all_rewards_path = Path::from("gamification.all_rewards");
    let typed_path = all_rewards_path.typed(LinkTypes::AllRewards)?;
    typed_path.ensure()?;
    let path_hash = typed_path.path.path_entry_hash()?;

    let links = get_links(
        LinkQuery::try_new(path_hash, LinkTypes::AllRewards)?,
        GetStrategy::Local,
    )?;

    let mut rewards = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                rewards.push(record);
            }
        }
    }

    Ok(rewards)
}

// ============== Gamification Summary ==============

/// Get a complete gamification summary for current learner
#[hdk_extern]
pub fn get_gamification_summary(_: ()) -> ExternResult<GamificationSummary> {
    let xp_record = get_or_create_xp(())?;
    let xp: LearnerXp = xp_record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid XP entry".into())))?;

    let streak_record = get_or_create_streak(())?;
    let streak: LearnerStreak = streak_record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid streak entry".into())))?;

    let badges = get_my_badges(())?;

    Ok(GamificationSummary {
        total_xp: xp.total_xp,
        level: xp.level,
        xp_to_next_level: xp_to_next_level(xp.total_xp),
        level_progress_permille: level_progress_permille(xp.total_xp),
        current_streak: streak.current_streak,
        longest_streak: streak.longest_streak,
        streak_bonus_permille: streak.streak_bonus_permille,
        badges_earned: badges.len() as u32,
        freezes_remaining: streak.freezes_remaining,
    })
}

/// Summary of gamification stats
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GamificationSummary {
    pub total_xp: u64,
    pub level: u32,
    pub xp_to_next_level: u64,
    pub level_progress_permille: u16,
    pub current_streak: u32,
    pub longest_streak: u32,
    pub streak_bonus_permille: u16,
    pub badges_earned: u32,
    pub freezes_remaining: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamification_summary_serialization() {
        let summary = GamificationSummary {
            total_xp: 1000,
            level: 3,
            xp_to_next_level: 600,
            level_progress_permille: 250,
            current_streak: 7,
            longest_streak: 14,
            streak_bonus_permille: 1100,
            badges_earned: 5,
            freezes_remaining: 2,
        };

        let json = serde_json::to_string(&summary).unwrap();
        let parsed: GamificationSummary = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.total_xp, 1000);
        assert_eq!(parsed.level, 3);
        assert_eq!(parsed.current_streak, 7);
    }
}
