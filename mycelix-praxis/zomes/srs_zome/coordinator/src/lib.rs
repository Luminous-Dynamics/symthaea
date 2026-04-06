// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spaced Repetition System (SRS) Coordinator Zome
//!
//! Implements the SuperMemo SM-2 algorithm for optimal learning retention.
//! This zome provides all the business logic for managing review cards,
//! scheduling reviews, and tracking learning statistics.

use hdk::prelude::*;
use srs_integrity::*;
use praxis_core::errors::{srs_errors, EduNetError};

/// Convert an EduNetError to a WasmError
fn to_wasm_error(err: EduNetError) -> WasmError {
    wasm_error!(WasmErrorInner::Guest(err.to_message()))
}

/// Default ease factor for new cards (2.5)
const DEFAULT_EASE_PERMILLE: u16 = 2500;
/// Minimum ease factor (1.3)
const MIN_EASE_PERMILLE: u16 = 1300;
/// Maximum ease factor (5.0)
const MAX_EASE_PERMILLE: u16 = 5000;
/// Minutes in a day
const MINUTES_PER_DAY: u32 = 1440;

// ============== Helper Functions ==============

/// Convert a Holochain Timestamp to i64 (microseconds)
fn timestamp_to_i64(ts: Timestamp) -> i64 {
    ts.as_micros()
}

/// Get current time as i64
fn current_time() -> ExternResult<i64> {
    Ok(timestamp_to_i64(sys_time()?))
}

/// Get learner anchor hash
fn learner_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("srs.learner.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToCards)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Apply the SM-2 algorithm to calculate new ease factor and interval
///
/// Returns (new_ease_permille, new_interval_minutes, new_repetitions, new_status)
fn sm2_algorithm(
    quality: u8,
    current_ease_permille: u16,
    current_interval_minutes: u32,
    current_repetitions: u32,
    config: &SrsConfig,
) -> (u16, u32, u32, CardStatus) {
    // Quality must be 0-5
    let q = quality.min(5) as f64;

    // Calculate new ease factor using SM-2 formula
    // EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    let current_ef = current_ease_permille as f64 / 1000.0;
    let adjustment = 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02);
    let new_ef = (current_ef + adjustment).max(MIN_EASE_PERMILLE as f64 / 1000.0)
        .min(MAX_EASE_PERMILLE as f64 / 1000.0);
    let new_ease_permille = (new_ef * 1000.0) as u16;

    if quality < 3 {
        // Failed review - reset to learning phase
        (new_ease_permille, config.learning_steps_minutes.first().copied().unwrap_or(1), 0, CardStatus::Learning)
    } else {
        // Successful review
        let new_repetitions = current_repetitions + 1;

        let new_interval_minutes = if current_repetitions == 0 {
            // First successful review - use graduating interval
            config.graduating_interval_days * MINUTES_PER_DAY
        } else if current_repetitions == 1 {
            // Second successful review - 6 days
            6 * MINUTES_PER_DAY
        } else {
            // Subsequent reviews - multiply by ease factor
            let interval_days = (current_interval_minutes as f64 / MINUTES_PER_DAY as f64 * new_ef) as u32;
            let capped_days = interval_days.min(config.max_interval_days);
            capped_days * MINUTES_PER_DAY
        };

        // Apply interval modifier
        let modified_interval = (new_interval_minutes as f64
            * (config.interval_modifier_permille as f64 / 1000.0)) as u32;

        // Determine status
        let new_status = if modified_interval >= 365 * MINUTES_PER_DAY {
            CardStatus::Graduated
        } else {
            CardStatus::Review
        };

        (new_ease_permille, modified_interval, new_repetitions, new_status)
    }
}

// ============== Pure Business Logic (HDK-free, testable) ==============

/// Pure SM-2 ease factor update. Returns new ease factor as permille (1300-5000).
///
/// Implements: EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
pub fn sm2_ease_factor(current_ease_permille: u16, quality: u8) -> u16 {
    let q = quality.min(5) as f64;
    let current_ef = current_ease_permille as f64 / 1000.0;
    let adjustment = 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02);
    let new_ef = (current_ef + adjustment)
        .max(MIN_EASE_PERMILLE as f64 / 1000.0)
        .min(MAX_EASE_PERMILLE as f64 / 1000.0);
    (new_ef * 1000.0) as u16
}

/// Pure SM-2 interval calculation (in minutes).
///
/// For repetitions == 0: graduating_interval_days * 1440
/// For repetitions == 1: 6 * 1440
/// For repetitions >= 2: previous_interval * ease_factor, capped at max_interval_days
///
/// The result is further scaled by interval_modifier_permille / 1000.
pub fn sm2_interval(
    repetitions: u32,
    current_interval_minutes: u32,
    ease_factor_permille: u16,
    graduating_interval_days: u32,
    max_interval_days: u32,
    interval_modifier_permille: u16,
) -> u32 {
    let ef = ease_factor_permille as f64 / 1000.0;
    let raw = if repetitions == 0 {
        graduating_interval_days * MINUTES_PER_DAY
    } else if repetitions == 1 {
        6 * MINUTES_PER_DAY
    } else {
        let interval_days = (current_interval_minutes as f64 / MINUTES_PER_DAY as f64 * ef) as u32;
        interval_days.min(max_interval_days) * MINUTES_PER_DAY
    };
    (raw as f64 * (interval_modifier_permille as f64 / 1000.0)) as u32
}

/// Check if a card qualifies as a leech (too many lapses).
pub fn is_leech(lapses: u32, threshold: u32) -> bool {
    lapses >= threshold
}

// ============== Card Management ==============

/// Create a new review card for a knowledge node
#[hdk_extern]
pub fn create_card(input: CreateCardInput) -> ExternResult<Record> {
    let now = current_time()?;
    let learner = agent_info()?.agent_initial_pubkey;

    let card = ReviewCard {
        node_hash: input.node_hash.clone(),
        learner: learner.clone(),
        ease_factor_permille: DEFAULT_EASE_PERMILLE,
        interval_minutes: 0,
        repetitions: 0,
        status: CardStatus::New,
        due_at: now, // Due immediately
        last_reviewed_at: None,
        total_reviews: 0,
        correct_count: 0,
        incorrect_count: 0,
        lapses: 0,
        avg_time_seconds: 0,
        custom_front: input.custom_front,
        custom_back: input.custom_back,
        tags: input.tags,
        created_at: now,
        modified_at: now,
    };

    let action_hash = create_entry(EntryTypes::ReviewCard(card.clone()))?;

    // Link from learner anchor
    let anchor_hash = learner_anchor()?;
    create_link(
        anchor_hash,
        action_hash.clone(),
        LinkTypes::LearnerToCards,
        (),
    )?;

    // Link from node
    create_link(
        input.node_hash,
        action_hash.clone(),
        LinkTypes::NodeToCards,
        (),
    )?;

    // Link to deck if specified
    if let Some(deck_hash) = input.deck_hash {
        create_link(
            deck_hash,
            action_hash.clone(),
            LinkTypes::DeckToCards,
            (),
        )?;
    }

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created card".into())))?;

    Ok(record)
}

/// Input for creating a new card
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateCardInput {
    pub node_hash: ActionHash,
    pub deck_hash: Option<ActionHash>,
    pub custom_front: Option<String>,
    pub custom_back: Option<String>,
    pub tags: Vec<String>,
}

/// Get a card by its action hash
#[hdk_extern]
pub fn get_card(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all cards for the current learner
#[hdk_extern]
pub fn get_my_cards(_: ()) -> ExternResult<Vec<Record>> {
    let anchor_hash = learner_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash, LinkTypes::LearnerToCards)?,
        GetStrategy::Local,
    )?;

    let mut cards = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                cards.push(record);
            }
        }
    }

    Ok(cards)
}

/// Get cards due for review
#[hdk_extern]
pub fn get_due_cards(limit: u32) -> ExternResult<Vec<Record>> {
    let now = current_time()?;
    let all_cards = get_my_cards(())?;

    let mut due_cards: Vec<(Record, i64)> = all_cards
        .into_iter()
        .filter_map(|record| {
            record.entry()
                .to_app_option::<ReviewCard>()
                .ok()
                .flatten()
                .filter(|card| card.due_at <= now && card.status != CardStatus::Suspended && card.status != CardStatus::Buried)
                .map(|card| (record, card.due_at))
        })
        .collect();

    // Sort by due date (oldest first)
    due_cards.sort_by_key(|(_, due_at)| *due_at);

    // Return limited number
    Ok(due_cards.into_iter().take(limit as usize).map(|(r, _)| r).collect())
}

/// Suspend a card
#[hdk_extern]
pub fn suspend_card(action_hash: ActionHash) -> ExternResult<Record> {
    let hash_str = action_hash.to_string();
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or_else(|| to_wasm_error(srs_errors::card_not_found(&hash_str)))?;

    let mut card: ReviewCard = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| to_wasm_error(
            praxis_core::errors::errors::invalid_entry("ReviewCard", &hash_str)
        ))?;

    card.status = CardStatus::Suspended;
    card.modified_at = current_time()?;

    let new_hash = update_entry(action_hash, card)?;

    get(new_hash, GetOptions::default())?
        .ok_or_else(|| to_wasm_error(
            praxis_core::errors::errors::update_failed("ReviewCard", "suspend")
        ))
}

/// Unsuspend a card
#[hdk_extern]
pub fn unsuspend_card(action_hash: ActionHash) -> ExternResult<Record> {
    let hash_str = action_hash.to_string();
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or_else(|| to_wasm_error(srs_errors::card_not_found(&hash_str)))?;

    let mut card: ReviewCard = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| to_wasm_error(
            praxis_core::errors::errors::invalid_entry("ReviewCard", &hash_str)
        ))?;

    card.status = CardStatus::Review;
    card.modified_at = current_time()?;

    let new_hash = update_entry(action_hash, card)?;

    get(new_hash, GetOptions::default())?
        .ok_or_else(|| to_wasm_error(
            praxis_core::errors::errors::update_failed("ReviewCard", "unsuspend")
        ))
}

/// Bury a card until tomorrow
#[hdk_extern]
pub fn bury_card(action_hash: ActionHash) -> ExternResult<Record> {
    let hash_str = action_hash.to_string();
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or_else(|| to_wasm_error(srs_errors::card_not_found(&hash_str)))?;

    let mut card: ReviewCard = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| to_wasm_error(
            praxis_core::errors::errors::invalid_entry("ReviewCard", &hash_str)
        ))?;

    let now = current_time()?;
    card.status = CardStatus::Buried;
    card.due_at = now + (MINUTES_PER_DAY as i64 * 60 * 1_000_000); // Tomorrow in microseconds
    card.modified_at = now;

    let new_hash = update_entry(action_hash, card)?;

    get(new_hash, GetOptions::default())?
        .ok_or_else(|| to_wasm_error(
            praxis_core::errors::errors::update_failed("ReviewCard", "bury")
        ))
}

// ============== Review Processing ==============

/// Submit a review for a card
#[hdk_extern]
pub fn submit_review(input: SubmitReviewInput) -> ExternResult<Record> {
    // Validate input quality (must be 0-5 for SM-2)
    if input.quality > 5 {
        return Err(to_wasm_error(srs_errors::invalid_recall_quality(input.quality)));
    }

    let now = current_time()?;
    let learner = agent_info()?.agent_initial_pubkey;
    let hash_str = input.card_hash.to_string();

    // Get the card
    let record = get(input.card_hash.clone(), GetOptions::default())?
        .ok_or_else(|| to_wasm_error(srs_errors::card_not_found(&hash_str)))?;

    let mut card: ReviewCard = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| to_wasm_error(
            praxis_core::errors::errors::invalid_entry("ReviewCard", &hash_str)
        ))?;

    // Check if card is suspended (shouldn't be reviewed)
    if card.status == CardStatus::Suspended {
        return Err(to_wasm_error(
            EduNetError::new(
                praxis_core::errors::ErrorCode::InvalidEntityState,
                "ReviewCard",
                "review",
                "Cannot review a suspended card"
            )
            .with_context(&hash_str)
            .with_hint("Unsuspend the card first using unsuspend_card")
        ));
    }

    // Get learner config
    let config = get_or_create_config(())?;
    let config_entry: SrsConfig = config.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| to_wasm_error(
            praxis_core::errors::errors::invalid_entry("SrsConfig", "default")
        ))?;

    // Store old values for the event
    let ease_before = card.ease_factor_permille;
    let interval_before = card.interval_minutes;

    // Apply SM-2 algorithm
    let (new_ease, new_interval, new_reps, new_status) = sm2_algorithm(
        input.quality,
        card.ease_factor_permille,
        card.interval_minutes,
        card.repetitions,
        &config_entry,
    );

    // Update card
    card.ease_factor_permille = new_ease;
    card.interval_minutes = new_interval;
    card.repetitions = new_reps;
    card.status = new_status.clone();
    card.last_reviewed_at = Some(now);
    card.due_at = now + (new_interval as i64 * 60 * 1_000_000); // Convert minutes to microseconds
    card.total_reviews += 1;

    if input.quality >= 3 {
        card.correct_count += 1;
    } else {
        card.incorrect_count += 1;
        if card.repetitions > 0 {
            card.lapses += 1;
        }
    }

    // Update average time
    if card.total_reviews == 1 {
        card.avg_time_seconds = (input.response_time_ms / 1000) as u32;
    } else {
        let total_time = card.avg_time_seconds * (card.total_reviews - 1);
        card.avg_time_seconds = (total_time + (input.response_time_ms / 1000) as u32) / card.total_reviews;
    }

    // Check for leech
    if card.lapses >= config_entry.leech_threshold {
        match config_entry.leech_action {
            LeechAction::Suspend => {
                card.status = CardStatus::Suspended;
                card.tags.push("leech".to_string());
            }
            LeechAction::TagOnly => {
                if !card.tags.contains(&"leech".to_string()) {
                    card.tags.push("leech".to_string());
                }
            }
            LeechAction::Simplify => {
                card.tags.push("leech".to_string());
                // Could trigger card simplification in the future
            }
        }
    }

    card.modified_at = now;

    // Create review event
    let event = ReviewEvent {
        card_hash: input.card_hash.clone(),
        learner: learner.clone(),
        quality: input.quality,
        response_time_ms: input.response_time_ms,
        ease_before_permille: ease_before,
        ease_after_permille: new_ease,
        interval_before_minutes: interval_before,
        interval_after_minutes: new_interval,
        reviewed_at: now,
    };

    let event_hash = create_entry(EntryTypes::ReviewEvent(event))?;

    // Link event to card
    create_link(
        input.card_hash.clone(),
        event_hash,
        LinkTypes::CardToEvents,
        (),
    )?;

    // Update card
    let new_hash = update_entry(input.card_hash, card)?;

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated card".into())))
}

/// Input for submitting a review
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitReviewInput {
    pub card_hash: ActionHash,
    pub quality: u8,  // 0-5
    pub response_time_ms: u32,
}

/// Get review history for a card
#[hdk_extern]
pub fn get_card_reviews(card_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(card_hash, LinkTypes::CardToEvents)?,
        GetStrategy::Local,
    )?;

    let mut events = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                events.push(record);
            }
        }
    }

    Ok(events)
}

// ============== Review Sessions ==============

/// Start a new review session
#[hdk_extern]
pub fn start_session(pod_hash: Option<ActionHash>) -> ExternResult<Record> {
    let now = current_time()?;
    let learner = agent_info()?.agent_initial_pubkey;

    let session = ReviewSession {
        learner,
        cards_reviewed: Vec::new(),
        duration_seconds: 0,
        correct_count: 0,
        incorrect_count: 0,
        started_at: now,
        ended_at: now, // Will be updated when session ends
        pod_hash,
    };

    let action_hash = create_entry(EntryTypes::ReviewSession(session))?;

    // Link to learner
    let anchor_hash = learner_anchor()?;
    create_link(
        anchor_hash,
        action_hash.clone(),
        LinkTypes::LearnerToSessions,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created session".into())))
}

/// End a review session
#[hdk_extern]
pub fn end_session(input: EndSessionInput) -> ExternResult<Record> {
    let now = current_time()?;

    let record = get(input.session_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Session not found".into())))?;

    let mut session: ReviewSession = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid session entry".into())))?;

    session.cards_reviewed = input.cards_reviewed;
    session.correct_count = input.correct_count;
    session.incorrect_count = input.incorrect_count;
    session.ended_at = now;
    session.duration_seconds = ((now - session.started_at) / 1_000_000) as u32;

    // Save duration before session is moved
    let duration = session.duration_seconds;

    let new_hash = update_entry(input.session_hash, session)?;

    // Update daily stats
    update_daily_stats(input.correct_count, input.incorrect_count, duration)?;

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated session".into())))
}

/// Input for ending a session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EndSessionInput {
    pub session_hash: ActionHash,
    pub cards_reviewed: Vec<ActionHash>,
    pub correct_count: u32,
    pub incorrect_count: u32,
}

/// Get recent sessions
#[hdk_extern]
pub fn get_my_sessions(limit: u32) -> ExternResult<Vec<Record>> {
    let anchor_hash = learner_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash, LinkTypes::LearnerToSessions)?,
        GetStrategy::Local,
    )?;

    let mut sessions: Vec<(Record, i64)> = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(session) = record.entry().to_app_option::<ReviewSession>().ok().flatten() {
                    sessions.push((record, session.started_at));
                }
            }
        }
    }

    // Sort by start time (newest first)
    sessions.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(sessions.into_iter().take(limit as usize).map(|(r, _)| r).collect())
}

// ============== Statistics ==============

/// Update daily stats
fn update_daily_stats(correct: u32, incorrect: u32, duration_seconds: u32) -> ExternResult<()> {
    let now = current_time()?;
    let learner = agent_info()?.agent_initial_pubkey;

    // Calculate date as YYYYMMDD
    // Simple calculation - could be improved with timezone awareness
    let days_since_epoch = (now / (24 * 60 * 60 * 1_000_000)) as u32;
    let date = 19700101 + days_since_epoch; // Approximate

    // Try to find existing stats for today
    let anchor_hash = learner_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor_hash.clone(), LinkTypes::LearnerToStats)?,
        GetStrategy::Local,
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(mut stats) = record.entry().to_app_option::<DailyStats>().ok().flatten() {
                    if stats.date == date {
                        // Update existing stats
                        stats.reviews += correct + incorrect;
                        stats.study_time_seconds += duration_seconds;
                        let total_attempts = stats.reviews;
                        let total_correct = (stats.retention_permille as u32 * (total_attempts - correct - incorrect) / 1000) + correct;
                        stats.retention_permille = if total_attempts > 0 {
                            ((total_correct * 1000) / total_attempts) as u16
                        } else {
                            0
                        };

                        update_entry(action_hash, stats)?;
                        return Ok(());
                    }
                }
            }
        }
    }

    // Create new stats for today
    let retention = if correct + incorrect > 0 {
        ((correct * 1000) / (correct + incorrect)) as u16
    } else {
        0
    };

    let stats = DailyStats {
        learner: learner.clone(),
        date,
        new_cards: 0,
        reviews: correct + incorrect,
        relearns: 0,
        study_time_seconds: duration_seconds,
        retention_permille: retention,
        streak_days: 1, // Would need to check previous day
    };

    let action_hash = create_entry(EntryTypes::DailyStats(stats))?;

    create_link(
        anchor_hash,
        action_hash,
        LinkTypes::LearnerToStats,
        (),
    )?;

    Ok(())
}

/// Get statistics for a date range
#[hdk_extern]
pub fn get_stats(input: GetStatsInput) -> ExternResult<Vec<Record>> {
    let anchor_hash = learner_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash, LinkTypes::LearnerToStats)?,
        GetStrategy::Local,
    )?;

    let mut stats = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(daily_stats) = record.entry().to_app_option::<DailyStats>().ok().flatten() {
                    if daily_stats.date >= input.from_date && daily_stats.date <= input.to_date {
                        stats.push(record);
                    }
                }
            }
        }
    }

    Ok(stats)
}

/// Input for getting stats
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetStatsInput {
    pub from_date: u32, // YYYYMMDD
    pub to_date: u32,   // YYYYMMDD
}

/// Get forecast of upcoming reviews
#[hdk_extern]
pub fn get_forecast(days: u32) -> ExternResult<Vec<ReviewForecast>> {
    let now = current_time()?;
    let all_cards = get_my_cards(())?;

    let mut forecasts: Vec<ReviewForecast> = Vec::new();

    for day in 0..days {
        let day_start = now + (day as i64 * 24 * 60 * 60 * 1_000_000);
        let day_end = day_start + (24 * 60 * 60 * 1_000_000);

        let mut review_count = 0u32;
        let mut new_count = 0u32;

        for record in &all_cards {
            if let Some(card) = record.entry().to_app_option::<ReviewCard>().ok().flatten() {
                if card.due_at >= day_start && card.due_at < day_end {
                    if card.status == CardStatus::New {
                        new_count += 1;
                    } else {
                        review_count += 1;
                    }
                }
            }
        }

        // Estimate time (assume 30 seconds per review, 60 seconds per new card)
        let estimated_minutes = (review_count * 30 + new_count * 60) / 60;

        // Calculate date
        let days_since_epoch = ((now / (24 * 60 * 60 * 1_000_000)) as u32) + day;
        let date = 19700101 + days_since_epoch;

        forecasts.push(ReviewForecast {
            date,
            review_count,
            new_count,
            estimated_minutes,
        });
    }

    Ok(forecasts)
}

// ============== Configuration ==============

/// Get or create learner's SRS configuration
#[hdk_extern]
pub fn get_or_create_config(_: ()) -> ExternResult<Record> {
    let learner = agent_info()?.agent_initial_pubkey;
    let anchor_hash = learner_anchor()?;

    // Check for existing config
    let links = get_links(
        LinkQuery::try_new(anchor_hash.clone(), LinkTypes::LearnerToConfig)?,
        GetStrategy::Local,
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                return Ok(record);
            }
        }
    }

    // Create default config
    let now = current_time()?;
    let config = SrsConfig {
        learner,
        new_cards_per_day: 20,
        learning_steps_minutes: vec![1, 10],
        graduating_interval_days: 1,
        easy_interval_days: 4,
        max_reviews_per_day: 100,
        easy_bonus_permille: 1300,
        interval_modifier_permille: 1000,
        max_interval_days: 365,
        relearning_steps_minutes: vec![10],
        lapse_min_interval_days: 1,
        leech_threshold: 8,
        leech_action: LeechAction::TagOnly,
        day_start_hour: 4, // 4 AM
        timezone_offset_minutes: 0,
        created_at: now,
        modified_at: now,
    };

    let action_hash = create_entry(EntryTypes::SrsConfig(config))?;

    create_link(
        anchor_hash,
        action_hash.clone(),
        LinkTypes::LearnerToConfig,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created config".into())))
}

/// Update SRS configuration
#[hdk_extern]
pub fn update_config(input: UpdateConfigInput) -> ExternResult<Record> {
    let record = get(input.config_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Config not found".into())))?;

    let mut config: SrsConfig = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid config entry".into())))?;

    // Apply updates
    if let Some(v) = input.new_cards_per_day { config.new_cards_per_day = v; }
    if let Some(v) = input.learning_steps_minutes { config.learning_steps_minutes = v; }
    if let Some(v) = input.graduating_interval_days { config.graduating_interval_days = v; }
    if let Some(v) = input.easy_interval_days { config.easy_interval_days = v; }
    if let Some(v) = input.max_reviews_per_day { config.max_reviews_per_day = v; }
    if let Some(v) = input.max_interval_days { config.max_interval_days = v; }
    if let Some(v) = input.leech_threshold { config.leech_threshold = v; }
    if let Some(v) = input.leech_action { config.leech_action = v; }
    if let Some(v) = input.day_start_hour { config.day_start_hour = v; }
    if let Some(v) = input.timezone_offset_minutes { config.timezone_offset_minutes = v; }

    config.modified_at = current_time()?;

    let new_hash = update_entry(input.config_hash, config)?;

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated config".into())))
}

/// Input for updating config
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateConfigInput {
    pub config_hash: ActionHash,
    pub new_cards_per_day: Option<u32>,
    pub learning_steps_minutes: Option<Vec<u32>>,
    pub graduating_interval_days: Option<u32>,
    pub easy_interval_days: Option<u32>,
    pub max_reviews_per_day: Option<u32>,
    pub max_interval_days: Option<u32>,
    pub leech_threshold: Option<u32>,
    pub leech_action: Option<LeechAction>,
    pub day_start_hour: Option<u8>,
    pub timezone_offset_minutes: Option<i16>,
}

// ============== Deck Management ==============

/// Create a new deck
#[hdk_extern]
pub fn create_deck(input: CreateDeckInput) -> ExternResult<Record> {
    let now = current_time()?;
    let owner = agent_info()?.agent_initial_pubkey;

    let deck = Deck {
        name: input.name,
        description: input.description,
        owner: owner.clone(),
        parent: input.parent,
        config_override: None,
        is_public: input.is_public,
        tags: input.tags,
        created_at: now,
    };

    let action_hash = create_entry(EntryTypes::Deck(deck))?;

    // Link to owner
    let anchor_hash = learner_anchor()?;
    create_link(
        anchor_hash,
        action_hash.clone(),
        LinkTypes::LearnerToDecks,
        (),
    )?;

    // Link to all decks if public
    if input.is_public {
        let all_decks_path = Path::from("srs.all_decks");
        let typed_path = all_decks_path.typed(LinkTypes::AllDecks)?;
        typed_path.ensure()?;

        create_link(
            typed_path.path.path_entry_hash()?,
            action_hash.clone(),
            LinkTypes::AllDecks,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created deck".into())))
}

/// Input for creating a deck
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateDeckInput {
    pub name: String,
    pub description: String,
    pub parent: Option<ActionHash>,
    pub is_public: bool,
    pub tags: Vec<String>,
}

/// Get all decks for current learner
#[hdk_extern]
pub fn get_my_decks(_: ()) -> ExternResult<Vec<Record>> {
    let anchor_hash = learner_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash, LinkTypes::LearnerToDecks)?,
        GetStrategy::Local,
    )?;

    let mut decks = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                decks.push(record);
            }
        }
    }

    Ok(decks)
}

/// Get all public decks
#[hdk_extern]
pub fn get_public_decks(_: ()) -> ExternResult<Vec<Record>> {
    let all_decks_path = Path::from("srs.all_decks");
    let typed_path = all_decks_path.typed(LinkTypes::AllDecks)?;
    typed_path.ensure()?;
    let path_hash = typed_path.path.path_entry_hash()?;

    let links = get_links(
        LinkQuery::try_new(path_hash, LinkTypes::AllDecks)?,
        GetStrategy::Local,
    )?;

    let mut decks = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                decks.push(record);
            }
        }
    }

    Ok(decks)
}

/// Get cards in a deck
#[hdk_extern]
pub fn get_deck_cards(deck_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(deck_hash, LinkTypes::DeckToCards)?,
        GetStrategy::Local,
    )?;

    let mut cards = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                cards.push(record);
            }
        }
    }

    Ok(cards)
}

/// Add a card to a deck
#[hdk_extern]
pub fn add_card_to_deck(input: AddCardToDeckInput) -> ExternResult<()> {
    create_link(
        input.deck_hash,
        input.card_hash,
        LinkTypes::DeckToCards,
        (),
    )?;
    Ok(())
}

/// Input for adding card to deck
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AddCardToDeckInput {
    pub deck_hash: ActionHash,
    pub card_hash: ActionHash,
}

// ============== Unit Tests ==============

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sm2_perfect_response() {
        let config = SrsConfig {
            learner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            new_cards_per_day: 20,
            learning_steps_minutes: vec![1, 10],
            graduating_interval_days: 1,
            easy_interval_days: 4,
            max_reviews_per_day: 100,
            easy_bonus_permille: 1300,
            interval_modifier_permille: 1000,
            max_interval_days: 365,
            relearning_steps_minutes: vec![10],
            lapse_min_interval_days: 1,
            leech_threshold: 8,
            leech_action: LeechAction::TagOnly,
            day_start_hour: 4,
            timezone_offset_minutes: 0,
            created_at: 0,
            modified_at: 0,
        };

        // Perfect response on first review
        let (ease, interval, reps, status) = sm2_algorithm(5, 2500, 0, 0, &config);

        // Should graduate with interval of 1 day
        assert_eq!(reps, 1);
        assert_eq!(interval, 1 * MINUTES_PER_DAY);
        assert_eq!(status, CardStatus::Review);
        // Ease should increase slightly for perfect response
        assert!(ease > 2500);
    }

    #[test]
    fn test_sm2_failed_response() {
        let config = SrsConfig {
            learner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            new_cards_per_day: 20,
            learning_steps_minutes: vec![1, 10],
            graduating_interval_days: 1,
            easy_interval_days: 4,
            max_reviews_per_day: 100,
            easy_bonus_permille: 1300,
            interval_modifier_permille: 1000,
            max_interval_days: 365,
            relearning_steps_minutes: vec![10],
            lapse_min_interval_days: 1,
            leech_threshold: 8,
            leech_action: LeechAction::TagOnly,
            day_start_hour: 4,
            timezone_offset_minutes: 0,
            created_at: 0,
            modified_at: 0,
        };

        // Failed response (quality = 2)
        let (ease, interval, reps, status) = sm2_algorithm(2, 2500, 6 * MINUTES_PER_DAY, 3, &config);

        // Should reset to learning phase
        assert_eq!(reps, 0);
        assert_eq!(interval, 1); // First learning step
        assert_eq!(status, CardStatus::Learning);
        // Ease should decrease for failed response
        assert!(ease < 2500);
    }

    #[test]
    fn test_sm2_ease_bounds() {
        let config = SrsConfig {
            learner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            new_cards_per_day: 20,
            learning_steps_minutes: vec![1, 10],
            graduating_interval_days: 1,
            easy_interval_days: 4,
            max_reviews_per_day: 100,
            easy_bonus_permille: 1300,
            interval_modifier_permille: 1000,
            max_interval_days: 365,
            relearning_steps_minutes: vec![10],
            lapse_min_interval_days: 1,
            leech_threshold: 8,
            leech_action: LeechAction::TagOnly,
            day_start_hour: 4,
            timezone_offset_minutes: 0,
            created_at: 0,
            modified_at: 0,
        };

        // Test minimum ease bound (multiple failures)
        let (ease, _, _, _) = sm2_algorithm(0, MIN_EASE_PERMILLE, 1, 0, &config);
        assert!(ease >= MIN_EASE_PERMILLE);

        // Test maximum ease bound (many perfect responses)
        let (ease, _, _, _) = sm2_algorithm(5, MAX_EASE_PERMILLE, 30 * MINUTES_PER_DAY, 10, &config);
        assert!(ease <= MAX_EASE_PERMILLE);
    }

    #[test]
    fn test_error_message_formatting() {
        // Test that our structured errors produce readable messages
        let err = srs_errors::card_not_found("Qm123abc");
        let msg = err.to_message();

        assert!(msg.contains("[E100]"), "Should contain error code E100");
        assert!(msg.contains("ReviewCard"), "Should mention entity type");
        assert!(msg.contains("Qm123abc"), "Should contain context");
        assert!(msg.contains("Hint:"), "Should include a hint");
    }

    #[test]
    fn test_invalid_quality_error() {
        let err = srs_errors::invalid_recall_quality(7);
        let msg = err.to_message();

        assert!(msg.contains("[E202]"), "Should be OutOfRange error");
        assert!(msg.contains("quality"), "Should mention the field");
        assert!(msg.contains("7"), "Should show the invalid value");
        assert!(msg.contains("0-5"), "Should show valid range");
    }

    #[test]
    fn test_deck_not_found_error() {
        let err = srs_errors::deck_not_found("uhCDk123");
        let msg = err.to_message();

        assert!(msg.contains("Deck"), "Should mention Deck entity");
        assert!(msg.contains("uhCDk123"), "Should include the hash");
    }

    #[test]
    fn test_session_error() {
        let err = srs_errors::session_not_found("session123");
        assert_eq!(err.entity_type, "ReviewSession");
        assert!(err.context.as_ref().unwrap().contains("session123"));
    }

    // ---- Pure SM-2 helper tests ----

    #[test]
    fn test_sm2_ease_factor_quality_0() {
        // Quality 0 (complete blackout): adjustment = 0.1 - 5*(0.08 + 5*0.02) = 0.1 - 0.9 = -0.8
        let ef = sm2_ease_factor(2500, 0);
        // 2.5 - 0.8 = 1.7 => 1700
        assert_eq!(ef, 1700);
    }

    #[test]
    fn test_sm2_ease_factor_quality_3() {
        // Quality 3 (correct with difficulty): adjustment = 0.1 - 2*(0.08 + 2*0.02) = 0.1 - 0.24 = -0.14
        let ef = sm2_ease_factor(2500, 3);
        // 2.5 - 0.14 = 2.36 => 2360
        assert_eq!(ef, 2360);
    }

    #[test]
    fn test_sm2_ease_factor_quality_5() {
        // Quality 5 (perfect): adjustment = 0.1 - 0*(0.08 + 0*0.02) = 0.1
        let ef = sm2_ease_factor(2500, 5);
        // 2.5 + 0.1 = 2.6 => 2600
        assert_eq!(ef, 2600);
    }

    #[test]
    fn test_sm2_ease_factor_clamps_at_minimum() {
        // Even with quality 0 from already-low ease, should not go below 1300
        let ef = sm2_ease_factor(MIN_EASE_PERMILLE, 0);
        assert_eq!(ef, MIN_EASE_PERMILLE);
    }

    #[test]
    fn test_sm2_ease_factor_clamps_at_maximum() {
        let ef = sm2_ease_factor(MAX_EASE_PERMILLE, 5);
        assert_eq!(ef, MAX_EASE_PERMILLE);
    }

    #[test]
    fn test_sm2_interval_first_review() {
        // First successful review (rep 0) -> graduating interval (1 day)
        let interval = sm2_interval(0, 0, 2500, 1, 365, 1000);
        assert_eq!(interval, 1 * MINUTES_PER_DAY);
    }

    #[test]
    fn test_sm2_interval_second_review() {
        // Second successful review (rep 1) -> 6 days
        let interval = sm2_interval(1, MINUTES_PER_DAY, 2500, 1, 365, 1000);
        assert_eq!(interval, 6 * MINUTES_PER_DAY);
    }

    #[test]
    fn test_sm2_interval_progression() {
        // Third review (rep 2): 6 days * 2.5 = 15 days
        let interval = sm2_interval(2, 6 * MINUTES_PER_DAY, 2500, 1, 365, 1000);
        assert_eq!(interval, 15 * MINUTES_PER_DAY);
    }

    #[test]
    fn test_sm2_interval_respects_max() {
        // Even with high ease and long interval, should not exceed max_interval_days
        let interval = sm2_interval(5, 300 * MINUTES_PER_DAY, 3000, 1, 365, 1000);
        assert!(interval <= 365 * MINUTES_PER_DAY);
    }

    #[test]
    fn test_sm2_interval_modifier_scales() {
        // With 1200 permille modifier (1.2x), interval should be 20% longer
        let base = sm2_interval(0, 0, 2500, 1, 365, 1000);
        let modified = sm2_interval(0, 0, 2500, 1, 365, 1200);
        assert_eq!(modified, (base as f64 * 1.2) as u32);
    }

    #[test]
    fn test_is_leech_at_threshold() {
        assert!(is_leech(8, 8));
    }

    #[test]
    fn test_is_leech_below_threshold() {
        assert!(!is_leech(7, 8));
    }

    #[test]
    fn test_is_leech_above_threshold() {
        assert!(is_leech(10, 8));
    }

    #[test]
    fn test_is_leech_zero_threshold() {
        // Edge case: threshold 0 means every card is a leech
        assert!(is_leech(0, 0));
    }
}
