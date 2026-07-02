// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Enhanced Multi-Round Reputation Tracker v2
//!
//! Provides EMA-based reputation tracking with sudden change detection
//! for catching sleeper agents in Byzantine-robust federated learning.
//!
//! Key Features:
//! - Exponential Moving Average (EMA) for smooth reputation decay
//! - Sudden change detection for sleeper agent activation
//! - Multi-round history tracking
//! - Configurable thresholds (standard, aggressive, conservative)
//! - Integration with CBD and gradient_storage zomes
//!
//! Author: Luminous Dynamics Research Team
//! Date: December 2025

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// Fixed-Point Arithmetic (Q16.16 format for DHT determinism)
// =============================================================================

/// Fixed-point type: Q16.16 (16 bits integer, 16 bits fraction)
pub type FixedPoint = i32;

/// Scale factor for Q16.16
const FP_SCALE: i32 = 65536; // 2^16

/// Convert f64 to fixed-point
#[inline]
fn fp_from_f64(x: f64) -> FixedPoint {
    (x * FP_SCALE as f64) as FixedPoint
}

/// Convert fixed-point to f64 (for display/logging only)
#[inline]
fn fp_to_f64(x: FixedPoint) -> f64 {
    x as f64 / FP_SCALE as f64
}

/// Multiply two fixed-point numbers
#[inline]
fn fp_mul(a: FixedPoint, b: FixedPoint) -> FixedPoint {
    ((a as i64 * b as i64) / FP_SCALE as i64) as FixedPoint
}

/// Divide two fixed-point numbers
#[inline]
fn fp_div(a: FixedPoint, b: FixedPoint) -> FixedPoint {
    if b == 0 {
        return if a >= 0 { i32::MAX } else { i32::MIN };
    }
    ((a as i64 * FP_SCALE as i64) / b as i64) as FixedPoint
}

/// Absolute value
#[inline]
fn fp_abs(x: FixedPoint) -> FixedPoint {
    if x < 0 { -x } else { x }
}

/// Clamp to [0, 1] range in fixed-point
#[inline]
fn fp_clamp_01(x: FixedPoint) -> FixedPoint {
    if x < 0 { 0 }
    else if x > FP_SCALE { FP_SCALE }
    else { x }
}

// =============================================================================
// Reputation State
// =============================================================================

/// Reputation state classification
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReputationState {
    /// Reputation >= 0.7 - Node is trusted
    Trusted,
    /// 0.3 <= Reputation < 0.7 - Node is suspicious
    Suspicious,
    /// Reputation < 0.3 - Node is untrusted (effectively blacklisted)
    Untrusted,
}

impl ReputationState {
    /// Get state from reputation score (fixed-point)
    pub fn from_reputation(reputation: FixedPoint) -> Self {
        let trusted_threshold = fp_from_f64(0.7);
        let untrusted_threshold = fp_from_f64(0.3);

        if reputation >= trusted_threshold {
            ReputationState::Trusted
        } else if reputation >= untrusted_threshold {
            ReputationState::Suspicious
        } else {
            ReputationState::Untrusted
        }
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Reputation tracker configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReputationConfig {
    /// EMA decay factor (alpha): higher = faster adaptation
    /// Standard: 0.3, Aggressive: 0.5, Conservative: 0.15
    pub ema_alpha: FixedPoint,

    /// Threshold for sudden change detection (z-score)
    pub sudden_change_threshold: FixedPoint,

    /// Threshold below which node becomes suspicious
    pub suspicious_threshold: FixedPoint,

    /// Threshold below which node becomes untrusted
    pub untrusted_threshold: FixedPoint,

    /// Number of rounds of history to maintain
    pub history_window: u32,

    /// Penalty factor for Byzantine behavior detection
    pub byzantine_penalty: FixedPoint,

    /// Recovery rate per honest round
    pub recovery_rate: FixedPoint,
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self::standard()
    }
}

impl ReputationConfig {
    /// Standard configuration
    pub fn standard() -> Self {
        ReputationConfig {
            ema_alpha: fp_from_f64(0.3),
            sudden_change_threshold: fp_from_f64(2.5),
            suspicious_threshold: fp_from_f64(0.5),
            untrusted_threshold: fp_from_f64(0.3),
            history_window: 10,
            byzantine_penalty: fp_from_f64(0.3),
            recovery_rate: fp_from_f64(0.05),
        }
    }

    /// Aggressive configuration (catches more, higher false positives)
    pub fn aggressive() -> Self {
        ReputationConfig {
            ema_alpha: fp_from_f64(0.5),
            sudden_change_threshold: fp_from_f64(2.0),
            suspicious_threshold: fp_from_f64(0.6),
            untrusted_threshold: fp_from_f64(0.4),
            history_window: 5,
            byzantine_penalty: fp_from_f64(0.4),
            recovery_rate: fp_from_f64(0.03),
        }
    }

    /// Conservative configuration (fewer false positives, may miss some)
    pub fn conservative() -> Self {
        ReputationConfig {
            ema_alpha: fp_from_f64(0.15),
            sudden_change_threshold: fp_from_f64(3.0),
            suspicious_threshold: fp_from_f64(0.4),
            untrusted_threshold: fp_from_f64(0.2),
            history_window: 15,
            byzantine_penalty: fp_from_f64(0.2),
            recovery_rate: fp_from_f64(0.08),
        }
    }
}

// =============================================================================
// Reputation Entry (DHT-stored)
// =============================================================================

/// Single round's behavior record
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RoundBehavior {
    /// Round number
    pub round: u64,
    /// Z-score for this round
    pub z_score: FixedPoint,
    /// Whether flagged this round
    pub was_flagged: bool,
    /// Reputation after this round
    pub reputation_after: FixedPoint,
}

/// Enhanced reputation entry stored in DHT
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ReputationEntryV2 {
    /// Agent public key of the node
    pub node_id: AgentPubKey,

    /// Current reputation score (0.0 - 1.0 in fixed-point)
    pub reputation: FixedPoint,

    /// Current state classification
    pub state: ReputationState,

    /// EMA of z-scores (for sudden change detection)
    pub z_score_ema: FixedPoint,

    /// Variance of z-scores (for sudden change detection)
    pub z_score_variance: FixedPoint,

    /// Number of times flagged
    pub times_flagged: u32,

    /// Total rounds participated
    pub rounds_participated: u32,

    /// Recent behavior history (last N rounds)
    pub history: Vec<RoundBehavior>,

    /// Current round number
    pub current_round: u64,

    /// Whether currently flagged
    pub is_flagged: bool,

    /// Reason for current flag (if any)
    pub flag_reason: Option<String>,

    /// Timestamp of creation
    pub created_at: Timestamp,

    /// Timestamp of last update
    pub updated_at: Timestamp,
}

// =============================================================================
// Reputation Update Result
// =============================================================================

/// Result of a reputation update
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReputationUpdate {
    /// Previous reputation
    pub old_reputation: FixedPoint,
    /// New reputation
    pub new_reputation: FixedPoint,
    /// Whether node was flagged this round
    pub flagged: bool,
    /// Reason for flagging (if applicable)
    pub reason: Option<String>,
    /// Whether sudden change was detected
    pub sudden_change_detected: bool,
    /// New state after update
    pub new_state: ReputationState,
}

// =============================================================================
// Input Types
// =============================================================================

/// Input for updating a node's reputation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateInput {
    /// Node to update
    pub node_id: AgentPubKey,
    /// Z-score from current round's behavior
    pub z_score: FixedPoint,
    /// Optional: force flag with reason
    pub force_flag: Option<String>,
}

/// Input for batch updates
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchUpdateInput {
    /// Round number
    pub round: u64,
    /// List of updates
    pub updates: Vec<UpdateInput>,
}

/// Input for getting detection metrics
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MetricsInput {
    /// Known Byzantine node IDs (ground truth)
    pub byzantine_nodes: Vec<AgentPubKey>,
}

// =============================================================================
// Output Types
// =============================================================================

/// Detection metrics result
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DetectionMetrics {
    pub true_positives: u32,
    pub false_positives: u32,
    pub false_negatives: u32,
    pub true_negatives: u32,
    pub precision: FixedPoint,
    pub recall: FixedPoint,
    pub f1_score: FixedPoint,
}

/// Summary of reputation states
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReputationSummary {
    pub total_nodes: u32,
    pub trusted_count: u32,
    pub suspicious_count: u32,
    pub untrusted_count: u32,
    pub flagged_count: u32,
    pub average_reputation: FixedPoint,
}

// =============================================================================
// Entry Types
// =============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    ReputationEntryV2(ReputationEntryV2),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Link from node agent key to reputation entry
    NodeToReputation,
    /// Link from round anchor to all entries updated that round
    RoundToEntries,
    /// Link to flagged nodes
    FlaggedNodes,
    /// Anchor links
    AnchorToNode,
}

// =============================================================================
// Zome Functions
// =============================================================================

/// Initialize the zome
#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

/// Update a node's reputation based on current round behavior
#[hdk_extern]
pub fn update_reputation(input: UpdateInput) -> ExternResult<ReputationUpdate> {
    let config = ReputationConfig::standard();
    update_reputation_with_config(input, config)
}

/// Update with custom configuration
#[hdk_extern]
pub fn update_reputation_configured(
    input: (UpdateInput, ReputationConfig),
) -> ExternResult<ReputationUpdate> {
    let (update_input, config) = input;
    update_reputation_with_config(update_input, config)
}

/// Internal update function
fn update_reputation_with_config(
    input: UpdateInput,
    config: ReputationConfig,
) -> ExternResult<ReputationUpdate> {
    // Get or create existing reputation
    let existing = get_reputation_entry(&input.node_id)?;
    let now = sys_time()?;

    let (mut entry, old_reputation) = match existing {
        Some(e) => {
            let old_rep = e.reputation;
            (e, old_rep)
        }
        None => {
            // New node starts with full reputation
            let new_entry = ReputationEntryV2 {
                node_id: input.node_id.clone(),
                reputation: FP_SCALE, // 1.0
                state: ReputationState::Trusted,
                z_score_ema: 0,
                z_score_variance: fp_from_f64(1.0),
                times_flagged: 0,
                rounds_participated: 0,
                history: Vec::new(),
                current_round: 0,
                is_flagged: false,
                flag_reason: None,
                created_at: now,
                updated_at: now,
            };
            (new_entry, FP_SCALE)
        }
    };

    // Update EMA of z-scores
    // new_ema = alpha * z_score + (1 - alpha) * old_ema
    let alpha = config.ema_alpha;
    let one_minus_alpha = FP_SCALE - alpha;
    let new_z_ema = fp_mul(alpha, input.z_score) + fp_mul(one_minus_alpha, entry.z_score_ema);

    // Update variance using EMA
    // variance_update = (z_score - old_ema)^2
    let deviation = input.z_score - entry.z_score_ema;
    let deviation_sq = fp_mul(deviation, deviation);
    let new_variance = fp_mul(alpha, deviation_sq) + fp_mul(one_minus_alpha, entry.z_score_variance);

    // Sudden change detection
    // Check if |z_score - z_ema| > threshold * sqrt(variance)
    let std_dev = fp_sqrt(new_variance);
    let change_threshold = fp_mul(config.sudden_change_threshold, std_dev);
    let sudden_change = fp_abs(input.z_score - entry.z_score_ema) > change_threshold
        && entry.rounds_participated >= 3; // Need history first

    // Determine if flagged
    let mut flagged = false;
    let mut flag_reason: Option<String> = None;

    // Check for forced flag
    if let Some(reason) = input.force_flag {
        flagged = true;
        flag_reason = Some(reason);
    }

    // Check for sudden change (sleeper agent detection)
    if sudden_change {
        flagged = true;
        flag_reason = Some("sudden_change_detected".to_string());
    }

    // Check for high z-score
    let high_z_threshold = fp_from_f64(3.0);
    if fp_abs(input.z_score) > high_z_threshold {
        flagged = true;
        flag_reason = flag_reason.or(Some("high_z_score".to_string()));
    }

    // Update reputation based on behavior
    let new_reputation = if flagged {
        // Penalty for Byzantine behavior
        let penalty = fp_mul(entry.reputation, config.byzantine_penalty);
        fp_clamp_01(entry.reputation - penalty)
    } else {
        // Slight recovery for honest behavior
        let recovery = fp_mul(FP_SCALE - entry.reputation, config.recovery_rate);
        fp_clamp_01(entry.reputation + recovery)
    };

    // Get new state
    let new_state = ReputationState::from_reputation(new_reputation);

    // Update entry
    entry.reputation = new_reputation;
    entry.state = new_state;
    entry.z_score_ema = new_z_ema;
    entry.z_score_variance = new_variance;
    entry.rounds_participated += 1;
    entry.current_round += 1;
    entry.updated_at = now;

    if flagged {
        entry.times_flagged += 1;
        entry.is_flagged = true;
        entry.flag_reason = flag_reason.clone();
    }

    // Add to history (keep last N rounds)
    let round_behavior = RoundBehavior {
        round: entry.current_round,
        z_score: input.z_score,
        was_flagged: flagged,
        reputation_after: new_reputation,
    };
    entry.history.push(round_behavior);
    if entry.history.len() > config.history_window as usize {
        entry.history.remove(0);
    }

    // Store updated entry
    let _action_hash = create_entry(EntryTypes::ReputationEntryV2(entry.clone()))?;
    let entry_hash = hash_entry(&entry)?;

    // Create link from node ID
    create_link(
        input.node_id.clone(),
        entry_hash.clone(),
        LinkTypes::NodeToReputation,
        (),
    )?;

    // If flagged, add to flagged nodes
    if flagged {
        let flagged_anchor = create_anchor("flagged_nodes")?;
        create_link(
            flagged_anchor,
            entry_hash,
            LinkTypes::FlaggedNodes,
            (),
        )?;
    }

    Ok(ReputationUpdate {
        old_reputation,
        new_reputation,
        flagged,
        reason: flag_reason,
        sudden_change_detected: sudden_change,
        new_state,
    })
}

/// Batch update multiple nodes
#[hdk_extern]
pub fn batch_update(input: BatchUpdateInput) -> ExternResult<Vec<ReputationUpdate>> {
    let config = ReputationConfig::standard();
    let mut results = Vec::with_capacity(input.updates.len());

    for update in input.updates {
        let result = update_reputation_with_config(update, config.clone())?;
        results.push(result);
    }

    Ok(results)
}

/// Get reputation for a specific node
#[hdk_extern]
pub fn get_reputation(node_id: AgentPubKey) -> ExternResult<Option<ReputationEntryV2>> {
    get_reputation_entry(&node_id)
}

/// Get all flagged nodes
#[hdk_extern]
pub fn get_flagged_nodes(_: ()) -> ExternResult<Vec<ReputationEntryV2>> {
    let flagged_anchor = create_anchor("flagged_nodes")?;
    let links = get_links(
        GetLinksInputBuilder::try_new(flagged_anchor, LinkTypes::FlaggedNodes)?.build()
    )?;

    let mut flagged = Vec::new();
    for link in links {
        if let Some(entry_hash) = link.target.into_entry_hash() {
            if let Some(record) = get(entry_hash, GetOptions::default())? {
                if let Some(entry) = record.entry().to_app_option::<ReputationEntryV2>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                    if entry.is_flagged {
                        flagged.push(entry);
                    }
                }
            }
        }
    }

    Ok(flagged)
}

/// Get detection metrics against ground truth
#[hdk_extern]
pub fn get_detection_metrics(input: MetricsInput) -> ExternResult<DetectionMetrics> {
    let flagged = get_flagged_nodes(())?;
    let flagged_ids: std::collections::HashSet<_> = flagged.iter()
        .map(|e| e.node_id.clone())
        .collect();
    let byzantine_set: std::collections::HashSet<_> = input.byzantine_nodes.into_iter().collect();

    let mut tp = 0u32;
    let mut fp = 0u32;
    let mut fn_ = 0u32;

    for node_id in &flagged_ids {
        if byzantine_set.contains(node_id) {
            tp += 1;
        } else {
            fp += 1;
        }
    }

    for node_id in &byzantine_set {
        if !flagged_ids.contains(node_id) {
            fn_ += 1;
        }
    }

    // TN would require knowing all honest nodes, approximated
    let tn = 0u32; // Not computed without full node list

    let precision = if tp + fp > 0 {
        fp_div(fp_from_f64(tp as f64), fp_from_f64((tp + fp) as f64))
    } else {
        0
    };

    let recall = if tp + fn_ > 0 {
        fp_div(fp_from_f64(tp as f64), fp_from_f64((tp + fn_) as f64))
    } else {
        0
    };

    let f1 = if precision + recall > 0 {
        fp_div(
            fp_mul(2 * FP_SCALE, fp_mul(precision, recall)),
            precision + recall
        )
    } else {
        0
    };

    Ok(DetectionMetrics {
        true_positives: tp,
        false_positives: fp,
        false_negatives: fn_,
        true_negatives: tn,
        precision,
        recall,
        f1_score: f1,
    })
}

/// Get summary of all reputation states
#[hdk_extern]
pub fn get_reputation_summary(_: ()) -> ExternResult<ReputationSummary> {
    // In production, would iterate through all known nodes
    // For now, return from flagged nodes
    let flagged = get_flagged_nodes(())?;

    let mut trusted = 0u32;
    let mut suspicious = 0u32;
    let mut untrusted = 0u32;
    let mut total_rep: i64 = 0;

    for entry in &flagged {
        total_rep += entry.reputation as i64;
        match entry.state {
            ReputationState::Trusted => trusted += 1,
            ReputationState::Suspicious => suspicious += 1,
            ReputationState::Untrusted => untrusted += 1,
        }
    }

    let total = flagged.len() as u32;
    let avg_rep = if total > 0 {
        (total_rep / total as i64) as FixedPoint
    } else {
        FP_SCALE // Default to 1.0
    };

    Ok(ReputationSummary {
        total_nodes: total,
        trusted_count: trusted,
        suspicious_count: suspicious,
        untrusted_count: untrusted,
        flagged_count: flagged.len() as u32,
        average_reputation: avg_rep,
    })
}

/// Clear flag for a node (for recovery)
#[hdk_extern]
pub fn clear_flag(node_id: AgentPubKey) -> ExternResult<bool> {
    let existing = get_reputation_entry(&node_id)?;

    if let Some(mut entry) = existing {
        entry.is_flagged = false;
        entry.flag_reason = None;
        entry.updated_at = sys_time()?;

        let _action_hash = create_entry(EntryTypes::ReputationEntryV2(entry.clone()))?;
        let entry_hash = hash_entry(&entry)?;

        create_link(
            node_id,
            entry_hash,
            LinkTypes::NodeToReputation,
            (),
        )?;

        return Ok(true);
    }

    Ok(false)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get reputation entry for a node
fn get_reputation_entry(node_id: &AgentPubKey) -> ExternResult<Option<ReputationEntryV2>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(node_id.clone(), LinkTypes::NodeToReputation)?.build()
    )?;

    // Get most recent entry
    if let Some(link) = links.into_iter().last() {
        if let Some(entry_hash) = link.target.into_entry_hash() {
            if let Some(record) = get(entry_hash, GetOptions::default())? {
                return record.entry().to_app_option()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())));
            }
        }
    }

    Ok(None)
}

/// Create an anchor for linking
fn create_anchor(name: &str) -> ExternResult<EntryHash> {
    // Use anchor function from hdk::hash_path::anchor
    use hdk::hash_path::anchor::anchor;
    anchor(LinkTypes::AnchorToNode, "reputation".to_string(), name.to_string())
}

/// Integer square root approximation for fixed-point
fn fp_sqrt(x: FixedPoint) -> FixedPoint {
    if x <= 0 {
        return 0;
    }

    // Newton-Raphson for sqrt with fixed-point
    // Start with x/2 as initial guess
    let mut guess = x / 2;
    if guess == 0 {
        guess = 1;
    }

    // 5 iterations is enough for Q16.16 precision
    for _ in 0..5 {
        // guess = (guess + x/guess) / 2
        let div = fp_div(x, guess);
        guess = (guess + div) / 2;
    }

    guess
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_basics() {
        let one = fp_from_f64(1.0);
        assert_eq!(one, FP_SCALE);

        let half = fp_from_f64(0.5);
        assert_eq!(half, FP_SCALE / 2);

        let product = fp_mul(half, half);
        assert_eq!(product, FP_SCALE / 4); // 0.25
    }

    #[test]
    fn test_fp_sqrt() {
        let four = fp_from_f64(4.0);
        let sqrt_four = fp_sqrt(four);
        let two = fp_from_f64(2.0);
        // Allow small error due to fixed-point
        assert!((sqrt_four - two).abs() < 100);

        let one = fp_from_f64(1.0);
        let sqrt_one = fp_sqrt(one);
        assert!((sqrt_one - one).abs() < 100);
    }

    #[test]
    fn test_reputation_state_classification() {
        assert_eq!(
            ReputationState::from_reputation(fp_from_f64(0.8)),
            ReputationState::Trusted
        );
        assert_eq!(
            ReputationState::from_reputation(fp_from_f64(0.5)),
            ReputationState::Suspicious
        );
        assert_eq!(
            ReputationState::from_reputation(fp_from_f64(0.2)),
            ReputationState::Untrusted
        );
    }

    #[test]
    fn test_config_modes() {
        let standard = ReputationConfig::standard();
        let aggressive = ReputationConfig::aggressive();
        let conservative = ReputationConfig::conservative();

        // Aggressive should have higher alpha (faster adaptation)
        assert!(aggressive.ema_alpha > standard.ema_alpha);
        assert!(standard.ema_alpha > conservative.ema_alpha);

        // Aggressive should have lower sudden change threshold
        assert!(aggressive.sudden_change_threshold < standard.sudden_change_threshold);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(fp_clamp_01(fp_from_f64(1.5)), FP_SCALE);
        assert_eq!(fp_clamp_01(fp_from_f64(-0.5)), 0);
        assert_eq!(fp_clamp_01(fp_from_f64(0.5)), FP_SCALE / 2);
    }

    #[test]
    fn test_ema_calculation() {
        // Simulate EMA update
        let alpha = fp_from_f64(0.3);
        let one_minus_alpha = FP_SCALE - alpha;

        let old_ema = fp_from_f64(0.0);
        let new_value = fp_from_f64(2.0);

        let new_ema = fp_mul(alpha, new_value) + fp_mul(one_minus_alpha, old_ema);

        // new_ema should be 0.3 * 2.0 + 0.7 * 0.0 = 0.6
        let expected = fp_from_f64(0.6);
        assert!((new_ema - expected).abs() < 100);
    }
}
