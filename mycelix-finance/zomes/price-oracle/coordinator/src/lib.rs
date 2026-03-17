#![deny(unsafe_code)]
//! Price Oracle Coordinator Zome
//!
//! Community-driven price discovery for TEND purchasing power.
//!
//! Flow:
//! 1. Members report real-world prices (`report_price`)
//! 2. Oracle computes trimmed median consensus (`get_consensus_price`)
//! 3. Basket definitions weight items into a composite index (`get_basket_index`)
//! 4. When volatility exceeds threshold, auto-escalate TEND limits
//!    via cross-zome call to `tend/update_oracle_state`
//!
//! Consciousness gating: Citizen+ tier (identity >= 0.25, reputation >= 0.1)
//! required for price reporting. Prevents sybil attacks.

use hdk::prelude::*;
use mycelix_finance_shared::{
    anchor_hash, follow_update_chain, rate_limit_anchor_key,
    verify_governance_or_bootstrap_from_links, GOVERNANCE_AGENTS_ANCHOR,
};

pub use price_oracle_integrity::*;

// =============================================================================
// CONSTANTS
// =============================================================================

const ITEM_REPORTS_ANCHOR_PREFIX: &str = "oracle:item:";
const BASKETS_ANCHOR: &str = "oracle:baskets";
const ALERTS_ANCHOR: &str = "oracle:alerts";
const REPORTER_ANCHOR_PREFIX: &str = "oracle:reporter:";
const ACCURACY_ANCHOR_PREFIX: &str = "oracle:accuracy:";
const ALL_REPORTS_ANCHOR: &str = "oracle:all_reports";

/// Default limit for get_recent_reports queries
const DEFAULT_RECENT_REPORTS_LIMIT: u32 = 20;

/// Maximum allowed limit for get_recent_reports
const MAX_RECENT_REPORTS_LIMIT: u32 = 100;

/// Minimum reporters for a valid consensus
const MIN_REPORTERS_FOR_CONSENSUS: usize = 2;

/// Rate limit: max 10 reports per minute per agent
const REPORT_RATE_LIMIT: u32 = 10;

/// Collected report data for consensus computation
struct ReportData {
    reporter_did: String,
    price: f64,
}

// =============================================================================
// INPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct ReportPriceInput {
    pub item: String,
    pub price_tend: f64,
    pub evidence: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetConsensusInput {
    pub item: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DefineBasketInput {
    pub name: String,
    pub items: Vec<BasketItemInput>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BasketItemInput {
    pub item: String,
    pub weight: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetBasketIndexInput {
    pub basket_name: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetRecentReportsInput {
    #[serde(default = "default_recent_limit")]
    pub limit: u32,
}

fn default_recent_limit() -> u32 {
    DEFAULT_RECENT_REPORTS_LIMIT
}

// =============================================================================
// OUTPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct ConsensusResult {
    pub item: String,
    pub median_price: f64,
    pub reporter_count: u32,
    pub std_dev: f64,
    pub window_start: Timestamp,
    /// Average accuracy of reporters contributing to this consensus (0.0-1.0).
    /// High signal_integrity means reporters have been historically accurate.
    pub signal_integrity: f64,
    /// Whether TEND limit escalation was triggered due to price volatility
    /// exceeding VOLATILITY_ESCALATION_THRESHOLD (20% weekly change).
    #[serde(default)]
    pub tend_escalated: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BasketIndexResult {
    pub basket_name: String,
    pub index: f64,
    pub item_prices: Vec<ItemPriceResult>,
    pub computed_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ItemPriceResult {
    pub item: String,
    pub price: f64,
    pub weight: f64,
    pub weighted_price: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VolatilityResult {
    pub basket_name: String,
    pub current_index: f64,
    pub previous_index: f64,
    pub weekly_change: f64,
    pub recommended_tier: String,
    pub escalated: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReporterAccuracyResult {
    pub reporter_did: String,
    pub accuracy_score: f64,
    pub report_count: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecentReportResult {
    pub id: String,
    pub item_name: String,
    pub price_tend: f64,
    pub evidence: String,
    pub reporter_did: String,
    pub timestamp: i64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetTopReportersInput {
    pub item: String,
    pub limit: u32,
}

// =============================================================================
// ACCURACY HELPERS
// =============================================================================

/// Fetch the current accuracy entry for a reporter, or return a default.
/// Returns (Some(original_action_hash), accuracy) if found, (None, default) if new.
fn get_or_create_accuracy(
    reporter_did: &str,
) -> ExternResult<(Option<ActionHash>, ReporterAccuracy)> {
    let acc_anchor = anchor_hash(&format!("{ACCURACY_ANCHOR_PREFIX}{reporter_did}"))?;
    let links = get_links(
        LinkQuery::try_new(acc_anchor, LinkTypes::ReporterToAccuracy)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(hash.clone())?;
            if let Some(acc) = record
                .entry()
                .to_app_option::<ReporterAccuracy>()
                .ok()
                .flatten()
            {
                // Return the original action hash for update_entry
                return Ok((Some(hash), acc));
            }
        }
    }

    // New reporter — return default
    Ok((
        None,
        ReporterAccuracy {
            reporter_did: reporter_did.to_string(),
            accuracy_score: ACCURACY_INITIAL_SCORE,
            report_count: 0,
            updated_at: sys_time()?,
        },
    ))
}

/// Update a reporter's accuracy score after consensus.
fn update_reporter_accuracy(
    reporter_did: &str,
    report_price: f64,
    consensus_median: f64,
) -> ExternResult<()> {
    let (existing_hash, mut acc) = get_or_create_accuracy(reporter_did)?;

    // Compute accuracy delta: how far was this report from consensus?
    // Guard against division by zero: if consensus_median is near zero,
    // use a small floor to avoid panic.
    let delta = if consensus_median.abs() < 0.0001 {
        (report_price - consensus_median).abs() / 0.0001
    } else {
        (report_price - consensus_median).abs() / consensus_median
    };
    let accuracy_signal = 1.0 - delta.min(1.0);
    let new_score =
        (1.0 - ACCURACY_EMA_ALPHA) * acc.accuracy_score + ACCURACY_EMA_ALPHA * accuracy_signal;

    acc.accuracy_score = new_score.clamp(0.0, 1.0);
    acc.report_count += 1;
    acc.updated_at = sys_time()?;

    if let Some(original_hash) = existing_hash {
        update_entry(original_hash, &EntryTypes::ReporterAccuracy(acc))?;
    } else {
        let action_hash = create_entry(&EntryTypes::ReporterAccuracy(acc))?;
        let acc_anchor = anchor_hash(&format!("{ACCURACY_ANCHOR_PREFIX}{reporter_did}"))?;
        create_link(acc_anchor, action_hash, LinkTypes::ReporterToAccuracy, ())?;
    }

    Ok(())
}

/// Compute the weighted median of (price, weight) pairs.
///
/// Sorts by price, walks through accumulating weight until
/// cumulative weight reaches total/2.
fn weighted_median(entries: &mut [(f64, f64)]) -> Option<f64> {
    if entries.is_empty() {
        return None;
    }
    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_weight: f64 = entries.iter().map(|(_, w)| w).sum();
    if total_weight <= 0.0 {
        return None;
    }
    let half = total_weight / 2.0;

    let mut cumulative = 0.0;
    for (i, &(price, weight)) in entries.iter().enumerate() {
        cumulative += weight;
        if cumulative >= half {
            // If we land exactly on the midpoint and there's a next entry, average.
            // Scale epsilon by total_weight so the comparison is meaningful for
            // large cumulative sums (bare f64::EPSILON is ~1e-16, easily lost).
            if (cumulative - half).abs() < f64::EPSILON * total_weight && i + 1 < entries.len() {
                return Some((price + entries[i + 1].0) / 2.0);
            }
            return Some(price);
        }
    }
    Some(entries.last()?.0)
}

// =============================================================================
// CONSCIOUSNESS GATING
// =============================================================================

/// Verify the caller meets Citizen+ consciousness tier.
///
/// Citizen tier requires:
/// - identity_score >= 0.25 (verified DID)
/// - reputation_score >= 0.10 (some community participation)
///
/// This is checked via cross-zome call to the identity cluster.
/// Falls back to governance agent check if identity cluster unavailable.
fn verify_citizen_tier() -> ExternResult<()> {
    // Try cross-zome consciousness check first
    match call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::from("consciousness_gating"),
        FunctionName::from("check_citizen_tier"),
        None,
        (),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            let passed = result.decode::<bool>().unwrap_or(false);
            if passed {
                Ok(())
            } else {
                Err(wasm_error!(WasmErrorInner::Guest(
                    "Citizen+ tier required for price reporting (identity >= 0.25, reputation >= 0.10)"
                        .into()
                )))
            }
        }
        // Identity cluster unavailable — fall back to governance agent check
        _ => verify_governance_or_bootstrap(),
    }
}

fn verify_governance_or_bootstrap() -> ExternResult<()> {
    let gov_links = get_links(
        LinkQuery::try_new(
            anchor_hash(GOVERNANCE_AGENTS_ANCHOR)?,
            LinkTypes::AnchorLinks,
        )?,
        GetStrategy::default(),
    )?;
    verify_governance_or_bootstrap_from_links(gov_links)
}

// =============================================================================
// PRICE REPORTING
// =============================================================================

/// Submit a price report for an item.
///
/// Requires Citizen+ consciousness tier. Rate-limited to 10/minute.
/// Reports are immutable once submitted — to correct, submit a new report.
#[hdk_extern]
pub fn report_price(input: ReportPriceInput) -> ExternResult<Record> {
    // Consciousness gate
    verify_citizen_tier()?;

    // Rate limit
    let my_info = agent_info()?;
    let now = sys_time()?;
    let rate_key = rate_limit_anchor_key(
        "oracle_report",
        &my_info.agent_initial_pubkey,
        now.as_micros(),
    );
    let rate_links = get_links(
        LinkQuery::try_new(anchor_hash(&rate_key)?, LinkTypes::AnchorLinks)?,
        GetStrategy::default(),
    )?;
    if rate_links.len() as u32 >= REPORT_RATE_LIMIT {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Rate limit exceeded: max 10 reports per minute".into()
        )));
    }

    // Validate
    let item = input.item.to_lowercase().trim().to_string();
    if item.is_empty() || item.len() > 128 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Item name must be 1-128 characters".into()
        )));
    }
    if !input.price_tend.is_finite() || input.price_tend <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Price must be a positive finite number".into()
        )));
    }

    let my_did = format!("did:holo:{}", my_info.agent_initial_pubkey);

    let report = PriceReport {
        item: item.clone(),
        price_tend: input.price_tend,
        evidence: input.evidence,
        reporter_did: my_did.clone(),
        observed_at: now,
        reported_at: now,
    };

    let action_hash = create_entry(&EntryTypes::PriceReport(report))?;

    // Link from item anchor
    let item_anchor = anchor_hash(&format!("{ITEM_REPORTS_ANCHOR_PREFIX}{item}"))?;
    create_link(
        item_anchor,
        action_hash.clone(),
        LinkTypes::ItemToReports,
        (),
    )?;

    // Link from reporter
    let reporter_anchor = anchor_hash(&format!("{REPORTER_ANCHOR_PREFIX}{my_did}"))?;
    create_link(
        reporter_anchor,
        action_hash.clone(),
        LinkTypes::ReporterToReports,
        (),
    )?;

    // Rate limit link
    create_link(
        anchor_hash(&rate_key)?,
        action_hash.clone(),
        LinkTypes::AnchorLinks,
        (),
    )?;

    // Global reports anchor (for get_recent_reports)
    create_link(
        anchor_hash(ALL_REPORTS_ANCHOR)?,
        action_hash.clone(),
        LinkTypes::AnchorLinks,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Failed to get created record".into()
    )))
}

// =============================================================================
// CONSENSUS COMPUTATION
// =============================================================================

/// Fetch the most recent stored consensus price for an item.
///
/// Returns `Ok(Some(median_price))` if a prior consensus exists, `Ok(None)` otherwise.
/// Used by `get_consensus_price` to detect volatility and trigger TEND escalation.
fn get_previous_consensus_price(item: &str) -> ExternResult<Option<f64>> {
    let consensus_anchor = anchor_hash(&format!("{ITEM_REPORTS_ANCHOR_PREFIX}{item}:consensus"))?;
    let links = get_links(
        LinkQuery::try_new(consensus_anchor, LinkTypes::ItemToConsensus)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(prev) = record
                    .entry()
                    .to_app_option::<PriceConsensus>()
                    .ok()
                    .flatten()
                {
                    return Ok(Some(prev.median_price));
                }
            }
        }
    }

    Ok(None)
}

/// Get the accuracy-weighted consensus price for an item.
///
/// 1. Collects all reports within the consensus window (1 week)
/// 2. Discards the top and bottom 10% by count (trimming)
/// 3. Fetches each reporter's accuracy EMA score
/// 4. Computes a weighted median where weight = max(accuracy, 0.05)
/// 5. Updates each reporter's accuracy score based on this consensus
/// 6. Checks weekly price change vs prior consensus — if above
///    VOLATILITY_ESCALATION_THRESHOLD (20%), auto-escalates TEND limits
///    via cross-zome call to `tend/update_oracle_state`
///
/// Requires at least 2 unique reporters for a valid consensus.
#[hdk_extern]
pub fn get_consensus_price(input: GetConsensusInput) -> ExternResult<ConsensusResult> {
    let item = input.item.to_lowercase().trim().to_string();
    let item_anchor = anchor_hash(&format!("{ITEM_REPORTS_ANCHOR_PREFIX}{item}"))?;

    let links = get_links(
        LinkQuery::try_new(item_anchor, LinkTypes::ItemToReports)?,
        GetStrategy::default(),
    )?;

    let now = sys_time()?;
    let now_us = now.as_micros();
    let window_start_us = now_us - CONSENSUS_WINDOW_US;
    let window_start = Timestamp::from_micros(window_start_us);

    // Collect reports with reporter identity (needed for accuracy weighting)
    let mut reports: Vec<ReportData> = Vec::new();
    let mut reporters: std::collections::HashSet<String> = std::collections::HashSet::new();

    for link in &links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(report) = record.entry().to_app_option::<PriceReport>().ok().flatten() {
                    if report.reported_at.as_micros() >= window_start_us {
                        reporters.insert(report.reporter_did.clone());
                        reports.push(ReportData {
                            reporter_did: report.reporter_did,
                            price: report.price_tend,
                        });
                    }
                }
            }
        }
    }

    if reporters.len() < MIN_REPORTERS_FOR_CONSENSUS {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Need at least {MIN_REPORTERS_FOR_CONSENSUS} reporters for consensus, got {}",
            reporters.len()
        ))));
    }

    // Sort by price for trimming
    reports.sort_by(|a, b| {
        a.price
            .partial_cmp(&b.price)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Trim top/bottom 10% by count
    let trim_count = (reports.len() as f64 * TRIM_PERCENT).floor() as usize;
    let trimmed = if trim_count > 0 && reports.len() > trim_count * 2 {
        &reports[trim_count..reports.len() - trim_count]
    } else {
        &reports[..]
    };

    if trimmed.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No valid prices after trimming".into()
        )));
    }

    // Fetch accuracy scores for each reporter in the trimmed set
    let mut accuracy_sum = 0.0;
    let mut accuracy_count = 0u32;
    let mut weighted_entries: Vec<(f64, f64)> = Vec::with_capacity(trimmed.len());

    for report in trimmed {
        let (_, acc) = get_or_create_accuracy(&report.reporter_did)?;
        let mut weight = acc.accuracy_score.max(ACCURACY_MIN_WEIGHT);
        // Sybil resistance: cap newcomer weight until they build history
        if acc.report_count < MIN_REPORTS_FOR_FULL_WEIGHT {
            weight = weight.min(NEWCOMER_WEIGHT_CAP);
        }
        weighted_entries.push((report.price, weight));
        accuracy_sum += acc.accuracy_score;
        accuracy_count += 1;
    }

    // Compute accuracy-weighted median
    let median = weighted_median(&mut weighted_entries).ok_or(wasm_error!(
        WasmErrorInner::Guest("Failed to compute weighted median".into())
    ))?;

    // Standard deviation (unweighted, on trimmed prices — for spread indication)
    let prices: Vec<f64> = trimmed.iter().map(|r| r.price).collect();
    let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
    let variance: f64 =
        prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
    let std_dev = variance.sqrt();

    // Signal integrity: average accuracy of contributing reporters
    let signal_integrity = if accuracy_count > 0 {
        accuracy_sum / accuracy_count as f64
    } else {
        ACCURACY_INITIAL_SCORE
    };

    // Store consensus entry
    let consensus = PriceConsensus {
        item: item.clone(),
        median_price: median,
        reporter_count: reporters.len() as u32,
        std_dev,
        window_start,
        computed_at: now,
    };

    let consensus_hash = create_entry(&EntryTypes::PriceConsensus(consensus))?;

    // Fetch the previous consensus price BEFORE replacing the link, so we
    // compare against the prior value (not the one we just stored).
    let previous_median = get_previous_consensus_price(&item)?;

    // Update latest consensus link (replace old if exists)
    let consensus_anchor = anchor_hash(&format!("{ITEM_REPORTS_ANCHOR_PREFIX}{item}:consensus"))?;
    let old_links = get_links(
        LinkQuery::try_new(consensus_anchor.clone(), LinkTypes::ItemToConsensus)?,
        GetStrategy::default(),
    )?;
    for old_link in old_links {
        delete_link(old_link.create_link_hash, GetOptions::default())?;
    }
    create_link(
        consensus_anchor,
        consensus_hash,
        LinkTypes::ItemToConsensus,
        (),
    )?;

    // Update accuracy scores for ALL reporters (not just trimmed)
    // so that trimmed-out reporters also get accuracy feedback
    for report in &reports {
        let _ = update_reporter_accuracy(&report.reporter_did, report.price, median);
    }

    // -------------------------------------------------------------------------
    // Volatility check: compare new consensus to previous consensus for this
    // item. If the weekly price change exceeds VOLATILITY_ESCALATION_THRESHOLD
    // (20%), auto-escalate TEND limits via cross-zome call.
    // -------------------------------------------------------------------------
    let mut tend_escalated = false;

    if let Some(previous_median) = previous_median {
        if previous_median > 0.0 {
            let weekly_change = ((median - previous_median) / previous_median).abs();
            if weekly_change > VOLATILITY_ESCALATION_THRESHOLD {
                let vitality = if weekly_change > 0.40 {
                    10u32 // Emergency
                } else if weekly_change > 0.30 {
                    30 // High
                } else {
                    50 // Elevated
                };

                // Cross-zome call to TEND oracle — same DNA, local call.
                // Failure is non-fatal: log warning but don't fail consensus.
                match call(
                    CallTargetCell::Local,
                    ZomeName::from("tend"),
                    FunctionName::from("update_oracle_state"),
                    None,
                    vitality,
                ) {
                    Ok(ZomeCallResponse::Ok(_)) => {
                        tend_escalated = true;
                    }
                    Ok(other) => {
                        debug!(
                            "price-oracle: TEND escalation returned non-Ok response: {:?}",
                            other
                        );
                    }
                    Err(e) => {
                        debug!(
                            "price-oracle: TEND escalation cross-zome call failed (non-fatal): {:?}",
                            e
                        );
                    }
                }
            }
        }
    }

    Ok(ConsensusResult {
        item,
        median_price: median,
        reporter_count: reporters.len() as u32,
        std_dev,
        window_start,
        signal_integrity,
        tend_escalated,
    })
}

// =============================================================================
// BASKET MANAGEMENT
// =============================================================================

/// Define a price basket (weighted set of items).
///
/// Requires governance authorization. Weights must sum to ~1.0.
#[hdk_extern]
pub fn define_basket(input: DefineBasketInput) -> ExternResult<Record> {
    verify_governance_or_bootstrap()?;

    let my_info = agent_info()?;
    let my_did = format!("did:holo:{}", my_info.agent_initial_pubkey);
    let now = sys_time()?;

    let items: Vec<BasketItemDef> = input
        .items
        .into_iter()
        .map(|i| BasketItemDef {
            item: i.item.to_lowercase().trim().to_string(),
            weight: i.weight,
        })
        .collect();

    let basket = BasketDefinition {
        name: input.name.clone(),
        items,
        creator_did: my_did,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::BasketDefinition(basket))?;

    // Link from baskets anchor
    create_link(
        anchor_hash(BASKETS_ANCHOR)?,
        action_hash.clone(),
        LinkTypes::AnchorToBasket,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Failed to get basket record".into()
    )))
}

/// Compute the composite index for a basket.
///
/// For each item in the basket, fetches the latest consensus price
/// and computes a weighted sum.
#[hdk_extern]
pub fn get_basket_index(input: GetBasketIndexInput) -> ExternResult<BasketIndexResult> {
    // Find the basket definition
    let basket_links = get_links(
        LinkQuery::try_new(anchor_hash(BASKETS_ANCHOR)?, LinkTypes::AnchorToBasket)?,
        GetStrategy::default(),
    )?;

    let mut basket_def: Option<BasketDefinition> = None;
    for link in &basket_links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(hash)?;
            if let Some(def) = record
                .entry()
                .to_app_option::<BasketDefinition>()
                .ok()
                .flatten()
            {
                if def.name == input.basket_name {
                    basket_def = Some(def);
                    break;
                }
            }
        }
    }

    let basket = basket_def.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Basket '{}' not found",
        input.basket_name
    ))))?;

    // Fetch consensus price for each item
    let mut item_prices = Vec::new();
    let mut total_index = 0.0;

    for basket_item in &basket.items {
        let consensus_anchor = anchor_hash(&format!(
            "{ITEM_REPORTS_ANCHOR_PREFIX}{}:consensus",
            basket_item.item
        ))?;

        let consensus_links = get_links(
            LinkQuery::try_new(consensus_anchor, LinkTypes::ItemToConsensus)?,
            GetStrategy::default(),
        )?;

        let price = if let Some(link) = consensus_links.last() {
            if let Some(hash) = link.target.clone().into_action_hash() {
                if let Some(record) = get(hash, GetOptions::default())? {
                    record
                        .entry()
                        .to_app_option::<PriceConsensus>()
                        .ok()
                        .flatten()
                        .map(|c| c.median_price)
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        let weighted = price * basket_item.weight;
        total_index += weighted;

        item_prices.push(ItemPriceResult {
            item: basket_item.item.clone(),
            price,
            weight: basket_item.weight,
            weighted_price: weighted,
        });
    }

    Ok(BasketIndexResult {
        basket_name: input.basket_name,
        index: total_index,
        item_prices,
        computed_at: sys_time()?,
    })
}

// =============================================================================
// VOLATILITY → TEND LIMIT ESCALATION
// =============================================================================

/// Check basket volatility and auto-escalate TEND limits if needed.
///
/// Computes weekly percentage change. If > 20%, calls
/// tend/update_oracle_state to escalate credit limits.
///
/// Returns volatility data and whether escalation occurred.
#[hdk_extern]
pub fn compute_volatility(input: GetBasketIndexInput) -> ExternResult<VolatilityResult> {
    let current = get_basket_index(GetBasketIndexInput {
        basket_name: input.basket_name.clone(),
    })?;

    // Get previous alerts to estimate prior index
    let alert_links = get_links(
        LinkQuery::try_new(anchor_hash(ALERTS_ANCHOR)?, LinkTypes::AnchorToAlerts)?,
        GetStrategy::default(),
    )?;

    let mut previous_index = current.index;
    for link in alert_links.iter().rev() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(alert) = record
                    .entry()
                    .to_app_option::<VolatilityAlert>()
                    .ok()
                    .flatten()
                {
                    if alert.basket_name == input.basket_name {
                        // Use the change to estimate prior index
                        previous_index = current.index / (1.0 + alert.weekly_change);
                        break;
                    }
                }
            }
        }
    }

    let weekly_change = if previous_index > 0.0 {
        (current.index - previous_index) / previous_index
    } else {
        0.0
    };

    let abs_change = weekly_change.abs();

    // Determine recommended tier based on volatility
    let recommended_tier = if abs_change > 0.40 {
        "Emergency"
    } else if abs_change > 0.30 {
        "High"
    } else if abs_change > VOLATILITY_ESCALATION_THRESHOLD {
        "Elevated"
    } else {
        "Normal"
    };

    let mut escalated = false;

    // Auto-escalate if above threshold
    if abs_change > VOLATILITY_ESCALATION_THRESHOLD {
        let vitality = match recommended_tier {
            "Emergency" => 10u32,
            "High" => 30,
            "Elevated" => 50,
            _ => 72,
        };

        // Cross-zome call to TEND oracle
        let result = call(
            CallTargetCell::Local,
            ZomeName::from("tend"),
            FunctionName::from("update_oracle_state"),
            None,
            vitality,
        );

        escalated = matches!(result, Ok(ZomeCallResponse::Ok(_)));

        // Record volatility alert
        let alert = VolatilityAlert {
            basket_name: input.basket_name.clone(),
            weekly_change,
            recommended_tier: recommended_tier.to_string(),
            detected_at: sys_time()?,
        };
        let alert_hash = create_entry(&EntryTypes::VolatilityAlert(alert))?;
        create_link(
            anchor_hash(ALERTS_ANCHOR)?,
            alert_hash,
            LinkTypes::AnchorToAlerts,
            (),
        )?;
    }

    Ok(VolatilityResult {
        basket_name: input.basket_name,
        current_index: current.index,
        previous_index,
        weekly_change,
        recommended_tier: recommended_tier.to_string(),
        escalated,
    })
}

// =============================================================================
// QUERY HELPERS
// =============================================================================

/// Get the N most recent price reports across all items, sorted by timestamp descending.
///
/// Used by the mesh bridge poller to relay price data to the mesh network.
/// Queries the global reports anchor and returns deserialized report data
/// with action hash IDs for deduplication.
#[hdk_extern]
pub fn get_recent_reports(input: GetRecentReportsInput) -> ExternResult<Vec<RecentReportResult>> {
    let limit = if input.limit == 0 {
        DEFAULT_RECENT_REPORTS_LIMIT
    } else {
        input.limit.min(MAX_RECENT_REPORTS_LIMIT)
    } as usize;

    let links = get_links(
        LinkQuery::try_new(anchor_hash(ALL_REPORTS_ANCHOR)?, LinkTypes::AnchorLinks)?,
        GetStrategy::default(),
    )?;

    let mut results: Vec<RecentReportResult> = Vec::new();

    for link in &links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(report) = record
                    .entry()
                    .to_app_option::<PriceReport>()
                    .ok()
                    .flatten()
                {
                    results.push(RecentReportResult {
                        id: hash.to_string(),
                        item_name: report.item,
                        price_tend: report.price_tend,
                        evidence: report.evidence,
                        reporter_did: report.reporter_did,
                        timestamp: report.reported_at.as_micros(),
                    });
                }
            }
        }
    }

    // Sort by timestamp descending (most recent first)
    results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    // Truncate to requested limit
    results.truncate(limit);

    Ok(results)
}

/// Get all price reports for an item (within consensus window).
#[hdk_extern]
pub fn get_item_reports(item: String) -> ExternResult<Vec<Record>> {
    let item = item.to_lowercase().trim().to_string();
    let item_anchor = anchor_hash(&format!("{ITEM_REPORTS_ANCHOR_PREFIX}{item}"))?;

    let links = get_links(
        LinkQuery::try_new(item_anchor, LinkTypes::ItemToReports)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    let now_us = sys_time()?.as_micros();
    let window_start_us = now_us - CONSENSUS_WINDOW_US;

    for link in &links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(report) = record.entry().to_app_option::<PriceReport>().ok().flatten() {
                    if report.reported_at.as_micros() >= window_start_us {
                        records.push(record);
                    }
                }
            }
        }
    }

    Ok(records)
}

/// Get a reporter's accuracy score.
#[hdk_extern]
pub fn get_reporter_accuracy(reporter_did: String) -> ExternResult<ReporterAccuracyResult> {
    let (_, acc) = get_or_create_accuracy(&reporter_did)?;
    Ok(ReporterAccuracyResult {
        reporter_did: acc.reporter_did,
        accuracy_score: acc.accuracy_score,
        report_count: acc.report_count,
    })
}

/// Get top reporters for an item, ranked by accuracy score.
#[hdk_extern]
pub fn get_top_reporters(input: GetTopReportersInput) -> ExternResult<Vec<ReporterAccuracyResult>> {
    let item = input.item.to_lowercase().trim().to_string();
    let item_anchor = anchor_hash(&format!("{ITEM_REPORTS_ANCHOR_PREFIX}{item}"))?;

    let links = get_links(
        LinkQuery::try_new(item_anchor, LinkTypes::ItemToReports)?,
        GetStrategy::default(),
    )?;

    // Collect unique reporter DIDs from reports
    let mut reporter_dids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for link in &links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(report) = record.entry().to_app_option::<PriceReport>().ok().flatten() {
                    reporter_dids.insert(report.reporter_did);
                }
            }
        }
    }

    // Fetch accuracy for each reporter
    let mut results: Vec<ReporterAccuracyResult> = Vec::new();
    for did in reporter_dids {
        let (_, acc) = get_or_create_accuracy(&did)?;
        results.push(ReporterAccuracyResult {
            reporter_did: acc.reporter_did,
            accuracy_score: acc.accuracy_score,
            report_count: acc.report_count,
        });
    }

    // Sort by accuracy descending
    results.sort_by(|a, b| {
        b.accuracy_score
            .partial_cmp(&a.accuracy_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Limit
    let limit = input.limit.min(100) as usize;
    results.truncate(limit);

    Ok(results)
}

/// Get all defined baskets.
#[hdk_extern]
pub fn get_all_baskets(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(BASKETS_ANCHOR)?, LinkTypes::AnchorToBasket)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in &links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(hash)?;
            records.push(record);
        }
    }

    Ok(records)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trimmed_median_odd() {
        let mut prices = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let trim_count = (prices.len() as f64 * TRIM_PERCENT).floor() as usize;
        let trimmed = &prices[trim_count..prices.len() - trim_count];
        // Trim 1 from each end: [2, 3, 4, 5, 6, 7, 8, 9]
        assert_eq!(trimmed, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        // Median of 8 elements = (5+6)/2 = 5.5
        let mid = trimmed.len() / 2;
        let median: f64 = (trimmed[mid - 1] + trimmed[mid]) / 2.0;
        assert!((median - 5.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trimmed_median_small() {
        // With < 10 reports, no trimming occurs
        let prices = vec![1.0, 3.0, 5.0];
        let trim_count = (prices.len() as f64 * TRIM_PERCENT).floor() as usize;
        assert_eq!(trim_count, 0);
        let trimmed = &prices[..];
        let median: f64 = trimmed[trimmed.len() / 2];
        assert!((median - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_volatility_tiers() {
        // Normal: < 20% change
        assert!(0.15 < VOLATILITY_ESCALATION_THRESHOLD);
        // Elevated: 20-30%
        assert!(0.25 > VOLATILITY_ESCALATION_THRESHOLD);
        // High: 30-40%
        // Emergency: > 40%
    }

    #[test]
    fn test_std_dev_computation() {
        let prices = vec![10.0, 10.0, 10.0];
        let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance: f64 =
            prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();
        assert!((std_dev - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_basket_weight_validation() {
        let items = vec![
            BasketItemDef {
                item: "bread".into(),
                weight: 0.3,
            },
            BasketItemDef {
                item: "diesel".into(),
                weight: 0.3,
            },
            BasketItemDef {
                item: "eggs".into(),
                weight: 0.4,
            },
        ];
        let total: f64 = items.iter().map(|i| i.weight).sum();
        assert!((total - 1.0).abs() < 0.05);
    }

    // =========================================================================
    // Weighted median tests
    // =========================================================================

    #[test]
    fn test_weighted_median_uniform_weights() {
        // Equal weights should produce same result as simple median
        let mut entries = vec![(1.0, 1.0), (3.0, 1.0), (5.0, 1.0), (7.0, 1.0), (9.0, 1.0)];
        let result = weighted_median(&mut entries).unwrap();
        assert!(
            (result - 5.0).abs() < f64::EPSILON,
            "uniform weights: got {result}"
        );
    }

    #[test]
    fn test_weighted_median_skewed_weights() {
        // Heavy weight on low price should pull median down
        let mut entries = vec![
            (1.0, 10.0), // very heavy
            (5.0, 1.0),
            (9.0, 1.0),
        ];
        // total = 12, half = 6. Cumulative: 10 >= 6 at first entry
        let result = weighted_median(&mut entries).unwrap();
        assert!(
            (result - 1.0).abs() < f64::EPSILON,
            "skewed low: got {result}"
        );

        // Heavy weight on high price should pull median up
        let mut entries2 = vec![
            (1.0, 1.0),
            (5.0, 1.0),
            (9.0, 10.0), // very heavy
        ];
        let result2 = weighted_median(&mut entries2).unwrap();
        assert!(
            (result2 - 9.0).abs() < f64::EPSILON,
            "skewed high: got {result2}"
        );
    }

    #[test]
    fn test_weighted_median_single_element() {
        let mut entries = vec![(42.0, 1.0)];
        let result = weighted_median(&mut entries).unwrap();
        assert!((result - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_median_empty() {
        let mut entries: Vec<(f64, f64)> = vec![];
        assert!(weighted_median(&mut entries).is_none());
    }

    #[test]
    fn test_weighted_median_two_equal_weights() {
        // Two elements with equal weight: should return the first one
        // that crosses the half-weight threshold
        let mut entries = vec![(2.0, 1.0), (8.0, 1.0)];
        // total = 2, half = 1. Cumulative at first: 1.0 >= 1.0 → exact midpoint
        // Since cumulative == half and there's a next element, average: (2+8)/2 = 5
        let result = weighted_median(&mut entries).unwrap();
        assert!(
            (result - 5.0).abs() < f64::EPSILON,
            "two equal: got {result}"
        );
    }

    // =========================================================================
    // Accuracy EMA tests
    // =========================================================================

    #[test]
    fn test_accuracy_ema_convergence() {
        // 10 perfectly accurate reports should keep score near 1.0
        let mut score: f64 = ACCURACY_INITIAL_SCORE;
        for _ in 0..10 {
            let delta: f64 = 0.0; // perfect accuracy
            let signal = 1.0 - delta.min(1.0);
            score = (1.0 - ACCURACY_EMA_ALPHA) * score + ACCURACY_EMA_ALPHA * signal;
        }
        assert!((score - 1.0).abs() < f64::EPSILON, "perfect: got {score}");
    }

    #[test]
    fn test_accuracy_ema_decay() {
        // Consistently wrong by 100% should drive score toward 0
        let mut score: f64 = ACCURACY_INITIAL_SCORE;
        for _ in 0..50 {
            let delta: f64 = 1.0; // 100% off
            let signal = 1.0 - delta.min(1.0);
            score = (1.0 - ACCURACY_EMA_ALPHA) * score + ACCURACY_EMA_ALPHA * signal;
        }
        // After 50 rounds of 100% error: score ≈ 0.005
        assert!(score < 0.01, "decayed: got {score}");
    }

    #[test]
    fn test_accuracy_ema_partial_error() {
        // Reporter is 20% off each time
        let mut score: f64 = ACCURACY_INITIAL_SCORE;
        for _ in 0..30 {
            let delta: f64 = 0.2; // 20% error
            let signal = 1.0 - delta.min(1.0);
            score = (1.0 - ACCURACY_EMA_ALPHA) * score + ACCURACY_EMA_ALPHA * signal;
        }
        // Should converge to 0.8
        assert!((score - 0.8).abs() < 0.01, "partial error: got {score}");
    }

    #[test]
    fn test_accuracy_min_weight_floor() {
        // Even a terrible reporter gets minimum weight
        let terrible_score: f64 = 0.01;
        let weight = terrible_score.max(ACCURACY_MIN_WEIGHT);
        assert!((weight - ACCURACY_MIN_WEIGHT).abs() < f64::EPSILON);
    }

    // =========================================================================
    // Full consensus simulation (no conductor needed)
    // =========================================================================

    /// Simulate the accuracy EMA update for a reporter given their price and the consensus.
    fn sim_accuracy_update(old_score: f64, report_price: f64, consensus: f64) -> f64 {
        let delta = if consensus.abs() < 0.0001 {
            (report_price - consensus).abs() / 0.0001
        } else {
            (report_price - consensus).abs() / consensus
        };
        let signal = 1.0 - delta.min(1.0);
        let new = (1.0 - ACCURACY_EMA_ALPHA) * old_score + ACCURACY_EMA_ALPHA * signal;
        new.clamp(0.0, 1.0)
    }

    /// Simulate the full trimmed + accuracy-weighted consensus from raw reports.
    /// Assumes all reporters are established (report_count >= MIN_REPORTS_FOR_FULL_WEIGHT).
    fn sim_consensus(reports: &[(f64, f64)], // (price, accuracy_score)
    ) -> (f64, f64) {
        // Treat all as established reporters
        let with_counts: Vec<(f64, f64, u32)> = reports
            .iter()
            .map(|(p, a)| (*p, *a, MIN_REPORTS_FOR_FULL_WEIGHT))
            .collect();
        sim_consensus_with_counts(&with_counts)
    }

    /// Simulate consensus with explicit report counts per reporter.
    /// (price, accuracy_score, report_count)
    fn sim_consensus_with_counts(reports: &[(f64, f64, u32)]) -> (f64, f64) {
        // (weighted_median, signal_integrity)
        // Sort by price for trimming
        let mut sorted: Vec<(f64, f64, u32)> = reports.to_vec();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Trim 10%
        let trim_count = (sorted.len() as f64 * TRIM_PERCENT).floor() as usize;
        let trimmed = if trim_count > 0 && sorted.len() > trim_count * 2 {
            &sorted[trim_count..sorted.len() - trim_count]
        } else {
            &sorted[..]
        };

        // Build weighted entries with newcomer cap
        let mut entries: Vec<(f64, f64)> = trimmed
            .iter()
            .map(|(price, acc, count)| {
                let mut weight = acc.max(ACCURACY_MIN_WEIGHT);
                if *count < MIN_REPORTS_FOR_FULL_WEIGHT {
                    weight = weight.min(NEWCOMER_WEIGHT_CAP);
                }
                (*price, weight)
            })
            .collect();

        let median = weighted_median(&mut entries).unwrap();
        let si = trimmed.iter().map(|(_, a, _)| a).sum::<f64>() / trimmed.len() as f64;

        (median, si)
    }

    #[test]
    fn test_sim_two_accurate_reporters() {
        // Alice: 0.15, Bob: 0.16 — both accurate (score 1.0)
        let reports = vec![(0.15, 1.0), (0.16, 1.0)];
        let (median, si) = sim_consensus(&reports);

        // Equal weights → midpoint
        assert!((median - 0.155).abs() < 0.001, "median: {median}");
        assert!((si - 1.0).abs() < f64::EPSILON, "si: {si}");

        // After consensus, both scores should stay high
        let alice_new = sim_accuracy_update(1.0, 0.15, median);
        let bob_new = sim_accuracy_update(1.0, 0.16, median);
        assert!(alice_new > 0.95, "alice: {alice_new}");
        assert!(bob_new > 0.95, "bob: {bob_new}");
    }

    #[test]
    fn test_sim_two_reporters_first_round_symmetric() {
        // With only 2 reporters at equal accuracy, the first round can't differentiate
        // This is CORRECT behavior — you need either more reporters or multiple rounds
        let reports = vec![(0.50, 1.0), (5.00, 1.0)];
        let (median, _) = sim_consensus(&reports);

        // Midpoint = 2.75. Both are equidistant in relative terms
        let alice_new = sim_accuracy_update(1.0, 0.50, median);
        let bob_new = sim_accuracy_update(1.0, 5.00, median);

        // Both lose accuracy equally on round 1
        assert!(
            (alice_new - bob_new).abs() < 0.01,
            "round 1 is symmetric: alice={alice_new}, bob={bob_new}"
        );
        assert!(alice_new < 1.0, "both should lose some accuracy");
    }

    #[test]
    fn test_sim_accuracy_diverges_with_majority() {
        // With 3 reporters where 2 agree, the system differentiates immediately
        let mut alice_acc = ACCURACY_INITIAL_SCORE; // reports 0.50
        let mut carol_acc = ACCURACY_INITIAL_SCORE; // reports 0.52 (agrees with Alice)
        let mut bob_acc = ACCURACY_INITIAL_SCORE; // reports 5.00 (outlier)

        for _ in 0..5 {
            let reports = vec![(0.50, alice_acc), (0.52, carol_acc), (5.00, bob_acc)];
            let (median, _) = sim_consensus(&reports);

            alice_acc = sim_accuracy_update(alice_acc, 0.50, median);
            carol_acc = sim_accuracy_update(carol_acc, 0.52, median);
            bob_acc = sim_accuracy_update(bob_acc, 5.00, median);
        }

        // After 5 rounds, Alice and Carol should be much more accurate than Bob
        assert!(alice_acc > 0.9, "alice after 5 rounds: {alice_acc}");
        assert!(carol_acc > 0.9, "carol after 5 rounds: {carol_acc}");
        assert!(
            bob_acc < alice_acc,
            "bob ({bob_acc}) should be less accurate than alice ({alice_acc})"
        );
        assert!(bob_acc < 0.7, "bob after 5 rounds: {bob_acc}");

        // In round 6, the weighted median should be pulled toward Alice/Carol
        let reports = vec![(0.50, alice_acc), (0.52, carol_acc), (5.00, bob_acc)];
        let (median, _) = sim_consensus(&reports);
        assert!(
            median < 1.0,
            "accuracy-weighted median ({median}) should favor honest reporters"
        );
    }

    #[test]
    fn test_sim_trimming_removes_outliers() {
        // 10 reporters: 8 report ~0.50, 2 report 100.0 (manipulation)
        let reports = vec![
            (0.48, 1.0),
            (0.49, 1.0),
            (0.50, 1.0),
            (0.50, 1.0),
            (0.51, 1.0),
            (0.51, 1.0),
            (0.52, 1.0),
            (0.53, 1.0),
            (100.0, 1.0),
            (100.0, 1.0), // manipulators
        ];
        let (median, _) = sim_consensus(&reports);

        // Trim 10% = 1 from each end. Removes one 100.0 and the 0.48.
        // Remaining: [0.49, 0.50, 0.50, 0.51, 0.51, 0.52, 0.53, 100.0]
        // Even with one remaining outlier, the median of 8 elements should be reasonable
        // Median of [0.49, 0.50, 0.50, 0.51, 0.51, 0.52, 0.53, 100.0] = (0.51+0.51)/2 = 0.51
        assert!(
            median < 1.0,
            "trimmed median ({median}) should resist manipulation"
        );
    }

    #[test]
    fn test_sim_accuracy_weighted_resists_sybil() {
        // 3 honest reporters (high accuracy) at 0.50
        // 5 sybil reporters (low accuracy from prior bad reports) at 10.0
        let reports = vec![
            (0.50, 0.95),
            (0.50, 0.92),
            (0.50, 0.90), // honest
            (10.0, 0.15),
            (10.0, 0.12),
            (10.0, 0.10),
            (10.0, 0.08),
            (10.0, 0.05), // sybil
        ];
        let (median, si) = sim_consensus(&reports);

        // Trimming removes 1 from each end (0.05 and 0.95 weight entries)
        // Remaining 6: honest reporters have much higher weight
        // Median should be pulled toward honest reporters
        assert!(
            median < 5.0,
            "accuracy-weighted ({median}) should resist sybil attack"
        );
        assert!(
            si < 0.6,
            "signal integrity ({si}) should reflect low-quality reporters"
        );
    }

    #[test]
    fn test_sim_new_reporter_starts_trusted() {
        // New reporter (1.0) reports alongside two established reporters
        let reports = vec![
            (0.50, 0.98),
            (0.51, 0.95),
            (0.52, ACCURACY_INITIAL_SCORE), // new reporter
        ];
        let (median, si) = sim_consensus(&reports);

        // All close together → median near 0.51
        assert!((median - 0.51).abs() < 0.02, "median: {median}");
        assert!(si > 0.95, "signal integrity: {si}");
    }

    #[test]
    fn test_sim_newcomer_weight_cap_resists_sybil_flood() {
        // 3 established honest reporters at ~0.50
        // 3 new sybil accounts (report_count=0, accuracy=1.0) reporting 10.0
        // Without the newcomer cap, sybils have 3×1.0 = 3.0 weight.
        // With the cap, each sybil gets at most 0.5 → 3×0.5 = 1.5 total.
        // Honest total: 0.95 + 0.92 + 0.90 = 2.77 → honest dominate.
        let reports = vec![
            (0.50, 0.95, MIN_REPORTS_FOR_FULL_WEIGHT),     // established honest
            (0.51, 0.92, MIN_REPORTS_FOR_FULL_WEIGHT + 3), // established honest
            (0.52, 0.90, MIN_REPORTS_FOR_FULL_WEIGHT + 1), // established honest
            (10.0, ACCURACY_INITIAL_SCORE, 0),              // sybil newcomer
            (10.0, ACCURACY_INITIAL_SCORE, 1),              // sybil newcomer
            (10.0, ACCURACY_INITIAL_SCORE, 2),              // sybil newcomer
        ];
        let (median, _si) = sim_consensus_with_counts(&reports);

        // Sorted by price: 0.50(0.95), 0.51(0.92), 0.52(0.90), 10.0(0.5), 10.0(0.5), 10.0(0.5)
        // Total weight: 2.77 + 1.5 = 4.27. Half = 2.135.
        // Cumulative: 0.95, 1.87, 2.77 >= 2.135 at price 0.52
        // Median should be near the honest price.
        assert!(
            median < 1.0,
            "newcomer-capped median ({median}) should resist sybil flood"
        );

        // Compare: without newcomer cap (all treated as established), sybils would
        // have full weight and pull the median higher.
        let uncapped_reports: Vec<(f64, f64)> = reports
            .iter()
            .map(|(p, a, _)| (*p, *a))
            .collect();
        let (uncapped_median, _) = sim_consensus(&uncapped_reports);
        assert!(
            median <= uncapped_median,
            "capped median ({median}) should be <= uncapped ({uncapped_median})"
        );
    }

    #[test]
    fn test_sim_10_round_convergence() {
        // 5 reporters, one is consistently 100% off (reports double price)
        // Track how the median shifts and accuracy diverges over 10 rounds
        let true_price = 0.50;
        let mut accuracies = vec![1.0_f64; 5]; // all start at 1.0

        let mut last_median = 0.0;
        for round in 0..10 {
            let reports: Vec<(f64, f64)> = vec![
                (true_price, accuracies[0]),        // accurate
                (true_price * 1.02, accuracies[1]), // slightly off
                (true_price * 0.98, accuracies[2]), // slightly off other way
                (true_price * 1.01, accuracies[3]), // very close
                (true_price * 2.00, accuracies[4]), // 100% off consistently
            ];

            let (median, _) = sim_consensus(&reports);
            last_median = median;

            // Update all accuracies
            let prices = [
                true_price,
                true_price * 1.02,
                true_price * 0.98,
                true_price * 1.01,
                true_price * 2.00,
            ];
            for i in 0..5 {
                accuracies[i] = sim_accuracy_update(accuracies[i], prices[i], median);
            }

            if round == 9 {
                // After 10 rounds, the bad reporter should have noticeably lower accuracy
                assert!(
                    accuracies[4] < accuracies[0],
                    "bad reporter ({}) should be less accurate than honest ({})",
                    accuracies[4],
                    accuracies[0]
                );
                // Accurate reporters should remain high
                assert!(
                    accuracies[0] > 0.9,
                    "accurate reporter after 10 rounds: {}",
                    accuracies[0]
                );
            }
        }

        // Final median should be close to true price
        assert!(
            (last_median - true_price).abs() < 0.15,
            "final median ({last_median}) should be near true price ({true_price})"
        );
    }

    // =========================================================================
    // get_recent_reports input/output tests
    // =========================================================================

    #[test]
    fn test_recent_reports_input_default_limit() {
        let input: GetRecentReportsInput =
            serde_json::from_str("{}").expect("empty object should use default limit");
        assert_eq!(input.limit, DEFAULT_RECENT_REPORTS_LIMIT);
    }

    #[test]
    fn test_recent_reports_input_custom_limit() {
        let input: GetRecentReportsInput =
            serde_json::from_str(r#"{"limit": 5}"#).expect("should parse custom limit");
        assert_eq!(input.limit, 5);
    }

    #[test]
    fn test_recent_reports_limit_clamping() {
        // Verify the clamping logic used in get_recent_reports
        let over_limit = 200u32;
        let clamped = over_limit.min(MAX_RECENT_REPORTS_LIMIT) as usize;
        assert_eq!(clamped, MAX_RECENT_REPORTS_LIMIT as usize);

        let zero_limit = 0u32;
        let resolved = if zero_limit == 0 {
            DEFAULT_RECENT_REPORTS_LIMIT
        } else {
            zero_limit.min(MAX_RECENT_REPORTS_LIMIT)
        } as usize;
        assert_eq!(resolved, DEFAULT_RECENT_REPORTS_LIMIT as usize);
    }

    #[test]
    fn test_recent_report_result_serialization() {
        let result = RecentReportResult {
            id: "uhCkk_abc123".to_string(),
            item_name: "bread_750g".to_string(),
            price_tend: 0.15,
            evidence: "Pick n Pay Roodepoort".to_string(),
            reporter_did: "did:holo:abc123".to_string(),
            timestamp: 1710000000000000,
        };

        let json = serde_json::to_value(&result).expect("should serialize");
        assert_eq!(json["id"], "uhCkk_abc123");
        assert_eq!(json["item_name"], "bread_750g");
        assert_eq!(json["price_tend"], 0.15);
        assert_eq!(json["evidence"], "Pick n Pay Roodepoort");
        assert_eq!(json["reporter_did"], "did:holo:abc123");
        assert_eq!(json["timestamp"], 1710000000000000i64);
    }

    #[test]
    fn test_recent_reports_sorting() {
        // Verify that sorting by timestamp descending works correctly
        let mut results = vec![
            RecentReportResult {
                id: "a".into(),
                item_name: "bread".into(),
                price_tend: 0.1,
                evidence: "".into(),
                reporter_did: "".into(),
                timestamp: 100,
            },
            RecentReportResult {
                id: "b".into(),
                item_name: "eggs".into(),
                price_tend: 0.2,
                evidence: "".into(),
                reporter_did: "".into(),
                timestamp: 300,
            },
            RecentReportResult {
                id: "c".into(),
                item_name: "diesel".into(),
                price_tend: 0.3,
                evidence: "".into(),
                reporter_did: "".into(),
                timestamp: 200,
            },
        ];
        results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        assert_eq!(results[0].id, "b"); // 300 — most recent
        assert_eq!(results[1].id, "c"); // 200
        assert_eq!(results[2].id, "a"); // 100 — oldest
    }
}
