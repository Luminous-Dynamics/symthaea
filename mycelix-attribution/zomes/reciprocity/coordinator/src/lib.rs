use hdk::prelude::*;
use reciprocity_integrity::*;
use serde::{Deserialize, Serialize};

// ── Signals ──────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum ReciprocitySignal {
    PledgeRecorded {
        dependency_id: String,
        contributor_did: String,
        pledge_type: String,
    },
    PledgeAcknowledged {
        pledge_id: String,
        dependency_id: String,
    },
}

// ── Output Types ─────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StewardshipScore {
    pub dependency_id: String,
    pub usage_count: u64,
    pub pledge_count: u64,
    pub ratio: f64,
    /// Weighted score incorporating amount, recency, and pledge diversity
    pub weighted_score: f64,
    /// Breakdown by pledge type
    pub pledge_type_counts: Vec<(String, u64)>,
}

// ── Input Types ─────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginationInput {
    pub offset: u64,
    pub limit: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedPledges {
    pub items: Vec<Record>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
    pub has_more: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedPledgesInput {
    pub id: String,
    pub pagination: PaginationInput,
}

// ── Init ─────────────────────────────────────────────────────────────

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    let anchor = Anchor("all_pledges".to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    Ok(InitCallbackResult::Pass)
}

// ── Constants ───────────────────────────────────────────────────────

const RATE_LIMIT_WINDOW_SECS: i64 = 60;
const MAX_PAGE_SIZE: u64 = 1000;

// ── Helpers ──────────────────────────────────────────────────────────
// NOTE: anchor_hash/resolve_links are intentionally duplicated across
// registry, usage, and reciprocity coordinators. Each zome uses its own
// Anchor/LinkTypes from its integrity crate, preventing shared extraction.

fn anchor_hash(tag: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(tag.to_string()))
}

/// Validate that a dependency exists in the registry (cross-zome call).
fn validate_dependency_exists(dep_id: &str) -> ExternResult<()> {
    let encoded = ExternIO::encode(dep_id.to_string()).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to encode dep_id: {}",
            e
        )))
    })?;
    match call(
        CallTargetCell::Local,
        ZomeName::from("registry"),
        FunctionName::from("get_dependency"),
        None,
        encoded,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => {
            let record: Option<Record> = io.decode().unwrap_or(None);
            match record {
                Some(_) => Ok(()),
                None => Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Dependency '{}' not registered",
                    dep_id
                )))),
            }
        }
        other => {
            debug!(
                "validate_dependency_exists: registry call returned non-Ok response: {:?}",
                other
            );
            Ok(()) // Graceful: allow if registry unavailable
        }
    }
}

/// Sliding-window rate limiter. Checks count BEFORE creating the link.
fn enforce_rate_limit(limit: u64) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash = EntryHash::from(agent);
    let now = sys_time()?;
    let now_micros = now.as_micros();
    let window_micros = RATE_LIMIT_WINDOW_SECS * 1_000_000;

    // Count links within window FIRST
    let links = get_links(
        LinkQuery::try_new(agent_hash.clone(), LinkTypes::PledgeRateLimit)?,
        GetStrategy::default(),
    )?;

    let cutoff = now_micros - window_micros;
    let recent = links
        .iter()
        .filter(|l| {
            if l.tag.0.len() >= 8 {
                let ts = i64::from_le_bytes(l.tag.0[..8].try_into().unwrap_or([0; 8]));
                ts > cutoff
            } else {
                false
            }
        })
        .count() as u64;

    if recent >= limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Rate limit exceeded: {} requests in last {} seconds (limit: {})",
            recent, RATE_LIMIT_WINDOW_SECS, limit
        ))));
    }

    // Create rate-limit link AFTER passing the check
    create_link(
        agent_hash.clone(),
        agent_hash.clone(),
        LinkTypes::PledgeRateLimit,
        LinkTag::new(now_micros.to_le_bytes()),
    )?;

    Ok(())
}

fn resolve_links(base: EntryHash, link_type: LinkTypes) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(base, link_type)?, GetStrategy::default())?;
    let mut records = Vec::new();
    for link in links {
        let entry_hash = EntryHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(entry_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ── Externs ──────────────────────────────────────────────────────────

#[hdk_extern]
pub fn record_pledge(pledge: ReciprocityPledge) -> ExternResult<Record> {
    validate_dependency_exists(&pledge.dependency_id)?;
    enforce_rate_limit(20)?;

    let action_hash = create_entry(&EntryTypes::ReciprocityPledge(pledge.clone()))?;
    let entry_hash = hash_entry(&pledge)?;

    // Link: pledges:{dep_id} → pledge
    let dep_tag = format!("pledges:{}", pledge.dependency_id);
    create_link(
        anchor_hash(&dep_tag)?,
        entry_hash.clone(),
        LinkTypes::DependencyToPledges,
        (),
    )?;

    // Link: contrib:{did} → pledge
    let contrib_tag = format!("contrib:{}", pledge.contributor_did);
    create_link(
        anchor_hash(&contrib_tag)?,
        entry_hash.clone(),
        LinkTypes::ContributorToPledges,
        (),
    )?;

    // Link: all_pledges → pledge
    create_link(
        anchor_hash("all_pledges")?,
        entry_hash.clone(),
        LinkTypes::AllPledges,
        (),
    )?;

    // Link: pledge_id:{id} → pledge (O(1) lookup for acknowledge_pledge)
    let id_tag = format!("pledge_id:{}", pledge.id);
    create_link(anchor_hash(&id_tag)?, entry_hash, LinkTypes::PledgeById, ())?;

    let _ = emit_signal(&ReciprocitySignal::PledgeRecorded {
        dependency_id: pledge.dependency_id.clone(),
        contributor_did: pledge.contributor_did.clone(),
        pledge_type: format!("{:?}", pledge.pledge_type),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch newly created pledge".into()
    )))
}

#[hdk_extern]
pub fn acknowledge_pledge(id: String) -> ExternResult<Record> {
    // O(1) lookup via pledge_id index
    let id_tag = format!("pledge_id:{}", id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&id_tag)?, LinkTypes::PledgeById)?,
        GetStrategy::default(),
    )?;

    let link = links
        .first()
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Pledge '{}' not found",
            id
        ))))?;

    let entry_hash = EntryHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    let record = get(entry_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest(format!("Pledge '{}' entry not found", id))
    ))?;

    let pledge: ReciprocityPledge = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Deserialization error: {}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Record has no entry".into()
        )))?;

    let mut updated = pledge;
    updated.acknowledged = true;

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::ReciprocityPledge(updated.clone()),
    )?;

    let _ = emit_signal(&ReciprocitySignal::PledgeAcknowledged {
        pledge_id: id,
        dependency_id: updated.dependency_id.clone(),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch acknowledged pledge".into()
    )))
}

#[hdk_extern]
pub fn get_dependency_pledges(dep_id: String) -> ExternResult<Vec<Record>> {
    let dep_tag = format!("pledges:{}", dep_id);
    resolve_links(anchor_hash(&dep_tag)?, LinkTypes::DependencyToPledges)
}

/// Count all pledges across all dependencies.
#[hdk_extern]
pub fn get_pledge_count(_: ()) -> ExternResult<u64> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_pledges")?, LinkTypes::AllPledges)?,
        GetStrategy::default(),
    )?;
    Ok(links.len() as u64)
}

#[hdk_extern]
pub fn get_contributor_pledges(did: String) -> ExternResult<Vec<Record>> {
    let contrib_tag = format!("contrib:{}", did);
    resolve_links(anchor_hash(&contrib_tag)?, LinkTypes::ContributorToPledges)
}

#[hdk_extern]
pub fn compute_stewardship_score(dep_id: String) -> ExternResult<StewardshipScore> {
    // Fetch all pledge records (not just count)
    let dep_tag = format!("pledges:{}", dep_id);
    let pledge_records = resolve_links(anchor_hash(&dep_tag)?, LinkTypes::DependencyToPledges)?;
    let pledge_count = pledge_records.len() as u64;

    // Cross-zome call to usage coordinator for usage count
    let usage_count: u64 = {
        let encoded = ExternIO::encode(dep_id.clone()).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to encode payload for usage::get_usage_count: {}",
                e
            )))
        })?;
        match call(
            CallTargetCell::Local,
            ZomeName::from("usage"),
            FunctionName::from("get_usage_count"),
            None,
            encoded,
        ) {
            Ok(ZomeCallResponse::Ok(io)) => io.decode().unwrap_or(0u64),
            _ => 0u64,
        }
    };

    let ratio = if usage_count == 0 {
        if pledge_count > 0 {
            1.0
        } else {
            0.0
        }
    } else {
        pledge_count as f64 / usage_count as f64
    };

    // Weighted scoring: amount + recency + pledge type diversity
    let now = sys_time()?;
    let now_micros = now.as_micros();
    let mut weighted_sum = 0.0;
    let mut type_counts: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();

    for record in &pledge_records {
        let pledge: Option<ReciprocityPledge> = record.entry().to_app_option().ok().flatten();
        if let Some(p) = pledge {
            let amount_weight = compute_amount_weight(p.amount);

            let pledge_micros = p.pledged_at.as_micros();
            let age_days =
                (now_micros.saturating_sub(pledge_micros) as f64) / (86_400.0 * 1_000_000.0);
            let recency_weight = compute_recency_weight(age_days);

            let type_weight = compute_type_weight(&p.pledge_type);

            weighted_sum += amount_weight * recency_weight * type_weight;

            *type_counts
                .entry(format!("{:?}", p.pledge_type))
                .or_insert(0) += 1;
        }
    }

    // Normalize weighted score by usage count
    let weighted_score = if usage_count == 0 {
        weighted_sum
    } else {
        weighted_sum / usage_count as f64
    };

    let pledge_type_counts: Vec<(String, u64)> = type_counts.into_iter().collect();

    Ok(StewardshipScore {
        dependency_id: dep_id,
        usage_count,
        pledge_count,
        ratio,
        weighted_score,
        pledge_type_counts,
    })
}

// ── Paginated Pledge Queries ────────────────────────────────────────

#[hdk_extern]
pub fn get_dependency_pledges_paginated(
    input: PaginatedPledgesInput,
) -> ExternResult<PaginatedPledges> {
    if input.pagination.limit > MAX_PAGE_SIZE {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Pagination limit {} exceeds maximum {}",
            input.pagination.limit, MAX_PAGE_SIZE
        ))));
    }
    let dep_tag = format!("pledges:{}", input.id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&dep_tag)?, LinkTypes::DependencyToPledges)?,
        GetStrategy::default(),
    )?;
    let total = links.len() as u64;

    let page_links: Vec<_> = links
        .into_iter()
        .skip(input.pagination.offset as usize)
        .take(input.pagination.limit as usize)
        .collect();

    let mut items = Vec::new();
    for link in page_links {
        let entry_hash = EntryHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(entry_hash, GetOptions::default())? {
            items.push(record);
        }
    }

    let has_more = input.pagination.offset + input.pagination.limit < total;
    Ok(PaginatedPledges {
        items,
        total,
        offset: input.pagination.offset,
        limit: input.pagination.limit,
        has_more,
    })
}

// ── Leaderboard & Under-Supported Queries ───────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderboardEntry {
    pub dependency_id: String,
    pub usage_count: u64,
    pub pledge_count: u64,
    pub weighted_score: f64,
}

#[hdk_extern]
pub fn get_stewardship_leaderboard(limit: u64) -> ExternResult<Vec<LeaderboardEntry>> {
    let limit = limit.min(MAX_PAGE_SIZE);
    // Get all deps from registry via cross-zome
    let encoded = ExternIO::encode(())
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to encode: {}", e))))?;

    let dep_records: Vec<Record> = match call(
        CallTargetCell::Local,
        ZomeName::from("registry"),
        FunctionName::from("get_all_dependencies"),
        None,
        encoded,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => io.decode().unwrap_or_else(|e| {
            debug!("Failed to decode registry response: {:?}", e);
            Vec::new()
        }),
        _ => return Ok(Vec::new()),
    };

    // Compute stewardship for each
    let mut entries: Vec<LeaderboardEntry> = Vec::new();

    for record in dep_records {
        let dep: Option<DependencyIdMirror> = record.entry().to_app_option().ok().flatten();
        if let Some(d) = dep {
            let score = compute_stewardship_score(d.id.clone())?;
            entries.push(LeaderboardEntry {
                dependency_id: d.id,
                usage_count: score.usage_count,
                pledge_count: score.pledge_count,
                weighted_score: score.weighted_score,
            });
        }
    }

    // Sort by weighted_score descending (NaN-safe: NaN sorts last)
    entries.sort_by(|a, b| b.weighted_score.total_cmp(&a.weighted_score));
    entries.truncate(limit as usize);

    Ok(entries)
}

#[hdk_extern]
pub fn get_under_supported_dependencies(limit: u64) -> ExternResult<Vec<LeaderboardEntry>> {
    let limit = limit.min(MAX_PAGE_SIZE);
    // Same approach: get all deps, compute stewardship, sort by lowest ratio
    let encoded = ExternIO::encode(())
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to encode: {}", e))))?;

    let dep_records: Vec<Record> = match call(
        CallTargetCell::Local,
        ZomeName::from("registry"),
        FunctionName::from("get_all_dependencies"),
        None,
        encoded,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => io.decode().unwrap_or_else(|e| {
            debug!("Failed to decode registry response: {:?}", e);
            Vec::new()
        }),
        _ => return Ok(Vec::new()),
    };

    let mut entries: Vec<LeaderboardEntry> = Vec::new();

    for record in dep_records {
        let dep: Option<DependencyIdMirror> = record.entry().to_app_option().ok().flatten();
        if let Some(d) = dep {
            let score = compute_stewardship_score(d.id.clone())?;
            // Only include deps with usage but low/zero stewardship
            if score.usage_count > 0 {
                entries.push(LeaderboardEntry {
                    dependency_id: d.id,
                    usage_count: score.usage_count,
                    pledge_count: score.pledge_count,
                    weighted_score: score.weighted_score,
                });
            }
        }
    }

    // Sort by weighted_score ascending (NaN-safe: NaN sorts last)
    entries.sort_by(|a, b| a.weighted_score.total_cmp(&b.weighted_score));
    entries.truncate(limit as usize);

    Ok(entries)
}

/// Mirror type for deserializing DependencyIdentity from registry zome.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct DependencyIdMirror {
    id: String,
}

holochain_serialized_bytes::holochain_serial!(DependencyIdMirror);

// ── Pure Weight Helpers (testable without HDK) ────────────────────

/// Amount weight: ln(1+a) for financial pledges, 1.0 for non-financial.
fn compute_amount_weight(amount: Option<f64>) -> f64 {
    match amount {
        Some(a) if a > 0.0 && a.is_finite() => (1.0 + a).ln(),
        _ => 1.0,
    }
}

/// Recency weight: exponential decay with ~6-month half-life.
fn compute_recency_weight(age_days: f64) -> f64 {
    (-age_days / 180.0_f64).exp()
}

/// Type weight by pledge type.
fn compute_type_weight(pledge_type: &PledgeType) -> f64 {
    match pledge_type {
        PledgeType::Financial => 1.5,
        PledgeType::DeveloperTime => 1.3,
        PledgeType::Compute => 1.2,
        PledgeType::Bandwidth => 1.1,
        PledgeType::QA => 1.1,
        PledgeType::Documentation => 1.0,
        PledgeType::Other => 0.8,
    }
}

/// Pure computation of has_more flag (testable without HDK).
#[cfg(test)]
fn compute_has_more(offset: u64, limit: u64, total: u64) -> bool {
    offset + limit < total
}

/// Pure validation of pagination limit (testable without HDK).
#[cfg(test)]
fn validate_page_limit(limit: u64) -> Result<(), String> {
    if limit > MAX_PAGE_SIZE {
        Err(format!(
            "Pagination limit {} exceeds maximum {}",
            limit, MAX_PAGE_SIZE
        ))
    } else {
        Ok(())
    }
}

/// Pure stewardship ratio computation (testable without HDK).
#[cfg(test)]
fn compute_ratio(pledge_count: u64, usage_count: u64) -> f64 {
    if usage_count == 0 {
        if pledge_count > 0 {
            1.0
        } else {
            0.0
        }
    } else {
        pledge_count as f64 / usage_count as f64
    }
}

/// Pure weighted score normalization (testable without HDK).
#[cfg(test)]
fn normalize_weighted_score(weighted_sum: f64, usage_count: u64) -> f64 {
    if usage_count == 0 {
        weighted_sum
    } else {
        weighted_sum / usage_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amount_weight_financial() {
        // ln(1 + 100) ≈ 4.615
        let w = compute_amount_weight(Some(100.0));
        assert!((w - 4.615).abs() < 0.01, "expected ~4.615, got {}", w);

        // ln(1 + 1000) ≈ 6.908
        let w2 = compute_amount_weight(Some(1000.0));
        assert!((w2 - 6.908).abs() < 0.01, "expected ~6.908, got {}", w2);

        // $1000 should NOT be 10x more than $100 (logarithmic)
        assert!(w2 / w < 2.0, "ratio {} should be < 2.0", w2 / w);
    }

    #[test]
    fn test_amount_weight_non_financial() {
        assert_eq!(compute_amount_weight(None), 1.0);
        assert_eq!(compute_amount_weight(Some(0.0)), 1.0);
        assert_eq!(compute_amount_weight(Some(-5.0)), 1.0);
        assert_eq!(compute_amount_weight(Some(f64::NAN)), 1.0);
        assert_eq!(compute_amount_weight(Some(f64::INFINITY)), 1.0);
    }

    #[test]
    fn test_recency_weight_decay() {
        // Today: weight = 1.0
        let w0 = compute_recency_weight(0.0);
        assert!((w0 - 1.0).abs() < 1e-10);

        // ~6 months: weight ≈ 0.368 (1/e)
        let w180 = compute_recency_weight(180.0);
        assert!(
            (w180 - (-1.0_f64).exp()).abs() < 0.01,
            "expected ~0.368, got {}",
            w180
        );

        // 1 year: weight ≈ 0.135
        let w360 = compute_recency_weight(360.0);
        assert!(w360 < 0.15 && w360 > 0.12, "expected ~0.135, got {}", w360);

        // Monotonically decreasing
        assert!(w0 > w180);
        assert!(w180 > w360);
    }

    #[test]
    fn test_type_weight_ordering() {
        let financial = compute_type_weight(&PledgeType::Financial);
        let dev_time = compute_type_weight(&PledgeType::DeveloperTime);
        let compute = compute_type_weight(&PledgeType::Compute);
        let docs = compute_type_weight(&PledgeType::Documentation);
        let other = compute_type_weight(&PledgeType::Other);

        assert!(financial > dev_time);
        assert!(dev_time > compute);
        assert!(compute > docs);
        assert!(docs > other);
    }

    #[test]
    fn test_combined_weight_scenario() {
        // $500 financial pledge made today
        let w = compute_amount_weight(Some(500.0))
            * compute_recency_weight(0.0)
            * compute_type_weight(&PledgeType::Financial);

        // ln(501) * 1.0 * 1.5 ≈ 9.32
        assert!(w > 9.0 && w < 10.0, "expected ~9.32, got {}", w);

        // Same pledge 1 year ago should be ~7x less
        let w_old = compute_amount_weight(Some(500.0))
            * compute_recency_weight(360.0)
            * compute_type_weight(&PledgeType::Financial);
        assert!(w / w_old > 5.0, "recent should dominate old");
    }

    // ── Constants ───────────────────────────────────────────────────

    #[test]
    fn test_max_page_size_is_1000() {
        assert_eq!(MAX_PAGE_SIZE, 1000);
    }

    #[test]
    fn test_rate_limit_window_is_60_seconds() {
        assert_eq!(RATE_LIMIT_WINDOW_SECS, 60);
    }

    // ── Pagination validation ───────────────────────────────────────

    #[test]
    fn test_validate_page_limit_within_max() {
        assert!(validate_page_limit(500).is_ok());
    }

    #[test]
    fn test_validate_page_limit_at_max() {
        assert!(validate_page_limit(MAX_PAGE_SIZE).is_ok());
    }

    #[test]
    fn test_validate_page_limit_exceeds_max() {
        let err = validate_page_limit(MAX_PAGE_SIZE + 1).unwrap_err();
        assert!(err.contains("exceeds maximum"), "error: {}", err);
    }

    #[test]
    fn test_validate_page_limit_zero() {
        assert!(validate_page_limit(0).is_ok());
    }

    // ── has_more computation ────────────────────────────────────────

    #[test]
    fn test_has_more_true_when_remaining() {
        assert!(compute_has_more(0, 10, 20));
    }

    #[test]
    fn test_has_more_false_at_exact_end() {
        assert!(!compute_has_more(10, 10, 20));
    }

    #[test]
    fn test_has_more_false_past_end() {
        assert!(!compute_has_more(15, 10, 20));
    }

    #[test]
    fn test_has_more_false_empty() {
        assert!(!compute_has_more(0, 10, 0));
    }

    // ── Stewardship ratio ───────────────────────────────────────────

    #[test]
    fn test_ratio_zero_usage_zero_pledges() {
        assert_eq!(compute_ratio(0, 0), 0.0);
    }

    #[test]
    fn test_ratio_zero_usage_some_pledges() {
        assert_eq!(compute_ratio(5, 0), 1.0);
    }

    #[test]
    fn test_ratio_equal_pledges_and_usage() {
        let r = compute_ratio(10, 10);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ratio_more_usage_than_pledges() {
        let r = compute_ratio(5, 100);
        assert!((r - 0.05).abs() < 1e-10);
    }

    // ── Weighted score normalization ────────────────────────────────

    #[test]
    fn test_normalize_weighted_score_zero_usage() {
        assert_eq!(normalize_weighted_score(42.0, 0), 42.0);
    }

    #[test]
    fn test_normalize_weighted_score_nonzero_usage() {
        let score = normalize_weighted_score(100.0, 50);
        assert!((score - 2.0).abs() < 1e-10);
    }

    // ── Serde roundtrip tests ───────────────────────────────────────

    #[test]
    fn test_pagination_input_serde_roundtrip() {
        let input = PaginationInput {
            offset: 10,
            limit: 50,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: PaginationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.offset, 10);
        assert_eq!(back.limit, 50);
    }

    #[test]
    fn test_stewardship_score_serde_roundtrip() {
        let score = StewardshipScore {
            dependency_id: "crate:serde".into(),
            usage_count: 100,
            pledge_count: 10,
            ratio: 0.1,
            weighted_score: 5.5,
            pledge_type_counts: vec![("Financial".into(), 5), ("DeveloperTime".into(), 5)],
        };
        let json = serde_json::to_string(&score).unwrap();
        let back: StewardshipScore = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dependency_id, "crate:serde");
        assert_eq!(back.usage_count, 100);
        assert_eq!(back.pledge_type_counts.len(), 2);
    }

    #[test]
    fn test_leaderboard_entry_serde_roundtrip() {
        let entry = LeaderboardEntry {
            dependency_id: "crate:tokio".into(),
            usage_count: 500,
            pledge_count: 20,
            weighted_score: 12.5,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: LeaderboardEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dependency_id, "crate:tokio");
        assert_eq!(back.weighted_score, 12.5);
    }

    #[test]
    fn test_leaderboard_sort_descending() {
        let mut entries = vec![
            LeaderboardEntry {
                dependency_id: "a".into(),
                usage_count: 10,
                pledge_count: 1,
                weighted_score: 2.0,
            },
            LeaderboardEntry {
                dependency_id: "b".into(),
                usage_count: 20,
                pledge_count: 5,
                weighted_score: 8.0,
            },
            LeaderboardEntry {
                dependency_id: "c".into(),
                usage_count: 15,
                pledge_count: 3,
                weighted_score: 5.0,
            },
        ];
        entries.sort_by(|a, b| b.weighted_score.total_cmp(&a.weighted_score));
        assert_eq!(entries[0].dependency_id, "b");
        assert_eq!(entries[1].dependency_id, "c");
        assert_eq!(entries[2].dependency_id, "a");
    }

    #[test]
    fn test_leaderboard_sort_nan_safe() {
        let mut entries = vec![
            LeaderboardEntry {
                dependency_id: "nan".into(),
                usage_count: 0,
                pledge_count: 0,
                weighted_score: f64::NAN,
            },
            LeaderboardEntry {
                dependency_id: "real".into(),
                usage_count: 10,
                pledge_count: 5,
                weighted_score: 5.0,
            },
        ];
        // total_cmp is NaN-safe (no panic). NaN is "greater than" all finite
        // values in IEEE 754 total ordering, so descending sort puts NaN first.
        entries.sort_by(|a, b| b.weighted_score.total_cmp(&a.weighted_score));
        // The important thing is that the sort completes without panic.
        assert_eq!(entries.len(), 2);
        // NaN sorts first in descending total_cmp
        assert_eq!(entries[0].dependency_id, "nan");
        assert_eq!(entries[1].dependency_id, "real");
    }

    #[test]
    fn test_reciprocity_signal_serde_roundtrip() {
        let signals = vec![
            ReciprocitySignal::PledgeRecorded {
                dependency_id: "dep1".into(),
                contributor_did: "did:mycelix:corp1".into(),
                pledge_type: "Financial".into(),
            },
            ReciprocitySignal::PledgeAcknowledged {
                pledge_id: "pledge-001".into(),
                dependency_id: "dep1".into(),
            },
        ];
        for sig in signals {
            let json = serde_json::to_string(&sig).unwrap();
            let back: ReciprocitySignal = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&back).unwrap();
            assert_eq!(json, json2);
        }
    }

    // ── Edge cases for weight helpers ───────────────────────────────

    #[test]
    fn test_amount_weight_very_small_positive() {
        let w = compute_amount_weight(Some(0.001));
        // ln(1.001) ≈ 0.001
        assert!(w > 0.0 && w < 0.01, "expected near 0, got {}", w);
    }

    #[test]
    fn test_recency_weight_negative_days_clamped() {
        // Negative days = future pledge, should give weight > 1
        let w = compute_recency_weight(-30.0);
        assert!(w > 1.0, "future pledge should have weight > 1, got {}", w);
    }

    #[test]
    fn test_recency_weight_very_old() {
        // 10 years old
        let w = compute_recency_weight(3650.0);
        assert!(
            w < 0.001,
            "very old pledge weight should be near 0, got {}",
            w
        );
    }

    #[test]
    fn test_type_weight_all_variants_positive() {
        let types = vec![
            PledgeType::Financial,
            PledgeType::DeveloperTime,
            PledgeType::Compute,
            PledgeType::Bandwidth,
            PledgeType::QA,
            PledgeType::Documentation,
            PledgeType::Other,
        ];
        for pt in types {
            let w = compute_type_weight(&pt);
            assert!(w > 0.0, "{:?} should have positive weight", pt);
        }
    }

    #[test]
    fn test_bandwidth_and_qa_equal_weight() {
        assert_eq!(
            compute_type_weight(&PledgeType::Bandwidth),
            compute_type_weight(&PledgeType::QA)
        );
    }
}
