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

// ── Helpers ──────────────────────────────────────────────────────────

fn anchor_hash(tag: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(tag.to_string()))
}

fn resolve_links(base: EntryHash, link_type: LinkTypes) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(base, link_type)?,
        GetStrategy::default(),
    )?;
    let mut records = Vec::new();
    for link in links {
        let entry_hash = EntryHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
        })?;
        if let Some(record) = get(entry_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ── Externs ──────────────────────────────────────────────────────────

#[hdk_extern]
pub fn record_pledge(pledge: ReciprocityPledge) -> ExternResult<Record> {
    let action_hash =
        create_entry(&EntryTypes::ReciprocityPledge(pledge.clone()))?;
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
        entry_hash,
        LinkTypes::AllPledges,
        (),
    )?;

    let _ = emit_signal(&ReciprocitySignal::PledgeRecorded {
        dependency_id: pledge.dependency_id.clone(),
        contributor_did: pledge.contributor_did.clone(),
        pledge_type: format!("{:?}", pledge.pledge_type),
    });

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not fetch newly created pledge".into()
        )))
}

#[hdk_extern]
pub fn acknowledge_pledge(id: String) -> ExternResult<Record> {
    // Find pledge by scanning all_pledges links
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_pledges")?, LinkTypes::AllPledges)?,
        GetStrategy::default(),
    )?;

    for link in links {
        let entry_hash = EntryHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
        })?;
        if let Some(record) = get(entry_hash, GetOptions::default())? {
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

            if pledge.id == id {
                let mut updated = pledge;
                updated.acknowledged = true;

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::ReciprocityPledge(updated.clone()),
                )?;

                let _ =
                    emit_signal(&ReciprocitySignal::PledgeAcknowledged {
                        pledge_id: id,
                        dependency_id: updated.dependency_id.clone(),
                    });

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Could not fetch acknowledged pledge".into()
                    )));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "Pledge '{}' not found",
        id
    ))))
}

#[hdk_extern]
pub fn get_dependency_pledges(dep_id: String) -> ExternResult<Vec<Record>> {
    let dep_tag = format!("pledges:{}", dep_id);
    resolve_links(anchor_hash(&dep_tag)?, LinkTypes::DependencyToPledges)
}

#[hdk_extern]
pub fn get_contributor_pledges(did: String) -> ExternResult<Vec<Record>> {
    let contrib_tag = format!("contrib:{}", did);
    resolve_links(
        anchor_hash(&contrib_tag)?,
        LinkTypes::ContributorToPledges,
    )
}

#[hdk_extern]
pub fn compute_stewardship_score(
    dep_id: String,
) -> ExternResult<StewardshipScore> {
    // Fetch all pledge records (not just count)
    let dep_tag = format!("pledges:{}", dep_id);
    let pledge_records = resolve_links(
        anchor_hash(&dep_tag)?,
        LinkTypes::DependencyToPledges,
    )?;
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
        let pledge: Option<ReciprocityPledge> = record
            .entry()
            .to_app_option()
            .ok()
            .flatten();
        if let Some(p) = pledge {
            // Amount weight: log1p so $1000 doesn't dominate 10x over $100
            let amount_weight = match p.amount {
                Some(a) if a > 0.0 && a.is_finite() => (1.0 + a).ln(),
                _ => 1.0, // non-financial pledges get base weight
            };

            // Recency weight: exponential decay, half-life ~6 months
            let pledge_micros = p.pledged_at.as_micros();
            let age_days = (now_micros.saturating_sub(pledge_micros) as f64)
                / (86_400.0 * 1_000_000.0);
            let recency_weight = (-age_days / 180.0_f64).exp();

            // Type weight: Financial > DeveloperTime > Compute > others
            let type_weight = match p.pledge_type {
                PledgeType::Financial => 1.5,
                PledgeType::DeveloperTime => 1.3,
                PledgeType::Compute => 1.2,
                PledgeType::Bandwidth => 1.1,
                PledgeType::QA => 1.1,
                PledgeType::Documentation => 1.0,
                PledgeType::Other => 0.8,
            };

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

    let pledge_type_counts: Vec<(String, u64)> =
        type_counts.into_iter().collect();

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
    let dep_tag = format!("pledges:{}", input.id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&dep_tag)?,
            LinkTypes::DependencyToPledges,
        )?,
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
        let entry_hash = EntryHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
        })?;
        if let Some(record) = get(entry_hash, GetOptions::default())? {
            items.push(record);
        }
    }

    Ok(PaginatedPledges {
        items,
        total,
        offset: input.pagination.offset,
        limit: input.pagination.limit,
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
pub fn get_stewardship_leaderboard(
    limit: u64,
) -> ExternResult<Vec<LeaderboardEntry>> {
    // Get all deps from registry via cross-zome
    let encoded = ExternIO::encode(()).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to encode: {}",
            e
        )))
    })?;

    let dep_records: Vec<Record> = match call(
        CallTargetCell::Local,
        ZomeName::from("registry"),
        FunctionName::from("get_all_dependencies"),
        None,
        encoded,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => io.decode().unwrap_or_default(),
        _ => return Ok(Vec::new()),
    };

    // Compute stewardship for each
    let mut entries: Vec<LeaderboardEntry> = Vec::new();

    for record in dep_records {
        let dep: Option<DependencyIdMirror> = record
            .entry()
            .to_app_option()
            .ok()
            .flatten();
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

    // Sort by weighted_score descending
    entries.sort_by(|a, b| {
        b.weighted_score
            .partial_cmp(&a.weighted_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(limit as usize);

    Ok(entries)
}

#[hdk_extern]
pub fn get_under_supported_dependencies(
    limit: u64,
) -> ExternResult<Vec<LeaderboardEntry>> {
    // Same approach: get all deps, compute stewardship, sort by lowest ratio
    let encoded = ExternIO::encode(()).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to encode: {}",
            e
        )))
    })?;

    let dep_records: Vec<Record> = match call(
        CallTargetCell::Local,
        ZomeName::from("registry"),
        FunctionName::from("get_all_dependencies"),
        None,
        encoded,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => io.decode().unwrap_or_default(),
        _ => return Ok(Vec::new()),
    };

    let mut entries: Vec<LeaderboardEntry> = Vec::new();

    for record in dep_records {
        let dep: Option<DependencyIdMirror> = record
            .entry()
            .to_app_option()
            .ok()
            .flatten();
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

    // Sort by weighted_score ascending (most under-supported first)
    entries.sort_by(|a, b| {
        a.weighted_score
            .partial_cmp(&b.weighted_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(limit as usize);

    Ok(entries)
}

/// Mirror type for deserializing DependencyIdentity from registry zome.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct DependencyIdMirror {
    id: String,
}

holochain_serialized_bytes::holochain_serial!(DependencyIdMirror);
