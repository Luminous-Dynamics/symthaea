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
    // Count pledges directly
    let dep_tag = format!("pledges:{}", dep_id);
    let pledge_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&dep_tag)?,
            LinkTypes::DependencyToPledges,
        )?,
        GetStrategy::default(),
    )?;
    let pledge_count = pledge_links.len() as u64;

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

    Ok(StewardshipScore {
        dependency_id: dep_id,
        usage_count,
        pledge_count,
        ratio,
    })
}
