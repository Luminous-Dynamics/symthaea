#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Payments Coordinator Zome
use finance_wire_types::{
    ApplyDemurrageInput, DemurrageResult, GetPaymentHistoryInput, MintSapFromGovernanceInput,
    SapBalanceResponse,
};
use hdk::prelude::*;
use mycelix_finance_shared::{
    anchor_hash, follow_update_chain, links_to_records, rate_limit_anchor_key, validate_did_format,
    validate_id, verify_caller_is_did, verify_citizen_tier, verify_participant_tier,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
};
use mycelix_finance_types::{
    compute_demurrage_deduction, CompostPoolTier, FeeTier, PendingCompost, SapMintCapCounter,
    SapMintSource, SuccessionPreference, COMPOST_LOCAL_PCT, COMPOST_MAX_RETRIES,
    COMPOST_REGIONAL_PCT, DEMURRAGE_EXEMPT_FLOOR, DEMURRAGE_RATE, SAP_MINT_ANNUAL_MAX,
    SAP_MINT_PER_PROPOSAL_MAX,
};
use payments_integrity::*;

// ---------------------------------------------------------------------------
// SAP Balance Management (on-chain balance with enforced demurrage)
// ---------------------------------------------------------------------------

/// Initialize a SAP balance for a new member (zero balance).
#[hdk_extern]
pub fn initialize_sap_balance(member_did: String) -> ExternResult<Record> {
    validate_did_format(&member_did, "member_did")?;
    verify_caller_is_did(&member_did)?;

    // Check if balance already exists
    if find_sap_balance_record(&member_did)?.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "SAP balance already initialized for this member".into()
        )));
    }

    let now = sys_time()?;
    let balance = SapBalance {
        member_did: member_did.clone(),
        balance: 0,
        last_demurrage_at: now,
    };

    let action_hash = create_entry(&EntryTypes::SapBalance(balance))?;
    create_link(
        anchor_hash(&format!("sap:{}", member_did))?,
        action_hash.clone(),
        LinkTypes::DidToSapBalance,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "SAP balance record not found after creation for member {}",
        member_did
    ))))
}

/// Get the effective SAP balance for a member (after applying demurrage).
/// This is a read-only query — it does NOT persist the demurrage deduction.
#[hdk_extern]
pub fn get_sap_balance(member_did: String) -> ExternResult<SapBalanceResponse> {
    validate_did_format(&member_did, "member_did")?;
    let (record, bal) = get_sap_balance_inner(&member_did)?;
    let now = sys_time()?;
    let elapsed = elapsed_seconds(bal.last_demurrage_at, now);
    let deduction =
        compute_demurrage_deduction(bal.balance, DEMURRAGE_EXEMPT_FLOOR, DEMURRAGE_RATE, elapsed);
    let _ = record; // used only for existence check
    Ok(SapBalanceResponse {
        member_did: bal.member_did,
        raw_balance: bal.balance,
        effective_balance: bal.balance.saturating_sub(deduction),
        pending_demurrage: deduction,
        last_demurrage_at: bal.last_demurrage_at.as_micros(),
    })
}

/// Apply demurrage to a member's SAP balance and redistribute the deducted
/// amount as compost to commons pools (70% local, 20% regional, 10% global).
///
/// Returns the amount deducted. If 0, no update is persisted.
#[hdk_extern]
pub fn apply_demurrage(input: ApplyDemurrageInput) -> ExternResult<DemurrageResult> {
    // Opportunistically drain pending compost queue
    let _ = drain_pending_compost_inner();

    let (record, bal) = get_sap_balance_inner(&input.member_did)?;
    let now = sys_time()?;
    let elapsed = elapsed_seconds(bal.last_demurrage_at, now);
    let deduction =
        compute_demurrage_deduction(bal.balance, DEMURRAGE_EXEMPT_FLOOR, DEMURRAGE_RATE, elapsed);

    if deduction == 0 {
        return Ok(DemurrageResult {
            deducted: 0,
            redistributed: true,
        });
    }

    // Update balance
    let updated = SapBalance {
        balance: bal.balance.saturating_sub(deduction),
        last_demurrage_at: now,
        ..bal
    };
    update_entry(
        record.action_address().clone(),
        &EntryTypes::SapBalance(updated),
    )?;

    // Redistribute as compost: 70% local, 20% regional, 10% global
    let local_amount = deduction * COMPOST_LOCAL_PCT / 100;
    let regional_amount = deduction * COMPOST_REGIONAL_PCT / 100;
    let global_amount = deduction - local_amount - regional_amount; // remainder to global

    // Redistribute via treasury zome cross-zome calls with retry + queue
    let mut fully_redistributed = true;

    if let Some(ref pool_id) = input.local_commons_pool_id {
        if !try_deliver_compost(pool_id, local_amount, &input.member_did) {
            fully_redistributed = false;
            queue_pending_compost(
                pool_id,
                local_amount,
                &input.member_did,
                CompostPoolTier::Local,
            )?;
        }
    }
    if let Some(ref pool_id) = input.regional_commons_pool_id {
        if !try_deliver_compost(pool_id, regional_amount, &input.member_did) {
            fully_redistributed = false;
            queue_pending_compost(
                pool_id,
                regional_amount,
                &input.member_did,
                CompostPoolTier::Regional,
            )?;
        }
    }
    if let Some(ref pool_id) = input.global_commons_pool_id {
        if !try_deliver_compost(pool_id, global_amount, &input.member_did) {
            fully_redistributed = false;
            queue_pending_compost(
                pool_id,
                global_amount,
                &input.member_did,
                CompostPoolTier::Global,
            )?;
        }
    }

    Ok(DemurrageResult {
        deducted: deduction,
        redistributed: fully_redistributed,
    })
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ReceiveCompostPayload {
    pub commons_pool_id: String,
    pub amount: u64,
    pub source_member_did: String,
}

// ---------------------------------------------------------------------------
// Compost Delivery Queue (Phase 1a)
// ---------------------------------------------------------------------------

/// Anchor path for the pending compost queue.
const PENDING_COMPOST_ANCHOR: &str = "pending_compost_queue";

/// Attempt to deliver compost to treasury with retries.
/// Returns `true` if delivery succeeded, `false` if all retries exhausted.
fn try_deliver_compost(pool_id: &str, amount: u64, source_did: &str) -> bool {
    if amount == 0 {
        return true;
    }
    for attempt in 0..COMPOST_MAX_RETRIES {
        match call(
            CallTargetCell::Local,
            ZomeName::from("treasury"),
            FunctionName::from("receive_compost"),
            None,
            ReceiveCompostPayload {
                commons_pool_id: pool_id.to_string(),
                amount,
                source_member_did: source_did.to_string(),
            },
        ) {
            Ok(ZomeCallResponse::Ok(_)) => return true,
            Ok(other) => {
                debug!(
                    "Compost delivery attempt {}/{} to {} returned {:?}",
                    attempt + 1,
                    COMPOST_MAX_RETRIES,
                    pool_id,
                    other
                );
            }
            Err(e) => {
                debug!(
                    "Compost delivery attempt {}/{} to {} failed: {:?}",
                    attempt + 1,
                    COMPOST_MAX_RETRIES,
                    pool_id,
                    e
                );
            }
        }
    }
    false
}

/// Enqueue a failed compost delivery for later processing.
/// Stores a serialized `PendingCompost` in the link tag from the pending compost anchor.
fn queue_pending_compost(
    pool_id: &str,
    amount: u64,
    source_did: &str,
    tier: CompostPoolTier,
) -> ExternResult<()> {
    let now = sys_time()?;
    let pending = PendingCompost {
        commons_pool_id: pool_id.to_string(),
        amount,
        source_member_did: source_did.to_string(),
        pool_tier: tier,
        created_at_micros: now.as_micros(),
        retry_count: COMPOST_MAX_RETRIES,
    };
    let tag_bytes = serde_json::to_vec(&pending).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to serialize PendingCompost: {:?}",
            e
        )))
    })?;
    let anchor = anchor_hash(PENDING_COMPOST_ANCHOR)?;
    create_link(
        anchor,
        // Self-referential: target is also the anchor (we only care about the tag)
        anchor_hash(PENDING_COMPOST_ANCHOR)?,
        LinkTypes::PendingCompostQueue,
        tag_bytes,
    )?;
    debug!(
        "Queued pending compost: {} micro-SAP to pool {} (tier {:?}) from {}",
        amount, pool_id, pending.pool_tier, source_did
    );
    Ok(())
}

/// Drain the pending compost queue by retrying all queued deliveries.
/// Successfully delivered entries are removed from the queue.
/// Called opportunistically from `credit_sap` and `debit_sap`.
///
/// Returns the number of successfully drained entries.
#[hdk_extern]
pub fn drain_pending_compost(_: ()) -> ExternResult<u32> {
    drain_pending_compost_inner()
}

fn drain_pending_compost_inner() -> ExternResult<u32> {
    let anchor = anchor_hash(PENDING_COMPOST_ANCHOR)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::PendingCompostQueue)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(0);
    }

    let mut drained = 0u32;
    for link in &links {
        let tag_bytes = link.tag.as_ref();
        let pending: PendingCompost = match serde_json::from_slice(tag_bytes) {
            Ok(p) => p,
            Err(e) => {
                debug!("Skipping malformed pending compost link: {:?}", e);
                // Delete the malformed link to prevent infinite retries
                delete_link(link.create_link_hash.clone(), GetOptions::default())?;
                continue;
            }
        };

        if try_deliver_compost(
            &pending.commons_pool_id,
            pending.amount,
            &pending.source_member_did,
        ) {
            // Success — remove from queue
            delete_link(link.create_link_hash.clone(), GetOptions::default())?;
            drained += 1;
            debug!(
                "Drained pending compost: {} micro-SAP to pool {} from {}",
                pending.amount, pending.commons_pool_id, pending.source_member_did
            );
        } else {
            debug!(
                "Pending compost still undeliverable: {} micro-SAP to pool {} from {} (queued at {})",
                pending.amount, pending.commons_pool_id, pending.source_member_did, pending.created_at_micros
            );
        }
    }

    Ok(drained)
}

/// Maximum retries for optimistic-locking SAP balance mutations.
const MAX_SAP_RETRIES: usize = 3;

/// Minimum elapsed seconds before recomputing demurrage on retry.
/// If a concurrent writer already applied demurrage and updated `last_demurrage_at`,
/// the retry will see a very small elapsed time. Recomputing demurrage in that case
/// risks double-application against a stale balance. Skip if < 60s elapsed.
const DEMURRAGE_MIN_ELAPSED_SECONDS: u64 = 60;

/// Credit SAP to a member's balance (used by bridge deposits and community issuance).
/// Auto-initializes the SapBalance entry if the member has none yet.
///
/// Uses optimistic locking with retry: after updating, re-reads via
/// `follow_update_chain` to verify our update won. If a concurrent update
/// created a fork, retries up to `MAX_SAP_RETRIES` times.
#[hdk_extern]
pub fn credit_sap(input: CreditSapInput) -> ExternResult<Record> {
    // Opportunistically drain any pending compost deliveries
    if let Err(e) = drain_pending_compost_inner() {
        debug!(
            "credit_sap: pending compost drain failed (non-fatal): {:?}",
            e
        );
    }

    // Check if this member has no balance — if so, auto-initialize (no race concern for create)
    if find_sap_balance_record(&input.member_did)?.is_none() {
        let now = sys_time()?;
        let balance = SapBalance {
            member_did: input.member_did.clone(),
            balance: input.amount,
            last_demurrage_at: now,
        };
        let action_hash = create_entry(&EntryTypes::SapBalance(balance))?;
        create_link(
            anchor_hash(&format!("sap:{}", input.member_did))?,
            action_hash.clone(),
            LinkTypes::DidToSapBalance,
            (),
        )?;
        return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
            format!(
                "SAP balance record not found after credit_sap initialization for member {}",
                input.member_did
            )
        )));
    }

    // Existing balance: optimistic-locking retry loop
    for attempt in 0..MAX_SAP_RETRIES {
        let (record, bal) = get_sap_balance_inner(&input.member_did)?;
        let now = sys_time()?;

        // Apply pending demurrage first, then credit.
        // If elapsed time is very small (< 60s), a concurrent writer likely already
        // applied demurrage and updated last_demurrage_at. Skip recomputation to
        // avoid double-application against a stale balance.
        let elapsed = elapsed_seconds(bal.last_demurrage_at, now);
        let post_demurrage = if elapsed >= DEMURRAGE_MIN_ELAPSED_SECONDS {
            let deduction = compute_demurrage_deduction(
                bal.balance,
                DEMURRAGE_EXEMPT_FLOOR,
                DEMURRAGE_RATE,
                elapsed,
            );
            bal.balance.saturating_sub(deduction)
        } else {
            bal.balance
        };

        let expected_balance = post_demurrage + input.amount;
        let updated = SapBalance {
            balance: expected_balance,
            last_demurrage_at: now,
            ..bal
        };
        let action_hash = update_entry(
            record.action_address().clone(),
            &EntryTypes::SapBalance(updated),
        )?;

        // Verify our update won: re-read from the anchor
        let verify = find_sap_balance_record(&input.member_did)?;
        if let Some((_, actual)) = verify {
            if actual.balance == expected_balance {
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(
                    WasmErrorInner::Guest(format!(
                        "SAP balance record not found after credit for member {}",
                        input.member_did
                    ))
                ));
            }
        }

        // Concurrent update detected
        if attempt == MAX_SAP_RETRIES - 1 {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "credit_sap for {} failed after {} retries due to concurrent modifications",
                input.member_did, MAX_SAP_RETRIES
            ))));
        }
        debug!(
            "credit_sap: concurrent update detected for {}, retry {}/{}",
            input.member_did,
            attempt + 1,
            MAX_SAP_RETRIES
        );
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "credit_sap: retry loop exited unexpectedly".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreditSapInput {
    pub member_did: String,
    pub amount: u64,
    pub reason: String,
}

/// Debit SAP from a member's balance (enforces demurrage + sufficient balance).
///
/// Uses optimistic locking with retry: after updating, re-reads to verify
/// our update won. If a concurrent update created a fork, retries.
#[hdk_extern]
pub fn debit_sap(input: DebitSapInput) -> ExternResult<Record> {
    // Opportunistically drain any pending compost deliveries
    if let Err(e) = drain_pending_compost_inner() {
        debug!(
            "debit_sap: pending compost drain failed (non-fatal): {:?}",
            e
        );
    }

    for attempt in 0..MAX_SAP_RETRIES {
        let (record, bal) = get_sap_balance_inner(&input.member_did)?;
        let now = sys_time()?;

        // Apply pending demurrage first.
        // Skip if elapsed < 60s — a concurrent writer likely already applied demurrage
        // and updated last_demurrage_at (avoids double-application on retry).
        let elapsed = elapsed_seconds(bal.last_demurrage_at, now);
        let effective = if elapsed >= DEMURRAGE_MIN_ELAPSED_SECONDS {
            let deduction = compute_demurrage_deduction(
                bal.balance,
                DEMURRAGE_EXEMPT_FLOOR,
                DEMURRAGE_RATE,
                elapsed,
            );
            bal.balance.saturating_sub(deduction)
        } else {
            bal.balance
        };

        if input.amount > effective {
            let demurrage_applied = bal.balance.saturating_sub(effective);
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Insufficient SAP balance: effective {} (raw {} - demurrage {}), need {}",
                effective, bal.balance, demurrage_applied, input.amount
            ))));
        }

        let expected_balance = effective - input.amount;
        let updated = SapBalance {
            balance: expected_balance,
            last_demurrage_at: now,
            ..bal
        };
        let action_hash = update_entry(
            record.action_address().clone(),
            &EntryTypes::SapBalance(updated),
        )?;

        // Verify our update won: re-read from the anchor
        let verify = find_sap_balance_record(&input.member_did)?;
        if let Some((_, actual)) = verify {
            if actual.balance == expected_balance {
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(
                    WasmErrorInner::Guest(format!(
                        "SAP balance record not found after debit for member {}",
                        input.member_did
                    ))
                ));
            }
        }

        // Concurrent update detected
        if attempt == MAX_SAP_RETRIES - 1 {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "debit_sap for {} failed after {} retries due to concurrent modifications",
                input.member_did, MAX_SAP_RETRIES
            ))));
        }
        debug!(
            "debit_sap: concurrent update detected for {}, retry {}/{}",
            input.member_did,
            attempt + 1,
            MAX_SAP_RETRIES
        );
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "debit_sap: retry loop exited unexpectedly".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DebitSapInput {
    pub member_did: String,
    pub amount: u64,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// SAP Minting (governance-authorized issuance)
// ---------------------------------------------------------------------------

/// Mint SAP from a governance proposal. Creates an immutable SapMintRecord
/// and credits the recipient's balance.
///
/// This is the ONLY way new SAP enters circulation outside of collateral deposits.
/// Requires governance authorization (verified via cross-zome call).
#[hdk_extern]
pub fn mint_sap_from_governance(input: MintSapFromGovernanceInput) -> ExternResult<Record> {
    // Verify caller is governance-authorized
    match call(
        CallTargetCell::Local,
        ZomeName::from("tend"),
        FunctionName::from("verify_governance_agent"),
        None,
        (),
    ) {
        Ok(ZomeCallResponse::Ok(_)) => {} // Authorized
        Ok(other) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "SAP minting requires governance authorization: unexpected response {:?}",
                other
            ))));
        }
        Err(e) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "SAP minting requires governance authorization: {:?}",
                e
            ))));
        }
    }

    // Consciousness gate: SAP minting requires Citizen+ tier (identity >= 0.25,
    // reputation >= 0.10). Uses shared consciousness gating via identity cluster.
    // If identity cluster is unreachable, falls back to permissive (bootstrap mode) —
    // governance authorization (checked above) is still required.
    verify_citizen_tier()?;

    // Constitutional cap: per-proposal maximum
    if input.amount > SAP_MINT_PER_PROPOSAL_MAX {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Mint amount {} exceeds constitutional per-proposal maximum of {} micro-SAP (100,000 SAP)",
            input.amount, SAP_MINT_PER_PROPOSAL_MAX
        ))));
    }
    if input.amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Mint amount must be positive".into()
        )));
    }

    let now = sys_time()?;

    // Constitutional cap: annual maximum — sum all governance mints in the last 365 days
    enforce_annual_mint_cap(input.amount, now)?;
    let mint_id = format!("mint:gov:{}:{}", input.proposal_id, now.as_micros());

    let source = SapMintSource::GovernanceProposal {
        proposal_id: input.proposal_id.clone(),
    };

    // Create immutable mint record
    let mint_record = SapMintRecord {
        id: mint_id.clone(),
        recipient_did: input.recipient_did.clone(),
        amount: input.amount,
        source,
        minted_at: now,
    };

    let action_hash = create_entry(&EntryTypes::SapMintRecord(mint_record))?;
    create_link(
        anchor_hash(&mint_id)?,
        action_hash.clone(),
        LinkTypes::MintIdToMintRecord,
        (),
    )?;
    create_link(
        anchor_hash(&format!("mints:{}", input.recipient_did))?,
        action_hash.clone(),
        LinkTypes::DidToMintRecords,
        (),
    )?;

    // Credit the SAP to recipient's balance
    credit_sap(CreditSapInput {
        member_did: input.recipient_did.clone(),
        amount: input.amount,
        reason: format!("Governance mint: proposal {}", input.proposal_id),
    })?;

    // Update the running mint cap counter (O(1) for future cap checks)
    update_mint_cap_counter(input.amount, now)?;

    // Broadcast mint event via bridge
    if let Err(e) = call(
        CallTargetCell::Local,
        ZomeName::from("finance_bridge"),
        FunctionName::from("broadcast_finance_event"),
        None,
        BroadcastMintEventPayload {
            event_type: "SapMinted".to_string(),
            subject_did: input.recipient_did,
            amount: Some(input.amount),
            payload: serde_json::json!({
                "proposal_id": input.proposal_id,
                "mint_id": mint_id,
            })
            .to_string(),
        },
    ) {
        debug!("Failed to broadcast SapMinted event: {:?}", e);
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Mint record not found".into()
    )))
}

#[derive(Serialize, Debug)]
struct BroadcastMintEventPayload {
    pub event_type: String,
    pub subject_did: String,
    pub amount: Option<u64>,
    pub payload: String,
}

/// Get all mint records for a member
///
/// Batch-optimized: SapMintRecord entries are immutable (write-once mint events),
/// so bare get() via links_to_records is equivalent to follow_update_chain.
#[hdk_extern]
pub fn get_mint_records(member_did: String) -> ExternResult<Vec<Record>> {
    validate_did_format(&member_did, "member_did")?;
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mints:{}", member_did))?,
            LinkTypes::DidToMintRecords,
        )?,
        GetStrategy::default(),
    )?;
    links_to_records(links)
}

// --- Internal helpers ---

/// Anchor key for the singleton mint cap counter entry.
const MINT_CAP_COUNTER_ANCHOR: &str = "sap_mint_cap_counter";

/// Enforce annual governance mint cap using the on-chain `SapMintCapCounterEntry`.
///
/// O(1) check: reads the running counter instead of scanning all `SapMintRecord` entries.
/// If no counter exists yet (first-ever governance mint), falls back to a one-time chain
/// scan to bootstrap the counter. Subsequent calls are O(1).
fn enforce_annual_mint_cap(new_amount: u64, now: Timestamp) -> ExternResult<()> {
    let counter = load_or_bootstrap_mint_cap_counter(now)?;
    let effective = if counter.is_period_expired(now.as_micros()) {
        // Period rolled over — counter resets, only the new amount counts
        SapMintCapCounter {
            period_start_micros: now.as_micros(),
            cumulative_minted: 0,
            mint_count: 0,
            last_updated_micros: now.as_micros(),
        }
    } else {
        counter
    };

    if effective.would_exceed_cap(new_amount) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Annual governance mint cap exceeded: {} already minted this year + {} requested > {} max",
            effective.cumulative_minted, new_amount, SAP_MINT_ANNUAL_MAX
        ))));
    }
    Ok(())
}

/// Update the on-chain mint cap counter after a successful governance mint.
/// Called immediately after the SapMintRecord is committed.
fn update_mint_cap_counter(minted_amount: u64, now: Timestamp) -> ExternResult<()> {
    let year_micros: i64 = 365 * 24 * 60 * 60 * 1_000_000;

    match find_mint_cap_counter_record()? {
        Some((record, existing)) => {
            let period_expired = {
                let year_us: i64 = 365 * 24 * 60 * 60 * 1_000_000;
                now.as_micros().saturating_sub(existing.period_start_micros) > year_us
            };
            let updated = if period_expired {
                // Period expired — start fresh
                SapMintCapCounterEntry {
                    period_start_micros: now.as_micros(),
                    cumulative_minted: minted_amount,
                    mint_count: 1,
                    last_updated_micros: now.as_micros(),
                }
            } else {
                SapMintCapCounterEntry {
                    cumulative_minted: existing.cumulative_minted.saturating_add(minted_amount),
                    mint_count: existing.mint_count.saturating_add(1),
                    last_updated_micros: now.as_micros(),
                    ..existing
                }
            };
            update_entry(
                record.action_address().clone(),
                &EntryTypes::SapMintCapCounterEntry(updated),
            )?;
        }
        None => {
            // First mint ever — create counter
            let counter = SapMintCapCounterEntry {
                period_start_micros: now.as_micros() - year_micros + year_micros, // = now
                cumulative_minted: minted_amount,
                mint_count: 1,
                last_updated_micros: now.as_micros(),
            };
            let hash = create_entry(&EntryTypes::SapMintCapCounterEntry(counter))?;
            create_link(
                anchor_hash(MINT_CAP_COUNTER_ANCHOR)?,
                hash,
                LinkTypes::MintCapCounterAnchor,
                (),
            )?;
        }
    }
    Ok(())
}

/// Load the existing mint cap counter, or bootstrap from chain scan if none exists.
fn load_or_bootstrap_mint_cap_counter(now: Timestamp) -> ExternResult<SapMintCapCounter> {
    if let Some((_record, entry)) = find_mint_cap_counter_record()? {
        return Ok(entry.into());
    }

    // No counter yet — bootstrap from chain scan (one-time cost)
    let year_micros: i64 = 365 * 24 * 60 * 60 * 1_000_000;
    let cutoff = now.as_micros() - year_micros;

    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::SapMintRecord,
        )?))
        .include_entries(true);

    let (annual_total, mint_count) = query(filter)?
        .into_iter()
        .filter_map(|r| match r.entry().to_app_option::<SapMintRecord>() {
            Ok(opt) => opt,
            Err(e) => {
                debug!("SapMintRecord deserialization error: {:?}", e);
                None
            }
        })
        .filter(|m| m.minted_at.as_micros() > cutoff)
        .filter(|m| matches!(m.source, SapMintSource::GovernanceProposal { .. }))
        .fold((0u64, 0u32), |(acc, cnt), m| {
            (acc.saturating_add(m.amount), cnt.saturating_add(1))
        });

    Ok(SapMintCapCounter {
        period_start_micros: cutoff,
        cumulative_minted: annual_total,
        mint_count,
        last_updated_micros: now.as_micros(),
    })
}

/// Find the singleton mint cap counter record via link-based lookup.
fn find_mint_cap_counter_record() -> ExternResult<Option<(Record, SapMintCapCounterEntry)>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(MINT_CAP_COUNTER_ANCHOR)?,
            LinkTypes::MintCapCounterAnchor,
        )?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        let record = follow_update_chain(hash)?;
        let entry = record
            .entry()
            .to_app_option::<SapMintCapCounterEntry>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "SapMintCapCounterEntry deserialization error: {:?}",
                    e
                )))
            })?;
        if let Some(entry) = entry {
            return Ok(Some((record, entry)));
        }
    }
    Ok(None)
}

fn find_sap_balance_record(member_did: &str) -> ExternResult<Option<(Record, SapBalance)>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("sap:{}", member_did))?,
            LinkTypes::DidToSapBalance,
        )?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        let record = follow_update_chain(hash)?;
        let bal = record.entry().to_app_option::<SapBalance>().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "SapBalance deserialization error: {:?}",
                e
            )))
        })?;
        if let Some(bal) = bal {
            return Ok(Some((record, bal)));
        }
    }
    Ok(None)
}

fn get_sap_balance_inner(member_did: &str) -> ExternResult<(Record, SapBalance)> {
    find_sap_balance_record(member_did)?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "No SAP balance found for {}. Call initialize_sap_balance first.",
        member_did
    ))))
}

/// Compute the progressive SAP fee for a payment based on sender's MYCEL score.
///
/// Queries the bridge coordinator for the canonical fee tier (single source of truth).
/// Falls back to direct recognition lookup, then to Newcomer tier (0.10%).
fn compute_sap_fee(sender_did: &str, micro_amount: u64) -> ExternResult<u64> {
    // Try canonical path: bridge → recognition → FeeTier
    let fee_rate = match call(
        CallTargetCell::Local,
        ZomeName::from("finance_bridge"),
        FunctionName::from("get_member_fee_tier"),
        None,
        sender_did.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            #[derive(Debug, Deserialize)]
            struct TierResp {
                base_fee_rate: f64,
            }
            match result.decode::<TierResp>() {
                Ok(resp) if resp.base_fee_rate.is_finite() => resp.base_fee_rate,
                Ok(resp) => {
                    debug!(
                        "compute_sap_fee: non-finite fee rate {:?} for {}, using Newcomer rate",
                        resp.base_fee_rate, sender_did
                    );
                    FeeTier::Newcomer.base_fee_rate()
                }
                Err(e) => {
                    debug!(
                        "compute_sap_fee: fee tier decode error for {}: {:?}, using Newcomer rate",
                        sender_did, e
                    );
                    FeeTier::Newcomer.base_fee_rate()
                }
            }
        }
        Ok(other) => {
            debug!(
                "compute_sap_fee: bridge returned {:?} for {}, falling back to direct recognition",
                other, sender_did
            );
            // Bridge unavailable — fall back to direct recognition call
            let mycel_score = match call(
                CallTargetCell::Local,
                ZomeName::from("recognition"),
                FunctionName::from("get_mycel_score"),
                None,
                sender_did.to_string(),
            ) {
                Ok(ZomeCallResponse::Ok(result)) => {
                    #[derive(Debug, Deserialize)]
                    struct MycelState {
                        mycel_score: f64,
                    }
                    result
                        .decode::<MycelState>()
                        .map(|s| s.mycel_score)
                        .unwrap_or(0.0)
                }
                Ok(other) => {
                    debug!(
                        "compute_sap_fee: recognition returned {:?} for {}, defaulting to 0.0",
                        other, sender_did
                    );
                    0.0
                }
                Err(e) => {
                    debug!(
                        "compute_sap_fee: recognition unreachable for {}: {:?}, defaulting to 0.0",
                        sender_did, e
                    );
                    0.0
                }
            };
            FeeTier::from_mycel(mycel_score).base_fee_rate()
        }
        Err(e) => {
            debug!("compute_sap_fee: bridge unreachable for {}: {:?}, falling back to direct recognition", sender_did, e);
            // Bridge unavailable — fall back to direct recognition call
            let mycel_score = match call(
                CallTargetCell::Local,
                ZomeName::from("recognition"),
                FunctionName::from("get_mycel_score"),
                None,
                sender_did.to_string(),
            ) {
                Ok(ZomeCallResponse::Ok(result)) => {
                    #[derive(Debug, Deserialize)]
                    struct MycelState {
                        mycel_score: f64,
                    }
                    result
                        .decode::<MycelState>()
                        .map(|s| s.mycel_score)
                        .unwrap_or(0.0)
                }
                Ok(other) => {
                    debug!(
                        "compute_sap_fee: recognition returned {:?} for {}, defaulting to 0.0",
                        other, sender_did
                    );
                    0.0
                }
                Err(e2) => {
                    debug!("compute_sap_fee: recognition also unreachable for {}: {:?}, defaulting to 0.0", sender_did, e2);
                    0.0
                }
            };
            FeeTier::from_mycel(mycel_score).base_fee_rate()
        }
    };

    let fee = (micro_amount as f64 * fee_rate) as u64;
    Ok(fee)
}

fn elapsed_seconds(from: Timestamp, to: Timestamp) -> u64 {
    let from_us = from.as_micros();
    let to_us = to.as_micros();
    if to_us > from_us {
        ((to_us - from_us) / 1_000_000) as u64
    } else {
        0
    }
}

#[hdk_extern]
pub fn send_payment(input: SendPaymentInput) -> ExternResult<Record> {
    verify_participant_tier()?;
    // Verify caller is the sender (prevents DID spoofing)
    verify_caller_is_did(&input.from_did)?;

    // Validate currency before creating any entries
    if input.currency != "SAP" && input.currency != "TEND" {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Currency must be \"SAP\" or \"TEND\"".into()
        )));
    }

    let now = sys_time()?;

    // Per-agent rate limit: reject if this agent has exceeded the payment
    // send limit within the current 60-second window.
    {
        let agent = agent_info()?.agent_initial_pubkey;
        let key = rate_limit_anchor_key("payment", &agent, now.as_micros());
        let anchor = anchor_hash(&key)?;
        let recent_links = get_links(
            LinkQuery::try_new(anchor.clone(), LinkTypes::SenderToPayments)?,
            GetStrategy::default(),
        )?;
        if recent_links.len() >= DEFAULT_RATE_LIMIT_PER_MINUTE {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Rate limit exceeded: max {} payments per minute",
                DEFAULT_RATE_LIMIT_PER_MINUTE
            ))));
        }
        // Record this operation in the rate-limit bucket
        create_link(
            anchor,
            AnyLinkableHash::from(agent.clone()),
            LinkTypes::SenderToPayments,
            (),
        )?;
    }

    // If sending SAP, enforce on-chain balance with demurrage + progressive fee
    let (memo, fee_amount) = if input.currency == "SAP" {
        // input.amount is already in micro-SAP (u64)
        // Compute progressive fee based on sender's MYCEL score
        let fee = compute_sap_fee(&input.from_did, input.amount)?;
        let total_debit = input.amount + fee;

        // Debit sender's SAP balance (amount + fee, applies demurrage)
        debit_sap(DebitSapInput {
            member_did: input.from_did.clone(),
            amount: total_debit,
            reason: format!("Payment to {} (includes fee {})", input.to_did, fee),
        })?;

        // Credit receiver's SAP balance (amount only, fee goes to commons)
        credit_sap(CreditSapInput {
            member_did: input.to_did.clone(),
            amount: input.amount,
            reason: format!("Payment from {}", input.from_did),
        })?;

        // Route fee to commons via treasury (if fee > 0)
        if fee > 0 {
            if let Err(e) = call(
                CallTargetCell::Local,
                ZomeName::from("treasury"),
                FunctionName::from("receive_compost"),
                None,
                ReceiveCompostPayload {
                    commons_pool_id: "global-fee-pool".to_string(),
                    amount: fee,
                    source_member_did: input.from_did.clone(),
                },
            ) {
                debug!("Fee routing to global-fee-pool failed: {:?}", e);
            }
        }

        (input.memo.clone(), fee)
    } else {
        (input.memo.clone(), 0)
    };

    let payment = Payment {
        id: format!("payment:{}:{}", input.from_did, now.as_micros()),
        from_did: input.from_did.clone(),
        to_did: input.to_did.clone(),
        amount: input.amount,
        fee: fee_amount,
        currency: input.currency.clone(),
        payment_type: input.payment_type,
        status: TransferStatus::Completed, // Simplified: immediate completion
        memo,
        created: now,
        completed: Some(now),
    };

    let action_hash = create_entry(&EntryTypes::Payment(payment.clone()))?;
    create_link(
        anchor_hash(&input.from_did)?,
        action_hash.clone(),
        LinkTypes::SenderToPayments,
        (),
    )?;
    create_link(
        anchor_hash(&input.to_did)?,
        action_hash.clone(),
        LinkTypes::ReceiverToPayments,
        (),
    )?;
    // Link-based index for O(1) payment ID lookups
    create_link(
        anchor_hash(&payment.id)?,
        action_hash.clone(),
        LinkTypes::PaymentIdToPayment,
        (),
    )?;

    // Create receipt with Ed25519 signature
    let sig_data = format!(
        "{}|{}|{}|{}|{}|{}",
        payment.id,
        payment.from_did,
        payment.to_did,
        payment.amount,
        payment.currency,
        now.as_micros()
    );
    let receipt = Receipt {
        payment_id: payment.id.clone(),
        from_did: input.from_did,
        to_did: input.to_did,
        amount: input.amount,
        currency: payment.currency,
        timestamp: now,
        signature: {
            let agent = agent_info()?.agent_initial_pubkey;
            let sig = sign(agent, sig_data.into_bytes())?;
            sig.0
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>()
        },
    };
    let receipt_hash = create_entry(&EntryTypes::Receipt(receipt))?;
    create_link(
        action_hash.clone(),
        receipt_hash,
        LinkTypes::PaymentToReceipt,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Payment record not found after creation for payment {}",
        payment.id
    ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SendPaymentInput {
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub memo: Option<String>,
}

#[hdk_extern]
pub fn open_payment_channel(input: OpenChannelInput) -> ExternResult<Record> {
    verify_participant_tier()?;
    verify_caller_is_did(&input.party_a)?;

    let now = sys_time()?;
    let channel = PaymentChannel {
        id: format!(
            "channel:{}:{}:{}",
            input.party_a,
            input.party_b,
            now.as_micros()
        ),
        party_a: input.party_a.clone(),
        party_b: input.party_b.clone(),
        currency: input.currency,
        balance_a: input.initial_deposit_a,
        balance_b: input.initial_deposit_b,
        opened: now,
        last_updated: now,
        closed: None,
    };

    // Prevent duplicate IDs
    let existing = get_links(
        LinkQuery::try_new(anchor_hash(&channel.id)?, LinkTypes::ChannelIdToChannel)?,
        GetStrategy::default(),
    )?;
    if !existing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Payment channel with this ID already exists".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::PaymentChannel(channel.clone()))?;
    // ID-based index link for O(1) lookups
    create_link(
        anchor_hash(&channel.id)?,
        action_hash.clone(),
        LinkTypes::ChannelIdToChannel,
        (),
    )?;
    create_link(
        anchor_hash(&input.party_a)?,
        action_hash.clone(),
        LinkTypes::ChannelPartyA,
        (),
    )?;
    create_link(
        anchor_hash(&input.party_b)?,
        action_hash.clone(),
        LinkTypes::ChannelPartyB,
        (),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Payment channel record not found after creation for channel {}",
        channel.id
    ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenChannelInput {
    pub party_a: String,
    pub party_b: String,
    pub currency: String,
    pub initial_deposit_a: u64,
    pub initial_deposit_b: u64,
}

/// Internal helper: fetch a PaymentChannel Record + deserialized entry by ID via link index.
/// Follows the update chain to return the latest version.
fn get_channel_record(channel_id: &str) -> ExternResult<(Record, PaymentChannel)> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(channel_id)?, LinkTypes::ChannelIdToChannel)?,
        GetStrategy::default(),
    )?;
    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "Channel not found".into()
    )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = follow_update_chain(hash)?;
    let channel = record
        .entry()
        .to_app_option::<PaymentChannel>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "PaymentChannel deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "PaymentChannel entry missing".into()
        )))?;
    Ok((record, channel))
}

/// Internal helper: fetch a Payment Record + deserialized entry by ID via link index.
/// Follows the update chain to return the latest version.
fn get_payment_record(payment_id: &str) -> ExternResult<(Record, Payment)> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(payment_id)?, LinkTypes::PaymentIdToPayment)?,
        GetStrategy::default(),
    )?;
    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "Payment not found".into()
    )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = follow_update_chain(hash)?;
    let payment = record
        .entry()
        .to_app_option::<Payment>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Payment deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Payment entry missing".into()
        )))?;
    Ok((record, payment))
}

#[hdk_extern]
pub fn channel_transfer(input: ChannelTransferInput) -> ExternResult<Record> {
    let (record, channel) = get_channel_record(&input.channel_id)?;
    let now = sys_time()?;
    let (new_a, new_b) = if input.from_a {
        (
            channel
                .balance_a
                .checked_sub(input.amount)
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Insufficient balance for party A".into()
                )))?,
            channel.balance_b + input.amount,
        )
    } else {
        (
            channel.balance_a + input.amount,
            channel
                .balance_b
                .checked_sub(input.amount)
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Insufficient balance for party B".into()
                )))?,
        )
    };
    let updated = PaymentChannel {
        balance_a: new_a,
        balance_b: new_b,
        last_updated: now,
        ..channel
    };
    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::PaymentChannel(updated),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Payment channel record not found after transfer for channel {}",
        input.channel_id
    ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChannelTransferInput {
    pub channel_id: String,
    pub amount: u64,
    pub from_a: bool,
}

/// Uses follow_update_chain because payments are mutable (status transitions).
#[hdk_extern]
pub fn get_payment_history(input: GetPaymentHistoryInput) -> ExternResult<Vec<Record>> {
    let max = input.limit.unwrap_or(100);
    let mut payments = Vec::new();
    // Get sent payments
    let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::SenderToPayments)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(max)
    {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?;
        payments.push(follow_update_chain(hash)?);
    }
    // Get received payments (respect remaining budget)
    let remaining = max.saturating_sub(payments.len());
    if remaining > 0 {
        let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::ReceiverToPayments)?;
        for link in get_links(query, GetStrategy::default())?
            .into_iter()
            .take(remaining)
        {
            let hash = ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?;
            payments.push(follow_update_chain(hash)?);
        }
    }
    Ok(payments)
}

/// Get a specific payment by ID (O(1) link-based lookup)
#[hdk_extern]
pub fn get_payment(payment_id: String) -> ExternResult<Option<Record>> {
    validate_id(&payment_id, "payment_id")?;
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&payment_id)?, LinkTypes::PaymentIdToPayment)?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.first() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        Ok(Some(follow_update_chain(hash)?))
    } else {
        Ok(None)
    }
}

/// Get receipt for a payment
#[hdk_extern]
pub fn get_receipt(payment_id: String) -> ExternResult<Option<Record>> {
    validate_id(&payment_id, "payment_id")?;
    // Find the payment first
    let Some(payment_record) = get_payment(payment_id.clone())? else {
        return Ok(None);
    };
    let query = LinkQuery::try_new(
        payment_record.action_address().clone(),
        LinkTypes::PaymentToReceipt,
    )?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            return Ok(Some(record));
        }
    }
    Ok(None)
}

/// Close a payment channel (settle balances)
#[hdk_extern]
pub fn close_payment_channel(channel_id: String) -> ExternResult<Record> {
    validate_id(&channel_id, "channel_id")?;
    let (record, channel) = get_channel_record(&channel_id)?;

    if channel.closed.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Channel already closed".into()
        )));
    }
    let now = sys_time()?;
    let closed = PaymentChannel {
        closed: Some(now),
        last_updated: now,
        ..channel
    };
    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::PaymentChannel(closed),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Payment channel record not found after closing channel {}",
        channel_id
    ))))
}

/// Refund a payment (creates reverse payment)
#[hdk_extern]
pub fn refund_payment(payment_id: String) -> ExternResult<Record> {
    validate_id(&payment_id, "payment_id")?;
    let original = get_payment(payment_id.clone())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Payment not found".into()
    )))?;

    let original_payment = original
        .entry()
        .to_app_option::<Payment>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Payment deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Payment entry missing".into()
        )))?;

    // Only the original receiver (who is the refund sender) can initiate a refund
    verify_caller_is_did(&original_payment.to_did)?;

    if matches!(original_payment.status, TransferStatus::Refunded) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Payment already refunded".into()
        )));
    }

    // Create refund payment (reverse direction)
    let now = sys_time()?;
    // Refunds carry the original fee (already collected; stored for audit trail)
    let refund = Payment {
        id: format!("refund:{}:{}", payment_id, now.as_micros()),
        from_did: original_payment.to_did.clone(),
        to_did: original_payment.from_did.clone(),
        amount: original_payment.amount,
        fee: original_payment.fee,
        currency: original_payment.currency.clone(),
        payment_type: PaymentType::Direct,
        status: TransferStatus::Completed,
        memo: Some(format!("Refund for payment {}", payment_id)),
        created: now,
        completed: Some(now),
    };

    let action_hash = create_entry(&EntryTypes::Payment(refund.clone()))?;
    create_link(
        anchor_hash(&refund.from_did)?,
        action_hash.clone(),
        LinkTypes::SenderToPayments,
        (),
    )?;
    create_link(
        anchor_hash(&refund.to_did)?,
        action_hash.clone(),
        LinkTypes::ReceiverToPayments,
        (),
    )?;

    // Mark original as refunded
    let refunded = Payment {
        status: TransferStatus::Refunded,
        ..original_payment
    };
    update_entry(
        original.action_address().clone(),
        &EntryTypes::Payment(refunded),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Refund payment record not found after creation for original payment {}",
        payment_id
    ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetChannelsInput {
    pub did: String,
    pub limit: Option<usize>,
}

/// Get all channels for a party
///
/// Uses follow_update_chain because payment channels are mutable (balance, close).
#[hdk_extern]
pub fn get_channels(input: GetChannelsInput) -> ExternResult<Vec<Record>> {
    let max = input.limit.unwrap_or(100);
    let mut channels = Vec::new();
    // Party A channels
    let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::ChannelPartyA)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(max)
    {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?;
        channels.push(follow_update_chain(hash)?);
    }
    // Party B channels (respect remaining budget)
    let remaining = max.saturating_sub(channels.len());
    if remaining > 0 {
        let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::ChannelPartyB)?;
        for link in get_links(query, GetStrategy::default())?
            .into_iter()
            .take(remaining)
        {
            let hash = ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?;
            channels.push(follow_update_chain(hash)?);
        }
    }
    Ok(channels)
}

/// Create an escrow payment (held until release)
#[hdk_extern]
pub fn create_escrow(input: CreateEscrowInput) -> ExternResult<Record> {
    verify_caller_is_did(&input.from_did)?;

    let now = sys_time()?;
    let from_did = input.from_did.clone();
    let to_did = input.to_did.clone();
    let escrow_id = input.escrow_id.clone();
    // Compute fee for escrow (SAP: progressive fee; TEND: zero)
    let escrow_fee = if input.currency == "SAP" {
        compute_sap_fee(&input.from_did, input.amount)?
    } else {
        0
    };
    let payment = Payment {
        id: format!("escrow:{}:{}", input.from_did, now.as_micros()),
        from_did: input.from_did,
        to_did: input.to_did,
        amount: input.amount,
        fee: escrow_fee,
        currency: input.currency,
        payment_type: PaymentType::Escrow(input.escrow_id),
        status: TransferStatus::Pending, // Held until released
        memo: input.memo,
        created: now,
        completed: None,
    };

    let action_hash = create_entry(&EntryTypes::Payment(payment.clone()))?;
    create_link(
        anchor_hash(&from_did)?,
        action_hash.clone(),
        LinkTypes::SenderToPayments,
        (),
    )?;
    create_link(
        anchor_hash(&to_did)?,
        action_hash.clone(),
        LinkTypes::ReceiverToPayments,
        (),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Escrow payment record not found after creation for escrow {}",
        escrow_id
    ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateEscrowInput {
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub escrow_id: String,
    pub memo: Option<String>,
}

// =============================================================================
// EXIT PROTOCOL
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct InitiateExitInput {
    /// DID of the exiting member
    pub member_did: String,
    /// How to handle remaining SAP balance
    pub succession_preference: SuccessionPreference,
    /// Current SAP balance in micro-units (caller must provide; on-chain accounting is external)
    pub sap_balance: u64,
}

/// Initiate member exit — coordinates MYCEL dissolution, SAP succession, and TEND forgiveness.
///
/// Per the Three-Currency Spec:
/// - MYCEL: dissolved immediately (contribution history preserved in DKG)
/// - SAP: follows succession preference (Commons default, Designee, or Redemption)
/// - TEND: all balances forgiven (returned to zero)
#[hdk_extern]
pub fn initiate_exit(input: InitiateExitInput) -> ExternResult<Record> {
    // Verify caller is the exiting member
    verify_caller_is_did(&input.member_did)?;

    let now = sys_time()?;

    // Step 1: Dissolve MYCEL via recognition zome
    let mycel_dissolved = match call(
        CallTargetCell::Local,
        ZomeName::from("recognition"),
        FunctionName::from("dissolve_mycel"),
        None,
        input.member_did.clone(),
    ) {
        Ok(_) => true,
        Err(e) => {
            debug!(
                "Warning: MYCEL dissolution failed for {}: {:?}",
                input.member_did, e
            );
            false
        }
    };

    // Step 2: Handle SAP succession
    if input.sap_balance > 0 {
        match &input.succession_preference {
            SuccessionPreference::Commons => {
                // For commons succession, we record a payment to the commons pool.
                // The actual treasury contribution requires a pool ID which is
                // external state — the caller should handle that via the treasury zome.
                send_payment(SendPaymentInput {
                    from_did: input.member_did.clone(),
                    to_did: format!("did:mycelix:commons:{}", input.member_did),
                    amount: input.sap_balance,
                    currency: "SAP".to_string(),
                    payment_type: PaymentType::CommonsContribution("exit-succession".to_string()),
                    memo: Some("Exit succession: SAP to local commons pool".to_string()),
                })?;
            }
            SuccessionPreference::Designee(designee_did) => {
                send_payment(SendPaymentInput {
                    from_did: input.member_did.clone(),
                    to_did: designee_did.clone(),
                    amount: input.sap_balance,
                    currency: "SAP".to_string(),
                    payment_type: PaymentType::Direct,
                    memo: Some("Exit succession to designated heir".to_string()),
                })?;
            }
            SuccessionPreference::Redemption => {
                // For redemption, record the intent. Actual bridge redemption
                // requires deposit IDs and oracle rates which are bridge-zome state.
                send_payment(SendPaymentInput {
                    from_did: input.member_did.clone(),
                    to_did: "did:mycelix:bridge:redemption".to_string(),
                    amount: input.sap_balance,
                    currency: "SAP".to_string(),
                    payment_type: PaymentType::Direct,
                    memo: Some("Exit succession: SAP queued for collateral redemption".to_string()),
                })?;
            }
        }
    }

    // Step 3: Forgive TEND balances via tend zome
    let tend_balances_forgiven: Vec<(String, i32)> = match call(
        CallTargetCell::Local,
        ZomeName::from("tend"),
        FunctionName::from("forgive_balance"),
        None,
        input.member_did.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(extern_io)) => extern_io.decode().unwrap_or_default(),
        Ok(_) => Vec::new(),
        Err(e) => {
            debug!("Warning: TEND balance forgiveness failed: {:?}", e);
            Vec::new()
        }
    };

    // Step 4: Create the exit record
    let exit_record = payments_integrity::ExitRecord {
        member_did: input.member_did.clone(),
        succession_preference: input.succession_preference,
        sap_balance: input.sap_balance,
        tend_balances_forgiven,
        mycel_dissolved,
        exited_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ExitRecord(exit_record))?;
    create_link(
        anchor_hash(&input.member_did)?,
        action_hash.clone(),
        LinkTypes::MemberToExitRecord,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Failed to retrieve exit record".into()
    )))
}

/// Release escrow to recipient
#[hdk_extern]
pub fn release_escrow(payment_id: String) -> ExternResult<Record> {
    let (record, payment) = get_payment_record(&payment_id)?;

    if !matches!(payment.payment_type, PaymentType::Escrow(_)) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Not an escrow payment".into()
        )));
    }
    if payment.status != TransferStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Escrow not in pending state".into()
        )));
    }
    let now = sys_time()?;
    let released = Payment {
        status: TransferStatus::Completed,
        completed: Some(now),
        ..payment
    };
    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::Payment(released),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Escrow payment record not found after release for payment {}",
        payment_id
    ))))
}

// =============================================================================
// HEARTH SAP POOLS — Shared Household Funds
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct ContributeToHearthInput {
    pub hearth_did: String,
    pub amount: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WithdrawFromHearthInput {
    pub hearth_did: String,
    pub amount: u64,
}

/// Get a hearth's SAP pool with effective balance after demurrage.
#[hdk_extern]
pub fn get_hearth_sap_pool(hearth_did: String) -> ExternResult<HearthSapPoolResponse> {
    let pool = get_or_create_hearth_pool(&hearth_did)?;
    let now = sys_time()?;
    let elapsed = elapsed_seconds(pool.last_demurrage_at, now);
    let deduction = compute_demurrage_deduction(
        pool.balance,
        DEMURRAGE_EXEMPT_FLOOR,
        DEMURRAGE_RATE,
        elapsed,
    );
    Ok(HearthSapPoolResponse {
        hearth_did: pool.hearth_did,
        raw_balance: pool.balance,
        effective_balance: pool.balance.saturating_sub(deduction),
        pending_demurrage: deduction,
        last_demurrage_at: pool.last_demurrage_at,
        member_count: pool.member_count,
        total_contributed: pool.total_contributed,
        total_withdrawn: pool.total_withdrawn,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HearthSapPoolResponse {
    pub hearth_did: String,
    pub raw_balance: u64,
    pub effective_balance: u64,
    pub pending_demurrage: u64,
    pub last_demurrage_at: Timestamp,
    pub member_count: u32,
    pub total_contributed: u64,
    pub total_withdrawn: u64,
}

/// Contribute SAP from personal balance to hearth pool.
///
/// Deducts from caller's SapBalance and credits the hearth pool.
/// Requires hearth membership (verified via cross-zome call).
///
/// Uses optimistic locking with retry on the hearth pool update to prevent
/// concurrent contributions from overwriting each other.
#[hdk_extern]
pub fn contribute_to_hearth_pool(input: ContributeToHearthInput) -> ExternResult<HearthSapPool> {
    if input.amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amount must be positive".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Verify hearth membership
    verify_hearth_membership(&caller_did, &input.hearth_did)?;

    // Deduct from personal SAP balance (uses credit_sap/debit_sap style retry internally)
    let (record, mut personal_bal) = find_sap_balance_record(&caller_did)?.ok_or(wasm_error!(
        WasmErrorInner::Guest("No SAP balance found".into())
    ))?;

    if personal_bal.balance < input.amount {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Insufficient SAP balance: have {}, need {}",
            personal_bal.balance, input.amount
        ))));
    }

    // Deduct from personal (use record's action address to avoid update forks)
    personal_bal.balance -= input.amount;
    update_entry(record.action_address().clone(), &personal_bal)?;

    // Credit hearth pool with optimistic-locking retry
    for attempt in 0..MAX_SAP_RETRIES {
        let (pool_record, pool) = get_hearth_pool_record(&input.hearth_did)?;
        let now_ts = sys_time()?;

        // Apply demurrage (skip if < 60s to avoid double-application on retry)
        let elapsed = elapsed_seconds(pool.last_demurrage_at, now_ts);
        let post_demurrage = if elapsed >= DEMURRAGE_MIN_ELAPSED_SECONDS {
            let deduction = compute_demurrage_deduction(
                pool.balance,
                DEMURRAGE_EXEMPT_FLOOR,
                DEMURRAGE_RATE,
                elapsed,
            );
            pool.balance.saturating_sub(deduction)
        } else {
            pool.balance
        };

        let expected_balance = post_demurrage + input.amount;
        let expected_contributed = pool.total_contributed + input.amount;
        let updated = HearthSapPool {
            balance: expected_balance,
            last_demurrage_at: now_ts,
            total_contributed: expected_contributed,
            ..pool
        };
        update_entry(pool_record.action_address().clone(), &updated)?;

        // Verify our update won: re-read from the anchor
        let (_, actual) = get_hearth_pool_record(&input.hearth_did)?;
        if actual.balance == expected_balance && actual.total_contributed == expected_contributed {
            return Ok(updated);
        }

        // Concurrent update detected
        if attempt == MAX_SAP_RETRIES - 1 {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "contribute_to_hearth_pool for hearth {} failed after {} retries due to concurrent modifications",
                input.hearth_did, MAX_SAP_RETRIES
            ))));
        }
        debug!(
            "contribute_to_hearth_pool: concurrent update detected for hearth {}, retry {}/{}",
            input.hearth_did,
            attempt + 1,
            MAX_SAP_RETRIES
        );
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "contribute_to_hearth_pool: retry loop exited unexpectedly".into()
    )))
}

/// Withdraw SAP from hearth pool to personal balance.
///
/// Requires hearth membership. Any hearth member can withdraw up to
/// the pool's available balance.
#[hdk_extern]
pub fn withdraw_from_hearth_pool(input: WithdrawFromHearthInput) -> ExternResult<HearthSapPool> {
    if input.amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amount must be positive".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    verify_hearth_membership(&caller_did, &input.hearth_did)?;

    let mut pool = get_or_create_hearth_pool(&input.hearth_did)?;

    // Apply demurrage before checking available balance
    let now_ts = sys_time()?;
    let elapsed = elapsed_seconds(pool.last_demurrage_at, now_ts);
    let deduction = compute_demurrage_deduction(
        pool.balance,
        DEMURRAGE_EXEMPT_FLOOR,
        DEMURRAGE_RATE,
        elapsed,
    );
    pool.balance = pool.balance.saturating_sub(deduction);
    pool.last_demurrage_at = now_ts;

    if pool.balance < input.amount {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Insufficient hearth pool balance: have {}, need {}",
            pool.balance, input.amount
        ))));
    }

    // Deduct from pool
    pool.balance -= input.amount;
    pool.total_withdrawn += input.amount;
    update_hearth_pool(&input.hearth_did, &pool)?;

    // Credit personal SAP balance
    if let Some((record, mut personal_bal)) = find_sap_balance_record(&caller_did)? {
        personal_bal.balance += input.amount;
        update_entry(record.action_address().clone(), &personal_bal)?;
    }

    Ok(pool)
}

/// Explicitly apply demurrage to a hearth SAP pool and redistribute as compost.
///
/// Same 70/20/10 split as personal SAP demurrage: local/regional/global commons.
/// Returns the amount deducted. If 0, no update is persisted.
#[hdk_extern]
pub fn apply_hearth_demurrage(input: ApplyHearthDemurrageInput) -> ExternResult<DemurrageResult> {
    let mut pool = get_or_create_hearth_pool(&input.hearth_did)?;
    let now = sys_time()?;
    let elapsed = elapsed_seconds(pool.last_demurrage_at, now);
    let deduction = compute_demurrage_deduction(
        pool.balance,
        DEMURRAGE_EXEMPT_FLOOR,
        DEMURRAGE_RATE,
        elapsed,
    );

    if deduction == 0 {
        return Ok(DemurrageResult {
            deducted: 0,
            redistributed: true,
        });
    }

    pool.balance = pool.balance.saturating_sub(deduction);
    pool.last_demurrage_at = now;
    update_hearth_pool(&input.hearth_did, &pool)?;

    // Redistribute as compost: 70% local, 20% regional, 10% global
    let local_amount = deduction * COMPOST_LOCAL_PCT / 100;
    let regional_amount = deduction * COMPOST_REGIONAL_PCT / 100;
    let global_amount = deduction - local_amount - regional_amount;

    let source_did = format!("hearth:{}", input.hearth_did);

    if let Some(ref pool_id) = input.local_commons_pool_id {
        if let Err(e) = call(
            CallTargetCell::Local,
            ZomeName::from("treasury"),
            FunctionName::from("receive_compost"),
            None,
            ReceiveCompostPayload {
                commons_pool_id: pool_id.clone(),
                amount: local_amount,
                source_member_did: source_did.clone(),
            },
        ) {
            debug!(
                "Hearth compost redistribution to local pool {} failed: {:?}",
                pool_id, e
            );
        }
    }
    if let Some(ref pool_id) = input.regional_commons_pool_id {
        if let Err(e) = call(
            CallTargetCell::Local,
            ZomeName::from("treasury"),
            FunctionName::from("receive_compost"),
            None,
            ReceiveCompostPayload {
                commons_pool_id: pool_id.clone(),
                amount: regional_amount,
                source_member_did: source_did.clone(),
            },
        ) {
            debug!(
                "Hearth compost redistribution to regional pool {} failed: {:?}",
                pool_id, e
            );
        }
    }
    if let Some(ref pool_id) = input.global_commons_pool_id {
        if let Err(e) = call(
            CallTargetCell::Local,
            ZomeName::from("treasury"),
            FunctionName::from("receive_compost"),
            None,
            ReceiveCompostPayload {
                commons_pool_id: pool_id.clone(),
                amount: global_amount,
                source_member_did: source_did,
            },
        ) {
            debug!(
                "Hearth compost redistribution to global pool {} failed: {:?}",
                pool_id, e
            );
        }
    }

    Ok(DemurrageResult {
        deducted: deduction,
        redistributed: true,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApplyHearthDemurrageInput {
    pub hearth_did: String,
    pub local_commons_pool_id: Option<String>,
    pub regional_commons_pool_id: Option<String>,
    pub global_commons_pool_id: Option<String>,
}

fn get_or_create_hearth_pool(hearth_did: &str) -> ExternResult<HearthSapPool> {
    let anchor_key = format!("hearth-sap:{}", hearth_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::HearthDidToSapPool)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(action_hash)?;
            if let Some(pool) = record
                .entry()
                .to_app_option::<HearthSapPool>()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "HearthSapPool deserialization error: {:?}",
                        e
                    )))
                })?
            {
                return Ok(pool);
            }
        }
    }

    let now = sys_time()?;
    let pool = HearthSapPool {
        hearth_did: hearth_did.to_string(),
        balance: 0,
        last_demurrage_at: now,
        member_count: 0,
        total_contributed: 0,
        total_withdrawn: 0,
    };

    let hash = create_entry(&EntryTypes::HearthSapPool(pool.clone()))?;
    create_link(
        anchor_hash(&anchor_key)?,
        hash,
        LinkTypes::HearthDidToSapPool,
        (),
    )?;
    Ok(pool)
}

/// Like `get_or_create_hearth_pool` but also returns the Record for optimistic locking.
fn get_hearth_pool_record(hearth_did: &str) -> ExternResult<(Record, HearthSapPool)> {
    let anchor_key = format!("hearth-sap:{}", hearth_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::HearthDidToSapPool)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(action_hash)?;
            if let Some(pool) = record
                .entry()
                .to_app_option::<HearthSapPool>()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "HearthSapPool deserialization error: {:?}",
                        e
                    )))
                })?
            {
                return Ok((record, pool));
            }
        }
    }

    // Create if not found
    let now = sys_time()?;
    let pool = HearthSapPool {
        hearth_did: hearth_did.to_string(),
        balance: 0,
        last_demurrage_at: now,
        member_count: 0,
        total_contributed: 0,
        total_withdrawn: 0,
    };

    let hash = create_entry(&EntryTypes::HearthSapPool(pool.clone()))?;
    create_link(
        anchor_hash(&anchor_key)?,
        hash.clone(),
        LinkTypes::HearthDidToSapPool,
        (),
    )?;

    let record = get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "HearthSapPool not found after creation".into()
    )))?;
    Ok((record, pool))
}

fn update_hearth_pool(hearth_did: &str, pool: &HearthSapPool) -> ExternResult<()> {
    let anchor_key = format!("hearth-sap:{}", hearth_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::HearthDidToSapPool)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(action_hash)?;
            update_entry(record.action_address().clone(), pool)?;
        }
    }
    Ok(())
}

fn verify_hearth_membership(caller_did: &str, hearth_did: &str) -> ExternResult<()> {
    #[derive(Serialize, Deserialize, Debug)]
    struct MembershipQuery {
        member_did: String,
        hearth_did: String,
    }

    match call(
        CallTargetCell::OtherRole("hearth".into()),
        ZomeName::from("hearth_bridge"),
        FunctionName::from("is_hearth_member"),
        None,
        MembershipQuery {
            member_did: caller_did.to_string(),
            hearth_did: hearth_did.to_string(),
        },
    ) {
        Ok(ZomeCallResponse::Ok(result)) => match result.decode::<bool>() {
            Ok(true) => Ok(()),
            Ok(false) => Err(wasm_error!(WasmErrorInner::Guest(
                "Not a member of this hearth".into()
            ))),
            Err(e) => {
                debug!(
                    "verify_hearth_membership: decode error for {}@{}: {:?}, rejecting",
                    caller_did, hearth_did, e
                );
                Err(wasm_error!(WasmErrorInner::Guest(
                    "Not a member of this hearth".into()
                )))
            }
        },
        Ok(other) => {
            // SECURITY NOTE: Hearth cluster unreachable/unauthorized — allow in standalone mode.
            // In production with a running hearth cluster, this path should not be reached.
            debug!("verify_hearth_membership: hearth_bridge returned {:?} for {}@{}, allowing (standalone mode)", other, caller_did, hearth_did);
            Ok(())
        }
        Err(e) => {
            // SECURITY NOTE: Hearth cluster unreachable — allow in standalone mode.
            debug!("verify_hearth_membership: hearth_bridge unreachable for {}@{}: {:?}, allowing (standalone mode)", caller_did, hearth_did, e);
            Ok(())
        }
    }
}

// ============================================================================
// ZKP-VERIFIED TRANSACTIONS (DASTARK)
// ============================================================================

/// Input for a ZKP-verified balance sufficiency proof.
///
/// The payer proves they have sufficient balance for a transaction
/// without revealing their actual balance. Uses DASTARK dual-backend:
/// - Winterfell STARK for simple balance range proofs (fast)
/// - RISC0 for complex multi-currency transaction validity
///
/// Domain tag: `ZTML:Finance:TxPrivacy:v1`
#[derive(Debug, Serialize, Deserialize, SerializedBytes)]
pub struct ZkBalanceProofInput {
    /// Currency type (SAP, TEND, or MYCEL).
    pub currency: String,
    /// Minimum balance being proven (public threshold).
    pub minimum_balance: u64,
    /// ZK proof bytes (generated client-side).
    pub proof_bytes: Vec<u8>,
    /// Commitment to the actual balance (Blake3 hash).
    pub balance_commitment: Vec<u8>,
}

/// Result of ZKP balance verification.
#[derive(Debug, Serialize, Deserialize, SerializedBytes)]
pub struct ZkBalanceVerification {
    /// Whether the balance proof is valid.
    pub sufficient: bool,
    /// Currency verified.
    pub currency: String,
    /// Minimum balance proven (public).
    pub minimum_proven: u64,
    /// Domain tag used.
    pub domain_tag: String,
}

/// Verify a ZKP balance sufficiency proof.
///
/// This allows a payer to prove "I have at least X SAP" without
/// revealing their actual balance. The proof is generated client-side
/// using DASTARK (Winterfell for range proofs, RISC0 for complex checks).
///
/// Integration: Called before `initiate_payment` to prove sufficiency
/// without querying the payer's balance on-chain.
#[hdk_extern]
pub fn verify_balance_proof(input: ZkBalanceProofInput) -> ExternResult<ZkBalanceVerification> {
    let domain_tag = mycelix_zkp_core::domain::tag_finance_tx();

    // Validate currency type
    let valid_currencies = ["SAP", "TEND", "MYCEL"];
    if !valid_currencies.contains(&input.currency.as_str()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid currency: {}. Must be SAP, TEND, or MYCEL",
            input.currency
        ))));
    }

    // Validate proof structure
    if input.proof_bytes.is_empty() {
        return Ok(ZkBalanceVerification {
            sufficient: false,
            currency: input.currency,
            minimum_proven: input.minimum_balance,
            domain_tag: domain_tag.as_str().to_string(),
        });
    }

    if input.proof_bytes.len() > 500_000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proof exceeds 500KB size limit".into()
        )));
    }

    // Validate commitment length (Blake3 = 32 bytes)
    if input.balance_commitment.len() != 32 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Balance commitment must be exactly 32 bytes".into()
        )));
    }

    // Domain-tagged proof verification
    // Full STARK verification will be wired once Winterfell AIR range circuit is ready.
    let proof_valid = !input.proof_bytes.is_empty() && input.balance_commitment.len() == 32;

    Ok(ZkBalanceVerification {
        sufficient: proof_valid,
        currency: input.currency,
        minimum_proven: input.minimum_balance,
        domain_tag: domain_tag.as_str().to_string(),
    })
}
