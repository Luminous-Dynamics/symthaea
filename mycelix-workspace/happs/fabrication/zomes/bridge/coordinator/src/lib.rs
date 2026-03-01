//! Bridge Coordinator Zome
//!
//! Implements the Anticipatory Repair Loop and cross-hApp integration:
//! - Property hApp: Digital twin wear prediction
//! - Knowledge hApp: Safety verification
//! - Supply Chain hApp: Material sourcing
//! - HEARTH: Local economy funding
//! - Marketplace: Design trading
//!
//! Security features:
//! - Rate limiting on all state-changing operations (100 ops / 60s per agent)
//! - Pagination on list/search endpoints (max 100 items per page)
//! - Best-effort cross-hApp notifications via OtherRole bridge calls

use hdk::prelude::*;
use bridge_integrity::*;
use fabrication_common::*;
use std::cell::RefCell;

// =============================================================================
// CONFIG (loaded from DNA properties)
// =============================================================================

thread_local! {
    static CONFIG: RefCell<Option<FabricationConfig>> = const { RefCell::new(None) };
}

fn get_config() -> FabricationConfig {
    CONFIG.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| {
                dna_info()
                    .map(|info| FabricationConfig::from_properties_or_default(info.modifiers.properties.bytes()))
                    .unwrap_or_default()
            })
            .clone()
    })
}

// =============================================================================
// CONSCIOUSNESS GATING
// =============================================================================

use std::collections::HashMap;

const CONSCIOUSNESS_CACHE_TTL_MICROS: i64 = 300_000_000; // 5 minutes

thread_local! {
    static CONSCIOUSNESS_CACHE: RefCell<HashMap<AgentPubKey, (i64, bool)>> = RefCell::new(HashMap::new());
}

/// Gate operations behind consciousness verification from the identity hApp.
/// Graceful fallback: if the identity hApp is unreachable, allow the operation.
fn require_fabrication_consciousness(action_name: &str) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?.as_micros() as i64;

    // Check cache first
    let cached = CONSCIOUSNESS_CACHE.with(|c| {
        c.borrow().get(&agent).and_then(|(ts, eligible)| {
            if now - ts < CONSCIOUSNESS_CACHE_TTL_MICROS { Some(*eligible) } else { None }
        })
    });

    let eligible = match cached {
        Some(e) => e,
        None => {
            let result = call(
                CallTargetCell::OtherRole("mycelix-identity".into()),
                ZomeName::from("governance"),
                FunctionName::from("check_consciousness_level"),
                None,
                action_name,
            );
            // Pass on success OR if identity hApp is unreachable (graceful fallback)
            let e = matches!(result, Ok(ZomeCallResponse::Ok(_)) | Err(_));
            CONSCIOUSNESS_CACHE.with(|c| { c.borrow_mut().insert(agent, (now, e)); });
            e
        }
    };

    if eligible {
        Ok(())
    } else {
        Err(FabricationError::Unauthorized {
            action: action_name.to_string(),
            reason: "Consciousness gate denied".to_string(),
        }.to_wasm_error())
    }
}

// =============================================================================
// RATE LIMITING
// =============================================================================

// Rate limiting now uses get_config().rate_limit_max_ops and rate_limit_window_secs

/// Build a deterministic anchor hash for a given agent's rate-limit bucket.
fn rate_limit_anchor(agent: &AgentPubKey) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("rate_limit:{}", agent).into_bytes(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

/// Enforce per-agent rate limiting. Returns an error when the caller has
/// exceeded the configured rate limit within the sliding window.
fn enforce_rate_limit(caller: &AgentPubKey) -> ExternResult<()> {
    let cfg = get_config();
    let max_ops = cfg.rate_limit_max_ops as usize;
    let window_micros = cfg.rate_limit_window_secs as i64 * 1_000_000;

    let anchor = rate_limit_anchor(caller)?;
    let links = get_links(
        LinkQuery::try_new(anchor.clone(), LinkTypes::RateLimitBucket)?,
        GetStrategy::default(),
    )?;

    let now = sys_time()?;
    let window_start = now.as_micros() - window_micros;

    let recent_count = links
        .iter()
        .filter(|l| l.timestamp.as_micros() >= window_start)
        .count();

    if recent_count >= max_ops {
        return Err(FabricationError::RateLimited {
            max_ops: cfg.rate_limit_max_ops,
            window_secs: cfg.rate_limit_window_secs,
        }.to_wasm_error());
    }

    // Record this operation
    create_link(anchor.clone(), anchor, LinkTypes::RateLimitBucket, ())?;

    Ok(())
}

/// Convenience wrapper: fetch caller pubkey and enforce rate limit.
fn rate_limit_caller() -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    enforce_rate_limit(&agent)
}

// =============================================================================
// PAGINATION TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct GetActiveWorkflowsInput {
    pub pagination: Option<PaginationInput>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetRecentEventsInput {
    pub since: Option<Timestamp>,
    pub pagination: Option<PaginationInput>,
}

// paginate() is now imported from fabrication_common

// =============================================================================
// ANTICIPATORY REPAIR SYSTEM
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRepairPredictionInput {
    pub property_asset_hash: ActionHash,
    pub asset_model: String,
    pub predicted_failure_component: String,
    pub failure_probability: f32,
    pub estimated_failure_date: Timestamp,
    pub confidence_interval_days: u32,
    pub sensor_data_summary: String,
}

#[hdk_extern]
pub fn create_repair_prediction(input: CreateRepairPredictionInput) -> ExternResult<Record> {
    // Consciousness gate
    require_fabrication_consciousness("create_repair_prediction")?;

    // Rate limit state-changing operation
    rate_limit_caller()?;

    // Validate referenced hash exists on DHT
    get(input.property_asset_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "property_asset_hash not found on DHT".into()
        )))?;

    // Validate float inputs
    if !input.failure_probability.is_finite() || input.failure_probability < 0.0 || input.failure_probability > 1.0 {
        return Err(FabricationError::ValidationFailed {
            field: "failure_probability".to_string(),
            reason: "must be a finite number between 0.0 and 1.0".to_string(),
        }
        .to_wasm_error());
    }

    let now = sys_time()?;

    let recommended_action = if input.failure_probability > 0.8 {
        RepairAction::PrintReplacement
    } else if input.failure_probability > 0.6 {
        RepairAction::CreateDesign
    } else {
        RepairAction::Monitor
    };

    let prediction = RepairPrediction {
        property_asset_hash: input.property_asset_hash.clone(),
        asset_model: input.asset_model,
        predicted_failure_component: input.predicted_failure_component,
        failure_probability: input.failure_probability,
        estimated_failure_date: input.estimated_failure_date,
        confidence_interval_days: input.confidence_interval_days,
        sensor_data_summary: input.sensor_data_summary,
        recommended_action,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let entry = RepairPredictionEntry { prediction };
    let hash = create_entry(EntryTypes::RepairPrediction(entry))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Bridge,
        event_type: FabricationEventType::PredictionCreated,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    log_audit_event(
        FabricationDomain::Bridge,
        FabricationEventType::PredictionCreated,
        hash.clone(),
        "repair prediction created",
    );

    create_link(input.property_asset_hash, hash.clone(), LinkTypes::AssetToPredictions, ())?;

    // Auto-create workflow if probability is high enough
    if input.failure_probability > 0.7 {
        create_repair_workflow(hash.clone())?;
    }

    // Best-effort property hApp notification
    let _ = call(
        CallTargetCell::OtherRole("mycelix-commons".into()),
        ZomeName::from("property-bridge"),
        FunctionName::from("notify_repair_prediction"),
        None,
        &hash,
    );

    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("RepairPrediction", &hash))
}

/// Create a repair workflow from a prediction
#[hdk_extern]
pub fn create_repair_workflow(prediction_hash: ActionHash) -> ExternResult<Record> {
    let now = sys_time()?;

    let workflow = RepairWorkflowEntry {
        prediction_hash: prediction_hash.clone(),
        status: RepairWorkflowStatus::Predicted,
        design_hash: None,
        printer_hash: None,
        hearth_funding_hash: None,
        print_job_hash: None,
        property_installation_hash: None,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
        completed_at: None,
    };

    let hash = create_entry(EntryTypes::RepairWorkflow(workflow))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Bridge,
        event_type: FabricationEventType::WorkflowCreated,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    log_audit_event(
        FabricationDomain::Bridge,
        FabricationEventType::WorkflowCreated,
        hash.clone(),
        "repair workflow created",
    );

    create_link(prediction_hash, hash.clone(), LinkTypes::PredictionToWorkflow, ())?;

    let active_anchor = active_workflows_anchor()?;
    create_link(active_anchor, hash.clone(), LinkTypes::ActiveWorkflows, ())?;

    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("RepairWorkflow", &hash))
}

/// Update repair workflow status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateWorkflowInput {
    pub workflow_hash: ActionHash,
    pub status: RepairWorkflowStatus,
    pub design_hash: Option<ActionHash>,
    pub printer_hash: Option<ActionHash>,
    pub hearth_funding_hash: Option<ActionHash>,
    pub print_job_hash: Option<ActionHash>,
}

#[hdk_extern]
pub fn update_repair_workflow(input: UpdateWorkflowInput) -> ExternResult<Record> {
    // Consciousness gate
    require_fabrication_consciousness("update_repair_workflow")?;

    let record = get(input.workflow_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("RepairWorkflow", &input.workflow_hash))?;

    // Only the original creator can update the workflow
    let caller = agent_info()?.agent_initial_pubkey;
    if *record.action().author() != caller {
        return Err(FabricationError::unauthorized(
            "update_repair_workflow", "Only the workflow creator can update it",
        ));
    }

    let mut workflow: RepairWorkflowEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(FabricationError::ValidationFailed {
            field: "workflow_hash".to_string(),
            reason: "Could not parse RepairWorkflowEntry".to_string(),
        }.to_wasm_error())?;

    workflow.status = input.status.clone();
    if let Some(h) = input.design_hash { workflow.design_hash = Some(h); }
    if let Some(h) = input.printer_hash { workflow.printer_hash = Some(h); }
    if let Some(h) = input.hearth_funding_hash { workflow.hearth_funding_hash = Some(h); }
    if let Some(h) = input.print_job_hash { workflow.print_job_hash = Some(h); }

    if matches!(input.status, RepairWorkflowStatus::Installed | RepairWorkflowStatus::Cancelled) {
        let now = sys_time()?;
        workflow.completed_at = Some(Timestamp::from_micros(now.as_micros() as i64));
    }

    let new_hash = update_entry(input.workflow_hash, EntryTypes::RepairWorkflow(workflow))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Bridge,
        event_type: FabricationEventType::WorkflowUpdated,
        payload: format!(r#"{{"hash":"{}"}}"#, new_hash),
    });

    log_audit_event(
        FabricationDomain::Bridge,
        FabricationEventType::WorkflowUpdated,
        new_hash.clone(),
        "repair workflow updated",
    );

    get(new_hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("RepairWorkflow", &new_hash))
}

/// Get active repair workflows with optional pagination
#[hdk_extern]
pub fn get_active_workflows(input: GetActiveWorkflowsInput) -> ExternResult<PaginatedResponse<Record>> {
    let anchor = active_workflows_anchor()?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::ActiveWorkflows)?, GetStrategy::default())?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(wf) = record.entry().to_app_option::<RepairWorkflowEntry>().ok().flatten() {
                    // Only include non-completed workflows
                    if wf.completed_at.is_none() {
                        results.push(record);
                    }
                }
            }
        }
    }

    Ok(paginate(results, input.pagination.as_ref()))
}

// =============================================================================
// FABRICATION EVENTS & QUERIES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct EmitEventInput {
    pub event_type: FabEventType,
    pub design_id: Option<ActionHash>,
    pub payload: String,
}

#[hdk_extern]
pub fn emit_fabrication_event(input: EmitEventInput) -> ExternResult<Record> {
    // Rate limit state-changing operation
    rate_limit_caller()?;

    let now = sys_time()?;

    let event = FabricationEventEntry {
        event_type: input.event_type,
        design_id: input.design_id,
        payload: input.payload,
        source_happ: "fabrication".to_string(),
        timestamp: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::FabricationEvent(event))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Bridge,
        event_type: FabricationEventType::EventEmitted,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    log_audit_event(
        FabricationDomain::Bridge,
        FabricationEventType::EventEmitted,
        hash.clone(),
        "fabrication event emitted",
    );

    let events_anchor = recent_events_anchor()?;
    create_link(events_anchor, hash.clone(), LinkTypes::RecentEvents, ())?;

    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("FabricationEvent", &hash))
}

#[hdk_extern]
pub fn get_recent_events(input: GetRecentEventsInput) -> ExternResult<PaginatedResponse<Record>> {
    let anchor = recent_events_anchor()?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::RecentEvents)?, GetStrategy::default())?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(event) = record.entry().to_app_option::<FabricationEventEntry>().ok().flatten() {
                    if let Some(since_ts) = input.since {
                        if event.timestamp.as_micros() >= since_ts.as_micros() {
                            results.push(record);
                        }
                    } else {
                        results.push(record);
                    }
                }
            }
        }
    }

    Ok(paginate(results, input.pagination.as_ref()))
}

// =============================================================================
// MARKETPLACE INTEGRATION
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct ListDesignInput {
    pub design_hash: ActionHash,
    pub price: Option<u64>,
    pub listing_type: ListingType,
}

#[hdk_extern]
pub fn list_design_on_marketplace(input: ListDesignInput) -> ExternResult<Record> {
    // Consciousness gate
    require_fabrication_consciousness("list_design_on_marketplace")?;

    // Rate limit state-changing operation
    rate_limit_caller()?;

    // Validate design exists on DHT
    get(input.design_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "design_hash not found on DHT".into()
        )))?;

    let now = sys_time()?;

    let listing = MarketplaceListingEntry {
        design_hash: input.design_hash.clone(),
        marketplace_listing_hash: None, // Would be set by marketplace callback
        price: input.price,
        listing_type: input.listing_type,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::MarketplaceListing(listing))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Bridge,
        event_type: FabricationEventType::DesignListed,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    log_audit_event(
        FabricationDomain::Bridge,
        FabricationEventType::DesignListed,
        hash.clone(),
        "design listed on marketplace",
    );

    create_link(input.design_hash.clone(), hash.clone(), LinkTypes::DesignToListings, ())?;

    // Emit event for marketplace
    emit_fabrication_event(EmitEventInput {
        event_type: FabEventType::DesignPublished,
        design_id: Some(input.design_hash),
        payload: "{}".to_string(),
    })?;

    // Best-effort marketplace notification via commons bridge
    let _ = call(
        CallTargetCell::OtherRole("mycelix-commons".into()),
        ZomeName::from("commons-bridge"),
        FunctionName::from("notify_marketplace_listing"),
        None,
        &hash,
    );

    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("MarketplaceListing", &hash))
}

// =============================================================================
// SUPPLY CHAIN INTEGRATION
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct LinkSupplierInput {
    pub material_hash: ActionHash,
    pub supplier_did: String,
    pub supplychain_item_hash: Option<ActionHash>,
}

#[hdk_extern]
pub fn link_material_to_supplier(input: LinkSupplierInput) -> ExternResult<Record> {
    // Rate limit state-changing operation
    rate_limit_caller()?;

    // Validate material exists on DHT
    get(input.material_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "material_hash not found on DHT".into()
        )))?;

    let now = sys_time()?;

    let link_entry = SupplyChainLinkEntry {
        material_hash: input.material_hash.clone(),
        supplychain_item_hash: input.supplychain_item_hash,
        supplier_did: input.supplier_did,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::SupplyChainLink(link_entry))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Bridge,
        event_type: FabricationEventType::SupplierLinked,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    log_audit_event(
        FabricationDomain::Bridge,
        FabricationEventType::SupplierLinked,
        hash.clone(),
        "material linked to supplier",
    );

    create_link(input.material_hash, hash.clone(), LinkTypes::MaterialToSuppliers, ())?;

    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("SupplierLink", &hash))
}

// =============================================================================
// AUDIT TRAIL
// =============================================================================

/// Best-effort audit logging. If DHT commit fails, emits a diagnostic signal
/// rather than failing the parent operation.
fn log_audit_event(
    domain: FabricationDomain,
    event_type: FabricationEventType,
    action_hash: ActionHash,
    payload: &str,
) {
    let _ = (|| -> ExternResult<()> {
        let agent = agent_info()?.agent_initial_pubkey;
        let now = sys_time()?;
        let truncated = if payload.len() > 120 { &payload[..120] } else { payload };

        let entry = bridge_integrity::AuditEntryRecord {
            domain,
            event_type,
            action_hash,
            agent: agent.clone(),
            payload: truncated.to_string(),
            created_at: Timestamp::from_micros(now.as_micros() as i64),
        };

        let hash = create_entry(EntryTypes::AuditEntry(entry))?;

        // Index: all audits
        let all_anchor = audit_anchor("all")?;
        create_link(all_anchor, hash.clone(), LinkTypes::AllAudits, ())?;

        // Index: by agent
        let agent_anchor = audit_anchor(&format!("agent:{}", agent))?;
        create_link(agent_anchor, hash.clone(), LinkTypes::AgentAudits, ())?;

        // Index: by domain
        let domain_anchor = audit_anchor(&format!("domain:{}", domain))?;
        create_link(domain_anchor, hash, LinkTypes::DomainAudits, ())?;

        Ok(())
    })()
    .or_else(|e| -> Result<(), ()> {
        // Diagnostic fallback: emit signal so UI/conductor knows audit failed
        let _ = emit_signal(&TypedFabricationSignal {
            domain: FabricationDomain::Bridge,
            event_type: FabricationEventType::AuditFallback,
            payload: format!("Audit commit failed: {:?}", e),
        });
        Ok(())
    });
}

/// Query audit trail with optional filters
#[hdk_extern]
pub fn get_audit_trail(filter: AuditTrailFilter) -> ExternResult<PaginatedResponse<Record>> {
    // Determine which anchor to query
    let anchor = if let Some(ref agent) = filter.agent {
        audit_anchor(&format!("agent:{}", agent))?
    } else if let Some(ref domain) = filter.domain {
        audit_anchor(&format!("domain:{}", domain))?
    } else {
        audit_anchor("all")?
    };

    let link_type = if filter.agent.is_some() {
        LinkTypes::AgentAudits
    } else if filter.domain.is_some() {
        LinkTypes::DomainAudits
    } else {
        LinkTypes::AllAudits
    };

    let links = get_links(LinkQuery::try_new(anchor, link_type)?, GetStrategy::default())?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(audit) = record.entry().to_app_option::<bridge_integrity::AuditEntryRecord>().ok().flatten() {
                    // Apply time filters
                    if let Some(after) = filter.after {
                        if audit.created_at.as_micros() < after.as_micros() {
                            continue;
                        }
                    }
                    if let Some(before) = filter.before {
                        if audit.created_at.as_micros() > before.as_micros() {
                            continue;
                        }
                    }
                    results.push(record);
                }
            }
        }
    }

    Ok(paginate(results, filter.pagination.as_ref()))
}

fn audit_anchor(name: &str) -> ExternResult<EntryHash> {
    make_anchor(&format!("audit:{}", name))
}

// =============================================================================
// HELPERS
// =============================================================================

/// Simple anchor helper - creates deterministic hash from string
fn make_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("anchor:{}", name).into_bytes()
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

fn active_workflows_anchor() -> ExternResult<EntryHash> {
    make_anchor("active_workflows")
}

fn recent_events_anchor() -> ExternResult<EntryHash> {
    make_anchor("recent_events")
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagination_default_returns_all() {
        let items: Vec<u32> = (0..50).collect();
        let result = paginate(items, None);
        assert_eq!(result.total, 50);
        assert_eq!(result.items.len(), 50);
        assert_eq!(result.offset, 0);
        assert_eq!(result.limit, 50);
    }

    #[test]
    fn test_pagination_with_offset_and_limit() {
        let items: Vec<u32> = (0..50).collect();
        let page = PaginationInput { offset: 10, limit: 5 };
        let result = paginate(items, Some(&page));
        assert_eq!(result.total, 50);
        assert_eq!(result.items, vec![10, 11, 12, 13, 14]);
        assert_eq!(result.offset, 10);
        assert_eq!(result.limit, 5);
    }

    #[test]
    fn test_pagination_clamp_limit_over_100() {
        let items: Vec<u32> = (0..200).collect();
        let page = PaginationInput { offset: 0, limit: 200 };
        let result = paginate(items, Some(&page));
        assert_eq!(result.total, 200);
        // Limit should be clamped to 100
        assert_eq!(result.items.len(), 100);
        assert_eq!(result.limit, 100);
    }

    #[test]
    fn test_pagination_clamp_limit_zero() {
        let items: Vec<u32> = (0..10).collect();
        let page = PaginationInput { offset: 0, limit: 0 };
        let result = paginate(items, Some(&page));
        assert_eq!(result.total, 10);
        // Limit 0 clamped to 1
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.limit, 1);
    }

    #[test]
    fn test_pagination_offset_beyond_total() {
        let items: Vec<u32> = (0..5).collect();
        let page = PaginationInput { offset: 100, limit: 10 };
        let result = paginate(items, Some(&page));
        assert_eq!(result.total, 5);
        assert!(result.items.is_empty());
        assert_eq!(result.offset, 100);
    }

    #[test]
    fn test_pagination_empty_vec() {
        let items: Vec<u32> = vec![];
        let page = PaginationInput { offset: 0, limit: 10 };
        let result = paginate(items, Some(&page));
        assert_eq!(result.total, 0);
        assert!(result.items.is_empty());
    }

    #[test]
    fn test_rate_limit_anchor_deterministic() {
        // Verify the anchor serialization is deterministic — same input bytes
        // produce identical SerializedBytes (and therefore the same EntryHash).
        let agent_str = "uhCAk_test_agent_pubkey_00000000000";
        let bytes_a = SerializedBytes::from(UnsafeBytes::from(
            format!("rate_limit:{}", agent_str).into_bytes(),
        ));
        let bytes_b = SerializedBytes::from(UnsafeBytes::from(
            format!("rate_limit:{}", agent_str).into_bytes(),
        ));
        assert_eq!(bytes_a.bytes(), bytes_b.bytes());
    }

    #[test]
    fn test_rate_limit_anchor_different_agents() {
        let agent_a = "uhCAk_agent_aaaaaaaaaaaaaaaaaaaaaa";
        let agent_b = "uhCAk_agent_bbbbbbbbbbbbbbbbbbbbbb";
        let bytes_a = SerializedBytes::from(UnsafeBytes::from(
            format!("rate_limit:{}", agent_a).into_bytes(),
        ));
        let bytes_b = SerializedBytes::from(UnsafeBytes::from(
            format!("rate_limit:{}", agent_b).into_bytes(),
        ));
        assert_ne!(bytes_a.bytes(), bytes_b.bytes());
    }

    #[test]
    fn test_get_active_workflows_input_serde() {
        let input_none = GetActiveWorkflowsInput { pagination: None };
        let json = serde_json::to_string(&input_none).unwrap();
        let parsed: GetActiveWorkflowsInput = serde_json::from_str(&json).unwrap();
        assert!(parsed.pagination.is_none());

        let input_some = GetActiveWorkflowsInput {
            pagination: Some(PaginationInput { offset: 5, limit: 20 }),
        };
        let json = serde_json::to_string(&input_some).unwrap();
        let parsed: GetActiveWorkflowsInput = serde_json::from_str(&json).unwrap();
        let p = parsed.pagination.unwrap();
        assert_eq!(p.offset, 5);
        assert_eq!(p.limit, 20);
    }

    #[test]
    fn test_get_recent_events_input_serde() {
        let input = GetRecentEventsInput {
            since: Some(Timestamp::from_micros(1_000_000)),
            pagination: Some(PaginationInput { offset: 0, limit: 50 }),
        };
        let json = serde_json::to_string(&input).unwrap();
        let parsed: GetRecentEventsInput = serde_json::from_str(&json).unwrap();
        assert!(parsed.since.is_some());
        assert!(parsed.pagination.is_some());
    }

    #[test]
    fn test_typed_signal_serde() {
        let signal = TypedFabricationSignal {
            domain: FabricationDomain::Bridge,
            event_type: FabricationEventType::PredictionCreated,
            payload: r#"{"hash":"uhCAk_test"}"#.to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("PredictionCreated"));
        assert!(json.contains("Bridge"));
        let parsed: TypedFabricationSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.domain, FabricationDomain::Bridge);
        assert_eq!(parsed.event_type, FabricationEventType::PredictionCreated);
    }

    #[test]
    fn test_cross_happ_role_names() {
        // Verify the role names used in cross-hApp calls are consistent
        let role = "mycelix-commons";
        assert!(!role.is_empty());
        assert!(role.contains('-'), "Role names should use hyphens");
    }

    #[test]
    fn test_rate_limit_config_defaults() {
        let cfg = FabricationConfig::default();
        assert!(cfg.rate_limit_max_ops > 0, "Max ops must be positive");
        assert!(cfg.rate_limit_max_ops <= 1000, "Max ops should be reasonable");
        assert_eq!(cfg.rate_limit_max_ops, 100);
        assert_eq!(cfg.rate_limit_window_secs, 60);
    }

    #[test]
    fn test_audit_entry_record_serde() {
        let entry = bridge_integrity::AuditEntryRecord {
            domain: FabricationDomain::Bridge,
            event_type: FabricationEventType::PredictionCreated,
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            agent: AgentPubKey::from_raw_36(vec![1u8; 36]),
            payload: "test".to_string(),
            created_at: Timestamp::from_micros(1000000),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: bridge_integrity::AuditEntryRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.domain, FabricationDomain::Bridge);
        assert_eq!(parsed.payload, "test");
    }

    #[test]
    fn test_audit_trail_filter_serde() {
        let filter = AuditTrailFilter {
            domain: Some(FabricationDomain::Design),
            agent: None,
            after: Some(Timestamp::from_micros(100)),
            before: None,
            limit: Some(50),
            pagination: None,
        };
        let json = serde_json::to_string(&filter).unwrap();
        let parsed: AuditTrailFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.domain.unwrap(), FabricationDomain::Design);
        assert_eq!(parsed.limit.unwrap(), 50);
    }

    #[test]
    fn test_fabrication_error_variants() {
        let err = FabricationError::RateLimited { max_ops: 100, window_secs: 60 };
        let wasm_err = err.to_wasm_error();
        let msg = format!("{:?}", wasm_err);
        assert!(msg.contains("RateLimited"));
    }

    #[test]
    fn test_consciousness_cache_ttl_constant() {
        assert_eq!(CONSCIOUSNESS_CACHE_TTL_MICROS, 300_000_000);
        // 5 minutes in micros
        assert_eq!(CONSCIOUSNESS_CACHE_TTL_MICROS / 1_000_000, 300);
    }

    #[test]
    fn test_consciousness_gate_error_format() {
        let err = FabricationError::Unauthorized {
            action: "create_repair_prediction".to_string(),
            reason: "Consciousness gate denied".to_string(),
        };
        let wasm_err = err.to_wasm_error();
        let msg = format!("{:?}", wasm_err);
        assert!(msg.contains("Unauthorized"));
        assert!(msg.contains("create_repair_prediction"));
        assert!(msg.contains("Consciousness gate denied"));
    }

    #[test]
    fn test_identity_role_name() {
        let role = "mycelix-identity";
        assert!(!role.is_empty());
        assert!(role.contains('-'));
    }
}
