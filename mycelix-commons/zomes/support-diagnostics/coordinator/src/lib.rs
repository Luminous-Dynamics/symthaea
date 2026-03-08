//! Support Diagnostics Coordinator Zome
//! Business logic for diagnostic results, privacy preferences, and cognitive
//! updates in the Mycelix support domain.

use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};
use support_diagnostics_integrity::*;
use support_types::*;

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

// ============================================================================
// SIGNAL
// ============================================================================

/// Signal emitted when a cognitive update is published, allowing the
/// Symthaea bridge to absorb new resolution patterns.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    pub event_type: String,
    pub payload_hash: ActionHash,
    pub category: SupportCategory,
    pub phi: f64,
}

// ============================================================================
// HELPERS
// ============================================================================

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// DIAGNOSTIC RESULTS
// ============================================================================

/// Run a diagnostic and store the result, linking it to the time-sharded
/// index, the agent, and optionally the originating ticket.
#[hdk_extern]
pub fn run_diagnostic(diag: DiagnosticResult) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "run_diagnostic")?;
    let action_hash = create_entry(&EntryTypes::DiagnosticResult(diag.clone()))?;

    // Time-sharded diagnostics link
    let shard = sharded_anchor("support", "diagnostics", &diag.created_at);
    create_entry(&EntryTypes::Anchor(Anchor(shard.clone())))?;
    create_link(
        anchor_hash(&shard)?,
        action_hash.clone(),
        LinkTypes::ShardedDiagnostics,
        (),
    )?;

    // All diagnostics anchor (for list_diagnostics)
    create_entry(&EntryTypes::Anchor(Anchor("all_diagnostics".to_string())))?;
    create_link(
        anchor_hash("all_diagnostics")?,
        action_hash.clone(),
        LinkTypes::ShardedDiagnostics,
        (),
    )?;

    // Agent to diagnostic link
    create_link(
        diag.agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToDiagnostic,
        (),
    )?;

    // Optional ticket to diagnostic link
    if let Some(ref ticket_hash) = diag.ticket_hash {
        create_link(
            ticket_hash.clone(),
            action_hash.clone(),
            LinkTypes::TicketToDiagnostic,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created diagnostic result".into()
    )))
}

/// Get a single diagnostic result by its action hash.
#[hdk_extern]
pub fn get_diagnostic(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// List all diagnostic results from the "all_diagnostics" anchor.
#[hdk_extern]
pub fn list_diagnostics(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("all_diagnostics")?,
            LinkTypes::ShardedDiagnostics,
        )?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// PRIVACY PREFERENCES
// ============================================================================

/// Set (or update) the calling agent's privacy preferences.
#[hdk_extern]
pub fn set_privacy_preference(pref: PrivacyPreference) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "set_privacy_preference")?;
    let action_hash = create_entry(&EntryTypes::PrivacyPreference(pref.clone()))?;

    create_link(
        pref.agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToPrivacyPreference,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created privacy preference".into()
    )))
}

/// Get the most recent privacy preference for an agent.
#[hdk_extern]
pub fn get_privacy_preference(agent: AgentPubKey) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToPrivacyPreference)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Return the latest link (highest timestamp)
    let mut sorted = links;
    sorted.sort_by_key(|l| l.timestamp);
    let latest = sorted.last().unwrap();

    let action_hash = ActionHash::try_from(latest.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    get(action_hash, GetOptions::default())
}

// ============================================================================
// SHAREABLE DIAGNOSTICS (filtered by privacy preference)
// ============================================================================

/// Get an agent's diagnostics filtered by their privacy preference sharing tier.
/// Returns all diagnostics if sharing tier is Full, scrubbed-only if Anonymized,
/// and empty if LocalOnly.
#[hdk_extern]
pub fn get_shareable_diagnostics(agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    // First get the agent's privacy preference
    let pref_record = get_privacy_preference(agent.clone())?;
    let sharing_tier = if let Some(record) = pref_record {
        let pref: PrivacyPreference = record
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid privacy preference entry".into()
            )))?;
        pref.sharing_tier
    } else {
        // Default to LocalOnly if no preference set
        SharingTier::LocalOnly
    };

    match sharing_tier {
        SharingTier::LocalOnly => Ok(vec![]),
        SharingTier::Anonymized => {
            // Return only scrubbed diagnostics
            let links = get_links(
                LinkQuery::try_new(agent, LinkTypes::AgentToDiagnostic)?,
                GetStrategy::default(),
            )?;
            let all_records = records_from_links(links)?;
            let mut scrubbed = Vec::new();
            for record in all_records {
                if let Some(diag) = record
                    .entry()
                    .to_app_option::<DiagnosticResult>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                {
                    if diag.scrubbed {
                        scrubbed.push(record);
                    }
                }
            }
            Ok(scrubbed)
        }
        SharingTier::Full => {
            let links = get_links(
                LinkQuery::try_new(agent, LinkTypes::AgentToDiagnostic)?,
                GetStrategy::default(),
            )?;
            records_from_links(links)
        }
    }
}

// ============================================================================
// HELPER PROFILES
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateAvailInput {
    pub helper_hash: ActionHash,
    pub available: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HelperWorkload {
    pub agent: AgentPubKey,
    pub active_tickets: u32,
    pub max_concurrent: u32,
}

/// Register a new helper profile, linking it to the "all_helpers" anchor.
#[hdk_extern]
pub fn register_helper(profile: HelperProfile) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "register_helper")?;
    let action_hash = create_entry(&EntryTypes::HelperProfile(profile))?;
    create_entry(&EntryTypes::Anchor(Anchor("all_helpers".to_string())))?;
    create_link(
        anchor_hash("all_helpers")?,
        action_hash.clone(),
        LinkTypes::AllHelpers,
        (),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created helper profile".into()
    )))
}

/// Update a helper's availability status.
#[hdk_extern]
pub fn update_availability(input: UpdateAvailInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "update_availability")?;
    let record = get(input.helper_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Helper profile not found".into())
    ))?;
    let mut profile: HelperProfile = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Not a HelperProfile".into()
        )))?;
    profile.available = input.available;
    let new_hash = update_entry(input.helper_hash, &EntryTypes::HelperProfile(profile))?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated helper profile".into()
    )))
}

/// Get all available helpers, optionally filtered by support category.
#[hdk_extern]
pub fn get_available_helpers(category: Option<SupportCategory>) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_helpers")?, LinkTypes::AllHelpers)?,
        GetStrategy::default(),
    )?;
    let all_records = records_from_links(links)?;
    let mut available = Vec::new();
    for record in all_records {
        if let Some(profile) = record
            .entry()
            .to_app_option::<HelperProfile>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if profile.available {
                if let Some(ref cat) = category {
                    if profile.expertise_categories.contains(cat) {
                        available.push(record);
                    }
                } else {
                    available.push(record);
                }
            }
        }
    }
    Ok(available)
}

// ============================================================================
// COGNITIVE UPDATES
// ============================================================================

/// Publish a cognitive update (resolution pattern encoded as BinaryHV).
/// Creates links to the category anchor and the all-updates anchor,
/// then emits a BridgeEventSignal for the Symthaea bridge.
#[hdk_extern]
pub fn publish_cognitive_update(update: CognitiveUpdate) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "publish_cognitive_update")?;
    let action_hash = create_entry(&EntryTypes::CognitiveUpdate(update.clone()))?;

    // Category-specific time-sharded link
    let cat_label = format!("{:?}", update.category).to_lowercase();
    let cat_anchor = sharded_anchor(
        "support",
        &format!("cognitive_{}", cat_label),
        &update.created_at,
    );
    create_entry(&EntryTypes::Anchor(Anchor(cat_anchor.clone())))?;
    create_link(
        anchor_hash(&cat_anchor)?,
        action_hash.clone(),
        LinkTypes::CategoryToCognitiveUpdate,
        (),
    )?;

    // Static category anchor (for queries without time range)
    let static_cat_anchor = format!("cognitive_updates:{}", cat_label);
    create_entry(&EntryTypes::Anchor(Anchor(static_cat_anchor.clone())))?;
    create_link(
        anchor_hash(&static_cat_anchor)?,
        action_hash.clone(),
        LinkTypes::CategoryToCognitiveUpdate,
        (),
    )?;

    // All cognitive updates link
    let all_anchor = sharded_anchor("support", "cognitive_all", &update.created_at);
    create_entry(&EntryTypes::Anchor(Anchor(all_anchor.clone())))?;
    create_link(
        anchor_hash(&all_anchor)?,
        action_hash.clone(),
        LinkTypes::AllCognitiveUpdates,
        (),
    )?;

    // Emit bridge event signal
    let signal = BridgeEventSignal {
        event_type: "cognitive_update_published".to_string(),
        payload_hash: action_hash.clone(),
        category: update.category.clone(),
        phi: update.phi,
    };
    emit_signal(&signal)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created cognitive update".into()
    )))
}

/// Get all cognitive updates for a given support category.
#[hdk_extern]
pub fn get_cognitive_updates_by_category(category: SupportCategory) -> ExternResult<Vec<Record>> {
    let cat_label = format!("{:?}", category).to_lowercase();
    let static_cat_anchor = format!("cognitive_updates:{}", cat_label);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&static_cat_anchor)?,
            LinkTypes::CategoryToCognitiveUpdate,
        )?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Absorb a cognitive update by retrieving its record. Actual absorption
/// (injecting the BinaryHV into the local Symthaea instance) happens in the
/// symthaea-support sub-crate, not in the zome.
#[hdk_extern]
pub fn absorb_cognitive_update(hash: ActionHash) -> ExternResult<Record> {
    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Cognitive update not found".into()
    )))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xdb; 36])
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xab; 36])
    }

    // ========================================================================
    // BridgeEventSignal serde roundtrip
    // ========================================================================

    #[test]
    fn bridge_event_signal_serde_roundtrip() {
        let signal = BridgeEventSignal {
            event_type: "cognitive_update_published".to_string(),
            payload_hash: fake_action_hash(),
            category: SupportCategory::Network,
            phi: 0.42,
        };
        let json = serde_json::to_string(&signal).unwrap();
        let back: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type, "cognitive_update_published");
        assert_eq!(back.category, SupportCategory::Network);
        assert_eq!(back.phi, 0.42);
    }

    #[test]
    fn bridge_event_signal_all_categories() {
        for category in [
            SupportCategory::Network,
            SupportCategory::Hardware,
            SupportCategory::Software,
            SupportCategory::Holochain,
            SupportCategory::Mycelix,
            SupportCategory::Security,
            SupportCategory::General,
        ] {
            let signal = BridgeEventSignal {
                event_type: "cognitive_update_published".to_string(),
                payload_hash: fake_action_hash(),
                category: category.clone(),
                phi: 1.0,
            };
            let json = serde_json::to_string(&signal).unwrap();
            let back: BridgeEventSignal = serde_json::from_str(&json).unwrap();
            assert_eq!(back.category, category);
        }
    }

    #[test]
    fn bridge_event_signal_zero_phi() {
        let signal = BridgeEventSignal {
            event_type: "test".to_string(),
            payload_hash: fake_action_hash(),
            category: SupportCategory::General,
            phi: 0.0,
        };
        let json = serde_json::to_string(&signal).unwrap();
        let back: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.phi, 0.0);
    }

    #[test]
    fn bridge_event_signal_negative_phi() {
        let signal = BridgeEventSignal {
            event_type: "test".to_string(),
            payload_hash: fake_action_hash(),
            category: SupportCategory::General,
            phi: -0.5,
        };
        let json = serde_json::to_string(&signal).unwrap();
        let back: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.phi, -0.5);
    }

    // ========================================================================
    // Integrity entry type serde roundtrips (from coordinator perspective)
    // ========================================================================

    #[test]
    fn diagnostic_result_serde_roundtrip() {
        let diag = DiagnosticResult {
            ticket_hash: Some(fake_action_hash()),
            diagnostic_type: DiagnosticType::NetworkCheck,
            findings: r#"{"status":"ok"}"#.to_string(),
            severity: DiagnosticSeverity::Healthy,
            recommendations: vec!["All clear".to_string()],
            agent: fake_agent(),
            scrubbed: false,
            created_at: Timestamp::from_micros(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&diag).unwrap();
        let back: DiagnosticResult = serde_json::from_str(&json).unwrap();
        assert_eq!(diag, back);
    }

    #[test]
    fn privacy_preference_serde_roundtrip() {
        let pref = PrivacyPreference {
            agent: fake_agent(),
            sharing_tier: SharingTier::Full,
            allowed_categories: vec![SupportCategory::Network],
            share_system_info: true,
            share_resolution_patterns: false,
            share_cognitive_updates: true,
            updated_at: Timestamp::from_micros(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&pref).unwrap();
        let back: PrivacyPreference = serde_json::from_str(&json).unwrap();
        assert_eq!(pref, back);
    }

    #[test]
    fn cognitive_update_serde_roundtrip() {
        let cu = CognitiveUpdate {
            category: SupportCategory::Software,
            encoding: vec![0u8; 2048],
            phi: 0.42,
            resolution_pattern: "Restart the service".to_string(),
            source_agent: fake_agent(),
            created_at: Timestamp::from_micros(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&cu).unwrap();
        let back: CognitiveUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(cu, back);
    }

    // ========================================================================
    // UpdateAvailInput serde roundtrip
    // ========================================================================

    #[test]
    fn update_avail_input_serde_roundtrip() {
        let input = UpdateAvailInput {
            helper_hash: fake_action_hash(),
            available: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: UpdateAvailInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.available, true);
    }

    // ========================================================================
    // HelperWorkload serde roundtrip
    // ========================================================================

    #[test]
    fn helper_workload_serde_roundtrip() {
        let workload = HelperWorkload {
            agent: fake_agent(),
            active_tickets: 2,
            max_concurrent: 5,
        };
        let json = serde_json::to_string(&workload).unwrap();
        let back: HelperWorkload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.active_tickets, 2);
        assert_eq!(back.max_concurrent, 5);
    }

    #[test]
    fn diagnostic_result_no_ticket_roundtrip() {
        let diag = DiagnosticResult {
            ticket_hash: None,
            diagnostic_type: DiagnosticType::DiskSpace,
            findings: r#"{"disk_free_gb":120}"#.to_string(),
            severity: DiagnosticSeverity::Warning,
            recommendations: vec![
                "Consider cleanup".to_string(),
                "Archive old logs".to_string(),
            ],
            agent: fake_agent(),
            scrubbed: true,
            created_at: Timestamp::from_micros(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&diag).unwrap();
        let back: DiagnosticResult = serde_json::from_str(&json).unwrap();
        assert!(back.ticket_hash.is_none());
        assert!(back.scrubbed);
        assert_eq!(back.recommendations.len(), 2);
    }
}
