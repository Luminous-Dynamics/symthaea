// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fact-Check Coordinator Zome
use hdk::prelude::*;
use media_factcheck_integrity::*;
use mycelix_bridge_common::{
    civic_requirement_proposal, GovernanceEligibility,
};
use mycelix_zome_helpers::{get_latest_record};


/// Helper function to create an anchor entry and return its hash
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    if let Err(e) = create_entry(&EntryTypes::Anchor(anchor.clone())) { debug!("Anchor creation warning: {:?}", e); }
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn submit_fact_check(input: SubmitFactCheckInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "submit_fact_check")?;
    let now = sys_time()?;
    let check = FactCheck {
        id: format!("factcheck:{}:{}", input.publication_id, now.as_micros()),
        publication_id: input.publication_id.clone(),
        claim_text: input.claim_text.clone(),
        claim_location: input.claim_location,
        epistemic_position: input.epistemic_position,
        verdict: input.verdict,
        evidence: input.evidence,
        checker_did: input.checker_did.clone(),
        checked: now,
    };

    let action_hash = create_entry(&EntryTypes::FactCheck(check))?;
    create_link(
        anchor_hash(&input.publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToFactChecks,
        (),
    )?;
    create_link(
        anchor_hash(&input.checker_did)?,
        action_hash.clone(),
        LinkTypes::CheckerToFactChecks,
        (),
    )?;

    // Link claim text for cross-reference
    create_link(
        anchor_hash(&input.claim_text)?,
        action_hash.clone(),
        LinkTypes::ClaimToFactCheck,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitFactCheckInput {
    pub publication_id: String,
    pub claim_text: String,
    pub claim_location: String,
    pub epistemic_position: EpistemicPosition,
    pub verdict: FactCheckVerdict,
    pub evidence: Vec<EvidenceItem>,
    pub checker_did: String,
}

#[hdk_extern]
pub fn get_publication_fact_checks(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut checks = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::PublicationToFactChecks as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            checks.push(record);
        }
    }
    Ok(checks)
}

#[hdk_extern]
pub fn search_fact_checks_for_claim(claim_text: String) -> ExternResult<Vec<Record>> {
    let mut checks = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&claim_text)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::ClaimToFactCheck as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            checks.push(record);
        }
    }
    Ok(checks)
}

#[hdk_extern]
pub fn update_source_credibility(input: UpdateCredibilityInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "update_source_credibility")?;
    let now = sys_time()?;
    let source = SourceCredibility {
        source_id: input.source_id,
        source_type: input.source_type,
        credibility_score: input.credibility_score,
        verification_count: input.verification_count,
        dispute_count: input.dispute_count,
        last_assessed: now,
    };

    let action_hash = create_entry(&EntryTypes::SourceCredibility(source))?;
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateCredibilityInput {
    pub source_id: String,
    pub source_type: SourceType,
    pub credibility_score: f64,
    pub verification_count: u32,
    pub dispute_count: u32,
}

/// Get a specific fact check by ID
#[hdk_extern]
pub fn get_fact_check(fact_check_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheck,
        )?))
        .include_entries(true);

    // Take the LAST match — update_entry appends newer versions later in the chain
    let mut found: Option<Record> = None;
    for record in query(filter)? {
        if let Some(check) = record.entry().to_app_option::<FactCheck>().ok().flatten() {
            if check.id == fact_check_id {
                found = Some(record);
            }
        }
    }
    Ok(found)
}

/// Get fact checks by checker
#[hdk_extern]
pub fn get_checker_fact_checks(checker_did: String) -> ExternResult<Vec<Record>> {
    let mut checks = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&checker_did)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::CheckerToFactChecks as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            checks.push(record);
        }
    }
    Ok(checks)
}

/// Get fact checks by verdict
#[hdk_extern]
pub fn get_fact_checks_by_verdict(verdict: FactCheckVerdict) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheck,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(check) = record.entry().to_app_option::<FactCheck>().ok().flatten() {
            if check.verdict == verdict {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Dispute a fact check
#[hdk_extern]
pub fn dispute_fact_check(input: DisputeFactCheckInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "dispute_fact_check")?;
    let now = sys_time()?;
    let dispute = FactCheckDispute {
        id: format!("dispute:{}:{}", input.fact_check_id, now.as_micros()),
        fact_check_id: input.fact_check_id.clone(),
        disputer_did: input.disputer_did,
        reason: input.reason,
        counter_evidence: input.counter_evidence,
        status: DisputeStatus::Pending,
        created: now,
        resolved: None,
    };

    let action_hash = create_entry(&EntryTypes::FactCheckDispute(dispute))?;
    create_link(
        anchor_hash(&input.fact_check_id)?,
        action_hash.clone(),
        LinkTypes::FactCheckToDisputes,
        (),
    )?;
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DisputeFactCheckInput {
    pub fact_check_id: String,
    pub disputer_did: String,
    pub reason: String,
    pub counter_evidence: Vec<EvidenceItem>,
}

/// Resolve a fact check dispute
#[hdk_extern]
pub fn resolve_dispute(input: ResolveDisputeInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "resolve_dispute")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheckDispute,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(dispute) = record
            .entry()
            .to_app_option::<FactCheckDispute>()
            .ok()
            .flatten()
        {
            if dispute.id == input.dispute_id {
                let now = sys_time()?;
                let updated = FactCheckDispute {
                    status: input.resolution,
                    resolved: Some(now),
                    ..dispute
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::FactCheckDispute(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Dispute not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveDisputeInput {
    pub dispute_id: String,
    pub resolution: DisputeStatus,
}

/// Get disputes for a fact check
#[hdk_extern]
pub fn get_fact_check_disputes(fact_check_id: String) -> ExternResult<Vec<Record>> {
    let mut disputes = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&fact_check_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::FactCheckToDisputes as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            disputes.push(record);
        }
    }
    Ok(disputes)
}

/// Get source credibility by ID
#[hdk_extern]
pub fn get_source_credibility(source_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::SourceCredibility,
        )?))
        .include_entries(true);

    // Take the LAST match — update_entry appends newer versions later in the chain
    let mut found: Option<Record> = None;
    for record in query(filter)? {
        if let Some(source) = record
            .entry()
            .to_app_option::<SourceCredibility>()
            .ok()
            .flatten()
        {
            if source.source_id == source_id {
                found = Some(record);
            }
        }
    }
    Ok(found)
}

/// Get checker stats
#[hdk_extern]
pub fn get_checker_stats(checker_did: String) -> ExternResult<CheckerStats> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheck,
        )?))
        .include_entries(true);

    let mut total_checks = 0;
    let mut true_count = 0;
    let mut false_count = 0;
    let mut mixed_count = 0;

    for record in query(filter)? {
        if let Some(check) = record.entry().to_app_option::<FactCheck>().ok().flatten() {
            if check.checker_did == checker_did {
                total_checks += 1;
                match check.verdict {
                    FactCheckVerdict::True => true_count += 1,
                    FactCheckVerdict::False => false_count += 1,
                    FactCheckVerdict::PartiallyTrue | FactCheckVerdict::Misleading => {
                        mixed_count += 1
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(CheckerStats {
        checker_did,
        total_checks,
        true_count,
        false_count,
        mixed_count,
    })
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct CheckerStats {
    pub checker_did: String,
    pub total_checks: u32,
    pub true_count: u32,
    pub false_count: u32,
    pub mixed_count: u32,
}

/// Classify a single FactCheckVerdict into the stats buckets.
///
/// Returns (is_true, is_false, is_mixed) -- exactly one will be true,
/// or all false for verdicts that fall into "other" (not counted).
/// Extracted as a pure function for testability.
pub fn classify_verdict(verdict: &FactCheckVerdict) -> (bool, bool, bool) {
    match verdict {
        FactCheckVerdict::True => (true, false, false),
        FactCheckVerdict::False => (false, true, false),
        FactCheckVerdict::PartiallyTrue | FactCheckVerdict::Misleading => (false, false, true),
        _ => (false, false, false),
    }
}

/// Aggregate a list of verdicts into CheckerStats.
/// Pure function: no HDK calls.
pub fn aggregate_checker_stats(checker_did: String, verdicts: &[FactCheckVerdict]) -> CheckerStats {
    let mut total_checks = 0u32;
    let mut true_count = 0u32;
    let mut false_count = 0u32;
    let mut mixed_count = 0u32;

    for verdict in verdicts {
        total_checks += 1;
        let (t, f, m) = classify_verdict(verdict);
        if t {
            true_count += 1;
        }
        if f {
            false_count += 1;
        }
        if m {
            mixed_count += 1;
        }
    }

    CheckerStats {
        checker_did,
        total_checks,
        true_count,
        false_count,
        mixed_count,
    }
}

/// Add additional evidence to fact check
#[hdk_extern]
pub fn add_evidence(input: AddEvidenceInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "add_evidence")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheck,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(check) = record.entry().to_app_option::<FactCheck>().ok().flatten() {
            if check.id == input.fact_check_id {
                // Only checker can add evidence
                if check.checker_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only checker can add evidence".into()
                    )));
                }

                let mut evidence = check.evidence.clone();
                evidence.extend(input.new_evidence);

                let updated = FactCheck { evidence, ..check };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::FactCheck(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Fact check not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddEvidenceInput {
    pub fact_check_id: String,
    pub requester_did: String,
    pub new_evidence: Vec<EvidenceItem>,
}

/// Update fact check verdict
#[hdk_extern]
pub fn update_verdict(input: UpdateVerdictInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "update_verdict")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheck,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(check) = record.entry().to_app_option::<FactCheck>().ok().flatten() {
            if check.id == input.fact_check_id {
                if check.checker_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only checker can update verdict".into()
                    )));
                }

                let updated = FactCheck {
                    verdict: input.new_verdict,
                    ..check
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::FactCheck(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Fact check not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateVerdictInput {
    pub fact_check_id: String,
    pub requester_did: String,
    pub new_verdict: FactCheckVerdict,
}

// ============================================================================
// Cross-domain: Verify disaster claims via emergency_incidents
// ============================================================================

/// Wire-compatible copy of emergency Disaster for deserialization.
/// Must match emergency_incidents_integrity::Disaster field-for-field.
#[derive(Serialize, Deserialize, Debug, Clone, SerializedBytes)]
struct LocalDisaster {
    pub id: String,
    pub disaster_type: LocalDisasterType,
    pub title: String,
    pub description: String,
    pub severity: LocalSeverityLevel,
    pub declared_by: AgentPubKey,
    pub declared_at: Timestamp,
    pub affected_area: LocalAffectedArea,
    pub status: LocalDisasterStatus,
    pub estimated_affected: u32,
    pub coordination_lead: Option<AgentPubKey>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalDisasterType {
    Hurricane,
    Earthquake,
    Wildfire,
    Flood,
    Tornado,
    Pandemic,
    Industrial,
    MassCasualty,
    CyberAttack,
    Infrastructure,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalSeverityLevel {
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct LocalAffectedArea {
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_km: f32,
    pub boundary: Option<Vec<(f64, f64)>>,
    pub zones: Vec<LocalOperationalZone>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct LocalOperationalZone {
    pub id: String,
    pub name: String,
    pub boundary: Vec<(f64, f64)>,
    pub priority: LocalZonePriority,
    pub status: LocalZoneStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalZonePriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalZoneStatus {
    Unassessed,
    Active,
    Cleared,
    Hazardous,
    Evacuated,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum LocalDisasterStatus {
    Declared,
    Active,
    Recovery,
    Closed,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyDisasterClaimInput {
    /// Disaster ID to look up, if known.
    pub disaster_id: Option<String>,
    /// Keyword to search in disaster titles/descriptions.
    pub claim_keyword: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DisasterVerificationResult {
    pub verified: bool,
    pub disaster_id: Option<String>,
    pub disaster_title: Option<String>,
    pub severity: Option<String>,
    pub active_disaster_count: u32,
    pub error: Option<String>,
}

/// Verify a disaster claim by checking active disasters in emergency_incidents.
///
/// This is a concrete cross-domain call: media-factcheck queries
/// emergency_incidents directly using `call(CallTargetCell::Local, ...)`.
/// Before rating a disaster-related claim, verify the disaster actually exists.
#[hdk_extern]
pub fn verify_disaster_claim(
    input: VerifyDisasterClaimInput,
) -> ExternResult<DisasterVerificationResult> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("emergency_incidents"),
        FunctionName::from("get_active_disasters"),
        None,
        (),
    );

    match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let records: Vec<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;

            let active_count = records.len() as u32;

            // Search for matching disaster
            for record in &records {
                if let Some(disaster) = record
                    .entry()
                    .to_app_option::<LocalDisaster>()
                    .ok()
                    .flatten()
                {
                    let id_match = input
                        .disaster_id
                        .as_ref()
                        .map(|id| disaster.id == *id)
                        .unwrap_or(false);

                    let keyword_match = disaster
                        .title
                        .to_lowercase()
                        .contains(&input.claim_keyword.to_lowercase())
                        || disaster
                            .description
                            .to_lowercase()
                            .contains(&input.claim_keyword.to_lowercase());

                    if id_match || keyword_match {
                        return Ok(DisasterVerificationResult {
                            verified: true,
                            disaster_id: Some(disaster.id),
                            disaster_title: Some(disaster.title),
                            severity: Some(format!("{:?}", disaster.severity)),
                            active_disaster_count: active_count,
                            error: None,
                        });
                    }
                }
            }

            Ok(DisasterVerificationResult {
                verified: false,
                disaster_id: None,
                disaster_title: None,
                severity: None,
                active_disaster_count: active_count,
                error: Some(format!(
                    "No matching disaster found for '{}' among {} active disasters",
                    input.claim_keyword, active_count
                )),
            })
        }
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(DisasterVerificationResult {
            verified: false,
            disaster_id: None,
            disaster_title: None,
            severity: None,
            active_disaster_count: 0,
            error: Some(format!("Network error: {}", err)),
        }),
        _ => Ok(DisasterVerificationResult {
            verified: false,
            disaster_id: None,
            disaster_title: None,
            severity: None,
            active_disaster_count: 0,
            error: Some("Failed to query emergency incidents".into()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // VERDICT CLASSIFICATION TESTS
    // ========================================================================

    #[test]
    fn classify_verdict_true() {
        let (t, f, m) = classify_verdict(&FactCheckVerdict::True);
        assert!(t);
        assert!(!f);
        assert!(!m);
    }

    #[test]
    fn classify_verdict_false() {
        let (t, f, m) = classify_verdict(&FactCheckVerdict::False);
        assert!(!t);
        assert!(f);
        assert!(!m);
    }

    #[test]
    fn classify_verdict_partially_true_is_mixed() {
        let (t, f, m) = classify_verdict(&FactCheckVerdict::PartiallyTrue);
        assert!(!t);
        assert!(!f);
        assert!(m);
    }

    #[test]
    fn classify_verdict_misleading_is_mixed() {
        let (t, f, m) = classify_verdict(&FactCheckVerdict::Misleading);
        assert!(!t);
        assert!(!f);
        assert!(m);
    }

    #[test]
    fn classify_verdict_other_variants_are_uncounted() {
        let others = vec![
            FactCheckVerdict::MostlyTrue,
            FactCheckVerdict::HalfTrue,
            FactCheckVerdict::MostlyFalse,
            FactCheckVerdict::Unverifiable,
            FactCheckVerdict::OutOfContext,
            FactCheckVerdict::Satire,
            FactCheckVerdict::Opinion,
        ];
        for v in others {
            let (t, f, m) = classify_verdict(&v);
            assert!(!t && !f && !m, "Expected (false,false,false) for {:?}", v);
        }
    }

    // ========================================================================
    // AGGREGATE CHECKER STATS TESTS
    // ========================================================================

    #[test]
    fn aggregate_empty_verdicts() {
        let stats = aggregate_checker_stats("did:example:checker".into(), &[]);
        assert_eq!(stats.total_checks, 0);
        assert_eq!(stats.true_count, 0);
        assert_eq!(stats.false_count, 0);
        assert_eq!(stats.mixed_count, 0);
    }

    #[test]
    fn aggregate_all_true() {
        let verdicts = vec![FactCheckVerdict::True; 5];
        let stats = aggregate_checker_stats("did:example:checker".into(), &verdicts);
        assert_eq!(stats.total_checks, 5);
        assert_eq!(stats.true_count, 5);
        assert_eq!(stats.false_count, 0);
        assert_eq!(stats.mixed_count, 0);
    }

    #[test]
    fn aggregate_all_false() {
        let verdicts = vec![FactCheckVerdict::False; 3];
        let stats = aggregate_checker_stats("did:example:checker".into(), &verdicts);
        assert_eq!(stats.total_checks, 3);
        assert_eq!(stats.true_count, 0);
        assert_eq!(stats.false_count, 3);
        assert_eq!(stats.mixed_count, 0);
    }

    #[test]
    fn aggregate_mixed_verdicts() {
        let verdicts = vec![
            FactCheckVerdict::True,
            FactCheckVerdict::False,
            FactCheckVerdict::PartiallyTrue,
            FactCheckVerdict::Misleading,
            FactCheckVerdict::Opinion,      // other
            FactCheckVerdict::Unverifiable, // other
            FactCheckVerdict::True,
        ];
        let stats = aggregate_checker_stats("did:example:checker".into(), &verdicts);
        assert_eq!(stats.total_checks, 7);
        assert_eq!(stats.true_count, 2);
        assert_eq!(stats.false_count, 1);
        assert_eq!(stats.mixed_count, 2);
    }

    #[test]
    fn aggregate_preserves_checker_did() {
        let stats = aggregate_checker_stats(
            "did:example:specific-checker".into(),
            &[FactCheckVerdict::True],
        );
        assert_eq!(stats.checker_did, "did:example:specific-checker");
    }

    // ========================================================================
    // CHECKER STATS SERDE TESTS
    // ========================================================================

    #[test]
    fn checker_stats_serde_roundtrip() {
        let stats = CheckerStats {
            checker_did: "did:example:checker".into(),
            total_checks: 10,
            true_count: 5,
            false_count: 3,
            mixed_count: 2,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let parsed: CheckerStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, stats);
    }

    #[test]
    fn checker_stats_zero_values_serde() {
        let stats = CheckerStats {
            checker_did: "did:example:newchecker".into(),
            total_checks: 0,
            true_count: 0,
            false_count: 0,
            mixed_count: 0,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let parsed: CheckerStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, stats);
    }

    // ========================================================================
    // FACT CHECK VERDICT SERDE TESTS
    // ========================================================================

    #[test]
    fn all_verdict_variants_serde_roundtrip() {
        let variants = vec![
            FactCheckVerdict::True,
            FactCheckVerdict::MostlyTrue,
            FactCheckVerdict::HalfTrue,
            FactCheckVerdict::MostlyFalse,
            FactCheckVerdict::False,
            FactCheckVerdict::Unverifiable,
            FactCheckVerdict::OutOfContext,
            FactCheckVerdict::Satire,
            FactCheckVerdict::Opinion,
            FactCheckVerdict::PartiallyTrue,
            FactCheckVerdict::Misleading,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: FactCheckVerdict = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant, "Failed for {:?}", variant);
        }
    }

    // ========================================================================
    // DISPUTE STATUS SERDE TESTS
    // ========================================================================

    #[test]
    fn all_dispute_status_variants_serde_roundtrip() {
        let variants = vec![
            DisputeStatus::Pending,
            DisputeStatus::Upheld,
            DisputeStatus::Rejected,
            DisputeStatus::PartiallyUpheld,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: DisputeStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // INPUT STRUCT SERDE TESTS
    // ========================================================================

    #[test]
    fn submit_fact_check_input_serde_roundtrip() {
        let input = SubmitFactCheckInput {
            publication_id: "pub-1".into(),
            claim_text: "Earth is round".into(),
            claim_location: "paragraph 1".into(),
            epistemic_position: EpistemicPosition {
                empirical: 0.95,
                normative: 0.1,
                mythic: 0.0,
            },
            verdict: FactCheckVerdict::True,
            evidence: vec![EvidenceItem {
                source_type: SourceType::ScientificStudy,
                source_url: Some("https://nasa.gov".into()),
                source_did: None,
                description: "NASA data".into(),
                supports_claim: true,
            }],
            checker_did: "did:example:checker".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: SubmitFactCheckInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.publication_id, "pub-1");
        assert_eq!(parsed.claim_text, "Earth is round");
        assert_eq!(parsed.evidence.len(), 1);
    }

    #[test]
    fn disaster_verification_result_serde_roundtrip() {
        let result = DisasterVerificationResult {
            verified: true,
            disaster_id: Some("D-001".into()),
            disaster_title: Some("Hurricane Test".into()),
            severity: Some("Level5".into()),
            active_disaster_count: 3,
            error: None,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        let parsed: DisasterVerificationResult = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.verified);
        assert_eq!(parsed.disaster_id, Some("D-001".into()));
        assert_eq!(parsed.active_disaster_count, 3);
        assert!(parsed.error.is_none());
    }

    #[test]
    fn disaster_verification_result_not_verified_serde() {
        let result = DisasterVerificationResult {
            verified: false,
            disaster_id: None,
            disaster_title: None,
            severity: None,
            active_disaster_count: 0,
            error: Some("No disasters found".into()),
        };
        let json = serde_json::to_string(&result).expect("serialize");
        let parsed: DisasterVerificationResult = serde_json::from_str(&json).expect("deserialize");
        assert!(!parsed.verified);
        assert!(parsed.error.is_some());
    }

    #[test]
    fn update_credibility_input_serde_roundtrip() {
        let input = UpdateCredibilityInput {
            source_id: "src-1".into(),
            source_type: SourceType::PrimarySource,
            credibility_score: 0.85,
            verification_count: 10,
            dispute_count: 2,
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UpdateCredibilityInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.source_id, "src-1");
        assert!((parsed.credibility_score - 0.85).abs() < 1e-10);
    }

    #[test]
    fn dispute_fact_check_input_serde_roundtrip() {
        let input = DisputeFactCheckInput {
            fact_check_id: "fc-1".into(),
            disputer_did: "did:example:disputer".into(),
            reason: "Outdated evidence".into(),
            counter_evidence: vec![],
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: DisputeFactCheckInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.fact_check_id, "fc-1");
        assert_eq!(parsed.reason, "Outdated evidence");
        assert!(parsed.counter_evidence.is_empty());
    }

    // ========================================================================
    // Additional edge-case and boundary tests
    // ========================================================================

    #[test]
    fn submit_fact_check_input_empty_evidence() {
        let input = SubmitFactCheckInput {
            publication_id: "pub-no-evidence".into(),
            claim_text: "Unsubstantiated claim".into(),
            claim_location: "header".into(),
            epistemic_position: EpistemicPosition {
                empirical: 0.0,
                normative: 0.0,
                mythic: 0.0,
            },
            verdict: FactCheckVerdict::Unverifiable,
            evidence: vec![],
            checker_did: "did:example:checker".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: SubmitFactCheckInput = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.evidence.is_empty());
        assert_eq!(parsed.verdict, FactCheckVerdict::Unverifiable);
    }

    #[test]
    fn epistemic_position_boundary_values() {
        let positions = vec![
            EpistemicPosition {
                empirical: 0.0,
                normative: 0.0,
                mythic: 0.0,
            },
            EpistemicPosition {
                empirical: 1.0,
                normative: 1.0,
                mythic: 1.0,
            },
            EpistemicPosition {
                empirical: 0.5,
                normative: 0.5,
                mythic: 0.5,
            },
        ];
        for pos in positions {
            let json = serde_json::to_string(&pos).expect("serialize");
            let parsed: EpistemicPosition = serde_json::from_str(&json).expect("deserialize");
            assert!((parsed.empirical - pos.empirical).abs() < 1e-10);
            assert!((parsed.normative - pos.normative).abs() < 1e-10);
            assert!((parsed.mythic - pos.mythic).abs() < 1e-10);
        }
    }

    #[test]
    fn verify_disaster_claim_input_serde_roundtrip() {
        let input = VerifyDisasterClaimInput {
            disaster_id: Some("DISASTER-42".into()),
            claim_keyword: "hurricane".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: VerifyDisasterClaimInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.disaster_id, Some("DISASTER-42".into()));
        assert_eq!(parsed.claim_keyword, "hurricane");
    }

    #[test]
    fn verify_disaster_claim_input_no_disaster_id() {
        let input = VerifyDisasterClaimInput {
            disaster_id: None,
            claim_keyword: "earthquake".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: VerifyDisasterClaimInput = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.disaster_id.is_none());
        assert_eq!(parsed.claim_keyword, "earthquake");
    }

    #[test]
    fn source_type_all_variants_serde() {
        let variants = vec![
            SourceType::PrimarySource,
            SourceType::SecondarySource,
            SourceType::ExpertOpinion,
            SourceType::OfficialDocument,
            SourceType::ScientificStudy,
            SourceType::EyewitnessAccount,
            SourceType::DataAnalysis,
            SourceType::Other("social_media_post".into()),
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: SourceType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant, "Failed for {:?}", variant);
        }
    }

    #[test]
    fn aggregate_only_other_verdicts() {
        // All verdicts that fall into "other" (not counted in true/false/mixed)
        let verdicts = vec![
            FactCheckVerdict::MostlyTrue,
            FactCheckVerdict::HalfTrue,
            FactCheckVerdict::MostlyFalse,
            FactCheckVerdict::Unverifiable,
            FactCheckVerdict::OutOfContext,
            FactCheckVerdict::Satire,
            FactCheckVerdict::Opinion,
        ];
        let stats = aggregate_checker_stats("did:example:other-only".into(), &verdicts);
        assert_eq!(stats.total_checks, 7);
        assert_eq!(stats.true_count, 0);
        assert_eq!(stats.false_count, 0);
        assert_eq!(stats.mixed_count, 0);
    }

    #[test]
    fn dispute_fact_check_input_with_counter_evidence() {
        let input = DisputeFactCheckInput {
            fact_check_id: "fc-2".into(),
            disputer_did: "did:example:disputer2".into(),
            reason: "Newer study contradicts".into(),
            counter_evidence: vec![
                EvidenceItem {
                    source_type: SourceType::ScientificStudy,
                    source_url: Some("https://example.com/study".into()),
                    source_did: None,
                    description: "2026 meta-analysis".into(),
                    supports_claim: false,
                },
                EvidenceItem {
                    source_type: SourceType::ExpertOpinion,
                    source_url: None,
                    source_did: Some("did:example:expert".into()),
                    description: "Expert testimony".into(),
                    supports_claim: false,
                },
            ],
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: DisputeFactCheckInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.counter_evidence.len(), 2);
        assert!(!parsed.counter_evidence[0].supports_claim);
        assert_eq!(
            parsed.counter_evidence[1].source_did.as_deref(),
            Some("did:example:expert")
        );
    }
}
