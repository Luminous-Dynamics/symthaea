// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Community Land Trust Coordinator Zome
//! Business logic for land trusts, ground leases, resale calculations,
//! and affordability reporting.

use hdk::prelude::*;
use housing_clt_integrity::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::get_latest_record;


/// Input for verifying a property before creating a CLT lease
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerifyPropertyForLeaseInput {
    pub property_id: String,
    pub expected_owner_did: Option<String>,
}

/// Result of property verification for CLT lease
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PropertyVerificationResult {
    pub verified: bool,
    pub owner_did: Option<String>,
    pub has_clear_title: bool,
    pub error: Option<String>,
}

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Create a new community land trust

#[hdk_extern]
pub fn create_land_trust(trust: LandTrust) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_land_trust")?;
    if trust.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Trust name must be at most 256 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::LandTrust(trust.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_trusts".to_string())))?;
    create_link(
        anchor_hash("all_trusts")?,
        action_hash.clone(),
        LinkTypes::AllTrusts,
        (),
    )?;

    // Geohash spatial index (centroid of boundary polygon)
    if !trust.boundary.is_empty() {
        let n = trust.boundary.len() as f64;
        let centroid_lat = trust.boundary.iter().map(|(lat, _)| lat).sum::<f64>() / n;
        let centroid_lon = trust.boundary.iter().map(|(_, lon)| lon).sum::<f64>() / n;
        let geo_hash = commons_types::geo::geohash_encode(centroid_lat, centroid_lon, 6);
        let geo_anchor_str = format!("geo:{}", geo_hash);
        create_entry(&EntryTypes::Anchor(Anchor(geo_anchor_str.clone())))?;
        create_link(anchor_hash(&geo_anchor_str)?, action_hash.clone(), LinkTypes::GeoIndex, geo_hash.as_bytes().to_vec())?;
    }

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created trust".into()
    )))
}

// ============================================================================
// UPDATE FUNCTIONS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateLandTrustInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: LandTrust,
}

/// Update a land trust entry (general-purpose update replacing the whole entry)
#[hdk_extern]
pub fn update_land_trust(input: UpdateLandTrustInput) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_land_trust")?;
    update_entry(
        input.original_action_hash,
        &EntryTypes::LandTrust(input.updated_entry),
    )
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateGroundLeaseInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: GroundLease,
}

/// Update a ground lease entry (general-purpose update replacing the whole entry)
#[hdk_extern]
pub fn update_ground_lease(input: UpdateGroundLeaseInput) -> ExternResult<ActionHash> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_ground_lease")?;
    update_entry(
        input.original_action_hash,
        &EntryTypes::GroundLease(input.updated_entry),
    )
}

// ============================================================================
// Cross-domain: Property ownership verification via bridge dispatch
// ============================================================================

/// Verify property ownership by calling property_registry within the same DNA.
///
/// This is a concrete cross-domain call example: housing-clt queries
/// property-registry directly using `call(CallTargetCell::Local, ...)`.
#[hdk_extern]
pub fn verify_property_for_lease(
    input: VerifyPropertyForLeaseInput,
) -> ExternResult<PropertyVerificationResult> {
    // 1. Check if property exists in the registry
    let get_response = call(
        CallTargetCell::Local,
        ZomeName::from("property_registry"),
        FunctionName::from("get_property"),
        None,
        input.property_id.clone(),
    );

    let property_exists = match &get_response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            // Decode as Option<Record> — None means not found
            let record: Option<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;
            record.is_some()
        }
        _ => false,
    };

    if !property_exists {
        return Ok(PropertyVerificationResult {
            verified: false,
            owner_did: None,
            has_clear_title: false,
            error: Some(format!(
                "Property '{}' not found in registry",
                input.property_id
            )),
        });
    }

    // 2. Check if property has clear title (no encumbrances)
    let title_response = call(
        CallTargetCell::Local,
        ZomeName::from("property_registry"),
        FunctionName::from("has_clear_title"),
        None,
        input.property_id.clone(),
    );

    let has_clear_title = match &title_response {
        Ok(ZomeCallResponse::Ok(extern_io)) => extern_io.decode::<bool>().unwrap_or(false),
        _ => false,
    };

    // 3. Optionally verify ownership matches expected DID
    let mut owner_did = None;
    let mut ownership_match = true;

    if let Some(ref expected) = input.expected_owner_did {
        let verify_response = call(
            CallTargetCell::Local,
            ZomeName::from("property_registry"),
            FunctionName::from("verify_ownership"),
            None,
            serde_json::json!({
                "property_id": input.property_id,
                "expected_owner": expected,
            })
            .to_string(),
        );

        match &verify_response {
            Ok(ZomeCallResponse::Ok(extern_io)) => {
                ownership_match = extern_io.decode::<bool>().unwrap_or(false);
                owner_did = Some(expected.clone());
            }
            _ => {
                ownership_match = false;
            }
        }
    }

    Ok(PropertyVerificationResult {
        verified: property_exists && has_clear_title && ownership_match,
        owner_did,
        has_clear_title,
        error: if !has_clear_title {
            Some("Property has encumbrances — clear title required for CLT lease".to_string())
        } else if !ownership_match {
            Some("Property ownership does not match expected trust DID".to_string())
        } else {
            None
        },
    })
}

// ============================================================================
// Lease Management
// ============================================================================

/// Issue a ground lease for a unit under the trust
#[hdk_extern]
pub fn issue_ground_lease(lease: GroundLease) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "issue_ground_lease")?;
    let action_hash = create_entry(&EntryTypes::GroundLease(lease.clone()))?;

    // Link trust to lease
    create_link(
        lease.trust_hash,
        action_hash.clone(),
        LinkTypes::TrustToLease,
        (),
    )?;

    // Link leaseholder to lease
    create_link(
        lease.leaseholder,
        action_hash.clone(),
        LinkTypes::LeaseholderToLease,
        (),
    )?;

    // Link unit to lease
    create_link(
        lease.unit_hash,
        action_hash.clone(),
        LinkTypes::UnitToLease,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created lease".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CalculateResaleInput {
    pub lease_hash: ActionHash,
    pub original_price_cents: u64,
    pub years_held: u32,
    pub improvements_value_cents: u64,
    pub ami_at_purchase: Option<u64>,
    pub current_ami: Option<u64>,
}

/// Calculate the maximum resale price under the ground lease formula.
///
/// Formula types:
/// - AppreciationCap: original * (1 + rate)^years + improvement_credit
/// - AreaMedianIncome: min(appreciated_value, ami_cap_percent * current_ami / 12 * affordability_factor)
/// - ConsumerPriceIndex: original * (1 + 0.03)^years + improvement_credit (3% assumed CPI)
/// - Hybrid: min(appreciation_cap_result, ami_result)
#[hdk_extern]
pub fn calculate_max_resale_price(input: CalculateResaleInput) -> ExternResult<Record> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "calculate_max_resale_price")?;
    let lease_record = get(input.lease_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Lease not found".into())))?;

    let lease: GroundLease = lease_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid lease entry".into()
        )))?;

    let formula = &lease.resale_formula;
    let improvement_credit_pct = formula.improvement_credit_percent.unwrap_or(0) as u64;
    let improvement_credit = input.improvements_value_cents * improvement_credit_pct / 100;

    let calculated_max_price_cents = match formula.formula_type {
        FormulaType::AppreciationCap => {
            let annual_rate = formula.max_appreciation_percent_annual.unwrap_or(2) as f64 / 100.0;
            let appreciated = input.original_price_cents as f64
                * (1.0 + annual_rate).powi(input.years_held as i32);
            appreciated as u64 + improvement_credit
        }
        FormulaType::AreaMedianIncome => {
            let ami_cap_pct = formula.ami_cap_percent.unwrap_or(80) as f64 / 100.0;
            let current_ami = input.current_ami.unwrap_or(6_000_000) as f64;
            // Maximum price = AMI cap % * annual AMI * affordability factor (assume 3x annual income)
            let ami_based_max = (ami_cap_pct * current_ami * 3.0) as u64;

            // Also calculate simple appreciation as a floor
            let appreciated =
                input.original_price_cents as f64 * (1.02_f64).powi(input.years_held as i32);
            let appreciated_with_credit = appreciated as u64 + improvement_credit;

            // Take the lesser of AMI-based and appreciated value
            ami_based_max.min(appreciated_with_credit)
        }
        FormulaType::ConsumerPriceIndex => {
            // Use 3% as assumed CPI rate
            let appreciated =
                input.original_price_cents as f64 * (1.03_f64).powi(input.years_held as i32);
            appreciated as u64 + improvement_credit
        }
        FormulaType::Hybrid => {
            // Appreciation cap calculation
            let annual_rate = formula.max_appreciation_percent_annual.unwrap_or(2) as f64 / 100.0;
            let appreciation_cap = input.original_price_cents as f64
                * (1.0 + annual_rate).powi(input.years_held as i32);
            let appreciation_result = appreciation_cap as u64 + improvement_credit;

            // AMI-based calculation
            let ami_cap_pct = formula.ami_cap_percent.unwrap_or(80) as f64 / 100.0;
            let current_ami = input.current_ami.unwrap_or(6_000_000) as f64;
            let ami_result = (ami_cap_pct * current_ami * 3.0) as u64;

            // Hybrid takes the minimum of both to ensure maximum affordability
            appreciation_result.min(ami_result)
        }
    };

    let calc = ResaleCalculation {
        lease_hash: input.lease_hash.clone(),
        original_price_cents: input.original_price_cents,
        years_held: input.years_held,
        improvements_value_cents: input.improvements_value_cents,
        calculated_max_price_cents,
        ami_at_purchase: input.ami_at_purchase,
        current_ami: input.current_ami,
    };

    let action_hash = create_entry(&EntryTypes::ResaleCalculation(calc))?;

    // Link lease to calculation
    create_link(
        input.lease_hash,
        action_hash.clone(),
        LinkTypes::LeaseToResaleCalc,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created resale calculation".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferLeaseInput {
    pub lease_hash: ActionHash,
    pub new_leaseholder: AgentPubKey,
}

/// Transfer a ground lease to a new leaseholder
#[hdk_extern]
pub fn transfer_lease(input: TransferLeaseInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "transfer_lease")?;
    let record = get(input.lease_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Lease not found".into())))?;

    let mut lease: GroundLease = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid lease entry".into()
        )))?;

    let old_leaseholder = lease.leaseholder.clone();
    lease.leaseholder = input.new_leaseholder.clone();

    let new_hash = update_entry(input.lease_hash.clone(), &EntryTypes::GroundLease(lease))?;

    // Remove old leaseholder link
    let links = get_links(
        LinkQuery::try_new(old_leaseholder, LinkTypes::LeaseholderToLease)?,
        GetStrategy::default(),
    )?;
    for link in links {
        let target = ActionHash::try_from(link.target.clone());
        if let Ok(target_hash) = target {
            if target_hash == input.lease_hash {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    // Add new leaseholder link
    create_link(
        input.new_leaseholder,
        new_hash.clone(),
        LinkTypes::LeaseholderToLease,
        (),
    )?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated lease".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GenerateAffordabilityInput {
    pub trust_hash: ActionHash,
    pub total_units: u32,
    pub affordable_units: u32,
    pub average_monthly_cost_cents: u64,
    pub median_area_income_cents: u64,
}

/// Generate an affordability report for a trust
#[hdk_extern]
pub fn generate_affordability_report(input: GenerateAffordabilityInput) -> ExternResult<Record> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "generate_affordability_report")?;
    let now = sys_time()?;

    // Affordability ratio = (average monthly cost * 12) / median annual income
    // A ratio <= 0.30 is considered affordable (30% rule)
    let annual_cost = input.average_monthly_cost_cents as f64 * 12.0;
    let affordability_ratio = if input.median_area_income_cents > 0 {
        (annual_cost / input.median_area_income_cents as f64) as f32
    } else {
        1.0
    };

    let report = AffordabilityReport {
        trust_hash: input.trust_hash.clone(),
        report_date: now,
        total_units: input.total_units,
        affordable_units: input.affordable_units,
        average_monthly_cost_cents: input.average_monthly_cost_cents,
        median_area_income_cents: input.median_area_income_cents,
        affordability_ratio,
    };

    let action_hash = create_entry(&EntryTypes::AffordabilityReport(report))?;

    create_link(
        input.trust_hash,
        action_hash.clone(),
        LinkTypes::TrustToReport,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created report".into()
    )))
}

/// Get all ground leases for a trust
#[hdk_extern]
pub fn get_trust_leases(trust_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(trust_hash, LinkTypes::TrustToLease)?,
        GetStrategy::default(),
    )?;

    let mut leases = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            leases.push(record);
        }
    }

    Ok(leases)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateTrustBoardInput {
    pub trust_hash: ActionHash,
    pub new_board: Vec<AgentPubKey>,
}

/// Update the stewardship board of a land trust
#[hdk_extern]
pub fn update_trust_board(input: UpdateTrustBoardInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_trust_board")?;
    if input.new_board.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Board must have at least one member".into()
        )));
    }

    let record = get(input.trust_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Trust not found".into())))?;

    let mut trust: LandTrust = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid trust entry".into()
        )))?;

    trust.stewardship_board = input.new_board;

    let new_hash = update_entry(input.trust_hash, &EntryTypes::LandTrust(trust))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated trust".into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    // ── Coordinator struct serde roundtrips ────────────────────────────

    #[test]
    fn calculate_resale_input_serde_roundtrip() {
        let input = CalculateResaleInput {
            lease_hash: fake_action_hash(),
            original_price_cents: 30000000,
            years_held: 5,
            improvements_value_cents: 2000000,
            ami_at_purchase: Some(6000000),
            current_ami: Some(6500000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CalculateResaleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_price_cents, 30000000);
        assert_eq!(decoded.years_held, 5);
        assert_eq!(decoded.improvements_value_cents, 2000000);
        assert_eq!(decoded.ami_at_purchase, Some(6000000));
        assert_eq!(decoded.current_ami, Some(6500000));
    }

    #[test]
    fn calculate_resale_input_no_ami_serde() {
        let input = CalculateResaleInput {
            lease_hash: fake_action_hash(),
            original_price_cents: 25000000,
            years_held: 10,
            improvements_value_cents: 0,
            ami_at_purchase: None,
            current_ami: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CalculateResaleInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.ami_at_purchase.is_none());
        assert!(decoded.current_ami.is_none());
        assert_eq!(decoded.improvements_value_cents, 0);
    }

    #[test]
    fn transfer_lease_input_serde_roundtrip() {
        let input = TransferLeaseInput {
            lease_hash: fake_action_hash(),
            new_leaseholder: fake_agent(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferLeaseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_leaseholder, fake_agent());
    }

    #[test]
    fn generate_affordability_input_serde_roundtrip() {
        let input = GenerateAffordabilityInput {
            trust_hash: fake_action_hash(),
            total_units: 100,
            affordable_units: 85,
            average_monthly_cost_cents: 120000,
            median_area_income_cents: 6000000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: GenerateAffordabilityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_units, 100);
        assert_eq!(decoded.affordable_units, 85);
        assert_eq!(decoded.average_monthly_cost_cents, 120000);
        assert_eq!(decoded.median_area_income_cents, 6000000);
    }

    #[test]
    fn update_trust_board_input_serde_roundtrip() {
        let agent_a = AgentPubKey::from_raw_36(vec![1u8; 36]);
        let agent_b = AgentPubKey::from_raw_36(vec![2u8; 36]);
        let input = UpdateTrustBoardInput {
            trust_hash: fake_action_hash(),
            new_board: vec![agent_a.clone(), agent_b.clone()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateTrustBoardInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_board.len(), 2);
        assert_eq!(decoded.new_board[0], agent_a);
        assert_eq!(decoded.new_board[1], agent_b);
    }

    #[test]
    fn update_trust_board_input_single_member_serde() {
        let input = UpdateTrustBoardInput {
            trust_hash: fake_action_hash(),
            new_board: vec![fake_agent()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateTrustBoardInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_board.len(), 1);
    }

    // ── Cross-domain types serde roundtrips ────────────────────────────

    #[test]
    fn verify_property_input_serde_roundtrip() {
        let input = VerifyPropertyForLeaseInput {
            property_id: "PROP-123".to_string(),
            expected_owner_did: Some("did:key:z6Mk...".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VerifyPropertyForLeaseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "PROP-123");
        assert_eq!(
            decoded.expected_owner_did,
            Some("did:key:z6Mk...".to_string())
        );
    }

    #[test]
    fn property_verification_result_serde_roundtrip() {
        let result = PropertyVerificationResult {
            verified: true,
            owner_did: Some("did:key:z6Mk...".to_string()),
            has_clear_title: true,
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PropertyVerificationResult = serde_json::from_str(&json).unwrap();
        assert!(decoded.verified);
        assert!(decoded.has_clear_title);
        assert!(decoded.error.is_none());
    }

    #[test]
    fn property_verification_result_failed_serde() {
        let result = PropertyVerificationResult {
            verified: false,
            owner_did: None,
            has_clear_title: false,
            error: Some("Property not found".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PropertyVerificationResult = serde_json::from_str(&json).unwrap();
        assert!(!decoded.verified);
        assert!(!decoded.has_clear_title);
        assert!(decoded.error.as_ref().unwrap().contains("not found"));
    }

    // ── Integrity enum serde roundtrips ────────────────────────────────

    #[test]
    fn formula_type_all_variants_serde() {
        let variants = vec![
            FormulaType::AppreciationCap,
            FormulaType::AreaMedianIncome,
            FormulaType::ConsumerPriceIndex,
            FormulaType::Hybrid,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: FormulaType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    // ── Integrity entry struct serde roundtrips ────────────────────────

    #[test]
    fn land_trust_serde_roundtrip() {
        let trust = LandTrust {
            id: "trust-001".to_string(),
            name: "Community Housing Trust".to_string(),
            mission: "Affordable housing for all".to_string(),
            boundary: vec![(45.0, -122.0), (45.1, -122.0), (45.1, -122.1)],
            charter_hash: Some(fake_action_hash()),
            stewardship_board: vec![fake_agent()],
            affordability_target_ami_percent: 80,
            created_at: Timestamp::from_micros(1000),
        };
        let json = serde_json::to_string(&trust).unwrap();
        let decoded: LandTrust = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, trust);
    }

    #[test]
    fn resale_formula_serde_roundtrip() {
        let formula = ResaleFormula {
            formula_type: FormulaType::Hybrid,
            max_appreciation_percent_annual: Some(3),
            ami_cap_percent: Some(80),
            improvement_credit_percent: Some(100),
        };
        let json = serde_json::to_string(&formula).unwrap();
        let decoded: ResaleFormula = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, formula);
    }

    #[test]
    fn ground_lease_serde_roundtrip() {
        let lease = GroundLease {
            trust_hash: fake_action_hash(),
            unit_hash: fake_action_hash(),
            leaseholder: fake_agent(),
            lease_term_years: 99,
            ground_rent_monthly_cents: 50000,
            resale_formula: ResaleFormula {
                formula_type: FormulaType::AppreciationCap,
                max_appreciation_percent_annual: Some(2),
                ami_cap_percent: None,
                improvement_credit_percent: Some(75),
            },
            started_at: Timestamp::from_micros(1000),
            expires_at: Timestamp::from_micros(2000000),
        };
        let json = serde_json::to_string(&lease).unwrap();
        let decoded: GroundLease = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, lease);
    }

    #[test]
    fn resale_calculation_serde_roundtrip() {
        let calc = ResaleCalculation {
            lease_hash: fake_action_hash(),
            original_price_cents: 30000000,
            years_held: 5,
            improvements_value_cents: 2000000,
            calculated_max_price_cents: 35000000,
            ami_at_purchase: Some(6000000),
            current_ami: Some(6500000),
        };
        let json = serde_json::to_string(&calc).unwrap();
        let decoded: ResaleCalculation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, calc);
    }

    #[test]
    fn affordability_report_serde_roundtrip() {
        let report = AffordabilityReport {
            trust_hash: fake_action_hash(),
            report_date: Timestamp::from_micros(2000),
            total_units: 100,
            affordable_units: 85,
            average_monthly_cost_cents: 120000,
            median_area_income_cents: 6000000,
            affordability_ratio: 0.24,
        };
        let json = serde_json::to_string(&report).unwrap();
        let decoded: AffordabilityReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, report);
    }

    // ── VerifyPropertyForLeaseInput edge cases ──────────────────────

    #[test]
    fn verify_property_input_no_expected_owner_serde() {
        let input = VerifyPropertyForLeaseInput {
            property_id: "PROP-NO-OWNER".to_string(),
            expected_owner_did: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VerifyPropertyForLeaseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.property_id, "PROP-NO-OWNER");
        assert!(decoded.expected_owner_did.is_none());
    }

    #[test]
    fn verify_property_input_empty_property_id_serde() {
        let input = VerifyPropertyForLeaseInput {
            property_id: "".to_string(),
            expected_owner_did: Some("did:key:z6MkOwner".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VerifyPropertyForLeaseInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.property_id.is_empty());
        assert!(decoded.expected_owner_did.is_some());
    }

    // ── PropertyVerificationResult edge cases ───────────────────────

    #[test]
    fn property_verification_result_encumbrance_error_serde() {
        let result = PropertyVerificationResult {
            verified: false,
            owner_did: Some("did:key:z6MkOwner".to_string()),
            has_clear_title: false,
            error: Some(
                "Property has encumbrances \u{2014} clear title required for CLT lease".to_string(),
            ),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PropertyVerificationResult = serde_json::from_str(&json).unwrap();
        assert!(!decoded.verified);
        assert!(!decoded.has_clear_title);
        assert!(decoded.error.as_ref().unwrap().contains("encumbrances"));
        assert!(decoded.owner_did.is_some());
    }

    #[test]
    fn property_verification_result_equality() {
        let a = PropertyVerificationResult {
            verified: true,
            owner_did: Some("did:key:z6MkA".to_string()),
            has_clear_title: true,
            error: None,
        };
        let b = PropertyVerificationResult {
            verified: true,
            owner_did: Some("did:key:z6MkA".to_string()),
            has_clear_title: true,
            error: None,
        };
        assert_eq!(a, b);

        let c = PropertyVerificationResult {
            verified: false,
            owner_did: Some("did:key:z6MkA".to_string()),
            has_clear_title: true,
            error: None,
        };
        assert_ne!(a, c);
    }

    // ── Trust board composition ─────────────────────────────────────

    #[test]
    fn update_trust_board_many_members_serde() {
        let board: Vec<AgentPubKey> = (0..25)
            .map(|i| AgentPubKey::from_raw_36(vec![i; 36]))
            .collect();
        let input = UpdateTrustBoardInput {
            trust_hash: fake_action_hash(),
            new_board: board.clone(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateTrustBoardInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_board.len(), 25);
        assert_eq!(decoded.new_board[0], board[0]);
        assert_eq!(decoded.new_board[24], board[24]);
    }

    #[test]
    fn update_trust_board_duplicate_members_serde() {
        // The coordinator doesn't prevent duplicates, just requires non-empty
        let agent = fake_agent();
        let input = UpdateTrustBoardInput {
            trust_hash: fake_action_hash(),
            new_board: vec![agent.clone(), agent.clone(), agent.clone()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateTrustBoardInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_board.len(), 3);
        assert_eq!(decoded.new_board[0], decoded.new_board[1]);
    }

    // ── Ground lease with zero rent serde ───────────────────────────

    #[test]
    fn ground_lease_zero_rent_serde_roundtrip() {
        let lease = GroundLease {
            trust_hash: fake_action_hash(),
            unit_hash: fake_action_hash(),
            leaseholder: fake_agent(),
            lease_term_years: 99,
            ground_rent_monthly_cents: 0,
            resale_formula: ResaleFormula {
                formula_type: FormulaType::ConsumerPriceIndex,
                max_appreciation_percent_annual: None,
                ami_cap_percent: None,
                improvement_credit_percent: None,
            },
            started_at: Timestamp::from_micros(1000),
            expires_at: Timestamp::from_micros(2000000),
        };
        let json = serde_json::to_string(&lease).unwrap();
        let decoded: GroundLease = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.ground_rent_monthly_cents, 0);
        assert_eq!(
            decoded.resale_formula.formula_type,
            FormulaType::ConsumerPriceIndex
        );
        assert!(decoded
            .resale_formula
            .max_appreciation_percent_annual
            .is_none());
    }

    // ── Land trust with all optional fields ─────────────────────────

    #[test]
    fn land_trust_no_charter_serde_roundtrip() {
        let trust = LandTrust {
            id: "trust-no-charter".to_string(),
            name: "Nascent Trust".to_string(),
            mission: "Building community".to_string(),
            boundary: vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            charter_hash: None,
            stewardship_board: vec![fake_agent()],
            affordability_target_ami_percent: 120,
            created_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&trust).unwrap();
        let decoded: LandTrust = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, trust);
        assert!(decoded.charter_hash.is_none());
        assert_eq!(decoded.affordability_target_ami_percent, 120);
    }

    // ── CalculateResaleInput boundary values ────────────────────────

    #[test]
    fn calculate_resale_input_zero_years_zero_improvements_serde() {
        let input = CalculateResaleInput {
            lease_hash: fake_action_hash(),
            original_price_cents: 20000000,
            years_held: 0,
            improvements_value_cents: 0,
            ami_at_purchase: None,
            current_ami: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CalculateResaleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.years_held, 0);
        assert_eq!(decoded.improvements_value_cents, 0);
        assert_eq!(decoded.original_price_cents, 20000000);
    }

    #[test]
    fn calculate_resale_input_large_values_serde() {
        let input = CalculateResaleInput {
            lease_hash: fake_action_hash(),
            original_price_cents: u64::MAX,
            years_held: u32::MAX,
            improvements_value_cents: u64::MAX,
            ami_at_purchase: Some(u64::MAX),
            current_ami: Some(u64::MAX),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CalculateResaleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_price_cents, u64::MAX);
        assert_eq!(decoded.years_held, u32::MAX);
        assert_eq!(decoded.ami_at_purchase, Some(u64::MAX));
    }

    // ── GenerateAffordabilityInput boundary values ──────────────────

    #[test]
    fn generate_affordability_input_single_unit_serde() {
        let input = GenerateAffordabilityInput {
            trust_hash: fake_action_hash(),
            total_units: 1,
            affordable_units: 1,
            average_monthly_cost_cents: 80000,
            median_area_income_cents: 5000000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: GenerateAffordabilityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_units, 1);
        assert_eq!(decoded.affordable_units, 1);
    }

    #[test]
    fn generate_affordability_input_zero_affordable_serde() {
        let input = GenerateAffordabilityInput {
            trust_hash: fake_action_hash(),
            total_units: 500,
            affordable_units: 0,
            average_monthly_cost_cents: 300000,
            median_area_income_cents: 4000000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: GenerateAffordabilityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.affordable_units, 0);
        assert_eq!(decoded.total_units, 500);
    }

    // ── TransferLeaseInput with distinct hashes ─────────────────────

    #[test]
    fn transfer_lease_input_distinct_agent_serde() {
        let new_leaseholder = AgentPubKey::from_raw_36(vec![0xAB; 36]);
        let input = TransferLeaseInput {
            lease_hash: ActionHash::from_raw_36(vec![0xCD; 36]),
            new_leaseholder: new_leaseholder.clone(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferLeaseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_leaseholder, new_leaseholder);
    }

    // ── Resale formula all variants serde ───────────────────────────

    #[test]
    fn resale_formula_all_variants_serde_from_coordinator() {
        let formulas = vec![
            ResaleFormula {
                formula_type: FormulaType::AppreciationCap,
                max_appreciation_percent_annual: Some(2),
                ami_cap_percent: None,
                improvement_credit_percent: Some(75),
            },
            ResaleFormula {
                formula_type: FormulaType::AreaMedianIncome,
                max_appreciation_percent_annual: None,
                ami_cap_percent: Some(80),
                improvement_credit_percent: Some(100),
            },
            ResaleFormula {
                formula_type: FormulaType::ConsumerPriceIndex,
                max_appreciation_percent_annual: None,
                ami_cap_percent: None,
                improvement_credit_percent: None,
            },
            ResaleFormula {
                formula_type: FormulaType::Hybrid,
                max_appreciation_percent_annual: Some(3),
                ami_cap_percent: Some(120),
                improvement_credit_percent: Some(50),
            },
        ];
        for formula in formulas {
            let json = serde_json::to_string(&formula).unwrap();
            let decoded: ResaleFormula = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, formula);
        }
    }

    // ── AffordabilityReport boundary ratio values ───────────────────

    #[test]
    fn affordability_report_ratio_thirty_percent_rule() {
        // 30% rule: ratio = 0.30 means housing is at the threshold of affordable
        let report = AffordabilityReport {
            trust_hash: fake_action_hash(),
            report_date: Timestamp::from_micros(3000),
            total_units: 200,
            affordable_units: 180,
            average_monthly_cost_cents: 150000,
            median_area_income_cents: 6000000,
            affordability_ratio: 0.30,
        };
        let json = serde_json::to_string(&report).unwrap();
        let decoded: AffordabilityReport = serde_json::from_str(&json).unwrap();
        assert!((decoded.affordability_ratio - 0.30).abs() < f32::EPSILON);
        assert_eq!(decoded.affordable_units, 180);
    }

    #[test]
    fn affordability_report_zero_ratio_serde() {
        let report = AffordabilityReport {
            trust_hash: fake_action_hash(),
            report_date: Timestamp::from_micros(4000),
            total_units: 50,
            affordable_units: 50,
            average_monthly_cost_cents: 0,
            median_area_income_cents: 6000000,
            affordability_ratio: 0.0,
        };
        let json = serde_json::to_string(&report).unwrap();
        let decoded: AffordabilityReport = serde_json::from_str(&json).unwrap();
        assert!((decoded.affordability_ratio - 0.0).abs() < f32::EPSILON);
    }

    // ── ResaleCalculation with no AMI data ──────────────────────────

    #[test]
    fn resale_calculation_no_ami_serde_roundtrip() {
        let calc = ResaleCalculation {
            lease_hash: fake_action_hash(),
            original_price_cents: 25000000,
            years_held: 10,
            improvements_value_cents: 5000000,
            calculated_max_price_cents: 32000000,
            ami_at_purchase: None,
            current_ami: None,
        };
        let json = serde_json::to_string(&calc).unwrap();
        let decoded: ResaleCalculation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, calc);
        assert!(decoded.ami_at_purchase.is_none());
        assert!(decoded.current_ami.is_none());
    }

    // ── UpdateLandTrustInput tests ──────────────────────────────────

    #[test]
    fn update_land_trust_input_struct_construction() {
        let input = UpdateLandTrustInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: LandTrust {
                id: "trust-updated".to_string(),
                name: "Updated Trust".to_string(),
                mission: "Updated mission".to_string(),
                boundary: vec![(45.0, -122.0), (45.1, -122.0)],
                charter_hash: None,
                stewardship_board: vec![fake_agent()],
                affordability_target_ami_percent: 60,
                created_at: Timestamp::from_micros(1000),
            },
        };
        assert_eq!(
            input.original_action_hash,
            ActionHash::from_raw_36(vec![0xdb; 36])
        );
        assert_eq!(input.updated_entry.name, "Updated Trust");
        assert_eq!(input.updated_entry.affordability_target_ami_percent, 60);
    }

    #[test]
    fn update_land_trust_input_serde_roundtrip() {
        let input = UpdateLandTrustInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            updated_entry: LandTrust {
                id: "trust-001".to_string(),
                name: "Community Trust".to_string(),
                mission: "Affordable housing for all".to_string(),
                boundary: vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
                charter_hash: Some(fake_action_hash()),
                stewardship_board: vec![fake_agent()],
                affordability_target_ami_percent: 80,
                created_at: Timestamp::from_micros(500),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateLandTrustInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_action_hash, input.original_action_hash);
        assert_eq!(decoded.updated_entry.id, "trust-001");
        assert_eq!(decoded.updated_entry.boundary.len(), 3);
    }

    #[test]
    fn update_land_trust_input_clone_is_equal() {
        let input = UpdateLandTrustInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xef; 36]),
            updated_entry: LandTrust {
                id: "trust-clone".to_string(),
                name: "Cloneable Trust".to_string(),
                mission: "Test cloning".to_string(),
                boundary: vec![],
                charter_hash: None,
                stewardship_board: vec![fake_agent()],
                affordability_target_ami_percent: 100,
                created_at: Timestamp::from_micros(0),
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry, input.updated_entry);
    }

    // ── UpdateGroundLeaseInput tests ────────────────────────────────

    #[test]
    fn update_ground_lease_input_struct_construction() {
        let input = UpdateGroundLeaseInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: GroundLease {
                trust_hash: fake_action_hash(),
                unit_hash: fake_action_hash(),
                leaseholder: fake_agent(),
                lease_term_years: 99,
                ground_rent_monthly_cents: 60000,
                resale_formula: ResaleFormula {
                    formula_type: FormulaType::AppreciationCap,
                    max_appreciation_percent_annual: Some(3),
                    ami_cap_percent: None,
                    improvement_credit_percent: Some(80),
                },
                started_at: Timestamp::from_micros(1000),
                expires_at: Timestamp::from_micros(3000000),
            },
        };
        assert_eq!(
            input.original_action_hash,
            ActionHash::from_raw_36(vec![0xdb; 36])
        );
        assert_eq!(input.updated_entry.ground_rent_monthly_cents, 60000);
        assert_eq!(input.updated_entry.lease_term_years, 99);
    }

    #[test]
    fn update_ground_lease_input_serde_roundtrip() {
        let input = UpdateGroundLeaseInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xcc; 36]),
            updated_entry: GroundLease {
                trust_hash: fake_action_hash(),
                unit_hash: fake_action_hash(),
                leaseholder: fake_agent(),
                lease_term_years: 50,
                ground_rent_monthly_cents: 0,
                resale_formula: ResaleFormula {
                    formula_type: FormulaType::Hybrid,
                    max_appreciation_percent_annual: Some(2),
                    ami_cap_percent: Some(80),
                    improvement_credit_percent: Some(100),
                },
                started_at: Timestamp::from_micros(500),
                expires_at: Timestamp::from_micros(2000000),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateGroundLeaseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_action_hash, input.original_action_hash);
        assert_eq!(decoded.updated_entry.lease_term_years, 50);
        assert_eq!(
            decoded.updated_entry.resale_formula.formula_type,
            FormulaType::Hybrid
        );
    }

    #[test]
    fn update_ground_lease_input_clone_is_equal() {
        let input = UpdateGroundLeaseInput {
            original_action_hash: ActionHash::from_raw_36(vec![0x11; 36]),
            updated_entry: GroundLease {
                trust_hash: fake_action_hash(),
                unit_hash: fake_action_hash(),
                leaseholder: fake_agent(),
                lease_term_years: 25,
                ground_rent_monthly_cents: 30000,
                resale_formula: ResaleFormula {
                    formula_type: FormulaType::ConsumerPriceIndex,
                    max_appreciation_percent_annual: None,
                    ami_cap_percent: None,
                    improvement_credit_percent: None,
                },
                started_at: Timestamp::from_micros(100),
                expires_at: Timestamp::from_micros(1000000),
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry, input.updated_entry);
    }
}
