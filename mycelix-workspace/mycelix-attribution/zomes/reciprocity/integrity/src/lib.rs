// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdi::prelude::*;
use serde::{Deserialize, Serialize};

// ── Entry Types ──────────────────────────────────────────────────────

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PledgeType {
    Financial,
    Compute,
    Bandwidth,
    DeveloperTime,
    QA,
    Documentation,
    /// Pledge tutoring hours to help future learners
    TutoringTime,
    /// Pledge translation time for curriculum localization
    ContentTranslation,
    /// Pledge to review/improve curriculum content
    CurriculumReview,
    Other,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Currency {
    USD,
    EUR,
    GBP,
    BTC,
    ETH,
    Other(String),
}

/// Voluntary record of giving back to a dependency.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReciprocityPledge {
    pub id: String,
    pub dependency_id: String,
    pub contributor_did: String,
    pub organization: Option<String>,
    pub pledge_type: PledgeType,
    pub amount: Option<f64>,
    pub currency: Option<Currency>,
    pub description: String,
    pub evidence_url: Option<String>,
    pub period: Option<String>,
    pub pledged_at: Timestamp,
    pub acknowledged: bool,
}

// ── Entry & Link Enums ───────────────────────────────────────────────

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    ReciprocityPledge(ReciprocityPledge),
}

#[hdk_link_types]
pub enum LinkTypes {
    DependencyToPledges,
    ContributorToPledges,
    AllPledges,
    PledgeRateLimit,
    PledgeById,
}

// ── Pure Validation Functions ────────────────────────────────────────

pub fn validate_create_pledge(
    _action: Create,
    pledge: ReciprocityPledge,
) -> ExternResult<ValidateCallbackResult> {
    if pledge.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Pledge id must not be empty".into(),
        ));
    }
    if pledge.dependency_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "dependency_id must not be empty".into(),
        ));
    }
    if !pledge.contributor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "contributor_did must start with 'did:'".into(),
        ));
    }
    if let Some(amount) = pledge.amount {
        if amount < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "amount must not be negative".into(),
            ));
        }
        if !amount.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "amount must be finite".into(),
            ));
        }
    }
    if pledge.pledge_type == PledgeType::Financial && pledge.currency.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Financial pledges must specify a currency".into(),
        ));
    }
    if pledge.description.len() > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "description must be at most 1000 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_update_pledge(
    _action: Update,
    pledge: ReciprocityPledge,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Same field validation as create
    if pledge.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Pledge id must not be empty".into(),
        ));
    }
    if pledge.dependency_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "dependency_id must not be empty".into(),
        ));
    }
    if !pledge.contributor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "contributor_did must start with 'did:'".into(),
        ));
    }
    if let Some(amount) = pledge.amount {
        if amount < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "amount must not be negative".into(),
            ));
        }
        if !amount.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "amount must be finite".into(),
            ));
        }
    }
    if pledge.pledge_type == PledgeType::Financial && pledge.currency.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Financial pledges must specify a currency".into(),
        ));
    }
    if pledge.description.len() > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "description must be at most 1000 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ── HDI Validation Callback ──────────────────────────────────────────

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ReciprocityPledge(pledge) => validate_create_pledge(action, pledge),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ReciprocityPledge(pledge) => {
                    validate_update_pledge(action, pledge, original_action_hash)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::DependencyToPledges
            | LinkTypes::ContributorToPledges
            | LinkTypes::AllPledges
            | LinkTypes::PledgeRateLimit
            | LinkTypes::PledgeById => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::DependencyToPledges
            | LinkTypes::ContributorToPledges
            | LinkTypes::AllPledges
            | LinkTypes::PledgeRateLimit
            | LinkTypes::PledgeById => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

// ── Unit Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_action() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::now(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex::from(0),
                0.into(),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    fn valid_pledge() -> ReciprocityPledge {
        ReciprocityPledge {
            id: "pledge-001".into(),
            dependency_id: "crate:serde:1.0".into(),
            contributor_did: "did:mycelix:corp456".into(),
            organization: Some("Acme Corp".into()),
            pledge_type: PledgeType::Financial,
            amount: Some(5000.0),
            currency: Some(Currency::USD),
            description: "Annual sponsorship for serde maintenance".into(),
            evidence_url: Some("https://opencollective.com/serde".into()),
            period: Some("2026-Q1".into()),
            pledged_at: Timestamp::now(),
            acknowledged: false,
        }
    }

    #[test]
    fn test_valid_pledge() {
        let result = validate_create_pledge(test_action(), valid_pledge()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_empty_pledge_id_rejected() {
        let mut p = valid_pledge();
        p.id = String::new();
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_empty_dependency_id_rejected() {
        let mut p = valid_pledge();
        p.dependency_id = String::new();
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_invalid_contributor_did_rejected() {
        let mut p = valid_pledge();
        p.contributor_did = "not-a-did".into();
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_negative_amount_rejected() {
        let mut p = valid_pledge();
        p.amount = Some(-100.0);
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_infinite_amount_rejected() {
        let mut p = valid_pledge();
        p.amount = Some(f64::INFINITY);
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_nan_amount_rejected() {
        let mut p = valid_pledge();
        p.amount = Some(f64::NAN);
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_financial_without_currency_rejected() {
        let mut p = valid_pledge();
        p.pledge_type = PledgeType::Financial;
        p.currency = None;
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_non_financial_without_currency_valid() {
        let mut p = valid_pledge();
        p.pledge_type = PledgeType::DeveloperTime;
        p.amount = None;
        p.currency = None;
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_description_too_long_rejected() {
        let mut p = valid_pledge();
        p.description = "x".repeat(1001);
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_neg_infinity_amount_rejected() {
        let mut p = valid_pledge();
        p.amount = Some(f64::NEG_INFINITY);
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_zero_amount_valid() {
        let mut p = valid_pledge();
        p.amount = Some(0.0);
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_description_exactly_1000_chars_valid() {
        let mut p = valid_pledge();
        p.description = "x".repeat(1000);
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_empty_dependency_id_pledge_rejected() {
        let mut p = valid_pledge();
        p.dependency_id = String::new();
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_pledge_types_roundtrip() {
        let types = vec![
            PledgeType::Financial,
            PledgeType::Compute,
            PledgeType::Bandwidth,
            PledgeType::DeveloperTime,
            PledgeType::QA,
            PledgeType::Documentation,
            PledgeType::Other,
        ];
        for pt in types {
            let json = serde_json::to_string(&pt).unwrap();
            let back: PledgeType = serde_json::from_str(&json).unwrap();
            assert_eq!(pt, back);
        }
    }

    #[test]
    fn test_currency_roundtrip() {
        let currencies = vec![
            Currency::USD,
            Currency::EUR,
            Currency::GBP,
            Currency::BTC,
            Currency::ETH,
            Currency::Other("DOGE".into()),
        ];
        for c in currencies {
            let json = serde_json::to_string(&c).unwrap();
            let back: Currency = serde_json::from_str(&json).unwrap();
            assert_eq!(c, back);
        }
    }

    // ── Update validation tests ─────────────────────────────────────

    fn test_update_action() -> Update {
        Update {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::now(),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0u8; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex::from(0),
                0.into(),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    #[test]
    fn test_valid_update_pledge() {
        let result = validate_update_pledge(
            test_update_action(),
            valid_pledge(),
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_empty_id_rejected() {
        let mut p = valid_pledge();
        p.id = String::new();
        let result = validate_update_pledge(
            test_update_action(),
            p,
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_update_invalid_did_rejected() {
        let mut p = valid_pledge();
        p.contributor_did = "bad".into();
        let result = validate_update_pledge(
            test_update_action(),
            p,
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_update_negative_amount_rejected() {
        let mut p = valid_pledge();
        p.amount = Some(-1.0);
        let result = validate_update_pledge(
            test_update_action(),
            p,
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_update_financial_without_currency_rejected() {
        let mut p = valid_pledge();
        p.pledge_type = PledgeType::Financial;
        p.currency = None;
        let result = validate_update_pledge(
            test_update_action(),
            p,
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_update_description_too_long_rejected() {
        let mut p = valid_pledge();
        p.description = "x".repeat(1001);
        let result = validate_update_pledge(
            test_update_action(),
            p,
            ActionHash::from_raw_36(vec![0u8; 36]),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── Additional edge cases ───────────────────────────────────────

    #[test]
    fn test_none_amount_valid() {
        let mut p = valid_pledge();
        p.amount = None;
        p.pledge_type = PledgeType::DeveloperTime;
        p.currency = None;
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_pledge_serde_roundtrip() {
        let p = valid_pledge();
        let json = serde_json::to_string(&p).unwrap();
        let back: ReciprocityPledge = serde_json::from_str(&json).unwrap();
        assert_eq!(p.id, back.id);
        assert_eq!(p.dependency_id, back.dependency_id);
        assert_eq!(p.pledge_type, back.pledge_type);
        assert_eq!(p.amount, back.amount);
        assert_eq!(p.currency, back.currency);
        assert_eq!(p.acknowledged, back.acknowledged);
    }

    #[test]
    fn test_anchor_serde_roundtrip() {
        let anchor = Anchor("all_pledges".into());
        let json = serde_json::to_string(&anchor).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(anchor, back);
    }

    #[test]
    fn test_description_exactly_1000_valid() {
        let mut p = valid_pledge();
        p.description = "x".repeat(1000);
        let result = validate_create_pledge(test_action(), p).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_all_pledge_types_valid_without_currency() {
        // Non-financial types should all be valid without currency
        let non_financial = vec![
            PledgeType::Compute,
            PledgeType::Bandwidth,
            PledgeType::DeveloperTime,
            PledgeType::QA,
            PledgeType::Documentation,
            PledgeType::Other,
        ];
        for pt in non_financial {
            let mut p = valid_pledge();
            p.pledge_type = pt.clone();
            p.currency = None;
            p.amount = None;
            let result = validate_create_pledge(test_action(), p).unwrap();
            assert_eq!(result, ValidateCallbackResult::Valid, "failed for {:?}", pt);
        }
    }
}
