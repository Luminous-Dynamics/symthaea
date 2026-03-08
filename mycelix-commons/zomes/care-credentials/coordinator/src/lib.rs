//! Credentials Coordinator Zome
//! Business logic for credential issuance, verification, and reference management.

use care_credentials_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
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

/// Issue a new credential to a holder
#[hdk_extern]
pub fn issue_credential(credential: CareCredential) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "issue_credential")?;
    let action_hash = create_entry(&EntryTypes::CareCredential(credential.clone()))?;

    // Link holder to credential
    let holder_anchor = ensure_anchor(&format!("agent_credentials:{}", credential.holder))?;
    create_link(
        holder_anchor,
        action_hash.clone(),
        LinkTypes::AgentToCredential,
        (),
    )?;

    // Link credential type to credential
    let type_anchor = ensure_anchor(&format!(
        "cred_type:{}",
        credential.credential_type.anchor_key()
    ))?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::TypeToCredential,
        (),
    )?;

    // If verified, link to all verified credentials
    if credential.verified {
        let verified_anchor = ensure_anchor("all_verified_credentials")?;
        create_link(
            verified_anchor,
            action_hash.clone(),
            LinkTypes::AllVerifiedCredentials,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created credential".into()
    )))
}

/// Input for verifying a credential
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyCredentialInput {
    pub credential_hash: ActionHash,
}

/// Mark a credential as verified
#[hdk_extern]
pub fn verify_credential(input: VerifyCredentialInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "verify_credential")?;
    let record = get(input.credential_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Credential not found".into())
    ))?;

    let mut credential: CareCredential = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid credential entry".into()
        )))?;

    if credential.verified {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential is already verified".into()
        )));
    }

    credential.verified = true;

    let updated_hash = update_entry(
        input.credential_hash,
        &EntryTypes::CareCredential(credential.clone()),
    )?;

    // Link to verified credentials
    let verified_anchor = ensure_anchor("all_verified_credentials")?;
    create_link(
        verified_anchor,
        updated_hash.clone(),
        LinkTypes::AllVerifiedCredentials,
        (),
    )?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated credential".into()
    )))
}

/// Add a reference for a care provider
#[hdk_extern]
pub fn add_reference(reference: CareReference) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "add_reference")?;
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is the one giving the reference
    if caller != reference.from_recipient {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only add references from your own agent".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::CareReference(reference.clone()))?;

    // Link provider to received reference
    let provider_anchor = ensure_anchor(&format!("agent_references:{}", reference.provider))?;
    create_link(
        provider_anchor,
        action_hash.clone(),
        LinkTypes::AgentToReference,
        (),
    )?;

    // Link recipient to given reference
    let giver_anchor = ensure_anchor(&format!("agent_given_refs:{}", reference.from_recipient))?;
    create_link(
        giver_anchor,
        action_hash.clone(),
        LinkTypes::AgentGivenReferences,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created reference".into()
    )))
}

/// Get all credentials for a provider
#[hdk_extern]
pub fn get_provider_credentials(provider: AgentPubKey) -> ExternResult<Vec<Record>> {
    let holder_anchor = anchor_hash(&format!("agent_credentials:{}", provider))?;
    let links = get_links(
        LinkQuery::try_new(holder_anchor, LinkTypes::AgentToCredential)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all references for a provider
#[hdk_extern]
pub fn get_provider_references(provider: AgentPubKey) -> ExternResult<Vec<Record>> {
    let provider_anchor = anchor_hash(&format!("agent_references:{}", provider))?;
    let links = get_links(
        LinkQuery::try_new(provider_anchor, LinkTypes::AgentToReference)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get my credentials
#[hdk_extern]
pub fn get_my_credentials(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    get_provider_credentials(caller)
}

/// Get my references (received)
#[hdk_extern]
pub fn get_my_references(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    get_provider_references(caller)
}

/// Get credentials by type
#[hdk_extern]
pub fn get_credentials_by_type(credential_type: CredentialType) -> ExternResult<Vec<Record>> {
    let type_anchor = anchor_hash(&format!("cred_type:{}", credential_type.anchor_key()))?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::TypeToCredential)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Provider reputation summary
#[derive(Serialize, Deserialize, Debug)]
pub struct ProviderReputation {
    pub provider: AgentPubKey,
    pub credential_count: u32,
    pub verified_credential_count: u32,
    pub reference_count: u32,
    pub average_rating: f32,
}

/// Get a provider's reputation summary
#[hdk_extern]
pub fn get_provider_reputation(provider: AgentPubKey) -> ExternResult<ProviderReputation> {
    let credentials = get_provider_credentials(provider.clone())?;
    let references = get_provider_references(provider.clone())?;

    let mut verified_count = 0u32;
    for record in &credentials {
        if let Some(cred) = record
            .entry()
            .to_app_option::<CareCredential>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if cred.verified {
                verified_count += 1;
            }
        }
    }

    let mut total_rating = 0u32;
    let mut rating_count = 0u32;
    for record in &references {
        if let Some(reference) = record
            .entry()
            .to_app_option::<CareReference>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            total_rating += reference.rating as u32;
            rating_count += 1;
        }
    }

    let average_rating = if rating_count > 0 {
        total_rating as f32 / rating_count as f32
    } else {
        0.0
    };

    Ok(ProviderReputation {
        provider,
        credential_count: credentials.len() as u32,
        verified_credential_count: verified_count,
        reference_count: references.len() as u32,
        average_rating,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_agent_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![2u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    // ── VerifyCredentialInput serde roundtrip ────────────────────────────

    #[test]
    fn verify_credential_input_serde_roundtrip() {
        let input = VerifyCredentialInput {
            credential_hash: fake_action_hash(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VerifyCredentialInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.credential_hash, input.credential_hash);
    }

    // ── ProviderReputation serde roundtrip ───────────────────────────────

    #[test]
    fn provider_reputation_serde_roundtrip() {
        let rep = ProviderReputation {
            provider: fake_agent(),
            credential_count: 5,
            verified_credential_count: 3,
            reference_count: 10,
            average_rating: 4.2,
        };
        let json = serde_json::to_string(&rep).unwrap();
        let decoded: ProviderReputation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.provider, rep.provider);
        assert_eq!(decoded.credential_count, 5);
        assert_eq!(decoded.verified_credential_count, 3);
        assert_eq!(decoded.reference_count, 10);
        assert!((decoded.average_rating - 4.2).abs() < f32::EPSILON);
    }

    #[test]
    fn provider_reputation_zero_values_roundtrip() {
        let rep = ProviderReputation {
            provider: fake_agent(),
            credential_count: 0,
            verified_credential_count: 0,
            reference_count: 0,
            average_rating: 0.0,
        };
        let json = serde_json::to_string(&rep).unwrap();
        let decoded: ProviderReputation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.credential_count, 0);
        assert_eq!(decoded.average_rating, 0.0);
    }

    #[test]
    fn provider_reputation_u32_max_counts() {
        let rep = ProviderReputation {
            provider: fake_agent(),
            credential_count: u32::MAX,
            verified_credential_count: u32::MAX,
            reference_count: u32::MAX,
            average_rating: 5.0,
        };
        let json = serde_json::to_string(&rep).unwrap();
        let decoded: ProviderReputation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.credential_count, u32::MAX);
        assert_eq!(decoded.verified_credential_count, u32::MAX);
        assert_eq!(decoded.reference_count, u32::MAX);
    }

    #[test]
    fn provider_reputation_fractional_rating() {
        let rep = ProviderReputation {
            provider: fake_agent(),
            credential_count: 1,
            verified_credential_count: 1,
            reference_count: 3,
            average_rating: 3.333_333_3,
        };
        let json = serde_json::to_string(&rep).unwrap();
        let decoded: ProviderReputation = serde_json::from_str(&json).unwrap();
        assert!((decoded.average_rating - 3.333_333_3).abs() < 1e-5);
    }

    // ── CredentialType serde roundtrip (all variants) ────────────────────

    #[test]
    fn credential_type_serde_all_variants() {
        let types = vec![
            CredentialType::FirstAid,
            CredentialType::CPR,
            CredentialType::ChildcareTraining,
            CredentialType::ElderCare,
            CredentialType::MentalHealthFirstAid,
            CredentialType::BackgroundCheck,
            CredentialType::DrivingLicense,
            CredentialType::SpecialNeeds,
            CredentialType::Other("Yoga Instructor".to_string()),
        ];
        for ct in &types {
            let json = serde_json::to_string(ct).unwrap();
            let decoded: CredentialType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, ct);
        }
    }

    // ── CredentialType::anchor_key pure function tests ───────────────────

    #[test]
    fn credential_type_anchor_key_known_variants() {
        assert_eq!(CredentialType::FirstAid.anchor_key(), "firstaid");
        assert_eq!(CredentialType::CPR.anchor_key(), "cpr");
        assert_eq!(
            CredentialType::ChildcareTraining.anchor_key(),
            "childcare_training"
        );
        assert_eq!(CredentialType::ElderCare.anchor_key(), "eldercare");
        assert_eq!(CredentialType::MentalHealthFirstAid.anchor_key(), "mhfa");
        assert_eq!(
            CredentialType::BackgroundCheck.anchor_key(),
            "background_check"
        );
        assert_eq!(CredentialType::DrivingLicense.anchor_key(), "driving");
        assert_eq!(CredentialType::SpecialNeeds.anchor_key(), "special_needs");
    }

    #[test]
    fn credential_type_anchor_key_other_lowercases_and_replaces_spaces() {
        let ct = CredentialType::Other("Pet First Aid".to_string());
        assert_eq!(ct.anchor_key(), "other_pet_first_aid");
    }

    #[test]
    fn credential_type_anchor_key_other_empty_string() {
        let ct = CredentialType::Other(String::new());
        assert_eq!(ct.anchor_key(), "other_");
    }

    #[test]
    fn credential_type_anchor_key_other_unicode() {
        let ct = CredentialType::Other("\u{00C9}ducation".to_string());
        let key = ct.anchor_key();
        assert!(key.starts_with("other_"));
        // lowercase of '\u{00C9}' is '\u{00E9}'
        assert!(key.contains('\u{00E9}'));
    }

    #[test]
    fn credential_type_anchor_key_other_multiple_spaces() {
        let ct = CredentialType::Other("A B  C   D".to_string());
        assert_eq!(ct.anchor_key(), "other_a_b__c___d");
    }

    // ── CareCredential serde roundtrip ──────────────────────────────────

    #[test]
    fn care_credential_serde_roundtrip() {
        let cred = CareCredential {
            holder: fake_agent(),
            credential_type: CredentialType::CPR,
            issuer: "Red Cross".to_string(),
            issued_at: Timestamp::from_micros(1000),
            expires_at: Some(Timestamp::from_micros(2000)),
            verified: true,
            metadata: "{}".to_string(),
        };
        let json = serde_json::to_string(&cred).unwrap();
        let decoded: CareCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, cred);
    }

    #[test]
    fn care_credential_no_expiry_serde_roundtrip() {
        let cred = CareCredential {
            holder: fake_agent(),
            credential_type: CredentialType::BackgroundCheck,
            issuer: "County Clerk".to_string(),
            issued_at: Timestamp::from_micros(500),
            expires_at: None,
            verified: false,
            metadata: String::new(),
        };
        let json = serde_json::to_string(&cred).unwrap();
        let decoded: CareCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.expires_at, None);
        assert_eq!(decoded.verified, false);
    }

    #[test]
    fn care_credential_other_type_roundtrip() {
        let cred = CareCredential {
            holder: fake_agent(),
            credential_type: CredentialType::Other("Doula Certification".to_string()),
            issuer: "DONA International".to_string(),
            issued_at: Timestamp::from_micros(0),
            expires_at: None,
            verified: true,
            metadata: r#"{"level":"advanced"}"#.to_string(),
        };
        let json = serde_json::to_string(&cred).unwrap();
        let decoded: CareCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.credential_type,
            CredentialType::Other("Doula Certification".to_string())
        );
        assert!(decoded.metadata.contains("advanced"));
    }

    #[test]
    fn care_credential_unicode_issuer() {
        let cred = CareCredential {
            holder: fake_agent(),
            credential_type: CredentialType::FirstAid,
            issuer: "\u{7D05}\u{5341}\u{5B57}\u{4F1A}".to_string(),
            issued_at: Timestamp::from_micros(0),
            expires_at: None,
            verified: false,
            metadata: "{}".to_string(),
        };
        let json = serde_json::to_string(&cred).unwrap();
        let decoded: CareCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.issuer, "\u{7D05}\u{5341}\u{5B57}\u{4F1A}");
    }

    #[test]
    fn care_credential_clone_equals() {
        let cred = CareCredential {
            holder: fake_agent(),
            credential_type: CredentialType::ElderCare,
            issuer: "State Board".to_string(),
            issued_at: Timestamp::from_micros(100),
            expires_at: Some(Timestamp::from_micros(200)),
            verified: true,
            metadata: "{}".to_string(),
        };
        let cloned = cred.clone();
        assert_eq!(cred, cloned);
    }

    // ── CareReference serde roundtrip ───────────────────────────────────

    #[test]
    fn care_reference_serde_roundtrip() {
        let reference = CareReference {
            provider: fake_agent(),
            from_recipient: fake_agent_2(),
            rating: 5,
            comment: "Excellent caregiver".to_string(),
            care_type: "Eldercare".to_string(),
            created_at: Timestamp::from_micros(3000),
        };
        let json = serde_json::to_string(&reference).unwrap();
        let decoded: CareReference = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, reference);
    }

    #[test]
    fn care_reference_min_rating() {
        let reference = CareReference {
            provider: fake_agent(),
            from_recipient: fake_agent_2(),
            rating: 1,
            comment: "Below expectations".to_string(),
            care_type: "Childcare".to_string(),
            created_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&reference).unwrap();
        let decoded: CareReference = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.rating, 1);
    }

    #[test]
    fn care_reference_boundary_ratings() {
        for rating in [1u8, 2, 3, 4, 5] {
            let reference = CareReference {
                provider: fake_agent(),
                from_recipient: fake_agent_2(),
                rating,
                comment: format!("Rating {}", rating),
                care_type: "General".to_string(),
                created_at: Timestamp::from_micros(0),
            };
            let json = serde_json::to_string(&reference).unwrap();
            let decoded: CareReference = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.rating, rating);
        }
    }

    #[test]
    fn care_reference_unicode_comment() {
        let reference = CareReference {
            provider: fake_agent(),
            from_recipient: fake_agent_2(),
            rating: 4,
            comment: "\u{2B50}\u{2B50}\u{2B50}\u{2B50} Great care!".to_string(),
            care_type: "\u{5065}\u{5EB7}".to_string(),
            created_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&reference).unwrap();
        let decoded: CareReference = serde_json::from_str(&json).unwrap();
        assert!(decoded.comment.contains('\u{2B50}'));
        assert_eq!(decoded.care_type, "\u{5065}\u{5EB7}");
    }

    #[test]
    fn care_reference_clone_equals() {
        let reference = CareReference {
            provider: fake_agent(),
            from_recipient: fake_agent_2(),
            rating: 3,
            comment: "Decent".to_string(),
            care_type: "General".to_string(),
            created_at: Timestamp::from_micros(0),
        };
        let cloned = reference.clone();
        assert_eq!(reference, cloned);
    }

    #[test]
    fn care_reference_empty_comment_and_care_type() {
        let reference = CareReference {
            provider: fake_agent(),
            from_recipient: fake_agent_2(),
            rating: 3,
            comment: String::new(),
            care_type: String::new(),
            created_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&reference).unwrap();
        let decoded: CareReference = serde_json::from_str(&json).unwrap();
        assert!(decoded.comment.is_empty());
        assert!(decoded.care_type.is_empty());
    }
}
