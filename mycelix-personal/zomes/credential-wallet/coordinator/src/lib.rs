//! Credential Wallet Coordinator Zome
//!
//! Store, present, and verify verifiable credentials.
//! Includes K-Vector trust credential issuance, attestation workflow,
//! and selective disclosure. Credentials are stored privately;
//! presentation is done via personal_bridge.

use hdk::prelude::*;
use credential_wallet_integrity::*;
use personal_types::{
    AttestationStatus, CredentialType, KVectorComponent, TrustScoreRange, TrustTier,
};

// ============================================================================
// Generic Credential Operations
// ============================================================================

/// Store a new credential in the wallet.
#[hdk_extern]
pub fn store_credential(credential: StoredCredential) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::StoredCredential(credential.clone()))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToCredentials, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve stored credential".into()
    )))
}

/// Create a proof from a stored credential.
#[hdk_extern]
pub fn create_proof(proof: CredentialProof) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::CredentialProof(proof))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToProofs, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created proof".into()
    )))
}

/// Get all credentials stored by this agent.
#[hdk_extern]
pub fn get_my_credentials(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToCredentials)?,
        GetStrategy::Local,
    )?;
    let mut records = Vec::new();
    for link in links {
        let target = ActionHash::try_from(link.target.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Invalid link target: {:?}", e))))?;
        if let Some(record) = get(target, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Get credentials of a specific type.
#[hdk_extern]
pub fn get_credentials_by_type(credential_type: CredentialType) -> ExternResult<Vec<StoredCredential>> {
    let all = get_my_credentials(())?;
    let mut matched = Vec::new();
    for record in all {
        if let Some(cred) = record
            .entry()
            .to_app_option::<StoredCredential>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if cred.credential_type == credential_type && !cred.revoked {
                matched.push(cred);
            }
        }
    }
    Ok(matched)
}

/// Present a credential for cross-cluster verification.
///
/// Returns the credential data as a JSON string. The personal_bridge
/// wraps this into a CredentialPresentation with appropriate scope filtering.
#[hdk_extern]
pub fn present_credential(credential_type: CredentialType) -> ExternResult<String> {
    let creds = get_credentials_by_type(credential_type.clone())?;
    match creds.first() {
        Some(cred) => Ok(cred.credential_data.clone()),
        None => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "No active credential of type {:?} found",
            credential_type
        )))),
    }
}

// ============================================================================
// K-Vector Trust Credential Operations
// ============================================================================

/// Helper to get an anchor entry hash using cryptographic hashing.
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let hash = holo_hash::blake2b_256(anchor_str.as_bytes());
    Ok(EntryHash::from_raw_32(hash.to_vec()))
}

/// Issue a new trust credential.
///
/// Creates a trust credential with K-Vector commitment and ZKP proof.
/// The caller must be the issuer (verified via agent_info).
#[hdk_extern]
pub fn issue_trust_credential(input: IssueTrustCredentialInput) -> ExternResult<Record> {
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Subject DID must be 1-256 characters".into()
        )));
    }
    if input.issuer_did.is_empty() || input.issuer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Issuer DID must be 1-256 characters".into()
        )));
    }
    if input.trust_score_lower < 0.0
        || input.trust_score_upper > 1.0
        || input.trust_score_lower > input.trust_score_upper
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Trust scores must be in [0.0, 1.0] with lower <= upper".into()
        )));
    }

    let now = sys_time()?;
    let cred_id = format!("trust-cred:{}:{}", input.subject_did, now.as_micros());
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is the issuer
    let caller_did = format!("did:mycelix:{}", caller);
    if input.issuer_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the issuer can issue trust credentials".into()
        )));
    }

    // Determine trust tier from the proven range
    let mid_score = (input.trust_score_lower as f64 + input.trust_score_upper as f64) / 2.0;
    let trust_tier = TrustTier::from_score(mid_score);

    let credential = TrustCredential {
        id: cred_id,
        subject_did: input.subject_did.clone(),
        issuer_did: input.issuer_did.clone(),
        kvector_commitment: input.kvector_commitment,
        range_proof: input.range_proof,
        trust_score_range: TrustScoreRange {
            lower: input.trust_score_lower,
            upper: input.trust_score_upper,
        },
        trust_tier: trust_tier.clone(),
        issued_at: now,
        expires_at: input.expires_at,
        revoked: false,
        revocation_reason: None,
        revoked_at: None,
        supersedes: input.supersedes,
    };

    let action_hash = create_entry(&EntryTypes::TrustCredential(credential))?;

    // Link subject to credential
    let subject_hash = anchor_hash(&format!("subject:{}", input.subject_did))?;
    create_link(
        subject_hash,
        action_hash.clone(),
        LinkTypes::SubjectToTrustCredential,
        (),
    )?;

    // Link issuer to credential
    let issuer_hash = anchor_hash(&format!("issuer:{}", input.issuer_did))?;
    create_link(
        issuer_hash,
        action_hash.clone(),
        LinkTypes::IssuerToTrustCredential,
        (),
    )?;

    // Link tier to credential
    let tier_name = format!("{:?}", trust_tier);
    let tier_hash = anchor_hash(&format!("tier:{}", tier_name))?;
    create_link(
        tier_hash,
        action_hash.clone(),
        LinkTypes::TierToTrustCredential,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find credential".into()
    )))
}

/// Input for issuing a trust credential.
#[derive(Serialize, Deserialize, Debug)]
pub struct IssueTrustCredentialInput {
    pub subject_did: String,
    pub issuer_did: String,
    pub kvector_commitment: Vec<u8>,
    pub range_proof: Vec<u8>,
    pub trust_score_lower: f32,
    pub trust_score_upper: f32,
    pub expires_at: Option<Timestamp>,
    pub supersedes: Option<String>,
}

/// Self-issue a trust credential (subject = issuer).
///
/// An agent self-attests their K-Vector with proof. Self-attested
/// credentials have lower trust weight in governance decisions.
#[hdk_extern]
pub fn self_attest_trust(input: SelfAttestTrustInput) -> ExternResult<Record> {
    if input.self_did.is_empty() || input.self_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Self DID must be 1-256 characters".into()
        )));
    }
    if input.trust_score_lower < 0.0
        || input.trust_score_upper > 1.0
        || input.trust_score_lower > input.trust_score_upper
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Trust scores must be in [0.0, 1.0] with lower <= upper".into()
        )));
    }

    let self_did = input.self_did.clone();
    issue_trust_credential(IssueTrustCredentialInput {
        subject_did: self_did.clone(),
        issuer_did: self_did,
        kvector_commitment: input.kvector_commitment,
        range_proof: input.range_proof,
        trust_score_lower: input.trust_score_lower,
        trust_score_upper: input.trust_score_upper,
        expires_at: input.expires_at,
        supersedes: input.supersedes,
    })
}

/// Input for self-attesting trust.
#[derive(Serialize, Deserialize, Debug)]
pub struct SelfAttestTrustInput {
    pub self_did: String,
    pub kvector_commitment: Vec<u8>,
    pub range_proof: Vec<u8>,
    pub trust_score_lower: f32,
    pub trust_score_upper: f32,
    pub expires_at: Option<Timestamp>,
    pub supersedes: Option<String>,
}

/// Revoke a trust credential.
///
/// Only the credential issuer can revoke. Revocation is irreversible.
#[hdk_extern]
pub fn revoke_trust_credential(input: RevokeTrustCredentialInput) -> ExternResult<Record> {
    if input.credential_id.is_empty() || input.credential_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential ID must be 1-256 characters".into()
        )));
    }
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Subject DID must be 1-256 characters".into()
        )));
    }
    if input.reason.is_empty() || input.reason.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reason must be 1-2048 characters".into()
        )));
    }

    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("subject:{}", input.subject_did))?,
            LinkTypes::SubjectToTrustCredential,
        )?,
        GetStrategy::default(),
    )?;

    let mut target_hash = None;
    let mut target_cred: Option<TrustCredential> = None;

    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah.clone(), GetOptions::default())? {
            if let Some(cred) = record
                .entry()
                .to_app_option::<TrustCredential>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if cred.id == input.credential_id && !cred.revoked {
                    target_hash = Some(ah);
                    target_cred = Some(cred);
                    break;
                }
            }
        }
    }

    let (original_hash, mut credential) = match (target_hash, target_cred) {
        (Some(ah), Some(c)) => (ah, c),
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Active credential not found".into()
            )))
        }
    };

    // Verify caller is the issuer
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if credential.issuer_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the credential issuer can revoke it".into()
        )));
    }

    let now = sys_time()?;
    credential.revoked = true;
    credential.revocation_reason = Some(input.reason);
    credential.revoked_at = Some(now);

    let action_hash = update_entry(original_hash, &EntryTypes::TrustCredential(credential))?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find revoked credential".into()
    )))
}

/// Input for revoking a trust credential.
#[derive(Serialize, Deserialize, Debug)]
pub struct RevokeTrustCredentialInput {
    pub credential_id: String,
    pub subject_did: String,
    pub reason: String,
}

/// Create a trust presentation (selective disclosure).
///
/// Allows an agent to prove their trust tier without revealing exact scores.
#[hdk_extern]
pub fn create_trust_presentation(input: CreateTrustPresentationInput) -> ExternResult<Record> {
    if input.credential_id.is_empty() || input.credential_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential ID must be 1-256 characters".into()
        )));
    }
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Subject DID must be 1-256 characters".into()
        )));
    }
    if input.purpose.is_empty() || input.purpose.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Purpose must be 1-2048 characters".into()
        )));
    }

    let now = sys_time()?;
    let pres_id = format!("pres:{}:{}", input.subject_did, now.as_micros());
    let nonce = now.as_micros().to_le_bytes().to_vec();

    let presentation = TrustPresentation {
        id: pres_id,
        credential_id: input.credential_id.clone(),
        subject_did: input.subject_did,
        disclosed_tier: input.disclosed_tier,
        disclosed_range: input.disclose_range.then_some(input.trust_range),
        presentation_proof: input.presentation_proof,
        verifier_did: input.verifier_did,
        purpose: input.purpose,
        presented_at: now,
        nonce,
    };

    let action_hash = create_entry(&EntryTypes::TrustPresentation(presentation))?;

    let cred_hash = anchor_hash(&format!("credential:{}", input.credential_id))?;
    create_link(
        cred_hash,
        action_hash.clone(),
        LinkTypes::CredentialToPresentation,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find presentation".into()
    )))
}

/// Input for creating a trust presentation.
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateTrustPresentationInput {
    pub credential_id: String,
    pub subject_did: String,
    pub disclosed_tier: TrustTier,
    pub disclose_range: bool,
    pub trust_range: TrustScoreRange,
    pub presentation_proof: Vec<u8>,
    pub verifier_did: Option<String>,
    pub purpose: String,
}

// ============================================================================
// Attestation Request Workflow
// ============================================================================

/// Request attestation from another agent.
///
/// Creates an attestation request that the subject can fulfill or decline.
#[hdk_extern]
pub fn request_attestation(input: RequestAttestationInput) -> ExternResult<Record> {
    if input.requester_did.is_empty() || input.requester_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Requester DID must be 1-256 characters".into()
        )));
    }
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Subject DID must be 1-256 characters".into()
        )));
    }
    if input.purpose.is_empty() || input.purpose.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Purpose must be 1-2048 characters".into()
        )));
    }
    if input.components.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "At least one component is required".into()
        )));
    }
    if let Some(score) = input.min_trust_score {
        if !(0.0..=1.0).contains(&score) {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Min trust score must be between 0.0 and 1.0".into()
            )));
        }
    }

    let now = sys_time()?;
    let req_id = format!(
        "req:{}:{}:{}",
        input.requester_did,
        input.subject_did,
        now.as_micros()
    );

    let request = AttestationRequest {
        id: req_id,
        requester_did: input.requester_did,
        subject_did: input.subject_did.clone(),
        components: input.components,
        min_trust_score: input.min_trust_score,
        min_tier: input.min_tier,
        purpose: input.purpose,
        expires_at: input.expires_at,
        status: AttestationStatus::Pending,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::AttestationRequest(request))?;

    let subject_hash = anchor_hash(&format!("requests:{}", input.subject_did))?;
    create_link(
        subject_hash,
        action_hash.clone(),
        LinkTypes::SubjectToRequest,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find request".into()
    )))
}

/// Input for requesting attestation.
#[derive(Serialize, Deserialize, Debug)]
pub struct RequestAttestationInput {
    pub requester_did: String,
    pub subject_did: String,
    pub components: Vec<KVectorComponent>,
    pub min_trust_score: Option<f32>,
    pub min_tier: Option<TrustTier>,
    pub purpose: String,
    pub expires_at: Timestamp,
}

/// Fulfill an attestation request.
///
/// The subject responds by providing a K-Vector commitment and range proof
/// that meets the request's requirements. Issues a trust credential and
/// marks the request as Fulfilled.
#[hdk_extern]
pub fn fulfill_attestation(input: FulfillAttestationInput) -> ExternResult<FulfillAttestationResult> {
    if input.request_id.is_empty() || input.request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into()
        )));
    }
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Subject DID must be 1-256 characters".into()
        )));
    }
    if input.trust_score_lower < 0.0
        || input.trust_score_upper > 1.0
        || input.trust_score_lower > input.trust_score_upper
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Trust scores must be in [0.0, 1.0] with lower <= upper".into()
        )));
    }

    let now = sys_time()?;

    // Find the attestation request
    let subject_hash = anchor_hash(&format!("requests:{}", input.subject_did))?;
    let links = get_links(
        LinkQuery::try_new(subject_hash, LinkTypes::SubjectToRequest)?,
        GetStrategy::default(),
    )?;

    let mut request_hash: Option<ActionHash> = None;
    let mut request: Option<AttestationRequest> = None;

    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah.clone(), GetOptions::default())? {
            if let Some(req) = record
                .entry()
                .to_app_option::<AttestationRequest>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if req.id == input.request_id {
                    request_hash = Some(ah);
                    request = Some(req);
                    break;
                }
            }
        }
    }

    let (original_hash, req) = match (request_hash, request) {
        (Some(ah), Some(r)) => (ah, r),
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Attestation request not found".into()
            )))
        }
    };

    if req.status != AttestationStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Request is not pending (status: {:?})",
            req.status
        ))));
    }

    if req.subject_did != input.subject_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the attestation subject can fulfill the request".into()
        )));
    }

    // Check request hasn't expired
    if now > req.expires_at {
        let mut expired_req = req.clone();
        expired_req.status = AttestationStatus::Expired;
        update_entry(original_hash, &EntryTypes::AttestationRequest(expired_req))?;
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Attestation request has expired".into()
        )));
    }

    // Verify the provided proof meets request requirements
    let mid_score = (input.trust_score_lower as f64 + input.trust_score_upper as f64) / 2.0;
    let trust_tier = TrustTier::from_score(mid_score);

    if let Some(min_score) = req.min_trust_score {
        if input.trust_score_lower < min_score {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Trust score lower bound ({}) is below the requested minimum ({})",
                input.trust_score_lower, min_score
            ))));
        }
    }

    if let Some(ref min_tier) = req.min_tier {
        if mid_score < min_tier.min_score() {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Trust tier {:?} does not meet the requested minimum {:?}",
                trust_tier, min_tier
            ))));
        }
    }

    // Issue the trust credential (subject self-attests in response to request)
    let credential_record = issue_trust_credential(IssueTrustCredentialInput {
        subject_did: input.subject_did.clone(),
        issuer_did: input.subject_did.clone(),
        kvector_commitment: input.kvector_commitment,
        range_proof: input.range_proof,
        trust_score_lower: input.trust_score_lower,
        trust_score_upper: input.trust_score_upper,
        expires_at: input.expires_at,
        supersedes: None,
    })?;

    // Update request status to Fulfilled
    let mut fulfilled_req = req;
    fulfilled_req.status = AttestationStatus::Fulfilled;
    update_entry(
        original_hash,
        &EntryTypes::AttestationRequest(fulfilled_req),
    )?;

    Ok(FulfillAttestationResult {
        credential_record,
        request_id: input.request_id,
    })
}

/// Input for fulfilling an attestation request.
#[derive(Serialize, Deserialize, Debug)]
pub struct FulfillAttestationInput {
    pub request_id: String,
    pub subject_did: String,
    pub kvector_commitment: Vec<u8>,
    pub range_proof: Vec<u8>,
    pub trust_score_lower: f32,
    pub trust_score_upper: f32,
    pub expires_at: Option<Timestamp>,
}

/// Result of fulfilling an attestation.
#[derive(Serialize, Deserialize, Debug)]
pub struct FulfillAttestationResult {
    pub credential_record: Record,
    pub request_id: String,
}

/// Decline an attestation request.
///
/// Only the attestation subject can decline.
#[hdk_extern]
pub fn decline_attestation(input: DeclineAttestationInput) -> ExternResult<Record> {
    if input.request_id.is_empty() || input.request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into()
        )));
    }
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Subject DID must be 1-256 characters".into()
        )));
    }

    let subject_hash = anchor_hash(&format!("requests:{}", input.subject_did))?;
    let links = get_links(
        LinkQuery::try_new(subject_hash, LinkTypes::SubjectToRequest)?,
        GetStrategy::default(),
    )?;

    let mut request_hash: Option<ActionHash> = None;
    let mut request: Option<AttestationRequest> = None;

    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah.clone(), GetOptions::default())? {
            if let Some(req) = record
                .entry()
                .to_app_option::<AttestationRequest>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if req.id == input.request_id && req.status == AttestationStatus::Pending {
                    request_hash = Some(ah);
                    request = Some(req);
                    break;
                }
            }
        }
    }

    let (original_hash, req) = match (request_hash, request) {
        (Some(ah), Some(r)) => (ah, r),
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Pending attestation request not found".into()
            )))
        }
    };

    if req.subject_did != input.subject_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the attestation subject can decline the request".into()
        )));
    }

    let mut declined_req = req;
    declined_req.status = AttestationStatus::Declined;
    let action_hash = update_entry(
        original_hash,
        &EntryTypes::AttestationRequest(declined_req),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated request".into()
    )))
}

/// Input for declining an attestation.
#[derive(Serialize, Deserialize, Debug)]
pub struct DeclineAttestationInput {
    pub request_id: String,
    pub subject_did: String,
}

// ============================================================================
// Trust Credential Queries
// ============================================================================

/// Get pending attestation requests for this agent.
#[hdk_extern]
pub fn get_pending_requests(subject_did: String) -> ExternResult<Vec<Record>> {
    if subject_did.is_empty() || subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Subject DID must be 1-256 characters".into()
        )));
    }
    let now = sys_time()?;
    let subject_hash = anchor_hash(&format!("requests:{}", subject_did))?;
    let links = get_links(
        LinkQuery::try_new(subject_hash, LinkTypes::SubjectToRequest)?,
        GetStrategy::default(),
    )?;

    let mut pending = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Some(req) = record
                .entry()
                .to_app_option::<AttestationRequest>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if req.status == AttestationStatus::Pending && now < req.expires_at {
                    pending.push(record);
                }
            }
        }
    }

    Ok(pending)
}

/// Get non-revoked trust credentials for a subject.
#[hdk_extern]
pub fn get_trust_credentials(subject_did: String) -> ExternResult<Vec<Record>> {
    if subject_did.is_empty() || subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Subject DID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("subject:{}", subject_did))?,
            LinkTypes::SubjectToTrustCredential,
        )?,
        GetStrategy::default(),
    )?;

    let mut credentials = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Some(cred) = record
                .entry()
                .to_app_option::<TrustCredential>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if !cred.revoked {
                    credentials.push(record);
                }
            }
        }
    }

    Ok(credentials)
}

/// Get trust credentials by tier.
#[hdk_extern]
pub fn get_trust_credentials_by_tier(tier: TrustTier) -> ExternResult<Vec<Record>> {
    let tier_name = format!("{:?}", tier);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("tier:{}", tier_name))?,
            LinkTypes::TierToTrustCredential,
        )?,
        GetStrategy::default(),
    )?;

    let mut credentials = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Some(cred) = record
                .entry()
                .to_app_option::<TrustCredential>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if !cred.revoked {
                    credentials.push(record);
                }
            }
        }
    }

    Ok(credentials)
}

/// Verify a trust credential's on-chain properties.
///
/// Checks commitment format, tier consistency, revocation, and expiration.
/// Full STARK proof verification is done off-chain due to WASM constraints.
#[hdk_extern]
pub fn verify_trust_credential(credential_id: String) -> ExternResult<VerificationResult> {
    if credential_id.is_empty() || credential_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential ID must be 1-256 characters".into()
        )));
    }

    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::TrustCredential,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let mut found_cred: Option<TrustCredential> = None;
    for record in records {
        if let Some(cred) = record
            .entry()
            .to_app_option::<TrustCredential>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if cred.id == credential_id {
                found_cred = Some(cred);
                break;
            }
        }
    }

    let cred = match found_cred {
        Some(c) => c,
        None => {
            return Ok(VerificationResult {
                credential_id,
                commitment_valid: false,
                tier_consistent: false,
                not_revoked: false,
                not_expired: false,
                proof_format_valid: false,
                message: "Credential not found".to_string(),
            });
        }
    };

    let not_revoked = !cred.revoked;

    let now = sys_time()?;
    let not_expired = match cred.expires_at {
        Some(exp) => now < exp,
        None => true,
    };

    let (commitment_valid, range_valid, tier_consistent, proof_format_valid) =
        verify_credential_pure(
            &cred.kvector_commitment,
            &cred.trust_score_range,
            &cred.trust_tier,
            &cred.range_proof,
        );

    let all_valid = not_revoked
        && not_expired
        && commitment_valid
        && range_valid
        && tier_consistent
        && proof_format_valid;
    let message = if all_valid {
        "On-chain verification passed. Run off-chain STARK verification for full proof.".into()
    } else {
        let mut issues = Vec::new();
        if !not_revoked {
            issues.push("credential revoked");
        }
        if !not_expired {
            issues.push("credential expired");
        }
        if !commitment_valid {
            issues.push("invalid commitment");
        }
        if !range_valid {
            issues.push("malformed score range");
        }
        if !tier_consistent {
            issues.push("tier/range mismatch");
        }
        if !proof_format_valid {
            issues.push("empty proof");
        }
        format!("Verification failed: {}", issues.join(", "))
    };

    Ok(VerificationResult {
        credential_id,
        commitment_valid,
        tier_consistent,
        not_revoked,
        not_expired,
        proof_format_valid,
        message,
    })
}

/// Result of credential verification.
#[derive(Serialize, Deserialize, Debug)]
pub struct VerificationResult {
    pub credential_id: String,
    pub commitment_valid: bool,
    pub tier_consistent: bool,
    pub not_revoked: bool,
    pub not_expired: bool,
    pub proof_format_valid: bool,
    pub message: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_types_all_exist() {
        let _cred = UnitEntryTypes::StoredCredential;
        let _proof = UnitEntryTypes::CredentialProof;
        let _trust = UnitEntryTypes::TrustCredential;
        let _req = UnitEntryTypes::AttestationRequest;
        let _pres = UnitEntryTypes::TrustPresentation;
    }

    #[test]
    fn link_types_all_exist() {
        let _creds = LinkTypes::AgentToCredentials;
        let _proofs = LinkTypes::AgentToProofs;
        let _type_to_cred = LinkTypes::CredentialTypeToCredential;
        let _subj = LinkTypes::SubjectToTrustCredential;
        let _issuer = LinkTypes::IssuerToTrustCredential;
        let _req = LinkTypes::SubjectToRequest;
        let _pres = LinkTypes::CredentialToPresentation;
        let _tier = LinkTypes::TierToTrustCredential;
    }

    #[test]
    fn verification_result_serde_roundtrip() {
        let vr = VerificationResult {
            credential_id: "test-cred".into(),
            commitment_valid: true,
            tier_consistent: true,
            not_revoked: true,
            not_expired: true,
            proof_format_valid: true,
            message: "OK".into(),
        };
        let json = serde_json::to_string(&vr).unwrap();
        let back: VerificationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.credential_id, "test-cred");
        assert!(back.commitment_valid);
    }

    #[test]
    fn issue_input_serde_roundtrip() {
        let input = IssueTrustCredentialInput {
            subject_did: "did:mycelix:alice".into(),
            issuer_did: "did:mycelix:bob".into(),
            kvector_commitment: vec![1u8; 32],
            range_proof: vec![1, 2, 3],
            trust_score_lower: 0.4,
            trust_score_upper: 0.6,
            expires_at: None,
            supersedes: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: IssueTrustCredentialInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.subject_did, "did:mycelix:alice");
    }

    #[test]
    fn fulfill_result_serde() {
        // Just check the struct compiles with proper derive
        let _type_check: fn(FulfillAttestationResult) = |r| {
            let _ = serde_json::to_string(&r);
        };
    }
}
