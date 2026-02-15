//! Trust Credential Coordinator Zome
//!
//! Business logic for K-Vector trust credentials with ZKP proofs.
//!
//! Workflow:
//! 1. Agent generates K-Vector off-chain using MATL formula
//! 2. Agent creates KVectorWitness and generates STARK proof (off-chain)
//! 3. Agent submits credential with commitment + proof to this zome
//! 4. Verifiers can check credential validity and trust tier
//! 5. Selective disclosure allows proving tier without revealing score

use hdk::prelude::*;
use trust_credential_integrity::*;

/// API version for cross-zome compatibility detection.
/// Increment when making breaking changes to extern signatures or types.
const API_VERSION: u16 = 1;

/// Returns the API version of this coordinator zome.
#[hdk_extern]
pub fn get_api_version(_: ()) -> ExternResult<u16> {
    Ok(API_VERSION)
}

/// Helper to get an anchor entry hash using cryptographic hashing
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let hash = holo_hash::blake2b_256(anchor_str.as_bytes());
    Ok(EntryHash::from_raw_32(hash.to_vec()))
}

/// Issue a new trust credential
///
/// Creates a trust credential with K-Vector commitment and ZKP proof.
/// The issuer vouches for the subject's trust properties.
#[hdk_extern]
pub fn issue_trust_credential(input: IssueTrustCredentialInput) -> ExternResult<Record> {
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Subject DID must be 1-256 characters".into())));
    }
    if input.issuer_did.is_empty() || input.issuer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Issuer DID must be 1-256 characters".into())));
    }
    if input.trust_score_lower < 0.0 || input.trust_score_upper > 1.0 || input.trust_score_lower > input.trust_score_upper {
        return Err(wasm_error!(WasmErrorInner::Guest("Trust scores must be in [0.0, 1.0] with lower <= upper".into())));
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
        id: cred_id.clone(),
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
        LinkTypes::SubjectToCredential,
        (),
    )?;

    // Link issuer to credential
    let issuer_hash = anchor_hash(&format!("issuer:{}", input.issuer_did))?;
    create_link(
        issuer_hash,
        action_hash.clone(),
        LinkTypes::IssuerToCredential,
        (),
    )?;

    // Link tier to credential
    let tier_name = format!("{:?}", trust_tier);
    let tier_hash = anchor_hash(&format!("tier:{}", tier_name))?;
    create_link(
        tier_hash,
        action_hash.clone(),
        LinkTypes::TierToCredential,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find credential".into()
        )))
}

/// Input for issuing a trust credential
#[derive(Serialize, Deserialize, Debug)]
pub struct IssueTrustCredentialInput {
    pub subject_did: String,
    pub issuer_did: String,
    /// K-Vector commitment (32 bytes from KVectorWitness.commitment())
    pub kvector_commitment: Vec<u8>,
    /// STARK range proof (serialized from KVectorRangeProof)
    pub range_proof: Vec<u8>,
    /// Proven lower bound of trust score
    pub trust_score_lower: f32,
    /// Proven upper bound of trust score
    pub trust_score_upper: f32,
    /// Optional expiration
    pub expires_at: Option<Timestamp>,
    /// Optional previous credential this supersedes
    pub supersedes: Option<String>,
}

/// Self-issue a trust credential
///
/// An agent can self-attest their K-Vector with proof.
/// Self-attested credentials have lower trust weight.
#[hdk_extern]
pub fn self_attest_trust(input: SelfAttestTrustInput) -> ExternResult<Record> {
    if input.self_did.is_empty() || input.self_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Self DID must be 1-256 characters".into())));
    }
    if input.trust_score_lower < 0.0 || input.trust_score_upper > 1.0 || input.trust_score_lower > input.trust_score_upper {
        return Err(wasm_error!(WasmErrorInner::Guest("Trust scores must be in [0.0, 1.0] with lower <= upper".into())));
    }
    // Self-attestation uses same issuance logic

    // For self-attestation, subject and issuer are the same
    let self_did = input.self_did.clone();

    let issue_input = IssueTrustCredentialInput {
        subject_did: self_did.clone(),
        issuer_did: self_did,
        kvector_commitment: input.kvector_commitment,
        range_proof: input.range_proof,
        trust_score_lower: input.trust_score_lower,
        trust_score_upper: input.trust_score_upper,
        expires_at: input.expires_at,
        supersedes: input.supersedes,
    };

    issue_trust_credential(issue_input)
}

/// Input for self-attesting trust
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

/// Revoke a trust credential
#[hdk_extern]
pub fn revoke_credential(input: RevokeCredentialInput) -> ExternResult<Record> {
    if input.credential_id.is_empty() || input.credential_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Credential ID must be 1-256 characters".into())));
    }
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Subject DID must be 1-256 characters".into())));
    }
    if input.reason.is_empty() || input.reason.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest("Reason must be 1-2048 characters".into())));
    }
    // Get the credential
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("subject:{}", input.subject_did))?,
            LinkTypes::SubjectToCredential,
        )?,
        GetStrategy::default(),
    )?;

    let mut target_hash = None;
    let mut target_cred: Option<TrustCredential> = None;

    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah.clone(), GetOptions::default())? {
            if let Some(cred) = record.entry().to_app_option::<TrustCredential>()
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
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            "Active credential not found".into()
        ))),
    };

    // Verify caller is the issuer of this credential
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if credential.issuer_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the credential issuer can revoke it".into()
        )));
    }

    // Update credential as revoked
    let now = sys_time()?;
    credential.revoked = true;
    credential.revocation_reason = Some(input.reason);
    credential.revoked_at = Some(now);

    let action_hash = update_entry(original_hash, &EntryTypes::TrustCredential(credential))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find revoked credential".into()
        )))
}

/// Input for revoking a credential
#[derive(Serialize, Deserialize, Debug)]
pub struct RevokeCredentialInput {
    pub credential_id: String,
    pub subject_did: String,
    pub reason: String,
}

/// Create a trust presentation (selective disclosure)
#[hdk_extern]
pub fn create_presentation(input: CreatePresentationInput) -> ExternResult<Record> {
    if input.credential_id.is_empty() || input.credential_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Credential ID must be 1-256 characters".into())));
    }
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Subject DID must be 1-256 characters".into())));
    }
    if input.purpose.is_empty() || input.purpose.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest("Purpose must be 1-2048 characters".into())));
    }
    let now = sys_time()?;
    let pres_id = format!("pres:{}:{}", input.subject_did, now.as_micros());

    // Generate a nonce for replay protection
    let nonce = now.as_micros().to_le_bytes().to_vec();

    let presentation = TrustPresentation {
        id: pres_id.clone(),
        credential_id: input.credential_id.clone(),
        subject_did: input.subject_did.clone(),
        disclosed_tier: input.disclosed_tier,
        disclosed_range: input.disclose_range.then_some(input.trust_range),
        presentation_proof: input.presentation_proof,
        verifier_did: input.verifier_did,
        purpose: input.purpose,
        presented_at: now,
        nonce,
    };

    let action_hash = create_entry(&EntryTypes::TrustPresentation(presentation))?;

    // Link credential to presentation
    let cred_hash = anchor_hash(&format!("credential:{}", input.credential_id))?;
    create_link(
        cred_hash,
        action_hash.clone(),
        LinkTypes::CredentialToPresentation,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find presentation".into()
        )))
}

/// Input for creating a presentation
#[derive(Serialize, Deserialize, Debug)]
pub struct CreatePresentationInput {
    pub credential_id: String,
    pub subject_did: String,
    pub disclosed_tier: TrustTier,
    pub disclose_range: bool,
    pub trust_range: TrustScoreRange,
    pub presentation_proof: Vec<u8>,
    pub verifier_did: Option<String>,
    pub purpose: String,
}

/// Request attestation from another agent
#[hdk_extern]
pub fn request_attestation(input: RequestAttestationInput) -> ExternResult<Record> {
    if input.requester_did.is_empty() || input.requester_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Requester DID must be 1-256 characters".into())));
    }
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Subject DID must be 1-256 characters".into())));
    }
    if input.purpose.is_empty() || input.purpose.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest("Purpose must be 1-2048 characters".into())));
    }
    if input.components.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("At least one component is required".into())));
    }
    if let Some(score) = input.min_trust_score {
        if !(0.0..=1.0).contains(&score) {
            return Err(wasm_error!(WasmErrorInner::Guest("Min trust score must be between 0.0 and 1.0".into())));
        }
    }
    let now = sys_time()?;
    let req_id = format!("req:{}:{}:{}", input.requester_did, input.subject_did, now.as_micros());

    let request = AttestationRequest {
        id: req_id.clone(),
        requester_did: input.requester_did.clone(),
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

    // Link subject to request
    let subject_hash = anchor_hash(&format!("requests:{}", input.subject_did))?;
    create_link(
        subject_hash,
        action_hash.clone(),
        LinkTypes::SubjectToRequest,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find request".into()
        )))
}

/// Input for requesting attestation
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

/// Fulfill an attestation request
///
/// The subject responds to an attestation request by providing a K-Vector
/// commitment and range proof that meets the request's requirements.
/// This issues a trust credential and marks the request as Fulfilled.
#[hdk_extern]
pub fn fulfill_attestation(input: FulfillAttestationInput) -> ExternResult<FulfillAttestationResult> {
    if input.request_id.is_empty() || input.request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Request ID must be 1-256 characters".into())));
    }
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Subject DID must be 1-256 characters".into())));
    }
    if input.trust_score_lower < 0.0 || input.trust_score_upper > 1.0 || input.trust_score_lower > input.trust_score_upper {
        return Err(wasm_error!(WasmErrorInner::Guest("Trust scores must be in [0.0, 1.0] with lower <= upper".into())));
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
            if let Some(req) = record.entry().to_app_option::<AttestationRequest>()
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
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            "Attestation request not found".into()
        ))),
    };

    // Verify request is still pending
    if req.status != AttestationStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Request is not pending (status: {:?})", req.status)
        )));
    }

    // Verify the fulfiller matches the subject
    if req.subject_did != input.subject_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the attestation subject can fulfill the request".into()
        )));
    }

    // Check request hasn't expired
    if now > req.expires_at {
        // Update request to expired
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
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Trust score lower bound ({}) is below the requested minimum ({})",
                    input.trust_score_lower, min_score)
            )));
        }
    }

    if let Some(ref min_tier) = req.min_tier {
        if (mid_score) < min_tier.min_score() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Trust tier {:?} does not meet the requested minimum {:?}",
                    trust_tier, min_tier)
            )));
        }
    }

    // Issue the trust credential (subject self-attests in response to request)
    let credential_record = issue_trust_credential(IssueTrustCredentialInput {
        subject_did: input.subject_did.clone(),
        issuer_did: input.subject_did.clone(), // Self-attestation
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
    update_entry(original_hash, &EntryTypes::AttestationRequest(fulfilled_req))?;

    Ok(FulfillAttestationResult {
        credential_record,
        request_id: input.request_id,
    })
}

/// Input for fulfilling an attestation request
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

/// Result of fulfilling an attestation
#[derive(Serialize, Deserialize, Debug)]
pub struct FulfillAttestationResult {
    pub credential_record: Record,
    pub request_id: String,
}

/// Decline an attestation request
#[hdk_extern]
pub fn decline_attestation(input: DeclineAttestationInput) -> ExternResult<Record> {
    if input.request_id.is_empty() || input.request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Request ID must be 1-256 characters".into())));
    }
    if input.subject_did.is_empty() || input.subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Subject DID must be 1-256 characters".into())));
    }

    // Find the request
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
            if let Some(req) = record.entry().to_app_option::<AttestationRequest>()
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
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            "Pending attestation request not found".into()
        ))),
    };

    // Verify the decliner is the subject
    if req.subject_did != input.subject_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the attestation subject can decline the request".into()
        )));
    }

    // Update status to Declined
    let mut declined_req = req;
    declined_req.status = AttestationStatus::Declined;
    let action_hash = update_entry(original_hash, &EntryTypes::AttestationRequest(declined_req))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated request".into()
        )))
}

/// Input for declining an attestation
#[derive(Serialize, Deserialize, Debug)]
pub struct DeclineAttestationInput {
    pub request_id: String,
    pub subject_did: String,
}

/// Get pending attestation requests for a subject
#[hdk_extern]
pub fn get_pending_requests(subject_did: String) -> ExternResult<Vec<Record>> {
    if subject_did.is_empty() || subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Subject DID must be 1-256 characters".into())));
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
            if let Some(req) = record.entry().to_app_option::<AttestationRequest>()
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

/// Get credentials for a subject
#[hdk_extern]
pub fn get_subject_credentials(subject_did: String) -> ExternResult<Vec<Record>> {
    if subject_did.is_empty() || subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Subject DID must be 1-256 characters".into())));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("subject:{}", subject_did))?,
            LinkTypes::SubjectToCredential,
        )?,
        GetStrategy::default(),
    )?;

    let mut credentials = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            // Only include non-revoked credentials
            if let Some(cred) = record.entry().to_app_option::<TrustCredential>()
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

/// Get credentials by trust tier
#[hdk_extern]
pub fn get_credentials_by_tier(tier: TrustTier) -> ExternResult<Vec<Record>> {
    let tier_name = format!("{:?}", tier);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("tier:{}", tier_name))?,
            LinkTypes::TierToCredential,
        )?,
        GetStrategy::default(),
    )?;

    let mut credentials = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Some(cred) = record.entry().to_app_option::<TrustCredential>()
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

/// Verify a trust credential's proof
///
/// This performs on-chain verification that the commitment matches
/// and the trust tier is valid for the claimed range.
/// Full STARK proof verification would be done off-chain due to
/// WASM constraints.
#[hdk_extern]
pub fn verify_credential(credential_id: String) -> ExternResult<VerificationResult> {
    if credential_id.is_empty() || credential_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Credential ID must be 1-256 characters".into())));
    }
    // Find the credential by searching subject links
    // We need to search all credentials since we only have credential_id
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

    // Check revocation
    let not_revoked = !cred.revoked;

    // Check expiration
    let now = sys_time()?;
    let not_expired = match cred.expires_at {
        Some(exp) => now < exp,
        None => true, // No expiration set
    };

    // Check commitment format (must be exactly 32 bytes for SHA3-256)
    let commitment_valid = cred.kvector_commitment.len() == 32
        && cred.kvector_commitment.iter().any(|&b| b != 0); // reject all-zeros

    // Validate score range is well-formed
    let range_valid = cred.trust_score_range.lower >= 0.0
        && cred.trust_score_range.upper <= 1.0
        && cred.trust_score_range.lower <= cred.trust_score_range.upper
        && !cred.trust_score_range.lower.is_nan()
        && !cred.trust_score_range.upper.is_nan();

    // Check tier consistency: verify tier matches the score range
    let mid_score = (cred.trust_score_range.lower as f64 + cred.trust_score_range.upper as f64) / 2.0;
    let expected_tier = TrustTier::from_score(mid_score);
    let tier_consistent = range_valid && cred.trust_tier == expected_tier;

    // Check proof format (must be non-empty for real proofs)
    let proof_format_valid = !cred.range_proof.is_empty();

    let all_valid = not_revoked && not_expired && commitment_valid && range_valid && tier_consistent && proof_format_valid;
    let message = if all_valid {
        "On-chain verification passed. Run off-chain STARK verification for full proof.".to_string()
    } else {
        let mut issues = Vec::new();
        if !not_revoked { issues.push("credential revoked"); }
        if !not_expired { issues.push("credential expired"); }
        if !commitment_valid { issues.push("invalid commitment"); }
        if !range_valid { issues.push("malformed score range"); }
        if !tier_consistent { issues.push("tier/range mismatch"); }
        if !proof_format_valid { issues.push("empty proof"); }
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

/// Result of credential verification
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

/// Pure verification logic — no HDK calls, fully testable.
/// Returns (commitment_valid, range_valid, tier_consistent, proof_format_valid).
pub fn verify_credential_pure(
    kvector_commitment: &[u8],
    trust_score_range: &TrustScoreRange,
    trust_tier: &TrustTier,
    range_proof: &[u8],
) -> (bool, bool, bool, bool) {
    let commitment_valid =
        kvector_commitment.len() == 32 && kvector_commitment.iter().any(|&b| b != 0);

    let range_valid = trust_score_range.lower >= 0.0
        && trust_score_range.upper <= 1.0
        && trust_score_range.lower <= trust_score_range.upper
        && !trust_score_range.lower.is_nan()
        && !trust_score_range.upper.is_nan();

    let mid_score = (trust_score_range.lower as f64 + trust_score_range.upper as f64) / 2.0;
    let expected_tier = TrustTier::from_score(mid_score);
    let tier_consistent = range_valid && *trust_tier == expected_tier;

    let proof_format_valid = !range_proof.is_empty();

    (commitment_valid, range_valid, tier_consistent, proof_format_valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_commitment() -> Vec<u8> {
        let mut c = vec![0u8; 32];
        c[0] = 1;
        c
    }

    #[test]
    fn test_valid_credential() {
        let (cv, rv, tc, pv) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.3, upper: 0.39 },
            &TrustTier::Basic, // mid = 0.345 → Basic
            &[1, 2, 3],
        );
        assert!(cv && rv && tc && pv);
    }

    #[test]
    fn test_commitment_too_short() {
        let (cv, _, _, _) = verify_credential_pure(
            &[1u8; 16],
            &TrustScoreRange { lower: 0.0, upper: 0.2 },
            &TrustTier::Observer,
            &[1],
        );
        assert!(!cv);
    }

    #[test]
    fn test_commitment_all_zeros() {
        let (cv, _, _, _) = verify_credential_pure(
            &[0u8; 32],
            &TrustScoreRange { lower: 0.0, upper: 0.2 },
            &TrustTier::Observer,
            &[1],
        );
        assert!(!cv);
    }

    #[test]
    fn test_range_invalid_nan() {
        let (_, rv, _, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: f32::NAN, upper: 0.5 },
            &TrustTier::Basic,
            &[1],
        );
        assert!(!rv);
    }

    #[test]
    fn test_range_inverted() {
        let (_, rv, _, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.8, upper: 0.2 },
            &TrustTier::Basic,
            &[1],
        );
        assert!(!rv);
    }

    #[test]
    fn test_range_exceeds_one() {
        let (_, rv, _, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.5, upper: 1.5 },
            &TrustTier::Elevated,
            &[1],
        );
        assert!(!rv);
    }

    #[test]
    fn test_tier_mismatch() {
        // mid = 0.5 → Standard, but claiming Guardian
        let (_, _, tc, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.4, upper: 0.59 },
            &TrustTier::Guardian,
            &[1],
        );
        assert!(!tc);
    }

    #[test]
    fn test_tier_observer() {
        let (_, _, tc, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.0, upper: 0.2 },
            &TrustTier::Observer, // mid = 0.1 → Observer
            &[1],
        );
        assert!(tc);
    }

    #[test]
    fn test_tier_guardian() {
        let (_, _, tc, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.85, upper: 0.95 },
            &TrustTier::Guardian, // mid = 0.9 → Guardian
            &[1],
        );
        assert!(tc);
    }

    #[test]
    fn test_tier_elevated() {
        let (_, _, tc, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.6, upper: 0.79 },
            &TrustTier::Elevated, // mid = 0.695 → Elevated
            &[1],
        );
        assert!(tc);
    }

    #[test]
    fn test_empty_proof() {
        let (_, _, _, pv) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.0, upper: 0.2 },
            &TrustTier::Observer,
            &[],
        );
        assert!(!pv);
    }

    #[test]
    fn test_all_tiers_from_score() {
        assert_eq!(TrustTier::from_score(0.0), TrustTier::Observer);
        assert_eq!(TrustTier::from_score(0.29), TrustTier::Observer);
        assert_eq!(TrustTier::from_score(0.3), TrustTier::Basic);
        assert_eq!(TrustTier::from_score(0.4), TrustTier::Standard);
        assert_eq!(TrustTier::from_score(0.6), TrustTier::Elevated);
        assert_eq!(TrustTier::from_score(0.8), TrustTier::Guardian);
        assert_eq!(TrustTier::from_score(1.0), TrustTier::Guardian);
    }
}
