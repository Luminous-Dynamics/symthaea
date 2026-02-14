//! LUCID Privacy Coordinator Zome

use hdk::prelude::*;
use privacy_integrity::*;

fn anchor_hash(s: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(s.to_string())))
}

fn create_anchor(s: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(s.to_string())))?;
    anchor_hash(s)
}

fn generate_uuid() -> String {
    let mut bytes = [0u8; 16];
    getrandom_03::fill(&mut bytes).expect("Failed to generate random bytes");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

/// Set sharing policy for a thought
#[hdk_extern]
pub fn set_sharing_policy(input: SetPolicyInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let policy = SharingPolicy {
        thought_id: input.thought_id.clone(),
        visibility: input.visibility,
        expires_at: input.expires_at,
        delegatable: input.delegatable.unwrap_or(false),
        created_at: now,
        updated_at: now,
    };

    let action_hash = create_entry(&EntryTypes::SharingPolicy(policy))?;

    let thought_anchor = format!("thought:{}", input.thought_id);
    create_anchor(&thought_anchor)?;
    create_link(
        anchor_hash(&thought_anchor)?,
        action_hash.clone(),
        LinkTypes::ThoughtToPolicy,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Policy not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SetPolicyInput {
    pub thought_id: String,
    pub visibility: Visibility,
    pub expires_at: Option<Timestamp>,
    pub delegatable: Option<bool>,
}

/// Get sharing policy for a thought
#[hdk_extern]
pub fn get_sharing_policy(thought_id: String) -> ExternResult<Option<Record>> {
    let thought_anchor = format!("thought:{}", thought_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&thought_anchor)?, LinkTypes::ThoughtToPolicy)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        return get(hash, GetOptions::default());
    }

    Ok(None)
}

/// Grant access to an agent
#[hdk_extern]
pub fn grant_access(input: GrantAccessInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let grant = AccessGrant {
        id: generate_uuid(),
        thought_id: input.thought_id.clone(),
        grantee: input.grantee.clone(),
        grantor: agent,
        permissions: input.permissions,
        expires_at: input.expires_at,
        can_delegate: input.can_delegate.unwrap_or(false),
        revoked: false,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::AccessGrant(grant))?;

    // Link thought to grant
    let thought_anchor = format!("thought:{}", input.thought_id);
    create_anchor(&thought_anchor)?;
    create_link(
        anchor_hash(&thought_anchor)?,
        action_hash.clone(),
        LinkTypes::ThoughtToGrants,
        (),
    )?;

    // Link grantee to grant (for their "shared with me" view)
    let grantee_anchor = format!("grantee:{}", input.grantee);
    create_anchor(&grantee_anchor)?;
    create_link(
        anchor_hash(&grantee_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToGrants,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Grant not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GrantAccessInput {
    pub thought_id: String,
    pub grantee: AgentPubKey,
    pub permissions: Vec<String>,
    pub expires_at: Option<Timestamp>,
    pub can_delegate: Option<bool>,
}

/// Get grants for a thought
#[hdk_extern]
pub fn get_thought_grants(thought_id: String) -> ExternResult<Vec<Record>> {
    let thought_anchor = format!("thought:{}", thought_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&thought_anchor)?, LinkTypes::ThoughtToGrants)?,
        GetStrategy::default(),
    )?;

    let mut grants = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            grants.push(record);
        }
    }

    Ok(grants)
}

/// Check if an agent has access to a thought
#[hdk_extern]
pub fn check_access(input: CheckAccessInput) -> ExternResult<bool> {
    // Get the policy
    if let Some(policy_record) = get_sharing_policy(input.thought_id.clone())? {
        if let Some(policy) = policy_record.entry().to_app_option::<SharingPolicy>().ok().flatten() {
            // Check expiration
            if let Some(expires) = policy.expires_at {
                let now = sys_time()?;
                if now > expires {
                    return Ok(false);
                }
            }

            // Check visibility
            match policy.visibility {
                Visibility::Public => return Ok(true),
                Visibility::Private => return Ok(false),
                Visibility::SharedWith(agents) => {
                    if agents.contains(&input.agent) {
                        return Ok(true);
                    }
                }
                Visibility::Federated(_) => {
                    // Would need cross-hApp verification
                    return Ok(false);
                }
                Visibility::GroupMembers(_) => {
                    // Would need group membership check
                    return Ok(false);
                }
            }
        }
    }

    // Check individual grants
    let grants = get_thought_grants(input.thought_id)?;
    for grant_record in grants {
        if let Some(grant) = grant_record.entry().to_app_option::<AccessGrant>().ok().flatten() {
            if grant.grantee == input.agent && !grant.revoked {
                // Check expiration
                if let Some(expires) = grant.expires_at {
                    let now = sys_time()?;
                    if now > expires {
                        continue;
                    }
                }
                return Ok(true);
            }
        }
    }

    Ok(false)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckAccessInput {
    pub thought_id: String,
    pub agent: AgentPubKey,
}

/// Log an access event
#[hdk_extern]
pub fn log_access(input: LogAccessInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let log = AccessLog {
        thought_id: input.thought_id.clone(),
        accessor: agent,
        access_type: input.access_type,
        timestamp: now,
    };

    let action_hash = create_entry(&EntryTypes::AccessLog(log))?;

    let thought_anchor = format!("thought:{}", input.thought_id);
    create_anchor(&thought_anchor)?;
    create_link(
        anchor_hash(&thought_anchor)?,
        action_hash.clone(),
        LinkTypes::ThoughtToAccessLog,
        (),
    )?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LogAccessInput {
    pub thought_id: String,
    pub access_type: String,
}

// ============================================================================
// Zero-Knowledge Proof Attestation Functions
// ============================================================================

/// Submit a ZK proof attestation after verification
#[hdk_extern]
pub fn submit_proof_attestation(input: SubmitAttestationInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let attestation = ZkProofAttestation {
        proof_id: generate_uuid(),
        proof_type: input.proof_type.clone(),
        proof_cid: input.proof_cid,
        public_inputs_hash: input.public_inputs_hash,
        subject_hash: input.subject_hash.clone(),
        verified_at: now,
        verifier: agent.clone(),
        verified: input.verified,
        expires_at: input.expires_at,
        metadata: input.metadata,
    };

    let action_hash = create_entry(&EntryTypes::ZkProofAttestation(attestation.clone()))?;

    // Link from subject to attestation
    let subject_anchor = format!("subject:{}", input.subject_hash);
    create_anchor(&subject_anchor)?;
    create_link(
        anchor_hash(&subject_anchor)?,
        action_hash.clone(),
        LinkTypes::SubjectToAttestation,
        (),
    )?;

    // Link from verifier to attestation
    let verifier_anchor = format!("verifier:{}", agent);
    create_anchor(&verifier_anchor)?;
    create_link(
        anchor_hash(&verifier_anchor)?,
        action_hash.clone(),
        LinkTypes::VerifierToAttestation,
        (),
    )?;

    // Link from proof type to attestation
    let type_anchor = format!("proof_type:{}", input.proof_type);
    create_anchor(&type_anchor)?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::ProofTypeToAttestation,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Attestation not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitAttestationInput {
    pub proof_type: String,
    pub proof_cid: String,
    pub public_inputs_hash: String,
    pub subject_hash: String,
    pub verified: bool,
    pub expires_at: Option<Timestamp>,
    pub metadata: Option<String>,
}

/// Get attestation by proof CID
#[hdk_extern]
pub fn get_proof_attestation(proof_cid: String) -> ExternResult<Option<Record>> {
    // Search through subject attestations - this is a simplified lookup
    // In production, you'd want an index by proof_cid
    let all_types = vec!["anonymous_belief", "reputation_range", "vote_eligibility"];

    for proof_type in all_types {
        let type_anchor = format!("proof_type:{}", proof_type);
        if let Ok(links) = get_links(
            LinkQuery::try_new(anchor_hash(&type_anchor)?, LinkTypes::ProofTypeToAttestation)?,
            GetStrategy::default(),
        ) {
            for link in links {
                let hash = ActionHash::try_from(link.target)
                    .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
                if let Some(record) = get(hash, GetOptions::default())? {
                    if let Some(att) = record.entry().to_app_option::<ZkProofAttestation>().ok().flatten() {
                        if att.proof_cid == proof_cid {
                            return Ok(Some(record));
                        }
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Get all attestations for a subject
#[hdk_extern]
pub fn get_subject_attestations(subject_hash: String) -> ExternResult<Vec<Record>> {
    let subject_anchor = format!("subject:{}", subject_hash);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&subject_anchor)?, LinkTypes::SubjectToAttestation)?,
        GetStrategy::default(),
    )?;

    let mut attestations = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            attestations.push(record);
        }
    }

    Ok(attestations)
}

/// Get attestations by proof type
#[hdk_extern]
pub fn get_attestations_by_type(proof_type: String) -> ExternResult<Vec<Record>> {
    let type_anchor = format!("proof_type:{}", proof_type);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&type_anchor)?, LinkTypes::ProofTypeToAttestation)?,
        GetStrategy::default(),
    )?;

    let mut attestations = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            attestations.push(record);
        }
    }

    Ok(attestations)
}

/// Get attestations created by a specific verifier
#[hdk_extern]
pub fn get_verifier_attestations(verifier: AgentPubKey) -> ExternResult<Vec<Record>> {
    let verifier_anchor = format!("verifier:{}", verifier);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&verifier_anchor)?, LinkTypes::VerifierToAttestation)?,
        GetStrategy::default(),
    )?;

    let mut attestations = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            attestations.push(record);
        }
    }

    Ok(attestations)
}

/// Check if a subject has a valid (non-expired, verified) attestation
#[hdk_extern]
pub fn has_valid_attestation(input: ValidAttestationInput) -> ExternResult<bool> {
    let attestations = get_subject_attestations(input.subject_hash)?;
    let now = sys_time()?;

    for record in attestations {
        if let Some(att) = record.entry().to_app_option::<ZkProofAttestation>().ok().flatten() {
            // Check if it matches the proof type (if specified)
            if let Some(ref required_type) = input.proof_type {
                if &att.proof_type != required_type {
                    continue;
                }
            }

            // Check if verified
            if !att.verified {
                continue;
            }

            // Check expiration
            if let Some(expires) = att.expires_at {
                if now > expires {
                    continue;
                }
            }

            return Ok(true);
        }
    }

    Ok(false)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ValidAttestationInput {
    pub subject_hash: String,
    pub proof_type: Option<String>,
}
