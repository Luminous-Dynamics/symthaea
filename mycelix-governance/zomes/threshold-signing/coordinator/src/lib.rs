//! Threshold Signing Coordinator Zome
//!
//! Business logic for DKG-based threshold signatures on governance decisions.
//!
//! Workflow:
//! 1. Create committee with threshold and member count
//! 2. Members register with their K-Vector trust scores
//! 3. Members run DKG ceremony off-chain using feldman-dkg
//! 4. Public commitments are submitted to advance ceremony
//! 5. Once complete, members can collectively sign governance decisions
//! 6. Threshold signatures are verified and stored

use hdk::prelude::*;
use k256::ecdsa::signature::Verifier;
use threshold_signing_integrity::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

/// Create a new signing committee
///
/// This initiates a DKG ceremony. Members must register and complete
/// the DKG protocol off-chain before the committee can sign.
#[hdk_extern]
pub fn create_committee(input: CreateCommitteeInput) -> ExternResult<Record> {
    // Validate threshold <= member_count (coordinator-side check supplements integrity)
    if input.threshold == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Threshold must be at least 1".into())));
    }
    if input.member_count < 3 {
        return Err(wasm_error!(WasmErrorInner::Guest("Committee must have at least 3 members".into())));
    }
    if input.threshold > input.member_count {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Threshold ({}) cannot exceed member count ({})",
            input.threshold, input.member_count
        ))));
    }

    let now = sys_time()?;
    let committee_id = format!("committee:{}:{}", input.name, now.as_micros());

    let committee = SigningCommittee {
        id: committee_id.clone(),
        name: input.name,
        threshold: input.threshold,
        member_count: input.member_count,
        phase: DkgPhase::Registration,
        public_key: None,
        commitments: Vec::new(),
        scope: input.scope,
        created_at: now,
        active: true,
        epoch: 1,
        min_phi: input.min_phi,
    };

    let action_hash = create_entry(&EntryTypes::SigningCommittee(committee))?;

    // Create anchor and link for committee lookup
    let committee_anchor = format!("committee:{}", committee_id);
    create_entry(&EntryTypes::Anchor(Anchor(committee_anchor.clone())))?;
    create_link(
        anchor_hash(&committee_anchor)?,
        action_hash.clone(),
        LinkTypes::EpochToCommittee,
        (),
    )?;

    // Link to all committees list
    let all_anchor = "all_committees";
    create_entry(&EntryTypes::Anchor(Anchor(all_anchor.to_string())))?;
    create_link(
        anchor_hash(all_anchor)?,
        action_hash.clone(),
        LinkTypes::EpochToCommittee,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find committee".into())))
}

/// Input for creating a signing committee
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCommitteeInput {
    pub name: String,
    pub threshold: u32,
    pub member_count: u32,
    pub scope: CommitteeScope,
    /// Minimum Φ score for committee membership (consciousness gate)
    #[serde(default)]
    pub min_phi: Option<f64>,
}

/// Register as a committee member
///
/// Called by validators who want to participate in the signing committee.
/// If the committee has a `min_phi` threshold, the caller's consciousness
/// score is checked via the governance bridge before registration is allowed.
#[hdk_extern]
pub fn register_member(input: RegisterMemberInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let caller = agent_info()?.agent_initial_pubkey;

    // If committee requires minimum Φ, verify consciousness gate
    if let Some(committee_record) = get_committee(input.committee_id.clone())? {
        if let Some(committee) = committee_record
            .entry()
            .to_app_option::<SigningCommittee>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if let Some(min_phi) = committee.min_phi {
                // Call governance bridge to verify consciousness gate
                let gate_result = call(
                    CallTargetCell::Local,
                    ZomeName::from("governance_bridge"),
                    FunctionName::from("verify_consciousness_gate"),
                    None,
                    serde_json::json!({
                        "action_type": "Voting",
                        "action_id": null
                    }),
                );

                match gate_result {
                    Ok(ZomeCallResponse::Ok(extern_io)) => {
                        // Parse the gate verification result
                        if let Ok(result) = extern_io.decode::<serde_json::Value>() {
                            let phi = result.get("phi").and_then(|p| p.as_f64()).unwrap_or(0.0);
                            if phi < min_phi {
                                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                                    "Consciousness gate failed: Φ score ({:.2}) below committee minimum ({:.2})",
                                    phi, min_phi
                                ))));
                            }
                        }
                    }
                    _ => {
                        // Bridge unavailable — use trust_score as Φ proxy (degraded mode)
                        if input.trust_score < min_phi {
                            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                                "Trust score ({:.2}) below committee Φ minimum ({:.2}) (bridge unavailable)",
                                input.trust_score, min_phi
                            ))));
                        }
                    }
                }
            }
        }
    }

    let member = CommitteeMember {
        committee_id: input.committee_id.clone(),
        participant_id: input.participant_id,
        agent: caller.clone(),
        member_did: input.member_did,
        trust_score: input.trust_score,
        public_share: None,
        vss_commitment: None,
        deal_submitted: false,
        qualified: false,
        registered_at: now,
    };

    let action_hash = create_entry(&EntryTypes::CommitteeMember(member))?;

    // Link committee to member
    let committee_anchor = format!("committee:{}", input.committee_id);
    create_link(
        anchor_hash(&committee_anchor)?,
        action_hash.clone(),
        LinkTypes::CommitteeToMember,
        (),
    )?;

    // Link agent to committee
    create_link(
        AnyLinkableHash::from(caller),
        action_hash.clone(),
        LinkTypes::AgentToCommittee,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find member".into())))
}

/// Input for registering as a committee member
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterMemberInput {
    pub committee_id: String,
    pub participant_id: u32,
    pub member_did: String,
    pub trust_score: f64,
}

/// Submit DKG deal (public commitments from off-chain DKG)
///
/// Called by members after running DKG dealing phase off-chain.
#[hdk_extern]
pub fn submit_dkg_deal(input: SubmitDkgDealInput) -> ExternResult<Record> {
    // Get member's record
    let member_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("committee:{}", input.committee_id))?,
            LinkTypes::CommitteeToMember,
        )?,
        GetStrategy::default(),
    )?;

    let caller = agent_info()?.agent_initial_pubkey;
    let mut member_action_hash = None;
    let mut member: Option<CommitteeMember> = None;

    for link in member_links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah.clone(), GetOptions::default())? {
            if let Some(m) = record.entry().to_app_option::<CommitteeMember>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.agent == caller {
                    member_action_hash = Some(ah);
                    member = Some(m);
                    break;
                }
            }
        }
    }

    let (original_hash, mut member) = match (member_action_hash, member) {
        (Some(ah), Some(m)) => (ah, m),
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            "Caller is not a member of this committee".into()
        ))),
    };

    // Validate the VSS commitment is a valid CommitmentSet before storing
    let cs = feldman_dkg::CommitmentSet::from_bytes(&input.vss_commitment)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Invalid VSS commitment: {}", e))))?;
    if cs.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "VSS commitment set must contain at least one commitment".into()
        )));
    }

    // Update member with deal info
    member.vss_commitment = Some(input.vss_commitment);
    member.deal_submitted = true;

    let action_hash = update_entry(original_hash, &EntryTypes::CommitteeMember(member))?;

    // Auto-advance phase: check if all registered members have submitted deals
    let all_member_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("committee:{}", input.committee_id))?,
            LinkTypes::CommitteeToMember,
        )?,
        GetStrategy::default(),
    )?;

    let mut all_submitted = true;
    let mut member_count = 0u32;
    for link in &all_member_links {
        let ah = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Some(m) = record.entry().to_app_option::<CommitteeMember>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                member_count += 1;
                if !m.deal_submitted {
                    all_submitted = false;
                }
            }
        }
    }

    // If all members submitted, advance committee phase to Dealing
    if all_submitted && member_count > 0 {
        if let Some(committee_record) = get_committee(input.committee_id)? {
            if let Some(mut committee) = committee_record
                .entry()
                .to_app_option::<SigningCommittee>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if committee.phase == DkgPhase::Registration {
                    committee.phase = DkgPhase::Dealing;
                    update_entry(
                        committee_record.action_address().clone(),
                        &EntryTypes::SigningCommittee(committee),
                    )?;
                }
            }
        }
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated member".into())))
}

/// Input for submitting DKG deal
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitDkgDealInput {
    pub committee_id: String,
    pub vss_commitment: Vec<u8>,
}

/// Finalize DKG ceremony
///
/// Called after all members have submitted valid deals.
/// Updates committee with combined public key.
/// Only callable by a registered member of the committee.
#[hdk_extern]
pub fn finalize_dkg(input: FinalizeDkgInput) -> ExternResult<Record> {
    // Verify caller is a registered committee member
    let caller = agent_info()?.agent_initial_pubkey;
    let member_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("committee:{}", input.committee_id))?,
            LinkTypes::CommitteeToMember,
        )?,
        GetStrategy::default(),
    )?;

    let mut caller_is_member = false;
    for link in &member_links {
        let ah = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Some(m) = record.entry().to_app_option::<CommitteeMember>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.agent == caller {
                    caller_is_member = true;
                    break;
                }
            }
        }
    }

    if !caller_is_member {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only registered committee members can finalize DKG".into()
        )));
    }

    // Get committee
    let committee_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("committee:{}", input.committee_id))?,
            LinkTypes::EpochToCommittee,
        )?,
        GetStrategy::default(),
    )?;

    let latest_link = committee_links.into_iter().max_by_key(|l| l.timestamp);
    let committee_hash = match latest_link {
        Some(link) => ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?,
        None => return Err(wasm_error!(WasmErrorInner::Guest("Committee not found".into()))),
    };

    let record = get(committee_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Committee not found".into())))?;

    let mut committee: SigningCommittee = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid committee entry".into())))?;

    // Guard: prevent double-finalize
    if committee.phase == DkgPhase::Complete {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Committee DKG is already complete".into()
        )));
    }

    // Validate combined public key is a valid secp256k1 point
    feldman_dkg::Commitment::from_bytes(&input.combined_public_key)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Invalid combined public key: {}", e)
        )))?;

    // Validate each public commitment set
    for (i, cs_bytes) in input.public_commitments.iter().enumerate() {
        feldman_dkg::CommitmentSet::from_bytes(cs_bytes)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                format!("Invalid commitment set at index {}: {}", i, e)
            )))?;
    }

    // Validate sufficient qualified members
    if (input.qualified_members.len() as u32) < committee.threshold {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Need at least {} qualified members, got {}",
            committee.threshold,
            input.qualified_members.len()
        ))));
    }

    // Update committee with DKG result
    committee.phase = DkgPhase::Complete;
    committee.public_key = Some(input.combined_public_key);
    committee.commitments = input.public_commitments;

    // Mark all qualified members
    let member_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("committee:{}", input.committee_id))?,
            LinkTypes::CommitteeToMember,
        )?,
        GetStrategy::default(),
    )?;

    for member_id in &input.qualified_members {
        for link in &member_links {
            let ah = ActionHash::try_from(link.target.clone())
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
            if let Some(record) = get(ah.clone(), GetOptions::default())? {
                if let Some(mut m) = record.entry().to_app_option::<CommitteeMember>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                {
                    if m.participant_id == *member_id {
                        m.qualified = true;
                        update_entry(ah, &EntryTypes::CommitteeMember(m))?;
                        break;
                    }
                }
            }
        }
    }

    let action_hash = update_entry(committee_hash, &EntryTypes::SigningCommittee(committee))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated committee".into())))
}

/// Input for finalizing DKG
#[derive(Serialize, Deserialize, Debug)]
pub struct FinalizeDkgInput {
    pub committee_id: String,
    pub combined_public_key: Vec<u8>,
    pub public_commitments: Vec<Vec<u8>>,
    pub qualified_members: Vec<u32>,
}

/// Submit a signature share
///
/// Called by committee members to contribute their partial signature.
#[hdk_extern]
pub fn submit_signature_share(input: SubmitSignatureShareInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let caller = agent_info()?.agent_initial_pubkey;

    let share = SignatureShare {
        signature_id: input.signature_id.clone(),
        participant_id: input.participant_id,
        signer: caller,
        share: input.share,
        content_hash: input.content_hash,
        submitted_at: now,
    };

    let action_hash = create_entry(&EntryTypes::SignatureShare(share))?;

    // Link to signature
    let sig_anchor = format!("signature:{}", input.signature_id);
    create_entry(&EntryTypes::Anchor(Anchor(sig_anchor.clone())))?;
    create_link(
        anchor_hash(&sig_anchor)?,
        action_hash.clone(),
        LinkTypes::SignatureToShare,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find share".into())))
}

/// Input for submitting a signature share
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitSignatureShareInput {
    pub signature_id: String,
    pub participant_id: u32,
    pub share: Vec<u8>,
    pub content_hash: Vec<u8>,
}

/// Combine signature shares into threshold signature
///
/// Called when enough shares have been collected to meet threshold.
/// Performs ECDSA verification against the committee's combined public key.
#[hdk_extern]
pub fn combine_signatures(input: CombineSignaturesInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let sig_id = format!("sig:{}:{}", input.committee_id, now.as_micros());

    // Verify the threshold signature against the committee's public key
    let mut verified = false;
    let committee_record = get_committee(input.committee_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Committee not found".into())))?;
    let committee: SigningCommittee = committee_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid committee entry".into())))?;

    if let Some(ref pk_bytes) = committee.public_key {
        // Construct verifying key from 33-byte compressed SEC1 point
        let vkey = k256::ecdsa::VerifyingKey::from_sec1_bytes(pk_bytes)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                format!("Invalid committee public key: {}", e)
            )))?;

        // Parse signature (compact r||s format)
        let sig = k256::ecdsa::Signature::from_slice(&input.combined_signature)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                format!("Invalid signature format: {}", e)
            )))?;

        // Verify signature against the content hash
        vkey.verify(&input.content_hash, &sig)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest(
                "Threshold signature verification failed".into()
            )))?;

        verified = true;
    }

    // Capture content_description before moving into struct
    let content_description = input.content_description;
    let is_proposal = content_description.starts_with("MIP-");
    let proposal_id = if is_proposal { Some(content_description.clone()) } else { None };

    let signature = ThresholdSignature {
        id: sig_id.clone(),
        committee_id: input.committee_id.clone(),
        signed_content_hash: input.content_hash,
        signed_content_description: content_description,
        signature: input.combined_signature,
        signer_count: input.signers.len() as u32,
        signers: input.signers,
        verified,
        signed_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ThresholdSignature(signature))?;

    // Link committee to signature
    let committee_anchor = format!("committee:{}", input.committee_id);
    create_link(
        anchor_hash(&committee_anchor)?,
        action_hash.clone(),
        LinkTypes::CommitteeToSignature,
        (),
    )?;

    // Create signature anchor for lookups
    let sig_anchor = format!("signature:{}", sig_id);
    create_entry(&EntryTypes::Anchor(Anchor(sig_anchor.clone())))?;

    // Link proposal to signature (if content_description is a proposal ID)
    if let Some(ref pid) = proposal_id {
        let proposal_sig_anchor = format!("proposal_sig:{}", pid);
        create_entry(&EntryTypes::Anchor(Anchor(proposal_sig_anchor.clone())))?;
        create_link(
            anchor_hash(&proposal_sig_anchor)?,
            action_hash.clone(),
            LinkTypes::ProposalToSignature,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find signature".into())))
}

/// Input for combining signatures
#[derive(Serialize, Deserialize, Debug)]
pub struct CombineSignaturesInput {
    pub committee_id: String,
    pub content_hash: Vec<u8>,
    pub content_description: String,
    pub combined_signature: Vec<u8>,
    pub signers: Vec<u32>,
    pub verified: bool,
}

/// Get committee by ID
#[hdk_extern]
pub fn get_committee(committee_id: String) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("committee:{}", committee_id))?,
            LinkTypes::EpochToCommittee,
        )?,
        GetStrategy::default(),
    )?;

    let latest = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(ah, GetOptions::default());
    }

    Ok(None)
}

/// Get committee members
#[hdk_extern]
pub fn get_committee_members(committee_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("committee:{}", committee_id))?,
            LinkTypes::CommitteeToMember,
        )?,
        GetStrategy::default(),
    )?;

    let mut members = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            members.push(record);
        }
    }

    Ok(members)
}

/// Get signature shares for a signature
#[hdk_extern]
pub fn get_signature_shares(signature_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("signature:{}", signature_id))?,
            LinkTypes::SignatureToShare,
        )?,
        GetStrategy::default(),
    )?;

    let mut shares = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            shares.push(record);
        }
    }

    Ok(shares)
}

/// Get all active committees
#[hdk_extern]
pub fn get_all_committees(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_committees")?, LinkTypes::EpochToCommittee)?,
        GetStrategy::default(),
    )?;

    let mut committees = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            // Only include active committees
            if let Some(committee) = record.entry().to_app_option::<SigningCommittee>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if committee.active {
                    committees.push(record);
                }
            }
        }
    }

    Ok(committees)
}

/// Initiate key rotation for a committee
///
/// Creates a new epoch and initiates a fresh DKG ceremony.
#[hdk_extern]
pub fn rotate_committee_keys(committee_id: String) -> ExternResult<Record> {
    // Get current committee
    let current = get_committee(committee_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Committee not found".into())))?;

    let current_committee: SigningCommittee = current
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid committee entry".into())))?;

    // Guard: can only rotate a completed committee (not mid-DKG)
    if current_committee.phase != DkgPhase::Complete {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only rotate keys for a completed committee".into()
        )));
    }

    // Deactivate current committee
    let mut old_committee = current_committee.clone();
    old_committee.active = false;
    update_entry(
        current.action_address().clone(),
        &EntryTypes::SigningCommittee(old_committee),
    )?;

    // Create new committee for next epoch
    let now = sys_time()?;
    let new_committee = SigningCommittee {
        id: format!("{}:epoch:{}", committee_id, current_committee.epoch + 1),
        name: current_committee.name,
        threshold: current_committee.threshold,
        member_count: current_committee.member_count,
        phase: DkgPhase::Registration,
        public_key: None,
        commitments: Vec::new(),
        scope: current_committee.scope,
        created_at: now,
        active: true,
        epoch: current_committee.epoch + 1,
        min_phi: current_committee.min_phi,
    };

    let action_hash = create_entry(&EntryTypes::SigningCommittee(new_committee.clone()))?;

    // Link to committee ID
    create_link(
        anchor_hash(&format!("committee:{}", committee_id))?,
        action_hash.clone(),
        LinkTypes::EpochToCommittee,
        (),
    )?;

    // Link to epoch anchor
    create_link(
        anchor_hash(&format!("epoch:{}", new_committee.epoch))?,
        action_hash.clone(),
        LinkTypes::EpochToCommittee,
        (),
    )?;

    // Carry forward previously-qualified members (re-register with reset DKG state)
    let member_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("committee:{}", committee_id))?,
            LinkTypes::CommitteeToMember,
        )?,
        GetStrategy::default(),
    )?;

    let new_committee_anchor = anchor_hash(&format!("committee:{}", new_committee.id))?;
    create_entry(&EntryTypes::Anchor(Anchor(format!("committee:{}", new_committee.id))))?;

    for link in &member_links {
        let ah = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Some(m) = record.entry().to_app_option::<CommitteeMember>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.qualified {
                    let new_member = CommitteeMember {
                        committee_id: new_committee.id.clone(),
                        participant_id: m.participant_id,
                        agent: m.agent,
                        member_did: m.member_did,
                        trust_score: m.trust_score,
                        public_share: None,
                        vss_commitment: None,
                        deal_submitted: false,
                        qualified: false,
                        registered_at: now,
                    };
                    let member_hash = create_entry(&EntryTypes::CommitteeMember(new_member))?;
                    create_link(
                        new_committee_anchor.clone(),
                        member_hash,
                        LinkTypes::CommitteeToMember,
                        (),
                    )?;
                }
            }
        }
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find new committee".into())))
}

/// Get all epochs (history) for a committee, sorted oldest-first
#[hdk_extern]
pub fn get_committee_history(committee_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("committee:{}", committee_id))?,
            LinkTypes::EpochToCommittee,
        )?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            records.push(record);
        }
    }

    // Sort by epoch (oldest first)
    records.sort_by_key(|r| {
        r.entry()
            .to_app_option::<SigningCommittee>()
            .ok()
            .flatten()
            .map(|c| c.epoch)
            .unwrap_or(0)
    });

    Ok(records)
}

/// Get verified threshold signature for a proposal
///
/// Looks up signatures linked to the proposal ID. Returns the first
/// verified signature found, or None if no verified signature exists.
#[hdk_extern]
pub fn get_proposal_signature(proposal_id: String) -> ExternResult<Option<Record>> {
    let proposal_sig_anchor = format!("proposal_sig:{}", proposal_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&proposal_sig_anchor)?,
            LinkTypes::ProposalToSignature,
        )?,
        GetStrategy::default(),
    )?;

    // Return the most recent verified signature
    let mut best: Option<(Timestamp, Record)> = None;
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Some(sig) = record
                .entry()
                .to_app_option::<ThresholdSignature>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if sig.verified {
                    match &best {
                        None => best = Some((sig.signed_at, record)),
                        Some((ts, _)) if sig.signed_at > *ts => {
                            best = Some((sig.signed_at, record));
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    Ok(best.map(|(_, r)| r))
}
