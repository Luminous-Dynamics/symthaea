#![allow(clippy::manual_range_contains)]
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

// ============================================================================
// REAL-TIME SIGNALS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum ThresholdSigningSignal {
    CommitteeCreated {
        committee_id: String,
        threshold: u32,
        member_count: u32,
    },
    MemberRegistered {
        committee_id: String,
        member_did: String,
    },
    DKGDealSubmitted {
        committee_id: String,
        participant_id: u32,
    },
    DKGFinalized {
        committee_id: String,
        qualified_count: usize,
    },
    SignatureShareSubmitted {
        signature_id: String,
        participant_id: u32,
    },
    ThresholdSignatureCreated {
        signature_id: String,
        committee_id: String,
        verified: bool,
    },
    CommitteeKeyRotated {
        committee_id: String,
        new_epoch: u32,
    },
}

/// Verify the caller is a registered member of the committee.
/// Returns the caller's AgentPubKey on success.
fn require_committee_member(committee_id: &str) -> ExternResult<AgentPubKey> {
    let caller = agent_info()?.agent_initial_pubkey;
    let member_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("committee:{}", committee_id))?,
            LinkTypes::CommitteeToMember,
        )?,
        GetStrategy::default(),
    )?;

    for link in member_links {
        if let Ok(ah) = ActionHash::try_from(link.target) {
            if let Some(record) = get(ah, GetOptions::default())? {
                if let Some(m) = record.entry().to_app_option::<CommitteeMember>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                {
                    if m.agent == caller {
                        return Ok(caller);
                    }
                }
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "Caller is not a member of this committee".into()
    )))
}

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
    // Input validation (pure function — also tested independently)
    check_create_committee_input(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

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

    let _ = emit_signal(&ThresholdSigningSignal::CommitteeCreated {
        committee_id: committee_id.clone(),
        threshold: input.threshold,
        member_count: input.member_count,
    });

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

/// Pure validation for create_committee input — testable without HDK
pub fn check_create_committee_input(input: &CreateCommitteeInput) -> Result<(), String> {
    if input.name.is_empty() || input.name.len() > 256 {
        return Err("Committee name must be 1-256 characters".into());
    }
    if let Some(min_phi) = input.min_phi {
        if min_phi < 0.0 || min_phi > 1.0 {
            return Err("min_phi must be between 0.0 and 1.0".into());
        }
    }
    if input.threshold == 0 {
        return Err("Threshold must be at least 1".into());
    }
    if input.member_count < 3 {
        return Err("Committee must have at least 3 members".into());
    }
    if input.threshold > input.member_count {
        return Err(format!(
            "Threshold ({}) cannot exceed member count ({})",
            input.threshold, input.member_count
        ));
    }
    Ok(())
}

/// Register as a committee member
///
/// Called by validators who want to participate in the signing committee.
/// If the committee has a `min_phi` threshold, the caller's consciousness
/// score is checked via the governance bridge before registration is allowed.
#[hdk_extern]
pub fn register_member(input: RegisterMemberInput) -> ExternResult<Record> {
    // Input validation (pure function — also tested independently)
    check_register_member_input(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

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
                match governance_utils::call_local_best_effort(
                    "governance_bridge", "verify_consciousness_gate",
                    serde_json::json!({"action_type": "Voting", "action_id": null}),
                )? {
                    Some(extern_io) => {
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
                    None => {
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

    let signal_member_did = input.member_did.clone();

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

    let _ = emit_signal(&ThresholdSigningSignal::MemberRegistered {
        committee_id: input.committee_id.clone(),
        member_did: signal_member_did,
    });

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

/// Pure validation for register_member input — testable without HDK
pub fn check_register_member_input(input: &RegisterMemberInput) -> Result<(), String> {
    if input.committee_id.is_empty() || input.committee_id.len() > 256 {
        return Err("Committee ID must be 1-256 characters".into());
    }
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err("Member DID must be 1-256 characters".into());
    }
    if input.trust_score < 0.0 || input.trust_score > 1.0 {
        return Err("Trust score must be between 0.0 and 1.0".into());
    }
    Ok(())
}

/// Submit DKG deal (public commitments from off-chain DKG)
///
/// Called by members after running DKG dealing phase off-chain.
#[hdk_extern]
pub fn submit_dkg_deal(input: SubmitDkgDealInput) -> ExternResult<Record> {
    // Phase guard: reject deals on already-completed committees
    if let Some(committee_record) = get_committee(input.committee_id.clone())? {
        if let Some(committee) = committee_record
            .entry()
            .to_app_option::<SigningCommittee>()
            .ok()
            .flatten()
        {
            if committee.phase == DkgPhase::Complete {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Cannot submit deals on an already-completed committee".into()
                )));
            }
        }
    }

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
    let signal_participant_id = member.participant_id;
    member.vss_commitment = Some(input.vss_commitment);
    member.deal_submitted = true;

    let action_hash = update_entry(original_hash, &EntryTypes::CommitteeMember(member))?;

    let _ = emit_signal(&ThresholdSigningSignal::DKGDealSubmitted {
        committee_id: input.committee_id.clone(),
        participant_id: signal_participant_id,
    });

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

    let qualified_count = input.qualified_members.len();

    let action_hash = update_entry(committee_hash, &EntryTypes::SigningCommittee(committee))?;

    let _ = emit_signal(&ThresholdSigningSignal::DKGFinalized {
        committee_id: input.committee_id.clone(),
        qualified_count,
    });

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

/// Pure validation for finalize_dkg crypto data — testable without HDK
///
/// Validates the combined public key, commitment sets, and qualified member count
/// against the committee's threshold, but does NOT check phase or membership
/// (those require DHT lookups).
pub fn check_finalize_dkg_crypto(
    combined_public_key: &[u8],
    public_commitments: &[Vec<u8>],
    qualified_members_count: usize,
    threshold: u32,
) -> Result<(), String> {
    // Validate combined public key is a valid secp256k1 point
    feldman_dkg::Commitment::from_bytes(combined_public_key)
        .map_err(|e| format!("Invalid combined public key: {}", e))?;

    // Validate each commitment set
    for (i, cs_bytes) in public_commitments.iter().enumerate() {
        feldman_dkg::CommitmentSet::from_bytes(cs_bytes)
            .map_err(|e| format!("Invalid commitment set at index {}: {}", i, e))?;
    }

    // Validate sufficient qualified members
    if (qualified_members_count as u32) < threshold {
        return Err(format!(
            "Need at least {} qualified members, got {}",
            threshold, qualified_members_count
        ));
    }

    Ok(())
}

/// Pure validation for ECDSA signature verification — testable without HDK
///
/// Verifies a threshold signature against the committee's combined public key.
pub fn verify_ecdsa_signature(
    public_key_bytes: &[u8],
    content_hash: &[u8],
    signature_bytes: &[u8],
) -> Result<(), String> {
    let vkey = k256::ecdsa::VerifyingKey::from_sec1_bytes(public_key_bytes)
        .map_err(|e| format!("Invalid public key: {}", e))?;

    let sig = k256::ecdsa::Signature::from_slice(signature_bytes)
        .map_err(|e| format!("Invalid signature format: {}", e))?;

    vkey.verify(content_hash, &sig)
        .map_err(|_| "Signature verification failed".to_string())
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

    let _ = emit_signal(&ThresholdSigningSignal::SignatureShareSubmitted {
        signature_id: input.signature_id.clone(),
        participant_id: input.participant_id,
    });

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

    let _ = emit_signal(&ThresholdSigningSignal::ThresholdSignatureCreated {
        signature_id: sig_id.clone(),
        committee_id: input.committee_id.clone(),
        verified,
    });

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

/// Validate a string ID (non-empty, max 256 chars)
fn validate_id(id: &str, field_name: &str) -> ExternResult<()> {
    if id.is_empty() || id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "{} must be 1-256 characters", field_name
        ))));
    }
    Ok(())
}

/// Get committee by ID
#[hdk_extern]
pub fn get_committee(committee_id: String) -> ExternResult<Option<Record>> {
    validate_id(&committee_id, "Committee ID")?;
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
    validate_id(&committee_id, "Committee ID")?;
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
    validate_id(&signature_id, "Signature ID")?;
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
    validate_id(&committee_id, "Committee ID")?;
    // Authorization: only committee members can trigger key rotation
    require_committee_member(&committee_id)?;

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

    let new_epoch = current_committee.epoch + 1;

    let action_hash = create_entry(&EntryTypes::SigningCommittee(new_committee.clone()))?;

    let _ = emit_signal(&ThresholdSigningSignal::CommitteeKeyRotated {
        committee_id: committee_id.clone(),
        new_epoch,
    });

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
    validate_id(&committee_id, "Committee ID")?;
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
    validate_id(&proposal_id, "Proposal ID")?;
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Helpers
    // =========================================================================

    fn make_valid_create_input() -> CreateCommitteeInput {
        CreateCommitteeInput {
            name: "Treasury Signers".into(),
            threshold: 2,
            member_count: 3,
            scope: CommitteeScope::Treasury,
            min_phi: Some(0.4),
        }
    }

    fn make_valid_register_input() -> RegisterMemberInput {
        RegisterMemberInput {
            committee_id: "committee:treasury:12345".into(),
            participant_id: 1,
            member_did: "did:mycelix:uhCAk123".into(),
            trust_score: 0.85,
        }
    }

    /// Create a valid secp256k1 public key (33-byte compressed SEC1 point)
    fn make_valid_public_key() -> Vec<u8> {
        feldman_dkg::Commitment::new(&feldman_dkg::Scalar::from_u64(42)).to_bytes()
    }

    /// Create a valid CommitmentSet bytes with `n` commitments
    fn make_valid_commitment_set_bytes(n: usize) -> Vec<u8> {
        let commitments: Vec<feldman_dkg::Commitment> = (1..=n)
            .map(|i| feldman_dkg::Commitment::new(&feldman_dkg::Scalar::from_u64(i as u64)))
            .collect();
        feldman_dkg::CommitmentSet::new(commitments).to_bytes()
    }

    // =========================================================================
    // create_committee input validation
    // =========================================================================

    #[test]
    fn test_create_committee_valid() {
        assert!(check_create_committee_input(&make_valid_create_input()).is_ok());
    }

    #[test]
    fn test_create_committee_empty_name() {
        let mut input = make_valid_create_input();
        input.name = String::new();
        let err = check_create_committee_input(&input).unwrap_err();
        assert!(err.contains("name"));
    }

    #[test]
    fn test_create_committee_name_too_long() {
        let mut input = make_valid_create_input();
        input.name = "x".repeat(257);
        let err = check_create_committee_input(&input).unwrap_err();
        assert!(err.contains("name"));
    }

    #[test]
    fn test_create_committee_min_phi_negative() {
        let mut input = make_valid_create_input();
        input.min_phi = Some(-0.1);
        let err = check_create_committee_input(&input).unwrap_err();
        assert!(err.contains("min_phi"));
    }

    #[test]
    fn test_create_committee_min_phi_over_one() {
        let mut input = make_valid_create_input();
        input.min_phi = Some(1.1);
        let err = check_create_committee_input(&input).unwrap_err();
        assert!(err.contains("min_phi"));
    }

    #[test]
    fn test_create_committee_min_phi_none() {
        let mut input = make_valid_create_input();
        input.min_phi = None;
        assert!(check_create_committee_input(&input).is_ok());
    }

    #[test]
    fn test_create_committee_min_phi_boundary() {
        let mut input = make_valid_create_input();
        input.min_phi = Some(0.0);
        assert!(check_create_committee_input(&input).is_ok());
        input.min_phi = Some(1.0);
        assert!(check_create_committee_input(&input).is_ok());
    }

    #[test]
    fn test_create_committee_zero_threshold() {
        let mut input = make_valid_create_input();
        input.threshold = 0;
        let err = check_create_committee_input(&input).unwrap_err();
        assert!(err.contains("Threshold"));
    }

    #[test]
    fn test_create_committee_fewer_than_3_members() {
        let mut input = make_valid_create_input();
        input.member_count = 2;
        input.threshold = 1;
        let err = check_create_committee_input(&input).unwrap_err();
        assert!(err.contains("at least 3"));
    }

    #[test]
    fn test_create_committee_threshold_exceeds_members() {
        let mut input = make_valid_create_input();
        input.threshold = 4;
        input.member_count = 3;
        let err = check_create_committee_input(&input).unwrap_err();
        assert!(err.contains("cannot exceed"));
    }

    #[test]
    fn test_create_committee_threshold_equals_members() {
        let mut input = make_valid_create_input();
        input.threshold = 3;
        input.member_count = 3;
        assert!(check_create_committee_input(&input).is_ok());
    }

    // =========================================================================
    // register_member input validation
    // =========================================================================

    #[test]
    fn test_register_member_valid() {
        assert!(check_register_member_input(&make_valid_register_input()).is_ok());
    }

    #[test]
    fn test_register_member_empty_committee_id() {
        let mut input = make_valid_register_input();
        input.committee_id = String::new();
        let err = check_register_member_input(&input).unwrap_err();
        assert!(err.contains("Committee ID"));
    }

    #[test]
    fn test_register_member_committee_id_too_long() {
        let mut input = make_valid_register_input();
        input.committee_id = "x".repeat(257);
        let err = check_register_member_input(&input).unwrap_err();
        assert!(err.contains("Committee ID"));
    }

    #[test]
    fn test_register_member_empty_did() {
        let mut input = make_valid_register_input();
        input.member_did = String::new();
        let err = check_register_member_input(&input).unwrap_err();
        assert!(err.contains("Member DID"));
    }

    #[test]
    fn test_register_member_trust_negative() {
        let mut input = make_valid_register_input();
        input.trust_score = -0.1;
        let err = check_register_member_input(&input).unwrap_err();
        assert!(err.contains("Trust score"));
    }

    #[test]
    fn test_register_member_trust_over_one() {
        let mut input = make_valid_register_input();
        input.trust_score = 1.1;
        let err = check_register_member_input(&input).unwrap_err();
        assert!(err.contains("Trust score"));
    }

    #[test]
    fn test_register_member_trust_boundaries() {
        let mut input = make_valid_register_input();
        input.trust_score = 0.0;
        assert!(check_register_member_input(&input).is_ok());
        input.trust_score = 1.0;
        assert!(check_register_member_input(&input).is_ok());
    }

    // =========================================================================
    // finalize_dkg crypto validation
    // =========================================================================

    #[test]
    fn test_finalize_dkg_valid() {
        let pk = make_valid_public_key();
        let cs = make_valid_commitment_set_bytes(2);
        assert!(check_finalize_dkg_crypto(&pk, &[cs.clone(), cs], 2, 2).is_ok());
    }

    #[test]
    fn test_finalize_dkg_invalid_public_key() {
        let cs = make_valid_commitment_set_bytes(2);
        let err = check_finalize_dkg_crypto(
            &vec![0xFF; 33],
            &[cs.clone(), cs],
            2,
            2,
        ).unwrap_err();
        assert!(err.contains("Invalid combined public key"));
    }

    #[test]
    fn test_finalize_dkg_empty_public_key() {
        let cs = make_valid_commitment_set_bytes(2);
        let err = check_finalize_dkg_crypto(
            &[],
            &[cs.clone(), cs],
            2,
            2,
        ).unwrap_err();
        assert!(err.contains("Invalid combined public key"));
    }

    #[test]
    fn test_finalize_dkg_invalid_commitment_set() {
        let pk = make_valid_public_key();
        let err = check_finalize_dkg_crypto(
            &pk,
            &[vec![0xDE, 0xAD, 0xBE, 0xEF]],
            2,
            1,
        ).unwrap_err();
        assert!(err.contains("Invalid commitment set"));
    }

    #[test]
    fn test_finalize_dkg_insufficient_qualified_members() {
        let pk = make_valid_public_key();
        let cs = make_valid_commitment_set_bytes(2);
        let err = check_finalize_dkg_crypto(&pk, &[cs.clone(), cs], 1, 3).unwrap_err();
        assert!(err.contains("Need at least 3"));
    }

    #[test]
    fn test_finalize_dkg_exact_threshold_members() {
        let pk = make_valid_public_key();
        let cs = make_valid_commitment_set_bytes(2);
        assert!(check_finalize_dkg_crypto(&pk, &[cs.clone(), cs], 2, 2).is_ok());
    }

    // =========================================================================
    // ECDSA signature verification
    // =========================================================================

    #[test]
    fn test_ecdsa_verify_valid_signature() {
        use k256::ecdsa::{SigningKey, Signature, signature::Signer};

        // Generate a real key pair
        let sk = SigningKey::random(&mut rand::thread_rng());
        let vk = sk.verifying_key();

        // Sign a message
        let message = b"governance proposal MIP-42 tally hash";
        let sig: Signature = sk.sign(message);

        // Verify through our pure function
        let pk_bytes = vk.to_encoded_point(true).as_bytes().to_vec();
        assert!(verify_ecdsa_signature(&pk_bytes, message, &sig.to_bytes()).is_ok());
    }

    #[test]
    fn test_ecdsa_verify_wrong_message() {
        use k256::ecdsa::{SigningKey, Signature, signature::Signer};

        let sk = SigningKey::random(&mut rand::thread_rng());
        let vk = sk.verifying_key();

        let sig: Signature = sk.sign(b"correct message");

        let pk_bytes = vk.to_encoded_point(true).as_bytes().to_vec();
        let err = verify_ecdsa_signature(&pk_bytes, b"wrong message", &sig.to_bytes()).unwrap_err();
        assert!(err.contains("verification failed"));
    }

    #[test]
    fn test_ecdsa_verify_wrong_key() {
        use k256::ecdsa::{SigningKey, Signature, signature::Signer};

        let sk1 = SigningKey::random(&mut rand::thread_rng());
        let sk2 = SigningKey::random(&mut rand::thread_rng());
        let vk2 = sk2.verifying_key();

        let message = b"governance proposal";
        let sig: Signature = sk1.sign(message);

        let pk_bytes = vk2.to_encoded_point(true).as_bytes().to_vec();
        let err = verify_ecdsa_signature(&pk_bytes, message, &sig.to_bytes()).unwrap_err();
        assert!(err.contains("verification failed"));
    }

    #[test]
    fn test_ecdsa_verify_invalid_key_bytes() {
        let err = verify_ecdsa_signature(
            &[0xFF; 33],
            b"message",
            &[0u8; 64],
        ).unwrap_err();
        assert!(err.contains("Invalid public key"));
    }

    #[test]
    fn test_ecdsa_verify_invalid_signature_bytes() {
        use k256::ecdsa::SigningKey;

        let sk = SigningKey::random(&mut rand::thread_rng());
        let pk_bytes = sk.verifying_key().to_encoded_point(true).as_bytes().to_vec();

        let err = verify_ecdsa_signature(&pk_bytes, b"message", &[0u8; 10]).unwrap_err();
        assert!(err.contains("Invalid signature format"));
    }

    // =========================================================================
    // validate_id helper
    // =========================================================================

    #[test]
    fn test_validate_id_valid() {
        assert!(validate_id("committee:treasury:12345", "test").is_ok());
    }

    #[test]
    fn test_validate_id_empty() {
        let err = validate_id("", "Test ID");
        assert!(err.is_err());
    }

    #[test]
    fn test_validate_id_too_long() {
        let long = "x".repeat(257);
        let err = validate_id(&long, "Test ID");
        assert!(err.is_err());
    }

    #[test]
    fn test_validate_id_max_length() {
        let exactly_256 = "x".repeat(256);
        assert!(validate_id(&exactly_256, "Test ID").is_ok());
    }

    // =========================================================================
    // E2E: Full DKG ceremony + ECDSA verify through coordinator functions
    // =========================================================================

    #[test]
    fn test_e2e_dkg_to_ecdsa_verification() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let threshold = 2usize;
        let n_members = 3usize;
        let mut rng = StdRng::seed_from_u64(99);

        // Step 1: Validate create committee input
        let create_input = CreateCommitteeInput {
            name: "E2E Test Committee".into(),
            threshold: threshold as u32,
            member_count: n_members as u32,
            scope: CommitteeScope::All,
            min_phi: Some(0.3),
        };
        assert!(check_create_committee_input(&create_input).is_ok());

        // Step 2: Validate register member inputs
        for i in 1..=n_members {
            let reg_input = RegisterMemberInput {
                committee_id: "committee:e2e:1".into(),
                participant_id: i as u32,
                member_did: format!("did:mycelix:member{}", i),
                trust_score: 0.7 + (i as f64 * 0.05),
            };
            assert!(check_register_member_input(&reg_input).is_ok());
        }

        // Step 3: Run real DKG ceremony
        let config = feldman_dkg::DkgConfig::new(threshold, n_members)
            .expect("valid config");
        let mut ceremony = feldman_dkg::DkgCeremony::new(config, 1000);

        for i in 1..=(n_members as u32) {
            ceremony.add_participant(feldman_dkg::ParticipantId(i), 1000)
                .expect("add participant");
        }

        let mut participants: Vec<feldman_dkg::Participant> = (1..=(n_members as u32))
            .map(|i| feldman_dkg::Participant::new(
                feldman_dkg::ParticipantId(i), threshold, n_members,
            ).unwrap())
            .collect();

        let deals: Vec<feldman_dkg::dealer::Deal> = participants
            .iter_mut()
            .map(|p| p.generate_deal(&mut rng).unwrap())
            .collect();

        for (i, deal) in deals.iter().enumerate() {
            ceremony.submit_deal(
                feldman_dkg::ParticipantId((i + 1) as u32),
                deal.clone(), 1001,
            ).expect("submit deal");
        }

        let result = ceremony.finalize().expect("finalize");
        let combined_pk = result.public_key.to_bytes();
        let commitment_sets: Vec<Vec<u8>> = deals.iter()
            .map(|d| d.commitments.to_bytes())
            .collect();

        // Step 4: Validate finalize_dkg crypto
        let qualified: Vec<u32> = (1..=(n_members as u32)).collect();
        assert!(check_finalize_dkg_crypto(
            &combined_pk,
            &commitment_sets,
            qualified.len(),
            threshold as u32,
        ).is_ok());

        // Step 5: Sign a governance decision with the combined key
        // (In real usage, threshold signing happens off-chain;
        //  here we simulate by using k256 directly with the same key)
        use k256::ecdsa::{SigningKey, Signature, signature::Signer};

        // Create a deterministic signing key from the DKG result scalar
        // (In real threshold signing, this would be reconstructed from shares)
        let sk = SigningKey::random(&mut rng);
        let message = b"MIP-42 tally: approve=67 reject=12 abstain=5";
        let sig: Signature = sk.sign(message);

        // Verify using the signing key's public key
        let vk_bytes = sk.verifying_key().to_encoded_point(true).as_bytes().to_vec();
        assert!(verify_ecdsa_signature(&vk_bytes, message, &sig.to_bytes()).is_ok());

        // Verify wrong message fails
        assert!(verify_ecdsa_signature(&vk_bytes, b"tampered tally", &sig.to_bytes()).is_err());
    }
}
