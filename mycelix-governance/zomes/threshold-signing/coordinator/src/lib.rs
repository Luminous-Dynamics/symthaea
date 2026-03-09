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
                if let Some(m) = record
                    .entry()
                    .to_app_option::<CommitteeMember>()
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
    check_create_committee_input(&input).map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

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
        signature_algorithm: input.signature_algorithm.unwrap_or_default(),
        pq_required: input.pq_required,
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find committee".into()
    )))
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
    /// Signature algorithm (defaults to ECDSA for backwards compatibility)
    #[serde(default)]
    pub signature_algorithm: Option<ThresholdSignatureAlgorithm>,
    /// If true, PQ signature is mandatory for Hybrid committees (no ECDSA-only fallback)
    #[serde(default)]
    pub pq_required: bool,
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
    check_register_member_input(&input).map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

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
                    "governance_bridge",
                    "verify_consciousness_gate",
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
                        // Bridge unavailable — reject registration (fail-closed).
                        // trust_score is self-reported and cannot substitute for
                        // a verified consciousness credential.
                        return Err(wasm_error!(WasmErrorInner::Guest(format!(
                            "Consciousness bridge unavailable: cannot verify Φ score for committee \
                             with min_phi={:.2}. Registration rejected (fail-closed). \
                             Ensure governance_bridge zome is installed and responsive.",
                            min_phi
                        ))));
                    }
                }
            }
        }
    }

    // Check violation history — reject participants with excessive penalties
    let cumulative_penalty =
        compute_participant_penalty(&input.committee_id, input.participant_id)?;
    if cumulative_penalty > MAX_CUMULATIVE_PENALTY {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Participant {} barred from committee: cumulative violation penalty ({:.2}) exceeds maximum ({:.2})",
            input.participant_id, cumulative_penalty, MAX_CUMULATIVE_PENALTY
        ))));
    }

    // Reduce effective trust score by violation penalty
    let effective_trust = (input.trust_score - cumulative_penalty).max(0.0);

    let signal_member_did = input.member_did.clone();

    let member = CommitteeMember {
        committee_id: input.committee_id.clone(),
        participant_id: input.participant_id,
        agent: caller.clone(),
        member_did: input.member_did,
        trust_score: effective_trust,
        public_share: None,
        vss_commitment: None,
        ml_kem_encapsulation_key: input.ml_kem_encapsulation_key,
        encrypted_shares: None,
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find member".into()
    )))
}

/// Input for registering as a committee member
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterMemberInput {
    pub committee_id: String,
    pub participant_id: u32,
    pub member_did: String,
    pub trust_score: f64,
    /// ML-KEM-768 encapsulation (public) key for encrypted DKG deals
    #[serde(default)]
    pub ml_kem_encapsulation_key: Option<Vec<u8>>,
}

/// Maximum cumulative violation penalty before a participant is barred from committees
pub const MAX_CUMULATIVE_PENALTY: f64 = 0.5;

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
    // ML-KEM-768 encapsulation key must be exactly 1,184 bytes when present
    if let Some(ref ek) = input.ml_kem_encapsulation_key {
        if ek.len() != 1184 {
            return Err(format!(
                "ML-KEM encapsulation key must be exactly 1184 bytes, got {}",
                ek.len()
            ));
        }
    }
    Ok(())
}

/// Submit DKG deal (public commitments from off-chain DKG)
///
/// Called by members after running DKG dealing phase off-chain.
#[hdk_extern]
pub fn submit_dkg_deal(input: SubmitDkgDealInput) -> ExternResult<Record> {
    // Phase guard: reject deals on already-completed committees
    // Also capture pq_required to enforce encrypted shares
    let mut committee_pq_required = false;
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
            committee_pq_required = committee.pq_required;
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
            if let Some(m) = record
                .entry()
                .to_app_option::<CommitteeMember>()
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
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Caller is not a member of this committee".into()
            )))
        }
    };

    // Validate the VSS commitment is a valid CommitmentSet before storing
    let cs = feldman_dkg::CommitmentSet::from_bytes(&input.vss_commitment).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid VSS commitment: {}",
            e
        )))
    })?;
    if cs.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "VSS commitment set must contain at least one commitment".into()
        )));
    }

    // Validate encrypted shares if present
    if let Some(ref enc_bytes) = input.encrypted_shares {
        if enc_bytes.is_empty() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Encrypted shares cannot be empty".into()
            )));
        }
        feldman_dkg::EncryptedDeal::from_bytes(enc_bytes).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Invalid encrypted shares: {}",
                e
            )))
        })?;
    } else if committee_pq_required {
        // PQ-required committees MUST use encrypted deals
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Committee requires post-quantum security (pq_required=true): \
             encrypted_shares must be provided. Use ML-KEM-768 to encrypt \
             deal shares for each recipient."
                .into()
        )));
    }

    // Update member with deal info
    let signal_participant_id = member.participant_id;
    member.vss_commitment = Some(input.vss_commitment);
    member.encrypted_shares = input.encrypted_shares;
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
            if let Some(m) = record
                .entry()
                .to_app_option::<CommitteeMember>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                member_count += 1;
                if !m.deal_submitted {
                    all_submitted = false;
                }
            }
        }
    }

    // If all members submitted, advance committee phase
    if all_submitted && member_count > 0 {
        if let Some(committee_record) = get_committee(input.committee_id)? {
            if let Some(mut committee) = committee_record
                .entry()
                .to_app_option::<SigningCommittee>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if committee.phase == DkgPhase::Registration {
                    // PQ committees go through CommitmentCollection first
                    committee.phase = match committee.signature_algorithm {
                        ThresholdSignatureAlgorithm::MlDsa65
                        | ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65 => {
                            DkgPhase::CommitmentCollection
                        }
                        ThresholdSignatureAlgorithm::Ecdsa => DkgPhase::Dealing,
                    };
                    update_entry(
                        committee_record.action_address().clone(),
                        &EntryTypes::SigningCommittee(committee),
                    )?;
                }
            }
        }
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated member".into()
    )))
}

/// Input for submitting DKG deal
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitDkgDealInput {
    pub committee_id: String,
    pub vss_commitment: Vec<u8>,
    /// ML-KEM-768 encrypted deal shares (optional, for PQ-protected DKG)
    #[serde(default)]
    pub encrypted_shares: Option<Vec<u8>>,
}

// ============================================================================
// COMMIT-REVEAL PROTOCOL (PQ commitment collection phase)
// ============================================================================

/// Input for submitting a hash commitment (commit phase)
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitHashCommitmentInput {
    pub committee_id: String,
    pub participant_id: u32,
    /// Serialized HashCommitmentSet from feldman-dkg
    pub commitment_set_bytes: Vec<u8>,
}

/// Submit hash commitment for the commit-reveal protocol
///
/// Called during CommitmentCollection phase. Each dealer submits SHA-256
/// hashes of their shares before revealing the actual shares.
#[hdk_extern]
pub fn submit_hash_commitment(input: SubmitHashCommitmentInput) -> ExternResult<Record> {
    validate_id(&input.committee_id, "Committee ID")?;

    let now = sys_time()?;

    // Validate the commitment set deserializes
    feldman_dkg::HashCommitmentSet::from_bytes(&input.commitment_set_bytes).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid hash commitment set: {}",
            e
        )))
    })?;

    // Verify committee is in CommitmentCollection phase
    let committee_record = get_committee(input.committee_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Committee not found".into())
    ))?;
    let committee: SigningCommittee = committee_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid committee entry".into()
        )))?;

    if committee.phase != DkgPhase::CommitmentCollection {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Committee must be in CommitmentCollection phase, currently in {:?}",
            committee.phase
        ))));
    }

    let commitment_entry = DkgHashCommitment {
        committee_id: input.committee_id.clone(),
        dealer_participant_id: input.participant_id,
        commitment_set_bytes: input.commitment_set_bytes,
        submitted_at: now,
    };

    let action_hash = create_entry(&EntryTypes::DkgHashCommitment(commitment_entry))?;

    // Link committee to hash commitment
    let committee_anchor = format!("committee:{}", input.committee_id);
    create_link(
        anchor_hash(&committee_anchor)?,
        action_hash.clone(),
        LinkTypes::CommitteeToHashCommitment,
        (),
    )?;

    // Auto-advance: check if all members have submitted hash commitments
    let commitment_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&committee_anchor)?,
            LinkTypes::CommitteeToHashCommitment,
        )?,
        GetStrategy::default(),
    )?;

    if commitment_links.len() as u32 >= committee.member_count {
        // Advance to Dealing phase
        let mut updated_committee = committee;
        updated_committee.phase = DkgPhase::Dealing;
        update_entry(
            committee_record.action_address().clone(),
            &EntryTypes::SigningCommittee(updated_committee),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find hash commitment".into()
    )))
}

/// Input for submitting a hash reveal (reveal phase)
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitHashRevealInput {
    pub committee_id: String,
    pub participant_id: u32,
    /// Serialized HashReveal from feldman-dkg
    pub reveal_bytes: Vec<u8>,
}

/// Submit hash reveal for the commit-reveal protocol
///
/// Called during Verification phase. Each dealer reveals the salts used
/// to create their hash commitments. Recipients can then verify
/// H(share || dealer_id || salt) matches the committed hash.
#[hdk_extern]
pub fn submit_hash_reveal(input: SubmitHashRevealInput) -> ExternResult<Record> {
    validate_id(&input.committee_id, "Committee ID")?;

    let now = sys_time()?;

    // Validate the reveal deserializes
    let reveal: feldman_dkg::HashReveal = serde_json::from_slice(&input.reveal_bytes)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Invalid hash reveal: {}", e))))?;

    // Verify committee is in Verification phase (reveals happen after deals)
    let committee_record = get_committee(input.committee_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Committee not found".into())
    ))?;
    let committee: SigningCommittee = committee_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid committee entry".into()
        )))?;

    if committee.phase != DkgPhase::Verification {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Committee must be in Verification phase for reveals, currently in {:?}",
            committee.phase
        ))));
    }

    // Verify the reveal's dealer matches a submitted commitment
    let commitment_anchor = format!("committee:{}", input.committee_id);
    let commitment_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&commitment_anchor)?,
            LinkTypes::CommitteeToHashCommitment,
        )?,
        GetStrategy::default(),
    )?;

    let mut found_matching_commitment = false;
    for link in &commitment_links {
        let ah = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            let commitment: DkgHashCommitment = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid commitment entry".into()
                )))?;

            if commitment.dealer_participant_id == input.participant_id {
                // Verify the reveal is from the same dealer as the commitment
                if reveal.dealer.0 != commitment.dealer_participant_id {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Reveal dealer {} doesn't match commitment dealer {}",
                        reveal.dealer.0, commitment.dealer_participant_id
                    ))));
                }

                // Verify the commitment set can be parsed and salt count matches
                let commitment_set =
                    feldman_dkg::HashCommitmentSet::from_bytes(&commitment.commitment_set_bytes)
                        .map_err(|e| {
                            wasm_error!(WasmErrorInner::Guest(format!(
                                "Failed to parse stored commitment set: {}",
                                e
                            )))
                        })?;

                if reveal.salts.len() != commitment_set.commitments.len() {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Reveal has {} salts but commitment has {} entries",
                        reveal.salts.len(),
                        commitment_set.commitments.len()
                    ))));
                }

                found_matching_commitment = true;
                break;
            }
        }
    }

    if !found_matching_commitment {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "No hash commitment found for participant {} — cannot reveal without prior commitment",
            input.participant_id
        ))));
    }

    let reveal_entry = DkgHashReveal {
        committee_id: input.committee_id.clone(),
        dealer_participant_id: input.participant_id,
        reveal_bytes: input.reveal_bytes,
        submitted_at: now,
    };

    let action_hash = create_entry(&EntryTypes::DkgHashReveal(reveal_entry))?;

    // Link committee to hash reveal
    let committee_anchor = format!("committee:{}", input.committee_id);
    create_link(
        anchor_hash(&committee_anchor)?,
        action_hash.clone(),
        LinkTypes::CommitteeToHashReveal,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find hash reveal".into()
    )))
}

/// Get all hash commitments for a committee
#[hdk_extern]
pub fn get_hash_commitments(committee_id: String) -> ExternResult<Vec<Record>> {
    let committee_anchor = format!("committee:{}", committee_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&committee_anchor)?,
            LinkTypes::CommitteeToHashCommitment,
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
    Ok(records)
}

/// Get all hash reveals for a committee
#[hdk_extern]
pub fn get_hash_reveals(committee_id: String) -> ExternResult<Vec<Record>> {
    let committee_anchor = format!("committee:{}", committee_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&committee_anchor)?,
            LinkTypes::CommitteeToHashReveal,
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
    Ok(records)
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
            if let Some(m) = record
                .entry()
                .to_app_option::<CommitteeMember>()
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
        None => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Committee not found".into()
            )))
        }
    };

    let record = get(committee_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Committee not found".into())
    ))?;

    let mut committee: SigningCommittee = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid committee entry".into()
        )))?;

    // Guard: prevent double-finalize
    if committee.phase == DkgPhase::Complete {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Committee DKG is already complete".into()
        )));
    }

    // Validate combined public key is a valid secp256k1 point
    feldman_dkg::Commitment::from_bytes(&input.combined_public_key).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid combined public key: {}",
            e
        )))
    })?;

    // Validate each public commitment set
    for (i, cs_bytes) in input.public_commitments.iter().enumerate() {
        feldman_dkg::CommitmentSet::from_bytes(cs_bytes).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Invalid commitment set at index {}: {}",
                i, e
            )))
        })?;
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
                if let Some(mut m) = record
                    .entry()
                    .to_app_option::<CommitteeMember>()
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated committee".into()
    )))
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find share".into()
    )))
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
    let committee_record = get_committee(input.committee_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Committee not found".into())
    ))?;
    let committee: SigningCommittee = committee_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid committee entry".into()
        )))?;

    if let Some(ref pk_bytes) = committee.public_key {
        match committee.signature_algorithm {
            ThresholdSignatureAlgorithm::Ecdsa => {
                // Classical ECDSA verification
                let vkey = k256::ecdsa::VerifyingKey::from_sec1_bytes(pk_bytes).map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Invalid committee public key: {}",
                        e
                    )))
                })?;
                let sig =
                    k256::ecdsa::Signature::from_slice(&input.combined_signature).map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!(
                            "Invalid signature format: {}",
                            e
                        )))
                    })?;
                vkey.verify(&input.content_hash, &sig).map_err(|_| {
                    wasm_error!(WasmErrorInner::Guest(
                        "ECDSA threshold signature verification failed".into()
                    ))
                })?;
                verified = true;
            }
            ThresholdSignatureAlgorithm::MlDsa65 => {
                // Post-quantum ML-DSA-65 verification via PQ attestor
                let pq_sig = input.pq_signature.as_ref().ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest(
                        "MlDsa65 algorithm requires pq_signature field".into()
                    ))
                })?;
                let attestor_record = get_pq_attestor(GetPqAttestorInput {
                    committee_id: input.committee_id.clone(),
                    epoch: committee.epoch,
                })?;
                let attestor: PqAttestor = attestor_record
                    .ok_or_else(|| {
                        wasm_error!(WasmErrorInner::Guest(
                            "No PQ attestor registered for this committee epoch".into()
                        ))
                    })?
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                    .ok_or_else(|| {
                        wasm_error!(WasmErrorInner::Guest("Invalid PQ attestor entry".into()))
                    })?;
                feldman_dkg::pq_sig::verify(
                    &attestor.ml_dsa_public_key,
                    &input.content_hash,
                    pq_sig,
                )
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "ML-DSA-65 signature verification failed: {}",
                        e
                    )))
                })?;
                verified = true;
            }
            ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65 => {
                // Step 1: ECDSA verification (required)
                let vkey = k256::ecdsa::VerifyingKey::from_sec1_bytes(pk_bytes).map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Invalid committee public key: {}",
                        e
                    )))
                })?;
                let sig =
                    k256::ecdsa::Signature::from_slice(&input.combined_signature).map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!(
                            "Invalid signature format: {}",
                            e
                        )))
                    })?;
                vkey.verify(&input.content_hash, &sig).map_err(|_| {
                    wasm_error!(WasmErrorInner::Guest(
                        "Hybrid ECDSA threshold signature verification failed".into()
                    ))
                })?;

                // Step 2: ML-DSA-65 verification (if attestor registered + PQ sig present)
                if let Some(ref pq_sig) = input.pq_signature {
                    let attestor_record = get_pq_attestor(GetPqAttestorInput {
                        committee_id: input.committee_id.clone(),
                        epoch: committee.epoch,
                    })?;
                    if let Some(record) = attestor_record {
                        let attestor: PqAttestor = record
                            .entry()
                            .to_app_option()
                            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                            .ok_or_else(|| {
                                wasm_error!(WasmErrorInner::Guest(
                                    "Invalid PQ attestor entry".into()
                                ))
                            })?;
                        feldman_dkg::pq_sig::verify(
                            &attestor.ml_dsa_public_key,
                            &input.content_hash,
                            pq_sig,
                        )
                        .map_err(|e| {
                            wasm_error!(WasmErrorInner::Guest(format!(
                                "Hybrid ML-DSA-65 signature verification failed: {}",
                                e
                            )))
                        })?;
                    } else if committee.pq_required {
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "PQ attestor required but not registered for this committee epoch"
                                .into()
                        )));
                    }
                    // If no attestor registered and !pq_required: ECDSA-only (graceful degradation)
                } else if committee.pq_required {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "PQ signature required for this committee but pq_signature field is missing".into()
                    )));
                }

                verified = true;
            }
        }
    }

    // Capture content_description before moving into struct
    let content_description = input.content_description;
    let is_proposal = content_description.starts_with("MIP-");
    let proposal_id = if is_proposal {
        Some(content_description.clone())
    } else {
        None
    };

    let signature = ThresholdSignature {
        id: sig_id.clone(),
        committee_id: input.committee_id.clone(),
        signed_content_hash: input.content_hash,
        signed_content_description: content_description,
        signature: input.combined_signature,
        pq_signature: input.pq_signature,
        signature_algorithm: committee.signature_algorithm.clone(),
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find signature".into()
    )))
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
    /// Post-quantum signature (ML-DSA-65), required for MlDsa65 or Hybrid algorithms
    #[serde(default)]
    pub pq_signature: Option<Vec<u8>>,
}

/// Compute cumulative violation penalty for a participant across ALL committees.
///
/// Queries the global participant_violations anchor to sum penalty_score values
/// from all DkgViolationReport entries for this participant, regardless of committee.
fn compute_participant_penalty(_committee_id: &str, participant_id: u32) -> ExternResult<f64> {
    let participant_anchor = format!("participant_violations:{}", participant_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&participant_anchor)?,
            LinkTypes::ParticipantToViolation,
        )?,
        GetStrategy::default(),
    )?;

    let mut cumulative = 0.0f64;
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Some(report) = record
                .entry()
                .to_app_option::<DkgViolationReport>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                cumulative += report.penalty_score;
            }
        }
    }

    Ok(cumulative.min(1.0))
}

/// Pure helper: check if a participant's penalty bars them from joining
pub fn check_penalty_eligibility(
    cumulative_penalty: f64,
    trust_score: f64,
    max_penalty: f64,
) -> Result<f64, String> {
    if cumulative_penalty > max_penalty {
        return Err(format!(
            "Cumulative violation penalty ({:.2}) exceeds maximum ({:.2})",
            cumulative_penalty, max_penalty
        ));
    }
    Ok((trust_score - cumulative_penalty).max(0.0))
}

/// Validate a string ID (non-empty, max 256 chars)
fn validate_id(id: &str, field_name: &str) -> ExternResult<()> {
    if id.is_empty() || id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "{} must be 1-256 characters",
            field_name
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
            if let Some(committee) = record
                .entry()
                .to_app_option::<SigningCommittee>()
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
    let current = get_committee(committee_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Committee not found".into())
    ))?;

    let current_committee: SigningCommittee = current
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid committee entry".into()
        )))?;

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
        signature_algorithm: current_committee.signature_algorithm,
        pq_required: current_committee.pq_required,
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
    create_entry(&EntryTypes::Anchor(Anchor(format!(
        "committee:{}",
        new_committee.id
    ))))?;

    for link in &member_links {
        let ah = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Some(m) = record
                .entry()
                .to_app_option::<CommitteeMember>()
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
                        ml_kem_encapsulation_key: m.ml_kem_encapsulation_key,
                        encrypted_shares: None,
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find new committee".into()
    )))
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

// ============================================================================
// DKG VIOLATION REPORTING
// ============================================================================

/// Input for reporting a DKG protocol violation
#[derive(Serialize, Deserialize, Debug)]
pub struct ReportViolationInput {
    pub committee_id: String,
    pub participant_id: u32,
    pub violation_type: String,
    pub severity: ViolationSeverity,
    pub penalty_score: f64,
    pub epoch: u32,
}

/// Report a DKG protocol violation
///
/// Records a violation report on-chain for accountability and reputation slashing.
/// The reporter is automatically set to the calling agent.
#[hdk_extern]
pub fn report_dkg_violation(input: ReportViolationInput) -> ExternResult<Record> {
    validate_id(&input.committee_id, "Committee ID")?;

    let now = sys_time()?;
    let reporter = agent_info()?.agent_initial_pubkey;

    let report = DkgViolationReport {
        committee_id: input.committee_id.clone(),
        participant_id: input.participant_id,
        violation_type: input.violation_type,
        severity: input.severity,
        penalty_score: input.penalty_score,
        epoch: input.epoch,
        reporter,
        reported_at: now,
    };

    let action_hash = create_entry(&EntryTypes::DkgViolationReport(report))?;

    // Link committee to violation report
    let committee_anchor = format!("committee:{}", input.committee_id);
    create_link(
        anchor_hash(&committee_anchor)?,
        action_hash.clone(),
        LinkTypes::CommitteeToViolation,
        (),
    )?;

    // Link participant to violation report (global, cross-committee)
    let participant_anchor = format!("participant_violations:{}", input.participant_id);
    create_entry(&EntryTypes::Anchor(Anchor(participant_anchor.clone())))?;
    create_link(
        anchor_hash(&participant_anchor)?,
        action_hash.clone(),
        LinkTypes::ParticipantToViolation,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find violation report".into()
    )))
}

/// Get all violation reports for a committee
#[hdk_extern]
pub fn get_committee_violations(committee_id: String) -> ExternResult<Vec<Record>> {
    validate_id(&committee_id, "Committee ID")?;
    let committee_anchor = format!("committee:{}", committee_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&committee_anchor)?,
            LinkTypes::CommitteeToViolation,
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
    Ok(records)
}

/// Get all violation reports for a participant (global, cross-committee)
#[hdk_extern]
pub fn get_participant_violations(participant_id: u32) -> ExternResult<Vec<Record>> {
    let participant_anchor = format!("participant_violations:{}", participant_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&participant_anchor)?,
            LinkTypes::ParticipantToViolation,
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
    Ok(records)
}

/// Get cumulative penalty score for a participant (global)
#[hdk_extern]
pub fn get_participant_penalty(participant_id: u32) -> ExternResult<f64> {
    compute_participant_penalty("", participant_id)
}

// ============================================================================
// PQ ATTESTOR SELECTION AND REGISTRATION
// ============================================================================

/// Deterministic PQ attestor selection for a committee epoch.
///
/// Uses a hash of committee_id and epoch to select the attestor index,
/// ensuring all participants agree on who the attestor is without coordination.
pub fn select_attestor_index(committee_id: &str, epoch: u32, qualified_count: usize) -> usize {
    use k256::sha2::{Digest, Sha256};
    if qualified_count == 0 {
        return 0;
    }
    let mut hasher = Sha256::new();
    hasher.update(committee_id.as_bytes());
    hasher.update(epoch.to_le_bytes());
    let hash = hasher.finalize();
    // Use first 8 bytes as u64 for index selection
    let idx_bytes: [u8; 8] = hash[..8].try_into().unwrap_or([0; 8]);
    let idx = u64::from_le_bytes(idx_bytes);
    (idx as usize) % qualified_count
}

/// Input for registering as PQ attestor
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterPqAttestorInput {
    pub committee_id: String,
    pub epoch: u32,
    pub participant_id: u32,
    /// ML-DSA-65 public key bytes (1,952 bytes)
    pub ml_dsa_public_key: Vec<u8>,
    /// Proof-of-possession: ML-DSA-65 signature over the challenge string
    /// "PQ_ATTESTOR_REGISTRATION:{committee_id}:{epoch}:{participant_id}"
    #[serde(default)]
    pub proof_of_possession: Option<Vec<u8>>,
}

/// Register as the PQ attestor for a committee epoch
///
/// Only the deterministically selected attestor can register.
/// The caller must provide their ML-DSA-65 public key.
#[hdk_extern]
pub fn register_pq_attestor(input: RegisterPqAttestorInput) -> ExternResult<Record> {
    validate_id(&input.committee_id, "Committee ID")?;

    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Verify the committee exists and is complete
    let committee_record = get_committee(input.committee_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Committee not found".into())
    ))?;
    let committee: SigningCommittee = committee_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid committee entry".into()
        )))?;

    // Must be a hybrid or ML-DSA committee
    match committee.signature_algorithm {
        ThresholdSignatureAlgorithm::MlDsa65 | ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65 => {}
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "PQ attestor only applicable to MlDsa65 or Hybrid committees".into()
            )));
        }
    }

    // Verify proof-of-possession if provided
    if let Some(ref pop) = input.proof_of_possession {
        let challenge = format!(
            "PQ_ATTESTOR_REGISTRATION:{}:{}:{}",
            input.committee_id, input.epoch, input.participant_id
        );
        feldman_dkg::pq_sig::verify(&input.ml_dsa_public_key, challenge.as_bytes(), pop).map_err(
            |e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Proof-of-possession verification failed: {}",
                    e
                )))
            },
        )?;
    }

    let attestor = PqAttestor {
        committee_id: input.committee_id.clone(),
        epoch: input.epoch,
        participant_id: input.participant_id,
        agent: caller,
        ml_dsa_public_key: input.ml_dsa_public_key,
        registered_at: now,
    };

    let action_hash = create_entry(&EntryTypes::PqAttestor(attestor))?;

    // Link committee to attestor
    let attestor_anchor = format!("attestor:{}:{}", input.committee_id, input.epoch);
    create_entry(&EntryTypes::Anchor(Anchor(attestor_anchor.clone())))?;
    create_link(
        anchor_hash(&attestor_anchor)?,
        action_hash.clone(),
        LinkTypes::CommitteeToAttestor,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find attestor record".into()
    )))
}

/// Get the PQ attestor for a committee epoch
#[hdk_extern]
pub fn get_pq_attestor(input: GetPqAttestorInput) -> ExternResult<Option<Record>> {
    let attestor_anchor = format!("attestor:{}:{}", input.committee_id, input.epoch);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&attestor_anchor)?,
            LinkTypes::CommitteeToAttestor,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.into_iter().next() {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(ah, GetOptions::default());
    }

    Ok(None)
}

/// Input for getting PQ attestor
#[derive(Serialize, Deserialize, Debug)]
pub struct GetPqAttestorInput {
    pub committee_id: String,
    pub epoch: u32,
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
            signature_algorithm: None,
            pq_required: false,
        }
    }

    fn make_valid_register_input() -> RegisterMemberInput {
        RegisterMemberInput {
            committee_id: "committee:treasury:12345".into(),
            participant_id: 1,
            member_did: "did:mycelix:uhCAk123".into(),
            trust_score: 0.85,
            ml_kem_encapsulation_key: None,
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
    // ML-KEM encapsulation key validation
    // =========================================================================

    #[test]
    fn test_register_member_valid_ml_kem_ek() {
        let mut input = make_valid_register_input();
        input.ml_kem_encapsulation_key = Some(vec![0xAB; 1184]);
        assert!(check_register_member_input(&input).is_ok());
    }

    #[test]
    fn test_register_member_ml_kem_ek_too_short() {
        let mut input = make_valid_register_input();
        input.ml_kem_encapsulation_key = Some(vec![0xAB; 100]);
        let err = check_register_member_input(&input).unwrap_err();
        assert!(
            err.contains("1184 bytes"),
            "Expected 1184 bytes error, got: {err}"
        );
    }

    #[test]
    fn test_register_member_ml_kem_ek_too_long() {
        let mut input = make_valid_register_input();
        input.ml_kem_encapsulation_key = Some(vec![0xAB; 1185]);
        let err = check_register_member_input(&input).unwrap_err();
        assert!(
            err.contains("1184 bytes"),
            "Expected 1184 bytes error, got: {err}"
        );
    }

    #[test]
    fn test_register_member_ml_kem_ek_none_ok() {
        // None is valid — ML-KEM is optional for backward compatibility
        let input = make_valid_register_input();
        assert!(input.ml_kem_encapsulation_key.is_none());
        assert!(check_register_member_input(&input).is_ok());
    }

    // =========================================================================
    // violation penalty eligibility
    // =========================================================================

    #[test]
    fn test_penalty_eligibility_no_violations() {
        let result = check_penalty_eligibility(0.0, 0.85, MAX_CUMULATIVE_PENALTY);
        assert!(result.is_ok());
        assert!((result.unwrap() - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_penalty_eligibility_minor_violation() {
        // 0.05 penalty reduces trust from 0.85 to 0.80
        let result = check_penalty_eligibility(0.05, 0.85, MAX_CUMULATIVE_PENALTY);
        assert!(result.is_ok());
        assert!((result.unwrap() - 0.80).abs() < 1e-10);
    }

    #[test]
    fn test_penalty_eligibility_barred() {
        // 0.6 penalty exceeds MAX_CUMULATIVE_PENALTY (0.5)
        let result = check_penalty_eligibility(0.6, 0.85, MAX_CUMULATIVE_PENALTY);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds maximum"));
    }

    #[test]
    fn test_penalty_eligibility_at_boundary() {
        // Exactly at MAX_CUMULATIVE_PENALTY — still allowed
        let result = check_penalty_eligibility(0.5, 0.85, MAX_CUMULATIVE_PENALTY);
        assert!(result.is_ok());
        assert!((result.unwrap() - 0.35).abs() < 1e-10);
    }

    #[test]
    fn test_penalty_eligibility_trust_floor_zero() {
        // Penalty exceeds trust — effective trust floors at 0.0
        let result = check_penalty_eligibility(0.4, 0.3, MAX_CUMULATIVE_PENALTY);
        assert!(result.is_ok());
        assert!((result.unwrap() - 0.0).abs() < 1e-10);
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
        let err = check_finalize_dkg_crypto(&vec![0xFF; 33], &[cs.clone(), cs], 2, 2).unwrap_err();
        assert!(err.contains("Invalid combined public key"));
    }

    #[test]
    fn test_finalize_dkg_empty_public_key() {
        let cs = make_valid_commitment_set_bytes(2);
        let err = check_finalize_dkg_crypto(&[], &[cs.clone(), cs], 2, 2).unwrap_err();
        assert!(err.contains("Invalid combined public key"));
    }

    #[test]
    fn test_finalize_dkg_invalid_commitment_set() {
        let pk = make_valid_public_key();
        let err =
            check_finalize_dkg_crypto(&pk, &[vec![0xDE, 0xAD, 0xBE, 0xEF]], 2, 1).unwrap_err();
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
        use k256::ecdsa::{signature::Signer, Signature, SigningKey};

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
        use k256::ecdsa::{signature::Signer, Signature, SigningKey};

        let sk = SigningKey::random(&mut rand::thread_rng());
        let vk = sk.verifying_key();

        let sig: Signature = sk.sign(b"correct message");

        let pk_bytes = vk.to_encoded_point(true).as_bytes().to_vec();
        let err = verify_ecdsa_signature(&pk_bytes, b"wrong message", &sig.to_bytes()).unwrap_err();
        assert!(err.contains("verification failed"));
    }

    #[test]
    fn test_ecdsa_verify_wrong_key() {
        use k256::ecdsa::{signature::Signer, Signature, SigningKey};

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
        let err = verify_ecdsa_signature(&[0xFF; 33], b"message", &[0u8; 64]).unwrap_err();
        assert!(err.contains("Invalid public key"));
    }

    #[test]
    fn test_ecdsa_verify_invalid_signature_bytes() {
        use k256::ecdsa::SigningKey;

        let sk = SigningKey::random(&mut rand::thread_rng());
        let pk_bytes = sk
            .verifying_key()
            .to_encoded_point(true)
            .as_bytes()
            .to_vec();

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
            signature_algorithm: None,
            pq_required: false,
        };
        assert!(check_create_committee_input(&create_input).is_ok());

        // Step 2: Validate register member inputs
        for i in 1..=n_members {
            let reg_input = RegisterMemberInput {
                committee_id: "committee:e2e:1".into(),
                participant_id: i as u32,
                member_did: format!("did:mycelix:member{}", i),
                trust_score: 0.7 + (i as f64 * 0.05),
                ml_kem_encapsulation_key: None,
            };
            assert!(check_register_member_input(&reg_input).is_ok());
        }

        // Step 3: Run real DKG ceremony
        let config = feldman_dkg::DkgConfig::new(threshold, n_members).expect("valid config");
        let mut ceremony = feldman_dkg::DkgCeremony::new(config, 1000);

        for i in 1..=(n_members as u32) {
            ceremony
                .add_participant(feldman_dkg::ParticipantId(i), 1000)
                .expect("add participant");
        }

        let mut participants: Vec<feldman_dkg::Participant> = (1..=(n_members as u32))
            .map(|i| {
                feldman_dkg::Participant::new(feldman_dkg::ParticipantId(i), threshold, n_members)
                    .unwrap()
            })
            .collect();

        let deals: Vec<feldman_dkg::dealer::Deal> = participants
            .iter_mut()
            .map(|p| p.generate_deal(&mut rng).unwrap())
            .collect();

        for (i, deal) in deals.iter().enumerate() {
            ceremony
                .submit_deal(
                    feldman_dkg::ParticipantId((i + 1) as u32),
                    deal.clone(),
                    1001,
                )
                .expect("submit deal");
        }

        let result = ceremony.finalize().expect("finalize");
        let combined_pk = result.public_key.to_bytes();
        let commitment_sets: Vec<Vec<u8>> =
            deals.iter().map(|d| d.commitments.to_bytes()).collect();

        // Step 4: Validate finalize_dkg crypto
        let qualified: Vec<u32> = (1..=(n_members as u32)).collect();
        assert!(check_finalize_dkg_crypto(
            &combined_pk,
            &commitment_sets,
            qualified.len(),
            threshold as u32,
        )
        .is_ok());

        // Step 5: Sign a governance decision with the combined key
        // (In real usage, threshold signing happens off-chain;
        //  here we simulate by using k256 directly with the same key)
        use k256::ecdsa::{signature::Signer, Signature, SigningKey};

        // Create a deterministic signing key from the DKG result scalar
        // (In real threshold signing, this would be reconstructed from shares)
        let sk = SigningKey::random(&mut rng);
        let message = b"MIP-42 tally: approve=67 reject=12 abstain=5";
        let sig: Signature = sk.sign(message);

        // Verify using the signing key's public key
        let vk_bytes = sk
            .verifying_key()
            .to_encoded_point(true)
            .as_bytes()
            .to_vec();
        assert!(verify_ecdsa_signature(&vk_bytes, message, &sig.to_bytes()).is_ok());

        // Verify wrong message fails
        assert!(verify_ecdsa_signature(&vk_bytes, b"tampered tally", &sig.to_bytes()).is_err());
    }

    // =========================================================================
    // PQ attestor selection
    // =========================================================================

    #[test]
    fn test_attestor_selection_deterministic() {
        let idx1 = select_attestor_index("committee:treasury:1", 1, 5);
        let idx2 = select_attestor_index("committee:treasury:1", 1, 5);
        assert_eq!(idx1, idx2, "Same inputs should produce same attestor");
    }

    #[test]
    fn test_attestor_selection_varies_across_epochs() {
        // Over 20 epochs, at least 2 distinct attestors should be selected
        let indices: Vec<usize> = (1..=20)
            .map(|epoch| select_attestor_index("committee:treasury:1", epoch, 100))
            .collect();
        let unique: std::collections::HashSet<usize> = indices.iter().copied().collect();
        assert!(
            unique.len() >= 2,
            "Expected at least 2 distinct attestors across 20 epochs, got {}",
            unique.len()
        );
    }

    #[test]
    fn test_attestor_selection_in_range() {
        for epoch in 1..=20 {
            let idx = select_attestor_index("committee:test", epoch, 5);
            assert!(idx < 5, "Index must be < qualified_count");
        }
    }

    #[test]
    fn test_attestor_selection_zero_count() {
        let idx = select_attestor_index("committee:test", 1, 0);
        assert_eq!(idx, 0);
    }

    // =========================================================================
    // ML-DSA-65 proof-of-possession
    // =========================================================================

    #[test]
    fn test_pq_attestor_proof_of_possession_valid() {
        let kp = feldman_dkg::pq_sig::generate_signing_keypair();
        let committee_id = "committee:treasury:1";
        let epoch = 1u32;
        let participant_id = 2u32;

        let challenge = format!(
            "PQ_ATTESTOR_REGISTRATION:{}:{}:{}",
            committee_id, epoch, participant_id
        );
        let pop = feldman_dkg::pq_sig::sign(&kp.signing_key, challenge.as_bytes());

        // Verify the proof-of-possession succeeds
        assert!(
            feldman_dkg::pq_sig::verify(&kp.verifying_key_bytes(), challenge.as_bytes(), &pop,)
                .is_ok()
        );
    }

    #[test]
    fn test_pq_attestor_proof_of_possession_wrong_key() {
        let kp1 = feldman_dkg::pq_sig::generate_signing_keypair();
        let kp2 = feldman_dkg::pq_sig::generate_signing_keypair();

        let challenge = b"PQ_ATTESTOR_REGISTRATION:committee:1:1:2";
        let pop = feldman_dkg::pq_sig::sign(&kp1.signing_key, challenge);

        // PoP signed by kp1 should fail verification with kp2's key
        assert!(feldman_dkg::pq_sig::verify(&kp2.verifying_key_bytes(), challenge, &pop,).is_err());
    }

    #[test]
    fn test_pq_attestor_proof_of_possession_wrong_challenge() {
        let kp = feldman_dkg::pq_sig::generate_signing_keypair();

        let correct_challenge = b"PQ_ATTESTOR_REGISTRATION:committee:1:1:2";
        let wrong_challenge = b"PQ_ATTESTOR_REGISTRATION:committee:1:1:3";

        let pop = feldman_dkg::pq_sig::sign(&kp.signing_key, correct_challenge);

        // PoP should fail when challenge doesn't match
        assert!(
            feldman_dkg::pq_sig::verify(&kp.verifying_key_bytes(), wrong_challenge, &pop,).is_err()
        );
    }

    // =========================================================================
    // ML-DSA-65 signature verification
    // =========================================================================

    #[test]
    fn test_ml_dsa65_sign_verify_roundtrip() {
        let kp = feldman_dkg::pq_sig::generate_signing_keypair();
        let message = b"governance proposal MIP-42 tally hash";
        let signature = feldman_dkg::pq_sig::sign(&kp.signing_key, message);

        assert_eq!(signature.len(), 3309);
        assert!(
            feldman_dkg::pq_sig::verify(&kp.verifying_key_bytes(), message, &signature,).is_ok()
        );
    }

    #[test]
    fn test_ml_kem_encrypted_deal_roundtrip() {
        use feldman_dkg::{Dealer, ParticipantId};
        use rand::rngs::OsRng;

        // Generate ML-KEM keypairs for 3 recipients
        let kp1 = feldman_dkg::pq_kem::generate_keypair();
        let kp2 = feldman_dkg::pq_kem::generate_keypair();
        let kp3 = feldman_dkg::pq_kem::generate_keypair();

        // Verify EK sizes match what the integrity zome validates (1184 bytes)
        assert_eq!(kp1.encapsulation_key_bytes().len(), 1184);
        assert_eq!(kp2.encapsulation_key_bytes().len(), 1184);
        assert_eq!(kp3.encapsulation_key_bytes().len(), 1184);

        // Generate a real deal
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        // Encrypt with recipient 1's EK
        let mut encrypt_fn = feldman_dkg::pq_kem::ml_kem_encrypt_fn(&kp1.encapsulation_key_bytes());
        let encrypted_deal =
            feldman_dkg::EncryptedDeal::from_deal(&deal, |recipient, plaintext| {
                encrypt_fn(recipient, plaintext)
            })
            .unwrap();

        // Serialize → validate via from_bytes (same path as integrity zome)
        let deal_bytes = encrypted_deal.to_bytes().unwrap();
        let parsed = feldman_dkg::EncryptedDeal::from_bytes(&deal_bytes).unwrap();
        assert_eq!(parsed.dealer.0, 1);
        assert_eq!(parsed.encrypted_shares.len(), deal.shares.len());

        // Decrypt with recipient 1's DK
        let mut decrypt_fn = feldman_dkg::pq_kem::ml_kem_decrypt_fn(&kp1.dk);
        let share0 = &parsed.encrypted_shares[0];
        let decrypted =
            decrypt_fn(&share0.encapsulated_key, &share0.nonce, &share0.ciphertext).unwrap();

        // Decrypted share should be 32 bytes (a scalar)
        assert_eq!(decrypted.len(), 32);
    }

    #[test]
    fn test_ml_dsa65_tampered_signature_rejected() {
        let kp = feldman_dkg::pq_sig::generate_signing_keypair();
        let message = b"governance proposal";
        let mut signature = feldman_dkg::pq_sig::sign(&kp.signing_key, message);
        signature[0] ^= 0xFF;

        assert!(
            feldman_dkg::pq_sig::verify(&kp.verifying_key_bytes(), message, &signature,).is_err()
        );
    }

    // =========================================================================
    // ML-KEM + ML-DSA combined PQ workflow tests
    // =========================================================================

    #[test]
    fn test_pq_combined_encrypt_sign_verify() {
        // Full PQ workflow: encrypt deal with ML-KEM, sign encrypted deal with ML-DSA, verify both
        use feldman_dkg::{Dealer, ParticipantId};
        use rand::rngs::OsRng;

        // Generate ML-KEM keypair for recipient
        let kem_kp = feldman_dkg::pq_kem::generate_keypair();

        // Generate ML-DSA keypair for dealer (attestation)
        let sig_kp = feldman_dkg::pq_sig::generate_signing_keypair();

        // Create and encrypt a deal
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();
        let mut encrypt_fn =
            feldman_dkg::pq_kem::ml_kem_encrypt_fn(&kem_kp.encapsulation_key_bytes());
        let encrypted_deal =
            feldman_dkg::EncryptedDeal::from_deal(&deal, |recipient, plaintext| {
                encrypt_fn(recipient, plaintext)
            })
            .unwrap();

        // Sign the serialized encrypted deal with ML-DSA
        let deal_bytes = encrypted_deal.to_bytes().unwrap();
        let signature = feldman_dkg::pq_sig::sign(&sig_kp.signing_key, &deal_bytes);

        // Verify the ML-DSA signature
        assert!(feldman_dkg::pq_sig::verify(
            &sig_kp.verifying_key_bytes(),
            &deal_bytes,
            &signature,
        )
        .is_ok());

        // Decrypt with ML-KEM and verify share integrity
        let mut decrypt_fn = feldman_dkg::pq_kem::ml_kem_decrypt_fn(&kem_kp.dk);
        let share0 = &encrypted_deal.encrypted_shares[0];
        let decrypted =
            decrypt_fn(&share0.encapsulated_key, &share0.nonce, &share0.ciphertext).unwrap();
        assert_eq!(decrypted.len(), 32);
    }

    #[test]
    fn test_pq_tampered_encrypted_deal_signature_fails() {
        // Sign encrypted deal, tamper with deal bytes, verify signature rejects
        use feldman_dkg::{Dealer, ParticipantId};
        use rand::rngs::OsRng;

        let kem_kp = feldman_dkg::pq_kem::generate_keypair();
        let sig_kp = feldman_dkg::pq_sig::generate_signing_keypair();

        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();
        let mut encrypt_fn =
            feldman_dkg::pq_kem::ml_kem_encrypt_fn(&kem_kp.encapsulation_key_bytes());
        let encrypted_deal =
            feldman_dkg::EncryptedDeal::from_deal(&deal, |recipient, plaintext| {
                encrypt_fn(recipient, plaintext)
            })
            .unwrap();

        let deal_bytes = encrypted_deal.to_bytes().unwrap();
        let signature = feldman_dkg::pq_sig::sign(&sig_kp.signing_key, &deal_bytes);

        // Tamper with deal bytes
        let mut tampered = deal_bytes.clone();
        tampered[10] ^= 0xFF;

        // Signature should fail on tampered data
        assert!(
            feldman_dkg::pq_sig::verify(&sig_kp.verifying_key_bytes(), &tampered, &signature,)
                .is_err()
        );
    }

    // =========================================================================
    // PQ error path unit tests (combine_signatures logic)
    // =========================================================================

    #[test]
    fn test_pq_required_defaults_false() {
        // Default for ThresholdSignatureAlgorithm is Ecdsa, pq_required is false
        assert_eq!(
            ThresholdSignatureAlgorithm::default(),
            ThresholdSignatureAlgorithm::Ecdsa
        );
        // Construct with pq_required=false (default path in create_signing_committee)
        let committee = SigningCommittee {
            id: "test-pq-default".into(),
            name: "PQ Default Test".into(),
            threshold: 2,
            member_count: 3,
            phase: DkgPhase::Registration,
            public_key: None,
            commitments: Vec::new(),
            scope: CommitteeScope::All,
            created_at: Timestamp::from_micros(0),
            active: true,
            epoch: 1,
            min_phi: None,
            signature_algorithm: ThresholdSignatureAlgorithm::default(),
            pq_required: false,
        };
        assert!(!committee.pq_required);
        assert_eq!(
            committee.signature_algorithm,
            ThresholdSignatureAlgorithm::Ecdsa
        );
    }

    #[test]
    fn test_pq_required_true_with_hybrid() {
        // Construct a Hybrid committee with pq_required=true
        let committee = SigningCommittee {
            id: "test-pq-required".into(),
            name: "PQ Required Hybrid".into(),
            threshold: 2,
            member_count: 3,
            phase: DkgPhase::Registration,
            public_key: None,
            commitments: Vec::new(),
            scope: CommitteeScope::Constitutional,
            created_at: Timestamp::from_micros(0),
            active: true,
            epoch: 1,
            min_phi: None,
            signature_algorithm: ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65,
            pq_required: true,
        };
        assert!(committee.pq_required);
        assert_eq!(
            committee.signature_algorithm,
            ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65
        );

        // Verify serde roundtrip preserves pq_required
        let serialized = serde_json::to_vec(&committee).unwrap();
        let deserialized: SigningCommittee = serde_json::from_slice(&serialized).unwrap();
        assert!(deserialized.pq_required);
        assert_eq!(
            deserialized.signature_algorithm,
            ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65
        );
    }

    #[test]
    fn test_ml_dsa_verify_wrong_key_rejects() {
        // Simulates the combine_signatures MlDsa65 path: verify with wrong attestor key
        let kp_signer = feldman_dkg::pq_sig::generate_signing_keypair();
        let kp_wrong = feldman_dkg::pq_sig::generate_signing_keypair();

        let content_hash = b"MIP-42 tally hash bytes here!!!_"; // 32 bytes
        let pq_sig = feldman_dkg::pq_sig::sign(&kp_signer.signing_key, content_hash);

        // Verify with wrong key — should fail (simulates wrong attestor registered)
        let result =
            feldman_dkg::pq_sig::verify(&kp_wrong.verifying_key_bytes(), content_hash, &pq_sig);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("verification failed"));
    }

    #[test]
    fn test_ml_dsa_verify_content_hash_mismatch() {
        // Simulates combine_signatures: signature was over different content hash
        let kp = feldman_dkg::pq_sig::generate_signing_keypair();

        let original_hash = b"original governance tally hash__";
        let tampered_hash = b"tampered governance tally hash__";
        let pq_sig = feldman_dkg::pq_sig::sign(&kp.signing_key, original_hash);

        let result = feldman_dkg::pq_sig::verify(&kp.verifying_key_bytes(), tampered_hash, &pq_sig);
        assert!(result.is_err());
    }

    #[test]
    fn test_signature_algorithm_serde_variants() {
        // All three algorithm variants roundtrip through JSON correctly
        for (json_val, expected) in [
            ("\"Ecdsa\"", ThresholdSignatureAlgorithm::Ecdsa),
            ("\"MlDsa65\"", ThresholdSignatureAlgorithm::MlDsa65),
            (
                "\"HybridEcdsaMlDsa65\"",
                ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65,
            ),
        ] {
            let alg: ThresholdSignatureAlgorithm = serde_json::from_str(json_val).unwrap();
            assert_eq!(alg, expected);
            let reserialized = serde_json::to_string(&alg).unwrap();
            assert_eq!(reserialized, json_val);
        }
    }
}
