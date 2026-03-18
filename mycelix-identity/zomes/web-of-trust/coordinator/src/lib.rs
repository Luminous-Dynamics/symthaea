use hdk::prelude::*;
use web_of_trust_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_voting, requirement_for_basic,
    GovernanceEligibility, GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("identity_bridge", requirement, action_name)
}

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))
}

/// Helper to ensure an anchor entry exists and return its hash
fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))?;
    anchor_hash(anchor_str)
}

/// Attest trust to another agent (Citizen+).
#[hdk_extern]
pub fn attest_trust(attestation: TrustAttestation) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "attest_trust")?;

    let action_hash = create_entry(&EntryTypes::TrustAttestation(attestation.clone()))?;
    let agent = agent_info()?.agent_initial_pubkey;

    // Link from attestor
    create_link(agent, action_hash.clone(), LinkTypes::AttestorToAttestations, ())?;

    // Link from subject
    create_link(attestation.subject, action_hash.clone(), LinkTypes::SubjectToAttestations, ())?;

    // Link from all attestations anchor
    let all_anchor = ensure_anchor("all_attestations")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllAttestations, ())?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}

/// Revoke a trust attestation (attestor only).
#[hdk_extern]
pub fn revoke_trust(revocation: TrustRevocation) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "revoke_trust")?;

    // Verify caller is the attestor
    let attestation_record = get(revocation.attestation_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Attestation not found".into())))?;
    let attestor = attestation_record.action().author().clone();
    let caller = agent_info()?.agent_initial_pubkey;
    if attestor != caller {
        return Err(wasm_error!(WasmErrorInner::Guest("Only attestor can revoke".into())));
    }

    let action_hash = create_entry(&EntryTypes::TrustRevocation(revocation.clone()))?;
    create_link(
        revocation.attestation_hash,
        action_hash.clone(),
        LinkTypes::AttestationToRevocations,
        (),
    )?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TrustChainInput {
    pub from: AgentPubKey,
    pub to: AgentPubKey,
    pub max_hops: u32,
}

/// Get trust chain between two agents (BFS resolution).
#[hdk_extern]
pub fn get_trust_chain(input: TrustChainInput) -> ExternResult<Vec<TrustAttestation>> {
    let mut chain = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut queue = vec![(input.from.clone(), 0u32)];
    visited.insert(input.from.clone());

    while let Some((current, depth)) = queue.pop() {
        if depth > input.max_hops {
            continue;
        }
        let links = get_links(
            LinkQuery::try_new(current.clone(), LinkTypes::AttestorToAttestations)?,
            GetStrategy::default(),
        )?;

        for link in links {
            if let Some(target) = link.target.into_action_hash() {
                if let Some(record) = get(target, GetOptions::default())? {
                    if let Some(att) = record.entry().to_app_option::<TrustAttestation>().ok().flatten() {
                        if !visited.contains(&att.subject) {
                            visited.insert(att.subject.clone());
                            chain.push(att.clone());
                            if att.subject == input.to {
                                return Ok(chain);
                            }
                            queue.push((att.subject, depth + 1));
                        }
                    }
                }
            }
        }
    }
    Ok(chain)
}

/// Get aggregate trust score for an agent.
#[hdk_extern]
pub fn get_trust_score(subject: AgentPubKey) -> ExternResult<f64> {
    let links = get_links(
        LinkQuery::try_new(subject, LinkTypes::SubjectToAttestations)?,
        GetStrategy::default(),
    )?;

    let mut total_trust = 0.0;
    let mut count = 0u32;

    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target.clone(), GetOptions::default())? {
                if let Some(att) = record.entry().to_app_option::<TrustAttestation>().ok().flatten() {
                    // Check for revocations
                    let revocations = get_links(
                        LinkQuery::try_new(target, LinkTypes::AttestationToRevocations)?,
                        GetStrategy::default(),
                    )?;
                    if revocations.is_empty() {
                        total_trust += att.trust_score;
                        count += 1;
                    }
                }
            }
        }
    }

    Ok(if count > 0 { total_trust / count as f64 } else { 0.0 })
}

/// Offline-capable trust verification from cached chain.
#[hdk_extern]
pub fn offline_verify_trust(input: TrustChainInput) -> ExternResult<f64> {
    let chain = get_trust_chain(input)?;
    if chain.is_empty() {
        return Ok(0.0);
    }

    // Compute transitive trust with 0.8/hop decay
    let mut trust = 1.0;
    for att in &chain {
        trust *= att.trust_score * 0.8;
    }
    Ok(trust)
}
