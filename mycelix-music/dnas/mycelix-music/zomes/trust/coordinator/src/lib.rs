// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust Coordinator Zome
//!
//! Implements the trust layer for Mycelix Music:
//! - Web-of-trust artist verification
//! - CDN node reputation management
//! - Byzantine behavior detection and reporting
//! - Integration point for Mycelix-Core PoGQ
//! Holochain 0.6 compatible (hdk 0.6)

use hdk::prelude::*;
use trust_integrity::*;
use mycelix_bridge_common::{
    gate_civic, civic_requirement_basic, civic_requirement_voting, GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<()> {
    gate_civic("music_bridge", requirement, action_name).map(|_| ())
}

/// Helper to ensure a path exists and return its entry hash
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

/// Create a trust claim (vouch for another agent)
#[hdk_extern]
pub fn create_trust_claim(input: CreateTrustClaimInput) -> ExternResult<ActionHash> {
    require_consciousness(&civic_requirement_basic(), "create_trust_claim")?;
    let my_agent = agent_info()?.agent_initial_pubkey;

    let claim = TrustClaim {
        from: my_agent.clone(),
        to: input.to.clone(),
        claim_type: input.claim_type,
        confidence_bps: input.confidence_bps,
        evidence: input.evidence,
        created_at: sys_time()?,
        expires_at: input.expires_at,
        active: true,
    };

    let action_hash = create_entry(&EntryTypes::TrustClaim(claim))?;

    // Link from claimer
    let from_path = Path::from(format!("claims_made/{}", my_agent));
    let from_hash = ensure_path(from_path, LinkTypes::AgentToClaimsMade)?;
    create_link(
        from_hash,
        action_hash.clone(),
        LinkTypes::AgentToClaimsMade,
        (),
    )?;

    // Link to recipient
    let to_path = Path::from(format!("claims_received/{}", input.to));
    let to_hash = ensure_path(to_path, LinkTypes::AgentToClaimsReceived)?;
    create_link(
        to_hash,
        action_hash.clone(),
        LinkTypes::AgentToClaimsReceived,
        (),
    )?;

    // Recompute verification status for recipient
    recompute_verification(input.to)?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateTrustClaimInput {
    pub to: AgentPubKey,
    pub claim_type: TrustClaimType,
    pub confidence_bps: u32,
    pub evidence: Option<String>,
    pub expires_at: Option<Timestamp>,
}

/// Get trust claims about an agent
#[hdk_extern]
pub fn get_trust_claims(agent: AgentPubKey) -> ExternResult<Vec<TrustClaim>> {
    let to_path = Path::from(format!("claims_received/{}", agent));
    let typed_path = to_path.typed(LinkTypes::AgentToClaimsReceived)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToClaimsReceived)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(claim) = record
                    .entry()
                    .to_app_option::<TrustClaim>()
                    .map_err(|e| wasm_error!(e))?
                {
                    if claim.active {
                        claims.push(claim);
                    }
                }
            }
        }
    }

    Ok(claims)
}

/// Compute and store verification status
fn recompute_verification(agent: AgentPubKey) -> ExternResult<()> {
    let claims = get_trust_claims(agent.clone())?;

    // Calculate trust score
    let vouch_count = claims.len() as u32;
    let total_confidence: u32 = claims.iter().map(|c| c.confidence_bps).sum();
    let trust_score = if vouch_count > 0 {
        total_confidence / vouch_count
    } else {
        0
    };

    // Determine tier
    let tier = if vouch_count >= 10 && trust_score >= 800 {
        VerificationTier::Trusted
    } else if vouch_count >= 3 {
        VerificationTier::CommunityVerified
    } else {
        VerificationTier::Unverified
    };

    let status = VerificationStatus {
        artist: agent.clone(),
        trust_score,
        tier,
        vouch_count,
        computed_at: sys_time()?,
    };

    let action_hash = create_entry(&EntryTypes::VerificationStatus(status))?;

    // Link to agent
    let status_path = Path::from(format!("verification/{}", agent));
    let status_hash = ensure_path(status_path, LinkTypes::AgentToVerification)?;
    create_link(
        status_hash,
        action_hash,
        LinkTypes::AgentToVerification,
        (),
    )?;

    Ok(())
}

/// Get verification status for an agent
#[hdk_extern]
pub fn get_verification_status(agent: AgentPubKey) -> ExternResult<Option<VerificationStatus>> {
    let status_path = Path::from(format!("verification/{}", agent));
    let typed_path = status_path.typed(LinkTypes::AgentToVerification)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToVerification)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                return Ok(record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?);
            }
        }
    }

    Ok(None)
}

/// Register as a CDN node
#[hdk_extern]
pub fn register_cdn_node(input: RegisterCdnNodeInput) -> ExternResult<ActionHash> {
    require_consciousness(&civic_requirement_voting(), "register_cdn_node")?;
    let my_agent = agent_info()?.agent_initial_pubkey;

    let reputation = CdnNodeReputation {
        node: my_agent.clone(),
        eth_address: input.eth_address,
        ipfs_peer_id: input.ipfs_peer_id,
        region: input.region,
        bytes_served: 0,
        successful_requests: 0,
        failed_requests: 0,
        avg_latency_ms: 0,
        uptime_bps: 1000, // Start at 100%
        pogq_score: 1.0,  // Start with perfect score
        last_active: sys_time()?,
        stake_amount: input.stake_amount,
        slash_count: 0,
    };

    let action_hash = create_entry(&EntryTypes::CdnNodeReputation(reputation))?;

    // Link to node
    let node_path = Path::from(format!("cdn_node/{}", my_agent));
    let node_hash = ensure_path(node_path, LinkTypes::NodeToReputation)?;
    create_link(
        node_hash,
        action_hash.clone(),
        LinkTypes::NodeToReputation,
        (),
    )?;

    // Add to all nodes list
    let all_nodes_path = Path::from("all_cdn_nodes");
    let all_nodes_hash = ensure_path(all_nodes_path, LinkTypes::NodeToReputation)?;
    create_link(
        all_nodes_hash,
        action_hash.clone(),
        LinkTypes::NodeToReputation,
        (),
    )?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterCdnNodeInput {
    pub eth_address: String,
    pub ipfs_peer_id: String,
    pub region: String,
    pub stake_amount: u64,
}

/// Get CDN node reputation
#[hdk_extern]
pub fn get_cdn_reputation(node: AgentPubKey) -> ExternResult<Option<CdnNodeReputation>> {
    let node_path = Path::from(format!("cdn_node/{}", node));
    let typed_path = node_path.typed(LinkTypes::NodeToReputation)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::NodeToReputation)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                return Ok(record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?);
            }
        }
    }

    Ok(None)
}

/// Get all CDN nodes (for routing)
#[hdk_extern]
pub fn get_all_cdn_nodes(_: ()) -> ExternResult<Vec<CdnNodeReputation>> {
    let all_nodes_path = Path::from("all_cdn_nodes");
    let typed_path = all_nodes_path.typed(LinkTypes::NodeToReputation)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::NodeToReputation)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut nodes = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(rep) = record
                    .entry()
                    .to_app_option::<CdnNodeReputation>()
                    .map_err(|e| wasm_error!(e))?
                {
                    nodes.push(rep);
                }
            }
        }
    }

    Ok(nodes)
}

/// Submit a service quality report
#[hdk_extern]
pub fn submit_quality_report(input: SubmitQualityReportInput) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let report = ServiceQualityReport {
        reporter: my_agent.clone(),
        node: input.node.clone(),
        song_hash: input.song_hash,
        latency_ms: input.latency_ms,
        success: input.success,
        error_code: input.error_code,
        reported_at: sys_time()?,
    };

    let action_hash = create_entry(&EntryTypes::ServiceQualityReport(report))?;

    // Link to reporter
    let reports_path = Path::from(format!("quality_reports/{}", my_agent));
    let reports_hash = ensure_path(reports_path, LinkTypes::AgentToReports)?;
    create_link(
        reports_hash,
        action_hash.clone(),
        LinkTypes::AgentToReports,
        (),
    )?;

    // Update CDN node reputation based on report
    update_cdn_reputation(input.node, input.success, input.latency_ms)?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitQualityReportInput {
    pub node: AgentPubKey,
    pub song_hash: ActionHash,
    pub latency_ms: u32,
    pub success: bool,
    pub error_code: Option<String>,
}

/// Update CDN reputation based on service report
fn update_cdn_reputation(node: AgentPubKey, success: bool, latency_ms: u32) -> ExternResult<()> {
    let node_path = Path::from(format!("cdn_node/{}", node));
    let typed_path = node_path.typed(LinkTypes::NodeToReputation)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::NodeToReputation)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(mut rep) = record
                    .entry()
                    .to_app_option::<CdnNodeReputation>()
                    .map_err(|e| wasm_error!(e))?
                {
                    // Update stats
                    if success {
                        rep.successful_requests += 1;
                        // Rolling average for latency
                        let total_requests = rep.successful_requests + rep.failed_requests;
                        rep.avg_latency_ms = ((rep.avg_latency_ms as u64 * (total_requests - 1)
                            + latency_ms as u64)
                            / total_requests) as u32;
                    } else {
                        rep.failed_requests += 1;
                    }

                    // Recalculate uptime
                    let total = rep.successful_requests + rep.failed_requests;
                    rep.uptime_bps = ((rep.successful_requests as f64 / total as f64) * 1000.0) as u32;

                    // Simple PoGQ score based on uptime and latency
                    let uptime_factor = rep.uptime_bps as f64 / 1000.0;
                    let latency_factor = if rep.avg_latency_ms < 100 {
                        1.0
                    } else if rep.avg_latency_ms < 500 {
                        0.8
                    } else {
                        0.5
                    };
                    rep.pogq_score = uptime_factor * latency_factor;

                    rep.last_active = sys_time()?;

                    // Update entry
                    let new_hash =
                        update_entry(action_hash, &EntryTypes::CdnNodeReputation(rep))?;

                    // Update link
                    create_link(
                        typed_path.path_entry_hash()?,
                        new_hash,
                        LinkTypes::NodeToReputation,
                        (),
                    )?;
                }
            }
        }
    }

    Ok(())
}

/// Report Byzantine behavior
#[hdk_extern]
pub fn report_byzantine_behavior(input: ReportByzantineInput) -> ExternResult<ActionHash> {
    require_consciousness(&civic_requirement_voting(), "report_byzantine_behavior")?;
    let my_agent = agent_info()?.agent_initial_pubkey;

    let report = ByzantineReport {
        reporter: my_agent,
        accused: input.accused,
        behavior_type: input.behavior_type,
        evidence: input.evidence,
        severity: input.severity,
        reported_at: sys_time()?,
        status: ReportStatus::Pending,
    };

    let action_hash = create_entry(&EntryTypes::ByzantineReport(report))?;

    // Link to byzantine reports anchor
    let reports_path = Path::from("byzantine_reports");
    let reports_hash = ensure_path(reports_path, LinkTypes::ByzantineReports)?;
    create_link(
        reports_hash,
        action_hash.clone(),
        LinkTypes::ByzantineReports,
        (),
    )?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReportByzantineInput {
    pub accused: AgentPubKey,
    pub behavior_type: ByzantineBehavior,
    pub evidence: String,
    pub severity: u8,
}

/// Get best CDN nodes for a region (for content routing)
#[hdk_extern]
pub fn get_best_nodes_for_region(region: String) -> ExternResult<Vec<CdnNodeReputation>> {
    let all_nodes = get_all_cdn_nodes(())?;

    // Filter by region and sort by PoGQ score
    let mut regional_nodes: Vec<CdnNodeReputation> = all_nodes
        .into_iter()
        .filter(|n| n.region == region || region == "global")
        .collect();

    regional_nodes.sort_by(|a, b| {
        b.pogq_score
            .partial_cmp(&a.pogq_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return top 5
    Ok(regional_nodes.into_iter().take(5).collect())
}

/// Get my trust claims (made by me)
#[hdk_extern]
pub fn get_my_trust_claims(_: ()) -> ExternResult<Vec<TrustClaim>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let from_path = Path::from(format!("claims_made/{}", my_agent));
    let typed_path = from_path.typed(LinkTypes::AgentToClaimsMade)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToClaimsMade)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(claim) = record
                    .entry()
                    .to_app_option::<TrustClaim>()
                    .map_err(|e| wasm_error!(e))?
                {
                    claims.push(claim);
                }
            }
        }
    }

    Ok(claims)
}

/// Revoke a trust claim
#[hdk_extern]
pub fn revoke_trust_claim(claim_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(claim_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Claim not found".to_string())))?;

    let mut claim: TrustClaim = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid claim".to_string())))?;

    // Verify ownership
    let my_agent = agent_info()?.agent_initial_pubkey;
    if claim.from != my_agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only revoke own claims".to_string()
        )));
    }

    // Mark as inactive
    claim.active = false;

    let new_hash = update_entry(claim_hash, &EntryTypes::TrustClaim(claim.clone()))?;

    // Recompute verification for the recipient
    recompute_verification(claim.to)?;

    Ok(new_hash)
}
