// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;
use hdk::prelude::{hdk_extern, wasm_error, WasmErrorInner};
use serde::{Deserialize, Serialize};

// =============================================================================
// CONSCIOUSNESS METRICS COORDINATOR FUNCTIONS
// =============================================================================

/// Maximum self-reported consciousness level for unsigned snapshots.
///
/// ANTI-TYRANNY: Self-reported (unsigned) snapshots are capped below the
/// Guardian threshold (0.8) to prevent consciousness score gaming.
/// Only Ed25519-signed attestations can reach Guardian tier.
/// This means self-reported scores can never unlock veto power.
const UNSIGNED_SNAPSHOT_PHI_CAP: f64 = 0.6;

/// Record a consciousness snapshot from Symthaea (DEPRECATED for governance).
///
/// Stores the Φ measurement and related metrics for an agent at a point in time.
///
/// WARNING: This function accepts self-reported values without cryptographic
/// verification. For governance actions requiring Guardian tier (0.8+), use
/// `record_consciousness_attestation` instead, which requires Ed25519 signatures.
///
/// Self-reported consciousness_level is capped at 0.6 (Steward tier) to prevent
/// gaming. Only signed attestations can unlock Guardian-tier governance powers.
#[hdk_extern]
pub fn record_consciousness_snapshot(input: RecordSnapshotInput) -> ExternResult<Record> {
    check_snapshot_input(&input).map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Cap self-reported consciousness level to prevent gaming.
    // Guardian-tier (0.8+) requires signed attestation.
    let capped_level = input.consciousness_level.min(UNSIGNED_SNAPSHOT_PHI_CAP);

    let snapshot = ConsciousnessSnapshot {
        id: format!("snapshot:{}:{}", agent_did, now.as_micros()),
        agent_did: agent_did.clone(),
        consciousness_level: capped_level,
        meta_awareness: input.meta_awareness,
        self_model_accuracy: input.self_model_accuracy,
        coherence: input.coherence,
        affective_valence: input.affective_valence,
        care_activation: input.care_activation,
        captured_at: now,
        source: input.source.unwrap_or_else(|| SYMTHAEA_SOURCE.to_string()),
        consciousness_vector: None,
    };

    let action_hash = create_entry(&EntryTypes::ConsciousnessSnapshot(snapshot.clone()))?;

    let _ = emit_signal(&BridgeSignal::ConsciousnessSnapshotRecorded {
        agent_did: agent_did.clone(),
        consciousness_level: input.consciousness_level,
    });

    // Link from agent to snapshot
    let agent_anchor = format!("agent:{}", agent_did);
    create_entry(&EntryTypes::Anchor(Anchor(agent_anchor.clone())))?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToSnapshots,
        (),
    )?;

    // Link to recent snapshots
    create_entry(&EntryTypes::Anchor(Anchor("recent_snapshots".to_string())))?;
    create_link(
        anchor_hash("recent_snapshots")?,
        action_hash.clone(),
        LinkTypes::RecentSnapshots,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Snapshot not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordSnapshotInput {
    pub consciousness_level: f64,
    pub meta_awareness: f64,
    pub self_model_accuracy: f64,
    pub coherence: f64,
    pub affective_valence: f64,
    pub care_activation: f64,
    pub source: Option<String>,
}

/// Verify consciousness gate for a governance action
///
/// Checks if the agent's current Φ meets the threshold for the requested action type.
/// Returns a gate verification record that can be referenced by governance actions.
///
/// ## Verification paths
///
/// 1. **ZKP attestation** (preferred): If the agent has submitted a
///    `ConsciousnessAttestation` with valid STARK proof bytes, the gate
///    checks the proven tier against the required threshold.  This path
///    allows Guardian-tier access with cryptographic proof.
/// 2. **Plaintext snapshot** (legacy): Falls back to the self-reported
///    `ConsciousnessSnapshot.consciousness_level` comparison.  Capped at
///    Steward tier for unsigned snapshots.
#[hdk_extern]
pub fn verify_consciousness_gate(input: VerifyGateInput) -> ExternResult<GateVerificationResult> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Use dynamic (configurable) threshold
    let dynamic_threshold = get_dynamic_consciousness_gate(&input.action_type)?;

    // --- Path 1: Check for ZKP ConsciousnessAttestation ---
    if let Some(attestation_result) =
        try_verify_zkp_attestation(&input.attestation, &agent_did, dynamic_threshold, now)?
    {
        let passed = attestation_result.0;
        let consciousness_level = attestation_result.1;
        let snapshot_id = attestation_result.2;

        let failure_reason = if passed {
            None
        } else {
            Some(format!(
                "ZKP attestation tier below gate {} for {:?}",
                dynamic_threshold, input.action_type
            ))
        };

        let gate = ConsciousnessGate {
            id: format!("gate:{}:{}", agent_did, now.as_micros()),
            agent_did: agent_did.clone(),
            action_type: input.action_type,
            snapshot_id,
            consciousness_at_verification: consciousness_level,
            required_consciousness: dynamic_threshold,
            passed,
            failure_reason: failure_reason.clone(),
            action_id: input.action_id,
            verified_at: now,
        };

        let action_hash = create_entry(&EntryTypes::ConsciousnessGate(gate.clone()))?;

        let agent_anchor = format!("agent:{}", agent_did);
        create_entry(&EntryTypes::Anchor(Anchor(agent_anchor.clone())))?;
        create_link(
            anchor_hash(&agent_anchor)?,
            action_hash,
            LinkTypes::AgentToGates,
            (),
        )?;

        return Ok(GateVerificationResult {
            passed,
            consciousness_level,
            required_consciousness: dynamic_threshold,
            action_type: input.action_type,
            failure_reason,
            gate_id: gate.id,
        });
    }

    // --- Path 2: Legacy plaintext snapshot ---
    let latest_snapshot = get_latest_agent_snapshot(&agent_did)?;

    let (_snapshot, snapshot_id, consciousness_level) = match latest_snapshot {
        Some((_record, snap)) => (
            Some(snap.clone()),
            snap.id.clone(),
            snap.consciousness_level,
        ),
        None => {
            // No snapshot available - gate fails
            let gate = ConsciousnessGate {
                id: format!("gate:{}:{}", agent_did, now.as_micros()),
                agent_did: agent_did.clone(),
                action_type: input.action_type,
                snapshot_id: "none".to_string(),
                consciousness_at_verification: 0.0,
                required_consciousness: dynamic_threshold,
                passed: false,
                failure_reason: Some("No consciousness snapshot or attestation available".to_string()),
                action_id: input.action_id.clone(),
                verified_at: now,
            };

            let action_hash = create_entry(&EntryTypes::ConsciousnessGate(gate.clone()))?;

            // Link from agent to gate
            let agent_anchor = format!("agent:{}", agent_did);
            create_entry(&EntryTypes::Anchor(Anchor(agent_anchor.clone())))?;
            create_link(
                anchor_hash(&agent_anchor)?,
                action_hash,
                LinkTypes::AgentToGates,
                (),
            )?;

            return Ok(GateVerificationResult {
                passed: false,
                consciousness_level: 0.0,
                required_consciousness: dynamic_threshold,
                action_type: input.action_type,
                failure_reason: Some("No consciousness snapshot or attestation available".to_string()),
                gate_id: gate.id,
            });
        }
    };

    let required_consciousness = dynamic_threshold;
    let passed = consciousness_level >= required_consciousness;
    let failure_reason = if passed {
        None
    } else {
        Some(format!(
            "Consciousness {} below gate {} for {:?}",
            consciousness_level, required_consciousness, input.action_type
        ))
    };

    // Create gate verification record
    let action_type_str = format!("{:?}", input.action_type);
    let gate = ConsciousnessGate {
        id: format!("gate:{}:{}", agent_did, now.as_micros()),
        agent_did: agent_did.clone(),
        action_type: input.action_type,
        snapshot_id,
        consciousness_at_verification: consciousness_level,
        required_consciousness,
        passed,
        failure_reason: failure_reason.clone(),
        action_id: input.action_id,
        verified_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ConsciousnessGate(gate.clone()))?;

    let _ = emit_signal(&BridgeSignal::ConsciousnessGateVerified {
        agent_did: agent_did.clone(),
        passed,
        action_type: action_type_str,
    });

    // Link from agent to gate
    let agent_anchor = format!("agent:{}", agent_did);
    create_entry(&EntryTypes::Anchor(Anchor(agent_anchor.clone())))?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash,
        LinkTypes::AgentToGates,
        (),
    )?;

    Ok(GateVerificationResult {
        passed,
        consciousness_level,
        required_consciousness,
        action_type: input.action_type,
        failure_reason,
        gate_id: gate.id,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyGateInput {
    pub action_type: GovernanceActionType,
    pub action_id: Option<String>,
    /// Optional ZKP consciousness attestation.  When present, the gate
    /// verifies the attestation's structural integrity and checks the
    /// proven tier instead of relying on plaintext snapshot comparison.
    #[serde(default)]
    pub attestation: Option<mycelix_bridge_common::consciousness_zkp::ConsciousnessAttestation>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GateVerificationResult {
    pub passed: bool,
    pub consciousness_level: f64,
    pub required_consciousness: f64,
    pub action_type: GovernanceActionType,
    pub failure_reason: Option<String>,
    pub gate_id: String,
}

// =============================================================================
// ZKP ATTESTATION VERIFICATION
// =============================================================================

/// Verify a caller-provided ZKP consciousness attestation.
///
/// Validates structure (proof_bytes non-empty, valid tier, non-zero
/// commitment, not expired) and checks whether the proven tier meets
/// the required threshold.
///
/// Returns `Some((passed, consciousness_level, snapshot_id))` if the
/// attestation is present, `None` if no attestation was provided
/// (caller should fall back to plaintext snapshot).
fn try_verify_zkp_attestation(
    attestation: &Option<mycelix_bridge_common::consciousness_zkp::ConsciousnessAttestation>,
    agent_did: &str,
    required_threshold: f64,
    now: Timestamp,
) -> ExternResult<Option<(bool, f64, String)>> {
    let attestation = match attestation {
        Some(a) => a,
        None => return Ok(None),
    };

    // Structural validation (size, commitment, tier range)
    if let Err(e) = attestation.validate_structure() {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid attestation structure: {}",
            e
        ))));
    }

    // Expiry check
    let now_secs = now.as_micros() as u64 / 1_000_000;
    if attestation.is_expired(now_secs) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Consciousness attestation has expired".into()
        )));
    }

    // Map tier (0-4) to consciousness level using the attestation's
    // minimum threshold for that tier.
    let consciousness_level = attestation.min_threshold();
    let snapshot_id = format!(
        "attestation:{}:tier{}:{}",
        agent_did, attestation.tier, attestation.generated_at
    );

    // Log verification status.  Off-chain verified attestations
    // (with verifier_signature) provide the strongest guarantee.
    if !attestation.is_verified() {
        debug!(
            "Attestation for {} has valid structure but no off-chain verification (tier {})",
            agent_did, attestation.tier
        );
    }

    let passed = consciousness_level >= required_threshold;
    Ok(Some((passed, consciousness_level, snapshot_id)))
}

/// Assess value alignment of a proposal
///
/// Evaluates how well a proposal aligns with the Eight Harmonies,
/// using the agent's current consciousness state.
#[hdk_extern]
pub fn assess_value_alignment(input: AssessAlignmentInput) -> ExternResult<Record> {
    check_alignment_input(&input).map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get the latest snapshot for this agent
    let latest_snapshot = get_latest_agent_snapshot(&agent_did)?;

    let snapshot_id = match &latest_snapshot {
        Some((_, snap)) => snap.id.clone(),
        None => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "No consciousness snapshot available for alignment assessment".into()
            )));
        }
    };

    // Calculate harmony scores (in production, this would analyze the proposal content)
    let harmony_scores = calculate_harmony_scores(&input.proposal_content);

    // Calculate overall alignment
    let overall_alignment: f64 =
        harmony_scores.iter().map(|h| h.score).sum::<f64>() / harmony_scores.len() as f64;

    // Detect violations (scores below threshold)
    let violations: Vec<String> = harmony_scores
        .iter()
        .filter(|h| h.score < -0.3)
        .map(|h| format!("{}: severe misalignment ({:.2})", h.harmony, h.score))
        .collect();

    // Calculate authenticity (based on CARE activation if available)
    let authenticity = match &latest_snapshot {
        Some((_, snap)) => snap.care_activation,
        None => 0.5,
    };

    // Determine recommendation
    let recommendation = determine_recommendation(overall_alignment, authenticity, &violations);
    let recommendation_str = format!("{:?}", recommendation);

    // Create assessment entry
    let assessment = ValueAlignmentAssessment {
        id: format!(
            "alignment:{}:{}:{}",
            agent_did,
            input.proposal_id,
            now.as_micros()
        ),
        agent_did: agent_did.clone(),
        proposal_id: input.proposal_id.clone(),
        overall_alignment,
        harmony_scores,
        authenticity,
        violations,
        recommendation,
        snapshot_id,
        assessed_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ValueAlignmentAssessment(assessment))?;

    let _ = emit_signal(&BridgeSignal::ValueAlignmentAssessed {
        proposal_id: input.proposal_id.clone(),
        agent_did: agent_did.clone(),
        recommendation: recommendation_str,
    });

    // Link from proposal to alignment
    let proposal_anchor = format!("proposal:{}", input.proposal_id);
    create_entry(&EntryTypes::Anchor(Anchor(proposal_anchor.clone())))?;
    create_link(
        anchor_hash(&proposal_anchor)?,
        action_hash.clone(),
        LinkTypes::ProposalToAlignments,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Assessment not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AssessAlignmentInput {
    pub proposal_id: String,
    pub proposal_content: String,
}

/// Get agent's consciousness history
#[hdk_extern]
pub fn get_agent_consciousness_history(agent_did: String) -> ExternResult<Option<Record>> {
    let agent_anchor = format!("agent:history:{}", agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToHistory)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// Get recent consciousness snapshots for an agent
#[hdk_extern]
pub fn get_agent_snapshots(input: GetAgentSnapshotsInput) -> ExternResult<Vec<Record>> {
    let agent_anchor = format!("agent:{}", input.agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToSnapshots)?,
        GetStrategy::default(),
    )?;

    let mut snapshots = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            snapshots.push(record);
        }

        if snapshots.len() >= input.limit.unwrap_or(50) as usize {
            break;
        }
    }

    // Sort by timestamp (most recent first)
    snapshots.sort_by(|a, b| b.action().timestamp().cmp(&a.action().timestamp()));

    Ok(snapshots)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetAgentSnapshotsInput {
    pub agent_did: String,
    pub limit: Option<u32>,
}

/// Get value alignments for a proposal
#[hdk_extern]
pub fn get_proposal_alignments(proposal_id: String) -> ExternResult<Vec<Record>> {
    let proposal_anchor = format!("proposal:{}", proposal_id);
    let anchor = match anchor_hash(&proposal_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ProposalToAlignments)?,
        GetStrategy::default(),
    )?;

    let mut alignments = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            alignments.push(record);
        }
    }

    Ok(alignments)
}

/// Get current consciousness gate requirements (dynamic, reads from GovernanceConsciousnessConfig)
#[hdk_extern]
pub fn get_consciousness_thresholds(_: ()) -> ExternResult<ConsciousnessThresholdSummary> {
    let config = get_current_consciousness_config()?;
    Ok(ConsciousnessThresholdSummary {
        basic: config.consciousness_gate_basic,
        proposal_submission: config.consciousness_gate_proposal,
        voting: config.consciousness_gate_voting,
        constitutional: config.consciousness_gate_constitutional,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConsciousnessThresholdSummary {
    pub basic: f64,
    pub proposal_submission: f64,
    pub voting: f64,
    pub constitutional: f64,
}
