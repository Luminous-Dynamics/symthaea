use super::*;

// =============================================================================
// CONSCIOUSNESS METRICS COORDINATOR FUNCTIONS
// =============================================================================

/// Record a consciousness snapshot from Symthaea
///
/// Stores the Φ measurement and related metrics for an agent at a point in time.
/// Used to establish consciousness state before governance actions.
#[hdk_extern]
pub fn record_consciousness_snapshot(input: RecordSnapshotInput) -> ExternResult<Record> {
    check_snapshot_input(&input).map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    let snapshot = ConsciousnessSnapshot {
        id: format!("snapshot:{}:{}", agent_did, now.as_micros()),
        agent_did: agent_did.clone(),
        consciousness_level: input.consciousness_level,
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
#[hdk_extern]
pub fn verify_consciousness_gate(input: VerifyGateInput) -> ExternResult<GateVerificationResult> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get the latest snapshot for this agent
    let latest_snapshot = get_latest_agent_snapshot(&agent_did)?;

    // Use dynamic (configurable) threshold
    let dynamic_threshold = get_dynamic_consciousness_gate(&input.action_type)?;

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
                failure_reason: Some("No consciousness snapshot available".to_string()),
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
                failure_reason: Some("No consciousness snapshot available".to_string()),
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
