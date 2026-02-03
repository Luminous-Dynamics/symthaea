//! Councils Coordinator Zome
//! Holonic governance with collective sensing
//!
//! Philosophy: Governance is nested - councils within councils,
//! each a holon containing both individual wisdom and collective emergence.
//! The HolonicMirror reflects the fractal health of the whole.

use hdk::prelude::*;
use councils_integrity::*;

// ============================================================================
// REAL-TIME SIGNALS
// ============================================================================

/// Signal types for real-time council updates
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum CouncilSignal {
    /// A new council was created
    CouncilCreated {
        council_id: String,
        council_name: String,
        council_type: String,
        parent_id: Option<String>,
    },

    /// A member joined a council
    MemberJoined {
        council_id: String,
        member_did: String,
        role: String,
        phi_score: f64,
    },

    /// A holonic reflection was generated
    HolonicReflectionGenerated {
        council_id: String,
        reflection_id: String,
        health_score: f64,
        vitality_trend: String,
        risk_count: usize,
    },

    /// A council decision was recorded
    DecisionRecorded {
        council_id: String,
        decision_id: String,
        title: String,
        passed: bool,
        phi_weighted_result: f64,
    },

    /// Council status changed
    CouncilStatusChanged {
        council_id: String,
        old_status: String,
        new_status: String,
    },
}

/// Emit a council signal to connected clients
fn emit_council_signal(signal: CouncilSignal) -> ExternResult<()> {
    emit_signal(&signal)?;
    Ok(())
}

/// Helper to create anchor hashes
fn anchor_hash(anchor_name: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_name.to_string());
    hash_entry(&anchor)
}

/// Initialize the zome with anchor entries
#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Create root anchor for all councils
    let anchor = Anchor("all_councils".to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    Ok(InitCallbackResult::Pass)
}

// ============================================================================
// COUNCIL MANAGEMENT
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCouncilInput {
    pub name: String,
    pub purpose: String,
    pub council_type: CouncilType,
    pub parent_council_id: Option<String>,
    pub phi_threshold: f64,
    pub quorum: f64,
    pub supermajority: f64,
    pub can_spawn_children: bool,
    pub max_delegation_depth: u8,
}

/// Create a new council
#[hdk_extern]
pub fn create_council(input: CreateCouncilInput) -> ExternResult<Record> {
    // Input validation
    if input.name.is_empty() || input.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Council name must be 1-256 characters".into())));
    }
    if input.purpose.is_empty() || input.purpose.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Council purpose must be 1-4096 characters".into())));
    }
    if input.phi_threshold < 0.0 || input.phi_threshold > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Phi threshold must be between 0.0 and 1.0".into())));
    }
    if input.quorum <= 0.0 || input.quorum > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Quorum must be between 0.0 (exclusive) and 1.0 (inclusive)".into())));
    }
    if input.supermajority <= 0.0 || input.supermajority > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Supermajority must be between 0.0 (exclusive) and 1.0 (inclusive)".into())));
    }
    if let Some(ref parent_id) = input.parent_council_id {
        if parent_id.is_empty() || parent_id.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest("Parent council ID must be 1-256 characters".into())));
        }
    }

    let timestamp = sys_time()?;

    // Generate unique ID
    let id = format!(
        "council-{}-{}",
        input.name.to_lowercase().replace(' ', "-"),
        timestamp.as_micros()
    );

    let council = Council {
        id: id.clone(),
        name: input.name,
        purpose: input.purpose,
        council_type: input.council_type.clone(),
        parent_council_id: input.parent_council_id.clone(),
        phi_threshold: input.phi_threshold,
        quorum: input.quorum,
        supermajority: input.supermajority,
        can_spawn_children: input.can_spawn_children,
        max_delegation_depth: input.max_delegation_depth,
        status: CouncilStatus::Active,
        created_at: timestamp,
        last_activity: timestamp,
    };

    let action_hash = create_entry(&EntryTypes::Council(council.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created council".into())))?;

    // Link from root anchor
    let all_councils_hash = anchor_hash("all_councils")?;
    create_link(
        all_councils_hash.clone(),
        action_hash.clone(),
        LinkTypes::AllCouncils,
        id.as_bytes().to_vec(),
    )?;

    // Link from parent if sub-council
    if let Some(parent_id) = input.parent_council_id {
        if let Some(parent_record) = get_council_by_id(parent_id.clone())? {
            let parent_action_hash = parent_record.action_hashed().hash.clone();
            create_link(
                parent_action_hash,
                action_hash.clone(),
                LinkTypes::ParentToChild,
                id.as_bytes().to_vec(),
            )?;
        }
    }

    // Link by council type
    let type_anchor_name = match input.council_type {
        CouncilType::Root => "type_root",
        CouncilType::Domain { .. } => "type_domain",
        CouncilType::Regional { .. } => "type_regional",
        CouncilType::WorkingGroup { .. } => "type_working_group",
        CouncilType::Advisory => "type_advisory",
        CouncilType::Emergency { .. } => "type_emergency",
    };
    let type_anchor = Anchor(type_anchor_name.to_string());
    let type_hash = hash_entry(&type_anchor)?;
    create_entry(&EntryTypes::Anchor(type_anchor))?;
    create_link(
        type_hash,
        action_hash,
        LinkTypes::TypeToCouncil,
        id.as_bytes().to_vec(),
    )?;

    // Emit real-time signal for connected clients
    let _ = emit_council_signal(CouncilSignal::CouncilCreated {
        council_id: council.id.clone(),
        council_name: council.name.clone(),
        council_type: format!("{:?}", council.council_type),
        parent_id: council.parent_council_id.clone(),
    });

    Ok(record)
}

/// Get a council by ID
#[hdk_extern]
pub fn get_council_by_id(council_id: String) -> ExternResult<Option<Record>> {
    let all_councils_hash = anchor_hash("all_councils")?;

    let links = get_links(
        LinkQuery::try_new(all_councils_hash, LinkTypes::AllCouncils)?,
        GetStrategy::default(),
    )?;

    for link in links {
        // Check if the tag matches the council_id
        if let Ok(tag_str) = String::from_utf8(link.tag.0.clone()) {
            if tag_str == council_id {
                if let Some(target) = link.target.into_action_hash() {
                    if let Some(record) = get(target, GetOptions::default())? {
                        return Ok(Some(record));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Get all councils
#[hdk_extern]
pub fn get_all_councils(_: ()) -> ExternResult<Vec<Record>> {
    let all_councils_hash = anchor_hash("all_councils")?;

    let links = get_links(
        LinkQuery::try_new(all_councils_hash, LinkTypes::AllCouncils)?,
        GetStrategy::default(),
    )?;

    let mut councils = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                councils.push(record);
            }
        }
    }

    Ok(councils)
}

/// Get child councils
#[hdk_extern]
pub fn get_child_councils(council_id: String) -> ExternResult<Vec<Record>> {
    let parent_record = get_council_by_id(council_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Council not found".into())))?;

    let parent_hash = parent_record.action_hashed().hash.clone();

    let links = get_links(
        LinkQuery::try_new(parent_hash, LinkTypes::ParentToChild)?,
        GetStrategy::default(),
    )?;

    let mut children = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                children.push(record);
            }
        }
    }

    Ok(children)
}

// ============================================================================
// MEMBERSHIP MANAGEMENT
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct JoinCouncilInput {
    pub council_id: String,
    pub member_did: String,
    pub role: MemberRole,
    pub phi_score: f64,
}

/// Join a council
#[hdk_extern]
pub fn join_council(input: JoinCouncilInput) -> ExternResult<Record> {
    // Input validation
    if input.council_id.is_empty() || input.council_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Council ID must be 1-256 characters".into())));
    }
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Member DID must be 1-256 characters".into())));
    }
    if input.phi_score < 0.0 || input.phi_score > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Phi score must be between 0.0 and 1.0".into())));
    }

    let timestamp = sys_time()?;

    // Get council to verify it exists and check phi threshold
    let council_record = get_council_by_id(input.council_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Council not found".into())))?;

    let council: Council = council_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid council entry".into())))?;

    // Verify phi threshold
    if input.phi_score < council.phi_threshold {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Phi score {} below council threshold {}",
                input.phi_score, council.phi_threshold
            )
        )));
    }

    // Calculate voting weight (phi-weighted)
    let base_weight = match input.role {
        MemberRole::Member => 1.0,
        MemberRole::Facilitator => 1.0, // Facilitators don't get extra voting weight
        MemberRole::Steward => 1.0,
        MemberRole::Observer => 0.0,
        MemberRole::Delegate { .. } => 1.0,
    };
    let voting_weight = base_weight * input.phi_score;

    let membership_id = format!(
        "membership-{}-{}-{}",
        input.council_id, input.member_did, timestamp.as_micros()
    );

    // Capture role string for signal before moving
    let role_str = format!("{:?}", input.role);

    let membership = CouncilMembership {
        id: membership_id.clone(),
        council_id: input.council_id.clone(),
        member_did: input.member_did.clone(),
        role: input.role,
        phi_score: input.phi_score,
        voting_weight,
        can_delegate: true,
        status: MembershipStatus::Active,
        joined_at: timestamp,
        last_participation: timestamp,
    };

    let action_hash = create_entry(&EntryTypes::CouncilMembership(membership))?;
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created membership".into())))?;

    // Link council to member
    let council_hash = council_record.action_hashed().hash.clone();
    create_link(
        council_hash,
        action_hash.clone(),
        LinkTypes::CouncilToMember,
        input.member_did.as_bytes().to_vec(),
    )?;

    // Link member to council
    let member_anchor = Anchor(format!("member_{}", input.member_did));
    let member_hash = hash_entry(&member_anchor)?;
    create_entry(&EntryTypes::Anchor(member_anchor))?;
    create_link(
        member_hash,
        action_hash,
        LinkTypes::MemberToCouncil,
        input.council_id.as_bytes().to_vec(),
    )?;

    // Emit real-time signal for connected clients
    let _ = emit_council_signal(CouncilSignal::MemberJoined {
        council_id: input.council_id.clone(),
        member_did: input.member_did.clone(),
        role: role_str,
        phi_score: input.phi_score,
    });

    Ok(record)
}

/// Get council members
#[hdk_extern]
pub fn get_council_members(council_id: String) -> ExternResult<Vec<Record>> {
    let council_record = get_council_by_id(council_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Council not found".into())))?;

    let council_hash = council_record.action_hashed().hash.clone();

    let links = get_links(
        LinkQuery::try_new(council_hash, LinkTypes::CouncilToMember)?,
        GetStrategy::default(),
    )?;

    let mut members = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                members.push(record);
            }
        }
    }

    Ok(members)
}

// ============================================================================
// HOLONIC MIRROR - COLLECTIVE SENSING FOR NESTED GROUPS
// ============================================================================

/// Generate a holonic reflection for a council
///
/// Philosophy: The Mirror sees the fractal health of governance.
/// Each reflection includes both the council itself and aggregate
/// sensing of all child councils.
#[hdk_extern]
pub fn reflect_on_council(council_id: String) -> ExternResult<Record> {
    let timestamp = sys_time()?;

    // Get council
    let council_record = get_council_by_id(council_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Council not found".into())))?;

    let council: Council = council_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid council entry".into())))?;

    // Get members
    let member_records = get_council_members(council_id.clone())?;
    let members: Vec<CouncilMembership> = member_records
        .iter()
        .filter_map(|r| r.entry().to_app_option().ok().flatten())
        .collect();

    // Calculate member metrics
    let member_count = members.len() as u64;
    let active_members: Vec<&CouncilMembership> = members
        .iter()
        .filter(|m| matches!(m.status, MembershipStatus::Active))
        .collect();

    let phi_qualified_count = active_members
        .iter()
        .filter(|m| m.phi_score >= council.phi_threshold)
        .count() as u64;

    let phi_scores: Vec<f64> = active_members.iter().map(|m| m.phi_score).collect();
    let average_phi = if phi_scores.is_empty() {
        0.0
    } else {
        phi_scores.iter().sum::<f64>() / phi_scores.len() as f64
    };

    let phi_distribution = calculate_phi_distribution(&phi_scores);

    // Get child councils
    let child_records = get_child_councils(council_id.clone())?;
    let child_count = child_records.len() as u64;

    // Calculate child health if there are children
    let child_health = if child_count > 0 {
        Some(aggregate_child_health(&child_records)?)
    } else {
        None
    };

    // Calculate holon depth
    let holon_depth = calculate_holon_depth(&council);

    // Calculate participation rate (simplified - would need activity tracking)
    let participation_rate = if member_count > 0 {
        active_members.len() as f64 / member_count as f64
    } else {
        0.0
    };

    // Calculate harmony presence (placeholder - would need contribution analysis)
    let harmony_presence = vec![
        HarmonyPresence {
            harmony: "ResonantCoherence".to_string(),
            presence: 0.7,
            trend: TrendDirection::Stable,
        },
        HarmonyPresence {
            harmony: "PanSentientFlourishing".to_string(),
            presence: 0.6,
            trend: TrendDirection::Rising,
        },
        HarmonyPresence {
            harmony: "IntegralWisdom".to_string(),
            presence: 0.5,
            trend: TrendDirection::Stable,
        },
    ];

    let harmony_coverage = harmony_presence.iter().map(|h| h.presence).sum::<f64>()
        / 7.0; // 7 harmonies total

    let absent_harmonies = vec![
        "InfinitePlay".to_string(),
        "UniversalInterconnectedness".to_string(),
        "SacredReciprocity".to_string(),
        "EvolutionaryProgression".to_string(),
    ];

    // Identify risk factors
    let mut risk_factors = Vec::new();

    if participation_rate < 0.5 {
        risk_factors.push(RiskFactor {
            category: RiskCategory::Participation,
            severity: 1.0 - participation_rate,
            description: format!("Low participation rate: {:.1}%", participation_rate * 100.0),
        });
    }

    if average_phi < 0.4 {
        risk_factors.push(RiskFactor {
            category: RiskCategory::Consciousness,
            severity: (0.4 - average_phi) / 0.4,
            description: format!("Low average phi score: {:.2}", average_phi),
        });
    }

    if harmony_coverage < 0.5 {
        risk_factors.push(RiskFactor {
            category: RiskCategory::HarmonyGap,
            severity: 1.0 - harmony_coverage,
            description: format!("{} harmonies absent from deliberation", absent_harmonies.len()),
        });
    }

    if let Some(ref ch) = child_health {
        if ch.struggling_count > 0 {
            risk_factors.push(RiskFactor {
                category: RiskCategory::ChildHealth,
                severity: ch.struggling_count as f64 / child_count as f64,
                description: format!("{} child councils struggling", ch.struggling_count),
            });
        }
    }

    // Calculate vitality trend
    let vitality_trend = if risk_factors.is_empty() {
        VitalityTrend::Thriving
    } else if risk_factors.iter().any(|r| r.severity > 0.7) {
        VitalityTrend::Critical
    } else if risk_factors.iter().any(|r| r.severity > 0.4) {
        VitalityTrend::Declining
    } else {
        VitalityTrend::Stable
    };

    // Calculate overall health score
    let health_score = calculate_health_score(
        participation_rate,
        average_phi,
        harmony_coverage,
        &child_health,
        &risk_factors,
    );

    // Identify opportunities
    let opportunities = identify_opportunities(
        &harmony_presence,
        &absent_harmonies,
        &child_health,
        participation_rate,
    );

    // Generate summary
    let summary = generate_reflection_summary(
        &council,
        member_count,
        health_score,
        &vitality_trend,
        &risk_factors,
    );

    let reflection_id = format!(
        "reflection-{}-{}",
        council_id,
        timestamp.as_micros()
    );

    let reflection = HolonicReflection {
        id: reflection_id,
        council_id: council_id.clone(),
        timestamp,
        holon_depth,
        member_count,
        phi_qualified_count,
        average_phi,
        phi_distribution,
        participation_rate,
        decisions_last_30_days: 0, // Would need decision tracking
        average_consensus: 0.75, // Placeholder
        contention_ratio: 0.1, // Placeholder
        implementation_rate: 0.85, // Placeholder
        child_count,
        child_health,
        parent_coherence: council.parent_council_id.as_ref().map(|_| 0.8), // Placeholder
        collaboration_score: 0.6, // Placeholder
        harmony_presence,
        harmony_coverage,
        absent_harmonies,
        vitality_trend,
        risk_factors,
        opportunities,
        health_score,
        summary,
    };

    // Capture signal data before moving reflection
    let signal_council_id = reflection.council_id.clone();
    let signal_reflection_id = reflection.id.clone();
    let signal_health_score = reflection.health_score;
    let signal_vitality_trend = format!("{:?}", reflection.vitality_trend);
    let signal_risk_count = reflection.risk_factors.len();

    let action_hash = create_entry(&EntryTypes::HolonicReflection(reflection))?;
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created reflection".into())))?;

    // Link council to reflection
    let council_hash = council_record.action_hashed().hash.clone();
    create_link(
        council_hash,
        action_hash,
        LinkTypes::CouncilToReflection,
        (),
    )?;

    // Emit real-time signal for connected clients
    let _ = emit_council_signal(CouncilSignal::HolonicReflectionGenerated {
        council_id: signal_council_id,
        reflection_id: signal_reflection_id,
        health_score: signal_health_score,
        vitality_trend: signal_vitality_trend,
        risk_count: signal_risk_count,
    });

    Ok(record)
}

/// Get council reflections
#[hdk_extern]
pub fn get_council_reflections(council_id: String) -> ExternResult<Vec<Record>> {
    let council_record = get_council_by_id(council_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Council not found".into())))?;

    let council_hash = council_record.action_hashed().hash.clone();

    let links = get_links(
        LinkQuery::try_new(council_hash, LinkTypes::CouncilToReflection)?,
        GetStrategy::default(),
    )?;

    let mut reflections = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                reflections.push(record);
            }
        }
    }

    Ok(reflections)
}

/// Get the latest reflection for a council
#[hdk_extern]
pub fn get_latest_council_reflection(council_id: String) -> ExternResult<Option<Record>> {
    let reflections = get_council_reflections(council_id)?;

    // Find most recent
    let mut latest: Option<(Timestamp, Record)> = None;

    for record in reflections {
        if let Some(reflection) = record
            .entry()
            .to_app_option::<HolonicReflection>()
            .ok()
            .flatten()
        {
            match &latest {
                None => latest = Some((reflection.timestamp, record)),
                Some((ts, _)) if reflection.timestamp > *ts => {
                    latest = Some((reflection.timestamp, record))
                }
                _ => {}
            }
        }
    }

    Ok(latest.map(|(_, r)| r))
}

/// Get holonic health tree (recursive reflection of entire hierarchy)
#[hdk_extern]
pub fn get_holonic_tree(root_council_id: String) -> ExternResult<HolonicTreeNode> {
    build_holonic_tree(root_council_id, 0, 5)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HolonicTreeNode {
    pub council_id: String,
    pub council_name: String,
    pub depth: u8,
    pub health_score: f64,
    pub member_count: u64,
    pub children: Vec<HolonicTreeNode>,
    pub reflection_summary: Option<String>,
}

fn build_holonic_tree(council_id: String, depth: u8, max_depth: u8) -> ExternResult<HolonicTreeNode> {
    if depth > max_depth {
        return Err(wasm_error!(WasmErrorInner::Guest("Max depth exceeded".into())));
    }

    let council_record = get_council_by_id(council_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Council not found".into())))?;

    let council: Council = council_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid council entry".into())))?;

    let members = get_council_members(council_id.clone())?;

    let latest_reflection = get_latest_council_reflection(council_id.clone())?;
    let (health_score, reflection_summary) = if let Some(ref record) = latest_reflection {
        if let Some(reflection) = record
            .entry()
            .to_app_option::<HolonicReflection>()
            .ok()
            .flatten()
        {
            (reflection.health_score, Some(reflection.summary))
        } else {
            (0.5, None)
        }
    } else {
        (0.5, None)
    };

    // Build children recursively
    let child_records = get_child_councils(council_id.clone())?;
    let mut children = Vec::new();

    for child_record in child_records {
        if let Some(child_council) = child_record
            .entry()
            .to_app_option::<Council>()
            .ok()
            .flatten()
        {
            let child_tree = build_holonic_tree(child_council.id, depth + 1, max_depth)?;
            children.push(child_tree);
        }
    }

    Ok(HolonicTreeNode {
        council_id,
        council_name: council.name,
        depth,
        health_score,
        member_count: members.len() as u64,
        children,
        reflection_summary,
    })
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn calculate_phi_distribution(phi_scores: &[f64]) -> PhiDistribution {
    if phi_scores.is_empty() {
        return PhiDistribution {
            min: 0.0,
            p25: 0.0,
            median: 0.0,
            p75: 0.0,
            max: 0.0,
        };
    }

    let mut sorted: Vec<f64> = phi_scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = sorted.len();
    let min = sorted[0];
    let max = sorted[len - 1];
    let median = if len % 2 == 0 {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    };
    let p25 = sorted[len / 4];
    let p75 = sorted[(3 * len) / 4];

    PhiDistribution {
        min,
        p25,
        median,
        p75,
        max,
    }
}

fn calculate_holon_depth(council: &Council) -> u8 {
    // Simple depth calculation - would need recursive lookup for accurate depth
    if council.parent_council_id.is_none() {
        0
    } else {
        1 // Simplified - actual implementation would traverse up
    }
}

fn aggregate_child_health(child_records: &[Record]) -> ExternResult<AggregateChildHealth> {
    let mut healthy_count = 0u64;
    let mut struggling_count = 0u64;
    let mut dormant_count = 0u64;
    let mut health_scores = Vec::new();
    let mut children_needing_attention = Vec::new();

    for record in child_records {
        if let Some(council) = record
            .entry()
            .to_app_option::<Council>()
            .ok()
            .flatten()
        {
            match council.status {
                CouncilStatus::Active => {
                    // Would check latest reflection for actual health
                    healthy_count += 1;
                    health_scores.push(0.7); // Placeholder
                }
                CouncilStatus::Dormant => {
                    dormant_count += 1;
                    children_needing_attention.push(council.id);
                }
                CouncilStatus::Suspended => {
                    struggling_count += 1;
                    children_needing_attention.push(council.id);
                }
                CouncilStatus::Dissolved => {}
            }
        }
    }

    let average_health = if health_scores.is_empty() {
        0.0
    } else {
        health_scores.iter().sum::<f64>() / health_scores.len() as f64
    };

    Ok(AggregateChildHealth {
        healthy_count,
        struggling_count,
        dormant_count,
        average_health,
        children_needing_attention,
    })
}

fn calculate_health_score(
    participation_rate: f64,
    average_phi: f64,
    harmony_coverage: f64,
    child_health: &Option<AggregateChildHealth>,
    risk_factors: &[RiskFactor],
) -> f64 {
    // Base score from key metrics
    let base_score = (participation_rate * 0.25)
        + (average_phi * 0.25)
        + (harmony_coverage * 0.25);

    // Add child health contribution
    let child_contribution = match child_health {
        Some(ch) => ch.average_health * 0.25,
        None => 0.25, // Full points if no children
    };

    let total = base_score + child_contribution;

    // Apply risk penalty
    let risk_penalty: f64 = risk_factors.iter().map(|r| r.severity * 0.05).sum();

    (total - risk_penalty).max(0.0).min(1.0)
}

fn identify_opportunities(
    _harmony_presence: &[HarmonyPresence],
    absent_harmonies: &[String],
    child_health: &Option<AggregateChildHealth>,
    participation_rate: f64,
) -> Vec<String> {
    let mut opportunities = Vec::new();

    if !absent_harmonies.is_empty() {
        opportunities.push(format!(
            "Invite perspectives from {} harmonies: {}",
            absent_harmonies.len(),
            absent_harmonies.join(", ")
        ));
    }

    if participation_rate < 0.7 {
        opportunities.push("Increase engagement through discussion facilitation".to_string());
    }

    if let Some(ch) = child_health {
        if ch.dormant_count > 0 {
            opportunities.push(format!(
                "Re-energize {} dormant sub-councils",
                ch.dormant_count
            ));
        }
    }

    opportunities
}

fn generate_reflection_summary(
    council: &Council,
    member_count: u64,
    health_score: f64,
    vitality_trend: &VitalityTrend,
    risk_factors: &[RiskFactor],
) -> String {
    let trend_desc = match vitality_trend {
        VitalityTrend::Thriving => "thriving",
        VitalityTrend::Stable => "stable",
        VitalityTrend::Declining => "showing signs of decline",
        VitalityTrend::Critical => "in critical need of attention",
    };

    let risk_desc = if risk_factors.is_empty() {
        "No significant risk factors identified.".to_string()
    } else {
        format!(
            "{} risk factors: {}",
            risk_factors.len(),
            risk_factors
                .iter()
                .map(|r| r.description.clone())
                .collect::<Vec<_>>()
                .join("; ")
        )
    };

    format!(
        "Council '{}' ({} members) is currently {}. Health score: {:.0}%. {}",
        council.name,
        member_count,
        trend_desc,
        health_score * 100.0,
        risk_desc
    )
}

// ============================================================================
// DECISION MAKING
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordDecisionInput {
    pub council_id: String,
    pub proposal_id: Option<String>,
    pub title: String,
    pub content: String,
    pub decision_type: DecisionType,
    pub votes_for: u64,
    pub votes_against: u64,
    pub abstentions: u64,
    pub phi_weighted_result: f64,
}

/// Record a council decision
#[hdk_extern]
pub fn record_decision(input: RecordDecisionInput) -> ExternResult<Record> {
    // Input validation
    if input.council_id.is_empty() || input.council_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Council ID must be 1-256 characters".into())));
    }
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Decision title must be 1-256 characters".into())));
    }
    if input.content.is_empty() || input.content.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Decision content must be 1-4096 characters".into())));
    }
    if let Some(ref proposal_id) = input.proposal_id {
        if proposal_id.is_empty() || proposal_id.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest("Proposal ID must be 1-256 characters".into())));
        }
    }
    if input.phi_weighted_result < 0.0 || input.phi_weighted_result > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Phi weighted result must be between 0.0 and 1.0".into())));
    }

    let timestamp = sys_time()?;

    // Get council to verify existence
    let council_record = get_council_by_id(input.council_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Council not found".into())))?;

    let council: Council = council_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid council entry".into())))?;

    // Determine if decision passed
    let threshold = match input.decision_type {
        DecisionType::Constitutional => council.supermajority,
        _ => 0.5,
    };
    let passed = input.phi_weighted_result > threshold;

    let decision_id = format!(
        "decision-{}-{}",
        input.council_id,
        timestamp.as_micros()
    );

    // Capture signal data before moving
    let signal_council_id = input.council_id.clone();
    let signal_decision_id = decision_id.clone();
    let signal_title = input.title.clone();
    let signal_phi_weighted = input.phi_weighted_result;

    let decision = CouncilDecision {
        id: decision_id,
        council_id: input.council_id.clone(),
        proposal_id: input.proposal_id,
        title: input.title,
        content: input.content,
        decision_type: input.decision_type,
        votes_for: input.votes_for,
        votes_against: input.votes_against,
        abstentions: input.abstentions,
        phi_weighted_result: input.phi_weighted_result,
        passed,
        status: if passed {
            DecisionStatus::Approved
        } else {
            DecisionStatus::Rejected
        },
        created_at: timestamp,
        executed_at: None,
    };

    let action_hash = create_entry(&EntryTypes::CouncilDecision(decision))?;
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created decision".into())))?;

    // Link council to decision
    let council_hash = council_record.action_hashed().hash.clone();
    create_link(
        council_hash,
        action_hash,
        LinkTypes::CouncilToDecision,
        (),
    )?;

    // Emit real-time signal for connected clients
    let _ = emit_council_signal(CouncilSignal::DecisionRecorded {
        council_id: signal_council_id,
        decision_id: signal_decision_id,
        title: signal_title,
        passed,
        phi_weighted_result: signal_phi_weighted,
    });

    Ok(record)
}

/// Get council decisions
#[hdk_extern]
pub fn get_council_decisions(council_id: String) -> ExternResult<Vec<Record>> {
    let council_record = get_council_by_id(council_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Council not found".into())))?;

    let council_hash = council_record.action_hashed().hash.clone();

    let links = get_links(
        LinkQuery::try_new(council_hash, LinkTypes::CouncilToDecision)?,
        GetStrategy::default(),
    )?;

    let mut decisions = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                decisions.push(record);
            }
        }
    }

    Ok(decisions)
}
