//! Temporal Consciousness Coordinator Zome
//!
//! Implements belief trajectory tracking and consciousness evolution analysis.

use hdk::prelude::*;
use temporal_consciousness_integrity::*;

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

// ============================================================================
// TRAJECTORY MANAGEMENT
// ============================================================================

/// Record a snapshot of a belief's state
#[hdk_extern]
pub fn record_belief_snapshot(input: RecordSnapshotInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let snapshot = BeliefSnapshot {
        epistemic_code: input.epistemic_code,
        confidence: input.confidence,
        phi: input.phi,
        coherence: input.coherence,
        timestamp: now,
        trigger: input.trigger,
    };

    // Get or create trajectory
    let trajectory_hash = get_or_create_trajectory(&input.thought_id, &snapshot, now)?;

    get(trajectory_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Trajectory not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordSnapshotInput {
    pub thought_id: String,
    pub epistemic_code: String,
    pub confidence: f64,
    pub phi: f64,
    pub coherence: f64,
    pub trigger: SnapshotTrigger,
}

fn get_or_create_trajectory(
    thought_id: &str,
    snapshot: &BeliefSnapshot,
    now: Timestamp,
) -> ExternResult<ActionHash> {
    let thought_anchor = format!("thought_trajectory:{}", thought_id);

    // Check for existing trajectory
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&thought_anchor)?, LinkTypes::ThoughtToTrajectory)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        // Update existing trajectory
        let existing_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;

        if let Some(record) = get(existing_hash.clone(), GetOptions::default())? {
            if let Some(mut trajectory) = record.entry().to_app_option::<BeliefTrajectory>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                trajectory.snapshots.push(snapshot.clone());
                trajectory.last_updated = now;

                // Reclassify trajectory type
                trajectory.trajectory_type = classify_trajectory(&trajectory.snapshots);
                trajectory.entrenchment = detect_entrenchment(&trajectory.snapshots);

                // Create new version
                let new_hash = create_entry(&EntryTypes::BeliefTrajectory(trajectory))?;

                // Update link
                create_link(
                    anchor_hash(&thought_anchor)?,
                    new_hash.clone(),
                    LinkTypes::ThoughtToTrajectory,
                    (),
                )?;

                return Ok(new_hash);
            }
        }
    }

    // Create new trajectory
    let trajectory = BeliefTrajectory {
        thought_id: thought_id.to_string(),
        snapshots: vec![snapshot.clone()],
        trajectory_type: TrajectoryType::Insufficient,
        entrenchment: None,
        started_at: now,
        last_updated: now,
    };

    let action_hash = create_entry(&EntryTypes::BeliefTrajectory(trajectory))?;

    create_anchor(&thought_anchor)?;
    create_link(
        anchor_hash(&thought_anchor)?,
        action_hash.clone(),
        LinkTypes::ThoughtToTrajectory,
        (),
    )?;

    Ok(action_hash)
}

/// Get the trajectory for a thought
#[hdk_extern]
pub fn get_belief_trajectory(thought_id: String) -> ExternResult<Option<BeliefTrajectory>> {
    let thought_anchor = format!("thought_trajectory:{}", thought_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&thought_anchor)?, LinkTypes::ThoughtToTrajectory)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            return Ok(record.entry().to_app_option::<BeliefTrajectory>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?);
        }
    }

    Ok(None)
}

// ============================================================================
// TRAJECTORY ANALYSIS
// ============================================================================

/// Analyze a belief's trajectory
#[hdk_extern]
pub fn analyze_belief_trajectory(thought_id: String) -> ExternResult<TrajectoryAnalysis> {
    let now = sys_time()?;

    // Get trajectory
    let trajectory = get_belief_trajectory(thought_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Trajectory not found".into())))?;

    if trajectory.snapshots.len() < 2 {
        return Ok(TrajectoryAnalysis {
            id: generate_uuid(),
            thought_id,
            trajectory_type: TrajectoryType::Insufficient,
            confidence_trend: 0.0,
            phi_trend: 0.0,
            coherence_trend: 0.0,
            entrenchment: None,
            contradiction_count: 0,
            recommendation: "Not enough history to analyze. Continue tracking this belief.".to_string(),
            analyzed_at: now,
        });
    }

    // Calculate trends
    let confidence_trend = calculate_trend(trajectory.snapshots.iter().map(|s| s.confidence));
    let phi_trend = calculate_trend(trajectory.snapshots.iter().map(|s| s.phi));
    let coherence_trend = calculate_trend(trajectory.snapshots.iter().map(|s| s.coherence));

    // Count contradiction triggers
    let contradiction_count = trajectory.snapshots.iter()
        .filter(|s| matches!(s.trigger, SnapshotTrigger::ContradictionDetected))
        .count() as u32;

    // Detect entrenchment
    let entrenchment = detect_entrenchment(&trajectory.snapshots);

    // Classify trajectory
    let trajectory_type = classify_trajectory(&trajectory.snapshots);

    // Generate recommendation
    let recommendation = generate_recommendation(
        &trajectory_type,
        &entrenchment,
        confidence_trend,
        contradiction_count,
    );

    let analysis = TrajectoryAnalysis {
        id: generate_uuid(),
        thought_id: thought_id.clone(),
        trajectory_type,
        confidence_trend,
        phi_trend,
        coherence_trend,
        entrenchment,
        contradiction_count,
        recommendation,
        analyzed_at: now,
    };

    // Store the analysis
    let action_hash = create_entry(&EntryTypes::TrajectoryAnalysis(analysis.clone()))?;

    // Link to trajectory
    let thought_anchor = format!("thought_trajectory:{}", thought_id);
    create_link(
        anchor_hash(&thought_anchor)?,
        action_hash,
        LinkTypes::TrajectoryToAnalyses,
        (),
    )?;

    Ok(analysis)
}

/// Calculate a trend from a series of values (-1 to 1)
fn calculate_trend<I>(values: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let vals: Vec<f64> = values.collect();
    if vals.len() < 2 {
        return 0.0;
    }

    // Simple linear regression slope
    let n = vals.len() as f64;
    let sum_x: f64 = (0..vals.len()).map(|i| i as f64).sum();
    let sum_y: f64 = vals.iter().sum();
    let sum_xy: f64 = vals.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
    let sum_xx: f64 = (0..vals.len()).map(|i| (i as f64) * (i as f64)).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return 0.0;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;

    // Normalize to -1 to 1 range
    slope.clamp(-1.0, 1.0)
}

/// Classify a trajectory based on snapshots
fn classify_trajectory(snapshots: &[BeliefSnapshot]) -> TrajectoryType {
    if snapshots.len() < 3 {
        return TrajectoryType::Insufficient;
    }

    let confidence_trend = calculate_trend(snapshots.iter().map(|s| s.confidence));
    let coherence_trend = calculate_trend(snapshots.iter().map(|s| s.coherence));

    // Check for volatility (large changes)
    let confidence_changes: Vec<f64> = snapshots.windows(2)
        .map(|w| (w[1].confidence - w[0].confidence).abs())
        .collect();
    let avg_change = confidence_changes.iter().sum::<f64>() / confidence_changes.len() as f64;

    if avg_change > 0.3 {
        return TrajectoryType::Volatile;
    }

    // Check for oscillation (alternating direction)
    let direction_changes = snapshots.windows(3)
        .filter(|w| {
            let d1 = w[1].confidence - w[0].confidence;
            let d2 = w[2].confidence - w[1].confidence;
            d1 * d2 < 0.0 // Sign change
        })
        .count();

    if direction_changes > snapshots.len() / 2 {
        return TrajectoryType::Oscillating;
    }

    // Check for entrenchment
    let contradiction_count = snapshots.iter()
        .filter(|s| matches!(s.trigger, SnapshotTrigger::ContradictionDetected))
        .count();

    if contradiction_count > 0 && confidence_trend > 0.1 {
        return TrajectoryType::Entrenched;
    }

    // Classify by trend
    if confidence_trend.abs() < 0.1 && coherence_trend.abs() < 0.1 {
        TrajectoryType::Stable
    } else if confidence_trend > 0.1 {
        TrajectoryType::Growing
    } else if confidence_trend < -0.1 {
        TrajectoryType::Weakening
    } else {
        TrajectoryType::Stable
    }
}

/// Detect entrenchment level
fn detect_entrenchment(snapshots: &[BeliefSnapshot]) -> Option<EntrenchmentLevel> {
    let contradiction_count = snapshots.iter()
        .filter(|s| matches!(s.trigger, SnapshotTrigger::ContradictionDetected))
        .count();

    if contradiction_count == 0 {
        return Some(EntrenchmentLevel::None);
    }

    // Check if confidence increased despite contradictions
    let first_contradiction_idx = snapshots.iter()
        .position(|s| matches!(s.trigger, SnapshotTrigger::ContradictionDetected));

    if let Some(idx) = first_contradiction_idx {
        if idx + 1 < snapshots.len() {
            let confidence_before = snapshots[idx].confidence;
            let confidence_after = snapshots.last().unwrap().confidence;

            if confidence_after > confidence_before + 0.1 {
                // Confidence increased despite contradictions
                return match contradiction_count {
                    1 => Some(EntrenchmentLevel::Mild),
                    2 => Some(EntrenchmentLevel::Moderate),
                    3 => Some(EntrenchmentLevel::High),
                    _ => Some(EntrenchmentLevel::Severe),
                };
            } else if (confidence_after - confidence_before).abs() < 0.05 {
                // Confidence unchanged despite contradictions
                return match contradiction_count {
                    1 => Some(EntrenchmentLevel::Mild),
                    2..=3 => Some(EntrenchmentLevel::Moderate),
                    _ => Some(EntrenchmentLevel::High),
                };
            }
        }
    }

    Some(EntrenchmentLevel::None)
}

/// Generate a recommendation based on trajectory analysis
fn generate_recommendation(
    trajectory_type: &TrajectoryType,
    entrenchment: &Option<EntrenchmentLevel>,
    confidence_trend: f64,
    contradiction_count: u32,
) -> String {
    // Check entrenchment first (most concerning)
    if let Some(level) = entrenchment {
        match level {
            EntrenchmentLevel::High | EntrenchmentLevel::Severe => {
                return format!(
                    "Warning: This belief shows signs of {} entrenchment. You've encountered {} contradictions but haven't revised your position. Consider actively seeking disconfirming evidence and steel-manning opposing viewpoints.",
                    if matches!(level, EntrenchmentLevel::Severe) { "severe" } else { "high" },
                    contradiction_count
                );
            }
            EntrenchmentLevel::Moderate => {
                return "This belief may be entrenched. Consider whether you're giving fair weight to contradictory evidence.".to_string();
            }
            _ => {}
        }
    }

    match trajectory_type {
        TrajectoryType::Stable => {
            "This belief has been stable. Consider periodically reviewing it against new evidence.".to_string()
        }
        TrajectoryType::Growing => {
            if confidence_trend > 0.5 {
                "Your confidence in this belief is growing rapidly. Ensure this is based on evidence rather than confirmation bias.".to_string()
            } else {
                "This belief is strengthening, which may indicate accumulating supporting evidence.".to_string()
            }
        }
        TrajectoryType::Weakening => {
            "Your confidence in this belief is declining. This may be healthy epistemic revision in light of new evidence.".to_string()
        }
        TrajectoryType::Oscillating => {
            "This belief shows unstable confidence. Consider investigating what causes your certainty to fluctuate.".to_string()
        }
        TrajectoryType::Volatile => {
            "This belief is changing rapidly. You may want to defer strong conclusions until it stabilizes.".to_string()
        }
        TrajectoryType::Entrenched => {
            "This belief appears entrenched. Practice intellectual humility and consider alternative perspectives.".to_string()
        }
        TrajectoryType::Insufficient => {
            "Not enough history to provide recommendations. Continue tracking this belief.".to_string()
        }
    }
}

// ============================================================================
// CONSCIOUSNESS EVOLUTION
// ============================================================================

/// Record a consciousness evolution summary for a time period
#[hdk_extern]
pub fn record_consciousness_evolution(input: RecordEvolutionInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let evolution = ConsciousnessEvolution {
        id: generate_uuid(),
        period_start: input.period_start,
        period_end: input.period_end,
        avg_phi: input.avg_phi,
        phi_trend: input.phi_trend,
        avg_coherence: input.avg_coherence,
        coherence_trend: input.coherence_trend,
        stable_belief_count: input.stable_belief_count,
        growing_belief_count: input.growing_belief_count,
        weakening_belief_count: input.weakening_belief_count,
        entrenched_belief_count: input.entrenched_belief_count,
        insights: input.insights,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ConsciousnessEvolution(evolution))?;

    // Link to agent
    let agent_anchor = format!("agent_evolution:{}", agent);
    create_anchor(&agent_anchor)?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToEvolution,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Evolution record not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordEvolutionInput {
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub avg_phi: f64,
    pub phi_trend: f64,
    pub avg_coherence: f64,
    pub coherence_trend: f64,
    pub stable_belief_count: u32,
    pub growing_belief_count: u32,
    pub weakening_belief_count: u32,
    pub entrenched_belief_count: u32,
    pub insights: Vec<String>,
}

/// Get consciousness evolution history
#[hdk_extern]
pub fn get_my_evolution_history(_: ()) -> ExternResult<Vec<ConsciousnessEvolution>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_anchor = format!("agent_evolution:{}", agent);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&agent_anchor)?, LinkTypes::AgentToEvolution)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(evolution) = record.entry().to_app_option::<ConsciousnessEvolution>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                records.push(evolution);
            }
        }
    }

    // Sort by period_start
    records.sort_by_key(|e| e.period_start);

    Ok(records)
}
