//! Stateful, deterministic replay for TOPO-ADAPT.
//!
//! This module implements only the deterministic P0 sequential contract. It
//! is not a learned controller: callers provide one bounded action per tick.
//! Message and controller-update costs remain unmeasured until a policy
//! adapter supplies actual chargeable traces.

use super::{
    solve_flow, validate_episode, Edge, Episode, FlowResult, GraphSemantics, SinkDemand,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExogenousEventKind {
    EdgeCapacityChange { from: u16, to: u16, capacity_after: u64 },
    SinkDemandChange { node: u16, demand_after: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExogenousEvent {
    pub event_id: String,
    pub kind: ExogenousEventKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum PolicyAction {
    NoOp { action_id: String, expected_pre_state_digest: String },
    AddCandidateEdge {
        action_id: String,
        expected_pre_state_digest: String,
        candidate_id: String,
    },
    Defer { action_id: String, expected_pre_state_digest: String },
    Abstain { action_id: String, expected_pre_state_digest: String },
}

impl PolicyAction {
    fn action_id(&self) -> &str {
        match self {
            Self::NoOp { action_id, .. }
            | Self::AddCandidateEdge { action_id, .. }
            | Self::Defer { action_id, .. }
            | Self::Abstain { action_id, .. } => action_id,
        }
    }

    fn expected_pre_state_digest(&self) -> &str {
        match self {
            Self::NoOp { expected_pre_state_digest, .. }
            | Self::AddCandidateEdge { expected_pre_state_digest, .. }
            | Self::Defer { expected_pre_state_digest, .. }
            | Self::Abstain { expected_pre_state_digest, .. } => expected_pre_state_digest,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionDisposition {
    Accepted,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionReceipt {
    pub action_id: String,
    pub action_kind: String,
    pub pre_state_digest: String,
    pub post_state_digest: String,
    pub disposition: ActionDisposition,
    pub reason: Option<String>,
    pub applied_candidate_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickReceipt {
    pub tick: u64,
    pub event_batch_digest: String,
    pub pre_action_state_digest: String,
    pub post_state_digest: String,
    pub flow_before_action: FlowResult,
    pub flow_after_action: FlowResult,
    pub action: Option<ActionReceipt>,
    pub accepted_candidate_count: usize,
    pub cumulative_edit_cost_units: u64,
    pub maintenance_cost_this_tick: u64,
    pub cumulative_maintenance_cost_units: u64,
    pub remaining_edit_budget: usize,
    // None means unmeasured; it must not be interpreted as zero.
    pub message_bytes: Option<u64>,
    pub controller_updates: Option<u64>,
}

#[derive(Clone, Debug)]
struct PreparedTick {
    active_edges: Vec<Edge>,
    demands: Vec<SinkDemand>,
    applied_event_ids: BTreeSet<String>,
    event_batch_digest: String,
}

#[derive(Clone, Debug)]
pub struct EpisodeRuntimeState {
    episode: Episode,
    semantics: GraphSemantics,
    context_digest: String,
    current_tick: Option<u64>,
    active_edges: Vec<Edge>,
    demands: Vec<SinkDemand>,
    accepted_candidate_ids: BTreeSet<String>,
    used_action_ids: BTreeSet<String>,
    applied_event_ids: BTreeSet<String>,
    remaining_edit_budget: usize,
    cumulative_edit_cost_units: u64,
    cumulative_maintenance_cost_units: u64,
}

impl EpisodeRuntimeState {
    /// Create a runtime from a validated static episode. Accepted edits persist
    /// across ticks, unlike independent calls to the static evaluator.
    pub fn new(episode: &Episode, semantics: &GraphSemantics) -> Result<Self, String> {
        validate_episode(episode, semantics)?;
        Ok(Self {
            episode: episode.clone(),
            semantics: semantics.clone(),
            context_digest: digest_episode_context(episode, semantics),
            current_tick: None,
            active_edges: episode.active_edges.clone(),
            demands: Vec::new(),
            accepted_candidate_ids: BTreeSet::new(),
            used_action_ids: BTreeSet::new(),
            applied_event_ids: BTreeSet::new(),
            remaining_edit_budget: semantics.edit_budget,
            cumulative_edit_cost_units: 0,
            cumulative_maintenance_cost_units: 0,
        })
    }

    pub fn active_edges(&self) -> &[Edge] {
        &self.active_edges
    }

    pub fn current_demands(&self) -> &[SinkDemand] {
        &self.demands
    }

    pub fn remaining_edit_budget(&self) -> usize {
        self.remaining_edit_budget
    }

    pub fn accepted_candidate_ids(&self) -> &BTreeSet<String> {
        &self.accepted_candidate_ids
    }

    pub fn cumulative_edit_cost_units(&self) -> u64 {
        self.cumulative_edit_cost_units
    }

    pub fn cumulative_maintenance_cost_units(&self) -> u64 {
        self.cumulative_maintenance_cost_units
    }

    pub fn current_tick(&self) -> Option<u64> {
        self.current_tick
    }

    /// Current authoritative state digest. Before the first tick the demand
    /// schedule has not yet been applied; use the preview method to bind an
    /// action to a particular tick's event-updated state.
    pub fn state_digest(&self) -> String {
        digest_state(
            &self.context_digest,
            self.current_tick.unwrap_or(0),
            &self.active_edges,
            &self.demands,
            self.remaining_edit_budget,
            self.cumulative_edit_cost_units,
            self.cumulative_maintenance_cost_units,
            &self.accepted_candidate_ids,
            &self.used_action_ids,
            &self.applied_event_ids,
        )
    }

    /// Compute the exact pre-action digest without mutating the runtime.
    /// apply_tick uses the same event preparation path.
    pub fn preview_pre_action_state_digest(
        &self,
        tick: u64,
        events: &[ExogenousEvent],
    ) -> Result<String, String> {
        let prepared = self.prepare_tick(tick, events)?;
        Ok(digest_state(
            &self.context_digest,
            tick,
            &prepared.active_edges,
            &prepared.demands,
            self.remaining_edit_budget,
            self.cumulative_edit_cost_units,
            self.cumulative_maintenance_cost_units,
            &self.accepted_candidate_ids,
            &self.used_action_ids,
            &prepared.applied_event_ids,
        ))
    }

    /// Apply one deterministic environment tick and at most one controller
    /// action. Exogenous events commit even if the action is rejected.
    /// Invalid/conflicting event batches reject atomically.
    pub fn apply_tick(
        &mut self,
        tick: u64,
        events: &[ExogenousEvent],
        action: Option<PolicyAction>,
    ) -> Result<TickReceipt, String> {
        let prepared = self.prepare_tick(tick, events)?;
        let pre_action_state_digest = digest_state(
            &self.context_digest,
            tick,
            &prepared.active_edges,
            &prepared.demands,
            self.remaining_edit_budget,
            self.cumulative_edit_cost_units,
            self.cumulative_maintenance_cost_units,
            &self.accepted_candidate_ids,
            &self.used_action_ids,
            &prepared.applied_event_ids,
        );

        let mut next_edges = prepared.active_edges;
        let next_demands = prepared.demands;
        let mut next_accepted = self.accepted_candidate_ids.clone();
        let mut next_used_actions = self.used_action_ids.clone();
        let next_event_ids = prepared.applied_event_ids;
        let mut next_remaining_budget = self.remaining_edit_budget;
        let mut next_edit_cost = self.cumulative_edit_cost_units;

        let mut flow_before_action = solve_flow(&self.episode, &next_edges, &next_demands)?;
        decorate_flow(
            &mut flow_before_action,
            next_accepted.len(),
            next_edit_cost,
            self.cumulative_maintenance_cost_units,
        );

        let mut action_fields: Option<(
            String,
            String,
            ActionDisposition,
            Option<String>,
            Option<String>,
        )> = None;

        if let Some(proposed) = action.as_ref() {
            let action_id = proposed.action_id().to_string();
            let kind = action_kind(proposed).to_string();
            let mut disposition = ActionDisposition::Rejected;
            let mut reason = None;
            let mut applied_candidate_id = None;

            if action_id.trim().is_empty() {
                reason = Some("empty_action_id".to_string());
            } else if next_used_actions.contains(&action_id) {
                reason = Some("duplicate_action_id".to_string());
            } else {
                // Consume non-duplicate IDs even for rejected proposals, so a
                // retry cannot later mutate state under the same idempotency key.
                next_used_actions.insert(action_id.clone());

                if proposed.expected_pre_state_digest() != pre_action_state_digest {
                    reason = Some("pre_state_digest_mismatch".to_string());
                } else {
                    match proposed {
                        PolicyAction::NoOp { .. }
                        | PolicyAction::Defer { .. }
                        | PolicyAction::Abstain { .. } => {
                            disposition = ActionDisposition::Accepted;
                        }
                        PolicyAction::AddCandidateEdge { candidate_id, .. } => {
                            if next_remaining_budget == 0 {
                                reason = Some("cumulative_edit_budget_exhausted".to_string());
                            } else if next_edges.len() + 1
                                > self.semantics.accounting.maximum_active_edges
                            {
                                reason = Some("maximum_active_edge_count_exceeded".to_string());
                            } else if next_accepted.contains(candidate_id) {
                                reason = Some("candidate_already_accepted".to_string());
                            } else if let Some(candidate) = self
                                .episode
                                .candidate_edges
                                .iter()
                                .find(|edge| edge.id.as_deref() == Some(candidate_id.as_str()))
                            {
                                if next_edges.iter().any(|edge| {
                                    edge.from == candidate.from && edge.to == candidate.to
                                }) {
                                    reason = Some("candidate_duplicates_runtime_edge".to_string());
                                } else if let Some(cost) = candidate.edit_cost {
                                    if let Some(total) = next_edit_cost.checked_add(cost) {
                                        next_edges.push(candidate.clone());
                                        next_accepted.insert(candidate_id.clone());
                                        next_remaining_budget -= 1;
                                        next_edit_cost = total;
                                        applied_candidate_id = Some(candidate_id.clone());
                                        disposition = ActionDisposition::Accepted;
                                    } else {
                                        reason = Some("cumulative_edit_cost_overflow".to_string());
                                    }
                                } else {
                                    reason = Some("candidate_missing_edit_cost".to_string());
                                }
                            } else {
                                reason = Some("unknown_candidate_id".to_string());
                            }
                        }
                    }
                }
            }

            action_fields = Some((
                action_id,
                kind,
                disposition,
                reason,
                applied_candidate_id,
            ));
        }

        let maintenance_cost_this_tick = next_edges.iter().try_fold(0_u64, |sum, edge| {
            sum.checked_add(edge.maintenance_cost)
                .ok_or_else(|| "maintenance_cost_overflow".to_string())
        })?;
        let cumulative_maintenance_cost_units = self
            .cumulative_maintenance_cost_units
            .checked_add(maintenance_cost_this_tick)
            .ok_or_else(|| "cumulative_maintenance_cost_overflow".to_string())?;

        let mut flow_after_action = solve_flow(&self.episode, &next_edges, &next_demands)?;
        decorate_flow(
            &mut flow_after_action,
            next_accepted.len(),
            next_edit_cost,
            cumulative_maintenance_cost_units,
        );

        let post_state_digest = digest_state(
            &self.context_digest,
            tick,
            &next_edges,
            &next_demands,
            next_remaining_budget,
            next_edit_cost,
            cumulative_maintenance_cost_units,
            &next_accepted,
            &next_used_actions,
            &next_event_ids,
        );

        let action_receipt = action_fields.map(
            |(action_id, action_kind, disposition, reason, applied_candidate_id)| ActionReceipt {
                action_id,
                action_kind,
                pre_state_digest: pre_action_state_digest.clone(),
                post_state_digest: post_state_digest.clone(),
                disposition,
                reason,
                applied_candidate_id,
            },
        );

        // Commit only after both flow/cut references and checked accounting
        // have succeeded.
        self.current_tick = Some(tick);
        self.active_edges = next_edges;
        self.demands = next_demands;
        self.accepted_candidate_ids = next_accepted;
        self.used_action_ids = next_used_actions;
        self.applied_event_ids = next_event_ids;
        self.remaining_edit_budget = next_remaining_budget;
        self.cumulative_edit_cost_units = next_edit_cost;
        self.cumulative_maintenance_cost_units = cumulative_maintenance_cost_units;

        Ok(TickReceipt {
            tick,
            event_batch_digest: prepared.event_batch_digest,
            pre_action_state_digest,
            post_state_digest,
            flow_before_action,
            flow_after_action,
            action: action_receipt,
            accepted_candidate_count: self.accepted_candidate_ids.len(),
            cumulative_edit_cost_units: self.cumulative_edit_cost_units,
            maintenance_cost_this_tick,
            cumulative_maintenance_cost_units: self.cumulative_maintenance_cost_units,
            remaining_edit_budget: self.remaining_edit_budget,
            message_bytes: None,
            controller_updates: None,
        })
    }

    fn prepare_tick(
        &self,
        tick: u64,
        external_events: &[ExogenousEvent],
    ) -> Result<PreparedTick, String> {
        if self.current_tick.map_or(false, |previous| tick <= previous) {
            return Err("tick_not_strictly_increasing".to_string());
        }

        let demand_schedule = self
            .episode
            .demand_schedule
            .iter()
            .filter(|step| step.time <= tick)
            .max_by_key(|step| step.time)
            .ok_or_else(|| "no_demand_schedule_at_or_before_tick".to_string())?;

        let mut events = external_events.to_vec();
        for (index, intervention) in self.episode.interventions.iter().enumerate() {
            let is_due = intervention.time <= tick
                && self
                    .current_tick
                    .map_or(true, |previous| intervention.time > previous);
            if is_due {
                if intervention.kind != "edge_capacity_failure" {
                    return Err("unsupported_fixture_intervention_kind".to_string());
                }
                let from = intervention
                    .from
                    .ok_or_else(|| "fixture_intervention_missing_from".to_string())?;
                let to = intervention
                    .to
                    .ok_or_else(|| "fixture_intervention_missing_to".to_string())?;
                let capacity_after = intervention
                    .capacity_after
                    .ok_or_else(|| "fixture_intervention_missing_capacity".to_string())?;
                events.push(ExogenousEvent {
                    event_id: format!("fixture-intervention-{}-{index}", intervention.time),
                    kind: ExogenousEventKind::EdgeCapacityChange {
                        from,
                        to,
                        capacity_after,
                    },
                });
            }
        }

        events.sort_by(|left, right| left.event_id.cmp(&right.event_id));
        let mut event_ids = self.applied_event_ids.clone();
        let mut changed_resources = BTreeSet::new();
        let mut active_edges = self.active_edges.clone();
        let mut demands = demand_schedule.sinks.clone();

        for event in &events {
            if event.event_id.trim().is_empty() {
                return Err("empty_event_id".to_string());
            }
            if !event_ids.insert(event.event_id.clone()) {
                return Err(format!("duplicate_event_id:{}", event.event_id));
            }

            let resource_key = match &event.kind {
                ExogenousEventKind::EdgeCapacityChange { from, to, .. } => {
                    format!("edge:{from}:{to}")
                }
                ExogenousEventKind::SinkDemandChange { node, .. } => {
                    format!("sink-demand:{node}")
                }
            };
            if !changed_resources.insert(resource_key.clone()) {
                return Err(format!("conflicting_same_tick_event:{resource_key}"));
            }

            match &event.kind {
                ExogenousEventKind::EdgeCapacityChange {
                    from,
                    to,
                    capacity_after,
                } => {
                    let edge = active_edges
                        .iter_mut()
                        .find(|edge| edge.from == *from && edge.to == *to)
                        .ok_or_else(|| {
                            format!("event_references_unknown_runtime_edge:{from}:{to}")
                        })?;
                    edge.capacity = *capacity_after;
                }
                ExogenousEventKind::SinkDemandChange { node, demand_after } => {
                    let is_sink = self.episode.nodes.iter().any(|candidate| {
                        candidate.id == *node && candidate.role == "sink"
                    });
                    if !is_sink {
                        return Err(format!("event_references_non_sink:{node}"));
                    }
                    if let Some(existing) = demands.iter_mut().find(|sink| sink.node == *node) {
                        existing.demand = *demand_after;
                    } else {
                        demands.push(SinkDemand {
                            node: *node,
                            demand: *demand_after,
                        });
                    }
                }
            }
        }

        demands.sort_by_key(|sink| sink.node);
        let payload = serde_json::to_vec(&events)
            .map_err(|error| format!("event_digest_serialize:{error}"))?;
        let event_batch_digest = blake3::hash(&payload).to_hex().to_string();

        Ok(PreparedTick {
            active_edges,
            demands,
            applied_event_ids: event_ids,
            event_batch_digest,
        })
    }
}

fn action_kind(action: &PolicyAction) -> &'static str {
    match action {
        PolicyAction::NoOp { .. } => "no_op",
        PolicyAction::AddCandidateEdge { .. } => "add_candidate_edge",
        PolicyAction::Defer { .. } => "defer",
        PolicyAction::Abstain { .. } => "abstain",
    }
}

fn decorate_flow(
    result: &mut FlowResult,
    accepted_candidate_count: usize,
    cumulative_edit_cost_units: u64,
    cumulative_maintenance_cost_units: u64,
) {
    result.accepted_edge_additions = accepted_candidate_count;
    result.edit_cost_units = cumulative_edit_cost_units;
    result.edge_maintenance_cost_units = cumulative_maintenance_cost_units;
}

#[allow(clippy::too_many_arguments)]
fn digest_state(
    context_digest: &str,
    tick: u64,
    edges: &[Edge],
    demands: &[SinkDemand],
    remaining_edit_budget: usize,
    cumulative_edit_cost_units: u64,
    cumulative_maintenance_cost_units: u64,
    accepted_candidate_ids: &BTreeSet<String>,
    used_action_ids: &BTreeSet<String>,
    applied_event_ids: &BTreeSet<String>,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_optional_string(&mut hasher, Some(context_digest));
    hasher.update(&tick.to_le_bytes());
    hasher.update(&(remaining_edit_budget as u64).to_le_bytes());
    hasher.update(&cumulative_edit_cost_units.to_le_bytes());
    hasher.update(&cumulative_maintenance_cost_units.to_le_bytes());

    let mut canonical_edges = edges.to_vec();
    canonical_edges.sort_by(|left, right| {
        (left.from, left.to, left.id.as_deref()).cmp(&(
            right.from,
            right.to,
            right.id.as_deref(),
        ))
    });
    for edge in &canonical_edges {
        hasher.update(&edge.from.to_le_bytes());
        hasher.update(&edge.to.to_le_bytes());
        hasher.update(&edge.capacity.to_le_bytes());
        hasher.update(&edge.latency.to_le_bytes());
        hasher.update(&edge.maintenance_cost.to_le_bytes());
        hash_optional_string(&mut hasher, edge.id.as_deref());
    }

    let mut canonical_demands = demands.to_vec();
    canonical_demands.sort_by_key(|sink| sink.node);
    for sink in &canonical_demands {
        hasher.update(&sink.node.to_le_bytes());
        hasher.update(&sink.demand.to_le_bytes());
    }

    hash_string_set(&mut hasher, accepted_candidate_ids);
    hash_string_set(&mut hasher, used_action_ids);
    hash_string_set(&mut hasher, applied_event_ids);
    hasher.finalize().to_hex().to_string()
}

fn hash_string_set(hasher: &mut blake3::Hasher, values: &BTreeSet<String>) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hash_optional_string(hasher, Some(value.as_str()));
    }
}

fn hash_optional_string(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn digest_episode_context(episode: &Episode, semantics: &GraphSemantics) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_optional_string(&mut hasher, Some(&episode.id));
    hasher.update(&[u8::from(semantics.directed)]);
    hasher.update(&(semantics.edit_budget as u64).to_le_bytes());
    hasher.update(&(semantics.candidate_edges_per_episode as u64).to_le_bytes());
    hasher.update(&semantics.accounting.accepted_edge_addition_cost_units.to_le_bytes());
    hasher.update(&semantics.accounting.message_budget_bytes.to_le_bytes());
    hasher.update(&semantics.accounting.controller_update_budget.to_le_bytes());
    hasher.update(&(semantics.accounting.maximum_active_edges as u64).to_le_bytes());

    let mut nodes = episode.nodes.clone();
    nodes.sort_by_key(|node| node.id);
    for node in &nodes {
        hasher.update(&node.id.to_le_bytes());
        hash_optional_string(&mut hasher, Some(&node.role));
        hash_optional_u64(&mut hasher, node.supply);
        hash_optional_u64(&mut hasher, node.demand);
    }

    for edges in [&episode.active_edges, &episode.candidate_edges] {
        let mut canonical = edges.clone();
        canonical.sort_by(|left, right| {
            (left.from, left.to, left.id.as_deref()).cmp(&(
                right.from,
                right.to,
                right.id.as_deref(),
            ))
        });
        hasher.update(&(canonical.len() as u64).to_le_bytes());
        for edge in &canonical {
            hash_edge(&mut hasher, edge);
        }
    }

    let mut schedules = episode.demand_schedule.clone();
    schedules.sort_by_key(|step| step.time);
    for schedule in &schedules {
        hasher.update(&schedule.time.to_le_bytes());
        let mut sinks = schedule.sinks.clone();
        sinks.sort_by_key(|sink| sink.node);
        for sink in &sinks {
            hasher.update(&sink.node.to_le_bytes());
            hasher.update(&sink.demand.to_le_bytes());
        }
    }

    let mut interventions = episode.interventions.clone();
    interventions.sort_by(|left, right| {
        (left.time, left.kind.as_str(), left.from, left.to, left.capacity_after)
            .cmp(&(right.time, right.kind.as_str(), right.from, right.to, right.capacity_after))
    });
    for event in &interventions {
        hasher.update(&event.time.to_le_bytes());
        hash_optional_string(&mut hasher, Some(&event.kind));
        hash_optional_u16(&mut hasher, event.from);
        hash_optional_u16(&mut hasher, event.to);
        hash_optional_u64(&mut hasher, event.capacity_after);
    }
    hasher.finalize().to_hex().to_string()
}

fn hash_edge(hasher: &mut blake3::Hasher, edge: &Edge) {
    hash_optional_string(hasher, edge.id.as_deref());
    hasher.update(&edge.from.to_le_bytes());
    hasher.update(&edge.to.to_le_bytes());
    hasher.update(&edge.capacity.to_le_bytes());
    hasher.update(&edge.latency.to_le_bytes());
    hash_optional_u64(hasher, edge.edit_cost);
    hasher.update(&edge.maintenance_cost.to_le_bytes());
}

fn hash_optional_u64(hasher: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hash_optional_u16(hasher: &mut blake3::Hasher, value: Option<u16>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Accounting, DemandSchedule, Node};

    fn edge(
        id: Option<&str>,
        from: u16,
        to: u16,
        capacity: u64,
        edit_cost: Option<u64>,
    ) -> Edge {
        Edge {
            id: id.map(str::to_string),
            from,
            to,
            capacity,
            latency: 1,
            edit_cost,
            maintenance_cost: 1,
        }
    }

    fn episode() -> (Episode, GraphSemantics) {
        let mut nodes = vec![
            Node { id: 0, role: "source".to_string(), supply: Some(4), demand: None },
            Node { id: 1, role: "transit".to_string(), supply: None, demand: None },
            Node { id: 2, role: "transit".to_string(), supply: None, demand: None },
            Node { id: 3, role: "sink".to_string(), supply: None, demand: Some(4) },
            Node { id: 4, role: "transit".to_string(), supply: None, demand: None },
        ];
        for id in 5..12 {
            nodes.push(Node {
                id,
                role: "neutral".to_string(),
                supply: None,
                demand: None,
            });
        }

        let episode = Episode {
            id: "persistent_repeated_disruption".to_string(),
            nodes,
            active_edges: vec![
                edge(None, 0, 1, 4, None),
                edge(None, 1, 3, 4, None),
                edge(None, 0, 2, 4, None),
                edge(None, 0, 4, 4, None),
            ],
            candidate_edges: vec![
                edge(Some("route_b_out"), 2, 3, 4, Some(1)),
                edge(Some("route_c_out"), 4, 3, 4, Some(1)),
            ],
            demand_schedule: vec![DemandSchedule {
                time: 0,
                sinks: vec![SinkDemand { node: 3, demand: 4 }],
            }],
            interventions: Vec::new(),
        };
        let semantics = GraphSemantics {
            directed: true,
            edit_budget: 2,
            candidate_edges_per_episode: 2,
            accounting: Accounting {
                accepted_edge_addition_cost_units: 1,
                message_budget_bytes: 1024,
                controller_update_budget: 64,
                maximum_active_edges: 8,
            },
        };
        (episode, semantics)
    }

    fn expected_digest(
        runtime: &EpisodeRuntimeState,
        tick: u64,
        events: &[ExogenousEvent],
    ) -> String {
        runtime.preview_pre_action_state_digest(tick, events).unwrap()
    }

    fn add_action(
        runtime: &EpisodeRuntimeState,
        tick: u64,
        events: &[ExogenousEvent],
        action_id: &str,
        candidate_id: &str,
    ) -> PolicyAction {
        PolicyAction::AddCandidateEdge {
            action_id: action_id.to_string(),
            expected_pre_state_digest: expected_digest(runtime, tick, events),
            candidate_id: candidate_id.to_string(),
        }
    }

    #[test]
    fn repeated_disruption_preserves_runtime_edges_and_cumulative_budget() {
        let (episode, semantics) = episode();
        let mut runtime = EpisodeRuntimeState::new(&episode, &semantics).unwrap();

        let tick0_events: Vec<ExogenousEvent> = Vec::new();
        let digest0 = expected_digest(&runtime, 0, &tick0_events);
        let tick0_action = PolicyAction::NoOp {
            action_id: "a0".to_string(),
            expected_pre_state_digest: digest0,
        };
        let tick0 = runtime.apply_tick(0, &tick0_events, Some(tick0_action)).unwrap();
        assert_eq!(tick0.flow_after_action.max_flow, 4);
        assert_eq!(tick0.flow_after_action.min_cut, 4);
        assert_eq!(tick0.accepted_candidate_count, 0);

        let fail_a = vec![ExogenousEvent {
            event_id: "fail-route-a".to_string(),
            kind: ExogenousEventKind::EdgeCapacityChange {
                from: 1,
                to: 3,
                capacity_after: 0,
            },
        }];
        let action1 = add_action(&runtime, 1, &fail_a, "a1", "route_b_out");
        let tick1 = runtime.apply_tick(1, &fail_a, Some(action1)).unwrap();
        assert_eq!(tick1.flow_before_action.max_flow, 0);
        assert_eq!(tick1.flow_after_action.max_flow, 4);
        assert_eq!(tick1.flow_after_action.min_cut, 4);
        assert_eq!(tick1.accepted_candidate_count, 1);
        assert_eq!(tick1.remaining_edit_budget, 1);

        let fail_b = vec![ExogenousEvent {
            event_id: "fail-route-b".to_string(),
            kind: ExogenousEventKind::EdgeCapacityChange {
                from: 2,
                to: 3,
                capacity_after: 0,
            },
        }];
        let action2 = add_action(&runtime, 2, &fail_b, "a2", "route_c_out");
        let tick2 = runtime.apply_tick(2, &fail_b, Some(action2)).unwrap();
        assert_eq!(tick2.flow_before_action.max_flow, 0);
        assert_eq!(tick2.flow_after_action.max_flow, 4);
        assert_eq!(tick2.flow_after_action.min_cut, 4);
        assert_eq!(tick2.accepted_candidate_count, 2);
        assert_eq!(tick2.remaining_edit_budget, 0);
        assert_eq!(tick2.cumulative_edit_cost_units, 2);
        assert!(runtime
            .active_edges()
            .iter()
            .any(|edge| edge.id.as_deref() == Some("route_b_out") && edge.capacity == 0));
        assert!(runtime
            .active_edges()
            .iter()
            .any(|edge| edge.id.as_deref() == Some("route_c_out") && edge.capacity == 4));
    }

    #[test]
    fn conflicting_same_tick_events_reject_atomically() {
        let (episode, semantics) = episode();
        let mut runtime = EpisodeRuntimeState::new(&episode, &semantics).unwrap();
        let before = runtime.state_digest();
        let events = vec![
            ExogenousEvent {
                event_id: "edge-failure".to_string(),
                kind: ExogenousEventKind::EdgeCapacityChange { from: 1, to: 3, capacity_after: 0 },
            },
            ExogenousEvent {
                event_id: "edge-repair".to_string(),
                kind: ExogenousEventKind::EdgeCapacityChange { from: 1, to: 3, capacity_after: 4 },
            },
        ];
        assert!(runtime.apply_tick(0, &events, None).is_err());
        assert_eq!(runtime.state_digest(), before);
        assert_eq!(runtime.current_tick(), None);
    }

    #[test]
    fn nonconflicting_event_order_is_invariant() {
        let (episode, semantics) = episode();
        let runtime = EpisodeRuntimeState::new(&episode, &semantics).unwrap();
        let events = vec![
            ExogenousEvent {
                event_id: "edge-change".to_string(),
                kind: ExogenousEventKind::EdgeCapacityChange { from: 1, to: 3, capacity_after: 0 },
            },
            ExogenousEvent {
                event_id: "sink-demand".to_string(),
                kind: ExogenousEventKind::SinkDemandChange { node: 3, demand_after: 2 },
            },
        ];
        let mut reversed = events.clone();
        reversed.reverse();
        assert_eq!(
            runtime.preview_pre_action_state_digest(0, &events).unwrap(),
            runtime.preview_pre_action_state_digest(0, &reversed).unwrap()
        );
    }

    #[test]
    fn duplicate_action_id_never_applies_a_second_mutation() {
        let (episode, semantics) = episode();
        let mut runtime = EpisodeRuntimeState::new(&episode, &semantics).unwrap();
        let empty: Vec<ExogenousEvent> = Vec::new();
        let digest = expected_digest(&runtime, 0, &empty);
        let first = PolicyAction::NoOp {
            action_id: "same-id".to_string(),
            expected_pre_state_digest: digest,
        };
        runtime.apply_tick(0, &empty, Some(first)).unwrap();

        let next_empty: Vec<ExogenousEvent> = Vec::new();
        let second = add_action(&runtime, 1, &next_empty, "same-id", "route_b_out");
        let receipt = runtime.apply_tick(1, &next_empty, Some(second)).unwrap();
        assert_eq!(
            receipt.action.unwrap().reason.as_deref(),
            Some("duplicate_action_id")
        );
        assert_eq!(receipt.accepted_candidate_count, 0);
        assert_eq!(receipt.flow_after_action.max_flow, 4);
    }

    #[test]
    fn stale_pre_state_digest_rejects_action_without_graph_mutation() {
        let (episode, semantics) = episode();
        let mut runtime = EpisodeRuntimeState::new(&episode, &semantics).unwrap();
        let events = vec![ExogenousEvent {
            event_id: "fail-route-a".to_string(),
            kind: ExogenousEventKind::EdgeCapacityChange { from: 1, to: 3, capacity_after: 0 },
        }];
        let action = PolicyAction::AddCandidateEdge {
            action_id: "stale-action".to_string(),
            expected_pre_state_digest: "not-the-pre-state".to_string(),
            candidate_id: "route_b_out".to_string(),
        };
        let receipt = runtime.apply_tick(0, &events, Some(action)).unwrap();
        assert_eq!(
            receipt.action.unwrap().reason.as_deref(),
            Some("pre_state_digest_mismatch")
        );
        assert_eq!(receipt.flow_after_action.max_flow, 0);
        assert_eq!(receipt.accepted_candidate_count, 0);
    }

    #[test]
    fn repeated_replay_produces_identical_receipts() {
        let (episode, semantics) = episode();
        let mut left = EpisodeRuntimeState::new(&episode, &semantics).unwrap();
        let mut right = EpisodeRuntimeState::new(&episode, &semantics).unwrap();
        let events = vec![ExogenousEvent {
            event_id: "fail-route-a".to_string(),
            kind: ExogenousEventKind::EdgeCapacityChange { from: 1, to: 3, capacity_after: 0 },
        }];
        let left_action = add_action(&left, 0, &events, "a1", "route_b_out");
        let right_action = add_action(&right, 0, &events, "a1", "route_b_out");
        let left_receipt = left.apply_tick(0, &events, Some(left_action)).unwrap();
        let right_receipt = right.apply_tick(0, &events, Some(right_action)).unwrap();
        assert_eq!(left_receipt, right_receipt);
        assert_eq!(left.state_digest(), right.state_digest());
    }

    #[test]
    fn edit_budget_is_cumulative_across_ticks() {
        let (episode, mut semantics) = episode();
        semantics.edit_budget = 1;
        let mut runtime = EpisodeRuntimeState::new(&episode, &semantics).unwrap();
        let empty: Vec<ExogenousEvent> = Vec::new();
        let first_action = add_action(&runtime, 0, &empty, "a1", "route_b_out");
        let first = runtime.apply_tick(0, &empty, Some(first_action)).unwrap();
        assert_eq!(first.remaining_edit_budget, 0);

        let second_empty: Vec<ExogenousEvent> = Vec::new();
        let second_action = add_action(&runtime, 1, &second_empty, "a2", "route_c_out");
        let second = runtime.apply_tick(1, &second_empty, Some(second_action)).unwrap();
        assert_eq!(
            second.action.unwrap().reason.as_deref(),
            Some("cumulative_edit_budget_exhausted")
        );
        assert_eq!(second.accepted_candidate_count, 1);
        assert_eq!(second.flow_after_action.max_flow, 4);
    }
}
