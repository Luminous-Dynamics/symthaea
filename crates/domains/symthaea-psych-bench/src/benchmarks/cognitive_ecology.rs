// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic environment kernel for PsychBench Cognitive Ecology (PCE).
//!
//! This module is an environment primitive, not yet a validated psychological
//! benchmark or a `PsychBenchmark` implementation. It deliberately separates
//! the full evaluator state from the partial observation returned to an agent.
//! Every charged attempt produces a deterministic receipt that can be replayed
//! and audited independently of the agent implementation.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::cmp::Reverse;

/// Versioned benchmark-internal cost units. These are task costs, not joules,
/// wall-clock latency, API tokens, or monetary cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostModel {
    pub inspect: u32,
    pub move_between_sites: u32,
    pub interaction: u32,
    pub wait: u32,
    pub invalid_action_surcharge: u32,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            inspect: 2,
            move_between_sites: 1,
            interaction: 1,
            wait: 1,
            invalid_action_surcharge: 1,
        }
    }
}

/// Versioned scenario and execution limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioManifest {
    pub schema_version: u16,
    pub scenario_id: String,
    pub generator_version: String,
    /// Deterministically selects a generated layout variant (seed parity in v0.2).
    pub seed: u64,
    pub max_steps: u32,
    pub observation_budget: u32,
    pub cost_model: CostModel,
}

impl Default for ScenarioManifest {
    fn default() -> Self {
        Self {
            schema_version: 1,
            scenario_id: "pce.hidden-resource-delivery.v1".to_string(),
            generator_version: "pce-kernel-0.2.0".to_string(),
            seed: 42,
            max_steps: 64,
            observation_budget: 12,
            cost_model: CostModel::default(),
        }
    }
}

impl ScenarioManifest {
    /// Generate the hidden-resource layout for this manifest.
    pub fn layout(&self) -> ScenarioLayout {
        ScenarioLayout {
            filter_location: if self.seed & 1 == 0 { Site::Depot } else { Site::Outpost },
        }
    }

    /// Stable digest of the exact scenario configuration.
    pub fn digest(&self) -> Result<String, serde_json::Error> {
        let bytes = serde_json::to_vec(self)?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

/// Sites in the first hidden-resource delivery scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Site {
    Depot,
    Crossing,
    Outpost,
}

/// Generated layout for the current scenario family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioLayout {
    pub filter_location: Site,
}

/// Resources that can be discovered and collected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Item {
    WaterFilter,
    ValveKey,
}

/// The evaluator's gate state. Agents see it only after inspecting the crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GateState {
    Closed,
    Open,
    Failed,
}

/// Facts that become available only through the observation action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Clue {
    DeliveryGoal,
    FilterStoredAt(Site),
    ValveKeyStoredAtDepot,
    ValveKeyOpensCrossingGate,
    GateRequiresRecoveryAfterFailure,
    OutpostIsDeliveryDestination,
}

/// Agent-visible actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Inspect,
    Move(Site),
    Collect(Item),
    UseValveKey,
    Deliver,
    Wait,
}

/// Exogenous evaluator interventions are separate from agent actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Perturbation {
    FailCrossingGate,
}

/// Full evaluator-side world state. Do not pass this structure to an agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldState {
    pub tick: u32,
    pub location: Site,
    pub inventory: BTreeSet<Item>,
    pub items_at: BTreeSet<(Site, Item)>,
    pub inspected_sites: BTreeSet<Site>,
    pub known_clues: BTreeSet<Clue>,
    pub gate_state: GateState,
    pub delivered: bool,
    pub observation_budget_remaining: u32,
    pub event_log: Vec<Event>,
    pub manifest_digest: String,
}

impl WorldState {
    /// Create the evaluator's initial state for the manifest.
    pub fn initial(manifest: &ScenarioManifest) -> Result<Self, serde_json::Error> {
        let layout = manifest.layout();
        Ok(Self {
            tick: 0,
            location: Site::Depot,
            inventory: BTreeSet::new(),
            items_at: [
                (layout.filter_location, Item::WaterFilter),
                (Site::Depot, Item::ValveKey),
            ]
            .into_iter()
            .collect(),
            inspected_sites: BTreeSet::new(),
            known_clues: BTreeSet::new(),
            gate_state: GateState::Closed,
            delivered: false,
            observation_budget_remaining: manifest.observation_budget,
            event_log: Vec::new(),
            manifest_digest: manifest.digest()?,
        })
    }

    /// Stable digest of the full evaluator state, including its event history.
    pub fn digest(&self) -> Result<String, serde_json::Error> {
        let bytes = serde_json::to_vec(self)?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

/// Partial state returned to the agent after a transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub tick: u32,
    pub location: Site,
    pub goal: String,
    pub visible_items: Vec<Item>,
    pub known_clues: Vec<Clue>,
    pub gate_state: Option<GateState>,
    pub delivery_complete: Option<bool>,
    pub observation_budget_remaining: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    Inspected(Site),
    ActionRejected { action: Action, reason: String },
    Moved { from: Site, to: Site },
    Collected(Item),
    GateOpened,
    Delivered,
    Waited,
    Perturbed(Perturbation),
}

/// Ordered command trace used to deterministically replay a scenario.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordedCommand {
    Agent(Action),
    Evaluator(Perturbation),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub tick: u32,
    pub kind: EventKind,
}

/// One attempted transition's evidence receipt, including rejected actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepReceipt {
    pub manifest_digest: String,
    pub step_index: u32,
    pub before_state_digest: String,
    pub after_state_digest: String,
    pub event: Event,
    pub observation: Observation,
}

/// Rejected transitions are transactional and do not advance the world. A
/// future agent-facing runner must separately log and charge rejected attempts
/// so invalid-action probing cannot become a free oracle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionError {
    StepLimitReached,
    ObservationBudgetExhausted,
    NotAdjacent { from: Site, to: Site },
    GateNotTraversable(GateState),
    ItemUnavailable { site: Site, item: Item },
    SiteNotInspected(Site),
    MechanismNotDiscovered,
    MissingValveKey,
    WrongLocation { expected: Site, actual: Site },
    MissingWaterFilter,
    AlreadyDelivered,
    ManifestMismatch,
    Serialization(String),
    ReplayMismatch,
    PlannerExhausted,
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepLimitReached => write!(f, "scenario step limit reached"),
            Self::ObservationBudgetExhausted => write!(f, "observation budget exhausted"),
            Self::NotAdjacent { from, to } => write!(f, "no direct route from {from:?} to {to:?}"),
            Self::GateNotTraversable(state) => write!(f, "crossing gate is not traversable: {state:?}"),
            Self::ItemUnavailable { site, item } => write!(f, "{item:?} is not available at {site:?}"),
            Self::SiteNotInspected(site) => write!(f, "inspect {site:?} before collecting hidden resources"),
            Self::MechanismNotDiscovered => write!(f, "inspect the crossing to discover the gate mechanism"),
            Self::MissingValveKey => write!(f, "valve key is required to operate the crossing gate"),
            Self::WrongLocation { expected, actual } => write!(f, "action requires {expected:?}, currently at {actual:?}"),
            Self::MissingWaterFilter => write!(f, "water filter is required for delivery"),
            Self::AlreadyDelivered => write!(f, "delivery goal is already complete"),
            Self::ManifestMismatch => write!(f, "world state belongs to a different scenario manifest"),
            Self::Serialization(message) => write!(f, "could not serialize transition evidence: {message}"),
            Self::ReplayMismatch => write!(f, "recorded receipts do not match deterministic replay"),
            Self::PlannerExhausted => write!(f, "search oracle found no valid plan within the step budget"),
        }
    }
}

impl std::error::Error for TransitionError {}

/// Build the agent-visible observation from the evaluator-side state.
pub fn observe(state: &WorldState) -> Observation {
    let inspected = state.inspected_sites.contains(&state.location);
    let visible_items = if inspected {
        state
            .items_at
            .iter()
            .filter_map(|(site, item)| (*site == state.location).then_some(*item))
            .collect()
    } else {
        Vec::new()
    };

    Observation {
        tick: state.tick,
        location: state.location,
        goal: "Deliver the water filter to the outpost.".to_string(),
        visible_items,
        known_clues: state.known_clues.iter().copied().collect(),
        gate_state: (state.location == Site::Crossing && inspected).then_some(state.gate_state),
        delivery_complete: (state.location == Site::Outpost).then_some(state.delivered),
        observation_budget_remaining: state.observation_budget_remaining,
    }
}

/// Outcome of an agent action attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptOutcome {
    Accepted,
    Rejected(TransitionError),
}

/// Runner-facing action attempt. Unlike `step`, a rejected action is charged
/// one tick and recorded as an event. This prevents free invalid-action probing.
pub fn attempt(
    manifest: &ScenarioManifest,
    state: &WorldState,
    action: Action,
) -> Result<(WorldState, StepReceipt, AttemptOutcome), TransitionError> {
    match step(manifest, state, action.clone()) {
        Ok((next, receipt)) => Ok((next, receipt, AttemptOutcome::Accepted)),
        Err(error @ (TransitionError::StepLimitReached
            | TransitionError::ManifestMismatch
            | TransitionError::Serialization(_))) => Err(error),
        Err(error) => {
            let before_state_digest = state
                .digest()
                .map_err(|serialization| TransitionError::Serialization(serialization.to_string()))?;
            let mut next = state.clone();
            next.tick += 1;
            let event = Event {
                tick: next.tick,
                kind: EventKind::ActionRejected { action, reason: error.to_string() },
            };
            next.event_log.push(event.clone());
            let after_state_digest = next
                .digest()
                .map_err(|serialization| TransitionError::Serialization(serialization.to_string()))?;
            let receipt = StepReceipt {
                manifest_digest: next.manifest_digest.clone(),
                step_index: next.tick,
                before_state_digest,
                after_state_digest,
                event,
                observation: observe(&next),
            };
            Ok((next, receipt, AttemptOutcome::Rejected(error)))
        }
    }
}

/// Apply one agent action. Failure leaves the caller's state unchanged.
pub fn step(
    manifest: &ScenarioManifest,
    state: &WorldState,
    action: Action,
) -> Result<(WorldState, StepReceipt), TransitionError> {
    if state.tick >= manifest.max_steps {
        return Err(TransitionError::StepLimitReached);
    }

    let manifest_digest = manifest
        .digest()
        .map_err(|error| TransitionError::Serialization(error.to_string()))?;
    if manifest_digest != state.manifest_digest {
        return Err(TransitionError::ManifestMismatch);
    }
    let before_state_digest = state
        .digest()
        .map_err(|error| TransitionError::Serialization(error.to_string()))?;
    let mut next = state.clone();
    let next_tick = next.tick + 1;

    let kind = match action {
        Action::Inspect => {
            if next.observation_budget_remaining == 0 {
                return Err(TransitionError::ObservationBudgetExhausted);
            }
            next.observation_budget_remaining -= 1;
            next.inspected_sites.insert(next.location);
            match next.location {
                Site::Depot => {
                    next.known_clues.insert(Clue::DeliveryGoal);
                    if next.items_at.contains(&(Site::Depot, Item::WaterFilter)) {
                        next.known_clues.insert(Clue::FilterStoredAt(Site::Depot));
                    }
                    if next.items_at.contains(&(Site::Depot, Item::ValveKey)) {
                        next.known_clues.insert(Clue::ValveKeyStoredAtDepot);
                    }
                }
                Site::Crossing => {
                    next.known_clues.insert(Clue::ValveKeyOpensCrossingGate);
                    if next.gate_state == GateState::Failed {
                        next.known_clues.insert(Clue::GateRequiresRecoveryAfterFailure);
                    }
                }
                Site::Outpost => {
                    next.known_clues.insert(Clue::OutpostIsDeliveryDestination);
                    if next.items_at.contains(&(Site::Outpost, Item::WaterFilter)) {
                        next.known_clues.insert(Clue::FilterStoredAt(Site::Outpost));
                    }
                }
            }
            EventKind::Inspected(next.location)
        }
        Action::Move(destination) => {
            let from = next.location;
            let adjacent = matches!(
                (from, destination),
                (Site::Depot, Site::Crossing)
                    | (Site::Crossing, Site::Depot)
                    | (Site::Crossing, Site::Outpost)
                    | (Site::Outpost, Site::Crossing)
            );
            if !adjacent {
                return Err(TransitionError::NotAdjacent { from, to: destination });
            }
            if (from == Site::Crossing && destination == Site::Outpost)
                || (from == Site::Outpost && destination == Site::Crossing)
            {
                if next.gate_state != GateState::Open {
                    return Err(TransitionError::GateNotTraversable(next.gate_state));
                }
            }
            next.location = destination;
            EventKind::Moved { from, to: destination }
        }
        Action::Collect(item) => {
            if !next.inspected_sites.contains(&next.location) {
                return Err(TransitionError::SiteNotInspected(next.location));
            }
            if !next.items_at.remove(&(next.location, item)) {
                return Err(TransitionError::ItemUnavailable {
                    site: next.location,
                    item,
                });
            }
            next.inventory.insert(item);
            EventKind::Collected(item)
        }
        Action::UseValveKey => {
            if next.location != Site::Crossing {
                return Err(TransitionError::WrongLocation {
                    expected: Site::Crossing,
                    actual: next.location,
                });
            }
            if !next.known_clues.contains(&Clue::ValveKeyOpensCrossingGate) {
                return Err(TransitionError::MechanismNotDiscovered);
            }
            if !next.inventory.contains(&Item::ValveKey) {
                return Err(TransitionError::MissingValveKey);
            }
            next.gate_state = GateState::Open;
            EventKind::GateOpened
        }
        Action::Deliver => {
            if next.delivered {
                return Err(TransitionError::AlreadyDelivered);
            }
            if next.location != Site::Outpost {
                return Err(TransitionError::WrongLocation {
                    expected: Site::Outpost,
                    actual: next.location,
                });
            }
            if !next.inventory.remove(&Item::WaterFilter) {
                return Err(TransitionError::MissingWaterFilter);
            }
            next.delivered = true;
            EventKind::Delivered
        }
        Action::Wait => EventKind::Waited,
    };

    next.tick = next_tick;
    let event = Event { tick: next_tick, kind };
    next.event_log.push(event.clone());
    let after_state_digest = next
        .digest()
        .map_err(|error| TransitionError::Serialization(error.to_string()))?;
    let receipt = StepReceipt {
        manifest_digest: next.manifest_digest.clone(),
        step_index: next_tick,
        before_state_digest,
        after_state_digest,
        event,
        observation: observe(&next),
    };
    Ok((next, receipt))
}

/// Replay a complete agent/evaluator command trace from the manifest's initial state.
///
/// The returned receipts are suitable for independent digest comparison. The
/// manifest digest is checked on every transition, so replaying against a
/// modified scenario fails closed.
pub fn replay(
    manifest: &ScenarioManifest,
    commands: &[RecordedCommand],
) -> Result<(WorldState, Vec<StepReceipt>), TransitionError> {
    let mut state = WorldState::initial(manifest)
        .map_err(|error| TransitionError::Serialization(error.to_string()))?;
    let mut receipts = Vec::with_capacity(commands.len());

    for command in commands {
        let (next, receipt) = match command {
            RecordedCommand::Agent(action) => {
                let (next, receipt, _outcome) = attempt(manifest, &state, action.clone())?;
                (next, receipt)
            }
            RecordedCommand::Evaluator(perturbation) => perturb(manifest, &state, *perturbation)?,
        };
        state = next;
        receipts.push(receipt);
    }

    Ok((state, receipts))
}

/// Evaluator-computed outcome for a fully replayed command trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioScore {
    pub goal_completed: bool,
    pub transitions_used: u32,
    pub agent_attempts: u32,
    pub invalid_actions: u32,
    pub inspections: u32,
    pub evaluator_perturbations: u32,
    pub successful_recoveries: u32,
    pub optimal_agent_steps: u32,
    pub optimal_resource_cost_units: u64,
    /// 0 when the goal is incomplete; otherwise optimal steps / actual agent attempts, capped at 1.
    pub action_efficiency: f64,
    /// 0 when incomplete; otherwise minimum valid plan cost / actual cost, capped at 1.
    pub cost_efficiency: f64,
    /// Sum of benchmark-defined cost units across agent actions and rejected attempts.
    pub resource_cost_units: u64,
    /// 0 when incomplete; otherwise completed goal per benchmark-defined cost unit.
    pub goal_completion_per_cost_unit: f64,
}

/// Auditable output of a deterministic trace evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluatedTrace {
    pub scenario_id: String,
    pub manifest_digest: String,
    pub final_state_digest: String,
    pub receipts: Vec<StepReceipt>,
    pub score: ScenarioScore,
}

/// Minimal behavioral state used by the search oracle. Event history and
/// timestamps are intentionally omitted: BFS reaches equivalent states first
/// by the shortest action sequence, and a shorter path dominates a longer one.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PlannerKey {
    location: Site,
    inventory: BTreeSet<Item>,
    items_at: BTreeSet<(Site, Item)>,
    inspected_sites: BTreeSet<Site>,
    known_clues: BTreeSet<Clue>,
    gate_state: GateState,
    delivered: bool,
    observation_budget_remaining: u32,
}

impl From<&WorldState> for PlannerKey {
    fn from(state: &WorldState) -> Self {
        Self {
            location: state.location,
            inventory: state.inventory.clone(),
            items_at: state.items_at.clone(),
            inspected_sites: state.inspected_sites.clone(),
            known_clues: state.known_clues.clone(),
            gate_state: state.gate_state,
            delivered: state.delivered,
            observation_budget_remaining: state.observation_budget_remaining,
        }
    }
}

/// Convert a state-transition event to the manifest's benchmark-internal cost units.
fn event_cost_units(kind: &EventKind, model: CostModel) -> u64 {
    match kind {
        EventKind::Inspected(_) => model.inspect as u64,
        EventKind::Moved { .. } => model.move_between_sites as u64,
        EventKind::Collected(_) | EventKind::GateOpened | EventKind::Delivered => {
            model.interaction as u64
        }
        EventKind::Waited => model.wait as u64,
        EventKind::ActionRejected { action, .. } => {
            let action_cost = match action {
                Action::Inspect => model.inspect,
                Action::Move(_) => model.move_between_sites,
                Action::Collect(_) | Action::UseValveKey | Action::Deliver => model.interaction,
                Action::Wait => model.wait,
            };
            action_cost.saturating_add(model.invalid_action_surcharge) as u64
        }
        EventKind::Perturbed(_) => 0,
    }
}

/// Breadth-first search for a shortest valid action sequence in the current
/// fully specified world. This is an evaluator-only oracle and must never be
/// exposed to the agent being evaluated.
fn shortest_action_plan(manifest: &ScenarioManifest) -> Option<Vec<Action>> {
    let initial = WorldState::initial(manifest).ok()?;
    let mut frontier = VecDeque::from([(initial, Vec::<Action>::new())]);
    let mut visited = BTreeSet::new();
    visited.insert(PlannerKey::from(&frontier.front()?.0));

    let actions = [
        Action::Inspect,
        Action::Move(Site::Depot),
        Action::Move(Site::Crossing),
        Action::Move(Site::Outpost),
        Action::Collect(Item::WaterFilter),
        Action::Collect(Item::ValveKey),
        Action::UseValveKey,
        Action::Deliver,
        Action::Wait,
    ];

    while let Some((state, plan)) = frontier.pop_front() {
        if state.delivered {
            return Some(plan);
        }
        if state.tick >= manifest.max_steps {
            continue;
        }

        for action in &actions {
            let Ok((next, _receipt)) = step(manifest, &state, action.clone()) else {
                continue;
            };
            let key = PlannerKey::from(&next);
            if !visited.insert(key) {
                continue;
            }
            let mut next_plan = plan.clone();
            next_plan.push(action.clone());
            frontier.push_back((next, next_plan));
        }
    }

    None
}

/// Dijkstra search for a valid plan with minimum benchmark-defined action cost.
/// Ties are broken by fewer actions, then by the deterministic state ordering.
/// This uses full evaluator state and is an evaluator ceiling, not an agent policy.
fn least_cost_action_plan(manifest: &ScenarioManifest) -> Option<(Vec<Action>, u64)> {
    let initial = WorldState::initial(manifest).ok()?;
    let initial_key = PlannerKey::from(&initial);
    let mut frontier = BinaryHeap::new();
    frontier.push(Reverse((0_u64, 0_u32, initial_key.clone())));

    let mut best: BTreeMap<PlannerKey, (u64, u32, Vec<Action>, WorldState)> = BTreeMap::new();
    best.insert(initial_key, (0, 0, Vec::new(), initial));

    let actions = [
        Action::Inspect,
        Action::Move(Site::Depot),
        Action::Move(Site::Crossing),
        Action::Move(Site::Outpost),
        Action::Collect(Item::WaterFilter),
        Action::Collect(Item::ValveKey),
        Action::UseValveKey,
        Action::Deliver,
        Action::Wait,
    ];

    while let Some(Reverse((cost, steps, key))) = frontier.pop() {
        let Some((known_cost, known_steps, plan, state)) = best
            .get(&key)
            .map(|(known_cost, known_steps, plan, state)| {
                (*known_cost, *known_steps, plan.clone(), state.clone())
            })
        else {
            continue;
        };
        if (cost, steps) != (known_cost, known_steps) {
            continue;
        }
        if state.delivered {
            return Some((plan, cost));
        }
        if steps >= manifest.max_steps {
            continue;
        }

        for action in &actions {
            let Ok((next, receipt)) = step(manifest, &state, action.clone()) else {
                continue;
            };
            let next_cost = cost.saturating_add(event_cost_units(
                &receipt.event.kind,
                manifest.cost_model,
            ));
            let next_steps = steps + 1;
            let next_key = PlannerKey::from(&next);
            let improves = best
                .get(&next_key)
                .map(|(existing_cost, existing_steps, _, _)| {
                    (next_cost, next_steps) < (*existing_cost, *existing_steps)
                })
                .unwrap_or(true);
            if !improves {
                continue;
            }

            let mut next_plan = plan.clone();
            next_plan.push(action.clone());
            best.insert(
                next_key.clone(),
                (next_cost, next_steps, next_plan, next),
            );
            frontier.push(Reverse((next_cost, next_steps, next_key)));
        }
    }

    None
}

/// Search-based cost oracle for the current environment specification.
pub fn run_cost_oracle(
    manifest: &ScenarioManifest,
) -> Result<EvaluatedTrace, TransitionError> {
    let (plan, _minimum_cost) =
        least_cost_action_plan(manifest).ok_or(TransitionError::PlannerExhausted)?;
    let commands = plan
        .into_iter()
        .map(RecordedCommand::Agent)
        .collect::<Vec<_>>();
    evaluate_trace(manifest, &commands)
}

/// Search-based evaluator baseline for the current environment specification.
/// It has access to full evaluator state and is a ceiling/oracle, not a fair
/// black-box competitor.
pub fn run_search_oracle(
    manifest: &ScenarioManifest,
) -> Result<EvaluatedTrace, TransitionError> {
    let plan = shortest_action_plan(manifest).ok_or(TransitionError::PlannerExhausted)?;
    let commands = plan
        .into_iter()
        .map(RecordedCommand::Agent)
        .collect::<Vec<_>>();
    evaluate_trace(manifest, &commands)
}

fn score_replayed_state(
    manifest: &ScenarioManifest,
    state: &WorldState,
    commands: &[RecordedCommand],
) -> ScenarioScore {
    let agent_attempts = commands
        .iter()
        .filter(|command| matches!(command, RecordedCommand::Agent(_)))
        .count() as u32;
    let invalid_actions = state
        .event_log
        .iter()
        .filter(|event| matches!(&event.kind, EventKind::ActionRejected { .. }))
        .count() as u32;
    let inspections = state
        .event_log
        .iter()
        .filter(|event| matches!(&event.kind, EventKind::Inspected(_)))
        .count() as u32;
    let evaluator_perturbations = state
        .event_log
        .iter()
        .filter(|event| matches!(&event.kind, EventKind::Perturbed(_)))
        .count() as u32;

    let mut awaiting_recovery = false;
    let mut successful_recoveries = 0;
    for event in &state.event_log {
        match &event.kind {
            EventKind::Perturbed(_) => awaiting_recovery = true,
            EventKind::GateOpened if awaiting_recovery => {
                successful_recoveries += 1;
                awaiting_recovery = false;
            }
            _ => {}
        }
    }

    let optimal_agent_steps = shortest_action_plan(manifest)
        .map(|plan| plan.len() as u32)
        .unwrap_or(0);
    let optimal_resource_cost_units = least_cost_action_plan(manifest)
        .map(|(_, cost)| cost)
        .unwrap_or(0);
    let resource_cost_units = state
        .event_log
        .iter()
        .map(|event| event_cost_units(&event.kind, manifest.cost_model))
        .sum::<u64>();

    let action_efficiency = if state.delivered && agent_attempts > 0 && optimal_agent_steps > 0 {
        (optimal_agent_steps as f64 / agent_attempts as f64).min(1.0)
    } else {
        0.0
    };
    let cost_efficiency = if state.delivered
        && resource_cost_units > 0
        && optimal_resource_cost_units > 0
    {
        (optimal_resource_cost_units as f64 / resource_cost_units as f64).min(1.0)
    } else {
        0.0
    };
    let goal_completion_per_cost_unit = if state.delivered && resource_cost_units > 0 {
        1.0 / resource_cost_units as f64
    } else {
        0.0
    };

    ScenarioScore {
        goal_completed: state.delivered,
        transitions_used: state.tick,
        agent_attempts,
        invalid_actions,
        inspections,
        evaluator_perturbations,
        successful_recoveries,
        optimal_agent_steps,
        optimal_resource_cost_units,
        action_efficiency,
        cost_efficiency,
        resource_cost_units,
        goal_completion_per_cost_unit,
    }
}

/// Replay and score a command trace using evaluator-only state.
///
/// This function never trusts a caller-supplied final state or score. It
/// reconstructs the state from the manifest and commands before scoring.
pub fn evaluate_trace(
    manifest: &ScenarioManifest,
    commands: &[RecordedCommand],
) -> Result<EvaluatedTrace, TransitionError> {
    let (state, receipts) = replay(manifest, commands)?;
    let final_state_digest = state
        .digest()
        .map_err(|error| TransitionError::Serialization(error.to_string()))?;
    Ok(EvaluatedTrace {
        scenario_id: manifest.scenario_id.clone(),
        manifest_digest: manifest
            .digest()
            .map_err(|error| TransitionError::Serialization(error.to_string()))?,
        final_state_digest,
        score: score_replayed_state(manifest, &state, commands),
        receipts,
    })
}

/// Recompute a trace and reject any receipt mismatch.
pub fn verify_trace(
    manifest: &ScenarioManifest,
    commands: &[RecordedCommand],
    expected_receipts: &[StepReceipt],
) -> Result<EvaluatedTrace, TransitionError> {
    let evaluated = evaluate_trace(manifest, commands)?;
    if evaluated.receipts != expected_receipts {
        return Err(TransitionError::ReplayMismatch);
    }
    Ok(evaluated)
}

/// Export one JSON receipt per line for artifact storage.
pub fn receipts_to_jsonl(receipts: &[StepReceipt]) -> Result<String, serde_json::Error> {
    let mut output = String::new();
    for receipt in receipts {
        output.push_str(&serde_json::to_string(receipt)?);
        output.push('\n');
    }
    Ok(output)
}

fn push_agent_action(
    manifest: &ScenarioManifest,
    state: &mut WorldState,
    commands: &mut Vec<RecordedCommand>,
    action: Action,
) -> Result<(), TransitionError> {
    let (next, _receipt, outcome) = attempt(manifest, state, action.clone())?;
    *state = next;
    commands.push(RecordedCommand::Agent(action));
    match outcome {
        AttemptOutcome::Accepted => Ok(()),
        AttemptOutcome::Rejected(error) => Err(error),
    }
}

fn push_evaluator_perturbation(
    manifest: &ScenarioManifest,
    state: &mut WorldState,
    commands: &mut Vec<RecordedCommand>,
    perturbation: Perturbation,
) -> Result<(), TransitionError> {
    let (next, _receipt) = perturb(manifest, state, perturbation)?;
    *state = next;
    commands.push(RecordedCommand::Evaluator(perturbation));
    Ok(())
}

/// A transparent scripted baseline that solves the current generated layout.
/// When requested, the evaluator injects a gate failure and the policy recovers.
pub fn run_reference_policy(
    manifest: &ScenarioManifest,
    inject_gate_failure: bool,
) -> Result<EvaluatedTrace, TransitionError> {
    let mut state = WorldState::initial(manifest)
        .map_err(|error| TransitionError::Serialization(error.to_string()))?;
    let mut commands = Vec::new();

    push_agent_action(manifest, &mut state, &mut commands, Action::Inspect)?;
    push_agent_action(
        manifest,
        &mut state,
        &mut commands,
        Action::Collect(Item::ValveKey),
    )?;
    if manifest.layout().filter_location == Site::Depot {
        push_agent_action(
            manifest,
            &mut state,
            &mut commands,
            Action::Collect(Item::WaterFilter),
        )?;
    }
    push_agent_action(
        manifest,
        &mut state,
        &mut commands,
        Action::Move(Site::Crossing),
    )?;
    push_agent_action(manifest, &mut state, &mut commands, Action::Inspect)?;
    push_agent_action(manifest, &mut state, &mut commands, Action::UseValveKey)?;

    if inject_gate_failure {
        push_evaluator_perturbation(
            manifest,
            &mut state,
            &mut commands,
            Perturbation::FailCrossingGate,
        )?;
        push_agent_action(manifest, &mut state, &mut commands, Action::Inspect)?;
        push_agent_action(manifest, &mut state, &mut commands, Action::UseValveKey)?;
    }

    push_agent_action(
        manifest,
        &mut state,
        &mut commands,
        Action::Move(Site::Outpost),
    )?;
    if manifest.layout().filter_location == Site::Outpost {
        push_agent_action(manifest, &mut state, &mut commands, Action::Inspect)?;
        push_agent_action(
            manifest,
            &mut state,
            &mut commands,
            Action::Collect(Item::WaterFilter),
        )?;
    }
    push_agent_action(manifest, &mut state, &mut commands, Action::Deliver)?;
    evaluate_trace(manifest, &commands)
}

/// Deterministic random-action baseline. It sees only that the episode has
/// ended; action choices are generated from a fixed SplitMix64 stream.
pub fn run_random_policy(manifest: &ScenarioManifest) -> Result<EvaluatedTrace, TransitionError> {
    let mut state = WorldState::initial(manifest)
        .map_err(|error| TransitionError::Serialization(error.to_string()))?;
    let mut commands = Vec::new();
    let mut rng = manifest.seed ^ 0xD1B5_4A32_D192_ED03;

    for _ in 0..manifest.max_steps {
        if state.delivered {
            break;
        }
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = rng;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;

        let action = match value % 9 {
            0 => Action::Inspect,
            1 => Action::Move(Site::Depot),
            2 => Action::Move(Site::Crossing),
            3 => Action::Move(Site::Outpost),
            4 => Action::Collect(Item::WaterFilter),
            5 => Action::Collect(Item::ValveKey),
            6 => Action::UseValveKey,
            7 => Action::Deliver,
            _ => Action::Wait,
        };
        let (next, _receipt, _outcome) = attempt(manifest, &state, action.clone())?;
        state = next;
        commands.push(RecordedCommand::Agent(action));
    }

    evaluate_trace(manifest, &commands)
}

/// Apply an evaluator-only regime shift. It is not available to the agent.
pub fn perturb(
    manifest: &ScenarioManifest,
    state: &WorldState,
    perturbation: Perturbation,
) -> Result<(WorldState, StepReceipt), TransitionError> {
    if state.tick >= manifest.max_steps {
        return Err(TransitionError::StepLimitReached);
    }
    let manifest_digest = manifest
        .digest()
        .map_err(|error| TransitionError::Serialization(error.to_string()))?;
    if manifest_digest != state.manifest_digest {
        return Err(TransitionError::ManifestMismatch);
    }
    let before_state_digest = state
        .digest()
        .map_err(|error| TransitionError::Serialization(error.to_string()))?;
    let mut next = state.clone();
    next.tick += 1;
    match perturbation {
        Perturbation::FailCrossingGate => {
            next.gate_state = GateState::Failed;
            next.inspected_sites.remove(&Site::Crossing);
        }
    }
    let event = Event {
        tick: next.tick,
        kind: EventKind::Perturbed(perturbation),
    };
    next.event_log.push(event.clone());
    let after_state_digest = next
        .digest()
        .map_err(|error| TransitionError::Serialization(error.to_string()))?;
    let receipt = StepReceipt {
        manifest_digest: next.manifest_digest.clone(),
        step_index: next.tick,
        before_state_digest,
        after_state_digest,
        event,
        observation: observe(&next),
    };
    Ok((next, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_action(state: &WorldState, manifest: &ScenarioManifest, action: Action) -> WorldState {
        step(manifest, state, action)
            .expect("valid action")
            .0
    }

    #[test]
    fn hidden_resources_are_not_disclosed_before_inspection() {
        let manifest = ScenarioManifest::default();
        let state = WorldState::initial(&manifest).expect("initial state");
        let observation = observe(&state);
        assert!(observation.visible_items.is_empty());
        assert!(observation.known_clues.is_empty());

        let (inspected, receipt) = step(&manifest, &state, Action::Inspect).expect("inspect");
        assert_eq!(receipt.observation.visible_items.len(), 2);
        assert!(receipt.observation.known_clues.contains(&Clue::ValveKeyStoredAtDepot));
        assert_eq!(inspected.observation_budget_remaining, manifest.observation_budget - 1);
    }

    #[test]
    fn delivery_requires_discovery_collection_and_gate_operation() {
        let manifest = ScenarioManifest::default();
        let mut state = WorldState::initial(&manifest).expect("initial state");
        state = run_action(&state, &manifest, Action::Inspect);
        state = run_action(&state, &manifest, Action::Collect(Item::WaterFilter));
        state = run_action(&state, &manifest, Action::Collect(Item::ValveKey));
        state = run_action(&state, &manifest, Action::Move(Site::Crossing));

        assert!(matches!(
            step(&manifest, &state, Action::Move(Site::Outpost)),
            Err(TransitionError::GateNotTraversable(GateState::Closed))
        ));

        state = run_action(&state, &manifest, Action::Inspect);
        state = run_action(&state, &manifest, Action::UseValveKey);
        state = run_action(&state, &manifest, Action::Move(Site::Outpost));
        state = run_action(&state, &manifest, Action::Deliver);
        assert!(state.delivered);
        assert!(!state.inventory.contains(&Item::WaterFilter));
    }

    #[test]
    fn exogenous_gate_failure_requires_agent_recovery() {
        let manifest = ScenarioManifest::default();
        let mut state = WorldState::initial(&manifest).expect("initial state");
        state = run_action(&state, &manifest, Action::Inspect);
        state = run_action(&state, &manifest, Action::Collect(Item::ValveKey));
        state = run_action(&state, &manifest, Action::Move(Site::Crossing));
        state = run_action(&state, &manifest, Action::Inspect);
        state = run_action(&state, &manifest, Action::UseValveKey);
        let (failed, receipt) =
            perturb(&manifest, &state, Perturbation::FailCrossingGate).expect("perturbation");

        assert_eq!(failed.gate_state, GateState::Failed);
        assert!(matches!(
            step(&manifest, &failed, Action::Move(Site::Outpost)),
            Err(TransitionError::GateNotTraversable(GateState::Failed))
        ));
        let (inspected, inspect_receipt) =
            step(&manifest, &failed, Action::Inspect).expect("inspect failure");
        assert!(inspect_receipt
            .observation
            .known_clues
            .contains(&Clue::GateRequiresRecoveryAfterFailure));
        let recovered = run_action(&inspected, &manifest, Action::UseValveKey);
        assert_eq!(recovered.gate_state, GateState::Open);
        assert_eq!(receipt.manifest_digest, inspect_receipt.manifest_digest);
    }

    #[test]
    fn failed_actions_do_not_mutate_the_input_state() {
        let manifest = ScenarioManifest::default();
        let state = WorldState::initial(&manifest).expect("initial state");
        let before = state.digest().expect("digest");
        assert!(step(&manifest, &state, Action::Deliver).is_err());
        assert_eq!(state.digest().expect("digest"), before);
    }

    #[test]
    fn observation_budget_is_enforced() {
        let manifest = ScenarioManifest {
            observation_budget: 1,
            ..ScenarioManifest::default()
        };
        let state = WorldState::initial(&manifest).expect("initial state");
        let observed = run_action(&state, &manifest, Action::Inspect);
        assert_eq!(observed.observation_budget_remaining, 0);
        assert_eq!(
            step(&manifest, &observed, Action::Inspect),
            Err(TransitionError::ObservationBudgetExhausted)
        );
    }

    #[test]
    fn attempt_charges_observation_budget_exhaustion() {
        let manifest = ScenarioManifest {
            observation_budget: 1,
            ..ScenarioManifest::default()
        };
        let initial = WorldState::initial(&manifest).expect("initial state");
        let (observed, _, first_outcome) =
            attempt(&manifest, &initial, Action::Inspect).expect("first inspection");
        assert_eq!(first_outcome, AttemptOutcome::Accepted);
        let (rejected, receipt, outcome) =
            attempt(&manifest, &observed, Action::Inspect).expect("charged over-budget attempt");
        assert_eq!(rejected.tick, observed.tick + 1);
        assert_eq!(
            outcome,
            AttemptOutcome::Rejected(TransitionError::ObservationBudgetExhausted)
        );
        assert!(matches!(&receipt.event.kind, EventKind::ActionRejected { .. }));
        assert_eq!(rejected.observation_budget_remaining, 0);
    }

    #[test]
    fn scenario_step_limit_is_enforced() {
        let manifest = ScenarioManifest {
            max_steps: 1,
            ..ScenarioManifest::default()
        };
        let state = WorldState::initial(&manifest).expect("initial state");
        let state = run_action(&state, &manifest, Action::Wait);
        assert_eq!(
            step(&manifest, &state, Action::Wait),
            Err(TransitionError::StepLimitReached)
        );
    }

    #[test]
    fn scenario_seed_generates_two_hidden_filter_layouts() {
        let depot_manifest = ScenarioManifest { seed: 42, ..ScenarioManifest::default() };
        let outpost_manifest = ScenarioManifest { seed: 43, ..ScenarioManifest::default() };
        assert_eq!(depot_manifest.layout().filter_location, Site::Depot);
        assert_eq!(outpost_manifest.layout().filter_location, Site::Outpost);

        let depot_state = WorldState::initial(&depot_manifest).expect("depot layout");
        let outpost_state = WorldState::initial(&outpost_manifest).expect("outpost layout");
        assert!(depot_state.items_at.contains(&(Site::Depot, Item::WaterFilter)));
        assert!(outpost_state.items_at.contains(&(Site::Outpost, Item::WaterFilter)));
    }

    #[test]
    fn rejected_agent_actions_consume_budget_and_emit_receipts() {
        let manifest = ScenarioManifest::default();
        let state = WorldState::initial(&manifest).expect("initial state");
        let (next, receipt, outcome) =
            attempt(&manifest, &state, Action::Deliver).expect("charged invalid attempt");

        assert_eq!(state.tick, 0, "caller state remains immutable");
        assert_eq!(next.tick, 1, "rejected attempt consumes one tick");
        assert!(matches!(outcome, AttemptOutcome::Rejected(TransitionError::WrongLocation { .. })));
        assert!(matches!(receipt.event.kind, EventKind::ActionRejected { .. }));
        assert_eq!(receipt.step_index, 1);
        assert_ne!(receipt.before_state_digest, receipt.after_state_digest);
    }

    #[test]
    fn cost_oracle_matches_or_improves_shortest_path_cost() {
        for seed in [42, 43] {
            let manifest = ScenarioManifest { seed, ..ScenarioManifest::default() };
            let shortest = run_search_oracle(&manifest).expect("shortest-action oracle");
            let least_cost = run_cost_oracle(&manifest).expect("least-cost oracle");
            assert!(least_cost.score.goal_completed);
            assert_eq!(
                least_cost.score.resource_cost_units,
                least_cost.score.optimal_resource_cost_units
            );
            assert_eq!(least_cost.score.cost_efficiency, 1.0);
            assert!(
                least_cost.score.resource_cost_units <= shortest.score.resource_cost_units
            );
        }
    }

    #[test]
    fn cost_oracle_respects_manifest_action_costs() {
        let manifest = ScenarioManifest {
            cost_model: CostModel { inspect: 5, ..CostModel::default() },
            ..ScenarioManifest::default()
        };
        let run = run_cost_oracle(&manifest).expect("least-cost oracle with modified costs");
        assert_eq!(run.score.resource_cost_units, 16);
        assert_eq!(run.score.optimal_resource_cost_units, 16);
        assert_eq!(run.score.cost_efficiency, 1.0);
    }

    #[test]
    fn search_oracle_fails_closed_when_step_budget_is_insufficient() {
        let manifest = ScenarioManifest {
            max_steps: 1,
            ..ScenarioManifest::default()
        };
        assert_eq!(
            run_search_oracle(&manifest),
            Err(TransitionError::PlannerExhausted)
        );
    }

    #[test]
    fn search_oracle_matches_shortest_reference_action_counts() {
        for seed in [42, 43] {
            let manifest = ScenarioManifest { seed, ..ScenarioManifest::default() };
            let oracle = run_search_oracle(&manifest).expect("search oracle");
            let reference = run_reference_policy(&manifest, false).expect("reference policy");
            assert!(oracle.score.goal_completed);
            assert_eq!(
                oracle.score.agent_attempts,
                reference.score.agent_attempts,
                "scripted policy should match shortest plan for seed {seed}"
            );
            assert_eq!(
                oracle.score.optimal_agent_steps,
                oracle.score.agent_attempts
            );
            assert_eq!(
                oracle.score.resource_cost_units,
                reference.score.resource_cost_units
            );
        }
    }

    #[test]
    fn reference_policy_solves_both_seeded_layouts() {
        for seed in [42, 43] {
            let manifest = ScenarioManifest { seed, ..ScenarioManifest::default() };
            let run = run_reference_policy(&manifest, false).expect("reference policy");
            assert!(run.score.goal_completed);
            assert_eq!(run.score.invalid_actions, 0);
            assert_eq!(run.score.action_efficiency, 1.0);
            assert_eq!(run.score.agent_attempts, run.score.optimal_agent_steps);
            assert!(run.receipts.windows(2).all(|pair| {
                pair[0].after_state_digest == pair[1].before_state_digest
            }));
        }
    }

    #[test]
    fn reference_policy_reports_cost_units_for_both_layouts() {
        let depot = ScenarioManifest { seed: 42, ..ScenarioManifest::default() };
        let outpost = ScenarioManifest { seed: 43, ..ScenarioManifest::default() };
        let depot_run = run_reference_policy(&depot, false).expect("depot baseline");
        let outpost_run = run_reference_policy(&outpost, false).expect("outpost baseline");

        assert_eq!(depot_run.score.resource_cost_units, 10);
        assert_eq!(outpost_run.score.resource_cost_units, 12);
        assert!((depot_run.score.goal_completion_per_cost_unit - 0.1).abs() < 1e-12);
        assert!((outpost_run.score.goal_completion_per_cost_unit - (1.0 / 12.0)).abs() < 1e-12);
    }

    #[test]
    fn invalid_actions_pay_the_configured_surcharge() {
        let manifest = ScenarioManifest::default();
        let commands = vec![RecordedCommand::Agent(Action::Deliver)];
        let run = evaluate_trace(&manifest, &commands).expect("invalid attempt trace");
        assert_eq!(run.score.invalid_actions, 1);
        assert_eq!(
            run.score.resource_cost_units,
            manifest.cost_model.interaction as u64
                + manifest.cost_model.invalid_action_surcharge as u64
        );
        assert_eq!(run.score.goal_completion_per_cost_unit, 0.0);
    }

    #[test]
    fn manifest_cost_model_is_digest_bound_and_changes_scores() {
        let default_manifest = ScenarioManifest::default();
        let expensive_inspection = ScenarioManifest {
            cost_model: CostModel { inspect: 5, ..CostModel::default() },
            ..ScenarioManifest::default()
        };
        assert_ne!(
            default_manifest.digest().expect("default digest"),
            expensive_inspection.digest().expect("modified digest")
        );
        let default_run =
            run_reference_policy(&default_manifest, false).expect("default costs");
        let expensive_run =
            run_reference_policy(&expensive_inspection, false).expect("expensive inspect");
        assert_eq!(default_run.score.resource_cost_units, 10);
        assert_eq!(expensive_run.score.resource_cost_units, 16);
    }

    #[test]
    fn reference_policy_recovers_after_an_evaluator_intervention() {
        let manifest = ScenarioManifest::default();
        let run = run_reference_policy(&manifest, true).expect("recovery policy");
        assert!(run.score.goal_completed);
        assert_eq!(run.score.evaluator_perturbations, 1);
        assert_eq!(run.score.successful_recoveries, 1);
        assert_eq!(run.score.invalid_actions, 0);
        assert!(run.score.action_efficiency < 1.0);
    }

    #[test]
    fn random_baseline_is_deterministic_and_charges_invalid_actions() {
        let manifest = ScenarioManifest::default();
        let a = run_random_policy(&manifest).expect("random baseline A");
        let b = run_random_policy(&manifest).expect("random baseline B");
        assert_eq!(a, b);
        assert!(a.score.transitions_used <= manifest.max_steps);
        assert_eq!(a.score.agent_attempts, a.score.transitions_used);
        assert_eq!(a.score.invalid_actions, a.receipts.iter().filter(|receipt| {
            matches!(&receipt.event.kind, EventKind::ActionRejected { .. })
        }).count() as u32);
    }

    #[test]
    fn replay_charges_and_verifies_rejected_actions() {
        let manifest = ScenarioManifest::default();
        let commands = vec![
            RecordedCommand::Agent(Action::Deliver),
            RecordedCommand::Agent(Action::Inspect),
        ];
        let (_state, receipts) = replay(&manifest, &commands).expect("replay");
        let verified = verify_trace(&manifest, &commands, &receipts).expect("verify");
        assert_eq!(verified.score.invalid_actions, 1);
        assert_eq!(verified.score.transitions_used, 2);

        let mut corrupt = receipts.clone();
        corrupt[0].step_index += 100;
        assert_eq!(
            verify_trace(&manifest, &commands, &corrupt),
            Err(TransitionError::ReplayMismatch)
        );
    }

    #[test]
    fn jsonl_export_emits_one_line_per_receipt() {
        let run = run_reference_policy(&ScenarioManifest::default(), false).expect("reference");
        let jsonl = receipts_to_jsonl(&run.receipts).expect("serialize");
        assert_eq!(jsonl.lines().count(), run.receipts.len());
        assert!(jsonl.lines().all(|line| serde_json::from_str::<StepReceipt>(line).is_ok()));
    }

    #[test]
    fn replay_reproduces_every_receipt_and_final_state() {
        let manifest = ScenarioManifest::default();
        let commands = vec![
            RecordedCommand::Agent(Action::Inspect),
            RecordedCommand::Agent(Action::Collect(Item::ValveKey)),
            RecordedCommand::Agent(Action::Move(Site::Crossing)),
            RecordedCommand::Agent(Action::Inspect),
            RecordedCommand::Agent(Action::UseValveKey),
            RecordedCommand::Evaluator(Perturbation::FailCrossingGate),
            RecordedCommand::Agent(Action::Inspect),
            RecordedCommand::Agent(Action::UseValveKey),
        ];

        let (state_a, receipts_a) = replay(&manifest, &commands).expect("first replay");
        let (state_b, receipts_b) = replay(&manifest, &commands).expect("second replay");
        assert_eq!(state_a, state_b);
        assert_eq!(receipts_a, receipts_b);
        assert_eq!(state_a.event_log.len(), commands.len());
    }

    #[test]
    fn changed_manifest_cannot_reuse_an_existing_state() {
        let original = ScenarioManifest::default();
        let state = WorldState::initial(&original).expect("initial state");
        let mut changed = original.clone();
        changed.max_steps += 1;

        assert_eq!(
            step(&changed, &state, Action::Inspect),
            Err(TransitionError::ManifestMismatch)
        );
    }

    #[test]
    fn identical_action_traces_produce_identical_receipts() {
        let manifest = ScenarioManifest::default();
        let initial = WorldState::initial(&manifest).expect("initial state");
        let (a, receipt_a) = step(&manifest, &initial, Action::Inspect).expect("inspect a");
        let (b, receipt_b) = step(&manifest, &initial, Action::Inspect).expect("inspect b");
        assert_eq!(a, b);
        assert_eq!(receipt_a, receipt_b);
        assert_ne!(receipt_a.before_state_digest, receipt_a.after_state_digest);
    }
}
