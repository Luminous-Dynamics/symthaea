// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic environment kernel for PsychBench Cognitive Ecology (PCE).
//!
//! This module is an environment primitive, not yet a validated psychological
//! benchmark or a `PsychBenchmark` implementation. It deliberately separates
//! the full evaluator state from the partial observation returned to an agent.
//! Every successful transition produces a deterministic receipt that can be
//! replayed and audited independently of the agent implementation.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Versioned scenario and execution limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioManifest {
    pub schema_version: u16,
    pub scenario_id: String,
    pub generator_version: String,
    /// Reserved for the future scenario generator; the current fixed fixture ignores it.
    pub seed: u64,
    pub max_steps: u32,
    pub observation_budget: u32,
}

impl Default for ScenarioManifest {
    fn default() -> Self {
        Self {
            schema_version: 1,
            scenario_id: "pce.hidden-resource-delivery.v0".to_string(),
            generator_version: "pce-kernel-0.1.0".to_string(),
            seed: 42,
            max_steps: 64,
            observation_budget: 12,
        }
    }
}

impl ScenarioManifest {
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

/// Resources that can be discovered and collected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Item {
    WaterFilter,
    ValveKey,
}

/// The evaluator's gate state. Agents see it only after inspecting the crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateState {
    Closed,
    Open,
    Failed,
}

/// Facts that become available only through the observation action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Clue {
    DeliveryGoal,
    FilterStoredAtDepot,
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
        Ok(Self {
            tick: 0,
            location: Site::Depot,
            inventory: BTreeSet::new(),
            items_at: [
                (Site::Depot, Item::WaterFilter),
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

/// One successful transition's evidence receipt.
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
                    next.known_clues.insert(Clue::FilterStoredAtDepot);
                    next.known_clues.insert(Clue::ValveKeyStoredAtDepot);
                }
                Site::Crossing => {
                    next.known_clues.insert(Clue::ValveKeyOpensCrossingGate);
                    if next.gate_state == GateState::Failed {
                        next.known_clues.insert(Clue::GateRequiresRecoveryAfterFailure);
                    }
                }
                Site::Outpost => {
                    next.known_clues.insert(Clue::OutpostIsDeliveryDestination);
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
            RecordedCommand::Agent(action) => step(manifest, &state, action.clone())?,
            RecordedCommand::Evaluator(perturbation) => {
                perturb(manifest, &state, *perturbation)?
            }
        };
        state = next;
        receipts.push(receipt);
    }

    Ok((state, receipts))
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
