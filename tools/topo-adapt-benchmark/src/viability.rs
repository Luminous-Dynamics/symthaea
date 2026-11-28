#![forbid(unsafe_code)]
use std::collections::{BTreeMap, BTreeSet};
pub const MAX_STATES: usize = 64;
pub const MAX_ACTIONS: usize = 64;
pub const MAX_HORIZON_STEPS: usize = 32;
pub const MAX_BUDGET_UNITS: u64 = 128;
pub const MAX_TOTAL_OUTCOMES: usize = 16_384;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Action {
    pub id: String,
    pub cost_units: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub disturbance_id: String,
    pub next_state_id: String,
    pub resource_debit_units: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateAction {
    pub action_id: String,
    pub disturbance_ids: Vec<String>,
    pub outcomes: Vec<Outcome>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameState {
    pub id: String,
    pub observation_id: String,
    pub admissible: bool,
    pub terminal_admissible: bool,
    pub recovery_target: bool,
    pub maintenance_cost_units: u64,
    pub state_actions: Vec<StateAction>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteHorizonGame {
    pub horizon_steps: usize,
    pub budget_limit_units: u64,
    pub minimum_terminal_reserve_units: u64,
    pub actions: Vec<Action>,
    pub states: Vec<GameState>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViabilityLayer {
    pub time_step: usize,
    viable_by_state_and_budget: Vec<Vec<bool>>,
    witness_action_index_by_state_and_budget: Vec<Vec<Option<usize>>>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViabilityKernel {
    pub horizon_steps: usize,
    pub budget_limit_units: u64,
    pub state_ids: Vec<String>,
    pub action_ids: Vec<String>,
    pub layers: Vec<ViabilityLayer>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryLayer {
    pub steps_allowed: usize,
    viable_by_state_and_budget: Vec<Vec<bool>>,
    first_action_index_by_state_and_budget: Vec<Vec<Option<usize>>>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryReport {
    pub deadline_steps: usize,
    pub budget_limit_units: u64,
    pub state_ids: Vec<String>,
    pub action_ids: Vec<String>,
    pub layers: Vec<RecoveryLayer>,
}
impl ViabilityKernel {
    pub fn is_viable(
        &self,
        time_step: usize,
        state_id: &str,
        remaining_budget_units: u64,
    ) -> Option<bool> {
        let state_index = self.state_ids.binary_search_by(|id| id.as_str().cmp(state_id)).ok()?;
        let budget_index = usize::try_from(remaining_budget_units).ok()?;
        self.layers
            .get(time_step)?
            .viable_by_state_and_budget
            .get(state_index)?
            .get(budget_index)
            .copied()
    }
    pub fn witness_action(
        &self,
        time_step: usize,
        state_id: &str,
        remaining_budget_units: u64,
    ) -> Option<&str> {
        let state_index = self.state_ids.binary_search_by(|id| id.as_str().cmp(state_id)).ok()?;
        let budget_index = usize::try_from(remaining_budget_units).ok()?;
        let action_index = self
            .layers
            .get(time_step)?
            .witness_action_index_by_state_and_budget
            .get(state_index)?
            .get(budget_index)?
            .as_ref()?;
        self.action_ids.get(*action_index).map(String::as_str)
    }
}
impl RecoveryReport {
    pub fn recoverable_by_deadline(
        &self,
        state_id: &str,
        remaining_budget_units: u64,
    ) -> Option<bool> {
        let state_index = self.state_ids.binary_search_by(|id| id.as_str().cmp(state_id)).ok()?;
        let budget_index = usize::try_from(remaining_budget_units).ok()?;
        self.layers
            .get(self.deadline_steps)?
            .viable_by_state_and_budget
            .get(state_index)?
            .get(budget_index)
            .copied()
    }
    pub fn minimum_recovery_steps(
        &self,
        state_id: &str,
        remaining_budget_units: u64,
    ) -> Result<Option<usize>, String> {
        let state_index = self
            .state_ids
            .binary_search_by(|id| id.as_str().cmp(state_id))
            .map_err(|_| format!("unknown state ID {state_id}"))?;
        if remaining_budget_units > self.budget_limit_units {
            return Err("remaining budget exceeds the report budget limit".to_string());
        }
        let budget_index = usize::try_from(remaining_budget_units)
            .map_err(|_| "budget cannot be represented on this platform".to_string())?;
        Ok(self.layers.iter().find_map(|layer| {
            layer
                .viable_by_state_and_budget
                .get(state_index)?
                .get(budget_index)
                .copied()
                .filter(|viable| *viable)
                .map(|_| layer.steps_allowed)
        }))
    }
    pub fn minimum_start_budget_for_recovery(
        &self,
        state_id: &str,
    ) -> Result<Option<u64>, String> {
        if self.state_ids.binary_search_by(|id| id.as_str().cmp(state_id)).is_err() {
            return Err(format!("unknown state ID {state_id}"));
        }
        for budget in 0..=self.budget_limit_units {
            if self.recoverable_by_deadline(state_id, budget) == Some(true) {
                return Ok(Some(budget));
            }
        }
        Ok(None)
    }
    pub fn first_action(
        &self,
        steps_allowed: usize,
        state_id: &str,
        remaining_budget_units: u64,
    ) -> Option<&str> {
        let state_index = self.state_ids.binary_search_by(|id| id.as_str().cmp(state_id)).ok()?;
        let budget_index = usize::try_from(remaining_budget_units).ok()?;
        let action_index = self
            .layers
            .get(steps_allowed)?
            .first_action_index_by_state_and_budget
            .get(state_index)?
            .get(budget_index)?
            .as_ref()?;
        self.action_ids.get(*action_index).map(String::as_str)
    }
}
#[derive(Clone, Debug)]
struct PreparedIndex {
    state_slot_by_original_index: Vec<usize>,
    state_index_by_id: BTreeMap<String, usize>,
    action_order: Vec<usize>,
    observation_groups: Vec<Vec<usize>>,
    state_ids: Vec<String>,
    action_ids: Vec<String>,
}
impl FiniteHorizonGame {
    pub fn validate(&self) -> Result<(), String> {
        if self.horizon_steps > MAX_HORIZON_STEPS {
            return Err(format!(
                "horizon {} exceeds exact-oracle limit {}",
                self.horizon_steps, MAX_HORIZON_STEPS
            ));
        }
        if self.budget_limit_units > MAX_BUDGET_UNITS {
            return Err(format!(
                "budget {} exceeds exact-oracle limit {}",
                self.budget_limit_units, MAX_BUDGET_UNITS
            ));
        }
        if self.minimum_terminal_reserve_units > self.budget_limit_units {
            return Err("minimum terminal reserve exceeds the budget limit".to_string());
        }
        if self.states.is_empty() || self.states.len() > MAX_STATES {
            return Err(format!(
                "state count must be between 1 and {MAX_STATES}"
            ));
        }
        if self.actions.is_empty() || self.actions.len() > MAX_ACTIONS {
            return Err(format!(
                "action count must be between 1 and {MAX_ACTIONS}"
            ));
        }
        let mut action_ids = BTreeSet::new();
        for action in &self.actions {
            if action.id.trim().is_empty() {
                return Err("action ID must not be empty".to_string());
            }
            if !action_ids.insert(action.id.as_str()) {
                return Err(format!("duplicate action ID {}", action.id));
            }
        }
        let mut state_ids = BTreeSet::new();
        for state in &self.states {
            if state.id.trim().is_empty() {
                return Err("state ID must not be empty".to_string());
            }
            if state.observation_id.trim().is_empty() {
                return Err(format!("{}: observation ID must not be empty", state.id));
            }
            if !state_ids.insert(state.id.as_str()) {
                return Err(format!("duplicate state ID {}", state.id));
            }
        }
        let mut observation_target: BTreeMap<&str, bool> = BTreeMap::new();
        let mut total_outcomes = 0usize;
        for state in &self.states {
            match observation_target.get(state.observation_id.as_str()) {
                Some(existing) if *existing != state.recovery_target => {
                    return Err(format!(
                        "observation {} mixes recovery-target and non-target states",
                        state.observation_id
                    ));
                }
                Some(_) => {}
                None => {
                    observation_target.insert(
                        state.observation_id.as_str(),
                        state.recovery_target,
                    );
                }
            }
            let mut seen_actions = BTreeSet::new();
            for state_action in &state.state_actions {
                if !action_ids.contains(state_action.action_id.as_str()) {
                    return Err(format!(
                        "{}: unknown action {}",
                        state.id, state_action.action_id
                    ));
                }
                if !seen_actions.insert(state_action.action_id.as_str()) {
                    return Err(format!(
                        "{}: duplicate transition definition for action {}",
                        state.id, state_action.action_id
                    ));
                }
                total_outcomes = total_outcomes
                    .checked_add(state_action.outcomes.len())
                    .ok_or_else(|| "outcome count overflow".to_string())?;
                if total_outcomes > MAX_TOTAL_OUTCOMES {
                    return Err(format!(
                        "outcome count exceeds exact-oracle limit {MAX_TOTAL_OUTCOMES}"
                    ));
                }
                if state_action.disturbance_ids.is_empty() || state_action.outcomes.is_empty() {
                    return Err(format!(
                        "{} / {}: disturbance support and outcomes must be non-empty",
                        state.id, state_action.action_id
                    ));
                }
                let mut declared_disturbances = BTreeSet::new();
                for disturbance_id in &state_action.disturbance_ids {
                    if disturbance_id.trim().is_empty() {
                        return Err(format!(
                            "{} / {}: disturbance ID must not be empty",
                            state.id, state_action.action_id
                        ));
                    }
                    if !declared_disturbances.insert(disturbance_id.as_str()) {
                        return Err(format!(
                            "{} / {}: duplicate declared disturbance {}",
                            state.id, state_action.action_id, disturbance_id
                        ));
                    }
                }
                let mut outcome_disturbances = BTreeSet::new();
                for outcome in &state_action.outcomes {
                    if !declared_disturbances.contains(outcome.disturbance_id.as_str()) {
                        return Err(format!(
                            "{} / {}: outcome {} is outside declared disturbance support",
                            state.id, state_action.action_id, outcome.disturbance_id
                        ));
                    }
                    if !outcome_disturbances.insert(outcome.disturbance_id.as_str()) {
                        return Err(format!(
                            "{} / {}: duplicate outcome for disturbance {}",
                            state.id, state_action.action_id, outcome.disturbance_id
                        ));
                    }
                    if !state_ids.contains(outcome.next_state_id.as_str()) {
                        return Err(format!(
                            "{} / {}: unknown successor state {}",
                            state.id, state_action.action_id, outcome.next_state_id
                        ));
                    }
                    let action = self
                        .actions
                        .iter()
                        .find(|action| action.id == state_action.action_id)
                        .ok_or_else(|| format!("unknown action {}", state_action.action_id))?;
                    state
                        .maintenance_cost_units
                        .checked_add(action.cost_units)
                        .and_then(|value| value.checked_add(outcome.resource_debit_units))
                        .ok_or_else(|| {
                            format!(
                                "{} / {}: resource debit overflows u64",
                                state.id, state_action.action_id
                            )
                        })?;
                }
                if outcome_disturbances != declared_disturbances {
                    return Err(format!(
                        "{} / {}: outcome set does not exactly match declared disturbance support",
                        state.id, state_action.action_id
                    ));
                }
            }
        }
        Ok(())
    }
    fn prepare_index(&self) -> Result<PreparedIndex, String> {
        self.validate()?;
        let mut state_order: Vec<usize> = (0..self.states.len()).collect();
        state_order.sort_by(|left, right| self.states[*left].id.cmp(&self.states[*right].id));
        let state_ids: Vec<String> = state_order
            .iter()
            .map(|index| self.states[*index].id.clone())
            .collect();
        let mut state_slot_by_original_index = vec![0; self.states.len()];
        let mut state_index_by_id = BTreeMap::new();
        for (slot, original_index) in state_order.iter().copied().enumerate() {
            state_slot_by_original_index[original_index] = slot;
            state_index_by_id.insert(self.states[original_index].id.clone(), original_index);
        }
        let mut action_order: Vec<usize> = (0..self.actions.len()).collect();
        action_order.sort_by(|left, right| self.actions[*left].id.cmp(&self.actions[*right].id));
        let action_ids: Vec<String> = action_order
            .iter()
            .map(|index| self.actions[*index].id.clone())
            .collect();
        let mut groups_by_observation: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, state) in self.states.iter().enumerate() {
            groups_by_observation
                .entry(state.observation_id.clone())
                .or_default()
                .push(index);
        }
        let mut observation_groups: Vec<Vec<usize>> = groups_by_observation.into_values().collect();
        for group in &mut observation_groups {
            group.sort_by(|left, right| self.states[*left].id.cmp(&self.states[*right].id));
        }
        Ok(PreparedIndex {
            state_slot_by_original_index,
            state_index_by_id,
            action_order,
            observation_groups,
            state_ids,
            action_ids,
        })
    }
    pub fn solve_robust_viability(&self) -> Result<ViabilityKernel, String> {
        let index = self.prepare_index()?;
        let state_count = self.states.len();
        let budget_count = usize::try_from(self.budget_limit_units)
            .map_err(|_| "budget cannot be represented on this platform".to_string())?
            + 1;
        let budget_limit = self.budget_limit_units as usize;
        let mut values = vec![
            vec![vec![false; budget_count]; state_count];
            self.horizon_steps + 1
        ];
        let mut witnesses = vec![
            vec![vec![None; budget_count]; state_count];
            self.horizon_steps + 1
        ];
        for group in &index.observation_groups {
            for budget in 0..=budget_limit {
                let group_is_terminal = budget as u64 >= self.minimum_terminal_reserve_units
                    && group.iter().all(|state_index| {
                        let state = &self.states[*state_index];
                        state.admissible && state.terminal_admissible
                    });
                if group_is_terminal {
                    for state_index in group {
                        let slot = index.state_slot_by_original_index[*state_index];
                        values[self.horizon_steps][slot][budget] = true;
                    }
                }
            }
        }
        for time_step in (0..self.horizon_steps).rev() {
            for group in &index.observation_groups {
                for budget in 0..=budget_limit {
                    if !group.iter().all(|state_index| self.states[*state_index].admissible) {
                        continue;
                    }
                    for (action_slot, action_index) in index.action_order.iter().copied().enumerate() {
                        if group_action_is_robust(
                            self,
                            group,
                            action_index,
                            budget as u64,
                            &values[time_step + 1],
                            &index,
                        ) {
                            for state_index in group {
                                let slot = index.state_slot_by_original_index[*state_index];
                                values[time_step][slot][budget] = true;
                                witnesses[time_step][slot][budget] = Some(action_slot);
                            }
                            break;
                        }
                    }
                }
            }
        }
        let layers = (0..=self.horizon_steps)
            .map(|time_step| ViabilityLayer {
                time_step,
                viable_by_state_and_budget: values[time_step].clone(),
                witness_action_index_by_state_and_budget: witnesses[time_step].clone(),
            })
            .collect();
        Ok(ViabilityKernel {
            horizon_steps: self.horizon_steps,
            budget_limit_units: self.budget_limit_units,
            state_ids: index.state_ids,
            action_ids: index.action_ids,
            layers,
        })
    }
    pub fn solve_recoverability(&self, deadline_steps: usize) -> Result<RecoveryReport, String> {
        let index = self.prepare_index()?;
        if deadline_steps > MAX_HORIZON_STEPS {
            return Err(format!(
                "recovery deadline {} exceeds exact-oracle limit {}",
                deadline_steps, MAX_HORIZON_STEPS
            ));
        }
        let state_count = self.states.len();
        let budget_count = usize::try_from(self.budget_limit_units)
            .map_err(|_| "budget cannot be represented on this platform".to_string())?
            + 1;
        let budget_limit = self.budget_limit_units as usize;
        let mut values = vec![vec![vec![false; budget_count]; state_count]; deadline_steps + 1];
        let mut witnesses =
            vec![vec![vec![None; budget_count]; state_count]; deadline_steps + 1];
        for group in &index.observation_groups {
            for budget in 0..=budget_limit {
                let target_reached = budget as u64 >= self.minimum_terminal_reserve_units
                    && group.iter().all(|state_index| {
                        let state = &self.states[*state_index];
                        state.recovery_target && state.admissible && state.terminal_admissible
                    });
                if target_reached {
                    for state_index in group {
                        let slot = index.state_slot_by_original_index[*state_index];
                        values[0][slot][budget] = true;
                    }
                }
            }
        }
        for steps_allowed in 1..=deadline_steps {
            for group in &index.observation_groups {
                let target_group = group
                    .iter()
                    .all(|state_index| self.states[*state_index].recovery_target);
                for budget in 0..=budget_limit {
                    if target_group {
                        let target_is_valid = budget as u64 >= self.minimum_terminal_reserve_units
                            && group.iter().all(|state_index| {
                                let state = &self.states[*state_index];
                                state.admissible && state.terminal_admissible
                            });
                        if target_is_valid {
                            for state_index in group {
                                let slot = index.state_slot_by_original_index[*state_index];
                                values[steps_allowed][slot][budget] = true;
                            }
                        }
                        continue;
                    }
                    if !group.iter().all(|state_index| self.states[*state_index].admissible) {
                        continue;
                    }
                    for (action_slot, action_index) in index.action_order.iter().copied().enumerate() {
                        if group_action_is_robust(
                            self,
                            group,
                            action_index,
                            budget as u64,
                            &values[steps_allowed - 1],
                            &index,
                        ) {
                            for state_index in group {
                                let slot = index.state_slot_by_original_index[*state_index];
                                values[steps_allowed][slot][budget] = true;
                                witnesses[steps_allowed][slot][budget] = Some(action_slot);
                            }
                            break;
                        }
                    }
                }
            }
        }
        let layers = (0..=deadline_steps)
            .map(|steps_allowed| RecoveryLayer {
                steps_allowed,
                viable_by_state_and_budget: values[steps_allowed].clone(),
                first_action_index_by_state_and_budget: witnesses[steps_allowed].clone(),
            })
            .collect();
        Ok(RecoveryReport {
            deadline_steps,
            budget_limit_units: self.budget_limit_units,
            state_ids: index.state_ids,
            action_ids: index.action_ids,
            layers,
        })
    }
}
fn group_action_is_robust(
    game: &FiniteHorizonGame,
    group: &[usize],
    action_index: usize,
    remaining_budget_units: u64,
    next_layer: &[Vec<bool>],
    index: &PreparedIndex,
) -> bool {
    let action = &game.actions[action_index];
    for state_index in group {
        let state = &game.states[*state_index];
        let Some(state_action) = state
            .state_actions
            .iter()
            .find(|state_action| state_action.action_id == action.id)
        else {
            return false;
        };
        let Some(fixed_spend) = state
            .maintenance_cost_units
            .checked_add(action.cost_units)
        else {
            return false;
        };
        if fixed_spend > remaining_budget_units {
            return false;
        }
        for outcome in &state_action.outcomes {
            let Some(total_spend) = fixed_spend.checked_add(outcome.resource_debit_units) else {
                return false;
            };
            if total_spend > remaining_budget_units {
                return false;
            }
            let Some(next_state_index) = index.state_index_by_id.get(&outcome.next_state_id) else {
                return false;
            };
            let next_state_slot = index.state_slot_by_original_index[*next_state_index];
            let next_budget = (remaining_budget_units - total_spend) as usize;
            if !next_layer
                .get(next_state_slot)
                .and_then(|budgets| budgets.get(next_budget))
                .copied()
                .unwrap_or(false)
            {
                return false;
            }
        }
    }
    true
}
#[cfg(test)]
mod tests {
    use super::*;
    fn action(id: &str, cost_units: u64) -> Action {
        Action {
            id: id.to_string(),
            cost_units,
        }
    }
    fn state(
        id: &str,
        observation_id: &str,
        admissible: bool,
        terminal_admissible: bool,
        recovery_target: bool,
        maintenance_cost_units: u64,
        state_actions: Vec<StateAction>,
    ) -> GameState {
        GameState {
            id: id.to_string(),
            observation_id: observation_id.to_string(),
            admissible,
            terminal_admissible,
            recovery_target,
            maintenance_cost_units,
            state_actions,
        }
    }
    fn transition(action_id: &str, outcomes: Vec<Outcome>) -> StateAction {
        StateAction {
            action_id: action_id.to_string(),
            disturbance_ids: outcomes
                .iter()
                .map(|outcome| outcome.disturbance_id.clone())
                .collect(),
            outcomes,
        }
    }
    fn outcome(disturbance_id: &str, next_state_id: &str, debit: u64) -> Outcome {
        Outcome {
            disturbance_id: disturbance_id.to_string(),
            next_state_id: next_state_id.to_string(),
            resource_debit_units: debit,
        }
    }
    fn game(
        horizon_steps: usize,
        budget_limit_units: u64,
        minimum_terminal_reserve_units: u64,
        actions: Vec<Action>,
        states: Vec<GameState>,
    ) -> FiniteHorizonGame {
        FiniteHorizonGame {
            horizon_steps,
            budget_limit_units,
            minimum_terminal_reserve_units,
            actions,
            states,
        }
    }
    #[test]
    fn robust_kernel_rejects_an_action_with_one_unsafe_disturbance() {
        let input = game(
            1,
            0,
            0,
            vec![action("continue", 0)],
            vec![
                state(
                    "root",
                    "root-observation",
                    true,
                    false,
                    false,
                    0,
                    vec![transition(
                        "continue",
                        vec![outcome("quiet", "safe", 0), outcome("surge", "unsafe", 0)],
                    )],
                ),
                state("safe", "safe-observation", true, true, false, 0, vec![]),
                state("unsafe", "unsafe-observation", true, false, false, 0, vec![]),
            ],
        );
        let kernel = input.solve_robust_viability().expect("valid game");
        assert_eq!(kernel.is_viable(0, "root", 0), Some(false));
    }
    #[test]
    fn robust_kernel_accepts_only_when_every_disturbance_remains_viable() {
        let input = game(
            1,
            0,
            0,
            vec![action("continue", 0)],
            vec![
                state(
                    "root",
                    "root-observation",
                    true,
                    false,
                    false,
                    0,
                    vec![transition("continue", vec![outcome("quiet", "safe", 0)])],
                ),
                state("safe", "safe-observation", true, true, false, 0, vec![]),
            ],
        );
        let kernel = input.solve_robust_viability().expect("valid game");
        assert_eq!(kernel.is_viable(0, "root", 0), Some(true));
        assert_eq!(kernel.witness_action(0, "root", 0), Some("continue"));
    }
    #[test]
    fn indistinguishable_states_must_share_one_robust_action() {
        let common_action = || {
            vec![transition("continue", vec![outcome("only", "terminal", 0)])]
        };
        let input = game(
            1,
            0,
            0,
            vec![action("continue", 0)],
            vec![
                state("hidden-a", "same-observation", true, false, false, 0, common_action()),
                state("hidden-b", "same-observation", true, false, false, 0, common_action()),
                state("terminal", "terminal-observation", true, true, false, 0, vec![]),
            ],
        );
        let kernel = input.solve_robust_viability().expect("valid game");
        assert_eq!(kernel.is_viable(0, "hidden-a", 0), Some(true));
        assert_eq!(kernel.is_viable(0, "hidden-b", 0), Some(true));
        assert_eq!(kernel.witness_action(0, "hidden-a", 0), Some("continue"));
        assert_eq!(kernel.witness_action(0, "hidden-b", 0), Some("continue"));
    }
    #[test]
    fn hidden_state_that_requires_a_different_action_is_not_certified() {
        let input = game(
            1,
            0,
            0,
            vec![action("left", 0), action("right", 0)],
            vec![
                state(
                    "hidden-a",
                    "same-observation",
                    true,
                    false,
                    false,
                    0,
                    vec![transition("left", vec![outcome("only", "safe-a", 0)])],
                ),
                state(
                    "hidden-b",
                    "same-observation",
                    true,
                    false,
                    false,
                    0,
                    vec![transition("right", vec![outcome("only", "safe-b", 0)])],
                ),
                state("safe-a", "safe-a-observation", true, true, false, 0, vec![]),
                state("safe-b", "safe-b-observation", true, true, false, 0, vec![]),
            ],
        );
        let kernel = input.solve_robust_viability().expect("valid game");
        assert_eq!(kernel.is_viable(0, "hidden-a", 0), Some(false));
        assert_eq!(kernel.is_viable(0, "hidden-b", 0), Some(false));
    }
    #[test]
    fn maintenance_action_and_disturbance_debits_are_cumulative() {
        let input = game(
            1,
            3,
            0,
            vec![action("repair", 1)],
            vec![
                state(
                    "root",
                    "root-observation",
                    true,
                    false,
                    false,
                    1,
                    vec![transition("repair", vec![outcome("repair-cost", "target", 1)])],
                ),
                state("target", "target-observation", true, true, false, 0, vec![]),
            ],
        );
        let kernel = input.solve_robust_viability().expect("valid game");
        assert_eq!(kernel.is_viable(0, "root", 2), Some(false));
        assert_eq!(kernel.is_viable(0, "root", 3), Some(true));
    }
    #[test]
    fn terminal_reserve_is_not_spendable() {
        let input = game(
            1,
            2,
            1,
            vec![action("repair", 1)],
            vec![
                state(
                    "root",
                    "root-observation",
                    true,
                    false,
                    false,
                    0,
                    vec![transition("repair", vec![outcome("repair", "target", 0)])],
                ),
                state("target", "target-observation", true, true, false, 0, vec![]),
            ],
        );
        let kernel = input.solve_robust_viability().expect("valid game");
        assert_eq!(kernel.is_viable(0, "root", 1), Some(false));
        assert_eq!(kernel.is_viable(0, "root", 2), Some(true));
    }
    #[test]
    fn recoverability_reports_minimum_steps_and_first_action() {
        let input = game(
            0,
            3,
            0,
            vec![action("repair", 1)],
            vec![
                state(
                    "damaged",
                    "damaged-observation",
                    true,
                    false,
                    false,
                    1,
                    vec![transition(
                        "repair",
                        vec![outcome("known-failure", "restored", 1)],
                    )],
                ),
                state("restored", "restored-observation", true, true, true, 0, vec![]),
            ],
        );
        let report = input.solve_recoverability(2).expect("valid game");
        assert_eq!(report.minimum_recovery_steps("damaged", 2), Ok(None));
        assert_eq!(report.minimum_recovery_steps("damaged", 3), Ok(Some(1)));
        assert_eq!(report.recoverable_by_deadline("damaged", 3), Some(true));
        assert_eq!(report.first_action(1, "damaged", 3), Some("repair"));
        assert_eq!(report.minimum_start_budget_for_recovery("damaged"), Ok(Some(3)));
        assert_eq!(report.minimum_recovery_steps("restored", 3), Ok(Some(0)));
    }
    #[test]
    fn recovery_does_not_assign_a_finite_time_to_an_unreachable_target() {
        let input = game(
            0,
            2,
            0,
            vec![action("wait", 0)],
            vec![state(
                "isolated",
                "isolated-observation",
                true,
                false,
                false,
                0,
                vec![transition("wait", vec![outcome("only", "isolated", 0)])],
            )],
        );
        let report = input.solve_recoverability(3).expect("valid game");
        assert_eq!(report.minimum_recovery_steps("isolated", 2), Ok(None));
        assert_eq!(report.recoverable_by_deadline("isolated", 2), Some(false));
    }
    #[test]
    fn action_and_state_input_order_do_not_change_the_kernel() {
        let first = game(
            1,
            0,
            0,
            vec![action("z-action", 0), action("a-action", 0)],
            vec![
                state(
                    "root",
                    "root-observation",
                    true,
                    false,
                    false,
                    0,
                    vec![
                        transition("z-action", vec![outcome("only", "safe", 0)]),
                        transition("a-action", vec![outcome("only", "safe", 0)]),
                    ],
                ),
                state("safe", "safe-observation", true, true, false, 0, vec![]),
            ],
        );
        let mut second = first.clone();
        second.actions.reverse();
        second.states.reverse();
        second.states[1].state_actions.reverse();
        let left = first.solve_robust_viability().expect("valid first game");
        let right = second.solve_robust_viability().expect("valid second game");
        assert_eq!(left, right);
        assert_eq!(left.witness_action(0, "root", 0), Some("a-action"));
    }
    #[test]
    fn declared_disturbance_support_must_match_outcomes_exactly() {
        let mut input = game(
            1,
            0,
            0,
            vec![action("continue", 0)],
            vec![
                state(
                    "root",
                    "root-observation",
                    true,
                    false,
                    false,
                    0,
                    vec![transition("continue", vec![outcome("quiet", "terminal", 0)])],
                ),
                state("terminal", "terminal-observation", true, true, false, 0, vec![]),
            ],
        );
        input.states[0].state_actions[0]
            .disturbance_ids
            .push("unmodeled-storm".to_string());
        assert!(input.validate().is_err());
    }
    #[test]
    fn target_membership_must_be_observable() {
        let input = game(
            0,
            0,
            0,
            vec![action("unused", 0)],
            vec![
                state("target", "same-observation", true, true, true, 0, vec![]),
                state("not-target", "same-observation", true, true, false, 0, vec![]),
            ],
        );
        assert!(input.validate().is_err());
    }
    #[test]
    fn unknown_successor_and_duplicate_disturbances_fail_closed() {
        let mut input = game(
            1,
            0,
            0,
            vec![action("continue", 0)],
            vec![state(
                "root",
                "root-observation",
                true,
                false,
                false,
                0,
                vec![transition("continue", vec![outcome("only", "missing", 0)])],
            )],
        );
        assert!(input.validate().is_err());
        input.states[0].state_actions[0].outcomes[0].next_state_id = "root".to_string();
        input.states[0].state_actions[0]
            .disturbance_ids
            .push("only".to_string());
        input.states[0].state_actions[0]
            .outcomes
            .push(outcome("only", "root", 0));
        assert!(input.validate().is_err());
    }
    #[test]
    fn resource_cost_overflow_is_rejected_before_solving() {
        let input = game(
            1,
            2,
            0,
            vec![action("overflow", u64::MAX)],
            vec![state(
                "root",
                "root-observation",
                true,
                false,
                false,
                1,
                vec![transition("overflow", vec![outcome("only", "root", 1)])],
            )],
        );
        assert!(input.validate().is_err());
    }
    fn brute_force_full_information(
        game: &FiniteHorizonGame,
        state_id: &str,
        remaining_budget: u64,
        steps_left: usize,
    ) -> bool {
        let Some(state) = game.states.iter().find(|state| state.id == state_id) else {
            return false;
        };
        if !state.admissible {
            return false;
        }
        if steps_left == 0 {
            return state.terminal_admissible
                && remaining_budget >= game.minimum_terminal_reserve_units;
        }
        game.actions.iter().any(|action| {
            let Some(state_action) = state
                .state_actions
                .iter()
                .find(|state_action| state_action.action_id == action.id)
            else {
                return false;
            };
            state_action.outcomes.iter().all(|outcome| {
                let Some(spend) = state
                    .maintenance_cost_units
                    .checked_add(action.cost_units)
                    .and_then(|value| value.checked_add(outcome.resource_debit_units))
                else {
                    return false;
                };
                spend <= remaining_budget
                    && brute_force_full_information(
                        game,
                        &outcome.next_state_id,
                        remaining_budget - spend,
                        steps_left - 1,
                    )
            })
        })
    }
    #[test]
    fn dynamic_program_matches_independent_recursive_minimax_on_small_games() {
        let input = game(
            2,
            3,
            1,
            vec![action("repair", 1), action("wait", 0)],
            vec![
                state(
                    "root",
                    "root-only",
                    true,
                    false,
                    false,
                    1,
                    vec![
                        transition(
                            "wait",
                            vec![outcome("quiet", "good-a", 0), outcome("storm", "bad", 0)],
                        ),
                        transition(
                            "repair",
                            vec![outcome("quiet", "good-a", 0), outcome("storm", "good-b", 0)],
                        ),
                    ],
                ),
                state(
                    "good-a",
                    "good-a-only",
                    true,
                    false,
                    false,
                    0,
                    vec![transition("wait", vec![outcome("quiet", "terminal-a", 0)])],
                ),
                state(
                    "good-b",
                    "good-b-only",
                    true,
                    false,
                    false,
                    0,
                    vec![transition("wait", vec![outcome("quiet", "terminal-b", 0)])],
                ),
                state("bad", "bad-only", true, false, false, 0, vec![]),
                state("terminal-a", "terminal-a-only", true, true, false, 0, vec![]),
                state("terminal-b", "terminal-b-only", true, true, false, 0, vec![]),
            ],
        );
        let kernel = input.solve_robust_viability().expect("valid game");
        for time_step in 0..=input.horizon_steps {
            let steps_left = input.horizon_steps - time_step;
            for state in &input.states {
                for budget in 0..=input.budget_limit_units {
                    assert_eq!(
                        kernel.is_viable(time_step, &state.id, budget),
                        Some(brute_force_full_information(
                            &input,
                            &state.id,
                            budget,
                            steps_left,
                        )),
                        "state={}, budget={}, time={}",
                        state.id,
                        budget,
                        time_step
                    );
                }
            }
        }
    }
    #[test]
    fn current_safety_is_required_even_when_a_future_path_exists() {
        let input = game(
            1,
            0,
            0,
            vec![action("continue", 0)],
            vec![state(
                "unsafe-now",
                "unsafe-observation",
                false,
                false,
                false,
                0,
                vec![transition("continue", vec![outcome("only", "safe", 0)])],
            ), state("safe", "safe-observation", true, true, false, 0, vec![])],
        );
        let kernel = input.solve_robust_viability().expect("valid game");
        assert_eq!(kernel.is_viable(0, "unsafe-now", 0), Some(false));
    }
}