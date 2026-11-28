//! Paired hidden-mechanics intervention fixture (GameBench F6).
//!
//! Two worlds have identical initial observations but different switch rules.
//! A paid intervention reveals the active rule through its visible consequence.
//! The hidden rule itself is never included in candidate observations.

use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

use crate::{Action, Position, TerminalStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwitchEffect {
    OpenNorthGate,
    OpenEastGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HiddenAction {
    Move(Action),
    UseSwitch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HiddenMechanicsState {
    pub position: Position,
    pub north_gate_open: bool,
    pub east_gate_open: bool,
    pub switch_used: bool,
    pub elapsed_actions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HiddenMechanicsObservation {
    pub position: Position,
    pub north_gate_open: bool,
    pub east_gate_open: bool,
    pub switch_available: bool,
    pub remaining_actions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HiddenMechanicsTransition {
    pub state: HiddenMechanicsState,
    pub moved: bool,
    pub rule_intervened: bool,
    pub status: TerminalStatus,
    pub observation: HiddenMechanicsObservation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HiddenMechanicsError {
    ZeroActionLimit,
    AlreadyTerminal(TerminalStatus),
}

impl Display for HiddenMechanicsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroActionLimit => write!(f, "hidden-mechanics action limit must be nonzero"),
            Self::AlreadyTerminal(status) => {
                write!(f, "cannot step hidden-mechanics world after {status:?}")
            }
        }
    }
}

impl Error for HiddenMechanicsError {}

#[derive(Clone, PartialEq, Eq)]
pub struct HiddenMechanicsGame {
    // Private evaluator-selected rule. No getter and no rule field in
    // HiddenMechanicsObservation: the only candidate-visible evidence is the
    // observed state change caused by an intervention.
    hidden_effect: SwitchEffect,
    max_actions: u32,
    state: HiddenMechanicsState,
    status: TerminalStatus,
}

impl Debug for HiddenMechanicsGame {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Deliberately omit hidden_effect so diagnostics cannot leak the
        // evaluator-selected rule to the candidate.
        f.debug_struct("HiddenMechanicsGame")
            .field("max_actions", &self.max_actions)
            .field("state", &self.state)
            .field("status", &self.status)
            .finish()
    }
}

impl HiddenMechanicsGame {
    pub fn new(
        hidden_effect: SwitchEffect,
        max_actions: u32,
    ) -> Result<Self, HiddenMechanicsError> {
        if max_actions == 0 {
            return Err(HiddenMechanicsError::ZeroActionLimit);
        }

        let state = HiddenMechanicsState {
            position: Position::new(1, 1),
            north_gate_open: false,
            east_gate_open: false,
            switch_used: false,
            elapsed_actions: 0,
        };

        Ok(Self {
            hidden_effect,
            max_actions,
            state,
            status: TerminalStatus::Running,
        })
    }

    pub const fn state(&self) -> HiddenMechanicsState {
        self.state
    }

    pub const fn status(&self) -> TerminalStatus {
        self.status
    }

    pub const fn observe(&self) -> HiddenMechanicsObservation {
        HiddenMechanicsObservation {
            position: self.state.position,
            north_gate_open: self.state.north_gate_open,
            east_gate_open: self.state.east_gate_open,
            switch_available: !self.state.switch_used,
            remaining_actions: self.max_actions - self.state.elapsed_actions,
        }
    }

    pub fn step(
        &mut self,
        action: HiddenAction,
    ) -> Result<HiddenMechanicsTransition, HiddenMechanicsError> {
        if self.status != TerminalStatus::Running {
            return Err(HiddenMechanicsError::AlreadyTerminal(self.status));
        }

        let mut moved = false;
        let mut rule_intervened = false;

        match action {
            HiddenAction::UseSwitch if !self.state.switch_used => {
                match self.hidden_effect {
                    SwitchEffect::OpenNorthGate => self.state.north_gate_open = true,
                    SwitchEffect::OpenEastGate => self.state.east_gate_open = true,
                }
                self.state.switch_used = true;
                rule_intervened = true;
            }
            HiddenAction::UseSwitch => {}
            HiddenAction::Move(direction) => {
                if let Some(destination) = adjacent(self.state.position, direction) {
                    let north_gate = Position::new(1, 0);
                    let east_gate = Position::new(2, 1);
                    let blocked_by_closed_gate =
                        (destination == north_gate && !self.state.north_gate_open)
                            || (destination == east_gate && !self.state.east_gate_open);
                    let in_bounds = destination.x < 3 && destination.y < 2;
                    if in_bounds && !blocked_by_closed_gate {
                        self.state.position = destination;
                        moved = true;
                    }
                }
            }
        }

        self.state.elapsed_actions += 1;
        self.status = if self.state.position == Position::new(2, 0) {
            TerminalStatus::Success
        } else if self.state.elapsed_actions >= self.max_actions {
            TerminalStatus::ActionLimit
        } else {
            TerminalStatus::Running
        };

        Ok(HiddenMechanicsTransition {
            state: self.state,
            moved,
            rule_intervened,
            status: self.status,
            observation: self.observe(),
        })
    }
}

fn adjacent(position: Position, action: Action) -> Option<Position> {
    match action {
        Action::North => Some(Position::new(position.x, position.y.checked_sub(1)?)),
        Action::East => Some(Position::new(position.x.checked_add(1)?, position.y)),
        Action::South => Some(Position::new(position.x, position.y.checked_add(1)?)),
        Action::West => Some(Position::new(position.x.checked_sub(1)?, position.y)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_worlds_have_identical_initial_observations() {
        let north = HiddenMechanicsGame::new(SwitchEffect::OpenNorthGate, 8).expect("world A");
        let east = HiddenMechanicsGame::new(SwitchEffect::OpenEastGate, 8).expect("world B");
        assert_eq!(north.observe(), east.observe());
        assert!(!north.observe().north_gate_open);
        assert!(!north.observe().east_gate_open);
        assert!(north.observe().switch_available);
    }

    #[test]
    fn paid_intervention_reveals_rule_through_gate_state() {
        let mut north = HiddenMechanicsGame::new(SwitchEffect::OpenNorthGate, 8).expect("world A");
        let mut east = HiddenMechanicsGame::new(SwitchEffect::OpenEastGate, 8).expect("world B");

        let north_result = north.step(HiddenAction::UseSwitch).expect("switch intervention");
        let east_result = east.step(HiddenAction::UseSwitch).expect("switch intervention");

        assert!(north_result.rule_intervened);
        assert!(east_result.rule_intervened);
        assert!(north_result.observation.north_gate_open);
        assert!(!north_result.observation.east_gate_open);
        assert!(!east_result.observation.north_gate_open);
        assert!(east_result.observation.east_gate_open);
    }

    #[test]
    fn no_probe_control_does_not_allow_entry_through_closed_gate() {
        let mut game = HiddenMechanicsGame::new(SwitchEffect::OpenNorthGate, 4).expect("world");
        let result = game
            .step(HiddenAction::Move(Action::North))
            .expect("blocked move is a valid attempt");
        assert!(!result.moved);
        assert_eq!(result.state.position, Position::new(1, 1));
        assert_eq!(result.state.elapsed_actions, 1);
    }

    #[test]
    fn correct_intervention_then_route_completes_goal() {
        let mut north = HiddenMechanicsGame::new(SwitchEffect::OpenNorthGate, 8).expect("world");
        north.step(HiddenAction::UseSwitch).expect("identify north rule");
        north.step(HiddenAction::Move(Action::North)).expect("enter north gate");
        let complete = north
            .step(HiddenAction::Move(Action::East))
            .expect("enter goal");
        assert_eq!(complete.status, TerminalStatus::Success);

        let mut east = HiddenMechanicsGame::new(SwitchEffect::OpenEastGate, 8).expect("world");
        east.step(HiddenAction::UseSwitch).expect("identify east rule");
        east.step(HiddenAction::Move(Action::East)).expect("enter east gate");
        let complete = east
            .step(HiddenAction::Move(Action::North))
            .expect("enter goal");
        assert_eq!(complete.status, TerminalStatus::Success);
    }

    #[test]
    fn switch_is_single_use_and_action_budget_is_enforced() {
        let mut game = HiddenMechanicsGame::new(SwitchEffect::OpenEastGate, 2).expect("world");
        game.step(HiddenAction::UseSwitch).expect("first use");
        let second = game.step(HiddenAction::UseSwitch).expect("second use consumes time");
        assert!(!second.rule_intervened);
        assert_eq!(second.status, TerminalStatus::ActionLimit);
        assert_eq!(game.step(HiddenAction::UseSwitch), Err(HiddenMechanicsError::AlreadyTerminal(TerminalStatus::ActionLimit)));
    }

    #[test]
    fn debug_representation_does_not_leak_hidden_rule() {
        let game = HiddenMechanicsGame::new(SwitchEffect::OpenNorthGate, 8).expect("world");
        let debug = format!("{game:?}");
        assert!(!debug.contains("OpenNorthGate"));
        assert!(!debug.contains("OpenEastGate"));
    }

    #[test]
    fn rejects_zero_action_budget() {
        assert_eq!(
            HiddenMechanicsGame::new(SwitchEffect::OpenNorthGate, 0),
            Err(HiddenMechanicsError::ZeroActionLimit)
        );
    }
}
