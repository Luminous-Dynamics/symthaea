// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical public schema for EUREKA-002 V2.
//!
//! This module owns only target-visible benchmark semantics. It contains no
//! partition, seed, schedule ordinal, evaluator oracle state, learned-model
//! state, or outcome information. Construct generation, target adaptation, and
//! future runners must all consume these definitions rather than re-declaring
//! their own family/context/action/count grammars.

use super::hidden_world::PublicAction;

pub(super) const V2_PUBLIC_SCHEMA_REVISION: &str = "EUREKA.002.V2.PUBLIC_SCHEMA.v1";
pub(super) const V2_COUNT_DENOMINATOR: i32 = 31;
pub(super) const V2_CONTEXT_DENOMINATOR: i32 = 7;
pub(super) const V2_OBSERVATION_DIM: usize = 4;
pub(super) const V2_REQUIRED_ACTIONS: usize = 4;
pub(super) const V2_PUBLIC_MODES_PER_FAMILY: u8 = 4;
pub(super) const V2_COUNT_CHANNELS: usize = 3;

pub(super) const V2_CANONICAL_ACTIONS: [PublicAction; V2_REQUIRED_ACTIONS] = [
    PublicAction::NoOp,
    PublicAction::Pulse { slot: 0 },
    PublicAction::Pulse { slot: 1 },
    PublicAction::Pulse { slot: 2 },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum V2PublicFamily {
    PublicFlowV2,
    PublicRelayV2,
}

impl V2PublicFamily {
    pub(super) const ALL: [Self; 2] = [Self::PublicFlowV2, Self::PublicRelayV2];

    pub(super) const fn tag(self) -> u8 {
        match self {
            Self::PublicFlowV2 => 1,
            Self::PublicRelayV2 => 2,
        }
    }

    pub(super) fn context(self, mode: u8) -> Result<i32, V2PublicSchemaError> {
        if mode >= V2_PUBLIC_MODES_PER_FAMILY {
            return Err(V2PublicSchemaError::ModeOutOfRange { mode });
        }
        Ok(match self {
            Self::PublicFlowV2 => i32::from(mode),
            Self::PublicRelayV2 => 4 + i32::from(mode),
        })
    }

    pub(super) const fn context_bounds(self) -> (i32, i32) {
        match self {
            Self::PublicFlowV2 => (0, 3),
            Self::PublicRelayV2 => (4, 7),
        }
    }

    pub(super) const fn contains_context(self, context: i32) -> bool {
        let (min, max) = self.context_bounds();
        context >= min && context <= max
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct V2PublicState {
    fields: [i32; V2_OBSERVATION_DIM],
}

impl V2PublicState {
    pub(super) fn new(
        fields: [i32; V2_OBSERVATION_DIM],
    ) -> Result<Self, V2PublicSchemaError> {
        for (index, value) in fields[..V2_COUNT_CHANNELS].iter().copied().enumerate() {
            if !(0..=V2_COUNT_DENOMINATOR).contains(&value) {
                return Err(V2PublicSchemaError::CountOutOfRange { index, value });
            }
        }
        let context = fields[3];
        if !(0..=V2_CONTEXT_DENOMINATOR).contains(&context) {
            return Err(V2PublicSchemaError::ContextOutOfRange { value: context });
        }
        Ok(Self { fields })
    }

    pub(super) const fn fields(self) -> [i32; V2_OBSERVATION_DIM] {
        self.fields
    }

    pub(super) const fn context(self) -> i32 {
        self.fields[3]
    }

    pub(super) const fn belongs_to(self, family: V2PublicFamily) -> bool {
        family.contains_context(self.context())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2PublicSchemaError {
    CountOutOfRange { index: usize, value: i32 },
    ContextOutOfRange { value: i32 },
    ModeOutOfRange { mode: u8 },
    UnsupportedAction,
    ActionIndexOutOfRange { index: usize },
}

pub(super) const fn action_index(action: PublicAction) -> Result<usize, V2PublicSchemaError> {
    match action {
        PublicAction::NoOp => Ok(0),
        PublicAction::Pulse { slot: 0 } => Ok(1),
        PublicAction::Pulse { slot: 1 } => Ok(2),
        PublicAction::Pulse { slot: 2 } => Ok(3),
        _ => Err(V2PublicSchemaError::UnsupportedAction),
    }
}

pub(super) const fn action_from_index(
    index: usize,
) -> Result<PublicAction, V2PublicSchemaError> {
    match index {
        0 => Ok(PublicAction::NoOp),
        1 => Ok(PublicAction::Pulse { slot: 0 }),
        2 => Ok(PublicAction::Pulse { slot: 1 }),
        3 => Ok(PublicAction::Pulse { slot: 2 }),
        _ => Err(V2PublicSchemaError::ActionIndexOutOfRange { index }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_context_bands_are_disjoint_complete_and_named_accurately() {
        for mode in 0..V2_PUBLIC_MODES_PER_FAMILY {
            let flow = V2PublicFamily::PublicFlowV2.context(mode).unwrap();
            let relay = V2PublicFamily::PublicRelayV2.context(mode).unwrap();
            assert_eq!(flow, i32::from(mode));
            assert_eq!(relay, 4 + i32::from(mode));
            assert!(V2PublicFamily::PublicFlowV2.contains_context(flow));
            assert!(!V2PublicFamily::PublicRelayV2.contains_context(flow));
            assert!(V2PublicFamily::PublicRelayV2.contains_context(relay));
            assert!(!V2PublicFamily::PublicFlowV2.contains_context(relay));
        }
        assert_eq!(
            V2PublicFamily::PublicFlowV2.context(4),
            Err(V2PublicSchemaError::ModeOutOfRange { mode: 4 })
        );
    }

    #[test]
    fn action_vocabulary_is_exact_bijection() {
        for (index, action) in V2_CANONICAL_ACTIONS.into_iter().enumerate() {
            assert_eq!(action_index(action), Ok(index));
            assert_eq!(action_from_index(index), Ok(action));
        }
        assert_eq!(
            action_index(PublicAction::Pulse { slot: 3 }),
            Err(V2PublicSchemaError::UnsupportedAction)
        );
        assert_eq!(
            action_from_index(V2_REQUIRED_ACTIONS),
            Err(V2PublicSchemaError::ActionIndexOutOfRange {
                index: V2_REQUIRED_ACTIONS
            })
        );
    }

    #[test]
    fn public_state_rejects_out_of_schema_values() {
        assert!(V2PublicState::new([0, 31, 17, 7]).is_ok());
        assert_eq!(
            V2PublicState::new([32, 0, 0, 0]),
            Err(V2PublicSchemaError::CountOutOfRange {
                index: 0,
                value: 32
            })
        );
        assert_eq!(
            V2PublicState::new([0, 0, 0, 8]),
            Err(V2PublicSchemaError::ContextOutOfRange { value: 8 })
        );
    }
}
