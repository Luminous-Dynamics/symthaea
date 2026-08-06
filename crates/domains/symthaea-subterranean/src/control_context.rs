// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared controller input contract for training and deployment.
//!
//! A controller trained on a sensor-state hypervector must not be deployed on
//! an unrelated thought hypervector. Both paths therefore use the same bound
//! representation: role-bound perception plus role-bound intent. Training uses
//! a deterministic neutral intent when no mission intent is supplied.

use crate::encoder::SubterraneanHdcEncoder;
use crate::mission::SubterraneanMissionIntent;
use crate::types::SubterraneanState;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

pub struct SubterraneanControlContextEncoder {
    perception_encoder: SubterraneanHdcEncoder,
    perception_role: ContinuousHV,
    intent_role: ContinuousHV,
    mission_role: ContinuousHV,
    neutral_intent: ContinuousHV,
    mission_intents: Vec<ContinuousHV>,
}

impl SubterraneanControlContextEncoder {
    pub fn new(genesis: &GenesisSeed, levels: usize) -> Self {
        Self {
            perception_encoder: SubterraneanHdcEncoder::new(genesis, levels),
            perception_role: ContinuousHV::from_genesis(
                genesis,
                "subterranean::role::perception",
                DIM,
            ),
            intent_role: ContinuousHV::from_genesis(genesis, "subterranean::role::intent", DIM),
            mission_role: ContinuousHV::from_genesis(genesis, "subterranean::role::mission", DIM),
            neutral_intent: ContinuousHV::from_genesis(
                genesis,
                "subterranean::intent::neutral",
                DIM,
            ),
            mission_intents: SubterraneanMissionIntent::ALL
                .into_iter()
                .map(|intent| {
                    ContinuousHV::from_genesis(
                        genesis,
                        &format!("subterranean::mission::{}", intent.label()),
                        DIM,
                    )
                })
                .collect(),
        }
    }

    pub fn encode(
        &mut self,
        state: &SubterraneanState,
        intent: Option<&ContinuousHV>,
        mission: SubterraneanMissionIntent,
    ) -> ContinuousHV {
        // Each role-bound term is normalized to unit magnitude before
        // bundling so perception/intent/mission contribute equally to the
        // context regardless of the raw magnitude of the vectors used to
        // build them. `perception` is already unit-normalized by
        // `SubterraneanHdcEncoder::encode`, but `intent`/`neutral_intent`
        // (raw `ContinuousHV::random`/`from_genesis` output, component
        // magnitude O(1)) and the mission symbols (same construction) have
        // norm ~O(sqrt(DIM)) before binding -- roughly 70x larger than the
        // perception term at DIM=16384. Left unnormalized, that imbalance
        // makes the bundled context nearly blind to sensor state: two
        // encodings that differ only in physical state end up with cosine
        // similarity effectively 1.0, since intent+mission dominate the sum.
        let perception = self.perception_encoder.encode(state);
        let mut context = perception.bind(&self.perception_role).normalize();
        let intent = intent.unwrap_or(&self.neutral_intent);
        context.add_in_place(&intent.bind(&self.intent_role).normalize());
        context.add_in_place(
            &self.mission_intents[mission.index()]
                .bind(&self.mission_role)
                .normalize(),
        );
        context.normalize()
    }

    pub fn encode_perception(&mut self, state: &SubterraneanState) -> ContinuousHV {
        self.perception_encoder.encode(state)
    }

    pub fn reset(&mut self) {
        self.perception_encoder.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_contains_both_state_and_intent() {
        let genesis = GenesisSeed::from_phrase("control-context-test");
        let mut encoder = SubterraneanControlContextEncoder::new(&genesis, 32);
        let state = SubterraneanState::home();
        let intent_a = ContinuousHV::random(DIM, 10);
        let intent_b = ContinuousHV::random(DIM, 11);

        let a = encoder.encode(&state, Some(&intent_a), SubterraneanMissionIntent::Explore);
        let b = encoder.encode(&state, Some(&intent_b), SubterraneanMissionIntent::Explore);
        assert_eq!(a.dim(), DIM);
        assert!(a.similarity(&b) < 0.9999);

        let mut changed_state = state.clone();
        changed_state.channels[crate::types::CUTTER_TEMP_C] = 150.0;
        let c = encoder.encode(
            &changed_state,
            Some(&intent_a),
            SubterraneanMissionIntent::Explore,
        );
        assert!(a.similarity(&c) < 0.9999);
    }

    #[test]
    fn neutral_intent_is_deterministic() {
        let genesis = GenesisSeed::from_phrase("control-context-neutral");
        let mut a = SubterraneanControlContextEncoder::new(&genesis, 32);
        let mut b = SubterraneanControlContextEncoder::new(&genesis, 32);
        let state = SubterraneanState::home();
        assert_eq!(
            a.encode(&state, None, SubterraneanMissionIntent::Explore)
                .as_slice(),
            b.encode(&state, None, SubterraneanMissionIntent::Explore)
                .as_slice()
        );
    }

    #[test]
    fn mission_symbol_changes_controller_context() {
        let genesis = GenesisSeed::from_phrase("control-context-mission");
        let mut encoder = SubterraneanControlContextEncoder::new(&genesis, 32);
        let state = SubterraneanState::home();
        let explore = encoder.encode(&state, None, SubterraneanMissionIntent::Explore);
        let return_home = encoder.encode(&state, None, SubterraneanMissionIntent::ReturnHome);
        assert!(explore.similarity(&return_home) < 0.9999);
    }
}
