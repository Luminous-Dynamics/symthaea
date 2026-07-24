// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic browser-independent Listen journey reducer.

use serde::{Deserialize, Serialize};
use symthaea_muse_protocol::ArtifactIdentity;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JourneyPolicy {
    Resonance,
    Discovery,
    Contrast,
}

impl JourneyPolicy {
    pub const ALL: [Self; 3] = [Self::Resonance, Self::Discovery, Self::Contrast];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Resonance => "Resonance",
            Self::Discovery => "Discovery",
            Self::Contrast => "Contrast",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JourneyArtifact {
    pub identity: ArtifactIdentity,
    pub candidate_id: u64,
    pub title: String,
    pub style: String,
    pub relation_from_previous: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingComposition {
    pub journey_step_id: u64,
    pub composition_request_id: String,
    pub prefetch_epoch: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JourneyState {
    pub journey_id: String,
    pub policy: JourneyPolicy,
    pub traversal_seed: u64,
    pub rng_position: u64,
    pub step_sequence: u64,
    pub prefetch_epoch: u64,
    pub previous: Vec<JourneyArtifact>,
    pub current: Option<JourneyArtifact>,
    pub next: Option<JourneyArtifact>,
    pub pending: Option<PendingComposition>,
    pub recent_compositions: Vec<String>,
    pub error: Option<String>,
}

impl JourneyState {
    pub fn new(journey_id: String, policy: JourneyPolicy, traversal_seed: u64) -> Self {
        Self {
            journey_id,
            policy,
            traversal_seed,
            rng_position: 0,
            step_sequence: 0,
            prefetch_epoch: 0,
            previous: Vec::new(),
            current: None,
            next: None,
            pending: None,
            recent_compositions: Vec::new(),
            error: None,
        }
    }

    pub fn reduce(&mut self, command: JourneyCommand) -> Vec<JourneyEffect> {
        match command {
            JourneyCommand::RequestNext if self.pending.is_none() && self.next.is_none() => {
                vec![self.begin_request()]
            }
            JourneyCommand::PrefetchCompleted {
                composition_request_id,
                prefetch_epoch,
                selected,
            } if self.pending.as_ref().is_some_and(|pending| {
                pending.composition_request_id == composition_request_id
                    && pending.prefetch_epoch == prefetch_epoch
            }) =>
            {
                self.pending = None;
                self.error = None;
                self.rng_position = self.rng_position.wrapping_add(1);
                self.remember(&selected);
                if self.current.is_none() {
                    self.current = Some(selected);
                    vec![JourneyEffect::CurrentChanged]
                } else {
                    self.next = Some(selected);
                    Vec::new()
                }
            }
            JourneyCommand::CompositionFailed {
                composition_request_id,
                prefetch_epoch,
                message,
            } if self.pending.as_ref().is_some_and(|pending| {
                pending.composition_request_id == composition_request_id
                    && pending.prefetch_epoch == prefetch_epoch
            }) =>
            {
                self.pending = None;
                self.error = Some(message);
                Vec::new()
            }
            JourneyCommand::Advance => {
                let Some(next) = self.next.take() else {
                    return if self.pending.is_none() {
                        vec![self.begin_request()]
                    } else {
                        Vec::new()
                    };
                };
                if let Some(current) = self.current.replace(next) {
                    self.previous.push(current);
                }
                self.step_sequence = self.step_sequence.wrapping_add(1);
                let mut effects = vec![JourneyEffect::CurrentChanged];
                if self.pending.is_none() {
                    effects.push(self.begin_request());
                }
                effects
            }
            JourneyCommand::ReturnToPrevious => {
                let Some(previous) = self.previous.pop() else {
                    return Vec::new();
                };
                self.invalidate_prefetch();
                self.next = self.current.replace(previous);
                self.step_sequence = self.step_sequence.wrapping_add(1);
                vec![JourneyEffect::CurrentChanged]
            }
            JourneyCommand::ChangePolicy(policy) if policy != self.policy => {
                self.policy = policy;
                self.next = None;
                self.invalidate_prefetch();
                vec![self.begin_request()]
            }
            JourneyCommand::Restore(restored) => {
                *self = *restored;
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    fn begin_request(&mut self) -> JourneyEffect {
        self.prefetch_epoch = self.prefetch_epoch.wrapping_add(1).max(1);
        let request = PendingComposition {
            journey_step_id: self.step_sequence,
            composition_request_id: format!(
                "{}-{}-{}-{}",
                self.journey_id, self.step_sequence, self.prefetch_epoch, self.rng_position
            ),
            prefetch_epoch: self.prefetch_epoch,
        };
        self.pending = Some(request.clone());
        JourneyEffect::ComposeNext {
            request,
            policy: self.policy,
            traversal_seed: self.traversal_seed,
            rng_position: self.rng_position,
            recent_compositions: self.recent_compositions.clone(),
        }
    }

    fn invalidate_prefetch(&mut self) {
        self.prefetch_epoch = self.prefetch_epoch.wrapping_add(1).max(1);
        self.pending = None;
    }

    fn remember(&mut self, artifact: &JourneyArtifact) {
        self.recent_compositions
            .push(artifact.identity.composition.0.clone());
        const HISTORY_LIMIT: usize = 32;
        if self.recent_compositions.len() > HISTORY_LIMIT {
            self.recent_compositions
                .drain(..self.recent_compositions.len() - HISTORY_LIMIT);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum JourneyCommand {
    RequestNext,
    PrefetchCompleted {
        composition_request_id: String,
        prefetch_epoch: u64,
        /// Already selected by the authoritative Rust server policy.
        selected: JourneyArtifact,
    },
    CompositionFailed {
        composition_request_id: String,
        prefetch_epoch: u64,
        message: String,
    },
    Advance,
    ReturnToPrevious,
    ChangePolicy(JourneyPolicy),
    Restore(Box<JourneyState>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum JourneyEffect {
    ComposeNext {
        request: PendingComposition,
        policy: JourneyPolicy,
        traversal_seed: u64,
        rng_position: u64,
        recent_compositions: Vec<String>,
    },
    CurrentChanged,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_muse_protocol::{
        CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId,
    };

    fn artifact(id: &str) -> JourneyArtifact {
        JourneyArtifact {
            identity: ArtifactIdentity {
                score_content: ScoreContentArtifactId(format!("scr_{id}")),
                composition: CompositionArtifactId(format!("cmp_{id}")),
                rendition: RenditionArtifactId(format!("rnd_{id}")),
            },
            candidate_id: 1,
            title: id.into(),
            style: "Classical".into(),
            relation_from_previous: "test".into(),
        }
    }

    fn request(effect: &JourneyEffect) -> PendingComposition {
        match effect {
            JourneyEffect::ComposeNext { request, .. } => request.clone(),
            _ => panic!("expected composition request"),
        }
    }

    #[test]
    fn stale_prefetch_cannot_enter_queue_after_policy_change() {
        let mut state = JourneyState::new("j".into(), JourneyPolicy::Resonance, 7);
        let old = request(&state.reduce(JourneyCommand::RequestNext)[0]);
        state.reduce(JourneyCommand::ChangePolicy(JourneyPolicy::Contrast));
        state.reduce(JourneyCommand::PrefetchCompleted {
            composition_request_id: old.composition_request_id,
            prefetch_epoch: old.prefetch_epoch,
            selected: artifact("stale"),
        });
        assert!(state.current.is_none());
        assert!(state.next.is_none());
    }

    #[test]
    fn advance_is_coherent_and_requests_replenishment() {
        let mut state = JourneyState::new("j".into(), JourneyPolicy::Discovery, 9);
        let first = request(&state.reduce(JourneyCommand::RequestNext)[0]);
        state.reduce(JourneyCommand::PrefetchCompleted {
            composition_request_id: first.composition_request_id,
            prefetch_epoch: first.prefetch_epoch,
            selected: artifact("a"),
        });
        let second = request(&state.reduce(JourneyCommand::RequestNext)[0]);
        state.reduce(JourneyCommand::PrefetchCompleted {
            composition_request_id: second.composition_request_id,
            prefetch_epoch: second.prefetch_epoch,
            selected: artifact("b"),
        });
        let effects = state.reduce(JourneyCommand::Advance);
        assert_eq!(state.previous[0].title, "a");
        assert_eq!(state.current.as_ref().unwrap().title, "b");
        assert!(effects.contains(&JourneyEffect::CurrentChanged));
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, JourneyEffect::ComposeNext { .. }))
        );
    }

    #[test]
    fn event_replay_is_deterministic() {
        let commands = |state: &mut JourneyState| {
            let first = request(&state.reduce(JourneyCommand::RequestNext)[0]);
            state.reduce(JourneyCommand::PrefetchCompleted {
                composition_request_id: first.composition_request_id,
                prefetch_epoch: first.prefetch_epoch,
                selected: artifact("a"),
            });
            let second = request(&state.reduce(JourneyCommand::RequestNext)[0]);
            state.reduce(JourneyCommand::PrefetchCompleted {
                composition_request_id: second.composition_request_id,
                prefetch_epoch: second.prefetch_epoch,
                selected: artifact("b"),
            });
            state.reduce(JourneyCommand::Advance);
        };
        let mut left = JourneyState::new("journey".into(), JourneyPolicy::Resonance, 42);
        let mut right = left.clone();
        commands(&mut left);
        commands(&mut right);
        assert_eq!(left, right);
    }
}
