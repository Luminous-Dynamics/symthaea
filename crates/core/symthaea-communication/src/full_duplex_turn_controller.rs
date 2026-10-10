// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full-duplex spoken turn-taking controller.
//!
//! This module controls conversational floor/output semantics only. It does not
//! perform ASR, TTS, acoustic processing, likeness authorization, or physical action.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const FULL_DUPLEX_TURN_CONTROLLER_SCHEMA_V1: &str =
    "symthaea.communication.full-duplex-turn-controller.v1";
const MAX_TRACKED_IDS_V1: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FullDuplexUserFloorStateV1 {
    Silent,
    Speaking {
        utterance_id: String,
        started_at_ns: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FullDuplexYieldReasonV1 {
    UserInterruption,
    ExplicitStop,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FullDuplexAssistantOutputStateV1 {
    Silent,
    Speaking {
        turn_id: String,
        started_at_ns: u64,
    },
    YieldRequested {
        turn_id: String,
        requested_at_ns: u64,
        reason: FullDuplexYieldReasonV1,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullDuplexStopLatchV1 {
    pub event_id: String,
    pub requested_at_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullDuplexInterruptionReceiptV1 {
    pub session_epoch: u64,
    pub turn_id: String,
    pub reason: FullDuplexYieldReasonV1,
    pub requested_at_ns: u64,
    pub output_stopped_at_ns: u64,
    pub latency_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullDuplexStopReceiptV1 {
    pub session_epoch: u64,
    pub event_id: String,
    pub turn_id: Option<String>,
    pub requested_at_ns: u64,
    pub output_stopped_at_ns: u64,
    pub latency_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullDuplexDeescalationReceiptV1 {
    pub session_epoch: u64,
    pub request_event_id: String,
    pub plan_ref: String,
    pub requested_at_ns: u64,
    pub acknowledged_at_ns: u64,
    pub latency_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullDuplexBackchannelReceiptV1 {
    pub session_epoch: u64,
    pub event_id: String,
    pub observed_at_ns: u64,
    pub assistant_output_active: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullDuplexAssistantSpeechPermitV1 {
    pub session_epoch: u64,
    pub turn_id: String,
    pub deescalation_required: bool,
    pub response_latency_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FullDuplexControlActionV1 {
    UserFloorOpened {
        utterance_id: String,
    },
    UserFloorReleased {
        utterance_id: String,
    },
    RequestAssistantYield {
        turn_id: String,
        reason: FullDuplexYieldReasonV1,
        requested_at_ns: u64,
    },
    BackchannelObserved {
        event_id: String,
        assistant_output_active: bool,
    },
    DeescalationRequired {
        event_id: String,
        newly_latched: bool,
    },
    StopLatched {
        event_id: String,
        newly_latched: bool,
    },
    AssistantSpeechPermitted(FullDuplexAssistantSpeechPermitV1),
    AssistantSpeechEnded {
        turn_id: String,
    },
    AssistantOutputStopped {
        turn_id: String,
        reason: FullDuplexYieldReasonV1,
        latency_ns: u64,
    },
    DeescalationAcknowledged {
        plan_ref: String,
        latency_ns: u64,
    },
    SessionReset {
        session_epoch: u64,
    },
}

#[derive(Debug)]
pub struct FullDuplexTurnControllerV1 {
    session_epoch: u64,
    user_floor: FullDuplexUserFloorStateV1,
    assistant_output: FullDuplexAssistantOutputStateV1,
    stop_latch: Option<FullDuplexStopLatchV1>,
    deescalation_request: Option<(String, u64)>,
    last_interruption_receipt: Option<FullDuplexInterruptionReceiptV1>,
    last_stop_receipt: Option<FullDuplexStopReceiptV1>,
    last_deescalation_receipt: Option<FullDuplexDeescalationReceiptV1>,
    backchannel_receipts: Vec<FullDuplexBackchannelReceiptV1>,
    last_user_floor_released_ns: Option<u64>,
    last_event_ns: Option<u64>,
    seen_event_ids: BTreeSet<String>,
    seen_turn_ids: BTreeSet<String>,
}

impl FullDuplexTurnControllerV1 {
    pub fn new(session_epoch: u64) -> Result<Self, FullDuplexTurnErrorV1> {
        if session_epoch == 0 {
            return Err(FullDuplexTurnErrorV1::InvalidSessionEpoch);
        }
        Ok(Self {
            session_epoch,
            user_floor: FullDuplexUserFloorStateV1::Silent,
            assistant_output: FullDuplexAssistantOutputStateV1::Silent,
            stop_latch: None,
            deescalation_request: None,
            last_interruption_receipt: None,
            last_stop_receipt: None,
            last_deescalation_receipt: None,
            backchannel_receipts: Vec::new(),
            last_user_floor_released_ns: None,
            last_event_ns: None,
            seen_event_ids: BTreeSet::new(),
            seen_turn_ids: BTreeSet::new(),
        })
    }

    pub const fn session_epoch(&self) -> u64 {
        self.session_epoch
    }

    pub fn user_floor(&self) -> &FullDuplexUserFloorStateV1 {
        &self.user_floor
    }

    pub fn assistant_output(&self) -> &FullDuplexAssistantOutputStateV1 {
        &self.assistant_output
    }

    pub fn stop_latch(&self) -> Option<&FullDuplexStopLatchV1> {
        self.stop_latch.as_ref()
    }

    pub const fn deescalation_required(&self) -> bool {
        self.deescalation_request.is_some()
    }

    pub fn last_interruption_receipt(&self) -> Option<&FullDuplexInterruptionReceiptV1> {
        self.last_interruption_receipt.as_ref()
    }

    pub fn last_stop_receipt(&self) -> Option<&FullDuplexStopReceiptV1> {
        self.last_stop_receipt.as_ref()
    }

    pub fn last_deescalation_receipt(&self) -> Option<&FullDuplexDeescalationReceiptV1> {
        self.last_deescalation_receipt.as_ref()
    }

    pub fn backchannel_receipts(&self) -> &[FullDuplexBackchannelReceiptV1] {
        &self.backchannel_receipts
    }

    pub fn user_speech_start(
        &mut self,
        utterance_id: impl Into<String>,
        now_ns: u64,
    ) -> Result<Vec<FullDuplexControlActionV1>, FullDuplexTurnErrorV1> {
        self.check_clock(now_ns)?;
        let utterance_id = canonical_id(utterance_id.into())?;
        if !matches!(self.user_floor, FullDuplexUserFloorStateV1::Silent) {
            return Err(FullDuplexTurnErrorV1::UserAlreadySpeaking);
        }
        self.track_event_id(&utterance_id)?;
        self.user_floor = FullDuplexUserFloorStateV1::Speaking {
            utterance_id: utterance_id.clone(),
            started_at_ns: now_ns,
        };

        let mut actions = vec![FullDuplexControlActionV1::UserFloorOpened {
            utterance_id,
        }];
        if let FullDuplexAssistantOutputStateV1::Speaking { turn_id, .. } =
            self.assistant_output.clone()
        {
            self.assistant_output = FullDuplexAssistantOutputStateV1::YieldRequested {
                turn_id: turn_id.clone(),
                requested_at_ns: now_ns,
                reason: FullDuplexYieldReasonV1::UserInterruption,
            };
            actions.push(FullDuplexControlActionV1::RequestAssistantYield {
                turn_id,
                reason: FullDuplexYieldReasonV1::UserInterruption,
                requested_at_ns: now_ns,
            });
        }
        self.commit_clock(now_ns);
        Ok(actions)
    }

    pub fn user_speech_end(
        &mut self,
        utterance_id: &str,
        now_ns: u64,
    ) -> Result<FullDuplexControlActionV1, FullDuplexTurnErrorV1> {
        self.check_clock(now_ns)?;
        let utterance_id = canonical_id(utterance_id.to_owned())?;
        match &self.user_floor {
            FullDuplexUserFloorStateV1::Speaking {
                utterance_id: active,
                ..
            } if active == &utterance_id => {}
            FullDuplexUserFloorStateV1::Speaking { .. } => {
                return Err(FullDuplexTurnErrorV1::UserUtteranceMismatch)
            }
            FullDuplexUserFloorStateV1::Silent => {
                return Err(FullDuplexTurnErrorV1::UserNotSpeaking)
            }
        }
        self.user_floor = FullDuplexUserFloorStateV1::Silent;
        self.last_user_floor_released_ns = Some(now_ns);
        self.commit_clock(now_ns);
        Ok(FullDuplexControlActionV1::UserFloorReleased { utterance_id })
    }

    pub fn user_backchannel(
        &mut self,
        event_id: impl Into<String>,
        now_ns: u64,
    ) -> Result<FullDuplexControlActionV1, FullDuplexTurnErrorV1> {
        self.check_clock(now_ns)?;
        let event_id = canonical_id(event_id.into())?;
        self.track_event_id(&event_id)?;
        let assistant_output_active =
            !matches!(self.assistant_output, FullDuplexAssistantOutputStateV1::Silent);
        self.backchannel_receipts.push(FullDuplexBackchannelReceiptV1 {
            session_epoch: self.session_epoch,
            event_id: event_id.clone(),
            observed_at_ns: now_ns,
            assistant_output_active,
        });
        self.commit_clock(now_ns);
        Ok(FullDuplexControlActionV1::BackchannelObserved {
            event_id,
            assistant_output_active,
        })
    }

    pub fn user_slow_down(
        &mut self,
        event_id: impl Into<String>,
        now_ns: u64,
    ) -> Result<FullDuplexControlActionV1, FullDuplexTurnErrorV1> {
        self.check_clock(now_ns)?;
        if self.stop_latch.is_some() {
            return Err(FullDuplexTurnErrorV1::StopIsLatched);
        }
        let event_id = canonical_id(event_id.into())?;
        if self
            .deescalation_request
            .as_ref()
            .is_some_and(|(active_id, _)| active_id == &event_id)
        {
            self.commit_clock(now_ns);
            return Ok(FullDuplexControlActionV1::DeescalationRequired {
                event_id,
                newly_latched: false,
            });
        }
        self.track_event_id(&event_id)?;
        let newly_latched = self.deescalation_request.is_none();
        if newly_latched {
            self.deescalation_request = Some((event_id.clone(), now_ns));
        }
        self.commit_clock(now_ns);
        Ok(FullDuplexControlActionV1::DeescalationRequired {
            event_id,
            newly_latched,
        })
    }

    pub fn acknowledge_deescalation(
        &mut self,
        plan_ref: impl Into<String>,
        now_ns: u64,
    ) -> Result<FullDuplexControlActionV1, FullDuplexTurnErrorV1> {
        self.check_clock(now_ns)?;
        if self.stop_latch.is_some() {
            return Err(FullDuplexTurnErrorV1::StopIsLatched);
        }
        let plan_ref = canonical_ref(plan_ref.into())?;
        let (request_event_id, requested_at_ns) = self
            .deescalation_request
            .clone()
            .ok_or(FullDuplexTurnErrorV1::NoDeescalationRequired)?;
        let latency_ns = now_ns - requested_at_ns;
        self.deescalation_request = None;
        self.last_deescalation_receipt = Some(FullDuplexDeescalationReceiptV1 {
            session_epoch: self.session_epoch,
            request_event_id,
            plan_ref: plan_ref.clone(),
            requested_at_ns,
            acknowledged_at_ns: now_ns,
            latency_ns,
        });
        self.commit_clock(now_ns);
        Ok(FullDuplexControlActionV1::DeescalationAcknowledged {
            plan_ref,
            latency_ns,
        })
    }

    pub fn user_stop(
        &mut self,
        event_id: impl Into<String>,
        now_ns: u64,
    ) -> Result<Vec<FullDuplexControlActionV1>, FullDuplexTurnErrorV1> {
        self.check_clock(now_ns)?;
        let event_id = canonical_id(event_id.into())?;
        if self
            .stop_latch
            .as_ref()
            .is_some_and(|active| active.event_id == event_id)
        {
            self.commit_clock(now_ns);
            return Ok(vec![FullDuplexControlActionV1::StopLatched {
                event_id,
                newly_latched: false,
            }]);
        }
        self.track_event_id(&event_id)?;

        if self.stop_latch.is_some() {
            self.commit_clock(now_ns);
            return Ok(vec![FullDuplexControlActionV1::StopLatched {
                event_id,
                newly_latched: false,
            }]);
        }

        self.stop_latch = Some(FullDuplexStopLatchV1 {
            event_id: event_id.clone(),
            requested_at_ns: now_ns,
        });
        let mut actions = vec![FullDuplexControlActionV1::StopLatched {
            event_id: event_id.clone(),
            newly_latched: true,
        }];

        match self.assistant_output.clone() {
            FullDuplexAssistantOutputStateV1::Silent => {
                self.last_stop_receipt = Some(FullDuplexStopReceiptV1 {
                    session_epoch: self.session_epoch,
                    event_id,
                    turn_id: None,
                    requested_at_ns: now_ns,
                    output_stopped_at_ns: now_ns,
                    latency_ns: 0,
                });
            }
            FullDuplexAssistantOutputStateV1::Speaking { turn_id, .. }
            | FullDuplexAssistantOutputStateV1::YieldRequested { turn_id, .. } => {
                self.assistant_output = FullDuplexAssistantOutputStateV1::YieldRequested {
                    turn_id: turn_id.clone(),
                    requested_at_ns: now_ns,
                    reason: FullDuplexYieldReasonV1::ExplicitStop,
                };
                actions.push(FullDuplexControlActionV1::RequestAssistantYield {
                    turn_id,
                    reason: FullDuplexYieldReasonV1::ExplicitStop,
                    requested_at_ns: now_ns,
                });
            }
        }
        self.commit_clock(now_ns);
        Ok(actions)
    }

    pub fn assistant_begin(
        &mut self,
        turn_id: impl Into<String>,
        now_ns: u64,
    ) -> Result<FullDuplexControlActionV1, FullDuplexTurnErrorV1> {
        self.check_clock(now_ns)?;
        if self.stop_latch.is_some() {
            return Err(FullDuplexTurnErrorV1::StopIsLatched);
        }
        if !matches!(self.user_floor, FullDuplexUserFloorStateV1::Silent) {
            return Err(FullDuplexTurnErrorV1::UserFloorOccupied);
        }
        if !matches!(self.assistant_output, FullDuplexAssistantOutputStateV1::Silent) {
            return Err(FullDuplexTurnErrorV1::AssistantOutputBusy);
        }
        let turn_id = canonical_id(turn_id.into())?;
        self.track_turn_id(&turn_id)?;
        let response_latency_ns = self
            .last_user_floor_released_ns
            .map(|released_at_ns| now_ns - released_at_ns);
        self.assistant_output = FullDuplexAssistantOutputStateV1::Speaking {
            turn_id: turn_id.clone(),
            started_at_ns: now_ns,
        };
        let action = FullDuplexControlActionV1::AssistantSpeechPermitted(
            FullDuplexAssistantSpeechPermitV1 {
                session_epoch: self.session_epoch,
                turn_id,
                deescalation_required: self.deescalation_required(),
                response_latency_ns,
            },
        );
        self.commit_clock(now_ns);
        Ok(action)
    }

    pub fn assistant_end(
        &mut self,
        turn_id: &str,
        now_ns: u64,
    ) -> Result<FullDuplexControlActionV1, FullDuplexTurnErrorV1> {
        self.check_clock(now_ns)?;
        let turn_id = canonical_id(turn_id.to_owned())?;
        match &self.assistant_output {
            FullDuplexAssistantOutputStateV1::Speaking { turn_id: active, .. }
                if active == &turn_id => {}
            FullDuplexAssistantOutputStateV1::YieldRequested { .. } => {
                return Err(FullDuplexTurnErrorV1::YieldAcknowledgementRequired)
            }
            FullDuplexAssistantOutputStateV1::Speaking { .. } => {
                return Err(FullDuplexTurnErrorV1::AssistantTurnMismatch)
            }
            FullDuplexAssistantOutputStateV1::Silent => {
                return Err(FullDuplexTurnErrorV1::AssistantNotSpeaking)
            }
        }
        self.assistant_output = FullDuplexAssistantOutputStateV1::Silent;
        self.commit_clock(now_ns);
        Ok(FullDuplexControlActionV1::AssistantSpeechEnded { turn_id })
    }

    pub fn acknowledge_output_stopped(
        &mut self,
        turn_id: &str,
        now_ns: u64,
    ) -> Result<FullDuplexControlActionV1, FullDuplexTurnErrorV1> {
        self.check_clock(now_ns)?;
        let turn_id = canonical_id(turn_id.to_owned())?;
        let (requested_turn_id, requested_at_ns, reason) = match &self.assistant_output {
            FullDuplexAssistantOutputStateV1::YieldRequested {
                turn_id,
                requested_at_ns,
                reason,
            } => (turn_id.clone(), *requested_at_ns, *reason),
            FullDuplexAssistantOutputStateV1::Speaking { .. } => {
                return Err(FullDuplexTurnErrorV1::NoYieldRequested)
            }
            FullDuplexAssistantOutputStateV1::Silent => {
                return Err(FullDuplexTurnErrorV1::AssistantNotSpeaking)
            }
        };
        if requested_turn_id != turn_id {
            return Err(FullDuplexTurnErrorV1::AssistantTurnMismatch);
        }
        let stop_latch = if reason == FullDuplexYieldReasonV1::ExplicitStop {
            Some(
                self.stop_latch
                    .clone()
                    .ok_or(FullDuplexTurnErrorV1::MissingStopLatch)?,
            )
        } else {
            None
        };
        let latency_ns = now_ns - requested_at_ns;
        self.assistant_output = FullDuplexAssistantOutputStateV1::Silent;

        match reason {
            FullDuplexYieldReasonV1::UserInterruption => {
                self.last_interruption_receipt = Some(FullDuplexInterruptionReceiptV1 {
                    session_epoch: self.session_epoch,
                    turn_id: turn_id.clone(),
                    reason,
                    requested_at_ns,
                    output_stopped_at_ns: now_ns,
                    latency_ns,
                });
            }
            FullDuplexYieldReasonV1::ExplicitStop => {
                let stop = stop_latch.expect("validated explicit stop has latch");
                self.last_stop_receipt = Some(FullDuplexStopReceiptV1 {
                    session_epoch: self.session_epoch,
                    event_id: stop.event_id,
                    turn_id: Some(turn_id.clone()),
                    requested_at_ns: stop.requested_at_ns,
                    output_stopped_at_ns: now_ns,
                    latency_ns: now_ns - stop.requested_at_ns,
                });
            }
        }

        self.commit_clock(now_ns);
        Ok(FullDuplexControlActionV1::AssistantOutputStopped {
            turn_id,
            reason,
            latency_ns,
        })
    }

    pub fn reset_session(
        &mut self,
        next_session_epoch: u64,
        now_ns: u64,
    ) -> Result<FullDuplexControlActionV1, FullDuplexTurnErrorV1> {
        self.check_clock(now_ns)?;
        if next_session_epoch <= self.session_epoch {
            return Err(FullDuplexTurnErrorV1::StaleSessionEpoch);
        }
        if !matches!(self.user_floor, FullDuplexUserFloorStateV1::Silent)
            || !matches!(self.assistant_output, FullDuplexAssistantOutputStateV1::Silent)
        {
            return Err(FullDuplexTurnErrorV1::SessionResetUnsafe);
        }
        self.session_epoch = next_session_epoch;
        self.stop_latch = None;
        self.deescalation_request = None;
        self.last_interruption_receipt = None;
        self.last_stop_receipt = None;
        self.last_deescalation_receipt = None;
        self.backchannel_receipts.clear();
        self.last_user_floor_released_ns = None;
        self.seen_event_ids.clear();
        self.seen_turn_ids.clear();
        self.commit_clock(now_ns);
        Ok(FullDuplexControlActionV1::SessionReset {
            session_epoch: next_session_epoch,
        })
    }

    fn check_clock(&self, now_ns: u64) -> Result<(), FullDuplexTurnErrorV1> {
        if self.last_event_ns.is_some_and(|previous| now_ns < previous) {
            return Err(FullDuplexTurnErrorV1::NonMonotonicTime);
        }
        Ok(())
    }

    fn commit_clock(&mut self, now_ns: u64) {
        self.last_event_ns = Some(now_ns);
    }

    fn track_event_id(&mut self, id: &str) -> Result<(), FullDuplexTurnErrorV1> {
        if self.seen_event_ids.contains(id) {
            return Err(FullDuplexTurnErrorV1::DuplicateEventId);
        }
        if self.seen_event_ids.len() >= MAX_TRACKED_IDS_V1 {
            return Err(FullDuplexTurnErrorV1::TooManyTrackedIds);
        }
        self.seen_event_ids.insert(id.to_owned());
        Ok(())
    }

    fn track_turn_id(&mut self, id: &str) -> Result<(), FullDuplexTurnErrorV1> {
        if self.seen_turn_ids.contains(id) {
            return Err(FullDuplexTurnErrorV1::DuplicateAssistantTurnId);
        }
        if self.seen_turn_ids.len() >= MAX_TRACKED_IDS_V1 {
            return Err(FullDuplexTurnErrorV1::TooManyTrackedIds);
        }
        self.seen_turn_ids.insert(id.to_owned());
        Ok(())
    }
}

fn canonical_id(value: String) -> Result<String, FullDuplexTurnErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 256 {
        return Err(FullDuplexTurnErrorV1::InvalidIdentity);
    }
    Ok(value)
}

fn canonical_ref(value: String) -> Result<String, FullDuplexTurnErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 1024 {
        return Err(FullDuplexTurnErrorV1::InvalidReference);
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FullDuplexTurnErrorV1 {
    InvalidSessionEpoch,
    InvalidIdentity,
    InvalidReference,
    NonMonotonicTime,
    TooManyTrackedIds,
    DuplicateEventId,
    DuplicateAssistantTurnId,
    UserAlreadySpeaking,
    UserNotSpeaking,
    UserUtteranceMismatch,
    UserFloorOccupied,
    AssistantOutputBusy,
    AssistantNotSpeaking,
    AssistantTurnMismatch,
    NoYieldRequested,
    YieldAcknowledgementRequired,
    StopIsLatched,
    MissingStopLatch,
    NoDeescalationRequired,
    StaleSessionEpoch,
    SessionResetUnsafe,
}
