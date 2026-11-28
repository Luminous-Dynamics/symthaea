// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Event-triggered, pixel-budgeted temporal foveation.
//!
//! The policy controls analysis of frames/ROIs already delivered to the
//! foveation subsystem. It does not open a camera, authorize sensor capture,
//! increase full-frame video FPS, or grant permission to retain/export media.
//! Callers must continue to enforce capture and privacy policy upstream.
//!
//! The design hypothesis is a low-rate global context stream plus short,
//! event-triggered ROI bursts. This is intended to recover fast nonverbal cues
//! without blindly feeding more full-frame video tokens into a model whose
//! cross-modal attention budget may already be saturated.

use std::collections::{HashSet, VecDeque};

pub const MAX_TRIGGER_IDS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalFoveationConfig {
    /// Baseline interval for ROI inspection when no burst is active.
    pub baseline_interval_us: u64,
    /// Minimum interval during a triggered ROI burst.
    pub burst_interval_us: u64,
    /// How long a valid high-urgency event keeps the burst active.
    pub burst_duration_us: u64,
    /// Minimum event urgency for activating a burst.
    pub trigger_urgency_threshold: f32,
    /// Pixel-work token capacity in the rolling bucket.
    pub pixel_token_capacity: u64,
    /// Pixel-work tokens replenished per second.
    pub pixel_tokens_per_second: u64,
    /// Maximum in-flight requests permitted under this policy.
    pub max_in_flight: usize,
}

impl Default for TemporalFoveationConfig {
    fn default() -> Self {
        Self {
            baseline_interval_us: 1_000_000,
            burst_interval_us: 250_000,
            burst_duration_us: 1_500_000,
            trigger_urgency_threshold: 0.70,
            pixel_token_capacity: 1_000_000,
            pixel_tokens_per_second: 500_000,
            max_in_flight: 2,
        }
    }
}

impl TemporalFoveationConfig {
    pub fn validate(&self) -> Result<(), TemporalFoveationError> {
        if self.baseline_interval_us == 0
            || self.burst_interval_us == 0
            || self.burst_interval_us > self.baseline_interval_us
            || self.burst_duration_us == 0
            || !self.trigger_urgency_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.trigger_urgency_threshold)
            || self.pixel_token_capacity == 0
            || self.pixel_tokens_per_second == 0
            || self.max_in_flight == 0
        {
            return Err(TemporalFoveationError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalTriggerKind {
    VisualSurprise,
    MotionOnset,
    AudioProsodyChange,
    TurnBoundaryCandidate,
    BackchannelCandidate,
    CrossModalDisagreement,
    UserDirectedAttention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemporalTrigger {
    pub id: String,
    /// Timestamp in the foveation source clock domain, in microseconds.
    pub timestamp_us: u64,
    pub kind: TemporalTriggerKind,
    pub urgency: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalSamplingMode {
    BaselineRoi,
    BurstRoi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalBudgetDecision {
    Dispatch {
        mode: TemporalSamplingMode,
        pixel_cost: u64,
        tokens_remaining: u64,
    },
    Defer {
        until_us: u64,
        mode: TemporalSamplingMode,
    },
    BudgetExhausted {
        required: u64,
        available: u64,
    },
    ConcurrencyLimited {
        in_flight: usize,
        maximum: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalFoveationError {
    InvalidConfig,
    InvalidTriggerId,
    InvalidUrgency,
    ClockRegression,
    DuplicateTrigger,
    TriggerCapacityReached,
    DispatchNotPermitted,
    PixelBudgetExceeded,
}

#[derive(Debug, Clone)]
pub struct TemporalFoveationBudget {
    config: TemporalFoveationConfig,
    tokens: u64,
    refill_remainder: u128,
    last_refill_us: u64,
    last_dispatch_us: Option<u64>,
    burst_until_us: Option<u64>,
    last_trigger_us: Option<u64>,
    trigger_ids: HashSet<String>,
    trigger_order: VecDeque<String>,
}

impl TemporalFoveationBudget {
    pub fn new(
        config: TemporalFoveationConfig,
        now_us: u64,
    ) -> Result<Self, TemporalFoveationError> {
        config.validate()?;
        Ok(Self {
            tokens: config.pixel_token_capacity,
            refill_remainder: 0,
            last_refill_us: now_us,
            last_dispatch_us: None,
            burst_until_us: None,
            last_trigger_us: None,
            trigger_ids: HashSet::new(),
            trigger_order: VecDeque::new(),
            config,
        })
    }

    /// Register one explicit temporal trigger. Duplicate trigger IDs do not
    /// extend a burst, preventing repeated polling of one event from creating
    /// an artificial high-rate stream.
    pub fn observe_trigger(
        &mut self,
        trigger: TemporalTrigger,
    ) -> Result<bool, TemporalFoveationError> {
        validate_trigger_id(&trigger.id)?;
        if !trigger.urgency.is_finite() || !(0.0..=1.0).contains(&trigger.urgency) {
            return Err(TemporalFoveationError::InvalidUrgency);
        }
        if trigger.timestamp_us < self.last_refill_us {
            return Err(TemporalFoveationError::ClockRegression);
        }
        if let Some(previous) = self.last_trigger_us {
            if trigger.timestamp_us < previous {
                return Err(TemporalFoveationError::ClockRegression);
            }
        }
        if self.trigger_ids.contains(&trigger.id) {
            return Err(TemporalFoveationError::DuplicateTrigger);
        }
        if self.trigger_order.len() >= MAX_TRIGGER_IDS {
            return Err(TemporalFoveationError::TriggerCapacityReached);
        }
        self.last_trigger_us = Some(trigger.timestamp_us);
        self.trigger_ids.insert(trigger.id.clone());
        self.trigger_order.push_back(trigger.id);
        if trigger.urgency >= self.config.trigger_urgency_threshold {
            let proposed_until = trigger
                .timestamp_us
                .saturating_add(self.config.burst_duration_us);
            self.burst_until_us = Some(
                self.burst_until_us
                    .map_or(proposed_until, |existing| existing.max(proposed_until)),
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn decide(
        &mut self,
        now_us: u64,
        roi_pixel_cost: u64,
        in_flight: usize,
    ) -> Result<TemporalBudgetDecision, TemporalFoveationError> {
        self.refill(now_us)?;
        if in_flight >= self.config.max_in_flight {
            return Ok(TemporalBudgetDecision::ConcurrencyLimited {
                in_flight,
                maximum: self.config.max_in_flight,
            });
        }
        let mode = self.mode_at(now_us);
        let interval = self.interval_for(mode);
        if let Some(last_dispatch) = self.last_dispatch_us {
            let earliest = last_dispatch.saturating_add(interval);
            if now_us < earliest {
                return Ok(TemporalBudgetDecision::Defer {
                    until_us: earliest,
                    mode,
                });
            }
        }
        if roi_pixel_cost > self.tokens {
            return Ok(TemporalBudgetDecision::BudgetExhausted {
                required: roi_pixel_cost,
                available: self.tokens,
            });
        }
        Ok(TemporalBudgetDecision::Dispatch {
            mode,
            pixel_cost: roi_pixel_cost,
            tokens_remaining: self.tokens - roi_pixel_cost,
        })
    }

    /// Commit a successfully enqueued ROI request. Call only after the
    /// background channel accepts it; queue failures must not spend budget.
    pub fn record_dispatch(
        &mut self,
        now_us: u64,
        roi_pixel_cost: u64,
    ) -> Result<(), TemporalFoveationError> {
        self.refill(now_us)?;
        let mode = self.mode_at(now_us);
        if let Some(last_dispatch) = self.last_dispatch_us {
            if now_us < last_dispatch.saturating_add(self.interval_for(mode)) {
                return Err(TemporalFoveationError::DispatchNotPermitted);
            }
        }
        if roi_pixel_cost > self.tokens {
            return Err(TemporalFoveationError::PixelBudgetExceeded);
        }
        self.tokens -= roi_pixel_cost;
        self.last_dispatch_us = Some(now_us);
        Ok(())
    }

    pub fn tokens_remaining(&self) -> u64 {
        self.tokens
    }

    pub fn burst_active(&self, now_us: u64) -> bool {
        self.burst_until_us.is_some_and(|until| now_us < until)
    }

    pub fn config(&self) -> TemporalFoveationConfig {
        self.config
    }

    fn mode_at(&self, now_us: u64) -> TemporalSamplingMode {
        if self.burst_active(now_us) {
            TemporalSamplingMode::BurstRoi
        } else {
            TemporalSamplingMode::BaselineRoi
        }
    }

    fn interval_for(&self, mode: TemporalSamplingMode) -> u64 {
        match mode {
            TemporalSamplingMode::BaselineRoi => self.config.baseline_interval_us,
            TemporalSamplingMode::BurstRoi => self.config.burst_interval_us,
        }
    }

    fn refill(&mut self, now_us: u64) -> Result<(), TemporalFoveationError> {
        if now_us < self.last_refill_us {
            return Err(TemporalFoveationError::ClockRegression);
        }
        let elapsed = now_us - self.last_refill_us;
        let numerator = (elapsed as u128)
            .saturating_mul(self.config.pixel_tokens_per_second as u128)
            .saturating_add(self.refill_remainder);
        let added = numerator / 1_000_000;
        self.refill_remainder = numerator % 1_000_000;
        self.tokens = self
            .tokens
            .saturating_add(added.min(u64::MAX as u128) as u64)
            .min(self.config.pixel_token_capacity);
        self.last_refill_us = now_us;
        Ok(())
    }
}

fn validate_trigger_id(value: &str) -> Result<(), TemporalFoveationError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(TemporalFoveationError::InvalidTriggerId);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trigger(id: &str, timestamp_us: u64, urgency: f32) -> TemporalTrigger {
        TemporalTrigger {
            id: id.to_owned(),
            timestamp_us,
            kind: TemporalTriggerKind::TurnBoundaryCandidate,
            urgency,
        }
    }

    #[test]
    fn baseline_stream_is_rate_limited_to_one_roi_per_second() {
        let config = TemporalFoveationConfig::default();
        let mut budget = TemporalFoveationBudget::new(config, 0).unwrap();
        assert!(matches!(
            budget.decide(0, 100_000, 0).unwrap(),
            TemporalBudgetDecision::Dispatch { mode: TemporalSamplingMode::BaselineRoi, .. }
        ));
        budget.record_dispatch(0, 100_000).unwrap();
        assert_eq!(
            budget.decide(999_999, 100_000, 0).unwrap(),
            TemporalBudgetDecision::Defer {
                until_us: 1_000_000,
                mode: TemporalSamplingMode::BaselineRoi
            }
        );
    }

    #[test]
    fn explicit_high_urgency_trigger_opens_short_roi_burst() {
        let config = TemporalFoveationConfig::default();
        let mut budget = TemporalFoveationBudget::new(config, 0).unwrap();
        assert!(budget.observe_trigger(trigger("turn-1", 100_000, 0.9)).unwrap());
        assert!(budget.burst_active(100_001));
        assert!(matches!(
            budget.decide(100_000, 100_000, 0).unwrap(),
            TemporalBudgetDecision::Dispatch { mode: TemporalSamplingMode::BurstRoi, .. }
        ));
        budget.record_dispatch(100_000, 100_000).unwrap();
        assert_eq!(
            budget.decide(349_999, 100_000, 0).unwrap(),
            TemporalBudgetDecision::Defer {
                until_us: 350_000,
                mode: TemporalSamplingMode::BurstRoi
            }
        );
    }

    #[test]
    fn low_urgency_trigger_does_not_open_burst() {
        let config = TemporalFoveationConfig::default();
        let mut budget = TemporalFoveationBudget::new(config, 0).unwrap();
        assert!(!budget.observe_trigger(trigger("low-1", 10, 0.2)).unwrap());
        assert!(!budget.burst_active(10));
    }

    #[test]
    fn repeated_event_id_cannot_extend_a_burst() {
        let config = TemporalFoveationConfig::default();
        let mut budget = TemporalFoveationBudget::new(config, 0).unwrap();
        budget.observe_trigger(trigger("turn-1", 100, 0.9)).unwrap();
        assert_eq!(
            budget.observe_trigger(trigger("turn-1", 200_000, 0.9)),
            Err(TemporalFoveationError::DuplicateTrigger)
        );
        assert!(budget.burst_active(1_500_099));
        assert!(!budget.burst_active(1_500_100));
    }

    #[test]
    fn trigger_time_regression_is_rejected() {
        let config = TemporalFoveationConfig::default();
        let mut budget = TemporalFoveationBudget::new(config, 0).unwrap();
        budget.observe_trigger(trigger("turn-2", 200, 0.9)).unwrap();
        assert_eq!(
            budget.observe_trigger(trigger("turn-1", 100, 0.9)),
            Err(TemporalFoveationError::ClockRegression)
        );
    }

    #[test]
    fn pixel_budget_caps_roi_work_even_during_a_burst() {
        let config = TemporalFoveationConfig {
            pixel_token_capacity: 200_000,
            pixel_tokens_per_second: 100_000,
            ..TemporalFoveationConfig::default()
        };
        let mut budget = TemporalFoveationBudget::new(config, 0).unwrap();
        budget.observe_trigger(trigger("surprise-1", 0, 1.0)).unwrap();
        budget.record_dispatch(0, 180_000).unwrap();
        assert_eq!(
            budget.decide(250_000, 100_000, 0).unwrap(),
            TemporalBudgetDecision::BudgetExhausted {
                required: 100_000,
                available: 45_000
            }
        );
    }

    #[test]
    fn token_refill_preserves_fractional_remainder() {
        let config = TemporalFoveationConfig {
            pixel_token_capacity: 10,
            pixel_tokens_per_second: 3,
            ..TemporalFoveationConfig::default()
        };
        let mut budget = TemporalFoveationBudget::new(config, 0).unwrap();
        budget.record_dispatch(0, 10).unwrap();
        budget.decide(100_000, 0, 0).unwrap();
        budget.decide(200_000, 0, 0).unwrap();
        budget.decide(300_000, 0, 0).unwrap();
        assert_eq!(budget.tokens_remaining(), 0);
        budget.decide(400_000, 0, 0).unwrap();
        assert_eq!(budget.tokens_remaining(), 1);
    }

    #[test]
    fn concurrency_limit_is_independent_of_pixel_budget() {
        let config = TemporalFoveationConfig {
            max_in_flight: 1,
            ..TemporalFoveationConfig::default()
        };
        let mut budget = TemporalFoveationBudget::new(config, 0).unwrap();
        assert_eq!(
            budget.decide(0, 1, 1).unwrap(),
            TemporalBudgetDecision::ConcurrencyLimited {
                in_flight: 1,
                maximum: 1
            }
        );
    }

    #[test]
    fn invalid_trigger_urgency_and_config_fail_closed() {
        let mut invalid_config = TemporalFoveationConfig::default();
        invalid_config.burst_interval_us = 2_000_000;
        assert_eq!(
            TemporalFoveationBudget::new(invalid_config, 0).err(),
            Some(TemporalFoveationError::InvalidConfig)
        );
        let mut budget =
            TemporalFoveationBudget::new(TemporalFoveationConfig::default(), 0).unwrap();
        assert_eq!(
            budget.observe_trigger(trigger("bad-1", 0, f32::NAN)),
            Err(TemporalFoveationError::InvalidUrgency)
        );
    }

    #[test]
    fn policy_rejects_clock_regression_during_refill() {
        let mut budget =
            TemporalFoveationBudget::new(TemporalFoveationConfig::default(), 100).unwrap();
        assert_eq!(
            budget.decide(99, 100, 0),
            Err(TemporalFoveationError::ClockRegression)
        );
    }
}
