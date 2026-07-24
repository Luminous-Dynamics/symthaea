// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Statistical self-consistency monitoring for navigation fusion.
//!
//! A finite covariance is not automatically credible. This monitor tracks
//! normalized innovation squared (NIS), covariance collapse/inflation, rejection
//! streaks, and recent rejection fraction. It is intentionally independent of
//! a particular EKF/UKF implementation and therefore provides a contract that a
//! future estimator backend can preserve.

use std::collections::VecDeque;

use crate::navigation_estimator::NavigationSource;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationConsistencyConfig {
    /// NIS above this value marks an accepted update as statistically suspect.
    pub warning_nis: f64,
    /// Position variance below this floor is treated as covariance collapse.
    pub minimum_credible_variance_m2: f64,
    /// Position variance above this ceiling is treated as unbounded uncertainty.
    pub maximum_credible_variance_m2: f64,
    /// Number of accepted/rejected decisions retained in the rolling window.
    pub rejection_window: usize,
    /// Rejection fraction that makes the estimator unreliable.
    pub maximum_rejection_fraction: f64,
    /// Consecutive inconsistent samples that make the estimator unreliable.
    pub maximum_consecutive_inconsistent: u32,
}

impl Default for NavigationConsistencyConfig {
    fn default() -> Self {
        Self {
            warning_nis: 7.815, // approximate chi-square 95% threshold for 3 DOF
            minimum_credible_variance_m2: 1.0e-7,
            maximum_credible_variance_m2: 100.0,
            rejection_window: 20,
            maximum_rejection_fraction: 0.5,
            maximum_consecutive_inconsistent: 5,
        }
    }
}

impl NavigationConsistencyConfig {
    pub fn validate(&self) -> Result<(), NavigationConsistencyError> {
        if !self.warning_nis.is_finite()
            || self.warning_nis <= 0.0
            || !self.minimum_credible_variance_m2.is_finite()
            || self.minimum_credible_variance_m2 <= 0.0
            || !self.maximum_credible_variance_m2.is_finite()
            || self.maximum_credible_variance_m2 <= self.minimum_credible_variance_m2
            || self.rejection_window == 0
            || !self.maximum_rejection_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_rejection_fraction)
            || self.maximum_consecutive_inconsistent == 0
        {
            return Err(NavigationConsistencyError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationConsistencyError {
    InvalidConfiguration,
    NonFiniteSample,
    NegativeVariance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NavigationConsistencyState {
    Healthy,
    Suspect,
    Unreliable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationConsistencySample {
    pub source: NavigationSource,
    /// None for initialization or malformed samples without a meaningful innovation.
    pub normalized_innovation_sq: Option<f64>,
    pub estimate_variance_m2: f64,
    pub measurement_variance_m2: f64,
    pub accepted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationConsistencyEvidence {
    pub state: NavigationConsistencyState,
    pub total_samples: u64,
    pub accepted_samples: u64,
    pub rejected_samples: u64,
    pub recent_rejection_fraction: f64,
    pub consecutive_inconsistent: u32,
    pub maximum_observed_nis: f64,
    pub covariance_collapse_count: u64,
    pub covariance_inflation_count: u64,
    pub last_source: Option<NavigationSource>,
}

#[derive(Debug, Clone)]
pub struct NavigationConsistencyMonitor {
    config: NavigationConsistencyConfig,
    recent_acceptance: VecDeque<bool>,
    evidence: NavigationConsistencyEvidence,
}

impl NavigationConsistencyMonitor {
    pub fn new(config: NavigationConsistencyConfig) -> Result<Self, NavigationConsistencyError> {
        config.validate()?;
        Ok(Self {
            config,
            recent_acceptance: VecDeque::with_capacity(config.rejection_window),
            evidence: NavigationConsistencyEvidence {
                state: NavigationConsistencyState::Healthy,
                total_samples: 0,
                accepted_samples: 0,
                rejected_samples: 0,
                recent_rejection_fraction: 0.0,
                consecutive_inconsistent: 0,
                maximum_observed_nis: 0.0,
                covariance_collapse_count: 0,
                covariance_inflation_count: 0,
                last_source: None,
            },
        })
    }

    pub fn evidence(&self) -> NavigationConsistencyEvidence {
        self.evidence
    }

    pub fn is_usable(&self) -> bool {
        self.evidence.state != NavigationConsistencyState::Unreliable
    }

    pub fn observe(
        &mut self,
        sample: NavigationConsistencySample,
    ) -> Result<NavigationConsistencyEvidence, NavigationConsistencyError> {
        self.config.validate()?;
        if !sample.estimate_variance_m2.is_finite()
            || !sample.measurement_variance_m2.is_finite()
            || sample
                .normalized_innovation_sq
                .is_some_and(|nis| !nis.is_finite())
        {
            return Err(NavigationConsistencyError::NonFiniteSample);
        }
        if sample.estimate_variance_m2 < 0.0 || sample.measurement_variance_m2 < 0.0 {
            return Err(NavigationConsistencyError::NegativeVariance);
        }

        self.evidence.total_samples = self.evidence.total_samples.saturating_add(1);
        self.evidence.last_source = Some(sample.source);
        if sample.accepted {
            self.evidence.accepted_samples = self.evidence.accepted_samples.saturating_add(1);
        } else {
            self.evidence.rejected_samples = self.evidence.rejected_samples.saturating_add(1);
        }
        self.recent_acceptance.push_back(sample.accepted);
        while self.recent_acceptance.len() > self.config.rejection_window {
            self.recent_acceptance.pop_front();
        }
        let rejected_recent = self
            .recent_acceptance
            .iter()
            .filter(|accepted| !**accepted)
            .count();
        self.evidence.recent_rejection_fraction =
            rejected_recent as f64 / self.recent_acceptance.len().max(1) as f64;

        let covariance_collapsed =
            sample.estimate_variance_m2 < self.config.minimum_credible_variance_m2;
        let covariance_inflated =
            sample.estimate_variance_m2 > self.config.maximum_credible_variance_m2;
        if covariance_collapsed {
            self.evidence.covariance_collapse_count =
                self.evidence.covariance_collapse_count.saturating_add(1);
        }
        if covariance_inflated {
            self.evidence.covariance_inflation_count =
                self.evidence.covariance_inflation_count.saturating_add(1);
        }
        if let Some(nis) = sample.normalized_innovation_sq {
            self.evidence.maximum_observed_nis = self.evidence.maximum_observed_nis.max(nis);
        }

        let nis_suspect = sample
            .normalized_innovation_sq
            .is_some_and(|nis| nis > self.config.warning_nis);
        let inconsistent =
            !sample.accepted || covariance_collapsed || covariance_inflated || nis_suspect;
        if inconsistent {
            self.evidence.consecutive_inconsistent =
                self.evidence.consecutive_inconsistent.saturating_add(1);
        } else {
            self.evidence.consecutive_inconsistent = 0;
        }

        self.evidence.state = if covariance_collapsed
            || covariance_inflated
            || self.evidence.consecutive_inconsistent
                >= self.config.maximum_consecutive_inconsistent
            || (self.recent_acceptance.len() == self.config.rejection_window
                && self.evidence.recent_rejection_fraction > self.config.maximum_rejection_fraction)
        {
            NavigationConsistencyState::Unreliable
        } else if nis_suspect
            || !sample.accepted
            || self.evidence.recent_rejection_fraction
                > self.config.maximum_rejection_fraction * 0.5
        {
            NavigationConsistencyState::Suspect
        } else {
            NavigationConsistencyState::Healthy
        };

        Ok(self.evidence)
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.config)
            .expect("an already validated navigation consistency config must remain valid");
    }
}

impl Default for NavigationConsistencyMonitor {
    fn default() -> Self {
        Self::new(NavigationConsistencyConfig::default())
            .expect("default navigation consistency config must remain valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(accepted: bool, nis: f64) -> NavigationConsistencySample {
        NavigationConsistencySample {
            source: NavigationSource::Gnss,
            normalized_innovation_sq: Some(nis),
            estimate_variance_m2: 1.0,
            measurement_variance_m2: 1.0,
            accepted,
        }
    }

    #[test]
    fn healthy_updates_remain_usable() {
        let mut monitor = NavigationConsistencyMonitor::default();
        for _ in 0..20 {
            monitor.observe(sample(true, 1.0)).unwrap();
        }
        assert_eq!(
            monitor.evidence().state,
            NavigationConsistencyState::Healthy
        );
        assert!(monitor.is_usable());
    }

    #[test]
    fn repeated_rejections_make_estimator_unreliable() {
        let mut monitor = NavigationConsistencyMonitor::default();
        for _ in 0..5 {
            monitor.observe(sample(false, 30.0)).unwrap();
        }
        assert_eq!(
            monitor.evidence().state,
            NavigationConsistencyState::Unreliable
        );
        assert!(!monitor.is_usable());
    }

    #[test]
    fn covariance_collapse_fails_immediately() {
        let mut monitor = NavigationConsistencyMonitor::default();
        let mut collapsed = sample(true, 0.1);
        collapsed.estimate_variance_m2 = 1.0e-12;
        let evidence = monitor.observe(collapsed).unwrap();
        assert_eq!(evidence.state, NavigationConsistencyState::Unreliable);
        assert_eq!(evidence.covariance_collapse_count, 1);
    }

    #[test]
    fn rolling_rejection_fraction_is_bounded() {
        let config = NavigationConsistencyConfig {
            rejection_window: 4,
            maximum_consecutive_inconsistent: 10,
            maximum_rejection_fraction: 0.5,
            ..NavigationConsistencyConfig::default()
        };
        let mut monitor = NavigationConsistencyMonitor::new(config).unwrap();
        for accepted in [true, false, false, false] {
            monitor
                .observe(sample(accepted, if accepted { 1.0 } else { 20.0 }))
                .unwrap();
        }
        assert_eq!(monitor.evidence().recent_rejection_fraction, 0.75);
        assert_eq!(
            monitor.evidence().state,
            NavigationConsistencyState::Unreliable
        );
    }
}
