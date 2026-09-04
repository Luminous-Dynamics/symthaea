// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Policy/configuration for the non-authoritative Symthaea boot observer.
//!
//! The observer reads structured systemd state and translates it into the
//! intentionally lossy `symthaea-boot-protocol`. It does not parse journal text
//! and it never participates in systemd's dependency/health authority.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use symthaea_boot_protocol::{BootDomain, BootHealth, BootPhase, Criticality, DomainState};

pub const DEFAULT_OUTPUT_SOCKET: &str = "/run/symthaea/boot-events.sock";
pub const DEFAULT_STATE_PATH: &str = "/run/symthaea-boot/state-v1.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObserverConfig {
    pub output_socket: PathBuf,
    pub state_path: PathBuf,
    pub watched_units: Vec<WatchedUnit>,
}

impl ObserverConfig {
    pub fn builtin() -> Self {
        Self {
            output_socket: PathBuf::from(DEFAULT_OUTPUT_SOCKET),
            state_path: PathBuf::from(DEFAULT_STATE_PATH),
            watched_units: vec![
                WatchedUnit::new(
                    "local-fs-pre.target",
                    BootDomain::Storage,
                    Some(BootPhase::Storage),
                    Criticality::Critical,
                    false,
                ),
                WatchedUnit::new(
                    "local-fs.target",
                    BootDomain::Filesystems,
                    Some(BootPhase::Filesystems),
                    Criticality::Critical,
                    false,
                ),
                WatchedUnit::new(
                    "network.target",
                    BootDomain::Network,
                    Some(BootPhase::Network),
                    Criticality::NonCritical,
                    false,
                ),
                WatchedUnit::new(
                    "multi-user.target",
                    BootDomain::Services,
                    Some(BootPhase::Services),
                    Criticality::Critical,
                    false,
                ),
                WatchedUnit::new(
                    "display-manager.service",
                    BootDomain::Graphics,
                    Some(BootPhase::Graphics),
                    Criticality::Critical,
                    false,
                ),
                // Keep graphical.target separate from the display-manager
                // domain. One unit's recovery must never visually erase the
                // other's failure. The target represents session readiness.
                WatchedUnit::new(
                    "graphical.target",
                    BootDomain::Session,
                    Some(BootPhase::Ready),
                    Criticality::Critical,
                    true,
                ),
            ],
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_absolute(&self.output_socket, "output_socket")?;
        validate_absolute(&self.state_path, "state_path")?;

        if self.watched_units.is_empty() {
            return Err(ConfigError::NoWatchedUnits);
        }

        let mut names = BTreeSet::new();
        // The live reducer stores one aggregate state per BootDomain, not one
        // state per watched unit. Until the observer carries per-unit state, two
        // units sharing a domain are ambiguous: recovery of one could otherwise
        // erase a still-failed sibling. Reject that configuration fail-closed.
        let mut domains = [false; BootDomain::COUNT];
        let mut boot_ready_count = 0usize;
        for watched in &self.watched_units {
            watched.validate()?;
            if !names.insert(watched.unit.as_str()) {
                return Err(ConfigError::DuplicateUnit(watched.unit.clone()));
            }
            let domain_index = watched.domain.index();
            if domains[domain_index] {
                return Err(ConfigError::DuplicateDomain(watched.domain));
            }
            domains[domain_index] = true;
            if watched.boot_ready {
                boot_ready_count += 1;
                if watched.phase != Some(BootPhase::Ready) {
                    return Err(ConfigError::BootReadyMustEnterReady(watched.unit.clone()));
                }
            }
        }

        if boot_ready_count > 1 {
            return Err(ConfigError::MultipleBootReadyUnits(boot_ready_count));
        }
        Ok(())
    }

    pub fn find(&self, unit: &str) -> Option<&WatchedUnit> {
        self.watched_units.iter().find(|watched| watched.unit == unit)
    }
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self::builtin()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatchedUnit {
    pub unit: String,
    pub domain: BootDomain,
    pub phase: Option<BootPhase>,
    pub criticality: Criticality,
    #[serde(default)]
    pub boot_ready: bool,
}

impl WatchedUnit {
    pub fn new(
        unit: impl Into<String>,
        domain: BootDomain,
        phase: Option<BootPhase>,
        criticality: Criticality,
        boot_ready: bool,
    ) -> Self {
        Self {
            unit: unit.into(),
            domain,
            phase,
            criticality,
            boot_ready,
        }
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.unit.is_empty()
            || self.unit.len() > 255
            || self.unit.chars().any(|c| c.is_control() || c.is_whitespace())
            || !self.unit.contains('.')
        {
            return Err(ConfigError::InvalidUnit(self.unit.clone()));
        }
        Ok(())
    }
}

pub fn domain_state_from_active_state(active_state: &str) -> Option<DomainState> {
    match active_state {
        "active" | "reloading" | "refreshing" => Some(DomainState::Ready),
        "activating" => Some(DomainState::Starting),
        "failed" => Some(DomainState::Failed),
        "inactive" | "deactivating" | "maintenance" => Some(DomainState::Pending),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobOutcome {
    QueryCurrentState,
    Failed,
    Ignore,
}

pub fn classify_job_result(result: &str) -> JobOutcome {
    match result {
        "done" => JobOutcome::QueryCurrentState,
        "failed" | "timeout" | "dependency" => JobOutcome::Failed,
        "canceled" | "skipped" => JobOutcome::Ignore,
        _ => JobOutcome::Ignore,
    }
}

/// Entering the protocol's Ready phase is a readiness fact, not a health proof.
///
/// In particular, absent health evidence remains `Unknown`; presentation must
/// never infer `Normal` merely because the boot-ready unit became active.
pub const fn health_at_boot_ready(current: BootHealth) -> BootHealth {
    current
}

fn validate_absolute(path: &Path, field: &'static str) -> Result<(), ConfigError> {
    if !path.is_absolute() {
        return Err(ConfigError::PathNotAbsolute(field));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    PathNotAbsolute(&'static str),
    NoWatchedUnits,
    InvalidUnit(String),
    DuplicateUnit(String),
    DuplicateDomain(BootDomain),
    MultipleBootReadyUnits(usize),
    BootReadyMustEnterReady(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PathNotAbsolute(field) => write!(f, "{field} must be an absolute path"),
            Self::NoWatchedUnits => write!(f, "at least one watched unit is required"),
            Self::InvalidUnit(unit) => write!(f, "invalid systemd unit name: {unit:?}"),
            Self::DuplicateUnit(unit) => write!(f, "systemd unit is watched more than once: {unit}"),
            Self::DuplicateDomain(domain) => {
                write!(f, "boot domain {domain:?} is watched by more than one unit")
            }
            Self::MultipleBootReadyUnits(n) => {
                write!(f, "at most one boot-ready unit is allowed, found {n}")
            }
            Self::BootReadyMustEnterReady(unit) => {
                write!(f, "boot-ready unit {unit} must enter BootPhase::Ready")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_configuration_is_valid() {
        ObserverConfig::builtin().validate().unwrap();
    }

    #[test]
    fn builtin_boot_ready_uses_session_domain() {
        let config = ObserverConfig::builtin();
        let ready = config
            .watched_units
            .iter()
            .find(|unit| unit.boot_ready)
            .expect("builtin config must have a boot-ready unit");
        assert_eq!(ready.unit, "graphical.target");
        assert_eq!(ready.domain, BootDomain::Session);
        assert_eq!(ready.phase, Some(BootPhase::Ready));
    }

    #[test]
    fn duplicate_units_are_rejected() {
        let mut config = ObserverConfig::builtin();
        config.watched_units.push(config.watched_units[0].clone());
        assert!(matches!(config.validate(), Err(ConfigError::DuplicateUnit(_))));
    }

    #[test]
    fn duplicate_domains_are_rejected_until_per_unit_aggregation_exists() {
        let mut config = ObserverConfig::builtin();
        config.watched_units.push(WatchedUnit::new(
            "network-online.target",
            BootDomain::Network,
            Some(BootPhase::Network),
            Criticality::NonCritical,
            false,
        ));
        assert!(matches!(
            config.validate(),
            Err(ConfigError::DuplicateDomain(BootDomain::Network))
        ));
    }

    #[test]
    fn boot_ready_does_not_upgrade_unknown_health() {
        assert_eq!(health_at_boot_ready(BootHealth::Unknown), BootHealth::Unknown);
        assert_eq!(health_at_boot_ready(BootHealth::Normal), BootHealth::Normal);
        assert_eq!(health_at_boot_ready(BootHealth::Failed), BootHealth::Failed);
        assert_eq!(health_at_boot_ready(BootHealth::Delayed), BootHealth::Delayed);
        assert_eq!(health_at_boot_ready(BootHealth::Degraded), BootHealth::Degraded);
    }

    #[test]
    fn systemd_states_are_intentionally_lossy() {
        assert_eq!(domain_state_from_active_state("active"), Some(DomainState::Ready));
        assert_eq!(domain_state_from_active_state("failed"), Some(DomainState::Failed));
        assert_eq!(domain_state_from_active_state("unknown-future-state"), None);
    }

    #[test]
    fn canceled_jobs_do_not_claim_failure() {
        assert_eq!(classify_job_result("canceled"), JobOutcome::Ignore);
        assert_eq!(classify_job_result("failed"), JobOutcome::Failed);
        assert_eq!(classify_job_result("done"), JobOutcome::QueryCurrentState);
    }
}
