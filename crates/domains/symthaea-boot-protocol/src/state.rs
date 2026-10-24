// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! A small reducer for consumers that want a current presentation snapshot.
//! It does not invent authority: callers decide which observer events are
//! authoritative and may replace the state with a validated snapshot at any time.

use crate::{
    BootDomain, BootEvent, BootHealth, BootPhase, BootSnapshot, Criticality, DomainSnapshot,
    DomainState, PROTOCOL_VERSION, ProtocolError,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct BootStateReducer {
    sequence: u64,
    elapsed_ms: u64,
    phase: BootPhase,
    health: BootHealth,
    domains: BTreeMap<u8, DomainSnapshot>,
}

impl Default for BootStateReducer {
    fn default() -> Self {
        Self {
            sequence: 0,
            elapsed_ms: 0,
            phase: BootPhase::Kernel,
            health: BootHealth::Unknown,
            domains: BTreeMap::new(),
        }
    }
}

impl BootStateReducer {
    /// Apply a validated, chronologically newer event.
    ///
    /// Returns `Ok(false)` for a duplicate or older sequence number. Malformed
    /// events and elapsed-time regressions are explicit errors so callers do not
    /// accidentally turn invalid telemetry into presentation truth.
    pub fn try_apply(&mut self, event: &BootEvent) -> Result<bool, ProtocolError> {
        event.validate()?;

        if event.sequence() <= self.sequence {
            return Ok(false);
        }
        if event.elapsed_ms() < self.elapsed_ms {
            return Err(ProtocolError::ElapsedRegressed {
                previous_ms: self.elapsed_ms,
                observed_ms: event.elapsed_ms(),
            });
        }

        self.sequence = event.sequence();
        self.elapsed_ms = event.elapsed_ms();

        match event {
            BootEvent::PhaseEntered { phase, .. } => {
                self.phase = *phase;
            }
            BootEvent::DomainStarting { domain, .. } => {
                self.set_domain(*domain, DomainState::Starting, Some(event.elapsed_ms()));
            }
            BootEvent::DomainReady { domain, .. } => {
                self.set_domain(*domain, DomainState::Ready, Some(event.elapsed_ms()));
            }
            BootEvent::DomainDelayed { domain, .. } => {
                self.set_domain(*domain, DomainState::Delayed, Some(event.elapsed_ms()));
                self.raise_health(BootHealth::Delayed);
            }
            BootEvent::DomainDegraded {
                domain,
                criticality,
                ..
            } => {
                self.set_domain(*domain, DomainState::Degraded, Some(event.elapsed_ms()));
                self.raise_health(match criticality {
                    Criticality::Critical => BootHealth::Failed,
                    _ => BootHealth::Degraded,
                });
            }
            BootEvent::DomainFailed { domain, .. } => {
                self.set_domain(*domain, DomainState::Failed, Some(event.elapsed_ms()));
                self.raise_health(BootHealth::Failed);
            }
            BootEvent::DomainRecovered { domain, .. } => {
                self.set_domain(*domain, DomainState::Ready, Some(event.elapsed_ms()));
                // Deliberately do not lower health here. A recovery event says a
                // domain recovered; only authoritative snapshot/BootReady data may
                // declare the whole boot healthy again.
            }
            BootEvent::BootReady { health, .. } => {
                self.phase = BootPhase::Ready;
                self.health = *health;
            }
        }
        Ok(true)
    }

    /// Backward-compatible convenience wrapper for trusted in-process producers.
    /// Untrusted/wire consumers should use `try_apply` and surface validation
    /// failures to diagnostics rather than silently discarding them.
    pub fn apply(&mut self, event: &BootEvent) -> bool {
        self.try_apply(event).unwrap_or(false)
    }

    /// Replace reducer state with a validated authoritative snapshot.
    ///
    /// Older snapshots are ignored. Equal-sequence snapshots are allowed because
    /// an authoritative snapshot may intentionally refine the reduced state at
    /// the same observation point.
    pub fn try_replace(&mut self, snapshot: BootSnapshot) -> Result<bool, ProtocolError> {
        snapshot.validate()?;

        if snapshot.sequence < self.sequence {
            return Ok(false);
        }
        if snapshot.elapsed_ms < self.elapsed_ms {
            return Err(ProtocolError::ElapsedRegressed {
                previous_ms: self.elapsed_ms,
                observed_ms: snapshot.elapsed_ms,
            });
        }

        self.sequence = snapshot.sequence;
        self.elapsed_ms = snapshot.elapsed_ms;
        self.phase = snapshot.phase;
        self.health = snapshot.health;
        self.domains.clear();
        for domain in snapshot.domains {
            self.domains.insert(domain_key(domain.domain), domain);
        }
        Ok(true)
    }

    /// Convenience wrapper matching the original reducer API style.
    pub fn replace(&mut self, snapshot: BootSnapshot) -> bool {
        self.try_replace(snapshot).unwrap_or(false)
    }

    pub fn snapshot(&self) -> BootSnapshot {
        BootSnapshot {
            protocol_version: PROTOCOL_VERSION,
            sequence: self.sequence,
            elapsed_ms: self.elapsed_ms,
            phase: self.phase,
            health: self.health,
            domains: self.domains.values().cloned().collect(),
        }
    }

    fn set_domain(&mut self, domain: BootDomain, state: DomainState, elapsed_ms: Option<u64>) {
        self.domains.insert(
            domain_key(domain),
            DomainSnapshot {
                domain,
                state,
                elapsed_ms,
            },
        );
    }

    fn raise_health(&mut self, health: BootHealth) {
        if health.severity() > self.health.severity() {
            self.health = health;
        }
    }
}

const fn domain_key(domain: BootDomain) -> u8 {
    match domain {
        BootDomain::Kernel => 0,
        BootDomain::Initrd => 1,
        BootDomain::Storage => 2,
        BootDomain::Filesystems => 3,
        BootDomain::Security => 4,
        BootDomain::Network => 5,
        BootDomain::Services => 6,
        BootDomain::Graphics => 7,
        BootDomain::Session => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn ignores_duplicate_or_out_of_order_events() {
        let mut reducer = BootStateReducer::default();
        let first = BootEvent::DomainReady {
            sequence: 2,
            elapsed_ms: 10,
            domain: BootDomain::Storage,
        };
        let stale = BootEvent::DomainFailed {
            sequence: 1,
            elapsed_ms: 11,
            domain: BootDomain::Storage,
            criticality: Criticality::Critical,
            detail: None,
        };

        assert!(reducer.try_apply(&first).unwrap());
        assert!(!reducer.try_apply(&stale).unwrap());
        assert_eq!(reducer.snapshot().health, BootHealth::Unknown);
    }

    #[test]
    fn rejects_elapsed_time_regression() {
        let mut reducer = BootStateReducer::default();
        reducer
            .try_apply(&BootEvent::DomainReady {
                sequence: 1,
                elapsed_ms: 50,
                domain: BootDomain::Kernel,
            })
            .unwrap();

        assert!(matches!(
            reducer.try_apply(&BootEvent::DomainReady {
                sequence: 2,
                elapsed_ms: 49,
                domain: BootDomain::Initrd,
            }),
            Err(ProtocolError::ElapsedRegressed { .. })
        ));
        assert_eq!(reducer.snapshot().sequence, 1);
    }

    #[test]
    fn recovery_does_not_silently_mark_boot_healthy() {
        let mut reducer = BootStateReducer::default();
        reducer
            .try_apply(&BootEvent::DomainFailed {
                sequence: 1,
                elapsed_ms: 10,
                domain: BootDomain::Network,
                criticality: Criticality::NonCritical,
                detail: None,
            })
            .unwrap();
        reducer
            .try_apply(&BootEvent::DomainRecovered {
                sequence: 2,
                elapsed_ms: 20,
                domain: BootDomain::Network,
            })
            .unwrap();

        assert_eq!(reducer.snapshot().health, BootHealth::Failed);
    }

    #[test]
    fn boot_ready_can_authoritatively_resolve_health() {
        let mut reducer = BootStateReducer::default();
        reducer
            .try_apply(&BootEvent::DomainDelayed {
                sequence: 1,
                elapsed_ms: 200,
                domain: BootDomain::Network,
            })
            .unwrap();
        reducer
            .try_apply(&BootEvent::BootReady {
                sequence: 2,
                elapsed_ms: 300,
                health: BootHealth::Normal,
            })
            .unwrap();

        let snapshot = reducer.snapshot();
        assert_eq!(snapshot.phase, BootPhase::Ready);
        assert_eq!(snapshot.health, BootHealth::Normal);
    }

    #[test]
    fn authoritative_snapshot_must_validate_and_not_go_backwards() {
        let mut reducer = BootStateReducer::default();
        reducer
            .try_apply(&BootEvent::DomainReady {
                sequence: 4,
                elapsed_ms: 40,
                domain: BootDomain::Kernel,
            })
            .unwrap();

        let stale = BootSnapshot::new(3, Duration::from_millis(50), BootPhase::Initrd);
        assert!(!reducer.try_replace(stale).unwrap());

        let regressed = BootSnapshot::new(4, Duration::from_millis(39), BootPhase::Initrd);
        assert!(matches!(
            reducer.try_replace(regressed),
            Err(ProtocolError::ElapsedRegressed { .. })
        ));

        let fresh = BootSnapshot::new(4, Duration::from_millis(40), BootPhase::Initrd);
        assert!(reducer.try_replace(fresh).unwrap());
        assert_eq!(reducer.snapshot().phase, BootPhase::Initrd);
    }
}
