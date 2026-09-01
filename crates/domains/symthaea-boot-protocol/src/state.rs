// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! A small reducer for consumers that want a current presentation snapshot.
//! It does not invent authority: callers decide which observer events are
//! authoritative and may replace the state with a validated snapshot at any time.

use crate::{
    BootDomain, BootEvent, BootHealth, BootPhase, BootSnapshot, Criticality, DomainSnapshot,
    DomainState, PROTOCOL_VERSION,
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
    pub fn apply(&mut self, event: &BootEvent) -> bool {
        if event.sequence() <= self.sequence {
            return false;
        }
        self.sequence = event.sequence();

        match event {
            BootEvent::PhaseEntered {
                elapsed_ms, phase, ..
            } => {
                self.elapsed_ms = *elapsed_ms;
                self.phase = *phase;
            }
            BootEvent::DomainStarting {
                elapsed_ms, domain, ..
            } => self.set_domain(*domain, DomainState::Starting, Some(*elapsed_ms)),
            BootEvent::DomainReady {
                elapsed_ms, domain, ..
            } => self.set_domain(*domain, DomainState::Ready, Some(*elapsed_ms)),
            BootEvent::DomainDelayed {
                elapsed_ms, domain, ..
            } => {
                self.elapsed_ms = *elapsed_ms;
                self.set_domain(*domain, DomainState::Delayed, Some(*elapsed_ms));
                self.raise_health(BootHealth::Delayed);
            }
            BootEvent::DomainDegraded {
                elapsed_ms,
                domain,
                criticality,
                ..
            } => {
                self.elapsed_ms = *elapsed_ms;
                self.set_domain(*domain, DomainState::Degraded, Some(*elapsed_ms));
                self.raise_health(match criticality {
                    Criticality::Critical => BootHealth::Failed,
                    _ => BootHealth::Degraded,
                });
            }
            BootEvent::DomainFailed {
                elapsed_ms, domain, ..
            } => {
                self.elapsed_ms = *elapsed_ms;
                self.set_domain(*domain, DomainState::Failed, Some(*elapsed_ms));
                self.raise_health(BootHealth::Failed);
            }
            BootEvent::DomainRecovered {
                elapsed_ms, domain, ..
            } => {
                self.elapsed_ms = *elapsed_ms;
                self.set_domain(*domain, DomainState::Ready, Some(*elapsed_ms));
                // Deliberately do not lower health here. A recovery event says a
                // domain recovered; only authoritative snapshot/BootReady data may
                // declare the whole boot healthy again.
            }
            BootEvent::BootReady {
                elapsed_ms, health, ..
            } => {
                self.elapsed_ms = *elapsed_ms;
                self.phase = BootPhase::Ready;
                self.health = *health;
            }
        }
        true
    }

    pub fn replace(&mut self, snapshot: BootSnapshot) {
        self.sequence = snapshot.sequence;
        self.elapsed_ms = snapshot.elapsed_ms;
        self.phase = snapshot.phase;
        self.health = snapshot.health;
        self.domains.clear();
        for domain in snapshot.domains {
            self.domains.insert(domain_key(domain.domain), domain);
        }
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

        assert!(reducer.apply(&first));
        assert!(!reducer.apply(&stale));
        assert_eq!(reducer.snapshot().health, BootHealth::Unknown);
    }

    #[test]
    fn recovery_does_not_silently_mark_boot_healthy() {
        let mut reducer = BootStateReducer::default();
        reducer.apply(&BootEvent::DomainFailed {
            sequence: 1,
            elapsed_ms: 10,
            domain: BootDomain::Network,
            criticality: Criticality::NonCritical,
            detail: None,
        });
        reducer.apply(&BootEvent::DomainRecovered {
            sequence: 2,
            elapsed_ms: 20,
            domain: BootDomain::Network,
        });

        assert_eq!(reducer.snapshot().health, BootHealth::Failed);
    }

    #[test]
    fn boot_ready_can_authoritatively_resolve_health() {
        let mut reducer = BootStateReducer::default();
        reducer.apply(&BootEvent::DomainDelayed {
            sequence: 1,
            elapsed_ms: 200,
            domain: BootDomain::Network,
        });
        reducer.apply(&BootEvent::BootReady {
            sequence: 2,
            elapsed_ms: 300,
            health: BootHealth::Normal,
        });

        let snapshot = reducer.snapshot();
        assert_eq!(snapshot.phase, BootPhase::Ready);
        assert_eq!(snapshot.health, BootHealth::Normal);
    }
}
