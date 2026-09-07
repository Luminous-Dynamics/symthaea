// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stateful fail-closed admission for multiscale operating envelopes.
//!
//! `symthaea-operating-envelope` defines a pure, inspectable authority value. This
//! crate adds runtime lease admission semantics: only the immediate parent may
//! supply an executable lease, generations must advance monotonically, and two
//! different envelopes may not claim the same generation for one issuer/subject
//! authority lineage. Generation watermarks survive lease expiry.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use symthaea_operating_envelope::OperatingEnvelope;
use symthaea_resource_hierarchy::ResourceHierarchy;
use thiserror::Error;

/// Result of attempting to admit an operating lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseAdmission {
    Accepted,
    Idempotent,
}

/// Last accepted runtime authority for one direct child.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedLease {
    pub envelope: OperatingEnvelope,
    pub accepted_at: DateTime<Utc>,
}

/// Stateful runtime admission registry.
///
/// Active leases are keyed by subject because each structural child has one current
/// parent. Replay watermarks are keyed by `(issuer, subject)` so expiry never erases
/// monotone-generation evidence, while an explicit hierarchy re-parenting can begin
/// a distinct authority lineage.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OperatingAuthorityRegistry {
    active_by_subject: BTreeMap<String, AcceptedLease>,
    watermark_by_lineage: BTreeMap<(String, String), OperatingEnvelope>,
}

impl OperatingAuthorityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit a runtime envelope if and only if it is current, comes from the
    /// subject's immediate parent, and advances authority monotonically.
    pub fn admit(
        &mut self,
        hierarchy: &ResourceHierarchy,
        envelope: OperatingEnvelope,
        now: DateTime<Utc>,
    ) -> Result<LeaseAdmission, AuthorityError> {
        if envelope.id.is_empty() {
            return Err(AuthorityError::EmptyEnvelopeId);
        }
        envelope
            .validate_structure()
            .map_err(|error| AuthorityError::InvalidEnvelope(error.to_string()))?;
        envelope
            .validate_authority(hierarchy)
            .map_err(|error| AuthorityError::InvalidEnvelope(error.to_string()))?;

        let parent = hierarchy
            .parent(&envelope.subject_node_id)
            .ok_or_else(|| AuthorityError::SubjectHasNoParent(envelope.subject_node_id.clone()))?;
        if parent != envelope.issuer_node_id {
            return Err(AuthorityError::IssuerNotImmediateParent {
                subject: envelope.subject_node_id.clone(),
                expected: parent.to_owned(),
                actual: envelope.issuer_node_id.clone(),
            });
        }

        if now < envelope.valid_from {
            return Err(AuthorityError::NotYetValid {
                envelope_id: envelope.id.clone(),
            });
        }
        if now >= envelope.valid_until {
            return Err(AuthorityError::Expired {
                envelope_id: envelope.id.clone(),
            });
        }

        let lineage = (
            envelope.issuer_node_id.clone(),
            envelope.subject_node_id.clone(),
        );
        if let Some(watermark) = self.watermark_by_lineage.get(&lineage) {
            if envelope.generation < watermark.generation {
                return Err(AuthorityError::StaleGeneration {
                    issuer: envelope.issuer_node_id.clone(),
                    subject: envelope.subject_node_id.clone(),
                    current: watermark.generation,
                    proposed: envelope.generation,
                });
            }
            if envelope.generation == watermark.generation {
                if envelope == *watermark {
                    return Ok(LeaseAdmission::Idempotent);
                }
                return Err(AuthorityError::GenerationEquivocation {
                    issuer: envelope.issuer_node_id.clone(),
                    subject: envelope.subject_node_id.clone(),
                    generation: envelope.generation,
                    current_envelope_id: watermark.id.clone(),
                    proposed_envelope_id: envelope.id.clone(),
                });
            }
        }

        let subject = envelope.subject_node_id.clone();
        self.watermark_by_lineage
            .insert(lineage, envelope.clone());
        self.active_by_subject.insert(
            subject,
            AcceptedLease {
                envelope,
                accepted_at: now,
            },
        );
        Ok(LeaseAdmission::Accepted)
    }

    /// Return the currently usable lease for `subject` at `now`.
    ///
    /// The current hierarchy is rechecked so a lease from a former parent cannot
    /// remain executable after explicit re-parenting.
    pub fn active<'a>(
        &'a self,
        hierarchy: &ResourceHierarchy,
        subject: &str,
        now: DateTime<Utc>,
    ) -> Option<&'a OperatingEnvelope> {
        self.active_by_subject.get(subject).and_then(|lease| {
            let current_parent = hierarchy.parent(subject)?;
            (current_parent == lease.envelope.issuer_node_id
                && now >= lease.envelope.valid_from
                && now < lease.envelope.valid_until)
                .then_some(&lease.envelope)
        })
    }

    /// Highest generation admitted for one issuer/subject authority lineage.
    /// This survives active-lease expiry and purge.
    pub fn latest_generation(&self, issuer: &str, subject: &str) -> Option<u64> {
        self.watermark_by_lineage
            .get(&(issuer.to_owned(), subject.to_owned()))
            .map(|envelope| envelope.generation)
    }

    /// Remove expired executable leases while retaining replay watermarks.
    pub fn purge_expired(&mut self, now: DateTime<Utc>) -> Vec<String> {
        let expired: Vec<String> = self
            .active_by_subject
            .iter()
            .filter(|(_, lease)| now >= lease.envelope.valid_until)
            .map(|(subject, _)| subject.clone())
            .collect();
        for subject in &expired {
            self.active_by_subject.remove(subject);
        }
        expired
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AuthorityError {
    #[error("operating envelope id must not be empty")]
    EmptyEnvelopeId,
    #[error("invalid operating envelope: {0}")]
    InvalidEnvelope(String),
    #[error("subject {0} has no structural parent")]
    SubjectHasNoParent(String),
    #[error("issuer {actual} is not immediate parent {expected} of subject {subject}")]
    IssuerNotImmediateParent {
        subject: String,
        expected: String,
        actual: String,
    },
    #[error("operating envelope {envelope_id} is not yet valid")]
    NotYetValid { envelope_id: String },
    #[error("operating envelope {envelope_id} is expired")]
    Expired { envelope_id: String },
    #[error(
        "stale generation {proposed} for authority lineage {issuer}->{subject}; current generation is {current}"
    )]
    StaleGeneration {
        issuer: String,
        subject: String,
        current: u64,
        proposed: u64,
    },
    #[error(
        "generation {generation} equivocation for authority lineage {issuer}->{subject}: current {current_envelope_id}, proposed {proposed_envelope_id}"
    )]
    GenerationEquivocation {
        issuer: String,
        subject: String,
        generation: u64,
        current_envelope_id: String,
        proposed_envelope_id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use std::collections::BTreeSet;
    use symthaea_operating_envelope::OperatingMode;
    use symthaea_resource_hierarchy::{NodeScale, ResourceNode};
    use symthaea_resource_model::ResourceEnvelope;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn hierarchy() -> ResourceHierarchy {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "region",
                "Region",
                NodeScale::Region,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "region",
                ResourceNode::new(
                    "site",
                    "Site",
                    NodeScale::Site,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack",
                    "Rack",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
    }

    fn envelope(
        id: &str,
        issuer: &str,
        subject: &str,
        generation: u64,
    ) -> OperatingEnvelope {
        let mut allowed_modes = BTreeSet::new();
        allowed_modes.insert(OperatingMode::Normal);
        OperatingEnvelope {
            id: id.into(),
            issuer_node_id: issuer.into(),
            subject_node_id: subject.into(),
            generation,
            valid_from: t0(),
            valid_until: t0() + Duration::hours(1),
            allowed_modes,
            resource_constraints: vec![],
            critical_service_floor: 0.8,
            max_shed_fraction: 0.2,
        }
    }

    #[test]
    fn immediate_parent_can_admit_current_lease() {
        let hierarchy = hierarchy();
        let mut registry = OperatingAuthorityRegistry::new();
        assert_eq!(
            registry
                .admit(
                    &hierarchy,
                    envelope("e1", "site", "rack", 1),
                    t0() + Duration::minutes(1),
                )
                .unwrap(),
            LeaseAdmission::Accepted
        );
        assert_eq!(registry.latest_generation("site", "rack"), Some(1));
    }

    #[test]
    fn ancestor_cannot_bypass_immediate_parent_at_runtime() {
        let hierarchy = hierarchy();
        let mut registry = OperatingAuthorityRegistry::new();
        let result = registry.admit(
            &hierarchy,
            envelope("e1", "region", "rack", 1),
            t0() + Duration::minutes(1),
        );
        assert!(matches!(
            result,
            Err(AuthorityError::IssuerNotImmediateParent { expected, actual, .. })
                if expected == "site" && actual == "region"
        ));
    }

    #[test]
    fn higher_generation_replaces_current_lease() {
        let hierarchy = hierarchy();
        let mut registry = OperatingAuthorityRegistry::new();
        registry
            .admit(
                &hierarchy,
                envelope("e1", "site", "rack", 1),
                t0() + Duration::minutes(1),
            )
            .unwrap();
        registry
            .admit(
                &hierarchy,
                envelope("e2", "site", "rack", 3),
                t0() + Duration::minutes(2),
            )
            .unwrap();
        assert_eq!(registry.latest_generation("site", "rack"), Some(3));
        assert_eq!(
            registry
                .active(&hierarchy, "rack", t0() + Duration::minutes(3))
                .unwrap()
                .id,
            "e2"
        );
    }

    #[test]
    fn stale_generation_is_rejected_even_while_time_valid() {
        let hierarchy = hierarchy();
        let mut registry = OperatingAuthorityRegistry::new();
        registry
            .admit(
                &hierarchy,
                envelope("e3", "site", "rack", 3),
                t0() + Duration::minutes(1),
            )
            .unwrap();
        let result = registry.admit(
            &hierarchy,
            envelope("e2", "site", "rack", 2),
            t0() + Duration::minutes(2),
        );
        assert!(matches!(
            result,
            Err(AuthorityError::StaleGeneration {
                current: 3,
                proposed: 2,
                ..
            })
        ));
    }

    #[test]
    fn same_generation_different_envelope_is_equivocation() {
        let hierarchy = hierarchy();
        let mut registry = OperatingAuthorityRegistry::new();
        registry
            .admit(
                &hierarchy,
                envelope("e1", "site", "rack", 7),
                t0() + Duration::minutes(1),
            )
            .unwrap();
        let result = registry.admit(
            &hierarchy,
            envelope("e2", "site", "rack", 7),
            t0() + Duration::minutes(2),
        );
        assert!(matches!(
            result,
            Err(AuthorityError::GenerationEquivocation { generation: 7, .. })
        ));
    }

    #[test]
    fn exact_replay_is_idempotent() {
        let hierarchy = hierarchy();
        let mut registry = OperatingAuthorityRegistry::new();
        let lease = envelope("e1", "site", "rack", 1);
        registry
            .admit(
                &hierarchy,
                lease.clone(),
                t0() + Duration::minutes(1),
            )
            .unwrap();
        assert_eq!(
            registry
                .admit(&hierarchy, lease, t0() + Duration::minutes(2))
                .unwrap(),
            LeaseAdmission::Idempotent
        );
    }

    #[test]
    fn expired_or_future_lease_is_not_admitted() {
        let hierarchy = hierarchy();
        let mut registry = OperatingAuthorityRegistry::new();
        assert!(matches!(
            registry.admit(
                &hierarchy,
                envelope("future", "site", "rack", 1),
                t0() - Duration::minutes(1),
            ),
            Err(AuthorityError::NotYetValid { .. })
        ));
        assert!(matches!(
            registry.admit(
                &hierarchy,
                envelope("expired", "site", "rack", 1),
                t0() + Duration::hours(1),
            ),
            Err(AuthorityError::Expired { .. })
        ));
    }

    #[test]
    fn expiry_purge_retains_generation_watermark() {
        let hierarchy = hierarchy();
        let mut registry = OperatingAuthorityRegistry::new();
        registry
            .admit(
                &hierarchy,
                envelope("e1", "site", "rack", 3),
                t0() + Duration::minutes(1),
            )
            .unwrap();
        assert_eq!(
            registry.purge_expired(t0() + Duration::hours(1)),
            vec!["rack".to_string()]
        );
        assert!(registry
            .active(&hierarchy, "rack", t0() + Duration::hours(1))
            .is_none());
        assert_eq!(registry.latest_generation("site", "rack"), Some(3));

        let mut replay = envelope("replay", "site", "rack", 2);
        replay.valid_from = t0() + Duration::hours(1);
        replay.valid_until = t0() + Duration::hours(2);
        assert!(matches!(
            registry.admit(&hierarchy, replay, t0() + Duration::hours(1)),
            Err(AuthorityError::StaleGeneration { current: 3, proposed: 2, .. })
        ));
    }
}
