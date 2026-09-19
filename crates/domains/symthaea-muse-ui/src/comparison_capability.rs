// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability truth for semantic A/B comparison.
//!
//! Auditionability and semantically synchronized comparison are deliberately
//! different capabilities. A legacy keeper can still be heard through the shared
//! transport, but if no authoritative musical timeline survives for it the UI
//! must not silently substitute normalized duration or equal wall-clock seconds.

use std::fmt;

use crate::comparison::ComparisonSide;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonTimelineEvidenceKind {
    /// A current in-session composition bundle fetched from the live piece API.
    LiveCompositionBundle,
    /// A durable keeper-side semantic bundle persisted with the saved artifact.
    PersistedKeeperSemanticBundle,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComparisonTimelineAuthority {
    pub kind: ComparisonTimelineEvidenceKind,
    pub bundle_version: u32,
}

impl ComparisonTimelineAuthority {
    pub fn live(bundle_version: u32) -> Self {
        Self {
            kind: ComparisonTimelineEvidenceKind::LiveCompositionBundle,
            bundle_version,
        }
    }

    pub fn persisted_keeper(bundle_version: u32) -> Self {
        Self {
            kind: ComparisonTimelineEvidenceKind::PersistedKeeperSemanticBundle,
            bundle_version,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonTimelineUnavailableReason {
    /// A historical keeper predates durable semantic timeline sidecars.
    LegacyKeeperWithoutSemanticBundle,
    /// An external/review artifact can be played but has no authoritative
    /// symbolic timeline in this workspace.
    NoAuthoritativeMusicalTimeline,
    /// A timeline was expected but could not be loaded or parsed.
    TimelineBundleUnavailable,
    /// Persisted identity/timeline evidence contradicted another stored fact.
    EvidenceMismatch,
    /// The available bundle version is not supported by this client.
    UnsupportedBundleVersion,
}

impl fmt::Display for ComparisonTimelineUnavailableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyKeeperWithoutSemanticBundle => {
                write!(f, "saved artifact predates persisted semantic timeline evidence")
            }
            Self::NoAuthoritativeMusicalTimeline => {
                write!(f, "artifact has no authoritative musical timeline")
            }
            Self::TimelineBundleUnavailable => {
                write!(f, "semantic timeline bundle is unavailable")
            }
            Self::EvidenceMismatch => {
                write!(f, "semantic timeline evidence conflicts with persisted identity")
            }
            Self::UnsupportedBundleVersion => {
                write!(f, "semantic timeline bundle version is unsupported")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonTimelineCapability {
    Available(ComparisonTimelineAuthority),
    Unavailable(ComparisonTimelineUnavailableReason),
}

impl ComparisonTimelineCapability {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComparisonCapabilitySet {
    pub a: ComparisonTimelineCapability,
    pub b: ComparisonTimelineCapability,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticSyncPermit {
    pub a: ComparisonTimelineAuthority,
    pub b: ComparisonTimelineAuthority,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticSyncUnavailable {
    pub side: ComparisonSide,
    pub reason: ComparisonTimelineUnavailableReason,
}

impl fmt::Display for SemanticSyncUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let side = match self.side {
            ComparisonSide::A => "A",
            ComparisonSide::B => "B",
        };
        write!(f, "comparison side {side}: {}", self.reason)
    }
}

impl std::error::Error for SemanticSyncUnavailable {}

impl ComparisonCapabilitySet {
    /// Grant semantically synchronized A/B only when BOTH subjects have an
    /// authoritative timeline. This method says nothing about whether either
    /// subject can be heard; ordinary audition remains a separate capability.
    pub fn semantic_sync_permit(&self) -> Result<SemanticSyncPermit, SemanticSyncUnavailable> {
        let a = match &self.a {
            ComparisonTimelineCapability::Available(authority) => authority.clone(),
            ComparisonTimelineCapability::Unavailable(reason) => {
                return Err(SemanticSyncUnavailable {
                    side: ComparisonSide::A,
                    reason: reason.clone(),
                });
            }
        };
        let b = match &self.b {
            ComparisonTimelineCapability::Available(authority) => authority.clone(),
            ComparisonTimelineCapability::Unavailable(reason) => {
                return Err(SemanticSyncUnavailable {
                    side: ComparisonSide::B,
                    reason: reason.clone(),
                });
            }
        };
        Ok(SemanticSyncPermit { a, b })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_and_persisted_authoritative_timelines_allow_semantic_sync() {
        let capabilities = ComparisonCapabilitySet {
            a: ComparisonTimelineCapability::Available(ComparisonTimelineAuthority::live(2)),
            b: ComparisonTimelineCapability::Available(
                ComparisonTimelineAuthority::persisted_keeper(2),
            ),
        };
        let permit = capabilities.semantic_sync_permit().unwrap();
        assert_eq!(
            permit.a.kind,
            ComparisonTimelineEvidenceKind::LiveCompositionBundle
        );
        assert_eq!(
            permit.b.kind,
            ComparisonTimelineEvidenceKind::PersistedKeeperSemanticBundle
        );
    }

    #[test]
    fn legacy_keeper_remains_auditionable_conceptually_but_blocks_semantic_sync() {
        let capabilities = ComparisonCapabilitySet {
            a: ComparisonTimelineCapability::Available(ComparisonTimelineAuthority::live(2)),
            b: ComparisonTimelineCapability::Unavailable(
                ComparisonTimelineUnavailableReason::LegacyKeeperWithoutSemanticBundle,
            ),
        };
        assert_eq!(
            capabilities.semantic_sync_permit(),
            Err(SemanticSyncUnavailable {
                side: ComparisonSide::B,
                reason: ComparisonTimelineUnavailableReason::LegacyKeeperWithoutSemanticBundle,
            })
        );
    }

    #[test]
    fn first_unavailable_side_is_reported_with_its_exact_reason() {
        let capabilities = ComparisonCapabilitySet {
            a: ComparisonTimelineCapability::Unavailable(
                ComparisonTimelineUnavailableReason::EvidenceMismatch,
            ),
            b: ComparisonTimelineCapability::Unavailable(
                ComparisonTimelineUnavailableReason::TimelineBundleUnavailable,
            ),
        };
        assert_eq!(
            capabilities.semantic_sync_permit(),
            Err(SemanticSyncUnavailable {
                side: ComparisonSide::A,
                reason: ComparisonTimelineUnavailableReason::EvidenceMismatch,
            })
        );
    }

    #[test]
    fn capability_availability_does_not_conflate_with_auditionability() {
        let unavailable = ComparisonTimelineCapability::Unavailable(
            ComparisonTimelineUnavailableReason::NoAuthoritativeMusicalTimeline,
        );
        assert!(!unavailable.is_available());
        // There is deliberately no `can_play` field or method here. Playback
        // capability belongs to PlaybackSource/Presentation, not timeline truth.
    }
}
