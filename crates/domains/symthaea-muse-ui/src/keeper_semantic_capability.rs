// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy adapter from keeper semantic-evidence fetch outcomes into the typed
//! comparison capability model.
//!
//! The distinction matters to UX and authority: a historical keeper with no
//! sidecar, a temporarily unavailable endpoint, an unsupported future bundle,
//! and contradictory persisted evidence are all different states. None should
//! silently become a normalized-duration comparison fallback.

use crate::comparison_capability::{
    ComparisonTimelineAuthority, ComparisonTimelineCapability,
    ComparisonTimelineUnavailableReason,
};
use crate::keeper_semantic::{
    KeeperSemanticEvidenceError, VerifiedKeeperSemanticBundle,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeeperSemanticCapabilityResolution {
    pub capability: ComparisonTimelineCapability,
    /// Preserves the exact integrity/transport diagnostic for UI inspection or
    /// logging. Historical absence (`Ok(None)`) is not an error and therefore
    /// leaves this field empty.
    pub diagnostic: Option<KeeperSemanticEvidenceError>,
}

/// Convert the semantic-sidecar fetch result into comparison capability truth.
///
/// This function never upgrades a failed verification into availability. It also
/// keeps legacy absence distinct from operational failure and evidence conflict.
pub fn resolve_keeper_semantic_capability(
    result: Result<Option<VerifiedKeeperSemanticBundle>, KeeperSemanticEvidenceError>,
) -> KeeperSemanticCapabilityResolution {
    classify_authority_result(result.map(|bundle| bundle.map(|verified| verified.authority)))
}

fn classify_authority_result(
    result: Result<Option<ComparisonTimelineAuthority>, KeeperSemanticEvidenceError>,
) -> KeeperSemanticCapabilityResolution {
    match result {
        Ok(Some(authority)) => KeeperSemanticCapabilityResolution {
            capability: ComparisonTimelineCapability::Available(authority),
            diagnostic: None,
        },
        Ok(None) => KeeperSemanticCapabilityResolution {
            capability: ComparisonTimelineCapability::Unavailable(
                ComparisonTimelineUnavailableReason::LegacyKeeperWithoutSemanticBundle,
            ),
            diagnostic: None,
        },
        Err(error) => {
            let reason = unavailable_reason(&error);
            KeeperSemanticCapabilityResolution {
                capability: ComparisonTimelineCapability::Unavailable(reason),
                diagnostic: Some(error),
            }
        }
    }
}

fn unavailable_reason(error: &KeeperSemanticEvidenceError) -> ComparisonTimelineUnavailableReason {
    match error {
        KeeperSemanticEvidenceError::UnsupportedSchemaVersion(_)
        | KeeperSemanticEvidenceError::UnsupportedListenBundleVersion(_) => {
            ComparisonTimelineUnavailableReason::UnsupportedBundleVersion
        }
        KeeperSemanticEvidenceError::RequestFailed(_)
        | KeeperSemanticEvidenceError::HttpStatus(_) => {
            ComparisonTimelineUnavailableReason::TimelineBundleUnavailable
        }
        KeeperSemanticEvidenceError::InvalidArtifactKey
        | KeeperSemanticEvidenceError::ParseFailed(_)
        | KeeperSemanticEvidenceError::MalformedCommitment(_)
        | KeeperSemanticEvidenceError::EvidenceMismatch(_)
        | KeeperSemanticEvidenceError::InvalidTimeline(_) => {
            ComparisonTimelineUnavailableReason::EvidenceMismatch
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison_timeline::ComparisonAnchorResolutionError;

    #[test]
    fn verified_authority_remains_available_without_diagnostic() {
        let authority = ComparisonTimelineAuthority::persisted_keeper(2);
        let resolution = classify_authority_result(Ok(Some(authority.clone())));
        assert_eq!(
            resolution,
            KeeperSemanticCapabilityResolution {
                capability: ComparisonTimelineCapability::Available(authority),
                diagnostic: None,
            }
        );
    }

    #[test]
    fn historical_absence_is_legacy_unavailable_not_an_error() {
        let resolution = classify_authority_result(Ok(None));
        assert_eq!(
            resolution,
            KeeperSemanticCapabilityResolution {
                capability: ComparisonTimelineCapability::Unavailable(
                    ComparisonTimelineUnavailableReason::LegacyKeeperWithoutSemanticBundle,
                ),
                diagnostic: None,
            }
        );
    }

    #[test]
    fn transport_failure_is_distinct_from_evidence_conflict() {
        let error = KeeperSemanticEvidenceError::RequestFailed("offline".into());
        let resolution = classify_authority_result(Err(error.clone()));
        assert_eq!(
            resolution.capability,
            ComparisonTimelineCapability::Unavailable(
                ComparisonTimelineUnavailableReason::TimelineBundleUnavailable,
            )
        );
        assert_eq!(resolution.diagnostic, Some(error));
    }

    #[test]
    fn unsupported_versions_remain_a_typed_compatibility_state() {
        for error in [
            KeeperSemanticEvidenceError::UnsupportedSchemaVersion(9),
            KeeperSemanticEvidenceError::UnsupportedListenBundleVersion(7),
        ] {
            let resolution = classify_authority_result(Err(error.clone()));
            assert_eq!(
                resolution.capability,
                ComparisonTimelineCapability::Unavailable(
                    ComparisonTimelineUnavailableReason::UnsupportedBundleVersion,
                )
            );
            assert_eq!(resolution.diagnostic, Some(error));
        }
    }

    #[test]
    fn malformed_or_contradictory_stored_evidence_never_looks_legacy() {
        let errors = [
            KeeperSemanticEvidenceError::InvalidArtifactKey,
            KeeperSemanticEvidenceError::ParseFailed("bad json".into()),
            KeeperSemanticEvidenceError::MalformedCommitment("audio"),
            KeeperSemanticEvidenceError::EvidenceMismatch("score"),
            KeeperSemanticEvidenceError::InvalidTimeline(
                ComparisonAnchorResolutionError::TempoMapDiscontinuous,
            ),
        ];

        for error in errors {
            let resolution = classify_authority_result(Err(error.clone()));
            assert_eq!(
                resolution.capability,
                ComparisonTimelineCapability::Unavailable(
                    ComparisonTimelineUnavailableReason::EvidenceMismatch,
                )
            );
            assert_eq!(resolution.diagnostic, Some(error));
        }
    }
}
