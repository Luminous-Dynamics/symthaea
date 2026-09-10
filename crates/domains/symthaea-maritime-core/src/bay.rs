// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

fn canonical_field(value: &str) -> bool {
    !value.trim().is_empty()
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BayMediumState {
    Dry,
    Flooded,
    Transitioning,
    Unknown,
}

/// Lower-level pressure/interlock qualification consumed by maritime-core.
///
/// This deliberately does not expose pressure-control commands or thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BayPressureQualification {
    QualifiedForInnerBoundary,
    QualifiedForOuterBoundary,
    Unqualified,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryPosition {
    Open,
    Closed,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BayOccupancy {
    Empty,
    Occupied { client_platform_id: String },
    Unknown,
}

impl BayOccupancy {
    fn validate(&self) -> Result<(), &'static str> {
        if let Self::Occupied { client_platform_id } = self {
            if !canonical_field(client_platform_id) {
                return Err("occupied bay requires canonical client platform id");
            }
        }
        Ok(())
    }
}

/// Evidence-bearing wet/dry bay observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaritimeBayObservation {
    pub bay_id: String,
    pub host_platform_id: String,
    pub medium: BayMediumState,
    pub pressure_qualification: BayPressureQualification,
    pub inner_boundary: BoundaryPosition,
    pub outer_boundary: BoundaryPosition,
    pub occupancy: BayOccupancy,
    pub service_isolated: bool,
    pub handling_volume_clear: bool,
    pub faulted: bool,
    pub evidence_binding: String,
}

impl MaritimeBayObservation {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !canonical_field(&self.bay_id)
            || !canonical_field(&self.host_platform_id)
            || !canonical_field(&self.evidence_binding)
        {
            return Err("bay observation contains a malformed canonical field");
        }
        self.occupancy.validate()?;

        if self.inner_boundary == BoundaryPosition::Open
            && self.outer_boundary == BoundaryPosition::Open
        {
            return Err("inner and outer bay boundaries must not both be open");
        }

        if self.outer_boundary == BoundaryPosition::Open
            && (self.medium != BayMediumState::Flooded
                || self.pressure_qualification
                    != BayPressureQualification::QualifiedForOuterBoundary)
        {
            return Err("open outer boundary requires flooded outer-qualified bay state");
        }

        if self.inner_boundary == BoundaryPosition::Open
            && (self.medium != BayMediumState::Dry
                || self.pressure_qualification
                    != BayPressureQualification::QualifiedForInnerBoundary)
        {
            return Err("open inner boundary requires dry inner-qualified bay state");
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BayBoundary {
    Inner,
    Outer,
}

/// Fresh local facts that raw serialized bay observations cannot recreate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BayBoundaryGateContext {
    pub authority_permitted: bool,
    pub local_interlocks_clear: bool,
    pub authenticated_occupant_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BayBoundaryRefusal {
    MalformedObservation,
    BayFaulted,
    AuthorityDenied,
    LocalInterlockBlocked,
    UnknownBoundaryState,
    OppositeBoundaryNotClosed,
    ServiceNotIsolated,
    HandlingVolumeBlocked,
    MediumNotQualified,
    PressureNotQualified,
    UnknownOccupancy,
    OccupantIdentityMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BayBoundaryDecision {
    Permitted,
    Refused(BayBoundaryRefusal),
}

/// Evaluate whether one bay boundary may be opened.
///
/// Physical pumping, ballast, valve and door control remain below this layer.
pub fn evaluate_bay_boundary_open(
    requested: BayBoundary,
    observation: &MaritimeBayObservation,
    context: &BayBoundaryGateContext,
) -> BayBoundaryDecision {
    if observation.validate().is_err() {
        return BayBoundaryDecision::Refused(BayBoundaryRefusal::MalformedObservation);
    }
    if observation.faulted {
        return BayBoundaryDecision::Refused(BayBoundaryRefusal::BayFaulted);
    }
    if !context.authority_permitted {
        return BayBoundaryDecision::Refused(BayBoundaryRefusal::AuthorityDenied);
    }
    if !context.local_interlocks_clear {
        return BayBoundaryDecision::Refused(BayBoundaryRefusal::LocalInterlockBlocked);
    }
    if observation.inner_boundary == BoundaryPosition::Unknown
        || observation.outer_boundary == BoundaryPosition::Unknown
    {
        return BayBoundaryDecision::Refused(BayBoundaryRefusal::UnknownBoundaryState);
    }

    match &observation.occupancy {
        BayOccupancy::Unknown => {
            return BayBoundaryDecision::Refused(BayBoundaryRefusal::UnknownOccupancy);
        }
        BayOccupancy::Occupied { client_platform_id }
            if context.authenticated_occupant_id.as_deref() != Some(client_platform_id.as_str()) =>
        {
            return BayBoundaryDecision::Refused(BayBoundaryRefusal::OccupantIdentityMismatch);
        }
        BayOccupancy::Empty | BayOccupancy::Occupied { .. } => {}
    }

    if !observation.service_isolated {
        return BayBoundaryDecision::Refused(BayBoundaryRefusal::ServiceNotIsolated);
    }
    if !observation.handling_volume_clear {
        return BayBoundaryDecision::Refused(BayBoundaryRefusal::HandlingVolumeBlocked);
    }

    let (opposite_boundary, required_medium, required_pressure) = match requested {
        BayBoundary::Inner => (
            observation.outer_boundary,
            BayMediumState::Dry,
            BayPressureQualification::QualifiedForInnerBoundary,
        ),
        BayBoundary::Outer => (
            observation.inner_boundary,
            BayMediumState::Flooded,
            BayPressureQualification::QualifiedForOuterBoundary,
        ),
    };

    if opposite_boundary != BoundaryPosition::Closed {
        return BayBoundaryDecision::Refused(BayBoundaryRefusal::OppositeBoundaryNotClosed);
    }
    if observation.medium != required_medium {
        return BayBoundaryDecision::Refused(BayBoundaryRefusal::MediumNotQualified);
    }
    if observation.pressure_qualification != required_pressure {
        return BayBoundaryDecision::Refused(BayBoundaryRefusal::PressureNotQualified);
    }

    BayBoundaryDecision::Permitted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_dry_bay() -> MaritimeBayObservation {
        MaritimeBayObservation {
            bay_id: "bay-1".into(),
            host_platform_id: "tender-1".into(),
            medium: BayMediumState::Dry,
            pressure_qualification: BayPressureQualification::QualifiedForInnerBoundary,
            inner_boundary: BoundaryPosition::Closed,
            outer_boundary: BoundaryPosition::Closed,
            occupancy: BayOccupancy::Empty,
            service_isolated: true,
            handling_volume_clear: true,
            faulted: false,
            evidence_binding: "evidence:bay-8".into(),
        }
    }

    fn gate() -> BayBoundaryGateContext {
        BayBoundaryGateContext {
            authority_permitted: true,
            local_interlocks_clear: true,
            authenticated_occupant_id: None,
        }
    }

    #[test]
    fn dry_inner_qualified_bay_can_open_inner_boundary() {
        assert_eq!(
            evaluate_bay_boundary_open(BayBoundary::Inner, &empty_dry_bay(), &gate()),
            BayBoundaryDecision::Permitted
        );
    }

    #[test]
    fn outer_boundary_requires_flooded_outer_qualified_state() {
        assert_eq!(
            evaluate_bay_boundary_open(BayBoundary::Outer, &empty_dry_bay(), &gate()),
            BayBoundaryDecision::Refused(BayBoundaryRefusal::MediumNotQualified)
        );

        let mut flooded = empty_dry_bay();
        flooded.medium = BayMediumState::Flooded;
        flooded.pressure_qualification = BayPressureQualification::QualifiedForOuterBoundary;
        assert_eq!(
            evaluate_bay_boundary_open(BayBoundary::Outer, &flooded, &gate()),
            BayBoundaryDecision::Permitted
        );
    }

    #[test]
    fn unsafe_observed_boundary_combinations_are_rejected() {
        let mut impossible = empty_dry_bay();
        impossible.inner_boundary = BoundaryPosition::Open;
        impossible.outer_boundary = BoundaryPosition::Open;
        assert!(impossible.validate().is_err());

        let mut wrong_medium = empty_dry_bay();
        wrong_medium.outer_boundary = BoundaryPosition::Open;
        assert!(wrong_medium.validate().is_err());
    }

    #[test]
    fn service_and_handling_must_be isolated_before_opening() {
        let mut obs = empty_dry_bay();
        obs.service_isolated = false;
        assert_eq!(
            evaluate_bay_boundary_open(BayBoundary::Inner, &obs, &gate()),
            BayBoundaryDecision::Refused(BayBoundaryRefusal::ServiceNotIsolated)
        );

        obs.service_isolated = true;
        obs.handling_volume_clear = false;
        assert_eq!(
            evaluate_bay_boundary_open(BayBoundary::Inner, &obs, &gate()),
            BayBoundaryDecision::Refused(BayBoundaryRefusal::HandlingVolumeBlocked)
        );
    }

    #[test]
    fn occupied_bay_requires_exact_authenticated_occupant() {
        let mut obs = empty_dry_bay();
        obs.occupancy = BayOccupancy::Occupied {
            client_platform_id: "auv-4".into(),
        };
        assert_eq!(
            evaluate_bay_boundary_open(BayBoundary::Inner, &obs, &gate()),
            BayBoundaryDecision::Refused(BayBoundaryRefusal::OccupantIdentityMismatch)
        );

        let mut local = gate();
        local.authenticated_occupant_id = Some("auv-4".into());
        assert_eq!(
            evaluate_bay_boundary_open(BayBoundary::Inner, &obs, &local),
            BayBoundaryDecision::Permitted
        );
    }

    #[test]
    fn authority_and_local_interlocks_fail_closed() {
        let mut local = gate();
        local.authority_permitted = false;
        assert_eq!(
            evaluate_bay_boundary_open(BayBoundary::Inner, &empty_dry_bay(), &local),
            BayBoundaryDecision::Refused(BayBoundaryRefusal::AuthorityDenied)
        );

        local.authority_permitted = true;
        local.local_interlocks_clear = false;
        assert_eq!(
            evaluate_bay_boundary_open(BayBoundary::Inner, &empty_dry_bay(), &local),
            BayBoundaryDecision::Refused(BayBoundaryRefusal::LocalInterlockBlocked)
        );
    }
}
