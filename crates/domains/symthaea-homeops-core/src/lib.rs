#![forbid(unsafe_code)]

//! Typed, fail-closed domain contracts for residential package receiving and
//! internal material flow. This crate deliberately has no hardware authority.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DimensionsMm {
    pub length_mm: u32,
    pub width_mm: u32,
    pub height_mm: u32,
}

impl DimensionsMm {
    pub fn validate(&self) -> bool {
        self.length_mm > 0 && self.width_mm > 0 && self.height_mm > 0
    }

    /// Conservative axis-aligned fit; does not rotate or solve 3D packing.
    pub fn fits_within(&self, limit: &Self) -> bool {
        self.length_mm <= limit.length_mm
            && self.width_mm <= limit.width_mm
            && self.height_mm <= limit.height_mm
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemperatureBandMilliC {
    pub minimum: i32,
    pub maximum: i32,
}

impl TemperatureBandMilliC {
    pub fn validate(&self) -> bool {
        self.minimum <= self.maximum
    }

    pub fn contains_temperature(&self, temperature: i32) -> bool {
        self.minimum <= temperature && temperature <= self.maximum
    }

    pub fn contains_band(&self, required: &Self) -> bool {
        self.minimum <= required.minimum && required.maximum <= self.maximum
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HandlingClass {
    Ambient,
    Chilled,
    Frozen,
    Fragile,
    Restricted,
    HumanReview,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SealState {
    Intact,
    Damaged,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TransportMode {
    Road,
    GroundRobot,
    Aerial {
        aircraft_configuration_digest: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageManifest {
    pub package_id: String,
    pub mass_g: u64,
    pub dimensions: DimensionsMm,
    /// Optional package-local centre of mass, measured from one package corner.
    pub centre_of_mass_mm: Option<[u32; 3]>,
    pub handling_class: HandlingClass,
    pub required_temperature: Option<TemperatureBandMilliC>,
    pub seal_state: SealState,
    pub allowed_transport_modes: BTreeSet<TransportMode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    EmptyPackageId,
    ZeroMass,
    InvalidDimensions,
    CentreOfMassOutsidePackage,
    MissingTemperatureBand,
    InvalidTemperatureBand,
    NoAllowedTransportModes,
    EmptyAircraftConfigurationDigest,
}

impl PackageManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.package_id.trim().is_empty() {
            return Err(ManifestError::EmptyPackageId);
        }
        if self.mass_g == 0 {
            return Err(ManifestError::ZeroMass);
        }
        if !self.dimensions.validate() {
            return Err(ManifestError::InvalidDimensions);
        }
        if let Some([x, y, z]) = self.centre_of_mass_mm {
            if x > self.dimensions.length_mm
                || y > self.dimensions.width_mm
                || z > self.dimensions.height_mm
            {
                return Err(ManifestError::CentreOfMassOutsidePackage);
            }
        }
        if matches!(&self.handling_class, HandlingClass::Chilled | HandlingClass::Frozen)
            && self.required_temperature.is_none()
        {
            return Err(ManifestError::MissingTemperatureBand);
        }
        if let Some(band) = self.required_temperature {
            if !band.validate() {
                return Err(ManifestError::InvalidTemperatureBand);
            }
        }
        if self.allowed_transport_modes.is_empty() {
            return Err(ManifestError::NoAllowedTransportModes);
        }
        for mode in &self.allowed_transport_modes {
            if let TransportMode::Aerial {
                aircraft_configuration_digest,
            } = mode
            {
                if aircraft_configuration_digest.trim().is_empty() {
                    return Err(ManifestError::EmptyAircraftConfigurationDigest);
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageClass {
    Ambient,
    Chilled,
    Frozen,
    Quarantine,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageCompartment {
    pub compartment_id: String,
    pub storage_class: StorageClass,
    pub mass_capacity_g: u64,
    pub occupied_mass_g: u64,
    pub maximum_item_dimensions: DimensionsMm,
    pub supported_temperature: Option<TemperatureBandMilliC>,
    pub supports_fragile: bool,
    pub available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferGate {
    Ready,
    Busy,
    Interlocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentObservation {
    pub sensor_fresh: bool,
    pub temperature_millic: Option<i32>,
    pub gate: TransferGate,
    pub faulted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementHoldReason {
    CompartmentUnavailable,
    SensorStale,
    TransferNotReady,
    EquipmentFault,
    SealRequiresReview,
    CapacityUnavailable,
    TemperatureUnavailable,
    TemperatureOutOfRange,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementRejectReason {
    InvalidManifest(ManifestError),
    InvalidCompartment,
    DimensionsDoNotFit,
    StorageClassMismatch,
    UnsupportedFragileHandling,
    TemperatureCapabilityMismatch,
    RestrictedItemRequiresQuarantine,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementDecision {
    Accept,
    Hold(PlacementHoldReason),
    Reject(PlacementRejectReason),
}

/// Evaluates only declared manifest and compartment constraints. It does not
/// certify hygiene, package contents, building safety, or physical hardware.
pub fn assess_placement(
    package: &PackageManifest,
    compartment: &StorageCompartment,
    observation: &EnvironmentObservation,
) -> PlacementDecision {
    if let Err(error) = package.validate() {
        return PlacementDecision::Reject(PlacementRejectReason::InvalidManifest(error));
    }
    if compartment.compartment_id.trim().is_empty()
        || !compartment.maximum_item_dimensions.validate()
        || compartment.mass_capacity_g == 0
        || compartment.occupied_mass_g > compartment.mass_capacity_g
    {
        return PlacementDecision::Reject(PlacementRejectReason::InvalidCompartment);
    }
    if !compartment.available {
        return PlacementDecision::Hold(PlacementHoldReason::CompartmentUnavailable);
    }
    if observation.faulted {
        return PlacementDecision::Hold(PlacementHoldReason::EquipmentFault);
    }
    if !observation.sensor_fresh {
        return PlacementDecision::Hold(PlacementHoldReason::SensorStale);
    }
    if observation.gate != TransferGate::Ready {
        return PlacementDecision::Hold(PlacementHoldReason::TransferNotReady);
    }
    if package.seal_state != SealState::Intact && compartment.storage_class != StorageClass::Quarantine {
        return PlacementDecision::Hold(PlacementHoldReason::SealRequiresReview);
    }
    if !package.dimensions.fits_within(&compartment.maximum_item_dimensions) {
        return PlacementDecision::Reject(PlacementRejectReason::DimensionsDoNotFit);
    }
    let next_mass = compartment.occupied_mass_g.checked_add(package.mass_g);
    if next_mass.is_none_or(|mass| mass > compartment.mass_capacity_g) {
        return PlacementDecision::Hold(PlacementHoldReason::CapacityUnavailable);
    }

    match &package.handling_class {
        HandlingClass::Restricted | HandlingClass::HumanReview => {
            if compartment.storage_class != StorageClass::Quarantine {
                return PlacementDecision::Reject(
                    PlacementRejectReason::RestrictedItemRequiresQuarantine,
                );
            }
        }
        HandlingClass::Chilled => {
            if compartment.storage_class != StorageClass::Chilled
                && compartment.storage_class != StorageClass::Quarantine
            {
                return PlacementDecision::Reject(PlacementRejectReason::StorageClassMismatch);
            }
        }
        HandlingClass::Frozen => {
            if compartment.storage_class != StorageClass::Frozen
                && compartment.storage_class != StorageClass::Quarantine
            {
                return PlacementDecision::Reject(PlacementRejectReason::StorageClassMismatch);
            }
        }
        HandlingClass::Ambient | HandlingClass::Fragile => {
            if compartment.storage_class != StorageClass::Ambient
                && compartment.storage_class != StorageClass::Quarantine
            {
                return PlacementDecision::Reject(PlacementRejectReason::StorageClassMismatch);
            }
        }
    }

    if package.handling_class == HandlingClass::Fragile && !compartment.supports_fragile {
        return PlacementDecision::Reject(PlacementRejectReason::UnsupportedFragileHandling);
    }

    if let Some(required) = package.required_temperature {
        let Some(supported) = compartment.supported_temperature else {
            return PlacementDecision::Reject(PlacementRejectReason::TemperatureCapabilityMismatch);
        };
        if !supported.contains_band(&required) {
            return PlacementDecision::Reject(PlacementRejectReason::TemperatureCapabilityMismatch);
        }
        let Some(current) = observation.temperature_millic else {
            return PlacementDecision::Hold(PlacementHoldReason::TemperatureUnavailable);
        };
        if !required.contains_temperature(current) {
            return PlacementDecision::Hold(PlacementHoldReason::TemperatureOutOfRange);
        }
    }

    PlacementDecision::Accept
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToteSpec {
    pub tote_id: String,
    pub mass_capacity_g: u64,
    pub internal_dimensions: DimensionsMm,
    pub allowed_handling_classes: BTreeSet<HandlingClass>,
    pub requires_intact_seals: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToteManifest {
    pub spec: ToteSpec,
    pub package_ids: BTreeSet<String>,
    pub occupied_mass_g: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToteError {
    InvalidSpec,
    InvalidPackage(ManifestError),
    DuplicatePackage,
    UnsupportedHandlingClass,
    SealRequiresReview,
    PackageDoesNotFit,
    CapacityExceeded,
}

impl ToteManifest {
    pub fn new(spec: ToteSpec) -> Result<Self, ToteError> {
        if spec.tote_id.trim().is_empty()
            || spec.mass_capacity_g == 0
            || !spec.internal_dimensions.validate()
            || spec.allowed_handling_classes.is_empty()
        {
            return Err(ToteError::InvalidSpec);
        }
        Ok(Self {
            spec,
            package_ids: BTreeSet::new(),
            occupied_mass_g: 0,
        })
    }

    /// Adds a sealed package reference to a tote. This is a conservative
    /// dimension check, not a 3D packing proof or automated unboxing operation.
    pub fn add_package(&mut self, package: &PackageManifest) -> Result<(), ToteError> {
        package.validate().map_err(ToteError::InvalidPackage)?;
        if self.package_ids.contains(&package.package_id) {
            return Err(ToteError::DuplicatePackage);
        }
        if !self
            .spec
            .allowed_handling_classes
            .contains(&package.handling_class)
        {
            return Err(ToteError::UnsupportedHandlingClass);
        }
        if self.spec.requires_intact_seals && package.seal_state != SealState::Intact {
            return Err(ToteError::SealRequiresReview);
        }
        if !package.dimensions.fits_within(&self.spec.internal_dimensions) {
            return Err(ToteError::PackageDoesNotFit);
        }
        let Some(next_mass) = self.occupied_mass_g.checked_add(package.mass_g) else {
            return Err(ToteError::CapacityExceeded);
        };
        if next_mass > self.spec.mass_capacity_g {
            return Err(ToteError::CapacityExceeded);
        }
        self.package_ids.insert(package.package_id.clone());
        self.occupied_mass_g = next_mass;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryState {
    Expected,
    Receiving,
    IdentityPending,
    Accepted,
    Quarantined,
    Sorting,
    Moving,
    Containerizing,
    StoragePending,
    Stored,
    ReadyForCollection,
    Collected,
    ReturnPending,
    Returned,
    Faulted,
    ManualRecovery,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionError {
    InvalidTransition,
    TimestampNotMonotonic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeliveryTask {
    pub task_id: String,
    pub package_id: String,
    pub state: DeliveryState,
    pub last_event_time_ms: u64,
}

impl DeliveryTask {
    pub fn advance(
        &mut self,
        next: DeliveryState,
        event_time_ms: u64,
    ) -> Result<(), TransitionError> {
        if event_time_ms <= self.last_event_time_ms {
            return Err(TransitionError::TimestampNotMonotonic);
        }
        if !transition_allowed(&self.state, &next) {
            return Err(TransitionError::InvalidTransition);
        }
        self.state = next;
        self.last_event_time_ms = event_time_ms;
        Ok(())
    }
}

fn transition_allowed(current: &DeliveryState, next: &DeliveryState) -> bool {
    use DeliveryState::*;
    matches!(
        (current, next),
        (Expected, Receiving | Faulted)
            | (Receiving, IdentityPending | Quarantined | Faulted)
            | (IdentityPending, Accepted | Quarantined | Faulted)
            | (Accepted, Sorting | Moving | Containerizing | StoragePending | Quarantined | ReturnPending | Faulted)
            | (Quarantined, Sorting | ReturnPending | ManualRecovery | Faulted)
            | (Sorting, Moving | Containerizing | StoragePending | Quarantined | Faulted)
            | (Moving, Containerizing | StoragePending | Quarantined | Faulted)
            | (Containerizing, StoragePending | Quarantined | Faulted)
            | (StoragePending, Stored | Quarantined | Faulted)
            | (Stored, ReadyForCollection | ReturnPending | Faulted)
            | (ReadyForCollection, Collected | ReturnPending | Faulted)
            | (ReturnPending, Returned | ManualRecovery | Faulted)
            | (Faulted, ManualRecovery)
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffReceipt {
    pub package_id: String,
    pub releasing_party: String,
    pub receiving_party: String,
    pub event_time_ms: u64,
    pub evidence_ref: String,
    pub accepted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandoffError {
    MissingIdentity,
    SameParty,
    MissingEvidence,
}

impl HandoffReceipt {
    pub fn validate(&self) -> Result<(), HandoffError> {
        if self.package_id.trim().is_empty()
            || self.releasing_party.trim().is_empty()
            || self.receiving_party.trim().is_empty()
        {
            return Err(HandoffError::MissingIdentity);
        }
        if self.releasing_party == self.receiving_party {
            return Err(HandoffError::SameParty);
        }
        if self.accepted && self.evidence_ref.trim().is_empty() {
            return Err(HandoffError::MissingEvidence);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package(
        id: &str,
        mass_g: u64,
        handling_class: HandlingClass,
        required_temperature: Option<TemperatureBandMilliC>,
    ) -> PackageManifest {
        PackageManifest {
            package_id: id.to_string(),
            mass_g,
            dimensions: DimensionsMm {
                length_mm: 200,
                width_mm: 150,
                height_mm: 100,
            },
            centre_of_mass_mm: Some([100, 75, 50]),
            handling_class,
            required_temperature,
            seal_state: SealState::Intact,
            allowed_transport_modes: [TransportMode::Road, TransportMode::GroundRobot]
                .into_iter()
                .collect(),
        }
    }

    fn ambient_compartment() -> StorageCompartment {
        StorageCompartment {
            compartment_id: "ambient-01".to_string(),
            storage_class: StorageClass::Ambient,
            mass_capacity_g: 5_000,
            occupied_mass_g: 0,
            maximum_item_dimensions: DimensionsMm {
                length_mm: 300,
                width_mm: 250,
                height_mm: 200,
            },
            supported_temperature: None,
            supports_fragile: true,
            available: true,
        }
    }

    fn fresh_observation() -> EnvironmentObservation {
        EnvironmentObservation {
            sensor_fresh: true,
            temperature_millic: None,
            gate: TransferGate::Ready,
            faulted: false,
        }
    }

    #[test]
    fn rejects_zero_mass_and_invalid_dimensions() {
        let mut p = package("p-1", 0, HandlingClass::Ambient, None);
        assert_eq!(p.validate(), Err(ManifestError::ZeroMass));
        p.mass_g = 10;
        p.dimensions.length_mm = 0;
        assert_eq!(p.validate(), Err(ManifestError::InvalidDimensions));
    }

    #[test]
    fn rejects_centre_of_mass_outside_package() {
        let mut p = package("p-1", 100, HandlingClass::Ambient, None);
        p.centre_of_mass_mm = Some([201, 20, 20]);
        assert_eq!(p.validate(), Err(ManifestError::CentreOfMassOutsidePackage));
    }

    #[test]
    fn stale_sensor_holds_placement() {
        let p = package("p-1", 100, HandlingClass::Ambient, None);
        let mut observation = fresh_observation();
        observation.sensor_fresh = false;
        assert_eq!(
            assess_placement(&p, &ambient_compartment(), &observation),
            PlacementDecision::Hold(PlacementHoldReason::SensorStale)
        );
    }

    #[test]
    fn damaged_seal_does_not_enter_ordinary_storage() {
        let mut p = package("p-1", 100, HandlingClass::Ambient, None);
        p.seal_state = SealState::Damaged;
        assert_eq!(
            assess_placement(&p, &ambient_compartment(), &fresh_observation()),
            PlacementDecision::Hold(PlacementHoldReason::SealRequiresReview)
        );
    }

    #[test]
    fn package_over_capacity_is_held() {
        let p = package("p-1", 100, HandlingClass::Ambient, None);
        let mut c = ambient_compartment();
        c.occupied_mass_g = 4_950;
        assert_eq!(
            assess_placement(&p, &c, &fresh_observation()),
            PlacementDecision::Hold(PlacementHoldReason::CapacityUnavailable)
        );
    }

    #[test]
    fn chilled_package_requires_supported_band_and_fresh_temperature() {
        let required = TemperatureBandMilliC {
            minimum: 2_000,
            maximum: 8_000,
        };
        let p = package("milk-1", 900, HandlingClass::Chilled, Some(required));
        assert_eq!(
            p.validate(),
            Ok(())
        );
        let c = StorageCompartment {
            compartment_id: "chiller-01".to_string(),
            storage_class: StorageClass::Chilled,
            mass_capacity_g: 10_000,
            occupied_mass_g: 0,
            maximum_item_dimensions: DimensionsMm {
                length_mm: 400,
                width_mm: 300,
                height_mm: 300,
            },
            supported_temperature: Some(TemperatureBandMilliC {
                minimum: 1_000,
                maximum: 9_000,
            }),
            supports_fragile: false,
            available: true,
        };
        let mut observation = fresh_observation();
        observation.temperature_millic = Some(5_000);
        assert_eq!(
            assess_placement(&p, &c, &observation),
            PlacementDecision::Accept
        );
        observation.temperature_millic = Some(9_000);
        assert_eq!(
            assess_placement(&p, &c, &observation),
            PlacementDecision::Hold(PlacementHoldReason::TemperatureOutOfRange)
        );
    }

    #[test]
    fn restricted_package_can_only_route_to_quarantine() {
        let p = package("restricted-1", 300, HandlingClass::Restricted, None);
        assert!(matches!(
            assess_placement(&p, &ambient_compartment(), &fresh_observation()),
            PlacementDecision::Reject(PlacementRejectReason::RestrictedItemRequiresQuarantine)
        ));
    }

    #[test]
    fn tote_rejects_duplicate_and_over_capacity_packages() {
        let spec = ToteSpec {
            tote_id: "tote-1".to_string(),
            mass_capacity_g: 1_000,
            internal_dimensions: DimensionsMm {
                length_mm: 300,
                width_mm: 250,
                height_mm: 200,
            },
            allowed_handling_classes: [HandlingClass::Ambient].into_iter().collect(),
            requires_intact_seals: true,
        };
        let mut tote = ToteManifest::new(spec).unwrap();
        let p = package("p-1", 800, HandlingClass::Ambient, None);
        tote.add_package(&p).unwrap();
        assert_eq!(tote.add_package(&p), Err(ToteError::DuplicatePackage));
        let p2 = package("p-2", 300, HandlingClass::Ambient, None);
        assert_eq!(tote.add_package(&p2), Err(ToteError::CapacityExceeded));
    }

    #[test]
    fn handoff_requires_two_distinct_parties_and_evidence_on_acceptance() {
        let mut receipt = HandoffReceipt {
            package_id: "p-1".to_string(),
            releasing_party: "carrier-1".to_string(),
            receiving_party: "home-1".to_string(),
            event_time_ms: 100,
            evidence_ref: String::new(),
            accepted: true,
        };
        assert_eq!(receipt.validate(), Err(HandoffError::MissingEvidence));
        receipt.evidence_ref = "sha256:receipt".to_string();
        assert_eq!(receipt.validate(), Ok(()));
        receipt.receiving_party = receipt.releasing_party.clone();
        assert_eq!(receipt.validate(), Err(HandoffError::SameParty));
    }

    #[test]
    fn delivery_state_rejects_skipping_identity_and_non_monotonic_time() {
        let mut task = DeliveryTask {
            task_id: "task-1".to_string(),
            package_id: "p-1".to_string(),
            state: DeliveryState::Expected,
            last_event_time_ms: 100,
        };
        assert_eq!(
            task.advance(DeliveryState::Stored, 101),
            Err(TransitionError::InvalidTransition)
        );
        assert_eq!(
            task.advance(DeliveryState::Receiving, 100),
            Err(TransitionError::TimestampNotMonotonic)
        );
        task.advance(DeliveryState::Receiving, 101).unwrap();
        task.advance(DeliveryState::IdentityPending, 102).unwrap();
        task.advance(DeliveryState::Accepted, 103).unwrap();
        task.advance(DeliveryState::Sorting, 104).unwrap();
        task.advance(DeliveryState::Containerizing, 105).unwrap();
        task.advance(DeliveryState::StoragePending, 106).unwrap();
        task.advance(DeliveryState::Stored, 107).unwrap();
        assert_eq!(task.state, DeliveryState::Stored);
    }
}
