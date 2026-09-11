// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transactional rover/habitat -> suit resource transfer for SX-019.
//!
//! All dual-meter checks and capability admissions are validated before a
//! cloned suit state is modified. The cloned state is committed only after all
//! transfer operations succeed and protected survival energy is proven not to
//! decrease. This is a simulation/accounting layer, not hardware valve control.

use serde::{Deserialize, Serialize};

use crate::plss::PlssReferenceTwin;
use crate::power::{MultiBusPowerSystem, PowerError, PowerRequest};
use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceResourceKind {
    ElectricalEnergy,
    PrimaryOxygen,
    SecondaryOxygen,
    ThermalService,
    Co2Regeneration,
    HumidityRegeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeteredAmount {
    /// Amount reported by the service node.
    pub source_amount: f64,
    /// Amount independently reported by the suit.
    pub suit_amount: f64,
    /// Maximum permitted relative disagreement [0,1].
    pub max_relative_disagreement: f64,
}

impl MeteredAmount {
    pub const fn zero() -> Self {
        Self {
            source_amount: 0.0,
            suit_amount: 0.0,
            max_relative_disagreement: 0.02,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.source_amount.is_finite()
            && self.source_amount >= 0.0
            && self.suit_amount.is_finite()
            && self.suit_amount >= 0.0
            && self.max_relative_disagreement.is_finite()
            && (0.0..=1.0).contains(&self.max_relative_disagreement)
    }

    /// Use the smaller agreed reading so accounting never credits more
    /// resource than either side observed.
    pub fn conservative_agreed_amount(&self) -> Option<f64> {
        if !self.is_valid() {
            return None;
        }
        let scale = self.source_amount.max(self.suit_amount).max(1e-12);
        let disagreement = (self.source_amount - self.suit_amount).abs() / scale;
        (disagreement <= self.max_relative_disagreement)
            .then_some(self.source_amount.min(self.suit_amount))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceTransferPermissions {
    pub electrical_charge: bool,
    pub oxygen_replenish: bool,
    pub coolant_service: bool,
    pub plss_regeneration: bool,
}

impl ResourceTransferPermissions {
    pub const fn none() -> Self {
        Self {
            electrical_charge: false,
            oxygen_replenish: false,
            coolant_service: false,
            plss_regeneration: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceTransferRequest {
    pub duration_s: f64,
    /// Electrical energy offered, Wh.
    pub electrical_energy_wh: MeteredAmount,
    /// Standard litres.
    pub primary_oxygen_l: MeteredAmount,
    /// Standard litres.
    pub secondary_oxygen_l: MeteredAmount,
    /// Heat removed from positive PLSS thermal store, J.
    pub thermal_service_j: MeteredAmount,
    /// CO2 burden removed, litres-equivalent.
    pub co2_regeneration_l: MeteredAmount,
    /// Normalized humidity burden removed.
    pub humidity_regeneration: MeteredAmount,
    /// Fraction of the verified transfer actually completed before disconnect.
    /// This allows interrupted-transfer accounting without fabricating a full
    /// successful transfer.
    pub completion_fraction: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl ResourceTransferRequest {
    pub fn is_valid(&self) -> bool {
        self.duration_s.is_finite()
            && self.duration_s > 0.0
            && self.completion_fraction.is_finite()
            && (0.0..=1.0).contains(&self.completion_fraction)
            && [
                self.electrical_energy_wh,
                self.primary_oxygen_l,
                self.secondary_oxygen_l,
                self.thermal_service_j,
                self.co2_regeneration_l,
                self.humidity_regeneration,
            ]
            .iter()
            .all(MeteredAmount::is_valid)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceTransferReceipt {
    pub electrical_energy_accepted_wh: f64,
    pub primary_oxygen_accepted_l: f64,
    pub secondary_oxygen_accepted_l: f64,
    pub thermal_energy_removed_j: f64,
    pub co2_removed_l: f64,
    pub humidity_removed: f64,
    pub survival_energy_before_wh: f64,
    pub survival_energy_after_wh: f64,
    pub transfer_complete: bool,
    pub evidence: ExosuitEvidenceLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceTransferError {
    InvalidRequest,
    MeterDisagreement(ServiceResourceKind),
    ServiceNotAdmitted(ServiceResourceKind),
    PowerModel(PowerError),
    InvalidPlssState,
    ProtectedReserveInvariant,
}

pub fn apply_resource_transfer(
    power: &mut MultiBusPowerSystem,
    plss: &mut PlssReferenceTwin,
    request: ResourceTransferRequest,
    permissions: ResourceTransferPermissions,
) -> Result<ResourceTransferReceipt, ResourceTransferError> {
    if !request.is_valid() {
        return Err(ResourceTransferError::InvalidRequest);
    }

    let electrical_wh = agreed(
        ServiceResourceKind::ElectricalEnergy,
        request.electrical_energy_wh,
    )? * request.completion_fraction;
    let primary_o2_l = agreed(
        ServiceResourceKind::PrimaryOxygen,
        request.primary_oxygen_l,
    )? * request.completion_fraction;
    let secondary_o2_l = agreed(
        ServiceResourceKind::SecondaryOxygen,
        request.secondary_oxygen_l,
    )? * request.completion_fraction;
    let thermal_j = agreed(
        ServiceResourceKind::ThermalService,
        request.thermal_service_j,
    )? * request.completion_fraction;
    let co2_l = agreed(
        ServiceResourceKind::Co2Regeneration,
        request.co2_regeneration_l,
    )? * request.completion_fraction;
    let humidity = agreed(
        ServiceResourceKind::HumidityRegeneration,
        request.humidity_regeneration,
    )? * request.completion_fraction;

    require_permission(
        electrical_wh,
        permissions.electrical_charge,
        ServiceResourceKind::ElectricalEnergy,
    )?;
    require_permission(
        primary_o2_l + secondary_o2_l,
        permissions.oxygen_replenish,
        ServiceResourceKind::PrimaryOxygen,
    )?;
    require_permission(
        thermal_j,
        permissions.coolant_service,
        ServiceResourceKind::ThermalService,
    )?;
    require_permission(
        co2_l + humidity,
        permissions.plss_regeneration,
        ServiceResourceKind::Co2Regeneration,
    )?;

    // Transactional staging: nothing touches caller state until every operation
    // and invariant check succeeds.
    let mut next_power = power.clone();
    let mut next_plss = plss.clone();

    let survival_before = next_power.config().survival.energy_wh;
    let offered_w = electrical_wh * 3600.0 / request.duration_s;
    let power_receipt = next_power
        .step(PowerRequest {
            survival_w: 0.0,
            mobility_w: 0.0,
            mission_w: 0.0,
            regenerative_w: 0.0,
            external_charger_w: offered_w,
            dt_s: request.duration_s,
        })
        .map_err(ResourceTransferError::PowerModel)?;
    let electrical_energy_accepted_wh =
        power_receipt.external_charge_accepted_w * request.duration_s / 3600.0;

    let config = *next_plss.config();
    let state = next_plss.state_mut_for_fault_injection();
    if !state.primary_o2_remaining_l.is_finite()
        || !state.secondary_o2_remaining_l.is_finite()
        || !state.co2_burden_l.is_finite()
        || !state.humidity_burden.is_finite()
        || !state.thermal_store_k.is_finite()
    {
        return Err(ResourceTransferError::InvalidPlssState);
    }

    let primary_headroom = (config.primary_o2_l - state.primary_o2_remaining_l).max(0.0);
    let secondary_headroom = (config.secondary_o2_l - state.secondary_o2_remaining_l).max(0.0);
    let primary_oxygen_accepted_l = primary_o2_l.min(primary_headroom);
    let secondary_oxygen_accepted_l = secondary_o2_l.min(secondary_headroom);
    state.primary_o2_remaining_l += primary_oxygen_accepted_l;
    state.secondary_o2_remaining_l += secondary_oxygen_accepted_l;

    let co2_removed_l = co2_l.min(state.co2_burden_l.max(0.0));
    let humidity_removed = humidity.min(state.humidity_burden.max(0.0));
    state.co2_burden_l -= co2_removed_l;
    state.humidity_burden -= humidity_removed;

    let positive_thermal_energy_j =
        state.thermal_store_k.max(0.0) * config.thermal_capacitance_j_k;
    let thermal_energy_removed_j = thermal_j.min(positive_thermal_energy_j);
    state.thermal_store_k -= thermal_energy_removed_j / config.thermal_capacitance_j_k;

    let survival_after = next_power.config().survival.energy_wh;
    if survival_after + 1e-9 < survival_before {
        return Err(ResourceTransferError::ProtectedReserveInvariant);
    }

    *power = next_power;
    *plss = next_plss;

    Ok(ResourceTransferReceipt {
        electrical_energy_accepted_wh,
        primary_oxygen_accepted_l,
        secondary_oxygen_accepted_l,
        thermal_energy_removed_j,
        co2_removed_l,
        humidity_removed,
        survival_energy_before_wh: survival_before,
        survival_energy_after_wh: survival_after,
        transfer_complete: request.completion_fraction >= 1.0 - 1e-12,
        evidence: request.evidence,
    })
}

fn agreed(
    kind: ServiceResourceKind,
    amount: MeteredAmount,
) -> Result<f64, ResourceTransferError> {
    amount
        .conservative_agreed_amount()
        .ok_or(ResourceTransferError::MeterDisagreement(kind))
}

fn require_permission(
    requested: f64,
    permitted: bool,
    kind: ServiceResourceKind,
) -> Result<(), ResourceTransferError> {
    if requested > 0.0 && !permitted {
        Err(ResourceTransferError::ServiceNotAdmitted(kind))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metered(value: f64) -> MeteredAmount {
        MeteredAmount {
            source_amount: value,
            suit_amount: value,
            max_relative_disagreement: 0.02,
        }
    }

    fn request() -> ResourceTransferRequest {
        ResourceTransferRequest {
            duration_s: 600.0,
            electrical_energy_wh: metered(100.0),
            primary_oxygen_l: metered(100.0),
            secondary_oxygen_l: MeteredAmount::zero(),
            thermal_service_j: MeteredAmount::zero(),
            co2_regeneration_l: MeteredAmount::zero(),
            humidity_regeneration: MeteredAmount::zero(),
            completion_fraction: 1.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    fn permissions() -> ResourceTransferPermissions {
        ResourceTransferPermissions {
            electrical_charge: true,
            oxygen_replenish: true,
            coolant_service: true,
            plss_regeneration: true,
        }
    }

    #[test]
    fn inbound_service_never_drains_protected_survival_energy() {
        let mut power = MultiBusPowerSystem::simulation_reference();
        let mut plss = PlssReferenceTwin::simulation_reference();
        power.config_mut_for_fault_injection().survival.energy_wh = 500.0;
        plss.state_mut_for_fault_injection().primary_o2_remaining_l = 1_000.0;
        let before = power.config().survival.energy_wh;
        let receipt = apply_resource_transfer(&mut power, &mut plss, request(), permissions()).unwrap();
        assert!(receipt.survival_energy_after_wh >= before);
        assert!(receipt.primary_oxygen_accepted_l > 0.0);
    }

    #[test]
    fn meter_disagreement_rolls_back_everything() {
        let mut power = MultiBusPowerSystem::simulation_reference();
        let mut plss = PlssReferenceTwin::simulation_reference();
        power.config_mut_for_fault_injection().survival.energy_wh = 500.0;
        plss.state_mut_for_fault_injection().primary_o2_remaining_l = 1_000.0;
        let power_before = power.config().survival.energy_wh;
        let o2_before = plss.state().primary_o2_remaining_l;
        let mut bad = request();
        bad.primary_oxygen_l.source_amount = 100.0;
        bad.primary_oxygen_l.suit_amount = 50.0;
        let result = apply_resource_transfer(&mut power, &mut plss, bad, permissions());
        assert_eq!(
            result,
            Err(ResourceTransferError::MeterDisagreement(
                ServiceResourceKind::PrimaryOxygen
            ))
        );
        assert_eq!(power.config().survival.energy_wh, power_before);
        assert_eq!(plss.state().primary_o2_remaining_l, o2_before);
    }

    #[test]
    fn unadmitted_optional_service_cannot_transfer() {
        let mut power = MultiBusPowerSystem::simulation_reference();
        let mut plss = PlssReferenceTwin::simulation_reference();
        let mut denied = permissions();
        denied.oxygen_replenish = false;
        let result = apply_resource_transfer(&mut power, &mut plss, request(), denied);
        assert_eq!(
            result,
            Err(ResourceTransferError::ServiceNotAdmitted(
                ServiceResourceKind::PrimaryOxygen
            ))
        );
    }

    #[test]
    fn interrupted_transfer_is_accounted_as_partial() {
        let mut power = MultiBusPowerSystem::simulation_reference();
        let mut plss = PlssReferenceTwin::simulation_reference();
        power.config_mut_for_fault_injection().survival.energy_wh = 500.0;
        plss.state_mut_for_fault_injection().primary_o2_remaining_l = 1_000.0;
        let mut partial = request();
        partial.completion_fraction = 0.25;
        let receipt = apply_resource_transfer(&mut power, &mut plss, partial, permissions()).unwrap();
        assert!(!receipt.transfer_complete);
        assert!(receipt.primary_oxygen_accepted_l <= 25.0 + 1e-9);
        assert!(receipt.electrical_energy_accepted_wh <= 25.0 + 1e-9);
    }
}
