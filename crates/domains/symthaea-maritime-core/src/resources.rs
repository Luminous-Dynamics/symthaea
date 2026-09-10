// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

fn canonical_field(value: &str) -> bool {
    !value.trim().is_empty()
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

/// Evidence-bearing snapshot of resources that upper maritime software may consume.
///
/// `protected_reserve_energy_j` is intentionally not allocatable through this
/// contract. It belongs to lower-level safety/recovery policy and remains outside
/// ordinary mission/service scheduling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaritimeResourceEnvelope {
    pub available_power_w: u64,
    pub stored_energy_j: u64,
    pub protected_reserve_energy_j: u64,
    pub thermal_rejection_margin_w: u64,
    pub cooling_available: bool,
    pub evidence_binding: String,
}

impl MaritimeResourceEnvelope {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !canonical_field(&self.evidence_binding) {
            return Err("resource envelope requires a canonical evidence binding");
        }
        if self.protected_reserve_energy_j > self.stored_energy_j {
            return Err("protected reserve cannot exceed stored energy");
        }
        Ok(())
    }

    pub fn discretionary_energy_j(&self) -> u64 {
        self.stored_energy_j
            .saturating_sub(self.protected_reserve_energy_j)
    }
}

/// Scheduling hint only. Priority never weakens hard power, energy, thermal,
/// authority, interlock, or reserve constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ResourcePriority {
    Opportunistic,
    Service,
    MissionEssential,
    RecoveryEssential,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaritimeResourceRequest {
    pub request_id: String,
    pub power_w: u64,
    pub energy_j: u64,
    pub thermal_load_w: u64,
    pub priority: ResourcePriority,
}

impl MaritimeResourceRequest {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !canonical_field(&self.request_id) {
            return Err("resource request requires a canonical request id");
        }
        if self.power_w == 0 && self.energy_j == 0 && self.thermal_load_w == 0 {
            return Err("resource request must consume at least one resource");
        }
        Ok(())
    }
}

/// Fresh local authorization/interlock state. Deliberately non-serializable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceGateContext {
    pub authority_permitted: bool,
    pub local_interlocks_clear: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceRefusal {
    MalformedEnvelope,
    MalformedRequest,
    AuthorityDenied,
    LocalInterlockBlocked,
    CoolingUnavailable,
    InsufficientPower,
    InsufficientDiscretionaryEnergy,
    InsufficientThermalMargin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceAllocationProjection {
    pub remaining_power_w: u64,
    pub remaining_discretionary_energy_j: u64,
    pub remaining_thermal_margin_w: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceDecision {
    Permitted(ResourceAllocationProjection),
    Refused(ResourceRefusal),
}

/// Evaluate one resource request against the current evidence-backed envelope.
///
/// This function never consumes protected reserve energy. A caller cannot bypass
/// that rule by claiming a higher scheduling priority; reserve use, if ever
/// permitted, must occur through a separate lower-level safety/recovery mechanism.
pub fn evaluate_resource_request(
    envelope: &MaritimeResourceEnvelope,
    request: &MaritimeResourceRequest,
    context: ResourceGateContext,
) -> ResourceDecision {
    if envelope.validate().is_err() {
        return ResourceDecision::Refused(ResourceRefusal::MalformedEnvelope);
    }
    if request.validate().is_err() {
        return ResourceDecision::Refused(ResourceRefusal::MalformedRequest);
    }
    if !context.authority_permitted {
        return ResourceDecision::Refused(ResourceRefusal::AuthorityDenied);
    }
    if !context.local_interlocks_clear {
        return ResourceDecision::Refused(ResourceRefusal::LocalInterlockBlocked);
    }
    if request.thermal_load_w > 0 && !envelope.cooling_available {
        return ResourceDecision::Refused(ResourceRefusal::CoolingUnavailable);
    }
    if request.power_w > envelope.available_power_w {
        return ResourceDecision::Refused(ResourceRefusal::InsufficientPower);
    }
    let discretionary_energy_j = envelope.discretionary_energy_j();
    if request.energy_j > discretionary_energy_j {
        return ResourceDecision::Refused(ResourceRefusal::InsufficientDiscretionaryEnergy);
    }
    if request.thermal_load_w > envelope.thermal_rejection_margin_w {
        return ResourceDecision::Refused(ResourceRefusal::InsufficientThermalMargin);
    }

    ResourceDecision::Permitted(ResourceAllocationProjection {
        remaining_power_w: envelope.available_power_w - request.power_w,
        remaining_discretionary_energy_j: discretionary_energy_j - request.energy_j,
        remaining_thermal_margin_w: envelope.thermal_rejection_margin_w - request.thermal_load_w,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope() -> MaritimeResourceEnvelope {
        MaritimeResourceEnvelope {
            available_power_w: 100_000,
            stored_energy_j: 5_000_000,
            protected_reserve_energy_j: 1_000_000,
            thermal_rejection_margin_w: 40_000,
            cooling_available: true,
            evidence_binding: "evidence:plant-7".into(),
        }
    }

    fn request() -> MaritimeResourceRequest {
        MaritimeResourceRequest {
            request_id: "charge-auv-4".into(),
            power_w: 20_000,
            energy_j: 500_000,
            thermal_load_w: 8_000,
            priority: ResourcePriority::Service,
        }
    }

    fn gate() -> ResourceGateContext {
        ResourceGateContext {
            authority_permitted: true,
            local_interlocks_clear: true,
        }
    }

    #[test]
    fn independent_resource_axes_are_projected() {
        assert_eq!(
            evaluate_resource_request(&envelope(), &request(), gate()),
            ResourceDecision::Permitted(ResourceAllocationProjection {
                remaining_power_w: 80_000,
                remaining_discretionary_energy_j: 3_500_000,
                remaining_thermal_margin_w: 32_000,
            })
        );
    }

    #[test]
    fn plentiful_power_does_not_hide_insufficient_thermal_margin() {
        let mut req = request();
        req.power_w = 1;
        req.thermal_load_w = 40_001;
        assert_eq!(
            evaluate_resource_request(&envelope(), &req, gate()),
            ResourceDecision::Refused(ResourceRefusal::InsufficientThermalMargin)
        );
    }

    #[test]
    fn protected_reserve_is_never_allocated_even_to_high_priority_request() {
        let mut req = request();
        req.priority = ResourcePriority::RecoveryEssential;
        req.energy_j = 4_000_001;
        assert_eq!(
            evaluate_resource_request(&envelope(), &req, gate()),
            ResourceDecision::Refused(ResourceRefusal::InsufficientDiscretionaryEnergy)
        );
    }

    #[test]
    fn cooling_loss_cannot_expand_permission() {
        let req = request();
        assert!(matches!(
            evaluate_resource_request(&envelope(), &req, gate()),
            ResourceDecision::Permitted(_)
        ));

        let mut degraded = envelope();
        degraded.cooling_available = false;
        assert_eq!(
            evaluate_resource_request(&degraded, &req, gate()),
            ResourceDecision::Refused(ResourceRefusal::CoolingUnavailable)
        );
    }

    #[test]
    fn malformed_reserve_or_evidence_fails_closed() {
        let mut malformed = envelope();
        malformed.protected_reserve_energy_j = malformed.stored_energy_j + 1;
        assert_eq!(
            evaluate_resource_request(&malformed, &request(), gate()),
            ResourceDecision::Refused(ResourceRefusal::MalformedEnvelope)
        );

        malformed = envelope();
        malformed.evidence_binding = " plant-7".into();
        assert_eq!(
            evaluate_resource_request(&malformed, &request(), gate()),
            ResourceDecision::Refused(ResourceRefusal::MalformedEnvelope)
        );
    }

    #[test]
    fn authority_and_interlocks_are_point_of_use_gates() {
        let mut context = gate();
        context.authority_permitted = false;
        assert_eq!(
            evaluate_resource_request(&envelope(), &request(), context),
            ResourceDecision::Refused(ResourceRefusal::AuthorityDenied)
        );

        context.authority_permitted = true;
        context.local_interlocks_clear = false;
        assert_eq!(
            evaluate_resource_request(&envelope(), &request(), context),
            ResourceDecision::Refused(ResourceRefusal::LocalInterlockBlocked)
        );
    }
}
