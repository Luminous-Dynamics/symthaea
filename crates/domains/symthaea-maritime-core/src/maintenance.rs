// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

fn canonical_field(value: &str) -> bool {
    !value.trim().is_empty()
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

/// Ordered maintenance evidence stages. Later stages never imply that earlier
/// evidence existed; `MaintenanceRecord::validate` requires the full prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaintenanceStage {
    FaultObserved,
    Diagnosed,
    ActionCompleted,
    VerificationPassed,
    Requalified,
}

impl MaintenanceStage {
    pub const ALL: [Self; 5] = [
        Self::FaultObserved,
        Self::Diagnosed,
        Self::ActionCompleted,
        Self::VerificationPassed,
        Self::Requalified,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaintenanceDisposition {
    Monitor,
    Defer,
    RepairOnboard,
    ReplaceModule,
    ExternalServiceRequired,
    RetireFromService,
}

/// Evidence-bearing maintenance lineage for one component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceRecord {
    pub record_id: String,
    pub platform_id: String,
    pub component_id: String,
    pub disposition: MaintenanceDisposition,
    pub stage: MaintenanceStage,
    /// Each completed stage binds to its own evidence. Validation requires a
    /// contiguous prefix through `stage`; a late-stage claim cannot skip diagnosis
    /// or verification evidence.
    pub stage_evidence: BTreeMap<MaintenanceStage, String>,
}

impl MaintenanceRecord {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !canonical_field(&self.record_id)
            || !canonical_field(&self.platform_id)
            || !canonical_field(&self.component_id)
        {
            return Err("maintenance record contains a malformed canonical field");
        }

        for required in MaintenanceStage::ALL {
            if required > self.stage {
                break;
            }
            let Some(binding) = self.stage_evidence.get(&required) else {
                return Err("maintenance record is missing required stage evidence");
            };
            if !canonical_field(binding) {
                return Err("maintenance stage evidence binding is malformed");
            }
        }

        if self
            .stage_evidence
            .keys()
            .any(|stage| *stage > self.stage)
        {
            return Err("maintenance record contains evidence for a future stage");
        }

        Ok(())
    }
}

/// Fresh point-of-use facts required before return to service.
/// Deliberately non-serializable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReturnToServiceGateContext {
    pub authority_permitted: bool,
    pub local_interlocks_clear: bool,
    pub current_health_acceptable: bool,
    pub current_operational_limits_within: bool,
    pub independent_verification_present: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReturnToServiceRefusal {
    MalformedMaintenanceRecord,
    NotRequalified,
    RetiredFromService,
    AuthorityDenied,
    LocalInterlockBlocked,
    CurrentHealthUnacceptable,
    CurrentOperationalLimitsNotWithin,
    MissingIndependentVerification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReturnToServiceDecision {
    Permitted,
    Refused(ReturnToServiceRefusal),
}

/// Decide whether one maintained component may participate in a platform's
/// return-to-service transition.
///
/// A diagnosis, repair, replacement or even completed verification is not enough:
/// the maintenance lineage must reach `Requalified`, and fresh current facts must
/// still confirm acceptable health, operating limits, local interlocks and authority.
pub fn evaluate_return_to_service(
    record: &MaintenanceRecord,
    context: ReturnToServiceGateContext,
) -> ReturnToServiceDecision {
    if record.validate().is_err() {
        return ReturnToServiceDecision::Refused(
            ReturnToServiceRefusal::MalformedMaintenanceRecord,
        );
    }
    if record.stage != MaintenanceStage::Requalified {
        return ReturnToServiceDecision::Refused(ReturnToServiceRefusal::NotRequalified);
    }
    if record.disposition == MaintenanceDisposition::RetireFromService {
        return ReturnToServiceDecision::Refused(ReturnToServiceRefusal::RetiredFromService);
    }
    if !context.authority_permitted {
        return ReturnToServiceDecision::Refused(ReturnToServiceRefusal::AuthorityDenied);
    }
    if !context.local_interlocks_clear {
        return ReturnToServiceDecision::Refused(ReturnToServiceRefusal::LocalInterlockBlocked);
    }
    if !context.current_health_acceptable {
        return ReturnToServiceDecision::Refused(
            ReturnToServiceRefusal::CurrentHealthUnacceptable,
        );
    }
    if !context.current_operational_limits_within {
        return ReturnToServiceDecision::Refused(
            ReturnToServiceRefusal::CurrentOperationalLimitsNotWithin,
        );
    }
    if !context.independent_verification_present {
        return ReturnToServiceDecision::Refused(
            ReturnToServiceRefusal::MissingIndependentVerification,
        );
    }

    ReturnToServiceDecision::Permitted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_record(stage: MaintenanceStage) -> MaintenanceRecord {
        let mut stage_evidence = BTreeMap::new();
        for candidate in MaintenanceStage::ALL {
            if candidate > stage {
                break;
            }
            stage_evidence.insert(candidate, format!("evidence:{candidate:?}"));
        }
        MaintenanceRecord {
            record_id: "maint-7".into(),
            platform_id: "auv-4".into(),
            component_id: "thruster-2".into(),
            disposition: MaintenanceDisposition::RepairOnboard,
            stage,
            stage_evidence,
        }
    }

    fn gate() -> ReturnToServiceGateContext {
        ReturnToServiceGateContext {
            authority_permitted: true,
            local_interlocks_clear: true,
            current_health_acceptable: true,
            current_operational_limits_within: true,
            independent_verification_present: true,
        }
    }

    #[test]
    fn requalified_lineage_with_fresh_checks_can_return_to_service() {
        assert_eq!(
            evaluate_return_to_service(&full_record(MaintenanceStage::Requalified), gate()),
            ReturnToServiceDecision::Permitted
        );
    }

    #[test]
    fn diagnosis_or_repair_is_not_requalification() {
        assert_eq!(
            evaluate_return_to_service(&full_record(MaintenanceStage::Diagnosed), gate()),
            ReturnToServiceDecision::Refused(ReturnToServiceRefusal::NotRequalified)
        );
        assert_eq!(
            evaluate_return_to_service(&full_record(MaintenanceStage::ActionCompleted), gate()),
            ReturnToServiceDecision::Refused(ReturnToServiceRefusal::NotRequalified)
        );
    }

    #[test]
    fn skipped_stage_evidence_fails_closed() {
        let mut record = full_record(MaintenanceStage::Requalified);
        record.stage_evidence.remove(&MaintenanceStage::VerificationPassed);
        assert_eq!(
            evaluate_return_to_service(&record, gate()),
            ReturnToServiceDecision::Refused(
                ReturnToServiceRefusal::MalformedMaintenanceRecord,
            )
        );
    }

    #[test]
    fn future_stage_evidence_is_rejected() {
        let mut record = full_record(MaintenanceStage::Diagnosed);
        record.stage_evidence.insert(
            MaintenanceStage::Requalified,
            "evidence:premature".into(),
        );
        assert!(record.validate().is_err());
    }

    #[test]
    fn retirement_cannot_be_overridden_by_fresh_good_state() {
        let mut record = full_record(MaintenanceStage::Requalified);
        record.disposition = MaintenanceDisposition::RetireFromService;
        assert_eq!(
            evaluate_return_to_service(&record, gate()),
            ReturnToServiceDecision::Refused(ReturnToServiceRefusal::RetiredFromService)
        );
    }

    #[test]
    fn fresh_health_limits_interlocks_authority_and_verification_all_gate_return() {
        let record = full_record(MaintenanceStage::Requalified);

        let mut context = gate();
        context.current_health_acceptable = false;
        assert_eq!(
            evaluate_return_to_service(&record, context),
            ReturnToServiceDecision::Refused(
                ReturnToServiceRefusal::CurrentHealthUnacceptable,
            )
        );

        context = gate();
        context.current_operational_limits_within = false;
        assert_eq!(
            evaluate_return_to_service(&record, context),
            ReturnToServiceDecision::Refused(
                ReturnToServiceRefusal::CurrentOperationalLimitsNotWithin,
            )
        );

        context = gate();
        context.independent_verification_present = false;
        assert_eq!(
            evaluate_return_to_service(&record, context),
            ReturnToServiceDecision::Refused(
                ReturnToServiceRefusal::MissingIndependentVerification,
            )
        );
    }
}
