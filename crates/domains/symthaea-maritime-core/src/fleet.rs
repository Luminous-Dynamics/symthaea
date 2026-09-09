// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::{HealthSeverity, OperatingEnvelope, PlatformHealth};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetMemberReport {
    pub platform_id: String,
    pub generation: u64,
    pub admitted: bool,
    pub quarantined: bool,
    pub envelope: OperatingEnvelope,
    pub health: PlatformHealth,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetAssuranceReport {
    pub fleet_id: String,
    pub expected_generation: u64,
    pub members: BTreeMap<String, FleetMemberReport>,
}

fn member_identity_consistent(key: &str, member: &FleetMemberReport) -> bool {
    !key.trim().is_empty()
        && key == member.platform_id
        && member.platform_id == member.health.platform_id
        && member.platform_id.trim() == member.platform_id
}

impl FleetAssuranceReport {
    /// Admission is intentionally all-members, not an aggregate score. Missing or
    /// unknown health evidence cannot satisfy a full-admission claim.
    ///
    /// The map key, member identity and nested health identity must also agree so a
    /// healthy report for one platform cannot be accidentally attributed to another.
    pub fn all_members_admitted(&self) -> bool {
        !self.members.is_empty()
            && !self.fleet_id.trim().is_empty()
            && self.members.iter().all(|(key, member)| {
                member_identity_consistent(key, member)
                    && member.admitted
                    && !member.quarantined
                    && member.generation == self.expected_generation
                    && member.envelope == OperatingEnvelope::Normal
                    && member.health.is_fully_observed_nominal()
            })
    }

    pub fn degraded_members(&self) -> Vec<&FleetMemberReport> {
        self.members
            .iter()
            .filter_map(|(key, member)| {
                ( !member_identity_consistent(key, member)
                    || member.quarantined
                    || !member.admitted
                    || member.generation != self.expected_generation
                    || member.envelope != OperatingEnvelope::Normal
                    || member.health.worst_severity() >= HealthSeverity::Unknown)
                    .then_some(member)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComponentHealth, HealthSeverity};

    fn member(id: &str, severity: HealthSeverity) -> FleetMemberReport {
        FleetMemberReport {
            platform_id: id.into(),
            generation: 4,
            admitted: true,
            quarantined: false,
            envelope: OperatingEnvelope::Normal,
            health: PlatformHealth {
                platform_id: id.into(),
                observed_at_ms: 1,
                components: vec![ComponentHealth {
                    component_id: "core".into(),
                    severity,
                    code: "test".into(),
                    detail: String::new(),
                }],
            },
        }
    }

    #[test]
    fn one_unsafe_member_prevents_full_admission() {
        let mut members = BTreeMap::new();
        members.insert("a".into(), member("a", HealthSeverity::Healthy));
        members.insert("b".into(), member("b", HealthSeverity::Unsafe));
        let report = FleetAssuranceReport {
            fleet_id: "fleet".into(),
            expected_generation: 4,
            members,
        };
        assert!(!report.all_members_admitted());
        assert_eq!(report.degraded_members().len(), 1);
    }

    #[test]
    fn missing_member_health_blocks_full_admission() {
        let mut m = member("a", HealthSeverity::Healthy);
        m.health.components.clear();
        let report = FleetAssuranceReport {
            fleet_id: "fleet".into(),
            expected_generation: 4,
            members: BTreeMap::from([("a".into(), m)]),
        };
        assert!(!report.all_members_admitted());
        assert_eq!(report.degraded_members().len(), 1);
    }

    #[test]
    fn map_key_or_nested_health_identity_mismatch_fails_closed() {
        let keyed_wrong = FleetAssuranceReport {
            fleet_id: "fleet".into(),
            expected_generation: 4,
            members: BTreeMap::from([("b".into(), member("a", HealthSeverity::Healthy))]),
        };
        assert!(!keyed_wrong.all_members_admitted());
        assert_eq!(keyed_wrong.degraded_members().len(), 1);

        let mut nested_wrong = member("a", HealthSeverity::Healthy);
        nested_wrong.health.platform_id = "b".into();
        let report = FleetAssuranceReport {
            fleet_id: "fleet".into(),
            expected_generation: 4,
            members: BTreeMap::from([("a".into(), nested_wrong)]),
        };
        assert!(!report.all_members_admitted());
        assert_eq!(report.degraded_members().len(), 1);
    }

    #[test]
    fn empty_fleet_identity_cannot_claim_full_admission() {
        let report = FleetAssuranceReport {
            fleet_id: String::new(),
            expected_generation: 4,
            members: BTreeMap::from([("a".into(), member("a", HealthSeverity::Healthy))]),
        };
        assert!(!report.all_members_admitted());
    }
}
