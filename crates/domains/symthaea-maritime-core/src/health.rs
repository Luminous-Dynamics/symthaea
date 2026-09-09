// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Severity is ordered from fully healthy to locally unsafe.
/// `Unknown` is deliberately worse than an advisory: absence of health evidence
/// must not silently become a healthy result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HealthSeverity {
    Healthy,
    Advisory,
    Unknown,
    Degraded,
    Critical,
    Unsafe,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub component_id: String,
    pub severity: HealthSeverity,
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlatformHealth {
    pub platform_id: String,
    pub observed_at_ms: u64,
    pub components: Vec<ComponentHealth>,
}

impl PlatformHealth {
    /// Validate the identity/evidence shape without converting absent evidence into health.
    ///
    /// An empty component set remains representable and maps to `Unknown`; when components are
    /// present their identities must be canonical and unique so duplicate evidence cannot make a
    /// malformed report look fully observed.
    pub fn validate_shape(&self) -> Result<(), &'static str> {
        if self.platform_id.trim().is_empty() {
            return Err("platform_id must not be empty");
        }
        if self.platform_id.trim() != self.platform_id {
            return Err("platform_id must not contain outer whitespace");
        }

        let mut seen = BTreeSet::new();
        for component in &self.components {
            if component.component_id.trim().is_empty() || component.code.trim().is_empty() {
                return Err("component_id and code must not be empty");
            }
            if component.component_id.trim() != component.component_id
                || component.code.trim() != component.code
            {
                return Err("component_id and code must not contain outer whitespace");
            }
            if !seen.insert(component.component_id.as_str()) {
                return Err("component_id values must be unique within one health report");
            }
        }
        Ok(())
    }

    pub fn worst_severity(&self) -> HealthSeverity {
        self.components
            .iter()
            .map(|component| component.severity)
            .max()
            .unwrap_or(HealthSeverity::Unknown)
    }

    /// Fleet summaries must never hide a locally unsafe component behind an average.
    pub fn is_locally_safe(&self) -> bool {
        self.validate_shape().is_ok() && self.worst_severity() < HealthSeverity::Unsafe
    }

    /// Full-health claims require positive, well-formed evidence, not merely absence of faults.
    pub fn is_fully_observed_nominal(&self) -> bool {
        self.validate_shape().is_ok()
            && !self.components.is_empty()
            && self.worst_severity() <= HealthSeverity::Advisory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worst_component_controls_platform_health() {
        let report = PlatformHealth {
            platform_id: "usv-7".into(),
            observed_at_ms: 10,
            components: vec![
                ComponentHealth {
                    component_id: "nav".into(),
                    severity: HealthSeverity::Healthy,
                    code: "ok".into(),
                    detail: String::new(),
                },
                ComponentHealth {
                    component_id: "steering".into(),
                    severity: HealthSeverity::Unsafe,
                    code: "feedback_lost".into(),
                    detail: "rudder feedback unavailable".into(),
                },
            ],
        };
        assert_eq!(report.worst_severity(), HealthSeverity::Unsafe);
        assert!(!report.is_locally_safe());
        assert!(!report.is_fully_observed_nominal());
    }

    #[test]
    fn missing_health_evidence_is_unknown_not_healthy() {
        let report = PlatformHealth {
            platform_id: "usv-8".into(),
            observed_at_ms: 10,
            components: Vec::new(),
        };
        assert_eq!(report.validate_shape(), Ok(()));
        assert_eq!(report.worst_severity(), HealthSeverity::Unknown);
        assert!(!report.is_fully_observed_nominal());
    }

    #[test]
    fn duplicate_or_ambiguous_component_evidence_cannot_claim_nominal_health() {
        let duplicate = ComponentHealth {
            component_id: "nav".into(),
            severity: HealthSeverity::Healthy,
            code: "ok".into(),
            detail: String::new(),
        };
        let report = PlatformHealth {
            platform_id: "usv-9".into(),
            observed_at_ms: 10,
            components: vec![duplicate.clone(), duplicate],
        };
        assert!(report.validate_shape().is_err());
        assert!(!report.is_fully_observed_nominal());

        let malformed = PlatformHealth {
            platform_id: " usv-9".into(),
            observed_at_ms: 10,
            components: vec![ComponentHealth {
                component_id: "nav".into(),
                severity: HealthSeverity::Healthy,
                code: "ok".into(),
                detail: String::new(),
            }],
        };
        assert!(malformed.validate_shape().is_err());
        assert!(!malformed.is_locally_safe());
    }
}
