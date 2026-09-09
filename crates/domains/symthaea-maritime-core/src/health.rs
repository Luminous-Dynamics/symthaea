// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

/// Severity is ordered from fully healthy to locally unsafe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HealthSeverity {
    Healthy,
    Advisory,
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
    pub fn worst_severity(&self) -> HealthSeverity {
        self.components
            .iter()
            .map(|component| component.severity)
            .max()
            .unwrap_or(HealthSeverity::Healthy)
    }

    /// Fleet summaries must never hide a locally unsafe component behind an average.
    pub fn is_locally_safe(&self) -> bool {
        self.worst_severity() < HealthSeverity::Unsafe
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
    }
}
