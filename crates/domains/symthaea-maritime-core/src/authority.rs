// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Mission-neutral capabilities intentionally exposed by maritime-core.
///
/// Lethal-force and target-engagement capabilities are deliberately absent from
/// this vocabulary; downstream applications cannot obtain them from this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaritimeCapability {
    ObserveEnvironment,
    Navigate,
    HoldPosition,
    ReturnToRecoveryPoint,
    Surface,
    Dock,
    RelayCommunications,
    CarryCargo,
    InspectInfrastructure,
    PerformMaintenance,
    PublishTelemetry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityContext {
    pub now_ms: u64,
    pub authority_epoch: u64,
    /// A lease expiry check is not meaningful when the platform cannot establish
    /// trusted time. Callers must fall back to platform-local safe behavior instead.
    pub trusted_time_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityLease {
    pub lease_id: String,
    pub platform_id: String,
    pub authority_epoch: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub capabilities: BTreeSet<MaritimeCapability>,
    /// Opaque binding to an externally authenticated authority/evidence record.
    pub evidence_binding: String,
}

impl AuthorityLease {
    pub fn permits(&self, capability: MaritimeCapability, context: AuthorityContext) -> bool {
        context.trusted_time_available
            && context.authority_epoch == self.authority_epoch
            && context.now_ms >= self.issued_at_ms
            && context.now_ms < self.expires_at_ms
            && self.capabilities.contains(&capability)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.lease_id.trim().is_empty() || self.platform_id.trim().is_empty() {
            return Err("lease_id and platform_id must not be empty");
        }
        if self.expires_at_ms <= self.issued_at_ms {
            return Err("authority lease must have a positive validity interval");
        }
        if self.capabilities.is_empty() {
            return Err("authority lease must contain at least one capability");
        }
        if self.evidence_binding.trim().is_empty() {
            return Err("evidence_binding must not be empty");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(now_ms: u64, epoch: u64) -> AuthorityContext {
        AuthorityContext {
            now_ms,
            authority_epoch: epoch,
            trusted_time_available: true,
        }
    }

    #[test]
    fn lease_is_epoch_time_and_capability_bounded() {
        let lease = AuthorityLease {
            lease_id: "lease-1".into(),
            platform_id: "usv-1".into(),
            authority_epoch: 9,
            issued_at_ms: 100,
            expires_at_ms: 200,
            capabilities: BTreeSet::from([MaritimeCapability::Navigate]),
            evidence_binding: "sha256:example".into(),
        };
        assert!(lease.permits(MaritimeCapability::Navigate, context(150, 9)));
        assert!(!lease.permits(MaritimeCapability::Navigate, context(200, 9)));
        assert!(!lease.permits(MaritimeCapability::Navigate, context(150, 10)));
        assert!(!lease.permits(MaritimeCapability::Dock, context(150, 9)));
    }

    #[test]
    fn lease_fails_closed_without_trusted_time() {
        let lease = AuthorityLease {
            lease_id: "lease-1".into(),
            platform_id: "usv-1".into(),
            authority_epoch: 9,
            issued_at_ms: 100,
            expires_at_ms: 200,
            capabilities: BTreeSet::from([MaritimeCapability::Navigate]),
            evidence_binding: "sha256:example".into(),
        };
        let mut ctx = context(150, 9);
        ctx.trusted_time_available = false;
        assert!(!lease.permits(MaritimeCapability::Navigate, ctx));
    }
}
