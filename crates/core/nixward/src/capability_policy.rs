// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Default-deny executable-to-network capability policy for Nixward.
//!
//! This module does not configure nftables, eBPF, a service mesh, or the
//! Network Twin. It answers one narrow question: given a resolved executable
//! identity, principal, purpose, time, and opaque network capability id, does an
//! explicit grant exist? Adapters remain responsible for resolving executable
//! symlinks and enforcing the resulting decision.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PrincipalRef(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NetworkCapabilityId(pub String);

/// Executable identity as observed by the host adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableIdentity {
    /// Exact resolved executable path under `/nix/store`.
    NixStorePath(String),
    /// An executable that could not be resolved to a trusted declared store path.
    Unknown {
        observed_path: Option<String>,
    },
}

impl ExecutableIdentity {
    pub fn concrete_nix_store_path(&self) -> Option<&str> {
        match self {
            Self::NixStorePath(path)
                if path.starts_with("/nix/store/")
                    && path.len() > "/nix/store/".len()
                    && !path.split('/').any(|part| part == "..") =>
            {
                Some(path)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrincipalSelector {
    Exact(PrincipalRef),
    AnyAuthenticated,
}

/// Time-bounded explicit authority for one executable identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityGrant {
    pub rule_id: String,
    /// Must be an exact resolved `/nix/store/...` executable path.
    pub executable_store_path: String,
    pub principal: PrincipalSelector,
    #[serde(default)]
    pub capabilities: BTreeSet<NetworkCapabilityId>,
    /// Optional exact purpose strings. Empty means the grant is not purpose-restricted.
    #[serde(default)]
    pub purposes: BTreeSet<String>,
    pub valid_from_unix_ms: Option<u64>,
    pub valid_until_unix_ms: Option<u64>,
}

impl CapabilityGrant {
    pub fn is_well_formed(&self) -> bool {
        let executable = ExecutableIdentity::NixStorePath(self.executable_store_path.clone());
        !self.rule_id.trim().is_empty()
            && executable.concrete_nix_store_path().is_some()
            && !self.capabilities.is_empty()
            && match (self.valid_from_unix_ms, self.valid_until_unix_ms) {
                (Some(start), Some(end)) => start <= end,
                _ => true,
            }
    }

    fn matches(&self, request: &CapabilityRequest, executable_store_path: &str) -> bool {
        if !self.is_well_formed() || self.executable_store_path != executable_store_path {
            return false;
        }

        let principal_matches = match &self.principal {
            PrincipalSelector::Exact(expected) => expected == &request.principal,
            PrincipalSelector::AnyAuthenticated => !request.principal.0.trim().is_empty(),
        };
        if !principal_matches || !self.capabilities.contains(&request.capability) {
            return false;
        }

        if !self.purposes.is_empty() && !self.purposes.contains(&request.purpose) {
            return false;
        }

        if let Some(start) = self.valid_from_unix_ms
            && request.observed_at_unix_ms < start
        {
            return false;
        }
        if let Some(end) = self.valid_until_unix_ms
            && request.observed_at_unix_ms > end
        {
            return false;
        }

        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityRequest {
    pub executable: ExecutableIdentity,
    pub principal: PrincipalRef,
    pub capability: NetworkCapabilityId,
    pub purpose: String,
    pub observed_at_unix_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PolicyMode {
    /// Compute and record decisions; adapter must not mutate enforcement state.
    Observe,
    /// Decision is eligible for enforcement by a separately authorized adapter.
    Enforce,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DenyReason {
    UnknownExecutableIdentity,
    InvalidExecutableIdentity,
    EmptyPrincipal,
    EmptyCapability,
    NoMatchingGrant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CapabilityDecision {
    Allow { rule_id: String },
    Deny { reason: DenyReason },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    pub mode: PolicyMode,
    pub decision: CapabilityDecision,
}

impl PolicyEvaluation {
    pub fn should_enforce_deny(&self) -> bool {
        self.mode == PolicyMode::Enforce
            && matches!(&self.decision, CapabilityDecision::Deny { .. })
    }
}

/// Policy is default-deny by construction; there is no allow-all switch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityPolicy {
    pub mode: PolicyMode,
    #[serde(default)]
    pub grants: Vec<CapabilityGrant>,
}

impl CapabilityPolicy {
    pub fn evaluate(&self, request: &CapabilityRequest) -> PolicyEvaluation {
        if request.principal.0.trim().is_empty() {
            return PolicyEvaluation {
                mode: self.mode,
                decision: CapabilityDecision::Deny {
                    reason: DenyReason::EmptyPrincipal,
                },
            };
        }

        if request.capability.0.trim().is_empty() {
            return PolicyEvaluation {
                mode: self.mode,
                decision: CapabilityDecision::Deny {
                    reason: DenyReason::EmptyCapability,
                },
            };
        }

        let executable_store_path = match &request.executable {
            ExecutableIdentity::Unknown { .. } => {
                return PolicyEvaluation {
                    mode: self.mode,
                    decision: CapabilityDecision::Deny {
                        reason: DenyReason::UnknownExecutableIdentity,
                    },
                };
            }
            identity => match identity.concrete_nix_store_path() {
                Some(path) => path,
                None => {
                    return PolicyEvaluation {
                        mode: self.mode,
                        decision: CapabilityDecision::Deny {
                            reason: DenyReason::InvalidExecutableIdentity,
                        },
                    };
                }
            },
        };

        let matching_rule = self
            .grants
            .iter()
            .filter(|grant| grant.matches(request, executable_store_path))
            .min_by(|left, right| left.rule_id.cmp(&right.rule_id));

        let decision = match matching_rule {
            Some(grant) => CapabilityDecision::Allow {
                rule_id: grant.rule_id.clone(),
            },
            None => CapabilityDecision::Deny {
                reason: DenyReason::NoMatchingGrant,
            },
        };

        PolicyEvaluation {
            mode: self.mode,
            decision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capability(value: &str) -> NetworkCapabilityId {
        NetworkCapabilityId(value.to_string())
    }

    fn request(path: ExecutableIdentity) -> CapabilityRequest {
        CapabilityRequest {
            executable: path,
            principal: PrincipalRef("did:example:alice".to_string()),
            capability: capability("svc:case-management"),
            purpose: "case-work".to_string(),
            observed_at_unix_ms: 1_500,
        }
    }

    fn policy(mode: PolicyMode) -> CapabilityPolicy {
        CapabilityPolicy {
            mode,
            grants: vec![CapabilityGrant {
                rule_id: "allow-case-client".to_string(),
                executable_store_path:
                    "/nix/store/abc123-case-client/bin/case-client".to_string(),
                principal: PrincipalSelector::Exact(PrincipalRef(
                    "did:example:alice".to_string(),
                )),
                capabilities: BTreeSet::from([capability("svc:case-management")]),
                purposes: BTreeSet::from(["case-work".to_string()]),
                valid_from_unix_ms: Some(1_000),
                valid_until_unix_ms: Some(2_000),
            }],
        }
    }

    #[test]
    fn exact_store_identity_and_grant_are_allowed() {
        let evaluation = policy(PolicyMode::Observe).evaluate(&request(
            ExecutableIdentity::NixStorePath(
                "/nix/store/abc123-case-client/bin/case-client".to_string(),
            ),
        ));

        assert_eq!(
            evaluation.decision,
            CapabilityDecision::Allow {
                rule_id: "allow-case-client".to_string()
            }
        );
        assert!(!evaluation.should_enforce_deny());
    }

    #[test]
    fn unknown_executable_is_default_denied() {
        let evaluation = policy(PolicyMode::Observe).evaluate(&request(
            ExecutableIdentity::Unknown {
                observed_path: Some("/tmp/ransomware".to_string()),
            },
        ));

        assert_eq!(
            evaluation.decision,
            CapabilityDecision::Deny {
                reason: DenyReason::UnknownExecutableIdentity
            }
        );
    }

    #[test]
    fn different_store_executable_does_not_inherit_grant() {
        let evaluation = policy(PolicyMode::Observe).evaluate(&request(
            ExecutableIdentity::NixStorePath(
                "/nix/store/other-tool/bin/other-tool".to_string(),
            ),
        ));

        assert_eq!(
            evaluation.decision,
            CapabilityDecision::Deny {
                reason: DenyReason::NoMatchingGrant
            }
        );
    }

    #[test]
    fn purpose_mismatch_is_denied() {
        let mut req = request(ExecutableIdentity::NixStorePath(
            "/nix/store/abc123-case-client/bin/case-client".to_string(),
        ));
        req.purpose = "bulk-export".to_string();

        assert!(matches!(
            policy(PolicyMode::Observe).evaluate(&req).decision,
            CapabilityDecision::Deny {
                reason: DenyReason::NoMatchingGrant
            }
        ));
    }

    #[test]
    fn expired_grant_is_denied() {
        let mut req = request(ExecutableIdentity::NixStorePath(
            "/nix/store/abc123-case-client/bin/case-client".to_string(),
        ));
        req.observed_at_unix_ms = 2_001;

        assert!(matches!(
            policy(PolicyMode::Observe).evaluate(&req).decision,
            CapabilityDecision::Deny {
                reason: DenyReason::NoMatchingGrant
            }
        ));
    }

    #[test]
    fn enforcement_mode_only_marks_denies_for_adapter_enforcement() {
        let evaluation = policy(PolicyMode::Enforce).evaluate(&request(
            ExecutableIdentity::Unknown {
                observed_path: Some("/tmp/unknown".to_string()),
            },
        ));

        assert!(evaluation.should_enforce_deny());
    }

    #[test]
    fn malformed_store_identity_is_denied() {
        let evaluation = policy(PolicyMode::Observe).evaluate(&request(
            ExecutableIdentity::NixStorePath("/usr/local/bin/tool".to_string()),
        ));

        assert_eq!(
            evaluation.decision,
            CapabilityDecision::Deny {
                reason: DenyReason::InvalidExecutableIdentity
            }
        );
    }

    #[test]
    fn invalid_grant_window_never_matches() {
        let mut p = policy(PolicyMode::Observe);
        p.grants[0].valid_from_unix_ms = Some(2_000);
        p.grants[0].valid_until_unix_ms = Some(1_000);

        let evaluation = p.evaluate(&request(ExecutableIdentity::NixStorePath(
            "/nix/store/abc123-case-client/bin/case-client".to_string(),
        )));

        assert!(matches!(
            evaluation.decision,
            CapabilityDecision::Deny {
                reason: DenyReason::NoMatchingGrant
            }
        ));
    }
}
