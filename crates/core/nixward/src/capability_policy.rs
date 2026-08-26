// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Default-deny executable-to-network capability policy for Nixward.
//!
//! This module does not configure nftables, eBPF, a service mesh, or the
//! Network Twin. It answers one narrow question: given a resolved executable
//! identity, authenticated principal, purpose, time, and opaque network
//! capability id, does one unambiguous explicit grant exist? Adapters remain
//! responsible for resolving executable symlinks and enforcing the resulting
//! decision.
//!
//! A policy decision is deliberately distinct from enforcement evidence. A
//! [`CapabilityDecisionReceipt`] proves what this evaluator decided for an
//! exact request and policy revision. A [`CapabilityEnforcementReceipt`] only
//! proves an enforced deny when an external adapter records that it actually
//! applied the deny and supplies evidence for that application.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CAPABILITY_DECISION_RECEIPT_SCHEMA_VERSION: u16 = 1;

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
    Unknown { observed_path: Option<String> },
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
            PrincipalSelector::AnyAuthenticated => true,
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
    /// Opaque locator/digest/reference proving the principal was authenticated
    /// by the authoritative identity/session boundary. A non-empty principal
    /// string by itself is never treated as authentication.
    pub principal_authentication_evidence: Option<String>,
    pub capability: NetworkCapabilityId,
    pub purpose: String,
    pub observed_at_unix_ms: u64,
}

impl CapabilityRequest {
    fn has_principal_authentication_evidence(&self) -> bool {
        self.principal_authentication_evidence
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
    }
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
    MissingPrincipalAuthenticationEvidence,
    EmptyCapability,
    EmptyPurpose,
    NoMatchingGrant,
    /// Multiple grants matched the same request. Fail closed instead of
    /// selecting one silently, because the authority provenance is ambiguous.
    AmbiguousMatchingGrant,
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

        if !request.has_principal_authentication_evidence() {
            return PolicyEvaluation {
                mode: self.mode,
                decision: CapabilityDecision::Deny {
                    reason: DenyReason::MissingPrincipalAuthenticationEvidence,
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

        if request.purpose.trim().is_empty() {
            return PolicyEvaluation {
                mode: self.mode,
                decision: CapabilityDecision::Deny {
                    reason: DenyReason::EmptyPurpose,
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

        let matching_rules: Vec<&CapabilityGrant> = self
            .grants
            .iter()
            .filter(|grant| grant.matches(request, executable_store_path))
            .collect();

        let decision = match matching_rules.as_slice() {
            [] => CapabilityDecision::Deny {
                reason: DenyReason::NoMatchingGrant,
            },
            [grant] => CapabilityDecision::Allow {
                rule_id: grant.rule_id.clone(),
            },
            _ => CapabilityDecision::Deny {
                reason: DenyReason::AmbiguousMatchingGrant,
            },
        };

        PolicyEvaluation {
            mode: self.mode,
            decision,
        }
    }

    /// Evaluate and bind the decision to an exact request id and immutable
    /// policy revision. This receipt records evaluator intent; it does not by
    /// itself prove that an external enforcement mechanism applied the result.
    pub fn evaluate_with_receipt(
        &self,
        request_id: impl Into<String>,
        policy_revision: impl Into<String>,
        request: &CapabilityRequest,
    ) -> CapabilityDecisionReceipt {
        CapabilityDecisionReceipt {
            schema_version: CAPABILITY_DECISION_RECEIPT_SCHEMA_VERSION,
            request_id: request_id.into(),
            policy_revision: policy_revision.into(),
            request: request.clone(),
            evaluation: self.evaluate(request),
        }
    }
}

/// Evidence of the policy evaluator's exact input and output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityDecisionReceipt {
    pub schema_version: u16,
    pub request_id: String,
    /// Git revision, policy digest, signed manifest id, or equivalent immutable
    /// identity for the policy evaluated.
    pub policy_revision: String,
    pub request: CapabilityRequest,
    pub evaluation: PolicyEvaluation,
}

impl CapabilityDecisionReceipt {
    pub fn is_complete(&self) -> bool {
        self.schema_version == CAPABILITY_DECISION_RECEIPT_SCHEMA_VERSION
            && !self.request_id.trim().is_empty()
            && !self.policy_revision.trim().is_empty()
    }
}

/// What an external enforcement adapter did with a policy decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EnforcementResult {
    /// Observe mode, authorization missing, or adapter deliberately did not act.
    NotAttempted,
    /// Adapter reports that the requested policy result was applied.
    Applied,
    /// Adapter attempted application and failed.
    Failed,
}

/// Adapter-owned evidence layered on top of a Nixward decision receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEnforcementReceipt {
    pub decision_receipt: CapabilityDecisionReceipt,
    /// Immutable build/revision identity of the enforcing adapter.
    pub adapter_revision: String,
    pub result: EnforcementResult,
    /// Locator and digest for independently inspectable enforcement evidence.
    pub evidence_locator: Option<String>,
    pub evidence_digest: Option<String>,
}

impl CapabilityEnforcementReceipt {
    /// True only when a complete enforce-mode deny decision was actually
    /// applied by an identified adapter and carries inspectable evidence.
    pub fn proves_enforced_deny(&self) -> bool {
        let evidence_complete = self
            .evidence_locator
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
            && self
                .evidence_digest
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty());

        self.decision_receipt.is_complete()
            && !self.adapter_revision.trim().is_empty()
            && self.result == EnforcementResult::Applied
            && evidence_complete
            && self.decision_receipt.evaluation.mode == PolicyMode::Enforce
            && matches!(
                &self.decision_receipt.evaluation.decision,
                CapabilityDecision::Deny { .. }
            )
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
            principal_authentication_evidence: Some("xenia-session:receipt-001".to_string()),
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
    fn exact_store_identity_authenticated_principal_and_grant_are_allowed() {
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
    fn missing_principal_authentication_evidence_is_denied() {
        let mut req = request(ExecutableIdentity::NixStorePath(
            "/nix/store/abc123-case-client/bin/case-client".to_string(),
        ));
        req.principal_authentication_evidence = None;

        assert_eq!(
            policy(PolicyMode::Observe).evaluate(&req).decision,
            CapabilityDecision::Deny {
                reason: DenyReason::MissingPrincipalAuthenticationEvidence
            }
        );
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
    fn empty_purpose_is_denied_even_for_unrestricted_grant() {
        let mut p = policy(PolicyMode::Observe);
        p.grants[0].purposes.clear();
        let mut req = request(ExecutableIdentity::NixStorePath(
            "/nix/store/abc123-case-client/bin/case-client".to_string(),
        ));
        req.purpose.clear();

        assert_eq!(
            p.evaluate(&req).decision,
            CapabilityDecision::Deny {
                reason: DenyReason::EmptyPurpose
            }
        );
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
    fn overlapping_grants_fail_closed_as_ambiguous() {
        let mut p = policy(PolicyMode::Observe);
        let mut second = p.grants[0].clone();
        second.rule_id = "also-allows-case-client".to_string();
        p.grants.push(second);

        let evaluation = p.evaluate(&request(ExecutableIdentity::NixStorePath(
            "/nix/store/abc123-case-client/bin/case-client".to_string(),
        )));

        assert_eq!(
            evaluation.decision,
            CapabilityDecision::Deny {
                reason: DenyReason::AmbiguousMatchingGrant
            }
        );
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

    #[test]
    fn decision_receipt_binds_request_and_policy_revision() {
        let req = request(ExecutableIdentity::NixStorePath(
            "/nix/store/abc123-case-client/bin/case-client".to_string(),
        ));
        let receipt = policy(PolicyMode::Observe).evaluate_with_receipt(
            "request-001",
            "git:policy-abc123",
            &req,
        );

        assert!(receipt.is_complete());
        assert_eq!(receipt.request, req);
        assert_eq!(receipt.policy_revision, "git:policy-abc123");
        assert!(matches!(receipt.evaluation.decision, CapabilityDecision::Allow { .. }));
    }

    #[test]
    fn decision_receipt_alone_does_not_prove_enforcement() {
        let req = request(ExecutableIdentity::Unknown {
            observed_path: Some("/tmp/unknown".to_string()),
        });
        let decision = policy(PolicyMode::Enforce).evaluate_with_receipt(
            "request-002",
            "git:policy-abc123",
            &req,
        );
        assert!(decision.evaluation.should_enforce_deny());

        let enforcement = CapabilityEnforcementReceipt {
            decision_receipt: decision,
            adapter_revision: "git:adapter-def456".to_string(),
            result: EnforcementResult::NotAttempted,
            evidence_locator: None,
            evidence_digest: None,
        };

        assert!(!enforcement.proves_enforced_deny());
    }

    #[test]
    fn applied_enforce_mode_deny_with_evidence_proves_enforcement() {
        let req = request(ExecutableIdentity::Unknown {
            observed_path: Some("/tmp/unknown".to_string()),
        });
        let decision = policy(PolicyMode::Enforce).evaluate_with_receipt(
            "request-003",
            "git:policy-abc123",
            &req,
        );
        let enforcement = CapabilityEnforcementReceipt {
            decision_receipt: decision,
            adapter_revision: "git:adapter-def456".to_string(),
            result: EnforcementResult::Applied,
            evidence_locator: Some("receipt:nftables-001".to_string()),
            evidence_digest: Some("sha256:deadbeef".to_string()),
        };

        assert!(enforcement.proves_enforced_deny());
    }
}
