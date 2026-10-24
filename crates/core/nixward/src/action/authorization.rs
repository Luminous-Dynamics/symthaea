// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! In-process, one-shot authorization primitives for NixOS command dispatch.
//!
//! A permit is bound to the exact executable and ordered argv vector,
//! principal, action scope, authority kind, and a unique decision ID. A ledger
//! consumes a permit on its first dispatch attempt, including a mismatched
//! attempt, so an attacker cannot probe multiple commands with one permit.
//!
//! IMPORTANT BOUNDARY: this is an in-memory, same-process primitive. It is
//! not a signed credential, does not authenticate the code that calls issue,
//! and does not itself intercept NixOSExecutor process dispatch. Callers must
//! issue permits only after their trusted approval/policy gate. Cross-process
//! use requires a separately designed authenticated receipt protocol.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT_LEDGER_ID: AtomicU64 = AtomicU64::new(1);

/// Default maximum time between issuance and dispatch of an in-process permit.
pub const DEFAULT_MAX_PERMIT_LIFETIME: Duration = Duration::from_secs(30);

/// The action domain covered by a permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionScope {
    ReadOnly,
    UserModify,
    SystemModify,
    SystemCritical,
    Destructive,
    Rollback,
    Recovery,
}

/// The source of authority for a particular decision.
///
/// These variants intentionally distinguish operator approvals from
/// compensating actions. A rollback permit is not a human approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuthorizationAuthority {
    HumanOperator,
    WatchdogOperator,
    ReadOnlyPolicy,
    RollbackPolicy,
    RecoveryPolicy,
}

/// Exact process invocation. Arguments are kept as an ordered vector; they are
/// never joined into a shell string for comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CommandBinding {
    executable: String,
    args: Vec<String>,
}

impl CommandBinding {
    pub fn new<I, S>(
        executable: impl Into<String>,
        args: I,
    ) -> Result<Self, AuthorizationError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let executable = executable.into();
        let args = args.into_iter().map(Into::into).collect::<Vec<_>>();

        if executable.trim().is_empty() {
            return Err(AuthorizationError::EmptyExecutable);
        }
        if executable.contains('\0') || args.iter().any(|arg| arg.contains('\0')) {
            return Err(AuthorizationError::EmbeddedNul);
        }

        Ok(Self { executable, args })
    }

    pub fn executable(&self) -> &str {
        &self.executable
    }

    pub fn args(&self) -> &[String] {
        &self.args
    }
}

/// Request for a short-lived capability.
///
/// Construct this only after the trusted decision point has established the
/// approval/policy facts. The ledger enforces shape, lifetime, uniqueness and
/// use semantics; it cannot prove that the supplied decision is truthful.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizationRequest {
    binding: CommandBinding,
    principal: String,
    scope: ActionScope,
    authority: AuthorizationAuthority,
    decision_id: String,
    lifetime: Duration,
}

impl AuthorizationRequest {
    pub fn new(
        binding: CommandBinding,
        principal: impl Into<String>,
        scope: ActionScope,
        authority: AuthorizationAuthority,
        decision_id: impl Into<String>,
        lifetime: Duration,
    ) -> Result<Self, AuthorizationError> {
        let principal = principal.into();
        let decision_id = decision_id.into();

        if principal.trim().is_empty() {
            return Err(AuthorizationError::EmptyPrincipal);
        }
        if decision_id.trim().is_empty() {
            return Err(AuthorizationError::EmptyDecisionId);
        }
        if lifetime.is_zero() {
            return Err(AuthorizationError::ZeroLifetime);
        }

        Ok(Self {
            binding,
            principal,
            scope,
            authority,
            decision_id,
            lifetime,
        })
    }
}

/// What the dispatcher is actually about to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizationAttempt {
    binding: CommandBinding,
    principal: String,
    scope: ActionScope,
    authority: AuthorizationAuthority,
}

impl AuthorizationAttempt {
    pub fn new(
        binding: CommandBinding,
        principal: impl Into<String>,
        scope: ActionScope,
        authority: AuthorizationAuthority,
    ) -> Result<Self, AuthorizationError> {
        let principal = principal.into();
        if principal.trim().is_empty() {
            return Err(AuthorizationError::EmptyPrincipal);
        }

        Ok(Self {
            binding,
            principal,
            scope,
            authority,
        })
    }
}

/// Opaque, non-cloneable handle to a ledger entry.
///
/// The handle is process-local and intentionally does not implement Serialize
/// or Clone. The ledger is still authoritative: it rejects wrong-ledger use,
/// revoked decisions, expired permits and repeated consumption.
#[derive(Debug)]
pub struct ExecutionPermit {
    ledger_id: u64,
    permit_id: u64,
}

/// Local evidence returned after a successful one-shot consumption.
///
/// This is useful for an execution record but is not a cryptographic
/// attestation and must not be accepted as one by another process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizationReceipt {
    decision_id: String,
    principal: String,
    scope: ActionScope,
    authority: AuthorizationAuthority,
    binding: CommandBinding,
}

impl AuthorizationReceipt {
    pub fn decision_id(&self) -> &str {
        &self.decision_id
    }

    pub fn principal(&self) -> &str {
        &self.principal
    }

    pub fn scope(&self) -> ActionScope {
        self.scope
    }

    pub fn authority(&self) -> AuthorizationAuthority {
        self.authority
    }

    pub fn binding(&self) -> &CommandBinding {
        &self.binding
    }
}

#[derive(Debug)]
struct PermitRecord {
    request: AuthorizationRequest,
    expires_at: Instant,
}

/// Process-local issuance, revocation and one-shot consumption ledger.
#[derive(Debug)]
pub struct ExecutionAuthorizationLedger {
    ledger_id: u64,
    max_lifetime: Duration,
    next_permit_id: u64,
    records: HashMap<u64, PermitRecord>,
    issued_decisions: HashSet<String>,
    revoked_decisions: HashSet<String>,
    consumed: HashSet<u64>,
}

impl Default for ExecutionAuthorizationLedger {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_PERMIT_LIFETIME)
            .expect("the default permit lifetime is valid")
    }
}

impl ExecutionAuthorizationLedger {
    pub fn new(max_lifetime: Duration) -> Result<Self, AuthorizationError> {
        if max_lifetime.is_zero() {
            return Err(AuthorizationError::ZeroMaximumLifetime);
        }

        let ledger_id = NEXT_LEDGER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| AuthorizationError::LedgerIdExhausted)?;

        Ok(Self {
            ledger_id,
            max_lifetime,
            next_permit_id: 1,
            records: HashMap::new(),
            issued_decisions: HashSet::new(),
            revoked_decisions: HashSet::new(),
            consumed: HashSet::new(),
        })
    }

    /// Issue a local permit after the trusted decision point approves a
    /// command. Decision IDs may be issued only once in this ledger.
    pub fn issue(
        &mut self,
        request: AuthorizationRequest,
    ) -> Result<ExecutionPermit, AuthorizationError> {
        if request.lifetime > self.max_lifetime {
            return Err(AuthorizationError::LifetimeTooLong);
        }
        if self.revoked_decisions.contains(&request.decision_id) {
            return Err(AuthorizationError::Revoked);
        }
        if self.issued_decisions.contains(&request.decision_id) {
            return Err(AuthorizationError::DuplicateDecision);
        }

        let expires_at = Instant::now()
            .checked_add(request.lifetime)
            .ok_or(AuthorizationError::LifetimeOverflow)?;
        let permit_id = self.next_permit_id;
        let next_permit_id = permit_id
            .checked_add(1)
            .ok_or(AuthorizationError::PermitIdExhausted)?;

        self.issued_decisions.insert(request.decision_id.clone());
        self.next_permit_id = next_permit_id;
        self.records.insert(
            permit_id,
            PermitRecord {
                request,
                expires_at,
            },
        );

        Ok(ExecutionPermit {
            ledger_id: self.ledger_id,
            permit_id,
        })
    }

    /// Revoke all permits for one decision ID. A consumed permit cannot be
    /// made usable again by revoking or re-issuing the decision.
    pub fn revoke_decision(&mut self, decision_id: &str) -> bool {
        if !self.issued_decisions.contains(decision_id) {
            return false;
        }
        self.revoked_decisions.insert(decision_id.to_owned())
    }

    /// Consume a permit against the invocation the dispatcher is about to run.
    ///
    /// The permit is marked consumed before checking the request fields. Thus
    /// any failed dispatch attempt burns it (fail-closed) and cannot be used to
    /// probe for a matching command, principal, scope or authority.
    pub fn consume(
        &mut self,
        permit: &ExecutionPermit,
        attempt: &AuthorizationAttempt,
    ) -> Result<AuthorizationReceipt, AuthorizationError> {
        if permit.ledger_id != self.ledger_id {
            return Err(AuthorizationError::WrongLedger);
        }
        if !self.records.contains_key(&permit.permit_id) {
            return Err(AuthorizationError::UnknownPermit);
        }
        if self.consumed.contains(&permit.permit_id) {
            return Err(AuthorizationError::Replay);
        }
        // Burn on the first attempt, including expired, revoked, or mismatched
        // attempts, so the same capability can never be retried.
        self.consumed.insert(permit.permit_id);

        let record = self
            .records
            .get(&permit.permit_id)
            .expect("record existence checked above");
        let request = &record.request;

        if self.revoked_decisions.contains(&request.decision_id) {
            return Err(AuthorizationError::Revoked);
        }
        if Instant::now() >= record.expires_at {
            return Err(AuthorizationError::Expired);
        }
        if request.binding != attempt.binding {
            return Err(AuthorizationError::CommandMismatch);
        }
        if request.principal != attempt.principal {
            return Err(AuthorizationError::PrincipalMismatch);
        }
        if request.scope != attempt.scope {
            return Err(AuthorizationError::ScopeMismatch);
        }
        if request.authority != attempt.authority {
            return Err(AuthorizationError::AuthorityMismatch);
        }

        Ok(AuthorizationReceipt {
            decision_id: request.decision_id.clone(),
            principal: request.principal.clone(),
            scope: request.scope,
            authority: request.authority,
            binding: request.binding.clone(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AuthorizationError {
    #[error("the executable must not be empty")]
    EmptyExecutable,
    #[error("executable or argument contains an OS-invalid NUL byte")]
    EmbeddedNul,
    #[error("the principal must not be empty")]
    EmptyPrincipal,
    #[error("the decision ID must not be empty")]
    EmptyDecisionId,
    #[error("permit lifetime must be positive")]
    ZeroLifetime,
    #[error("maximum permit lifetime must be positive")]
    ZeroMaximumLifetime,
    #[error("requested permit lifetime exceeds the ledger maximum")]
    LifetimeTooLong,
    #[error("permit lifetime overflowed the monotonic clock representation")]
    LifetimeOverflow,
    #[error("this decision ID has already been issued")]
    DuplicateDecision,
    #[error("this decision has been revoked")]
    Revoked,
    #[error("the permit expired before dispatch")]
    Expired,
    #[error("permit belongs to a different ledger")]
    WrongLedger,
    #[error("permit does not exist in this ledger")]
    UnknownPermit,
    #[error("permit was already consumed")]
    Replay,
    #[error("actual executable or ordered arguments differ from the permit")]
    CommandMismatch,
    #[error("actual principal differs from the permit")]
    PrincipalMismatch,
    #[error("actual action scope differs from the permit")]
    ScopeMismatch,
    #[error("actual authority kind differs from the permit")]
    AuthorityMismatch,
    #[error("permit identifier space exhausted")]
    PermitIdExhausted,
    #[error("ledger identifier space exhausted")]
    LedgerIdExhausted,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(args: &[&str]) -> CommandBinding {
        CommandBinding::new("nixos-rebuild", args.iter().copied()).unwrap()
    }

    fn request(
        command: CommandBinding,
        decision_id: &str,
        authority: AuthorizationAuthority,
        scope: ActionScope,
    ) -> AuthorizationRequest {
        AuthorizationRequest::new(
            command,
            "operator:alice",
            scope,
            authority,
            decision_id,
            Duration::from_secs(5),
        )
        .unwrap()
    }

    fn attempt(
        command: CommandBinding,
        authority: AuthorizationAuthority,
        scope: ActionScope,
    ) -> AuthorizationAttempt {
        AuthorizationAttempt::new(command, "operator:alice", scope, authority).unwrap()
    }

    #[test]
    fn binding_preserves_argument_boundaries_and_order() {
        let first = binding(&["switch", "--flake", ".#host", "--", "arg with spaces"]);
        let same = binding(&["switch", "--flake", ".#host", "--", "arg with spaces"]);
        let different = binding(&["switch", "--flake", ".#host", "--", "arg", "with", "spaces"]);

        assert_eq!(first, same);
        assert_ne!(first, different);
        assert_eq!(first.args().last().map(String::as_str), Some("arg with spaces"));
    }

    #[test]
    fn valid_permit_is_consumed_exactly_once() {
        let command = binding(&["switch", "--flake", ".#host"]);
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger
            .issue(request(
                command.clone(),
                "decision-001",
                AuthorizationAuthority::HumanOperator,
                ActionScope::SystemCritical,
            ))
            .unwrap();

        let receipt = ledger
            .consume(
                &permit,
                &attempt(
                    command.clone(),
                    AuthorizationAuthority::HumanOperator,
                    ActionScope::SystemCritical,
                ),
            )
            .unwrap();

        assert_eq!(receipt.decision_id(), "decision-001");
        assert_eq!(receipt.binding(), &command);
        assert_eq!(
            ledger.consume(
                &permit,
                &attempt(
                    command,
                    AuthorizationAuthority::HumanOperator,
                    ActionScope::SystemCritical,
                ),
            ),
            Err(AuthorizationError::Replay)
        );
    }

    #[test]
    fn command_mismatch_fails_closed_and_burns_the_permit() {
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger
            .issue(request(
                binding(&["switch", "--flake", ".#approved"]),
                "decision-002",
                AuthorizationAuthority::HumanOperator,
                ActionScope::SystemCritical,
            ))
            .unwrap();

        assert_eq!(
            ledger.consume(
                &permit,
                &attempt(
                    binding(&["switch", "--flake", ".#different"]),
                    AuthorizationAuthority::HumanOperator,
                    ActionScope::SystemCritical,
                ),
            ),
            Err(AuthorizationError::CommandMismatch)
        );
        assert_eq!(
            ledger.consume(
                &permit,
                &attempt(
                    binding(&["switch", "--flake", ".#approved"]),
                    AuthorizationAuthority::HumanOperator,
                    ActionScope::SystemCritical,
                ),
            ),
            Err(AuthorizationError::Replay)
        );
    }

    #[test]
    fn principal_mismatch_fails_closed() {
        let command = binding(&["-d", "--delete-old"]);
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger
            .issue(request(
                command.clone(),
                "decision-003",
                AuthorizationAuthority::HumanOperator,
                ActionScope::Destructive,
            ))
            .unwrap();

        let wrong_principal = AuthorizationAttempt::new(
            command,
            "operator:bob",
            ActionScope::Destructive,
            AuthorizationAuthority::HumanOperator,
        )
        .unwrap();
        assert_eq!(
            ledger.consume(&permit, &wrong_principal),
            Err(AuthorizationError::PrincipalMismatch)
        );
    }

    #[test]
    fn scope_mismatch_fails_closed() {
        let command = binding(&["switch"]);
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger
            .issue(request(
                command.clone(),
                "decision-004",
                AuthorizationAuthority::HumanOperator,
                ActionScope::SystemCritical,
            ))
            .unwrap();
        assert_eq!(
            ledger.consume(
                &permit,
                &attempt(
                    command,
                    AuthorizationAuthority::HumanOperator,
                    ActionScope::UserModify,
                ),
            ),
            Err(AuthorizationError::ScopeMismatch)
        );
    }

    #[test]
    fn rollback_authority_cannot_substitute_for_operator_authority() {
        let command = binding(&["switch", "--rollback"]);
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger
            .issue(request(
                command.clone(),
                "decision-005",
                AuthorizationAuthority::RollbackPolicy,
                ActionScope::Rollback,
            ))
            .unwrap();

        assert_eq!(
            ledger.consume(
                &permit,
                &attempt(
                    command,
                    AuthorizationAuthority::HumanOperator,
                    ActionScope::Rollback,
                ),
            ),
            Err(AuthorizationError::AuthorityMismatch)
        );
    }

    #[test]
    fn revoked_decision_cannot_be_consumed() {
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger
            .issue(request(
                binding(&["switch"]),
                "decision-006",
                AuthorizationAuthority::WatchdogOperator,
                ActionScope::SystemCritical,
            ))
            .unwrap();

        assert!(ledger.revoke_decision("decision-006"));
        assert_eq!(
            ledger.consume(
                &permit,
                &attempt(
                    binding(&["switch"]),
                    AuthorizationAuthority::WatchdogOperator,
                    ActionScope::SystemCritical,
                ),
            ),
            Err(AuthorizationError::Revoked)
        );
    }

    #[test]
    fn expired_permit_is_rejected() {
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger
            .issue(request(
                binding(&["switch"]),
                "decision-007",
                AuthorizationAuthority::HumanOperator,
                ActionScope::SystemCritical,
            ))
            .unwrap();
        ledger.records.get_mut(&permit.permit_id).unwrap().expires_at =
            Instant::now() - Duration::from_millis(1);

        assert_eq!(
            ledger.consume(
                &permit,
                &attempt(
                    binding(&["switch"]),
                    AuthorizationAuthority::HumanOperator,
                    ActionScope::SystemCritical,
                ),
            ),
            Err(AuthorizationError::Expired)
        );
    }

    #[test]
    fn duplicate_decision_id_cannot_mint_another_permit() {
        let mut ledger = ExecutionAuthorizationLedger::default();
        ledger
            .issue(request(
                binding(&["switch", "--flake", ".#host"]),
                "decision-008",
                AuthorizationAuthority::HumanOperator,
                ActionScope::SystemCritical,
            ))
            .unwrap();

        assert_eq!(
            ledger
                .issue(request(
                    binding(&["switch", "--flake", ".#other"]),
                    "decision-008",
                    AuthorizationAuthority::HumanOperator,
                    ActionScope::SystemCritical,
                ))
                .unwrap_err(),
            AuthorizationError::DuplicateDecision
        );
    }

    #[test]
    fn permits_are_bound_to_the_ledger_that_issued_them() {
        let mut first = ExecutionAuthorizationLedger::default();
        let mut second = ExecutionAuthorizationLedger::default();
        let command = binding(&["search", "vim"]);
        let permit = first
            .issue(request(
                command.clone(),
                "decision-009",
                AuthorizationAuthority::ReadOnlyPolicy,
                ActionScope::ReadOnly,
            ))
            .unwrap();

        assert_eq!(
            second.consume(
                &permit,
                &attempt(
                    command,
                    AuthorizationAuthority::ReadOnlyPolicy,
                    ActionScope::ReadOnly,
                ),
            ),
            Err(AuthorizationError::WrongLedger)
        );
    }

    #[test]
    fn invalid_executable_arguments_and_lifetimes_are_rejected() {
        assert_eq!(
            CommandBinding::new("  ", Vec::<String>::new()),
            Err(AuthorizationError::EmptyExecutable)
        );
        assert_eq!(
            CommandBinding::new("nix", ["flake\0update"]),
            Err(AuthorizationError::EmbeddedNul)
        );
        assert_eq!(
            AuthorizationRequest::new(
                binding(&["switch"]),
                "operator:alice",
                ActionScope::SystemCritical,
                AuthorizationAuthority::HumanOperator,
                "decision-010",
                Duration::ZERO,
            ),
            Err(AuthorizationError::ZeroLifetime)
        );
        assert_eq!(
            ExecutionAuthorizationLedger::new(Duration::ZERO).unwrap_err(),
            AuthorizationError::ZeroMaximumLifetime
        );
    }
}
