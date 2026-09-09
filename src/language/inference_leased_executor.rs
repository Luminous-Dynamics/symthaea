// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final leased remote-provider execution boundary.
//!
//! The wrapper owns current-profile execution plus the combined IF-7/IF-12 resource
//! authority. Resource reservation and IF-2 permit preparation are joined at the
//! public boundary so a failed permit cannot strand a naked active lease.

#[cfg(not(test))]
use super::inference_binding::QuotaStateBinding;
#[cfg(not(test))]
use super::inference_contract::{AdmittedInferenceRoute, InferencePolicy, InferenceRequest};
#[cfg(not(test))]
use super::inference_current_profile_executor::CurrentProfileCredentialExecutor;
#[cfg(not(test))]
use super::inference_execution_envelope::InferenceGenerationControls;
#[cfg(not(test))]
use super::inference_executor::{
    InferenceExecutionFailure, InferenceExecutionOutcome, InferenceExecutorFatalError,
    InferenceTickSource,
};
#[cfg(not(test))]
use super::inference_permit::{
    InferenceExecutionBinding, InferencePermitError, InferencePermitIssuer,
    PreparedInferenceExecution,
};
#[cfg(not(test))]
use super::inference_profile_execution::ProviderProfileResolver;
#[cfg(not(test))]
use super::inference_receipt::{InferenceFailureClass, InferenceReceipt, InferenceReceiptError};
#[cfg(not(test))]
use super::inference_resource_execution::{
    InferenceResourceExecutionError, admit_and_bind_resource_execution,
};
#[cfg(not(test))]
use super::inference_resource_guard::InferenceResourceGuard;
#[cfg(not(test))]
use super::inference_resource_lease::{
    InferenceExecutionResourceAuthority, InferenceExecutionResourceLease,
    InferenceResourceLeaseError, bind_active_resource_lease,
};
#[cfg(not(test))]
use super::inference_resource_scope::InferenceResourceScopeRegistry;

#[cfg(test)]
use crate::inference_binding::QuotaStateBinding;
#[cfg(test)]
use crate::inference_contract::{AdmittedInferenceRoute, InferencePolicy, InferenceRequest};
#[cfg(test)]
use crate::inference_current_profile_executor::CurrentProfileCredentialExecutor;
#[cfg(test)]
use crate::inference_execution_envelope::InferenceGenerationControls;
#[cfg(test)]
use crate::inference_executor::{
    InferenceExecutionFailure, InferenceExecutionOutcome, InferenceExecutorFatalError,
    InferenceTickSource,
};
#[cfg(test)]
use crate::inference_permit::{
    InferenceExecutionBinding, InferencePermitError, InferencePermitIssuer,
    PreparedInferenceExecution,
};
#[cfg(test)]
use crate::inference_profile_execution::ProviderProfileResolver;
#[cfg(test)]
use crate::inference_receipt::{InferenceFailureClass, InferenceReceipt, InferenceReceiptError};
#[cfg(test)]
use crate::inference_resource_execution::{
    InferenceResourceExecutionError, admit_and_bind_resource_execution,
};
#[cfg(test)]
use crate::inference_resource_guard::InferenceResourceGuard;
#[cfg(test)]
use crate::inference_resource_lease::{
    InferenceExecutionResourceAuthority, InferenceExecutionResourceLease,
    InferenceResourceLeaseError, bind_active_resource_lease,
};
#[cfg(test)]
use crate::inference_resource_scope::InferenceResourceScopeRegistry;

use std::fmt;
use std::sync::{Arc, Mutex};

/// Trusted monotonic millisecond source used only for resource guard cooldown and
/// settlement timing. It is separate from IF-11's abstract trusted tick source.
pub trait InferenceResourceMillisSource: Send + Sync {
    fn now_millis(&self) -> u64;
}

/// Crate-internal intermediate. Public callers never receive a naked active lease;
/// `prepare_leased_current` immediately joins it to a prepared IF-2 capability.
pub(crate) struct LeasedResourceExecutableAdmission {
    route: AdmittedInferenceRoute,
    binding: InferenceExecutionBinding,
    lease: InferenceExecutionResourceLease,
}

impl fmt::Debug for LeasedResourceExecutableAdmission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeasedResourceExecutableAdmission")
            .field("route", &self.route)
            .field("binding", &self.binding)
            .field("lease", &self.lease)
            .finish()
    }
}

impl LeasedResourceExecutableAdmission {
    pub(crate) fn route(&self) -> &AdmittedInferenceRoute {
        &self.route
    }

    pub(crate) const fn binding(&self) -> &InferenceExecutionBinding {
        &self.binding
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        AdmittedInferenceRoute,
        InferenceExecutionBinding,
        InferenceExecutionResourceLease,
    ) {
        (self.route, self.binding, self.lease)
    }
}

/// Public authority-bearing capability. It joins the exact prepared IF-2 attempt
/// to the one IF-15 resource lease that reserved its request/token exposure.
/// Intentionally non-Clone/non-Serde.
pub struct PreparedLeasedInferenceExecution {
    route: AdmittedInferenceRoute,
    prepared: PreparedInferenceExecution,
    lease: InferenceExecutionResourceLease,
}

impl fmt::Debug for PreparedLeasedInferenceExecution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedLeasedInferenceExecution")
            .field("route", &self.route)
            .field("prepared", &self.prepared)
            .field("lease", &self.lease)
            .finish()
    }
}

impl PreparedLeasedInferenceExecution {
    pub fn route(&self) -> &AdmittedInferenceRoute {
        &self.route
    }

    pub const fn binding(&self) -> &InferenceExecutionBinding {
        self.prepared.binding()
    }

    pub const fn permit_generation(&self) -> u64 {
        self.prepared.generation()
    }
}

/// Highest-level v1 remote execution wrapper. Neither the resource guard nor scope
/// registry is exposed mutably after construction.
pub struct LeasedCurrentProfileExecutor<C: ?Sized, R, T: ?Sized> {
    current: CurrentProfileCredentialExecutor<C, R>,
    authority: Mutex<InferenceExecutionResourceAuthority>,
    resource_clock: Arc<T>,
}

impl<C: InferenceTickSource + ?Sized, R, T: InferenceResourceMillisSource + ?Sized> fmt::Debug
    for LeasedCurrentProfileExecutor<C, R, T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let active = self
            .authority
            .lock()
            .map(|authority| authority.active_lease_count())
            .ok();
        f.debug_struct("LeasedCurrentProfileExecutor")
            .field("current", &self.current)
            .field("active_lease_count", &active)
            .field("resource_clock_present", &true)
            .finish()
    }
}

impl<
        C: InferenceTickSource + ?Sized,
        R: ProviderProfileResolver,
        T: InferenceResourceMillisSource + ?Sized,
    > LeasedCurrentProfileExecutor<C, R, T>
{
    pub fn new(
        current: CurrentProfileCredentialExecutor<C, R>,
        authority: InferenceExecutionResourceAuthority,
        resource_clock: Arc<T>,
    ) -> Self {
        Self {
            current,
            authority: Mutex::new(authority),
            resource_clock,
        }
    }

    pub fn active_lease_count(&self) -> Result<usize, InferenceLeasedExecutorError> {
        Ok(self.lock_authority()?.active_lease_count())
    }

    pub fn effective_remaining_requests(
        &self,
    ) -> Result<Option<u64>, InferenceLeasedExecutorError> {
        Ok(self.lock_authority()?.effective_remaining_requests())
    }

    pub fn effective_remaining_tokens(
        &self,
    ) -> Result<Option<u64>, InferenceLeasedExecutorError> {
        Ok(self.lock_authority()?.effective_remaining_tokens())
    }

    /// Replace guard + scope authority as one state transition. Active leases make
    /// replacement fail closed, freezing their local resource authority across I/O.
    pub fn replace_resource_state(
        &self,
        guard: InferenceResourceGuard,
        scopes: InferenceResourceScopeRegistry,
    ) -> Result<(), InferenceLeasedExecutorError> {
        self.lock_authority()?
            .replace_state(guard, scopes)
            .map_err(InferenceLeasedExecutorError::Lease)
    }

    /// Public preparation boundary. Admission, conservative reservation, permit
    /// issuance, and permit preparation either all succeed or the lease is retired
    /// without refund. Callers therefore never coordinate a naked reservation.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_leased_current(
        &self,
        issuer: &mut InferencePermitIssuer,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
        ttl_ticks: u64,
        nonce: [u8; 32],
    ) -> Result<PreparedLeasedInferenceExecution, InferenceLeasedExecutorError> {
        let admission = self.bind_and_lease_current(policy, request, quota, controls)?;
        let (route, binding, lease) = admission.into_parts();
        let issued_at_tick = self.current.trusted_now_tick();
        let permit = match issuer.issue(binding, issued_at_tick, ttl_ticks, nonce) {
            Ok(permit) => permit,
            Err(error) => return Err(self.cleanup_failed_permit(lease, error)),
        };
        let prepared_at_tick = self.current.trusted_now_tick();
        let prepared = match issuer.prepare_execution(permit, binding, prepared_at_tick) {
            Ok(prepared) => prepared,
            Err(error) => return Err(self.cleanup_failed_permit(lease, error)),
        };
        Ok(PreparedLeasedInferenceExecution {
            route,
            prepared,
            lease,
        })
    }

    /// Explicit local cancellation path. It consumes the prepared capability,
    /// retires the reservation without refund, and records cancellation evidence.
    pub fn cancel_prepared_leased_current(
        &self,
        capability: PreparedLeasedInferenceExecution,
    ) -> Result<InferenceReceipt, InferenceLeasedExecutionError> {
        if self.capability_belongs_elsewhere(&capability) {
            return Err(InferenceLeasedExecutionError::WrongAuthority(capability));
        }
        let PreparedLeasedInferenceExecution {
            prepared,
            lease,
            ..
        } = capability;
        let completed_at_tick = self.current.trusted_now_tick();
        self.lock_authority()
            .map_err(InferenceLeasedExecutionError::PreWire)?
            .abandon(lease)
            .map_err(|error| {
                InferenceLeasedExecutionError::PreWire(InferenceLeasedExecutorError::Lease(error))
            })?;
        InferenceReceipt::failure(
            prepared,
            InferenceFailureClass::Cancelled,
            completed_at_tick,
        )
        .map_err(|error| {
            InferenceLeasedExecutionError::PreWire(InferenceLeasedExecutorError::Receipt(error))
        })
    }

    /// Public non-streaming execution accepts only the joined prepared capability.
    /// A capability presented to a different authority is returned intact before
    /// provider resolution so its origin can still settle/retire it.
    pub async fn execute_prepared_leased_current(
        &self,
        capability: PreparedLeasedInferenceExecution,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
    ) -> Result<InferenceExecutionOutcome, InferenceLeasedExecutionError> {
        if self.capability_belongs_elsewhere(&capability) {
            return Err(InferenceLeasedExecutionError::WrongAuthority(capability));
        }
        let PreparedLeasedInferenceExecution {
            prepared,
            lease,
            ..
        } = capability;
        self.execute_leased_current(prepared, lease, policy, request, quota, controls)
            .await
    }

    /// Streaming equivalent of `execute_prepared_leased_current`.
    pub async fn execute_streaming_prepared_leased_current(
        &self,
        capability: PreparedLeasedInferenceExecution,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<InferenceExecutionOutcome, InferenceLeasedExecutionError> {
        if self.capability_belongs_elsewhere(&capability) {
            return Err(InferenceLeasedExecutionError::WrongAuthority(capability));
        }
        let PreparedLeasedInferenceExecution {
            prepared,
            lease,
            ..
        } = capability;
        self.execute_streaming_leased_current(
            prepared, lease, policy, request, quota, controls, on_token,
        )
        .await
    }

    /// Crate-internal intermediate used only to compose the public prepared
    /// capability. It is never returned from the public safe surface.
    fn bind_and_lease_current(
        &self,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
    ) -> Result<LeasedResourceExecutableAdmission, InferenceLeasedExecutorError> {
        let now_tick = self.current.trusted_now_tick();
        let now_millis = self.resource_clock.now_millis();
        let profile = self
            .current
            .resolve_current_profile()
            .map_err(|_| InferenceLeasedExecutorError::ProviderProfileRejected)?;
        let reserved_tokens = request
            .requirements
            .estimated_input_tokens
            .checked_add(request.requirements.max_output_tokens)
            .ok_or(InferenceLeasedExecutorError::TokenExposureOverflow)?;

        let mut authority = self.lock_authority()?;
        let scope = authority
            .verify_scope(&profile, self.current.credential_binding(), quota)
            .map_err(InferenceLeasedExecutorError::Lease)?;
        let v4 = admit_and_bind_resource_execution(
            policy,
            request,
            &profile,
            self.current.credential_binding(),
            quota,
            self.current.endpoint_binding(),
            controls,
            now_tick,
            &scope,
        )
        .map_err(InferenceLeasedExecutorError::ResourceExecution)?;
        let lease = authority
            .reserve_verified(
                &profile,
                self.current.credential_binding(),
                quota,
                &scope,
                reserved_tokens,
                now_millis,
            )
            .map_err(InferenceLeasedExecutorError::Lease)?;

        let binding = match bind_active_resource_lease(*v4.binding(), &lease) {
            Ok(binding) => binding,
            Err(error) => {
                let cleanup = authority.abandon(lease);
                return Err(match cleanup {
                    Ok(()) => InferenceLeasedExecutorError::Lease(error),
                    Err(cleanup) => InferenceLeasedExecutorError::LeaseCleanup(cleanup),
                });
            }
        };

        Ok(LeasedResourceExecutableAdmission {
            route: v4.route().clone(),
            binding,
            lease,
        })
    }

    fn cleanup_failed_permit(
        &self,
        lease: InferenceExecutionResourceLease,
        permit: InferencePermitError,
    ) -> InferenceLeasedExecutorError {
        match self.authority.lock() {
            Ok(mut authority) => match authority.abandon(lease) {
                Ok(()) => InferenceLeasedExecutorError::Permit(permit),
                Err(cleanup) => InferenceLeasedExecutorError::PermitAndLeaseCleanup {
                    permit,
                    cleanup,
                },
            },
            Err(_) => InferenceLeasedExecutorError::PermitAndAuthorityPoisoned(permit),
        }
    }

    fn capability_belongs_elsewhere(
        &self,
        capability: &PreparedLeasedInferenceExecution,
    ) -> bool {
        let authority = match self.authority.lock() {
            Ok(authority) => authority,
            Err(_) => return false,
        };
        matches!(
            authority.validate_lease_owner(&capability.lease),
            Err(
                InferenceResourceLeaseError::AuthorityMismatch
                    | InferenceResourceLeaseError::AuthorityEpochMismatch
                    | InferenceResourceLeaseError::InactiveLease
            )
        )
    }

    pub(crate) async fn execute_leased_current(
        &self,
        prepared: PreparedInferenceExecution,
        lease: InferenceExecutionResourceLease,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
    ) -> Result<InferenceExecutionOutcome, InferenceLeasedExecutionError> {
        let now_tick = self.current.trusted_now_tick();
        let expected_binding = match self.derive_expected_binding(
            &lease,
            policy,
            request,
            quota,
            controls,
        ) {
            Ok(binding) => binding,
            Err(error @ InferenceLeasedExecutorError::AuthorityPoisoned) => {
                return Err(InferenceLeasedExecutionError::PreWire(error));
            }
            Err(error) => {
                let failure = prewire_failure_class(&error);
                return self
                    .local_failure(prepared, lease, now_tick, failure)
                    .map_err(InferenceLeasedExecutionError::PreWire);
            }
        };

        let execution = self
            .current
            .inner()
            .execute_preverified_binding(prepared, expected_binding, request, controls)
            .await;
        self.finish_execution(lease, execution)
    }

    pub(crate) async fn execute_streaming_leased_current(
        &self,
        prepared: PreparedInferenceExecution,
        lease: InferenceExecutionResourceLease,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<InferenceExecutionOutcome, InferenceLeasedExecutionError> {
        let now_tick = self.current.trusted_now_tick();
        let expected_binding = match self.derive_expected_binding(
            &lease,
            policy,
            request,
            quota,
            controls,
        ) {
            Ok(binding) => binding,
            Err(error @ InferenceLeasedExecutorError::AuthorityPoisoned) => {
                return Err(InferenceLeasedExecutionError::PreWire(error));
            }
            Err(error) => {
                let failure = prewire_failure_class(&error);
                return self
                    .local_failure(prepared, lease, now_tick, failure)
                    .map_err(InferenceLeasedExecutionError::PreWire);
            }
        };

        let execution = self
            .current
            .inner()
            .execute_streaming_preverified_binding(
                prepared,
                expected_binding,
                request,
                controls,
                on_token,
            )
            .await;
        self.finish_execution(lease, execution)
    }

    fn derive_expected_binding(
        &self,
        lease: &InferenceExecutionResourceLease,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
    ) -> Result<InferenceExecutionBinding, InferenceLeasedExecutorError> {
        let now_tick = self.current.trusted_now_tick();
        let profile = self
            .current
            .resolve_current_profile()
            .map_err(|_| InferenceLeasedExecutorError::ProviderProfileRejected)?;
        let authority = self.lock_authority()?;
        let scope = authority
            .revalidate(
                lease,
                &profile,
                self.current.credential_binding(),
                quota,
            )
            .map_err(InferenceLeasedExecutorError::Lease)?;
        let v4 = admit_and_bind_resource_execution(
            policy,
            request,
            &profile,
            self.current.credential_binding(),
            quota,
            self.current.endpoint_binding(),
            controls,
            now_tick,
            &scope,
        )
        .map_err(InferenceLeasedExecutorError::ResourceExecution)?;
        bind_active_resource_lease(*v4.binding(), lease).map_err(InferenceLeasedExecutorError::Lease)
    }

    fn finish_execution(
        &self,
        lease: InferenceExecutionResourceLease,
        execution: Result<InferenceExecutionOutcome, InferenceExecutorFatalError>,
    ) -> Result<InferenceExecutionOutcome, InferenceLeasedExecutionError> {
        let now_millis = self.resource_clock.now_millis();
        match execution {
            Ok(outcome) => {
                let mut authority = match self.authority.lock() {
                    Ok(authority) => authority,
                    Err(_) => {
                        return Err(InferenceLeasedExecutionError::OutcomeAuthorityPoisoned(
                            Box::new(outcome),
                        ));
                    }
                };
                match authority.settle_outcome(lease, &outcome, now_millis) {
                    Ok(_) => Ok(outcome),
                    Err(error) => Err(InferenceLeasedExecutionError::Settlement {
                        outcome: Box::new(outcome),
                        error,
                    }),
                }
            }
            Err(executor) => {
                let mut authority = match self.authority.lock() {
                    Ok(authority) => authority,
                    Err(_) => {
                        return Err(
                            InferenceLeasedExecutionError::ExecutorAndAuthorityPoisoned(executor),
                        );
                    }
                };
                match authority.settle_executor_fatal(lease, now_millis) {
                    Ok(()) => Err(InferenceLeasedExecutionError::Executor(executor)),
                    Err(settlement) => Err(InferenceLeasedExecutionError::ExecutorAndSettlement {
                        executor,
                        settlement,
                    }),
                }
            }
        }
    }

    fn local_failure(
        &self,
        prepared: PreparedInferenceExecution,
        lease: InferenceExecutionResourceLease,
        completed_at_tick: u64,
        failure: InferenceExecutionFailure,
    ) -> Result<InferenceExecutionOutcome, InferenceLeasedExecutorError> {
        // Retire resource capability first. Even if receipt construction then fails,
        // the failed local attempt cannot pin active authority indefinitely.
        self.lock_authority()?
            .abandon(lease)
            .map_err(InferenceLeasedExecutorError::Lease)?;
        let receipt = InferenceReceipt::failure(
            prepared,
            InferenceFailureClass::VerificationRejected,
            completed_at_tick,
        )
        .map_err(InferenceLeasedExecutorError::Receipt)?;
        Ok(InferenceExecutionOutcome {
            response_text: None,
            receipt,
            failure: Some(failure),
            wire_observation: None,
        })
    }

    fn lock_authority(
        &self,
    ) -> Result<
        std::sync::MutexGuard<'_, InferenceExecutionResourceAuthority>,
        InferenceLeasedExecutorError,
    > {
        self.authority
            .lock()
            .map_err(|_| InferenceLeasedExecutorError::AuthorityPoisoned)
    }
}

fn prewire_failure_class(error: &InferenceLeasedExecutorError) -> InferenceExecutionFailure {
    match error {
        InferenceLeasedExecutorError::ProviderProfileRejected => {
            InferenceExecutionFailure::ProviderProfileRejected
        }
        _ => InferenceExecutionFailure::ResourceScopeRejected,
    }
}

#[derive(Debug)]
pub enum InferenceLeasedExecutorError {
    AuthorityPoisoned,
    ProviderProfileRejected,
    TokenExposureOverflow,
    Permit(InferencePermitError),
    PermitAndLeaseCleanup {
        permit: InferencePermitError,
        cleanup: InferenceResourceLeaseError,
    },
    PermitAndAuthorityPoisoned(InferencePermitError),
    Lease(InferenceResourceLeaseError),
    LeaseCleanup(InferenceResourceLeaseError),
    ResourceExecution(InferenceResourceExecutionError),
    Receipt(InferenceReceiptError),
}

impl fmt::Display for InferenceLeasedExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthorityPoisoned => write!(f, "resource lease authority mutex is poisoned"),
            Self::ProviderProfileRejected => {
                write!(f, "current provider profile rejected lease binding")
            }
            Self::TokenExposureOverflow => write!(f, "request token exposure overflow"),
            Self::Permit(error) => write!(f, "leased permit preparation failed: {error}"),
            Self::PermitAndLeaseCleanup { permit, cleanup } => write!(
                f,
                "leased permit preparation failed ({permit}) and lease cleanup failed ({cleanup})"
            ),
            Self::PermitAndAuthorityPoisoned(permit) => write!(
                f,
                "leased permit preparation failed ({permit}) and resource authority is poisoned"
            ),
            Self::Lease(error) => write!(f, "resource lease rejected: {error}"),
            Self::LeaseCleanup(error) => write!(f, "resource lease cleanup failed: {error}"),
            Self::ResourceExecution(error) => write!(f, "v4 resource binding rejected: {error}"),
            Self::Receipt(error) => write!(f, "local leased failure receipt failed: {error}"),
        }
    }
}

impl std::error::Error for InferenceLeasedExecutorError {}

#[derive(Debug)]
pub enum InferenceLeasedExecutionError {
    WrongAuthority(PreparedLeasedInferenceExecution),
    PreWire(InferenceLeasedExecutorError),
    Executor(InferenceExecutorFatalError),
    Settlement {
        outcome: Box<InferenceExecutionOutcome>,
        error: InferenceResourceLeaseError,
    },
    OutcomeAuthorityPoisoned(Box<InferenceExecutionOutcome>),
    ExecutorAndAuthorityPoisoned(InferenceExecutorFatalError),
    ExecutorAndSettlement {
        executor: InferenceExecutorFatalError,
        settlement: InferenceResourceLeaseError,
    },
}

impl InferenceLeasedExecutionError {
    pub fn into_outcome(self) -> Option<InferenceExecutionOutcome> {
        match self {
            Self::Settlement { outcome, .. } | Self::OutcomeAuthorityPoisoned(outcome) => {
                Some(*outcome)
            }
            _ => None,
        }
    }

    pub fn into_capability(self) -> Option<PreparedLeasedInferenceExecution> {
        match self {
            Self::WrongAuthority(capability) => Some(capability),
            _ => None,
        }
    }
}

impl fmt::Display for InferenceLeasedExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongAuthority(_) => {
                write!(f, "leased inference capability belongs to a different authority")
            }
            Self::PreWire(error) => write!(f, "leased execution rejected before I/O: {error}"),
            Self::Executor(error) => write!(f, "leased inference executor failed: {error}"),
            Self::Settlement { error, .. } => {
                write!(f, "inference completed but resource settlement failed: {error}")
            }
            Self::OutcomeAuthorityPoisoned(_) => {
                write!(f, "inference completed but resource authority mutex is poisoned")
            }
            Self::ExecutorAndAuthorityPoisoned(error) => write!(
                f,
                "inference executor failed and resource authority mutex is poisoned: {error}"
            ),
            Self::ExecutorAndSettlement {
                executor,
                settlement,
            } => write!(
                f,
                "inference executor failed ({executor}) and resource settlement failed ({settlement})"
            ),
        }
    }
}

impl std::error::Error for InferenceLeasedExecutionError {}
