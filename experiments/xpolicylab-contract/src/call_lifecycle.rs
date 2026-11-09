//! Policy-server RPC lifecycle contract for XPolicyLab reconnect semantics.
//!
//! This models only policy-serving requests. It is distinct from the
//! Symtropy controller request/ticket ledger and has no actuator authority.
//!
//! The upstream XPolicyLab protocol documents request-ID reuse across
//! reconnects, duplicate-response caching, fatal ambiguous timeout, and
//! explicit server-restart abort. This in-memory fixture keeps those cases
//! separate and binds a request ID to its exact RPC kind and payload digest.

use std::collections::BTreeMap;

use super::Digest32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyRpcKind {
    UpdateObservation,
    UpdateObservationBatch,
    GetAction,
    GetActionBatch,
    Reset,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyRequestIdentity {
    pub request_id: String,
    pub episode_id: String,
    pub policy_server_instance_id: String,
    pub schema_version: String,
    pub rpc_kind: PolicyRpcKind,
    /// Digest of the exact serialized request payload, including observations
    /// or batch/environment indices as sent to the policy server.
    pub payload_digest: Digest32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeginDisposition {
    /// First time this request identity is being submitted.
    DispatchNew,
    /// Same ID and same request identity while the original call is unresolved.
    /// Transport may reconnect/replay this exact ID; it must not create a new
    /// model invocation with a new ID.
    ReplaySameInFlightRequest,
    /// The server already cached this exact request's response.
    ReturnCachedResponse(Digest32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialAbortReason {
    AmbiguousTimeout,
    PolicyServerRestarted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallLifecycleError {
    EmptyIdentity,
    InvalidPayloadDigest,
    RequestIdentityMismatch,
    RequestIdReusedWithDifferentPayload,
    UnknownRequest(String),
    ConflictingCachedResponse,
    RequestAlreadyCompleted,
    RequestBudgetExhausted,
    TrialAborted(TrialAbortReason),
    LateResponseAfterAbort(TrialAbortReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestState {
    InFlight,
    ResponseCached(Digest32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RequestRecord {
    identity: PolicyRequestIdentity,
    state: RequestState,
}

/// Bounded by the episode's declared request budget. The fixture deliberately
/// does not implement persistence or cross-process anti-replay.
#[derive(Debug, Clone)]
pub struct PolicyCallLedger {
    episode_id: String,
    policy_server_instance_id: String,
    max_requests: usize,
    requests: BTreeMap<String, RequestRecord>,
    aborted: Option<TrialAbortReason>,
}

impl PolicyCallLedger {
    pub fn new(
        episode_id: impl Into<String>,
        policy_server_instance_id: impl Into<String>,
        max_requests: usize,
    ) -> Result<Self, CallLifecycleError> {
        let episode_id = episode_id.into();
        let policy_server_instance_id = policy_server_instance_id.into();
        if episode_id.trim().is_empty()
            || policy_server_instance_id.trim().is_empty()
            || max_requests == 0
        {
            return Err(CallLifecycleError::EmptyIdentity);
        }
        Ok(Self {
            episode_id,
            policy_server_instance_id,
            max_requests,
            requests: BTreeMap::new(),
            aborted: None,
        })
    }

    /// Reserve a policy RPC identity or return the disposition for an exact
    /// reconnect replay. A request ID cannot be rebound to changed input.
    pub fn begin(
        &mut self,
        identity: PolicyRequestIdentity,
    ) -> Result<BeginDisposition, CallLifecycleError> {
        self.ensure_active()?;
        validate_identity(&identity)?;

        if identity.episode_id != self.episode_id
            || identity.policy_server_instance_id != self.policy_server_instance_id
        {
            return Err(CallLifecycleError::RequestIdentityMismatch);
        }

        if let Some(existing) = self.requests.get(&identity.request_id) {
            if existing.identity != identity {
                return Err(CallLifecycleError::RequestIdReusedWithDifferentPayload);
            }
            return Ok(match existing.state {
                RequestState::InFlight => BeginDisposition::ReplaySameInFlightRequest,
                RequestState::ResponseCached(digest) => {
                    BeginDisposition::ReturnCachedResponse(digest)
                }
            });
        }

        if self.requests.len() >= self.max_requests {
            return Err(CallLifecycleError::RequestBudgetExhausted);
        }

        self.requests.insert(
            identity.request_id.clone(),
            RequestRecord {
                identity,
                state: RequestState::InFlight,
            },
        );
        Ok(BeginDisposition::DispatchNew)
    }

    /// Commit one server response to the exact request. An exact duplicate is
    /// idempotent; different content for the same request ID is rejected.
    pub fn record_response(
        &mut self,
        request_id: &str,
        response_digest: Digest32,
    ) -> Result<(), CallLifecycleError> {
        if let Some(reason) = self.aborted {
            return Err(CallLifecycleError::LateResponseAfterAbort(reason));
        }
        let record = self
            .requests
            .get_mut(request_id)
            .ok_or_else(|| CallLifecycleError::UnknownRequest(request_id.to_owned()))?;
        match record.state {
            RequestState::InFlight => {
                record.state = RequestState::ResponseCached(response_digest);
                Ok(())
            }
            RequestState::ResponseCached(existing) if existing == response_digest => Ok(()),
            RequestState::ResponseCached(_) => {
                Err(CallLifecycleError::ConflictingCachedResponse)
            }
        }
    }

    /// An ambiguous timeout terminates the policy trial. No subsequent request
    /// may proceed in this episode, and a late response cannot be used.
    pub fn mark_timeout(&mut self, request_id: &str) -> Result<(), CallLifecycleError> {
        self.ensure_active()?;
        let record = self
            .requests
            .get(request_id)
            .ok_or_else(|| CallLifecycleError::UnknownRequest(request_id.to_owned()))?;
        if matches!(record.state, RequestState::ResponseCached(_)) {
            return Err(CallLifecycleError::RequestAlreadyCompleted);
        }
        self.aborted = Some(TrialAbortReason::AmbiguousTimeout);
        Ok(())
    }

    /// A changed policy-server process loses model-side state and terminates
    /// the active episode. A caller must start a separately identified episode.
    pub fn mark_server_restarted(&mut self) -> Result<(), CallLifecycleError> {
        self.ensure_active()?;
        self.aborted = Some(TrialAbortReason::PolicyServerRestarted);
        Ok(())
    }

    pub fn abort_reason(&self) -> Option<TrialAbortReason> {
        self.aborted
    }

    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    fn ensure_active(&self) -> Result<(), CallLifecycleError> {
        match self.aborted {
            Some(reason) => Err(CallLifecycleError::TrialAborted(reason)),
            None => Ok(()),
        }
    }
}

fn validate_identity(identity: &PolicyRequestIdentity) -> Result<(), CallLifecycleError> {
    if identity.request_id.trim().is_empty()
        || identity.episode_id.trim().is_empty()
        || identity.policy_server_instance_id.trim().is_empty()
        || identity.schema_version.trim().is_empty()
    {
        return Err(CallLifecycleError::EmptyIdentity);
    }
    if identity.payload_digest.0.iter().all(|byte| *byte == 0) {
        return Err(CallLifecycleError::InvalidPayloadDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> Digest32 {
        Digest32::from_bytes([value; 32]).unwrap()
    }

    fn identity(request_id: &str, payload: u8) -> PolicyRequestIdentity {
        PolicyRequestIdentity {
            request_id: request_id.into(),
            episode_id: "episode-a".into(),
            policy_server_instance_id: "policy-process-1".into(),
            schema_version: "xpl-schema-v1".into(),
            rpc_kind: PolicyRpcKind::GetActionBatch,
            payload_digest: digest(payload),
        }
    }

    fn ledger() -> PolicyCallLedger {
        PolicyCallLedger::new("episode-a", "policy-process-1", 8).unwrap()
    }

    #[test]
    fn first_request_is_dispatched_once() {
        let mut state = ledger();
        assert_eq!(
            state.begin(identity("request-1", 1)),
            Ok(BeginDisposition::DispatchNew)
        );
        assert_eq!(state.request_count(), 1);
    }

    #[test]
    fn reconnect_reuses_same_request_identity_while_in_flight() {
        let mut state = ledger();
        let request = identity("request-1", 1);
        assert_eq!(state.begin(request.clone()), Ok(BeginDisposition::DispatchNew));
        assert_eq!(
            state.begin(request),
            Ok(BeginDisposition::ReplaySameInFlightRequest)
        );
        assert_eq!(state.request_count(), 1);
    }

    #[test]
    fn same_request_returns_cached_response_after_completion() {
        let mut state = ledger();
        let request = identity("request-1", 1);
        assert_eq!(state.begin(request.clone()), Ok(BeginDisposition::DispatchNew));
        state.record_response("request-1", digest(9)).unwrap();
        assert_eq!(
            state.begin(request),
            Ok(BeginDisposition::ReturnCachedResponse(digest(9)))
        );
    }

    #[test]
    fn same_request_id_cannot_be_rebound_to_different_payload_or_rpc() {
        let mut state = ledger();
        assert_eq!(
            state.begin(identity("request-1", 1)),
            Ok(BeginDisposition::DispatchNew)
        );
        assert_eq!(
            state.begin(identity("request-1", 2)),
            Err(CallLifecycleError::RequestIdReusedWithDifferentPayload)
        );

        let mut changed_kind = identity("request-1", 1);
        changed_kind.rpc_kind = PolicyRpcKind::UpdateObservation;
        assert_eq!(
            state.begin(changed_kind),
            Err(CallLifecycleError::RequestIdReusedWithDifferentPayload)
        );
    }

    #[test]
    fn timeout_aborts_episode_and_rejects_late_response() {
        let mut state = ledger();
        state.begin(identity("request-1", 1)).unwrap();
        state.mark_timeout("request-1").unwrap();
        assert_eq!(
            state.record_response("request-1", digest(9)),
            Err(CallLifecycleError::LateResponseAfterAbort(
                TrialAbortReason::AmbiguousTimeout
            ))
        );
        assert_eq!(
            state.begin(identity("request-2", 2)),
            Err(CallLifecycleError::TrialAborted(
                TrialAbortReason::AmbiguousTimeout
            ))
        );
    }

    #[test]
    fn policy_server_restart_aborts_model_state_for_episode() {
        let mut state = ledger();
        state.begin(identity("request-1", 1)).unwrap();
        state.mark_server_restarted().unwrap();
        assert_eq!(
            state.begin(identity("request-2", 2)),
            Err(CallLifecycleError::TrialAborted(
                TrialAbortReason::PolicyServerRestarted
            ))
        );
    }

    #[test]
    fn rejects_cross_episode_and_cross_process_requests() {
        let mut state = ledger();
        let mut wrong_episode = identity("request-1", 1);
        wrong_episode.episode_id = "episode-b".into();
        assert_eq!(
            state.begin(wrong_episode),
            Err(CallLifecycleError::RequestIdentityMismatch)
        );

        let mut wrong_process = identity("request-2", 2);
        wrong_process.policy_server_instance_id = "policy-process-2".into();
        assert_eq!(
            state.begin(wrong_process),
            Err(CallLifecycleError::RequestIdentityMismatch)
        );
    }

    #[test]
    fn duplicate_response_is_idempotent_but_conflicting_response_rejects() {
        let mut state = ledger();
        state.begin(identity("request-1", 1)).unwrap();
        state.record_response("request-1", digest(9)).unwrap();
        state.record_response("request-1", digest(9)).unwrap();
        assert_eq!(
            state.record_response("request-1", digest(8)),
            Err(CallLifecycleError::ConflictingCachedResponse)
        );
    }

    #[test]
    fn rejects_timeout_after_completed_response_and_unknown_request() {
        let mut state = ledger();
        assert_eq!(
            state.mark_timeout("missing"),
            Err(CallLifecycleError::UnknownRequest("missing".into()))
        );

        state.begin(identity("request-1", 1)).unwrap();
        state.record_response("request-1", digest(9)).unwrap();
        assert_eq!(
            state.mark_timeout("request-1"),
            Err(CallLifecycleError::RequestAlreadyCompleted)
        );
    }

    #[test]
    fn request_budget_is_bounded() {
        let mut state = PolicyCallLedger::new("episode-a", "policy-process-1", 1).unwrap();
        assert_eq!(
            state.begin(identity("request-1", 1)),
            Ok(BeginDisposition::DispatchNew)
        );
        assert_eq!(
            state.begin(identity("request-2", 2)),
            Err(CallLifecycleError::RequestBudgetExhausted)
        );
    }
}
