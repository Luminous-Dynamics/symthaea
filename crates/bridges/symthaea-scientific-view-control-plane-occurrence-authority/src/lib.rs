//! Deployment-owned store-authority qualification for SCI-014 control-plane occurrences.
//!
//! The authority chain is deliberately non-collapsible:
//!
//! ```text
//! store-authority binding descriptor
//!     != opaque trusted-store handle
//!     != backend-observed commit
//!     != store-authority-qualified commit
//!     != historical committed occurrence
//!     != current control-plane head
//! ```
//!
//! The trusted handle has no public production constructor in this tranche.
//! A future deployment/provisioning boundary must establish that one actual
//! backend instance is the store named by the binding before production code can
//! mint the handle.
//!
//! Architecture: SCI-014R2B / #3457.

#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use std::error::Error as StdError;
use symthaea_scientific_view_control_plane_occurrence::{
    attempt_observed_control_plane_commit, reconcile_observed_control_plane_commit,
    ControlPlaneCommitPlanV1, ControlPlaneCommitProtocolError,
    ControlPlaneOccurrenceStoreBindingV1, ControlPlaneOccurrenceStoreV1,
    ObservedControlPlaneCommitAmbiguityV1, ObservedControlPlaneCommitOutcomeV1,
    ObservedControlPlaneNonCommitV1, ObservedExactControlPlaneCommitV1,
};
use symthaea_scientific_view_profile::Commitment32;
use thiserror::Error;

const STORE_AUTHORITY_BINDING_DOMAIN: &[u8] =
    b"symthaea.science.view.control-plane.occurrence-store-authority-binding.v1";
const STORE_AUTHORITY_QUALIFIED_COMMIT_DOMAIN: &[u8] =
    b"symthaea.science.view.control-plane.store-authority-qualified-commit.v1";
const HISTORICAL_OCCURRENCE_DOMAIN: &[u8] =
    b"symthaea.science.view.control-plane.historical-committed-occurrence.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlPlaneOccurrenceStoreAuthorityBindingV1 {
    deployment_id: String,
    view_namespace: String,
    occurrence_store_binding_commitment: Commitment32,
    store_provisioning_epoch: u64,
    store_instance_binding_commitment: Commitment32,
    store_configuration_commitment: Commitment32,
    store_trust_root_commitment: Commitment32,
    provisioning_evidence_commitment: Commitment32,
    commitment: Commitment32,
}

impl ControlPlaneOccurrenceStoreAuthorityBindingV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        occurrence_store_binding: &ControlPlaneOccurrenceStoreBindingV1,
        store_instance_binding_commitment: Commitment32,
        store_configuration_commitment: Commitment32,
        store_trust_root_commitment: Commitment32,
        provisioning_evidence_commitment: Commitment32,
    ) -> Result<Self, ControlPlaneOccurrenceStoreAuthorityError> {
        for (field, value) in [
            ("store_instance_binding_commitment", store_instance_binding_commitment),
            ("store_configuration_commitment", store_configuration_commitment),
            ("store_trust_root_commitment", store_trust_root_commitment),
            ("provisioning_evidence_commitment", provisioning_evidence_commitment),
        ] {
            if value.is_zero() {
                return Err(ControlPlaneOccurrenceStoreAuthorityError::ZeroCommitment { field });
            }
        }
        let commitment = hash_with(STORE_AUTHORITY_BINDING_DOMAIN, |hasher| {
            put_text(hasher, occurrence_store_binding.deployment_id());
            put_text(hasher, occurrence_store_binding.view_namespace());
            put_commitment(hasher, occurrence_store_binding.commitment());
            put_u64(hasher, occurrence_store_binding.provisioning_epoch());
            put_commitment(hasher, store_instance_binding_commitment);
            put_commitment(hasher, store_configuration_commitment);
            put_commitment(hasher, store_trust_root_commitment);
            put_commitment(hasher, provisioning_evidence_commitment);
        });
        Ok(Self {
            deployment_id: occurrence_store_binding.deployment_id().to_owned(),
            view_namespace: occurrence_store_binding.view_namespace().to_owned(),
            occurrence_store_binding_commitment: occurrence_store_binding.commitment(),
            store_provisioning_epoch: occurrence_store_binding.provisioning_epoch(),
            store_instance_binding_commitment,
            store_configuration_commitment,
            store_trust_root_commitment,
            provisioning_evidence_commitment,
            commitment,
        })
    }

    pub fn deployment_id(&self) -> &str { &self.deployment_id }
    pub fn view_namespace(&self) -> &str { &self.view_namespace }
    pub const fn occurrence_store_binding_commitment(&self) -> Commitment32 { self.occurrence_store_binding_commitment }
    pub const fn store_provisioning_epoch(&self) -> u64 { self.store_provisioning_epoch }
    pub const fn store_instance_binding_commitment(&self) -> Commitment32 { self.store_instance_binding_commitment }
    pub const fn store_configuration_commitment(&self) -> Commitment32 { self.store_configuration_commitment }
    pub const fn store_trust_root_commitment(&self) -> Commitment32 { self.store_trust_root_commitment }
    pub const fn provisioning_evidence_commitment(&self) -> Commitment32 { self.provisioning_evidence_commitment }
    pub const fn commitment(&self) -> Commitment32 { self.commitment }

    fn validate_store_binding(&self, store_binding: &ControlPlaneOccurrenceStoreBindingV1) -> Result<(), ControlPlaneOccurrenceStoreAuthorityError> {
        if self.deployment_id != store_binding.deployment_id()
            || self.view_namespace != store_binding.view_namespace()
            || self.occurrence_store_binding_commitment != store_binding.commitment()
            || self.store_provisioning_epoch != store_binding.provisioning_epoch()
        {
            return Err(ControlPlaneOccurrenceStoreAuthorityError::StoreBindingMismatch);
        }
        Ok(())
    }
}

pub struct TrustedControlPlaneOccurrenceStoreV1<S> {
    binding: ControlPlaneOccurrenceStoreAuthorityBindingV1,
    store: S,
}

impl<S> TrustedControlPlaneOccurrenceStoreV1<S> {
    pub fn binding(&self) -> &ControlPlaneOccurrenceStoreAuthorityBindingV1 { &self.binding }
}

#[derive(Debug, PartialEq, Eq)]
pub struct StoreAuthorityQualifiedControlPlaneCommitV1 {
    observed: ObservedExactControlPlaneCommitV1,
    authority_binding: ControlPlaneOccurrenceStoreAuthorityBindingV1,
    qualified_commitment: Commitment32,
}

impl StoreAuthorityQualifiedControlPlaneCommitV1 {
    pub fn observed(&self) -> &ObservedExactControlPlaneCommitV1 { &self.observed }
    pub fn authority_binding(&self) -> &ControlPlaneOccurrenceStoreAuthorityBindingV1 { &self.authority_binding }
    pub const fn qualified_commitment(&self) -> Commitment32 { self.qualified_commitment }
}

#[derive(Debug, PartialEq, Eq)]
pub struct HistoricalCommittedControlPlaneTransitionV1 {
    qualified_commit: StoreAuthorityQualifiedControlPlaneCommitV1,
    historical_occurrence_commitment: Commitment32,
}

impl HistoricalCommittedControlPlaneTransitionV1 {
    pub fn qualified_commit(&self) -> &StoreAuthorityQualifiedControlPlaneCommitV1 { &self.qualified_commit }
    pub const fn historical_occurrence_commitment(&self) -> Commitment32 { self.historical_occurrence_commitment }
}

#[derive(Debug)]
pub struct TrustedControlPlaneCommitAmbiguityV1<E>
where E: StdError + Send + Sync + 'static,
{
    authority_binding_commitment: Commitment32,
    inner: ObservedControlPlaneCommitAmbiguityV1<E>,
}

impl<E> TrustedControlPlaneCommitAmbiguityV1<E>
where E: StdError + Send + Sync + 'static,
{
    pub const fn authority_binding_commitment(&self) -> Commitment32 { self.authority_binding_commitment }
    pub fn operation_id(&self) -> symthaea_scientific_view_control_plane_occurrence::ControlPlaneCommitOperationIdV1 { self.inner.operation_id() }
}

#[derive(Debug)]
pub enum TrustedControlPlaneCommitOutcomeV1<E>
where E: StdError + Send + Sync + 'static,
{
    HistoricalCommitted(HistoricalCommittedControlPlaneTransitionV1),
    ProvenNotCommitted {
        authority_binding_commitment: Commitment32,
        observed: ObservedControlPlaneNonCommitV1,
    },
    OutcomeUnknown(TrustedControlPlaneCommitAmbiguityV1<E>),
}

impl<E> TrustedControlPlaneCommitOutcomeV1<E>
where E: StdError + Send + Sync + 'static,
{
    pub fn requires_reconciliation(&self) -> bool { matches!(self, Self::OutcomeUnknown(_)) }
}

pub fn attempt_trusted_control_plane_commit<S>(
    trusted: &mut TrustedControlPlaneOccurrenceStoreV1<S>,
    plan: ControlPlaneCommitPlanV1,
) -> Result<TrustedControlPlaneCommitOutcomeV1<S::Error>, TrustedControlPlaneCommitError<S::Error>>
where S: ControlPlaneOccurrenceStoreV1,
{
    trusted.binding.validate_store_binding(plan.store_binding())?;
    let observed = attempt_observed_control_plane_commit(&mut trusted.store, plan)?;
    Ok(map_observed(&trusted.binding, observed))
}

pub fn reconcile_trusted_control_plane_commit<S>(
    trusted: &TrustedControlPlaneOccurrenceStoreV1<S>,
    ambiguity: TrustedControlPlaneCommitAmbiguityV1<S::Error>,
) -> Result<TrustedControlPlaneCommitOutcomeV1<S::Error>, TrustedControlPlaneCommitError<S::Error>>
where S: ControlPlaneOccurrenceStoreV1,
{
    if ambiguity.authority_binding_commitment != trusted.binding.commitment() {
        return Err(TrustedControlPlaneCommitError::AuthorityBindingMismatch);
    }
    let observed = reconcile_observed_control_plane_commit(&trusted.store, ambiguity.inner);
    Ok(map_observed(&trusted.binding, observed))
}

fn map_observed<E>(
    binding: &ControlPlaneOccurrenceStoreAuthorityBindingV1,
    outcome: ObservedControlPlaneCommitOutcomeV1<E>,
) -> TrustedControlPlaneCommitOutcomeV1<E>
where E: StdError + Send + Sync + 'static,
{
    match outcome {
        ObservedControlPlaneCommitOutcomeV1::CommittedObserved(observed) => {
            let qualified_commitment = hash_with(STORE_AUTHORITY_QUALIFIED_COMMIT_DOMAIN, |hasher| {
                put_commitment(hasher, binding.commitment());
                put_commitment(hasher, observed.occurrence().commitment());
                put_text(hasher, observed.store_reference());
            });
            let qualified = StoreAuthorityQualifiedControlPlaneCommitV1 {
                observed,
                authority_binding: binding.clone(),
                qualified_commitment,
            };
            let historical_occurrence_commitment = hash_with(HISTORICAL_OCCURRENCE_DOMAIN, |hasher| {
                put_commitment(hasher, qualified.qualified_commitment);
                put_commitment(hasher, binding.commitment());
                put_commitment(hasher, qualified.observed.occurrence().commitment());
            });
            TrustedControlPlaneCommitOutcomeV1::HistoricalCommitted(
                HistoricalCommittedControlPlaneTransitionV1 {
                    qualified_commit: qualified,
                    historical_occurrence_commitment,
                },
            )
        }
        ObservedControlPlaneCommitOutcomeV1::ProvenNotCommitted(observed) => {
            TrustedControlPlaneCommitOutcomeV1::ProvenNotCommitted {
                authority_binding_commitment: binding.commitment(),
                observed,
            }
        }
        ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(inner) => {
            TrustedControlPlaneCommitOutcomeV1::OutcomeUnknown(
                TrustedControlPlaneCommitAmbiguityV1 {
                    authority_binding_commitment: binding.commitment(),
                    inner,
                },
            )
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ControlPlaneOccurrenceStoreAuthorityError {
    #[error("zero commitment in field {field}")]
    ZeroCommitment { field: &'static str },
    #[error("store binding does not match occurrence-store authority binding")]
    StoreBindingMismatch,
}

#[derive(Debug, Error)]
pub enum TrustedControlPlaneCommitError<E>
where E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Authority(#[from] ControlPlaneOccurrenceStoreAuthorityError),
    #[error("trusted control-plane ambiguity belongs to a different store-authority binding")]
    AuthorityBindingMismatch,
    #[error(transparent)]
    ObservedProtocol(#[from] ControlPlaneCommitProtocolError<E>),
}

fn hash_with(domain: &[u8], write_fields: impl FnOnce(&mut Sha256)) -> Commitment32 {
    let mut hasher = Sha256::new();
    put_bytes(&mut hasher, domain);
    write_fields(&mut hasher);
    Commitment32::from_bytes(hasher.finalize().into())
}
fn put_bytes(hasher: &mut Sha256, value: &[u8]) { put_u64(hasher, value.len() as u64); hasher.update(value); }
fn put_text(hasher: &mut Sha256, value: &str) { put_bytes(hasher, value.as_bytes()); }
fn put_u64(hasher: &mut Sha256, value: u64) { hasher.update(value.to_be_bytes()); }
fn put_commitment(hasher: &mut Sha256, value: Commitment32) { hasher.update(value.as_bytes()); }

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fmt;
    use std::sync::{Arc, Mutex};
    use symthaea_scientific_view_control_plane::{CandidateScientificViewControlPlaneTransitionV1, DeploymentBootstrapEvidenceRefV1};
    use symthaea_scientific_view_control_plane_occurrence::{
        ControlPlaneOccurrenceHeadV1, ControlPlaneStoreCasResultV1,
        ControlPlaneStoreOperationResolutionV1, ProposedControlPlaneOccurrenceV1,
        RawControlPlaneOccurrenceRecordV1,
    };
    use symthaea_scientific_view_profile::{
        AuthoritySourceBindingV1, RoleSemanticRevisionV1,
        ScientificAuthorityRoleV1 as Role, ScientificViewDeploymentBindingV1,
        ScientificViewSemanticProfileV1,
    };

    fn c(byte: u8) -> Commitment32 { Commitment32::from_bytes([byte; 32]) }
    fn expected(hex: &str) -> Commitment32 {
        let mut bytes = [0u8; 32];
        for (i, slot) in bytes.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        Commitment32::from_bytes(bytes)
    }
    fn candidate() -> CandidateScientificViewControlPlaneTransitionV1 {
        let p = ScientificViewSemanticProfileV1::new(
            "lunar/site01", "site01-confirmatory",
            vec![
                RoleSemanticRevisionV1::new(Role::ResearchSemanticHead, c(11)).unwrap(),
                RoleSemanticRevisionV1::new(Role::VerifierPolicyHead, c(12)).unwrap(),
            ], c(201), c(202), c(203), c(204)
        ).unwrap();
        let b = ScientificViewDeploymentBindingV1::new(
            "deployment/site01-a", &p,
            vec![
                AuthoritySourceBindingV1::new(Role::ResearchSemanticHead, "research/store-a", 7, c(21)).unwrap(),
                AuthoritySourceBindingV1::new(Role::VerifierPolicyHead, "verifier-policy/store-a", 3, c(22)).unwrap(),
            ]
        ).unwrap();
        let bootstrap = DeploymentBootstrapEvidenceRefV1::new(&p, &b, c(41), c(42)).unwrap();
        CandidateScientificViewControlPlaneTransitionV1::bootstrap_candidate(&p, &b, &bootstrap).unwrap()
    }
    fn store_binding(epoch: u64) -> ControlPlaneOccurrenceStoreBindingV1 {
        ControlPlaneOccurrenceStoreBindingV1::new(
            "deployment/site01-a", "lunar/site01", "control-plane/store-a", epoch, c(71)
        ).unwrap()
    }
    fn authority_binding(store: &ControlPlaneOccurrenceStoreBindingV1, trust: u8) -> ControlPlaneOccurrenceStoreAuthorityBindingV1 {
        ControlPlaneOccurrenceStoreAuthorityBindingV1::new(store, c(81), c(82), c(trust), c(84)).unwrap()
    }

    #[derive(Debug, Clone, Copy)] enum ModelError { Ambiguous }
    impl fmt::Display for ModelError { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "ambiguous") } }
    impl StdError for ModelError {}
    #[derive(Clone, Copy, PartialEq, Eq)] enum Mode { Normal, CommitThenError }
    #[derive(Default)] struct State {
        frontier: Option<RawControlPlaneOccurrenceRecordV1>,
        ops: BTreeMap<Commitment32, RawControlPlaneOccurrenceRecordV1>,
    }
    #[derive(Clone)] struct Store { state: Arc<Mutex<State>>, mode: Mode }
    impl Store { fn new(mode: Mode) -> Self { Self { state: Arc::new(Mutex::new(State::default())), mode } } }
    impl ControlPlaneOccurrenceStoreV1 for Store {
        type Error = ModelError;
        fn load_frontier(&self, _: &ControlPlaneOccurrenceStoreBindingV1) -> Result<Option<RawControlPlaneOccurrenceRecordV1>, Self::Error> {
            Ok(self.state.lock().unwrap().frontier.clone())
        }
        fn compare_and_swap(&mut self, _: &ControlPlaneOccurrenceStoreBindingV1, expected: Option<ControlPlaneOccurrenceHeadV1>, proposed: &ProposedControlPlaneOccurrenceV1) -> Result<ControlPlaneStoreCasResultV1, Self::Error> {
            let mut s = self.state.lock().unwrap();
            let current = s.frontier.as_ref().map(|r| r.head());
            if current != expected { return Ok(ControlPlaneStoreCasResultV1::Conflict { actual_frontier: current }); }
            let record = RawControlPlaneOccurrenceRecordV1::new_unqualified(proposed.clone(), "store-ref-1").unwrap();
            s.ops.insert(proposed.operation_id().commitment(), record.clone());
            s.frontier = Some(record);
            if self.mode == Mode::CommitThenError { return Err(ModelError::Ambiguous); }
            Ok(ControlPlaneStoreCasResultV1::Applied { store_reference: "store-ref-1".into() })
        }
        fn resolve_operation(&self, _: &ControlPlaneOccurrenceStoreBindingV1, op: symthaea_scientific_view_control_plane_occurrence::ControlPlaneCommitOperationIdV1) -> Result<ControlPlaneStoreOperationResolutionV1, Self::Error> {
            let s = self.state.lock().unwrap();
            if let Some(record) = s.ops.get(&op.commitment()) { return Ok(ControlPlaneStoreOperationResolutionV1::Found(record.clone())); }
            Ok(ControlPlaneStoreOperationResolutionV1::ProvenAbsent { current_frontier: s.frontier.as_ref().map(|r| r.head()) })
        }
    }

    #[test]
    fn authority_binding_and_historical_ids_match_oracle() {
        let store_id = store_binding(7);
        let binding = authority_binding(&store_id, 83);
        assert_eq!(binding.commitment(), expected("eba4dfa6626431008d986b01f026a5fdbabd034ed94baf8d7fc1c341df9e42e8"));
        let plan = ControlPlaneCommitPlanV1::prepare(store_id, &candidate(), None).unwrap();
        let mut trusted = TrustedControlPlaneOccurrenceStoreV1 { binding, store: Store::new(Mode::Normal) };
        match attempt_trusted_control_plane_commit(&mut trusted, plan).unwrap() {
            TrustedControlPlaneCommitOutcomeV1::HistoricalCommitted(h) => {
                assert_eq!(h.qualified_commit().qualified_commitment(), expected("686ef4d30dd20dcf19ed530a412542f094651d02d797e43024e7bcba0ac26d65"));
                assert_eq!(h.historical_occurrence_commitment(), expected("b2a1a5fa14083c083ba4e020f7410da140ece86eb4bdb471e2bdc97b4fc4c336"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn untrusted_descriptor_cannot_construct_public_trusted_handle() {
        let source = include_str!("lib.rs");
        assert!(!source.contains("pub fn new_trusted"));
        assert!(!source.contains("pub fn provision"));
        assert!(!source.contains("impl<S> From<"));
    }

    #[test]
    fn wrong_store_binding_is_rejected_before_backend_use() {
        let store7 = store_binding(7);
        let binding = authority_binding(&store7, 83);
        let plan8 = ControlPlaneCommitPlanV1::prepare(store_binding(8), &candidate(), None).unwrap();
        let mut trusted = TrustedControlPlaneOccurrenceStoreV1 { binding, store: Store::new(Mode::Normal) };
        assert!(matches!(attempt_trusted_control_plane_commit(&mut trusted, plan8), Err(TrustedControlPlaneCommitError::Authority(ControlPlaneOccurrenceStoreAuthorityError::StoreBindingMismatch))));
    }

    #[test]
    fn outcome_unknown_cannot_reconcile_through_another_authority_binding() {
        let store_id = store_binding(7);
        let binding = authority_binding(&store_id, 83);
        let plan = ControlPlaneCommitPlanV1::prepare(store_id, &candidate(), None).unwrap();
        let shared = Arc::new(Mutex::new(State::default()));
        let mut trusted = TrustedControlPlaneOccurrenceStoreV1 {
            binding: binding.clone(),
            store: Store { state: shared.clone(), mode: Mode::CommitThenError },
        };
        let ambiguity = match attempt_trusted_control_plane_commit(&mut trusted, plan).unwrap() {
            TrustedControlPlaneCommitOutcomeV1::OutcomeUnknown(a) => a,
            other => panic!("unexpected {other:?}"),
        };
        let wrong = TrustedControlPlaneOccurrenceStoreV1 {
            binding: authority_binding(&store_binding(7), 85),
            store: Store { state: shared, mode: Mode::Normal },
        };
        assert!(matches!(reconcile_trusted_control_plane_commit(&wrong, ambiguity), Err(TrustedControlPlaneCommitError::AuthorityBindingMismatch)));
    }

    #[test]
    fn raw_backend_observation_has_no_public_promotion_function() {
        let source = include_str!("lib.rs");
        assert!(!source.contains("pub fn qualify_observed"));
        assert!(!source.contains("pub fn historical_from_observed"));
        assert!(!source.contains("Serialize"));
        assert!(!source.contains("pub fn is_current"));
    }
}
