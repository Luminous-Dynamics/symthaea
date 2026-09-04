// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Outcome-unknown-safe external anchoring for qualification witness frontiers.
//!
//! This crate does not implement Xenia/TPM/SCITT itself. It consumes #456's
//! guarded local anchor permit and imposes the transport-independent rules that
//! every concrete anchor backend must satisfy:
//!
//! - one deterministic idempotency identity per source namespace + exact target;
//! - closed-world `Applied / ProvenNotDispatched / OutcomeUnknown` dispatch;
//! - no automatic retry after an ambiguous effect;
//! - an `Applied` claim must exactly describe the guarded local frontier;
//! - the external source verifier must still authenticate/freshen that claim;
//! - post-dispatch verification failure remains retry-unsafe.
//!
//! Evidence chronology remains separate from execution authority.

#![deny(unsafe_code)]

use symthaea_authority::Digest32;
use symthaea_qualification_witness_frontier::{
    verify_external_witness_frontier_v1, ExternalWitnessFrontierClaimV1,
    ExternalWitnessFrontierVerifier, VerifiedExternalWitnessFrontierV1,
    WitnessFrontierPointV1,
};
use symthaea_qualification_witness_frontier_sqlite::GuardedAnchorPermitV1;
use thiserror::Error;

pub const WITNESS_ANCHOR_RUNTIME_SCHEMA_VERSION: u16 = 1;

const OPERATION_DOMAIN: &[u8] = b"symthaea.qualification-witness.anchor-operation.v1\0";
const DIAGNOSTIC_DOMAIN: &[u8] = b"symthaea.qualification-witness.anchor-diagnostic.v1\0";
const ZERO32: [u8; 32] = [0; 32];

/// Stable source namespace configured by a concrete external-anchor adapter.
///
/// `anchor_policy_digest` should commit the source-specific append/currentness
/// policy (for example Xenia ledger identity + append schema + checkpoint
/// freshness rules). Changing that policy intentionally changes operation IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalAnchorSourceNamespaceV1 {
    pub source_id: [u8; 16],
    pub source_epoch: u64,
    pub anchor_policy_digest: Digest32,
}

impl ExternalAnchorSourceNamespaceV1 {
    pub fn validate(self) -> Result<(), WitnessAnchorRuntimeError> {
        if self.source_id == [0; 16]
            || self.source_epoch == 0
            || self.anchor_policy_digest.0 == ZERO32
        {
            return Err(WitnessAnchorRuntimeError::InvalidSourceNamespace);
        }
        Ok(())
    }
}

/// Deterministic write identity for one exact witness frontier in one external
/// source namespace. Backends MUST use `operation_id` as their idempotency key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessAnchorOperationV1 {
    pub schema_version: u16,
    pub operation_id: Digest32,
    pub source_id: [u8; 16],
    pub source_epoch: u64,
    pub anchor_policy_digest: Digest32,
    pub witness_id: [u8; 16],
    pub high_watermark: u64,
    pub reservation_head: Digest32,
    pub frontier_statement_digest: Digest32,
}

impl WitnessAnchorOperationV1 {
    pub fn from_guarded_permit(
        permit: &GuardedAnchorPermitV1<'_>,
        source: ExternalAnchorSourceNamespaceV1,
    ) -> Result<Self, WitnessAnchorRuntimeError> {
        source.validate()?;
        let frontier = permit.frontier();
        validate_frontier(frontier)?;
        if permit.witness_id() != frontier.witness_id {
            return Err(WitnessAnchorRuntimeError::PermitFrontierMismatch);
        }

        let operation_id = operation_digest(source, frontier);
        Ok(Self {
            schema_version: WITNESS_ANCHOR_RUNTIME_SCHEMA_VERSION,
            operation_id,
            source_id: source.source_id,
            source_epoch: source.source_epoch,
            anchor_policy_digest: source.anchor_policy_digest,
            witness_id: frontier.witness_id,
            high_watermark: frontier.high_watermark,
            reservation_head: frontier.reservation_head,
            frontier_statement_digest: frontier.statement_digest,
        })
    }

    pub fn validate(self) -> Result<(), WitnessAnchorRuntimeError> {
        if self.schema_version != WITNESS_ANCHOR_RUNTIME_SCHEMA_VERSION
            || self.source_id == [0; 16]
            || self.source_epoch == 0
            || self.anchor_policy_digest.0 == ZERO32
            || self.witness_id == [0; 16]
            || self.high_watermark == 0
            || self.reservation_head.0 == ZERO32
            || self.frontier_statement_digest.0 == ZERO32
        {
            return Err(WitnessAnchorRuntimeError::MalformedOperation);
        }
        let source = ExternalAnchorSourceNamespaceV1 {
            source_id: self.source_id,
            source_epoch: self.source_epoch,
            anchor_policy_digest: self.anchor_policy_digest,
        };
        let point = WitnessFrontierPointV1 {
            witness_id: self.witness_id,
            high_watermark: self.high_watermark,
            reservation_head: self.reservation_head,
            statement_digest: self.frontier_statement_digest,
        };
        source.validate()?;
        validate_frontier(point)?;
        if self.operation_id != operation_digest(source, point) {
            return Err(WitnessAnchorRuntimeError::OperationCommitmentMismatch);
        }
        Ok(())
    }

    pub fn frontier(&self) -> WitnessFrontierPointV1 {
        WitnessFrontierPointV1 {
            witness_id: self.witness_id,
            high_watermark: self.high_watermark,
            reservation_head: self.reservation_head,
            statement_digest: self.frontier_statement_digest,
        }
    }
}

/// Backends return an effect classification, never a generic transport `Err`.
/// If a backend cannot prove the request never crossed its effect boundary, it
/// MUST return `OutcomeUnknown`.
#[derive(Debug)]
pub enum ExternalAnchorDispatchOutcomeV1 {
    Applied(ExternalWitnessFrontierClaimV1),
    ProvenNotDispatched { diagnostic_digest: Digest32 },
    OutcomeUnknown { diagnostic_digest: Digest32 },
}

#[derive(Debug)]
pub enum ExternalAnchorReconciliationOutcomeV1 {
    Applied(ExternalWitnessFrontierClaimV1),
    ProvenNotApplied { diagnostic_digest: Digest32 },
    OutcomeUnknown { diagnostic_digest: Digest32 },
}

/// Concrete Xenia/TPM/transparency implementations must provide bounded calls,
/// use `operation_id` as an idempotency key, and enforce their own source-side
/// monotonic/CAS preconditions. This generic layer cannot manufacture those
/// source-specific guarantees.
pub trait ExternalWitnessAnchorBackend {
    fn dispatch(&mut self, operation: &WitnessAnchorOperationV1) -> ExternalAnchorDispatchOutcomeV1;

    fn reconcile(
        &mut self,
        operation: &WitnessAnchorOperationV1,
    ) -> ExternalAnchorReconciliationOutcomeV1;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorOutcomeUnknownReasonV1 {
    BackendReportedUnknown,
    BackendOutcomeMalformed,
    AppliedClaimMismatch,
    AppliedClaimVerificationRejected,
}

/// Dispatch result. Once `backend.dispatch` has been called, this API does not
/// return a generic error; every post-dispatch state is explicitly retry-safe or
/// retry-unsafe.
#[derive(Debug)]
pub enum GuardedAnchorDispatchResultV1 {
    VerifiedApplied {
        operation: WitnessAnchorOperationV1,
        verified_external: VerifiedExternalWitnessFrontierV1,
    },
    ProvenNotDispatched {
        operation: WitnessAnchorOperationV1,
        diagnostic_digest: Digest32,
    },
    OutcomeUnknown {
        operation: WitnessAnchorOperationV1,
        reason: AnchorOutcomeUnknownReasonV1,
        diagnostic_digest: Digest32,
    },
}

#[derive(Debug)]
pub enum AnchorReconciliationResultV1 {
    VerifiedApplied {
        operation: WitnessAnchorOperationV1,
        verified_external: VerifiedExternalWitnessFrontierV1,
    },
    ProvenNotApplied {
        operation: WitnessAnchorOperationV1,
        diagnostic_digest: Digest32,
    },
    OutcomeUnknown {
        operation: WitnessAnchorOperationV1,
        reason: AnchorOutcomeUnknownReasonV1,
        diagnostic_digest: Digest32,
    },
}

pub fn dispatch_guarded_anchor_v1<B, V>(
    permit: &GuardedAnchorPermitV1<'_>,
    source: ExternalAnchorSourceNamespaceV1,
    backend: &mut B,
    verifier: &V,
) -> Result<GuardedAnchorDispatchResultV1, WitnessAnchorRuntimeError>
where
    B: ExternalWitnessAnchorBackend,
    V: ExternalWitnessFrontierVerifier,
{
    let operation = WitnessAnchorOperationV1::from_guarded_permit(permit, source)?;
    operation.validate()?;

    let outcome = backend.dispatch(&operation);
    Ok(classify_dispatch_outcome(operation, outcome, verifier))
}

/// Reconcile one previously constructed deterministic operation. Callers must
/// persist/derive the exact operation they are reconciling and must not replace
/// reconciliation with a blind second dispatch after `OutcomeUnknown`.
pub fn reconcile_anchor_operation_v1<B, V>(
    operation: WitnessAnchorOperationV1,
    backend: &mut B,
    verifier: &V,
) -> Result<AnchorReconciliationResultV1, WitnessAnchorRuntimeError>
where
    B: ExternalWitnessAnchorBackend,
    V: ExternalWitnessFrontierVerifier,
{
    operation.validate()?;
    let outcome = backend.reconcile(&operation);
    Ok(classify_reconciliation_outcome(operation, outcome, verifier))
}

fn classify_dispatch_outcome<V: ExternalWitnessFrontierVerifier>(
    operation: WitnessAnchorOperationV1,
    outcome: ExternalAnchorDispatchOutcomeV1,
    verifier: &V,
) -> GuardedAnchorDispatchResultV1 {
    match outcome {
        ExternalAnchorDispatchOutcomeV1::Applied(claim) => {
            match verify_applied_claim(operation, claim, verifier) {
                Ok(verified_external) => GuardedAnchorDispatchResultV1::VerifiedApplied {
                    operation,
                    verified_external,
                },
                Err(AppliedClaimFailure::Mismatch(diagnostic_digest)) => {
                    GuardedAnchorDispatchResultV1::OutcomeUnknown {
                        operation,
                        reason: AnchorOutcomeUnknownReasonV1::AppliedClaimMismatch,
                        diagnostic_digest,
                    }
                }
                Err(AppliedClaimFailure::Verification(diagnostic_digest)) => {
                    GuardedAnchorDispatchResultV1::OutcomeUnknown {
                        operation,
                        reason: AnchorOutcomeUnknownReasonV1::AppliedClaimVerificationRejected,
                        diagnostic_digest,
                    }
                }
            }
        }
        ExternalAnchorDispatchOutcomeV1::ProvenNotDispatched { diagnostic_digest } => {
            if diagnostic_digest.0 == ZERO32 {
                GuardedAnchorDispatchResultV1::OutcomeUnknown {
                    operation,
                    reason: AnchorOutcomeUnknownReasonV1::BackendOutcomeMalformed,
                    diagnostic_digest: diagnostic_for_label(b"zero-proven-not-dispatched"),
                }
            } else {
                GuardedAnchorDispatchResultV1::ProvenNotDispatched {
                    operation,
                    diagnostic_digest,
                }
            }
        }
        ExternalAnchorDispatchOutcomeV1::OutcomeUnknown { diagnostic_digest } => {
            GuardedAnchorDispatchResultV1::OutcomeUnknown {
                operation,
                reason: if diagnostic_digest.0 == ZERO32 {
                    AnchorOutcomeUnknownReasonV1::BackendOutcomeMalformed
                } else {
                    AnchorOutcomeUnknownReasonV1::BackendReportedUnknown
                },
                diagnostic_digest: if diagnostic_digest.0 == ZERO32 {
                    diagnostic_for_label(b"zero-backend-outcome-unknown")
                } else {
                    diagnostic_digest
                },
            }
        }
    }
}

fn classify_reconciliation_outcome<V: ExternalWitnessFrontierVerifier>(
    operation: WitnessAnchorOperationV1,
    outcome: ExternalAnchorReconciliationOutcomeV1,
    verifier: &V,
) -> AnchorReconciliationResultV1 {
    match outcome {
        ExternalAnchorReconciliationOutcomeV1::Applied(claim) => {
            match verify_applied_claim(operation, claim, verifier) {
                Ok(verified_external) => AnchorReconciliationResultV1::VerifiedApplied {
                    operation,
                    verified_external,
                },
                Err(AppliedClaimFailure::Mismatch(diagnostic_digest)) => {
                    AnchorReconciliationResultV1::OutcomeUnknown {
                        operation,
                        reason: AnchorOutcomeUnknownReasonV1::AppliedClaimMismatch,
                        diagnostic_digest,
                    }
                }
                Err(AppliedClaimFailure::Verification(diagnostic_digest)) => {
                    AnchorReconciliationResultV1::OutcomeUnknown {
                        operation,
                        reason: AnchorOutcomeUnknownReasonV1::AppliedClaimVerificationRejected,
                        diagnostic_digest,
                    }
                }
            }
        }
        ExternalAnchorReconciliationOutcomeV1::ProvenNotApplied { diagnostic_digest } => {
            if diagnostic_digest.0 == ZERO32 {
                AnchorReconciliationResultV1::OutcomeUnknown {
                    operation,
                    reason: AnchorOutcomeUnknownReasonV1::BackendOutcomeMalformed,
                    diagnostic_digest: diagnostic_for_label(b"zero-proven-not-applied"),
                }
            } else {
                AnchorReconciliationResultV1::ProvenNotApplied {
                    operation,
                    diagnostic_digest,
                }
            }
        }
        ExternalAnchorReconciliationOutcomeV1::OutcomeUnknown { diagnostic_digest } => {
            AnchorReconciliationResultV1::OutcomeUnknown {
                operation,
                reason: if diagnostic_digest.0 == ZERO32 {
                    AnchorOutcomeUnknownReasonV1::BackendOutcomeMalformed
                } else {
                    AnchorOutcomeUnknownReasonV1::BackendReportedUnknown
                },
                diagnostic_digest: if diagnostic_digest.0 == ZERO32 {
                    diagnostic_for_label(b"zero-reconcile-unknown")
                } else {
                    diagnostic_digest
                },
            }
        }
    }
}

enum AppliedClaimFailure {
    Mismatch(Digest32),
    Verification(Digest32),
}

fn verify_applied_claim<V: ExternalWitnessFrontierVerifier>(
    operation: WitnessAnchorOperationV1,
    claim: ExternalWitnessFrontierClaimV1,
    verifier: &V,
) -> Result<VerifiedExternalWitnessFrontierV1, AppliedClaimFailure> {
    if claim.source_id != operation.source_id
        || claim.source_epoch != operation.source_epoch
        || claim.witness_id != operation.witness_id
        || claim.high_watermark != operation.high_watermark
        || claim.reservation_head != operation.reservation_head
        || claim.frontier_statement_digest != operation.frontier_statement_digest
    {
        return Err(AppliedClaimFailure::Mismatch(diagnostic_claim(&claim)));
    }

    verify_external_witness_frontier_v1(claim, verifier).map_err(|error| {
        AppliedClaimFailure::Verification(diagnostic_for_text(&error.to_string()))
    })
}

fn validate_frontier(point: WitnessFrontierPointV1) -> Result<(), WitnessAnchorRuntimeError> {
    point
        .validate()
        .map_err(|_| WitnessAnchorRuntimeError::PermitFrontierMismatch)
}

fn operation_digest(
    source: ExternalAnchorSourceNamespaceV1,
    frontier: WitnessFrontierPointV1,
) -> Digest32 {
    let mut transcript = Transcript::new(OPERATION_DOMAIN);
    transcript.u16(WITNESS_ANCHOR_RUNTIME_SCHEMA_VERSION);
    transcript.fixed(&source.source_id);
    transcript.u64(source.source_epoch);
    transcript.fixed(&source.anchor_policy_digest.0);
    transcript.fixed(&frontier.witness_id);
    transcript.u64(frontier.high_watermark);
    transcript.fixed(&frontier.reservation_head.0);
    transcript.fixed(&frontier.statement_digest.0);
    Digest32(transcript.finish())
}

fn diagnostic_claim(claim: &ExternalWitnessFrontierClaimV1) -> Digest32 {
    let mut transcript = Transcript::new(DIAGNOSTIC_DOMAIN);
    transcript.fixed(b"claim-mismatch\0");
    transcript.u16(claim.schema_version);
    transcript.fixed(&claim.source_id);
    transcript.u64(claim.source_epoch);
    transcript.u64(claim.source_sequence);
    transcript.fixed(&claim.witness_id);
    transcript.u64(claim.high_watermark);
    transcript.fixed(&claim.reservation_head.0);
    transcript.fixed(&claim.frontier_statement_digest.0);
    transcript.fixed(&claim.freshness_evidence_digest.0);
    Digest32(transcript.finish())
}

fn diagnostic_for_label(label: &[u8]) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DIAGNOSTIC_DOMAIN);
    hasher.update(label);
    Digest32(*hasher.finalize().as_bytes())
}

fn diagnostic_for_text(text: &str) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DIAGNOSTIC_DOMAIN);
    hasher.update(b"verification-error\0");
    hasher.update(text.as_bytes());
    Digest32(*hasher.finalize().as_bytes())
}

#[derive(Debug, Error)]
pub enum WitnessAnchorRuntimeError {
    #[error("invalid external anchor source namespace")]
    InvalidSourceNamespace,
    #[error("guarded anchor permit and local frontier disagree")]
    PermitFrontierMismatch,
    #[error("malformed witness anchor operation")]
    MalformedOperation,
    #[error("witness anchor operation commitment mismatch")]
    OperationCommitmentMismatch,
}

struct Transcript {
    bytes: Vec<u8>,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(domain.len() + 256);
        bytes.extend_from_slice(domain);
        Self { bytes }
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn fixed(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.bytes).as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::*;
    use symthaea_qualification_witness_frontier::{
        ExternalAnchorVerificationError, WitnessFrontierRecoveryRelationV1,
        EXTERNAL_ANCHOR_SCHEMA_VERSION,
    };
    use symthaea_qualification_witness_frontier_sqlite::SqliteWitnessFrontierPublicationGuard;
    use symthaea_qualification_witness_sequence::{
        SqliteWitnessSequenceStore, WitnessSequenceAttemptBindingV1,
    };

    fn path(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "symthaea-anchor-runtime-{name}-{}-{}.sqlite3",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        cleanup(&path);
        path
    }

    fn binding() -> WitnessSequenceAttemptBindingV1 {
        WitnessSequenceAttemptBindingV1 {
            attempt_id: [1; 16],
            witness_id: [0x51; 16],
            witness_epoch: 7,
            archive_sha256: Digest32([0x11; 32]),
            git_head: [0x22; 20],
            git_tree: [0x33; 20],
            verifier_digest: Digest32([0x44; 32]),
            witness_policy_digest: Digest32([0x55; 32]),
        }
    }

    fn source(epoch: u64) -> ExternalAnchorSourceNamespaceV1 {
        ExternalAnchorSourceNamespaceV1 {
            source_id: [0x61; 16],
            source_epoch: epoch,
            anchor_policy_digest: Digest32([0x62; 32]),
        }
    }

    struct AcceptVerifier;

    impl ExternalWitnessFrontierVerifier for AcceptVerifier {
        fn verify_current(
            &self,
            _claim: &ExternalWitnessFrontierClaimV1,
        ) -> Result<(), ExternalAnchorVerificationError> {
            Ok(())
        }
    }

    struct RejectVerifier;

    impl ExternalWitnessFrontierVerifier for RejectVerifier {
        fn verify_current(
            &self,
            _claim: &ExternalWitnessFrontierClaimV1,
        ) -> Result<(), ExternalAnchorVerificationError> {
            Err(ExternalAnchorVerificationError {
                reason: "freshness rejected".to_string(),
            })
        }
    }

    #[derive(Clone, Copy)]
    enum FakeMode {
        Applied,
        MismatchedApplied,
        NotDispatched,
        Unknown,
        ReconcileApplied,
    }

    struct FakeBackend {
        mode: FakeMode,
    }

    impl FakeBackend {
        fn claim(operation: &WitnessAnchorOperationV1) -> ExternalWitnessFrontierClaimV1 {
            ExternalWitnessFrontierClaimV1 {
                schema_version: EXTERNAL_ANCHOR_SCHEMA_VERSION,
                source_id: operation.source_id,
                source_epoch: operation.source_epoch,
                source_sequence: 10,
                witness_id: operation.witness_id,
                high_watermark: operation.high_watermark,
                reservation_head: operation.reservation_head,
                frontier_statement_digest: operation.frontier_statement_digest,
                freshness_evidence_digest: Digest32([0x77; 32]),
            }
        }
    }

    impl ExternalWitnessAnchorBackend for FakeBackend {
        fn dispatch(&mut self, operation: &WitnessAnchorOperationV1) -> ExternalAnchorDispatchOutcomeV1 {
            match self.mode {
                FakeMode::Applied | FakeMode::ReconcileApplied => {
                    ExternalAnchorDispatchOutcomeV1::Applied(Self::claim(operation))
                }
                FakeMode::MismatchedApplied => {
                    let mut claim = Self::claim(operation);
                    claim.reservation_head = Digest32([0x99; 32]);
                    ExternalAnchorDispatchOutcomeV1::Applied(claim)
                }
                FakeMode::NotDispatched => ExternalAnchorDispatchOutcomeV1::ProvenNotDispatched {
                    diagnostic_digest: Digest32([0x81; 32]),
                },
                FakeMode::Unknown => ExternalAnchorDispatchOutcomeV1::OutcomeUnknown {
                    diagnostic_digest: Digest32([0x82; 32]),
                },
            }
        }

        fn reconcile(
            &mut self,
            operation: &WitnessAnchorOperationV1,
        ) -> ExternalAnchorReconciliationOutcomeV1 {
            match self.mode {
                FakeMode::ReconcileApplied | FakeMode::Applied => {
                    ExternalAnchorReconciliationOutcomeV1::Applied(Self::claim(operation))
                }
                FakeMode::NotDispatched => {
                    ExternalAnchorReconciliationOutcomeV1::ProvenNotApplied {
                        diagnostic_digest: Digest32([0x83; 32]),
                    }
                }
                FakeMode::Unknown | FakeMode::MismatchedApplied => {
                    ExternalAnchorReconciliationOutcomeV1::OutcomeUnknown {
                        diagnostic_digest: Digest32([0x84; 32]),
                    }
                }
            }
        }
    }

    #[test]
    fn exact_applied_claim_becomes_verified_external_frontier() {
        let path = path("applied");
        let store = SqliteWitnessSequenceStore::open(&path).unwrap();
        store.reserve_attempt(binding()).unwrap();
        let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, [0x51; 16]).unwrap();
        let decision = guard.classify(None).unwrap();
        assert!(matches!(
            decision.relation(),
            WitnessFrontierRecoveryRelationV1::InitialAnchorRequired { .. }
        ));
        let permit = decision.anchor_permit().unwrap();
        let mut backend = FakeBackend { mode: FakeMode::Applied };
        let result = dispatch_guarded_anchor_v1(&permit, source(3), &mut backend, &AcceptVerifier).unwrap();
        let verified = match result {
            GuardedAnchorDispatchResultV1::VerifiedApplied { verified_external, .. } => verified_external,
            other => panic!("unexpected result: {other:?}"),
        };
        assert_eq!(verified.point(), permit.frontier());
        drop(permit);
        drop(decision);
        guard.release().unwrap();
        cleanup(&path);
    }

    #[test]
    fn operation_id_is_deterministic_and_policy_namespaced() {
        let path = path("operation-id");
        let store = SqliteWitnessSequenceStore::open(&path).unwrap();
        store.reserve_attempt(binding()).unwrap();
        let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, [0x51; 16]).unwrap();
        let decision = guard.classify(None).unwrap();
        let permit = decision.anchor_permit().unwrap();
        let a = WitnessAnchorOperationV1::from_guarded_permit(&permit, source(3)).unwrap();
        let b = WitnessAnchorOperationV1::from_guarded_permit(&permit, source(3)).unwrap();
        let c = WitnessAnchorOperationV1::from_guarded_permit(&permit, source(4)).unwrap();
        assert_eq!(a.operation_id, b.operation_id);
        assert_ne!(a.operation_id, c.operation_id);
        drop(permit);
        drop(decision);
        guard.release().unwrap();
        cleanup(&path);
    }

    #[test]
    fn mismatched_applied_claim_is_retry_unsafe_unknown() {
        let path = path("mismatch");
        let store = SqliteWitnessSequenceStore::open(&path).unwrap();
        store.reserve_attempt(binding()).unwrap();
        let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, [0x51; 16]).unwrap();
        let decision = guard.classify(None).unwrap();
        let permit = decision.anchor_permit().unwrap();
        let mut backend = FakeBackend { mode: FakeMode::MismatchedApplied };
        let result = dispatch_guarded_anchor_v1(&permit, source(3), &mut backend, &AcceptVerifier).unwrap();
        assert!(matches!(
            result,
            GuardedAnchorDispatchResultV1::OutcomeUnknown {
                reason: AnchorOutcomeUnknownReasonV1::AppliedClaimMismatch,
                ..
            }
        ));
        drop(permit);
        drop(decision);
        guard.release().unwrap();
        cleanup(&path);
    }

    #[test]
    fn verifier_rejection_after_applied_is_retry_unsafe_unknown() {
        let path = path("verify-reject");
        let store = SqliteWitnessSequenceStore::open(&path).unwrap();
        store.reserve_attempt(binding()).unwrap();
        let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, [0x51; 16]).unwrap();
        let decision = guard.classify(None).unwrap();
        let permit = decision.anchor_permit().unwrap();
        let mut backend = FakeBackend { mode: FakeMode::Applied };
        let result = dispatch_guarded_anchor_v1(&permit, source(3), &mut backend, &RejectVerifier).unwrap();
        assert!(matches!(
            result,
            GuardedAnchorDispatchResultV1::OutcomeUnknown {
                reason: AnchorOutcomeUnknownReasonV1::AppliedClaimVerificationRejected,
                ..
            }
        ));
        drop(permit);
        drop(decision);
        guard.release().unwrap();
        cleanup(&path);
    }

    #[test]
    fn unknown_dispatch_requires_explicit_reconciliation() {
        let path = path("reconcile");
        let store = SqliteWitnessSequenceStore::open(&path).unwrap();
        store.reserve_attempt(binding()).unwrap();
        let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, [0x51; 16]).unwrap();
        let decision = guard.classify(None).unwrap();
        let permit = decision.anchor_permit().unwrap();
        let mut backend = FakeBackend { mode: FakeMode::Unknown };
        let result = dispatch_guarded_anchor_v1(&permit, source(3), &mut backend, &AcceptVerifier).unwrap();
        let operation = match result {
            GuardedAnchorDispatchResultV1::OutcomeUnknown { operation, .. } => operation,
            other => panic!("unexpected result: {other:?}"),
        };
        drop(permit);
        drop(decision);
        guard.release().unwrap();

        backend.mode = FakeMode::ReconcileApplied;
        let reconciled = reconcile_anchor_operation_v1(operation, &mut backend, &AcceptVerifier).unwrap();
        assert!(matches!(
            reconciled,
            AnchorReconciliationResultV1::VerifiedApplied { .. }
        ));
        cleanup(&path);
    }

    #[test]
    fn proven_not_dispatched_is_the_only_retry_safe_dispatch_failure() {
        let path = path("not-dispatched");
        let store = SqliteWitnessSequenceStore::open(&path).unwrap();
        store.reserve_attempt(binding()).unwrap();
        let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, [0x51; 16]).unwrap();
        let decision = guard.classify(None).unwrap();
        let permit = decision.anchor_permit().unwrap();
        let mut backend = FakeBackend { mode: FakeMode::NotDispatched };
        let result = dispatch_guarded_anchor_v1(&permit, source(3), &mut backend, &AcceptVerifier).unwrap();
        assert!(matches!(
            result,
            GuardedAnchorDispatchResultV1::ProvenNotDispatched { .. }
        ));
        drop(permit);
        drop(decision);
        guard.release().unwrap();
        cleanup(&path);
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(format!("{}-wal", path.display()));
        let _ = fs::remove_file(format!("{}-shm", path.display()));
    }
}
