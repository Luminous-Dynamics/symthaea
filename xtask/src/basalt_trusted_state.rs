//! Pure reference state machine for BASALT trusted source-epoch advancement.
//!
//! The pure reference state machine models state-transition semantics only;
//! evidence variants are inputs from separately verified adapters. The optional
//! SQLite backend below implements a local WAL/FULL transaction adapter, but
//! this module does not verify DSSE signatures, signer authorization, Merkle
//! inclusion/consistency proofs, witness signatures, external freshness,
//! physical storage/VFS durability, or protection against malicious snapshot
//! rollback. Passing either the model or local adapter is not a trust verdict.
//!
//! The source cursor key is stable across observer-profile and boot changes:
//! (provisioned host identity, source ID, partition ID). Profile and boot
//! transitions require explicit authorization evidence and never reset the
//! source generation.

use anyhow::Context;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};

const MAX_ID_BYTES: usize = 128;
const MAX_SQLITE_LOG_CURSORS: usize = 128;
const MAX_SQLITE_SOURCE_CURSORS: usize = 4_096;
const MAX_SQLITE_ACCEPTED_RECEIPTS: usize = 100_000;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SourceKey {
    pub provisioned_host_id: String,
    pub source_id: String,
    pub partition_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogCheckpoint {
    pub log_id: String,
    pub tree_size: u64,
    pub root_digest: String,
    pub checkpoint_digest: String,
}

/// Evidence supplied only after the corresponding external verifier has run.
///
/// The reference model checks transition shape and linkage. It does not verify
/// the signatures, trust roots, inclusion paths, or consistency proof bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointEvidence {
    /// First use of a log requires an explicitly pinned checkpoint digest.
    VerifiedPinnedBootstrap { pinned_checkpoint_digest: String },
    /// A refreshed signed checkpoint commits to the same tree size and root.
    VerifiedSameTree,
    /// A larger tree must extend the previous root under a verified proof.
    VerifiedConsistentExtension { consistency_proof_digest: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyVersions {
    pub signer: u64,
    pub witness: u64,
    pub bundle_digest: String,
}

/// A candidate after envelope, signer-policy, inclusion, checkpoint and
/// witness adapters have produced their independent verification results.
///
/// expected_revision binds the candidate to the state snapshot that was
/// actually checked. A conflict requires a complete re-read and re-verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub expected_revision: u64,
    pub source: SourceKey,
    pub generation: u64,
    pub previous_seal_digest: Option<String>,
    pub seal_digest: String,
    /// Digest of the exact immutable DSSE envelope bytes, not re-serialized JSON.
    pub receipt_digest: String,
    pub observer_profile_digest: String,
    pub boot_incarnation_digest: String,
    /// Required only when the observer profile changes for an existing cursor.
    pub profile_transition_evidence_digest: Option<String>,
    /// Required only when the boot incarnation changes for an existing cursor.
    pub boot_transition_evidence_digest: Option<String>,
    pub policy_versions: PolicyVersions,
    /// Required for policy bootstrap or monotonic policy advancement.
    pub policy_transition_evidence_digest: Option<String>,
    pub checkpoint: LogCheckpoint,
    pub checkpoint_evidence: CheckpointEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceCursor {
    pub generation: u64,
    pub seal_digest: String,
    pub receipt_digest: String,
    pub observer_profile_digest: String,
    pub boot_incarnation_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedState {
    revision: u64,
    log_cursors: BTreeMap<String, LogCheckpoint>,
    source_cursors: BTreeMap<SourceKey, SourceCursor>,
    accepted_receipts: BTreeMap<String, AcceptedReceipt>,
    policy_versions: Option<PolicyVersions>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AcceptedReceipt {
    source: SourceKey,
    generation: u64,
    previous_seal_digest: Option<String>,
    seal_digest: String,
    receipt_digest: String,
    observer_profile_digest: String,
    boot_incarnation_digest: String,
    checkpoint: LogCheckpoint,
    policy_versions: PolicyVersions,
}

impl AcceptedReceipt {
    fn matches(&self, candidate: &Candidate) -> bool {
        // The checkpoint and policy values are the original acceptance context,
        // not the immutable receipt identity. A retry may carry refreshed proof
        // material or a newer policy view; it must discover the prior decision
        // without advancing either cursor again.
        self.source == candidate.source
            && self.generation == candidate.generation
            && self.previous_seal_digest == candidate.previous_seal_digest
            && self.seal_digest == candidate.seal_digest
            && self.receipt_digest == candidate.receipt_digest
            && self.observer_profile_digest == candidate.observer_profile_digest
            && self.boot_incarnation_digest == candidate.boot_incarnation_digest
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdvancePlan {
    expected_revision: u64,
    expected_state: TrustedState,
    next_state: TrustedState,
    receipt_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransitionDecision {
    Advance(AdvancePlan),
    AlreadyCommitted { revision: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransitionError {
    MalformedCandidate,
    CompareConflict { expected: u64, actual: u64 },
    StateSnapshotMismatch { revision: u64 },
    RevisionOverflow,
    GenerationZero,
    GenerationBootstrapInvalid,
    GenerationGap { expected: u64, actual: u64 },
    StaleGeneration { current: u64, candidate: u64 },
    GenerationEquivocation { generation: u64 },
    GenerationOverflow,
    PredecessorMismatch,
    UnauthorizedProfileTransition,
    UnexpectedProfileTransitionEvidence,
    UnauthorizedBootTransition,
    UnexpectedBootTransitionEvidence,
    LogBootstrapRequired,
    LogBootstrapMismatch,
    LogOriginMismatch,
    StaleCheckpoint { current: u64, candidate: u64 },
    CheckpointFork { tree_size: u64 },
    CheckpointEvidenceMismatch,
    InvalidConsistencyProofReference,
    PolicyBootstrapRequired,
    PolicyRollback,
    PolicyVersionEquivocation,
    PolicyTransitionUnauthorized,
    UnexpectedPolicyTransitionEvidence,
    ReceiptDigestCollision,
    InvalidAdvancePlan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitOutcome {
    Committed { revision: u64 },
}

impl Default for TrustedState {
    fn default() -> Self {
        Self::new()
    }
}

impl TrustedState {
    pub fn new() -> Self {
        Self {
            revision: 0,
            log_cursors: BTreeMap::new(),
            source_cursors: BTreeMap::new(),
            accepted_receipts: BTreeMap::new(),
            policy_versions: None,
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn source_cursor(&self, key: &SourceKey) -> Option<&SourceCursor> {
        self.source_cursors.get(key)
    }

    pub fn log_cursor(&self, log_id: &str) -> Option<&LogCheckpoint> {
        self.log_cursors.get(log_id)
    }

    pub fn accepted_receipt_count(&self) -> usize {
        self.accepted_receipts.len()
    }

    pub fn policy_versions(&self) -> Option<&PolicyVersions> {
        self.policy_versions.as_ref()
    }

    /// Prepare a pure, revision-bound state transition.
    ///
    /// Exact retries are recognized before the revision comparison so a
    /// client recovering a lost commit response can distinguish a committed
    /// receipt from an uncommitted candidate. A reused receipt digest with
    /// different semantics is rejected as a collision/equivocation.
    pub fn preview(&self, candidate: &Candidate) -> Result<TransitionDecision, TransitionError> {
        validate_candidate(candidate)?;

        if let Some(accepted) = self.accepted_receipts.get(&candidate.receipt_digest) {
            return if accepted.matches(candidate) {
                Ok(TransitionDecision::AlreadyCommitted { revision: self.revision })
            } else {
                Err(TransitionError::ReceiptDigestCollision)
            };
        }

        if candidate.expected_revision != self.revision {
            return Err(TransitionError::CompareConflict {
                expected: candidate.expected_revision,
                actual: self.revision,
            });
        }

        let next_revision = self.revision.checked_add(1).ok_or(TransitionError::RevisionOverflow)?;
        validate_source_transition(self, candidate)?;
        validate_policy_transition(self, candidate)?;
        validate_checkpoint_transition(self, candidate)?;

        let mut next_state = self.clone();
        next_state.revision = next_revision;
        next_state.log_cursors.insert(
            candidate.checkpoint.log_id.clone(),
            candidate.checkpoint.clone(),
        );

        next_state.source_cursors.insert(
            candidate.source.clone(),
            SourceCursor {
                generation: candidate.generation,
                seal_digest: candidate.seal_digest.clone(),
                receipt_digest: candidate.receipt_digest.clone(),
                observer_profile_digest: candidate.observer_profile_digest.clone(),
                boot_incarnation_digest: candidate.boot_incarnation_digest.clone(),
            },
        );

        next_state.policy_versions = Some(candidate.policy_versions.clone());
        next_state.accepted_receipts.insert(
            candidate.receipt_digest.clone(),
            AcceptedReceipt {
                source: candidate.source.clone(),
                generation: candidate.generation,
                previous_seal_digest: candidate.previous_seal_digest.clone(),
                seal_digest: candidate.seal_digest.clone(),
                receipt_digest: candidate.receipt_digest.clone(),
                observer_profile_digest: candidate.observer_profile_digest.clone(),
                boot_incarnation_digest: candidate.boot_incarnation_digest.clone(),
                checkpoint: candidate.checkpoint.clone(),
                policy_versions: candidate.policy_versions.clone(),
            },
        );

        Ok(TransitionDecision::Advance(AdvancePlan {
            expected_revision: self.revision,
            expected_state: self.clone(),
            next_state,
            receipt_digest: candidate.receipt_digest.clone(),
        }))
    }

    /// Apply a prepared plan using a compare-and-swap on the state revision.
    ///
    /// This models a linearizable backend transaction. It does not itself
    /// provide storage durability or protection against restoring old state.
    pub fn commit(&mut self, plan: AdvancePlan) -> Result<CommitOutcome, TransitionError> {
        if self.revision != plan.expected_revision {
            return Err(TransitionError::CompareConflict {
                expected: plan.expected_revision,
                actual: self.revision,
            });
        }
        if self != &plan.expected_state {
            return Err(TransitionError::StateSnapshotMismatch {
                revision: self.revision,
            });
        }

        let next_revision = self.revision.checked_add(1).ok_or(TransitionError::RevisionOverflow)?;
        if plan.next_state.revision != next_revision
            || !plan.next_state.accepted_receipts.contains_key(&plan.receipt_digest)
        {
            return Err(TransitionError::InvalidAdvancePlan);
        }

        *self = plan.next_state;
        Ok(CommitOutcome::Committed { revision: self.revision })
    }
}

fn validate_candidate(candidate: &Candidate) -> Result<(), TransitionError> {
    if candidate.generation == 0 {
        return Err(TransitionError::GenerationZero);
    }
    if !valid_id(&candidate.source.provisioned_host_id)
        || !valid_id(&candidate.source.source_id)
        || !valid_id(&candidate.source.partition_id)
        || !valid_digest(&candidate.seal_digest)
        || !valid_digest(&candidate.receipt_digest)
        || !valid_digest(&candidate.observer_profile_digest)
        || !valid_digest(&candidate.boot_incarnation_digest)
        || !valid_id(&candidate.checkpoint.log_id)
        || candidate.checkpoint.tree_size == 0
        || !valid_digest(&candidate.checkpoint.root_digest)
        || !valid_digest(&candidate.checkpoint.checkpoint_digest)
        || candidate.policy_versions.signer == 0
        || candidate.policy_versions.witness == 0
        || !valid_digest(&candidate.policy_versions.bundle_digest)
    {
        return Err(TransitionError::MalformedCandidate);
    }
    Ok(())
}

fn validate_source_transition(
    state: &TrustedState,
    candidate: &Candidate,
) -> Result<(), TransitionError> {
    match state.source_cursors.get(&candidate.source) {
        None => {
            if candidate.generation != 1 || candidate.previous_seal_digest.is_some() {
                return Err(TransitionError::GenerationBootstrapInvalid);
            }
            if candidate.profile_transition_evidence_digest.is_some()
                || candidate.boot_transition_evidence_digest.is_some()
            {
                return Err(TransitionError::UnexpectedProfileTransitionEvidence);
            }
        }
        Some(current) => {
            let expected = current.generation.checked_add(1).ok_or(TransitionError::GenerationOverflow)?;
            if candidate.generation < current.generation {
                return Err(TransitionError::StaleGeneration {
                    current: current.generation,
                    candidate: candidate.generation,
                });
            }
            if candidate.generation == current.generation {
                return Err(TransitionError::GenerationEquivocation {
                    generation: candidate.generation,
                });
            }
            if candidate.generation != expected {
                return Err(TransitionError::GenerationGap {
                    expected,
                    actual: candidate.generation,
                });
            }
            if candidate.previous_seal_digest.as_deref() != Some(current.seal_digest.as_str()) {
                return Err(TransitionError::PredecessorMismatch);
            }

            let profile_changed = candidate.observer_profile_digest != current.observer_profile_digest;
            match (profile_changed, candidate.profile_transition_evidence_digest.as_deref()) {
                (true, Some(digest)) if valid_digest(digest) => {}
                (true, _) => return Err(TransitionError::UnauthorizedProfileTransition),
                (false, Some(_)) => return Err(TransitionError::UnexpectedProfileTransitionEvidence),
                (false, None) => {}
            }

            let boot_changed = candidate.boot_incarnation_digest != current.boot_incarnation_digest;
            match (boot_changed, candidate.boot_transition_evidence_digest.as_deref()) {
                (true, Some(digest)) if valid_digest(digest) => {}
                (true, _) => return Err(TransitionError::UnauthorizedBootTransition),
                (false, Some(_)) => return Err(TransitionError::UnexpectedBootTransitionEvidence),
                (false, None) => {}
            }
        }
    }
    Ok(())
}

fn validate_policy_transition(
    state: &TrustedState,
    candidate: &Candidate,
) -> Result<(), TransitionError> {
    match state.policy_versions.as_ref() {
        None => {
            if candidate.policy_transition_evidence_digest.as_deref().is_some_and(valid_digest) {
                Ok(())
            } else {
                Err(TransitionError::PolicyBootstrapRequired)
            }
        }
        Some(current) => {
            if candidate.policy_versions.signer < current.signer
                || candidate.policy_versions.witness < current.witness
            {
                return Err(TransitionError::PolicyRollback);
            }

            let same_versions = candidate.policy_versions.signer == current.signer
                && candidate.policy_versions.witness == current.witness;

            if same_versions {
                if candidate.policy_versions.bundle_digest != current.bundle_digest {
                    return Err(TransitionError::PolicyVersionEquivocation);
                }
                if candidate.policy_transition_evidence_digest.is_some() {
                    return Err(TransitionError::UnexpectedPolicyTransitionEvidence);
                }
                return Ok(());
            }

            if candidate.policy_transition_evidence_digest.as_deref().is_some_and(valid_digest) {
                Ok(())
            } else {
                Err(TransitionError::PolicyTransitionUnauthorized)
            }
        }
    }
}

fn validate_checkpoint_transition(
    state: &TrustedState,
    candidate: &Candidate,
) -> Result<(), TransitionError> {
    match state.log_cursors.get(&candidate.checkpoint.log_id) {
        None => match &candidate.checkpoint_evidence {
            CheckpointEvidence::VerifiedPinnedBootstrap { pinned_checkpoint_digest }
                if pinned_checkpoint_digest == &candidate.checkpoint.checkpoint_digest =>
            {
                Ok(())
            }
            CheckpointEvidence::VerifiedPinnedBootstrap { .. } => {
                Err(TransitionError::LogBootstrapMismatch)
            }
            _ => Err(TransitionError::LogBootstrapRequired),
        },
        Some(current) => {
            if current.log_id != candidate.checkpoint.log_id {
                return Err(TransitionError::LogOriginMismatch);
            }
            if candidate.checkpoint.tree_size < current.tree_size {
                return Err(TransitionError::StaleCheckpoint {
                    current: current.tree_size,
                    candidate: candidate.checkpoint.tree_size,
                });
            }
            if candidate.checkpoint.tree_size == current.tree_size {
                if candidate.checkpoint.root_digest != current.root_digest {
                    return Err(TransitionError::CheckpointFork {
                        tree_size: candidate.checkpoint.tree_size,
                    });
                }
                return match &candidate.checkpoint_evidence {
                    CheckpointEvidence::VerifiedSameTree => Ok(()),
                    _ => Err(TransitionError::CheckpointEvidenceMismatch),
                };
            }

            match &candidate.checkpoint_evidence {
                CheckpointEvidence::VerifiedConsistentExtension { consistency_proof_digest }
                    if valid_digest(consistency_proof_digest) =>
                {
                    Ok(())
                }
                CheckpointEvidence::VerifiedConsistentExtension { .. } => {
                    Err(TransitionError::InvalidConsistencyProofReference)
                }
                _ => Err(TransitionError::CheckpointEvidenceMismatch),
            }
        }
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_ID_BYTES && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

/// Result of a built-in conformance smoke suite. This is a model check, not a
/// storage, cryptographic, or production anti-rollback qualification.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ConformanceReport {
    pub report_version: u32,
    pub passed: bool,
    pub scenarios_run: usize,
    pub scenarios_passed: usize,
    pub scenarios: Vec<ConformanceCase>,
    pub anti_rollback_qualified: bool,
    pub signature_verification_performed: bool,
    pub storage_durability_qualified: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ConformanceCase {
    pub id: String,
    pub passed: bool,
    pub expected: String,
    pub observed: String,
}

fn fixture_candidate() -> Candidate {
    Candidate {
        expected_revision: 0,
        source: SourceKey {
            provisioned_host_id: "host-A".to_string(),
            source_id: "bpf-process-exit".to_string(),
            partition_id: "cpu-0".to_string(),
        },
        generation: 1,
        previous_seal_digest: None,
        seal_digest: test_digest(1),
        receipt_digest: test_digest(2),
        observer_profile_digest: test_digest(3),
        boot_incarnation_digest: test_digest(4),
        profile_transition_evidence_digest: None,
        boot_transition_evidence_digest: None,
        policy_versions: PolicyVersions {
            signer: 2,
            witness: 2,
            bundle_digest: test_digest(5),
        },
        policy_transition_evidence_digest: Some(test_digest(6)),
        checkpoint: LogCheckpoint {
            log_id: "basalt-log-prod".to_string(),
            tree_size: 10,
            root_digest: test_digest(7),
            checkpoint_digest: test_digest(8),
        },
        checkpoint_evidence: CheckpointEvidence::VerifiedPinnedBootstrap {
            pinned_checkpoint_digest: test_digest(8),
        },
    }
}

fn successor_candidate(state: &TrustedState, generation: u64, receipt_seed: u8) -> Candidate {
    let current = state
        .source_cursor(&fixture_source_key())
        .expect("fixture source cursor exists");
    let checkpoint = state
        .log_cursor("basalt-log-prod")
        .expect("fixture log cursor exists")
        .clone();

    Candidate {
        expected_revision: state.revision(),
        source: fixture_source_key(),
        generation,
        previous_seal_digest: Some(current.seal_digest.clone()),
        seal_digest: test_digest(receipt_seed.wrapping_add(20)),
        receipt_digest: test_digest(receipt_seed),
        observer_profile_digest: current.observer_profile_digest.clone(),
        boot_incarnation_digest: current.boot_incarnation_digest.clone(),
        profile_transition_evidence_digest: None,
        boot_transition_evidence_digest: None,
        policy_versions: state.policy_versions().expect("fixture policy").clone(),
        policy_transition_evidence_digest: None,
        checkpoint,
        checkpoint_evidence: CheckpointEvidence::VerifiedSameTree,
    }
}

fn secondary_source_candidate(state: &TrustedState) -> Candidate {
    Candidate {
        expected_revision: state.revision(),
        source: SourceKey {
            provisioned_host_id: "host-A".to_string(),
            source_id: "bpf-process-exec".to_string(),
            partition_id: "logical-partition-0".to_string(),
        },
        generation: 1,
        previous_seal_digest: None,
        seal_digest: test_digest(70),
        receipt_digest: test_digest(71),
        observer_profile_digest: test_digest(72),
        boot_incarnation_digest: test_digest(73),
        profile_transition_evidence_digest: None,
        boot_transition_evidence_digest: None,
        policy_versions: state.policy_versions().expect("fixture policy").clone(),
        policy_transition_evidence_digest: None,
        checkpoint: state
            .log_cursor("basalt-log-prod")
            .expect("fixture log cursor")
            .clone(),
        checkpoint_evidence: CheckpointEvidence::VerifiedSameTree,
    }
}

fn fixture_source_key() -> SourceKey {
    SourceKey {
        provisioned_host_id: "host-A".to_string(),
        source_id: "bpf-process-exit".to_string(),
        partition_id: "cpu-0".to_string(),
    }
}

fn test_digest(seed: u8) -> String {
    format!("sha256:{:064x}", seed)
}

fn record_case(
    cases: &mut Vec<ConformanceCase>,
    id: &str,
    expected: &str,
    observed: impl Into<String>,
    passed: bool,
) {
    cases.push(ConformanceCase {
        id: id.to_string(),
        passed,
        expected: expected.to_string(),
        observed: observed.into(),
    });
}

/// Runs deterministic transition-model checks without an external runner.
pub fn run_conformance() -> ConformanceReport {
    let mut cases = Vec::new();

    let mut state = TrustedState::new();
    let initial = fixture_candidate();
    let first_plan = state.preview(&initial);
    let first_passed = match first_plan {
        Ok(TransitionDecision::Advance(plan)) => {
            matches!(state.commit(plan), Ok(CommitOutcome::Committed { revision: 1 }))
                && state.source_cursor(&fixture_source_key()).is_some()
                && state.log_cursor("basalt-log-prod").is_some()
                && state.accepted_receipt_count() == 1
        }
        _ => false,
    };
    record_case(&mut cases, "TSTATE-MODEL-001", "bootstrap commits source+log+receipt atomically", format!("{first_passed}"), first_passed);

    let retry = state.preview(&initial);
    let retry_passed = matches!(retry, Ok(TransitionDecision::AlreadyCommitted { revision: 1 }));
    record_case(&mut cases, "TSTATE-MODEL-002", "exact receipt retry is idempotent", format!("{retry:?}"), retry_passed);

    let next = successor_candidate(&state, 2, 9);
    let plan_a = state.preview(&next);
    let plan_b = state.preview(&next);
    let cas_passed = match (plan_a, plan_b) {
        (Ok(TransitionDecision::Advance(a)), Ok(TransitionDecision::Advance(b))) => {
            state.commit(a).is_ok()
                && matches!(state.commit(b), Err(TransitionError::CompareConflict { .. }))
                && state.revision() == 2
        }
        _ => false,
    };
    record_case(&mut cases, "TSTATE-MODEL-003", "racing identical plans produce one commit and one CAS conflict", format!("{cas_passed}"), cas_passed);

    let retry_after_conflict = state.preview(&next);
    let retry_after_conflict_passed = matches!(
        retry_after_conflict,
        Ok(TransitionDecision::AlreadyCommitted { revision: 2 })
    );
    record_case(
        &mut cases,
        "TSTATE-MODEL-004",
        "exact retry after a stale CAS discovers the durable accepted receipt",
        format!("{retry_after_conflict:?}"),
        retry_after_conflict_passed,
    );

    let gap = successor_candidate(&state, 4, 10);
    let gap_result = state.preview(&gap);
    let gap_passed = matches!(gap_result, Err(TransitionError::GenerationGap { .. }));
    record_case(&mut cases, "TSTATE-MODEL-005", "generation gaps fail closed", format!("{gap_result:?}"), gap_passed);

    let mut fork = successor_candidate(&state, 3, 11);
    fork.checkpoint.tree_size = 10;
    fork.checkpoint.root_digest = test_digest(77);
    fork.checkpoint_evidence = CheckpointEvidence::VerifiedSameTree;
    let fork_result = state.preview(&fork);
    let fork_passed = matches!(fork_result, Err(TransitionError::CheckpointFork { .. }));
    record_case(&mut cases, "TSTATE-MODEL-006", "same-size different-root checkpoint is rejected", format!("{fork_result:?}"), fork_passed);

    let mut stale = successor_candidate(&state, 3, 12);
    stale.checkpoint.tree_size = 9;
    stale.checkpoint_evidence = CheckpointEvidence::VerifiedConsistentExtension {
        consistency_proof_digest: test_digest(13),
    };
    let stale_result = state.preview(&stale);
    let stale_passed = matches!(stale_result, Err(TransitionError::StaleCheckpoint { .. }));
    record_case(&mut cases, "TSTATE-MODEL-007", "smaller checkpoint is rejected", format!("{stale_result:?}"), stale_passed);

    let mut missing_consistency = successor_candidate(&state, 3, 14);
    missing_consistency.checkpoint.tree_size = 11;
    missing_consistency.checkpoint.root_digest = test_digest(15);
    missing_consistency.checkpoint.checkpoint_digest = test_digest(16);
    missing_consistency.checkpoint_evidence = CheckpointEvidence::VerifiedConsistentExtension {
        consistency_proof_digest: "not-a-digest".to_string(),
    };
    let consistency_result = state.preview(&missing_consistency);
    let consistency_passed = matches!(consistency_result, Err(TransitionError::InvalidConsistencyProofReference));
    record_case(&mut cases, "TSTATE-MODEL-008", "malformed consistency proof reference is rejected", format!("{consistency_result:?}"), consistency_passed);

    let mut profile_change = successor_candidate(&state, 3, 17);
    profile_change.observer_profile_digest = test_digest(18);
    let profile_result = state.preview(&profile_change);
    let profile_rejected = matches!(profile_result, Err(TransitionError::UnauthorizedProfileTransition));
    record_case(&mut cases, "TSTATE-MODEL-009", "profile change without authorization is rejected", format!("{profile_result:?}"), profile_rejected);

    profile_change.profile_transition_evidence_digest = Some(test_digest(19));
    let authorized_profile_result = state.preview(&profile_change);
    let authorized_profile_passed = matches!(authorized_profile_result, Ok(TransitionDecision::Advance(_)));
    record_case(&mut cases, "TSTATE-MODEL-010", "authorized profile change preserves successor generation", format!("{authorized_profile_result:?}"), authorized_profile_passed);

    let mut boot_change = successor_candidate(&state, 3, 21);
    boot_change.boot_incarnation_digest = test_digest(22);
    let boot_result = state.preview(&boot_change);
    let boot_rejected = matches!(boot_result, Err(TransitionError::UnauthorizedBootTransition));
    record_case(&mut cases, "TSTATE-MODEL-011", "boot change without authorization is rejected", format!("{boot_result:?}"), boot_rejected);

    boot_change.boot_transition_evidence_digest = Some(test_digest(23));
    let authorized_boot_result = state.preview(&boot_change);
    let authorized_boot_passed = matches!(authorized_boot_result, Ok(TransitionDecision::Advance(_)));
    record_case(&mut cases, "TSTATE-MODEL-012", "authorized boot transition preserves successor generation", format!("{authorized_boot_result:?}"), authorized_boot_passed);

    let mut policy_rollback = successor_candidate(&state, 3, 24);
    policy_rollback.policy_versions.signer = 1;
    let policy_result = state.preview(&policy_rollback);
    let policy_passed = matches!(policy_result, Err(TransitionError::PolicyRollback));
    record_case(&mut cases, "TSTATE-MODEL-013", "policy versions cannot decrease", format!("{policy_result:?}"), policy_passed);

    let mut receipt_collision = successor_candidate(&state, 3, 25);
    receipt_collision.receipt_digest = initial.receipt_digest.clone();
    let collision_result = state.preview(&receipt_collision);
    let collision_passed = matches!(collision_result, Err(TransitionError::ReceiptDigestCollision));
    record_case(&mut cases, "TSTATE-MODEL-014", "receipt digest reuse with different semantics is rejected", format!("{collision_result:?}"), collision_passed);

    let mut stale_revision = successor_candidate(&state, 3, 26);
    stale_revision.expected_revision = 1;
    let revision_result = state.preview(&stale_revision);
    let revision_passed = matches!(revision_result, Err(TransitionError::CompareConflict { .. }));
    record_case(&mut cases, "TSTATE-MODEL-015", "candidate checked against stale revision is rejected", format!("{revision_result:?}"), revision_passed);

    let mut competing_source = secondary_source_candidate(&state);
    let first_stream = successor_candidate(&state, 3, 27);
    let first_plan = state.preview(&first_stream);
    let competing_plan = state.preview(&competing_source);
    let competing_sources_passed = match (first_plan, competing_plan) {
        (Ok(TransitionDecision::Advance(first)), Ok(TransitionDecision::Advance(second))) => {
            let first_committed = state.commit(first).is_ok();
            let stale_conflict = matches!(
                state.commit(second),
                Err(TransitionError::CompareConflict { .. })
            );
            competing_source.expected_revision = state.revision();
            let rebased = state.preview(&competing_source);
            let second_committed = match rebased {
                Ok(TransitionDecision::Advance(plan)) => state.commit(plan).is_ok(),
                _ => false,
            };
            first_committed
                && stale_conflict
                && second_committed
                && state.source_cursor(&fixture_source_key()).is_some_and(|cursor| cursor.generation == 3)
                && state.source_cursor(&competing_source.source).is_some_and(|cursor| cursor.generation == 1)
        }
        _ => false,
    };
    record_case(
        &mut cases,
        "TSTATE-MODEL-016",
        "independent streams re-verify after shared checkpoint CAS conflict without lost updates",
        format!("{competing_sources_passed}"),
        competing_sources_passed,
    );

    use std::sync::{Arc, Mutex};
    use std::thread;

    let mut concurrent_state = TrustedState::new();
    let initial_plan = match concurrent_state.preview(&fixture_candidate()) {
        Ok(TransitionDecision::Advance(plan)) => Some(plan),
        _ => None,
    };
    let bootstrap_passed = initial_plan
        .and_then(|plan| concurrent_state.commit(plan).ok())
        .is_some();
    let thread_race_passed = if bootstrap_passed {
        let concurrent_candidate = successor_candidate(&concurrent_state, 2, 28);
        let concurrent_plan = concurrent_state.preview(&concurrent_candidate);
        match concurrent_plan {
            Ok(TransitionDecision::Advance(plan)) => {
                let shared = Arc::new(Mutex::new(concurrent_state));
                let handles: Vec<_> = (0..16)
                    .map(|_| {
                        let shared = Arc::clone(&shared);
                        let plan = plan.clone();
                        thread::spawn(move || shared.lock().unwrap().commit(plan))
                    })
                    .collect();
                let outcomes: Vec<_> = handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect();
                let committed = outcomes
                    .iter()
                    .filter(|result| matches!(result, Ok(CommitOutcome::Committed { .. })))
                    .count();
                let conflicts = outcomes
                    .iter()
                    .filter(|result| matches!(result, Err(TransitionError::CompareConflict { .. })))
                    .count();
                committed == 1 && conflicts == 15 && shared.lock().unwrap().revision() == 2
            }
            _ => false,
        }
    } else {
        false
    };
    record_case(
        &mut cases,
        "TSTATE-MODEL-017",
        "16 concurrent identical plans produce one winner and 15 compare conflicts",
        format!("{thread_race_passed}"),
        thread_race_passed,
    );

    let mut forked_snapshot = state.clone();
    if let Some(checkpoint) = forked_snapshot.log_cursors.get_mut("basalt-log-prod") {
        checkpoint.root_digest = test_digest(88);
    }
    let fork_candidate = successor_candidate(&state, 3, 29);
    let fork_plan = state.preview(&fork_candidate);
    let snapshot_fork_passed = match fork_plan {
        Ok(TransitionDecision::Advance(plan)) => matches!(
            forked_snapshot.commit(plan),
            Err(TransitionError::StateSnapshotMismatch { revision })
                if revision == state.revision()
        ),
        _ => false,
    };
    record_case(
        &mut cases,
        "TSTATE-MODEL-018",
        "same revision but divergent state snapshot cannot accept a stale plan",
        format!("{snapshot_fork_passed}"),
        snapshot_fork_passed,
    );

    let passed = cases.iter().all(|case| case.passed);
    let scenarios_run = cases.len();
    let scenarios_passed = cases.iter().filter(|case| case.passed).count();
    ConformanceReport {
        report_version: 1,
        passed,
        scenarios_run,
        scenarios_passed,
        scenarios: cases,
        anti_rollback_qualified: false,
        signature_verification_performed: false,
        storage_durability_qualified: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bootstrap() -> TrustedState {
        let mut state = TrustedState::new();
        let candidate = fixture_candidate();
        let plan = match state.preview(&candidate).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("empty state cannot contain a retry"),
        };
        state.commit(plan).unwrap();
        state
    }

    #[test]
    fn reference_conformance_suite_passes_without_promoting_external_trust() {
        let report = run_conformance();
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.scenarios_run, report.scenarios_passed);
        assert!(!report.anti_rollback_qualified);
        assert!(!report.signature_verification_performed);
        assert!(!report.storage_durability_qualified);
    }

    #[test]
    fn exact_receipt_retry_remains_idempotent_after_log_checkpoint_advances() {
        let mut state = bootstrap();
        let original = fixture_candidate();
        let mut successor = successor_candidate(&state, 2, 62);
        successor.checkpoint.tree_size = 11;
        successor.checkpoint.root_digest = test_digest(63);
        successor.checkpoint.checkpoint_digest = test_digest(64);
        successor.checkpoint_evidence = CheckpointEvidence::VerifiedConsistentExtension {
            consistency_proof_digest: test_digest(65),
        };
        let plan = match state.preview(&successor).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("successor is new"),
        };
        state.commit(plan).unwrap();
        assert_eq!(
            state.preview(&original),
            Ok(TransitionDecision::AlreadyCommitted { revision: 2 })
        );
    }

    #[test]
    fn profile_change_does_not_reset_stable_source_generation() {
        let state = bootstrap();
        let mut candidate = successor_candidate(&state, 2, 40);
        candidate.observer_profile_digest = test_digest(41);
        assert_eq!(
            state.preview(&candidate),
            Err(TransitionError::UnauthorizedProfileTransition)
        );
        candidate.profile_transition_evidence_digest = Some(test_digest(42));
        assert!(matches!(state.preview(&candidate), Ok(TransitionDecision::Advance(_))));
    }

    #[test]
    fn boot_transition_does_not_reset_source_generation() {
        let state = bootstrap();
        let mut candidate = successor_candidate(&state, 2, 43);
        candidate.boot_incarnation_digest = test_digest(44);
        assert_eq!(
            state.preview(&candidate),
            Err(TransitionError::UnauthorizedBootTransition)
        );
        candidate.boot_transition_evidence_digest = Some(test_digest(45));
        assert!(matches!(state.preview(&candidate), Ok(TransitionDecision::Advance(_))));
    }

    #[test]
    fn policy_versions_never_decrease_or_change_digest_without_version_change() {
        let state = bootstrap();
        let mut candidate = successor_candidate(&state, 2, 46);
        candidate.policy_versions.signer = 1;
        assert_eq!(state.preview(&candidate), Err(TransitionError::PolicyRollback));

        candidate.policy_versions.signer = 3;
        assert_eq!(
            state.preview(&candidate),
            Err(TransitionError::PolicyTransitionUnauthorized)
        );
        candidate.policy_transition_evidence_digest = Some(test_digest(47));
        assert!(matches!(state.preview(&candidate), Ok(TransitionDecision::Advance(_))));
    }

    #[test]
    fn source_generation_equivocation_is_rejected() {
        let state = bootstrap();
        let mut candidate = fixture_candidate();
        candidate.receipt_digest = test_digest(52);
        candidate.seal_digest = test_digest(53);
        candidate.expected_revision = state.revision();
        assert_eq!(
            state.preview(&candidate),
            Err(TransitionError::GenerationEquivocation { generation: 1 })
        );
    }

    #[test]
    fn older_generation_is_rejected_after_a_successor_commits() {
        let mut state = bootstrap();
        let successor = successor_candidate(&state, 2, 54);
        let plan = match state.preview(&successor).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("successor is new"),
        };
        state.commit(plan).unwrap();
        let mut old = successor_candidate(&state, 1, 55);
        old.previous_seal_digest = None;
        assert!(matches!(
            state.preview(&old),
            Err(TransitionError::StaleGeneration { .. })
        ));
    }

    #[test]
    fn verified_log_extension_advances_checkpoint_with_source_cursor() {
        let state = bootstrap();
        let mut candidate = successor_candidate(&state, 2, 56);
        candidate.checkpoint.tree_size = 11;
        candidate.checkpoint.root_digest = test_digest(57);
        candidate.checkpoint.checkpoint_digest = test_digest(58);
        candidate.checkpoint_evidence = CheckpointEvidence::VerifiedConsistentExtension {
            consistency_proof_digest: test_digest(59),
        };
        assert!(matches!(
            state.preview(&candidate),
            Ok(TransitionDecision::Advance(_))
        ));
    }

    #[test]
    fn policy_bundle_cannot_change_without_a_version_transition() {
        let state = bootstrap();
        let mut candidate = successor_candidate(&state, 2, 60);
        candidate.policy_versions.bundle_digest = test_digest(61);
        assert_eq!(
            state.preview(&candidate),
            Err(TransitionError::PolicyVersionEquivocation)
        );
    }

    #[test]
    fn new_log_requires_an_explicit_pinned_checkpoint() {
        let mut state = TrustedState::new();
        let mut candidate = fixture_candidate();
        candidate.checkpoint_evidence = CheckpointEvidence::VerifiedSameTree;
        assert_eq!(
            state.preview(&candidate),
            Err(TransitionError::LogBootstrapRequired)
        );
    }

    #[test]
    fn generation_overflow_is_rejected_without_mutation() {
        let mut state = bootstrap();
        let cursor = state.source_cursors.get_mut(&fixture_source_key()).unwrap();
        cursor.generation = u64::MAX;
        let before = state.clone();
        let candidate = successor_candidate(&state, u64::MAX, 48);
        assert_eq!(state.preview(&candidate), Err(TransitionError::GenerationOverflow));
        assert_eq!(state, before);
    }

    #[test]
    fn revision_overflow_is_rejected_without_mutation() {
        let mut state = bootstrap();
        state.revision = u64::MAX;
        let before = state.clone();
        let candidate = successor_candidate(&state, 2, 49);
        assert_eq!(state.preview(&candidate), Err(TransitionError::RevisionOverflow));
        assert_eq!(state, before);
    }

    #[test]
    fn failed_compare_and_swap_leaves_all_state_unchanged() {
        let mut state = bootstrap();
        let candidate = successor_candidate(&state, 2, 50);
        let plan = match state.preview(&candidate).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("candidate is new"),
        };
        state.revision += 1;
        let before = state.clone();
        assert!(matches!(
            state.commit(plan),
            Err(TransitionError::CompareConflict { .. })
        ));
        assert_eq!(state, before);
    }

    #[test]
    fn sixteen_concurrent_same_plan_commits_have_exactly_one_winner() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let snapshot = bootstrap();
        let candidate = successor_candidate(&snapshot, 2, 74);
        let plan = match snapshot.preview(&candidate).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("candidate is new"),
        };
        let state = Arc::new(Mutex::new(snapshot));
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let shared = Arc::clone(&state);
                let plan = plan.clone();
                thread::spawn(move || shared.lock().unwrap().commit(plan))
            })
            .collect();

        let outcomes: Vec<_> = handles.into_iter().map(|handle| handle.join().unwrap()).collect();
        let committed = outcomes
            .iter()
            .filter(|result| matches!(result, Ok(CommitOutcome::Committed { .. })))
            .count();
        let conflicts = outcomes
            .iter()
            .filter(|result| matches!(result, Err(TransitionError::CompareConflict { .. })))
            .count();

        assert_eq!(committed, 1);
        assert_eq!(conflicts, 15);
        assert_eq!(state.lock().unwrap().revision(), 2);
    }

    #[test]
    fn distinct_sources_reverify_after_checkpoint_cas_conflict() {
        let mut state = bootstrap();
        let first = successor_candidate(&state, 2, 75);
        let second = secondary_source_candidate(&state);
        let first_plan = match state.preview(&first).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("first stream is new"),
        };
        let second_plan = match state.preview(&second).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("second stream is new"),
        };

        state.commit(first_plan).unwrap();
        assert!(matches!(
            state.commit(second_plan),
            Err(TransitionError::CompareConflict { .. })
        ));

        let mut rebased = second;
        rebased.expected_revision = state.revision();
        let rebased_plan = match state.preview(&rebased).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("second stream was not committed"),
        };
        state.commit(rebased_plan).unwrap();

        assert_eq!(state.source_cursor(&fixture_source_key()).unwrap().generation, 2);
        assert_eq!(state.source_cursor(&rebased.source).unwrap().generation, 1);
        assert_eq!(state.accepted_receipt_count(), 3);
        assert_eq!(state.revision(), 3);
    }

    #[test]
    fn same_revision_divergent_snapshot_rejects_prepared_plan() {
        let state = bootstrap();
        let candidate = successor_candidate(&state, 2, 76);
        let plan = match state.preview(&candidate).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("candidate is new"),
        };

        let mut divergent = state.clone();
        divergent
            .log_cursors
            .get_mut("basalt-log-prod")
            .unwrap()
            .root_digest = test_digest(77);
        let before = divergent.clone();

        assert_eq!(
            divergent.commit(plan),
            Err(TransitionError::StateSnapshotMismatch { revision: state.revision() })
        );
        assert_eq!(divergent, before);
    }

    #[test]
    fn source_log_and_receipt_advance_as_one_model_transition() {
        let mut state = bootstrap();
        let candidate = successor_candidate(&state, 2, 51);
        let old_log = state.log_cursor("basalt-log-prod").unwrap().clone();
        let old_source = state.source_cursor(&fixture_source_key()).unwrap().clone();
        let plan = match state.preview(&candidate).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("candidate is new"),
        };
        assert_eq!(state.log_cursor("basalt-log-prod"), Some(&old_log));
        assert_eq!(state.source_cursor(&fixture_source_key()), Some(&old_source));
        state.commit(plan).unwrap();
        assert_eq!(state.source_cursor(&fixture_source_key()).unwrap().generation, 2);
        assert_eq!(state.accepted_receipt_count(), 2);
    }

    #[test]
    fn sqlite_store_reopens_with_exact_source_log_and_receipt_state() {
        let path = temporary_sqlite_path("reopen");
        let candidate = fixture_candidate();
        {
            let mut store = SqliteTrustedStateStore::open(&path).unwrap();
            let plan = match store.preview(&candidate).unwrap() {
                TransitionDecision::Advance(plan) => plan,
                TransitionDecision::AlreadyCommitted { .. } => panic!("fresh database has no receipt"),
            };
            assert_eq!(
                store.commit_plan(plan).unwrap(),
                CommitOutcome::Committed { revision: 1 }
            );
            let report = store.report().unwrap();
            assert_eq!(report.journal_mode.to_ascii_lowercase(), "wal");
            assert_eq!(report.synchronous_level, 2);
            assert!(!report.anti_rollback_qualified);
            assert!(!report.signature_verification_performed);
            assert!(!report.external_freshness_established);
        }
        {
            let store = SqliteTrustedStateStore::open(&path).unwrap();
            let state = store.read_state().unwrap();
            assert_eq!(state.revision(), 1);
            assert_eq!(state.source_cursor(&fixture_source_key()).unwrap().generation, 1);
            assert_eq!(state.accepted_receipt_count(), 1);
            assert_eq!(
                store.preview(&candidate).unwrap(),
                TransitionDecision::AlreadyCommitted { revision: 1 }
            );
        }
        cleanup_sqlite_path(&path);
    }

    #[test]
    fn sqlite_write_boundary_failures_rollback_all_rows() {
        let stages = [
            SqliteCommitStage::SourceCursorWritten,
            SqliteCommitStage::LogCursorWritten,
            SqliteCommitStage::ReceiptWritten,
            SqliteCommitStage::MetaWritten,
        ];
        for (index, fail_stage) in stages.into_iter().enumerate() {
            let path = temporary_sqlite_path(&format!("rollback-{index}"));
            let mut store = SqliteTrustedStateStore::open(&path).unwrap();
            let bootstrap_candidate = fixture_candidate();
            let bootstrap_plan = match store.preview(&bootstrap_candidate).unwrap() {
                TransitionDecision::Advance(plan) => plan,
                TransitionDecision::AlreadyCommitted { .. } => panic!("fresh database has no receipt"),
            };
            store.commit_plan(bootstrap_plan).unwrap();
            let before = store.read_state().unwrap();
            let candidate = successor_candidate(&before, 2, 90 + index as u8);
            let plan = match store.preview(&candidate).unwrap() {
                TransitionDecision::Advance(plan) => plan,
                TransitionDecision::AlreadyCommitted { .. } => panic!("successor receipt is new"),
            };
            let result = store.commit_plan_with_hook(plan, |stage| {
                if stage == fail_stage {
                    anyhow::bail!("injected transactional abort at {stage:?}");
                }
                Ok(())
            });
            assert!(matches!(result, Err(DurableStoreError::Storage(_))));
            drop(store);

            let reopened = SqliteTrustedStateStore::open(&path).unwrap();
            assert_eq!(reopened.read_state().unwrap(), before);
            let quick_check: String = reopened
                .connection
                .query_row("PRAGMA quick_check", [], |row| row.get(0))
                .unwrap();
            assert_eq!(quick_check, "ok");
            drop(reopened);
            cleanup_sqlite_path(&path);
        }
    }

    #[test]
    fn sqlite_exact_receipt_retry_after_restart_discovers_committed_state() {
        let path = temporary_sqlite_path("lost-ack");
        let candidate = fixture_candidate();
        {
            let mut store = SqliteTrustedStateStore::open(&path).unwrap();
            let plan = match store.preview(&candidate).unwrap() {
                TransitionDecision::Advance(plan) => plan,
                TransitionDecision::AlreadyCommitted { .. } => panic!("fresh database has no receipt"),
            };
            store.commit_plan(plan).unwrap();
        }
        {
            let reopened = SqliteTrustedStateStore::open(&path).unwrap();
            assert_eq!(
                reopened.preview(&candidate).unwrap(),
                TransitionDecision::AlreadyCommitted { revision: 1 }
            );
            assert_eq!(reopened.read_state().unwrap().accepted_receipt_count(), 1);
        }
        cleanup_sqlite_path(&path);
    }

    #[test]
    fn sqlite_two_connections_recover_exact_retry_after_compare_conflict() {
        let path = temporary_sqlite_path("two-connection");
        let mut first = SqliteTrustedStateStore::open(&path).unwrap();
        let mut second = SqliteTrustedStateStore::open(&path).unwrap();
        let candidate = fixture_candidate();
        let first_plan = match first.preview(&candidate).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("fresh database has no receipt"),
        };
        let second_plan = match second.preview(&candidate).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("second connection read before commit"),
        };
        first.commit_plan(first_plan).unwrap();
        assert!(matches!(
            second.commit_plan(second_plan),
            Err(DurableStoreError::Transition(TransitionError::CompareConflict { .. }))
        ));
        let reloaded = second.read_state().unwrap();
        assert_eq!(
            reloaded.preview(&candidate).unwrap(),
            TransitionDecision::AlreadyCommitted { revision: 1 }
        );
        drop(first);
        drop(second);
        cleanup_sqlite_path(&path);
    }

    #[test]
    fn sqlite_rejects_plan_from_divergent_same_revision_database_snapshot() {
        let path_a = temporary_sqlite_path("fork-a");
        let path_b = temporary_sqlite_path("fork-b");
        let mut store_a = SqliteTrustedStateStore::open(&path_a).unwrap();
        let mut store_b = SqliteTrustedStateStore::open(&path_b).unwrap();

        let candidate_a = fixture_candidate();
        let mut candidate_b = fixture_candidate();
        candidate_b.receipt_digest = test_digest(92);
        candidate_b.seal_digest = test_digest(93);
        candidate_b.checkpoint.root_digest = test_digest(94);
        candidate_b.checkpoint.checkpoint_digest = test_digest(95);
        candidate_b.checkpoint_evidence = CheckpointEvidence::VerifiedPinnedBootstrap {
            pinned_checkpoint_digest: test_digest(95),
        };

        let plan_a = match store_a.preview(&candidate_a).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("fresh database A"),
        };
        let plan_b = match store_b.preview(&candidate_b).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("fresh database B"),
        };
        store_a.commit_plan(plan_a).unwrap();
        store_b.commit_plan(plan_b).unwrap();

        let candidate_next = successor_candidate(&store_a.read_state().unwrap(), 2, 96);
        let foreign_plan = match store_a.preview(&candidate_next).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("successor is new"),
        };
        assert!(matches!(
            store_b.commit_plan(foreign_plan),
            Err(DurableStoreError::Transition(TransitionError::StateSnapshotMismatch { .. }))
        ));

        drop(store_a);
        drop(store_b);
        cleanup_sqlite_path(&path_a);
        cleanup_sqlite_path(&path_b);
    }

    #[test]
    fn sqlite_eight_connections_have_one_winner_for_the_same_plan() {
        use std::thread;

        let path = temporary_sqlite_path("concurrent");
        let mut seed = SqliteTrustedStateStore::open(&path).unwrap();
        let bootstrap_candidate = fixture_candidate();
        let bootstrap_plan = match seed.preview(&bootstrap_candidate).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("fresh database"),
        };
        seed.commit_plan(bootstrap_plan).unwrap();

        let candidate = successor_candidate(&seed.read_state().unwrap(), 2, 97);
        let shared_plan = match seed.preview(&candidate).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("candidate is new"),
        };
        let stores: Vec<_> = (0..8)
            .map(|_| SqliteTrustedStateStore::open(&path).unwrap())
            .collect();
        let handles: Vec<_> = stores
            .into_iter()
            .map(|mut store| {
                let plan = shared_plan.clone();
                thread::spawn(move || store.commit_plan(plan))
            })
            .collect();
        let outcomes: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        let committed = outcomes
            .iter()
            .filter(|result| matches!(result, Ok(CommitOutcome::Committed { .. })))
            .count();
        let conflicts = outcomes
            .iter()
            .filter(|result| {
                matches!(
                    result,
                    Err(DurableStoreError::Transition(TransitionError::CompareConflict { .. }))
                )
            })
            .count();
        assert_eq!(committed, 1);
        assert_eq!(conflicts, 7);
        assert_eq!(seed.read_state().unwrap().revision(), 2);
        drop(seed);
        cleanup_sqlite_path(&path);
    }

    #[test]
    fn sqlite_unkeyed_state_digest_detects_inconsistent_database_edits() {
        let path = temporary_sqlite_path("digest-corruption");
        let mut store = SqliteTrustedStateStore::open(&path).unwrap();
        let candidate = fixture_candidate();
        let plan = match store.preview(&candidate).unwrap() {
            TransitionDecision::Advance(plan) => plan,
            TransitionDecision::AlreadyCommitted { .. } => panic!("fresh database"),
        };
        store.commit_plan(plan).unwrap();
        store
            .connection
            .execute(
                "UPDATE basalt_state_meta SET state_digest=?1 WHERE singleton=1",
                params![test_digest(99)],
            )
            .unwrap();
        drop(store);
        assert!(SqliteTrustedStateStore::open(&path).is_err());
        cleanup_sqlite_path(&path);
    }
}


// ---------------------------------------------------------------------------
// SQLite durable backend prototype
// ---------------------------------------------------------------------------
//
// This backend makes the local transaction boundary executable. It is not an
// external anti-rollback anchor: an attacker who can restore an old database
// snapshot can restore an old, internally consistent state. The state digest
// detects accidental/inconsistent modification only; it is unkeyed.
//
// One SQLite row per logical record keeps source/log/receipt history queryable.
// Every accepted receipt transition is written under one BEGIN IMMEDIATE
// transaction. WAL + synchronous=FULL is set and checked on every connection.
// The exact expected in-memory snapshot is compared after acquiring the writer
// transaction; the revision alone is not sufficient.

const SQLITE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS basalt_state_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    revision TEXT NOT NULL,
    signer_policy_version TEXT,
    witness_policy_version TEXT,
    policy_bundle_digest TEXT,
    state_digest TEXT NOT NULL,
    CHECK (
        (signer_policy_version IS NULL AND witness_policy_version IS NULL AND policy_bundle_digest IS NULL)
        OR
        (signer_policy_version IS NOT NULL AND witness_policy_version IS NOT NULL AND policy_bundle_digest IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS basalt_log_cursors (
    log_id TEXT PRIMARY KEY,
    tree_size TEXT NOT NULL,
    root_digest TEXT NOT NULL,
    checkpoint_digest TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS basalt_source_cursors (
    host_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    partition_id TEXT NOT NULL,
    generation TEXT NOT NULL,
    seal_digest TEXT NOT NULL,
    receipt_digest TEXT NOT NULL,
    observer_profile_digest TEXT NOT NULL,
    boot_incarnation_digest TEXT NOT NULL,
    PRIMARY KEY (host_id, source_id, partition_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS basalt_accepted_receipts (
    receipt_digest TEXT PRIMARY KEY,
    host_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    partition_id TEXT NOT NULL,
    generation TEXT NOT NULL,
    previous_seal_digest TEXT,
    seal_digest TEXT NOT NULL,
    observer_profile_digest TEXT NOT NULL,
    boot_incarnation_digest TEXT NOT NULL,
    log_id TEXT NOT NULL,
    tree_size TEXT NOT NULL,
    root_digest TEXT NOT NULL,
    checkpoint_digest TEXT NOT NULL,
    signer_policy_version TEXT NOT NULL,
    witness_policy_version TEXT NOT NULL,
    policy_bundle_digest TEXT NOT NULL,
    UNIQUE (host_id, source_id, partition_id, generation),
    FOREIGN KEY (host_id, source_id, partition_id)
        REFERENCES basalt_source_cursors(host_id, source_id, partition_id),
    FOREIGN KEY (log_id) REFERENCES basalt_log_cursors(log_id)
) WITHOUT ROWID;
";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurableStoreError {
    Transition(TransitionError),
    Storage(String),
    CommitOutcomeIndeterminate(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SqliteCommitStage {
    SourceCursorWritten,
    LogCursorWritten,
    ReceiptWritten,
    MetaWritten,
}

pub struct SqliteTrustedStateStore {
    connection: Connection,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SqliteStoreReport {
    pub report_version: u32,
    pub backend: String,
    pub journal_mode: String,
    pub synchronous_level: i64,
    pub revision: u64,
    pub log_cursor_count: usize,
    pub source_cursor_count: usize,
    pub accepted_receipt_count: usize,
    pub state_digest: String,
    pub local_atomicity_backend_configured: bool,
    pub local_snapshot_digest_consistent: bool,
    pub anti_rollback_qualified: bool,
    pub signature_verification_performed: bool,
    pub external_freshness_established: bool,
}

impl SqliteTrustedStateStore {
    /// Opens or initializes the local state store with a pinned durability
    /// configuration. The parent directory must already exist.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        // Refuse an existing symlink or non-regular file. The containing
        // directory must itself be trusted against path replacement.
        match std::fs::symlink_metadata(path.as_ref()) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                anyhow::bail!("BASALT SQLite state path must be a regular non-symlink file");
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }

        let open_flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NOFOLLOW;
        let mut connection = Connection::open_with_flags(path.as_ref(), open_flags)
            .with_context(|| format!("open BASALT SQLite state store at {}", path.as_ref().display()))?;
        connection.busy_timeout(Duration::from_secs(5))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::symlink_metadata(path.as_ref())?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                anyhow::bail!("BASALT SQLite state path changed to a non-regular file");
            }
            std::fs::set_permissions(path.as_ref(), std::fs::Permissions::from_mode(0o600))?;
        }

        connection.pragma_update(None, "journal_mode", "WAL")?;
        let journal_mode: String =
            connection.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
        if !journal_mode.eq_ignore_ascii_case("wal") {
            anyhow::bail!("BASALT SQLite backend requires WAL mode, got {journal_mode}");
        }

        // SQLite documents synchronous=FULL as syncing the WAL at each
        // transaction commit. This is a local durability choice, not rollback
        // protection against restoration of the complete database snapshot.
        connection.pragma_update(None, "synchronous", 2_i64)?;
        let synchronous_level: i64 =
            connection.pragma_query_value(None, "synchronous", |row| row.get(0))?;
        if synchronous_level != 2 {
            anyhow::bail!(
                "BASALT SQLite backend requires synchronous=FULL (2), got {synchronous_level}"
            );
        }
        connection.pragma_update(None, "foreign_keys", 1_i64)?;

        connection.execute_batch(SQLITE_SCHEMA)?;
        {
            let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let existing: Option<i64> = tx
                .query_row(
                    "SELECT singleton FROM basalt_state_meta WHERE singleton = 1",
                    [],
                    |row| row.get(0),
                )
                .optional()?;
            if existing.is_none() {
                let initial = TrustedState::new();
                tx.execute(
                    "INSERT INTO basalt_state_meta
                     (singleton, revision, signer_policy_version, witness_policy_version,
                      policy_bundle_digest, state_digest)
                     VALUES (1, '0', NULL, NULL, NULL, ?1)",
                    params![trusted_state_digest(&initial)],
                )?;
            }
            tx.commit()?;
        }

        // Fail closed on malformed, inconsistent or tampered persisted state.
        let _ = read_sqlite_state(&connection)?;
        Ok(Self { connection })
    }

    pub fn read_state(&self) -> anyhow::Result<TrustedState> {
        read_sqlite_state(&self.connection)
    }

    /// Prepares against the exact state currently read from the database.
    /// The returned plan is still only a proposal until commit_plan succeeds.
    pub fn preview(&self, candidate: &Candidate) -> anyhow::Result<TransitionDecision> {
        self.read_state()?
            .preview(candidate)
            .map_err(|error| anyhow::anyhow!("BASALT state transition rejected: {error:?}"))
    }

    pub fn commit_plan(
        &mut self,
        plan: AdvancePlan,
    ) -> Result<CommitOutcome, DurableStoreError> {
        self.commit_plan_with_hook(plan, |_| Ok(()))
    }

    fn commit_plan_with_hook<F>(
        &mut self,
        plan: AdvancePlan,
        mut after_stage: F,
    ) -> Result<CommitOutcome, DurableStoreError>
    where
        F: FnMut(SqliteCommitStage) -> anyhow::Result<()>,
    {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| DurableStoreError::Storage(error.to_string()))?;

        let current = read_sqlite_state(&tx)
            .map_err(|error| DurableStoreError::Storage(error.to_string()))?;

        if current.revision != plan.expected_revision {
            return Err(DurableStoreError::Transition(
                TransitionError::CompareConflict {
                    expected: plan.expected_revision,
                    actual: current.revision,
                },
            ));
        }
        if current != plan.expected_state {
            return Err(DurableStoreError::Transition(
                TransitionError::StateSnapshotMismatch {
                    revision: current.revision,
                },
            ));
        }
        let Some(planned_receipt) = plan.next_state.accepted_receipts.get(&plan.receipt_digest) else {
            return Err(DurableStoreError::Transition(
                TransitionError::InvalidAdvancePlan,
            ));
        };
        if current.log_cursors.len() >= MAX_SQLITE_LOG_CURSORS
            && !current.log_cursors.contains_key(&planned_receipt.checkpoint.log_id)
        {
            return Err(DurableStoreError::Storage(
                "BASALT SQLite log-cursor limit reached".to_string(),
            ));
        }
        if current.source_cursors.len() >= MAX_SQLITE_SOURCE_CURSORS
            && !current.source_cursors.contains_key(&planned_receipt.source)
        {
            return Err(DurableStoreError::Storage(
                "BASALT SQLite source-cursor limit reached".to_string(),
            ));
        }
        if current.accepted_receipts.len() >= MAX_SQLITE_ACCEPTED_RECEIPTS {
            return Err(DurableStoreError::Storage(
                "BASALT SQLite accepted-receipt limit reached".to_string(),
            ));
        }

        let next_revision = current
            .revision
            .checked_add(1)
            .ok_or_else(|| DurableStoreError::Transition(TransitionError::RevisionOverflow))?;
        if plan.next_state.revision != next_revision
            || !plan.next_state.accepted_receipts.contains_key(&plan.receipt_digest)
        {
            return Err(DurableStoreError::Transition(
                TransitionError::InvalidAdvancePlan,
            ));
        }

        write_sqlite_transition(&tx, &plan, &mut after_stage)
            .map_err(|error| DurableStoreError::Storage(error.to_string()))?;

        tx.commit().map_err(|error| {
            DurableStoreError::CommitOutcomeIndeterminate(error.to_string())
        })?;
        Ok(CommitOutcome::Committed {
            revision: plan.next_state.revision,
        })
    }

    pub fn report(&self) -> anyhow::Result<SqliteStoreReport> {
        let state = self.read_state()?;
        let journal_mode: String =
            self.connection
                .pragma_query_value(None, "journal_mode", |row| row.get(0))?;
        let synchronous_level: i64 =
            self.connection
                .pragma_query_value(None, "synchronous", |row| row.get(0))?;
        Ok(SqliteStoreReport {
            report_version: 1,
            backend: "sqlite-wal-full-v1".to_string(),
            journal_mode,
            synchronous_level,
            revision: state.revision,
            log_cursor_count: state.log_cursors.len(),
            source_cursor_count: state.source_cursors.len(),
            accepted_receipt_count: state.accepted_receipts.len(),
            state_digest: trusted_state_digest(&state),
            local_atomicity_backend_configured: synchronous_level == 2,
            local_snapshot_digest_consistent: true,
            // No independent anti-rollback/freshness anchor is connected here.
            anti_rollback_qualified: false,
            signature_verification_performed: false,
            external_freshness_established: false,
        })
    }
}

fn parse_sql_u64(value: &str, field: &str) -> anyhow::Result<u64> {
    value
        .parse::<u64>()
        .with_context(|| format!("invalid persisted u64 in {field}"))
}

fn read_sqlite_state(connection: &Connection) -> anyhow::Result<TrustedState> {
    let (
        revision_text,
        signer_policy_text,
        witness_policy_text,
        policy_bundle_digest,
        expected_state_digest,
    ): (String, Option<String>, Option<String>, Option<String>, String) =
        connection.query_row(
            "SELECT revision, signer_policy_version, witness_policy_version,
                    policy_bundle_digest, state_digest
             FROM basalt_state_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        )?;

    let revision = parse_sql_u64(&revision_text, "revision")?;

    let log_count: i64 = connection.query_row(
        "SELECT COUNT(*) FROM basalt_log_cursors",
        [],
        |row| row.get(0),
    )?;
    let source_count: i64 = connection.query_row(
        "SELECT COUNT(*) FROM basalt_source_cursors",
        [],
        |row| row.get(0),
    )?;
    let receipt_count: i64 = connection.query_row(
        "SELECT COUNT(*) FROM basalt_accepted_receipts",
        [],
        |row| row.get(0),
    )?;
    if log_count < 0
        || source_count < 0
        || receipt_count < 0
        || log_count as usize > MAX_SQLITE_LOG_CURSORS
        || source_count as usize > MAX_SQLITE_SOURCE_CURSORS
        || receipt_count as usize > MAX_SQLITE_ACCEPTED_RECEIPTS
    {
        anyhow::bail!("persisted BASALT SQLite state exceeds configured record bounds");
    }

    let policy_versions = match (
        signer_policy_text,
        witness_policy_text,
        policy_bundle_digest,
    ) {
        (None, None, None) => None,
        (Some(signer), Some(witness), Some(bundle_digest)) => {
            let signer = parse_sql_u64(&signer, "signer_policy_version")?;
            let witness = parse_sql_u64(&witness, "witness_policy_version")?;
            if signer == 0 || witness == 0 || !valid_digest(&bundle_digest) {
                anyhow::bail!("invalid persisted policy version or digest");
            }
            Some(PolicyVersions {
                signer,
                witness,
                bundle_digest,
            })
        }
        _ => anyhow::bail!("partial persisted policy-version tuple"),
    };

    let mut state = TrustedState {
        revision,
        log_cursors: BTreeMap::new(),
        source_cursors: BTreeMap::new(),
        accepted_receipts: BTreeMap::new(),
        policy_versions,
    };

    {
        let mut statement = connection.prepare(
            "SELECT log_id, tree_size, root_digest, checkpoint_digest
             FROM basalt_log_cursors ORDER BY log_id",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        for row in rows {
            let (log_id, tree_size, root_digest, checkpoint_digest) = row?;
            let checkpoint = LogCheckpoint {
                log_id: log_id.clone(),
                tree_size: parse_sql_u64(&tree_size, "tree_size")?,
                root_digest,
                checkpoint_digest,
            };
            if !valid_id(&checkpoint.log_id)
                || checkpoint.tree_size == 0
                || !valid_digest(&checkpoint.root_digest)
                || !valid_digest(&checkpoint.checkpoint_digest)
                || state.log_cursors.insert(log_id, checkpoint).is_some()
            {
                anyhow::bail!("invalid or duplicate persisted log cursor");
            }
        }
    }

    {
        let mut statement = connection.prepare(
            "SELECT host_id, source_id, partition_id, generation, seal_digest,
                    receipt_digest, observer_profile_digest, boot_incarnation_digest
             FROM basalt_source_cursors ORDER BY host_id, source_id, partition_id",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
            ))
        })?;
        for row in rows {
            let (
                host_id,
                source_id,
                partition_id,
                generation,
                seal_digest,
                receipt_digest,
                observer_profile_digest,
                boot_incarnation_digest,
            ) = row?;
            let key = SourceKey {
                provisioned_host_id: host_id,
                source_id,
                partition_id,
            };
            let cursor = SourceCursor {
                generation: parse_sql_u64(&generation, "source generation")?,
                seal_digest,
                receipt_digest,
                observer_profile_digest,
                boot_incarnation_digest,
            };
            if !valid_id(&key.provisioned_host_id)
                || !valid_id(&key.source_id)
                || !valid_id(&key.partition_id)
                || cursor.generation == 0
                || !valid_digest(&cursor.seal_digest)
                || !valid_digest(&cursor.receipt_digest)
                || !valid_digest(&cursor.observer_profile_digest)
                || !valid_digest(&cursor.boot_incarnation_digest)
                || state.source_cursors.insert(key, cursor).is_some()
            {
                anyhow::bail!("invalid or duplicate persisted source cursor");
            }
        }
    }

    {
        let mut statement = connection.prepare(
            "SELECT receipt_digest, host_id, source_id, partition_id, generation,
                    previous_seal_digest, seal_digest, observer_profile_digest,
                    boot_incarnation_digest, log_id, tree_size, root_digest,
                    checkpoint_digest, signer_policy_version, witness_policy_version,
                    policy_bundle_digest
             FROM basalt_accepted_receipts ORDER BY receipt_digest",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, String>(10)?,
                row.get::<_, String>(11)?,
                row.get::<_, String>(12)?,
                row.get::<_, String>(13)?,
                row.get::<_, String>(14)?,
                row.get::<_, String>(15)?,
            ))
        })?;
        for row in rows {
            let (
                receipt_digest,
                host_id,
                source_id,
                partition_id,
                generation,
                previous_seal_digest,
                seal_digest,
                observer_profile_digest,
                boot_incarnation_digest,
                log_id,
                tree_size,
                root_digest,
                checkpoint_digest,
                signer_policy,
                witness_policy,
                bundle_digest,
            ) = row?;
            let generation = parse_sql_u64(&generation, "accepted receipt generation")?;
            let previous_valid = if generation == 1 {
                previous_seal_digest.is_none()
            } else {
                previous_seal_digest.as_deref().is_some_and(valid_digest)
            };
            let receipt = AcceptedReceipt {
                source: SourceKey {
                    provisioned_host_id: host_id,
                    source_id,
                    partition_id,
                },
                generation,
                previous_seal_digest,
                seal_digest,
                receipt_digest: receipt_digest.clone(),
                observer_profile_digest,
                boot_incarnation_digest,
                checkpoint: LogCheckpoint {
                    log_id,
                    tree_size: parse_sql_u64(&tree_size, "accepted tree_size")?,
                    root_digest,
                    checkpoint_digest,
                },
                policy_versions: PolicyVersions {
                    signer: parse_sql_u64(&signer_policy, "accepted signer policy")?,
                    witness: parse_sql_u64(&witness_policy, "accepted witness policy")?,
                    bundle_digest,
                },
            };
            if !valid_digest(&receipt_digest)
                || !valid_id(&receipt.source.provisioned_host_id)
                || !valid_id(&receipt.source.source_id)
                || !valid_id(&receipt.source.partition_id)
                || !previous_valid
                || !valid_digest(&receipt.seal_digest)
                || !valid_digest(&receipt.observer_profile_digest)
                || !valid_digest(&receipt.boot_incarnation_digest)
                || !valid_id(&receipt.checkpoint.log_id)
                || receipt.checkpoint.tree_size == 0
                || !valid_digest(&receipt.checkpoint.root_digest)
                || !valid_digest(&receipt.checkpoint.checkpoint_digest)
                || receipt.policy_versions.signer == 0
                || receipt.policy_versions.witness == 0
                || !valid_digest(&receipt.policy_versions.bundle_digest)
                || state.accepted_receipts.insert(receipt_digest, receipt).is_some()
            {
                anyhow::bail!("invalid or duplicate persisted accepted receipt");
            }
        }
    }

    for (key, cursor) in &state.source_cursors {
        let accepted = state
            .accepted_receipts
            .get(&cursor.receipt_digest)
            .ok_or_else(|| anyhow::anyhow!("source cursor has no accepted receipt record"))?;
        if &accepted.source != key
            || accepted.generation != cursor.generation
            || accepted.seal_digest != cursor.seal_digest
            || accepted.observer_profile_digest != cursor.observer_profile_digest
            || accepted.boot_incarnation_digest != cursor.boot_incarnation_digest
        {
            anyhow::bail!("source cursor does not match its accepted receipt");
        }
    }

    for (log_id, cursor) in &state.log_cursors {
        if !state.accepted_receipts.values().any(|receipt| {
            receipt.checkpoint.log_id == *log_id && receipt.checkpoint == *cursor
        }) {
            anyhow::bail!("log cursor has no matching accepted receipt checkpoint");
        }
    }

    let actual_state_digest = trusted_state_digest(&state);
    if actual_state_digest != expected_state_digest {
        anyhow::bail!("persisted trusted-state digest mismatch");
    }
    Ok(state)
}

fn write_sqlite_transition<F>(
    transaction: &Transaction<'_>,
    plan: &AdvancePlan,
    after_stage: &mut F,
) -> anyhow::Result<()>
where
    F: FnMut(SqliteCommitStage) -> anyhow::Result<()>,
{
    let receipt = plan
        .next_state
        .accepted_receipts
        .get(&plan.receipt_digest)
        .ok_or_else(|| anyhow::anyhow!("advance plan receipt is absent from next state"))?;
    let source = plan
        .next_state
        .source_cursors
        .get(&receipt.source)
        .ok_or_else(|| anyhow::anyhow!("advance plan source cursor is absent from next state"))?;
    let checkpoint = plan
        .next_state
        .log_cursors
        .get(&receipt.checkpoint.log_id)
        .ok_or_else(|| anyhow::anyhow!("advance plan log cursor is absent from next state"))?;
    let policies = plan
        .next_state
        .policy_versions
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("advance plan policy versions are absent"))?;

    transaction.execute(
        "INSERT INTO basalt_source_cursors
         (host_id, source_id, partition_id, generation, seal_digest, receipt_digest,
          observer_profile_digest, boot_incarnation_digest)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(host_id, source_id, partition_id) DO UPDATE SET
          generation=excluded.generation,
          seal_digest=excluded.seal_digest,
          receipt_digest=excluded.receipt_digest,
          observer_profile_digest=excluded.observer_profile_digest,
          boot_incarnation_digest=excluded.boot_incarnation_digest",
        params![
            receipt.source.provisioned_host_id,
            receipt.source.source_id,
            receipt.source.partition_id,
            source.generation.to_string(),
            source.seal_digest,
            source.receipt_digest,
            source.observer_profile_digest,
            source.boot_incarnation_digest,
        ],
    )?;
    after_stage(SqliteCommitStage::SourceCursorWritten)?;

    transaction.execute(
        "INSERT INTO basalt_log_cursors (log_id, tree_size, root_digest, checkpoint_digest)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(log_id) DO UPDATE SET
          tree_size=excluded.tree_size,
          root_digest=excluded.root_digest,
          checkpoint_digest=excluded.checkpoint_digest",
        params![
            checkpoint.log_id,
            checkpoint.tree_size.to_string(),
            checkpoint.root_digest,
            checkpoint.checkpoint_digest,
        ],
    )?;
    after_stage(SqliteCommitStage::LogCursorWritten)?;

    transaction.execute(
        "INSERT INTO basalt_accepted_receipts
         (receipt_digest, host_id, source_id, partition_id, generation,
          previous_seal_digest, seal_digest, observer_profile_digest,
          boot_incarnation_digest, log_id, tree_size, root_digest,
          checkpoint_digest, signer_policy_version, witness_policy_version,
          policy_bundle_digest)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
        params![
            receipt.receipt_digest,
            receipt.source.provisioned_host_id,
            receipt.source.source_id,
            receipt.source.partition_id,
            receipt.generation.to_string(),
            receipt.previous_seal_digest,
            receipt.seal_digest,
            receipt.observer_profile_digest,
            receipt.boot_incarnation_digest,
            receipt.checkpoint.log_id,
            receipt.checkpoint.tree_size.to_string(),
            receipt.checkpoint.root_digest,
            receipt.checkpoint.checkpoint_digest,
            receipt.policy_versions.signer.to_string(),
            receipt.policy_versions.witness.to_string(),
            receipt.policy_versions.bundle_digest,
        ],
    )?;
    after_stage(SqliteCommitStage::ReceiptWritten)?;

    let updated = transaction.execute(
        "UPDATE basalt_state_meta
         SET revision=?1, signer_policy_version=?2, witness_policy_version=?3,
             policy_bundle_digest=?4, state_digest=?5
         WHERE singleton=1",
        params![
            plan.next_state.revision.to_string(),
            policies.signer.to_string(),
            policies.witness.to_string(),
            policies.bundle_digest,
            trusted_state_digest(&plan.next_state),
        ],
    )?;
    if updated != 1 {
        anyhow::bail!("expected exactly one BASALT state-meta row to advance");
    }
    after_stage(SqliteCommitStage::MetaWritten)?;
    Ok(())
}

fn hash_state_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hash_state_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_be_bytes());
}

fn hash_state_optional_string(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1_u8]);
            hash_state_string(hasher, value);
        }
        None => hasher.update([0_u8]),
    }
}

/// Deterministic, domain-separated digest of the normalized local state.
/// It detects accidental inconsistency; it is not keyed and cannot prevent a
/// capable attacker from restoring/recomputing an older complete database.
pub fn trusted_state_digest(state: &TrustedState) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"basalt.trusted-state.snapshot.v1\0");
    hash_state_u64(&mut hasher, state.revision);

    match &state.policy_versions {
        Some(policy) => {
            hasher.update([1_u8]);
            hash_state_u64(&mut hasher, policy.signer);
            hash_state_u64(&mut hasher, policy.witness);
            hash_state_string(&mut hasher, &policy.bundle_digest);
        }
        None => hasher.update([0_u8]),
    }

    hash_state_u64(&mut hasher, state.log_cursors.len() as u64);
    for (log_id, checkpoint) in &state.log_cursors {
        hash_state_string(&mut hasher, log_id);
        hash_state_u64(&mut hasher, checkpoint.tree_size);
        hash_state_string(&mut hasher, &checkpoint.root_digest);
        hash_state_string(&mut hasher, &checkpoint.checkpoint_digest);
    }

    hash_state_u64(&mut hasher, state.source_cursors.len() as u64);
    for (key, cursor) in &state.source_cursors {
        hash_state_string(&mut hasher, &key.provisioned_host_id);
        hash_state_string(&mut hasher, &key.source_id);
        hash_state_string(&mut hasher, &key.partition_id);
        hash_state_u64(&mut hasher, cursor.generation);
        hash_state_string(&mut hasher, &cursor.seal_digest);
        hash_state_string(&mut hasher, &cursor.receipt_digest);
        hash_state_string(&mut hasher, &cursor.observer_profile_digest);
        hash_state_string(&mut hasher, &cursor.boot_incarnation_digest);
    }

    hash_state_u64(&mut hasher, state.accepted_receipts.len() as u64);
    for (digest, receipt) in &state.accepted_receipts {
        hash_state_string(&mut hasher, digest);
        hash_state_string(&mut hasher, &receipt.source.provisioned_host_id);
        hash_state_string(&mut hasher, &receipt.source.source_id);
        hash_state_string(&mut hasher, &receipt.source.partition_id);
        hash_state_u64(&mut hasher, receipt.generation);
        hash_state_optional_string(&mut hasher, receipt.previous_seal_digest.as_deref());
        hash_state_string(&mut hasher, &receipt.seal_digest);
        hash_state_string(&mut hasher, &receipt.receipt_digest);
        hash_state_string(&mut hasher, &receipt.observer_profile_digest);
        hash_state_string(&mut hasher, &receipt.boot_incarnation_digest);
        hash_state_string(&mut hasher, &receipt.checkpoint.log_id);
        hash_state_u64(&mut hasher, receipt.checkpoint.tree_size);
        hash_state_string(&mut hasher, &receipt.checkpoint.root_digest);
        hash_state_string(&mut hasher, &receipt.checkpoint.checkpoint_digest);
        hash_state_u64(&mut hasher, receipt.policy_versions.signer);
        hash_state_u64(&mut hasher, receipt.policy_versions.witness);
        hash_state_string(&mut hasher, &receipt.policy_versions.bundle_digest);
    }

    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("sha256:{hex}")
}

#[cfg(test)]
fn temporary_sqlite_path(label: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "basalt-{label}-{}-{nonce}-{}.sqlite",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

#[cfg(test)]
fn cleanup_sqlite_path(path: &Path) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(path.with_extension("sqlite-wal"));
    let _ = std::fs::remove_file(path.with_extension("sqlite-shm"));
}

