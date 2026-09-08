// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent content reconstruction for resource-campaign execution receipts.
//!
//! This crate is deliberately downstream of the structural receipt matcher. It
//! consumes captured bytes and structured observations, reconstructs the canonical
//! commitments named by an execution receipt, and compares those reconstructed
//! values with the already profile-matched claim.
//!
//! Its strongest theorem is intentionally narrow:
//!
//! `captured evidence reconstructs every committed receipt field`.
//!
//! It does not authenticate the capture source, query a live host, prove that a
//! benchmark actually ran, or establish scientific superiority.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use blake3::Hasher;
use symthaea_resource_campaign_execution_receipt::{
    ExecutionReceiptId, ProfileMatchedExecutionReceipt,
};
use thiserror::Error;

pub const RESOURCE_CAMPAIGN_EXECUTION_VERIFIER_V1: &str =
    "symthaea.resource-campaign-execution-verifier.v1";

const CLOSURE_OBJECT_CONTENT_V1: &[u8] = b"symthaea.execution.closure-object-content.v1";
const CLOSURE_MEMBERSHIP_V1: &[u8] = b"symthaea.execution.recursive-closure.v1";
const CLOSURE_CENSUS_V1: &[u8] = b"symthaea.execution.closure-content-census.v1";
const CLOSURE_REFERENCE_GRAPH_V1: &[u8] = b"symthaea.execution.closure-reference-graph.v1";
const EXECUTABLE_CONTENT_V1: &[u8] = b"symthaea.execution.executable-content.v1";
const VERSION_OUTPUT_V1: &[u8] = b"symthaea.execution.exact-version-output.v1";
const PLATFORM_IDENTITY_V1: &[u8] = b"symthaea.execution.platform-identity.v1";
const RUNNER_IDENTITY_V1: &[u8] = b"symthaea.execution.runner-identity.v1";
const PROCESS_ENVIRONMENT_V1: &[u8] = b"symthaea.execution.process-environment.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutionVerificationId([u8; 32]);

impl ExecutionVerificationId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptVerificationState {
    ReceiptCommitmentsVerifiedOnly,
}

/// One captured Nix closure object.
///
/// `nar_bytes` is the exact canonical object byte representation supplied to the
/// verifier. The verifier hashes these bytes itself; callers do not provide the
/// resulting object commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureObjectCapture {
    pub store_path: String,
    pub nar_bytes: Vec<u8>,
    pub references: Vec<String>,
}

/// Captured execution evidence used to reconstruct an execution receipt.
///
/// Identity maps are canonicalized by their `BTreeMap` ordering. The version
/// commitment includes the executable path, exact argv, and exact output bytes so
/// a version string cannot float away from the command that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionCapture {
    pub root_store_path: String,
    pub closure_objects: Vec<ClosureObjectCapture>,
    pub executable_path: String,
    pub executable_bytes: Vec<u8>,
    pub version_argv: Vec<String>,
    pub exact_version_output_bytes: Vec<u8>,
    pub target_triple: String,
    pub cpu_feature_profile: String,
    pub platform_identity: BTreeMap<String, String>,
    pub runner_identity: BTreeMap<String, String>,
    pub process_environment: BTreeMap<String, String>,
    pub nix_version: String,
    pub capture_evidence_ref: String,
}

/// Canonical commitments independently reconstructed from captured evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructedExecutionCommitments {
    root_store_path: String,
    executable_path: String,
    closure_object_digests: BTreeMap<String, [u8; 32]>,
    closure_reference_graph: BTreeMap<String, Vec<String>>,
    recursive_runtime_closure_commitment: [u8; 32],
    closure_content_census_commitment: [u8; 32],
    closure_reference_graph_commitment: [u8; 32],
    executable_content_commitment: [u8; 32],
    exact_version_output_commitment: [u8; 32],
    platform_identity_commitment: [u8; 32],
    runner_identity_commitment: [u8; 32],
    process_environment_commitment: [u8; 32],
    target_triple: String,
    cpu_feature_profile: String,
    nix_version: String,
}

impl ReconstructedExecutionCommitments {
    pub fn root_store_path(&self) -> &str {
        &self.root_store_path
    }

    pub fn executable_path(&self) -> &str {
        &self.executable_path
    }

    pub fn closure_object_digests(&self) -> &BTreeMap<String, [u8; 32]> {
        &self.closure_object_digests
    }

    pub fn closure_reference_graph(&self) -> &BTreeMap<String, Vec<String>> {
        &self.closure_reference_graph
    }

    pub fn recursive_runtime_closure_commitment(&self) -> &[u8; 32] {
        &self.recursive_runtime_closure_commitment
    }

    pub fn closure_content_census_commitment(&self) -> &[u8; 32] {
        &self.closure_content_census_commitment
    }

    pub fn closure_reference_graph_commitment(&self) -> &[u8; 32] {
        &self.closure_reference_graph_commitment
    }

    pub fn executable_content_commitment(&self) -> &[u8; 32] {
        &self.executable_content_commitment
    }

    pub fn exact_version_output_commitment(&self) -> &[u8; 32] {
        &self.exact_version_output_commitment
    }

    pub fn platform_identity_commitment(&self) -> &[u8; 32] {
        &self.platform_identity_commitment
    }

    pub fn runner_identity_commitment(&self) -> &[u8; 32] {
        &self.runner_identity_commitment
    }

    pub fn process_environment_commitment(&self) -> &[u8; 32] {
        &self.process_environment_commitment
    }

    pub fn target_triple(&self) -> &str {
        &self.target_triple
    }

    pub fn cpu_feature_profile(&self) -> &str {
        &self.cpu_feature_profile
    }

    pub fn nix_version(&self) -> &str {
        &self.nix_version
    }
}

/// Positive receipt-content verification witness.
///
/// Fields are private so downstream code cannot manufacture a verified receipt by
/// copying claim fields. This still does not prove benchmark execution; it proves
/// only that captured evidence reconstructs the receipt commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExecutionReceipt {
    matched_receipt: ProfileMatchedExecutionReceipt,
    reconstructed: ReconstructedExecutionCommitments,
    capture_evidence_ref: String,
    verification_id: ExecutionVerificationId,
}

impl VerifiedExecutionReceipt {
    pub fn matched_receipt(&self) -> &ProfileMatchedExecutionReceipt {
        &self.matched_receipt
    }

    pub fn reconstructed(&self) -> &ReconstructedExecutionCommitments {
        &self.reconstructed
    }

    pub fn capture_evidence_ref(&self) -> &str {
        &self.capture_evidence_ref
    }

    pub fn verification_id(&self) -> ExecutionVerificationId {
        self.verification_id
    }

    pub fn qualification_state(&self) -> ReceiptVerificationState {
        ReceiptVerificationState::ReceiptCommitmentsVerifiedOnly
    }
}

/// Reconstruct every receipt commitment from captured bytes/observations.
pub fn reconstruct_execution_commitments(
    capture: &ExecutionCapture,
) -> Result<ReconstructedExecutionCommitments, ExecutionVerificationError> {
    validate_nonblank("root_store_path", &capture.root_store_path)?;
    validate_store_path(&capture.root_store_path)?;
    validate_nonblank("executable_path", &capture.executable_path)?;
    validate_store_path(&capture.executable_path)?;
    validate_nonblank("target_triple", &capture.target_triple)?;
    validate_nonblank("cpu_feature_profile", &capture.cpu_feature_profile)?;
    validate_nonblank("nix_version", &capture.nix_version)?;
    validate_nonblank("capture_evidence_ref", &capture.capture_evidence_ref)?;

    if capture.closure_objects.is_empty() {
        return Err(ExecutionVerificationError::EmptyClosure);
    }
    if capture.executable_bytes.is_empty() {
        return Err(ExecutionVerificationError::EmptyExecutableBytes);
    }
    if capture.exact_version_output_bytes.is_empty() {
        return Err(ExecutionVerificationError::EmptyVersionOutput);
    }
    if capture.version_argv.is_empty() {
        return Err(ExecutionVerificationError::EmptyVersionArgv);
    }
    if capture.version_argv.iter().any(|arg| arg.trim().is_empty()) {
        return Err(ExecutionVerificationError::BlankVersionArgument);
    }
    if capture.version_argv.first().map(String::as_str) != Some(capture.executable_path.as_str()) {
        return Err(ExecutionVerificationError::VersionCommandExecutableMismatch);
    }

    let executable_prefix = format!("{}/", capture.root_store_path.trim_end_matches('/'));
    if capture.executable_path != capture.root_store_path
        && !capture.executable_path.starts_with(&executable_prefix)
    {
        return Err(ExecutionVerificationError::ExecutableOutsideRoot {
            root: capture.root_store_path.clone(),
            executable: capture.executable_path.clone(),
        });
    }

    validate_identity_map("platform_identity", &capture.platform_identity)?;
    validate_identity_map("runner_identity", &capture.runner_identity)?;
    validate_identity_map("process_environment", &capture.process_environment)?;

    let mut closure_object_digests = BTreeMap::new();
    let mut closure_reference_graph = BTreeMap::new();

    for object in &capture.closure_objects {
        validate_nonblank("closure_store_path", &object.store_path)?;
        validate_store_path(&object.store_path)?;
        if object.nar_bytes.is_empty() {
            return Err(ExecutionVerificationError::EmptyClosureObjectBytes(
                object.store_path.clone(),
            ));
        }
        if closure_object_digests.contains_key(&object.store_path) {
            return Err(ExecutionVerificationError::DuplicateClosureObject(
                object.store_path.clone(),
            ));
        }

        let mut references = BTreeSet::new();
        for reference in &object.references {
            validate_nonblank("closure_reference", reference)?;
            validate_store_path(reference)?;
            if !references.insert(reference.clone()) {
                return Err(ExecutionVerificationError::DuplicateReference {
                    store_path: object.store_path.clone(),
                    reference: reference.clone(),
                });
            }
        }

        closure_object_digests.insert(
            object.store_path.clone(),
            hash_bytes(CLOSURE_OBJECT_CONTENT_V1, &object.nar_bytes),
        );
        closure_reference_graph.insert(
            object.store_path.clone(),
            references.into_iter().collect(),
        );
    }

    if !closure_object_digests.contains_key(&capture.root_store_path) {
        return Err(ExecutionVerificationError::RootMissingFromClosure(
            capture.root_store_path.clone(),
        ));
    }

    for (store_path, references) in &closure_reference_graph {
        for reference in references {
            if !closure_object_digests.contains_key(reference) {
                return Err(ExecutionVerificationError::MissingReferencedObject {
                    store_path: store_path.clone(),
                    reference: reference.clone(),
                });
            }
        }
    }

    let recursive_runtime_closure_commitment = hash_closure_membership(
        &capture.root_store_path,
        closure_object_digests.keys(),
    );
    let closure_content_census_commitment =
        hash_closure_census(&closure_object_digests);
    let closure_reference_graph_commitment =
        hash_reference_graph(&closure_reference_graph);
    let executable_content_commitment = hash_executable(
        &capture.executable_path,
        &capture.executable_bytes,
    );
    let exact_version_output_commitment = hash_version_output(
        &capture.executable_path,
        &capture.version_argv,
        &capture.exact_version_output_bytes,
    );
    let platform_identity_commitment =
        hash_identity_map(PLATFORM_IDENTITY_V1, &capture.platform_identity);
    let runner_identity_commitment =
        hash_identity_map(RUNNER_IDENTITY_V1, &capture.runner_identity);
    let process_environment_commitment =
        hash_identity_map(PROCESS_ENVIRONMENT_V1, &capture.process_environment);

    Ok(ReconstructedExecutionCommitments {
        root_store_path: capture.root_store_path.clone(),
        executable_path: capture.executable_path.clone(),
        closure_object_digests,
        closure_reference_graph,
        recursive_runtime_closure_commitment,
        closure_content_census_commitment,
        closure_reference_graph_commitment,
        executable_content_commitment,
        exact_version_output_commitment,
        platform_identity_commitment,
        runner_identity_commitment,
        process_environment_commitment,
        target_triple: capture.target_triple.clone(),
        cpu_feature_profile: capture.cpu_feature_profile.clone(),
        nix_version: capture.nix_version.clone(),
    })
}

/// Verify one already profile-matched receipt against independently captured data.
pub fn verify_execution_receipt(
    matched_receipt: &ProfileMatchedExecutionReceipt,
    capture: &ExecutionCapture,
) -> Result<VerifiedExecutionReceipt, ExecutionVerificationError> {
    let reconstructed = reconstruct_execution_commitments(capture)?;
    let claim = matched_receipt.claim();

    if reconstructed.target_triple != claim.target_triple {
        return Err(ExecutionVerificationError::TargetTripleMismatch {
            expected: claim.target_triple.clone(),
            actual: reconstructed.target_triple.clone(),
        });
    }
    if reconstructed.cpu_feature_profile != claim.cpu_feature_profile {
        return Err(ExecutionVerificationError::CpuFeatureProfileMismatch {
            expected: claim.cpu_feature_profile.clone(),
            actual: reconstructed.cpu_feature_profile.clone(),
        });
    }
    if reconstructed.nix_version != claim.nix_version {
        return Err(ExecutionVerificationError::NixVersionMismatch {
            expected: claim.nix_version.clone(),
            actual: reconstructed.nix_version.clone(),
        });
    }

    compare_commitment(
        "recursive_runtime_closure",
        reconstructed.recursive_runtime_closure_commitment,
        claim.recursive_runtime_closure_commitment,
    )?;
    compare_commitment(
        "closure_content_census",
        reconstructed.closure_content_census_commitment,
        claim.closure_content_census_commitment,
    )?;
    compare_commitment(
        "closure_reference_graph",
        reconstructed.closure_reference_graph_commitment,
        claim.closure_reference_graph_commitment,
    )?;
    compare_commitment(
        "executable_content",
        reconstructed.executable_content_commitment,
        claim.executable_content_commitment,
    )?;
    compare_commitment(
        "exact_version_output",
        reconstructed.exact_version_output_commitment,
        claim.exact_version_output_commitment,
    )?;
    compare_commitment(
        "platform_identity",
        reconstructed.platform_identity_commitment,
        claim.platform_identity_commitment,
    )?;
    compare_commitment(
        "runner_identity",
        reconstructed.runner_identity_commitment,
        claim.runner_identity_commitment,
    )?;
    compare_commitment(
        "process_environment",
        reconstructed.process_environment_commitment,
        claim.process_environment_commitment,
    )?;

    if reconstructed.process_environment_commitment
        != *matched_receipt
            .selection_profile()
            .process_environment_commitment()
    {
        return Err(ExecutionVerificationError::SelectionEnvironmentMismatch);
    }

    let verification_id = ExecutionVerificationId(hash_verification(
        matched_receipt.receipt_id(),
        &reconstructed,
        &capture.capture_evidence_ref,
    ));

    Ok(VerifiedExecutionReceipt {
        matched_receipt: matched_receipt.clone(),
        reconstructed,
        capture_evidence_ref: capture.capture_evidence_ref.clone(),
        verification_id,
    })
}

fn validate_nonblank(
    field: &'static str,
    value: &str,
) -> Result<(), ExecutionVerificationError> {
    if value.trim().is_empty() {
        Err(ExecutionVerificationError::BlankField { field })
    } else {
        Ok(())
    }
}

fn validate_store_path(path: &str) -> Result<(), ExecutionVerificationError> {
    if !path.starts_with("/nix/store/") || path.len() <= "/nix/store/".len() {
        Err(ExecutionVerificationError::InvalidStorePath(path.to_owned()))
    } else {
        Ok(())
    }
}

fn validate_identity_map(
    field: &'static str,
    values: &BTreeMap<String, String>,
) -> Result<(), ExecutionVerificationError> {
    if values.is_empty() {
        return Err(ExecutionVerificationError::EmptyIdentityMap { field });
    }
    for key in values.keys() {
        if key.trim().is_empty() {
            return Err(ExecutionVerificationError::BlankIdentityKey { field });
        }
    }
    Ok(())
}

fn compare_commitment(
    field: &'static str,
    reconstructed: [u8; 32],
    claimed: [u8; 32],
) -> Result<(), ExecutionVerificationError> {
    if reconstructed == claimed {
        Ok(())
    } else {
        Err(ExecutionVerificationError::CommitmentMismatch { field })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExecutionVerificationError {
    #[error("execution capture field {field} must not be blank")]
    BlankField { field: &'static str },
    #[error("invalid Nix store path {0}")]
    InvalidStorePath(String),
    #[error("execution capture must contain at least one closure object")]
    EmptyClosure,
    #[error("captured executable bytes must not be empty")]
    EmptyExecutableBytes,
    #[error("captured exact version output must not be empty")]
    EmptyVersionOutput,
    #[error("version command argv must not be empty")]
    EmptyVersionArgv,
    #[error("version command argv contains a blank argument")]
    BlankVersionArgument,
    #[error("version command argv[0] must equal the captured executable path")]
    VersionCommandExecutableMismatch,
    #[error("captured executable {executable} is outside selected root {root}")]
    ExecutableOutsideRoot { root: String, executable: String },
    #[error("identity map {field} must not be empty")]
    EmptyIdentityMap { field: &'static str },
    #[error("identity map {field} contains a blank key")]
    BlankIdentityKey { field: &'static str },
    #[error("closure object {0} has empty canonical bytes")]
    EmptyClosureObjectBytes(String),
    #[error("duplicate closure object {0}")]
    DuplicateClosureObject(String),
    #[error("closure object {store_path} repeats reference {reference}")]
    DuplicateReference {
        store_path: String,
        reference: String,
    },
    #[error("selected root {0} is missing from the captured recursive closure")]
    RootMissingFromClosure(String),
    #[error("closure object {store_path} references missing object {reference}")]
    MissingReferencedObject {
        store_path: String,
        reference: String,
    },
    #[error("captured target triple {actual} does not match receipt {expected}")]
    TargetTripleMismatch { expected: String, actual: String },
    #[error("captured CPU feature profile {actual} does not match receipt {expected}")]
    CpuFeatureProfileMismatch { expected: String, actual: String },
    #[error("captured Nix version {actual} does not match receipt {expected}")]
    NixVersionMismatch { expected: String, actual: String },
    #[error("reconstructed commitment {field} does not match the receipt claim")]
    CommitmentMismatch { field: &'static str },
    #[error("reconstructed process environment does not match the selection profile")]
    SelectionEnvironmentMismatch,
}

fn hash_bytes(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, domain);
    frame(&mut hasher, bytes);
    *hasher.finalize().as_bytes()
}

fn hash_closure_membership<'a>(
    root: &str,
    paths: impl Iterator<Item = &'a String>,
) -> [u8; 32] {
    let paths: Vec<&String> = paths.collect();
    let mut hasher = Hasher::new();
    frame(&mut hasher, CLOSURE_MEMBERSHIP_V1);
    frame(&mut hasher, root.as_bytes());
    hasher.update(&(paths.len() as u64).to_le_bytes());
    for path in paths {
        frame(&mut hasher, path.as_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn hash_closure_census(digests: &BTreeMap<String, [u8; 32]>) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, CLOSURE_CENSUS_V1);
    hasher.update(&(digests.len() as u64).to_le_bytes());
    for (path, digest) in digests {
        frame(&mut hasher, path.as_bytes());
        hasher.update(digest);
    }
    *hasher.finalize().as_bytes()
}

fn hash_reference_graph(graph: &BTreeMap<String, Vec<String>>) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, CLOSURE_REFERENCE_GRAPH_V1);
    hasher.update(&(graph.len() as u64).to_le_bytes());
    for (path, references) in graph {
        frame(&mut hasher, path.as_bytes());
        hasher.update(&(references.len() as u64).to_le_bytes());
        for reference in references {
            frame(&mut hasher, reference.as_bytes());
        }
    }
    *hasher.finalize().as_bytes()
}

fn hash_executable(path: &str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, EXECUTABLE_CONTENT_V1);
    frame(&mut hasher, path.as_bytes());
    frame(&mut hasher, bytes);
    *hasher.finalize().as_bytes()
}

fn hash_version_output(path: &str, argv: &[String], output: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, VERSION_OUTPUT_V1);
    frame(&mut hasher, path.as_bytes());
    hasher.update(&(argv.len() as u64).to_le_bytes());
    for arg in argv {
        frame(&mut hasher, arg.as_bytes());
    }
    frame(&mut hasher, output);
    *hasher.finalize().as_bytes()
}

fn hash_identity_map(domain: &[u8], values: &BTreeMap<String, String>) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, domain);
    hasher.update(&(values.len() as u64).to_le_bytes());
    for (key, value) in values {
        frame(&mut hasher, key.as_bytes());
        frame(&mut hasher, value.as_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn hash_verification(
    receipt_id: ExecutionReceiptId,
    reconstructed: &ReconstructedExecutionCommitments,
    capture_evidence_ref: &str,
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(
        &mut hasher,
        RESOURCE_CAMPAIGN_EXECUTION_VERIFIER_V1.as_bytes(),
    );
    hasher.update(receipt_id.as_bytes());
    for commitment in [
        reconstructed.recursive_runtime_closure_commitment,
        reconstructed.closure_content_census_commitment,
        reconstructed.closure_reference_graph_commitment,
        reconstructed.executable_content_commitment,
        reconstructed.exact_version_output_commitment,
        reconstructed.platform_identity_commitment,
        reconstructed.runner_identity_commitment,
        reconstructed.process_environment_commitment,
    ] {
        hasher.update(&commitment);
    }
    frame(&mut hasher, reconstructed.target_triple.as_bytes());
    frame(&mut hasher, reconstructed.cpu_feature_profile.as_bytes());
    frame(&mut hasher, reconstructed.nix_version.as_bytes());
    frame(&mut hasher, capture_evidence_ref.as_bytes());
    *hasher.finalize().as_bytes()
}

fn frame(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_resource_campaign_execution_receipt::{
        match_execution_receipt, ExecutionReceiptClaim,
    };
    use symthaea_resource_campaign_execution_profile::ExecutionSelectionProfile;

    fn identity(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    }

    fn capture() -> ExecutionCapture {
        let root = "/nix/store/aaaaaaaa-root".to_owned();
        let dep_a = "/nix/store/bbbbbbbb-dep-a".to_owned();
        let dep_b = "/nix/store/cccccccc-dep-b".to_owned();
        let executable = format!("{root}/bin/resource-bench");
        ExecutionCapture {
            root_store_path: root.clone(),
            closure_objects: vec![
                ClosureObjectCapture {
                    store_path: dep_b.clone(),
                    nar_bytes: b"nar-dep-b".to_vec(),
                    references: vec![],
                },
                ClosureObjectCapture {
                    store_path: root,
                    nar_bytes: b"nar-root".to_vec(),
                    references: vec![dep_b, dep_a.clone()],
                },
                ClosureObjectCapture {
                    store_path: dep_a,
                    nar_bytes: b"nar-dep-a".to_vec(),
                    references: vec![],
                },
            ],
            executable_path: executable.clone(),
            executable_bytes: b"ELF-resource-bench".to_vec(),
            version_argv: vec![executable, "--version".into()],
            exact_version_output_bytes: b"resource-bench 0.1.0\n".to_vec(),
            target_triple: "x86_64-unknown-linux-gnu".into(),
            cpu_feature_profile: "x86_64-baseline-v1".into(),
            platform_identity: identity(&[
                ("kernel", "linux-6.x"),
                ("machine", "x86_64"),
            ]),
            runner_identity: identity(&[
                ("provider", "github-actions"),
                ("image", "ubuntu-24.04"),
            ]),
            process_environment: identity(&[
                ("LANG", "C"),
                ("LC_ALL", "C"),
                ("TZ", "UTC0"),
            ]),
            nix_version: "nix 2.x".into(),
            capture_evidence_ref: "evidence://independent-capture".into(),
        }
    }

    fn matched_receipt(capture: &ExecutionCapture) -> ProfileMatchedExecutionReceipt {
        let reconstructed = reconstruct_execution_commitments(capture).unwrap();
        let profile = ExecutionSelectionProfile::new(
            "git:deadbeef",
            [1; 32],
            [2; 32],
            [3; 32],
            [4; 32],
            capture.target_triple.clone(),
            capture.cpu_feature_profile.clone(),
            [5; 32],
            *reconstructed.process_environment_commitment(),
        )
        .unwrap();
        let claim = ExecutionReceiptClaim {
            selection_profile_id: profile.profile_id(),
            recursive_runtime_closure_commitment: *reconstructed
                .recursive_runtime_closure_commitment(),
            closure_content_census_commitment: *reconstructed
                .closure_content_census_commitment(),
            closure_reference_graph_commitment: *reconstructed
                .closure_reference_graph_commitment(),
            executable_content_commitment: *reconstructed.executable_content_commitment(),
            exact_version_output_commitment: *reconstructed.exact_version_output_commitment(),
            target_triple: capture.target_triple.clone(),
            cpu_feature_profile: capture.cpu_feature_profile.clone(),
            platform_identity_commitment: *reconstructed.platform_identity_commitment(),
            runner_identity_commitment: *reconstructed.runner_identity_commitment(),
            process_environment_commitment: *reconstructed.process_environment_commitment(),
            nix_version: capture.nix_version.clone(),
            evidence_ref: "evidence://untrusted-receipt-claim".into(),
        };
        match_execution_receipt(&profile, claim).unwrap()
    }

    #[test]
    fn exact_capture_reconstructs_a_verified_receipt_only() {
        let capture = capture();
        let matched = matched_receipt(&capture);
        let verified = verify_execution_receipt(&matched, &capture).unwrap();
        assert_eq!(
            verified.qualification_state(),
            ReceiptVerificationState::ReceiptCommitmentsVerifiedOnly
        );
        assert_eq!(
            verified.capture_evidence_ref(),
            "evidence://independent-capture"
        );
    }

    #[test]
    fn closure_input_order_does_not_change_commitments() {
        let a = capture();
        let mut b = a.clone();
        b.closure_objects.reverse();
        b.closure_objects
            .iter_mut()
            .find(|object| object.store_path == b.root_store_path)
            .unwrap()
            .references
            .reverse();
        assert_eq!(
            reconstruct_execution_commitments(&a).unwrap(),
            reconstruct_execution_commitments(&b).unwrap()
        );
    }

    #[test]
    fn missing_recursive_reference_fails_before_receipt_comparison() {
        let mut capture = capture();
        let root = capture.root_store_path.clone();
        capture
            .closure_objects
            .iter_mut()
            .find(|object| object.store_path == root)
            .unwrap()
            .references
            .push("/nix/store/dddddddd-missing".into());
        assert!(matches!(
            reconstruct_execution_commitments(&capture),
            Err(ExecutionVerificationError::MissingReferencedObject { .. })
        ));
    }

    #[test]
    fn executable_or_version_drift_breaks_the_claim() {
        let original = capture();
        let matched = matched_receipt(&original);

        let mut executable_drift = original.clone();
        executable_drift.executable_bytes.push(0xff);
        assert!(matches!(
            verify_execution_receipt(&matched, &executable_drift),
            Err(ExecutionVerificationError::CommitmentMismatch {
                field: "executable_content"
            })
        ));

        let mut version_drift = original;
        version_drift.exact_version_output_bytes = b"resource-bench 0.2.0\n".to_vec();
        assert!(matches!(
            verify_execution_receipt(&matched, &version_drift),
            Err(ExecutionVerificationError::CommitmentMismatch {
                field: "exact_version_output"
            })
        ));
    }

    #[test]
    fn reference_graph_drift_is_independently_detected() {
        let original = capture();
        let matched = matched_receipt(&original);
        let mut changed = original;
        let root = changed.root_store_path.clone();
        let dep_b = "/nix/store/cccccccc-dep-b";
        changed
            .closure_objects
            .iter_mut()
            .find(|object| object.store_path == root)
            .unwrap()
            .references
            .retain(|reference| reference != dep_b);
        assert!(matches!(
            verify_execution_receipt(&matched, &changed),
            Err(ExecutionVerificationError::CommitmentMismatch {
                field: "closure_reference_graph"
            })
        ));
    }

    #[test]
    fn runner_and_environment_drift_fail_closed() {
        let original = capture();
        let matched = matched_receipt(&original);

        let mut runner_drift = original.clone();
        runner_drift
            .runner_identity
            .insert("image".into(), "ubuntu-next".into());
        assert!(matches!(
            verify_execution_receipt(&matched, &runner_drift),
            Err(ExecutionVerificationError::CommitmentMismatch {
                field: "runner_identity"
            })
        ));

        let mut environment_drift = original;
        environment_drift
            .process_environment
            .insert("TZ".into(), "Africa/Johannesburg".into());
        assert!(matches!(
            verify_execution_receipt(&matched, &environment_drift),
            Err(ExecutionVerificationError::CommitmentMismatch {
                field: "process_environment"
            })
        ));
    }

    #[test]
    fn version_command_is_bound_to_the_captured_executable() {
        let mut capture = capture();
        capture.version_argv[0] = "/nix/store/eeeeeeee-other/bin/tool".into();
        assert!(matches!(
            reconstruct_execution_commitments(&capture),
            Err(ExecutionVerificationError::VersionCommandExecutableMismatch)
        ));
    }
}
