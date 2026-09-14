// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict parser/self-consistency verifier for EUREKA-002 V2 backend
//! qualification receipts.
//!
//! This module deliberately contains no trusted qualifier hash and cannot mint
//! execution authority. It establishes only receipt-v2 grammar and internal
//! consistency for the checked-in qualification-contract revision.

#![allow(dead_code)]

use std::collections::BTreeMap;

pub(super) const V2_QUALIFICATION_RECEIPT_SCHEMA: &str =
    "EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v2";
pub(super) const V2_QUALIFICATION_REVISION: &str =
    "EUREKA.002.V2.BACKEND_QUALIFICATION.v2";
pub(super) const V2_QUALIFICATION_COMMAND_CONTRACT_REVISION: &str =
    "EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v2";
pub(super) const V2_QUALIFICATION_CLAIM_SCOPE: &str = "backend-build-test-lint-only";
const EXPECTED_REPOSITORY: &str = "Luminous-Dynamics/symthaea";
const WORKFLOW_REF_PREFIX: &str =
    "Luminous-Dynamics/symthaea/.github/workflows/eureka-v2-backend-qualification.yml@";

const REQUIRED_KEYS: [&str; 29] = [
    "receipt_schema_revision",
    "qualification_revision",
    "command_contract_revision",
    "repository",
    "event",
    "github_run_id",
    "github_run_attempt",
    "github_workflow_ref",
    "expected_subject_head",
    "subject_head",
    "subject_tree",
    "cargo_lock_sha256",
    "workflow_sha256",
    "command_contract_sha256",
    "rustc_version",
    "cargo_version",
    "checkout_clean_before",
    "claim_scope",
    "execution_authority_granted",
    "real_canary_executed",
    "heldout_executed",
    "confirmatory_evidence_minted",
    "postflight_head",
    "postflight_tree",
    "postflight_cargo_lock_sha256",
    "postflight_workflow_sha256",
    "postflight_command_contract_sha256",
    "checkout_clean_after",
    "qualification_result",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2QualificationReceipt {
    repository: String,
    event: String,
    github_run_id: u64,
    github_run_attempt: u32,
    github_workflow_ref: String,
    subject_head: String,
    subject_tree: String,
    cargo_lock_sha256: String,
    workflow_sha256: String,
    command_contract_sha256: String,
    rustc_version: String,
    cargo_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2QualificationReceiptError {
    MalformedLine,
    EmptyKey,
    EmptyValue,
    UnknownKey,
    DuplicateKey,
    MissingKey,
    WrongReceiptSchema,
    WrongQualificationRevision,
    WrongCommandContractRevision,
    WrongRepository,
    WrongEvent,
    WrongWorkflowRef,
    InvalidRunId,
    InvalidRunAttempt,
    InvalidBoolean,
    InvalidHex,
    WrongClaimScope,
    NotPass,
    CheckoutNotClean,
    ForbiddenAuthorityOrExecution,
    SubjectHeadMismatch,
    SubjectTreeMismatch,
    CargoLockMismatch,
    WorkflowMismatch,
    CommandContractMismatch,
}

impl V2QualificationReceipt {
    pub(super) fn parse_and_verify(input: &str) -> Result<Self, V2QualificationReceiptError> {
        let fields = parse_fields(input)?;
        ensure_required_keys(&fields)?;

        require_exact(
            &fields,
            "receipt_schema_revision",
            V2_QUALIFICATION_RECEIPT_SCHEMA,
            V2QualificationReceiptError::WrongReceiptSchema,
        )?;
        require_exact(
            &fields,
            "qualification_revision",
            V2_QUALIFICATION_REVISION,
            V2QualificationReceiptError::WrongQualificationRevision,
        )?;
        require_exact(
            &fields,
            "command_contract_revision",
            V2_QUALIFICATION_COMMAND_CONTRACT_REVISION,
            V2QualificationReceiptError::WrongCommandContractRevision,
        )?;
        require_exact(
            &fields,
            "repository",
            EXPECTED_REPOSITORY,
            V2QualificationReceiptError::WrongRepository,
        )?;
        require_exact(
            &fields,
            "claim_scope",
            V2_QUALIFICATION_CLAIM_SCOPE,
            V2QualificationReceiptError::WrongClaimScope,
        )?;
        require_exact(
            &fields,
            "qualification_result",
            "PASS",
            V2QualificationReceiptError::NotPass,
        )?;

        let event = required(&fields, "event")?;
        if !matches!(event, "pull_request" | "workflow_dispatch") {
            return Err(V2QualificationReceiptError::WrongEvent);
        }

        let workflow_ref = required(&fields, "github_workflow_ref")?;
        if !workflow_ref.starts_with(WORKFLOW_REF_PREFIX) {
            return Err(V2QualificationReceiptError::WrongWorkflowRef);
        }

        let github_run_id = parse_positive_u64(required(&fields, "github_run_id")?)?;
        let github_run_attempt = parse_positive_u32(required(&fields, "github_run_attempt")?)?;

        for key in ["checkout_clean_before", "checkout_clean_after"] {
            if !parse_canonical_bool(required(&fields, key)?)? {
                return Err(V2QualificationReceiptError::CheckoutNotClean);
            }
        }
        for key in [
            "execution_authority_granted",
            "real_canary_executed",
            "heldout_executed",
            "confirmatory_evidence_minted",
        ] {
            if parse_canonical_bool(required(&fields, key)?)? {
                return Err(V2QualificationReceiptError::ForbiddenAuthorityOrExecution);
            }
        }

        let expected_subject_head = checked_hex(required(&fields, "expected_subject_head")?, 40)?;
        let subject_head = checked_hex(required(&fields, "subject_head")?, 40)?;
        let postflight_head = checked_hex(required(&fields, "postflight_head")?, 40)?;
        if expected_subject_head != subject_head || subject_head != postflight_head {
            return Err(V2QualificationReceiptError::SubjectHeadMismatch);
        }

        let subject_tree = checked_hex(required(&fields, "subject_tree")?, 40)?;
        let postflight_tree = checked_hex(required(&fields, "postflight_tree")?, 40)?;
        if subject_tree != postflight_tree {
            return Err(V2QualificationReceiptError::SubjectTreeMismatch);
        }

        let cargo_lock = checked_hex(required(&fields, "cargo_lock_sha256")?, 64)?;
        let postflight_lock = checked_hex(required(&fields, "postflight_cargo_lock_sha256")?, 64)?;
        if cargo_lock != postflight_lock {
            return Err(V2QualificationReceiptError::CargoLockMismatch);
        }

        let workflow = checked_hex(required(&fields, "workflow_sha256")?, 64)?;
        let postflight_workflow =
            checked_hex(required(&fields, "postflight_workflow_sha256")?, 64)?;
        if workflow != postflight_workflow {
            return Err(V2QualificationReceiptError::WorkflowMismatch);
        }

        let contract = checked_hex(required(&fields, "command_contract_sha256")?, 64)?;
        let postflight_contract = checked_hex(
            required(&fields, "postflight_command_contract_sha256")?,
            64,
        )?;
        if contract != postflight_contract {
            return Err(V2QualificationReceiptError::CommandContractMismatch);
        }

        let rustc_version = required(&fields, "rustc_version")?;
        let cargo_version = required(&fields, "cargo_version")?;

        Ok(Self {
            repository: EXPECTED_REPOSITORY.to_string(),
            event: event.to_string(),
            github_run_id,
            github_run_attempt,
            github_workflow_ref: workflow_ref.to_string(),
            subject_head: subject_head.to_string(),
            subject_tree: subject_tree.to_string(),
            cargo_lock_sha256: cargo_lock.to_string(),
            workflow_sha256: workflow.to_string(),
            command_contract_sha256: contract.to_string(),
            rustc_version: rustc_version.to_string(),
            cargo_version: cargo_version.to_string(),
        })
    }

    pub(super) const fn github_run_id(&self) -> u64 {
        self.github_run_id
    }

    pub(super) const fn github_run_attempt(&self) -> u32 {
        self.github_run_attempt
    }

    pub(super) fn subject_head(&self) -> &str {
        &self.subject_head
    }

    pub(super) fn workflow_sha256(&self) -> &str {
        &self.workflow_sha256
    }

    pub(super) fn command_contract_sha256(&self) -> &str {
        &self.command_contract_sha256
    }
}

fn parse_fields(input: &str) -> Result<BTreeMap<&str, &str>, V2QualificationReceiptError> {
    let mut fields = BTreeMap::new();
    for line in input.lines() {
        if line.is_empty() {
            return Err(V2QualificationReceiptError::MalformedLine);
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(V2QualificationReceiptError::MalformedLine);
        };
        if key.is_empty() {
            return Err(V2QualificationReceiptError::EmptyKey);
        }
        if value.is_empty() {
            return Err(V2QualificationReceiptError::EmptyValue);
        }
        if !REQUIRED_KEYS.contains(&key) {
            return Err(V2QualificationReceiptError::UnknownKey);
        }
        if fields.insert(key, value).is_some() {
            return Err(V2QualificationReceiptError::DuplicateKey);
        }
    }
    Ok(fields)
}

fn ensure_required_keys(
    fields: &BTreeMap<&str, &str>,
) -> Result<(), V2QualificationReceiptError> {
    if REQUIRED_KEYS.iter().any(|key| !fields.contains_key(key)) {
        return Err(V2QualificationReceiptError::MissingKey);
    }
    Ok(())
}

fn required<'a>(
    fields: &BTreeMap<&'a str, &'a str>,
    key: &str,
) -> Result<&'a str, V2QualificationReceiptError> {
    fields
        .get(key)
        .copied()
        .ok_or(V2QualificationReceiptError::MissingKey)
}

fn require_exact(
    fields: &BTreeMap<&str, &str>,
    key: &str,
    expected: &str,
    error: V2QualificationReceiptError,
) -> Result<(), V2QualificationReceiptError> {
    if required(fields, key)? != expected {
        return Err(error);
    }
    Ok(())
}

fn parse_positive_u64(value: &str) -> Result<u64, V2QualificationReceiptError> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| V2QualificationReceiptError::InvalidRunId)?;
    if parsed == 0 {
        return Err(V2QualificationReceiptError::InvalidRunId);
    }
    Ok(parsed)
}

fn parse_positive_u32(value: &str) -> Result<u32, V2QualificationReceiptError> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| V2QualificationReceiptError::InvalidRunAttempt)?;
    if parsed == 0 {
        return Err(V2QualificationReceiptError::InvalidRunAttempt);
    }
    Ok(parsed)
}

fn parse_canonical_bool(value: &str) -> Result<bool, V2QualificationReceiptError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(V2QualificationReceiptError::InvalidBoolean),
    }
}

fn checked_hex(value: &str, digits: usize) -> Result<&str, V2QualificationReceiptError> {
    if value.len() != digits
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2QualificationReceiptError::InvalidHex);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_receipt() -> String {
        let head = "1".repeat(40);
        let tree = "2".repeat(40);
        let lock = "a".repeat(64);
        let workflow = "b".repeat(64);
        let contract = "c".repeat(64);
        format!(
            "receipt_schema_revision={V2_QUALIFICATION_RECEIPT_SCHEMA}\n\
qualification_revision={V2_QUALIFICATION_REVISION}\n\
command_contract_revision={V2_QUALIFICATION_COMMAND_CONTRACT_REVISION}\n\
repository={EXPECTED_REPOSITORY}\n\
event=pull_request\n\
github_run_id=34829175771\n\
github_run_attempt=1\n\
github_workflow_ref={WORKFLOW_REF_PREFIX}refs/pull/2894/merge\n\
expected_subject_head={head}\n\
subject_head={head}\n\
subject_tree={tree}\n\
cargo_lock_sha256={lock}\n\
workflow_sha256={workflow}\n\
command_contract_sha256={contract}\n\
rustc_version=rustc 1.96.0 (fixture)\n\
cargo_version=cargo 1.96.0 (fixture)\n\
checkout_clean_before=true\n\
claim_scope={V2_QUALIFICATION_CLAIM_SCOPE}\n\
execution_authority_granted=false\n\
real_canary_executed=false\n\
heldout_executed=false\n\
confirmatory_evidence_minted=false\n\
postflight_head={head}\n\
postflight_tree={tree}\n\
postflight_cargo_lock_sha256={lock}\n\
postflight_workflow_sha256={workflow}\n\
postflight_command_contract_sha256={contract}\n\
checkout_clean_after=true\n\
qualification_result=PASS\n"
        )
    }

    fn replace_value(receipt: &str, key: &str, value: &str) -> String {
        let prefix = format!("{key}=");
        let lines = receipt.lines().map(|line| {
            if line.starts_with(&prefix) {
                format!("{key}={value}")
            } else {
                line.to_string()
            }
        });
        format!("{}\n", lines.collect::<Vec<_>>().join("\n"))
    }

    #[test]
    fn canonical_receipt_parses_and_self_verifies() {
        let parsed = V2QualificationReceipt::parse_and_verify(&valid_receipt()).unwrap();
        assert_eq!(parsed.github_run_id(), 34_829_175_771);
        assert_eq!(parsed.github_run_attempt(), 1);
        assert_eq!(parsed.subject_head(), "1".repeat(40));
        assert_eq!(parsed.workflow_sha256(), "b".repeat(64));
        assert_eq!(parsed.command_contract_sha256(), "c".repeat(64));
    }

    #[test]
    fn predecessor_command_contract_revision_is_rejected() {
        let mutated = replace_value(
            &valid_receipt(),
            "command_contract_revision",
            "EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v1",
        );
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&mutated),
            Err(V2QualificationReceiptError::WrongCommandContractRevision)
        );
    }

    #[test]
    fn duplicate_unknown_and_missing_keys_fail_closed() {
        let mut duplicate = valid_receipt();
        duplicate.push_str("qualification_result=PASS\n");
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&duplicate),
            Err(V2QualificationReceiptError::DuplicateKey)
        );

        let mut unknown = valid_receipt();
        unknown.push_str("future_optional_field=ignored\n");
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&unknown),
            Err(V2QualificationReceiptError::UnknownKey)
        );

        let missing = valid_receipt()
            .lines()
            .filter(|line| !line.starts_with("github_run_attempt="))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&missing),
            Err(V2QualificationReceiptError::MissingKey)
        );
    }

    #[test]
    fn run_identity_and_boolean_grammar_fail_closed() {
        for (key, value, expected) in [
            ("github_run_id", "0", V2QualificationReceiptError::InvalidRunId),
            (
                "github_run_attempt",
                "not-a-number",
                V2QualificationReceiptError::InvalidRunAttempt,
            ),
            (
                "checkout_clean_before",
                "TRUE",
                V2QualificationReceiptError::InvalidBoolean,
            ),
        ] {
            let mutated = replace_value(&valid_receipt(), key, value);
            assert_eq!(V2QualificationReceipt::parse_and_verify(&mutated), Err(expected));
        }
    }

    #[test]
    fn authority_or_execution_flags_can_never_be_true() {
        for key in [
            "execution_authority_granted",
            "real_canary_executed",
            "heldout_executed",
            "confirmatory_evidence_minted",
        ] {
            let mutated = replace_value(&valid_receipt(), key, "true");
            assert_eq!(
                V2QualificationReceipt::parse_and_verify(&mutated),
                Err(V2QualificationReceiptError::ForbiddenAuthorityOrExecution)
            );
        }
    }

    #[test]
    fn identity_mutations_fail_closed() {
        let uppercase = replace_value(&valid_receipt(), "workflow_sha256", &"B".repeat(64));
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&uppercase),
            Err(V2QualificationReceiptError::InvalidHex)
        );

        let workflow_mismatch = replace_value(
            &valid_receipt(),
            "postflight_workflow_sha256",
            &"d".repeat(64),
        );
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&workflow_mismatch),
            Err(V2QualificationReceiptError::WorkflowMismatch)
        );

        let head_mismatch = replace_value(&valid_receipt(), "postflight_head", &"3".repeat(40));
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&head_mismatch),
            Err(V2QualificationReceiptError::SubjectHeadMismatch)
        );
    }

    #[test]
    fn scope_schema_pass_and_event_are_exact() {
        for (key, value, expected) in [
            ("qualification_result", "FAIL", V2QualificationReceiptError::NotPass),
            ("claim_scope", "all", V2QualificationReceiptError::WrongClaimScope),
            (
                "receipt_schema_revision",
                "EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v3",
                V2QualificationReceiptError::WrongReceiptSchema,
            ),
            ("event", "push", V2QualificationReceiptError::WrongEvent),
        ] {
            let mutated = replace_value(&valid_receipt(), key, value);
            assert_eq!(V2QualificationReceipt::parse_and_verify(&mutated), Err(expected));
        }
    }
}
