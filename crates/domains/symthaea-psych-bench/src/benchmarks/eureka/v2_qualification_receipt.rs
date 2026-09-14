// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict parser/self-consistency verifier for EUREKA-002 V2 backend
//! qualification receipts.
//!
//! This module deliberately does not contain a trusted qualifier workflow hash
//! and cannot mint execution authority. It proves only that raw receipt bytes
//! satisfy the frozen v2 grammar and are internally self-consistent.

#![allow(dead_code)]

use std::collections::BTreeMap;

pub(super) const V2_QUALIFICATION_RECEIPT_SCHEMA: &str =
    "EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v2";
pub(super) const V2_QUALIFICATION_REVISION: &str =
    "EUREKA.002.V2.BACKEND_QUALIFICATION.v2";
pub(super) const V2_QUALIFICATION_COMMAND_CONTRACT_REVISION: &str =
    "EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v1";
pub(super) const V2_QUALIFICATION_CLAIM_SCOPE: &str =
    "backend-build-test-lint-only";
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

        let github_run_id = required(&fields, "github_run_id")?
            .parse::<u64>()
            .map_err(|_| V2QualificationReceiptError::InvalidRunId)?;
        if github_run_id == 0 {
            return Err(V2QualificationReceiptError::InvalidRunId);
        }
        let github_run_attempt = required(&fields, "github_run_attempt")?
            .parse::<u32>()
            .map_err(|_| V2QualificationReceiptError::InvalidRunAttempt)?;
        if github_run_attempt == 0 {
            return Err(V2QualificationReceiptError::InvalidRunAttempt);
        }

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

        let expected_subject_head = required(&fields, "expected_subject_head")?;
        let subject_head = required(&fields, "subject_head")?;
        let postflight_head = required(&fields, "postflight_head")?;
        validate_lower_hex(expected_subject_head, 40)?;
        validate_lower_hex(subject_head, 40)?;
        validate_lower_hex(postflight_head, 40)?;
        if expected_subject_head != subject_head || subject_head != postflight_head {
            return Err(V2QualificationReceiptError::SubjectHeadMismatch);
        }

        let subject_tree = required(&fields, "subject_tree")?;
        let postflight_tree = required(&fields, "postflight_tree")?;
        validate_lower_hex(subject_tree, 40)?;
        validate_lower_hex(postflight_tree, 40)?;
        if subject_tree != postflight_tree {
            return Err(V2QualificationReceiptError::SubjectTreeMismatch);
        }

        let cargo_lock_sha256 = required(&fields, "cargo_lock_sha256")?;
        let postflight_cargo_lock_sha256 = required(&fields, "postflight_cargo_lock_sha256")?;
        validate_lower_hex(cargo_lock_sha256, 64)?;
        validate_lower_hex(postflight_cargo_lock_sha256, 64)?;
        if cargo_lock_sha256 != postflight_cargo_lock_sha256 {
            return Err(V2QualificationReceiptError::CargoLockMismatch);
        }

        let workflow_sha256 = required(&fields, "workflow_sha256")?;
        let postflight_workflow_sha256 = required(&fields, "postflight_workflow_sha256")?;
        validate_lower_hex(workflow_sha256, 64)?;
        validate_lower_hex(postflight_workflow_sha256, 64)?;
        if workflow_sha256 != postflight_workflow_sha256 {
            return Err(V2QualificationReceiptError::WorkflowMismatch);
        }

        let command_contract_sha256 = required(&fields, "command_contract_sha256")?;
        let postflight_command_contract_sha256 =
            required(&fields, "postflight_command_contract_sha256")?;
        validate_lower_hex(command_contract_sha256, 64)?;
        validate_lower_hex(postflight_command_contract_sha256, 64)?;
        if command_contract_sha256 != postflight_command_contract_sha256 {
            return Err(V2QualificationReceiptError::CommandContractMismatch);
        }

        let rustc_version = required(&fields, "rustc_version")?;
        let cargo_version = required(&fields, "cargo_version")?;
        if rustc_version.is_empty() || cargo_version.is_empty() {
            return Err(V2QualificationReceiptError::EmptyValue);
        }

        Ok(Self {
            repository: EXPECTED_REPOSITORY.to_string(),
            event: event.to_string(),
            github_run_id,
            github_run_attempt,
            github_workflow_ref: workflow_ref.to_string(),
            subject_head: subject_head.to_string(),
            subject_tree: subject_tree.to_string(),
            cargo_lock_sha256: cargo_lock_sha256.to_string(),
            workflow_sha256: workflow_sha256.to_string(),
            command_contract_sha256: command_contract_sha256.to_string(),
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

fn parse_fields<'a>(
    input: &'a str,
) -> Result<BTreeMap<&'a str, &'a str>, V2QualificationReceiptError> {
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

fn parse_canonical_bool(value: &str) -> Result<bool, V2QualificationReceiptError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(V2QualificationReceiptError::InvalidBoolean),
    }
}

fn validate_lower_hex(
    value: &str,
    expected_digits: usize,
) -> Result<(), V2QualificationReceiptError> {
    if value.len() != expected_digits
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2QualificationReceiptError::InvalidHex);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "0123456789abcdef0123456789abcdef01234567";
    const TREE: &str = "89abcdef0123456789abcdef0123456789abcdef";
    const LOCK: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const WORKFLOW: &str = "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0";
    const CONTRACT: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn valid_receipt() -> String {
        format!(
            "receipt_schema_revision={V2_QUALIFICATION_RECEIPT_SCHEMA}\n\
qualification_revision={V2_QUALIFICATION_REVISION}\n\
command_contract_revision={V2_QUALIFICATION_COMMAND_CONTRACT_REVISION}\n\
repository={EXPECTED_REPOSITORY}\n\
event=pull_request\n\
github_run_id=34828668196\n\
github_run_attempt=1\n\
github_workflow_ref={WORKFLOW_REF_PREFIX}refs/pull/2889/merge\n\
expected_subject_head={HEAD}\n\
subject_head={HEAD}\n\
subject_tree={TREE}\n\
cargo_lock_sha256={LOCK}\n\
workflow_sha256={WORKFLOW}\n\
command_contract_sha256={CONTRACT}\n\
rustc_version=rustc 1.96.0 (fixture)\n\
cargo_version=cargo 1.96.0 (fixture)\n\
checkout_clean_before=true\n\
claim_scope={V2_QUALIFICATION_CLAIM_SCOPE}\n\
execution_authority_granted=false\n\
real_canary_executed=false\n\
heldout_executed=false\n\
confirmatory_evidence_minted=false\n\
postflight_head={HEAD}\n\
postflight_tree={TREE}\n\
postflight_cargo_lock_sha256={LOCK}\n\
postflight_workflow_sha256={WORKFLOW}\n\
postflight_command_contract_sha256={CONTRACT}\n\
checkout_clean_after=true\n\
qualification_result=PASS\n"
        )
    }

    fn replace_value(receipt: &str, key: &str, value: &str) -> String {
        let prefix = format!("{key}=");
        let mut lines = Vec::new();
        for line in receipt.lines() {
            if line.starts_with(&prefix) {
                lines.push(format!("{key}={value}"));
            } else {
                lines.push(line.to_string());
            }
        }
        format!("{}\n", lines.join("\n"))
    }

    #[test]
    fn canonical_v2_receipt_parses_and_self_verifies() {
        let parsed = V2QualificationReceipt::parse_and_verify(&valid_receipt()).unwrap();
        assert_eq!(parsed.github_run_id(), 34_828_668_196);
        assert_eq!(parsed.github_run_attempt(), 1);
        assert_eq!(parsed.subject_head(), HEAD);
        assert_eq!(parsed.workflow_sha256(), WORKFLOW);
        assert_eq!(parsed.command_contract_sha256(), CONTRACT);
    }

    #[test]
    fn duplicate_and_unknown_keys_fail_closed() {
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
    }

    #[test]
    fn missing_or_malformed_run_identity_fails_closed() {
        let missing = valid_receipt()
            .lines()
            .filter(|line| !line.starts_with("github_run_attempt="))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&missing),
            Err(V2QualificationReceiptError::MissingKey)
        );

        let malformed = replace_value(&valid_receipt(), "github_run_id", "not-a-number");
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&malformed),
            Err(V2QualificationReceiptError::InvalidRunId)
        );
    }

    #[test]
    fn authority_or_execution_flags_can_never_be_true() {
        for key in [
            "execution_authority_granted",
            "real_canary_executed",
            "heldout_executed",
            "confirmatory_evidence_minted",
        ] {
            let changed = replace_value(&valid_receipt(), key, "true");
            assert_eq!(
                V2QualificationReceipt::parse_and_verify(&changed),
                Err(V2QualificationReceiptError::ForbiddenAuthorityOrExecution)
            );
        }
    }

    #[test]
    fn booleans_are_canonical_lowercase_only() {
        let changed = replace_value(&valid_receipt(), "checkout_clean_after", "TRUE");
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&changed),
            Err(V2QualificationReceiptError::InvalidBoolean)
        );
    }

    #[test]
    fn subject_and_postflight_identity_mismatches_fail_closed() {
        let changed_head = replace_value(
            &valid_receipt(),
            "postflight_head",
            "1123456789abcdef0123456789abcdef01234567",
        );
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&changed_head),
            Err(V2QualificationReceiptError::SubjectHeadMismatch)
        );

        let changed_workflow = replace_value(
            &valid_receipt(),
            "postflight_workflow_sha256",
            "023456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0",
        );
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&changed_workflow),
            Err(V2QualificationReceiptError::WorkflowMismatch)
        );
    }

    #[test]
    fn malformed_or_noncanonical_hex_fails_closed() {
        let uppercase = replace_value(
            &valid_receipt(),
            "workflow_sha256",
            "A23456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0",
        );
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&uppercase),
            Err(V2QualificationReceiptError::InvalidHex)
        );

        let short = replace_value(&valid_receipt(), "subject_tree", "abcd");
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&short),
            Err(V2QualificationReceiptError::InvalidHex)
        );
    }

    #[test]
    fn pass_scope_and_schema_are_exact() {
        let fail = replace_value(&valid_receipt(), "qualification_result", "FAIL");
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&fail),
            Err(V2QualificationReceiptError::NotPass)
        );

        let scope = replace_value(&valid_receipt(), "claim_scope", "execution-authorized");
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&scope),
            Err(V2QualificationReceiptError::WrongClaimScope)
        );

        let schema = replace_value(&valid_receipt(), "receipt_schema_revision", "v3");
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&schema),
            Err(V2QualificationReceiptError::WrongReceiptSchema)
        );
    }

    #[test]
    fn malformed_lines_and_empty_values_fail_closed() {
        let malformed = valid_receipt().replace(
            "repository=Luminous-Dynamics/symthaea",
            "repository",
        );
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&malformed),
            Err(V2QualificationReceiptError::MalformedLine)
        );

        let empty = replace_value(&valid_receipt(), "cargo_version", "");
        assert_eq!(
            V2QualificationReceipt::parse_and_verify(&empty),
            Err(V2QualificationReceiptError::EmptyValue)
        );
    }
}
