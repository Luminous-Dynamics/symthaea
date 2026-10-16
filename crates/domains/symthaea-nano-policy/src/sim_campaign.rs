//! Campaign-level reconciliation of per-run simulator-budget ledgers.
//!
//! This layer checks that paired policy arms contain the same frozen seed roster,
//! and that each ledger file is digest-bound, identity-matched, and passes the
//! single-run simulator-call auditor. It does not evaluate scientific performance.
use crate::sim_budget::{
    audit_json_ledger, is_hex_digest, sha256_hex, AuditDisposition, SimulatorBudgetLedger,
    MAX_SIMULATOR_CALLS, MAX_PROPOSAL_ATTEMPTS,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Campaign-manifest contract version.
pub const CAMPAIGN_MANIFEST_SCHEMA_VERSION: &str = "nano-simulator-campaign-manifest-v1";
/// Campaign-audit report contract version.
pub const CAMPAIGN_AUDIT_SCHEMA_VERSION: &str = "nano-simulator-campaign-audit-v1";
/// Maximum encoded campaign-manifest size.
pub const MAX_CAMPAIGN_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
/// Maximum combined raw run-ledger bytes audited in one campaign.
pub const MAX_CAMPAIGN_TOTAL_LEDGER_BYTES: usize = 512 * 1024 * 1024;
/// Maximum number of policy arms.
pub const MAX_CAMPAIGN_ARMS: usize = 32;
/// Maximum seed values in one declared roster.
pub const MAX_CAMPAIGN_SEEDS: usize = 1_000;
/// Maximum ledger references per policy arm.
pub const MAX_LEDGER_REFERENCES_PER_ARM: usize = 1_000;
/// Maximum bytes for identifiers and relative ledger paths.
pub const MAX_CAMPAIGN_TEXT_BYTES: usize = 512;

/// Immutable campaign manifest describing the expected seed roster and policy arms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CampaignManifest {
    /// Must equal CAMPAIGN_MANIFEST_SCHEMA_VERSION.
    pub schema_version: String,
    /// Shared benchmark/task family, identical across every run.
    pub benchmark_id: String,
    /// Unique campaign identifier.
    pub campaign_id: String,
    /// Required actual simulator calls per run.
    pub expected_simulator_calls: usize,
    /// Separate cap on generation/proposal attempts per run.
    pub proposal_attempt_cap: usize,
    /// Common unit for proposal and simulator costs.
    pub cost_unit: String,
    /// Frozen objective, units, constraints and eligibility contract.
    pub task_contract_sha256: String,
    /// Frozen target/system/split/seed-group manifest.
    pub evaluation_manifest_sha256: String,
    /// Frozen simulator recipe and settings.
    pub simulator_recipe_sha256: String,
    /// Exact seed roster that every arm must contain once.
    pub expected_seeds: Vec<u64>,
    /// All policy arms to reconcile.
    pub arms: Vec<CampaignArmManifest>,
}

/// One comparison arm and its per-seed run-ledger references.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CampaignArmManifest {
    /// Unique arm identifier, e.g. random_seeded.
    pub arm_id: String,
    /// Human-readable policy identifier.
    pub policy_id: String,
    /// Immutable lowercase 40-character policy/source revision.
    pub policy_revision: String,
    /// One entry per expected seed. Missing files are reported by the audit rather than
    /// silently removed; omitted seed references are reported as missing provenance.
    pub ledgers: Vec<CampaignLedgerReference>,
}

/// A run-ledger file pinned by relative path, expected seed and raw-byte SHA-256.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CampaignLedgerReference {
    /// Expected integer seed for this run.
    pub seed: u64,
    /// Safe path relative to the caller-supplied ledger root; no absolute, dot, or parent components.
    pub relative_path: String,
    /// SHA-256 of the exact raw run-ledger JSON bytes.
    pub sha256: String,
}

/// Status of an individual reference during campaign audit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CampaignRunStatus {
    /// File was present, digest and identity matched, and the run's call-budget audit passed.
    Verified,
    /// The manifest references a file that was not available to the auditor.
    MissingFile,
    /// File bytes were available but did not match the pinned SHA-256.
    DigestMismatch,
    /// File was not a valid simulator-budget ledger.
    InvalidLedger,
    /// The ledger parsed but failed the single-run budget audit.
    RunAuditFailed,
    /// Ledger did not match the declared campaign/arm/seed/task/configuration identity.
    IdentityMismatch,
}

/// Audit details for one per-seed run-ledger reference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CampaignRunAudit {
    /// Declared path relative to the external ledger root.
    pub relative_path: String,
    /// Seed pinned by the campaign manifest.
    pub expected_seed: u64,
    /// Seed found in the parsed run ledger, if it parsed.
    pub observed_seed: Option<u64>,
    /// SHA-256 pinned by the campaign manifest.
    pub expected_sha256: String,
    /// SHA-256 calculated from exact observed file bytes, if present.
    pub observed_sha256: Option<String>,
    /// Raw file bytes match the campaign manifest.
    pub digest_matches: bool,
    /// Run ID from the ledger, if parsed.
    pub run_id: Option<String>,
    /// Number of calls recomputed by the single-run auditor, if valid.
    pub observed_simulator_calls: Option<usize>,
    /// Disposition from the single-run auditor, if valid.
    pub run_disposition: Option<AuditDisposition>,
    /// Result for this reference after digest, identity and budget checks.
    pub status: CampaignRunStatus,
    /// Reasons this run reference did not pass.
    pub blockers: Vec<String>,
}

/// Seed roster reconciliation for one policy arm.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CampaignArmAudit {
    /// Arm identity.
    pub arm_id: String,
    /// Policy identity.
    pub policy_id: String,
    /// Policy revision.
    pub policy_revision: String,
    /// Size of the expected seed roster.
    pub expected_seed_count: usize,
    /// Number of ledger references in the campaign manifest.
    pub referenced_run_count: usize,
    /// Number of fully verified run ledgers.
    pub verified_run_count: usize,
    /// Expected seeds absent from the arm's ledger references entirely.
    pub missing_seed_references: Vec<u64>,
    /// Expected seeds lacking exactly one fully verified run.
    pub missing_verified_seeds: Vec<u64>,
    /// Repeated seed references in this arm.
    pub duplicate_seed_references: Vec<u64>,
    /// Referenced seeds not in the declared expected roster.
    pub unexpected_seed_references: Vec<u64>,
    /// All run-level audit outputs for the arm.
    pub runs: Vec<CampaignRunAudit>,
    /// True only when the expected seed roster is represented exactly once and all runs pass.
    pub audit_pass: bool,
}

/// Full campaign-level run/seed/call-budget report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CampaignAuditReport {
    /// Must equal CAMPAIGN_AUDIT_SCHEMA_VERSION.
    pub schema_version: String,
    /// SHA-256 of the exact raw campaign-manifest JSON bytes.
    pub manifest_sha256: String,
    /// Campaign and benchmark identity.
    pub campaign_id: String,
    pub benchmark_id: String,
    /// Expected shared budget and seed roster.
    pub expected_simulator_calls: usize,
    pub proposal_attempt_cap: usize,
    pub cost_unit: String,
    pub expected_seeds: Vec<u64>,
    pub expected_seed_count: usize,
    pub expected_run_count: usize,
    pub referenced_run_count: usize,
    pub verified_run_count: usize,
    /// Duplicate run IDs across arms/runs invalidate a paired campaign.
    pub duplicate_run_ids: Vec<String>,
    /// Arm-by-arm seed and ledger reconciliation.
    pub arms: Vec<CampaignArmAudit>,
    /// True only when every arm matches the seed roster and all ledgers pass.
    pub audit_pass: bool,
    /// Claim limits retained with the report.
    pub limitations: Vec<String>,
}

/// Parse and validate a campaign manifest. The CLI uses this to determine which files
/// must be loaded beneath a chosen artifact root.
pub fn parse_campaign_manifest(raw_manifest: &[u8]) -> Result<CampaignManifest, String> {
    if raw_manifest.len() > MAX_CAMPAIGN_MANIFEST_BYTES {
        return Err(format!(
            "campaign manifest exceeds the {MAX_CAMPAIGN_MANIFEST_BYTES}-byte limit"
        ));
    }
    let manifest: CampaignManifest = serde_json::from_slice(raw_manifest)
        .map_err(|error| format!("invalid campaign-manifest JSON: {error}"))?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

/// Reconcile the exact seed roster across arms and verify every referenced run-ledger byte
/// string against its SHA-256. The map keys are safe, manifest-relative paths. A missing
/// key is reported as a missing ledger reference, not silently dropped.
pub fn audit_campaign_json(
    raw_manifest: &[u8],
    ledger_bytes: &BTreeMap<String, Vec<u8>>,
) -> Result<CampaignAuditReport, String> {
    let manifest = parse_campaign_manifest(raw_manifest)?;
    let total_ledger_bytes = ledger_bytes.values().try_fold(0usize, |total, raw| {
        total
            .checked_add(raw.len())
            .ok_or_else(|| "combined ledger byte count overflow".to_string())
    })?;
    if total_ledger_bytes > MAX_CAMPAIGN_TOTAL_LEDGER_BYTES {
        return Err(format!(
            "combined run ledgers exceed the {MAX_CAMPAIGN_TOTAL_LEDGER_BYTES}-byte campaign limit"
        ));
    }
    let manifest_sha256 = sha256_hex(raw_manifest);
    let expected_seeds: BTreeSet<u64> = manifest.expected_seeds.iter().copied().collect();
    let mut global_run_ids = HashSet::new();
    let mut duplicate_run_ids = BTreeSet::new();
    let mut arm_reports = Vec::with_capacity(manifest.arms.len());
    let mut referenced_run_count = 0usize;
    let mut verified_run_count = 0usize;

    for arm in &manifest.arms {
        let mut seed_reference_counts: HashMap<u64, usize> = HashMap::new();
        let mut verified_seed_counts: HashMap<u64, usize> = HashMap::new();
        let mut arm_run_reports = Vec::with_capacity(arm.ledgers.len());

        for reference in &arm.ledgers {
            referenced_run_count += 1;
            *seed_reference_counts.entry(reference.seed).or_default() += 1;
            let mut report = CampaignRunAudit {
                relative_path: reference.relative_path.clone(),
                expected_seed: reference.seed,
                observed_seed: None,
                expected_sha256: reference.sha256.clone(),
                observed_sha256: None,
                digest_matches: false,
                run_id: None,
                observed_simulator_calls: None,
                run_disposition: None,
                status: CampaignRunStatus::MissingFile,
                blockers: Vec::new(),
            };

            let Some(raw_ledger) = ledger_bytes.get(&reference.relative_path) else {
                report.blockers.push("referenced ledger file is missing".to_string());
                arm_run_reports.push(report);
                continue;
            };
            let observed_sha256 = sha256_hex(raw_ledger);
            report.digest_matches = observed_sha256 == reference.sha256;
            report.observed_sha256 = Some(observed_sha256);
            if !report.digest_matches {
                report.status = CampaignRunStatus::DigestMismatch;
                report
                    .blockers
                    .push("raw ledger SHA-256 does not match campaign manifest".to_string());
            }

            let ledger: SimulatorBudgetLedger = match serde_json::from_slice(raw_ledger) {
                Ok(ledger) => ledger,
                Err(error) => {
                    report.status = CampaignRunStatus::InvalidLedger;
                    report
                        .blockers
                        .push(format!("invalid run-ledger JSON: {error}"));
                    arm_run_reports.push(report);
                    continue;
                }
            };
            report.observed_seed = Some(ledger.seed);
            report.run_id = Some(ledger.run_id.clone());
            let duplicate_run_id = !global_run_ids.insert(ledger.run_id.clone());
            if duplicate_run_id {
                duplicate_run_ids.insert(ledger.run_id.clone());
                report.blockers.push("run_id is duplicated elsewhere in the campaign".to_string());
            }

            let run_audit = match audit_json_ledger(raw_ledger) {
                Ok(audit) => audit,
                Err(error) => {
                    report.status = CampaignRunStatus::InvalidLedger;
                    report
                        .blockers
                        .push(format!("single-run budget auditor rejected ledger: {error}"));
                    arm_run_reports.push(report);
                    continue;
                }
            };
            report.observed_simulator_calls = Some(run_audit.observed_simulator_calls);
            report.run_disposition = Some(run_audit.disposition);

            let mut identity_blockers = Vec::new();
            if ledger.benchmark_id != manifest.benchmark_id {
                identity_blockers.push("benchmark_id differs from campaign".to_string());
            }
            if ledger.seed != reference.seed {
                identity_blockers.push(
                    "ledger seed differs from the seed pinned by its reference".to_string(),
                );
            }
            if ledger.arm_id != arm.arm_id {
                identity_blockers.push("arm_id differs from campaign arm".to_string());
            }
            if ledger.policy_id != arm.policy_id {
                identity_blockers.push("policy_id differs from campaign arm".to_string());
            }
            if ledger.policy_revision != arm.policy_revision {
                identity_blockers.push("policy_revision differs from campaign arm".to_string());
            }
            if ledger.expected_simulator_calls != manifest.expected_simulator_calls {
                identity_blockers.push(
                    "expected simulator-call budget differs from campaign".to_string(),
                );
            }
            if ledger.proposal_attempt_cap != manifest.proposal_attempt_cap {
                identity_blockers.push("proposal-attempt cap differs from campaign".to_string());
            }
            if ledger.cost_unit != manifest.cost_unit {
                identity_blockers.push("cost unit differs from campaign".to_string());
            }
            if ledger.task_contract_sha256 != manifest.task_contract_sha256 {
                identity_blockers.push("task-contract digest differs from campaign".to_string());
            }
            if ledger.evaluation_manifest_sha256 != manifest.evaluation_manifest_sha256 {
                identity_blockers.push(
                    "evaluation-manifest digest differs from campaign".to_string(),
                );
            }
            if ledger.simulator_recipe_sha256 != manifest.simulator_recipe_sha256 {
                identity_blockers.push("simulator-recipe digest differs from campaign".to_string());
            }
            if !report.digest_matches {
                report.status = CampaignRunStatus::DigestMismatch;
                report.blockers.extend(identity_blockers);
            } else if duplicate_run_id || !identity_blockers.is_empty() {
                report.status = CampaignRunStatus::IdentityMismatch;
                report.blockers.extend(identity_blockers);
            } else if !run_audit.audit_pass {
                report.status = CampaignRunStatus::RunAuditFailed;
                report
                    .blockers
                    .push("single-run simulator-call budget audit did not pass".to_string());
            } else {
                report.status = CampaignRunStatus::Verified;
                *verified_seed_counts.entry(ledger.seed).or_default() += 1;
                verified_run_count += 1;
            }
            arm_run_reports.push(report);
        }

        let missing_seed_references: Vec<u64> = manifest
            .expected_seeds
            .iter()
            .copied()
            .filter(|seed| !seed_reference_counts.contains_key(seed))
            .collect();
        let missing_verified_seeds: Vec<u64> = manifest
            .expected_seeds
            .iter()
            .copied()
            .filter(|seed| verified_seed_counts.get(seed).copied().unwrap_or(0) != 1)
            .collect();
        let duplicate_seed_references: Vec<u64> = seed_reference_counts
            .iter()
            .filter_map(|(seed, count)| (*count > 1).then_some(*seed))
            .collect();
        let unexpected_seed_references: Vec<u64> = seed_reference_counts
            .keys()
            .filter(|seed| !expected_seeds.contains(seed))
            .copied()
            .collect();

        let all_runs_pass = arm_run_reports
            .iter()
            .all(|run| run.status == CampaignRunStatus::Verified);
        let audit_pass = missing_seed_references.is_empty()
            && missing_verified_seeds.is_empty()
            && duplicate_seed_references.is_empty()
            && unexpected_seed_references.is_empty()
            && all_runs_pass;
        arm_reports.push(CampaignArmAudit {
            arm_id: arm.arm_id.clone(),
            policy_id: arm.policy_id.clone(),
            policy_revision: arm.policy_revision.clone(),
            expected_seed_count: manifest.expected_seeds.len(),
            referenced_run_count: arm.ledgers.len(),
            verified_run_count: arm_run_reports
                .iter()
                .filter(|run| run.status == CampaignRunStatus::Verified)
                .count(),
            missing_seed_references,
            missing_verified_seeds,
            duplicate_seed_references,
            unexpected_seed_references,
            runs: arm_run_reports,
            audit_pass,
        });
    }

    let audit_pass = duplicate_run_ids.is_empty() && arm_reports.iter().all(|arm| arm.audit_pass);
    Ok(CampaignAuditReport {
        schema_version: CAMPAIGN_AUDIT_SCHEMA_VERSION.to_string(),
        manifest_sha256,
        campaign_id: manifest.campaign_id,
        benchmark_id: manifest.benchmark_id,
        expected_simulator_calls: manifest.expected_simulator_calls,
        proposal_attempt_cap: manifest.proposal_attempt_cap,
        cost_unit: manifest.cost_unit,
        expected_seeds: manifest.expected_seeds.clone(),
        expected_seed_count: manifest.expected_seeds.len(),
        expected_run_count: manifest.expected_seeds.len() * manifest.arms.len(),
        referenced_run_count,
        verified_run_count,
        duplicate_run_ids: duplicate_run_ids.into_iter().collect(),
        arms: arm_reports,
        audit_pass,
        limitations: vec![
            "A complete campaign seed roster is a provenance/counting condition, not proof of adequate statistical power or scientific superiority.".to_string(),
            "Each run ledger's receipt digest strings are syntax-checked, but the auditor does not dereference those simulator start/terminal/result artifacts.".to_string(),
            "The campaign manifest's source/model/license decisions must be reviewed separately; this auditor does not certify rights.".to_string(),
            "Declared group/seed independence is not proven by identifiers alone; use a preregistered scientific grouping and power rationale.".to_string(),
            "A successful budget/seed audit does not imply equal wall time, equal total cost, valid physics, or physical-device performance.".to_string(),
        ],
    })
}

fn validate_manifest(manifest: &CampaignManifest) -> Result<(), String> {
    if manifest.schema_version != CAMPAIGN_MANIFEST_SCHEMA_VERSION {
        return Err(format!(
            "schema_version must be {CAMPAIGN_MANIFEST_SCHEMA_VERSION}"
        ));
    }
    validate_text(&manifest.benchmark_id, "benchmark_id")?;
    validate_text(&manifest.campaign_id, "campaign_id")?;
    validate_text(&manifest.cost_unit, "cost_unit")?;
    if !is_hex_digest(&manifest.task_contract_sha256, 64)
        || !is_hex_digest(&manifest.evaluation_manifest_sha256, 64)
        || !is_hex_digest(&manifest.simulator_recipe_sha256, 64)
    {
        return Err("task/evaluation/recipe manifest digests must be 64 lowercase hexadecimal characters".to_string());
    }
    if manifest.expected_simulator_calls == 0
        || manifest.expected_simulator_calls > MAX_SIMULATOR_CALLS
    {
        return Err(format!(
            "expected_simulator_calls must be in 1..={MAX_SIMULATOR_CALLS}"
        ));
    }
    if manifest.proposal_attempt_cap < manifest.expected_simulator_calls
        || manifest.proposal_attempt_cap > MAX_PROPOSAL_ATTEMPTS
    {
        return Err(format!(
            "proposal_attempt_cap must be >= expected_simulator_calls and <= {MAX_PROPOSAL_ATTEMPTS}"
        ));
    }
    if manifest.expected_seeds.is_empty() || manifest.expected_seeds.len() > MAX_CAMPAIGN_SEEDS {
        return Err(format!("expected_seeds must contain 1..={MAX_CAMPAIGN_SEEDS} values"));
    }
    let expected: BTreeSet<u64> = manifest.expected_seeds.iter().copied().collect();
    if expected.len() != manifest.expected_seeds.len() {
        return Err("expected_seeds contains duplicate values".to_string());
    }
    if manifest.arms.len() < 2 || manifest.arms.len() > MAX_CAMPAIGN_ARMS {
        return Err(format!("arms must contain 2..={MAX_CAMPAIGN_ARMS} policies"));
    }
    let mut arm_ids = HashSet::with_capacity(manifest.arms.len());
    let mut all_paths = HashSet::new();
    for arm in &manifest.arms {
        validate_text(&arm.arm_id, "arm_id")?;
        validate_text(&arm.policy_id, "policy_id")?;
        if !is_hex_digest(&arm.policy_revision, 40) {
            return Err(format!(
                "policy_revision for arm {:?} must be 40 lowercase hexadecimal characters",
                arm.arm_id
            ));
        }
        if !arm_ids.insert(arm.arm_id.as_str()) {
            return Err(format!("duplicate arm_id {:?}", arm.arm_id));
        }
        if arm.ledgers.len() > MAX_LEDGER_REFERENCES_PER_ARM {
            return Err(format!(
                "arm {:?} exceeds {MAX_LEDGER_REFERENCES_PER_ARM} ledger references",
                arm.arm_id
            ));
        }
        for reference in &arm.ledgers {
            validate_relative_path(&reference.relative_path)?;
            if !is_hex_digest(&reference.sha256, 64) {
                return Err(format!(
                    "ledger reference {:?} must have a 64-character lowercase SHA-256",
                    reference.relative_path
                ));
            }
            if !all_paths.insert(reference.relative_path.as_str()) {
                return Err(format!(
                    "ledger path {:?} is referenced more than once in the campaign",
                    reference.relative_path
                ));
            }
        }
    }
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<(), String> {
    validate_text(path, "relative_path")?;
    if path.len() > MAX_CAMPAIGN_TEXT_BYTES
        || path.starts_with('/')
        || path.contains('\\')
        || path.split('/').any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(format!(
            "relative_path {path:?} must be a safe forward-slash relative path with no empty, dot, or parent components"
        ));
    }
    Ok(())
}

fn validate_text(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty()
        || value.len() > MAX_CAMPAIGN_TEXT_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(format!(
            "{field} must be non-empty, at most {MAX_CAMPAIGN_TEXT_BYTES} bytes, and contain no control characters"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim_budget::{
        PreflightStatus, ProposalAttempt, ProposalStatus, RunStatus,
        SimulatorBudgetLedger, SimulatorCall, SimulatorTerminalStatus,
    };

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn test_call(
        id: &str,
        status: SimulatorTerminalStatus,
        result_digest: Option<String>,
        cost: f64,
    ) -> SimulatorCall {
        SimulatorCall {
            invocation_id: id.to_string(),
            started_receipt_sha256: digest('d'),
            terminal_receipt_sha256: digest('e'),
            terminal_status: status,
            result_artifact_sha256: result_digest,
            elapsed_ms: 100,
            cost,
        }
    }

    fn test_attempt(
        sequence: usize,
        id: &str,
        candidate_id: &str,
        candidate_digest: char,
        call: SimulatorCall,
    ) -> ProposalAttempt {
        ProposalAttempt {
            sequence,
            attempt_id: id.to_string(),
            candidate_id: Some(candidate_id.to_string()),
            candidate_sha256: Some(digest(candidate_digest)),
            proposal_status: ProposalStatus::Generated,
            preflight_status: PreflightStatus::Passed,
            attempt_reason: None,
            not_called_reason: None,
            proposal_cost: 0.1,
            proposal_elapsed_ms: 5,
            simulator_call: Some(call),
        }
    }

    fn test_ledger(
        seed: u64,
        run_id: &str,
        arm_id: &str,
        policy_id: &str,
        policy_revision: String,
    ) -> Vec<u8> {
        let ledger = SimulatorBudgetLedger {
            schema_version: crate::sim_budget::SIMULATOR_LEDGER_SCHEMA_VERSION.to_string(),
            benchmark_id: "campaign-synthetic".to_string(),
            run_id: run_id.to_string(),
            seed,
            arm_id: arm_id.to_string(),
            policy_id: policy_id.to_string(),
            policy_revision,
            run_manifest_sha256: digest('1'),
            task_contract_sha256: digest('2'),
            evaluation_manifest_sha256: digest('3'),
            source_manifest_sha256: digest('4'),
            simulator_recipe_sha256: digest('5'),
            expected_simulator_calls: 2,
            reported_simulator_calls: 2,
            proposal_attempt_cap: 10,
            cost_unit: "cost_units".to_string(),
            run_status: RunStatus::Completed,
            attempts: vec![
                test_attempt(
                    1,
                    &format!("{run_id}-attempt-1"),
                    &format!("{run_id}-candidate-1"),
                    'a',
                    test_call(
                        &format!("{run_id}-call-1"),
                        SimulatorTerminalStatus::ReturnedFiniteResult,
                        Some(digest('f')),
                        1.0,
                    ),
                ),
                test_attempt(
                    2,
                    &format!("{run_id}-attempt-2"),
                    &format!("{run_id}-candidate-2"),
                    'b',
                    test_call(
                        &format!("{run_id}-call-2"),
                        SimulatorTerminalStatus::Failed,
                        None,
                        1.0,
                    ),
                ),
            ],
        };
        serde_json::to_vec_pretty(&ledger).expect("test ledger serializes")
    }

    fn add_run(
        manifest: &mut CampaignManifest,
        bytes: &mut BTreeMap<String, Vec<u8>>,
        arm_index: usize,
        seed: u64,
        path: &str,
    ) {
        let arm = &manifest.arms[arm_index];
        let run_id = format!("run-{}-{seed}", arm.arm_id);
        let raw = test_ledger(
            seed,
            &run_id,
            &arm.arm_id,
            &arm.policy_id,
            arm.policy_revision.clone(),
        );
        manifest.arms[arm_index].ledgers.push(CampaignLedgerReference {
            seed,
            relative_path: path.to_string(),
            sha256: sha256_hex(&raw),
        });
        bytes.insert(path.to_string(), raw);
    }

    fn sample_campaign() -> (CampaignManifest, BTreeMap<String, Vec<u8>>) {
        let mut manifest = CampaignManifest {
            schema_version: CAMPAIGN_MANIFEST_SCHEMA_VERSION.to_string(),
            benchmark_id: "campaign-synthetic".to_string(),
            campaign_id: "campaign-fixture-001".to_string(),
            expected_simulator_calls: 2,
            proposal_attempt_cap: 10,
            cost_unit: "cost_units".to_string(),
            task_contract_sha256: digest('2'),
            evaluation_manifest_sha256: digest('3'),
            simulator_recipe_sha256: digest('5'),
            expected_seeds: vec![42, 43],
            arms: vec![
                CampaignArmManifest {
                    arm_id: "random".to_string(),
                    policy_id: "random_seeded".to_string(),
                    policy_revision: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
                    ledgers: Vec::new(),
                },
                CampaignArmManifest {
                    arm_id: "maxmin".to_string(),
                    policy_id: "farthest_first".to_string(),
                    policy_revision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
                    ledgers: Vec::new(),
                },
            ],
        };
        let mut bytes = BTreeMap::new();
        add_run(
            &mut manifest,
            &mut bytes,
            0,
            42,
            "runs/random/seed-42.json",
        );
        add_run(
            &mut manifest,
            &mut bytes,
            0,
            43,
            "runs/random/seed-43.json",
        );
        add_run(
            &mut manifest,
            &mut bytes,
            1,
            42,
            "runs/maxmin/seed-42.json",
        );
        add_run(
            &mut manifest,
            &mut bytes,
            1,
            43,
            "runs/maxmin/seed-43.json",
        );
        (manifest, bytes)
    }

    fn raw_manifest(manifest: &CampaignManifest) -> Vec<u8> {
        serde_json::to_vec_pretty(manifest).expect("test manifest serializes")
    }

    #[test]
    fn matching_seed_rosters_and_ledgers_pass_campaign_audit() {
        let (manifest, bytes) = sample_campaign();
        let report = audit_campaign_json(&raw_manifest(&manifest), &bytes)
            .expect("valid campaign manifest audits");
        assert!(report.audit_pass);
        assert_eq!(report.expected_run_count, 4);
        assert_eq!(report.referenced_run_count, 4);
        assert_eq!(report.verified_run_count, 4);
        assert!(report.arms.iter().all(|arm| arm.audit_pass));
        assert_eq!(report.arms[0].missing_verified_seeds, Vec::<u64>::new());
        assert_eq!(report.arms[1].missing_verified_seeds, Vec::<u64>::new());
    }

    #[test]
    fn missing_seed_reference_is_reported_instead_of_shrinking_roster() {
        let (mut manifest, bytes) = sample_campaign();
        manifest.arms[0].ledgers.retain(|reference| reference.seed != 43);
        let report = audit_campaign_json(&raw_manifest(&manifest), &bytes)
            .expect("valid but incomplete roster audits");
        assert!(!report.audit_pass);
        assert_eq!(report.arms[0].missing_seed_references, vec![43]);
        assert_eq!(report.arms[0].missing_verified_seeds, vec![43]);
        assert!(report.arms[1].audit_pass);
    }

    #[test]
    fn referenced_but_missing_file_is_not_counted_as_verified_seed() {
        let (manifest, mut bytes) = sample_campaign();
        bytes.remove("runs/maxmin/seed-43.json");
        let report = audit_campaign_json(&raw_manifest(&manifest), &bytes)
            .expect("missing file is reported as audit failure");
        assert!(!report.audit_pass);
        assert_eq!(report.arms[1].missing_verified_seeds, vec![43]);
        let missing = report.arms[1]
            .runs
            .iter()
            .find(|run| run.expected_seed == 43)
            .expect("seed-43 run report");
        assert_eq!(missing.status, CampaignRunStatus::MissingFile);
    }

    #[test]
    fn raw_ledger_hash_mismatch_blocks_campaign_pass() {
        let (mut manifest, bytes) = sample_campaign();
        manifest.arms[0].ledgers[0].sha256 = digest('9');
        let report = audit_campaign_json(&raw_manifest(&manifest), &bytes)
            .expect("digest mismatch is a reportable campaign audit failure");
        assert!(!report.audit_pass);
        let run = report.arms[0]
            .runs
            .iter()
            .find(|run| run.expected_seed == 42)
            .expect("seed-42 report");
        assert_eq!(run.status, CampaignRunStatus::DigestMismatch);
        assert!(!run.digest_matches);
    }

    #[test]
    fn declared_seed_must_match_seed_inside_run_ledger() {
        let (mut manifest, mut bytes) = sample_campaign();
        let path = "runs/random/seed-42.json";
        let raw = test_ledger(
            99,
            "run-random-99",
            "random",
            "random_seeded",
            manifest.arms[0].policy_revision.clone(),
        );
        bytes.insert(path.to_string(), raw.clone());
        manifest.arms[0].ledgers[0].sha256 = sha256_hex(&raw);
        let report = audit_campaign_json(&raw_manifest(&manifest), &bytes)
            .expect("seed identity mismatch is reportable");
        assert!(!report.audit_pass);
        let run = report.arms[0]
            .runs
            .iter()
            .find(|run| run.expected_seed == 42)
            .expect("seed-42 report");
        assert_eq!(run.observed_seed, Some(99));
        assert_eq!(run.status, CampaignRunStatus::IdentityMismatch);
        assert!(run.blockers.iter().any(|message| message.contains("seed")));
    }

    #[test]
    fn repeated_seed_reference_and_duplicate_run_id_fail_campaign() {
        let (mut manifest, mut bytes) = sample_campaign();
        let duplicate = manifest.arms[0].ledgers[0].clone();
        let new_path = "runs/random/seed-42-duplicate.json";
        bytes.insert(
            new_path.to_string(),
            bytes.get(&duplicate.relative_path).unwrap().clone(),
        );
        manifest.arms[0].ledgers.push(CampaignLedgerReference {
            seed: duplicate.seed,
            relative_path: new_path.to_string(),
            sha256: duplicate.sha256,
        });
        let report = audit_campaign_json(&raw_manifest(&manifest), &bytes)
            .expect("duplicate seed is reportable");
        assert!(!report.audit_pass);
        assert_eq!(report.arms[0].duplicate_seed_references, vec![42]);
        assert_eq!(report.duplicate_run_ids, vec!["run-random-42".to_string()]);
    }

    #[test]
    fn parent_directory_paths_are_rejected() {
        let (mut manifest, bytes) = sample_campaign();
        manifest.arms[0].ledgers[0].relative_path = "../outside.json".to_string();
        let error = audit_campaign_json(&raw_manifest(&manifest), &bytes).unwrap_err();
        assert!(error.contains("safe forward-slash relative path"));
    }

    #[test]
    fn unreferenced_extra_ledger_bytes_do_not_enter_campaign_denominator() {
        let (manifest, mut bytes) = sample_campaign();
        bytes.insert("extra.json".to_string(), b"not part of the manifest".to_vec());
        let report = audit_campaign_json(&raw_manifest(&manifest), &bytes)
            .expect("extra external files do not expand campaign");
        assert!(report.audit_pass);
        assert_eq!(report.referenced_run_count, 4);
    }
}
