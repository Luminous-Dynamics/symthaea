//! Typed composition of operational-risk and statistical-confidence budgets.
//!
//! This crate verifies contract compatibility, complete family membership,
//! append-only evidence receipt continuity, and exact union-bound arithmetic.
//! It does NOT verify component theorems, calibration assumptions, signatures,
//! or real-world truth. A successful report is bookkeeping evidence only.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PPB: u64 = 1_000_000_000;
const PLAN_DOMAIN: &[u8] = b"SYMTHAEA-RISK-BUDGET-PLAN-V0.1";
const RECEIPT_DOMAIN: &[u8] = b"SYMTHAEA-RISK-BUDGET-EVIDENCE-RECEIPT-V0.1";
const LEDGER_DOMAIN: &[u8] = b"SYMTHAEA-RISK-BUDGET-LEDGER-V0.1";
const REPORT_DOMAIN: &[u8] = b"SYMTHAEA-RISK-BUDGET-REPORT-V0.1";
const MAX_CONTRACTS: usize = 100_000;
const MAX_ACTIONS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiskLedgerError {
    pub field: String,
    pub message: String,
}
impl RiskLedgerError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}
impl std::fmt::Display for RiskLedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl std::error::Error for RiskLedgerError {}
fn fail(field: impl Into<String>, message: impl Into<String>) -> RiskLedgerError {
    RiskLedgerError::new(field, message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskGuaranteeKind {
    MarginalFiniteSample,
    ActionConditional,
    SequentialAnytime,
    ComputeStopping,
    EmpiricalOnly,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskCompositionMode {
    /// Each contract bounds an unconditional event probability on the same
    /// declared population and execution scope. The group uses a union bound.
    MarginalUnionBound,
    /// Each event family is bounded conditional on every action in one
    /// complete, frozen action partition. The group bound is max_a sum_j alpha(j,a).
    CompleteActionPartition,
    /// Contracts are recorded for audit but receive no formal risk/confidence credit.
    NoFormalCredit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskCompositionGroup {
    pub group_id: String,
    pub mode: RiskCompositionMode,
    /// Required only for CompleteActionPartition.
    pub action_support_sha256: Option<String>,
    /// Complete, strictly increasing action support. Required only for
    /// CompleteActionPartition.
    pub action_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskContract {
    pub claim_id: String,
    /// Identifies the adverse event across conditions. For an action-partition
    /// group, each action must declare the same set of event_family_id values.
    pub event_family_id: String,
    pub group_id: String,
    pub guarantee_kind: RiskGuaranteeKind,
    /// Upper bound for the operational adverse-event probability, in PPB.
    pub operational_alpha_ppb: u64,
    /// Failure probability of the statistical certification statement, in PPB.
    /// This is a separate budget from operational alpha.
    pub statistical_delta_ppb: u64,
    pub target_population_sha256: String,
    /// Digest of the complete execution episode / horizon to which the claim applies.
    /// All contracts in a plan must use the same execution scope.
    pub execution_scope_sha256: String,
    /// Digest of the agent, controller, or policy whose behavior is bounded.
    pub subject_sha256: String,
    /// Digest identifying the joint certification experiment or sampling protocol.
    pub certification_experiment_sha256: String,
    pub calibration_data_sha256: String,
    pub assumptions_sha256: String,
    pub method_id: String,
    pub method_version: String,
    pub paper_or_spec_url: String,
    pub source_repository_url: String,
    /// Pinned source revision (40-character lowercase Git commit SHA).
    pub source_revision: String,
    /// Set only for sequential-anytime claims. It binds the filtration identity.
    pub filtration_sha256: Option<String>,
    /// Set only for action-conditional claims.
    pub action_id: Option<u32>,
    pub action_support_sha256: Option<String>,
    /// Optional digest from an upstream canonical evidence ledger (e.g. #9465).
    /// This is an integrity reference, not verification of that upstream ledger.
    pub source_receipt_sha256: Option<String>,
}

impl RiskContract {
    pub fn validate(&self) -> Result<(), RiskLedgerError> {
        for (field, value) in [
            ("claim_id", self.claim_id.as_str()),
            ("event_family_id", self.event_family_id.as_str()),
            ("group_id", self.group_id.as_str()),
            ("method_id", self.method_id.as_str()),
            ("method_version", self.method_version.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(fail(field, "must be non-empty"));
            }
        }
        for (field, digest) in [
            ("target_population_sha256", &self.target_population_sha256),
            ("execution_scope_sha256", &self.execution_scope_sha256),
            ("subject_sha256", &self.subject_sha256),
            ("certification_experiment_sha256", &self.certification_experiment_sha256),
            ("calibration_data_sha256", &self.calibration_data_sha256),
            ("assumptions_sha256", &self.assumptions_sha256),
        ] {
            if !is_sha256(digest) {
                return Err(fail(field, "must be lowercase hexadecimal SHA-256"));
            }
        }
        if self.operational_alpha_ppb > PPB || self.statistical_delta_ppb > PPB {
            return Err(fail("risk_allocation", "alpha and delta must each be in [0, 1] PPB"));
        }
        if !self.paper_or_spec_url.starts_with("https://") {
            return Err(fail("paper_or_spec_url", "must use HTTPS"));
        }
        if !self.source_repository_url.starts_with("https://") {
            return Err(fail("source_repository_url", "must use HTTPS"));
        }
        if !is_git_sha(&self.source_revision) {
            return Err(fail("source_revision", "must be a 40-character lowercase Git SHA"));
        }
        if let Some(digest) = &self.source_receipt_sha256 {
            if !is_sha256(digest) {
                return Err(fail("source_receipt_sha256", "must be lowercase hexadecimal SHA-256"));
            }
        }

        match self.guarantee_kind {
            RiskGuaranteeKind::MarginalFiniteSample | RiskGuaranteeKind::ComputeStopping => {
                if self.action_id.is_some() || self.action_support_sha256.is_some() {
                    return Err(fail("conditioning", "marginal contracts cannot carry action conditioning"));
                }
                if self.filtration_sha256.is_some() {
                    return Err(fail("filtration_sha256", "fixed-horizon marginal contract cannot declare a sequential filtration"));
                }
                if self.operational_alpha_ppb == 0 {
                    return Err(fail("operational_alpha_ppb", "qualified operational claim must declare a positive alpha bound"));
                }
            }
            RiskGuaranteeKind::SequentialAnytime => {
                if self.action_id.is_some() || self.action_support_sha256.is_some() {
                    return Err(fail("conditioning", "sequential marginal contract cannot carry action conditioning"));
                }
                let filtration = self.filtration_sha256.as_ref()
                    .ok_or_else(|| fail("filtration_sha256", "sequential-anytime claim requires a filtration digest"))?;
                if !is_sha256(filtration) {
                    return Err(fail("filtration_sha256", "must be lowercase hexadecimal SHA-256"));
                }
                if self.operational_alpha_ppb == 0 {
                    return Err(fail("operational_alpha_ppb", "qualified operational claim must declare a positive alpha bound"));
                }
            }
            RiskGuaranteeKind::ActionConditional => {
                if self.action_id.is_none() {
                    return Err(fail("action_id", "action-conditional claim requires an action ID"));
                }
                let support = self.action_support_sha256.as_ref()
                    .ok_or_else(|| fail("action_support_sha256", "action-conditional claim requires a complete support digest"))?;
                if !is_sha256(support) {
                    return Err(fail("action_support_sha256", "must be lowercase hexadecimal SHA-256"));
                }
                if self.filtration_sha256.is_some() {
                    return Err(fail("filtration_sha256", "v0.1 action-conditional claims are fixed-horizon only"));
                }
                if self.operational_alpha_ppb == 0 {
                    return Err(fail("operational_alpha_ppb", "qualified operational claim must declare a positive alpha bound"));
                }
            }
            RiskGuaranteeKind::EmpiricalOnly | RiskGuaranteeKind::Unsupported => {
                if self.operational_alpha_ppb != 0 || self.statistical_delta_ppb != 0 {
                    return Err(fail("risk_allocation", "empirical-only/unsupported claims receive zero formal alpha and delta credit"));
                }
                if self.action_id.is_some() || self.action_support_sha256.is_some() || self.filtration_sha256.is_some() {
                    return Err(fail("conditioning", "no-credit claims cannot participate in conditional/sequential composition"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskBudgetPlan {
    pub schema: String,
    pub version: String,
    pub family_id: String,
    pub target_population_sha256: String,
    pub execution_scope_sha256: String,
    pub subject_sha256: String,
    pub certification_experiment_sha256: String,
    pub operational_alpha_budget_ppb: u64,
    pub statistical_delta_budget_ppb: u64,
    /// Sorted, complete list of composition groups.
    pub groups: Vec<RiskCompositionGroup>,
    /// Sorted, complete list of preregistered contracts.
    pub contracts: Vec<RiskContract>,
    pub plan_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskEvidenceReceipt {
    pub sequence_index: u64,
    pub claim_id: String,
    pub contract_sha256: String,
    pub evidence_sha256: String,
    pub method_receipt_sha256: String,
    pub source_receipt_sha256: Option<String>,
    pub previous_receipt_sha256: Option<String>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskBudgetLedger {
    pub schema: String,
    pub version: String,
    pub plan: RiskBudgetPlan,
    /// Receipts append in the exact preregistered contract order.
    pub receipts: Vec<RiskEvidenceReceipt>,
    pub ledger_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompositionStatus {
    ArithmeticValidComponentsUnverified,
    UnsupportedComponentPreventsFullComposition,
    OperationalBudgetExceeded,
    StatisticalConfidenceBudgetExceeded,
    BothBudgetsExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionBound {
    pub action_id: u32,
    pub operational_alpha_ppb: u128,
    pub contributing_claim_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionGroupReceipt {
    pub group_id: String,
    pub mode: RiskCompositionMode,
    /// None means this group has no formal risk credit.
    pub operational_alpha_bound_ppb: Option<u128>,
    pub action_bounds: Vec<ActionBound>,
    pub claim_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskCompositionReport {
    pub schema: String,
    pub status: CompositionStatus,
    /// This v0.1 bookkeeping layer never claims it proved component theorems.
    pub formal_system_guarantee_claimed: bool,
    pub operational_alpha_bound_ppb: Option<u128>,
    pub operational_alpha_budget_ppb: u64,
    pub statistical_delta_sum_ppb: u128,
    pub statistical_delta_budget_ppb: u64,
    pub groups: Vec<CompositionGroupReceipt>,
    pub plan_sha256: String,
    pub ledger_sha256: String,
    pub report_sha256: String,
    pub claim_ceiling: String,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn is_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    let value = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in value {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}
pub fn sha256_hex(bytes: &[u8]) -> String {
    let value = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in value {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}
fn canonical_json<T: Serialize>(value: &T, field: &str) -> Result<Vec<u8>, RiskLedgerError> {
    serde_json::to_vec(value).map_err(|error| fail(field, format!("canonical JSON serialization failed: {error}")))
}

fn validate_plan_fields(plan: &RiskBudgetPlan) -> Result<(), RiskLedgerError> {
    if plan.schema != "symthaea.risk_budget_plan" || plan.version != "0.1.0" {
        return Err(fail("schema_version", "unsupported risk-budget plan schema/version"));
    }
    if plan.family_id.trim().is_empty() {
        return Err(fail("family_id", "must be non-empty"));
    }
    for (field, value) in [
        ("target_population_sha256", &plan.target_population_sha256),
        ("execution_scope_sha256", &plan.execution_scope_sha256),
        ("subject_sha256", &plan.subject_sha256),
        ("certification_experiment_sha256", &plan.certification_experiment_sha256),
    ] {
        if !is_sha256(value) {
            return Err(fail(field, "must be lowercase hexadecimal SHA-256"));
        }
    }
    if plan.operational_alpha_budget_ppb == 0 || plan.operational_alpha_budget_ppb > PPB {
        return Err(fail("operational_alpha_budget_ppb", "must be in 1..=1,000,000,000"));
    }
    if plan.statistical_delta_budget_ppb > PPB {
        return Err(fail("statistical_delta_budget_ppb", "must be in 0..=1,000,000,000"));
    }
    if plan.groups.is_empty() || plan.groups.len() > MAX_CONTRACTS {
        return Err(fail("groups", "must contain a bounded non-empty set"));
    }
    if plan.contracts.is_empty() || plan.contracts.len() > MAX_CONTRACTS {
        return Err(fail("contracts", "must contain a bounded non-empty complete family"));
    }

    let mut group_ids = BTreeSet::new();
    let mut previous_group: Option<&str> = None;
    for (index, group) in plan.groups.iter().enumerate() {
        if group.group_id.trim().is_empty() {
            return Err(fail(format!("groups[{index}].group_id"), "must be non-empty"));
        }
        if let Some(previous) = previous_group {
            if previous >= group.group_id.as_str() {
                return Err(fail("groups", "group IDs must be unique and strictly sorted"));
            }
        }
        previous_group = Some(&group.group_id);
        if !group_ids.insert(group.group_id.as_str()) {
            return Err(fail("groups", "duplicate group ID"));
        }
        match group.mode {
            RiskCompositionMode::MarginalUnionBound | RiskCompositionMode::NoFormalCredit => {
                if group.action_support_sha256.is_some() || !group.action_ids.is_empty() {
                    return Err(fail(format!("groups[{index}].action_support"), "non-action group cannot declare an action partition"));
                }
            }
            RiskCompositionMode::CompleteActionPartition => {
                let support = group.action_support_sha256.as_ref()
                    .ok_or_else(|| fail(format!("groups[{index}].action_support_sha256"), "complete action partition requires a support digest"))?;
                if !is_sha256(support) {
                    return Err(fail(format!("groups[{index}].action_support_sha256"), "must be lowercase hexadecimal SHA-256"));
                }
                if group.action_ids.is_empty() || group.action_ids.len() > MAX_ACTIONS {
                    return Err(fail(format!("groups[{index}].action_ids"), "must declare a bounded non-empty complete action support"));
                }
                if group.action_ids.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(fail(format!("groups[{index}].action_ids"), "action IDs must be unique and strictly increasing"));
                }
            }
        }
    }

    let mut claim_ids = BTreeSet::new();
    let mut previous_claim: Option<&str> = None;
    let mut contracts_by_group: BTreeMap<&str, Vec<&RiskContract>> = BTreeMap::new();
    for (index, contract) in plan.contracts.iter().enumerate() {
        contract.validate()?;
        if let Some(previous) = previous_claim {
            if previous >= contract.claim_id.as_str() {
                return Err(fail("contracts", "claim IDs must be unique and strictly sorted"));
            }
        }
        previous_claim = Some(&contract.claim_id);
        if !claim_ids.insert(contract.claim_id.as_str()) {
            return Err(fail("contracts", "duplicate claim ID"));
        }
        if !group_ids.contains(contract.group_id.as_str()) {
            return Err(fail(format!("contracts[{index}].group_id"), "claim references an undeclared group"));
        }
        if contract.target_population_sha256 != plan.target_population_sha256
            || contract.execution_scope_sha256 != plan.execution_scope_sha256
            || contract.subject_sha256 != plan.subject_sha256
            || contract.certification_experiment_sha256 != plan.certification_experiment_sha256 {
            return Err(fail(format!("contracts[{index}].scope"), "population, execution, subject, and certification experiment must exactly match the frozen plan"));
        }
        contracts_by_group.entry(contract.group_id.as_str()).or_default().push(contract);
    }

    for group in &plan.groups {
        let members = contracts_by_group.get(group.group_id.as_str())
            .ok_or_else(|| fail("groups", format!("declared group {:?} has no contracts", group.group_id)))?;
        if members.is_empty() {
            return Err(fail("groups", "every declared group must have at least one contract"));
        }
        match group.mode {
            RiskCompositionMode::MarginalUnionBound => {
                let mut event_families = BTreeSet::new();
                for contract in members {
                    if !matches!(contract.guarantee_kind,
                        RiskGuaranteeKind::MarginalFiniteSample
                        | RiskGuaranteeKind::SequentialAnytime
                        | RiskGuaranteeKind::ComputeStopping) {
                        return Err(fail("groups", format!("marginal group {:?} contains non-marginal guarantee type", group.group_id)));
                    }
                    if !event_families.insert(contract.event_family_id.as_str()) {
                        return Err(fail("groups", format!("marginal group {:?} repeats an event family", group.group_id)));
                    }
                }
            }
            RiskCompositionMode::CompleteActionPartition => {
                let support_digest = group.action_support_sha256.as_ref().expect("validated above");
                let declared_actions: BTreeSet<u32> = group.action_ids.iter().copied().collect();
                let mut seen_pairs = BTreeSet::new();
                let mut families_by_action: BTreeMap<u32, BTreeSet<&str>> = BTreeMap::new();
                for contract in members {
                    if contract.guarantee_kind != RiskGuaranteeKind::ActionConditional {
                        return Err(fail("groups", format!("action partition {:?} contains a non-action-conditional claim", group.group_id)));
                    }
                    if contract.action_support_sha256.as_ref() != Some(support_digest) {
                        return Err(fail("groups", "action-conditional claim support digest differs from frozen complete action partition"));
                    }
                    let action_id = contract.action_id.expect("validated action-conditional contract");
                    if !declared_actions.contains(&action_id) {
                        return Err(fail("groups", format!("action {action_id} is outside the declared complete support")));
                    }
                    if !seen_pairs.insert((action_id, contract.event_family_id.as_str())) {
                        return Err(fail("groups", "duplicate event-family/action conditional claim"));
                    }
                    families_by_action.entry(action_id).or_default().insert(contract.event_family_id.as_str());
                }
                let expected_families = families_by_action.values().next()
                    .ok_or_else(|| fail("groups", "action partition has no action families"))?;
                for action_id in &group.action_ids {
                    let action_families = families_by_action.get(action_id)
                        .ok_or_else(|| fail("groups", format!("complete action partition is missing action {action_id}")))?;
                    if action_families != expected_families {
                        return Err(fail("groups", format!("action {action_id} does not contain the same complete event-family set as every other action")));
                    }
                }
                if families_by_action.len() != declared_actions.len() {
                    return Err(fail("groups", "action partition contains undeclared or incomplete action coverage"));
                }
            }
            RiskCompositionMode::NoFormalCredit => {
                for contract in members {
                    if !matches!(contract.guarantee_kind, RiskGuaranteeKind::EmpiricalOnly | RiskGuaranteeKind::Unsupported) {
                        return Err(fail("groups", format!("no-credit group {:?} contains a formal guarantee type", group.group_id)));
                    }
                    if contract.operational_alpha_ppb != 0 || contract.statistical_delta_ppb != 0 {
                        return Err(fail("groups", "no-credit group must have zero formal alpha and delta allocations"));
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn seal_risk_budget_plan(mut plan: RiskBudgetPlan) -> Result<RiskBudgetPlan, RiskLedgerError> {
    plan.plan_sha256.clear();
    validate_plan_fields(&plan)?;
    let bytes = canonical_json(&plan, "plan")?;
    plan.plan_sha256 = digest(PLAN_DOMAIN, &bytes);
    Ok(plan)
}

pub fn validate_risk_budget_plan(plan: &RiskBudgetPlan) -> Result<(), RiskLedgerError> {
    if !is_sha256(&plan.plan_sha256) {
        return Err(fail("plan_sha256", "must be lowercase hexadecimal SHA-256"));
    }
    let mut copy = plan.clone();
    let recorded = copy.plan_sha256.clone();
    copy.plan_sha256.clear();
    validate_plan_fields(&copy)?;
    let expected = digest(PLAN_DOMAIN, &canonical_json(&copy, "plan")?);
    if expected != recorded {
        return Err(fail("plan_sha256", "does not match the complete plan and allocation manifest"));
    }
    Ok(())
}

pub fn create_risk_budget_ledger(plan: RiskBudgetPlan) -> Result<RiskBudgetLedger, RiskLedgerError> {
    validate_risk_budget_plan(&plan)?;
    let mut ledger = RiskBudgetLedger {
        schema: "symthaea.risk_budget_ledger".to_string(),
        version: "0.1.0".to_string(),
        plan,
        receipts: Vec::new(),
        ledger_sha256: String::new(),
    };
    ledger.ledger_sha256 = compute_ledger_digest(&ledger)?;
    Ok(ledger)
}

/// Append evidence for the next preregistered claim. Evidence is accepted only
/// in frozen plan order. The source receipt digest must exactly match the
/// optional upstream-ledger digest bound into that claim.
pub fn append_risk_evidence(
    ledger: &mut RiskBudgetLedger,
    claim_id: &str,
    evidence_sha256: &str,
    method_receipt_sha256: &str,
    source_receipt_sha256: Option<&str>,
) -> Result<RiskEvidenceReceipt, RiskLedgerError> {
    verify_risk_budget_ledger(ledger)?;
    if !is_sha256(evidence_sha256) || !is_sha256(method_receipt_sha256) {
        return Err(fail("evidence_digest", "evidence and method receipt must be SHA-256 digests"));
    }
    if let Some(source) = source_receipt_sha256 {
        if !is_sha256(source) {
            return Err(fail("source_receipt_sha256", "must be lowercase hexadecimal SHA-256"));
        }
    }
    let expected_index = ledger.receipts.len();
    let contract = ledger.plan.contracts.get(expected_index)
        .ok_or_else(|| fail("claim_id", "all preregistered contract slots already have evidence receipts"))?;
    if contract.claim_id != claim_id {
        return Err(fail("claim_id", format!("next preregistered claim is {:?}, not {claim_id:?}", contract.claim_id)));
    }
    if contract.source_receipt_sha256.as_deref() != source_receipt_sha256 {
        return Err(fail("source_receipt_sha256", "does not match the upstream evidence digest preregistered in this contract"));
    }
    let previous_receipt_sha256 = ledger.receipts.last().map(|x| x.receipt_sha256.clone());
    let mut receipt = RiskEvidenceReceipt {
        sequence_index: expected_index as u64,
        claim_id: claim_id.to_string(),
        contract_sha256: sha256_hex(&canonical_json(contract, "contract")?),
        evidence_sha256: evidence_sha256.to_string(),
        method_receipt_sha256: method_receipt_sha256.to_string(),
        source_receipt_sha256: source_receipt_sha256.map(str::to_string),
        previous_receipt_sha256,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = compute_receipt_digest(&receipt)?;
    let mut candidate_ledger = ledger.clone();
    candidate_ledger.receipts.push(receipt.clone());
    candidate_ledger.ledger_sha256 = compute_ledger_digest(&candidate_ledger)?;
    *ledger = candidate_ledger;
    Ok(receipt)
}

pub fn verify_risk_budget_ledger(ledger: &RiskBudgetLedger) -> Result<(), RiskLedgerError> {
    if ledger.schema != "symthaea.risk_budget_ledger" || ledger.version != "0.1.0" {
        return Err(fail("schema_version", "unsupported risk ledger schema/version"));
    }
    validate_risk_budget_plan(&ledger.plan)?;
    if ledger.receipts.len() > ledger.plan.contracts.len() {
        return Err(fail("receipts", "ledger has more receipts than preregistered contracts"));
    }
    let mut previous: Option<String> = None;
    for (index, receipt) in ledger.receipts.iter().enumerate() {
        let contract = &ledger.plan.contracts[index];
        if receipt.sequence_index != index as u64 || receipt.claim_id != contract.claim_id {
            return Err(fail(format!("receipts[{index}]"), "receipt order/identity differs from frozen contract order"));
        }
        let expected_contract_sha = sha256_hex(&canonical_json(contract, "contract")?);
        if receipt.contract_sha256 != expected_contract_sha {
            return Err(fail(format!("receipts[{index}].contract_sha256"), "does not match the preregistered contract"));
        }
        if !is_sha256(&receipt.evidence_sha256)
            || !is_sha256(&receipt.method_receipt_sha256)
            || !is_sha256(&receipt.contract_sha256)
            || !is_sha256(&receipt.receipt_sha256) {
            return Err(fail(format!("receipts[{index}]"), "contains malformed digest"));
        }
        if receipt.source_receipt_sha256 != contract.source_receipt_sha256 {
            return Err(fail(format!("receipts[{index}].source_receipt_sha256"), "differs from preregistered upstream reference"));
        }
        if receipt.previous_receipt_sha256 != previous {
            return Err(fail(format!("receipts[{index}].previous_receipt_sha256"), "does not link to the preceding receipt"));
        }
        let expected = compute_receipt_digest(receipt)?;
        if expected != receipt.receipt_sha256 {
            return Err(fail(format!("receipts[{index}].receipt_sha256"), "does not match the receipt contents"));
        }
        previous = Some(receipt.receipt_sha256.clone());
    }
    let expected_ledger_sha = compute_ledger_digest(ledger)?;
    if ledger.ledger_sha256 != expected_ledger_sha {
        return Err(fail("ledger_sha256", "does not match the plan and ordered evidence receipt chain"));
    }
    Ok(())
}

fn compute_receipt_digest(receipt: &RiskEvidenceReceipt) -> Result<String, RiskLedgerError> {
    let mut copy = receipt.clone();
    copy.receipt_sha256.clear();
    Ok(digest(RECEIPT_DOMAIN, &canonical_json(&copy, "receipt")?))
}
fn compute_ledger_digest(ledger: &RiskBudgetLedger) -> Result<String, RiskLedgerError> {
    let mut copy = ledger.clone();
    copy.ledger_sha256.clear();
    Ok(digest(LEDGER_DOMAIN, &canonical_json(&copy, "ledger")?))
}

fn checked_sum(values: impl IntoIterator<Item = u64>, field: &str) -> Result<u128, RiskLedgerError> {
    values.into_iter().try_fold(0u128, |total, value| {
        total.checked_add(u128::from(value)).ok_or_else(|| fail(field, "integer sum overflow"))
    })
}

pub fn compose_risk_budget(ledger: &RiskBudgetLedger) -> Result<RiskCompositionReport, RiskLedgerError> {
    verify_risk_budget_ledger(ledger)?;
    if ledger.receipts.len() != ledger.plan.contracts.len() {
        return Err(fail("receipts", "cannot compose before every preregistered contract has one evidence receipt"));
    }

    let mut group_receipts = Vec::new();
    let mut system_alpha = 0u128;
    let mut delta_values = Vec::with_capacity(ledger.plan.contracts.len());
    let mut unsupported = false;

    for group in &ledger.plan.groups {
        let members: Vec<&RiskContract> = ledger.plan.contracts.iter()
            .filter(|contract| contract.group_id == group.group_id)
            .collect();
        let claim_ids = members.iter().map(|x| x.claim_id.clone()).collect::<Vec<_>>();
        let (group_alpha, action_bounds) = match group.mode {
            RiskCompositionMode::MarginalUnionBound => {
                let sum = checked_sum(members.iter().map(|x| x.operational_alpha_ppb), "group_alpha")?;
                (Some(sum), Vec::new())
            }
            RiskCompositionMode::CompleteActionPartition => {
                let mut action_bounds = Vec::new();
                let mut maximum = 0u128;
                for action_id in &group.action_ids {
                    let claims: Vec<&RiskContract> = members.iter()
                        .copied()
                        .filter(|contract| contract.action_id == Some(*action_id))
                        .collect();
                    let alpha = checked_sum(claims.iter().map(|x| x.operational_alpha_ppb), "action_alpha")?;
                    maximum = maximum.max(alpha);
                    action_bounds.push(ActionBound {
                        action_id: *action_id,
                        operational_alpha_ppb: alpha,
                        contributing_claim_ids: claims.iter().map(|x| x.claim_id.clone()).collect(),
                    });
                }
                (Some(maximum), action_bounds)
            }
            RiskCompositionMode::NoFormalCredit => {
                unsupported = true;
                (None, Vec::new())
            }
        };
        if let Some(alpha) = group_alpha {
            system_alpha = system_alpha.checked_add(alpha)
                .ok_or_else(|| fail("system_alpha", "integer sum overflow"))?;
        }
        group_receipts.push(CompositionGroupReceipt {
            group_id: group.group_id.clone(),
            mode: group.mode,
            operational_alpha_bound_ppb: group_alpha,
            action_bounds,
            claim_ids,
        });
    }

    for contract in &ledger.plan.contracts {
        delta_values.push(contract.statistical_delta_ppb);
    }
    let delta_sum = checked_sum(delta_values, "statistical_delta")?;
    let alpha_over = system_alpha > u128::from(ledger.plan.operational_alpha_budget_ppb);
    let delta_over = delta_sum > u128::from(ledger.plan.statistical_delta_budget_ppb);

    let status = if unsupported {
        CompositionStatus::UnsupportedComponentPreventsFullComposition
    } else if alpha_over && delta_over {
        CompositionStatus::BothBudgetsExceeded
    } else if alpha_over {
        CompositionStatus::OperationalBudgetExceeded
    } else if delta_over {
        CompositionStatus::StatisticalConfidenceBudgetExceeded
    } else {
        CompositionStatus::ArithmeticValidComponentsUnverified
    };

    let mut report = RiskCompositionReport {
        schema: "symthaea.risk_composition_report".to_string(),
        status,
        formal_system_guarantee_claimed: false,
        operational_alpha_bound_ppb: if unsupported { None } else { Some(system_alpha) },
        operational_alpha_budget_ppb: ledger.plan.operational_alpha_budget_ppb,
        statistical_delta_sum_ppb: delta_sum,
        statistical_delta_budget_ppb: ledger.plan.statistical_delta_budget_ppb,
        groups: group_receipts,
        plan_sha256: ledger.plan.plan_sha256.clone(),
        ledger_sha256: ledger.ledger_sha256.clone(),
        report_sha256: String::new(),
        claim_ceiling: "bookkeeping and compatibility verification only; component theorem, data assumptions, distribution validity, and external attestation are not verified by this ledger".to_string(),
    };
    report.report_sha256 = compute_report_digest(&report)?;
    Ok(report)
}

fn compute_report_digest(report: &RiskCompositionReport) -> Result<String, RiskLedgerError> {
    let mut copy = report.clone();
    copy.report_sha256.clear();
    Ok(digest(REPORT_DOMAIN, &canonical_json(&copy, "report")?))
}

/// Verify the report's digest and its non-authoritative claim ceiling.
/// This checks report integrity only; it does not verify component guarantees.
pub fn verify_risk_composition_report(report: &RiskCompositionReport) -> Result<(), RiskLedgerError> {
    if report.schema != "symthaea.risk_composition_report" {
        return Err(fail("schema", "unsupported composition report schema"));
    }
    if report.formal_system_guarantee_claimed {
        return Err(fail("formal_system_guarantee_claimed", "v0.1 report must never claim theorem-level authority"));
    }
    if !is_sha256(&report.plan_sha256) || !is_sha256(&report.ledger_sha256) || !is_sha256(&report.report_sha256) {
        return Err(fail("report_digest", "plan, ledger, and report digests must be valid SHA-256 values"));
    }
    let expected = compute_report_digest(report)?;
    if expected != report.report_sha256 {
        return Err(fail("report_sha256", "does not match report contents"));
    }
    Ok(())
}

pub const BUILTIN_PLAN_JSON: &str = include_str!("../fixtures/compositional_risk_plan_v0.1.json");
pub fn load_builtin_plan() -> Result<RiskBudgetPlan, RiskLedgerError> {
    let mut plan: RiskBudgetPlan = serde_json::from_str(BUILTIN_PLAN_JSON)
        .map_err(|error| fail("fixture", format!("risk plan JSON parse failed: {error}")))?;
    if plan.plan_sha256.is_empty() {
        plan = seal_risk_budget_plan(plan)?;
    } else {
        validate_risk_budget_plan(&plan)?;
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(ch: char) -> String { std::iter::repeat(ch).take(64).collect() }
    fn git(ch: char) -> String { std::iter::repeat(ch).take(40).collect() }
    fn contract(
        id: &str, event: &str, group: &str, kind: RiskGuaranteeKind,
        alpha: u64, delta: u64,
    ) -> RiskContract {
        let mut c = RiskContract {
            claim_id:id.into(), event_family_id:event.into(), group_id:group.into(),
            guarantee_kind:kind, operational_alpha_ppb:alpha, statistical_delta_ppb:delta,
            target_population_sha256:h('a'), execution_scope_sha256:h('b'), subject_sha256:h('c'),
            certification_experiment_sha256:h('d'), calibration_data_sha256:h('e'),
            assumptions_sha256:h('f'), method_id:"synthetic-test-method".into(), method_version:"0.1".into(),
            paper_or_spec_url:"https://example.org/paper".into(), source_repository_url:"https://github.com/example/research".into(), source_revision:git('1'),
            filtration_sha256:None, action_id:None, action_support_sha256:None, source_receipt_sha256:None,
        };
        if kind == RiskGuaranteeKind::SequentialAnytime { c.filtration_sha256=Some(h('9')); c.source_receipt_sha256=Some(h('8')); }
        c
    }
    fn plan_with(contracts:Vec<RiskContract>,groups:Vec<RiskCompositionGroup>,alpha:u64,delta:u64)->RiskBudgetPlan{
        seal_risk_budget_plan(RiskBudgetPlan{
            schema:"symthaea.risk_budget_plan".into(),version:"0.1.0".into(),family_id:"unit-test-family".into(),
            target_population_sha256:h('a'),execution_scope_sha256:h('b'),subject_sha256:h('c'),
            certification_experiment_sha256:h('d'),operational_alpha_budget_ppb:alpha,
            statistical_delta_budget_ppb:delta,groups,contracts,plan_sha256:String::new()
        }).unwrap()
    }
    fn marginal_group(id:&str)->RiskCompositionGroup{
        RiskCompositionGroup{group_id:id.into(),mode:RiskCompositionMode::MarginalUnionBound,action_support_sha256:None,action_ids:vec![]}
    }
    fn action_group(id:&str)->RiskCompositionGroup{
        RiskCompositionGroup{group_id:id.into(),mode:RiskCompositionMode::CompleteActionPartition,action_support_sha256:Some(h('7')),action_ids:vec![0,1]}
    }
    fn append_all(ledger:&mut RiskBudgetLedger)->Result<(),RiskLedgerError>{
        let contracts=ledger.plan.contracts.clone();
        for c in contracts{
            let evidence=sha256_hex(format!("evidence:{}",c.claim_id).as_bytes());
            let method=sha256_hex(format!("method:{}",c.claim_id).as_bytes());
            append_risk_evidence(ledger,&c.claim_id,&evidence,&method,c.source_receipt_sha256.as_deref())?;
        }
        Ok(())
    }

    #[test]
    fn marginal_and_sequential_bounds_union_and_delta_stays_separate(){
        let mut contracts=vec![
            contract("compute","compute_wrong","marginal",RiskGuaranteeKind::ComputeStopping,5,2),
            contract("prediction","prediction_wrong","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,3),
            contract("temporal","temporal_harm","marginal",RiskGuaranteeKind::SequentialAnytime,20,4),
        ];
        contracts.sort_by(|a,b|a.claim_id.cmp(&b.claim_id));
        let plan=plan_with(contracts,vec![marginal_group("marginal")],40,20);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();append_all(&mut ledger).unwrap();
        let report=compose_risk_budget(&ledger).unwrap();
        assert_eq!(report.operational_alpha_bound_ppb,Some(35));
        assert_eq!(report.statistical_delta_sum_ppb,9);
        assert_eq!(report.status,CompositionStatus::ArithmeticValidComponentsUnverified);
        assert!(!report.formal_system_guarantee_claimed);
    }

    #[test]
    fn complete_action_partition_uses_max_not_sum_across_actions(){
        let mut contracts=vec![
            contract("a0-risk-x","risk-x","actions",RiskGuaranteeKind::ActionConditional,10,1),
            contract("a0-risk-y","risk-y","actions",RiskGuaranteeKind::ActionConditional,20,1),
            contract("a1-risk-x","risk-x","actions",RiskGuaranteeKind::ActionConditional,40,1),
            contract("a1-risk-y","risk-y","actions",RiskGuaranteeKind::ActionConditional,10,1),
        ];
        for c in &mut contracts{c.action_support_sha256=Some(h('7'));}
        contracts[0].action_id=Some(0);contracts[1].action_id=Some(0);
        contracts[2].action_id=Some(1);contracts[3].action_id=Some(1);
        contracts.sort_by(|a,b|a.claim_id.cmp(&b.claim_id));
        let plan=plan_with(contracts,vec![action_group("actions")],50,10);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();append_all(&mut ledger).unwrap();
        let report=compose_risk_budget(&ledger).unwrap();
        assert_eq!(report.operational_alpha_bound_ppb,Some(50));
        let group=&report.groups[0];
        assert_eq!(group.action_bounds[0].operational_alpha_ppb,30);
        assert_eq!(group.action_bounds[1].operational_alpha_ppb,50);
    }

    #[test]
    fn action_partition_composes_with_marginal_and_compute_groups(){
        let mut contracts=vec![
            contract("compute","compute_error","marginal",RiskGuaranteeKind::ComputeStopping,5,1),
            contract("prediction","prediction_error","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,1),
            contract("action-0","unsafe_action","actions",RiskGuaranteeKind::ActionConditional,20,1),
            contract("action-1","unsafe_action","actions",RiskGuaranteeKind::ActionConditional,30,1),
        ];
        contracts[2].action_id=Some(0);contracts[3].action_id=Some(1);
        contracts[2].action_support_sha256=Some(h('7'));contracts[3].action_support_sha256=Some(h('7'));
        contracts.sort_by(|a,b|a.claim_id.cmp(&b.claim_id));
        let plan=plan_with(contracts,vec![action_group("actions"),marginal_group("marginal")],50,10);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();append_all(&mut ledger).unwrap();
        let report=compose_risk_budget(&ledger).unwrap();
        assert_eq!(report.operational_alpha_bound_ppb,Some(45));
        assert_eq!(report.statistical_delta_sum_ppb,4);
    }

    #[test]
    fn empirical_only_components_receive_no_formal_credit(){
        let c=contract("telemetry","observed_error","empirical",RiskGuaranteeKind::EmpiricalOnly,0,0);
        let plan=plan_with(vec![c],vec![RiskCompositionGroup{
            group_id:"empirical".into(),mode:RiskCompositionMode::NoFormalCredit,action_support_sha256:None,action_ids:vec![]
        }],100,100);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();append_all(&mut ledger).unwrap();
        let report=compose_risk_budget(&ledger).unwrap();
        assert_eq!(report.status,CompositionStatus::UnsupportedComponentPreventsFullComposition);
        assert_eq!(report.operational_alpha_bound_ppb,None);
    }

    #[test]
    fn missing_action_in_complete_partition_fails_closed(){
        let mut c=contract("only-action-zero","unsafe_action","actions",RiskGuaranteeKind::ActionConditional,10,1);
        c.action_id=Some(0);c.action_support_sha256=Some(h('7'));
        let plan=RiskBudgetPlan{
            schema:"symthaea.risk_budget_plan".into(),version:"0.1.0".into(),family_id:"bad".into(),
            target_population_sha256:h('a'),execution_scope_sha256:h('b'),subject_sha256:h('c'),
            certification_experiment_sha256:h('d'),operational_alpha_budget_ppb:50,statistical_delta_budget_ppb:10,
            groups:vec![action_group("actions")],contracts:vec![c],plan_sha256:String::new()
        };
        assert!(seal_risk_budget_plan(plan).is_err());
    }

    #[test]
    fn action_partition_requires_same_event_families_for_every_action(){
        let mut a=contract("a0-x","x","actions",RiskGuaranteeKind::ActionConditional,10,1);
        let mut b=contract("a1-y","y","actions",RiskGuaranteeKind::ActionConditional,10,1);
        for c in [&mut a,&mut b]{c.action_support_sha256=Some(h('7'));}
        a.action_id=Some(0);b.action_id=Some(1);
        assert!(seal_risk_budget_plan(RiskBudgetPlan{
            schema:"symthaea.risk_budget_plan".into(),version:"0.1.0".into(),family_id:"bad".into(),
            target_population_sha256:h('a'),execution_scope_sha256:h('b'),subject_sha256:h('c'),
            certification_experiment_sha256:h('d'),operational_alpha_budget_ppb:50,statistical_delta_budget_ppb:10,
            groups:vec![action_group("actions")],contracts:vec![a,b],plan_sha256:String::new()
        }).is_err());
    }

    #[test]
    fn alpha_and_delta_budgets_are_reported_separately(){
        let c=contract("prediction","prediction_error","marginal",RiskGuaranteeKind::MarginalFiniteSample,80,70);
        let plan=plan_with(vec![c],vec![marginal_group("marginal")],50,60);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();append_all(&mut ledger).unwrap();
        let report=compose_risk_budget(&ledger).unwrap();
        assert_eq!(report.status,CompositionStatus::BothBudgetsExceeded);
        assert_eq!(report.operational_alpha_bound_ppb,Some(80));
        assert_eq!(report.statistical_delta_sum_ppb,70);
    }

    #[test]
    fn missing_evidence_receipt_prevents_composition(){
        let c=contract("prediction","prediction_error","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,5);
        let plan=plan_with(vec![c],vec![marginal_group("marginal")],20,10);
        let ledger=create_risk_budget_ledger(plan).unwrap();
        assert!(compose_risk_budget(&ledger).is_err());
    }

    #[test]
    fn append_order_is_frozen_and_duplicate_claim_cannot_be_spent_twice(){
        let a=contract("a","event-a","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,1);
        let b=contract("b","event-b","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,1);
        let plan=plan_with(vec![a,b],vec![marginal_group("marginal")],30,10);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();
        let e=sha256_hex(b"e");let m=sha256_hex(b"m");
        assert!(append_risk_evidence(&mut ledger,"b",&e,&m,None).is_err());
        append_risk_evidence(&mut ledger,"a",&e,&m,None).unwrap();
        assert!(append_risk_evidence(&mut ledger,"a",&e,&m,None).is_err());
    }

    #[test]
    fn receipt_mutation_is_detected_by_chain_verification(){
        let c=contract("prediction","prediction_error","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,1);
        let plan=plan_with(vec![c],vec![marginal_group("marginal")],20,10);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();append_all(&mut ledger).unwrap();
        ledger.receipts[0].evidence_sha256=h('9');
        assert!(verify_risk_budget_ledger(&ledger).is_err());
    }

    #[test]
    fn changed_policy_or_population_digest_invalidates_frozen_plan(){
        let c=contract("prediction","prediction_error","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,1);
        let plan=plan_with(vec![c],vec![marginal_group("marginal")],20,10);
        let mut changed=plan.clone();changed.contracts[0].subject_sha256=h('8');
        assert!(validate_risk_budget_plan(&changed).is_err());
        let mut changed=plan.clone();changed.target_population_sha256=h('8');
        assert!(validate_risk_budget_plan(&changed).is_err());
    }

    #[test]
    fn wrong_execution_scope_cannot_be_combined(){
        let a=contract("a","event-a","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,1);
        let mut b=contract("b","event-b","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,1);
        b.execution_scope_sha256=h('8');
        assert!(seal_risk_budget_plan(RiskBudgetPlan{
            schema:"symthaea.risk_budget_plan".into(),version:"0.1.0".into(),family_id:"bad".into(),
            target_population_sha256:h('a'),execution_scope_sha256:h('b'),subject_sha256:h('c'),
            certification_experiment_sha256:h('d'),operational_alpha_budget_ppb:50,statistical_delta_budget_ppb:10,
            groups:vec![marginal_group("marginal")],contracts:vec![a,b],plan_sha256:String::new()
        }).is_err());
    }

    #[test]
    fn sequential_claim_requires_filtration_but_upstream_receipt_is_optional(){
        let mut c=contract("seq","sequential_harm","marginal",RiskGuaranteeKind::SequentialAnytime,10,1);
        c.filtration_sha256=None;
        assert!(c.validate().is_err());
        let mut c=contract("seq","sequential_harm","marginal",RiskGuaranteeKind::SequentialAnytime,10,1);
        c.source_receipt_sha256=None;
        let plan=plan_with(vec![c],vec![marginal_group("marginal")],20,10);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();
        let e=sha256_hex(b"e");let m=sha256_hex(b"m");
        assert!(append_risk_evidence(&mut ledger,"seq",&e,&m,None).is_ok());
    }

    #[test]
    fn upstream_receipt_digest_must_match_preregistered_value(){
        let c=contract("seq","sequential_harm","marginal",RiskGuaranteeKind::SequentialAnytime,10,1);
        let plan=plan_with(vec![c],vec![marginal_group("marginal")],20,10);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();
        let e=sha256_hex(b"e");let m=sha256_hex(b"m");
        assert!(append_risk_evidence(&mut ledger,"seq",&e,&m,Some(&h('7'))).is_err());
        append_all(&mut ledger).unwrap();
        assert!(verify_risk_budget_ledger(&ledger).is_ok());
    }

    #[test]
    fn no_formal_credit_contract_cannot_claim_nonzero_budgets(){
        let c=contract("telemetry","telemetry","empirical",RiskGuaranteeKind::EmpiricalOnly,1,0);
        let mut plan=RiskBudgetPlan{
            schema:"symthaea.risk_budget_plan".into(),version:"0.1.0".into(),family_id:"bad".into(),
            target_population_sha256:h('a'),execution_scope_sha256:h('b'),subject_sha256:h('c'),
            certification_experiment_sha256:h('d'),operational_alpha_budget_ppb:50,statistical_delta_budget_ppb:10,
            groups:vec![RiskCompositionGroup{group_id:"empirical".into(),mode:RiskCompositionMode::NoFormalCredit,action_support_sha256:None,action_ids:vec![]}],
            contracts:vec![c],plan_sha256:String::new()
        };
        assert!(seal_risk_budget_plan(plan.clone()).is_err());
        plan.contracts[0].operational_alpha_ppb=0;
        assert!(seal_risk_budget_plan(plan).is_ok());
    }

    #[test]
    fn canonical_plan_and_receipt_digests_are_stable(){
        let c=contract("a","event-a","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,1);
        let plan=plan_with(vec![c],vec![marginal_group("marginal")],20,10);
        validate_risk_budget_plan(&plan).unwrap();
        let mut ledger=create_risk_budget_ledger(plan).unwrap();
        append_all(&mut ledger).unwrap();
        let clone=ledger.clone();
        verify_risk_budget_ledger(&clone).unwrap();
        assert_eq!(clone.ledger_sha256,ledger.ledger_sha256);
    }

    #[test]
    fn composition_report_digest_detects_tampering(){
        let c=contract("prediction","prediction_error","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,1);
        let plan=plan_with(vec![c],vec![marginal_group("marginal")],20,10);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();append_all(&mut ledger).unwrap();
        let report=compose_risk_budget(&ledger).unwrap();
        verify_risk_composition_report(&report).unwrap();
        let mut changed=report.clone();
        changed.operational_alpha_bound_ppb=Some(999);
        assert!(verify_risk_composition_report(&changed).is_err());
    }

    #[test]
    fn synthetic_fixture_expected_alpha_and_delta_composition(){
        let plan=load_builtin_plan().unwrap();
        let mut ledger=create_risk_budget_ledger(plan).unwrap();
        append_all(&mut ledger).unwrap();
        let report=compose_risk_budget(&ledger).unwrap();
        assert_eq!(report.operational_alpha_bound_ppb,Some(42_000_000));
        assert_eq!(report.operational_alpha_budget_ppb,50_000_000);
        assert_eq!(report.statistical_delta_sum_ppb,10_000_000);
        assert_eq!(report.statistical_delta_budget_ppb,20_000_000);
        assert_eq!(report.status,CompositionStatus::ArithmeticValidComponentsUnverified);
        verify_risk_composition_report(&report).unwrap();
    }

    #[test]
    fn append_failure_is_transactional(){
        let contract=contract("a","event-a","marginal",RiskGuaranteeKind::MarginalFiniteSample,10,1);
        let plan=plan_with(vec![contract],vec![marginal_group("marginal")],20,10);
        let mut ledger=create_risk_budget_ledger(plan).unwrap();
        let before=ledger.clone();
        let bad=sha256_hex(b"valid");
        assert!(append_risk_evidence(&mut ledger,"a",&bad,"bad-digest",None).is_err());
        assert_eq!(ledger,before);
    }

    #[test]
    fn fixture_plan_parses_and_validates(){
        let plan=load_builtin_plan().unwrap();
        assert_eq!(plan.contracts.len(),7);
        assert_eq!(plan.groups.len(),2);
    }
}
