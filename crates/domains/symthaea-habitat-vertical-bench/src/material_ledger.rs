// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic-only quantity ledger and per-transaction material conservation.
//!
//! The ledger tracks a single textile-fiber component stream in integer mg.
//! It deliberately refuses to infer a conversion from the base fixture's
//! piece-equivalent count, and it is not a whole-product mass or LCA model.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::Value;

pub const EMBEDDED_MATERIAL_LEDGER: &str =
    include_str!("../../../../docs/research/fixtures/habitat-material-ledger-v0.1.json");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerIssue {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialLedgerReport {
    pub issues: Vec<LedgerIssue>,
    pub account_count: usize,
    pub transaction_count: usize,
    pub opening_mass_mg: u64,
    pub closing_mass_mg: u64,
    pub nonzero_closing_account_count: usize,
}

impl MaterialLedgerReport {
    /// True means only that the synthetic ledger and its known answer agree.
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

#[derive(Debug, Clone)]
struct AccountSpec {
    account_class: String,
    unit: String,
    opening: u64,
    expected_closing: u64,
}

#[derive(Debug, Clone)]
struct QuantityLine {
    account_id: String,
    quantity: u64,
    unit: String,
}

pub fn validate_material_ledger_str(
    input: &str,
    base_fixture: &str,
    event_fixture: &str,
) -> Result<MaterialLedgerReport, serde_json::Error> {
    let root: Value = serde_json::from_str(input)?;
    let base: Value = serde_json::from_str(base_fixture)?;
    let events: Value = serde_json::from_str(event_fixture)?;
    Ok(validate_material_ledger(&root, &base, &events))
}

pub fn validate_material_ledger(
    root: &Value,
    base_fixture: &Value,
    event_fixture: &Value,
) -> MaterialLedgerReport {
    let mut issues = Vec::new();
    expect_string(root, "schema_version", "0.1.0", &mut issues);
    expect_string(root, "ledger_id", "HABITAT-MATERIAL-LEDGER-001", &mut issues);
    expect_string(root, "program_id", "HABITAT-VERTICAL-001", &mut issues);
    expect_string(
        root,
        "evidence_status",
        "synthetic_contract_only_not_executed",
        &mut issues,
    );

    if root.pointer("/quantity_basis/unit").and_then(Value::as_str) != Some("mg")
        || root.pointer("/quantity_basis/cross_unit_reconciliation_status")
            .and_then(Value::as_str) != Some("blocked_not_crosswalked")
        || root.pointer("/quantity_basis/quantity_authenticity")
            .and_then(Value::as_str) != Some("synthetic_only")
    {
        push(
            &mut issues,
            "QUANTITY_BASIS_OVERCLAIM",
            "quantity_basis",
            "unit, synthetic-only status, and blocked cross-unit reconciliation must be explicit",
        );
    }
    if !root.get("claim_ceiling").and_then(Value::as_str)
        .is_some_and(|s| s.contains("No physical material quantity"))
    {
        push(
            &mut issues,
            "CLAIM_CEILING_MISSING",
            "claim_ceiling",
            "physical quantity and product mass claim ceiling must be explicit",
        );
    }

    let known_nodes: HashSet<String> = base
        .get("traceability_nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|node| node.get("node_id").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .collect();
    let known_events: HashMap<String, Value> = event_fixture
        .get("events")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|event| {
            let id = event.get("event_id").and_then(Value::as_str)?;
            Some((id.to_owned(), event.clone()))
        })
        .collect();

    let mut accounts = HashMap::<String, AccountSpec>::new();
    let Some(account_rows) = root.get("accounts").and_then(Value::as_array) else {
        push(
            &mut issues,
            "ACCOUNTS_ARRAY_MISSING",
            "accounts",
            "accounts must be an array",
        );
        return report(issues, 0, 0, 0, 0, 0);
    };
    for (index, account) in account_rows.iter().enumerate() {
        let path = format!("accounts[{index}]");
        let Some(id) = account.get("account_id").and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
        else {
            push(&mut issues, "ACCOUNT_ID_MISSING", format!("{path}.account_id"), "account_id must be a non-empty string");
            continue;
        };
        let account_class = account.get("account_class").and_then(Value::as_str).unwrap_or_default().to_owned();
        if account_class.trim().is_empty() {
            push(&mut issues, "ACCOUNT_CLASS_MISSING", format!("{path}.account_class"), "account_class must be a non-empty string");
        }
        let unit = account.get("unit").and_then(Value::as_str).unwrap_or_default().to_owned();
        if unit != "mg" {
            push(&mut issues, "UNIT_MISMATCH", format!("{path}.unit"), "this ledger schema only permits mg");
        }
        let opening = account.get("opening_balance_mg").and_then(Value::as_u64);
        let expected_closing = account.get("expected_closing_balance_mg").and_then(Value::as_u64);
        if opening.is_none() || expected_closing.is_none() {
            push(&mut issues, "INVALID_ACCOUNT_BALANCE", &path, "opening and expected closing balances must be nonnegative integers");
        }
        let traceability_node_id = match account.get("traceability_node_id") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) if !value.trim().is_empty() => Some(value.clone()),
            Some(_) => {
                push(&mut issues, "INVALID_TRACEABILITY_NODE", format!("{path}.traceability_node_id"), "traceability node must be null or a non-empty string");
                None
            }
        };
        if let Some(node_id) = traceability_node_id.as_deref() {
            if !known_nodes.contains(node_id) {
                push(&mut issues, "UNKNOWN_TRACEABILITY_NODE", format!("{path}.traceability_node_id"), format!("traceability node {node_id:?} does not exist in the base fixture"));
            }
        }
        let spec = AccountSpec {
            account_class,
            unit,
            opening: opening.unwrap_or(0),
            expected_closing: expected_closing.unwrap_or(0),
        };
        if accounts.insert(id.to_owned(), spec).is_some() {
            push(&mut issues, "DUPLICATE_ACCOUNT_ID", format!("{path}.account_id"), format!("account id {id:?} is repeated"));
        }
    }

    let opening_mass_mg = match checked_account_total(accounts.values().map(|a| a.opening)) {
        Some(total) => total,
        None => {
            push(&mut issues, "QUANTITY_OVERFLOW", "accounts", "opening mass total overflowed u64");
            0
        }
    };
    if root.pointer("/expected_known_answer/opening_mass_mg").and_then(Value::as_u64)
        != Some(opening_mass_mg)
    {
        push(&mut issues, "OPENING_MASS_MISMATCH", "expected_known_answer.opening_mass_mg", "declared opening mass does not match account opening balances");
    }

    let mut transactions = Vec::<(u64, String, Value)>::new();
    let Some(transaction_rows) = root.get("transactions").and_then(Value::as_array) else {
        push(&mut issues, "TRANSACTIONS_ARRAY_MISSING", "transactions", "transactions must be an array");
        return report(issues, accounts.len(), 0, opening_mass_mg, 0, 0);
    };
    let mut seen_transaction_ids = HashSet::new();
    let mut seen_sequences = HashSet::new();
    for (index, transaction) in transaction_rows.iter().enumerate() {
        let path = format!("transactions[{index}]");
        let Some(id) = transaction.get("transaction_id").and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
        else {
            push(&mut issues, "TRANSACTION_ID_MISSING", format!("{path}.transaction_id"), "transaction_id must be a non-empty string");
            continue;
        };
        if !seen_transaction_ids.insert(id.to_owned()) {
            push(&mut issues, "DUPLICATE_TRANSACTION_ID", format!("{path}.transaction_id"), format!("transaction id {id:?} is repeated"));
        }
        let Some(sequence) = transaction.get("sequence").and_then(Value::as_u64) else {
            push(&mut issues, "INVALID_SEQUENCE", format!("{path}.sequence"), "sequence must be a nonnegative integer");
            continue;
        };
        if sequence == 0 {
            push(&mut issues, "INVALID_SEQUENCE", format!("{path}.sequence"), "sequence numbers start at one");
        }
        if !seen_sequences.insert(sequence) {
            push(&mut issues, "DUPLICATE_SEQUENCE", format!("{path}.sequence"), format!("sequence {sequence} is repeated"));
        }
        transactions.push((sequence, id.to_owned(), transaction.clone()));
    }
    transactions.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    let mut balances: BTreeMap<String, u64> = accounts
        .iter()
        .map(|(id, spec)| (id.clone(), spec.opening))
        .collect();
    let mut last_sequence = 0_u64;
    for (sequence, id, transaction) in &transactions {
        let path = format!("transactions[{id}]");
        if *sequence <= last_sequence {
            if *sequence != last_sequence {
                push(&mut issues, "NON_MONOTONIC_SEQUENCE", format!("{path}.sequence"), "transactions are not in increasing sequence order");
            }
        }
        last_sequence = *sequence;

        let kind = transaction.get("kind").and_then(Value::as_str).unwrap_or_default();
        if !["transformation", "recovery", "quality_transfer", "packing", "dispatch"].contains(&kind) {
            push(&mut issues, "UNKNOWN_TRANSACTION_KIND", format!("{path}.kind"), format!("transaction kind {kind:?} is not allowed"));
        }
        if transaction.get("unit").and_then(Value::as_str) != Some("mg") {
            push(&mut issues, "UNIT_MISMATCH", format!("{path}.unit"), "transaction unit must be mg");
        }

        let source_ids = read_string_array(&transaction, "source_event_ids", &format!("{path}.source_event_ids"), &mut issues);
        let source_link_status = transaction.get("source_link_status").and_then(Value::as_str).unwrap_or_default();
        if source_ids.is_empty() {
            if source_link_status != "unlinked_synthetic_scenario" {
                push(&mut issues, "SOURCE_LINK_STATUS_MISMATCH", format!("{path}.source_link_status"), "transactions without source events must declare unlinked_synthetic_scenario");
            }
        } else {
            if !["context_only", "context_only_component_mass"].contains(&source_link_status) {
                push(&mut issues, "SOURCE_LINK_STATUS_MISMATCH", format!("{path}.source_link_status"), "source-event links are context-only and must be explicitly typed");
            }
            for source_id in &source_ids {
                match known_events.get(source_id) {
                    None => push(&mut issues, "UNKNOWN_SOURCE_EVENT", format!("{path}.source_event_ids"), format!("source event {source_id:?} is not in the event-replay fixture")),
                    Some(source) => {
                        if let Some(expected_operation_id) = transaction.get("expected_operation_id").and_then(Value::as_str) {
                            if source.get("operation_id").and_then(Value::as_str) != Some(expected_operation_id) {
                                push(&mut issues, "SOURCE_OPERATION_MISMATCH", format!("{path}.expected_operation_id"), format!("source event {source_id:?} does not name expected operation {expected_operation_id:?}"));
                            }
                        }
                    }
                }
            }
        }

        let inputs = read_lines(&transaction, "inputs", &path, &accounts, &mut issues);
        let outputs = read_lines(&transaction, "outputs", &path, &accounts, &mut issues);
        let inputs_unique = validate_unique_lines(&inputs, &path, "inputs", &mut issues);
        let outputs_unique = validate_unique_lines(&outputs, &path, "outputs", &mut issues);

        let input_total = sum_lines(&inputs);
        let output_total = sum_lines(&outputs);
        match (input_total, output_total) {
            (Some(input), Some(output)) if input != output => push(
                &mut issues,
                "TRANSACTION_MASS_IMBALANCE",
                &path,
                format!("inputs total {input} mg but outputs/dispositions total {output} mg"),
            ),
            (None, _) | (_, None) => push(
                &mut issues,
                "QUANTITY_OVERFLOW",
                &path,
                "input or output quantity sum overflowed u64",
            ),
            _ => {}
        }

        let input_accounts: HashSet<String> = inputs.iter().map(|line| line.account_id.clone()).collect();
        let output_accounts: HashSet<String> = outputs.iter().map(|line| line.account_id.clone()).collect();
        if input_accounts.iter().any(|account| output_accounts.contains(account)) {
            push(&mut issues, "ACCOUNT_USED_ON_BOTH_SIDES", &path, "the same account may not appear on both sides of a transaction");
        }

        let lines_well_formed = inputs.iter().chain(outputs.iter()).all(|line| {
            line.quantity > 0
                && line.unit == "mg"
                && accounts.get(&line.account_id).is_some_and(|account| account.unit == line.unit)
        });
        let mut can_apply = input_total.is_some() && output_total.is_some() && input_total == output_total
            && lines_well_formed
            && inputs_unique
            && outputs_unique
            && !input_accounts.iter().any(|account| output_accounts.contains(account));
        let mut consumed_by_account = HashMap::<String, u64>::new();
        for line in &inputs {
            let entry = consumed_by_account.entry(line.account_id.clone()).or_insert(0);
            match entry.checked_add(line.quantity) {
                Some(total) => *entry = total,
                None => {
                    can_apply = false;
                    push(&mut issues, "QUANTITY_OVERFLOW", &path, "consumption total overflowed u64");
                }
            }
        }
        for (account_id, quantity) in &consumed_by_account {
            match balances.get(account_id).copied() {
                Some(available) if available >= *quantity => {}
                Some(available) => {
                    can_apply = false;
                    push(&mut issues, "ACCOUNT_OVERDRAW", &path, format!("account {account_id:?} has {available} mg but transaction consumes {quantity} mg"));
                }
                None => can_apply = false,
            }
        }

        if can_apply {
            for (account_id, quantity) in consumed_by_account {
                if let Some(balance) = balances.get_mut(&account_id) {
                    *balance -= quantity;
                }
            }
            let mut produced_by_account = HashMap::<String, u64>::new();
            for line in &outputs {
                let entry = produced_by_account.entry(line.account_id.clone()).or_insert(0);
                match entry.checked_add(line.quantity) {
                    Some(total) => *entry = total,
                    None => {
                        can_apply = false;
                        push(&mut issues, "QUANTITY_OVERFLOW", &path, "production total overflowed u64");
                    }
                }
            }
            for (account_id, quantity) in produced_by_account {
                match balances.get(&account_id).copied().and_then(|balance| balance.checked_add(quantity)) {
                    Some(updated) => { balances.insert(account_id, updated); }
                    None => {
                        can_apply = false;
                        push(&mut issues, "QUANTITY_OVERFLOW", &path, format!("account {account_id:?} balance overflowed u64"));
                    }
                }
            }
            if can_apply {
                let current_total = checked_account_total(balances.values().copied()).unwrap_or(u64::MAX);
                if current_total != opening_mass_mg {
                    push(&mut issues, "GLOBAL_MASS_NOT_CONSERVED", &path, format!("tracked mass changed from {opening_mass_mg} mg to {current_total} mg"));
                }
            }
        }
    }

    let closing_mass_mg = match checked_account_total(balances.values().copied()) {
        Some(total) => total,
        None => {
            push(&mut issues, "QUANTITY_OVERFLOW", "balances", "closing mass total overflowed u64");
            0
        }
    };
    let mut nonzero_closing_account_count = 0;
    for (account_id, spec) in &accounts {
        let actual = balances.get(account_id).copied().unwrap_or(0);
        if actual > 0 {
            nonzero_closing_account_count += 1;
        }
        if actual != spec.expected_closing {
            push(
                &mut issues,
                "CLOSING_BALANCE_MISMATCH",
                format!("accounts[{account_id}].expected_closing_balance_mg"),
                format!("expected {} mg, replay produced {actual} mg", spec.expected_closing),
            );
        }
    }
    if root.pointer("/expected_known_answer/closing_mass_mg").and_then(Value::as_u64) != Some(closing_mass_mg) {
        push(&mut issues, "CLOSING_MASS_MISMATCH", "expected_known_answer.closing_mass_mg", "declared closing mass does not match replayed balances");
    }
    if root.pointer("/expected_known_answer/transaction_count").and_then(Value::as_u64) != Some(transactions.len() as u64) {
        push(&mut issues, "KNOWN_ANSWER_TRANSACTION_COUNT_MISMATCH", "expected_known_answer.transaction_count", "transaction count differs from the frozen known answer");
    }
    if root.pointer("/expected_known_answer/nonzero_closing_account_count").and_then(Value::as_u64) != Some(nonzero_closing_account_count as u64) {
        push(&mut issues, "KNOWN_ANSWER_NONZERO_ACCOUNT_COUNT_MISMATCH", "expected_known_answer.nonzero_closing_account_count", "nonzero closing account count differs from the frozen known answer");
    }

    let class_total = |classes: &[&str]| -> Option<u64> {
        checked_account_total(accounts.iter().filter_map(|(id, spec)| {
            if classes.contains(&spec.account_class.as_str()) {
                balances.get(id).copied()
            } else {
                None
            }
        }))
    };
    let delivered_component = class_total(&["delivered_component"]);
    let recoverable_material = class_total(&["recoverable_offcut", "recovered_feedstock"]);
    let scrap_and_residue = class_total(&["scrap", "residue"]);
    for (field, actual) in [
        ("accepted_component_mass_mg", delivered_component),
        ("recoverable_material_mg", recoverable_material),
        ("scrap_and_residue_mg", scrap_and_residue),
    ] {
        let expected = root.pointer(&format!("/expected_known_answer/{field}")).and_then(Value::as_u64);
        if actual.is_none() || expected != actual {
            push(
                &mut issues,
                "KNOWN_ANSWER_CLASS_TOTAL_MISMATCH",
                format!("expected_known_answer.{field}"),
                format!("declared value {expected:?} does not match replayed account-class total {actual:?}"),
            );
        }
    }

    report(issues, accounts.len(), transactions.len(), opening_mass_mg, closing_mass_mg, nonzero_closing_account_count)
}

fn read_lines(
    transaction: &Value,
    field: &str,
    path: &str,
    accounts: &HashMap<String, AccountSpec>,
    issues: &mut Vec<LedgerIssue>,
) -> Vec<QuantityLine> {
    let Some(rows) = transaction.get(field).and_then(Value::as_array) else {
        push(issues, "QUANTITY_LINES_MISSING", format!("{path}.{field}"), "quantity line list must be an array");
        return Vec::new();
    };
    if rows.is_empty() {
        push(issues, "QUANTITY_LINES_EMPTY", format!("{path}.{field}"), "input and output sides must each contain at least one quantity line");
    }
    let tx_unit = transaction.get("unit").and_then(Value::as_str).unwrap_or_default();
    let mut result = Vec::new();
    for (index, line) in rows.iter().enumerate() {
        let line_path = format!("{path}.{field}[{index}]");
        let Some(account_id) = line.get("account_id").and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
        else {
            push(issues, "ACCOUNT_ID_MISSING", format!("{line_path}.account_id"), "account_id must be a non-empty string");
            continue;
        };
        let Some(quantity) = line.get("quantity_mg").and_then(Value::as_u64) else {
            push(issues, "INVALID_QUANTITY", format!("{line_path}.quantity_mg"), "quantity_mg must be a nonnegative integer");
            continue;
        };
        if quantity == 0 {
            push(issues, "INVALID_QUANTITY", format!("{line_path}.quantity_mg"), "zero quantity lines are not permitted");
        }
        let unit = line.get("unit").and_then(Value::as_str).unwrap_or_default().to_owned();
        if unit != tx_unit {
            push(issues, "UNIT_MISMATCH", format!("{line_path}.unit"), "line unit differs from transaction unit");
        }
        match accounts.get(account_id) {
            None => push(issues, "UNKNOWN_ACCOUNT", format!("{line_path}.account_id"), format!("account {account_id:?} is not declared")),
            Some(spec) if spec.unit != unit => push(issues, "UNIT_MISMATCH", format!("{line_path}.unit"), "line unit differs from account unit"),
            Some(_) => {}
        }
        result.push(QuantityLine { account_id: account_id.to_owned(), quantity, unit });
    }
    result
}

fn validate_unique_lines(lines: &[QuantityLine], path: &str, side: &str, issues: &mut Vec<LedgerIssue>) -> bool {
    let mut seen = HashSet::new();
    let mut unique = true;
    for line in lines {
        if !seen.insert(line.account_id.clone()) {
            unique = false;
            push(issues, "DUPLICATE_ACCOUNT_LINE", format!("{path}.{side}"), format!("account {:?} appears more than once on the same side", line.account_id));
        }
    }
    unique
}

fn sum_lines(lines: &[QuantityLine]) -> Option<u64> {
    lines.iter().try_fold(0_u64, |total, line| total.checked_add(line.quantity))
}

fn checked_account_total<I: Iterator<Item = u64>>(values: I) -> Option<u64> {
    values.try_fold(0_u64, |total, value| total.checked_add(value))
}

fn read_string_array(root: &Value, field: &str, path: &str, issues: &mut Vec<LedgerIssue>) -> Vec<String> {
    let Some(values) = root.get(field).and_then(Value::as_array) else {
        push(issues, "STRING_ARRAY_MISSING", path, "required source-event list must be an array");
        return Vec::new();
    };
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    for (index, value) in values.iter().enumerate() {
        match value.as_str().filter(|s| !s.trim().is_empty()) {
            Some(id) => {
                if !seen.insert(id.to_owned()) {
                    push(issues, "DUPLICATE_SOURCE_EVENT", format!("{path}[{index}]"), format!("source event {id:?} appears more than once"));
                }
                result.push(id.to_owned());
            }
            None => push(issues, "INVALID_SOURCE_EVENT", format!("{path}[{index}]"), "source event identifiers must be non-empty strings"),
        }
    }
    result
}

fn expect_string(root: &Value, field: &str, expected: &str, issues: &mut Vec<LedgerIssue>) {
    if root.get(field).and_then(Value::as_str) != Some(expected) {
        push(issues, "EXPECTED_STRING_MISMATCH", field, format!("required value is {expected:?}"));
    }
}

fn push(issues: &mut Vec<LedgerIssue>, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
    issues.push(LedgerIssue { code, path: path.into(), message: message.into() });
}

fn report(
    mut issues: Vec<LedgerIssue>,
    account_count: usize,
    transaction_count: usize,
    opening_mass_mg: u64,
    closing_mass_mg: u64,
    nonzero_closing_account_count: usize,
) -> MaterialLedgerReport {
    issues.sort_by(|left, right| left.code.cmp(right.code).then_with(|| left.path.cmp(&right.path)).then_with(|| left.message.cmp(&right.message)));
    MaterialLedgerReport { issues, account_count, transaction_count, opening_mass_mg, closing_mass_mg, nonzero_closing_account_count }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_pair() -> (Value, Value, Value) {
        (
            serde_json::from_str(EMBEDDED_MATERIAL_LEDGER).expect("material ledger JSON"),
            serde_json::from_str(crate::EMBEDDED_FIXTURE).expect("base fixture JSON"),
            serde_json::from_str(crate::event_replay::EMBEDDED_EVENT_LOG).expect("event fixture JSON"),
        )
    }

    fn tx_index(root: &Value, id: &str) -> usize {
        root["transactions"].as_array().expect("transactions")
            .iter().position(|tx| tx["transaction_id"].as_str() == Some(id))
            .expect("transaction exists")
    }

    fn has_code(report: &MaterialLedgerReport, code: &str) -> bool {
        report.issues.iter().any(|issue| issue.code == code)
    }

    #[test]
    fn embedded_ledger_matches_mass_known_answer() {
        let (root, base, events) = fixture_pair();
        let report = validate_material_ledger(&root, &base, &events);
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.account_count, 16);
        assert_eq!(report.transaction_count, 9);
        assert_eq!(report.opening_mass_mg, 1_000_000);
        assert_eq!(report.closing_mass_mg, 1_000_000);
        assert_eq!(report.nonzero_closing_account_count, 8);
    }

    #[test]
    fn transaction_input_permutation_preserves_result() {
        let (mut root, base, events) = fixture_pair();
        let before = validate_material_ledger(&root, &base, &events);
        root["transactions"].as_array_mut().expect("transactions").reverse();
        let after = validate_material_ledger(&root, &base, &events);
        assert!(before.is_valid(), "{:?}", before.issues);
        assert!(after.is_valid(), "{:?}", after.issues);
        assert_eq!(before.opening_mass_mg, after.opening_mass_mg);
        assert_eq!(before.closing_mass_mg, after.closing_mass_mg);
    }

    #[test]
    fn exact_mass_is_conserved_after_every_transaction() {
        let (root, base, events) = fixture_pair();
        let report = validate_material_ledger(&root, &base, &events);
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.opening_mass_mg, report.closing_mass_mg);
    }

    #[test]
    fn overdrawn_source_account_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        let index = tx_index(&root, "ML-01-CUT-FABRIC");
        root["transactions"][index]["inputs"][0]["quantity_mg"] = Value::from(1_000_001_u64);
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "ACCOUNT_OVERDRAW"));
    }

    #[test]
    fn imbalanced_transformation_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        let index = tx_index(&root, "ML-01-CUT-FABRIC");
        root["transactions"][index]["outputs"][0]["quantity_mg"] = Value::from(479_999_u64);
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "TRANSACTION_MASS_IMBALANCE"));
    }

    #[test]
    fn unknown_account_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        let index = tx_index(&root, "ML-01-CUT-FABRIC");
        root["transactions"][index]["inputs"][0]["account_id"] = Value::String("ACC-DOES-NOT-EXIST".to_owned());
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "UNKNOWN_ACCOUNT"));
    }

    #[test]
    fn unknown_source_event_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        let index = tx_index(&root, "ML-01-CUT-FABRIC");
        root["transactions"][index]["source_event_ids"] = serde_json::json!(["EV-MISSING"]);
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "UNKNOWN_SOURCE_EVENT"));
    }

    #[test]
    fn mixed_unit_line_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        let index = tx_index(&root, "ML-01-CUT-FABRIC");
        root["transactions"][index]["outputs"][0]["unit"] = Value::String("g".to_owned());
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "UNIT_MISMATCH"));
    }

    #[test]
    fn duplicate_transaction_sequence_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        let first_sequence = root["transactions"][0]["sequence"].clone();
        root["transactions"][1]["sequence"] = first_sequence;
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "DUPLICATE_SEQUENCE"));
    }

    #[test]
    fn unknown_traceability_node_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        root["accounts"][0]["traceability_node_id"] = Value::String("NODE-NOT-DECLARED".to_owned());
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "UNKNOWN_TRACEABILITY_NODE"));
    }

    #[test]
    fn incorrect_expected_closing_balance_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        root["accounts"][0]["expected_closing_balance_mg"] = Value::from(400_001_u64);
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "CLOSING_BALANCE_MISMATCH"));
    }

    #[test]
    fn duplicate_account_lines_are_rejected() {
        let (mut root, base, events) = fixture_pair();
        let index = tx_index(&root, "ML-01-CUT-FABRIC");
        let duplicate = root["transactions"][index]["inputs"][0].clone();
        root["transactions"][index]["inputs"].as_array_mut().expect("inputs").push(duplicate);
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "DUPLICATE_ACCOUNT_LINE"));
    }

    #[test]
    fn absent_cross_unit_boundary_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        root["quantity_basis"]["cross_unit_reconciliation_status"] = Value::String("reconciled".to_owned());
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "QUANTITY_BASIS_OVERCLAIM"));
    }

    #[test]
    fn source_operation_mismatch_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        let index = tx_index(&root, "ML-01-CUT-FABRIC");
        root["transactions"][index]["expected_operation_id"] = Value::String("OP-70-PACK".to_owned());
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "SOURCE_OPERATION_MISMATCH"));
    }

    #[test]
    fn physical_quantity_claim_is_rejected() {
        let (mut root, base, events) = fixture_pair();
        root["evidence_status"] = Value::String("observed_physical".to_owned());
        let report = validate_material_ledger(&root, &base, &events);
        assert!(has_code(&report, "EXPECTED_STRING_MISMATCH"));
    }
}
