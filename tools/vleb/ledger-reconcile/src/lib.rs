//! Reference implementation for the synthetic VLEB ledger-reconciliation contract.
//!
//! This crate uses only the Rust standard library so the reference implementation is
//! reproducible without fetching dependencies. It is a fixture for testing the benchmark
//! contract, not a banking ledger or production accounting system.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

const EXPECTED_HEADER: [&str; 4] = ["transaction_id", "account_id", "side", "amount_minor"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerError {
    message: String,
}

impl LedgerError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LedgerError {}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Side {
    Credit,
    Debit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Transaction {
    transaction_id: String,
    account_id: String,
    side: Side,
    amount_minor: u64,
}

#[derive(Debug, Default)]
struct AccountTotals {
    credits_minor: u64,
    debits_minor: u64,
    unique_transactions: u64,
}

#[derive(Debug, Default)]
struct Totals {
    credits_minor: u64,
    debits_minor: u64,
    unique_transactions: u64,
    exact_duplicate_rows_ignored: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccountReport {
    pub account_id: String,
    pub credits_minor: u64,
    pub debits_minor: u64,
    pub net_minor: i128,
    pub unique_transactions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerReport {
    pub accounts: Vec<AccountReport>,
    pub credits_minor: u64,
    pub debits_minor: u64,
    pub net_minor: i128,
    pub unique_transactions: u64,
    pub exact_duplicate_rows_ignored: u64,
}

impl LedgerReport {
    /// Return deterministic JSON matching the VLEB fixture contract.
    pub fn to_json(&self) -> String {
        let mut output = String::new();
        output.push_str("{\n");
        output.push_str("  \"schema_version\": \"vleb-ledger-output-v1\",\n");
        output.push_str("  \"currency_semantics\": \"integer_minor_units_unspecified_currency\",\n");
        output.push_str("  \"accounts\": [");

        if self.accounts.is_empty() {
            output.push_str("],\n");
        } else {
            output.push('\n');
            for (index, account) in self.accounts.iter().enumerate() {
                output.push_str("    {\n");
                output.push_str("      \"account_id\": ");
                push_json_string(&mut output, &account.account_id);
                output.push_str(",\n");
                output.push_str(&format!("      \"credits_minor\": {},\n", account.credits_minor));
                output.push_str(&format!("      \"debits_minor\": {},\n", account.debits_minor));
                output.push_str(&format!("      \"net_minor\": {},\n", account.net_minor));
                output.push_str(&format!("      \"unique_transactions\": {}\n", account.unique_transactions));
                output.push_str("    }");
                if index + 1 != self.accounts.len() {
                    output.push(',');
                }
                output.push('\n');
            }
            output.push_str("  ],\n");
        }

        output.push_str("  \"totals\": {\n");
        output.push_str(&format!("    \"credits_minor\": {},\n", self.credits_minor));
        output.push_str(&format!("    \"debits_minor\": {},\n", self.debits_minor));
        output.push_str(&format!("    \"net_minor\": {},\n", self.net_minor));
        output.push_str(&format!("    \"unique_transactions\": {},\n", self.unique_transactions));
        output.push_str(&format!("    \"exact_duplicate_rows_ignored\": {}\n", self.exact_duplicate_rows_ignored));
        output.push_str("  }\n");
        output.push_str("}\n");
        output
    }
}

/// Parse, validate and reconcile the entire CSV input.
///
/// This function has no externally visible side effects. It returns a report only
/// after every row has validated, all duplicate-ID relationships are consistent, and
/// all arithmetic has completed with checked operations. A caller must not emit a
/// partial report on an error.
pub fn reconcile_csv(input: &str) -> Result<LedgerReport, LedgerError> {
    let rows = parse_csv(input)?;
    let Some(header) = rows.first() else {
        return Err(LedgerError::new("CSV is empty; expected header row"));
    };

    if header.len() != EXPECTED_HEADER.len()
        || !header
            .iter()
            .zip(EXPECTED_HEADER.iter())
            .all(|(actual, expected)| actual.as_str() == *expected)
    {
        return Err(LedgerError::new(
            "invalid CSV header; expected transaction_id,account_id,side,amount_minor",
        ));
    }

    let mut seen: HashMap<String, Transaction> = HashMap::new();
    let mut accounts: BTreeMap<String, AccountTotals> = BTreeMap::new();
    let mut totals = Totals::default();

    for (row_index, row) in rows.iter().enumerate().skip(1) {
        if row.len() != EXPECTED_HEADER.len() {
            return Err(LedgerError::new(format!(
                "row {} has {} fields; expected {}",
                row_index + 1,
                row.len(),
                EXPECTED_HEADER.len()
            )));
        }

        let transaction_id = row[0].clone();
        let account_id = row[1].clone();
        validate_identifier("transaction_id", &transaction_id, row_index + 1)?;
        validate_identifier("account_id", &account_id, row_index + 1)?;

        let side = match row[2].as_str() {
            "CREDIT" => Side::Credit,
            "DEBIT" => Side::Debit,
            other => {
                return Err(LedgerError::new(format!(
                    "row {} has invalid side {other:?}; expected CREDIT or DEBIT",
                    row_index + 1
                )));
            }
        };

        if row[3].is_empty() || !row[3].bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(LedgerError::new(format!(
                "row {} amount_minor must contain only ASCII digits",
                row_index + 1
            )));
        }
        let amount_minor = row[3].parse::<u64>().map_err(|_| {
            LedgerError::new(format!(
                "row {} amount_minor is outside the supported unsigned integer range",
                row_index + 1
            ))
        })?;
        if amount_minor == 0 {
            return Err(LedgerError::new(format!(
                "row {} amount_minor must be greater than zero",
                row_index + 1
            )));
        }

        let transaction = Transaction {
            transaction_id: transaction_id.clone(),
            account_id: account_id.clone(),
            side: side.clone(),
            amount_minor,
        };

        if let Some(previous) = seen.get(&transaction_id) {
            if previous == &transaction {
                totals.exact_duplicate_rows_ignored = totals
                    .exact_duplicate_rows_ignored
                    .checked_add(1)
                    .ok_or_else(|| LedgerError::new("duplicate-row counter overflow"))?;
                continue;
            }
            return Err(LedgerError::new(format!(
                "conflicting duplicate transaction_id {transaction_id:?}; input rejected"
            )));
        }

        // Insert only after the whole row validates.
        seen.insert(transaction_id, transaction);
        let account = accounts.entry(account_id).or_default();

        match side {
            Side::Credit => {
                account.credits_minor = account.credits_minor
                    .checked_add(amount_minor)
                    .ok_or_else(|| LedgerError::new("account credit total overflow"))?;
                totals.credits_minor = totals.credits_minor
                    .checked_add(amount_minor)
                    .ok_or_else(|| LedgerError::new("global credit total overflow"))?;
            }
            Side::Debit => {
                account.debits_minor = account.debits_minor
                    .checked_add(amount_minor)
                    .ok_or_else(|| LedgerError::new("account debit total overflow"))?;
                totals.debits_minor = totals.debits_minor
                    .checked_add(amount_minor)
                    .ok_or_else(|| LedgerError::new("global debit total overflow"))?;
            }
        }

        account.unique_transactions = account.unique_transactions
            .checked_add(1)
            .ok_or_else(|| LedgerError::new("account transaction count overflow"))?;
        totals.unique_transactions = totals.unique_transactions
            .checked_add(1)
            .ok_or_else(|| LedgerError::new("global transaction count overflow"))?;
    }

    let report_accounts = accounts
        .into_iter()
        .map(|(account_id, account)| AccountReport {
            account_id,
            credits_minor: account.credits_minor,
            debits_minor: account.debits_minor,
            net_minor: i128::from(account.credits_minor) - i128::from(account.debits_minor),
            unique_transactions: account.unique_transactions,
        })
        .collect();

    Ok(LedgerReport {
        accounts: report_accounts,
        credits_minor: totals.credits_minor,
        debits_minor: totals.debits_minor,
        net_minor: i128::from(totals.credits_minor) - i128::from(totals.debits_minor),
        unique_transactions: totals.unique_transactions,
        exact_duplicate_rows_ignored: totals.exact_duplicate_rows_ignored,
    })
}

fn validate_identifier(name: &str, value: &str, row_number: usize) -> Result<(), LedgerError> {
    if value.trim().is_empty() {
        return Err(LedgerError::new(format!("row {row_number} {name} must not be empty")));
    }
    if value != value.trim() {
        return Err(LedgerError::new(format!(
            "row {row_number} {name} must not have leading or trailing whitespace"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(LedgerError::new(format!(
            "row {row_number} {name} must not contain control characters"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CsvState {
    FieldStart,
    Unquoted,
    Quoted,
    QuoteClosed,
}

/// Strict CSV parser supporting quoted fields and doubled quotes.
/// CRLF record endings are accepted; bare CR and characters after a closing quote
/// are rejected. Quoted line breaks are normalized to LF.
fn parse_csv(input: &str) -> Result<Vec<Vec<String>>, LedgerError> {
    let bytes = input.as_bytes();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field: Vec<u8> = Vec::new();
    let mut state = CsvState::FieldStart;
    let mut touched = false;
    let mut index = 0usize;

    while index < bytes.len() {
        let byte = bytes[index];
        match state {
            CsvState::FieldStart => match byte {
                b'"' => {
                    state = CsvState::Quoted;
                    touched = true;
                    index += 1;
                }
                b',' => {
                    row.push(String::new());
                    touched = true;
                    index += 1;
                }
                b'\n' => {
                    row.push(String::from_utf8(std::mem::take(&mut field))
                        .map_err(|_| LedgerError::new("CSV contains invalid UTF-8"))?);
                    rows.push(std::mem::take(&mut row));
                    touched = false;
                    index += 1;
                }
                b'\r' => {
                    if bytes.get(index + 1) != Some(&b'\n') {
                        return Err(LedgerError::new(format!("bare carriage return at byte {index}")));
                    }
                    row.push(String::from_utf8(std::mem::take(&mut field))
                        .map_err(|_| LedgerError::new("CSV contains invalid UTF-8"))?);
                    rows.push(std::mem::take(&mut row));
                    touched = false;
                    index += 2;
                }
                _ => {
                    field.push(byte);
                    state = CsvState::Unquoted;
                    touched = true;
                    index += 1;
                }
            },
            CsvState::Unquoted => match byte {
                b',' => {
                    row.push(String::from_utf8(std::mem::take(&mut field))
                        .map_err(|_| LedgerError::new("CSV contains invalid UTF-8"))?);
                    state = CsvState::FieldStart;
                    touched = true;
                    index += 1;
                }
                b'\n' => {
                    row.push(String::from_utf8(std::mem::take(&mut field))
                        .map_err(|_| LedgerError::new("CSV contains invalid UTF-8"))?);
                    rows.push(std::mem::take(&mut row));
                    state = CsvState::FieldStart;
                    touched = false;
                    index += 1;
                }
                b'\r' => {
                    if bytes.get(index + 1) != Some(&b'\n') {
                        return Err(LedgerError::new(format!("bare carriage return at byte {index}")));
                    }
                    row.push(String::from_utf8(std::mem::take(&mut field))
                        .map_err(|_| LedgerError::new("CSV contains invalid UTF-8"))?);
                    rows.push(std::mem::take(&mut row));
                    state = CsvState::FieldStart;
                    touched = false;
                    index += 2;
                }
                b'"' => {
                    return Err(LedgerError::new(format!(
                        "unexpected quote in unquoted field at byte {index}"
                    )));
                }
                _ => {
                    field.push(byte);
                    index += 1;
                }
            },
            CsvState::Quoted => match byte {
                b'"' => {
                    if bytes.get(index + 1) == Some(&b'"') {
                        field.push(b'"');
                        index += 2;
                    } else {
                        state = CsvState::QuoteClosed;
                        index += 1;
                    }
                }
                b'\n' => {
                    field.push(b'\n');
                    index += 1;
                }
                b'\r' => {
                    if bytes.get(index + 1) != Some(&b'\n') {
                        return Err(LedgerError::new(format!(
                            "bare carriage return inside quoted field at byte {index}"
                        )));
                    }
                    field.push(b'\n');
                    index += 2;
                }
                _ => {
                    field.push(byte);
                    index += 1;
                }
            },
            CsvState::QuoteClosed => match byte {
                b',' => {
                    row.push(String::from_utf8(std::mem::take(&mut field))
                        .map_err(|_| LedgerError::new("CSV contains invalid UTF-8"))?);
                    state = CsvState::FieldStart;
                    touched = true;
                    index += 1;
                }
                b'\n' => {
                    row.push(String::from_utf8(std::mem::take(&mut field))
                        .map_err(|_| LedgerError::new("CSV contains invalid UTF-8"))?);
                    rows.push(std::mem::take(&mut row));
                    state = CsvState::FieldStart;
                    touched = false;
                    index += 1;
                }
                b'\r' => {
                    if bytes.get(index + 1) != Some(&b'\n') {
                        return Err(LedgerError::new(format!("bare carriage return at byte {index}")));
                    }
                    row.push(String::from_utf8(std::mem::take(&mut field))
                        .map_err(|_| LedgerError::new("CSV contains invalid UTF-8"))?);
                    rows.push(std::mem::take(&mut row));
                    state = CsvState::FieldStart;
                    touched = false;
                    index += 2;
                }
                _ => {
                    return Err(LedgerError::new(format!(
                        "unexpected character after closing quote at byte {index}"
                    )));
                }
            },
        }
    }

    if state == CsvState::Quoted {
        return Err(LedgerError::new("unterminated quoted CSV field"));
    }

    if touched || !field.is_empty() || !row.is_empty() || state == CsvState::QuoteClosed {
        row.push(String::from_utf8(field)
            .map_err(|_| LedgerError::new("CSV contains invalid UTF-8"))?);
        rows.push(row);
    }

    Ok(rows)
}

fn push_json_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000C}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{001F}' => {
                output.push_str(&format!("\\u{:04x}", u32::from(character)));
            }
            character => output.push(character),
        }
    }
    output.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT: &str = include_str!("../../examples/ledger-reconciliation/input.csv");
    const EXPECTED: &str = include_str!("../../examples/ledger-reconciliation/expected.json");
    const CONFLICT: &str = include_str!("../../examples/ledger-reconciliation/conflicting-duplicate.csv");

    #[test]
    fn fixture_matches_exact_expected_output() {
        let report = reconcile_csv(INPUT).expect("normal fixture should reconcile");
        assert_eq!(report.to_json(), EXPECTED);
    }

    #[test]
    fn conflicting_transaction_id_is_rejected() {
        let error = reconcile_csv(CONFLICT).expect_err("conflicting ID must fail");
        assert!(error.to_string().contains("conflicting duplicate transaction_id"));
    }

    #[test]
    fn zero_and_negative_amounts_are_rejected() {
        for amount in ["0", "-1"] {
            let input = format!(
                "transaction_id,account_id,side,amount_minor\nTX-1,AC-1,CREDIT,{amount}\n"
            );
            assert!(reconcile_csv(&input).is_err(), "amount {amount} should fail");
        }
    }

    #[test]
    fn unknown_side_is_rejected() {
        let input = "transaction_id,account_id,side,amount_minor\nTX-1,AC-1,TRANSFER,1\n";
        assert!(reconcile_csv(input).is_err());
    }

    #[test]
    fn wrong_header_and_wrong_field_count_are_rejected() {
        assert!(reconcile_csv("transaction_id,amount\nTX-1,10\n").is_err());
        assert!(reconcile_csv(
            "transaction_id,account_id,side,amount_minor\nTX-1,AC-1,CREDIT\n"
        ).is_err());
    }

    #[test]
    fn duplicate_id_with_different_account_or_side_is_a_conflict() {
        let input = concat!(
            "transaction_id,account_id,side,amount_minor\n",
            "TX-1,AC-1,CREDIT,100\n",
            "TX-1,AC-2,CREDIT,100\n"
        );
        assert!(reconcile_csv(input).is_err());
    }

    #[test]
    fn arithmetic_overflow_is_rejected() {
        let input = format!(
            "transaction_id,account_id,side,amount_minor\nTX-1,AC-1,CREDIT,{}\nTX-2,AC-1,CREDIT,1\n",
            u64::MAX
        );
        let error = reconcile_csv(&input).expect_err("credit total overflow must fail");
        assert!(error.to_string().contains("overflow"));
    }

    #[test]
    fn empty_ledger_with_header_has_zero_totals() {
        let report = reconcile_csv("transaction_id,account_id,side,amount_minor\n")
            .expect("header-only ledger is a valid empty input");
        assert!(report.accounts.is_empty());
        assert_eq!(report.credits_minor, 0);
        assert_eq!(report.debits_minor, 0);
        assert_eq!(report.net_minor, 0);
    }

    #[test]
    fn quoted_fields_and_crlf_are_supported() {
        let valid = "transaction_id,account_id,side,amount_minor\r\n\"TX-\"\"1\"\"\",AC-1,CREDIT,5\r\n";
        let report = reconcile_csv(valid).expect("valid escaped quote should parse");
        assert_eq!(report.accounts[0].account_id, "AC-1");
        assert_eq!(report.accounts[0].unique_transactions, 1);
    }

    #[test]
    fn amount_tokens_must_be_canonical_ascii_integers() {
        for amount in ["+1", " 1", "1 ", "1.0", "1e2", "١"] {
            let input = format!(
                "transaction_id,account_id,side,amount_minor\nTX-1,AC-1,CREDIT,{amount}\n"
            );
            assert!(
                reconcile_csv(&input).is_err(),
                "amount token {amount:?} should be rejected"
            );
        }
    }

    #[test]
    fn equal_maximum_credit_and_debit_are_valid_without_net_overflow() {
        let input = format!(
            "transaction_id,account_id,side,amount_minor\nTX-C,AC-1,CREDIT,{}\nTX-D,AC-1,DEBIT,{}\n",
            u64::MAX,
            u64::MAX
        );
        let report = reconcile_csv(&input).expect("separate checked totals fit");
        assert_eq!(report.credits_minor, u64::MAX);
        assert_eq!(report.debits_minor, u64::MAX);
        assert_eq!(report.net_minor, 0);
        assert_eq!(report.accounts[0].net_minor, 0);
    }

    #[test]
    fn transaction_order_does_not_change_the_result() {
        let lines: Vec<&str> = INPUT.trim_end_matches('\n').lines().collect();
        let mut reversed = String::new();
        reversed.push_str(lines[0]);
        reversed.push('\n');
        for line in lines[1..].iter().rev() {
            reversed.push_str(line);
            reversed.push('\n');
        }
        let original = reconcile_csv(INPUT).expect("original fixture is valid");
        let reordered = reconcile_csv(&reversed).expect("permuted fixture is valid");
        assert_eq!(original, reordered);
    }

    #[test]
    fn presence_of_final_line_ending_does_not_change_semantics() {
        let with_newline = reconcile_csv(INPUT).expect("fixture with final newline is valid");
        let without_newline = reconcile_csv(INPUT.trim_end_matches('\n'))
            .expect("fixture without final newline is valid");
        assert_eq!(with_newline, without_newline);
    }

    #[test]
    fn repeated_exact_duplicate_is_idempotent_for_arbitrary_multiplicity() {
        let header = "transaction_id,account_id,side,amount_minor\n";
        let row = "TX-1,AC-1,CREDIT,10\n";
        let mut input = header.to_owned();
        for _ in 0..100 {
            input.push_str(row);
        }
        let report = reconcile_csv(&input).expect("repeated exact rows are valid");
        assert_eq!(report.credits_minor, 10);
        assert_eq!(report.unique_transactions, 1);
        assert_eq!(report.exact_duplicate_rows_ignored, 99);
    }

    #[test]
    fn deterministic_generated_ledgers_match_independent_reference_reducer() {
        // This reference reducer deliberately uses a linear Vec lookup instead of the
        // implementation's HashMap/BTreeMap path. A fixed LCG seed makes every case
        // reproducible while exploring many account/side/amount combinations.
        for case in 0u64..64 {
            let mut seed = 0x5eed_cafe_f00d_u64 ^ case;
            let mut generated: Vec<(String, String, Side, u64)> = Vec::new();
            let count = 20 + (case % 45) as usize;

            for index in 0..count {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let account_id = format!("AC-{:02}", (seed >> 32) % 7);
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let side = if seed & 1 == 0 { Side::Credit } else { Side::Debit };
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let amount = (seed % 1_000_000) + 1;
                generated.push((
                    format!("TX-{case:02}-{index:04}"),
                    account_id,
                    side,
                    amount,
                ));
                if index % 9 == 0 {
                    let duplicate = generated.last().expect("just inserted").clone();
                    generated.push(duplicate);
                }
            }

            let mut csv = String::from("transaction_id,account_id,side,amount_minor\n");
            for (transaction_id, account_id, side, amount) in &generated {
                let side_text = match side {
                    Side::Credit => "CREDIT",
                    Side::Debit => "DEBIT",
                };
                csv.push_str(&format!(
                    "{transaction_id},{account_id},{side_text},{amount}\n"
                ));
            }

            let mut accepted: Vec<(String, String, Side, u64)> = Vec::new();
            let mut duplicate_count = 0u64;
            let mut oracle_accounts: BTreeMap<String, (u64, u64, u64)> = BTreeMap::new();

            for transaction in &generated {
                if let Some(previous) = accepted
                    .iter()
                    .find(|previous| previous.0 == transaction.0)
                {
                    assert_eq!(previous, transaction, "generator only emits exact duplicates");
                    duplicate_count += 1;
                    continue;
                }

                accepted.push(transaction.clone());
                let account = oracle_accounts
                    .entry(transaction.1.clone())
                    .or_insert((0, 0, 0));
                match &transaction.2 {
                    Side::Credit => account.0 += transaction.3,
                    Side::Debit => account.1 += transaction.3,
                }
                account.2 += 1;
            }

            let expected_accounts: Vec<AccountReport> = oracle_accounts
                .into_iter()
                .map(|(account_id, (credits, debits, count))| AccountReport {
                    account_id,
                    credits_minor: credits,
                    debits_minor: debits,
                    net_minor: i128::from(credits) - i128::from(debits),
                    unique_transactions: count,
                })
                .collect();
            let expected_credits: u64 = expected_accounts.iter().map(|a| a.credits_minor).sum();
            let expected_debits: u64 = expected_accounts.iter().map(|a| a.debits_minor).sum();
            let expected_unique: u64 = expected_accounts.iter().map(|a| a.unique_transactions).sum();

            let report = reconcile_csv(&csv).expect("generated ledger should reconcile");
            assert_eq!(report.accounts, expected_accounts, "case {case}");
            assert_eq!(report.credits_minor, expected_credits, "case {case}");
            assert_eq!(report.debits_minor, expected_debits, "case {case}");
            assert_eq!(
                report.net_minor,
                i128::from(expected_credits) - i128::from(expected_debits),
                "case {case}"
            );
            assert_eq!(report.unique_transactions, expected_unique, "case {case}");
            assert_eq!(
                report.exact_duplicate_rows_ignored,
                duplicate_count,
                "case {case}"
            );
        }
    }

    #[test]
    fn malformed_quoting_and_unterminated_fields_are_rejected() {
        for input in [
            "transaction_id,account_id,side,amount_minor\nTX-1,AC-1,CREDIT,1\"x\n",
            "transaction_id,account_id,side,amount_minor\n\"TX-1,AC-1,CREDIT,1\n",
            "transaction_id,account_id,side,amount_minor\nTX-1,AC-1,CREDIT,\"1",
        ] {
            assert!(reconcile_csv(input).is_err(), "malformed CSV should fail: {input:?}");
        }
    }

    #[test]
    fn bare_carriage_return_is_rejected() {
        let input = "transaction_id,account_id,side,amount_minor\rTX-1,AC-1,CREDIT,1";
        assert!(reconcile_csv(input).is_err());
    }
}
