use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const INPUT: &str = include_str!("../../examples/ledger-reconciliation/input.csv");
const EXPECTED: &str = include_str!("../../examples/ledger-reconciliation/expected.json");
const CONFLICT: &str =
    include_str!("../../examples/ledger-reconciliation/conflicting-duplicate.csv");

fn run_cli(input: &str) -> (Output, PathBuf) {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "vleb-ledger-reconcile-{}-{sequence}.csv",
        std::process::id()
    ));
    fs::write(&path, input).expect("write temporary CSV input");

    let output = Command::new(env!("CARGO_BIN_EXE_ledger-reconcile"))
        .arg("--input")
        .arg(&path)
        .output()
        .expect("launch ledger-reconcile CLI");

    (output, path)
}

#[test]
fn cli_emits_exact_golden_output_for_fixture() {
    let (output, path) = run_cli(INPUT);
    let _ = fs::remove_file(path);

    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), EXPECTED);
    assert!(output.stderr.is_empty());
}

#[test]
fn conflicting_duplicate_exits_nonzero_without_partial_stdout() {
    let (output, path) = run_cli(CONFLICT);
    let _ = fs::remove_file(path);

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("conflicting duplicate transaction_id")
    );
}

#[test]
fn late_invalid_row_exits_nonzero_without_partial_stdout() {
    let input = concat!(
        "transaction_id,account_id,side,amount_minor\n",
        "TX-1,AC-1,CREDIT,10\n",
        "TX-2,AC-1,CREDIT,not-a-number\n"
    );
    let (output, path) = run_cli(input);
    let _ = fs::remove_file(path);

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn missing_input_argument_is_rejected() {
    let output = Command::new(env!("CARGO_BIN_EXE_ledger-reconcile"))
        .output()
        .expect("launch ledger-reconcile CLI");

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}
