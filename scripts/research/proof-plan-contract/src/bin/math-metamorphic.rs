use proof_plan_contract::metamorphic::{parse_dataset, validation_receipts};
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: math-metamorphic <dataset.json>");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("usage: math-metamorphic <dataset.json>");
        return ExitCode::from(2);
    }

    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            return ExitCode::from(1);
        }
    };
    let dataset = match parse_dataset(&input) {
        Ok(dataset) => dataset,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(1);
        }
    };
    let receipts = match validation_receipts(&dataset) {
        Ok(receipts) => receipts,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(1);
        }
    };

    for receipt in &receipts {
        println!(
            "MATH_METAMORPHIC_PAIR_VALID pair_id={} class={:?} relation={:?}",
            receipt.pair_id, receipt.transformation_class, receipt.oracle_relation
        );
    }
    println!("MATH_METAMORPHIC_DATASET_VALID pairs={}", receipts.len());
    ExitCode::SUCCESS
}
