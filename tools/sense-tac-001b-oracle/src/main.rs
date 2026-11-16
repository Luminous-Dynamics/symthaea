use sense_tac_001b_oracle::validate_tracked;
use std::process::ExitCode;

fn main() -> ExitCode {
    match validate_tracked() {
        Ok(report) => {
            println!("suite=sense_tac_001b_synthetic_fault_oracle_v0_1");
            println!("fixtures={}", report.fixture_count);
            println!("unique_fault_kinds={}", report.unique_fault_kinds);
            println!("known_answers_verified={}", report.cases_verified);
            println!("result=all_known_answers_matched");
            println!("claim_ceiling=synthetic_fault_and_representation_contract_only");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("result=invalid");
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
