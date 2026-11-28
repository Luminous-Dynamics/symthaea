use std::process::ExitCode;
use typed_research_state_contract::validate_examples;

fn main() -> ExitCode {
    match validate_examples() {
        Ok(report) => {
            println!(
                "PASS: {} ResearchState examples, {} claims, {} evidence artifacts, {} claim-source relations",
                report.example_count,
                report.claim_count,
                report.evidence_count,
                report.relation_count
            );
            ExitCode::SUCCESS
        }
        Err(issues) => {
            for issue in issues {
                eprintln!("{}: {}", issue.path, issue.message);
            }
            ExitCode::FAILURE
        }
    }
}
