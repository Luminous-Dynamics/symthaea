use plasma_active_candidate_audit::{CandidateRun, AuditDisposition, audit_candidate};
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1) else {
        eprintln!("usage: plasma-active-candidate-audit <candidate.json>");
        return ExitCode::from(2);
    };
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("could not read candidate: {error}");
            return ExitCode::from(2);
        }
    };
    let candidate: CandidateRun = match serde_json::from_slice(&bytes) {
        Ok(candidate) => candidate,
        Err(error) => {
            eprintln!("candidate schema parse failed: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match audit_candidate(&candidate) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("candidate audit failed: {error}");
            return ExitCode::from(2);
        }
    };
    match serde_json::to_string_pretty(&report) {
        Ok(json) => {
            println!("{json}");
            match report.body.disposition {
                AuditDisposition::Admissible => ExitCode::SUCCESS,
                AuditDisposition::Rejected => ExitCode::from(1),
            }
        }
        Err(error) => {
            eprintln!("could not serialize candidate audit report: {error}");
            ExitCode::from(2)
        }
    }
}
