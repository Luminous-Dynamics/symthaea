use crate::global_ledger::{GlobalClaimStatus, GlobalEpistemicLedger};
use std::path::Path;
use std::process::Command;

pub struct FormalWatchdog;

impl FormalWatchdog {
    /// Audits the ledger against actual Lean 4 proof repository state.
    pub fn audit_claims(ledger: &GlobalEpistemicLedger) -> Vec<String> {
        let mut violations = Vec::new();

        for claim in &ledger.claims {
            if claim.status == GlobalClaimStatus::Proven {
                if let Some(path) = &claim.formal_proof_path {
                    if !Path::new(path).exists() {
                        violations.push(format!(
                            "Proven claim '{}' missing proof file at: {}",
                            claim.name, path
                        ));
                    } else {
                        // verify the file compiles in lean 4
                        if !Self::check_lean_compilation(path) {
                            violations.push(format!(
                                "Proven claim '{}' failed formal verification at: {}",
                                claim.name, path
                            ));
                        }
                    }
                } else {
                    violations.push(format!(
                        "Proven claim '{}' has no associated proof path.",
                        claim.name
                    ));
                }
            }
        }
        violations
    }

    fn check_lean_compilation(path: &str) -> bool {
        // Assume lake is used for compilation/verification
        let status = Command::new("lake").arg("lean").arg(path).status();

        match status {
            Ok(s) => s.success(),
            Err(_) => false,
        }
    }
}
