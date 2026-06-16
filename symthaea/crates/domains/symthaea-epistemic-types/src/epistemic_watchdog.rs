use crate::global_ledger::{GlobalClaimStatus, GlobalEpistemicLedger};
use std::process::Command;

pub struct EpistemicWatchdog;

impl EpistemicWatchdog {
    /// Audits the entire workspace for claim consistency.
    /// Fails if any 'Proven' claim lacks a valid proof file.
    pub fn audit(ledger: &GlobalEpistemicLedger) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        for claim in &ledger.claims {
            if claim.status == GlobalClaimStatus::Proven {
                match &claim.formal_proof_path {
                    Some(path) => {
                        if !std::path::Path::new(path).exists() {
                            errors.push(format!("Proof missing for '{}': {}", claim.name, path));
                        } else if !Self::verify_formal_proof(path) {
                            errors.push(format!("Proof failed verification for '{}': {}", claim.name, path));
                        }
                    }
                    None => {
                        errors.push(format!("Claim '{}' is marked Proven but lacks proof path.", claim.name));
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn verify_formal_proof(path: &str) -> bool {
        // Invokes Lean 4 compiler (lake) for formal verification
        let output = Command::new("lake")
            .arg("lean")
            .arg(path)
            .status();
        
        output.map(|s| s.success()).unwrap_or(false)
    }
}
