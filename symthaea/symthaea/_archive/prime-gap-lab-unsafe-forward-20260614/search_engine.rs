use crate::claim_ledger::{Claim, ClaimKind, ClaimScope, ClaimStatus, EvidenceKind};
use crate::hardy_littlewood::calculate_singular_series;
use crate::tuples::PrimeTuple;

pub struct SearchEngine {
    pub max_k: usize,
    pub width: u64,
}

impl SearchEngine {
    pub fn new(max_k: usize, width: u64) -> Self {
        Self { max_k, width }
    }

    pub fn run_proof_search(&self, ledger: &mut crate::claim_ledger::ClaimLedger) {
        for k in 2..=self.max_k {
            let candidates = crate::tuples::enumerate_admissible_tuples(k, self.width);
            for t in candidates {
                let score = calculate_singular_series(&t.elements, 100);

                // Only consider high-probability candidates for formal verification
                if score > 1.0 {
                    println!("Checking candidate: {:?}", t.elements);
                    // Automate call to formal verification pipeline
                    let is_provable = self.verify_formally(&t.elements);

                    if is_provable {
                        let claim = Claim {
                            name: format!("Tuple_{:?}_Admissible", t.elements),
                            status: ClaimStatus::Proven,
                            evidence: EvidenceKind::FormalProof,
                            kind: ClaimKind::Theorem,
                            scope: ClaimScope::Narrow,
                            assumptions: vec![],
                            dependencies: vec![],
                            caveats: vec![],
                        };
                        let _ = ledger.add_claim(claim);
                    }
                }
            }
        }
    }

    fn verify_formally(&self, tuple: &[u64]) -> bool {
        // Interface with verify_candidate.py via command-line call
        let output = std::process::Command::new("python3")
            .arg("crates/symthaea-prime-gap-lab/scripts/verify_candidate.py")
            .arg(format!("{:?}", tuple))
            .output();

        match output {
            Ok(o) => o.status.success(),
            Err(_) => false,
        }
    }
}
