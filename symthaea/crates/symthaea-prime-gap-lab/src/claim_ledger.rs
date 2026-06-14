#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum ClaimStatus {
    Proven,
    Conditional,
    Heuristic,
    Computational,
    Refuted,
    Open,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum EvidenceKind {
    FormalProof,
    Literature,
    Computation,
    HeuristicModel,
    Counterexample,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum ClaimKind {
    Conjecture,
    Lemma,
    Theorem,
    Heuristic,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum ClaimScope {
    Global,
    Narrow,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Dependency {
    pub description: String,
    pub source: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Claim {
    pub name: String,
    pub status: ClaimStatus,
    pub evidence: EvidenceKind,
    pub kind: ClaimKind,
    pub scope: ClaimScope,
    pub assumptions: Vec<String>,
    pub dependencies: Vec<Dependency>,
    pub caveats: Vec<String>,
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ClaimLedger {
    pub claims: Vec<Claim>,
    pub non_claims: Vec<String>,
}

impl ClaimLedger {
    pub fn new() -> Self {
        Self {
            claims: Vec::new(),
            non_claims: vec![
                "Does not prove twin primes.".to_string(),
                "Does not solve the parity problem.".to_string(),
                "Does not reproduce Zhang/Maynard-Tao.".to_string(),
                "Does not formalize analytic number theory.".to_string(),
                "Does not integrate into symthaea-core.".to_string(),
            ],
        }
    }

    pub fn add_claim(&mut self, claim: Claim) -> Result<(), String> {
        if claim.status != ClaimStatus::Proven && claim.caveats.is_empty() {
            return Err("Non-proven claims must have caveats.".to_string());
        }
        self.claims.push(claim);
        Ok(())
    }

    pub fn open_conjecture(name: &str, caveats: Vec<String>) -> Claim {
        Claim {
            name: name.to_string(),
            status: ClaimStatus::Open,
            evidence: EvidenceKind::Literature,
            kind: ClaimKind::Conjecture,
            scope: ClaimScope::Global,
            assumptions: vec![],
            dependencies: vec![],
            caveats,
        }
    }

    pub fn heuristic_claim(name: &str, caveats: Vec<String>) -> Claim {
        Claim {
            name: name.to_string(),
            status: ClaimStatus::Heuristic,
            evidence: EvidenceKind::HeuristicModel,
            kind: ClaimKind::Heuristic,
            scope: ClaimScope::Global,
            assumptions: vec![],
            dependencies: vec![],
            caveats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claim_ledger_caveats() {
        let mut ledger = ClaimLedger::new();
        let claim = Claim {
            name: "Unproven".to_string(),
            status: ClaimStatus::Open,
            evidence: EvidenceKind::Literature,
            kind: ClaimKind::Conjecture,
            scope: ClaimScope::Global,
            assumptions: vec![],
            dependencies: vec![],
            caveats: vec![],
        };
        assert!(ledger.add_claim(claim).is_err());
    }
}
