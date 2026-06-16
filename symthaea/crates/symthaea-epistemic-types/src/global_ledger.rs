use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum GlobalClaimStatus {
    Heuristic,
    Formalized,
    Proven,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GlobalClaim {
    pub domain: String,
    pub name: String,
    pub status: GlobalClaimStatus,
    pub formal_proof_path: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GlobalEpistemicLedger {
    pub claims: Vec<GlobalClaim>,
}

impl GlobalEpistemicLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn audit_all(&self) -> bool {
        self.claims.iter().all(|c| match c.status {
            GlobalClaimStatus::Proven => c.formal_proof_path.is_some(),
            _ => true,
        })
    }
}
