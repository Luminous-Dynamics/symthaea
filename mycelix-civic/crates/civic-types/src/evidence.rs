//! Evidence types shared across justice and media domains

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

/// Evidence record used by both justice and media fact-checking
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Evidence {
    pub id: String,
    pub case_id: String,
    pub content_hash: String,
    pub evidence_type: EvidenceType,
    pub submitted_by: String,
    pub submitted_at: Timestamp,
    pub description: String,
}

/// Types of evidence
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EvidenceType {
    Document,
    Testimony,
    Data,
    Media,
    SourceCitation,
}

/// Custody chain event for evidence integrity
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CustodyEvent {
    pub evidence_hash: String,
    pub action: CustodyAction,
    pub actor: String,
    pub timestamp: Timestamp,
    pub notes: String,
}

/// Actions in the custody chain
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CustodyAction {
    Submitted,
    Verified,
    Challenged,
    Sealed,
    Released,
}

/// Verification result for evidence
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EvidenceVerification {
    pub evidence_hash: String,
    pub verifier_did: String,
    pub verdict: Verdict,
    pub confidence: f64,
    pub timestamp: Timestamp,
}

/// Verdict for evidence verification or fact-checking
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Verdict {
    Verified,
    Unverified,
    Disputed,
    False,
}
