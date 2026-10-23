//! Canonical normalization of heterogeneous scientific-agent proposals.
//!
//! Normalization makes proposals comparable without treating the normalized
//! representation as evidence. The original payload digest remains attached.

use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalClaim {
    pub claim_id: String,
    pub statement: String,
    pub mechanism_id: String,
    pub assumptions: Vec<String>,
    pub predictions: Vec<String>,
    pub falsifiers: Vec<String>,
    pub required_evidence: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedClaim {
    pub claim_id: String,
    pub statement: String,
    pub mechanism_id: String,
    pub assumptions: Vec<String>,
    pub predictions: Vec<String>,
    pub falsifiers: Vec<String>,
    pub required_evidence: BTreeSet<String>,
    pub source_payload_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationError {
    EmptyField(&'static str),
    EmptyCollection(&'static str),
    MissingEvidenceRequirement,
}

pub fn normalize(
    claim: &ProposalClaim,
    source_payload_digest: impl Into<String>,
) -> Result<NormalizedClaim, NormalizationError> {
    let digest = source_payload_digest.into();
    if claim.claim_id.trim().is_empty() {
        return Err(NormalizationError::EmptyField("claim_id"));
    }
    if claim.statement.trim().is_empty() {
        return Err(NormalizationError::EmptyField("statement"));
    }
    if claim.mechanism_id.trim().is_empty() {
        return Err(NormalizationError::EmptyField("mechanism_id"));
    }
    if digest.trim().is_empty() {
        return Err(NormalizationError::EmptyField("source_payload_digest"));
    }
    if claim.assumptions.is_empty() {
        return Err(NormalizationError::EmptyCollection("assumptions"));
    }
    if claim.predictions.is_empty() {
        return Err(NormalizationError::EmptyCollection("predictions"));
    }
    if claim.falsifiers.is_empty() {
        return Err(NormalizationError::EmptyCollection("falsifiers"));
    }
    if claim.required_evidence.is_empty() {
        return Err(NormalizationError::MissingEvidenceRequirement);
    }

    Ok(NormalizedClaim {
        claim_id: claim.claim_id.clone(),
        statement: claim.statement.clone(),
        mechanism_id: claim.mechanism_id.clone(),
        assumptions: claim.assumptions.clone(),
        predictions: claim.predictions.clone(),
        falsifiers: claim.falsifiers.clone(),
        required_evidence: claim.required_evidence.clone(),
        source_payload_digest: digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim() -> ProposalClaim {
        ProposalClaim {
            claim_id: "claim-1".into(),
            statement: "mechanism may explain observation".into(),
            mechanism_id: "mechanism-1".into(),
            assumptions: vec!["assumption-1".into()],
            predictions: vec!["prediction-1".into()],
            falsifiers: vec!["falsifier-1".into()],
            required_evidence: ["ExternalExperimentalObservation".into()].into_iter().collect(),
        }
    }

    #[test]
    fn normalizes_without_losing_source_identity() {
        let n = normalize(&claim(), "sha256:source").unwrap();
        assert_eq!(n.source_payload_digest, "sha256:source");
        assert_eq!(n.predictions, vec!["prediction-1"]);
    }

    #[test]
    fn rejects_claim_without_falsifier() {
        let mut c = claim();
        c.falsifiers.clear();
        assert_eq!(normalize(&c, "sha256:x"), Err(NormalizationError::EmptyCollection("falsifiers")));
    }

    #[test]
    fn rejects_claim_without_evidence_requirement() {
        let mut c = claim();
        c.required_evidence.clear();
        assert_eq!(normalize(&c, "sha256:x"), Err(NormalizationError::MissingEvidenceRequirement));
    }

    #[test]
    fn rejects_missing_source_identity() {
        assert_eq!(normalize(&claim(), ""), Err(NormalizationError::EmptyField("source_payload_digest")));
    }

    #[test]
    fn rejects_empty_prediction_set() {
        let mut c = claim();
        c.predictions.clear();
        assert_eq!(normalize(&c, "sha256:x"), Err(NormalizationError::EmptyCollection("predictions")));
    }
}
