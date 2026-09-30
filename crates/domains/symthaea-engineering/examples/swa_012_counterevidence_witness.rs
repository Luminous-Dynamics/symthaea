//! SWA-012 — independently reproducible counterevidence witnesses.
//!
//! A claim may have supporting, qualifying, or contradicting evidence. This
//! fixture deliberately preserves all three relations without resolving them
//! into a confidence score or authority decision.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct WitnessRef {
    pub witness_id: String,
    pub input_fingerprint: String,
    pub artifact_fingerprint: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EvidenceRelation {
    Supports,
    Qualifies,
    Contradicts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CounterevidenceLink {
    pub relation: EvidenceRelation,
    pub witness: WitnessRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ClaimState {
    Supported,
    Qualified,
    Contested,
    Unsupported,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CounterevidenceRecord {
    pub claim_id: String,
    pub links: Vec<CounterevidenceLink>,
}

impl CounterevidenceRecord {
    pub fn canonicalize(mut self) -> Self {
        self.links.sort_by(|a, b| {
            relation_key(&a.relation)
                .cmp(relation_key(&b.relation))
                .then_with(|| a.witness.witness_id.cmp(&b.witness.witness_id))
        });
        self
    }

    pub fn state(&self) -> ClaimState {
        let has_support = self.links.iter().any(|l| matches!(l.relation, EvidenceRelation::Supports));
        let has_qualification = self.links.iter().any(|l| matches!(l.relation, EvidenceRelation::Qualifies));
        let has_contradiction = self.links.iter().any(|l| matches!(l.relation, EvidenceRelation::Contradicts));

        if self.links.is_empty() {
            ClaimState::Unsupported
        } else if has_contradiction {
            ClaimState::Contested
        } else if has_support && has_qualification {
            ClaimState::Qualified
        } else if has_support {
            ClaimState::Supported
        } else {
            ClaimState::Unknown
        }
    }

    pub fn witness_ids(&self) -> Vec<&str> {
        self.links.iter().map(|l| l.witness.witness_id.as_str()).collect()
    }

    pub fn has_independent_witnesses(&self) -> bool {
        self.links
            .iter()
            .map(|l| &l.witness.witness_id)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == self.links.len()
    }
}

fn relation_key(relation: &EvidenceRelation) -> u8 {
    match relation {
        EvidenceRelation::Supports => 0,
        EvidenceRelation::Qualifies => 1,
        EvidenceRelation::Contradicts => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn witness(id: &str) -> WitnessRef {
        WitnessRef {
            witness_id: id.into(),
            input_fingerprint: format!("input-{id}"),
            artifact_fingerprint: format!("artifact-{id}"),
        }
    }

    #[test]
    fn support_is_distinct_from_qualification() {
        let record = CounterevidenceRecord {
            claim_id: "claim-001".into(),
            links: vec![CounterevidenceLink {
                relation: EvidenceRelation::Qualifies,
                witness: witness("w-qualify"),
            }],
        };
        assert_eq!(record.state(), ClaimState::Unknown);
    }

    #[test]
    fn support_and_qualification_produce_qualified_state() {
        let record = CounterevidenceRecord {
            claim_id: "claim-001".into(),
            links: vec![
                CounterevidenceLink { relation: EvidenceRelation::Supports, witness: witness("w-support") },
                CounterevidenceLink { relation: EvidenceRelation::Qualifies, witness: witness("w-qualify") },
            ],
        };
        assert_eq!(record.state(), ClaimState::Qualified);
    }

    #[test]
    fn contradiction_is_preserved_without_resolution() {
        let record = CounterevidenceRecord {
            claim_id: "claim-001".into(),
            links: vec![
                CounterevidenceLink { relation: EvidenceRelation::Supports, witness: witness("w-support") },
                CounterevidenceLink { relation: EvidenceRelation::Contradicts, witness: witness("w-counter") },
            ],
        };
        assert_eq!(record.state(), ClaimState::Contested);
        assert_eq!(record.witness_ids(), vec!["w-support", "w-counter"]);
    }

    #[test]
    fn witnesses_must_be_independently_identified() {
        let record = CounterevidenceRecord {
            claim_id: "claim-001".into(),
            links: vec![
                CounterevidenceLink { relation: EvidenceRelation::Supports, witness: witness("w-1") },
                CounterevidenceLink { relation: EvidenceRelation::Contradicts, witness: witness("w-1") },
            ],
        };
        assert!(!record.has_independent_witnesses());
    }

    #[test]
    fn canonicalization_is_deterministic() {
        let a = CounterevidenceRecord {
            claim_id: "claim-001".into(),
            links: vec![
                CounterevidenceLink { relation: EvidenceRelation::Contradicts, witness: witness("w-c") },
                CounterevidenceLink { relation: EvidenceRelation::Supports, witness: witness("w-a") },
            ],
        };
        let b = CounterevidenceRecord {
            claim_id: "claim-001".into(),
            links: vec![
                CounterevidenceLink { relation: EvidenceRelation::Supports, witness: witness("w-a") },
                CounterevidenceLink { relation: EvidenceRelation::Contradicts, witness: witness("w-c") },
            ],
        };
        assert_eq!(a.canonicalize(), b.canonicalize());
    }

    #[test]
    fn no_link_grants_authority() {
        let record = CounterevidenceRecord { claim_id: "claim-001".into(), links: vec![] };
        assert_eq!(record.state(), ClaimState::Unsupported);
    }
}
