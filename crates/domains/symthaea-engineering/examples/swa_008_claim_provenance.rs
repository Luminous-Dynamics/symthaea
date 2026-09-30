//! SWA-008: claim-level provenance and contestability.
//!
//! Claims name the exact evidence they rely on and the dependency closure that
//! can make them stale. Support, contradiction, and qualification remain
//! distinct. Missing provenance never becomes Valid, and claims never authorize.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
enum DependencyKind {
    Model,
    Parameters,
    Solver,
    Scenario,
    Dataset,
    UncertaintyModel,
    ContextOfUse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct DependencyRevision {
    kind: DependencyKind,
    id: &'static str,
    revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct DependencyFingerprint {
    model: DependencyRevision,
    parameters: DependencyRevision,
    solver: DependencyRevision,
    scenario: DependencyRevision,
    dataset: DependencyRevision,
    uncertainty: DependencyRevision,
    context: DependencyRevision,
}

impl DependencyFingerprint {
    fn revision(model: u64, parameters: u64, solver: u64, scenario: u64, dataset: u64, uncertainty: u64, context: u64) -> Self {
        Self {
            model: DependencyRevision { kind: DependencyKind::Model, id: "model", revision: model },
            parameters: DependencyRevision { kind: DependencyKind::Parameters, id: "parameters", revision: parameters },
            solver: DependencyRevision { kind: DependencyKind::Solver, id: "solver", revision: solver },
            scenario: DependencyRevision { kind: DependencyKind::Scenario, id: "scenario", revision: scenario },
            dataset: DependencyRevision { kind: DependencyKind::Dataset, id: "dataset", revision: dataset },
            uncertainty: DependencyRevision { kind: DependencyKind::UncertaintyModel, id: "uncertainty", revision: uncertainty },
            context: DependencyRevision { kind: DependencyKind::ContextOfUse, id: "context", revision: context },
        }
    }

    fn get(self, kind: DependencyKind) -> DependencyRevision {
        match kind {
            DependencyKind::Model => self.model,
            DependencyKind::Parameters => self.parameters,
            DependencyKind::Solver => self.solver,
            DependencyKind::Scenario => self.scenario,
            DependencyKind::Dataset => self.dataset,
            DependencyKind::UncertaintyModel => self.uncertainty,
            DependencyKind::ContextOfUse => self.context,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct ValidationEvidence {
    id: &'static str,
    dependencies: DependencyFingerprint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum ClaimKind {
    PredictionValidated,
    UncertaintyCharacterized,
    ApplicabilitySupported,
}

impl ClaimKind {
    fn closure(self) -> &'static [DependencyKind] {
        match self {
            Self::PredictionValidated => &[
                DependencyKind::Model,
                DependencyKind::Parameters,
                DependencyKind::Solver,
                DependencyKind::Scenario,
                DependencyKind::Dataset,
            ],
            Self::UncertaintyCharacterized => &[
                DependencyKind::Model,
                DependencyKind::Parameters,
                DependencyKind::Solver,
                DependencyKind::Dataset,
                DependencyKind::UncertaintyModel,
            ],
            Self::ApplicabilitySupported => &[
                DependencyKind::Model,
                DependencyKind::Scenario,
                DependencyKind::Dataset,
                DependencyKind::ContextOfUse,
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum EvidenceRelation {
    Supports,
    Contradicts,
    Qualifies,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct EvidenceLink {
    evidence_id: &'static str,
    relation: EvidenceRelation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum ClaimStatus {
    Valid,
    Stale,
    Unknown,
    Unsupported,
    Contested,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct EvidenceClaim {
    id: &'static str,
    kind: ClaimKind,
    evidence: &'static [EvidenceLink],
    dependencies: DependencyFingerprint,
    is_authorization: bool,
}

fn status(claim: &EvidenceClaim, current: DependencyFingerprint) -> ClaimStatus {
    let mut changed = false;

    for &kind in claim.kind.closure() {
        let historical = claim.dependencies.get(kind);
        let now = current.get(kind);

        if historical.id != now.id || historical.kind != now.kind {
            return ClaimStatus::Unknown;
        }

        if historical.revision != now.revision {
            changed = true;
        }
    }

    if claim.evidence.is_empty() {
        return ClaimStatus::Unsupported;
    }

    let has_support = claim.evidence.iter().any(|link| link.relation == EvidenceRelation::Supports);
    let has_contradiction = claim.evidence.iter().any(|link| link.relation == EvidenceRelation::Contradicts);

    if has_support && has_contradiction {
        ClaimStatus::Contested
    } else if changed {
        ClaimStatus::Stale
    } else if has_support {
        ClaimStatus::Valid
    } else {
        ClaimStatus::Unsupported
    }
}

fn main() {
    let v1 = DependencyFingerprint::revision(1, 1, 1, 1, 1, 1, 1);
    let solver_only = DependencyFingerprint::revision(1, 1, 2, 1, 1, 1, 1);

    let validation = ValidationEvidence {
        id: "validation-001",
        dependencies: v1,
    };

    let claim = EvidenceClaim {
        id: "claim-001",
        kind: ClaimKind::PredictionValidated,
        evidence: &[EvidenceLink {
            evidence_id: validation.id,
            relation: EvidenceRelation::Supports,
        }],
        dependencies: validation.dependencies,
        is_authorization: false,
    };

    assert_eq!(status(&claim, solver_only), ClaimStatus::Stale);
    assert!(!claim.is_authorization);

    let resilience_like_applicability = EvidenceClaim {
        id: "claim-002",
        kind: ClaimKind::ApplicabilitySupported,
        ..claim
    };
    assert_eq!(status(&resilience_like_applicability, solver_only), ClaimStatus::Valid);

    let contested = EvidenceClaim {
        evidence: &[
            EvidenceLink { evidence_id: "validation-001", relation: EvidenceRelation::Supports },
            EvidenceLink { evidence_id: "validation-002", relation: EvidenceRelation::Contradicts },
        ],
        ..claim
    };
    assert_eq!(status(&contested, v1), ClaimStatus::Contested);

    println!("{}", serde_json::to_string(&claim).expect("claim serializes"));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim() -> EvidenceClaim {
        EvidenceClaim {
            id: "claim-test",
            kind: ClaimKind::PredictionValidated,
            evidence: &[EvidenceLink {
                evidence_id: "validation-test",
                relation: EvidenceRelation::Supports,
            }],
            dependencies: DependencyFingerprint::revision(1, 1, 1, 1, 1, 1, 1),
            is_authorization: false,
        }
    }

    #[test]
    fn missing_evidence_is_unsupported() {
        let mut value = claim();
        value.evidence = &[];
        assert_eq!(status(&value, value.dependencies), ClaimStatus::Unsupported);
    }

    #[test]
    fn contradictory_evidence_is_contested_not_resolved() {
        let value = EvidenceClaim {
            evidence: &[
                EvidenceLink { evidence_id: "support", relation: EvidenceRelation::Supports },
                EvidenceLink { evidence_id: "counter", relation: EvidenceRelation::Contradicts },
            ],
            ..claim()
        };
        assert_eq!(status(&value, value.dependencies), ClaimStatus::Contested);
    }

    #[test]
    fn qualification_is_not_contradiction() {
        let value = EvidenceClaim {
            evidence: &[
                EvidenceLink { evidence_id: "support", relation: EvidenceRelation::Supports },
                EvidenceLink { evidence_id: "boundary", relation: EvidenceRelation::Qualifies },
            ],
            ..claim()
        };
        assert_eq!(status(&value, value.dependencies), ClaimStatus::Valid);
    }

    #[test]
    fn unrelated_dependency_change_does_not_stale_claim() {
        let value = EvidenceClaim {
            kind: ClaimKind::ApplicabilitySupported,
            ..claim()
        };
        let current = DependencyFingerprint::revision(1, 1, 2, 1, 1, 1, 1);
        assert_eq!(status(&value, current), ClaimStatus::Valid);
    }

    #[test]
    fn dependency_change_stales_claim_without_mutating_history() {
        let value = claim();
        let current = DependencyFingerprint::revision(1, 2, 1, 1, 1, 1, 1);
        assert_eq!(status(&value, current), ClaimStatus::Stale);
        assert_eq!(value.dependencies.parameters.revision, 1);
    }

    #[test]
    fn incomplete_provenance_is_unknown() {
        let value = claim();
        let current = DependencyFingerprint {
            model: DependencyRevision { kind: DependencyKind::Model, id: "unknown", revision: 1 },
            ..value.dependencies
        };
        assert_eq!(status(&value, current), ClaimStatus::Unknown);
    }

    #[test]
    fn claims_cannot_be_authorization() {
        assert!(!claim().is_authorization);
    }
}
