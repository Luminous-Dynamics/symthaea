//! Adversarial, synthetic alternatives-assessment corpus.
//!
//! The cases intentionally contain trade-offs and epistemic defects. They are
//! not claims about real materials, chemicals, suppliers, or impacts.

use super::*;
use blake3::Hasher;
use serde::Serialize;
use std::collections::BTreeMap;

/// One frozen benchmark scenario.
#[derive(Debug, Clone)]
pub struct BenchmarkCase {
    /// Stable benchmark identifier.
    pub id: &'static str,
    /// Purpose of the adversarial case.
    pub purpose: &'static str,
    /// Functional requirement.
    pub requirement: FunctionalRequirement,
    /// Candidate pathways in deliberately non-canonical input order.
    pub candidates: Vec<CandidatePathway>,
    /// Incumbent identifier.
    pub incumbent_id: &'static str,
}

fn evidence(
    id: &str,
    source: &str,
    kind: EvidenceKind,
    stance: EvidenceStance,
    confidence: f64,
) -> EvidenceRecord {
    EvidenceRecord {
        id: id.into(),
        kind,
        stance,
        confidence,
        source_id: source.into(),
        scope: "benchmark:functional-unit-v1|global".into(),
        unit: Some("burden-unit".into()),
        as_of: Some("benchmark-v1".into()),
    }
}

fn burdens(
    evidence_ids: &[&str],
    hazard: Interval,
    water: Interval,
) -> BTreeMap<Dimension, BurdenEstimate> {
    Dimension::ALL
        .into_iter()
        .map(|dimension| {
            let interval = match dimension {
                Dimension::Hazard => hazard,
                Dimension::Water => water,
                _ => Interval::point(5.0).unwrap(),
            };
            (
                dimension,
                BurdenEstimate {
                    interval,
                    unit: "burden-unit".into(),
                    scope: "benchmark:functional-unit-v1|global".into(),
                    evidence_ids: evidence_ids.iter().map(|id| (*id).into()).collect(),
                },
            )
        })
        .collect()
}

fn candidate(
    id: &'static str,
    kind: PathwayKind,
    hazard: (f64, f64),
    water: (f64, f64),
    evidence: Vec<EvidenceRecord>,
) -> CandidatePathway {
    let evidence_ids = evidence.iter().map(|e| e.id.as_str()).collect::<Vec<_>>();
    CandidatePathway {
        id: id.into(),
        name: id.into(),
        kind,
        performance: BTreeMap::from([
            ("service_life_years".into(), 12.0),
            ("throughput_per_hour".into(), 120.0),
        ]),
        burdens: burdens(
            &evidence_ids,
            Interval::new(hazard.0, hazard.1).unwrap(),
            Interval::new(water.0, water.1).unwrap(),
        ),
        evidence,
    }
}

/// Canonical synthetic benchmark containing the five fundamental substitution
/// classes: incumbent, direct substitute, process substitute, product redesign,
/// and elimination.
pub fn five_pathway_adversarial_case() -> BenchmarkCase {
    BenchmarkCase {
        id: "industrial-alternatives/five-pathway-v1",
        purpose: "regrettable substitution + epistemic uncertainty + functional alternatives",
        requirement: FunctionalRequirement {
            id: "seal-v1".into(),
            description: "Provide a durable chemical-resistant seal.".into(),
            constraints: BTreeMap::from([
                ("service_life_years".into(), RequirementBound::AtLeast(10.0)),
                ("throughput_per_hour".into(), RequirementBound::AtLeast(100.0)),
            ]),
        },
        candidates: vec![
            // Intentionally shuffled: canonicalization must make output invariant
            // to candidate/evidence insertion order.
            candidate(
                "product-redesign",
                PathwayKind::ProductRedesign,
                (1.5, 2.5),
                (3.0, 5.0),
                vec![
                    evidence("r2", "source-r2", EvidenceKind::Reported, EvidenceStance::Supports, 0.8),
                    evidence("r1", "source-r1", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                ],
            ),
            candidate(
                "incumbent",
                PathwayKind::MaterialSubstitution,
                (9.0, 11.0),
                (9.0, 11.0),
                vec![
                    evidence("i1", "source-incumbent", EvidenceKind::Observed, EvidenceStance::Supports, 0.95),
                    evidence("i2", "source-incumbent-2", EvidenceKind::LifecycleAssessed, EvidenceStance::Supports, 0.9),
                ],
            ),
            candidate(
                "elimination",
                PathwayKind::Elimination,
                (0.5, 2.0),
                (0.5, 2.0),
                vec![
                    evidence("e1", "hypothesis", EvidenceKind::Hypothesis, EvidenceStance::Supports, 0.2),
                ],
            ),
            candidate(
                "direct-substitute",
                PathwayKind::MaterialSubstitution,
                (2.0, 3.0),
                (28.0, 32.0),
                vec![
                    evidence("d3", "source-lca", EvidenceKind::LifecycleAssessed, EvidenceStance::Supports, 0.9),
                    evidence("d1", "source-measure", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                    evidence("d2", "source-report", EvidenceKind::Reported, EvidenceStance::Supports, 0.9),
                ],
            ),
            candidate(
                "process-substitute",
                PathwayKind::ProcessSubstitution,
                (3.0, 4.0),
                (3.0, 4.0),
                vec![
                    evidence("p1", "source-process", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                    evidence("p2", "source-process-2", EvidenceKind::Reported, EvidenceStance::Supports, 0.9),
                    evidence("p3", "source-process-lca", EvidenceKind::LifecycleAssessed, EvidenceStance::Supports, 0.9),
                ],
            ),
        ],
        incumbent_id: "incumbent",
    }
}

/// Frozen identity manifest for a benchmark case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BenchmarkManifest {
    /// Manifest schema version.
    pub schema_version: u16,
    /// Benchmark case identifier.
    pub case_id: String,
    /// Assessment algorithm version.
    pub algorithm_version: String,
    /// Functional requirement identifier.
    pub requirement_id: String,
    /// Candidate identifiers in canonical order.
    pub candidate_ids: Vec<String>,
    /// Incumbent identifier.
    pub incumbent_id: String,
}

impl BenchmarkCase {
    /// Build the canonical manifest identity for this case.
    pub fn manifest(&self) -> BenchmarkManifest {
        let mut candidate_ids = self
            .candidates
            .iter()
            .map(|candidate| candidate.id.clone())
            .collect::<Vec<_>>();
        candidate_ids.sort();

        BenchmarkManifest {
            schema_version: SCHEMA_VERSION,
            case_id: self.id.into(),
            algorithm_version: ALGORITHM_VERSION.into(),
            requirement_id: self.requirement.id.clone(),
            candidate_ids,
            incumbent_id: self.incumbent_id.into(),
        }
    }

    /// Compute a stable BLAKE3 identity for the frozen benchmark topology.
    pub fn manifest_hash(&self) -> String {
        let bytes = serde_json::to_vec(&self.manifest()).expect("manifest is serializable");
        let mut hasher = Hasher::new();
        hasher.update(&bytes);
        hasher.finalize().to_hex().to_string()
    }
}

/// Run every case with the deterministic engine.
pub fn run_case(case: &BenchmarkCase) -> Result<AssessmentResult, AssessmentError> {
    AlternativesEngine.assess(&case.requirement, &case.candidates, Some(case.incumbent_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_pathway_case_contains_all_pathway_classes() {
        let case = five_pathway_adversarial_case();
        let kinds = case
            .candidates
            .iter()
            .map(|candidate| candidate.kind)
            .collect::<Vec<_>>();

        assert!(kinds.contains(&PathwayKind::MaterialSubstitution));
        assert!(kinds.contains(&PathwayKind::ProcessSubstitution));
        assert!(kinds.contains(&PathwayKind::ProductRedesign));
        assert!(kinds.contains(&PathwayKind::Elimination));
        assert_eq!(case.candidates.len(), 5);
    }

    #[test]
    fn five_pathway_case_flags_regrettable_direct_substitute() {
        let result = run_case(&five_pathway_adversarial_case()).unwrap();
        let direct = result
            .burden_transfers
            .iter()
            .find(|transfer| transfer.candidate_id == "direct-substitute")
            .unwrap();

        assert!(direct.is_regrettable_substitution());
        assert!(direct.clearly_better.contains(&Dimension::Hazard));
        assert!(direct.clearly_worse.contains(&Dimension::Water));
    }

    #[test]
    fn five_pathway_case_keeps_elimination_as_hypothesis() {
        let result = run_case(&five_pathway_adversarial_case()).unwrap();
        let elimination = result
            .candidates
            .iter()
            .find(|candidate| candidate.candidate_id == "elimination")
            .unwrap();

        assert_eq!(elimination.qualification, QualificationState::Hypothesis);
        assert!(!elimination.frontier_blocked);
    }

    #[test]
    fn five_pathway_case_is_reproducible() {
        let case = five_pathway_adversarial_case();
        let mut reversed = case.clone();
        reversed.candidates.reverse();
        reversed
            .candidates
            .iter_mut()
            .for_each(|candidate| candidate.evidence.reverse());

        let first = run_case(&case).unwrap();
        let second = run_case(&reversed).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.receipt.schema_version, SCHEMA_VERSION);
        assert_eq!(first.receipt.algorithm_version, ALGORITHM_VERSION);
    }

    #[test]
    fn manifest_identity_is_input_order_independent() {
        let case = five_pathway_adversarial_case();
        let mut reversed = case.clone();
        reversed.candidates.reverse();

        assert_eq!(case.manifest(), reversed.manifest());
        assert_eq!(case.manifest_hash(), reversed.manifest_hash());
        assert!(!case.manifest_hash().is_empty());
    }

    #[test]
    fn five_pathway_case_does_not_make_regrettable_substitution_disappear() {
        let result = run_case(&five_pathway_adversarial_case()).unwrap();
        let direct = result
            .candidates
            .iter()
            .find(|candidate| candidate.candidate_id == "direct-substitute")
            .unwrap();

        assert!(direct.evidence_conflict == false);
        assert_eq!(direct.qualification, QualificationState::LifecycleQualified);
    }
    }
}
