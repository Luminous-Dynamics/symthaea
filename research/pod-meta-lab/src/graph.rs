//! Synthetic AND/OR proof graph generator with a private source-truth oracle.
//! The planner view contains assertions and provenance metadata, never oracle labels.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Polarity { Supports, Refutes }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceIntegrity { Verified, Unverified, Invalid }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceLifecycle { Active, Retracted }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSplit { Development, Calibration, Test }

impl DatasetSplit {
    fn label(self) -> &'static str {
        match self {
            Self::Development => "dev",
            Self::Calibration => "cal",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryProfile {
    pub valid_at: u64,
    pub known_at: u64,
    pub required_scope: String,
    pub required_authority: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRevision {
    pub id: String,
    pub source_id: String,
    pub source_family_id: String,
    pub independence_group_id: String,
    pub subject_claim_id: String,
    pub asserted_polarity: Polarity,
    pub valid_from: u64,
    pub valid_until: Option<u64>,
    pub known_from: u64,
    pub known_until: Option<u64>,
    pub scope_id: String,
    pub authority_id: String,
    pub integrity: SourceIntegrity,
    pub lifecycle: EvidenceLifecycle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofRequirement {
    Evidence(String),
    Claim { claim_id: String, polarity: Polarity },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofAlternative {
    pub id: String,
    pub conclusion: Polarity,
    /// Requirements are ANDed. Alternatives on one claim node are ORed.
    pub requirements: Vec<ProofRequirement>,
    /// Minimum distinct groups among this alternative's direct Evidence requirements.
    /// Each dependent Claim is independently checked against its own alternatives.
    pub min_independent_groups: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimNode {
    pub id: String,
    pub alternatives: Vec<ProofAlternative>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerCase {
    pub id: String,
    pub query: QueryProfile,
    pub target_claim_ids: Vec<String>,
    pub claims: Vec<ClaimNode>,
    pub evidence: Vec<EvidenceRevision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimDisposition {
    Supported { alternative_id: String },
    Refuted { alternative_id: String },
    Contested,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofPacket {
    pub case_id: String,
    pub claim_dispositions: BTreeMap<String, ClaimDisposition>,
    /// Ranked retrieval results (best first); distinct from final proof citations.
    pub retrieval_ranking: Vec<String>,
    pub cited_evidence_ids: BTreeSet<String>,
}

impl ProofPacket {
    pub fn abstain_all(case: &PlannerCase) -> Self {
        Self {
            case_id: case.id.clone(),
            claim_dispositions: case.target_claim_ids.iter()
                .map(|id| (id.clone(), ClaimDisposition::Abstain)).collect(),
            retrieval_ranking: Vec::new(),
            cited_evidence_ids: BTreeSet::new(),
        }
    }
}

/// Fixed cutoff for the synthetic retrieval contract. Confirmatory TEMPO runs
/// must use that benchmark's official cutoffs and evaluator.
pub const TEMPORAL_RETRIEVAL_K: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CaseScore {
    pub case_id: String,
    pub target_claims: usize,
    pub correct_dispositions: usize,
    pub incorrect_dispositions: usize,
    pub false_supported_claims: usize,
    pub false_refuted_claims: usize,
    pub missed_resolvable_claims: usize,
    pub unsupported_proof_paths: usize,
    pub missing_required_citations: usize,
    pub invalid_citations: usize,
    pub temporal_violations: usize,
    pub scope_violations: usize,
    pub authority_violations: usize,
    pub integrity_violations: usize,
    pub retracted_citations: usize,
    pub correlation_violations: usize,
    pub abstained_claims: usize,
    /// Number of hidden-oracle-relevant, time-admissible graph evidence items.
    pub temporal_relevant_evidence: usize,
    /// Unique relevant evidence found among the first TEMPORAL_RETRIEVAL_K ranked results.
    pub temporal_relevant_retrieved_at_k: usize,
    /// Number of ranked result slots evaluated (at most TEMPORAL_RETRIEVAL_K).
    pub temporal_retrieved_at_k: usize,
}

impl CaseScore {
    pub fn temporal_coverage_at_k(&self) -> Option<f64> {
        (self.temporal_relevant_evidence > 0).then_some(
            self.temporal_relevant_retrieved_at_k as f64 / self.temporal_relevant_evidence as f64
        )
    }

    pub fn temporal_precision_at_k(&self) -> Option<f64> {
        (self.temporal_retrieved_at_k > 0).then_some(
            self.temporal_relevant_retrieved_at_k as f64 / self.temporal_retrieved_at_k as f64
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GraphEvaluationReport {
    pub case_scores: Vec<CaseScore>,
    pub duplicate_prediction_packets: usize,
    pub unknown_prediction_packets: usize,
}

impl GraphEvaluationReport {
    pub fn case_count(&self) -> usize { self.case_scores.len() }
    pub fn correct_dispositions(&self) -> usize {
        self.case_scores.iter().map(|s| s.correct_dispositions).sum()
    }
    pub fn false_supported_claims(&self) -> usize {
        self.case_scores.iter().map(|s| s.false_supported_claims).sum()
    }
    pub fn false_refuted_claims(&self) -> usize {
        self.case_scores.iter().map(|s| s.false_refuted_claims).sum()
    }
    pub fn invalid_citations(&self) -> usize {
        self.case_scores.iter().map(|s| s.invalid_citations).sum()
    }
    pub fn all_hard_invariants_hold(&self) -> bool {
        self.false_supported_claims() == 0
            && self.false_refuted_claims() == 0
            && self.invalid_citations() == 0
            && self.case_scores.iter().all(|s| {
                s.unsupported_proof_paths == 0 && s.temporal_violations == 0
                    && s.scope_violations == 0 && s.authority_violations == 0
                    && s.integrity_violations == 0 && s.retracted_citations == 0
                    && s.correlation_violations == 0 && s.missing_required_citations == 0
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceMutation {
    Retract,
    FlipAssertedPolarity,
    SetKnownFrom(u64),
    SetValidFrom(u64),
    SetScope(String),
    SetAuthority(String),
    SetIntegrity(SourceIntegrity),
    SetIndependenceGroup(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HiddenTruth { SupportsAssertion, RefutesAssertion, Irrelevant }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OracleState { Supported, Refuted, Contested, Unknown }

#[derive(Debug, Clone, Default)]
struct Witness {
    evidence_ids: BTreeSet<String>,
    independence_groups: BTreeSet<String>,
}

#[derive(Debug, Clone, Default)]
struct ClaimEval {
    support: BTreeMap<String, Witness>,
    refute: BTreeMap<String, Witness>,
    correlation_rejected: BTreeSet<String>,
}

impl ClaimEval {
    fn state(&self) -> OracleState {
        match (!self.support.is_empty(), !self.refute.is_empty()) {
            (true, true) => OracleState::Contested,
            (true, false) => OracleState::Supported,
            (false, true) => OracleState::Refuted,
            (false, false) => OracleState::Unknown,
        }
    }
    fn witnesses(&self, polarity: Polarity) -> &BTreeMap<String, Witness> {
        match polarity { Polarity::Supports => &self.support, Polarity::Refutes => &self.refute }
    }
}

#[derive(Clone)]
pub struct GraphSuite {
    planner_cases: Vec<PlannerCase>,
    // Private and deliberately omitted from all planner-facing records.
    oracle: BTreeMap<String, BTreeMap<String, HiddenTruth>>,
}

impl GraphSuite {
    pub fn planner_cases(&self) -> &[PlannerCase] { &self.planner_cases }
    pub fn case_count(&self) -> usize { self.planner_cases.len() }

    /// Apply a controlled mutation to planner-facing evidence metadata.
    /// The private oracle label is never changed, so the mutation can be scored.
    pub fn mutate_evidence(
        &mut self,
        case_id: &str,
        evidence_id: &str,
        mutation: EvidenceMutation,
    ) -> bool {
        let Some(case) = self.planner_cases.iter_mut().find(|c| c.id == case_id) else { return false; };
        let Some(evidence) = case.evidence.iter_mut().find(|e| e.id == evidence_id) else { return false; };
        match mutation {
            EvidenceMutation::Retract => evidence.lifecycle = EvidenceLifecycle::Retracted,
            EvidenceMutation::FlipAssertedPolarity => {
                evidence.asserted_polarity = match evidence.asserted_polarity {
                    Polarity::Supports => Polarity::Refutes,
                    Polarity::Refutes => Polarity::Supports,
                };
            }
            EvidenceMutation::SetKnownFrom(value) => evidence.known_from = value,
            EvidenceMutation::SetValidFrom(value) => evidence.valid_from = value,
            EvidenceMutation::SetScope(value) => evidence.scope_id = value,
            EvidenceMutation::SetAuthority(value) => evidence.authority_id = value,
            EvidenceMutation::SetIntegrity(value) => evidence.integrity = value,
            EvidenceMutation::SetIndependenceGroup(value) => evidence.independence_group_id = value,
        }
        true
    }

    /// Convenience wrapper for the common retraction mutation.
    pub fn retract_evidence(&mut self, case_id: &str, evidence_id: &str) -> bool {
        self.mutate_evidence(case_id, evidence_id, EvidenceMutation::Retract)
    }

    /// Score public predictions using the private recursive oracle.
    /// Missing predictions abstain; duplicate and unknown packet IDs are counted.
    pub fn evaluate(&self, packets: &[ProofPacket]) -> GraphEvaluationReport {
        let mut by_case = BTreeMap::<String, &ProofPacket>::new();
        let mut duplicates = 0;
        for packet in packets {
            if by_case.insert(packet.case_id.clone(), packet).is_some() { duplicates += 1; }
        }
        let mut case_scores = Vec::with_capacity(self.planner_cases.len());
        for case in &self.planner_cases {
            let labels = self.oracle.get(&case.id).expect("oracle labels exist for generated case");
            let abstain;
            let packet = match by_case.remove(&case.id) {
                Some(packet) => packet,
                None => { abstain = ProofPacket::abstain_all(case); &abstain }
            };
            case_scores.push(score_case(case, labels, packet));
        }
        GraphEvaluationReport {
            case_scores,
            duplicate_prediction_packets: duplicates,
            unknown_prediction_packets: by_case.len(),
        }
    }
}

fn temporal(e: &EvidenceRevision, q: &QueryProfile) -> bool {
    e.valid_from <= q.valid_at && e.valid_until.map_or(true, |end| q.valid_at < end)
        && e.known_from <= q.known_at && e.known_until.map_or(true, |end| q.known_at < end)
}

fn admissible(e: &EvidenceRevision, q: &QueryProfile) -> bool {
    temporal(e, q) && e.scope_id == q.required_scope
        && e.authority_id == q.required_authority
        && e.integrity == SourceIntegrity::Verified
        && e.lifecycle == EvidenceLifecycle::Active
}

fn truth_matches(e: &EvidenceRevision, labels: &BTreeMap<String, HiddenTruth>) -> bool {
    matches!((e.asserted_polarity, labels.get(&e.id).copied()),
        (Polarity::Supports, Some(HiddenTruth::SupportsAssertion))
        | (Polarity::Refutes, Some(HiddenTruth::RefutesAssertion)))
}

fn resolve_claim(
    case: &PlannerCase,
    labels: &BTreeMap<String, HiddenTruth>,
    claim_id: &str,
    stack: &mut BTreeSet<String>,
    memo: &mut BTreeMap<String, ClaimEval>,
) -> ClaimEval {
    if let Some(cached) = memo.get(claim_id) { return cached.clone(); }
    if stack.contains(claim_id) { return ClaimEval::default(); }
    let Some(node) = case.claims.iter().find(|n| n.id == claim_id) else { return ClaimEval::default(); };
    stack.insert(claim_id.to_string());
    let evidence_by_id: BTreeMap<&str, &EvidenceRevision> =
        case.evidence.iter().map(|e| (e.id.as_str(), e)).collect();
    let mut result = ClaimEval::default();

    for alt in &node.alternatives {
        let mut witness = Witness::default();
        let mut direct_independence_groups = BTreeSet::new();
        let mut valid = true;
        for req in &alt.requirements {
            match req {
                ProofRequirement::Evidence(id) => {
                    let Some(e) = evidence_by_id.get(id.as_str()) else { valid = false; break; };
                    if e.subject_claim_id != claim_id || !admissible(e, &case.query) || !truth_matches(e, labels) {
                        valid = false;
                        break;
                    }
                    witness.evidence_ids.insert(e.id.clone());
                    witness.independence_groups.insert(e.independence_group_id.clone());
                    direct_independence_groups.insert(e.independence_group_id.clone());
                }
                ProofRequirement::Claim { claim_id: child_id, polarity } => {
                    let child = resolve_claim(case, labels, child_id, stack, memo);
                    let Some(child_witness) = child.witnesses(*polarity).values().next() else {
                        valid = false;
                        break;
                    };
                    witness.evidence_ids.extend(child_witness.evidence_ids.iter().cloned());
                    witness.independence_groups.extend(child_witness.independence_groups.iter().cloned());
                }
            }
        }
        if !valid { continue; }
        if direct_independence_groups.len() < alt.min_independent_groups {
            result.correlation_rejected.insert(alt.id.clone());
            continue;
        }
        match alt.conclusion {
            Polarity::Supports => { result.support.insert(alt.id.clone(), witness); }
            Polarity::Refutes => { result.refute.insert(alt.id.clone(), witness); }
        }
    }
    stack.remove(claim_id);
    memo.insert(claim_id.to_string(), result.clone());
    result
}

fn collect_reachable_evidence_requirements(
    case: &PlannerCase,
    claim_id: &str,
    visited_claims: &mut BTreeSet<String>,
    requirements: &mut BTreeSet<(String, String)>,
) {
    if !visited_claims.insert(claim_id.to_string()) {
        return;
    }
    let Some(node) = case.claims.iter().find(|node| node.id == claim_id) else {
        return;
    };
    for alternative in &node.alternatives {
        for requirement in &alternative.requirements {
            match requirement {
                ProofRequirement::Evidence(evidence_id) => {
                    requirements.insert((claim_id.to_string(), evidence_id.clone()));
                }
                ProofRequirement::Claim { claim_id: child, .. } => {
                    collect_reachable_evidence_requirements(case, child, visited_claims, requirements);
                }
            }
        }
    }
}

fn oracle_relevant_evidence_ids(
    case: &PlannerCase,
    labels: &BTreeMap<String, HiddenTruth>,
) -> BTreeSet<String> {
    let mut requirements = BTreeSet::<(String, String)>::new();
    let mut visited = BTreeSet::new();
    for target in &case.target_claim_ids {
        collect_reachable_evidence_requirements(case, target, &mut visited, &mut requirements);
    }

    let evidence_by_id: BTreeMap<&str, &EvidenceRevision> =
        case.evidence.iter().map(|e| (e.id.as_str(), e)).collect();
    requirements.into_iter().filter_map(|(claim_id, evidence_id)| {
        let evidence = evidence_by_id.get(evidence_id.as_str())?;
        (evidence.subject_claim_id == claim_id
            && admissible(evidence, &case.query)
            && truth_matches(evidence, labels))
            .then_some(evidence_id)
    }).collect()
}

fn score_case(
    case: &PlannerCase,
    labels: &BTreeMap<String, HiddenTruth>,
    packet: &ProofPacket,
) -> CaseScore {
    let evidence_by_id: BTreeMap<&str, &EvidenceRevision> =
        case.evidence.iter().map(|e| (e.id.as_str(), e)).collect();
    let mut memo = BTreeMap::new();
    let relevant_evidence = oracle_relevant_evidence_ids(case, labels);
    let mut score = CaseScore {
        case_id: case.id.clone(),
        target_claims: case.target_claim_ids.len(),
        temporal_relevant_evidence: relevant_evidence.len(),
        ..CaseScore::default()
    };
    let mut seen_retrieval = BTreeSet::new();
    for evidence_id in packet.retrieval_ranking.iter().take(TEMPORAL_RETRIEVAL_K) {
        score.temporal_retrieved_at_k += 1;
        if seen_retrieval.insert(evidence_id.clone())
            && relevant_evidence.contains(evidence_id)
        {
            score.temporal_relevant_retrieved_at_k += 1;
        }
    }

    for id in &case.target_claim_ids {
        let evaluation = resolve_claim(case, labels, id, &mut BTreeSet::new(), &mut memo);
        let state = evaluation.state();
        let disposition = packet.claim_dispositions.get(id).unwrap_or(&ClaimDisposition::Abstain);
        let mut correct = false;
        let mut witness: Option<&Witness> = None;
        match disposition {
            ClaimDisposition::Supported { alternative_id } => {
                witness = evaluation.support.get(alternative_id);
                correct = state == OracleState::Supported && witness.is_some();
                if !correct {
                    score.false_supported_claims += 1;
                    if witness.is_none() { score.unsupported_proof_paths += 1; }
                    if evaluation.correlation_rejected.contains(alternative_id) { score.correlation_violations += 1; }
                }
            }
            ClaimDisposition::Refuted { alternative_id } => {
                witness = evaluation.refute.get(alternative_id);
                correct = state == OracleState::Refuted && witness.is_some();
                if !correct {
                    score.false_refuted_claims += 1;
                    if witness.is_none() { score.unsupported_proof_paths += 1; }
                }
            }
            ClaimDisposition::Contested => { correct = state == OracleState::Contested; }
            ClaimDisposition::Abstain => {
                score.abstained_claims += 1;
                correct = state == OracleState::Unknown;
                if !correct { score.missed_resolvable_claims += 1; }
            }
        }
        if correct { score.correct_dispositions += 1; }
        else {
            score.incorrect_dispositions += 1;
            if !matches!(disposition, ClaimDisposition::Abstain) && state != OracleState::Unknown {
                score.missed_resolvable_claims += 1;
            }
        }
        if let Some(w) = witness {
            for required in &w.evidence_ids {
                if !packet.cited_evidence_ids.contains(required) { score.missing_required_citations += 1; }
            }
        }
    }

    for citation in &packet.cited_evidence_ids {
        let Some(e) = evidence_by_id.get(citation.as_str()) else { score.invalid_citations += 1; continue; };
        let mut invalid = false;
        if !temporal(e, &case.query) { score.temporal_violations += 1; invalid = true; }
        if e.scope_id != case.query.required_scope { score.scope_violations += 1; invalid = true; }
        if e.authority_id != case.query.required_authority { score.authority_violations += 1; invalid = true; }
        if e.integrity != SourceIntegrity::Verified { score.integrity_violations += 1; invalid = true; }
        if e.lifecycle == EvidenceLifecycle::Retracted { score.retracted_citations += 1; invalid = true; }
        if !truth_matches(e, labels) { invalid = true; }
        if invalid { score.invalid_citations += 1; }
    }
    score
}

fn ev(
    split: DatasetSplit, family: &str, id: &str, claim: &str, polarity: Polarity, group: &str,
    valid: (u64, Option<u64>), known: (u64, Option<u64>), scope: &str, authority: &str,
) -> EvidenceRevision {
    EvidenceRevision {
        id: id.into(), source_id: format!("src-{}-{}-{family}-{id}", split.label(), valid.0),
        source_family_id: format!("{}-family-{family}", split.label()),
        independence_group_id: group.into(), subject_claim_id: claim.into(),
        asserted_polarity: polarity, valid_from: valid.0, valid_until: valid.1,
        known_from: known.0, known_until: known.1, scope_id: scope.into(),
        authority_id: authority.into(), integrity: SourceIntegrity::Verified,
        lifecycle: EvidenceLifecycle::Active,
    }
}

fn q(valid_at: u64, known_at: u64) -> QueryProfile {
    QueryProfile { valid_at, known_at, required_scope: "scope-a".into(), required_authority: "authority-a".into() }
}

fn alt(id: &str, conclusion: Polarity, requirements: Vec<ProofRequirement>, independent: usize) -> ProofAlternative {
    ProofAlternative { id: id.into(), conclusion, requirements, min_independent_groups: independent }
}

fn claim(id: &str, alternatives: Vec<ProofAlternative>) -> ClaimNode {
    ClaimNode { id: id.into(), alternatives }
}

fn add(
    suite: &mut GraphSuite, seed: u64, split: DatasetSplit, suffix: &str, query: QueryProfile,
    targets: &[&str], claims: Vec<ClaimNode>, evidence: Vec<EvidenceRevision>,
    labels: &[(&str, HiddenTruth)],
) {
    let id = format!("poar-{}-{}-{suffix}", split.label(), seed);
    suite.planner_cases.push(PlannerCase {
        id: id.clone(), query, target_claim_ids: targets.iter().map(|s| (*s).into()).collect(), claims, evidence,
    });
    suite.oracle.insert(id, labels.iter().map(|(id, label)| ((*id).into(), *label)).collect());
}

/// Fixed-template synthetic generator. Seed salts identities and time coordinates;
/// split prefixes guarantee disjoint synthetic source-family IDs.
pub fn generate_poar_graph_suite(seed: u64, split: DatasetSplit) -> GraphSuite {
    let t = (seed % 1_000_000) * 100;
    let mut suite = GraphSuite { planner_cases: Vec::new(), oracle: BTreeMap::new() };

    // AND: root needs a supported child claim and separate root evidence.
    add(&mut suite, seed, split, "and-chain", q(t+5,t+5), &["root"], vec![
        claim("leaf", vec![alt("leaf-support", Polarity::Supports, vec![ProofRequirement::Evidence("e-leaf".into())], 1)]),
        claim("root", vec![alt("and-path", Polarity::Supports, vec![
            ProofRequirement::Claim { claim_id: "leaf".into(), polarity: Polarity::Supports },
            ProofRequirement::Evidence("e-root".into()),
        ], 1)]),
    ], vec![
        ev(split,"and","e-leaf","leaf",Polarity::Supports,"g-leaf",(t,None),(t,None),"scope-a","authority-a"),
        ev(split,"and","e-root","root",Polarity::Supports,"g-root",(t,None),(t,None),"scope-a","authority-a"),
    ], &[("e-leaf",HiddenTruth::SupportsAssertion),("e-root",HiddenTruth::SupportsAssertion)]);

    // OR: old revision is no longer known; current path remains admissible.
    add(&mut suite, seed, split, "or-current-alternative", q(t+5,t+20), &["claim"], vec![
        claim("claim", vec![
            alt("old-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-old".into())],1),
            alt("current-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-current".into())],1),
        ]),
    ], vec![
        ev(split,"or","e-old","claim",Polarity::Supports,"g-old",(t,None),(t,Some(t+10)),"scope-a","authority-a"),
        ev(split,"or","e-current","claim",Polarity::Supports,"g-current",(t,None),(t+10,None),"scope-a","authority-a"),
    ], &[("e-old",HiddenTruth::SupportsAssertion),("e-current",HiddenTruth::SupportsAssertion)]);

    // Correlated echo: two record IDs but one independence group.
    add(&mut suite, seed, split, "correlated-echo", q(t+5,t+5), &["claim"], vec![
        claim("claim", vec![alt("echo-path",Polarity::Supports,vec![
            ProofRequirement::Evidence("e-echo-a".into()),ProofRequirement::Evidence("e-echo-b".into()),
        ],2)]),
    ], vec![
        ev(split,"echo","e-echo-a","claim",Polarity::Supports,"same-origin",(t,None),(t,None),"scope-a","authority-a"),
        ev(split,"echo","e-echo-b","claim",Polarity::Supports,"same-origin",(t,None),(t,None),"scope-a","authority-a"),
    ], &[("e-echo-a",HiddenTruth::SupportsAssertion),("e-echo-b",HiddenTruth::SupportsAssertion)]);

    // Bitemporal correction: both records are valid at the query time, but only
    // the revision known at the query's known_at coordinate may be used.
    for (suffix, known_at) in [("temporal-before-correction",t+5),("temporal-after-correction",t+15)] {
        add(&mut suite,seed,split,suffix,q(t+5,known_at),&["claim"],vec![
            claim("claim",vec![
                alt("revision-one",Polarity::Supports,vec![ProofRequirement::Evidence("e-rev-one".into())],1),
                alt("revision-two",Polarity::Refutes,vec![ProofRequirement::Evidence("e-rev-two".into())],1),
            ]),
        ],vec![
            ev(split,"temporal","e-rev-one","claim",Polarity::Supports,"g-rev-one",(t,Some(t+100)),(t,Some(t+10)),"scope-a","authority-a"),
            ev(split,"temporal","e-rev-two","claim",Polarity::Refutes,"g-rev-two",(t,Some(t+100)),(t+10,None),"scope-a","authority-a"),
        ],&[("e-rev-one",HiddenTruth::SupportsAssertion),("e-rev-two",HiddenTruth::RefutesAssertion)]);
    }

    // Transitive retraction: a root conclusion depends on a retractable leaf source.
    add(&mut suite,seed,split,"retraction-dependency",q(t+5,t+5),&["root","unaffected"],vec![
        claim("base",vec![alt("base-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-base".into())],1)]),
        claim("root",vec![alt("root-via-base",Polarity::Supports,vec![
            ProofRequirement::Claim{claim_id:"base".into(),polarity:Polarity::Supports},
        ],0)]),
        claim("unaffected",vec![alt("unaffected-path",Polarity::Supports,vec![
            ProofRequirement::Evidence("e-unaffected".into()),
        ],1)]),
    ],vec![
        ev(split,"retraction","e-base","base",Polarity::Supports,"g-base",(t,None),(t,None),"scope-a","authority-a"),
        ev(split,"retraction-independent","e-unaffected","unaffected",Polarity::Supports,"g-unaffected",(t,None),(t,None),"scope-a","authority-a"),
    ],&[("e-base",HiddenTruth::SupportsAssertion),("e-unaffected",HiddenTruth::SupportsAssertion)]);

    // Scope mismatch and authority mismatch are distinct fixtures.
    add(&mut suite,seed,split,"scope-mismatch",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![alt("scope-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-scope".into())],1)]),
    ],vec![ev(split,"scope","e-scope","claim",Polarity::Supports,"g-scope",(t,None),(t,None),"scope-b","authority-a")],
    &[("e-scope",HiddenTruth::SupportsAssertion)]);

    add(&mut suite,seed,split,"authority-mismatch",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![alt("authority-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-authority".into())],1)]),
    ],vec![ev(split,"authority","e-authority","claim",Polarity::Supports,"g-authority",(t,None),(t,None),"scope-a","authority-b")],
    &[("e-authority",HiddenTruth::SupportsAssertion)]);

    // Simultaneous positive and negative support must remain contested.
    add(&mut suite,seed,split,"positive-negative-conflict",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![
            alt("positive-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-positive".into())],1),
            alt("negative-path",Polarity::Refutes,vec![ProofRequirement::Evidence("e-negative".into())],1),
        ]),
    ],vec![
        ev(split,"conflict","e-positive","claim",Polarity::Supports,"g-plus",(t,None),(t,None),"scope-a","authority-a"),
        ev(split,"conflict","e-negative","claim",Polarity::Refutes,"g-minus",(t,None),(t,None),"scope-a","authority-a"),
    ],&[("e-positive",HiddenTruth::SupportsAssertion),("e-negative",HiddenTruth::RefutesAssertion)]);

    // Cycles cannot self-certify.
    add(&mut suite,seed,split,"cyclic-derivation",q(t+5,t+5),&["a"],vec![
        claim("a",vec![alt("a-via-b",Polarity::Supports,vec![ProofRequirement::Claim{claim_id:"b".into(),polarity:Polarity::Supports}],0)]),
        claim("b",vec![alt("b-via-a",Polarity::Supports,vec![ProofRequirement::Claim{claim_id:"a".into(),polarity:Polarity::Supports}],0)]),
    ],Vec::new(),&[]);

    // Future-known evidence cannot leak into the earlier as-known view.
    add(&mut suite,seed,split,"future-known-leak",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![alt("future-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-future".into())],1)]),
    ],vec![ev(split,"future","e-future","claim",Polarity::Supports,"g-future",(t,None),(t+20,None),"scope-a","authority-a")],
    &[("e-future",HiddenTruth::SupportsAssertion)]);

    // Difficulty is structurally stratified: dev=8, calibration=32, held-out test=64.
    {
        let depth = match split {
            DatasetSplit::Development => 8usize,
            DatasetSplit::Calibration => 32usize,
            DatasetSplit::Test => 64usize,
        };
        let mut claims = Vec::with_capacity(depth);
        for index in 0..depth {
            let id = format!("chain-{index:03}");
            let alternative = if index + 1 == depth {
                alt("chain-tail", Polarity::Supports,
                    vec![ProofRequirement::Evidence("e-chain-leaf".into())], 1)
            } else {
                alt(&format!("chain-step-{index:03}"), Polarity::Supports,
                    vec![ProofRequirement::Claim {
                        claim_id: format!("chain-{:03}", index + 1),
                        polarity: Polarity::Supports,
                    }], 0)
            };
            claims.push(ClaimNode { id, alternatives: vec![alternative] });
        }
        let leaf = format!("chain-{:03}", depth - 1);
        let suffix = format!("long-chain-{depth}");
        add(&mut suite, seed, split, &suffix, q(t+5,t+5), &["chain-000"], claims,
            vec![ev(split,"long-chain","e-chain-leaf",&leaf,Polarity::Supports,"g-chain-leaf",
                (t,None),(t,None),"scope-a","authority-a")],
            &[("e-chain-leaf",HiddenTruth::SupportsAssertion)]);
    }

    // The public assertion is plausible but the hidden source-truth label disagrees.
    add(&mut suite,seed,split,"assertion-content-mismatch",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![alt("assertion-path",Polarity::Supports,
            vec![ProofRequirement::Evidence("e-assertion".into())],1)]),
    ],vec![ev(split,"assertion","e-assertion","claim",Polarity::Supports,"g-assertion",
        (t,None),(t,None),"scope-a","authority-a")],
    &[("e-assertion",HiddenTruth::Irrelevant)]);

    // Truthful content is still inadmissible when the source-integrity floor fails.
    let mut unverified = ev(split,"integrity","e-unverified","claim",Polarity::Supports,"g-unverified",
        (t,None),(t,None),"scope-a","authority-a");
    unverified.integrity = SourceIntegrity::Unverified;
    add(&mut suite,seed,split,"unverified-source",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![alt("unverified-path",Polarity::Supports,
            vec![ProofRequirement::Evidence("e-unverified".into())],1)]),
    ],vec![unverified],&[("e-unverified",HiddenTruth::SupportsAssertion)]);

    // Independent pair is a mutation baseline for a later source-correlation merge.
    add(&mut suite,seed,split,"independent-pair",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![alt("independent-path",Polarity::Supports,vec![
            ProofRequirement::Evidence("e-independent-a".into()),
            ProofRequirement::Evidence("e-independent-b".into()),
        ],2)]),
    ],vec![
        ev(split,"independent","e-independent-a","claim",Polarity::Supports,"g-independent-a",(t,None),(t,None),"scope-a","authority-a"),
        ev(split,"independent","e-independent-b","claim",Polarity::Supports,"g-independent-b",(t,None),(t,None),"scope-a","authority-a"),
    ],&[("e-independent-a",HiddenTruth::SupportsAssertion),("e-independent-b",HiddenTruth::SupportsAssertion)]);

    suite
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find<'a>(suite: &'a GraphSuite, suffix: &str) -> &'a PlannerCase {
        suite.planner_cases().iter().find(|c| c.id.ends_with(suffix)).expect("fixture exists")
    }
    fn packet(c: &PlannerCase, claim: &str, disposition: ClaimDisposition, citations: &[&str]) -> ProofPacket {
        ProofPacket {
            case_id: c.id.clone(),
            claim_dispositions: BTreeMap::from([(claim.into(),disposition)]),
            retrieval_ranking: citations.iter().map(|id| (*id).into()).collect(),
            cited_evidence_ids: citations.iter().map(|id| (*id).into()).collect(),
        }
    }

    #[test]
    fn split_graph_depths_are_held_out_in_difficulty_order() {
        fn chain_depth(suite: &GraphSuite) -> usize {
            let case = suite.planner_cases().iter().find(|c| c.id.contains("long-chain-")).expect("chain fixture exists");
            case.claims.iter().filter(|c| c.id.starts_with("chain-")).count()
        }
        let dev = generate_poar_graph_suite(41,DatasetSplit::Development);
        let cal = generate_poar_graph_suite(41,DatasetSplit::Calibration);
        let test = generate_poar_graph_suite(41,DatasetSplit::Test);
        assert_eq!(chain_depth(&dev),8);
        assert_eq!(chain_depth(&cal),32);
        assert_eq!(chain_depth(&test),64);
    }

    #[test]
    fn source_families_are_disjoint_across_development_calibration_and_test() {
        let suites = [
            generate_poar_graph_suite(41,DatasetSplit::Development),
            generate_poar_graph_suite(41,DatasetSplit::Calibration),
            generate_poar_graph_suite(41,DatasetSplit::Test),
        ];
        let families = |s: &GraphSuite| s.planner_cases().iter()
            .flat_map(|c| c.evidence.iter().map(|e| e.source_family_id.clone())).collect::<BTreeSet<_>>();
        let sources = |s: &GraphSuite| s.planner_cases().iter()
            .flat_map(|c| c.evidence.iter().map(|e| e.source_id.clone())).collect::<BTreeSet<_>>();
        let a=families(&suites[0]); let b=families(&suites[1]); let c=families(&suites[2]);
        let sa=sources(&suites[0]); let sb=sources(&suites[1]); let sc=sources(&suites[2]);
        assert!(a.is_disjoint(&b) && a.is_disjoint(&c) && b.is_disjoint(&c));
        assert!(sa.is_disjoint(&sb) && sa.is_disjoint(&sc) && sb.is_disjoint(&sc));
    }

    #[test]
    fn same_seed_and_split_reproduce_identical_planner_cases() {
        let first=generate_poar_graph_suite(2026,DatasetSplit::Development);
        let second=generate_poar_graph_suite(2026,DatasetSplit::Development);
        assert_eq!(first.planner_cases(),second.planner_cases());
        let first_predictions=first.planner_cases().iter().map(ProofPacket::abstain_all).collect::<Vec<_>>();
        let second_predictions=second.planner_cases().iter().map(ProofPacket::abstain_all).collect::<Vec<_>>();
        assert_eq!(first.evaluate(&first_predictions),second.evaluate(&second_predictions));
    }

    #[test]
    fn source_ids_are_seed_specific_within_a_split() {
        let first=generate_poar_graph_suite(7,DatasetSplit::Development);
        let second=generate_poar_graph_suite(8,DatasetSplit::Development);
        let source_ids = |suite: &GraphSuite| suite.planner_cases().iter()
            .flat_map(|case| case.evidence.iter().map(|e| e.source_id.clone())).collect::<BTreeSet<_>>();
        let family_ids = |suite: &GraphSuite| suite.planner_cases().iter()
            .flat_map(|case| case.evidence.iter().map(|e| e.source_family_id.clone())).collect::<BTreeSet<_>>();
        assert!(source_ids(&first).is_disjoint(&source_ids(&second)));
        assert_eq!(family_ids(&first),family_ids(&second));
    }

    #[test]
    fn oracle_accepts_and_or_and_current_paths_but_rejects_stale_revision() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let and=find(&suite,"and-chain");
        let or=find(&suite,"or-current-alternative");
        let good=vec![
            packet(and,"root",ClaimDisposition::Supported{alternative_id:"and-path".into()},&["e-leaf","e-root"]),
            packet(or,"claim",ClaimDisposition::Supported{alternative_id:"current-path".into()},&["e-current"]),
        ];
        let report=suite.evaluate(&good);
        assert_eq!(report.case_scores[0].correct_dispositions,1);
        assert_eq!(report.case_scores[1].correct_dispositions,1);
        let stale=suite.evaluate(&[packet(or,"claim",ClaimDisposition::Supported{alternative_id:"old-path".into()},&["e-old"])]);
        assert_eq!(stale.case_scores[1].false_supported_claims,1);
        assert_eq!(stale.case_scores[1].temporal_violations,1);
    }

    #[test]
    fn correlated_source_echo_does_not_meet_independence_threshold() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let c=find(&suite,"correlated-echo");
        let report=suite.evaluate(&[packet(c,"claim",ClaimDisposition::Supported{alternative_id:"echo-path".into()},&["e-echo-a","e-echo-b"])]);
        assert_eq!(report.case_scores[2].false_supported_claims,1);
        assert_eq!(report.case_scores[2].correlation_violations,1);
        assert_eq!(report.case_scores[2].unsupported_proof_paths,1);
    }

    #[test]
    fn known_time_selects_correct_side_of_late_correction() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let before=find(&suite,"temporal-before-correction");
        let after=find(&suite,"temporal-after-correction");
        let mut before_packet=packet(before,"claim",ClaimDisposition::Supported{alternative_id:"revision-one".into()},&["e-rev-one"]);
        before_packet.retrieval_ranking=vec!["e-rev-two".into(),"e-rev-one".into()];
        let mut after_packet=packet(after,"claim",ClaimDisposition::Refuted{alternative_id:"revision-two".into()},&["e-rev-two"]);
        after_packet.retrieval_ranking=vec!["e-rev-one".into(),"e-rev-two".into()];
        let report=suite.evaluate(&[before_packet,after_packet]);
        assert_eq!(report.case_scores[3].correct_dispositions,1);
        assert_eq!(report.case_scores[4].correct_dispositions,1);
        assert_eq!(report.case_scores[3].temporal_violations,0);
        assert_eq!(report.case_scores[4].temporal_violations,0);
        assert_eq!(report.case_scores[3].temporal_relevant_evidence,1);
        assert_eq!(report.case_scores[4].temporal_relevant_evidence,1);
        assert_eq!(report.case_scores[3].temporal_coverage_at_k(),Some(1.0));
        assert_eq!(report.case_scores[4].temporal_coverage_at_k(),Some(1.0));
        assert_eq!(report.case_scores[3].temporal_precision_at_k(),Some(0.5));
        assert_eq!(report.case_scores[4].temporal_precision_at_k(),Some(0.5));
    }

    #[test]
    fn retraction_invalidates_dependents_but_preserves_unrelated_claims() {
        let mut suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let c=find(&suite,"retraction-dependency");
        let id=c.id.clone();
        let p=ProofPacket {
            case_id:id.clone(),
            claim_dispositions:BTreeMap::from([
                ("root".into(),ClaimDisposition::Supported{alternative_id:"root-via-base".into()}),
                ("unaffected".into(),ClaimDisposition::Supported{alternative_id:"unaffected-path".into()}),
            ]),
            retrieval_ranking:vec!["e-base".into(),"e-unaffected".into()],
            cited_evidence_ids:BTreeSet::from(["e-base".into(),"e-unaffected".into()]),
        };
        assert_eq!(suite.evaluate(&[p.clone()]).case_scores[5].correct_dispositions,2);
        assert!(suite.retract_evidence(&id,"e-base"));
        let report=suite.evaluate(&[p]);
        let score=&report.case_scores[5];
        assert_eq!(score.false_supported_claims,1);
        assert_eq!(score.correct_dispositions,1);
        assert_eq!(score.retracted_citations,1);
        assert_eq!(score.invalid_citations,1);
    }

    #[test]
    fn scope_and_authority_are_separate_hard_checks() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let scope=find(&suite,"scope-mismatch");
        let authority=find(&suite,"authority-mismatch");
        let report=suite.evaluate(&[
            packet(scope,"claim",ClaimDisposition::Supported{alternative_id:"scope-path".into()},&["e-scope"]),
            packet(authority,"claim",ClaimDisposition::Supported{alternative_id:"authority-path".into()},&["e-authority"]),
        ]);
        assert_eq!(report.case_scores[6].scope_violations,1);
        assert_eq!(report.case_scores[7].authority_violations,1);
    }

    #[test]
    fn contradictory_support_and_refutation_requires_contested_disposition() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let c=find(&suite,"positive-negative-conflict");
        let correct=suite.evaluate(&[packet(c,"claim",ClaimDisposition::Contested,&["e-positive","e-negative"])]);
        assert_eq!(correct.case_scores[8].correct_dispositions,1);
        let wrong=suite.evaluate(&[packet(c,"claim",ClaimDisposition::Supported{alternative_id:"positive-path".into()},&["e-positive"])]);
        assert_eq!(wrong.case_scores[8].false_supported_claims,1);
    }

    #[test]
    fn cyclic_claim_dependencies_fail_closed() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let c=find(&suite,"cyclic-derivation");
        let report=suite.evaluate(&[packet(c,"a",ClaimDisposition::Supported{alternative_id:"a-via-b".into()},&[])]);
        assert_eq!(report.case_scores[9].false_supported_claims,1);
        assert_eq!(report.case_scores[9].unsupported_proof_paths,1);
    }

    #[test]
    fn future_known_evidence_is_not_admissible() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let c=find(&suite,"future-known-leak");
        let report=suite.evaluate(&[packet(c,"claim",ClaimDisposition::Supported{alternative_id:"future-path".into()},&["e-future"])]);
        assert_eq!(report.case_scores[10].temporal_violations,1);
        assert_eq!(report.case_scores[10].false_supported_claims,1);
    }

    #[test]
    fn deep_dependency_chain_requires_terminal_evidence() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let c=find(&suite,"long-chain-64");
        let good=packet(c,"chain-000",ClaimDisposition::Supported{alternative_id:"chain-step-000".into()},&["e-chain-leaf"]);
        let report=suite.evaluate(&[good]);
        assert_eq!(report.case_scores[11].correct_dispositions,1);
        let missing=packet(c,"chain-000",ClaimDisposition::Supported{alternative_id:"chain-step-000".into()},&[]);
        let missing_report=suite.evaluate(&[missing]);
        assert_eq!(missing_report.case_scores[11].missing_required_citations,1);
    }

    #[test]
    fn asserted_polarity_does_not_replace_hidden_source_truth() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let c=find(&suite,"assertion-content-mismatch");
        let p=packet(c,"claim",ClaimDisposition::Supported{alternative_id:"assertion-path".into()},&["e-assertion"]);
        let report=suite.evaluate(&[p]);
        let score=&report.case_scores[12];
        assert_eq!(score.false_supported_claims,1);
        assert_eq!(score.invalid_citations,1);
    }

    #[test]
    fn unverified_source_cannot_discharge_a_claim() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let c=find(&suite,"unverified-source");
        let p=packet(c,"claim",ClaimDisposition::Supported{alternative_id:"unverified-path".into()},&["e-unverified"]);
        let report=suite.evaluate(&[p]);
        let score=&report.case_scores[13];
        assert_eq!(score.false_supported_claims,1);
        assert_eq!(score.integrity_violations,1);
        assert_eq!(score.invalid_citations,1);
    }

    #[test]
    fn mutation_time_polarity_integrity_and_independence_are_detected() {
        // Move a current record beyond known_at: the formerly valid proof must fail.
        let mut temporal_suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let temporal_case=find(&temporal_suite,"or-current-alternative");
        let temporal_id=temporal_case.id.clone();
        let temporal_packet=packet(temporal_case,"claim",ClaimDisposition::Supported{alternative_id:"current-path".into()},&["e-current"]);
        assert!(temporal_suite.mutate_evidence(&temporal_id,"e-current",EvidenceMutation::SetKnownFrom(9_999)));
        let temporal_score=temporal_suite.evaluate(&[temporal_packet]);
        assert_eq!(temporal_score.case_scores[1].temporal_violations,1);
        assert_eq!(temporal_score.case_scores[1].false_supported_claims,1);

        // A metadata polarity flip does not rewrite the hidden source-truth label.
        let mut polarity_suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let polarity_case=find(&polarity_suite,"and-chain");
        let polarity_id=polarity_case.id.clone();
        let polarity_packet=packet(polarity_case,"root",ClaimDisposition::Supported{alternative_id:"and-path".into()},&["e-leaf","e-root"]);
        assert!(polarity_suite.mutate_evidence(&polarity_id,"e-root",EvidenceMutation::FlipAssertedPolarity));
        let polarity_score=polarity_suite.evaluate(&[polarity_packet]);
        assert_eq!(polarity_score.case_scores[0].false_supported_claims,1);
        assert_eq!(polarity_score.case_scores[0].invalid_citations,1);

        // Integrity downgrade keeps content truth fixed but removes admissibility.
        let mut integrity_suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let integrity_case=find(&integrity_suite,"and-chain");
        let integrity_id=integrity_case.id.clone();
        let integrity_packet=packet(integrity_case,"root",ClaimDisposition::Supported{alternative_id:"and-path".into()},&["e-leaf","e-root"]);
        assert!(integrity_suite.mutate_evidence(&integrity_id,"e-root",EvidenceMutation::SetIntegrity(SourceIntegrity::Invalid)));
        let integrity_score=integrity_suite.evaluate(&[integrity_packet]);
        assert_eq!(integrity_score.case_scores[0].integrity_violations,1);
        assert_eq!(integrity_score.case_scores[0].false_supported_claims,1);

        // Two genuinely independent sources collapse to one group after mutation.
        let mut correlation_suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let correlation_case=find(&correlation_suite,"independent-pair");
        let correlation_id=correlation_case.id.clone();
        let correlation_packet=packet(correlation_case,"claim",ClaimDisposition::Supported{alternative_id:"independent-path".into()},&["e-independent-a","e-independent-b"]);
        assert_eq!(correlation_suite.evaluate(&[correlation_packet.clone()]).case_scores[14].correct_dispositions,1);
        assert!(correlation_suite.mutate_evidence(&correlation_id,"e-independent-b",EvidenceMutation::SetIndependenceGroup("g-independent-a".into())));
        let correlation_score=correlation_suite.evaluate(&[correlation_packet]);
        assert_eq!(correlation_score.case_scores[14].correlation_violations,1);
        assert_eq!(correlation_score.case_scores[14].false_supported_claims,1);
    }

    #[test]
    fn valid_time_scope_and_authority_mutations_are_detected() {
        // valid_from moves beyond the query's valid_at coordinate.
        let mut valid_suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let valid_case=find(&valid_suite,"or-current-alternative");
        let valid_id=valid_case.id.clone();
        let valid_packet=packet(valid_case,"claim",ClaimDisposition::Supported{alternative_id:"current-path".into()},&["e-current"]);
        assert!(valid_suite.mutate_evidence(&valid_id,"e-current",EvidenceMutation::SetValidFrom(9_999)));
        let valid_report=valid_suite.evaluate(&[valid_packet]);
        assert_eq!(valid_report.case_scores[1].temporal_violations,1);

        // A source can retain truthful content while leaving the required scope.
        let mut scope_suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let scope_case=find(&scope_suite,"or-current-alternative");
        let scope_id=scope_case.id.clone();
        let scope_packet=packet(scope_case,"claim",ClaimDisposition::Supported{alternative_id:"current-path".into()},&["e-current"]);
        assert!(scope_suite.mutate_evidence(&scope_id,"e-current",EvidenceMutation::SetScope("scope-outside".into())));
        let scope_report=scope_suite.evaluate(&[scope_packet]);
        assert_eq!(scope_report.case_scores[1].scope_violations,1);

        // Authority is independent of scope and content truth.
        let mut authority_suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let authority_case=find(&authority_suite,"or-current-alternative");
        let authority_id=authority_case.id.clone();
        let authority_packet=packet(authority_case,"claim",ClaimDisposition::Supported{alternative_id:"current-path".into()},&["e-current"]);
        assert!(authority_suite.mutate_evidence(&authority_id,"e-current",EvidenceMutation::SetAuthority("authority-revoked".into())));
        let authority_report=authority_suite.evaluate(&[authority_packet]);
        assert_eq!(authority_report.case_scores[1].authority_violations,1);
    }

    #[test]
    fn duplicate_and_unknown_prediction_packets_are_counted() {
        let suite=generate_poar_graph_suite(7,DatasetSplit::Test);
        let c=find(&suite,"and-chain");
        let p=ProofPacket::abstain_all(c);
        let unknown=ProofPacket{case_id:"unknown".into(),claim_dispositions:BTreeMap::new(),retrieval_ranking:Vec::new(),cited_evidence_ids:BTreeSet::new()};
        let report=suite.evaluate(&[p.clone(),p,unknown]);
        assert_eq!(report.duplicate_prediction_packets,1);
        assert_eq!(report.unknown_prediction_packets,1);
    }
}
