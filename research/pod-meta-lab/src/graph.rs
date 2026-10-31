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

/// Explicit complexity controls for the deterministic synthetic graph suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphGenerationConfig {
    pub chain_depth: usize,
}

impl GraphGenerationConfig {
    pub fn for_split(split: DatasetSplit) -> Self {
        let chain_depth = match split {
            DatasetSplit::Development => 8,
            DatasetSplit::Calibration => 32,
            DatasetSplit::Test => 64,
        };
        Self { chain_depth }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphGenerationError {
    ChainDepthOutOfRange { requested: usize, minimum: usize, maximum: usize },
}

impl std::fmt::Display for GraphGenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChainDepthOutOfRange { requested, minimum, maximum } => write!(
                f,
                "chain depth {requested} is outside the supported range {minimum}..={maximum}"
            ),
        }
    }
}

impl std::error::Error for GraphGenerationError {}

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

#[derive(Debug, Clone)]
pub struct GraphSuite {
    planner_cases: Vec<PlannerCase>,
    // Generator identity used when serializing deterministic benchmark artifacts.
    seed: u64,
    split: DatasetSplit,
    config: GraphGenerationConfig,
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

#[allow(clippy::too_many_arguments)] // Synthetic fixture fields are deliberately explicit at call sites.
fn ev(
    seed: u64, split: DatasetSplit, family: &str, id: &str, claim: &str, polarity: Polarity, group: &str,
    valid: (u64, Option<u64>), known: (u64, Option<u64>), scope: &str, authority: &str,
) -> EvidenceRevision {
    EvidenceRevision {
        id: id.into(), source_id: format!("src-{}-{seed}-{family}-{id}", split.label()),
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

#[allow(clippy::too_many_arguments)] // Fixture constructor keeps case, graph, evidence, and oracle labels visible.
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
    generate_poar_graph_suite_with_config(seed, split, GraphGenerationConfig::for_split(split))
        .expect("the built-in split profiles have valid chain depths")
}

/// Generate a suite with explicit, validated graph complexity. The default
/// wrapper preserves the fixed 8/32/64 split profiles for existing callers.
pub fn generate_poar_graph_suite_with_config(
    seed: u64,
    split: DatasetSplit,
    config: GraphGenerationConfig,
) -> Result<GraphSuite, GraphGenerationError> {
    const MIN_CHAIN_DEPTH: usize = 1;
    const MAX_CHAIN_DEPTH: usize = 256;
    if !(MIN_CHAIN_DEPTH..=MAX_CHAIN_DEPTH).contains(&config.chain_depth) {
        return Err(GraphGenerationError::ChainDepthOutOfRange {
            requested: config.chain_depth,
            minimum: MIN_CHAIN_DEPTH,
            maximum: MAX_CHAIN_DEPTH,
        });
    }

    let t = (seed % 1_000_000) * 100;
    let mut suite = GraphSuite {
        planner_cases: Vec::new(),
        seed,
        split,
        config,
        oracle: BTreeMap::new(),
    };

    // AND: root needs a supported child claim and separate root evidence.
    add(&mut suite, seed, split, "and-chain", q(t+5,t+5), &["root"], vec![
        claim("leaf", vec![alt("leaf-support", Polarity::Supports, vec![ProofRequirement::Evidence("e-leaf".into())], 1)]),
        claim("root", vec![alt("and-path", Polarity::Supports, vec![
            ProofRequirement::Claim { claim_id: "leaf".into(), polarity: Polarity::Supports },
            ProofRequirement::Evidence("e-root".into()),
        ], 1)]),
    ], vec![
        ev(seed,split,"and","e-leaf","leaf",Polarity::Supports,"g-leaf",(t,None),(t,None),"scope-a","authority-a"),
        ev(seed,split,"and","e-root","root",Polarity::Supports,"g-root",(t,None),(t,None),"scope-a","authority-a"),
    ], &[("e-leaf",HiddenTruth::SupportsAssertion),("e-root",HiddenTruth::SupportsAssertion)]);

    // OR: old revision is no longer known; current path remains admissible.
    add(&mut suite, seed, split, "or-current-alternative", q(t+5,t+20), &["claim"], vec![
        claim("claim", vec![
            alt("old-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-old".into())],1),
            alt("current-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-current".into())],1),
        ]),
    ], vec![
        ev(seed,split,"or","e-old","claim",Polarity::Supports,"g-old",(t,None),(t,Some(t+10)),"scope-a","authority-a"),
        ev(seed,split,"or","e-current","claim",Polarity::Supports,"g-current",(t,None),(t+10,None),"scope-a","authority-a"),
    ], &[("e-old",HiddenTruth::SupportsAssertion),("e-current",HiddenTruth::SupportsAssertion)]);

    // Correlated echo: two record IDs but one independence group.
    add(&mut suite, seed, split, "correlated-echo", q(t+5,t+5), &["claim"], vec![
        claim("claim", vec![alt("echo-path",Polarity::Supports,vec![
            ProofRequirement::Evidence("e-echo-a".into()),ProofRequirement::Evidence("e-echo-b".into()),
        ],2)]),
    ], vec![
        ev(seed,split,"echo","e-echo-a","claim",Polarity::Supports,"same-origin",(t,None),(t,None),"scope-a","authority-a"),
        ev(seed,split,"echo","e-echo-b","claim",Polarity::Supports,"same-origin",(t,None),(t,None),"scope-a","authority-a"),
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
            ev(seed,split,"temporal","e-rev-one","claim",Polarity::Supports,"g-rev-one",(t,Some(t+100)),(t,Some(t+10)),"scope-a","authority-a"),
            ev(seed,split,"temporal","e-rev-two","claim",Polarity::Refutes,"g-rev-two",(t,Some(t+100)),(t+10,None),"scope-a","authority-a"),
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
        ev(seed,split,"retraction","e-base","base",Polarity::Supports,"g-base",(t,None),(t,None),"scope-a","authority-a"),
        ev(seed,split,"retraction-independent","e-unaffected","unaffected",Polarity::Supports,"g-unaffected",(t,None),(t,None),"scope-a","authority-a"),
    ],&[("e-base",HiddenTruth::SupportsAssertion),("e-unaffected",HiddenTruth::SupportsAssertion)]);

    // Scope mismatch and authority mismatch are distinct fixtures.
    add(&mut suite,seed,split,"scope-mismatch",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![alt("scope-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-scope".into())],1)]),
    ],vec![ev(seed,split,"scope","e-scope","claim",Polarity::Supports,"g-scope",(t,None),(t,None),"scope-b","authority-a")],
    &[("e-scope",HiddenTruth::SupportsAssertion)]);

    add(&mut suite,seed,split,"authority-mismatch",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![alt("authority-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-authority".into())],1)]),
    ],vec![ev(seed,split,"authority","e-authority","claim",Polarity::Supports,"g-authority",(t,None),(t,None),"scope-a","authority-b")],
    &[("e-authority",HiddenTruth::SupportsAssertion)]);

    // Simultaneous positive and negative support must remain contested.
    add(&mut suite,seed,split,"positive-negative-conflict",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![
            alt("positive-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-positive".into())],1),
            alt("negative-path",Polarity::Refutes,vec![ProofRequirement::Evidence("e-negative".into())],1),
        ]),
    ],vec![
        ev(seed,split,"conflict","e-positive","claim",Polarity::Supports,"g-plus",(t,None),(t,None),"scope-a","authority-a"),
        ev(seed,split,"conflict","e-negative","claim",Polarity::Refutes,"g-minus",(t,None),(t,None),"scope-a","authority-a"),
    ],&[("e-positive",HiddenTruth::SupportsAssertion),("e-negative",HiddenTruth::RefutesAssertion)]);

    // Cycles cannot self-certify.
    add(&mut suite,seed,split,"cyclic-derivation",q(t+5,t+5),&["a"],vec![
        claim("a",vec![alt("a-via-b",Polarity::Supports,vec![ProofRequirement::Claim{claim_id:"b".into(),polarity:Polarity::Supports}],0)]),
        claim("b",vec![alt("b-via-a",Polarity::Supports,vec![ProofRequirement::Claim{claim_id:"a".into(),polarity:Polarity::Supports}],0)]),
    ],Vec::new(),&[]);

    // Future-known evidence cannot leak into the earlier as-known view.
    add(&mut suite,seed,split,"future-known-leak",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![alt("future-path",Polarity::Supports,vec![ProofRequirement::Evidence("e-future".into())],1)]),
    ],vec![ev(seed,split,"future","e-future","claim",Polarity::Supports,"g-future",(t,None),(t+20,None),"scope-a","authority-a")],
    &[("e-future",HiddenTruth::SupportsAssertion)]);

    // Difficulty is structurally stratified: dev=8, calibration=32, held-out test=64.
    {
        let depth = config.chain_depth;
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
            vec![ev(seed,split,"long-chain","e-chain-leaf",&leaf,Polarity::Supports,"g-chain-leaf",
                (t,None),(t,None),"scope-a","authority-a")],
            &[("e-chain-leaf",HiddenTruth::SupportsAssertion)]);
    }

    // The public assertion is plausible but the hidden source-truth label disagrees.
    add(&mut suite,seed,split,"assertion-content-mismatch",q(t+5,t+5),&["claim"],vec![
        claim("claim",vec![alt("assertion-path",Polarity::Supports,
            vec![ProofRequirement::Evidence("e-assertion".into())],1)]),
    ],vec![ev(seed,split,"assertion","e-assertion","claim",Polarity::Supports,"g-assertion",
        (t,None),(t,None),"scope-a","authority-a")],
    &[("e-assertion",HiddenTruth::Irrelevant)]);

    // Truthful content is still inadmissible when the source-integrity floor fails.
    let mut unverified = ev(seed,split,"integrity","e-unverified","claim",Polarity::Supports,"g-unverified",
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
        ev(seed,split,"independent","e-independent-a","claim",Polarity::Supports,"g-independent-a",(t,None),(t,None),"scope-a","authority-a"),
        ev(seed,split,"independent","e-independent-b","claim",Polarity::Supports,"g-independent-b",(t,None),(t,None),"scope-a","authority-a"),
    ],&[("e-independent-a",HiddenTruth::SupportsAssertion),("e-independent-b",HiddenTruth::SupportsAssertion)]);

    Ok(suite)
}

fn hex_field(value: &str) -> String {
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn polarity_field(polarity: Polarity) -> &'static str {
    match polarity {
        Polarity::Supports => "supports",
        Polarity::Refutes => "refutes",
    }
}

fn optional_u64_field(value: Option<u64>) -> String {
    value.map_or_else(|| "-".to_string(), |number| number.to_string())
}

fn integrity_field(integrity: SourceIntegrity) -> &'static str {
    match integrity {
        SourceIntegrity::Verified => "verified",
        SourceIntegrity::Unverified => "unverified",
        SourceIntegrity::Invalid => "invalid",
    }
}

fn lifecycle_field(lifecycle: EvidenceLifecycle) -> &'static str {
    match lifecycle {
        EvidenceLifecycle::Active => "active",
        EvidenceLifecycle::Retracted => "retracted",
    }
}

fn join_record(fields: &[String]) -> String {
    fields.join("\t")
}

/// Encode planner-visible claims, asserted evidence metadata, and graph
/// references only. Hidden source-truth labels and gold outcomes are omitted.
pub fn encode_planner_artifact_v1(
    suite: &GraphSuite,
    generator_revision: &str,
) -> Result<String, String> {
    if generator_revision.is_empty() {
        return Err("generator revision must not be empty".into());
    }

    let mut records = Vec::<String>::new();
    for case in &suite.planner_cases {
        let case_id = hex_field(&case.id);
        records.push(join_record(&[
            "CASE".into(),
            case_id.clone(),
            case.query.valid_at.to_string(),
            case.query.known_at.to_string(),
            hex_field(&case.query.required_scope),
            hex_field(&case.query.required_authority),
        ]));

        for target in &case.target_claim_ids {
            records.push(join_record(&[
                "TARGET".into(),
                case_id.clone(),
                hex_field(target),
            ]));
        }

        for claim in &case.claims {
            records.push(join_record(&[
                "CLAIM".into(),
                case_id.clone(),
                hex_field(&claim.id),
            ]));
            for alternative in &claim.alternatives {
                records.push(join_record(&[
                    "ALT".into(),
                    case_id.clone(),
                    hex_field(&claim.id),
                    hex_field(&alternative.id),
                    polarity_field(alternative.conclusion).into(),
                    alternative.min_independent_groups.to_string(),
                ]));

                for requirement in &alternative.requirements {
                    match requirement {
                        ProofRequirement::Evidence(evidence_id) => records.push(join_record(&[
                            "REQ_E".into(),
                            case_id.clone(),
                            hex_field(&claim.id),
                            hex_field(&alternative.id),
                            hex_field(evidence_id),
                        ])),
                        ProofRequirement::Claim { claim_id, polarity } => records.push(join_record(&[
                            "REQ_C".into(),
                            case_id.clone(),
                            hex_field(&claim.id),
                            hex_field(&alternative.id),
                            hex_field(claim_id),
                            polarity_field(*polarity).into(),
                        ])),
                    }
                }
            }
        }

        for evidence in &case.evidence {
            records.push(join_record(&[
                "EVIDENCE".into(),
                case_id.clone(),
                hex_field(&evidence.id),
                hex_field(&evidence.source_id),
                hex_field(&evidence.source_family_id),
                hex_field(&evidence.independence_group_id),
                hex_field(&evidence.subject_claim_id),
                polarity_field(evidence.asserted_polarity).into(),
                evidence.valid_from.to_string(),
                optional_u64_field(evidence.valid_until),
                evidence.known_from.to_string(),
                optional_u64_field(evidence.known_until),
                hex_field(&evidence.scope_id),
                hex_field(&evidence.authority_id),
                integrity_field(evidence.integrity).into(),
                lifecycle_field(evidence.lifecycle).into(),
            ]));
        }
    }

    records.sort_unstable();
    let mut output = format!(
        "HEADER\tPOAR-TSV/1\t{}\t{}\t{}\t{}\n",
        hex_field(generator_revision),
        suite.seed,
        suite.split.label(),
        suite.config.chain_depth
    );
    for record in records {
        output.push_str(&record);
        output.push('\n');
    }
    Ok(output)
}

fn hidden_truth_field(truth: HiddenTruth) -> &'static str {
    match truth {
        HiddenTruth::SupportsAssertion => "supports_assertion",
        HiddenTruth::RefutesAssertion => "refutes_assertion",
        HiddenTruth::Irrelevant => "irrelevant",
    }
}

fn oracle_state_field(state: OracleState) -> &'static str {
    match state {
        OracleState::Supported => "supported",
        OracleState::Refuted => "refuted",
        OracleState::Contested => "contested",
        OracleState::Unknown => "unknown",
    }
}

/// Encode hidden source labels and gold proof witnesses for evaluator-only use.
/// The planner digest must be the lowercase SHA-256 of the exact planner bytes;
/// the evaluator must verify it before accepting predictions.
pub fn encode_oracle_artifact_v1(
    suite: &GraphSuite,
    generator_revision: &str,
    planner_sha256: &str,
) -> Result<String, String> {
    if generator_revision.is_empty() {
        return Err("generator revision must not be empty".into());
    }
    if planner_sha256.len() != 64
        || !planner_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        || planner_sha256.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        return Err("planner SHA-256 must be exactly 64 lowercase hexadecimal characters".into());
    }

    let mut records = Vec::<String>::new();
    for case in &suite.planner_cases {
        let case_id = hex_field(&case.id);
        let labels = suite.oracle.get(&case.id)
            .ok_or_else(|| format!("missing oracle labels for case {}", case.id))?;

        for (evidence_id, truth) in labels {
            records.push(join_record(&[
                "TRUTH".into(),
                case_id.clone(),
                hex_field(evidence_id),
                hidden_truth_field(*truth).into(),
            ]));
        }

        let mut memo = BTreeMap::new();
        for claim_id in &case.target_claim_ids {
            let evaluation = resolve_claim(
                case,
                labels,
                claim_id,
                &mut BTreeSet::new(),
                &mut memo,
            );
            records.push(join_record(&[
                "GOLD_CLAIM".into(),
                case_id.clone(),
                hex_field(claim_id),
                oracle_state_field(evaluation.state()).into(),
            ]));

            for (polarity, witnesses) in [
                (Polarity::Supports, &evaluation.support),
                (Polarity::Refutes, &evaluation.refute),
            ] {
                for (alternative_id, witness) in witnesses {
                    let mut fields = vec![
                        "GOLD_PATH".into(),
                        case_id.clone(),
                        hex_field(claim_id),
                        polarity_field(polarity).into(),
                        hex_field(alternative_id),
                        witness.evidence_ids.len().to_string(),
                    ];
                    fields.extend(witness.evidence_ids.iter().map(|id| hex_field(id)));
                    records.push(join_record(&fields));
                }
            }
        }
    }

    records.sort_unstable();
    let mut output = format!(
        "ORACLE_HEADER\tPOAR-TSV/1\t{}\t{}\t{}\t{}\n",
        planner_sha256,
        hex_field(generator_revision),
        suite.seed,
        suite.split.label()
    );
    for record in records {
        output.push_str(&record);
        output.push('\n');
    }
    Ok(output)
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
    fn planner_artifact_excludes_truth_and_gold_records() {
        let suite=generate_poar_graph_suite(123,DatasetSplit::Test);
        let planner=encode_planner_artifact_v1(&suite,"unit-test-generator-revision").expect("planner encoding");
        let oracle=encode_oracle_artifact_v1(&suite,"unit-test-generator-revision",&"a".repeat(64)).expect("oracle encoding");
        assert!(planner.starts_with("HEADER\tPOAR-TSV/1\t"));
        assert!(!planner.lines().any(|line| line.starts_with("TRUTH\t")));
        assert!(!planner.lines().any(|line| line.starts_with("GOLD_CLAIM\t")));
        assert!(!planner.lines().any(|line| line.starts_with("GOLD_PATH\t")));
        assert!(!planner.contains("supports_assertion"));
        assert!(oracle.starts_with(&format!("ORACLE_HEADER\tPOAR-TSV/1\t{}\t","a".repeat(64))));
        assert!(oracle.lines().any(|line| line.starts_with("TRUTH\t")));
        assert!(oracle.lines().any(|line| line.starts_with("GOLD_CLAIM\t")));
        assert!(oracle.lines().any(|line| line.starts_with("GOLD_PATH\t")));
    }

    #[test]
    fn oracle_artifact_rejects_noncanonical_planner_digests() {
        let suite=generate_poar_graph_suite(123,DatasetSplit::Test);
        assert!(encode_oracle_artifact_v1(&suite,"rev","ABCDEF").is_err());
        assert!(encode_oracle_artifact_v1(&suite,"rev",&"g".repeat(64)).is_err());
        assert!(encode_oracle_artifact_v1(&suite,"rev",&"A".repeat(64)).is_err());
        assert!(encode_oracle_artifact_v1(&suite,"",&"a".repeat(64)).is_err());
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
    fn custom_generation_profiles_change_graph_depth_deterministically() {
        let config=GraphGenerationConfig{chain_depth:5};
        let first=generate_poar_graph_suite_with_config(99,DatasetSplit::Test,config).expect("valid config");
        let second=generate_poar_graph_suite_with_config(99,DatasetSplit::Test,config).expect("same valid config");
        assert_eq!(first.planner_cases(),second.planner_cases());
        let chain=first.planner_cases().iter().find(|case| case.id.contains("long-chain-")).expect("custom chain exists");
        assert_eq!(chain.claims.iter().filter(|claim| claim.id.starts_with("chain-")).count(),5);
        assert!(chain.id.ends_with("long-chain-5"));
    }

    #[test]
    fn invalid_generation_profiles_fail_closed() {
        for depth in [0,257] {
            let error=generate_poar_graph_suite_with_config(
                99,DatasetSplit::Test,GraphGenerationConfig{chain_depth:depth}
            ).expect_err("invalid depth must be rejected");
            assert_eq!(error,GraphGenerationError::ChainDepthOutOfRange{
                requested:depth,minimum:1,maximum:256
            });
        }
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


// ---------------------------------------------------------------------------
// POAR frozen-artifact parser and evaluator. This runs after inference: the
// planner emits predictions, then a separate process consumes these bytes.
// Parsing validates canonical form and cross-artifact identity before scoring.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoarPredictionIdentityV1 {
    pub planner_sha256: String,
    pub model_id: String,
    pub checkpoint_sha256: String,
    pub prompt_sha256: String,
    pub run_config_sha256: String,
    pub seed: u64,
    pub split: DatasetSplit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoarArtifactEvaluationV1 {
    pub prediction_identity: PoarPredictionIdentityV1,
    pub report: GraphEvaluationReport,
}

#[derive(Debug, Clone)]
struct ParsedPlannerV1 {
    generator_revision: String,
    seed: u64,
    split: DatasetSplit,
    config: GraphGenerationConfig,
    cases: Vec<PlannerCase>,
}

#[derive(Debug, Default)]
struct PlannerCaseBuilderV1 {
    query: Option<QueryProfile>,
    targets: BTreeSet<String>,
    declared_claims: BTreeSet<String>,
    alternatives: BTreeMap<(String, String), ProofAlternative>,
    evidence: BTreeMap<String, EvidenceRevision>,
}

#[derive(Debug)]
struct ParsedOracleV1 {
    planner_sha256: String,
    generator_revision: String,
    seed: u64,
    split: DatasetSplit,
    truths: BTreeMap<String, BTreeMap<String, HiddenTruth>>,
    gold_claims: BTreeMap<(String, String), OracleState>,
    gold_paths: BTreeMap<(String, String, Polarity, String), BTreeSet<String>>,
}

#[derive(Debug, Default)]
struct PredictionCaseBuilderV1 {
    dispositions: BTreeMap<String, ClaimDisposition>,
    retrieval: BTreeMap<u64, String>,
    retrieval_ids: BTreeSet<String>,
    citations: BTreeSet<String>,
}

#[derive(Debug)]
struct ParsedPredictionsV1 {
    identity: PoarPredictionIdentityV1,
    packets: Vec<ProofPacket>,
}

fn canonical_lines_v1(bytes: &[u8], label: &str) -> Result<Vec<String>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("{label}: artifact is not UTF-8: {error}"))?;
    if text.is_empty()
        || !text.ends_with('\n')
        || text.ends_with("\n\n")
        || text.starts_with('\u{feff}')
        || text.contains('\r')
        || text.contains('\0')
    {
        return Err(format!(
            "{label}: non-canonical byte stream (requires UTF-8, no BOM/CR/NUL, and exactly one terminal LF)"
        ));
    }
    let lines = text[..text.len() - 1]
        .split('\n')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if lines.iter().any(String::is_empty) {
        return Err(format!("{label}: empty records are not permitted"));
    }
    if lines.len() > 2_000_000 {
        return Err(format!("{label}: record limit exceeded"));
    }
    if lines.len() > 2
        && lines[1..].windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(format!("{label}: records must be strictly lexicographically sorted and unique"));
    }
    Ok(lines)
}

fn expect_fields_v1<'a>(
    fields: &'a [&'a str],
    expected: usize,
    label: &str,
    line_number: usize,
) -> Result<(), String> {
    if fields.len() != expected {
        return Err(format!(
            "{label}: record at line {line_number} has {} fields; expected {expected}",
            fields.len()
        ));
    }
    Ok(())
}

fn decode_hex_field_v1(value: &str, label: &str) -> Result<String, String> {
    if value.is_empty() || value.len() % 2 != 0 {
        return Err(format!("{label}: hex field must be non-empty and have even length"));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let digit = |byte: u8| -> Option<u8> {
            match byte {
                b'0'..=b'9' => Some(byte - b'0'),
                b'a'..=b'f' => Some(byte - b'a' + 10),
                _ => None,
            }
        };
        let high = digit(pair[0]).ok_or_else(|| format!("{label}: hex must be lowercase canonical hexadecimal"))?;
        let low = digit(pair[1]).ok_or_else(|| format!("{label}: hex must be lowercase canonical hexadecimal"))?;
        bytes.push((high << 4) | low);
    }
    let decoded = String::from_utf8(bytes)
        .map_err(|error| format!("{label}: hex does not decode to UTF-8: {error}"))?;
    if decoded.is_empty()
        || decoded.contains('\0')
        || decoded.contains('\t')
        || decoded.contains('\r')
        || decoded.contains('\n')
        || hex_field(&decoded) != value
    {
        return Err(format!("{label}: decoded string is empty, contains a forbidden control, or is non-canonical"));
    }
    Ok(decoded)
}

fn parse_u64_canonical_v1(value: &str, label: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|error| format!("{label}: invalid unsigned integer {value:?}: {error}"))?;
    if parsed.to_string() != value {
        return Err(format!("{label}: integer is not in canonical base-10 form"));
    }
    Ok(parsed)
}

fn parse_usize_canonical_v1(value: &str, label: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("{label}: invalid unsigned integer {value:?}: {error}"))?;
    if parsed.to_string() != value {
        return Err(format!("{label}: integer is not in canonical base-10 form"));
    }
    Ok(parsed)
}

fn parse_optional_u64_v1(value: &str, label: &str) -> Result<Option<u64>, String> {
    if value == "-" {
        Ok(None)
    } else {
        parse_u64_canonical_v1(value, label).map(Some)
    }
}

fn parse_split_v1(value: &str, label: &str) -> Result<DatasetSplit, String> {
    match value {
        "dev" => Ok(DatasetSplit::Development),
        "cal" => Ok(DatasetSplit::Calibration),
        "test" => Ok(DatasetSplit::Test),
        _ => Err(format!("{label}: unknown split {value:?}")),
    }
}

fn parse_polarity_v1(value: &str, label: &str) -> Result<Polarity, String> {
    match value {
        "supports" => Ok(Polarity::Supports),
        "refutes" => Ok(Polarity::Refutes),
        _ => Err(format!("{label}: unknown polarity {value:?}")),
    }
}

fn valid_sha256_v1(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_planner_artifact_v1(bytes: &[u8]) -> Result<ParsedPlannerV1, String> {
    let lines = canonical_lines_v1(bytes, "planner")?;
    let header = lines[0].split('\t').collect::<Vec<_>>();
    expect_fields_v1(&header, 6, "planner header", 1)?;
    if header[0] != "HEADER" || header[1] != "POAR-TSV/1" {
        return Err("planner: unsupported or missing POAR-TSV/1 header".into());
    }
    let generator_revision = decode_hex_field_v1(header[2], "planner generator revision")?;
    let seed = parse_u64_canonical_v1(header[3], "planner seed")?;
    let split = parse_split_v1(header[4], "planner split")?;
    let chain_depth = parse_usize_canonical_v1(header[5], "planner chain depth")?;
    if !(1..=256).contains(&chain_depth) {
        return Err("planner: chain depth must be in 1..=256".into());
    }

    let mut builders = BTreeMap::<String, PlannerCaseBuilderV1>::new();
    let mut requirements_seen = BTreeSet::<(String, String, String, String, String)>::new();
    for (offset, line) in lines.iter().enumerate().skip(1) {
        let line_number = offset + 1;
        let fields = line.split('\t').collect::<Vec<_>>();
        let tag = fields.first().copied().unwrap_or("");
        if tag == "HEADER" {
            return Err(format!("planner: duplicate header at line {line_number}"));
        }
        let case_id = fields
            .get(1)
            .ok_or_else(|| format!("planner: missing case ID at line {line_number}"))?;
        let case_id = decode_hex_field_v1(case_id, "planner case ID")?;
        let builder = builders.entry(case_id.clone()).or_default();

        match tag {
            "CASE" => {
                expect_fields_v1(&fields, 6, "CASE", line_number)?;
                if builder.query.is_some() {
                    return Err(format!("planner: duplicate CASE identity {case_id:?}"));
                }
                builder.query = Some(QueryProfile {
                    valid_at: parse_u64_canonical_v1(fields[2], "valid_at")?,
                    known_at: parse_u64_canonical_v1(fields[3], "known_at")?,
                    required_scope: decode_hex_field_v1(fields[4], "required scope")?,
                    required_authority: decode_hex_field_v1(fields[5], "required authority")?,
                });
            }
            "TARGET" => {
                expect_fields_v1(&fields, 3, "TARGET", line_number)?;
                let target = decode_hex_field_v1(fields[2], "target claim ID")?;
                if !builder.targets.insert(target.clone()) {
                    return Err(format!("planner: duplicate target {target:?} in {case_id:?}"));
                }
            }
            "CLAIM" => {
                expect_fields_v1(&fields, 3, "CLAIM", line_number)?;
                let claim = decode_hex_field_v1(fields[2], "claim ID")?;
                if !builder.declared_claims.insert(claim.clone()) {
                    return Err(format!("planner: duplicate claim {claim:?} in {case_id:?}"));
                }
            }
            "ALT" => {
                expect_fields_v1(&fields, 6, "ALT", line_number)?;
                let claim = decode_hex_field_v1(fields[2], "alternative claim ID")?;
                let alternative_id = decode_hex_field_v1(fields[3], "alternative ID")?;
                let alternative = ProofAlternative {
                    id: alternative_id.clone(),
                    conclusion: parse_polarity_v1(fields[4], "alternative conclusion")?,
                    requirements: Vec::new(),
                    min_independent_groups: parse_usize_canonical_v1(fields[5], "minimum independent groups")?,
                };
                if builder.alternatives.insert((claim.clone(), alternative_id.clone()), alternative).is_some() {
                    return Err(format!("planner: duplicate alternative {alternative_id:?} for {claim:?}"));
                }
            }
            "REQ_E" => {
                expect_fields_v1(&fields, 5, "REQ_E", line_number)?;
                let claim = decode_hex_field_v1(fields[2], "requirement claim ID")?;
                let alternative = decode_hex_field_v1(fields[3], "requirement alternative ID")?;
                let evidence = decode_hex_field_v1(fields[4], "requirement evidence ID")?;
                let key = (case_id.clone(), claim.clone(), alternative.clone(), "E".into(), evidence.clone());
                if !requirements_seen.insert(key) {
                    return Err(format!("planner: duplicate evidence requirement {evidence:?}"));
                }
                let target = builder.alternatives.get_mut(&(claim, alternative))
                    .ok_or_else(|| format!("planner: REQ_E references an undeclared alternative at line {line_number}"))?;
                target.requirements.push(ProofRequirement::Evidence(evidence));
            }
            "REQ_C" => {
                expect_fields_v1(&fields, 6, "REQ_C", line_number)?;
                let claim = decode_hex_field_v1(fields[2], "requirement claim ID")?;
                let alternative = decode_hex_field_v1(fields[3], "requirement alternative ID")?;
                let child = decode_hex_field_v1(fields[4], "child claim ID")?;
                let polarity = parse_polarity_v1(fields[5], "child claim polarity")?;
                let key = (case_id.clone(), claim.clone(), alternative.clone(), "C".into(), format!("{child}:{polarity:?}"));
                if !requirements_seen.insert(key) {
                    return Err(format!("planner: duplicate claim requirement {child:?}"));
                }
                let target = builder.alternatives.get_mut(&(claim, alternative))
                    .ok_or_else(|| format!("planner: REQ_C references an undeclared alternative at line {line_number}"))?;
                target.requirements.push(ProofRequirement::Claim { claim_id: child, polarity });
            }
            "EVIDENCE" => {
                expect_fields_v1(&fields, 16, "EVIDENCE", line_number)?;
                let evidence_id = decode_hex_field_v1(fields[2], "evidence ID")?;
                let evidence = EvidenceRevision {
                    id: evidence_id.clone(),
                    source_id: decode_hex_field_v1(fields[3], "source ID")?,
                    source_family_id: decode_hex_field_v1(fields[4], "source family ID")?,
                    independence_group_id: decode_hex_field_v1(fields[5], "independence group ID")?,
                    subject_claim_id: decode_hex_field_v1(fields[6], "evidence subject claim ID")?,
                    asserted_polarity: parse_polarity_v1(fields[7], "asserted polarity")?,
                    valid_from: parse_u64_canonical_v1(fields[8], "valid_from")?,
                    valid_until: parse_optional_u64_v1(fields[9], "valid_until")?,
                    known_from: parse_u64_canonical_v1(fields[10], "known_from")?,
                    known_until: parse_optional_u64_v1(fields[11], "known_until")?,
                    scope_id: decode_hex_field_v1(fields[12], "evidence scope")?,
                    authority_id: decode_hex_field_v1(fields[13], "evidence authority")?,
                    integrity: match fields[14] {
                        "verified" => SourceIntegrity::Verified,
                        "unverified" => SourceIntegrity::Unverified,
                        "invalid" => SourceIntegrity::Invalid,
                        other => return Err(format!("planner: unknown source integrity {other:?}")),
                    },
                    lifecycle: match fields[15] {
                        "active" => EvidenceLifecycle::Active,
                        "retracted" => EvidenceLifecycle::Retracted,
                        other => return Err(format!("planner: unknown evidence lifecycle {other:?}")),
                    },
                };
                if builder.evidence.insert(evidence_id.clone(), evidence).is_some() {
                    return Err(format!("planner: duplicate evidence ID {evidence_id:?} in {case_id:?}"));
                }
            }
            other => return Err(format!("planner: unknown record type {other:?} at line {line_number}")),
        }
    }

    if builders.is_empty() {
        return Err("planner: no cases found".into());
    }
    let mut cases = Vec::with_capacity(builders.len());
    for (case_id, builder) in builders {
        let query = builder.query.ok_or_else(|| format!("planner: case {case_id:?} has no CASE record"))?;
        if builder.targets.is_empty() || builder.declared_claims.is_empty() {
            return Err(format!("planner: case {case_id:?} must declare at least one target and claim"));
        }
        for target in &builder.targets {
            if !builder.declared_claims.contains(target) {
                return Err(format!("planner: target {target:?} is not a declared claim"));
            }
        }
        let evidence_ids = builder.evidence.keys().cloned().collect::<BTreeSet<_>>();
        let mut claims = builder.declared_claims.iter()
            .map(|id| (id.clone(), Vec::<ProofAlternative>::new()))
            .collect::<BTreeMap<_, _>>();
        for ((claim_id, _), alternative) in builder.alternatives {
            let alternatives = claims.get_mut(&claim_id)
                .ok_or_else(|| format!("planner: alternative belongs to undeclared claim {claim_id:?}"))?;
            for requirement in &alternative.requirements {
                match requirement {
                    ProofRequirement::Claim { claim_id: child, .. } => {
                        if !builder.declared_claims.contains(child) {
                            return Err(format!("planner: alternative references unknown child claim {child:?}"));
                        }
                    }
                    ProofRequirement::Evidence(evidence_id) => {
                        if !evidence_ids.contains(evidence_id) {
                            return Err(format!("planner: alternative references unknown evidence {evidence_id:?}"));
                        }
                    }
                }
            }
            alternatives.push(alternative);
        }
        for evidence in builder.evidence.values() {
            if !builder.declared_claims.contains(&evidence.subject_claim_id) {
                return Err(format!(
                    "planner: evidence {:?} references unknown subject claim {:?}",
                    evidence.id, evidence.subject_claim_id
                ));
            }
        }
        cases.push(PlannerCase {
            id: case_id,
            query,
            target_claim_ids: builder.targets.into_iter().collect(),
            claims: claims.into_iter().map(|(id, alternatives)| ClaimNode { id, alternatives }).collect(),
            evidence: builder.evidence.into_values().collect(),
        });
    }

    Ok(ParsedPlannerV1 {
        generator_revision,
        seed,
        split,
        config: GraphGenerationConfig { chain_depth },
        cases,
    })
}

fn parse_oracle_artifact_v1(bytes: &[u8]) -> Result<ParsedOracleV1, String> {
    let lines = canonical_lines_v1(bytes, "oracle")?;
    let header = lines[0].split('\t').collect::<Vec<_>>();
    expect_fields_v1(&header, 6, "oracle header", 1)?;
    if header[0] != "ORACLE_HEADER" || header[1] != "POAR-TSV/1" {
        return Err("oracle: unsupported or missing POAR-TSV/1 header".into());
    }
    if !valid_sha256_v1(header[2]) {
        return Err("oracle: planner digest is not canonical lowercase SHA-256".into());
    }
    let planner_sha256 = header[2].to_owned();
    let generator_revision = decode_hex_field_v1(header[3], "oracle generator revision")?;
    let seed = parse_u64_canonical_v1(header[4], "oracle seed")?;
    let split = parse_split_v1(header[5], "oracle split")?;

    let mut truths = BTreeMap::<String, BTreeMap<String, HiddenTruth>>::new();
    let mut gold_claims = BTreeMap::<(String, String), OracleState>::new();
    let mut gold_paths = BTreeMap::<(String, String, Polarity, String), BTreeSet<String>>::new();
    for (offset, line) in lines.iter().enumerate().skip(1) {
        let line_number = offset + 1;
        let fields = line.split('\t').collect::<Vec<_>>();
        let tag = fields.first().copied().unwrap_or("");
        if tag == "ORACLE_HEADER" {
            return Err(format!("oracle: duplicate header at line {line_number}"));
        }
        match tag {
            "TRUTH" => {
                expect_fields_v1(&fields, 4, "TRUTH", line_number)?;
                let case_id = decode_hex_field_v1(fields[1], "truth case ID")?;
                let evidence_id = decode_hex_field_v1(fields[2], "truth evidence ID")?;
                let truth = match fields[3] {
                    "supports_assertion" => HiddenTruth::SupportsAssertion,
                    "refutes_assertion" => HiddenTruth::RefutesAssertion,
                    "irrelevant" => HiddenTruth::Irrelevant,
                    other => return Err(format!("oracle: unknown truth label {other:?}")),
                };
                if truths.entry(case_id).or_default().insert(evidence_id.clone(), truth).is_some() {
                    return Err(format!("oracle: duplicate truth label for evidence {evidence_id:?}"));
                }
            }
            "GOLD_CLAIM" => {
                expect_fields_v1(&fields, 4, "GOLD_CLAIM", line_number)?;
                let key = (
                    decode_hex_field_v1(fields[1], "gold claim case ID")?,
                    decode_hex_field_v1(fields[2], "gold claim ID")?,
                );
                let state = match fields[3] {
                    "supported" => OracleState::Supported,
                    "refuted" => OracleState::Refuted,
                    "contested" => OracleState::Contested,
                    "unknown" => OracleState::Unknown,
                    other => return Err(format!("oracle: unknown gold claim state {other:?}")),
                };
                if gold_claims.insert(key, state).is_some() {
                    return Err("oracle: duplicate GOLD_CLAIM identity".into());
                }
            }
            "GOLD_PATH" => {
                if fields.len() < 6 {
                    return Err(format!("oracle: GOLD_PATH has too few fields at line {line_number}"));
                }
                let case_id = decode_hex_field_v1(fields[1], "gold path case ID")?;
                let claim_id = decode_hex_field_v1(fields[2], "gold path claim ID")?;
                let polarity = parse_polarity_v1(fields[3], "gold path polarity")?;
                let alternative_id = decode_hex_field_v1(fields[4], "gold path alternative ID")?;
                let count = parse_usize_canonical_v1(fields[5], "gold path evidence count")?;
                if fields.len() != 6 + count {
                    return Err(format!("oracle: GOLD_PATH count does not match its evidence fields at line {line_number}"));
                }
                let mut evidence_ids = BTreeSet::new();
                for encoded in &fields[6..] {
                    let evidence_id = decode_hex_field_v1(encoded, "gold path evidence ID")?;
                    if !evidence_ids.insert(evidence_id.clone()) {
                        return Err(format!("oracle: duplicate evidence {evidence_id:?} in GOLD_PATH"));
                    }
                }
                if gold_paths.insert((case_id, claim_id, polarity, alternative_id), evidence_ids).is_some() {
                    return Err("oracle: duplicate GOLD_PATH identity".into());
                }
            }
            other => return Err(format!("oracle: unknown record type {other:?} at line {line_number}")),
        }
    }
    if truths.is_empty() || gold_claims.is_empty() {
        return Err("oracle: truth and gold claim records are required".into());
    }
    Ok(ParsedOracleV1 {
        planner_sha256,
        generator_revision,
        seed,
        split,
        truths,
        gold_claims,
        gold_paths,
    })
}

pub fn encode_prediction_artifact_v1(
    identity: &PoarPredictionIdentityV1,
    packets: &[ProofPacket],
) -> Result<String, String> {
    if !valid_sha256_v1(&identity.planner_sha256)
        || !valid_sha256_v1(&identity.checkpoint_sha256)
        || !valid_sha256_v1(&identity.prompt_sha256)
        || !valid_sha256_v1(&identity.run_config_sha256)
        || identity.model_id.is_empty()
    {
        return Err("prediction identity requires non-empty model ID and four canonical lowercase SHA-256 digests".into());
    }
    let mut case_ids = BTreeSet::new();
    let mut records = Vec::<String>::new();
    for packet in packets {
        if packet.case_id.is_empty() || !case_ids.insert(packet.case_id.as_str()) {
            return Err("prediction packets must have unique, non-empty case IDs".into());
        }
        let case_id = hex_field(&packet.case_id);
        for (claim_id, disposition) in &packet.claim_dispositions {
            if claim_id.is_empty() {
                return Err("prediction claim ID must not be empty".into());
            }
            let (label, alternative) = match disposition {
                ClaimDisposition::Supported { alternative_id } => ("supported", hex_field(alternative_id)),
                ClaimDisposition::Refuted { alternative_id } => ("refuted", hex_field(alternative_id)),
                ClaimDisposition::Contested => ("contested", "-".into()),
                ClaimDisposition::Abstain => ("abstain", "-".into()),
            };
            records.push(format!(
                "DISPOSITION\t{}\t{}\t{}\t{}\n",
                case_id, hex_field(claim_id), label, alternative
            ).trim_end_matches('\n').to_owned());
        }
        let mut seen_retrieval = BTreeSet::new();
        for (index, evidence_id) in packet.retrieval_ranking.iter().enumerate() {
            if evidence_id.is_empty() || !seen_retrieval.insert(evidence_id) {
                return Err("retrieval ranking must contain unique, non-empty evidence IDs".into());
            }
            records.push(format!(
                "RETRIEVED\t{}\t{}\t{}",
                case_id, index + 1, hex_field(evidence_id)
            ));
        }
        for evidence_id in &packet.cited_evidence_ids {
            if evidence_id.is_empty() {
                return Err("citation evidence ID must not be empty".into());
            }
            records.push(format!("CITATION\t{}\t{}", case_id, hex_field(evidence_id)));
        }
    }
    records.sort_unstable();
    let mut output = format!(
        "PRED_HEADER\tPOAR-PRED/1\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        identity.planner_sha256,
        hex_field(&identity.model_id),
        identity.checkpoint_sha256,
        identity.prompt_sha256,
        identity.run_config_sha256,
        identity.seed,
        identity.split.label()
    );
    for record in records {
        output.push_str(&record);
        output.push('\n');
    }
    Ok(output)
}

fn parse_predictions_v1(
    bytes: &[u8],
    planner: &ParsedPlannerV1,
    planner_sha256: &str,
) -> Result<ParsedPredictionsV1, String> {
    let lines = canonical_lines_v1(bytes, "predictions")?;
    let header = lines[0].split('\t').collect::<Vec<_>>();
    expect_fields_v1(&header, 9, "prediction header", 1)?;
    if header[0] != "PRED_HEADER" || header[1] != "POAR-PRED/1" {
        return Err("predictions: unsupported or missing POAR-PRED/1 header".into());
    }
    if !valid_sha256_v1(header[2]) || header[2] != planner_sha256 {
        return Err("predictions: planner digest does not match exact planner bytes".into());
    }
    for (index, label) in [(4, "checkpoint"), (5, "prompt"), (6, "run config")] {
        if !valid_sha256_v1(header[index]) {
            return Err(format!("predictions: {label} digest is not canonical lowercase SHA-256"));
        }
    }
    let identity = PoarPredictionIdentityV1 {
        planner_sha256: header[2].to_owned(),
        model_id: decode_hex_field_v1(header[3], "prediction model ID")?,
        checkpoint_sha256: header[4].to_owned(),
        prompt_sha256: header[5].to_owned(),
        run_config_sha256: header[6].to_owned(),
        seed: parse_u64_canonical_v1(header[7], "prediction seed")?,
        split: parse_split_v1(header[8], "prediction split")?,
    };
    if identity.seed != planner.seed || identity.split != planner.split {
        return Err("predictions: seed/split identity does not match planner artifact".into());
    }

    let case_map = planner.cases.iter().map(|case| (case.id.as_str(), case)).collect::<BTreeMap<_, _>>();
    let mut builders = BTreeMap::<String, PredictionCaseBuilderV1>::new();
    for (offset, line) in lines.iter().enumerate().skip(1) {
        let line_number = offset + 1;
        let fields = line.split('\t').collect::<Vec<_>>();
        let tag = fields.first().copied().unwrap_or("");
        if tag == "PRED_HEADER" {
            return Err(format!("predictions: duplicate header at line {line_number}"));
        }
        let case_id = decode_hex_field_v1(
            fields.get(1).ok_or_else(|| format!("predictions: missing case ID at line {line_number}"))?,
            "prediction case ID"
        )?;
        let case = case_map.get(case_id.as_str()).copied()
            .ok_or_else(|| format!("predictions: unknown case {case_id:?}"))?;
        let builder = builders.entry(case_id.clone()).or_default();

        match tag {
            "DISPOSITION" => {
                expect_fields_v1(&fields, 5, "DISPOSITION", line_number)?;
                let claim_id = decode_hex_field_v1(fields[2], "prediction claim ID")?;
                if !case.target_claim_ids.contains(&claim_id) {
                    return Err(format!("predictions: claim {claim_id:?} is not a target of case {case_id:?}"));
                }
                let disposition = match fields[3] {
                    "supported" | "refuted" => {
                        let alternative_id = decode_hex_field_v1(fields[4], "prediction alternative ID")?;
                        let claim = case.claims.iter().find(|node| node.id == claim_id)
                            .ok_or_else(|| format!("predictions: unknown claim {claim_id:?}"))?;
                        let alternative = claim.alternatives.iter().find(|alt| alt.id == alternative_id)
                            .ok_or_else(|| format!("predictions: unknown alternative {alternative_id:?}"))?;
                        if fields[3] == "supported" {
                            if alternative.conclusion != Polarity::Supports {
                                return Err(format!("predictions: supported disposition references refuting alternative {alternative_id:?}"));
                            }
                            ClaimDisposition::Supported { alternative_id }
                        } else {
                            if alternative.conclusion != Polarity::Refutes {
                                return Err(format!("predictions: refuted disposition references supporting alternative {alternative_id:?}"));
                            }
                            ClaimDisposition::Refuted { alternative_id }
                        }
                    }
                    "contested" => {
                        if fields[4] != "-" {
                            return Err("predictions: contested disposition must use '-' for alternative".into());
                        }
                        ClaimDisposition::Contested
                    }
                    "abstain" => {
                        if fields[4] != "-" {
                            return Err("predictions: abstain disposition must use '-' for alternative".into());
                        }
                        ClaimDisposition::Abstain
                    }
                    other => return Err(format!("predictions: unknown disposition {other:?}")),
                };
                if builder.dispositions.insert(claim_id.clone(), disposition).is_some() {
                    return Err(format!("predictions: duplicate disposition for {claim_id:?}"));
                }
            }
            "RETRIEVED" => {
                expect_fields_v1(&fields, 4, "RETRIEVED", line_number)?;
                let rank = parse_u64_canonical_v1(fields[2], "retrieval rank")?;
                if rank == 0 {
                    return Err("predictions: retrieval ranks start at 1".into());
                }
                let evidence_id = decode_hex_field_v1(fields[3], "retrieved evidence ID")?;
                if !case.evidence.iter().any(|evidence| evidence.id == evidence_id) {
                    return Err(format!("predictions: retrieval references unknown evidence {evidence_id:?}"));
                }
                if !builder.retrieval_ids.insert(evidence_id.clone())
                    || builder.retrieval.insert(rank, evidence_id).is_some()
                {
                    return Err(format!("predictions: duplicate retrieval ID or rank in case {case_id:?}"));
                }
            }
            "CITATION" => {
                expect_fields_v1(&fields, 3, "CITATION", line_number)?;
                let evidence_id = decode_hex_field_v1(fields[2], "citation evidence ID")?;
                if !case.evidence.iter().any(|evidence| evidence.id == evidence_id) {
                    return Err(format!("predictions: citation references unknown evidence {evidence_id:?}"));
                }
                if !builder.citations.insert(evidence_id.clone()) {
                    return Err(format!("predictions: duplicate citation {evidence_id:?}"));
                }
            }
            other => return Err(format!("predictions: unknown record type {other:?} at line {line_number}")),
        }
    }

    let mut packets = Vec::with_capacity(planner.cases.len());
    for case in &planner.cases {
        let builder = builders.remove(&case.id)
            .ok_or_else(|| format!("predictions: no records for case {:?}", case.id))?;
        if builder.dispositions.len() != case.target_claim_ids.len()
            || case.target_claim_ids.iter().any(|target| !builder.dispositions.contains_key(target))
        {
            return Err(format!("predictions: every target in case {:?} needs exactly one disposition", case.id));
        }
        for (expected, actual) in (1_u64..).zip(builder.retrieval.keys().copied()) {
            if expected != actual {
                return Err(format!("predictions: retrieval ranks in case {:?} must be contiguous from 1", case.id));
            }
        }
        packets.push(ProofPacket {
            case_id: case.id.clone(),
            claim_dispositions: builder.dispositions,
            retrieval_ranking: builder.retrieval.into_values().collect(),
            cited_evidence_ids: builder.citations,
        });
    }
    if !builders.is_empty() {
        return Err("predictions: unconsumed or unknown case records".into());
    }
    Ok(ParsedPredictionsV1 { identity, packets })
}

fn validate_oracle_labels_v1(
    planner: &ParsedPlannerV1,
    oracle: &ParsedOracleV1,
) -> Result<(), String> {
    let mut expected_truths = BTreeMap::<String, BTreeSet<String>>::new();
    for case in &planner.cases {
        expected_truths.insert(
            case.id.clone(),
            case.evidence.iter().map(|evidence| evidence.id.clone()).collect()
        );
    }
    if oracle.truths.len() != expected_truths.len() {
        return Err("oracle: case coverage does not match planner".into());
    }
    for (case_id, expected_ids) in &expected_truths {
        let labels = oracle.truths.get(case_id)
            .ok_or_else(|| format!("oracle: missing truth labels for case {case_id:?}"))?;
        let actual_ids = labels.keys().cloned().collect::<BTreeSet<_>>();
        if &actual_ids != expected_ids {
            return Err(format!("oracle: truth evidence IDs do not exactly match planner case {case_id:?}"));
        }
    }

    let mut expected_claims = BTreeMap::<(String, String), OracleState>::new();
    let mut expected_paths = BTreeMap::<(String, String, Polarity, String), BTreeSet<String>>::new();
    for case in &planner.cases {
        let labels = oracle.truths.get(&case.id).expect("truth map validated above");
        let mut memo = BTreeMap::new();
        for claim_id in &case.target_claim_ids {
            let evaluation = resolve_claim(case, labels, claim_id, &mut BTreeSet::new(), &mut memo);
            expected_claims.insert((case.id.clone(), claim_id.clone()), evaluation.state());
            for (alternative_id, witness) in &evaluation.support {
                expected_paths.insert(
                    (case.id.clone(), claim_id.clone(), Polarity::Supports, alternative_id.clone()),
                    witness.evidence_ids.clone()
                );
            }
            for (alternative_id, witness) in &evaluation.refute {
                expected_paths.insert(
                    (case.id.clone(), claim_id.clone(), Polarity::Refutes, alternative_id.clone()),
                    witness.evidence_ids.clone()
                );
            }
        }
    }
    if expected_claims != oracle.gold_claims {
        return Err("oracle: GOLD_CLAIM labels are incomplete or inconsistent with its truth records".into());
    }
    if expected_paths != oracle.gold_paths {
        return Err("oracle: GOLD_PATH witnesses are incomplete or inconsistent with its truth records".into());
    }
    Ok(())
}

/// Evaluate three frozen artifacts after inference. The caller supplies the
/// SHA-256 computed from the exact planner bytes (the CLI uses sha256sum).
/// This function validates byte-format, digest, schema, references, and oracle
/// consistency before invoking the graph scorer.
pub fn evaluate_poar_artifacts_v1(
    planner_bytes: &[u8],
    oracle_bytes: &[u8],
    prediction_bytes: &[u8],
    actual_planner_sha256: &str,
) -> Result<PoarArtifactEvaluationV1, String> {
    if !valid_sha256_v1(actual_planner_sha256) {
        return Err("evaluator: supplied planner digest is not canonical lowercase SHA-256".into());
    }
    let planner = parse_planner_artifact_v1(planner_bytes)?;
    let oracle = parse_oracle_artifact_v1(oracle_bytes)?;
    if oracle.planner_sha256 != actual_planner_sha256 {
        return Err("oracle: planner digest does not match exact planner bytes".into());
    }
    if oracle.generator_revision != planner.generator_revision
        || oracle.seed != planner.seed
        || oracle.split != planner.split
    {
        return Err("oracle: generator, seed, or split identity does not match planner".into());
    }
    validate_oracle_labels_v1(&planner, &oracle)?;
    let predictions = parse_predictions_v1(prediction_bytes, &planner, actual_planner_sha256)?;
    let suite = GraphSuite {
        planner_cases: planner.cases,
        seed: planner.seed,
        split: planner.split,
        config: planner.config,
        oracle: oracle.truths,
    };
    let report = suite.evaluate(&predictions.packets);
    if report.duplicate_prediction_packets != 0 || report.unknown_prediction_packets != 0 {
        return Err("evaluator: parser/scorer disagreement on prediction packet identities".into());
    }
    Ok(PoarArtifactEvaluationV1 {
        prediction_identity: predictions.identity,
        report,
    })
}

#[cfg(test)]
mod artifact_evaluator_tests_v1 {
    use super::*;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn artifacts() -> (String, String, String, String) {
        let suite = generate_poar_graph_suite(1729, DatasetSplit::Test);
        let generator = "test-generator-revision";
        let planner = encode_planner_artifact_v1(&suite, generator).expect("planner encode");
        let oracle = encode_oracle_artifact_v1(&suite, generator, DIGEST).expect("oracle encode");
        let packets = suite.planner_cases().iter().map(ProofPacket::abstain_all).collect::<Vec<_>>();
        let identity = PoarPredictionIdentityV1 {
            planner_sha256: DIGEST.into(),
            model_id: "test-model".into(),
            checkpoint_sha256: DIGEST.into(),
            prompt_sha256: DIGEST.into(),
            run_config_sha256: DIGEST.into(),
            seed: 1729,
            split: DatasetSplit::Test,
        };
        let predictions = encode_prediction_artifact_v1(&identity, &packets).expect("prediction encode");
        (planner, oracle, predictions, DIGEST.into())
    }

    #[test]
    fn frozen_artifacts_round_trip_and_score_in_a_fresh_evaluator_api() {
        let (planner, oracle, predictions, digest) = artifacts();
        let evaluated = evaluate_poar_artifacts_v1(
            planner.as_bytes(), oracle.as_bytes(), predictions.as_bytes(), &digest
        ).expect("valid frozen artifacts");
        assert_eq!(evaluated.report.case_count(), 16);
        assert_eq!(evaluated.report.duplicate_prediction_packets, 0);
        assert_eq!(evaluated.report.unknown_prediction_packets, 0);
        assert_eq!(evaluated.prediction_identity.model_id, "test-model");
        assert_eq!(evaluated.report.case_scores.iter().map(|score| score.target_claims).sum::<usize>(), 18);
    }

    #[test]
    fn parser_rejects_missing_final_lf_crlf_bom_and_empty_records() {
        let (planner, oracle, predictions, digest) = artifacts();
        let no_lf = planner.trim_end_matches('\n').as_bytes().to_vec();
        assert!(evaluate_poar_artifacts_v1(&no_lf, oracle.as_bytes(), predictions.as_bytes(), &digest).is_err());
        let crlf = planner.replace('\n', "\r\n");
        assert!(evaluate_poar_artifacts_v1(crlf.as_bytes(), oracle.as_bytes(), predictions.as_bytes(), &digest).is_err());
        let bom = format!("\u{feff}{planner}");
        assert!(evaluate_poar_artifacts_v1(bom.as_bytes(), oracle.as_bytes(), predictions.as_bytes(), &digest).is_err());
        let blank = planner.replacen('\n', "\n\n", 1);
        assert!(evaluate_poar_artifacts_v1(blank.as_bytes(), oracle.as_bytes(), predictions.as_bytes(), &digest).is_err());
    }

    #[test]
    fn planner_oracle_and_prediction_digest_bindings_fail_closed() {
        let (planner, oracle, predictions, digest) = artifacts();
        let wrong = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        assert!(evaluate_poar_artifacts_v1(planner.as_bytes(), oracle.as_bytes(), predictions.as_bytes(), wrong).is_err());
        let wrong_pred = predictions.replace(&digest, wrong);
        assert!(evaluate_poar_artifacts_v1(planner.as_bytes(), oracle.as_bytes(), wrong_pred.as_bytes(), &digest).is_err());
        let wrong_oracle = oracle.replace(&digest, wrong);
        assert!(evaluate_poar_artifacts_v1(planner.as_bytes(), wrong_oracle.as_bytes(), predictions.as_bytes(), &digest).is_err());
    }

    #[test]
    fn prediction_parser_rejects_missing_or_duplicate_target_dispositions() {
        let (planner, oracle, predictions, digest) = artifacts();
        let without_first = predictions.lines().enumerate()
            .filter(|(index, _)| *index != 1)
            .map(|(_, line)| line).collect::<Vec<_>>().join("\n") + "\n";
        assert!(evaluate_poar_artifacts_v1(planner.as_bytes(), oracle.as_bytes(), without_first.as_bytes(), &digest).is_err());
        let first_record = predictions.lines().nth(1).expect("first prediction record");
        let duplicate = format!("{predictions}{first_record}\n");
        assert!(evaluate_poar_artifacts_v1(planner.as_bytes(), oracle.as_bytes(), duplicate.as_bytes(), &digest).is_err());
    }

    #[test]
    fn prediction_parser_rejects_retrieval_rank_gaps_and_unknown_evidence() {
        let (planner, oracle, predictions, digest) = artifacts();
        let case_id = hex_field("and-chain-1729-test");
        let rank_gap = format!("{predictions}RETRIEVED\t{case_id}\t2\t{}\n", hex_field("e-leaf"));
        assert!(evaluate_poar_artifacts_v1(planner.as_bytes(), oracle.as_bytes(), rank_gap.as_bytes(), &digest).is_err());
        let unknown = format!("{predictions}CITATION\t{case_id}\t{}\n", hex_field("not-in-planner"));
        assert!(evaluate_poar_artifacts_v1(planner.as_bytes(), oracle.as_bytes(), unknown.as_bytes(), &digest).is_err());
    }

    #[test]
    fn tampered_gold_claim_and_gold_path_artifacts_are_rejected() {
        let (planner, oracle, predictions, digest) = artifacts();
        let tampered_claim = oracle.replacen("GOLD_CLAIM", "GOLD_CLAIM", 1)
            .replacen("\tsupported\n", "\tunknown\n", 1);
        assert!(evaluate_poar_artifacts_v1(planner.as_bytes(), tampered_claim.as_bytes(), predictions.as_bytes(), &digest).is_err());
        let tampered_path = oracle.replacen("GOLD_PATH", "GOLD_PATH", 1)
            .replace("GOLD_PATH\t", "GOLD_PATH\t");
        let malformed_path = tampered_path.lines().map(|line| {
            if line.starts_with("GOLD_PATH\t") { line.trim_end().to_owned() + "\t00" } else { line.to_owned() }
        }).collect::<Vec<_>>().join("\n") + "\n";
        assert!(evaluate_poar_artifacts_v1(planner.as_bytes(), malformed_path.as_bytes(), predictions.as_bytes(), &digest).is_err());
    }

    #[test]
    fn frozen_predictions_are_scored_for_false_support_and_missing_proof_citations() {
        let suite = generate_poar_graph_suite(1729, DatasetSplit::Test);
        let planner = encode_planner_artifact_v1(&suite, "test-generator-revision").expect("planner");
        let oracle = encode_oracle_artifact_v1(&suite, "test-generator-revision", DIGEST).expect("oracle");
        let case = suite.planner_cases().iter().find(|case| case.id.ends_with("or-current-alternative")).expect("case");
        let packet = ProofPacket {
            case_id: case.id.clone(),
            claim_dispositions: BTreeMap::from([(
                "claim".into(),
                ClaimDisposition::Supported { alternative_id: "current-path".into() }
            )]),
            retrieval_ranking: vec![],
            cited_evidence_ids: BTreeSet::new(),
        };
        let mut packets = suite.planner_cases().iter().map(ProofPacket::abstain_all).collect::<Vec<_>>();
        let target = packets.iter_mut().find(|packet| packet.case_id == case.id).expect("packet");
        *target = packet;
        let identity = PoarPredictionIdentityV1 {
            planner_sha256: DIGEST.into(), model_id: "test-model".into(),
            checkpoint_sha256: DIGEST.into(), prompt_sha256: DIGEST.into(),
            run_config_sha256: DIGEST.into(), seed: 1729, split: DatasetSplit::Test,
        };
        let predictions = encode_prediction_artifact_v1(&identity, &packets).expect("predictions");
        let result = evaluate_poar_artifacts_v1(
            planner.as_bytes(), oracle.as_bytes(), predictions.as_bytes(), DIGEST
        ).expect("evaluation");
        let score = result.report.case_scores.iter().find(|score| score.case_id == case.id).expect("case score");
        assert_eq!(score.correct_dispositions, 1);
        assert_eq!(score.missing_required_citations, 1);
        assert!(!result.report.all_hard_invariants_hold());
    }
}
