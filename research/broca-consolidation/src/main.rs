use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::process;

const ACTIONS: [Action; 6] = [
    Action::Raw,
    Action::Provisional,
    Action::Summary,
    Action::Skill,
    Action::Rule,
    Action::Retire,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    Raw,
    Provisional,
    Summary,
    Skill,
    Rule,
    Retire,
}

impl Action {
    fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Provisional => "provisional",
            Self::Summary => "summary",
            Self::Skill => "skill",
            Self::Rule => "rule",
            Self::Retire => "retire",
        }
    }

    fn storage_cost(self) -> i32 {
        match self {
            Self::Raw => 5,
            Self::Provisional => 3,
            Self::Summary => 2,
            Self::Skill => 3,
            Self::Rule => 2,
            Self::Retire => 0,
        }
    }

    fn is_high_level_promotion(self) -> bool {
        matches!(self, Self::Summary | Self::Skill | Self::Rule)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CandidateKind {
    EpisodicFact,
    StableCluster,
    Procedure,
    Invariant,
    PoisonedRule,
    ObsoleteFact,
}

impl CandidateKind {
    const ALL: [Self; 6] = [
        Self::EpisodicFact,
        Self::StableCluster,
        Self::Procedure,
        Self::Invariant,
        Self::PoisonedRule,
        Self::ObsoleteFact,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::EpisodicFact => "episodic_fact",
            Self::StableCluster => "stable_cluster",
            Self::Procedure => "procedure",
            Self::Invariant => "invariant",
            Self::PoisonedRule => "poisoned_rule",
            Self::ObsoleteFact => "obsolete_fact",
        }
    }

    fn load_bearing(self) -> bool {
        matches!(
            self,
            Self::EpisodicFact | Self::StableCluster | Self::Procedure | Self::Invariant
        )
    }

    fn poisoned_or_obsolete(self) -> bool {
        matches!(self, Self::PoisonedRule | Self::ObsoleteFact)
    }

    fn task_reward(self, action: Action) -> i32 {
        match self {
            Self::EpisodicFact => match action {
                Action::Raw => 6,
                Action::Provisional => 5,
                Action::Summary => 2,
                Action::Skill => -1,
                Action::Rule => -2,
                Action::Retire => -6,
            },
            Self::StableCluster => match action {
                Action::Raw => 7,
                Action::Provisional => 6,
                Action::Summary => 10,
                Action::Skill => 6,
                Action::Rule => 5,
                Action::Retire => -7,
            },
            Self::Procedure => match action {
                Action::Raw => 6,
                Action::Provisional => 5,
                Action::Summary => 6,
                Action::Skill => 12,
                Action::Rule => 8,
                Action::Retire => -8,
            },
            Self::Invariant => match action {
                Action::Raw => 7,
                Action::Provisional => 6,
                Action::Summary => 9,
                Action::Skill => 8,
                Action::Rule => 13,
                Action::Retire => -9,
            },
            Self::PoisonedRule => match action {
                Action::Raw => 1,
                Action::Provisional => -2,
                Action::Summary => -6,
                Action::Skill => -12,
                Action::Rule => -15,
                Action::Retire => 0,
            },
            Self::ObsoleteFact => match action {
                Action::Raw => -1,
                Action::Provisional => -1,
                Action::Summary => -3,
                Action::Skill => -5,
                Action::Rule => -6,
                Action::Retire => 0,
            },
        }
    }

    fn net_utility(self, action: Action) -> i32 {
        self.task_reward(action) - action.storage_cost()
    }

    fn oracle_action(self) -> Action {
        let mut best = ACTIONS[0];
        let mut best_utility = self.net_utility(best);
        for action in ACTIONS.into_iter().skip(1) {
            let utility = self.net_utility(action);
            if utility > best_utility {
                best = action;
                best_utility = utility;
            }
        }
        best
    }
}

/// Observable evidence available to policies. It deliberately contains no
/// candidate kind, future-task label, oracle action, or future utility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ObservedFeatures {
    repeat_count: u8,
    distinct_episodes: u8,
    contradiction_count: u8,
    recency: u8,
    correction_seen: bool,
    independent_successes: u8,
    provenance_diversity: u8,
}

/// Evaluator-only labels. This type is never passed to choose_action().
#[derive(Clone, Copy, Debug)]
struct GroundTruth {
    kind: CandidateKind,
    load_bearing: bool,
    poisoned_or_obsolete: bool,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    id: u64,
    features: ObservedFeatures,
    truth: GroundTruth,
}

impl Candidate {
    fn new(id: u64, kind: CandidateKind) -> Self {
        let features = match kind {
            CandidateKind::EpisodicFact => ObservedFeatures {
                repeat_count: 1,
                distinct_episodes: 1,
                contradiction_count: 0,
                recency: 9,
                correction_seen: false,
                independent_successes: 1,
                provenance_diversity: 1,
            },
            CandidateKind::StableCluster => ObservedFeatures {
                repeat_count: 6,
                distinct_episodes: 5,
                contradiction_count: 0,
                recency: 8,
                correction_seen: false,
                independent_successes: 3,
                provenance_diversity: 4,
            },
            CandidateKind::Procedure => ObservedFeatures {
                repeat_count: 5,
                distinct_episodes: 4,
                contradiction_count: 0,
                recency: 7,
                correction_seen: false,
                independent_successes: 4,
                provenance_diversity: 3,
            },
            CandidateKind::Invariant => ObservedFeatures {
                repeat_count: 7,
                distinct_episodes: 6,
                contradiction_count: 0,
                recency: 8,
                correction_seen: false,
                independent_successes: 6,
                provenance_diversity: 5,
            },
            CandidateKind::PoisonedRule => ObservedFeatures {
                repeat_count: 5,
                distinct_episodes: 2,
                contradiction_count: 1,
                recency: 9,
                correction_seen: true,
                independent_successes: 4,
                provenance_diversity: 1,
            },
            CandidateKind::ObsoleteFact => ObservedFeatures {
                repeat_count: 6,
                distinct_episodes: 5,
                contradiction_count: 3,
                recency: 1,
                correction_seen: true,
                independent_successes: 4,
                provenance_diversity: 4,
            },
        };
        Self {
            id,
            features,
            truth: GroundTruth {
                kind,
                load_bearing: kind.load_bearing(),
                poisoned_or_obsolete: kind.poisoned_or_obsolete(),
            },
        }
    }

    /// Add seed-controlled noise to observable evidence without changing the
    /// evaluator-owned kind or the synthetic utility contract.
    fn with_feature_noise(mut self, rng: &mut DeterministicRng) -> Self {
        self.features.repeat_count = jitter(self.features.repeat_count, 2, rng);
        self.features.distinct_episodes = jitter(self.features.distinct_episodes, 2, rng);
        self.features.contradiction_count = jitter(self.features.contradiction_count, 1, rng);
        self.features.recency = jitter(self.features.recency, 2, rng);
        self.features.independent_successes = jitter(self.features.independent_successes, 2, rng);
        self.features.provenance_diversity = jitter(self.features.provenance_diversity, 2, rng);
        // Simulate one imperfectly observed correction signal without leaking
        // whether a candidate is actually poisoned or obsolete.
        if rng.next_u64() % 7 == 0 {
            self.features.correction_seen = !self.features.correction_seen;
        }
        self
    }
}

fn jitter(value: u8, magnitude: u8, rng: &mut DeterministicRng) -> u8 {
    let span = u64::from(magnitude) * 2 + 1;
    let delta = (rng.next_u64() % span) as i16 - i16::from(magnitude);
    (i16::from(value) + delta).clamp(0, 10) as u8
}

#[derive(Clone, Copy, Debug)]
enum Policy {
    AlwaysRaw,
    FixedInterval,
    DedupOnly,
    FeatureThreshold,
    TypedLifecycle,
    RetireEverything,
    PromoteEverything,
}

struct PolicySpec {
    name: &'static str,
    role: &'static str,
    policy: Policy,
}

const POLICIES: [PolicySpec; 7] = [
    PolicySpec { name: "always_raw", role: "baseline", policy: Policy::AlwaysRaw },
    PolicySpec { name: "repeat_threshold_summary", role: "baseline", policy: Policy::FixedInterval },
    PolicySpec { name: "dedup_only", role: "baseline", policy: Policy::DedupOnly },
    PolicySpec { name: "feature_threshold", role: "baseline", policy: Policy::FeatureThreshold },
    PolicySpec { name: "typed_lifecycle", role: "candidate", policy: Policy::TypedLifecycle },
    PolicySpec { name: "retire_everything", role: "anti_control", policy: Policy::RetireEverything },
    PolicySpec { name: "promote_everything", role: "mutation_control", policy: Policy::PromoteEverything },
];

/// The policy API accepts only observable features. Oracle labels and candidate
/// kind are kept in the evaluator and cannot be passed through this function.
fn choose_action(policy: Policy, f: &ObservedFeatures) -> Action {
    match policy {
        Policy::AlwaysRaw => Action::Raw,
        Policy::FixedInterval => {
            if f.repeat_count >= 3 { Action::Summary } else { Action::Raw }
        }
        Policy::DedupOnly => {
            if f.repeat_count >= 2 { Action::Summary } else { Action::Raw }
        }
        Policy::FeatureThreshold => {
            if f.correction_seen || f.contradiction_count >= 2 || f.recency <= 1 {
                Action::Retire
            } else if f.provenance_diversity >= 5
                && f.repeat_count >= 5
                && f.independent_successes >= 4
            {
                Action::Rule
            } else if f.repeat_count >= 3 {
                Action::Summary
            } else {
                Action::Provisional
            }
        }
        Policy::TypedLifecycle => {
            if f.correction_seen && f.contradiction_count > 0 {
                Action::Retire
            } else if f.distinct_episodes >= 5
                && f.provenance_diversity >= 5
                && f.independent_successes >= 5
            {
                Action::Rule
            } else if f.repeat_count >= 4
                && f.distinct_episodes >= 3
                && f.independent_successes >= 4
            {
                Action::Skill
            } else if f.repeat_count >= 4
                && f.distinct_episodes >= 4
                && f.independent_successes <= 3
            {
                Action::Summary
            } else if f.repeat_count >= 3 {
                Action::Summary
            } else {
                Action::Provisional
            }
        }
        Policy::RetireEverything => Action::Retire,
        Policy::PromoteEverything => Action::Rule,
    }
}

#[derive(Clone, Debug)]
struct Decision {
    candidate_id: u64,
    policy: &'static str,
    role: &'static str,
    kind: CandidateKind,
    features: ObservedFeatures,
    action: Action,
    oracle_action: Action,
    net_utility: i32,
    oracle_utility: i32,
    regret: i32,
    storage_units: i32,
    load_bearing: bool,
    covered: bool,
    false_promotion: bool,
    destructive_forgetting: bool,
}

impl Decision {
    fn from_candidate(candidate: &Candidate, spec: &PolicySpec, action: Action) -> Self {
        let kind = candidate.truth.kind;
        let utility = kind.net_utility(action);
        let oracle_action = kind.oracle_action();
        let oracle_utility = kind.net_utility(oracle_action);
        let covered = candidate.truth.load_bearing && action != Action::Retire;
        let false_promotion =
            candidate.truth.poisoned_or_obsolete && action.is_high_level_promotion();
        let destructive_forgetting = candidate.truth.load_bearing && action == Action::Retire;
        Self {
            candidate_id: candidate.id,
            policy: spec.name,
            role: spec.role,
            kind,
            features: candidate.features,
            action,
            oracle_action,
            net_utility: utility,
            oracle_utility,
            regret: oracle_utility - utility,
            storage_units: action.storage_cost(),
            load_bearing: candidate.truth.load_bearing,
            covered,
            false_promotion,
            destructive_forgetting,
        }
    }
}

#[derive(Default, Clone, Debug)]
struct PolicySummary {
    candidates: usize,
    total_utility: i64,
    total_regret: i64,
    total_storage_units: i64,
    load_bearing_candidates: usize,
    covered_load_bearing: usize,
    false_promotions: usize,
    destructive_forgetting: usize,
}

impl PolicySummary {
    fn add(&mut self, row: &Decision) {
        self.candidates += 1;
        self.total_utility += i64::from(row.net_utility);
        self.total_regret += i64::from(row.regret);
        self.total_storage_units += i64::from(row.storage_units);
        if row.load_bearing { self.load_bearing_candidates += 1; }
        if row.covered { self.covered_load_bearing += 1; }
        if row.false_promotion { self.false_promotions += 1; }
        if row.destructive_forgetting { self.destructive_forgetting += 1; }
    }

    fn mean_regret(&self) -> f64 {
        if self.candidates == 0 { 0.0 } else { self.total_regret as f64 / self.candidates as f64 }
    }

    fn load_bearing_coverage(&self) -> f64 {
        if self.load_bearing_candidates == 0 {
            0.0
        } else {
            self.covered_load_bearing as f64 / self.load_bearing_candidates as f64
        }
    }
}

fn build_fixture(seed: u64, count: usize) -> Vec<Candidate> {
    let mut rng = DeterministicRng::new(seed);
    let mut kinds = Vec::with_capacity(count);
    for index in 0..count {
        kinds.push(CandidateKind::ALL[index % CandidateKind::ALL.len()]);
    }

    // Shuffle ordering only; the same seed yields the same complete fixture.
    for index in (1..kinds.len()).rev() {
        let other = (rng.next_u64() % (index as u64 + 1)) as usize;
        kinds.swap(index, other);
    }

    kinds.into_iter().enumerate()
        .map(|(index, kind)| {
            Candidate::new(index as u64 + 1, kind).with_feature_noise(&mut rng)
        })
        .collect()
}

struct DeterministicRng { state: u64 }

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

#[derive(Clone, Debug)]
struct Config {
    seed: u64,
    candidates: usize,
    source_sha: String,
}

impl Default for Config {
    fn default() -> Self {
        Self { seed: 17, candidates: 120, source_sha: String::new() }
    }
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut config = Config::default();
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--help" {
            return Err("use --help as the only argument".into());
        }
        if index + 1 >= args.len() {
            return Err(format!("missing value for {}", args[index]));
        }
        let name = args[index].as_str();
        let value = args[index + 1].as_str();
        match name {
            "--seed" => config.seed = value.parse().map_err(|_| format!("invalid --seed: {value}"))?,
            "--candidates" => config.candidates = value.parse().map_err(|_| format!("invalid --candidates: {value}"))?,
            "--source-sha" => config.source_sha = value.to_owned(),
            _ => return Err(format!("unknown argument: {name}")),
        }
        index += 2;
    }

    if !(12..=100_000).contains(&config.candidates) {
        return Err("--candidates must be in 12..=100000".into());
    }
    if config.source_sha.len() != 40 || !config.source_sha.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("--source-sha is required and must be a 40-character Git commit SHA".into());
    }
    Ok(config)
}

fn validate_self_checks(candidates: &[Candidate], rows: &[Decision]) -> Result<(), String> {
    if candidates.len() < 12 {
        return Err("fixture too small for lifecycle categories".into());
    }
    for candidate in candidates {
        let expected = candidate.truth.kind.oracle_action();
        for action in ACTIONS {
            if candidate.truth.kind.net_utility(action) > candidate.truth.kind.net_utility(expected) {
                return Err(format!("oracle failed to maximize utility for candidate {}", candidate.id));
            }
        }
    }

    let expected_ids: BTreeSet<u64> =
        candidates.iter().map(|candidate| candidate.id).collect();
    if expected_ids.len() != candidates.len() {
        return Err("fixture contains duplicate candidate IDs".into());
    }

    let summaries = summarize(rows);
    for spec in &POLICIES {
        let summary = summaries
            .get(spec.name)
            .ok_or_else(|| format!("missing policy {}", spec.name))?;
        if summary.candidates != candidates.len() {
            return Err(format!(
                "policy {} produced {} decisions for {} candidates",
                spec.name, summary.candidates, candidates.len()
            ));
        }
        let policy_ids: BTreeSet<u64> = rows
            .iter()
            .filter(|row| row.policy == spec.name)
            .map(|row| row.candidate_id)
            .collect();
        if policy_ids != expected_ids {
            return Err(format!(
                "policy {} has missing or duplicate candidate decisions",
                spec.name
            ));
        }
    }
    let retire = summaries.get("retire_everything")
        .ok_or_else(|| "missing retire-everything anti-control".to_owned())?;
    if retire.covered_load_bearing != 0
        || retire.destructive_forgetting != retire.load_bearing_candidates
    {
        return Err("retire-everything anti-control did not expose degenerate forgetting".into());
    }

    let promote = summaries.get("promote_everything")
        .ok_or_else(|| "missing promote-everything mutation control".to_owned())?;
    if promote.false_promotions == 0 {
        return Err("promote-everything mutation control was not detected".into());
    }

    // Policy quality is an experimental outcome, not receipt validity. A
    // poorly performing candidate policy must still produce a valid report;
    // only evaluator/oracle and mutation-control failures invalidate the run.
    Ok(())
}

fn summarize(rows: &[Decision]) -> BTreeMap<&'static str, PolicySummary> {
    let mut summaries = BTreeMap::new();
    for row in rows {
        summaries.entry(row.policy).or_insert_with(PolicySummary::default).add(row);
    }
    summaries
}

fn fixture_fingerprint(candidates: &[Candidate]) -> String {
    // Stable, explicitly non-cryptographic FNV-1a fingerprint over the canonical
    // generator record. This is a receipt aid, not a dataset-integrity proof.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for c in candidates {
        let f = c.features;
        let canonical = format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}\n",
            c.id, c.truth.kind.name(), f.repeat_count, f.distinct_episodes,
            f.contradiction_count, f.recency, f.correction_seen,
            f.independent_successes, f.provenance_diversity
        );
        for byte in canonical.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn json_string(value: &str) -> String {
    // All fields passed here are controlled enums, fixed metadata, or a
    // validated hexadecimal commit SHA; none contains JSON-special characters.
    format!("\"{value}\"")
}

fn emit_candidate(row: &Decision, source_sha: &str, seed: u64, candidates: usize) {
    println!(
        "{{\"type\":\"decision\",\"protocol\":\"GMCv2\",\"source_sha\":{},\"seed\":{},\"candidate_count\":{},\"candidate_id\":{},\"observed_features\":{{\"repeat_count\":{},\"distinct_episodes\":{},\"contradiction_count\":{},\"recency\":{},\"correction_seen\":{},\"independent_successes\":{},\"provenance_diversity\":{}}},\"kind\":{},\"policy\":{},\"role\":{},\"action\":{},\"oracle_action\":{},\"net_utility\":{},\"oracle_utility\":{},\"decision_regret\":{},\"storage_units\":{},\"load_bearing\":{},\"covered\":{},\"false_promotion\":{},\"destructive_forgetting\":{}}}",
        json_string(source_sha), seed, candidates, row.candidate_id,
        row.features.repeat_count, row.features.distinct_episodes,
        row.features.contradiction_count, row.features.recency,
        row.features.correction_seen, row.features.independent_successes,
        row.features.provenance_diversity,
        json_string(row.kind.name()), json_string(row.policy), json_string(row.role),
        json_string(row.action.name()), json_string(row.oracle_action.name()),
        row.net_utility, row.oracle_utility, row.regret, row.storage_units,
        row.load_bearing, row.covered, row.false_promotion, row.destructive_forgetting
    );
}

fn emit_summary(policy: &str, summary: &PolicySummary, source_sha: &str, seed: u64) {
    println!(
        "{{\"type\":\"summary\",\"protocol\":\"GMCv2\",\"source_sha\":{},\"seed\":{},\"policy\":{},\"candidates\":{},\"total_utility\":{},\"mean_regret\":{:.6},\"total_regret\":{},\"total_storage_units\":{},\"load_bearing_candidates\":{},\"covered_load_bearing\":{},\"load_bearing_coverage\":{:.6},\"false_promotions\":{},\"destructive_forgetting\":{}}}",
        json_string(source_sha), seed, json_string(policy), summary.candidates,
        summary.total_utility, summary.mean_regret(), summary.total_regret,
        summary.total_storage_units, summary.load_bearing_candidates,
        summary.covered_load_bearing, summary.load_bearing_coverage(),
        summary.false_promotions, summary.destructive_forgetting
    );
}

fn run(config: Config) -> Result<(), String> {
    let candidates = build_fixture(config.seed, config.candidates);
    let mut rows = Vec::new();

    for spec in &POLICIES {
        for candidate in &candidates {
            let action = choose_action(spec.policy, &candidate.features);
            rows.push(Decision::from_candidate(candidate, spec, action));
        }
    }

    validate_self_checks(&candidates, &rows)?;
    let fingerprint = fixture_fingerprint(&candidates);
    println!(
        "{{\"type\":\"run\",\"protocol\":\"GMCv2\",\"generator\":\"broca-consolidation-v2\",\"status\":\"COMPLETED\",\"self_checks_passed\":true,\"source_sha\":{},\"seed\":{},\"candidate_count\":{},\"policy_count\":{},\"fixture_fnv1a64\":{},\"fingerprint_is_cryptographic\":false,\"feature_noise\":\"bounded_jitter_v1\",\"oracle_scope\":\"synthetic-generator-only\"}}",
        json_string(&config.source_sha), config.seed, config.candidates,
        POLICIES.len(), json_string(&fingerprint)
    );

    for row in &rows {
        emit_candidate(row, &config.source_sha, config.seed, config.candidates);
    }
    for (policy, summary) in summarize(&rows) {
        emit_summary(policy, &summary, &config.source_sha, config.seed);
    }

    let oracle_spec = PolicySpec {
        name: "oracle_diagnostic",
        role: "oracle_diagnostic",
        policy: Policy::AlwaysRaw,
    };
    let mut oracle_rows = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        let row = Decision::from_candidate(candidate, &oracle_spec, candidate.truth.kind.oracle_action());
        oracle_rows.push(row);
    }
    for row in &oracle_rows {
        emit_candidate(row, &config.source_sha, config.seed, config.candidates);
    }
    let mut oracle_summary = PolicySummary::default();
    for row in &oracle_rows {
        oracle_summary.add(row);
    }
    emit_summary("oracle_diagnostic", &oracle_summary, &config.source_sha, config.seed);
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!(
            "broca-consolidation-reference --source-sha <40-char-commit-sha> [--seed N] [--candidates N]\n\nEvaluates deterministic consolidation policies. Oracle and mutation controls are diagnostics, not deployable systems."
        );
        return;
    }
    match parse_args(&args).and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("broca-consolidation-reference: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate_all(candidates: &[Candidate]) -> Vec<Decision> {
        POLICIES.iter().flat_map(|spec| {
            candidates.iter().map(move |candidate| {
                Decision::from_candidate(candidate, spec, choose_action(spec.policy, &candidate.features))
            })
        }).collect()
    }

    #[test]
    fn oracle_action_maximizes_the_declared_synthetic_utility() {
        for kind in CandidateKind::ALL {
            let oracle = kind.oracle_action();
            for action in ACTIONS {
                assert!(kind.net_utility(oracle) >= kind.net_utility(action));
            }
        }
    }

    #[test]
    fn fixture_is_deterministic_and_seed_sensitive() {
        let a = build_fixture(17, 24);
        let b = build_fixture(17, 24);
        let c = build_fixture(18, 24);
        assert_eq!(fixture_fingerprint(&a), fixture_fingerprint(&b));
        assert_ne!(fixture_fingerprint(&a), fixture_fingerprint(&c));
    }

    #[test]
    fn observable_feature_profiles_vary_within_each_ground_truth_class() {
        let candidates = build_fixture(171, 240);
        for kind in CandidateKind::ALL {
            let profiles: Vec<_> = candidates
                .iter()
                .filter(|candidate| candidate.truth.kind == kind)
                .map(|candidate| candidate.features)
                .collect();
            assert!(profiles.len() >= 2);
            assert!(
                profiles.windows(2).any(|pair| pair[0] != pair[1]),
                "observable profiles did not vary for {:?}",
                kind
            );
        }
    }

    #[test]
    fn same_seed_is_reproducible_but_changes_observable_evidence_by_seed() {
        let first = build_fixture(1701, 240);
        let second = build_fixture(1702, 240);
        let changed_same_kind = first.iter().any(|left| {
            second.iter().any(|right| {
                left.id == right.id
                    && left.truth.kind == right.truth.kind
                    && left.features != right.features
            })
        });
        assert!(
            changed_same_kind,
            "different seeds changed only ordering, not matched observable profiles"
        );
        assert_eq!(
            fixture_fingerprint(&first),
            fixture_fingerprint(&build_fixture(1701, 240))
        );
    }

    #[test]
    fn policies_receive_features_not_ground_truth() {
        fn accepts_features_only(policy: Policy, features: &ObservedFeatures) -> Action {
            choose_action(policy, features)
        }
        let candidate = Candidate::new(1, CandidateKind::PoisonedRule);
        let _ = accepts_features_only(Policy::TypedLifecycle, &candidate.features);
        // GroundTruth is held by the evaluator; choose_action has no parameter
        // through which it could receive it.
    }

    #[test]
    fn retire_everything_is_flagged_as_degenerate_forgetting() {
        let candidates = build_fixture(21, 24);
        let summary = summarize(&evaluate_all(&candidates));
        let retire = summary.get("retire_everything").unwrap();
        assert_eq!(retire.covered_load_bearing, 0);
        assert_eq!(retire.destructive_forgetting, retire.load_bearing_candidates);
    }

    #[test]
    fn promote_everything_control_triggers_false_promotion() {
        let candidates = build_fixture(25, 24);
        let summary = summarize(&evaluate_all(&candidates));
        assert!(summary.get("promote_everything").unwrap().false_promotions > 0);
    }

    #[test]
    fn typed_lifecycle_beats_both_degenerate_extremes_on_noisy_features() {
        let candidates = build_fixture(5, 240);
        let rows = evaluate_all(&candidates);
        let summaries = summarize(&rows);
        let typed = summaries.get("typed_lifecycle").unwrap();
        let retire = summaries.get("retire_everything").unwrap();
        let promote = summaries.get("promote_everything").unwrap();

        assert!(typed.load_bearing_coverage() >= 0.75);
        assert!(typed.total_regret < retire.total_regret);
        assert!(typed.total_regret < promote.total_regret);
        assert!(typed.false_promotions < promote.false_promotions);
    }

    #[test]
    fn receipt_requires_exact_source_sha_and_valid_candidate_count() {
        let valid = vec![
            "--source-sha".to_owned(), "a".repeat(40),
            "--candidates".to_owned(), "12".to_owned(),
        ];
        assert!(parse_args(&valid).is_ok());
        assert!(parse_args(&Vec::new()).is_err());
        let invalid = vec!["--source-sha".to_owned(), "xyz".to_owned()];
        assert!(parse_args(&invalid).is_err());
        let too_small = vec![
            "--source-sha".to_owned(), "a".repeat(40),
            "--candidates".to_owned(), "11".to_owned(),
        ];
        assert!(parse_args(&too_small).is_err());
    }

    #[test]
    fn poor_candidate_quality_does_not_invalidate_measurement_receipt() {
        let candidates = build_fixture(350, 24);
        let mut rows = evaluate_all(&candidates);
        let typed_spec = POLICIES
            .iter()
            .find(|spec| spec.name == "typed_lifecycle")
            .unwrap();
        for row in rows.iter_mut().filter(|row| row.policy == "typed_lifecycle") {
            let candidate = candidates
                .iter()
                .find(|candidate| candidate.id == row.candidate_id)
                .unwrap();
            *row = Decision::from_candidate(candidate, typed_spec, Action::Retire);
        }
        // The anti-controls remain intact. Poor candidate performance should
        // be measured, not relabeled as an invalid benchmark run.
        assert!(validate_self_checks(&candidates, &rows).is_ok());
        assert_eq!(
            summarize(&rows).get("typed_lifecycle").unwrap().covered_load_bearing,
            0
        );
    }

    #[test]
    fn instrument_self_checks_accept_multiple_noisy_seeds() {
        for seed in 1..=32 {
            let candidates = build_fixture(seed, 24);
            let rows = evaluate_all(&candidates);
            assert!(
                validate_self_checks(&candidates, &rows).is_ok(),
                "instrument self-check failed for seed {seed}"
            );
        }
    }

    #[test]
    fn self_check_rejects_missing_mutation_controls() {
        let candidates = build_fixture(3, 12);
        let only_baseline: Vec<_> = candidates.iter().map(|candidate| {
            let spec = &POLICIES[0];
            Decision::from_candidate(candidate, spec, choose_action(spec.policy, &candidate.features))
        }).collect();
        assert!(validate_self_checks(&candidates, &only_baseline).is_err());
    }
}
