use std::collections::{BTreeMap, HashMap};
use std::env;
use std::process;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Key {
    episode: u32,
    entity: u32,
    attribute: u16,
}

#[derive(Clone, Debug)]
enum Action {
    Put {
        step: u64,
        key: Key,
        version: u32,
        value: u64,
    },
    Delete {
        step: u64,
        key: Key,
        version: u32,
    },
    Query {
        step: u64,
        query_id: u64,
        family: &'static str,
        key: Key,
        expected: Option<u64>,
    },
}

impl Action {
    fn canonical(&self) -> String {
        match self {
            Self::Put {
                step,
                key,
                version,
                value,
            } => format!(
                "P|{step}|{}|{}|{}|{version}|{value}\n",
                key.episode, key.entity, key.attribute
            ),
            Self::Delete {
                step,
                key,
                version,
            } => format!(
                "D|{step}|{}|{}|{}|{version}\n",
                key.episode, key.entity, key.attribute
            ),
            Self::Query {
                step,
                query_id,
                family,
                key,
                expected,
            } => format!(
                "Q|{step}|{query_id}|{family}|{}|{}|{}|{}\n",
                key.episode,
                key.entity,
                key.attribute,
                expected.map_or_else(|| "ABSTAIN".to_owned(), |v| v.to_string())
            ),
        }
    }
}

#[derive(Clone, Copy)]
struct TruthRecord {
    version: u32,
    value: Option<u64>,
}

struct FixtureBuilder {
    actions: Vec<Action>,
    truth: HashMap<Key, TruthRecord>,
    rng: DeterministicRng,
    next_step: u64,
    next_query_id: u64,
}

impl FixtureBuilder {
    fn new(seed: u64) -> Self {
        Self {
            actions: Vec::new(),
            truth: HashMap::new(),
            rng: DeterministicRng::new(seed),
            next_step: 0,
            next_query_id: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.next_step += 1;
        self.next_step
    }

    fn value(&mut self) -> u64 {
        self.rng.next_u64()
    }

    fn put(&mut self, key: Key, version: u32, value: u64) {
        let should_apply = self
            .truth
            .get(&key)
            .map_or(true, |current| version > current.version);
        if should_apply {
            self.truth.insert(
                key,
                TruthRecord {
                    version,
                    value: Some(value),
                },
            );
        }

        let step = self.tick();
        self.actions.push(Action::Put {
            step,
            key,
            version,
            value,
        });
    }

    fn delete(&mut self, key: Key, version: u32) {
        let should_apply = self
            .truth
            .get(&key)
            .map_or(true, |current| version > current.version);
        if should_apply {
            self.truth.insert(
                key,
                TruthRecord {
                    version,
                    value: None,
                },
            );
        }

        let step = self.tick();
        self.actions.push(Action::Delete {
            step,
            key,
            version,
        });
    }

    fn query(&mut self, family: &'static str, key: Key) {
        let expected = self.truth.get(&key).and_then(|record| record.value);
        let step = self.tick();
        let query_id = self.next_query_id;
        self.next_query_id += 1;
        self.actions.push(Action::Query {
            step,
            query_id,
            family,
            key,
            expected,
        });
    }
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*: fast deterministic fixture values, not cryptography.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn build_fixture(seed: u64, episodes: usize, distractors: usize) -> Vec<Action> {
    let mut builder = FixtureBuilder::new(seed);
    for episode_idx in 0..episodes {
        let episode = episode_idx as u32;

        // A. Retention under interference: a bounded cache may evict the target.
        let stable_key = Key {
            episode,
            entity: 1,
            attribute: 1,
        };
        let stable_value = builder.value();
        builder.put(stable_key, 1, stable_value);
        for i in 0..distractors {
            let key = Key {
                episode,
                entity: 1_000 + i as u32,
                attribute: (i % 11) as u16,
            };
            let value = builder.value();
            builder.put(key, 1, value);
        }
        builder.query("stable_retention", stable_key);

        // B. A replay of version 1 must not replace the newer version 2.
        let update_key = Key {
            episode,
            entity: 2,
            attribute: 1,
        };
        let old_value = builder.value();
        let new_value = builder.value();
        builder.put(update_key, 1, old_value);
        builder.put(update_key, 2, new_value);
        builder.put(update_key, 1, old_value);
        builder.query("stale_update", update_key);

        // C. An old value replay must not resurrect a deleted version.
        // The distractors deliberately pressure bounded tombstone retention.
        let delete_key = Key {
            episode,
            entity: 3,
            attribute: 2,
        };
        let deleted_value = builder.value();
        builder.put(delete_key, 1, deleted_value);
        builder.delete(delete_key, 2);
        for i in 0..distractors {
            let key = Key {
                episode,
                entity: 200_000 + i as u32,
                attribute: (i % 13) as u16,
            };
            let value = builder.value();
            builder.put(key, 1, value);
        }
        builder.put(delete_key, 1, deleted_value);
        builder.query("deletion_replay", delete_key);

        // D. A query in a different namespace must not see the source episode.
        let scoped_key = Key {
            episode,
            entity: 77,
            attribute: 1,
        };
        let scoped_value = builder.value();
        builder.put(scoped_key, 1, scoped_value);
        let absent_scope_key = Key {
            episode: 4_000_000_000u32.wrapping_add(episode),
            entity: scoped_key.entity,
            attribute: scoped_key.attribute,
        };
        builder.query("cross_namespace", absent_scope_key);
    }
    builder.actions
}

fn fixture_fingerprint(actions: &[Action]) -> String {
    // Stable, explicitly non-cryptographic FNV-1a fingerprint.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for action in actions {
        for byte in action.canonical().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

#[derive(Clone, Copy)]
struct Entry {
    version: u32,
    value: Option<u64>,
    last_touch: u64,
}

struct MemoryModel {
    name: &'static str,
    role: &'static str,
    capacity: Option<usize>,
    ignore_episode: bool,
    ignore_versions: bool,
    entries: HashMap<Key, Entry>,
    clock: u64,
}

impl MemoryModel {
    fn new(
        name: &'static str,
        role: &'static str,
        capacity: Option<usize>,
        ignore_episode: bool,
        ignore_versions: bool,
    ) -> Self {
        Self {
            name,
            role,
            capacity,
            ignore_episode,
            ignore_versions,
            entries: HashMap::new(),
            clock: 0,
        }
    }

    fn normalize_key(&self, mut key: Key) -> Key {
        if self.ignore_episode {
            key.episode = 0;
        }
        key
    }

    fn next_clock(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn mutate(&mut self, key: Key, version: u32, value: Option<u64>) {
        let touch = self.next_clock();
        let normalized = self.normalize_key(key);

        if !self.ignore_versions {
            if let Some(previous) = self.entries.get(&normalized) {
                if version <= previous.version {
                    return;
                }
            }
        }

        if !self.entries.contains_key(&normalized) {
            if let Some(capacity) = self.capacity {
                if self.entries.len() >= capacity {
                    let oldest = self
                        .entries
                        .iter()
                        .min_by_key(|(_, entry)| entry.last_touch)
                        .map(|(key, _)| *key);
                    if let Some(oldest_key) = oldest {
                        self.entries.remove(&oldest_key);
                    }
                }
            }
        }

        self.entries.insert(
            normalized,
            Entry {
                version,
                value,
                last_touch: touch,
            },
        );
    }

    fn query(&mut self, key: Key) -> Option<u64> {
        let touch = self.next_clock();
        let normalized = self.normalize_key(key);
        if let Some(entry) = self.entries.get_mut(&normalized) {
            entry.last_touch = touch;
            entry.value
        } else {
            None
        }
    }
}

#[derive(Clone)]
struct QueryResult {
    system: &'static str,
    role: &'static str,
    query_id: u64,
    step: u64,
    episode: u32,
    family: &'static str,
    expected: Option<u64>,
    actual: Option<u64>,
}

impl QueryResult {
    fn correct(&self) -> bool {
        self.expected == self.actual
    }
}

fn run_model(actions: &[Action], model: &mut MemoryModel) -> Vec<QueryResult> {
    let mut results = Vec::new();
    for action in actions {
        match action {
            Action::Put {
                key,
                version,
                value,
                ..
            } => model.mutate(*key, *version, Some(*value)),
            Action::Delete { key, version, .. } => model.mutate(*key, *version, None),
            Action::Query {
                step,
                query_id,
                family,
                key,
                expected,
            } => {
                results.push(QueryResult {
                    system: model.name,
                    role: model.role,
                    query_id: *query_id,
                    step: *step,
                    episode: key.episode,
                    family: *family,
                    expected: *expected,
                    actual: model.query(*key),
                });
            }
        }
    }
    results
}

#[derive(Default)]
struct Summary {
    queries: usize,
    correct: usize,
    stale_update_errors: usize,
    deletion_leaks: usize,
    cross_namespace_leaks: usize,
}

fn summarize(results: &[QueryResult]) -> BTreeMap<String, Summary> {
    let mut summaries: BTreeMap<String, Summary> = BTreeMap::new();
    for result in results {
        let summary = summaries.entry(result.family.to_owned()).or_default();
        summary.queries += 1;
        if result.correct() {
            summary.correct += 1;
        }
        if result.family == "stale_update"
            && result.actual.is_some()
            && result.actual != result.expected
        {
            summary.stale_update_errors += 1;
        }
        if result.family == "deletion_replay"
            && result.expected.is_none()
            && result.actual.is_some()
        {
            summary.deletion_leaks += 1;
        }
        if result.family == "cross_namespace"
            && result.expected.is_none()
            && result.actual.is_some()
        {
            summary.cross_namespace_leaks += 1;
        }
    }
    summaries
}

struct Config {
    seed: u64,
    episodes: usize,
    distractors: usize,
    capacity: usize,
    source_sha: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 7,
            episodes: 16,
            distractors: 32,
            capacity: 16,
            source_sha: String::new(),
        }
    }
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut config = Config::default();
    let mut i = 0;
    while i < args.len() {
        if i + 1 >= args.len() {
            return Err(format!("missing value for {}", args[i]));
        }
        let name = args[i].as_str();
        let value = args[i + 1].as_str();
        match name {
            "--seed" => {
                config.seed = value
                    .parse()
                    .map_err(|_| format!("invalid --seed: {value}"))?;
            }
            "--episodes" => {
                config.episodes = value
                    .parse()
                    .map_err(|_| format!("invalid --episodes: {value}"))?;
            }
            "--distractors" => {
                config.distractors = value
                    .parse()
                    .map_err(|_| format!("invalid --distractors: {value}"))?;
            }
            "--capacity" => {
                config.capacity = value
                    .parse()
                    .map_err(|_| format!("invalid --capacity: {value}"))?;
            }
            "--source-sha" => config.source_sha = value.to_owned(),
            _ => return Err(format!("unknown argument: {name}")),
        }
        i += 2;
    }

    if config.episodes == 0 || config.episodes > 10_000 {
        return Err("--episodes must be in 1..=10000".into());
    }
    if config.capacity == 0 || config.capacity > 1_000_000 {
        return Err("--capacity must be in 1..=1000000".into());
    }
    if config.distractors > 100_000 {
        return Err("--distractors must be <= 100000".into());
    }
    if config.episodes.saturating_mul(config.distractors) > 1_000_000 {
        return Err("episodes * distractors must be <= 1000000".into());
    }
    if config.source_sha.len() != 40
        || !config.source_sha.chars().all(|c| c.is_ascii_hexdigit())
    {
        return Err("--source-sha is required and must be a 40-character Git commit SHA".into());
    }
    Ok(config)
}

fn json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if c.is_control() => output.push_str(&format!("\\u{:04x}", c as u32)),
            c => output.push(c),
        }
    }
    output.push('"');
    output
}

fn option_json(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_owned(), |v| json_string(&v.to_string()))
}

fn emit_result(record: &QueryResult, source_sha: &str, seed: u64, episodes: usize, capacity: usize) {
    println!(
        "{{\"type\":\"query\",\"protocol\":\"MIREv1\",\"source_sha\":{},\"seed\":{},\"episodes\":{},\"capacity\":{},\"system\":{},\"role\":{},\"query_id\":{},\"step\":{},\"episode\":{},\"family\":{},\"expected\":{},\"actual\":{},\"correct\":{}}}",
        json_string(source_sha),
        seed,
        episodes,
        capacity,
        json_string(record.system),
        json_string(record.role),
        record.query_id,
        record.step,
        record.episode,
        json_string(record.family),
        option_json(record.expected),
        option_json(record.actual),
        record.correct()
    );
}

fn emit_summaries(
    results: &[QueryResult],
    source_sha: &str,
    seed: u64,
    episodes: usize,
    capacity: usize,
) {
    for (family, summary) in summarize(results) {
        let accuracy = if summary.queries == 0 {
            0.0
        } else {
            summary.correct as f64 / summary.queries as f64
        };
        println!(
            "{{\"type\":\"summary\",\"protocol\":\"MIREv1\",\"source_sha\":{},\"seed\":{},\"episodes\":{},\"capacity\":{},\"system\":{},\"role\":{},\"family\":{},\"queries\":{},\"correct\":{},\"accuracy\":{:.6},\"stale_update_errors\":{},\"deletion_leaks\":{},\"cross_namespace_leaks\":{}}}",
            json_string(source_sha),
            seed,
            episodes,
            capacity,
            json_string(results[0].system),
            json_string(results[0].role),
            json_string(&family),
            summary.queries,
            summary.correct,
            accuracy,
            summary.stale_update_errors,
            summary.deletion_leaks,
            summary.cross_namespace_leaks
        );
    }
}

fn system_specs(capacity: usize) -> Vec<MemoryModel> {
    vec![
        MemoryModel::new("exact_versioned_reference", "baseline", None, false, false),
        MemoryModel::new("bounded_lru_versioned", "baseline", Some(capacity), false, false),
        MemoryModel::new("mutation_namespace_blind", "mutation_control", None, true, false),
        MemoryModel::new("mutation_version_ignoring", "mutation_control", None, false, true),
    ]
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!(
            "broca-mire-reference --source-sha <40-char-commit-sha> [--seed N] [--episodes N] [--distractors N] [--capacity N]\n\nEmits deterministic JSONL records. Mutation controls are intentionally faulty and not competitive systems."
        );
        return Ok(());
    }

    let config = parse_args(&args)?;
    let actions = build_fixture(config.seed, config.episodes, config.distractors);
    let fingerprint = fixture_fingerprint(&actions);
    println!(
        "{{\"type\":\"run\",\"protocol\":\"MIREv1\",\"generator\":\"mire-reference-v1\",\"status\":\"COMPLETED\",\"source_sha\":{},\"seed\":{},\"episodes\":{},\"distractors\":{},\"capacity\":{},\"actions\":{},\"fixture_fnv1a64\":{},\"fingerprint_is_cryptographic\":false}}",
        json_string(&config.source_sha),
        config.seed,
        config.episodes,
        config.distractors,
        config.capacity,
        actions.len(),
        json_string(&fingerprint)
    );

    for mut model in system_specs(config.capacity) {
        let results = run_model(&actions, &mut model);
        for result in &results {
            emit_result(
                result,
                &config.source_sha,
                config.seed,
                config.episodes,
                config.capacity,
            );
        }
        emit_summaries(
            &results,
            &config.source_sha,
            config.seed,
            config.episodes,
            config.capacity,
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("broca-mire-reference: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &'static str, capacity: Option<usize>, ignore_episode: bool, ignore_versions: bool) -> MemoryModel {
        MemoryModel::new(name, "test", capacity, ignore_episode, ignore_versions)
    }

    #[test]
    fn exact_reference_matches_all_generated_labels() {
        let actions = build_fixture(7, 4, 8);
        let results = run_model(
            &actions,
            &mut model("exact", None, false, false),
        );
        assert!(!results.is_empty());
        assert!(results.iter().all(QueryResult::correct));
    }

    #[test]
    fn namespace_mutant_is_detected_by_cross_namespace_queries() {
        let actions = build_fixture(7, 3, 4);
        let results = run_model(
            &actions,
            &mut model("namespace_mutant", None, true, false),
        );
        assert!(results.iter().any(|result| {
            result.family == "cross_namespace" && !result.correct()
        }));
        assert!(summarize(&results)
            .values()
            .any(|summary| summary.cross_namespace_leaks > 0));
    }

    #[test]
    fn version_ignoring_mutant_is_detected_by_stale_replay() {
        let actions = build_fixture(7, 3, 4);
        let results = run_model(
            &actions,
            &mut model("version_mutant", None, false, true),
        );
        assert!(results.iter().any(|result| {
            result.family == "stale_update" && !result.correct()
        }));
        assert!(summarize(&results)
            .values()
            .any(|summary| summary.stale_update_errors > 0));
        assert!(summarize(&results)
            .values()
            .any(|summary| summary.deletion_leaks > 0));
    }

    #[test]
    fn bounded_lru_exposes_eviction_and_tombstone_resurrection_tradeoff() {
        let actions = build_fixture(11, 2, 24);
        let results = run_model(
            &actions,
            &mut model("lru", Some(8), false, false),
        );
        assert!(results.iter().any(|result| {
            result.family == "stable_retention" && !result.correct()
        }));
        assert!(results.iter().any(|result| {
            result.family == "deletion_replay" && result.actual.is_some()
        }));
    }

    #[test]
    fn fixture_fingerprint_is_deterministic_and_seed_sensitive() {
        let a = build_fixture(11, 2, 7);
        let b = build_fixture(11, 2, 7);
        let c = build_fixture(12, 2, 7);
        assert_eq!(fixture_fingerprint(&a), fixture_fingerprint(&b));
        assert_ne!(fixture_fingerprint(&a), fixture_fingerprint(&c));
    }

    #[test]
    fn receipt_requires_exact_source_revision() {
        let valid = vec![
            "--source-sha".to_owned(),
            "a".repeat(40),
        ];
        assert!(parse_args(&valid).is_ok());
        let missing = Vec::new();
        assert!(parse_args(&missing).is_err());
        let short = vec!["--source-sha".to_owned(), "deadbeef".to_owned()];
        assert!(parse_args(&short).is_err());
    }
}
