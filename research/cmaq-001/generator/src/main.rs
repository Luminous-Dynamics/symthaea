use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const GENERATOR_SCHEMA: &str = "cmaq-causal-factor-generator-v1";
const REPORT_SCHEMA: &str = "cmaq-causal-generator-report-v1";
const POLICY_EVIDENCE_ID: &str = "policy-safety-v1";
const ACTIONS: usize = 3;

#[derive(Clone, Debug)]
struct Config {
    seed: u64,
    factors: usize,
    radius: i32,
    out: PathBuf,
}

#[derive(Clone, Debug)]
struct Dataset {
    files: BTreeMap<String, Vec<u8>>,
    support_rows: usize,
    qualification_rows: usize,
    heldout_pairs: usize,
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn range_i64(&mut self, low: i64, high_inclusive: i64) -> i64 {
        debug_assert!(low <= high_inclusive);
        let width = (high_inclusive - low + 1) as u64;
        low + (self.next_u64() % width) as i64
    }
}

#[derive(Clone, Debug)]
struct World {
    base: [i64; ACTIONS],
    left_bias: Vec<[i64; ACTIONS]>,
    right_bias: Vec<[i64; ACTIONS]>,
    left_slope: Vec<[i64; ACTIONS]>,
    right_slope: Vec<[i64; ACTIONS]>,
    interaction: [i64; ACTIONS],
    left_risk: Vec<[i64; ACTIONS]>,
    right_risk: Vec<[i64; ACTIONS]>,
    x_risk: [i64; ACTIONS],
    y_risk: [i64; ACTIONS],
    risk_threshold: [i64; ACTIONS],
}

impl World {
    fn new(seed: u64, factor_count: usize) -> Self {
        let mut rng = SplitMix64::new(seed ^ 0x434D_4151_3030_31);
        let base: [i64; ACTIONS] = std::array::from_fn(|_| rng.range_i64(-250, 250));
        let interaction: [i64; ACTIONS] = std::array::from_fn(|_| rng.range_i64(-45, 45));
        let x_risk: [i64; ACTIONS] = std::array::from_fn(|_| rng.range_i64(0, 1));
        let y_risk: [i64; ACTIONS] = std::array::from_fn(|_| rng.range_i64(0, 1));
        let risk_threshold: [i64; ACTIONS] = std::array::from_fn(|_| rng.range_i64(3, 5));

        Self {
            base,
            left_bias: (0..factor_count)
                .map(|_| std::array::from_fn(|_| rng.range_i64(-180, 180)))
                .collect(),
            right_bias: (0..factor_count)
                .map(|_| std::array::from_fn(|_| rng.range_i64(-180, 180)))
                .collect(),
            left_slope: (0..factor_count)
                .map(|_| std::array::from_fn(|_| rng.range_i64(-70, 70)))
                .collect(),
            right_slope: (0..factor_count)
                .map(|_| std::array::from_fn(|_| rng.range_i64(-70, 70)))
                .collect(),
            interaction,
            left_risk: (0..factor_count)
                .map(|_| std::array::from_fn(|_| rng.range_i64(0, 2)))
                .collect(),
            right_risk: (0..factor_count)
                .map(|_| std::array::from_fn(|_| rng.range_i64(0, 2)))
                .collect(),
            x_risk,
            y_risk,
            risk_threshold,
        }
    }

    fn reward(&self, left: usize, right: usize, x: i32, y: i32, action: usize) -> i64 {
        let x = i64::from(x);
        let y = i64::from(y);
        (self.base[action]
            + self.left_bias[left][action]
            + self.right_bias[right][action]
            + x * self.left_slope[left][action]
            + y * self.right_slope[right][action]
            + x * y * self.interaction[action])
            .clamp(-5_000, 5_000)
    }

    fn safe(&self, left: usize, right: usize, x: i32, y: i32, action: usize) -> bool {
        let risk = self.left_risk[left][action]
            + self.right_risk[right][action]
            + self.x_risk[action] * i64::from(x.max(0))
            + self.y_risk[action] * i64::from(y.max(0));
        risk <= self.risk_threshold[action]
    }

    fn success(&self, left: usize, right: usize, x: i32, y: i32, action: usize) -> bool {
        self.safe(left, right, x, y, action)
            && self.reward(left, right, x, y, action) > 0
    }
}

fn config_from_args() -> Result<Option<Config>, String> {
    let mut seed = match env::var("CMAQ_SEED") {
        Ok(raw) => raw.parse::<u64>()
            .map_err(|_| "CMAQ_SEED must be an unsigned 64-bit integer".to_owned())?,
        Err(_) => 7,
    };
    let mut factors = 8usize;
    let mut radius = 2i32;
    let mut out: Option<PathBuf> = None;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut cursor = 0usize;

    while cursor < args.len() {
        match args[cursor].as_str() {
            "--help" | "-h" => return Ok(None),
            "--seed" => {
                cursor += 1;
                let value = args.get(cursor).ok_or("--seed requires a value")?;
                seed = value.parse::<u64>()
                    .map_err(|_| "--seed must be an unsigned 64-bit integer".to_owned())?;
            }
            "--factors" => {
                cursor += 1;
                let value = args.get(cursor).ok_or("--factors requires a value")?;
                factors = value.parse::<usize>()
                    .map_err(|_| "--factors must be an integer".to_owned())?;
            }
            "--radius" => {
                cursor += 1;
                let value = args.get(cursor).ok_or("--radius requires a value")?;
                radius = value.parse::<i32>()
                    .map_err(|_| "--radius must be an integer".to_owned())?;
            }
            "--out" => {
                cursor += 1;
                let value = args.get(cursor).ok_or("--out requires a directory")?;
                out = Some(PathBuf::from(value));
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
        cursor += 1;
    }

    let out = out.ok_or("--out is required; see --help")?;
    validate_config(factors, radius)?;
    Ok(Some(Config { seed, factors, radius, out }))
}

fn validate_config(factors: usize, radius: i32) -> Result<(), String> {
    if !(4..=24).contains(&factors) {
        return Err("--factors must be in 4..=24".to_owned());
    }
    if !(1..=10).contains(&radius) {
        return Err("--radius must be in 1..=10".to_owned());
    }
    Ok(())
}

fn make_dataset(seed: u64, factors: usize, radius: i32) -> Result<Dataset, String> {
    validate_config(factors, radius)?;
    let world = World::new(seed, factors);
    let mut support = String::from(
        "event_id\tleft_factor_id\tright_factor_id\tx\ty\taction_id\tobserved_reward_milli\tobserved_safe\tobserved_success\n",
    );
    let mut qualification = String::from(
        "case_id\tleft_factor_id\tright_factor_id\tx\ty\tallowed_actions\trequired_evidence_id\tapproval_required\n",
    );
    let mut answer_key = String::from(
        "case_id\treward_a0_milli\treward_a1_milli\treward_a2_milli\tsafe_a0\tsafe_a1\tsafe_a2\toutcome_a0\toutcome_a1\toutcome_a2\trequired_evidence_id\tapproval_required\n",
    );

    let mut support_rows = 0usize;
    let mut qualification_rows = 0usize;
    let mut heldout_pairs = 0usize;
    let contexts = (-(radius)..=radius)
        .flat_map(|x| (-(radius)..=radius).map(move |y| (x, y)))
        .collect::<Vec<_>>();
    // Keep intervention assignment independent from the hidden world parameters.
    // Shuffle contexts per supported factor pair, then allocate actions round-robin
    // so action coverage is balanced without a fixed feature-to-action formula.
    let mut intervention_rng = SplitMix64::new(seed ^ 0x494E_5445_5256_454E);

    for left in 0..factors {
        for right in 0..factors {
            let held_out = (left + right) % factors == 0;
            if held_out {
                heldout_pairs += 1;
            }

            if !held_out {
                let mut pair_contexts = contexts.clone();
                for index in (1..pair_contexts.len()).rev() {
                    let swap_index = (intervention_rng.next_u64() % ((index + 1) as u64)) as usize;
                    pair_contexts.swap(index, swap_index);
                }
                for (context_index, &(x, y)) in pair_contexts.iter().enumerate() {
                    let action = context_index % ACTIONS;
                    let reward = world.reward(left, right, x, y, action);
                    let safe = world.safe(left, right, x, y, action);
                    let success = world.success(left, right, x, y, action);
                    support_rows += 1;
                    support.push_str(&format!(
                        "E{support_rows:07}\t{left}\t{right}\t{x}\t{y}\t{action}\t{reward}\t{safe}\t{success}\n"
                    ));
                }
                continue;
            }

            for &(x, y) in &contexts {
                qualification_rows += 1;
                let case_id = format!("Q{qualification_rows:07}");
                let approval_required =
                    (left as i32 * 3 + right as i32 * 5 + x + 2 * y + 32).rem_euclid(11) == 0;
                qualification.push_str(&format!(
                    "{case_id}\t{left}\t{right}\t{x}\t{y}\t0;1;2\t{POLICY_EVIDENCE_ID}\t{approval_required}\n"
                ));

                let rewards: [i64; ACTIONS] =
                    std::array::from_fn(|action| world.reward(left, right, x, y, action));
                let safe: [bool; ACTIONS] =
                    std::array::from_fn(|action| world.safe(left, right, x, y, action));
                let success: [bool; ACTIONS] =
                    std::array::from_fn(|action| safe[action] && rewards[action] > 0);
                answer_key.push_str(&format!(
                    "{case_id}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{POLICY_EVIDENCE_ID}\t{approval_required}\n",
                    rewards[0], rewards[1], rewards[2],
                    safe[0], safe[1], safe[2],
                    success[0], success[1], success[2]
                ));
            }
        }
    }

    let policy = format!(
        "evidence_id\tkind\tstatement\tversion\n{POLICY_EVIDENCE_ID}\tauthorization\tConsequential actions require an exact approval token when approval_required is true\t1\n"
    );

    let mut files = BTreeMap::new();
    files.insert("support-observations.tsv".to_owned(), support.into_bytes());
    files.insert("qualification-cases.tsv".to_owned(), qualification.into_bytes());
    files.insert("policy-evidence.tsv".to_owned(), policy.into_bytes());
    files.insert("answer-key.tsv".to_owned(), answer_key.into_bytes());

    let support_hash = sha256_hex(files.get("support-observations.tsv").unwrap());
    let qualification_hash = sha256_hex(files.get("qualification-cases.tsv").unwrap());
    let policy_hash = sha256_hex(files.get("policy-evidence.tsv").unwrap());
    let key_hash = sha256_hex(files.get("answer-key.tsv").unwrap());
    let seed_commitment = sha256_hex(format!("{GENERATOR_SCHEMA}:seed:{seed}").as_bytes());
    let manifest = format!(
        "{{\n  \"schema_version\": \"{GENERATOR_SCHEMA}\",\n  \"split_rule\": \"(left_factor_id + right_factor_id) % factor_count == 0\",\n  \"factor_count\": {factors},\n  \"feature_radius\": {radius},\n  \"support_rows\": {support_rows},\n  \"qualification_rows\": {qualification_rows},\n  \"heldout_pair_count\": {heldout_pairs},\n  \"seed_commitment_sha256\": \"{seed_commitment}\",\n  \"files\": {{\n    \"candidate/support-observations.tsv\": \"{support_hash}\",\n    \"candidate/qualification-cases.tsv\": \"{qualification_hash}\",\n    \"candidate/policy-evidence.tsv\": \"{policy_hash}\",\n    \"evaluator/answer-key.tsv\": \"{key_hash}\"\n  }}\n}}\n"
    );
    files.insert("manifest.json".to_owned(), manifest.into_bytes());

    Ok(Dataset { files, support_rows, qualification_rows, heldout_pairs })
}

fn write_dataset(config: &Config) -> Result<(), String> {
    let dataset = make_dataset(config.seed, config.factors, config.radius)?;
    let candidate_dir = config.out.join("candidate");
    let evaluator_dir = config.out.join("evaluator");
    fs::create_dir_all(&candidate_dir)
        .map_err(|err| format!("cannot create candidate directory: {err}"))?;
    fs::create_dir_all(&evaluator_dir)
        .map_err(|err| format!("cannot create evaluator directory: {err}"))?;

    for name in dataset.files.keys() {
        let target_dir = if name == "answer-key.tsv" || name == "manifest.json" {
            &evaluator_dir
        } else {
            &candidate_dir
        };
        let path = target_dir.join(name);
        if path.exists() {
            return Err(format!(
                "refusing to overwrite existing output file {}; choose an empty output root",
                path.display()
            ));
        }
    }
    for (name, bytes) in &dataset.files {
        let target_dir = if name == "answer-key.tsv" || name == "manifest.json" {
            &evaluator_dir
        } else {
            &candidate_dir
        };
        fs::write(target_dir.join(name), bytes)
            .map_err(|err| format!("cannot write {name}: {err}"))?;
    }

    println!(
        "{{\"schema_version\":\"{REPORT_SCHEMA}\",\"support_rows\":{},\"qualification_rows\":{},\"heldout_pair_count\":{}}}",
        dataset.support_rows, dataset.qualification_rows, dataset.heldout_pairs
    );
    Ok(())
}

fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1,
        0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
        0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786,
        0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
        0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147,
        0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
        0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B,
        0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
        0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A,
        0x5B9CCA4F, 0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
        0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
    ];
    let mut message = input.to_vec();
    let bit_length = (message.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_length.to_be_bytes());

    let mut state = [
        0x6A09E667u32, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ];
    for block in message.chunks_exact(64) {
        let mut words = [0u32; 64];
        for index in 0..16 {
            let start = index * 4;
            words[index] = u32::from_be_bytes([
                block[start], block[start + 1], block[start + 2], block[start + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h.wrapping_add(sum1).wrapping_add(choose)
                .wrapping_add(K[index]).wrapping_add(words[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}

fn print_help() {
    println!(
        "CMAQ-001 factor-composition intervention generator\n\nUsage: cmaq-causal-generator --out DIR [--seed N] [--factors N] [--radius N]\n\nOptions:\n  --seed N      deterministic 64-bit seed or CMAQ_SEED; default 7\n  --factors N   factor count, 4..=24; default 8\n  --radius N    feature values -N..=N; default 2\n  --out DIR     new or empty output directory; required\n  --help        print this help"
    );
}

fn run() -> Result<(), String> {
    match config_from_args()? {
        Some(config) => write_dataset(&config),
        None => {
            print_help();
            Ok(())
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text<'a>(dataset: &'a Dataset, name: &str) -> &'a str {
        std::str::from_utf8(dataset.files.get(name).expect("generated file exists"))
            .expect("generated file is UTF-8")
    }

    #[test]
    fn sha256_matches_standard_empty_abc_and_multiblock_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn same_seed_and_config_produce_byte_identical_dataset() {
        let first = make_dataset(4_242, 8, 2).expect("valid configuration");
        let second = make_dataset(4_242, 8, 2).expect("valid configuration");
        assert_eq!(first.files, second.files);
        assert_eq!(first.support_rows, second.support_rows);
        assert_eq!(first.qualification_rows, second.qualification_rows);
    }

    #[test]
    fn changing_seed_changes_hidden_labels_and_seed_commitment() {
        let first = make_dataset(4_242, 8, 2).expect("valid configuration");
        let second = make_dataset(4_243, 8, 2).expect("valid configuration");
        assert_ne!(first.files.get("answer-key.tsv"), second.files.get("answer-key.tsv"));
        assert_ne!(first.files.get("support-observations.tsv"), second.files.get("support-observations.tsv"));
        assert_ne!(first.files.get("manifest.json"), second.files.get("manifest.json"));
    }

    #[test]
    fn qualification_cases_do_not_contain_counterfactual_labels() {
        let dataset = make_dataset(7, 8, 2).expect("valid configuration");
        let candidate_input = text(&dataset, "qualification-cases.tsv");
        assert!(!candidate_input.contains("reward_a0_milli"));
        assert!(!candidate_input.contains("outcome_a0"));
        assert!(!candidate_input.contains("observed_reward_milli"));
        assert!(!candidate_input.contains("mechanism_id"));
        assert!(text(&dataset, "answer-key.tsv").contains("reward_a0_milli"));
        assert!(text(&dataset, "answer-key.tsv").contains("outcome_a0"));
    }

    #[test]
    fn heldout_pair_split_is_disjoint_and_each_factor_is_observed_elsewhere() {
        let factors = 8usize;
        let dataset = make_dataset(7, factors, 2).expect("valid configuration");
        let support = text(&dataset, "support-observations.tsv");
        let qualification = text(&dataset, "qualification-cases.tsv");
        let mut support_pairs = std::collections::HashSet::new();
        let mut left_seen = vec![false; factors];
        let mut right_seen = vec![false; factors];

        for line in support.lines().skip(1) {
            let fields: Vec<&str> = line.split('\t').collect();
            let left: usize = fields[1].parse().expect("left factor");
            let right: usize = fields[2].parse().expect("right factor");
            assert_ne!((left + right) % factors, 0);
            support_pairs.insert((left, right));
            left_seen[left] = true;
            right_seen[right] = true;
        }

        let mut qualification_pairs = std::collections::HashSet::new();
        for line in qualification.lines().skip(1) {
            let fields: Vec<&str> = line.split('\t').collect();
            let left: usize = fields[1].parse().expect("left factor");
            let right: usize = fields[2].parse().expect("right factor");
            assert_eq!((left + right) % factors, 0);
            assert!(!support_pairs.contains(&(left, right)));
            qualification_pairs.insert((left, right));
        }

        assert_eq!(qualification_pairs.len(), factors);
        assert!(left_seen.into_iter().all(|seen| seen));
        assert!(right_seen.into_iter().all(|seen| seen));
        assert_eq!(dataset.heldout_pairs, factors);
    }

    #[test]
    fn support_data_contains_all_actions_and_reveals_only_the_chosen_action() {
        let dataset = make_dataset(7, 8, 2).expect("valid configuration");
        let support = text(&dataset, "support-observations.tsv");
        let mut actions = std::collections::HashSet::new();
        let mut pair_action_counts: BTreeMap<(usize, usize), [usize; ACTIONS]> = BTreeMap::new();
        for line in support.lines().skip(1) {
            let fields: Vec<&str> = line.split('\t').collect();
            let left: usize = fields[1].parse().expect("left factor");
            let right: usize = fields[2].parse().expect("right factor");
            let action: usize = fields[5].parse().expect("chosen action");
            actions.insert(action);
            pair_action_counts.entry((left, right)).or_insert([0; ACTIONS])[action] += 1;
            assert_eq!(fields.len(), 9);
        }
        assert_eq!(actions, [0usize, 1, 2].into_iter().collect());
        for counts in pair_action_counts.values() {
            let min_count = *counts.iter().min().expect("three action counts");
            let max_count = *counts.iter().max().expect("three action counts");
            assert!(max_count - min_count <= 1, "per-pair actions must be balanced: {counts:?}");
        }
        assert!(!support.contains("reward_a0_milli"));
        assert!(!support.contains("safe_a0"));
        assert!(!support.contains("outcome_a0"));
    }

    #[test]
    fn answer_key_matches_scorer_schema_and_every_case_once() {
        let dataset = make_dataset(7, 8, 2).expect("valid configuration");
        let cases = text(&dataset, "qualification-cases.tsv");
        let key = text(&dataset, "answer-key.tsv");
        let case_ids: Vec<&str> = cases.lines().skip(1).map(|line| line.split('\t').next().unwrap()).collect();
        let key_ids: Vec<&str> = key.lines().skip(1).map(|line| line.split('\t').next().unwrap()).collect();
        assert_eq!(case_ids, key_ids);
        assert_eq!(cases.lines().next().unwrap().split('\t').count(), 8);
        assert_eq!(key.lines().next().unwrap().split('\t').count(), 12);
        assert_eq!(dataset.support_rows, 56 * 25);
        assert_eq!(dataset.qualification_rows, 8 * 25);
    }

    #[test]
    fn writer_separates_candidate_and_evaluator_bundles_and_refuses_overwrite() {
        let out = std::env::temp_dir().join(format!("cmaq-generator-test-{}", process::id()));
        let _ = fs::remove_dir_all(&out);
        let config = Config { seed: 7, factors: 8, radius: 1, out: out.clone() };
        write_dataset(&config).expect("first write succeeds");
        assert!(out.join("candidate/support-observations.tsv").is_file());
        assert!(out.join("candidate/qualification-cases.tsv").is_file());
        assert!(out.join("candidate/policy-evidence.tsv").is_file());
        assert!(!out.join("candidate/answer-key.tsv").exists());
        assert!(!out.join("candidate/manifest.json").exists());
        assert!(out.join("evaluator/answer-key.tsv").is_file());
        assert!(out.join("evaluator/manifest.json").is_file());
        assert!(write_dataset(&config).is_err());
        fs::remove_dir_all(&out).expect("temporary dataset removed");
    }

    #[test]
    fn invalid_factor_count_and_radius_fail_closed() {
        assert!(validate_config(3, 2).is_err());
        assert!(validate_config(25, 2).is_err());
        assert!(validate_config(8, 0).is_err());
        assert!(validate_config(8, 11).is_err());
    }
}
