use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn range(&mut self, upper: usize) -> usize {
        ((self.next_u64() as u128 * upper as u128) >> 64) as usize
    }

    fn bit(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

#[derive(Debug)]
struct Config {
    seed: u64,
    factors: usize,
    radius: usize,
    out: PathBuf,
}

#[derive(Debug)]
struct Example {
    context: u64,
    action: bool,
    success: bool,
}

#[derive(Debug)]
struct GeneratedCase {
    id: String,
    context: u64,
    mask: u64,
    offset: bool,
    examples: Vec<Example>,
}

impl GeneratedCase {
    fn target_action(&self) -> bool {
        ((self.context & self.mask).count_ones() % 2 == 1) ^ self.offset
    }

    fn agent_json(&self, factors: usize) -> String {
        let mut history = String::new();
        for (i, sample) in self.examples.iter().enumerate() {
            if i > 0 {
                history.push(',');
            }
            write!(
                &mut history,
                concat!(
                    "{{\"event_id\":\"evt-{i:03}\",",
                    "\"event_type\":\"intervention\",",
                    "\"timestamp\":{i},",
                    "\"source_id\":\"synthetic-environment\",",
                    "\"source_revision\":\"v0\",",
                    "\"validity\":\"valid\",",
                    "\"payload\":{{\"context_bits\":\"{}\",",
                    "\"intervened_action\":{},",
                    "\"observed_success\":{}}}}}"
                ),
                bit_string(sample.context, factors),
                u8::from(sample.action),
                sample.success
            )
            .expect("writing into a String cannot fail");
        }
        format!(
            concat!(
                "{{\"schema_version\":\"cmaq-001-agent-case-v0.1\",",
                "\"case_id\":\"{}\",",
                "\"suite\":\"causal-transfer\",",
                "\"mechanism_family\":\"sparse-parity\",",
                "\"episode_id\":\"episode-{}\",",
                "\"step_id\":{},",
                "\"regime_id\":\"r0\",",
                "\"observation\":{{\"context_bits\":\"{}\",",
                "\"factor_count\":{},\"action_count\":2}},",
                "\"event_history\":[{}],",
                "\"allowed_actions\":[\"a0\",\"a1\"],",
                "\"budgets\":{{\"remaining_queries\":0,",
                "\"remaining_actions\":1,\"context_tokens\":4096}}}}"
            ),
            self.id,
            self.id,
            self.examples.len(),
            bit_string(self.context, factors),
            factors,
            history
        )
    }

    fn answer_json(&self, factors: usize) -> String {
        format!(
            concat!(
                "{{\"case_id\":\"{}\",",
                "\"mechanism_mask_bits\":\"{}\",",
                "\"parity_offset\":{},",
                "\"target_action\":{},",
                "\"outcome_a0\":{},\"outcome_a1\":{}}}"
            ),
            self.id,
            bit_string(self.mask, factors),
            u8::from(self.offset),
            u8::from(self.target_action()),
            !self.target_action(),
            self.target_action()
        )
    }
}

fn bit_string(value: u64, factors: usize) -> String {
    let mut result = String::with_capacity(factors);
    for i in 0..factors {
        result.push(if value & (1_u64 << i) == 0 { '0' } else { '1' });
    }
    result
}

fn sample_context(rng: &mut SplitMix64, factor_mask: u64) -> u64 {
    rng.next_u64() & factor_mask
}

fn sample_mask(rng: &mut SplitMix64, factors: usize, radius: usize) -> u64 {
    let mut indices: Vec<usize> = (0..factors).collect();
    for i in 0..radius {
        let j = i + rng.range(factors - i);
        indices.swap(i, j);
    }
    indices[..radius]
        .iter()
        .fold(0_u64, |mask, i| mask | (1_u64 << i))
}

fn generate_cases(seed: u64, factors: usize, radius: usize) -> Result<Vec<GeneratedCase>, String> {
    if !(6..=64).contains(&factors) {
        return Err("factors must be in 6..=64 to allow held-out binary contexts".to_owned());
    }
    if radius == 0 || radius > factors || radius > 8 {
        return Err("radius must be in 1..=min(factors, 8)".to_owned());
    }

    let factor_mask = if factors == 64 { u64::MAX } else { (1_u64 << factors) - 1 };
    let case_count = factors * 8;
    let history_count = (factors * 4).max(32);
    let mut rng = SplitMix64::new(seed);
    let mut cases = Vec::with_capacity(case_count);

    for case_index in 0..case_count {
        let mask = sample_mask(&mut rng, factors, radius);
        let offset = rng.bit();
        let target = |context: u64| ((context & mask).count_ones() % 2 == 1) ^ offset;

        let mut examples = Vec::with_capacity(history_count);
        let mut historical_contexts = HashSet::new();
        for _ in 0..history_count {
            let context = sample_context(&mut rng, factor_mask);
            let action = rng.bit();
            let success = action == target(context);
            historical_contexts.insert(context);
            examples.push(Example { context, action, success });
        }

        let mut current_context = sample_context(&mut rng, factor_mask);
        let mut attempts = 0;
        while historical_contexts.contains(&current_context) {
            current_context = sample_context(&mut rng, factor_mask);
            attempts += 1;
            if attempts > 10_000 {
                return Err(format!(
                    "could not sample a held-out context for case {case_index}; increase factors"
                ));
            }
        }

        cases.push(GeneratedCase {
            id: format!("case-{case_index:04}"),
            context: current_context,
            mask,
            offset,
            examples,
        });
    }
    Ok(cases)
}

fn generate_contents(seed: u64, factors: usize, radius: usize) -> Result<(String, String, usize), String> {
    let cases = generate_cases(seed, factors, radius)?;
    let mut agent = String::new();
    let mut key = String::new();
    for case in &cases {
        writeln!(&mut agent, "{}", case.agent_json(factors)).expect("String write");
        writeln!(&mut key, "{}", case.answer_json(factors)).expect("String write");
    }
    Ok((agent, key, cases.len()))
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut seed = None;
    let mut factors = None;
    let mut radius = None;
    let mut out = None;
    let mut i = 1;
    while i < args.len() {
        let name = args[i].as_str();
        let value = args.get(i + 1).ok_or_else(|| format!("missing value for {name}"))?;
        match name {
            "--seed" => seed = Some(value.parse::<u64>().map_err(|_| "seed must be a u64".to_owned())?),
            "--factors" => factors = Some(value.parse::<usize>().map_err(|_| "factors must be an integer".to_owned())?),
            "--radius" => radius = Some(value.parse::<usize>().map_err(|_| "radius must be an integer".to_owned())?),
            "--out" => out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {name:?}")),
        }
        i += 2;
    }

    Ok(Config {
        seed: seed.ok_or_else(|| "required argument --seed is missing".to_owned())?,
        factors: factors.ok_or_else(|| "required argument --factors is missing".to_owned())?,
        radius: radius.ok_or_else(|| "required argument --radius is missing".to_owned())?,
        out: out.ok_or_else(|| "required argument --out is missing".to_owned())?,
    })
}

fn sha256(input: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(input);
    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    output
}
fn hex(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut result, "{byte:02x}").expect("String write");
    }
    result
}

fn write_outputs(config: &Config) -> Result<String, String> {
    let (agent, key, case_count) = generate_contents(config.seed, config.factors, config.radius)?;
    let agent_sha = hex(&sha256(agent.as_bytes()));
    let key_sha = hex(&sha256(key.as_bytes()));

    let agent_dir = config.out.join("agent");
    let evaluator_dir = config.out.join("evaluator");
    fs::create_dir_all(&agent_dir).map_err(|error| format!("create {}: {error}", agent_dir.display()))?;
    fs::create_dir_all(&evaluator_dir).map_err(|error| format!("create {}: {error}", evaluator_dir.display()))?;
    fs::write(agent_dir.join("cases.jsonl"), agent.as_bytes())
        .map_err(|error| format!("write agent cases: {error}"))?;
    fs::write(evaluator_dir.join("answer-key.jsonl"), key.as_bytes())
        .map_err(|error| format!("write answer key: {error}"))?;

    let manifest = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"cmaq-causal-factor-generator-v1\",\n",
            "  \"generator\": \"cmaq-causal-generator\",\n",
            "  \"seed\": {},\n",
            "  \"factor_count\": {},\n",
            "  \"interaction_radius\": {},\n",
            "  \"case_count\": {},\n",
            "  \"history_count_per_case\": {},\n",
            "  \"mechanism_family\": \"binary_sparse_parity\",\n",
            "  \"agent_case_file\": \"agent/cases.jsonl\",\n",
            "  \"agent_case_sha256\": \"{}\",\n",
            "  \"evaluator_key_file\": \"evaluator/answer-key.jsonl\",\n",
            "  \"evaluator_key_sha256\": \"{}\",\n",
            "  \"agent_key_separation\": \"separate-files; answer key must never be shown to candidate\",\n",
            "  \"seed_is_qualification_secret\": false\n",
            "}}\n"
        ),
        config.seed,
        config.factors,
        config.radius,
        case_count,
        (config.factors * 4).max(32),
        agent_sha,
        key_sha
    );
    fs::write(evaluator_dir.join("manifest.json"), manifest.as_bytes())
        .map_err(|error| format!("write manifest: {error}"))?;

    let manifest_sha = hex(&sha256(manifest.as_bytes()));
    Ok(format!(
        concat!(
            "{{\"schema_version\":\"cmaq-causal-generator-report-v1\",",
            "\"seed\":{},\"factors\":{},\"radius\":{},\"cases\":{},",
            "\"agent_cases_sha256\":\"{}\",\"answer_key_sha256\":\"{}\",",
            "\"manifest_sha256\":\"{}\"}}"
        ),
        config.seed, config.factors, config.radius, case_count, agent_sha, key_sha, manifest_sha
    ))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let config = parse_args(&args)?;
    let report = write_outputs(&config)?;
    println!("{report}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cmaq-causal-generator: {error}");
        eprintln!("usage: cmaq-causal-generator --seed N --factors 6..64 --radius 1..8 --out DIR");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn generation_is_byte_deterministic() {
        let first = generate_contents(4242, 8, 2).expect("valid generation");
        let second = generate_contents(4242, 8, 2).expect("same seed");
        assert_eq!(first, second);
        assert_eq!(first.2, 64);
    }

    #[test]
    fn hidden_mechanisms_match_the_emitted_intervention_histories() {
        let cases = generate_cases(12345, 8, 2).expect("valid generation");
        for case in cases {
            assert_eq!(case.mask.count_ones() as usize, 2);
            let target = case.target_action();
            for sample in &case.examples {
                let historical_target =
                    ((sample.context & case.mask).count_ones() % 2 == 1) ^ case.offset;
                assert_eq!(sample.success, sample.action == historical_target);
            }
            assert_eq!(
                target,
                ((case.context & case.mask).count_ones() % 2 == 1) ^ case.offset
            );
            assert!(!case.examples.iter().any(|sample| sample.context == case.context));
        }
    }

    #[test]
    fn rejects_invalid_factor_dimensions_and_radius() {
        assert!(generate_cases(1, 5, 2).is_err());
        assert!(generate_cases(1, 8, 0).is_err());
        assert!(generate_cases(1, 8, 9).is_err());
    }

    #[test]
    fn mask_and_context_dimensions_are_preserved() {
        let cases = generate_cases(77, 64, 8).expect("64 factors are supported");
        assert!(cases.iter().all(|case| case.mask.count_ones() == 8));
        let top_bit = bit_string(1_u64 << 63, 64);
        assert_eq!(top_bit, format!("{}1", "0".repeat(63)));
        assert_eq!(bit_string(1, 6), "100000");
        assert_eq!(bit_string(32, 6), "000001");
    }
}
