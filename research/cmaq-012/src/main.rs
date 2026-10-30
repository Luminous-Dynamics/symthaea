use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process;

const SCHEMA: &str = "cmaq-012-prediction-advantage-report-v0.1";
const AUDITOR_VERSION: &str = env!("CARGO_PKG_VERSION");
const MIN_QUALIFICATION_PAIRS: usize = 20;
const BOOTSTRAP_REPLICATES: usize = 2_000;
const BOOTSTRAP_SEED: u64 = 0xC0A9_0012_2026_1030;
const HEADER: [&str; 16] = [
    "task_id", "capability", "split", "seed", "arm", "return", "success",
    "status", "initial_state_sha256", "proposer_sha256", "candidate_set_sha256",
    "environment_config_sha256", "candidate_count", "rollout_horizon",
    "planner_compute_units", "latency_ms",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Arm { D0, D1, W1, W2, W3 }
impl Arm {
    fn all() -> [Self; 5] { [Self::D0, Self::D1, Self::W1, Self::W2, Self::W3] }
    fn name(self) -> &'static str {
        match self { Self::D0 => "D0", Self::D1 => "D1", Self::W1 => "W1", Self::W2 => "W2", Self::W3 => "W3" }
    }
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "D0" => Ok(Self::D0), "D1" => Ok(Self::D1), "W1" => Ok(Self::W1),
            "W2" => Ok(Self::W2), "W3" => Ok(Self::W3),
            _ => Err(format!("unknown arm {value:?}; expected D0, D1, W1, W2 or W3")),
        }
    }
}

#[derive(Clone, Debug)]
struct Record {
    task_id: String, capability: String, split: String, seed: u64, arm: Arm,
    outcome: f64, success: bool, status: String,
    initial_state_sha256: String, proposer_sha256: String, candidate_set_sha256: String,
    environment_config_sha256: String, candidate_count: usize, rollout_horizon: usize,
    planner_compute_units: u64, latency_ms: f64,
}
type Episode = BTreeMap<Arm, Record>;
type EpisodeKey = (String, String, String, u64);
type SliceKey = (String, String, String);

#[derive(Clone, Copy, Debug)]
struct Estimate { mean: f64, low: f64, high: f64 }
#[derive(Clone, Copy, Debug)]
struct ArmSummary {
    mean_return: f64, success_rate: f64, latency_p50: f64, latency_p95: f64,
    complete: usize, failed: usize, timeout: usize,
}
#[derive(Clone, Debug)]
struct SliceSummary {
    task_id: String, capability: String, split: String, n_pairs: usize,
    qualification_floor_met: bool, reason: String, arms: BTreeMap<Arm, ArmSummary>,
    contrasts: BTreeMap<&'static str, (Estimate, Estimate)>,
}
#[derive(Clone, Copy, Debug)]
struct SplitMix64 { state: u64 }
impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn index(&mut self, upper: usize) -> usize {
        if upper == 0 { 0 } else { (self.next_u64() % upper as u64) as usize }
    }
}

fn parse_bool(value: &str, row: usize) -> Result<bool, String> {
    match value {
        "true" | "1" => Ok(true), "false" | "0" => Ok(false),
        _ => Err(format!("row {row}: success must be true/false or 1/0")),
    }
}
fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn parse_record(line: &str, row: usize) -> Result<Record, String> {
    let fields: Vec<&str> = line.trim_end_matches('\r').split('\t').collect();
    if fields.len() != HEADER.len() {
        return Err(format!("row {row}: expected {} tab-separated fields, got {}", HEADER.len(), fields.len()));
    }
    let nonempty = |index: usize, label: &str| -> Result<String, String> {
        if fields[index].is_empty() { Err(format!("row {row}: {label} must not be empty")) }
        else { Ok(fields[index].to_string()) }
    };
    let seed = fields[3].parse::<u64>().map_err(|_| format!("row {row}: invalid seed"))?;
    let outcome = fields[5].parse::<f64>().map_err(|_| format!("row {row}: invalid return"))?;
    let success = parse_bool(fields[6], row)?;
    let status = nonempty(7, "status")?;
    if !["complete", "failed", "timeout"].contains(&status.as_str()) {
        return Err(format!("row {row}: status must be complete, failed or timeout"));
    }
    if status != "complete" && success {
        return Err(format!("row {row}: failed/timeout episodes cannot be successful"));
    }
    if !outcome.is_finite() {
        return Err(format!("row {row}: return must be finite; failed/timeout rows need explicit numeric penalties"));
    }
    for (index, label) in [
        (8, "initial_state_sha256"), (9, "proposer_sha256"),
        (10, "candidate_set_sha256"), (11, "environment_config_sha256"),
    ] {
        if !valid_digest(fields[index]) {
            return Err(format!("row {row}: {label} must be exactly 64 hexadecimal characters"));
        }
    }
    let candidate_count = fields[12].parse::<usize>().map_err(|_| format!("row {row}: invalid candidate_count"))?;
    let rollout_horizon = fields[13].parse::<usize>().map_err(|_| format!("row {row}: invalid rollout_horizon"))?;
    let planner_compute_units = fields[14].parse::<u64>().map_err(|_| format!("row {row}: invalid planner_compute_units"))?;
    let latency_ms = fields[15].parse::<f64>().map_err(|_| format!("row {row}: invalid latency_ms"))?;
    if candidate_count == 0 || rollout_horizon == 0 || planner_compute_units == 0 {
        return Err(format!("row {row}: candidate_count, rollout_horizon and planner_compute_units must be positive"));
    }
    if !latency_ms.is_finite() || latency_ms < 0.0 {
        return Err(format!("row {row}: latency_ms must be finite and non-negative"));
    }
    let split = nonempty(2, "split")?;
    if split != "development" && split != "qualification" {
        return Err(format!("row {row}: split must be development or qualification"));
    }
    Ok(Record {
        task_id: nonempty(0, "task_id")?, capability: nonempty(1, "capability")?,
        split, seed, arm: Arm::parse(fields[4])?, outcome, success, status,
        initial_state_sha256: fields[8].to_ascii_lowercase(),
        proposer_sha256: fields[9].to_ascii_lowercase(),
        candidate_set_sha256: fields[10].to_ascii_lowercase(),
        environment_config_sha256: fields[11].to_ascii_lowercase(),
        candidate_count, rollout_horizon, planner_compute_units, latency_ms,
    })
}

fn parse_and_validate(input: &str) -> Result<BTreeMap<SliceKey, Vec<Episode>>, String> {
    let mut lines = input.lines();
    let header = lines.next().ok_or_else(|| "input is empty".to_string())?;
    let header_fields: Vec<&str> = header.trim_end_matches('\r').split('\t').collect();
    if header_fields != HEADER {
        return Err(format!("invalid TSV header; expected exactly: {}", HEADER.join("\t")));
    }
    let mut episodes: BTreeMap<EpisodeKey, Episode> = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        let line_no = index + 2;
        if line.trim().is_empty() { continue; }
        let record = parse_record(line, line_no)?;
        let key = (record.task_id.clone(), record.capability.clone(), record.split.clone(), record.seed);
        let episode = episodes.entry(key).or_default();
        if episode.insert(record.arm, record.clone()).is_some() {
            return Err(format!("row {line_no}: duplicate {} arm for the same task/capability/split/seed", record.arm.name()));
        }
    }
    if episodes.is_empty() { return Err("input has a header but no episode rows".to_string()); }

    let mut slices: BTreeMap<SliceKey, Vec<Episode>> = BTreeMap::new();
    for ((task, capability, split, seed), episode) in episodes {
        for arm in Arm::all() {
            if !episode.contains_key(&arm) {
                return Err(format!("missing {} arm for {task}/{capability}/{split}/seed={seed}", arm.name()));
            }
        }
        let reference = episode.get(&Arm::D0).expect("validated D0 arm");
        for arm in Arm::all().into_iter().skip(1) {
            let record = episode.get(&arm).expect("validated arm");
            if record.initial_state_sha256 != reference.initial_state_sha256
                || record.proposer_sha256 != reference.proposer_sha256
                || record.candidate_set_sha256 != reference.candidate_set_sha256
                || record.environment_config_sha256 != reference.environment_config_sha256
                || record.candidate_count != reference.candidate_count
                || record.rollout_horizon != reference.rollout_horizon
                || record.planner_compute_units != reference.planner_compute_units
            {
                return Err(format!("mismatched pairing, proposal or budget metadata for {task}/{capability}/{split}/seed={seed} arm={}", arm.name()));
            }
        }
        slices.entry((task, capability, split)).or_default().push(episode);
    }
    Ok(slices)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}
fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() { return 0.0; }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * q).ceil() as usize).saturating_sub(1).min(sorted.len() - 1);
    sorted[index]
}
fn bootstrap_estimate(values: &[f64], seed: u64) -> Estimate {
    if values.is_empty() { return Estimate { mean: 0.0, low: 0.0, high: 0.0 }; }
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut total = 0.0;
        for _ in 0..values.len() { total += values[rng.index(values.len())]; }
        samples.push(total / values.len() as f64);
    }
    Estimate { mean: mean(values), low: percentile(&samples, 0.025), high: percentile(&samples, 0.975) }
}
fn hash64(input: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in input.bytes() { hash ^= byte as u64; hash = hash.wrapping_mul(0x100000001b3); }
    hash
}
fn return_value(record: &Record) -> f64 { record.outcome }
fn success_value(record: &Record) -> f64 { if record.success { 1.0 } else { 0.0 } }
fn make_contrast(episodes: &[Episode], left: Arm, right: Arm, field: fn(&Record) -> f64, seed: u64) -> Estimate {
    let differences: Vec<f64> = episodes.iter().map(|episode| {
        field(episode.get(&left).expect("complete arm set")) - field(episode.get(&right).expect("complete arm set"))
    }).collect();
    bootstrap_estimate(&differences, seed ^ ((left as u64) << 8) ^ right as u64)
}
fn summarize_arm(episodes: &[Episode], arm: Arm) -> ArmSummary {
    let rows: Vec<&Record> = episodes.iter().map(|e| e.get(&arm).expect("complete arm set")).collect();
    let returns: Vec<f64> = rows.iter().map(|r| r.outcome).collect();
    let successes: Vec<f64> = rows.iter().map(|r| if r.success { 1.0 } else { 0.0 }).collect();
    let latencies: Vec<f64> = rows.iter().map(|r| r.latency_ms).collect();
    ArmSummary {
        mean_return: mean(&returns), success_rate: mean(&successes),
        latency_p50: percentile(&latencies, 0.50), latency_p95: percentile(&latencies, 0.95),
        complete: rows.iter().filter(|r| r.status == "complete").count(),
        failed: rows.iter().filter(|r| r.status == "failed").count(),
        timeout: rows.iter().filter(|r| r.status == "timeout").count(),
    }
}
fn summarize_slice(key: SliceKey, episodes: &[Episode]) -> SliceSummary {
    let (task_id, capability, split) = key;
    let n_pairs = episodes.len();
    let qualification_floor_met = split == "qualification" && n_pairs >= MIN_QUALIFICATION_PAIRS;
    let reason = if split != "qualification" {
        "development split is never qualification eligible"
    } else if n_pairs < MIN_QUALIFICATION_PAIRS {
        "below the minimum of 20 paired episodes; smoke-only"
    } else {
        "mechanical sample-count floor met; held-out split and preregistration must be independently verified"
    }.to_string();
    let mut arms = BTreeMap::new();
    for arm in Arm::all() { arms.insert(arm, summarize_arm(episodes, arm)); }
    let slice_seed = hash64(&format!("{task_id}|{capability}|{split}"));
    let contrasts = [
        ("aop_w1_minus_d1", Arm::W1, Arm::D1),
        ("deployment_w1_minus_d0", Arm::W1, Arm::D0),
        ("prediction_mediation_w1_minus_w2", Arm::W1, Arm::W2),
        ("oracle_gap_w3_minus_w1", Arm::W3, Arm::W1),
        ("compute_only_d1_minus_d0", Arm::D1, Arm::D0),
    ];
    let mut contrast_map = BTreeMap::new();
    for (name, left, right) in contrasts {
        let name_seed = hash64(name);
        let returns = make_contrast(episodes, left, right, return_value, BOOTSTRAP_SEED ^ slice_seed ^ name_seed);
        let success = make_contrast(episodes, left, right, success_value, BOOTSTRAP_SEED ^ slice_seed ^ name_seed ^ 0x51CC_E551);
        contrast_map.insert(name, (returns, success));
    }
    SliceSummary { task_id, capability, split, n_pairs, qualification_floor_met, reason, arms, contrasts: contrast_map }
}
fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"), '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push(' '), c => out.push(c),
        }
    }
    out.push('"'); out
}
fn number(value: f64) -> String {
    if value.is_finite() { format!("{value:.8}") } else { "null".to_string() }
}
fn estimate_json(value: Estimate) -> String {
    format!("{{\"mean\":{},\"ci95_low\":{},\"ci95_high\":{}}}", number(value.mean), number(value.low), number(value.high))
}
fn arm_json(arm: Arm, summary: ArmSummary) -> String {
    format!(
        concat!("\"{}\":{{\"mean_return\":{},\"success_rate\":{},",
        "\"latency_p50_ms\":{},\"latency_p95_ms\":{},\"complete\":{},\"failed\":{},\"timeout\":{}}}"),
        arm.name(), number(summary.mean_return), number(summary.success_rate),
        number(summary.latency_p50), number(summary.latency_p95),
        summary.complete, summary.failed, summary.timeout
    )
}
fn slice_json(slice: &SliceSummary) -> String {
    let arms = Arm::all().iter().map(|arm| arm_json(*arm, slice.arms[arm])).collect::<Vec<_>>().join(",");
    let contrasts = slice.contrasts.iter().map(|(name, (returns, success))| {
        format!("{}:{{\"return\":{},\"success_rate\":{}}}", json_string(name), estimate_json(*returns), estimate_json(*success))
    }).collect::<Vec<_>>().join(",");
    format!(
        concat!("{{\"task_id\":{},\"capability\":{},\"split\":{},\"n_pairs\":{},",
        "\"qualification_floor_met\":{},\"qualification_floor_reason\":{},\"arms\":{{{}}},\"contrasts\":{{{}}}}}"),
        json_string(&slice.task_id), json_string(&slice.capability), json_string(&slice.split),
        slice.n_pairs, slice.qualification_floor_met, json_string(&slice.reason), arms, contrasts
    )
}

fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let mut message = input.to_vec();
    let bit_len = (message.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 { message.push(0); }
    message.extend_from_slice(&bit_len.to_be_bytes());
    let mut h = [0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32;64];
        for (i, bytes) in chunk.chunks_exact(4).enumerate() { w[i] = u32::from_be_bytes([bytes[0],bytes[1],bytes[2],bytes[3]]); }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let [mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh=g; g=f; f=e; e=d.wrapping_add(t1); d=c; c=b; b=a; a=t1.wrapping_add(t2);
        }
        h[0]=h[0].wrapping_add(a); h[1]=h[1].wrapping_add(b); h[2]=h[2].wrapping_add(c); h[3]=h[3].wrapping_add(d);
        h[4]=h[4].wrapping_add(e); h[5]=h[5].wrapping_add(f); h[6]=h[6].wrapping_add(g); h[7]=h[7].wrapping_add(hh);
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}
fn render_report(input: &str, slices: &BTreeMap<SliceKey, Vec<Episode>>) -> String {
    let summaries: Vec<SliceSummary> = slices.iter().map(|(key, episodes)| summarize_slice(key.clone(), episodes)).collect();
    let summary_json = summaries.iter().map(slice_json).collect::<Vec<_>>().join(",\n    ");
    format!(
        concat!("{{\n  \"schema_version\":\"{}\",\n  \"auditor_version\":\"{}\",\n",
        "  \"input_sha256\":\"{}\",\n  \"minimum_qualification_pairs\":{},\n",
        "  \"bootstrap_replicates\":{},\n  \"claim_ceiling\":\"scoring_contract_only_not_environment_performance\",\n  \"slice_count\":{},\n  \"slices\":[\n    {}\n  ]\n}}\n"),
        SCHEMA, AUDITOR_VERSION, sha256_hex(input.as_bytes()), MIN_QUALIFICATION_PAIRS,
        BOOTSTRAP_REPLICATES, summaries.len(), summary_json
    )
}
fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("CMAQ-012 paired advantage-of-prediction auditor\n\nUsage: cmaq-012 <episodes.tsv>\n\nValidates D0/D1/W1/W2/W3 paired rows, then emits a deterministic JSON report.");
        return;
    }
    if args.len() != 1 { eprintln!("usage: cmaq-012 <episodes.tsv>"); process::exit(2); }
    let input = fs::read_to_string(&args[0]).unwrap_or_else(|error| {
        eprintln!("cannot read {}: {error}", args[0]); process::exit(2);
    });
    let slices = parse_and_validate(&input).unwrap_or_else(|error| {
        eprintln!("invalid paired benchmark input: {error}"); process::exit(1);
    });
    print!("{}", render_report(&input, &slices));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smoke_input() -> &'static str {
        include_str!("../fixtures/smoke.tsv")
    }

    fn qualification_input() -> String {
        smoke_input().replace("\tdevelopment\t", "\tqualification\t")
    }

    #[test]
    fn sha256_matches_known_answer() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn development_fixture_has_all_paired_arms_and_positive_aop() {
        let slices = parse_and_validate(smoke_input()).unwrap();
        assert_eq!(slices.len(), 1);
        let slice = summarize_slice(slices.keys().next().unwrap().clone(), slices.values().next().unwrap());
        assert_eq!(slice.n_pairs, 20);
        assert_eq!(slice.split, "development");
        assert!(!slice.qualification_floor_met);
        assert!(slice.reason.contains("development split"));
        assert!(slice.contrasts["aop_w1_minus_d1"].0.mean > 0.0);
        assert!(slice.contrasts["prediction_mediation_w1_minus_w2"].0.mean > 0.0);
    }

    #[test]
    fn duplicate_arm_is_rejected() {
        let input = smoke_input();
        let duplicate = input.lines().nth(1).unwrap();
        let mutated = format!("{input}{duplicate}\n");
        assert!(parse_and_validate(&mutated).unwrap_err().contains("duplicate"));
    }

    #[test]
    fn missing_arm_is_rejected() {
        let input = smoke_input();
        let mut lines: Vec<&str> = input.lines().collect();
        lines.remove(1);
        let mutated = format!("{}\n", lines.join("\n"));
        assert!(parse_and_validate(&mutated).unwrap_err().contains("missing D0 arm"));
    }

    #[test]
    fn mismatched_candidate_digest_is_rejected() {
        let mut lines: Vec<String> = smoke_input().lines().map(str::to_string).collect();
        let mut fields: Vec<String> = lines[2].split('\t').map(str::to_string).collect();
        fields[10] = "f".repeat(64);
        lines[2] = fields.join("\t");
        assert!(parse_and_validate(&lines.join("\n")).unwrap_err().contains("mismatched pairing"));
    }

    #[test]
    fn failed_and_timed_out_rows_are_retained_in_the_denominator() {
        // The checked-in smoke fixture already contains one explicit timeout
        // with its finite penalty. Retain it in the exact paired denominator.
        let slices = parse_and_validate(smoke_input()).unwrap();
        let slice = summarize_slice(slices.keys().next().unwrap().clone(), slices.values().next().unwrap());
        assert_eq!(slice.arms[&Arm::D0].timeout, 1);
        assert_eq!(slice.n_pairs, 20);
    }

    #[test]
    fn twenty_paired_rows_meet_only_the_mechanical_qualification_floor() {
        let input = qualification_input();
        let slices = parse_and_validate(&input).unwrap();
        let slice = summarize_slice(slices.keys().next().unwrap().clone(), slices.values().next().unwrap());
        assert!(slice.qualification_floor_met);
        assert!(slice.reason.contains("held-out split and preregistration must be independently verified"));
    }

    #[test]
    fn nineteen_pairs_do_not_meet_the_mechanical_qualification_floor() {
        let input = qualification_input();
        let lines: Vec<&str> = input.lines().take(1 + 19 * 5).collect();
        let mutated = format!("{}\n", lines.join("\n"));
        let slices = parse_and_validate(&mutated).unwrap();
        let slice = summarize_slice(slices.keys().next().unwrap().clone(), slices.values().next().unwrap());
        assert!(!slice.qualification_floor_met);
        assert!(slice.reason.contains("below the minimum"));
    }

    #[test]
    fn shuffled_future_negative_control_can_be_worse_than_world_model() {
        let mut lines: Vec<String> = smoke_input().lines().map(str::to_string).collect();
        for line in lines.iter_mut().skip(1) {
            let mut fields: Vec<String> = line.split('\t').map(str::to_string).collect();
            if fields[4] == "W2" {
                let prior = fields[5].parse::<f64>().unwrap();
                fields[5] = (prior - 0.25).to_string();
            }
            *line = fields.join("\t");
        }
        let slices = parse_and_validate(&lines.join("\n")).unwrap();
        let slice = summarize_slice(slices.keys().next().unwrap().clone(), slices.values().next().unwrap());
        assert!(slice.contrasts["prediction_mediation_w1_minus_w2"].0.mean > 0.20);
    }

    #[test]
    fn negative_aop_is_reported_not_relabelled() {
        let mut lines: Vec<String> = smoke_input().lines().map(str::to_string).collect();
        for line in lines.iter_mut().skip(1) {
            let mut fields: Vec<String> = line.split('\t').map(str::to_string).collect();
            if fields[4] == "W1" {
                let prior = fields[5].parse::<f64>().unwrap();
                fields[5] = (prior - 0.30).to_string();
            }
            *line = fields.join("\t");
        }
        let slices = parse_and_validate(&lines.join("\n")).unwrap();
        let slice = summarize_slice(slices.keys().next().unwrap().clone(), slices.values().next().unwrap());
        assert!(slice.contrasts["aop_w1_minus_d1"].0.mean < 0.0);
        assert!(slice.contrasts["aop_w1_minus_d1"].0.high < 0.0);
    }

    #[test]
    fn bootstrap_is_deterministic() {
        let a = bootstrap_estimate(&[0.2, 0.4, 0.6, 0.8], 42);
        let b = bootstrap_estimate(&[0.2, 0.4, 0.6, 0.8], 42);
        assert_eq!(a.mean, b.mean);
        assert_eq!(a.low, b.low);
        assert_eq!(a.high, b.high);
    }
}
