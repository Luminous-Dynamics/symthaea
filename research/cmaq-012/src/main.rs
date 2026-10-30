use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process;

const SCHEMA: &str = "cmaq-012-paired-aop-report-v0.1";
const AUDITOR_VERSION: &str = "0.1.0";
const MIN_QUALIFICATION_PAIRS: usize = 20;
const HEADER: &str = "task\tcapability\tsplit\tseed\tarm\tinitial_state_sha256\tproposer_sha256\tcandidate_set_sha256\tenvironment_config_sha256\tadapter_sha256\tsource_sha256\tcheckpoint_sha256\tcandidate_count\trollout_horizon\tplanner_compute_budget_units\tstatus\treturn\ttask_success\tlatency_ms";
const ARMS: [Arm; 5] = [Arm::D0, Arm::D1, Arm::W1, Arm::W2, Arm::W3];
const CONTRASTS: [(&str, Arm, Arm); 5] = [
    ("advantage_of_prediction", Arm::W1, Arm::D1),
    ("deployment_gain", Arm::W1, Arm::D0),
    ("prediction_mediation", Arm::W1, Arm::W2),
    ("oracle_gap", Arm::W3, Arm::W1),
    ("compute_only_gain", Arm::D1, Arm::D0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Arm { D0, D1, W1, W2, W3 }

impl Arm {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "D0" => Ok(Self::D0), "D1" => Ok(Self::D1), "W1" => Ok(Self::W1),
            "W2" => Ok(Self::W2), "W3" => Ok(Self::W3),
            _ => Err(format!("unknown arm {value:?}; expected D0, D1, W1, W2 or W3")),
        }
    }
    fn name(self) -> &'static str {
        match self { Self::D0 => "D0", Self::D1 => "D1", Self::W1 => "W1", Self::W2 => "W2", Self::W3 => "W3" }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status { Success, Failed, TimedOut }

impl Status {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "success" => Ok(Self::Success), "failed" => Ok(Self::Failed), "timed_out" => Ok(Self::TimedOut),
            _ => Err(format!("invalid status {value:?}; expected success, failed or timed_out")),
        }
    }
}

#[derive(Clone, Debug)]
struct Row {
    task: String, capability: String, split: String, seed: u64, arm: Arm,
    initial_state_sha256: String, proposer_sha256: String, candidate_set_sha256: String,
    environment_config_sha256: String, adapter_sha256: String, source_sha256: String,
    checkpoint_sha256: String, candidate_count: usize, rollout_horizon: usize,
    planner_compute_budget_units: u64, status: Status, episode_return: f64,
    task_success: bool, latency_ms: f64,
}

impl Row {
    fn parse(line: &str, line_number: usize) -> Result<Self, String> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 19 {
            return Err(format!("line {line_number}: expected 19 tab-separated fields, found {}", fields.len()));
        }
        let context = |message: String| format!("line {line_number}: {message}");
        let task = identifier(fields[0]).map_err(context)?;
        let capability = identifier(fields[1]).map_err(context)?;
        let split = identifier(fields[2]).map_err(context)?;
        let seed = fields[3].parse::<u64>().map_err(|_| context(format!("invalid seed {:?}", fields[3])))?;
        let arm = Arm::parse(fields[4]).map_err(context)?;
        let initial_state_sha256 = digest(fields[5]).map_err(context)?;
        let proposer_sha256 = digest(fields[6]).map_err(context)?;
        let candidate_set_sha256 = digest(fields[7]).map_err(context)?;
        let environment_config_sha256 = digest(fields[8]).map_err(context)?;
        let adapter_sha256 = digest(fields[9]).map_err(context)?;
        let source_sha256 = digest(fields[10]).map_err(context)?;
        let checkpoint_sha256 = digest(fields[11]).map_err(context)?;
        let candidate_count = fields[12].parse::<usize>().map_err(|_| context(format!("invalid candidate_count {:?}", fields[12])))?;
        let rollout_horizon = fields[13].parse::<usize>().map_err(|_| context(format!("invalid rollout_horizon {:?}", fields[13])))?;
        let planner_compute_budget_units = fields[14].parse::<u64>().map_err(|_| context(format!("invalid planner_compute_budget_units {:?}", fields[14])))?;
        if candidate_count == 0 || rollout_horizon == 0 || planner_compute_budget_units == 0 {
            return Err(context("candidate_count, rollout_horizon and planner_compute_budget_units must be positive".into()));
        }
        let status = Status::parse(fields[15]).map_err(context)?;
        let episode_return = finite_float(fields[16]).map_err(context)?;
        let task_success = match fields[17] {
            "0" => false, "1" => true,
            other => return Err(context(format!("task_success must be 0 or 1, found {other:?}"))),
        };
        if status != Status::Success && task_success {
            return Err(context("failed/timed_out rows must have task_success=0".into()));
        }
        let latency_ms = finite_float(fields[18]).map_err(context)?;
        if latency_ms < 0.0 { return Err(context("latency_ms cannot be negative".into())); }
        Ok(Self {
            task, capability, split, seed, arm, initial_state_sha256, proposer_sha256,
            candidate_set_sha256, environment_config_sha256, adapter_sha256, source_sha256,
            checkpoint_sha256, candidate_count, rollout_horizon, planner_compute_budget_units,
            status, episode_return, task_success, latency_ms,
        })
    }
}

fn identifier(value: &str) -> Result<String, String> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.') {
        return Err(format!("identifier {value:?} is empty or contains unsupported characters"));
    }
    Ok(value.to_string())
}
fn digest(value: &str) -> Result<String, String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("expected 64 hexadecimal SHA-256 characters, found {value:?}"));
    }
    Ok(value.to_ascii_lowercase())
}
fn finite_float(value: &str) -> Result<f64, String> {
    let parsed = value.parse::<f64>().map_err(|_| format!("invalid number {value:?}"))?;
    if !parsed.is_finite() { return Err(format!("non-finite numeric value {value:?}")); }
    Ok(parsed)
}

type EpisodeKey = (String, String, String, u64);
type ResultKey = (String, String, String);

fn parse_and_pair(input: &str) -> Result<Vec<(ResultKey, Vec<BTreeMap<Arm, Row>>)>, String> {
    let mut lines = input.lines();
    let header = lines.next().ok_or_else(|| "input is empty".to_string())?;
    if header != HEADER { return Err(format!("header mismatch; required exact header: {HEADER}")); }
    let mut episodes: BTreeMap<EpisodeKey, BTreeMap<Arm, Row>> = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        let line_number = index + 2;
        if line.is_empty() { return Err(format!("line {line_number}: blank rows are not allowed")); }
        let row = Row::parse(line, line_number)?;
        let key = (row.task.clone(), row.capability.clone(), row.split.clone(), row.seed);
        let arms = episodes.entry(key).or_default();
        if arms.insert(row.arm, row).is_some() {
            return Err(format!("duplicate arm row for paired episode at line {line_number}"));
        }
    }
    if episodes.is_empty() { return Err("input contains a header but no episode rows".into()); }

    let mut grouped: BTreeMap<ResultKey, Vec<BTreeMap<Arm, Row>>> = BTreeMap::new();
    for ((task, capability, split, seed), arms) in episodes {
        if arms.len() != ARMS.len() || ARMS.iter().any(|arm| !arms.contains_key(arm)) {
            let present = arms.keys().map(|arm| arm.name()).collect::<Vec<_>>().join(",");
            return Err(format!("unpaired episode {task}/{capability}/{split}/{seed}: expected D0,D1,W1,W2,W3; found [{present}]"));
        }
        validate_pair(&arms, &task, &capability, &split, seed)?;
        grouped.entry((task, capability, split)).or_default().push(arms);
    }
    Ok(grouped.into_iter().collect())
}

fn validate_pair(arms: &BTreeMap<Arm, Row>, task: &str, capability: &str, split: &str, seed: u64) -> Result<(), String> {
    let reference = arms.get(&Arm::D0).ok_or_else(|| "missing D0".to_string())?;
    for arm in ARMS.iter().skip(1) {
        let row = arms.get(arm).ok_or_else(|| format!("missing {}", arm.name()))?;
        let matching = row.initial_state_sha256 == reference.initial_state_sha256
            && row.proposer_sha256 == reference.proposer_sha256
            && row.candidate_set_sha256 == reference.candidate_set_sha256
            && row.environment_config_sha256 == reference.environment_config_sha256
            && row.adapter_sha256 == reference.adapter_sha256
            && row.candidate_count == reference.candidate_count
            && row.rollout_horizon == reference.rollout_horizon
            && row.planner_compute_budget_units == reference.planner_compute_budget_units;
        if !matching {
            return Err(format!("paired contract mismatch for {task}/{capability}/{split}/{seed} at arm {}: shared digests and compute/action budgets must match", arm.name()));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ArmMetrics {
    mean_return: f64, success_rate: f64, success_count: usize,
    failed_count: usize, timed_out_count: usize,
    mean_latency_ms: f64, p50_latency_ms: f64, p95_latency_ms: f64,
}
#[derive(Clone, Debug)]
struct Contrast {
    name: &'static str, numerator: &'static str, denominator: &'static str,
    return_mean_delta: f64, return_ci95_low: f64, return_ci95_high: f64,
    success_rate_delta: f64, success_ci95_low: f64, success_ci95_high: f64,
}
#[derive(Clone, Debug)]
struct GroupReport {
    task: String, capability: String, split: String, episode_count: usize,
    eligibility: &'static str, arms: BTreeMap<Arm, ArmMetrics>, contrasts: Vec<Contrast>,
}
#[derive(Clone, Debug)]
struct Report {
    input_sha256: String, bootstrap_replicates: usize, bootstrap_seed: u64,
    groups: Vec<GroupReport>,
}

fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let index = ((quantile * values.len() as f64).ceil() as usize).saturating_sub(1).min(values.len().saturating_sub(1));
    values[index]
}
fn summarize_arm(rows: &[BTreeMap<Arm, Row>], arm: Arm) -> ArmMetrics {
    let mut returns = Vec::with_capacity(rows.len());
    let mut latencies = Vec::with_capacity(rows.len());
    let mut success_count = 0usize;
    let mut failed_count = 0usize;
    let mut timed_out_count = 0usize;
    for episode in rows {
        let row = &episode[&arm];
        returns.push(row.episode_return);
        latencies.push(row.latency_ms);
        if row.task_success { success_count += 1; }
        match row.status { Status::Success => {}, Status::Failed => failed_count += 1, Status::TimedOut => timed_out_count += 1 }
    }
    let n = rows.len() as f64;
    let mean_return = returns.iter().sum::<f64>() / n;
    let success_rate = success_count as f64 / n;
    let mean_latency_ms = latencies.iter().sum::<f64>() / n;
    let p50_latency_ms = percentile(&mut latencies.clone(), 0.50);
    let p95_latency_ms = percentile(&mut latencies, 0.95);
    ArmMetrics { mean_return, success_rate, success_count, failed_count, timed_out_count, mean_latency_ms, p50_latency_ms, p95_latency_ms }
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
    fn index(&mut self, upper: usize) -> usize { (self.next_u64() % upper as u64) as usize }
}
fn mean(values: &[f64]) -> f64 { values.iter().sum::<f64>() / values.len() as f64 }
fn bootstrap_ci(values: &[f64], replicates: usize, seed: u64) -> (f64, f64) {
    let mut rng = SplitMix64::new(seed);
    let mut means = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut total = 0.0;
        for _ in 0..values.len() { total += values[rng.index(values.len())]; }
        means.push(total / values.len() as f64);
    }
    means.sort_by(|a, b| a.total_cmp(b));
    let low = percentile(&mut means, 0.025);
    let high = percentile(&mut means, 0.975);
    (low, high)
}

fn build_report(input: &str, replicates: usize, bootstrap_seed: u64) -> Result<Report, String> {
    if replicates < 100 { return Err("bootstrap replicates must be at least 100".into()); }
    let paired_groups = parse_and_pair(input)?;
    let mut groups = Vec::with_capacity(paired_groups.len());
    for (group_index, ((task, capability, split), episodes)) in paired_groups.into_iter().enumerate() {
        let episode_count = episodes.len();
        let eligibility = if split == "qualification" && episode_count >= MIN_QUALIFICATION_PAIRS {
            "qualification_candidate"
        } else { "smoke_only" };
        let mut arms = BTreeMap::new();
        for arm in ARMS { arms.insert(arm, summarize_arm(&episodes, arm)); }
        let mut contrasts = Vec::with_capacity(CONTRASTS.len());
        for (index, (name, numerator, denominator)) in CONTRASTS.iter().enumerate() {
            let mut return_deltas = Vec::with_capacity(episode_count);
            let mut success_deltas = Vec::with_capacity(episode_count);
            for episode in &episodes {
                let a = &episode[numerator];
                let b = &episode[denominator];
                return_deltas.push(a.episode_return - b.episode_return);
                success_deltas.push((a.task_success as i32 - b.task_success as i32) as f64);
            }
            let (return_ci95_low, return_ci95_high) = bootstrap_ci(
                &return_deltas, replicates, bootstrap_seed ^ (group_index as u64).rotate_left(11) ^ index as u64
            );
            let (success_ci95_low, success_ci95_high) = bootstrap_ci(
                &success_deltas, replicates, bootstrap_seed ^ (group_index as u64).rotate_left(17) ^ (index as u64 + 101)
            );
            contrasts.push(Contrast {
                name, numerator: numerator.name(), denominator: denominator.name(),
                return_mean_delta: mean(&return_deltas), return_ci95_low, return_ci95_high,
                success_rate_delta: mean(&success_deltas), success_ci95_low, success_ci95_high,
            });
        }
        groups.push(GroupReport { task, capability, split, episode_count, eligibility, arms, contrasts });
    }
    Ok(Report {
        input_sha256: sha256_hex(input.as_bytes()), bootstrap_replicates: replicates,
        bootstrap_seed, groups,
    })
}

fn number(value: f64) -> String {
    if value.is_finite() { format!("{value:.8}") } else { "null".to_string() }
}
fn render_report(report: &Report) -> String {
    let mut out = String::new();
    writeln!(&mut out, "{{").unwrap();
    writeln!(&mut out, "  \"schema_version\": \"{SCHEMA}\",").unwrap();
    writeln!(&mut out, "  \"auditor_version\": \"{AUDITOR_VERSION}\",").unwrap();
    writeln!(&mut out, "  \"input_sha256\": \"{}\",", report.input_sha256).unwrap();
    writeln!(&mut out, "  \"bootstrap_replicates\": {},", report.bootstrap_replicates).unwrap();
    writeln!(&mut out, "  \"bootstrap_seed\": {},", report.bootstrap_seed).unwrap();
    writeln!(&mut out, "  \"minimum_qualification_pairs\": {MIN_QUALIFICATION_PAIRS},").unwrap();
    writeln!(&mut out, "  \"groups\": [").unwrap();
    for (group_index, group) in report.groups.iter().enumerate() {
        let group_comma = if group_index + 1 == report.groups.len() { "" } else { "," };
        writeln!(&mut out, "    {{").unwrap();
        writeln!(&mut out, "      \"task\": \"{}\",", group.task).unwrap();
        writeln!(&mut out, "      \"capability\": \"{}\",", group.capability).unwrap();
        writeln!(&mut out, "      \"split\": \"{}\",", group.split).unwrap();
        writeln!(&mut out, "      \"episode_count\": {},", group.episode_count).unwrap();
        writeln!(&mut out, "      \"eligibility\": \"{}\",", group.eligibility).unwrap();
        writeln!(&mut out, "      \"arms\": {{").unwrap();
        for (arm_index, arm) in ARMS.iter().enumerate() {
            let m = &group.arms[arm];
            let comma = if arm_index + 1 == ARMS.len() { "" } else { "," };
            writeln!(&mut out,
                concat!("        \"{}\": {{\"mean_return\": {}, \"success_rate\": {}, ",
                "\"success_count\": {}, \"failed_count\": {}, \"timed_out_count\": {}, ",
                "\"mean_latency_ms\": {}, \"p50_latency_ms\": {}, \"p95_latency_ms\": {}}}{}"),
                arm.name(), number(m.mean_return), number(m.success_rate), m.success_count,
                m.failed_count, m.timed_out_count, number(m.mean_latency_ms),
                number(m.p50_latency_ms), number(m.p95_latency_ms), comma
            ).unwrap();
        }
        writeln!(&mut out, "      }},").unwrap();
        writeln!(&mut out, "      \"contrasts\": [").unwrap();
        for (contrast_index, contrast) in group.contrasts.iter().enumerate() {
            let comma = if contrast_index + 1 == group.contrasts.len() { "" } else { "," };
            writeln!(&mut out,
                concat!("        {{\"estimand\": \"{}\", \"numerator\": \"{}\", \"denominator\": \"{}\", ",
                "\"return_mean_delta\": {}, \"return_ci95_low\": {}, \"return_ci95_high\": {}, ",
                "\"success_rate_delta\": {}, \"success_ci95_low\": {}, \"success_ci95_high\": {}}}{}"),
                contrast.name, contrast.numerator, contrast.denominator,
                number(contrast.return_mean_delta), number(contrast.return_ci95_low),
                number(contrast.return_ci95_high), number(contrast.success_rate_delta),
                number(contrast.success_ci95_low), number(contrast.success_ci95_high), comma
            ).unwrap();
        }
        writeln!(&mut out, "      ]").unwrap();
        writeln!(&mut out, "    }}{group_comma}").unwrap();
    }
    writeln!(&mut out, "  ]").unwrap();
    writeln!(&mut out, "}}").unwrap();
    out
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
    let mut h = [0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut message = input.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 { message.push(0); }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let start = i * 4;
            w[i] = u32::from_be_bytes([chunk[start], chunk[start+1], chunk[start+2], chunk[start+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(choice).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            hh = g; g = f; f = e; e = d.wrapping_add(temp1);
            d = c; c = b; b = a; a = temp1.wrapping_add(temp2);
        }
        h[0]=h[0].wrapping_add(a); h[1]=h[1].wrapping_add(b); h[2]=h[2].wrapping_add(c); h[3]=h[3].wrapping_add(d);
        h[4]=h[4].wrapping_add(e); h[5]=h[5].wrapping_add(f); h[6]=h[6].wrapping_add(g); h[7]=h[7].wrapping_add(hh);
    }
    let mut output = String::with_capacity(64);
    for word in h { write!(&mut output, "{word:08x}").unwrap(); }
    output
}

fn parse_args() -> Result<(PathBuf, usize, u64), String> {
    let mut input = None;
    let mut replicates = 2000usize;
    let mut seed = 0xC0A0_0012_2026_1030u64;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--help" | "-h" => {
                println!("CMAQ-012 paired advantage-of-prediction auditor\n\nUsage: cargo run -- --input FILE [--replicates N] [--bootstrap-seed N]\n\nReads a strict TSV result ledger and emits paired return/success contrasts as JSON.");
                process::exit(0);
            }
            "--input" => { index += 1; input = Some(PathBuf::from(args.get(index).ok_or_else(|| "--input requires a path".to_string())?)); }
            "--replicates" => { index += 1; replicates = args.get(index).ok_or_else(|| "--replicates requires a number".to_string())?.parse().map_err(|_| "invalid --replicates value".to_string())?; }
            "--bootstrap-seed" => { index += 1; seed = args.get(index).ok_or_else(|| "--bootstrap-seed requires a number".to_string())?.parse().map_err(|_| "invalid --bootstrap-seed value".to_string())?; }
            other => return Err(format!("unknown argument {other:?}; use --help")),
        }
        index += 1;
    }
    Ok((input.ok_or_else(|| "--input is required".to_string())?, replicates, seed))
}

fn main() {
    let (path, replicates, seed) = match parse_args() {
        Ok(args) => args, Err(error) => { eprintln!("{error}"); process::exit(2); }
    };
    let input = match fs::read_to_string(&path) {
        Ok(input) => input, Err(error) => { eprintln!("failed to read {}: {error}", path.display()); process::exit(2); }
    };
    match build_report(&input, replicates, seed) {
        Ok(report) => print!("{}", render_report(&report)),
        Err(error) => { eprintln!("invalid CMAQ-012 input: {error}"); process::exit(1); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_digest(value: u64) -> String { format!("{value:064x}") }
    fn synthetic_tsv(n: usize, d0: f64, d1: f64, w1: f64, w2: f64, w3: f64) -> String {
        let mut text = format!("{HEADER}\n");
        let values = [d0, d1, w1, w2, w3];
        for seed in 0..n {
            for (index, arm) in ARMS.iter().enumerate() {
                let value = values[index];
                let success = if value > 0.0 { 1 } else { 0 };
                text.push_str(&format!(
                    "halfcheetah\tcontrol\tqualification\t{seed}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t64\t3\t100000\tsuccess\t{value:.4}\t{success}\t12.5\n",
                    arm.name(), test_digest(10 + seed as u64), test_digest(20),
                    test_digest(30), test_digest(40), test_digest(50),
                    test_digest(100 + index as u64), test_digest(200 + index as u64)
                ));
            }
        }
        text
    }
    fn with_timeout(input: &str) -> String {
        let mut lines: Vec<String> = input.lines().map(str::to_string).collect();
        for line in lines.iter_mut().skip(1) {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts[3] == "0" && parts[4] == "W1" {
                let mut fields: Vec<String> = parts.iter().map(|v| v.to_string()).collect();
                fields[15] = "timed_out".into();
                fields[16] = "-9.0".into();
                fields[17] = "0".into();
                *line = fields.join("\t");
                break;
            }
        }
        lines.join("\n") + "\n"
    }
    fn contrast<'a>(report: &'a Report, name: &str) -> &'a Contrast {
        report.groups[0].contrasts.iter().find(|row| row.name == name).unwrap()
    }
    fn alter_row_field(input: &str, line_index: usize, field_index: usize, value: &str) -> String {
        let mut lines: Vec<String> = input.lines().map(str::to_string).collect();
        let mut fields: Vec<String> = lines[line_index].split('\t').map(str::to_string).collect();
        fields[field_index] = value.to_string();
        lines[line_index] = fields.join("\t");
        lines.join("\n") + "\n"
    }

    #[test]
    fn sha256_known_answer() {
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
    #[test]
    fn positive_null_and_negative_advantage_are_distinguished() {
        let positive = build_report(&synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0),500,7).unwrap();
        let null = build_report(&synthetic_tsv(20,1.0,2.0,2.0,2.0,5.0),500,7).unwrap();
        let negative = build_report(&synthetic_tsv(20,1.0,2.0,1.0,2.0,5.0),500,7).unwrap();
        assert_eq!(contrast(&positive,"advantage_of_prediction").return_mean_delta,2.0);
        assert_eq!(contrast(&null,"advantage_of_prediction").return_mean_delta,0.0);
        assert_eq!(contrast(&negative,"advantage_of_prediction").return_mean_delta,-1.0);
    }
    #[test]
    fn negative_control_mediation_is_separate_from_compute_gain() {
        let report = build_report(&synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0),500,11).unwrap();
        assert_eq!(contrast(&report,"prediction_mediation").return_mean_delta,2.0);
        assert_eq!(contrast(&report,"compute_only_gain").return_mean_delta,1.0);
    }
    #[test]
    fn timed_out_episode_remains_in_the_paired_denominator() {
        let input = with_timeout(&synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0));
        let report = build_report(&input,500,11).unwrap();
        let metrics = &report.groups[0].arms[&Arm::W1];
        assert_eq!(metrics.timed_out_count,1);
        assert_eq!(metrics.success_count,19);
        assert_eq!(report.groups[0].episode_count,20);
        assert_eq!(contrast(&report,"advantage_of_prediction").return_mean_delta,1.35);
    }
    #[test]
    fn below_minimum_qualification_slice_is_smoke_only() {
        let report = build_report(&synthetic_tsv(4,1.0,2.0,4.0,2.0,5.0),500,11).unwrap();
        assert_eq!(report.groups[0].eligibility,"smoke_only");
    }
    #[test]
    fn mismatched_common_state_digest_is_rejected() {
        let input = synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0);
        let modified = alter_row_field(&input,2,5,&test_digest(9999));
        assert!(parse_and_pair(&modified).unwrap_err().contains("paired contract mismatch"));
    }
    #[test]
    fn missing_arm_and_duplicate_arm_are_rejected() {
        let input = synthetic_tsv(1,1.0,2.0,4.0,2.0,5.0);
        let lines: Vec<&str> = input.lines().collect();
        let missing = format!("{}\n{}\n",lines[0],lines[1]);
        assert!(parse_and_pair(&missing).unwrap_err().contains("expected D0,D1,W1,W2,W3"));
        let duplicate = format!("{}\n{}\n{}\n",lines[0],lines[1],lines[1]);
        assert!(parse_and_pair(&duplicate).unwrap_err().contains("duplicate arm row"));
    }
    #[test]
    fn unequal_compute_budgets_are_rejected() {
        let input = synthetic_tsv(1,1.0,2.0,4.0,2.0,5.0);
        let modified = alter_row_field(&input,2,14,"999");
        assert!(parse_and_pair(&modified).unwrap_err().contains("paired contract mismatch"));
    }
    #[test]
    fn bootstrap_is_deterministic() {
        let values = [1.0,2.0,4.0,8.0];
        assert_eq!(bootstrap_ci(&values,1000,123),bootstrap_ci(&values,1000,123));
    }
    #[test]
    fn deterministic_report_binds_exact_input_bytes() {
        let input = synthetic_tsv(4,1.0,2.0,4.0,2.0,5.0);
        let report = build_report(&input,500,42).unwrap();
        assert_eq!(report.input_sha256,sha256_hex(input.as_bytes()));
        assert_eq!(render_report(&report),render_report(&report));
    }
}
