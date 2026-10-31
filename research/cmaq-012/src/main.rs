use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const SCHEMA: &str = "cmaq-012-paired-aop-report-v0.2";
const AUDITOR_VERSION: &str = "0.2.0";
const MIN_QUALIFICATION_PAIRS: usize = 20;
const COMPUTE_WORK_TOLERANCE: f64 = 0.05;
const LATENCY_TOLERANCE: f64 = 0.05;
const MEMORY_TOLERANCE: f64 = 0.10;
const HEADER: &str = "task\tcapability\tsplit\tseed\tarm\tinitial_state_sha256\tproposer_sha256\tcandidate_set_sha256\tenvironment_config_sha256\tadapter_sha256\tsource_sha256\tcheckpoint_sha256\tcandidate_count\trollout_horizon\tplanner_compute_budget_units\tstatus\treturn\ttask_success\tlatency_ms\tresource_profile_sha256\tprofiling_receipt_sha256\tresource_measurement_method\tcompute_work_units\tpeak_memory_bytes\tenergy_millijoules\tdirect_value_calls\tworld_model_forward_calls\tworld_model_transition_calls\toracle_transition_calls\tenvironment_transition_calls\toptimizer_iterations";
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
enum ResourceMeasurementMethod { OperatorTrace, HardwareCounter, StaticEstimate, Unavailable }

impl ResourceMeasurementMethod {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "operator_trace" => Ok(Self::OperatorTrace),
            "hardware_counter" => Ok(Self::HardwareCounter),
            "static_estimate" => Ok(Self::StaticEstimate),
            "unavailable" => Ok(Self::Unavailable),
            _ => Err(format!("invalid resource_measurement_method {value:?}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::OperatorTrace => "operator_trace",
            Self::HardwareCounter => "hardware_counter",
            Self::StaticEstimate => "static_estimate",
            Self::Unavailable => "unavailable",
        }
    }

    fn supports_compute_matching(self) -> bool {
        matches!(self, Self::OperatorTrace | Self::HardwareCounter)
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
    resource_profile_sha256: String,
    profiling_receipt_sha256: String,
    resource_measurement_method: ResourceMeasurementMethod,
    compute_work_units: Option<u64>,
    peak_memory_bytes: Option<u64>,
    energy_millijoules: Option<f64>,
    direct_value_calls: u64,
    world_model_forward_calls: u64,
    world_model_transition_calls: u64,
    oracle_transition_calls: u64,
    environment_transition_calls: u64,
    optimizer_iterations: u64,
}

impl Row {
    fn parse(line: &str, line_number: usize) -> Result<Self, String> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 31 {
            return Err(format!("line {line_number}: expected 31 tab-separated fields, found {}", fields.len()));
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
        let resource_profile_sha256 = digest(fields[19]).map_err(context)?;
        let profiling_receipt_sha256 = digest(fields[20]).map_err(context)?;
        let resource_measurement_method =
            ResourceMeasurementMethod::parse(fields[21]).map_err(context)?;
        let compute_work_units = optional_u64(fields[22]).map_err(context)?;
        let peak_memory_bytes = optional_u64(fields[23]).map_err(context)?;
        let energy_millijoules = optional_float(fields[24]).map_err(context)?;
        if energy_millijoules.is_some_and(|value| value < 0.0) {
            return Err(context("energy_millijoules cannot be negative".into()));
        }
        let direct_value_calls = fields[25].parse::<u64>()
            .map_err(|_| context(format!("invalid direct_value_calls {:?}", fields[25])))?;
        let world_model_forward_calls = fields[26].parse::<u64>()
            .map_err(|_| context(format!("invalid world_model_forward_calls {:?}", fields[26])))?;
        let world_model_transition_calls = fields[27].parse::<u64>()
            .map_err(|_| context(format!("invalid world_model_transition_calls {:?}", fields[27])))?;
        let oracle_transition_calls = fields[28].parse::<u64>()
            .map_err(|_| context(format!("invalid oracle_transition_calls {:?}", fields[28])))?;
        let environment_transition_calls = fields[29].parse::<u64>()
            .map_err(|_| context(format!("invalid environment_transition_calls {:?}", fields[29])))?;
        let optimizer_iterations = fields[30].parse::<u64>()
            .map_err(|_| context(format!("invalid optimizer_iterations {:?}", fields[30])))?;

        if resource_measurement_method.supports_compute_matching()
            && (compute_work_units.unwrap_or(0) == 0 || peak_memory_bytes.unwrap_or(0) == 0)
        {
            return Err(context(
                "operator_trace/hardware_counter require positive compute_work_units and peak_memory_bytes".into()
            ));
        }
        if resource_measurement_method == ResourceMeasurementMethod::Unavailable
            && (compute_work_units.is_some() || peak_memory_bytes.is_some() || energy_millijoules.is_some())
        {
            return Err(context(
                "unavailable resource method requires null compute, memory and energy values".into()
            ));
        }

        Ok(Self {
            task, capability, split, seed, arm, initial_state_sha256, proposer_sha256,
            candidate_set_sha256, environment_config_sha256, adapter_sha256, source_sha256,
            checkpoint_sha256, candidate_count, rollout_horizon, planner_compute_budget_units,
            status, episode_return, task_success, latency_ms,
            resource_profile_sha256, profiling_receipt_sha256, resource_measurement_method,
            compute_work_units, peak_memory_bytes, energy_millijoules,
            direct_value_calls, world_model_forward_calls, world_model_transition_calls,
            oracle_transition_calls, environment_transition_calls, optimizer_iterations,
        })
    }
}


#[derive(Clone, Debug, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    fn get(&self, key: &str) -> Result<&JsonValue, String> {
        match self {
            Self::Object(values) => values.get(key).ok_or_else(|| format!("missing JSON field {key:?}")),
            _ => Err("expected JSON object".into()),
        }
    }

    fn as_str(&self, field: &str) -> Result<&str, String> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err(format!("JSON field {field:?} must be a string")),
        }
    }

    fn as_bool(&self, field: &str) -> Result<bool, String> {
        match self {
            Self::Bool(value) => Ok(*value),
            _ => Err(format!("JSON field {field:?} must be a boolean")),
        }
    }

    fn as_u64(&self, field: &str) -> Result<u64, String> {
        match self {
            Self::Number(value) if !value.contains(['.', 'e', 'E', '-']) => value.parse::<u64>()
                .map_err(|_| format!("JSON field {field:?} must be a non-negative integer")),
            _ => Err(format!("JSON field {field:?} must be a non-negative integer")),
        }
    }

    fn as_f64(&self, field: &str) -> Result<f64, String> {
        match self {
            Self::Number(value) => {
                let parsed = value.parse::<f64>().map_err(|_| format!("JSON field {field:?} is not numeric"))?;
                if parsed.is_finite() { Ok(parsed) } else { Err(format!("JSON field {field:?} is not finite")) }
            }
            _ => Err(format!("JSON field {field:?} must be numeric")),
        }
    }

    fn optional_u64(&self, field: &str) -> Result<Option<u64>, String> {
        match self {
            Self::Null => Ok(None),
            value => value.as_u64(field).map(Some),
        }
    }

    fn optional_f64(&self, field: &str) -> Result<Option<f64>, String> {
        match self {
            Self::Null => Ok(None),
            value => value.as_f64(field).map(Some),
        }
    }

    fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { bytes: input.as_bytes(), position: 0 }
    }

    fn whitespace(&mut self) {
        while self.position < self.bytes.len()
            && matches!(self.bytes[self.position], b' ' | b'\n' | b'\r' | b'\t')
        {
            self.position += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> Result<(), String> {
        self.whitespace();
        if self.bytes.get(self.position) == Some(&expected) {
            self.position += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at JSON byte {}", expected as char, self.position))
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.consume(b'"')?;
        let mut output = Vec::new();
        while self.position < self.bytes.len() {
            let byte = self.bytes[self.position];
            self.position += 1;
            match byte {
                b'"' => {
                    return String::from_utf8(output).map_err(|_| "JSON string is not valid UTF-8".into());
                }
                b'\\' => {
                    let escaped = *self.bytes.get(self.position)
                        .ok_or_else(|| "unterminated JSON escape".to_string())?;
                    self.position += 1;
                    match escaped {
                        b'"' => output.push(b'"'),
                        b'\\' => output.push(b'\\'),
                        b'/' => output.push(b'/'),
                        b'b' => output.push(0x08),
                        b'f' => output.push(0x0c),
                        b'n' => output.push(b'\n'),
                        b'r' => output.push(b'\r'),
                        b't' => output.push(b'\t'),
                        b'u' => {
                            let first = self.parse_hex_quad()?;
                            let codepoint = if (0xD800..=0xDBFF).contains(&first) {
                                if self.bytes.get(self.position..self.position + 2) != Some(b"\\u") {
                                    return Err("high surrogate must be followed by a low surrogate".into());
                                }
                                self.position += 2;
                                let second = self.parse_hex_quad()?;
                                if !(0xDC00..=0xDFFF).contains(&second) {
                                    return Err("invalid low surrogate in JSON string".into());
                                }
                                0x10000 + (((first - 0xD800) as u32) << 10) + (second - 0xDC00) as u32
                            } else if (0xDC00..=0xDFFF).contains(&first) {
                                return Err("unpaired low surrogate in JSON string".into());
                            } else {
                                first as u32
                            };
                            let character = char::from_u32(codepoint)
                                .ok_or_else(|| "invalid Unicode scalar in JSON string".to_string())?;
                            let mut buffer = [0u8; 4];
                            output.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
                        }
                        _ => return Err(format!("invalid JSON escape byte {escaped:?}")),
                    }
                }
                0x00..=0x1f => return Err("unescaped control byte in JSON string".into()),
                other => output.push(other),
            }
        }
        Err("unterminated JSON string".into())
    }

    fn parse_hex_quad(&mut self) -> Result<u16, String> {
        if self.position + 4 > self.bytes.len() {
            return Err("truncated Unicode escape".into());
        }
        let mut value = 0u16;
        for byte in &self.bytes[self.position..self.position + 4] {
            value = value.checked_mul(16).ok_or_else(|| "Unicode escape overflow".to_string())?;
            value += match byte {
                b'0'..=b'9' => (byte - b'0') as u16,
                b'a'..=b'f' => (byte - b'a' + 10) as u16,
                b'A'..=b'F' => (byte - b'A' + 10) as u16,
                _ => return Err("invalid hexadecimal digit in Unicode escape".into()),
            };
        }
        self.position += 4;
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.position;
        if self.bytes.get(self.position) == Some(&b'-') { self.position += 1; }
        match self.bytes.get(self.position) {
            Some(b'0') => self.position += 1,
            Some(b'1'..=b'9') => {
                self.position += 1;
                while self.bytes.get(self.position).is_some_and(u8::is_ascii_digit) {
                    self.position += 1;
                }
            }
            _ => return Err(format!("invalid JSON number at byte {start}")),
        }
        if self.bytes.get(self.position) == Some(&b'.') {
            self.position += 1;
            let fraction_start = self.position;
            while self.bytes.get(self.position).is_some_and(u8::is_ascii_digit) {
                self.position += 1;
            }
            if self.position == fraction_start {
                return Err("JSON number has an empty fractional part".into());
            }
        }
        if matches!(self.bytes.get(self.position), Some(b'e' | b'E')) {
            self.position += 1;
            if matches!(self.bytes.get(self.position), Some(b'+' | b'-')) { self.position += 1; }
            let exponent_start = self.position;
            while self.bytes.get(self.position).is_some_and(u8::is_ascii_digit) {
                self.position += 1;
            }
            if self.position == exponent_start {
                return Err("JSON number has an empty exponent".into());
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.position])
            .map_err(|_| "invalid UTF-8 in JSON number".to_string())?;
        let parsed = text.parse::<f64>().map_err(|_| format!("invalid JSON number {text:?}"))?;
        if !parsed.is_finite() {
            return Err("JSON number is outside finite f64 range".into());
        }
        Ok(JsonValue::Number(text.to_string()))
    }

    fn value(&mut self) -> Result<JsonValue, String> {
        self.whitespace();
        match self.bytes.get(self.position).copied() {
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b'{') => {
                self.position += 1;
                self.whitespace();
                let mut values = BTreeMap::new();
                if self.bytes.get(self.position) == Some(&b'}') {
                    self.position += 1;
                    return Ok(JsonValue::Object(values));
                }
                loop {
                    self.whitespace();
                    if self.bytes.get(self.position) != Some(&b'"') {
                        return Err(format!("JSON object key must be a string at byte {}", self.position));
                    }
                    let key = self.parse_string()?;
                    self.consume(b':')?;
                    let value = self.value()?;
                    if values.insert(key.clone(), value).is_some() {
                        return Err(format!("duplicate JSON object key {key:?}"));
                    }
                    self.whitespace();
                    match self.bytes.get(self.position) {
                        Some(b',') => self.position += 1,
                        Some(b'}') => { self.position += 1; break; }
                        _ => return Err(format!("expected comma or closing brace at byte {}", self.position)),
                    }
                }
                Ok(JsonValue::Object(values))
            }
            Some(b'[') => {
                self.position += 1;
                self.whitespace();
                let mut values = Vec::new();
                if self.bytes.get(self.position) == Some(&b']') {
                    self.position += 1;
                    return Ok(JsonValue::Array(values));
                }
                loop {
                    values.push(self.value()?);
                    self.whitespace();
                    match self.bytes.get(self.position) {
                        Some(b',') => self.position += 1,
                        Some(b']') => { self.position += 1; break; }
                        _ => return Err(format!("expected comma or closing bracket at byte {}", self.position)),
                    }
                }
                Ok(JsonValue::Array(values))
            }
            Some(b't') if self.bytes.get(self.position..self.position + 4) == Some(b"true") => {
                self.position += 4; Ok(JsonValue::Bool(true))
            }
            Some(b'f') if self.bytes.get(self.position..self.position + 5) == Some(b"false") => {
                self.position += 5; Ok(JsonValue::Bool(false))
            }
            Some(b'n') if self.bytes.get(self.position..self.position + 4) == Some(b"null") => {
                self.position += 4; Ok(JsonValue::Null)
            }
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(other) => Err(format!("unexpected JSON byte {other:?} at {}", self.position)),
            None => Err("unexpected end of JSON input".into()),
        }
    }
}

fn parse_json(input: &str) -> Result<JsonValue, String> {
    let mut parser = JsonParser::new(input);
    let value = parser.value()?;
    parser.whitespace();
    if parser.position != parser.bytes.len() {
        return Err(format!("trailing bytes after JSON value at {}", parser.position));
    }
    Ok(value)
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
fn optional_u64(value: &str) -> Result<Option<u64>, String> {
    if value == "null" { return Ok(None); }
    value.parse::<u64>().map(Some).map_err(|_| format!("invalid optional integer {value:?}"))
}
fn optional_float(value: &str) -> Result<Option<f64>, String> {
    if value == "null" { return Ok(None); }
    finite_float(value).map(Some)
}

type EpisodeKey = (String, String, String, u64);
type ResultKey = (String, String, String);

fn parse_and_pair(input: &str) -> Result<Vec<(ResultKey, Vec<BTreeMap<Arm, Row>>)>, String> {
    let mut lines = input.lines();
    let header = lines.next().ok_or_else(|| "input is empty".to_string())?;
    if header != HEADER { return Err(format!("header mismatch; required exact header: {HEADER}")); }
    let mut episodes: BTreeMap<EpisodeKey, BTreeMap<Arm, Row>> = BTreeMap::new();
    let mut profiling_receipts = BTreeSet::new();
    for (index, line) in lines.enumerate() {
        let line_number = index + 2;
        if line.is_empty() { return Err(format!("line {line_number}: blank rows are not allowed")); }
        let row = Row::parse(line, line_number)?;
        if !profiling_receipts.insert(row.profiling_receipt_sha256.clone()) {
            return Err(format!("line {line_number}: profiling receipt digest is reused"));
        }
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
    for ((task, capability, split), paired_episodes) in grouped.iter() {
        let reference = &paired_episodes[0][&Arm::D0];
        for (episode_index, episode) in paired_episodes.iter().enumerate() {
            for arm in ARMS {
                let row = &episode[&arm];
                if row.resource_profile_sha256 != reference.resource_profile_sha256
                    || row.resource_measurement_method != reference.resource_measurement_method
                {
                    return Err(format!(
                        "resource profile/method drift within {task}/{capability}/{split} at paired episode {episode_index}, arm {}",
                        arm.name()
                    ));
                }
            }
        }
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
            && row.planner_compute_budget_units == reference.planner_compute_budget_units
            && row.resource_profile_sha256 == reference.resource_profile_sha256
            && row.resource_measurement_method == reference.resource_measurement_method;
        if !matching {
            return Err(format!("paired contract mismatch for {task}/{capability}/{split}/{seed} at arm {}: shared digests and compute/action budgets must match", arm.name()));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ResourceMetrics {
    mean_compute_work_units: Option<f64>,
    p50_compute_work_units: Option<f64>,
    p95_compute_work_units: Option<f64>,
    p95_peak_memory_bytes: Option<f64>,
    mean_energy_millijoules: Option<f64>,
    energy_measured_episode_count: usize,
    mean_direct_value_calls: f64,
    mean_world_model_forward_calls: f64,
    mean_world_model_transition_calls: f64,
    mean_oracle_transition_calls: f64,
    mean_environment_transition_calls: f64,
    mean_optimizer_iterations: f64,
}

#[derive(Clone, Debug)]
struct ResourcePairDelta {
    compute_work_p50_relative: Option<f64>,
    compute_work_p95_relative: Option<f64>,
    latency_p50_relative: Option<f64>,
    latency_p95_relative: Option<f64>,
    memory_p95_relative: Option<f64>,
}

#[derive(Clone, Debug)]
struct ResourceMatch {
    status: &'static str,
    resource_measurement_method: &'static str,
    resource_profile_sha256: String,
    receipt_verification_status: &'static str,
    d1_role_valid: bool,
    learned_model_role_valid: bool,
    w2_workload_match_valid: bool,
    energy_status: &'static str,
    d1_vs_w1: ResourcePairDelta,
    w1_vs_w2: ResourcePairDelta,
    reasons: Vec<&'static str>,
}

#[derive(Clone, Debug)]
struct ArmMetrics {
    mean_return: f64, success_rate: f64, success_count: usize,
    failed_count: usize, timed_out_count: usize,
    mean_latency_ms: f64, p50_latency_ms: f64, p95_latency_ms: f64,
    resources: ResourceMetrics,
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
    eligibility: &'static str, resource_match: ResourceMatch,
    arms: BTreeMap<Arm, ArmMetrics>, contrasts: Vec<Contrast>,
}
#[derive(Clone, Debug)]
struct Report {
    input_sha256: String, bootstrap_replicates: usize, bootstrap_seed: u64,
    groups: Vec<GroupReport>,
}


fn relative_delta(left: f64, right: f64) -> Option<f64> {
    let denominator = left.abs().max(right.abs());
    if !left.is_finite() || !right.is_finite() {
        None
    } else if denominator == 0.0 {
        Some(0.0)
    } else {
        Some((left - right).abs() / denominator)
    }
}

fn paired_relative_deltas(
    rows: &[BTreeMap<Arm, Row>],
    left: Arm,
    right: Arm,
    select: fn(&Row) -> Option<f64>,
) -> Option<Vec<f64>> {
    rows.iter()
        .map(|episode| {
            let left_value = select(&episode[&left])?;
            let right_value = select(&episode[&right])?;
            relative_delta(left_value, right_value)
        })
        .collect()
}

fn relative_percentile(values: &Option<Vec<f64>>, quantile: f64) -> Option<f64> {
    values.as_ref().map(|values| percentile(&mut values.clone(), quantile))
}

fn compare_resource_arms(
    rows: &[BTreeMap<Arm, Row>],
    left: Arm,
    right: Arm,
) -> ResourcePairDelta {
    let compute = paired_relative_deltas(rows, left, right, |row| {
        row.compute_work_units.map(|value| value as f64)
    });
    let latency = paired_relative_deltas(rows, left, right, |row| Some(row.latency_ms));
    let memory = paired_relative_deltas(rows, left, right, |row| {
        row.peak_memory_bytes.map(|value| value as f64)
    });
    ResourcePairDelta {
        compute_work_p50_relative: relative_percentile(&compute, 0.50),
        compute_work_p95_relative: relative_percentile(&compute, 0.95),
        latency_p50_relative: relative_percentile(&latency, 0.50),
        latency_p95_relative: relative_percentile(&latency, 0.95),
        memory_p95_relative: relative_percentile(&memory, 0.95),
    }
}

fn pair_is_within_tolerance(delta: &ResourcePairDelta) -> bool {
    [
        delta.compute_work_p50_relative.map(|value| value <= COMPUTE_WORK_TOLERANCE),
        delta.compute_work_p95_relative.map(|value| value <= COMPUTE_WORK_TOLERANCE),
        delta.latency_p50_relative.map(|value| value <= LATENCY_TOLERANCE),
        delta.latency_p95_relative.map(|value| value <= LATENCY_TOLERANCE),
        delta.memory_p95_relative.map(|value| value <= MEMORY_TOLERANCE),
    ].iter().all(|value| *value == Some(true))
}

fn evaluate_resource_match(episodes: &[BTreeMap<Arm, Row>]) -> ResourceMatch {
    let reference = &episodes[0][&Arm::D0];
    let method = reference.resource_measurement_method;
    let d1_role_valid = episodes.iter().all(|episode| {
        let row = &episode[&Arm::D1];
        row.direct_value_calls > 0
            && row.world_model_forward_calls == 0
            && row.world_model_transition_calls == 0
    });
    let learned_model_role_valid = episodes.iter().all(|episode| {
        let w1 = &episode[&Arm::W1];
        let w2 = &episode[&Arm::W2];
        let w3 = &episode[&Arm::W3];
        w1.world_model_forward_calls > 0
            && w1.world_model_transition_calls > 0
            && w2.world_model_forward_calls > 0
            && w2.world_model_transition_calls > 0
            && w3.oracle_transition_calls > 0
            && w3.world_model_forward_calls == 0
            && w3.world_model_transition_calls == 0
    });
    let w2_workload_match_valid = episodes.iter().all(|episode| {
        let w1 = &episode[&Arm::W1];
        let w2 = &episode[&Arm::W2];
        w1.direct_value_calls == w2.direct_value_calls
            && w1.world_model_forward_calls == w2.world_model_forward_calls
            && w1.world_model_transition_calls == w2.world_model_transition_calls
            && w1.optimizer_iterations == w2.optimizer_iterations
    });

    let mut reasons = Vec::new();
    if !d1_role_valid { reasons.push("D1 must use direct value calls and zero learned-world-model calls"); }
    if !learned_model_role_valid { reasons.push("W1/W2 require learned-model calls and W3 requires evaluator-only oracle transitions"); }
    if !w2_workload_match_valid { reasons.push("W2 must preserve W1 direct/model call counts and optimizer iterations"); }

    let resource_data_complete = method.supports_compute_matching()
        && [Arm::D1, Arm::W1, Arm::W2].iter().all(|arm| {
            episodes.iter().all(|episode| {
                let row = &episode[arm];
                row.compute_work_units.is_some_and(|value| value > 0)
                    && row.peak_memory_bytes.is_some_and(|value| value > 0)
                    && row.latency_ms > 0.0
            })
        });

    let d1_vs_w1 = compare_resource_arms(episodes, Arm::D1, Arm::W1);
    let w1_vs_w2 = compare_resource_arms(episodes, Arm::W1, Arm::W2);
    let energy_complete = [Arm::D1, Arm::W1, Arm::W2].iter().all(|arm| {
        episodes.iter().all(|episode| episode[arm].energy_millijoules.is_some())
    });
    let energy_status = if !energy_complete {
        "unmeasured"
    } else {
        let d1_w1 = paired_relative_deltas(episodes, Arm::D1, Arm::W1, |row| row.energy_millijoules)
            .map(|values| mean(&values))
            .unwrap_or(f64::INFINITY);
        let w1_w2 = paired_relative_deltas(episodes, Arm::W1, Arm::W2, |row| row.energy_millijoules)
            .map(|values| mean(&values))
            .unwrap_or(f64::INFINITY);
        if d1_w1 <= 0.10 && w1_w2 <= 0.10 { "measured_and_within_10_percent" }
        else { "measured_but_outside_10_percent" }
    };

    let status = if !resource_data_complete {
        reasons.push("compute work, memory or latency lacks an accepted measurement");
        "resource_unmeasured"
    } else if !d1_role_valid || !learned_model_role_valid || !w2_workload_match_valid
        || !pair_is_within_tolerance(&d1_vs_w1)
        || !pair_is_within_tolerance(&w1_vs_w2)
        || energy_status == "measured_but_outside_10_percent"
    {
        reasons.push("D1/W1 or W1/W2 differs beyond the preregistered resource tolerance");
        "budget_unmatched"
    } else {
        "budget_matched"
    };

    ResourceMatch {
        status,
        resource_measurement_method: method.name(),
        resource_profile_sha256: reference.resource_profile_sha256.clone(),
        receipt_verification_status: "digest_only_not_independently_verified",
        d1_role_valid,
        learned_model_role_valid,
        w2_workload_match_valid,
        energy_status,
        d1_vs_w1,
        w1_vs_w2,
        reasons,
    }
}

fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let index = ((quantile * values.len() as f64).ceil() as usize).saturating_sub(1).min(values.len().saturating_sub(1));
    values[index]
}
fn complete_resource_values(
    rows: &[BTreeMap<Arm, Row>],
    arm: Arm,
    select: fn(&Row) -> Option<u64>,
) -> Option<Vec<f64>> {
    rows.iter().map(|episode| select(&episode[&arm]).map(|value| value as f64)).collect()
}

fn mean_resource_counter(
    rows: &[BTreeMap<Arm, Row>],
    arm: Arm,
    select: fn(&Row) -> u64,
) -> f64 {
    rows.iter().map(|episode| select(&episode[&arm]) as f64).sum::<f64>() / rows.len() as f64
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

    let compute_values = complete_resource_values(rows, arm, |row| row.compute_work_units);
    let memory_values = complete_resource_values(rows, arm, |row| row.peak_memory_bytes);
    let energy_values: Vec<f64> = rows.iter()
        .filter_map(|episode| episode[&arm].energy_millijoules)
        .collect();
    let resources = ResourceMetrics {
        mean_compute_work_units: compute_values.as_ref().map(|values| mean(values)),
        p50_compute_work_units: compute_values.as_ref().map(|values| percentile(&mut values.clone(), 0.50)),
        p95_compute_work_units: compute_values.as_ref().map(|values| percentile(&mut values.clone(), 0.95)),
        p95_peak_memory_bytes: memory_values.as_ref().map(|values| percentile(&mut values.clone(), 0.95)),
        mean_energy_millijoules: if energy_values.is_empty() { None } else { Some(mean(&energy_values)) },
        energy_measured_episode_count: energy_values.len(),
        mean_direct_value_calls: mean_resource_counter(rows, arm, |row| row.direct_value_calls),
        mean_world_model_forward_calls: mean_resource_counter(rows, arm, |row| row.world_model_forward_calls),
        mean_world_model_transition_calls: mean_resource_counter(rows, arm, |row| row.world_model_transition_calls),
        mean_oracle_transition_calls: mean_resource_counter(rows, arm, |row| row.oracle_transition_calls),
        mean_environment_transition_calls: mean_resource_counter(rows, arm, |row| row.environment_transition_calls),
        mean_optimizer_iterations: mean_resource_counter(rows, arm, |row| row.optimizer_iterations),
    };

    ArmMetrics {
        mean_return, success_rate, success_count, failed_count, timed_out_count,
        mean_latency_ms, p50_latency_ms, p95_latency_ms, resources,
    }
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
        let mut arms = BTreeMap::new();
        for arm in ARMS { arms.insert(arm, summarize_arm(&episodes, arm)); }
        let resource_match = evaluate_resource_match(&episodes);
        let eligibility = if split != "qualification" || episode_count < MIN_QUALIFICATION_PAIRS {
            "smoke_only"
        } else {
            match resource_match.status {
                "budget_matched" => "budget_match_unverified",
                "budget_unmatched" => "budget_unmatched",
                _ => "resource_unmeasured",
            }
        };
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
        groups.push(GroupReport { task, capability, split, episode_count, eligibility, resource_match, arms, contrasts });
    }
    Ok(Report {
        input_sha256: sha256_hex(input.as_bytes()), bootstrap_replicates: replicates,
        bootstrap_seed, groups,
    })
}

fn number(value: f64) -> String {
    if value.is_finite() { format!("{value:.8}") } else { "null".to_string() }
}
fn optional_number(value: Option<f64>) -> String {
    value.map(number).unwrap_or_else(|| "null".to_string())
}
fn render_resource_pair_delta(delta: &ResourcePairDelta) -> String {
    format!(
        "{{\"compute_work_p50_relative\":{},\"compute_work_p95_relative\":{},\"latency_p50_relative\":{},\"latency_p95_relative\":{},\"memory_p95_relative\":{}}}",
        optional_number(delta.compute_work_p50_relative),
        optional_number(delta.compute_work_p95_relative),
        optional_number(delta.latency_p50_relative),
        optional_number(delta.latency_p95_relative),
        optional_number(delta.memory_p95_relative),
    )
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
        let resource = &group.resource_match;
        writeln!(&mut out,
            "      \"resource_match\": {{\"status\": \"{}\", \"resource_measurement_method\": \"{}\", \"resource_profile_sha256\": \"{}\", \"receipt_verification_status\": \"{}\", \"d1_role_valid\": {}, \"learned_model_role_valid\": {}, \"w2_workload_match_valid\": {}, \"energy_status\": \"{}\", \"compute_work_tolerance_fraction\": {}, \"latency_tolerance_fraction\": {}, \"memory_tolerance_fraction\": {}, \"d1_vs_w1\": {}, \"w1_vs_w2\": {}, \"reasons\": [{}]}},",
            resource.status, resource.resource_measurement_method, resource.resource_profile_sha256,
            resource.receipt_verification_status, resource.d1_role_valid, resource.learned_model_role_valid, resource.w2_workload_match_valid,
            resource.energy_status, number(COMPUTE_WORK_TOLERANCE), number(LATENCY_TOLERANCE),
            number(MEMORY_TOLERANCE), render_resource_pair_delta(&resource.d1_vs_w1),
            render_resource_pair_delta(&resource.w1_vs_w2),
            resource.reasons.iter().map(|reason| format!("\"{}\"", reason)).collect::<Vec<_>>().join(",")
        ).unwrap();
        writeln!(&mut out, "      \"arms\": {{").unwrap();
        for (arm_index, arm) in ARMS.iter().enumerate() {
            let m = &group.arms[arm];
            let comma = if arm_index + 1 == ARMS.len() { "" } else { "," };
            writeln!(&mut out,
                concat!("        \"{}\": {{\"mean_return\": {}, \"success_rate\": {}, ",
                "\"success_count\": {}, \"failed_count\": {}, \"timed_out_count\": {}, ",
                "\"mean_latency_ms\": {}, \"p50_latency_ms\": {}, \"p95_latency_ms\": {}, ",
                "\"resource_metrics\": {{\"mean_compute_work_units\": {}, \"p50_compute_work_units\": {}, \"p95_compute_work_units\": {}, ",
                "\"p95_peak_memory_bytes\": {}, \"mean_energy_millijoules\": {}, \"energy_measured_episode_count\": {}, ",
                "\"mean_direct_value_calls\": {}, \"mean_world_model_forward_calls\": {}, \"mean_world_model_transition_calls\": {}, ",
                "\"mean_oracle_transition_calls\": {}, \"mean_environment_transition_calls\": {}, \"mean_optimizer_iterations\": {}}}}{}"),
                arm.name(), number(m.mean_return), number(m.success_rate), m.success_count,
                m.failed_count, m.timed_out_count, number(m.mean_latency_ms),
                number(m.p50_latency_ms), number(m.p95_latency_ms),
                optional_number(m.resources.mean_compute_work_units),
                optional_number(m.resources.p50_compute_work_units),
                optional_number(m.resources.p95_compute_work_units),
                optional_number(m.resources.p95_peak_memory_bytes),
                optional_number(m.resources.mean_energy_millijoules),
                m.resources.energy_measured_episode_count,
                number(m.resources.mean_direct_value_calls),
                number(m.resources.mean_world_model_forward_calls),
                number(m.resources.mean_world_model_transition_calls),
                number(m.resources.mean_oracle_transition_calls),
                number(m.resources.mean_environment_transition_calls),
                number(m.resources.mean_optimizer_iterations), comma
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


const PROFILE_SCHEMA_VERSION: &str = "cmaq-012-resource-profile-v0.1";
const RECEIPT_SCHEMA_VERSION: &str = "cmaq-012-profiling-receipt-v0.1";
const PROFILE_CLAIM_CEILING: &str = "measurement_profile_contract_only_receipt_bytes_must_be_independently_verified";
const RECEIPT_CLAIM_CEILING: &str = "artifact_content_integrity_verified_profiler_and_hardware_origin_unattested";
const ARTIFACT_MAX_BYTES: u64 = 1_048_576;

#[derive(Clone, Debug)]
struct ArtifactVerification {
    status: &'static str,
    profiles_verified: usize,
    receipts_verified: usize,
    fixture_only: bool,
    claim_ceiling: &'static str,
}

fn optional_fixture_only(value: &JsonValue) -> Result<bool, String> {
    match value {
        JsonValue::Object(fields) => match fields.get("fixture_only") {
            None => Ok(false),
            Some(value) => value.as_bool("fixture_only"),
        },
        _ => Err("artifact root must be a JSON object".into()),
    }
}

fn require_nonempty_string(value: &JsonValue, field: &str) -> Result<(), String> {
    let text = value.get(field)?.as_str(field)?;
    if text.trim().is_empty() {
        return Err(format!("JSON field {field:?} must not be empty"));
    }
    Ok(())
}

fn require_digest_field(value: &JsonValue, field: &str) -> Result<(), String> {
    let text = value.get(field)?.as_str(field)?;
    digest(text).map(|_| ()).map_err(|error| format!("JSON field {field:?}: {error}"))
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-9 * (1.0 + left.abs().max(right.abs()))
}

fn artifact_path(root: &Path, subdirectory: &str, digest_value: &str) -> PathBuf {
    root.join(subdirectory).join(format!("{digest_value}.json"))
}

fn read_verified_artifact(
    root: &Path,
    subdirectory: &str,
    expected_digest: &str,
    kind: &str,
) -> Result<Vec<u8>, String> {
    let path = artifact_path(root, subdirectory, expected_digest);
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("missing {kind} artifact {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{kind} artifact {} must be a regular non-symlink file", path.display()));
    }
    if metadata.len() == 0 || metadata.len() > ARTIFACT_MAX_BYTES {
        return Err(format!("{kind} artifact {} must be 1..={ARTIFACT_MAX_BYTES} bytes", path.display()));
    }
    let bytes = fs::read(&path)
        .map_err(|error| format!("failed to read {kind} artifact {}: {error}", path.display()))?;
    let actual = sha256_hex(&bytes);
    if actual != expected_digest {
        return Err(format!(
            "{kind} artifact digest mismatch at {}: expected {expected_digest}, computed {actual}",
            path.display()
        ));
    }
    Ok(bytes)
}

fn validate_profile_artifact(
    bytes: &[u8],
    expected_digest: &str,
    expected_method: ResourceMeasurementMethod,
    allow_fixtures: bool,
) -> Result<bool, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "resource profile is not UTF-8".to_string())?;
    let profile = parse_json(text).map_err(|error| format!("invalid resource profile JSON: {error}"))?;
    if profile.get("schema_version")?.as_str("schema_version")? != PROFILE_SCHEMA_VERSION {
        return Err("resource profile schema_version mismatch".into());
    }
    if profile.get("resource_measurement_method")?.as_str("resource_measurement_method")?
        != expected_method.name()
    {
        return Err("resource profile measurement method does not match TSV rows".into());
    }
    if profile.get("claim_ceiling")?.as_str("claim_ceiling")? != PROFILE_CLAIM_CEILING {
        return Err("resource profile claim_ceiling mismatch".into());
    }
    if digest(expected_digest)? != expected_digest {
        return Err("resource profile digest is not canonical lowercase hexadecimal".into());
    }

    let fixture_only = optional_fixture_only(&profile)?;
    if fixture_only && !allow_fixtures {
        return Err("synthetic fixture resource profile rejected without explicit --allow-fixtures".into());
    }

    let hardware = profile.get("hardware")?;
    for field in ["os_arch", "cpu_model", "device_clock_policy"] {
        require_nonempty_string(hardware, field)?;
    }
    if hardware.get("logical_cpu_count")?.as_u64("logical_cpu_count")? == 0
        || hardware.get("memory_bytes")?.as_u64("memory_bytes")? == 0
    {
        return Err("resource profile hardware counts and memory must be positive".into());
    }
    for field in ["accelerator_model", "accelerator_driver"] {
        let value = hardware.get(field)?;
        if !value.is_null() { value.as_str(field)?; }
    }

    let software = profile.get("software")?;
    require_nonempty_string(software, "source_commit")?;
    require_nonempty_string(software, "compiler")?;
    require_nonempty_string(software, "runtime")?;
    require_digest_field(software, "binary_sha256")?;
    require_digest_field(software, "dependency_lock_sha256")?;

    let instrumentation = profile.get("instrumentation")?;
    for field in ["profiler_name", "profiler_version", "measurement_scope",
                  "synchronization_policy", "instrumentation_overhead_policy"] {
        require_nonempty_string(instrumentation, field)?;
    }
    require_digest_field(instrumentation, "profiler_binary_sha256")?;
    if instrumentation.get("measured_repetitions")?.as_u64("measured_repetitions")? == 0 {
        return Err("resource profile measured_repetitions must be positive".into());
    }

    let compute_definition = profile.get("compute_work_unit_definition")?;
    if compute_definition.get("unit_name")?.as_str("unit_name")? != "normalized_compute_work_unit" {
        return Err("resource profile compute work unit name mismatch".into());
    }
    require_nonempty_string(compute_definition, "counting_semantics")?;
    let accepted = match compute_definition.get("accepted_methods")? {
        JsonValue::Array(values) => values,
        _ => return Err("resource profile accepted_methods must be an array".into()),
    };
    if !accepted.iter().any(|value| value.as_str("accepted_methods[]").ok() == Some(expected_method.name())) {
        return Err("resource profile does not declare the TSV measurement method as accepted".into());
    }

    if profile.get("latency_measurement")?.get("unit")?.as_str("latency_measurement.unit")? != "milliseconds"
        || profile.get("memory_measurement")?.get("unit")?.as_str("memory_measurement.unit")? != "bytes"
        || profile.get("energy_measurement")?.get("unit")?.as_str("energy_measurement.unit")? != "millijoules"
    {
        return Err("resource profile units must be milliseconds, bytes and millijoules".into());
    }
    for (section, field, expected) in [
        ("tolerances", "compute_work_relative", COMPUTE_WORK_TOLERANCE),
        ("tolerances", "latency_relative", LATENCY_TOLERANCE),
        ("tolerances", "peak_memory_relative", MEMORY_TOLERANCE),
    ] {
        let value = profile.get(section)?.get(field)?.as_f64(field)?;
        if !approximately_equal(value, expected) {
            return Err(format!("resource profile tolerance {field} does not match auditor contract"));
        }
    }
    Ok(fixture_only)
}

fn validate_receipt_row(
    receipt: &JsonValue,
    row: &Row,
    profile_digest: &str,
    profile_fixture_only: bool,
    allow_fixtures: bool,
) -> Result<(), String> {
    if receipt.get("schema_version")?.as_str("schema_version")? != RECEIPT_SCHEMA_VERSION {
        return Err("profiling receipt schema_version mismatch".into());
    }
    if receipt.get("claim_ceiling")?.as_str("claim_ceiling")? != RECEIPT_CLAIM_CEILING {
        return Err("profiling receipt claim_ceiling mismatch".into());
    }
    if receipt.get("resource_profile_sha256")?.as_str("resource_profile_sha256")? != profile_digest {
        return Err("profiling receipt resource profile digest does not match TSV row".into());
    }
    if receipt.get("task")?.as_str("task")? != row.task
        || receipt.get("capability")?.as_str("capability")? != row.capability
        || receipt.get("split")?.as_str("split")? != row.split
        || receipt.get("seed")?.as_u64("seed")? != row.seed
        || receipt.get("arm")?.as_str("arm")? != row.arm.name()
    {
        return Err("profiling receipt episode identity does not match TSV row".into());
    }
    if receipt.get("measurement_method")?.as_str("measurement_method")? != row.resource_measurement_method.name() {
        return Err("profiling receipt measurement method does not match TSV row".into());
    }
    let latency = receipt.get("latency_ms")?.as_f64("latency_ms")?;
    if !approximately_equal(latency, row.latency_ms) {
        return Err("profiling receipt latency_ms does not match TSV row".into());
    }
    if receipt.get("compute_work_units")?.optional_u64("compute_work_units")? != row.compute_work_units
        || receipt.get("peak_memory_bytes")?.optional_u64("peak_memory_bytes")? != row.peak_memory_bytes
    {
        return Err("profiling receipt compute or memory does not match TSV row".into());
    }
    let energy = receipt.get("energy_millijoules")?.optional_f64("energy_millijoules")?;
    match (energy, row.energy_millijoules) {
        (Some(a), Some(b)) if approximately_equal(a, b) => {}
        (None, None) => {}
        _ => return Err("profiling receipt energy does not match TSV row".into()),
    }
    for (field, expected) in [
        ("direct_value_calls", row.direct_value_calls),
        ("world_model_forward_calls", row.world_model_forward_calls),
        ("world_model_transition_calls", row.world_model_transition_calls),
        ("oracle_transition_calls", row.oracle_transition_calls),
        ("environment_transition_calls", row.environment_transition_calls),
        ("optimizer_iterations", row.optimizer_iterations),
    ] {
        if receipt.get(field)?.as_u64(field)? != expected {
            return Err(format!("profiling receipt {field} does not match TSV row"));
        }
    }
    require_nonempty_string(receipt, "recorder_id")?;
    if receipt.get("profiling_receipt_sha256").is_ok() {
        return Err("profiling receipt must not contain a self-referential digest field".into());
    }
    let fixture_only = optional_fixture_only(receipt)?;
    if fixture_only != profile_fixture_only {
        return Err("profiling receipt fixture_only flag differs from resource profile".into());
    }
    if fixture_only && !allow_fixtures {
        return Err("synthetic profiling receipt rejected without explicit --allow-fixtures".into());
    }
    Ok(())
}

fn verify_artifacts(
    groups: &[(ResultKey, Vec<BTreeMap<Arm, Row>>)],
    artifacts_dir: &Path,
    allow_fixtures: bool,
) -> Result<ArtifactVerification, String> {
    let mut verified_profiles = BTreeSet::new();
    let mut verified_receipts = BTreeSet::new();
    let mut fixture_mode: Option<bool> = None;

    for ((task, capability, split), episodes) in groups {
        let reference = &episodes[0][&Arm::D0];
        let profile_digest = &reference.resource_profile_sha256;
        let profile_bytes = read_verified_artifact(
            artifacts_dir, "profiles", profile_digest, "resource profile"
        )?;
        let profile_fixture = validate_profile_artifact(
            &profile_bytes, profile_digest, reference.resource_measurement_method, allow_fixtures
        )?;
        if fixture_mode.replace(profile_fixture).is_some_and(|previous| previous != profile_fixture) {
            return Err("artifact set mixes fixture-only and non-fixture profiles".into());
        }
        verified_profiles.insert(profile_digest.clone());

        for episode in episodes {
            for arm in ARMS {
                let row = &episode[&arm];
                if row.resource_profile_sha256 != *profile_digest {
                    return Err(format!("resource profile drift while verifying {task}/{capability}/{split}"));
                }
                let receipt_digest = &row.profiling_receipt_sha256;
                let receipt_bytes = read_verified_artifact(
                    artifacts_dir, "receipts", receipt_digest, "profiling receipt"
                )?;
                let receipt_text = std::str::from_utf8(&receipt_bytes)
                    .map_err(|_| format!("profiling receipt {receipt_digest} is not UTF-8"))?;
                let receipt = parse_json(receipt_text)
                    .map_err(|error| format!("invalid profiling receipt {receipt_digest}: {error}"))?;
                validate_receipt_row(&receipt, row, profile_digest, profile_fixture, allow_fixtures)
                    .map_err(|error| format!("profiling receipt {receipt_digest}: {error}"))?;
                verified_receipts.insert(receipt_digest.clone());
            }
        }
    }

    let fixture_only = fixture_mode.unwrap_or(false);
    let (status, claim_ceiling) = if fixture_only {
        ("synthetic_fixture_content_verified_origin_unattested",
         "synthetic_fixture_content_integrity_verified_no_hardware_claim")
    } else {
        ("content_verified_origin_unattested",
         "artifact_content_integrity_verified_profiler_and_hardware_origin_unattested")
    };
    Ok(ArtifactVerification {
        status,
        profiles_verified: verified_profiles.len(),
        receipts_verified: verified_receipts.len(),
        fixture_only,
        claim_ceiling,
    })
}

fn unverified_artifacts() -> ArtifactVerification {
    ArtifactVerification {
        status: "digest_only_not_independently_verified",
        profiles_verified: 0,
        receipts_verified: 0,
        fixture_only: false,
        claim_ceiling: "tsv_digest_syntax_only_no_artifact_bytes_or_hardware_origin_verified",
    }
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
                    "halfcheetah\tcontrol\tqualification\t{seed}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t64\t3\t100000\tsuccess\t{value:.4}\t{success}\t12.5\t{}\t{}\toperator_trace\t100000\t1048576\tnull\t{}\t{}\t{}\t{}\t3\t3\n",
                    arm.name(), test_digest(10 + seed as u64), test_digest(20),
                    test_digest(30), test_digest(40), test_digest(50),
                    test_digest(100 + index as u64), test_digest(200 + index as u64),
                    test_digest(500), test_digest(600 + seed as u64 * 10 + index as u64),
                    if *arm == Arm::D1 { 100 } else { 0 },
                    if matches!(*arm, Arm::W1 | Arm::W2) { 100 } else { 0 },
                    if matches!(*arm, Arm::W1 | Arm::W2) { 300 } else { 0 },
                    if *arm == Arm::W3 { 300 } else { 0 }
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

    fn alter_all_rows_field(input: &str, field_index: usize, value: &str) -> String {
        let mut lines: Vec<String> = input.lines().map(str::to_string).collect();
        for line in lines.iter_mut().skip(1) {
            let mut fields: Vec<String> = line.split('\t').map(str::to_string).collect();
            fields[field_index] = value.to_string();
            *line = fields.join("\t");
        }
        lines.join("\n") + "\n"
    }

    fn alter_arm_seed_field(input: &str, arm_name: &str, field_index: usize, by_seed: &[&str]) -> String {
        let mut lines: Vec<String> = input.lines().map(str::to_string).collect();
        for line in lines.iter_mut().skip(1) {
            let mut fields: Vec<String> = line.split('\t').map(str::to_string).collect();
            if fields[4] == arm_name {
                let seed = fields[3].parse::<usize>().unwrap();
                fields[field_index] = by_seed[seed].to_string();
                *line = fields.join("\t");
            }
        }
        lines.join("\n") + "\n"
    }

    fn alter_arm_rows_field(input: &str, arm_name: &str, field_index: usize, value: &str) -> String {
        let mut lines: Vec<String> = input.lines().map(str::to_string).collect();
        for line in lines.iter_mut().skip(1) {
            let mut fields: Vec<String> = line.split('\t').map(str::to_string).collect();
            if fields[4] == arm_name {
                fields[field_index] = value.to_string();
                *line = fields.join("\t");
            }
        }
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
    fn measured_resource_vector_and_d1_roles_are_required_for_compute_match() {
        let report = build_report(&synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0),500,11).unwrap();
        assert_eq!(report.groups[0].resource_match.status, "budget_matched");
        assert!(report.groups[0].resource_match.d1_role_valid);
        assert!(report.groups[0].resource_match.learned_model_role_valid);
        assert!(report.groups[0].resource_match.w2_workload_match_valid);
        assert_eq!(report.groups[0].eligibility, "budget_match_unverified");
        assert_eq!(report.groups[0].resource_match.receipt_verification_status, "digest_only_not_independently_verified");
    }

    #[test]
    fn d1_cannot_claim_compute_matching_while_using_learned_world_model_calls() {
        let input = synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0);
        let modified = alter_row_field(&input,2,26,"1");
        let report = build_report(&modified,500,11).unwrap();
        assert_eq!(report.groups[0].resource_match.status, "budget_unmatched");
        assert!(!report.groups[0].resource_match.d1_role_valid);
        assert_eq!(report.groups[0].eligibility, "budget_unmatched");
    }

    #[test]
    fn resource_work_outside_tolerance_is_budget_unmatched() {
        let input = synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0);
        let modified = alter_arm_rows_field(&input,"W1",22,"200000");
        let report = build_report(&modified,500,11).unwrap();
        assert_eq!(report.groups[0].resource_match.status, "budget_unmatched");
        assert_eq!(report.groups[0].eligibility, "budget_unmatched");
    }

    #[test]
    fn paired_cost_mismatch_is_rejected_even_when_marginal_distributions_match() {
        let input = synthetic_tsv(4,1.0,2.0,4.0,2.0,5.0);
        let d1 = ["100000", "200000", "100000", "200000"];
        let wm = ["200000", "100000", "200000", "100000"];
        let input = alter_arm_seed_field(&input, "D1", 22, &d1);
        let input = alter_arm_seed_field(&input, "W1", 22, &wm);
        let input = alter_arm_seed_field(&input, "W2", 22, &wm);
        let report = build_report(&input,500,11).unwrap();
        assert_eq!(report.groups[0].resource_match.status, "budget_unmatched");
        assert_eq!(report.groups[0].resource_match.d1_vs_w1.compute_work_p50_relative, Some(0.5));
        assert_eq!(report.groups[0].resource_match.d1_vs_w1.compute_work_p95_relative, Some(0.5));
    }

    #[test]
    fn w2_must_preserve_w1_predictive_work_and_optimizer_iterations() {
        let input = synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0);
        let modified = alter_row_field(&input,4,27,"999");
        let report = build_report(&modified,500,11).unwrap();
        assert!(!report.groups[0].resource_match.w2_workload_match_valid);
        assert_eq!(report.groups[0].resource_match.status, "budget_unmatched");
    }

    #[test]
    fn static_estimates_are_reported_but_never_qualify_compute_match() {
        let input = synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0);
        let modified = alter_all_rows_field(&input,21,"static_estimate");
        let report = build_report(&modified,500,11).unwrap();
        assert_eq!(report.groups[0].resource_match.status, "resource_unmeasured");
        assert_eq!(report.groups[0].eligibility, "resource_unmeasured");
    }

    #[test]
    fn missing_resource_counters_fail_closed_as_unmeasured() {
        let input = synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0);
        let modified = alter_all_rows_field(&input,21,"unavailable");
        let modified = alter_all_rows_field(&modified,22,"null");
        let modified = alter_all_rows_field(&modified,23,"null");
        let modified = alter_all_rows_field(&modified,24,"null");
        let report = build_report(&modified,500,11).unwrap();
        assert_eq!(report.groups[0].resource_match.status, "resource_unmeasured");
    }

    #[test]
    fn changed_resource_profile_is_rejected_within_a_paired_episode() {
        let input = synthetic_tsv(1,1.0,2.0,4.0,2.0,5.0);
        let modified = alter_row_field(&input,2,19,&test_digest(99999));
        assert!(parse_and_pair(&modified).unwrap_err().contains("paired contract mismatch"));
    }


    #[test]
    fn profiling_receipt_digests_must_be_unique() {
        let input = synthetic_tsv(1,1.0,2.0,4.0,2.0,5.0);
        let modified = alter_row_field(&input,2,20,&test_digest(600));
        assert!(parse_and_pair(&modified).unwrap_err().contains("profiling receipt digest is reused"));
    }

    #[test]
    fn measured_energy_mismatch_is_not_hidden_by_compute_match() {
        let input = synthetic_tsv(20,1.0,2.0,4.0,2.0,5.0);
        let input = alter_arm_rows_field(&input,"D1",24,"10");
        let input = alter_arm_rows_field(&input,"W1",24,"20");
        let input = alter_arm_rows_field(&input,"W2",24,"20");
        let report = build_report(&input,500,11).unwrap();
        assert_eq!(report.groups[0].resource_match.energy_status, "measured_but_outside_10_percent");
        assert_eq!(report.groups[0].resource_match.status, "budget_unmatched");
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
