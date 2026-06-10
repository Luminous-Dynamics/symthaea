// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use symthaea::action::bindings::{ActionContext, ActionRegistry, PrimitiveExecutor};
use symthaea::action::{ActionIR, ActionOutcome, PolicyBundle, SandboxRoot, SimpleExecutor};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ConsciousnessProfile};
use symthaea::language::{LLMOrgan, LLMOrganConfig, OllamaBackend};
use symthaea::mind::{
    ActivatedConcept, DomainContext, EmotionalTone, EpistemicStatus, ResponseType, SemanticIntent,
    StructuredThought,
};
use symthaea_core::hdc::primitive_system::PrimitiveSystem;
use symthaea_core::hdc::primitive_system::PrimitiveTier;
use symthaea_core::hdc::relational_consciousness::{RelationMode, RelationshipStage};

#[derive(Debug, Deserialize)]
struct AgentInput {
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    files: Option<Vec<Value>>,
    #[serde(default)]
    metadata: Option<Value>,
}

#[derive(Debug, Serialize)]
struct ActionLog {
    action: String,
    status: String,
    detail: Option<String>,
}

fn sanitize_id(raw: &str) -> String {
    raw.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn safe_join(root: &Path, relative: &str) -> PathBuf {
    let mut out = PathBuf::from(root);
    for comp in Path::new(relative).components() {
        if let std::path::Component::Normal(part) = comp {
            out.push(part);
        }
    }
    out
}

#[derive(Debug, Clone)]
struct CognitiveMeta {
    backend: String,
    profile: String,
    cycles: usize,
    genesis: Option<String>,
}

fn parse_profile(raw: Option<String>) -> (ConsciousnessProfile, String) {
    let value = raw.unwrap_or_else(|| "standard".to_string());
    let normalized = value.to_ascii_lowercase();
    match normalized.as_str() {
        "minimal" => (ConsciousnessProfile::Minimal, "minimal".to_string()),
        "full" => (ConsciousnessProfile::Full, "full".to_string()),
        "research" => (ConsciousnessProfile::Research, "research".to_string()),
        _ => (ConsciousnessProfile::Standard, "standard".to_string()),
    }
}

fn build_cognitive_loop() -> Option<(CognitiveLoopService, CognitiveMeta)> {
    let disabled = env::var("SYMTHAEA_AGENT_DISABLE_COGNITIVE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if disabled {
        return None;
    }

    let (profile, profile_label) = parse_profile(env::var("SYMTHAEA_AGENT_PROFILE").ok());
    let backend_raw = env::var("SYMTHAEA_AGENT_TEMPORAL_BACKEND").ok();
    let backend_key = backend_raw.as_deref().unwrap_or("cfc").to_ascii_lowercase();

    let (mut config, backend_label) = match backend_key.as_str() {
        "hdc-ltc-fast" | "hdc_ltc_fast" | "fast" => (
            CognitiveLoopConfig::with_hdc_ltc_fast(),
            "hdc-ltc-fast".to_string(),
        ),
        "hdc-ltc-accurate" | "hdc_ltc_accurate" | "accurate" => (
            CognitiveLoopConfig::with_hdc_ltc_accurate(),
            "hdc-ltc-accurate".to_string(),
        ),
        "hdc-ltc" | "hdc_ltc" | "unified" => (
            CognitiveLoopConfig::with_hdc_ltc_unified(),
            "hdc-ltc".to_string(),
        ),
        _ => (CognitiveLoopConfig::with_cfc(), "cfc".to_string()),
    };

    profile.apply(&mut config);

    let genesis = env::var("SYMTHAEA_AGENT_GENESIS")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    if let Some(ref phrase) = genesis {
        config.genesis_phrase = Some(phrase.clone());
    }

    let cycles = env::var("SYMTHAEA_AGENT_CYCLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    let service = CognitiveLoopService::new(config).ok()?;
    Some((
        service,
        CognitiveMeta {
            backend: backend_label,
            profile: profile_label,
            cycles,
            genesis,
        },
    ))
}

fn build_policy(sandbox: &SandboxRoot) -> PolicyBundle {
    let mut policy = PolicyBundle::restrictive();
    policy.name = "gaia-cyborg".to_string();

    let allowed = [
        "ls", "cat", "echo", "curl", "python3", "jq", "rg", "sed", "awk", "grep",
    ];

    policy.capabilities.shell.allowed_programs = allowed
        .iter()
        .map(|s| s.to_string())
        .collect::<BTreeSet<_>>();

    let root_str = sandbox.root().to_string_lossy().to_string();
    policy.capabilities.filesystem.read_patterns = vec![format!("{}/", root_str)];
    policy.capabilities.filesystem.write_patterns = vec![format!("{}/", root_str)];

    policy.budgets.shell_commands_per_session = env::var("SYMTHAEA_AGENT_SHELL_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    policy.budgets.file_writes_per_session = env::var("SYMTHAEA_AGENT_WRITE_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    policy.budgets.bytes_written_per_session = env::var("SYMTHAEA_AGENT_BYTES_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10 * 1024 * 1024);

    policy
}

fn parse_context_limit() -> usize {
    env::var("SYMTHAEA_AGENT_CONTEXT_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000)
}

fn append_truncated(buf: &mut String, label: &str, content: &str, limit: usize) {
    if buf.len() >= limit {
        return;
    }
    let remaining = limit - buf.len();
    if remaining == 0 {
        return;
    }
    let mut section = String::new();
    section.push_str(label);
    section.push_str(content);
    if section.len() > remaining {
        section.truncate(remaining);
    }
    buf.push_str(&section);
}

fn build_cognitive_input(
    task: &str,
    file_contents: &[(String, String)],
    web_content: Option<&str>,
    limit: usize,
) -> String {
    let mut buf = String::new();
    append_truncated(&mut buf, "TASK:\n", task, limit);

    for (path, content) in file_contents {
        let label = format!("\nFILE {}:\n", path);
        append_truncated(&mut buf, &label, content, limit);
    }

    if let Some(html) = web_content {
        append_truncated(&mut buf, "\nWEB:\n", html, limit);
    }

    if buf.is_empty() {
        task.to_string()
    } else {
        buf
    }
}

fn tiers_from_primitives(primitives: &[String]) -> Vec<String> {
    if primitives.is_empty() {
        return Vec::new();
    }
    let system = PrimitiveSystem::global();
    let mut tiers = BTreeSet::new();
    for prim in primitives {
        if let Some(entry) = system.get(prim) {
            tiers.insert(format!("{:?}", entry.tier));
        }
    }
    tiers.into_iter().collect()
}

fn concepts_from_primitives(primitives: &[String], base: f32) -> Vec<ActivatedConcept> {
    let activation = base.clamp(0.1, 0.95);
    primitives
        .iter()
        .take(6)
        .map(|name| ActivatedConcept {
            name: name.clone(),
            activation,
            relevance: activation,
            #[cfg(feature = "provenance")]
            source: None,
        })
        .collect()
}

fn resolve_epistemic_status(
    computed: bool,
    snapshot: Option<&symthaea::cognitive_loop::ConsciousnessSnapshot>,
) -> EpistemicStatus {
    if !computed {
        return EpistemicStatus::Unknown;
    }
    if let Some(state) = snapshot {
        if state.predictions_trustworthy && state.prediction_confidence >= 0.7 {
            EpistemicStatus::Certain
        } else if state.prediction_confidence >= 0.5 {
            EpistemicStatus::Probable
        } else if state.prediction_confidence >= 0.35 {
            EpistemicStatus::Uncertain
        } else {
            EpistemicStatus::Unknown
        }
    } else {
        EpistemicStatus::Certain
    }
}

fn extract_urls(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|tok| {
            let trimmed = tok
                .trim_matches(|c: char| c == ')' || c == '(' || c == ',' || c == '.' || c == ';');
            if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
                Some(trimmed.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn extract_expression(text: &str) -> Option<String> {
    let mut best: Option<String> = None;
    let mut current = String::new();
    let mut has_op = false;
    let mut has_digit = false;

    let flush = |current: &mut String,
                 has_op: &mut bool,
                 has_digit: &mut bool,
                 best: &mut Option<String>| {
        if *has_op && *has_digit && best.as_ref().map(|s| s.len()).unwrap_or(0) < current.len() {
            *best = Some(current.clone());
        }
        current.clear();
        *has_op = false;
        *has_digit = false;
    };

    for ch in text.chars() {
        if ch.is_ascii_digit() || matches!(ch, '+' | '-' | '*' | '/' | '(' | ')' | '.') {
            current.push(ch);
            if ch.is_ascii_digit() {
                has_digit = true;
            }
            if matches!(ch, '+' | '-' | '*' | '/') {
                has_op = true;
            }
        } else {
            flush(&mut current, &mut has_op, &mut has_digit, &mut best);
        }
    }

    if !current.is_empty() {
        flush(&mut current, &mut has_op, &mut has_digit, &mut best);
    }

    best
}

fn tokenize_keywords(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            if current.len() > 3 {
                out.push(current.clone());
            }
            current.clear();
        }
    }
    if !current.is_empty() && current.len() > 3 {
        out.push(current);
    }
    out.sort();
    out.dedup();
    out
}

fn score_line(line: &str, keywords: &[String]) -> usize {
    let lower = line.to_ascii_lowercase();
    keywords
        .iter()
        .filter(|k| lower.contains(k.as_str()))
        .count()
}

fn best_line_from_text(text: &str, keywords: &[String]) -> Option<String> {
    let mut best: Option<(usize, String)> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let score = score_line(trimmed, keywords);
        if score == 0 {
            continue;
        }
        match &mut best {
            Some((best_score, best_line)) => {
                if score > *best_score {
                    *best_score = score;
                    *best_line = trimmed.to_string();
                }
            }
            None => {
                best = Some((score, trimmed.to_string()));
            }
        }
    }
    best.map(|(_, line)| line)
}

fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    if let Some(start) = lower.find("<title>") {
        if let Some(end) = lower[start + 7..].find("</title>") {
            let title = &html[start + 7..start + 7 + end];
            return Some(title.trim().to_string());
        }
    }
    None
}

fn collect_numbers(text: &str) -> Vec<f64> {
    let mut numbers = Vec::new();
    let mut current = String::new();
    let mut has_digit = false;
    let mut prev: Option<char> = None;

    for ch in text.chars() {
        let is_number_char = ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+';
        if is_number_char {
            if (ch == '-' || ch == '+') && prev.map(|p| p.is_ascii_digit()).unwrap_or(false) {
                // treat sign after digit as separator
                if has_digit {
                    if let Ok(val) = current.parse::<f64>() {
                        numbers.push(val);
                    }
                }
                current.clear();
                has_digit = false;
            }
            current.push(ch);
            if ch.is_ascii_digit() {
                has_digit = true;
            }
        } else {
            if has_digit {
                if let Ok(val) = current.parse::<f64>() {
                    numbers.push(val);
                }
            }
            current.clear();
            has_digit = false;
        }
        prev = Some(ch);
    }
    if has_digit {
        if let Ok(val) = current.parse::<f64>() {
            numbers.push(val);
        }
    }
    numbers
}

fn summarize_numeric(task: &str, numbers: &[f64]) -> Option<String> {
    if numbers.is_empty() {
        return None;
    }
    let lower = task.to_ascii_lowercase();
    if lower.contains("average") || lower.contains("mean") {
        let sum: f64 = numbers.iter().sum();
        return Some(format_number(sum / numbers.len() as f64));
    }
    if lower.contains("sum") || lower.contains("total") {
        let sum: f64 = numbers.iter().sum();
        return Some(format_number(sum));
    }
    if lower.contains("max") || lower.contains("highest") || lower.contains("largest") {
        return numbers
            .iter()
            .cloned()
            .fold(None::<f64>, |acc, v| {
                Some(acc.map(|a| a.max(v)).unwrap_or(v))
            })
            .map(format_number);
    }
    if lower.contains("min") || lower.contains("lowest") || lower.contains("smallest") {
        return numbers
            .iter()
            .cloned()
            .fold(None::<f64>, |acc, v| {
                Some(acc.map(|a| a.min(v)).unwrap_or(v))
            })
            .map(format_number);
    }
    if lower.contains("count") || lower.contains("how many") || lower.contains("number of") {
        return Some(numbers.len().to_string());
    }
    None
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn new(expr: &str) -> Self {
        Self {
            chars: expr.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn parse_expr(&mut self) -> Option<f64> {
        let mut val = self.parse_term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('+') => {
                    self.next();
                    val += self.parse_term()?;
                }
                Some('-') => {
                    self.next();
                    val -= self.parse_term()?;
                }
                _ => break,
            }
        }
        Some(val)
    }

    fn parse_term(&mut self) -> Option<f64> {
        let mut val = self.parse_factor()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('*') => {
                    self.next();
                    val *= self.parse_factor()?;
                }
                Some('/') => {
                    self.next();
                    val /= self.parse_factor()?;
                }
                _ => break,
            }
        }
        Some(val)
    }

    fn parse_factor(&mut self) -> Option<f64> {
        self.skip_ws();
        match self.peek()? {
            '(' => {
                self.next();
                let val = self.parse_expr()?;
                self.skip_ws();
                if self.next()? != ')' {
                    return None;
                }
                Some(val)
            }
            _ => self.parse_number(),
        }
    }

    fn parse_number(&mut self) -> Option<f64> {
        self.skip_ws();
        let mut sign = 1.0;
        if let Some('-') = self.peek() {
            self.next();
            sign = -1.0;
        }

        let mut s = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                s.push(ch);
                self.next();
            } else {
                break;
            }
        }
        if s.is_empty() {
            return None;
        }
        s.parse::<f64>().ok().map(|v| v * sign)
    }
}

fn eval_expression(expr: &str) -> Option<f64> {
    let mut parser = Parser::new(expr);
    let val = parser.parse_expr()?;
    parser.skip_ws();
    if parser.peek().is_some() {
        return None;
    }
    Some(val)
}

fn format_number(val: f64) -> String {
    if (val - val.round()).abs() < 1e-9 {
        format!("{}", val.round() as i64)
    } else {
        let s = format!("{:.6}", val);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn classify_tiers(task: &str, used_files: bool, used_web: bool, used_math: bool) -> Vec<String> {
    let _ = PrimitiveSystem::global();

    let mut tiers = Vec::new();
    if used_math {
        tiers.push(format!("{:?}", PrimitiveTier::Mathematical));
    }
    if used_files {
        tiers.push(format!("{:?}", PrimitiveTier::Code));
    }
    if used_web {
        tiers.push(format!("{:?}", PrimitiveTier::Strategic));
    }
    if task.contains("plan") || task.contains("steps") {
        tiers.push(format!("{:?}", PrimitiveTier::Compositional));
    }
    if tiers.is_empty() {
        tiers.push(format!("{:?}", PrimitiveTier::MetaCognitive));
    }
    tiers
}

fn summarize_action(action: &ActionIR, outcome: &ActionOutcome) -> ActionLog {
    let action_label = match action {
        ActionIR::ReadFile { path, .. } => format!("ReadFile({})", path.display()),
        ActionIR::WriteFile { path, .. } => format!("WriteFile({})", path.display()),
        ActionIR::ListDirectory { path, .. } => format!("ListDirectory({})", path.display()),
        ActionIR::RunCommand { program, args, .. } => {
            format!("RunCommand({} {})", program, args.join(" "))
        }
        ActionIR::CreateDirectory { path, .. } => format!("CreateDirectory({})", path.display()),
        ActionIR::DeleteFile { path } => format!("DeleteFile({})", path.display()),
        ActionIR::ReadSensor { sensor_id, .. } => format!("ReadSensor({})", sensor_id),
        ActionIR::WriteServo { servo_id, value } => {
            format!("WriteServo({}:{:.3})", servo_id, value)
        }
        ActionIR::SwarmGossip { topic, payload } => {
            format!("SwarmGossip({}; {} bytes)", topic, payload.len())
        }
        ActionIR::Sequence(_) => "Sequence".to_string(),
        ActionIR::NoOp => "NoOp".to_string(),
        ActionIR::WasmSandbox {
            module_path,
            function_name,
            ..
        } => {
            format!("WasmSandbox({}::{})", module_path.display(), function_name)
        }
    };

    match outcome {
        ActionOutcome::Success => ActionLog {
            action: action_label,
            status: "success".to_string(),
            detail: None,
        },
        ActionOutcome::FileContent(bytes) => ActionLog {
            action: action_label,
            status: "file_content".to_string(),
            detail: Some(format!("{} bytes", bytes.len())),
        },
        ActionOutcome::DirectoryListing(entries) => ActionLog {
            action: action_label,
            status: "directory_listing".to_string(),
            detail: Some(format!("{} entries", entries.len())),
        },
        ActionOutcome::SensorData { sensor_id, values } => ActionLog {
            action: action_label,
            status: "sensor_data".to_string(),
            detail: Some(format!("{} channels from {}", values.len(), sensor_id)),
        },
        ActionOutcome::ServoStatus {
            servo_id,
            current_value,
        } => ActionLog {
            action: action_label,
            status: "servo_status".to_string(),
            detail: Some(format!("servo {} -> {}", servo_id, current_value)),
        },
        ActionOutcome::SimulatedCommand { program, args } => ActionLog {
            action: action_label,
            status: "simulated".to_string(),
            detail: Some(format!("{} {}", program, args.join(" "))),
        },
        ActionOutcome::CommandOutput {
            stdout,
            stderr,
            exit_code,
        } => ActionLog {
            action: action_label,
            status: format!("exit_{exit_code}"),
            detail: Some(format!(
                "stdout={} bytes stderr={} bytes",
                stdout.len(),
                stderr.len()
            )),
        },
        ActionOutcome::WasmResult { output, logs } => ActionLog {
            action: action_label,
            status: "wasm_result".to_string(),
            detail: Some(format!("{} bytes, {} logs", output.len(), logs.len())),
        },
    }
}

fn find_keyword(text: &str, keywords: &[&str]) -> bool {
    let lower = text.to_lowercase();
    keywords.iter().any(|k| lower.contains(k))
}

#[tokio::main]
#[allow(clippy::field_reassign_with_default)]
async fn main() {
    let mut input = String::new();
    if io::stdin().read_to_string(&mut input).is_err() {
        println!(
            "{}",
            json!({"answer": "I don't know", "status": "stdin_error"})
        );
        return;
    }

    let parsed: AgentInput = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(_) => {
            println!(
                "{}",
                json!({"answer": "I don't know", "status": "invalid_json"})
            );
            return;
        }
    };

    let task_id = parsed
        .task_id
        .clone()
        .unwrap_or_else(|| "gaia-task".to_string());
    let task_text = parsed.task.clone().unwrap_or_default();

    let session_id = sanitize_id(&task_id);
    let sandbox = match SandboxRoot::new(&session_id) {
        Ok(s) => s,
        Err(_) => {
            println!(
                "{}",
                json!({"answer": "I don't know", "status": "sandbox_error"})
            );
            return;
        }
    };
    let policy = build_policy(&sandbox);
    let mut executor = SimpleExecutor::from_env();
    let action_phi = 0.0f64;

    let mut action_logs = Vec::new();
    let mut file_contents: Vec<(String, String)> = Vec::new();
    let mut web_content: Option<String> = None;

    // Prepare files from input
    let mut prepared_files: Vec<PathBuf> = Vec::new();
    if let Some(files) = parsed.files {
        for (idx, entry) in files.into_iter().enumerate() {
            match entry {
                Value::String(path_str) => {
                    let safe_path = safe_join(sandbox.root(), &path_str);
                    prepared_files.push(safe_path);
                }
                Value::Object(map) => {
                    let default_name = format!("attachment-{}.txt", idx);
                    let path_val = map
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&default_name);
                    let content_val = map.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    let safe_path = safe_join(sandbox.root(), path_val);

                    let write_action = ActionIR::WriteFile {
                        path: safe_path.clone(),
                        content: content_val.as_bytes().to_vec(),
                        create_dirs: true,
                    };
                    match executor.execute(&write_action, &policy, &sandbox, action_phi) {
                        Ok(outcome) => {
                            action_logs.push(summarize_action(&write_action, &outcome.outcome))
                        }
                        Err(err) => action_logs.push(ActionLog {
                            action: format!("WriteFile({})", safe_path.display()),
                            status: "error".to_string(),
                            detail: Some(err.to_string()),
                        }),
                    }
                    prepared_files.push(safe_path);
                }
                _ => {}
            }
        }
    }

    // Use primitive registry for basic file/list actions
    let registry = ActionRegistry::standard();
    let primitive_executor = PrimitiveExecutor::new(registry);

    for path in &prepared_files {
        let ctx = ActionContext {
            target_path: Some(path.clone()),
            content: None,
            args: Vec::new(),
            env: HashMap::new(),
        };
        if let Ok(actions) = primitive_executor.translate(&["READ".to_string()], &ctx) {
            for action in actions {
                match executor.execute(&action, &policy, &sandbox, action_phi) {
                    Ok(outcome) => {
                        if let ActionOutcome::FileContent(bytes) = &outcome.outcome {
                            let text = String::from_utf8_lossy(bytes).to_string();
                            file_contents.push((path.display().to_string(), text));
                        }
                        action_logs.push(summarize_action(&action, &outcome.outcome));
                    }
                    Err(err) => action_logs.push(ActionLog {
                        action: format!("ReadFile({})", path.display()),
                        status: "error".to_string(),
                        detail: Some(err.to_string()),
                    }),
                }
            }
        }
    }

    if find_keyword(&task_text, &["list", "directory", "files"]) {
        let ctx = ActionContext {
            target_path: Some(sandbox.root().to_path_buf()),
            content: None,
            args: Vec::new(),
            env: HashMap::new(),
        };
        if let Ok(actions) = primitive_executor.translate(&["LIST".to_string()], &ctx) {
            for action in actions {
                if let Ok(outcome) = executor.execute(&action, &policy, &sandbox, action_phi) {
                    action_logs.push(summarize_action(&action, &outcome.outcome));
                }
            }
        }
    }

    let urls = extract_urls(&task_text);
    let allow_web = env::var("SYMTHAEA_AGENT_ALLOW_WEB")
        .map(|v| v == "1")
        .unwrap_or(false);
    if allow_web {
        if let Some(url) = urls.first() {
            let action = ActionIR::RunCommand {
                program: "curl".to_string(),
                args: vec!["-fsSL".to_string(), url.to_string()],
                env: BTreeMap::new(),
                working_dir: None,
            };
            match executor.execute(&action, &policy, &sandbox, action_phi) {
                Ok(outcome) => {
                    if let ActionOutcome::CommandOutput { stdout, .. } = &outcome.outcome {
                        let text = String::from_utf8_lossy(stdout).to_string();
                        web_content = Some(text);
                    }
                    action_logs.push(summarize_action(&action, &outcome.outcome));
                }
                Err(err) => action_logs.push(ActionLog {
                    action: format!("RunCommand(curl {})", url),
                    status: "error".to_string(),
                    detail: Some(err.to_string()),
                }),
            }
        }
    }

    let mut computed_answer: Option<String> = None;
    let mut used_math = false;

    if let Some(expr) = extract_expression(&task_text) {
        if let Some(value) = eval_expression(&expr) {
            computed_answer = Some(format_number(value));
            used_math = true;
        }
    }

    let has_files = !file_contents.is_empty();
    let has_web = web_content.is_some();

    if computed_answer.is_none() && has_files {
        let lower = task_text.to_lowercase();
        if lower.contains("line count")
            || lower.contains("lines")
            || lower.contains("number of lines")
        {
            if let Some((_, content)) = file_contents.first() {
                let count = content.lines().count();
                computed_answer = Some(count.to_string());
            }
        } else if lower.contains("word count") || lower.contains("words") {
            if let Some((_, content)) = file_contents.first() {
                let count = content.split_whitespace().count();
                computed_answer = Some(count.to_string());
            }
        } else if lower.contains("first line") {
            if let Some((_, content)) = file_contents.first() {
                let first = content.lines().next().unwrap_or("");
                computed_answer = Some(first.to_string());
            }
        } else if lower.contains("character count") || lower.contains("characters") {
            if let Some((_, content)) = file_contents.first() {
                let count = content.chars().count();
                computed_answer = Some(count.to_string());
            }
        }
    }

    if computed_answer.is_none() && has_files {
        let keywords = tokenize_keywords(&task_text);
        for (_, content) in &file_contents {
            if let Some(line) = best_line_from_text(content, &keywords) {
                computed_answer = Some(line);
                break;
            }
        }
    }

    if computed_answer.is_none() && has_files {
        for (_, content) in &file_contents {
            let numbers = collect_numbers(content);
            if let Some(answer) = summarize_numeric(&task_text, &numbers) {
                computed_answer = Some(answer);
                break;
            }
        }
    }

    if computed_answer.is_none() && has_web {
        if let Some(html) = &web_content {
            if find_keyword(&task_text, &["title", "page title"]) {
                if let Some(title) = extract_title(html) {
                    computed_answer = Some(title);
                }
            }
        }
    }

    if computed_answer.is_none() && has_web {
        if let Some(html) = &web_content {
            let keywords = tokenize_keywords(&task_text);
            if let Some(line) = best_line_from_text(html, &keywords) {
                computed_answer = Some(line);
            }
        }
    }

    if computed_answer.is_none() && has_web {
        if let Some(html) = &web_content {
            let numbers = collect_numbers(html);
            if let Some(answer) = summarize_numeric(&task_text, &numbers) {
                computed_answer = Some(answer);
            }
        }
    }

    let mut cognitive_meta: Option<CognitiveMeta> = None;
    let mut cognitive_snapshot: Option<symthaea::cognitive_loop::ConsciousnessSnapshot> = None;
    let mut cognitive_primitives: Vec<String> = Vec::new();
    let mut cognitive_tiers: Vec<String> = Vec::new();
    let mut cognitive_concepts: Vec<ActivatedConcept> = Vec::new();

    if let Some((mut cognitive, meta)) = build_cognitive_loop() {
        let limit = parse_context_limit();
        let input =
            build_cognitive_input(&task_text, &file_contents, web_content.as_deref(), limit);
        let mut last_cycle = None;
        for _ in 0..meta.cycles {
            last_cycle = Some(cognitive.cycle(&input));
        }
        let snapshot = cognitive.consciousness_snapshot();
        let base_activation = snapshot.prediction_confidence;
        if let Some(cycle) = &last_cycle {
            cognitive_primitives = cycle.detected_primitives.clone();
            cognitive_tiers = tiers_from_primitives(&cognitive_primitives);
            if !cognitive_primitives.is_empty() {
                cognitive_concepts =
                    concepts_from_primitives(&cognitive_primitives, base_activation);
            }
        }
        cognitive_snapshot = Some(snapshot);
        cognitive_meta = Some(meta);
    }

    let has_answer = computed_answer.is_some();
    let epistemic = resolve_epistemic_status(has_answer, cognitive_snapshot.as_ref());
    let intent = if has_answer {
        SemanticIntent::Answer
    } else {
        SemanticIntent::ExpressUncertainty
    };
    let mut tiers = if !cognitive_tiers.is_empty() {
        cognitive_tiers
    } else {
        classify_tiers(&task_text, has_files, has_web, used_math)
    };
    if tiers.is_empty() {
        tiers.push(format!("{:?}", PrimitiveTier::MetaCognitive));
    }

    let mut thought = StructuredThought::default();
    thought.semantic_intent = intent;
    thought.response_type = ResponseType::Statement;
    thought.epistemic_status = epistemic;
    thought.relationship_stage = RelationshipStage::Contact;
    thought.relation_mode = RelationMode::IIt;
    thought.activated_concepts = cognitive_concepts;
    if let Some(snapshot) = &cognitive_snapshot {
        let warmth = ((snapshot.unified_valence + 1.0) * 0.5).clamp(0.0, 1.0);
        thought.emotional_tone = EmotionalTone {
            valence: snapshot.unified_valence as f64,
            arousal: snapshot.unified_arousal as f64,
            warmth: warmth as f64,
        };
        thought.psi = snapshot.unified_psi as f64;
        thought.meta_awareness = snapshot.pattern_confidence as f64;
        thought.coherence = snapshot.temporal_coherence as f64;
    } else {
        thought.psi = if epistemic == EpistemicStatus::Certain {
            0.6
        } else {
            0.2
        };
        thought.meta_awareness = if epistemic == EpistemicStatus::Certain {
            0.5
        } else {
            0.2
        };
        thought.coherence = if epistemic == EpistemicStatus::Certain {
            0.6
        } else {
            0.3
        };
    }
    thought.original_input = Some(task_text.clone());
    thought.primitive_tiers = tiers;
    thought.primitives = cognitive_primitives;
    thought.domain_context = Some(DomainContext {
        domain: "gaia".to_string(),
        entities: Vec::new(),
        computed_answer: computed_answer.clone(),
        cube: None,
        psi: cognitive_snapshot.as_ref().map(|s| s.unified_psi as f64),
    });

    let model_id = env::var("SYMTHAEA_LLM_MODEL").unwrap_or_else(|_| "local".to_string());
    let llm_config = LLMOrganConfig {
        model_id,
        ..Default::default()
    };
    let backend = OllamaBackend::new();
    let mut llm = LLMOrgan::with_backend(llm_config, Arc::new(backend));

    let response = llm.translate_thought(&thought, 1.0).await;
    let meta_note = parsed
        .metadata
        .map(|m| format!("metadata={}", m))
        .unwrap_or_else(|| "no_metadata".to_string());
    let mut notes = vec!["cyborg_mind_gaia_agent".to_string()];
    if let Some(meta) = &cognitive_meta {
        notes.push(format!("brain=hdc+{}", meta.backend));
        notes.push(format!("profile={}", meta.profile));
        notes.push(format!("cycles={}", meta.cycles));
        if let Some(genesis) = &meta.genesis {
            notes.push(format!("genesis={}", genesis));
        }
    } else {
        notes.push("brain=heuristic".to_string());
    }
    notes.push(meta_note);
    let notes = notes.join("; ");
    let output = json!({
        "answer": response.text,
        "status": "ok",
        "actions": action_logs,
        "notes": notes
    });

    println!("{}", output);
}
