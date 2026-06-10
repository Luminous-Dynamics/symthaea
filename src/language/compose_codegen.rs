// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Docker Compose code generator (#3 of the "make this even better"
//! list — substrate-independence at the generator level).
//!
//! Mirrors the Nix generator pattern: prompt → intent classification
//! → idiom library → YAML emission. The YAML parser that validates
//! output lives in `compose_scorer.rs`.
//!
//! Scope: minimal-viable idiom library covering the canonical Compose
//! shapes (nginx, postgres, redis, a full web+db stack, monitoring).
//! Enough to exercise the full Scorer → Codegen pipeline on a second
//! substrate, matching `nix_codegen.rs`'s role.

/// Detected Compose intent from prompt. Narrower than NixIntent — Compose
/// deals primarily in service definitions, so the intent space is shallower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComposeIntent {
    /// Single service (nginx, postgres, redis, etc.).
    SingleService,
    /// Multi-service stack (web + db + cache).
    Stack,
    /// Unknown / fallthrough.
    Generic,
}

/// Result of Compose code generation.
#[derive(Debug, Clone)]
pub struct ComposeGenResult {
    pub prompt: String,
    pub intent: ComposeIntent,
    pub code: String,
    pub source: ComposeSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComposeSource {
    /// Matched an idiom in the library.
    Idiom,
    /// Fell through to a skeleton.
    Skeleton,
}

/// Top-level entrypoint: natural-language prompt → Compose YAML.
pub fn generate_compose(prompt: &str) -> ComposeGenResult {
    let lower = prompt.to_lowercase();
    let intent = classify_compose_intent(&lower);
    let code = compose_idiom_body(&lower).unwrap_or_else(|| skeleton_compose());
    let source = if compose_idiom_body(&lower).is_some() {
        ComposeSource::Idiom
    } else {
        ComposeSource::Skeleton
    };
    ComposeGenResult {
        prompt: prompt.to_string(),
        intent,
        code,
        source,
    }
}

fn classify_compose_intent(lower: &str) -> ComposeIntent {
    let service_words = ["nginx", "postgres", "redis", "mongodb", "mysql"];
    let stack_words = ["stack", "web and db", "with database", "full"];
    let has_service = service_words.iter().any(|w| lower.contains(w));
    let has_stack = stack_words.iter().any(|w| lower.contains(w));
    if has_service && has_stack {
        ComposeIntent::Stack
    } else if has_service {
        ComposeIntent::SingleService
    } else {
        ComposeIntent::Generic
    }
}

/// Match prompt to a compose idiom and return the YAML body.
/// Returns None if no idiom matches — caller falls through to skeleton.
pub fn compose_idiom_body(lower: &str) -> Option<String> {
    if lower.contains("nginx") && lower.contains("redis") {
        return Some(emit_nginx_plus_redis_stack());
    }
    if lower.contains("nginx") {
        return Some(emit_nginx_only());
    }
    if lower.contains("postgres") && (lower.contains("app") || lower.contains("web")) {
        return Some(emit_postgres_web_stack());
    }
    if lower.contains("postgres") {
        return Some(emit_postgres_only());
    }
    if lower.contains("redis") {
        return Some(emit_redis_only());
    }
    if lower.contains("prometheus") || lower.contains("monitoring") {
        return Some(emit_prometheus_monitoring());
    }
    None
}

fn emit_nginx_only() -> String {
    r#"services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
"#
    .to_string()
}

fn emit_nginx_plus_redis_stack() -> String {
    r#"services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
  redis:
    image: redis:latest
"#
    .to_string()
}

fn emit_postgres_only() -> String {
    r#"services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
"#
    .to_string()
}

fn emit_postgres_web_stack() -> String {
    r#"services:
  web:
    image: nginx:latest
    ports:
      - "80:80"
  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
"#
    .to_string()
}

fn emit_redis_only() -> String {
    r#"services:
  redis:
    image: redis:latest
"#
    .to_string()
}

fn emit_prometheus_monitoring() -> String {
    r#"services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
"#
    .to_string()
}

fn skeleton_compose() -> String {
    "services: {}\n".to_string()
}

/// Golden-reference Compose snippets for the minimal corpus. Keyed
/// by prompt — returns None if no golden exists yet.
pub fn compose_golden_for(prompt: &str) -> Option<&'static str> {
    match prompt {
        "basic nginx service" => Some(NGINX_GOLDEN),
        "redis cache service" => Some(REDIS_GOLDEN),
        "postgres database" => Some(POSTGRES_GOLDEN),
        "nginx web server with redis cache" => Some(NGINX_REDIS_GOLDEN),
        "prometheus monitoring" => Some(PROMETHEUS_GOLDEN),
        _ => None,
    }
}

pub fn compose_golden_prompts() -> &'static [&'static str] {
    &[
        "basic nginx service",
        "redis cache service",
        "postgres database",
        "nginx web server with redis cache",
        "prometheus monitoring",
    ]
}

const NGINX_GOLDEN: &str = r#"services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
"#;

const REDIS_GOLDEN: &str = r#"services:
  redis:
    image: redis:latest
"#;

const POSTGRES_GOLDEN: &str = r#"services:
  postgres:
    image: postgres:16
"#;

const NGINX_REDIS_GOLDEN: &str = r#"services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
  redis:
    image: redis:latest
"#;

const PROMETHEUS_GOLDEN: &str = r#"services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::compose_scorer::score;

    #[test]
    fn generate_nginx_matches_golden() {
        let result = generate_compose("basic nginx service");
        let golden = compose_golden_for("basic nginx service").unwrap();
        let verdict = score(&result.code, golden);
        assert!(
            verdict.pass(),
            "nginx generation should match golden; got {:?}",
            verdict
        );
    }

    #[test]
    fn generate_redis_matches_golden() {
        let result = generate_compose("redis cache service");
        let golden = compose_golden_for("redis cache service").unwrap();
        let verdict = score(&result.code, golden);
        assert!(verdict.pass(), "redis; got {:?}", verdict);
    }

    #[test]
    fn generate_nginx_redis_stack_matches_golden() {
        let result = generate_compose("nginx web server with redis cache");
        let golden = compose_golden_for("nginx web server with redis cache").unwrap();
        let verdict = score(&result.code, golden);
        assert!(verdict.pass(), "nginx+redis stack; got {:?}", verdict);
    }

    #[test]
    fn prometheus_monitoring_matches_golden() {
        let result = generate_compose("prometheus monitoring");
        let golden = compose_golden_for("prometheus monitoring").unwrap();
        let verdict = score(&result.code, golden);
        assert!(verdict.pass(), "prometheus; got {:?}", verdict);
    }

    #[test]
    fn unknown_prompt_falls_to_skeleton() {
        let result = generate_compose("something we don't know about");
        assert_eq!(result.source, ComposeSource::Skeleton);
        assert!(result.code.contains("services:"));
    }

    #[test]
    fn postgres_with_stack_intent_classification() {
        let intent = classify_compose_intent("postgres database with web app");
        assert_eq!(intent, ComposeIntent::Stack);
    }
}
