// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Structural scorer for Docker Compose YAML (Phase 3 of the
//! coding-AI roadmap — substrate-independence starter).
//!
//! Mirrors the architecture of `nix_scorer.rs` but for YAML: parse
//! both generated and golden, flatten each to dotted paths, compare
//! with missing/extraneous/mismatch verdicts.
//!
//! The flattening rule:
//! - scalars at `a.b.c = value` → one entry
//! - sequences at `a.b = [x, y]` → one entry with a canonical
//!   List(Vec<String>) value
//! - nested maps recurse with accumulated path prefix
//!
//! Pass criteria match `nix_scorer::StructuralVerdict::pass()`:
//! every required path present, no value mismatches, extraneous
//! paths warning-only.
//!
//! Why this module proves substrate-independence:
//! Nix's scorer works because `rnix` produces `NODE_ATTRPATH_VALUE`
//! leaves we can flatten. Compose's equivalent in YAML is the
//! key-at-each-level recursive descent. Different parsers, same
//! scoring shape. If repair loops work on this too, the architecture
//! claim is stable across substrates.

use std::collections::BTreeMap;

/// Canonicalized YAML value. Like `nix_scorer::CanonValue` but
/// tailored to YAML's type system (no Nix-specific "PackageList"
/// subset semantics, since Compose lists are typically ports or
/// volumes where extras can matter).
#[derive(Debug, Clone, PartialEq)]
pub enum YamlValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// Preserves order — relevant for `command:` and `entrypoint:`
    /// where array order is semantically meaningful.
    Seq(Vec<YamlValue>),
    /// Used when a value is a map and we've flattened its children;
    /// this is just the shape marker.
    MapMarker,
}

impl YamlValue {
    pub fn display(&self) -> String {
        match self {
            YamlValue::Null => "null".into(),
            YamlValue::Bool(b) => b.to_string(),
            YamlValue::Int(i) => i.to_string(),
            YamlValue::Float(f) => f.to_string(),
            YamlValue::Str(s) => format!("\"{}\"", s),
            YamlValue::Seq(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.display()).collect();
                format!("[{}]", parts.join(", "))
            }
            YamlValue::MapMarker => "<map>".into(),
        }
    }
}

/// Result of comparing two compose YAMLs. Shape-compatible with
/// `nix_scorer::StructuralVerdict` — makes it easy to reuse patterns
/// (pass criteria, summary printing, etc.) across substrates.
#[derive(Debug, Clone, Default)]
pub struct ComposeVerdict {
    pub path_jaccard: f32,
    pub value_mismatches: Vec<ComposeMismatch>,
    pub missing_required: Vec<String>,
    pub extraneous: Vec<String>,
    pub parse_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComposeMismatch {
    pub path: String,
    pub got: YamlValue,
    pub want: YamlValue,
}

impl ComposeVerdict {
    pub fn pass(&self) -> bool {
        self.parse_error.is_none()
            && self.value_mismatches.is_empty()
            && self.missing_required.is_empty()
    }

    pub fn summary(&self) -> String {
        if let Some(err) = &self.parse_error {
            return format!("PARSE-ERR: {}", err);
        }
        if self.pass() {
            "PASS".to_string()
        } else {
            format!(
                "FAIL: jaccard={:.2} mismatches={} missing={}",
                self.path_jaccard,
                self.value_mismatches.len(),
                self.missing_required.len()
            )
        }
    }
}

/// Parse the YAML document and flatten it into a path→value map.
/// Only top-level-mapping documents are accepted (Docker Compose is
/// always a mapping at the root). Other shapes return a parse error.
fn flatten(yaml: &str) -> Result<BTreeMap<String, YamlValue>, String> {
    let doc: serde_yaml::Value =
        serde_yaml::from_str(yaml).map_err(|e| format!("yaml parse: {}", e))?;
    let map = match doc {
        serde_yaml::Value::Mapping(m) => m,
        _ => return Err("compose document must be a mapping at root".into()),
    };
    let mut out = BTreeMap::new();
    for (k, v) in map {
        let key = scalar_to_string(&k)
            .ok_or_else(|| "top-level key must be a scalar string".to_string())?;
        flatten_value(&key, &v, &mut out);
    }
    Ok(out)
}

fn flatten_value(prefix: &str, v: &serde_yaml::Value, out: &mut BTreeMap<String, YamlValue>) {
    match v {
        serde_yaml::Value::Null => {
            out.insert(prefix.to_string(), YamlValue::Null);
        }
        serde_yaml::Value::Bool(b) => {
            out.insert(prefix.to_string(), YamlValue::Bool(*b));
        }
        serde_yaml::Value::Number(n) => {
            let val = if let Some(i) = n.as_i64() {
                YamlValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                YamlValue::Float(f)
            } else {
                YamlValue::Str(n.to_string())
            };
            out.insert(prefix.to_string(), val);
        }
        serde_yaml::Value::String(s) => {
            out.insert(prefix.to_string(), YamlValue::Str(s.clone()));
        }
        serde_yaml::Value::Sequence(seq) => {
            let items: Vec<YamlValue> = seq.iter().map(simple_value).collect();
            out.insert(prefix.to_string(), YamlValue::Seq(items));
        }
        serde_yaml::Value::Mapping(m) => {
            out.insert(prefix.to_string(), YamlValue::MapMarker);
            for (k, v) in m {
                let Some(key) = scalar_to_string(k) else {
                    continue;
                };
                let child_path = format!("{}.{}", prefix, key);
                flatten_value(&child_path, v, out);
            }
        }
        serde_yaml::Value::Tagged(tagged) => {
            // Drop the tag — we don't handle custom YAML tags.
            flatten_value(prefix, &tagged.value, out);
        }
    }
}

/// Best-effort: render any YAML value as a single `YamlValue`. Used
/// for sequence items where further nesting is less common in real
/// compose files. Nested maps inside sequences are flattened to
/// MapMarker; nested sequences keep their structure.
fn simple_value(v: &serde_yaml::Value) -> YamlValue {
    match v {
        serde_yaml::Value::Null => YamlValue::Null,
        serde_yaml::Value::Bool(b) => YamlValue::Bool(*b),
        serde_yaml::Value::Number(n) => n
            .as_i64()
            .map(YamlValue::Int)
            .or_else(|| n.as_f64().map(YamlValue::Float))
            .unwrap_or(YamlValue::Str(n.to_string())),
        serde_yaml::Value::String(s) => YamlValue::Str(s.clone()),
        serde_yaml::Value::Sequence(seq) => YamlValue::Seq(seq.iter().map(simple_value).collect()),
        serde_yaml::Value::Mapping(_) => YamlValue::MapMarker,
        serde_yaml::Value::Tagged(t) => simple_value(&t.value),
    }
}

fn scalar_to_string(v: &serde_yaml::Value) -> Option<String> {
    match v {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        serde_yaml::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Top-level entry. Parses both sides, compares.
pub fn score(generated: &str, golden: &str) -> ComposeVerdict {
    let mut verdict = ComposeVerdict::default();
    let gen_map = match flatten(generated) {
        Ok(m) => m,
        Err(e) => {
            verdict.parse_error = Some(format!("generated: {}", e));
            return verdict;
        }
    };
    let gold_map = match flatten(golden) {
        Ok(m) => m,
        Err(e) => {
            verdict.parse_error = Some(format!("golden: {}", e));
            return verdict;
        }
    };

    let gen_paths: std::collections::BTreeSet<&String> = gen_map.keys().collect();
    let gold_paths: std::collections::BTreeSet<&String> = gold_map.keys().collect();
    let intersection: std::collections::BTreeSet<&&String> =
        gen_paths.intersection(&gold_paths).collect();
    let union: std::collections::BTreeSet<&&String> = gen_paths.union(&gold_paths).collect();

    verdict.path_jaccard = if union.is_empty() {
        1.0
    } else {
        intersection.len() as f32 / union.len() as f32
    };

    for path in &intersection {
        let key: &String = **path;
        let got = &gen_map[key];
        let want = &gold_map[key];
        // MapMarker → MapMarker is "both sides have a sub-map here".
        // Don't flag as value mismatch — the actual content was
        // flattened into child paths.
        if matches!(got, YamlValue::MapMarker) && matches!(want, YamlValue::MapMarker) {
            continue;
        }
        if got != want {
            verdict.value_mismatches.push(ComposeMismatch {
                path: key.clone(),
                got: got.clone(),
                want: want.clone(),
            });
        }
    }
    for path in gold_paths.difference(&gen_paths) {
        verdict.missing_required.push((*path).clone());
    }
    for path in gen_paths.difference(&gold_paths) {
        verdict.extraneous.push((*path).clone());
    }

    verdict
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_compose_files_pass() {
        let yaml = r#"
services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
"#;
        let v = score(yaml, yaml);
        assert!(v.pass(), "identical should pass; got {:?}", v);
    }

    #[test]
    fn missing_service_key_fails() {
        let gen = r#"
services:
  nginx:
    image: nginx:latest
"#;
        let gold = r#"
services:
  nginx:
    image: nginx:latest
  redis:
    image: redis:latest
"#;
        let v = score(gen, gold);
        assert!(!v.pass());
        assert!(v.missing_required.iter().any(|p| p.contains("redis")));
    }

    #[test]
    fn different_image_version_fails() {
        let gen = r#"
services:
  nginx:
    image: nginx:1.25
"#;
        let gold = r#"
services:
  nginx:
    image: nginx:latest
"#;
        let v = score(gen, gold);
        assert!(!v.pass(), "different image tags must be caught");
        assert_eq!(v.value_mismatches.len(), 1);
        assert_eq!(v.value_mismatches[0].path, "services.nginx.image");
    }

    #[test]
    fn extra_service_is_warning_not_fail() {
        let gen = r#"
services:
  nginx:
    image: nginx:latest
  redis:
    image: redis:latest
"#;
        let gold = r#"
services:
  nginx:
    image: nginx:latest
"#;
        let v = score(gen, gold);
        assert!(
            v.pass(),
            "extra service should be warning only; got {:?}",
            v
        );
        assert!(!v.extraneous.is_empty());
    }

    #[test]
    fn integer_port_values_match() {
        let gen = r#"
services:
  web:
    image: nginx
    ports:
      - 8080
"#;
        let gold = r#"
services:
  web:
    image: nginx
    ports:
      - 8080
"#;
        let v = score(gen, gold);
        assert!(v.pass());
    }

    #[test]
    fn ports_list_mismatch_fails() {
        let gen = r#"
services:
  web:
    image: nginx
    ports:
      - "80:80"
"#;
        let gold = r#"
services:
  web:
    image: nginx
    ports:
      - "8080:80"
"#;
        let v = score(gen, gold);
        assert!(!v.pass(), "different port mapping must be flagged");
    }

    #[test]
    fn invalid_yaml_reports_parse_error() {
        let gen = r#"services:
  nginx:
    image: nginx:latest
    invalid: : : :
"#;
        let gold = r#"services:
  nginx:
    image: nginx:latest
"#;
        let v = score(gen, gold);
        assert!(!v.pass());
        assert!(v.parse_error.is_some());
    }

    #[test]
    fn version_2_and_version_3_differ() {
        // Compose spec versions matter semantically — scorer should
        // flag them as mismatched top-level values.
        let gen = r#"
version: "2"
services:
  nginx:
    image: nginx
"#;
        let gold = r#"
version: "3.8"
services:
  nginx:
    image: nginx
"#;
        let v = score(gen, gold);
        assert!(!v.pass());
        assert!(v.value_mismatches.iter().any(|m| m.path == "version"));
    }
}
