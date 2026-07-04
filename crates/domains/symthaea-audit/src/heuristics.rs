// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, non-LLM pre-pass that surfaces a specific class of dark-code
//! signal: a function called with a literal `None` argument in one place and a
//! non-`None` argument at the *same position* somewhere else — the exact shape of
//! this session's highest-value finding (`ConsciousnessEngine::new(finder, None, None,
//! None)` in production vs. `ConsciousnessEngine::new(finder, None, Some(eq), None)` in
//! tests).
//!
//! This is a heuristic, not ground truth: it groups call sites by (callee, arity),
//! which can still coincidentally merge unrelated same-named methods on different
//! types that happen to take the same number of arguments. Its output is explicitly
//! labeled to the model as unverified — the point is to save a weak model from having
//! to *discover* this class of pattern via multi-turn reasoning, not to replace
//! verification.
//!
//! Known limitation, confirmed while dogfooding this exact tool against the repo that
//! now contains its own source: since call-site extraction is plain text/regex-based
//! (no real Rust parser), string literals inside this crate's own doc comments and
//! test fixtures that happen to resemble real call syntax get scanned as if they were
//! live code, adding noise to whichever group they coincidentally match. This doesn't
//! affect audits of other repos — it's specific to this crate scanning itself — and a
//! full fix would require excluding comment/string-literal spans before scanning,
//! which is out of scope for a heuristic whose entire premise is "cheap, not exact."

use std::path::{Path, PathBuf};

use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub struct Arg {
    /// The literal source text of the argument as written at the call site — a bare
    /// identifier like `engine_mmi`, not resolved back to `None`, so hint output stays
    /// legible.
    pub text: String,
    /// True if the argument is the literal token `None`, or a bare identifier that was
    /// most recently assigned `let IDENT = None;` earlier in the same file — the shape
    /// real code actually uses when a constructor takes several `Option<T>` fields
    /// (named locals for readability instead of bare positional `None`s).
    pub is_none: bool,
    /// True when `is_none` came from resolving a local `let` binding rather than a
    /// literal `None` at the call site — surfaced in hint text for transparency.
    pub via_let: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallSite {
    pub callee: String,
    pub file: PathBuf,
    pub line: usize,
    pub args: Vec<Arg>,
    pub is_test_path: bool,
}

fn looks_like_test_path(path: &Path) -> bool {
    path.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        s == "tests" || s == "test"
    }) || path
        .file_stem()
        .map(|s| {
            let s = s.to_string_lossy();
            s == "tests" || s.starts_with("test_") || s.ends_with("_test") || s.ends_with(".test")
        })
        .unwrap_or(false)
}

/// Normalizes a fully/partially qualified call path (`super::foo::Bar::new`,
/// `Bar::new`) down to its last two segments (`Bar::new`) so different import
/// qualifications of the same call still group together.
fn normalize_callee(path: &str) -> String {
    let segments: Vec<&str> = path.split("::").collect();
    if segments.len() <= 2 {
        path.to_string()
    } else {
        segments[segments.len() - 2..].join("::")
    }
}

/// Generic stdlib/common wrapper types whose constructors are called constantly, with
/// completely unrelated arguments, all over any real codebase — grouping call sites by
/// name alone (as [`normalize_callee`] does) turns these into pure noise rather than
/// signal. Excluding them is a precision/recall tradeoff made deliberately in favor of
/// precision: a few real findings on obscure generic types may be missed, but without
/// this filter the top of the hint list is dominated by e.g. `Mutex::new` call sites
/// that have nothing to do with each other, burying anything actually interesting.
const GENERIC_TYPE_DENYLIST: &[&str] = &[
    "Mutex", "RwLock", "Arc", "Rc", "Box", "Cell", "RefCell", "Option", "Vec", "VecDeque",
    "HashMap", "HashSet", "BTreeMap", "BTreeSet", "String", "Default", "Some",
];

fn is_generic_wrapper_callee(callee: &str) -> bool {
    callee
        .split("::")
        .next()
        .map(|ty| GENERIC_TYPE_DENYLIST.contains(&ty))
        .unwrap_or(false)
}

/// Extracts the top-level (depth-0), comma-separated arguments from `text`, skipping
/// commas inside nested brackets and string/char literals.
fn split_top_level_args(text: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    let mut in_string = false;
    let mut in_char = false;

    while let Some(c) = chars.next() {
        if in_string {
            current.push(c);
            if c == '\\' {
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if in_char {
            current.push(c);
            if c == '\\' {
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            } else if c == '\'' {
                in_char = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                current.push(c);
            }
            '\'' => {
                in_char = true;
                current.push(c);
            }
            '(' | '[' | '{' => {
                depth += 1;
                current.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                args.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(c),
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        args.push(trimmed.to_string());
    }
    args
}

/// Finds the byte offset of the argument-list close paren matching the open paren at
/// `open_idx` in `text`, tracking string/char literals so parens inside them don't
/// confuse the depth count. Returns `None` if unbalanced.
fn find_matching_close_paren(text: &str, open_idx: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut i = open_idx;
    let mut in_string = false;
    let mut in_char = false;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if in_string {
            if c == '\\' {
                i += 2;
                continue;
            }
            if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if in_char {
            if c == '\\' {
                i += 2;
                continue;
            }
            if c == '\'' {
                in_char = false;
            }
            i += 1;
            continue;
        }
        match c {
            '"' => in_string = true,
            '\'' => in_char = true,
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn line_number_at(text: &str, byte_offset: usize) -> usize {
    text[..byte_offset.min(text.len())].matches('\n').count() + 1
}

fn is_bare_identifier(s: &str) -> bool {
    let s = s.trim();
    !s.is_empty()
        && s.chars()
            .next()
            .map(|c| c.is_alphabetic() || c == '_')
            .unwrap_or(false)
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Finds every `let [mut] IDENT [: Type] = None;` binding in `content`, returning
/// `(identifier, line_number)` pairs sorted by line. Real code frequently names a
/// `None` field via a local instead of writing the token positionally (e.g.
/// `let engine_mmi = None;` followed by `Foo::new(engine_smf, engine_mmi, ...)`), so
/// resolving these is necessary to catch the pattern this module targets at all.
fn find_none_let_bindings(content: &str) -> Vec<(String, usize)> {
    let let_re =
        Regex::new(r"\blet\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=;]+)?=\s*None\s*[;,)]")
            .unwrap();
    let mut bindings: Vec<(String, usize)> = let_re
        .captures_iter(content)
        .map(|c| {
            let ident = c.get(1).unwrap().as_str().to_string();
            let line = line_number_at(content, c.get(0).unwrap().start());
            (ident, line)
        })
        .collect();
    bindings.sort_by_key(|(_, line)| *line);
    bindings
}

/// Resolves raw argument text into [`Arg`]s: a literal `None` is `is_none = true`
/// directly; a bare identifier is `is_none = true` if the nearest preceding
/// `let IDENT = None;` binding in the same file (by line number) matches it.
fn classify_args(
    raw_args: Vec<String>,
    none_bindings: &[(String, usize)],
    call_line: usize,
) -> Vec<Arg> {
    raw_args
        .into_iter()
        .map(|text| {
            let trimmed = text.trim();
            if trimmed == "None" {
                return Arg {
                    text,
                    is_none: true,
                    via_let: false,
                };
            }
            if is_bare_identifier(trimmed) {
                let resolved = none_bindings
                    .iter()
                    .filter(|(ident, line)| ident == trimmed && *line <= call_line)
                    .next_back();
                if resolved.is_some() {
                    return Arg {
                        text,
                        is_none: true,
                        via_let: true,
                    };
                }
            }
            Arg {
                text,
                is_none: false,
                via_let: false,
            }
        })
        .collect()
}

/// Extracts every `Path::to::callee(` call site in `content`, with its full argument
/// list split at the top level and each argument classified None/non-None.
fn extract_call_sites(content: &str, file: &Path) -> Vec<CallSite> {
    // Matches a `::`-joined path immediately (modulo whitespace, including newlines)
    // followed by an opening paren, e.g. `ConsciousnessEngine::new(` or
    // `super::consciousness_engine::ConsciousnessEngine::new(`.
    let call_re =
        Regex::new(r"((?:[A-Za-z_][A-Za-z0-9_]*::)+[A-Za-z_][A-Za-z0-9_]*)\s*\(").unwrap();
    let is_test_path = looks_like_test_path(file);
    let none_bindings = find_none_let_bindings(content);
    let mut sites = Vec::new();

    for caps in call_re.captures_iter(content) {
        let whole_match = caps.get(0).unwrap();
        let callee = caps.get(1).unwrap().as_str().to_string();
        // The match spans "callee(", possibly with whitespace/newlines before the
        // paren; the paren is always the match's last byte.
        let open_paren_offset = whole_match.end() - 1;

        let Some(close_paren_offset) = find_matching_close_paren(content, open_paren_offset) else {
            continue;
        };
        let arg_text = &content[open_paren_offset + 1..close_paren_offset];
        let raw_args = split_top_level_args(arg_text);
        if raw_args.is_empty() {
            continue;
        }
        let line = line_number_at(content, open_paren_offset);
        let args = classify_args(raw_args, &none_bindings, line);
        sites.push(CallSite {
            callee: normalize_callee(&callee),
            file: file.to_path_buf(),
            line,
            args,
            is_test_path,
        });
    }
    sites
}

struct Mismatch {
    score: i32,
    text: String,
}

/// Compares call sites of the same callee (with matching arity) and reports positions
/// where some sites pass a literal `None` and others don't, prioritizing the
/// production-`None`-vs-test-`Some` shape.
fn find_mismatches(sites: &[CallSite]) -> Vec<Mismatch> {
    use std::collections::HashMap;
    // Group by (callee, arity) rather than callee alone: a single unrelated call site
    // sharing the same normalized name but a different argument count (a genuine
    // overload elsewhere in a large codebase, or — as found while dogfooding this
    // exact tool against a repo that now contains its own test fixtures — a string
    // literal inside this crate's own tests that happens to resemble a shorter call)
    // must not poison an otherwise-valid same-arity cluster.
    let mut groups: HashMap<(&str, usize), Vec<&CallSite>> = HashMap::new();
    for site in sites {
        groups
            .entry((site.callee.as_str(), site.args.len()))
            .or_default()
            .push(site);
    }

    let mut mismatches = Vec::new();
    for ((callee, arity), group) in groups {
        if group.len() < 2 || is_generic_wrapper_callee(callee) {
            continue;
        }
        for pos in 0..arity {
            let none_sites: Vec<&&CallSite> =
                group.iter().filter(|s| s.args[pos].is_none).collect();
            let some_sites: Vec<&&CallSite> =
                group.iter().filter(|s| !s.args[pos].is_none).collect();
            if none_sites.is_empty() || some_sites.is_empty() {
                continue;
            }
            let prod_none = none_sites.iter().any(|s| !s.is_test_path);
            let test_some = some_sites.iter().any(|s| s.is_test_path);
            let score = if prod_none && test_some { 2 } else { 1 };

            let describe = |s: &CallSite| {
                let arg = &s.args[pos];
                let via_let = if arg.via_let {
                    " (= None via local let)"
                } else {
                    ""
                };
                format!(
                    "{}:{} ({}) → arg[{pos}] = {}{via_let}",
                    s.file.display(),
                    s.line,
                    if s.is_test_path { "test" } else { "non-test" },
                    arg.text
                )
            };
            let mut lines = vec![format!(
                "`{callee}` called with argument {pos} sometimes `None`, sometimes not:"
            )];
            for s in none_sites.iter().chain(some_sites.iter()) {
                lines.push(format!("  - {}", describe(s)));
            }
            mismatches.push(Mismatch {
                score,
                text: lines.join("\n"),
            });
        }
    }
    mismatches.sort_by(|a, b| b.score.cmp(&a.score));
    mismatches
}

const MAX_HINTS: usize = 20;
/// A hard backstop against pathological inputs (e.g. a target containing millions of
/// files), not a realistic limit — a real single-language repo the size of this
/// monorepo (~7,200 `.rs` files) must fit comfortably under it. Counts only `.rs`
/// files actually scanned, not every walked directory entry.
const MAX_FILES_SCANNED: usize = 200_000;

/// Scans `.rs` files under `root` (via the same `.gitignore`-aware walk used
/// elsewhere) and returns up to [`MAX_HINTS`] formatted hint strings, highest-signal
/// first. Returns an empty vec if nothing interesting is found — callers should skip
/// adding an empty hints section rather than showing "no hints found" noise.
///
/// If [`MAX_FILES_SCANNED`] is actually hit, this prints a loud warning rather than
/// silently returning partial results — a scan that quietly stopped partway through a
/// large repo is worse than no scan, since it looks complete when it isn't.
pub fn find_none_arg_hints(root: &Path) -> Vec<String> {
    let mut all_sites = Vec::new();
    let mut scanned = 0usize;
    let walker = ignore::WalkBuilder::new(root).build();
    for entry in walker.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        if scanned >= MAX_FILES_SCANNED {
            eprintln!(
                "[symthaea-audit] WARNING: pre-scan hit its {MAX_FILES_SCANNED}-file cap \
                 under {} — hints below are INCOMPLETE, not a full-repo scan",
                root.display()
            );
            break;
        }
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        let rel = path.strip_prefix(root).unwrap_or(path);
        all_sites.extend(extract_call_sites(&content, rel));
        scanned += 1;
    }
    find_mismatches(&all_sites)
        .into_iter()
        .take(MAX_HINTS)
        .map(|m| m.text)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_top_level_args_handles_nested_and_strings() {
        let args = split_top_level_args("a, Some(b, c), \"x, y\", None");
        assert_eq!(args, vec!["a", "Some(b, c)", "\"x, y\"", "None"]);
    }

    #[test]
    fn extract_call_sites_finds_multiline_call() {
        let content = "Foo::new(\n    a,\n    None,\n    None,\n);";
        let sites = extract_call_sites(content, Path::new("src/foo.rs"));
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].callee, "Foo::new");
        let texts: Vec<&str> = sites[0].args.iter().map(|a| a.text.as_str()).collect();
        assert_eq!(texts, vec!["a", "None", "None"]);
        assert!(!sites[0].args[0].is_none);
        assert!(sites[0].args[1].is_none);
        assert!(sites[0].args[2].is_none);
    }

    #[test]
    fn resolves_none_via_local_let_binding() {
        let content = "fn build() {\n    let engine_smf = Finder::new();\n    let engine_mmi = None;\n    let engine_eq = None;\n    ConsciousnessEngine::new(engine_smf, engine_mmi, engine_eq);\n}";
        let sites = extract_call_sites(content, Path::new("src/constructor.rs"));
        assert_eq!(sites.len(), 1);
        let args = &sites[0].args;
        assert!(!args[0].is_none, "engine_smf should not resolve to None");
        assert!(
            args[1].is_none && args[1].via_let,
            "engine_mmi should resolve to None via let"
        );
        assert!(
            args[2].is_none && args[2].via_let,
            "engine_eq should resolve to None via let"
        );
    }

    #[test]
    fn detects_prod_none_via_let_vs_test_some_mismatch() {
        // Reproduces the actual real-world shape this heuristic was built for:
        // production code names its None fields via locals instead of writing the
        // bare token positionally.
        let mut sites = extract_call_sites(
            "fn build() {\n    let engine_mmi = None;\n    let engine_eq = None;\n    ConsciousnessEngine::new(finder, engine_mmi, engine_eq, engine_mmi);\n}",
            Path::new("src/cognitive_loop/constructor.rs"),
        );
        sites.extend(extract_call_sites(
            "ConsciousnessEngine::new(finder, None, Some(eq), None);",
            Path::new("src/cognitive_loop/consciousness_engine/tests.rs"),
        ));
        let mismatches = find_mismatches(&sites);
        assert!(
            !mismatches.is_empty(),
            "expected the let-bound None to be caught"
        );
        assert!(mismatches[0].text.contains("via local let"));
    }

    #[test]
    fn detects_prod_none_vs_test_some_mismatch() {
        let mut sites = extract_call_sites(
            "ConsciousnessEngine::new(finder, None, None, None);",
            Path::new("src/cognitive_loop/constructor.rs"),
        );
        sites.extend(extract_call_sites(
            "ConsciousnessEngine::new(finder, None, Some(eq), None);",
            Path::new("src/cognitive_loop/consciousness_engine/tests.rs"),
        ));
        let mismatches = find_mismatches(&sites);
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0].score, 2);
        assert!(mismatches[0].text.contains("ConsciousnessEngine::new"));
        assert!(mismatches[0].text.contains("constructor.rs"));
        assert!(mismatches[0].text.contains("tests.rs"));
    }

    #[test]
    fn no_mismatch_when_args_always_match() {
        let mut sites = extract_call_sites("Foo::new(a, None);", Path::new("src/a.rs"));
        sites.extend(extract_call_sites(
            "Foo::new(b, None);",
            Path::new("src/b.rs"),
        ));
        assert!(find_mismatches(&sites).is_empty());
    }

    #[test]
    fn different_arity_calls_are_not_compared() {
        let mut sites = extract_call_sites("Foo::new(a, None);", Path::new("src/a.rs"));
        sites.extend(extract_call_sites(
            "Foo::new(a, None, None);",
            Path::new("src/b.rs"),
        ));
        assert!(find_mismatches(&sites).is_empty());
    }

    #[test]
    fn normalize_callee_strips_prefix() {
        assert_eq!(
            normalize_callee("super::consciousness_engine::ConsciousnessEngine::new"),
            "ConsciousnessEngine::new"
        );
        assert_eq!(normalize_callee("Foo::new"), "Foo::new");
    }

    #[test]
    fn find_none_arg_hints_end_to_end_on_temp_dir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src/cognitive_loop/consciousness_engine"))
            .unwrap();
        std::fs::write(
            tmp.path().join("src/cognitive_loop/constructor.rs"),
            "fn build() { ConsciousnessEngine::new(finder, None, None, None); }",
        )
        .unwrap();
        std::fs::write(
            tmp.path()
                .join("src/cognitive_loop/consciousness_engine/tests.rs"),
            "fn t() { ConsciousnessEngine::new(finder, None, Some(eq), None); }",
        )
        .unwrap();
        let hints = find_none_arg_hints(tmp.path());
        assert_eq!(hints.len(), 1);
        assert!(hints[0].contains("ConsciousnessEngine::new"));
    }

    #[test]
    fn looks_like_test_path_detects_tests_dir_and_suffix() {
        assert!(looks_like_test_path(Path::new("crates/foo/tests/bar.rs")));
        assert!(looks_like_test_path(Path::new(
            "src/consciousness_engine/tests.rs"
        )));
        assert!(looks_like_test_path(Path::new("src/foo_test.rs")));
        assert!(!looks_like_test_path(Path::new(
            "src/consciousness_engine/constructor.rs"
        )));
    }

    #[test]
    fn generic_wrapper_types_are_excluded_from_mismatches() {
        let mut sites = extract_call_sites("Mutex::new(None);", Path::new("src/a.rs"));
        sites.extend(extract_call_sites(
            "Mutex::new(some_completely_unrelated_value);",
            Path::new("tests/b.rs"),
        ));
        assert!(
            find_mismatches(&sites).is_empty(),
            "Mutex::new should be filtered as a generic wrapper, not flagged as a real mismatch"
        );
    }

    #[test]
    fn non_generic_type_named_new_is_still_flagged() {
        let mut sites = extract_call_sites(
            "ConsciousnessEngine::new(finder, None);",
            Path::new("src/constructor.rs"),
        );
        sites.extend(extract_call_sites(
            "ConsciousnessEngine::new(finder, Some(eq));",
            Path::new("tests/t.rs"),
        ));
        assert!(!find_mismatches(&sites).is_empty());
    }
}
