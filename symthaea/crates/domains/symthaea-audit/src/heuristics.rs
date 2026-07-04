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
//! Also includes a second, independent detector: `pub fn`s defined outside test code
//! that are never referenced anywhere except (at most) in tests — a direct, cheap
//! signal for the report schema's CLAIMED BUT DARK / SHOULD DELETE categories, with no
//! argument analysis needed at all.
//!
//! Both detectors first strip comments and string literals from each file (replacing
//! them with equal-length whitespace so line numbers stay correct) before scanning —
//! without this, example call syntax written in a doc comment or a test's string
//! fixture gets scanned as if it were live code. This was originally found by
//! dogfooding the tool against a repo that happens to contain this crate's own source,
//! but the underlying problem is general: any target repo can have code-like text in
//! comments or string literals, and it isn't sound to skip fixing it just because it
//! was noticed here first.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use regex::Regex;

/// Replaces `//` line comments, `/* */` block comments (nesting-aware), regular
/// string literals, and raw string literals with whitespace of the same byte length —
/// newlines are preserved so line numbers computed against the result still match the
/// original file. Char literals are deliberately left untouched: distinguishing them
/// from lifetimes (`'a`) without a real tokenizer is unreliable, and unlike comments/
/// strings they rarely contain `(`/`)`/`::` sequences that would create false matches.
fn strip_comments_and_strings(content: &str) -> String {
    let bytes = content.as_bytes();
    let mut out = String::with_capacity(content.len());
    let mut i = 0;
    let blank = |c: u8| if c == b'\n' { '\n' } else { ' ' };

    while i < bytes.len() {
        let c = bytes[i];
        // Line comment: everything to end of line.
        if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        // Block comment: nesting-aware, since Rust allows /* /* */ */.
        if c == b'/' && bytes.get(i + 1) == Some(&b'*') {
            let mut depth = 1;
            out.push(' ');
            out.push(' ');
            i += 2;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                } else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                } else {
                    out.push(blank(bytes[i]));
                    i += 1;
                }
            }
            continue;
        }
        // Raw string: r"...", r#"..."#, r##"..."##, ... (optionally byte-prefixed,
        // e.g. br#"..."#, which this also handles since the leading b/r pass through
        // unchanged and the `r` immediately preceding the hashes/quote is what matters).
        if c == b'r' && is_raw_string_start(bytes, i) {
            let hashes = count_hashes_after_r(bytes, i);
            let quote_idx = i + 1 + hashes;
            out.push(' '); // 'r'
            for _ in 0..hashes {
                out.push(' ');
            }
            out.push(' '); // opening quote
            i = quote_idx + 1;
            let closer: Vec<u8> = std::iter::once(b'"')
                .chain(std::iter::repeat(b'#').take(hashes))
                .collect();
            while i < bytes.len() {
                if bytes[i..].starts_with(&closer[..]) {
                    for _ in 0..closer.len() {
                        out.push(' ');
                    }
                    i += closer.len();
                    break;
                }
                out.push(blank(bytes[i]));
                i += 1;
            }
            continue;
        }
        // Regular string literal, with backslash-escape handling.
        if c == b'"' {
            out.push(' ');
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    out.push(' ');
                    out.push(blank(bytes[i + 1]));
                    i += 2;
                    continue;
                }
                if bytes[i] == b'"' {
                    out.push(' ');
                    i += 1;
                    break;
                }
                out.push(blank(bytes[i]));
                i += 1;
            }
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// True if `bytes[i]` is an `r` that starts a raw string (`r"`, `r#"`, `r##"`, ...),
/// i.e. it's a standalone token (not part of a longer identifier) followed by zero or
/// more `#` then a `"`.
fn is_raw_string_start(bytes: &[u8], i: usize) -> bool {
    let prev_is_ident = i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
    if prev_is_ident {
        return false;
    }
    let hashes = count_hashes_after_r(bytes, i);
    bytes.get(i + 1 + hashes) == Some(&b'"')
}

fn count_hashes_after_r(bytes: &[u8], i: usize) -> usize {
    let mut n = 0;
    while bytes.get(i + 1 + n) == Some(&b'#') {
        n += 1;
    }
    n
}

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

/// Vendored dependencies, external benchmark/exercise corpora, and example binaries
/// where "only called from its own test" (or "only called from `main()`") is the
/// *expected*, correct shape rather than a real finding — e.g. an Exercism practice
/// exercise's implementation is supposed to only be exercised by its own test file.
/// Only excluded from the dead-`pub fn` detector: the None-arg detector still benefits
/// from scanning these paths (a real, useful finding turned up in `patches/iroh/`
/// during verification), it's specifically "is this pub fn ever used outside its own
/// tests" that's structurally the wrong question to ask about this kind of code.
fn looks_like_non_primary_path(path: &Path) -> bool {
    path.components().any(|c| {
        matches!(
            c.as_os_str().to_string_lossy().as_ref(),
            "vendor" | "benchmarks" | "examples" | "patches"
        )
    })
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
/// list split at the top level and each argument classified None/non-None. Compiles
/// its own regex — only used by tests; [`extract_call_sites_from`] is what
/// [`scan_repo`] uses when scanning many files so the regex isn't rebuilt each time.
#[cfg(test)]
fn extract_call_sites(content: &str, file: &Path) -> Vec<CallSite> {
    extract_call_sites_from(content, file, &call_site_regex())
}

fn extract_call_sites_from(content: &str, file: &Path, call_re: &Regex) -> Vec<CallSite> {
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
            // Cap how many example sites are shown per side — a common helper called
            // dozens of times with only one genuinely-differing call site is still
            // worth flagging, but printing all 40+ ordinary sites would drown out
            // every other hint's share of the model's context budget for no benefit
            // over showing a representative few plus a count of the rest.
            for (label, sites) in [("None", &none_sites), ("non-None", &some_sites)] {
                for s in sites.iter().take(MAX_EXAMPLE_SITES_PER_SIDE) {
                    lines.push(format!("  - {}", describe(s)));
                }
                if sites.len() > MAX_EXAMPLE_SITES_PER_SIDE {
                    lines.push(format!(
                        "  - ... and {} more {label} site(s)",
                        sites.len() - MAX_EXAMPLE_SITES_PER_SIDE
                    ));
                }
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
/// Per side (None-sites, non-None-sites) within a single mismatch hint. See the
/// comment at its use site for why this exists.
const MAX_EXAMPLE_SITES_PER_SIDE: usize = 4;
/// A hard backstop against pathological inputs (e.g. a target containing millions of
/// files), not a realistic limit — a real single-language repo the size of this
/// monorepo (~7,200 `.rs` files) must fit comfortably under it. Counts only `.rs`
/// files actually scanned, not every walked directory entry.
const MAX_FILES_SCANNED: usize = 200_000;

#[derive(Debug, Clone)]
pub struct FnDef {
    pub name: String,
    pub file: PathBuf,
    pub line: usize,
}

/// Everything both detectors need, gathered in a single walk over the target so
/// auditing a large repo doesn't mean scanning every file twice.
pub struct RepoScan {
    call_sites: Vec<CallSite>,
    /// `pub fn`/`pub(...) fn` definitions found outside test paths.
    pub_fn_defs: Vec<FnDef>,
    /// How many times each identifier token appears anywhere in non-test files,
    /// including at its own `fn NAME` definition (callers subtract known definition
    /// counts to test for "no usage beyond the definition itself").
    identifier_counts_non_test: HashMap<String, usize>,
    identifier_counts_test: HashMap<String, usize>,
}

/// Matches a `::`-joined path immediately (modulo whitespace, including newlines)
/// followed by an opening paren, e.g. `ConsciousnessEngine::new(` or
/// `super::consciousness_engine::ConsciousnessEngine::new(`.
fn call_site_regex() -> Regex {
    Regex::new(r"((?:[A-Za-z_][A-Za-z0-9_]*::)+[A-Za-z_][A-Za-z0-9_]*)\s*\(").unwrap()
}

/// Matches `fn NAME`, optionally preceded by a visibility modifier on the same
/// statement (`pub fn NAME`, `pub(crate) fn NAME`). Deliberately does not require
/// anything about what follows `NAME` (generics, then `(`), so generic function
/// definitions are still captured correctly.
fn fn_def_regex() -> Regex {
    Regex::new(r"(pub(?:\([^)]*\))?\s+)?\bfn\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap()
}

/// Every identifier-shaped token in the file — used for usage counting, deliberately
/// *not* requiring a following `(`, since real usage includes `.method()` dot calls,
/// `Type::method()` UFCS calls, bare `function()` calls, and non-call references
/// (function pointers, trait bounds) alike. Errs toward under-reporting (missing real
/// dead code because some other, unrelated item shares its bare name somewhere) rather
/// than over-reporting (flagging something as dead that's actually used) — the safer
/// direction for a hint that's presented as unverified.
fn identifier_token_regex() -> Regex {
    Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]*\b").unwrap()
}

/// Method names dispatched implicitly by the language (operator sugar, `for`/`.await`
/// desugaring, formatting macros) whose literal name frequently never appears as a
/// bare token anywhere in the source, even though the method is very much used —
/// flagging these as "unused" would be a reliable false positive, so they're excluded
/// from the dead-`pub fn` detector entirely.
const IMPLICIT_DISPATCH_METHOD_DENYLIST: &[&str] = &[
    "fmt",
    "eq",
    "ne",
    "cmp",
    "partial_cmp",
    "hash",
    "index",
    "index_mut",
    "deref",
    "deref_mut",
    "drop",
    "next",
    "poll",
    "add",
    "sub",
    "mul",
    "div",
    "rem",
    "neg",
    "not",
    "bitand",
    "bitor",
    "bitxor",
    "shl",
    "shr",
];

/// Scans `.rs` files under `root` once, gathering everything [`none_arg_mismatch_hints`]
/// and [`dead_pub_fn_hints`] need. See [`MAX_FILES_SCANNED`] for the truncation policy.
pub fn scan_repo(root: &Path) -> RepoScan {
    let call_re = call_site_regex();
    let fn_re = fn_def_regex();
    let ident_re = identifier_token_regex();

    let mut call_sites = Vec::new();
    let mut pub_fn_defs = Vec::new();
    let mut identifier_counts_non_test: HashMap<String, usize> = HashMap::new();
    let mut identifier_counts_test: HashMap<String, usize> = HashMap::new();
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
        let Ok(raw_content) = std::fs::read_to_string(path) else {
            continue;
        };
        let content = strip_comments_and_strings(&raw_content);
        let rel = path.strip_prefix(root).unwrap_or(path).to_path_buf();
        let is_test = looks_like_test_path(&rel);
        let is_primary = !looks_like_non_primary_path(&rel);

        call_sites.extend(extract_call_sites_from(&content, &rel, &call_re));

        for caps in fn_re.captures_iter(&content) {
            if caps.get(1).is_some() && !is_test && is_primary {
                let name = caps.get(2).unwrap().as_str().to_string();
                let line = line_number_at(&content, caps.get(0).unwrap().start());
                pub_fn_defs.push(FnDef {
                    name,
                    file: rel.clone(),
                    line,
                });
            }
        }

        let counts = if is_test {
            &mut identifier_counts_test
        } else {
            &mut identifier_counts_non_test
        };
        for m in ident_re.find_iter(&content) {
            *counts.entry(m.as_str().to_string()).or_insert(0) += 1;
        }
        scanned += 1;
    }

    RepoScan {
        call_sites,
        pub_fn_defs,
        identifier_counts_non_test,
        identifier_counts_test,
    }
}

/// Returns up to [`MAX_HINTS`] None-vs-non-None argument mismatch hints, highest-signal
/// first. Empty when nothing interesting was found.
pub fn none_arg_mismatch_hints(scan: &RepoScan) -> Vec<String> {
    find_mismatches(&scan.call_sites)
        .into_iter()
        .take(MAX_HINTS)
        .map(|m| m.text)
        .collect()
}

/// Returns up to [`MAX_HINTS`] `pub fn`s defined outside test code that appear to have
/// no non-test usage anywhere in the scanned repo, highest-signal first (test-only
/// usage — the deceptive "looks alive because of its own tests" shape — ranks above
/// apparently-unused-anywhere, since the latter is more likely a genuine untracked
/// utility that's simply rarely called, and the former is a stronger CLAIMED-BUT-DARK
/// signal). Empty when nothing was found.
pub fn dead_pub_fn_hints(scan: &RepoScan) -> Vec<String> {
    let mut def_counts_by_name: HashMap<&str, usize> = HashMap::new();
    for def in &scan.pub_fn_defs {
        *def_counts_by_name.entry(def.name.as_str()).or_insert(0) += 1;
    }

    struct Candidate {
        score: i32,
        text: String,
    }
    let mut candidates = Vec::new();
    let mut reported: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for def in &scan.pub_fn_defs {
        if IMPLICIT_DISPATCH_METHOD_DENYLIST.contains(&def.name.as_str())
            || !reported.insert(&def.name)
        {
            continue;
        }
        let own_defs = *def_counts_by_name.get(def.name.as_str()).unwrap_or(&0);
        let non_test_count = *scan.identifier_counts_non_test.get(&def.name).unwrap_or(&0);
        let test_count = *scan.identifier_counts_test.get(&def.name).unwrap_or(&0);
        // Each definition's own `fn NAME` text also matches the identifier-token
        // regex, so subtract known definitions before deciding "no other usage".
        if non_test_count > own_defs {
            continue;
        }
        let sites: Vec<&FnDef> = scan
            .pub_fn_defs
            .iter()
            .filter(|d| d.name == def.name)
            .collect();
        let where_defined = sites
            .iter()
            .map(|d| format!("{}:{}", d.file.display(), d.line))
            .collect::<Vec<_>>()
            .join(", ");
        if test_count > 0 {
            candidates.push(Candidate {
                score: 2,
                text: format!(
                    "`{}` is defined at {where_defined} but only ever referenced from test code — \
                     it looks alive because tests exercise it, but nothing in production calls it.",
                    def.name
                ),
            });
        } else {
            candidates.push(Candidate {
                score: 1,
                text: format!(
                    "`{}` is defined at {where_defined} but was not found referenced anywhere else in the scan.",
                    def.name
                ),
            });
        }
    }

    candidates.sort_by(|a, b| b.score.cmp(&a.score));
    candidates
        .into_iter()
        .take(MAX_HINTS)
        .map(|c| c.text)
        .collect()
}

/// Convenience wrapper for callers that only need the None-arg detector — runs its own
/// scan. Prefer [`scan_repo`] directly when also calling [`dead_pub_fn_hints`], so the
/// repo is only walked once.
pub fn find_none_arg_hints(root: &Path) -> Vec<String> {
    none_arg_mismatch_hints(&scan_repo(root))
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
    fn looks_like_non_primary_path_detects_vendor_benchmark_example_patch_dirs() {
        assert!(looks_like_non_primary_path(Path::new(
            "vendor/binius64/src/lib.rs"
        )));
        assert!(looks_like_non_primary_path(Path::new(
            "benchmarks/external/exercism-rust/exercises/foo/src/lib.rs"
        )));
        assert!(looks_like_non_primary_path(Path::new(
            "crates/foo/examples/demo.rs"
        )));
        assert!(looks_like_non_primary_path(Path::new(
            "patches/iroh/iroh/src/lib.rs"
        )));
        assert!(!looks_like_non_primary_path(Path::new(
            "src/cognitive_loop/constructor.rs"
        )));
    }

    #[test]
    fn large_groups_are_capped_per_side_with_a_remainder_note() {
        let mut sites = Vec::new();
        for i in 0..10 {
            sites.extend(extract_call_sites(
                &format!("Widget::build({i});"),
                Path::new("src/a.rs"),
            ));
        }
        sites.extend(extract_call_sites(
            "Widget::build(None);",
            Path::new("tests/b.rs"),
        ));
        let mismatches = find_mismatches(&sites);
        assert_eq!(mismatches.len(), 1);
        let shown = mismatches[0].text.matches(" → arg[0] = ").count();
        assert_eq!(
            shown,
            MAX_EXAMPLE_SITES_PER_SIDE + 1,
            "capped non-None side + the single None site"
        );
        assert!(mismatches[0].text.contains("more"));
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

    // -----------------------------------------------------------------------
    // strip_comments_and_strings
    // -----------------------------------------------------------------------

    #[test]
    fn strips_line_comments_preserving_newlines() {
        let input = "let a = 1; // Foo::new(x, None)\nlet b = 2;";
        let stripped = strip_comments_and_strings(input);
        assert!(!stripped.contains("Foo::new"));
        assert_eq!(stripped.lines().count(), input.lines().count());
    }

    #[test]
    fn strips_nested_block_comments() {
        let input = "/* outer /* inner Foo::new(x) */ still comment */ real_code();";
        let stripped = strip_comments_and_strings(input);
        assert!(!stripped.contains("Foo::new"));
        assert!(stripped.contains("real_code();"));
    }

    #[test]
    fn strips_doc_comments() {
        let input =
            "//! Example: `ConsciousnessEngine::new(finder, None, None, None)`\nfn real() {}";
        let stripped = strip_comments_and_strings(input);
        assert!(!stripped.contains("ConsciousnessEngine::new"));
        assert!(stripped.contains("fn real"));
    }

    #[test]
    fn strips_string_literals_with_escapes() {
        let input = r#"let s = "Foo::new(finder, None); \" still string"; real();"#;
        let stripped = strip_comments_and_strings(input);
        assert!(!stripped.contains("Foo::new"));
        assert!(stripped.contains("real();"));
    }

    #[test]
    fn strips_raw_strings_with_hashes() {
        let input = "let s = r#\"Foo::new(finder, None)\"#; real();";
        let stripped = strip_comments_and_strings(input);
        assert!(!stripped.contains("Foo::new"));
        assert!(stripped.contains("real();"));
    }

    #[test]
    fn does_not_touch_real_code_outside_comments_and_strings() {
        let input = "ConsciousnessEngine::new(finder, None, Some(eq), None);";
        let stripped = strip_comments_and_strings(input);
        assert_eq!(stripped, input);
    }

    #[test]
    fn line_numbers_survive_stripping() {
        let input = "a();\n// Foo::new(x, None)\nb();\n\"a string\nspanning lines\";\nc();";
        let stripped = strip_comments_and_strings(input);
        assert_eq!(stripped.lines().count(), input.lines().count());
        // `c();` should still be on the same line number in the stripped text.
        let original_line = input.lines().position(|l| l.contains("c();")).unwrap();
        let stripped_line = stripped.lines().position(|l| l.contains("c();")).unwrap();
        assert_eq!(original_line, stripped_line);
    }

    #[test]
    fn own_doc_comment_no_longer_pollutes_a_real_scan() {
        // Regression test for the exact bug found dogfooding this tool: this crate's
        // own module doc comment contains example call syntax that used to get
        // scanned as if it were live code.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(
            tmp.path().join("src/lib.rs"),
            "//! Example: `ConsciousnessEngine::new(finder, None, None, None)` vs\n\
             //! `ConsciousnessEngine::new(finder, None, Some(eq), None)` in tests.\n",
        )
        .unwrap();
        let hints = find_none_arg_hints(tmp.path());
        assert!(
            hints.is_empty(),
            "doc-comment example text should not produce a hint: {hints:?}"
        );
    }

    // -----------------------------------------------------------------------
    // dead_pub_fn_hints
    // -----------------------------------------------------------------------

    #[test]
    fn flags_pub_fn_used_only_in_tests() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::create_dir_all(tmp.path().join("tests")).unwrap();
        std::fs::write(
            tmp.path().join("src/lib.rs"),
            "pub fn orphaned_feature(x: u32) -> u32 { x + 1 }\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("tests/it.rs"),
            "fn t() { assert_eq!(orphaned_feature(1), 2); }\n",
        )
        .unwrap();
        let scan = scan_repo(tmp.path());
        let hints = dead_pub_fn_hints(&scan);
        assert_eq!(hints.len(), 1);
        assert!(hints[0].contains("orphaned_feature"));
        assert!(hints[0].contains("only ever referenced from test code"));
    }

    #[test]
    fn does_not_flag_pub_fn_used_in_production() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(
            tmp.path().join("src/lib.rs"),
            "pub fn real_feature(x: u32) -> u32 { x + 1 }\n\
             pub fn caller() -> u32 { real_feature(1) }\n",
        )
        .unwrap();
        let scan = scan_repo(tmp.path());
        let hints = dead_pub_fn_hints(&scan);
        assert!(
            hints.iter().all(|h| !h.contains("real_feature")),
            "real_feature is called from caller() and must not be flagged: {hints:?}"
        );
    }

    #[test]
    fn flags_pub_fn_unused_anywhere_with_lower_score_than_test_only() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::create_dir_all(tmp.path().join("tests")).unwrap();
        std::fs::write(
            tmp.path().join("src/lib.rs"),
            "pub fn truly_dead(x: u32) -> u32 { x }\n\
             pub fn test_only_alive(x: u32) -> u32 { x }\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("tests/it.rs"),
            "fn t() { test_only_alive(1); }\n",
        )
        .unwrap();
        let scan = scan_repo(tmp.path());
        let hints = dead_pub_fn_hints(&scan);
        assert_eq!(hints.len(), 2);
        // Test-only usage (score 2) ranks above unused-anywhere (score 1).
        assert!(hints[0].contains("test_only_alive"));
        assert!(hints[0].contains("only ever referenced from test code"));
        assert!(hints[1].contains("truly_dead"));
        assert!(hints[1].contains("was not found referenced anywhere else"));
    }

    #[test]
    fn implicit_dispatch_methods_are_never_flagged() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(
            tmp.path().join("src/lib.rs"),
            "impl std::fmt::Display for Foo {\n    pub fn fmt(&self) {}\n}\n",
        )
        .unwrap();
        let scan = scan_repo(tmp.path());
        let hints = dead_pub_fn_hints(&scan);
        assert!(
            hints.is_empty(),
            "fmt is operator/macro-dispatched and must be denylisted: {hints:?}"
        );
    }

    #[test]
    fn vendored_and_benchmark_paths_are_excluded_from_dead_fn_detection() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(
            tmp.path()
                .join("benchmarks/external/exercism-rust/exercises/foo/src"),
        )
        .unwrap();
        std::fs::create_dir_all(
            tmp.path()
                .join("benchmarks/external/exercism-rust/exercises/foo/tests"),
        )
        .unwrap();
        std::fs::write(
            tmp.path()
                .join("benchmarks/external/exercism-rust/exercises/foo/src/lib.rs"),
            "pub fn solve(x: u32) -> u32 { x }\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path()
                .join("benchmarks/external/exercism-rust/exercises/foo/tests/it.rs"),
            "fn t() { solve(1); }\n",
        )
        .unwrap();
        let scan = scan_repo(tmp.path());
        let hints = dead_pub_fn_hints(&scan);
        assert!(
            hints.is_empty(),
            "an exercise's own impl being only-tested-by-its-own-tests is the expected shape, not a finding: {hints:?}"
        );
    }

    #[test]
    fn scan_repo_powers_both_detectors_from_one_walk() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src/cognitive_loop/consciousness_engine"))
            .unwrap();
        std::fs::create_dir_all(tmp.path().join("tests")).unwrap();
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
        std::fs::write(
            tmp.path().join("src/lib.rs"),
            "pub fn orphaned(x: u32) -> u32 { x }\n",
        )
        .unwrap();
        std::fs::write(tmp.path().join("tests/it.rs"), "fn t() { orphaned(1); }\n").unwrap();

        let scan = scan_repo(tmp.path());
        let none_hints = none_arg_mismatch_hints(&scan);
        let dead_hints = dead_pub_fn_hints(&scan);
        assert_eq!(none_hints.len(), 1);
        assert_eq!(dead_hints.len(), 1);
    }
}
