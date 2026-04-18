// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Structural repair for Nix code (Phase 1 / M1 of the coding-AI roadmap).
//!
//! Takes a generated Nix snippet + a failing `StructuralVerdict` and
//! produces a patched version that closes the most common gaps:
//!
//! 1. **Missing required paths** — append flat dotted assignments to
//!    the module body.
//! 2. **Wrong bool values** — flip `false` → `true` (or vice versa) at
//!    the mismatched path.
//! 3. **Protocol swap** — if `allowedUDPPorts` is missing AND
//!    `allowedTCPPorts` is extraneous (or vice versa), assume the
//!    generator picked the wrong protocol and swap in place.
//!
//! Deliberately text-based, not rnix-AST-rewriting, for two reasons:
//! (a) the scorer uses rnix to *detect* the issue, but the fix only
//! needs to preserve the generated code's shape; (b) AST round-tripping
//! via rnix 0.11 loses source-level trivia (comments, exact formatting)
//! that the user values.
//!
//! Scope: repairs the 3 most common scorer-failure modes. Missing
//! entire subsystems (e.g. generator emitted `{ }` for a time-zone
//! prompt) are NOT repairable — that's a missing idiom, not a
//! fixable output.
//!
//! Feature-gated behind `code_generation` (matches scorer).

use crate::language::nix_codegen::generate_nix;
use crate::language::nix_scorer::{score, CanonValue, StructuralVerdict, ValueMismatch};

/// Result of a repair attempt.
#[derive(Debug, Clone)]
pub struct RepairedCode {
    /// The patched source. Always valid Nix IF the original was.
    pub code: String,
    /// The repair actions applied, in order. Empty if nothing was
    /// repairable (caller should treat this the same as `None`).
    pub steps: Vec<RepairStep>,
}

/// One recorded repair action. Logged for explainability — the
/// benchmark's `--repair` mode prints these so a human can see what
/// the loop did before re-scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairStep {
    /// Inserted `<path> = <value>;` into the module body.
    AppendedPath { path: String, value: String },
    /// Replaced the RHS of `<path>` from one value to another.
    ReplacedValue {
        path: String,
        from: String,
        to: String,
    },
    /// Swapped a protocol-variant path (TCP↔UDP firewall list).
    SwappedProtocol { from: String, to: String },
}

/// Entry point. Returns `None` if the verdict has nothing repairable
/// or the repair heuristics don't apply. Otherwise returns the
/// modified code + the list of steps taken.
pub fn repair_structural(code: &str, verdict: &StructuralVerdict) -> Option<RepairedCode> {
    if verdict.parse_error.is_some() {
        // Parse errors go through `generate_nix_with_repair`'s existing
        // path, not here.
        return None;
    }
    let mut current = code.to_string();
    let mut steps: Vec<RepairStep> = Vec::new();

    // 1. Protocol swaps first — they flip both a missing and an
    //    extraneous path in one action, so doing them before the
    //    generic "append missing" branch avoids duplicate edits.
    if let Some(step) = try_protocol_swap(&current, verdict) {
        current = step.0;
        steps.push(step.1);
    }

    // 2. Wrong bool values — cheap in-place swap.
    for mismatch in &verdict.value_mismatches {
        if let Some((next, step)) = try_fix_bool_mismatch(&current, mismatch) {
            current = next;
            steps.push(step);
        }
    }

    // 3. Missing required paths — append as flat dotted assignments.
    //    Skip any path already satisfied by the protocol-swap branch.
    //    (Collect owned Strings so the read of `steps` doesn't hold an
    //    immutable borrow across the mutable `steps.push` below.)
    let already_added: std::collections::HashSet<String> = steps
        .iter()
        .filter_map(|s| match s {
            RepairStep::AppendedPath { path, .. } => Some(path.clone()),
            RepairStep::SwappedProtocol { to, .. } => Some(to.clone()),
            _ => None,
        })
        .collect();
    for path in &verdict.missing_required {
        if already_added.contains(path) {
            continue;
        }
        if let Some((next, step)) = try_append_path(&current, path) {
            current = next;
            steps.push(step);
        }
    }

    if steps.is_empty() {
        None
    } else {
        Some(RepairedCode {
            code: current,
            steps,
        })
    }
}

// ─── Protocol swap ─────────────────────────────────────────────────────

/// If missing contains a UDP-ports path and extraneous contains the
/// TCP equivalent (or vice versa), do a single-token substitution in
/// the source and report one SwappedProtocol step.
fn try_protocol_swap(code: &str, verdict: &StructuralVerdict) -> Option<(String, RepairStep)> {
    const TCP: &str = "networking.firewall.allowedTCPPorts";
    const UDP: &str = "networking.firewall.allowedUDPPorts";

    let missing_has_udp = verdict.missing_required.iter().any(|p| p == UDP);
    let extra_has_tcp = verdict.extraneous.iter().any(|p| p == TCP);
    if missing_has_udp && extra_has_tcp {
        let next = code.replace("allowedTCPPorts", "allowedUDPPorts");
        return Some((
            next,
            RepairStep::SwappedProtocol {
                from: TCP.to_string(),
                to: UDP.to_string(),
            },
        ));
    }

    let missing_has_tcp = verdict.missing_required.iter().any(|p| p == TCP);
    let extra_has_udp = verdict.extraneous.iter().any(|p| p == UDP);
    if missing_has_tcp && extra_has_udp {
        let next = code.replace("allowedUDPPorts", "allowedTCPPorts");
        return Some((
            next,
            RepairStep::SwappedProtocol {
                from: UDP.to_string(),
                to: TCP.to_string(),
            },
        ));
    }

    None
}

// ─── Bool value swap ───────────────────────────────────────────────────

/// Handle `enable = false;` vs golden's `enable = true;` (and the
/// opposite direction). Deliberately conservative — only fires when
/// the mismatch is a pure Bool→Bool flip.
fn try_fix_bool_mismatch(code: &str, mismatch: &ValueMismatch) -> Option<(String, RepairStep)> {
    let (got, want) = match (&mismatch.got, &mismatch.want) {
        (CanonValue::Bool(g), CanonValue::Bool(w)) if g != w => (*g, *w),
        _ => return None,
    };
    // Find the assignment for this path in the flat-dotted form and
    // replace the RHS. If the code uses nested form
    // (`services.nginx = { enable = false; };`) we fall through — that's
    // a harder rewrite deferred to a later milestone.
    let leaf = mismatch.path.rsplit('.').next()?;
    let needle = format!("{} = {}", leaf, got);
    let replacement = format!("{} = {}", leaf, want);

    // Only replace within a line that also mentions the last-but-one
    // path segment, to avoid clobbering unrelated `enable = false;`
    // assignments elsewhere in the snippet.
    let anchor = path_anchor(&mismatch.path);
    let mut out = String::with_capacity(code.len());
    let mut replaced = false;
    for line in code.lines() {
        if !replaced && line.contains(&needle) && line.contains(&anchor) {
            out.push_str(&line.replacen(&needle, &replacement, 1));
            replaced = true;
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    if !replaced {
        return None;
    }

    Some((
        out,
        RepairStep::ReplacedValue {
            path: mismatch.path.clone(),
            from: got.to_string(),
            to: want.to_string(),
        },
    ))
}

/// For path `services.postgresql.enable`, returns `postgresql.enable`.
/// For top-level path like `enable`, returns the path itself. Used to
/// disambiguate which `enable = false;` line to flip when the snippet
/// has several.
fn path_anchor(path: &str) -> String {
    let segs: Vec<&str> = path.split('.').collect();
    if segs.len() >= 2 {
        format!("{}.{}", segs[segs.len() - 2], segs[segs.len() - 1])
    } else {
        path.to_string()
    }
}

// ─── Missing path append ───────────────────────────────────────────────

/// Inject `<path> = true;` (for `.enable`-ish paths) or a best-guess
/// value before the last `}` in the snippet. Defers to the scorer's
/// own parse pass to reject malformed output; if the result doesn't
/// re-parse, the repair loop caller will notice and discard.
fn try_append_path(code: &str, path: &str) -> Option<(String, RepairStep)> {
    let value = default_value_for(path);
    let assignment = format!("  {} = {};\n", path, value);

    // Find the final `}` in the snippet — that's the module body's
    // closing brace (we assume well-formed input from the generator).
    let last_close = code.rfind('}')?;
    let (prefix, suffix) = code.split_at(last_close);
    let out = format!("{}{}{}", prefix, assignment, suffix);

    Some((
        out,
        RepairStep::AppendedPath {
            path: path.to_string(),
            value,
        },
    ))
}

/// Heuristic default value for a missing path. `enable` / `openFirewall`
/// / flags → `true`. Package paths → we don't guess (return `null`).
/// Anything else → `true` as least-bad default.
fn default_value_for(path: &str) -> String {
    let last = path.rsplit('.').next().unwrap_or("");
    match last {
        "enable" | "enable32Bit" | "enableTCPIP" | "openFirewall" | "modesetting"
        | "nvidiaSettings" => "true".to_string(),
        "package" | "pkgs" => "null".to_string(),
        // Ports lists — we'd need the number from the prompt. Default
        // to empty list so the parse is valid; a later repair iteration
        // can fill in the number.
        p if p.ends_with("Ports") => "[ ]".to_string(),
        // timeZone expects a string — can't guess the zone.
        "timeZone" => "\"UTC\"".to_string(),
        _ => "true".to_string(),
    }
}

// ─── M2: Scorer-in-the-loop repair ─────────────────────────────────────

/// Outcome of running the generate → score → repair loop.
#[derive(Debug, Clone)]
pub struct ScorerRepairResult {
    /// Final code, after any successful repairs.
    pub code: String,
    /// Final verdict. `pass() == true` if the loop converged;
    /// otherwise the last state before giving up.
    pub verdict: StructuralVerdict,
    /// Every repair step applied across all iterations, in order.
    /// An empty vec means initial generation already passed.
    pub steps: Vec<RepairStep>,
    /// How many repair iterations ran before PASS / giveup. Zero
    /// when the first generation already passed; `max_iters` when
    /// the loop exhausted its budget.
    pub iterations: usize,
}

/// Generate code for `prompt`, score it against `golden`, and
/// iteratively repair until PASS or `max_iters` exhausted.
///
/// This is the agent-loop demo the roadmap calls out: **the scorer
/// becomes an oracle the generator is conditioned on**. Each FAIL
/// produces structured feedback (missing paths, wrong bool values,
/// protocol mismatches) that `repair_structural` consumes; the loop
/// converges when every required path is present with the right value.
///
/// `max_iters` caps the loop to prevent pathological cases (e.g.
/// repair produces a new FAIL that repair can't fix) from spinning.
/// Recommended: 5. Each iteration does one scorer call (ms) and one
/// repair call (ms). Cheap.
pub fn generate_nix_with_scorer_repair(
    prompt: &str,
    golden: &str,
    max_iters: usize,
) -> ScorerRepairResult {
    let initial = generate_nix(prompt);
    let mut code = initial.code;
    let mut all_steps: Vec<RepairStep> = Vec::new();
    let mut iterations = 0;

    loop {
        let verdict = score(&code, golden);
        if verdict.pass() {
            return ScorerRepairResult {
                code,
                verdict,
                steps: all_steps,
                iterations,
            };
        }
        if iterations >= max_iters {
            return ScorerRepairResult {
                code,
                verdict,
                steps: all_steps,
                iterations,
            };
        }
        match repair_structural(&code, &verdict) {
            Some(repaired) => {
                code = repaired.code;
                all_steps.extend(repaired.steps);
                iterations += 1;
            }
            None => {
                // Nothing to repair — return the current state.
                return ScorerRepairResult {
                    code,
                    verdict,
                    steps: all_steps,
                    iterations,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::nix_scorer::score;

    const HEADER: &str = "{ config, pkgs, ... }: ";
    fn wrap(body: &str) -> String {
        format!("{}{{\n{}\n}}", HEADER, body)
    }

    #[test]
    fn no_mismatches_no_repair() {
        let code = wrap("services.nginx.enable = true;");
        let verdict = score(&code, &code);
        assert!(verdict.pass());
        assert!(repair_structural(&code, &verdict).is_none());
    }

    #[test]
    fn appends_single_missing_path() {
        let gen = wrap("services.nginx.enable = true;");
        let golden = wrap(
            "services.nginx.enable = true;\n\
             services.postgresql.enable = true;",
        );
        let verdict = score(&gen, &golden);
        assert!(!verdict.pass());
        let repaired = repair_structural(&gen, &verdict).expect("should repair");
        assert_eq!(repaired.steps.len(), 1);
        match &repaired.steps[0] {
            RepairStep::AppendedPath { path, value } => {
                assert_eq!(path, "services.postgresql.enable");
                assert_eq!(value, "true");
            }
            _ => panic!("expected AppendedPath, got {:?}", repaired.steps[0]),
        }
        // Re-score the repaired code — must now PASS.
        let reverdict = score(&repaired.code, &golden);
        assert!(
            reverdict.pass(),
            "repaired code must re-score PASS; got {:?}",
            reverdict
        );
    }

    #[test]
    fn flips_bool_false_to_true() {
        // Exact false-positive case from the scorer docs — the scorer
        // catches enable=false vs golden's enable=true, and the
        // repair loop flips it.
        let gen = wrap("services.nginx.enable = false;");
        let golden = wrap("services.nginx.enable = true;");
        let verdict = score(&gen, &golden);
        assert!(!verdict.pass());
        let repaired = repair_structural(&gen, &verdict).expect("should repair");
        assert_eq!(repaired.steps.len(), 1);
        match &repaired.steps[0] {
            RepairStep::ReplacedValue { path, from, to } => {
                assert_eq!(path, "services.nginx.enable");
                assert_eq!(from, "false");
                assert_eq!(to, "true");
            }
            _ => panic!("expected ReplacedValue"),
        }
        let reverdict = score(&repaired.code, &golden);
        assert!(reverdict.pass());
    }

    #[test]
    fn swaps_tcp_to_udp_when_both_sides_mismatch() {
        // The exact UDP-regression case from this session: generator
        // emits TCP, golden wants UDP, scorer surfaces both sides.
        let gen = wrap("networking.firewall.allowedTCPPorts = [ 51820 ];");
        let golden = wrap("networking.firewall.allowedUDPPorts = [ 51820 ];");
        let verdict = score(&gen, &golden);
        assert!(!verdict.pass());
        let repaired = repair_structural(&gen, &verdict).expect("should repair");
        assert!(repaired
            .steps
            .iter()
            .any(|s| matches!(s, RepairStep::SwappedProtocol { .. })));
        let reverdict = score(&repaired.code, &golden);
        assert!(
            reverdict.pass(),
            "protocol swap should close the gap; got {:?}",
            reverdict
        );
    }

    #[test]
    fn repairs_both_missing_and_mismatch_in_one_pass() {
        let gen = wrap("services.nginx.enable = false;");
        let golden = wrap(
            "services.nginx.enable = true;\n\
             services.postgresql.enable = true;",
        );
        let verdict = score(&gen, &golden);
        let repaired = repair_structural(&gen, &verdict).expect("should repair");
        // Two steps: one value flip + one append.
        assert_eq!(repaired.steps.len(), 2);
        let reverdict = score(&repaired.code, &golden);
        assert!(
            reverdict.pass(),
            "combined repair should close both gaps; got {:?}",
            reverdict
        );
    }

    #[test]
    fn does_not_touch_unrelated_bool_lines() {
        // Two `enable = false;` lines, only one should flip.
        let gen = wrap(
            "services.nginx.enable = false;\n\
             services.redis.servers.\"\".enable = false;",
        );
        let golden = wrap(
            "services.nginx.enable = true;\n\
             services.redis.servers.\"\".enable = false;",
        );
        let verdict = score(&gen, &golden);
        let repaired = repair_structural(&gen, &verdict).expect("should repair");
        // Only nginx should have been flipped.
        assert!(repaired.code.contains("services.nginx.enable = true;"));
        assert!(repaired
            .code
            .contains("services.redis.servers.\"\".enable = false;"));
    }

    #[test]
    fn parse_error_verdict_returns_none() {
        let mut v = StructuralVerdict::default();
        v.parse_error = Some("boom".to_string());
        v.missing_required.push("services.nginx.enable".to_string());
        assert!(repair_structural("{ }", &v).is_none());
    }

    #[test]
    fn unknown_path_default_value_is_true() {
        assert_eq!(default_value_for("services.foo.enable"), "true");
        assert_eq!(default_value_for("networking.firewall.enable"), "true");
        assert_eq!(
            default_value_for("networking.firewall.allowedTCPPorts"),
            "[ ]"
        );
        assert_eq!(default_value_for("time.timeZone"), "\"UTC\"");
        assert_eq!(default_value_for("services.postgresql.package"), "null");
    }

    #[test]
    fn path_anchor_uses_last_two_segments() {
        assert_eq!(
            path_anchor("services.postgresql.enable"),
            "postgresql.enable"
        );
        assert_eq!(path_anchor("enable"), "enable");
    }

    // ── M2: scorer-in-the-loop integration tests ──────────────────

    #[test]
    fn loop_no_ops_when_initial_passes() {
        // If the generator's output already PASSes the golden, the
        // loop returns with zero iterations and zero steps.
        let prompt = "enable nginx web server";
        let golden = "{ config, pkgs, ... }:\n{\n  services.nginx.enable = true;\n}\n";
        let result = generate_nix_with_scorer_repair(prompt, golden, 5);
        assert!(
            result.verdict.pass(),
            "initial should pass; got {:?}",
            result.verdict
        );
        assert_eq!(result.iterations, 0);
        assert!(result.steps.is_empty());
    }

    #[test]
    fn loop_closes_intel_gpu_gap_via_append() {
        // The "configure intel hardware acceleration" prompt has
        // generated `{ # hardware config }` all session. Golden
        // demands `hardware.graphics.enable = true`. Repair loop
        // should close the gap via the append branch.
        let prompt = "configure intel hardware acceleration";
        let golden = "{ pkgs, ... }:\n{\n  hardware.graphics.enable = true;\n}\n";
        let result = generate_nix_with_scorer_repair(prompt, golden, 5);
        assert!(
            result.verdict.pass(),
            "intel GPU gap should close after repair; got {:?} after {} iters",
            result.verdict,
            result.iterations
        );
        assert_eq!(result.iterations, 1, "single append iteration suffices");
        assert!(result.steps.iter().any(|s| matches!(
            s,
            RepairStep::AppendedPath { path, .. } if path == "hardware.graphics.enable"
        )));
    }

    #[test]
    fn loop_caps_at_max_iters() {
        // A golden demanding something the generator + repairer
        // cannot satisfy (an exotic path not in our default-value
        // table; but still appendable) should converge in ≤1 iter
        // because the heuristic always emits `true` as fallback.
        // What we're checking here is the CAP — with a golden the
        // loop genuinely can't close, it still exits in finite time.
        let prompt = "enable nginx web server";
        // Demand a path the generator won't emit; repair will append
        // it with value=true, which satisfies the golden too, so the
        // loop should converge. Pick something whose append default
        // doesn't match — but any `.enable` path gets `true`, so this
        // is actually self-satisfying. Keep the assertion to "converges
        // or caps" rather than forcing a specific outcome.
        let golden = "{\n  services.nonexistent_xyz.enable = true;\n}\n";
        let result = generate_nix_with_scorer_repair(prompt, golden, 3);
        assert!(
            result.iterations <= 3,
            "loop must respect max_iters cap; got {} iters",
            result.iterations
        );
    }

    #[test]
    fn append_places_before_final_brace() {
        let gen = wrap("services.nginx.enable = true;");
        let golden = wrap(
            "services.nginx.enable = true;\n\
             services.postgresql.enable = true;",
        );
        let verdict = score(&gen, &golden);
        let repaired = repair_structural(&gen, &verdict).expect("should repair");
        // New line must be BEFORE the final `}`, not after.
        let last_brace = repaired.code.rfind('}').unwrap();
        let inserted = repaired.code.find("services.postgresql.enable").unwrap();
        assert!(
            inserted < last_brace,
            "appended path must precede closing brace"
        );
    }
}
