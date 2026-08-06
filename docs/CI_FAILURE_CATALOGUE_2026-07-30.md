# Independent verification of this catalogue (2026-07-30)

The body below was produced by a 9-agent parallel diagnosis. **It was not taken on trust.**
Six load-bearing claims were re-checked by hand against the CI logs and both trees. Result:
**four confirmed exactly, one confirmed after I initially mis-refuted it, one materially wrong.**

| Claim | Verdict | Evidence |
|---|---|---|
| Clippy: all 25 in `symthaea-alife`, zero in `symthaea-vision-manifold` | ✅ **confirmed** | Log breakdown is exactly 17 + 6 + 1 + 1 = 25, every `-->` path under `crates/domains/symthaea-alife`. `could not compile symthaea-alife` also confirms the early abort. |
| psych-bench `ArcChain` collapse | ✅ **confirmed verbatim** | `chain_3 1.0000 -> 0.0167 (-98.3%)`, `chain_4 1.0000 -> 0.1500 (-85.0%)`. See escalation below. |
| broca `enable_hard_mask` field never existed | ✅ **confirmed** | Zero definitions in `symthaea-broca/src/`; the only occurrence workspace-wide is the example that *sets* it (`b2052043d9`). |
| muse: 2 missing `pub mod`, patch already uncommitted locally | ✅ **confirmed** | `git diff HEAD` shows both lines as `+`. My first check used the working tree and looked like a refutation; checking HEAD inverted it. |
| `symthaea-perception` deliberately in workspace `exclude` | ✅ **confirmed — I was wrong** | I grepped line 33 and called it a member. `members` starts line 3, `exclude` line 17; line 33 is inside `exclude`, under an explicit `ARCHIVED 2026-07-16` comment. |
| wisdom: "24 of 28 src files are `// placeholder`" | ⚠️ **materially wrong** | True of the CI branch (28 files). **False of the monorepo** — `53ea9e9a48` already deleted them; only 4 real modules remain. See below. |

## Correction 1 — wisdom is not what the catalogue says, and is worse

The catalogue classifies wisdom as **D7, "implement or quarantine, multi-day-to-multi-week"**,
on the premise that 24 of 28 source files are placeholders. That premise is stale.

What is actually true:

- The 24 placeholder modules **were already deleted in the monorepo** by `53ea9e9a48`
  (*"delete 148 orphaned `// placeholder` files"*), after being added by patchset wave `0cc1ff8539`.
- The monorepo's `symthaea-wisdom` now has **4** source modules (`autopoiesis`, `harmonics`,
  `meta_cognition`, `lib`).
- **The 7 test files were not deleted with them.** They still import ~40 symbols from the removed
  subsystem.

Verified directly:

```
$ cargo check -p symthaea-wisdom --tests
error[E0432]: unresolved imports `symthaea_wisdom::ActionExecutionCoordinator`,
  `symthaea_wisdom::EvidenceLedger`, `symthaea_wisdom::TrustRegistry`, ... (24 listed)
error: could not compile `symthaea-wisdom` (test "archive_startup_invariants")
```

**Consequence, and it inverts the catalogue's advice:** a fresh full re-export (decision **D1**)
was expected to be the highest-leverage move partly because it might carry wisdom along. It does
not. The monorepo is broken *identically* — a cleanup pass deleted an implementation and left its
tests dangling. D7 remains a genuine decision, but its subject changed: not *"a crate full of
placeholders"* but *"tests orphaned from a deliberately deleted subsystem."* Deleting or
quarantining those 7 test files is now a defensible and much cheaper option than the catalogue's
framing suggests, because the thing they test was already judged not worth keeping.

**Do not check `docs/ops/*Patch_Sets*` for recovery here** (the catalogue's D7 advice, carried over
from the Symtropy pass). The content is not lost — it was *deliberately removed*, and restoring it
would revert someone's considered cleanup.

⚠️ **Concurrency:** another session is editing 5 of those 7 test files right now (an import-sort
pass, symbols reordered not removed). Do not touch `symthaea-wisdom/tests/` without coordinating.

## Correction 2 — `ArcChain` deserves escalation above "red job"

The catalogue rightly says a human must look before anyone regenerates the baseline. One detail
sharpens it into something that does not depend on trusting the baseline at all:

```
chain_3_accuracy: 0.0167
chain_4_accuracy: 0.1500
```

**`chain_3` scores below `chain_4`.** A 3-step chain is strictly easier than a 4-step chain, so
that ordering is not reachable by a sound scorer on a harder-is-lower task. Whatever the baseline
says, the *current run* is internally inconsistent — which is evidence of a live defect in
`ArcChain` scoring or generation, independent of the stale-baseline question. Meanwhile the
crate's own `test_chain_2_above_chance` still passes, so the internal sanity check does not cover
this.

Treat as: **a possible real capability regression with an independent internal red flag**, not a
CI-hygiene item. Regenerating `baselines/v0.9.0.json` first would silently bless it.

## Standing methodological note

Two traps fired during this verification and both are worth remembering, because both produce
**false negatives that read as clean results**:

1. **GitHub Actions logs carry a timestamp prefix** (`2026-07-27T18:09:25.123Z `) *in addition* to
   ANSI colour codes. `grep -E '^error'` silently returns nothing against either. Strip both:
   `sed -E 's/^[0-9T:.-]+Z //' | sed 's/\x1b\[[0-9;]*m//g'`.
2. **Grep reads the working tree; `git diff HEAD` reads the commit.** Checking a "is this
   committed?" question with grep produced a confident refutation of a correct claim.

---

# CI Failure Catalogue — run [30496274683](https://github.com/Luminous-Dynamics/symthaea/actions/runs/30496274683)

**Repo:** Luminous-Dynamics/symthaea · **PR:** [#32](https://github.com/Luminous-Dynamics/symthaea/pull/32) (OPEN) · **Branch:** `export/6a2fdb112e-20260727-180925` @ `b69c7ea365`
**Totals:** 100 success · **22 failure** · 33 cancelled · 2 skipped (157 jobs)

## Record corrections (MASTER_ROADMAP P0-#2)

| Roadmap says | Actual |
|---|---|
| "~10 real failures, not yet individually triaged" | **22 failures.** All 22 now diagnosed. |
| Tracked on PR #31 | **#31 is CLOSED.** #32 supersedes it — same branch, title: *"continued from #31 — CI trigger stopped firing reliably"*. |
| Still failing: Documentation Tests, Hardened API Regressions, Compliance (Robustness & Governance), Compliance (Safety & Ethics) | **All four now PASS.** |
| Still failing: `Test CI-safe (science-infra)`, `Test Feature Combinations (mesh)` | **Cancelled in this run, not fixed.** The entire `Test Feature Combinations` matrix (25 jobs) and entire `Test CI-safe` matrix (7 jobs) were cancelled. Status unknown. |
| Still failing: `symthaea-broca (test-helpers)` | That job no longer exists; the workflow now has `Test Sub-Crates (symthaea-broca)`, which fails for an unrelated reason. |
| `Workspace Target Integrity` passes | Confirmed — still passes. |

**Meta-finding:** none of the 22 is a fresh regression. All are pre-existing debt made visible for the first time by the 2026-07-27 workflow repair (the invalid `replace()` expression that had silently voided the whole workflow file). The "dominant cause = timeouts" hypothesis is dead: **zero of the 22 are infra or timeout failures.**

---

## 1. Catalogue

Ordered by classification so shared causes sit together.

### config-drift (10)

| Job | Root cause (one line) | Fix | Conf. |
|---|---|---|---|
| Embodiment Safety Composition | Branch tree is stale: `symthaea-gravcraft/src/embodiment.rs` lacks the `safety_override`/`moral_safety` composition that monorepo `d49a18aaa9` added; only the *lint script* was hand-copied onto the branch. | trivial | high |
| Orphan Module Check | Same stale-branch mechanism: branch got the post-cleanup 19-entry QUARANTINE list from `53ea9e9a48` but not the 148 `// placeholder` deletions ⇒ 118 ORPHAN + 1 STRANDED SUBTREE. | trivial | high |
| Test Sub-Crates (symthaea-orbital) | `orbital-mechanics` is a `../../../../mycelix-workspace/...` path dep, commented out by `export-to-standalone.sh`, but `src/` still `use`s it unconditionally ⇒ lib won't compile. | medium | high |
| Test Sub-Crates (symthaea-auv) | Identical: `positioning` path dep stripped on export, `src/navigation_bridge.rs` uses it unconditionally. | small | high |
| Test Sub-Crates (symthaea-evidence-plane) | `cargo test -p …` on a package absent from the tested tree — crate landed in monorepo `1f242dbd55`, not an ancestor of the export commit, but `ci.yml:1255` lists it. | trivial | high |
| Test Sub-Crates (symthaea-perception) | Same error string, different cause: crate is deliberately in the workspace `exclude` list (archived 2026-07-16) yet still in the CI matrix at `ci.yml:1355`. | trivial | high |
| Genesis Mission Benchmarks (temporal_unified, safety-agents) | Example uses `symthaea_physics`/`symthaea_nuclear_forensics`, both optional deps not enabled by `required-features` nor by the matrix feature string. Actual compiler error never emitted — `bash -e` kills the step before `echo "$output"`. | trivial | high |
| Hardened Nix Regressions | `nix/tests/eval-api-security.nix:26` `src = ../../../.` escapes the flake root ⇒ `path '/nix/store/' is not in the Nix store`; `sourceRoot = "source/symthaea"` is also invalid in the standalone layout. | medium | high |
| WASM Compatibility (Spore) | No compile error — artifact is 681,742 B vs a hard 512,000 B budget. Byte-identical across 3 runs; has never been green. `nixward` (~45K LOC) is pulled in by the `wasm` feature. | small | high |
| Feature Interactions (safety-agents,ssm_language) | Two real feature-gating holes in `symthaea` lib: unconditional `symthaea_causal_reasoning::counterfactual` import, and a 6-vs-5 arg mismatch at `cycle.rs:409` when `sentinel` is off. Masked by `default-mind`. | small | high |

### missing-symbol (4)

| Job | Root cause (one line) | Fix | Conf. |
|---|---|---|---|
| Muse (tests, studio, wasm UI) | `symthaea-muse/src/lib.rs` never declares `pub mod teaching_corpus;` / `pub mod symbolic_import;` — both files exist, all four called fns exist. Monorepo HEAD is equally broken. | trivial | high |
| Test Sub-Crates (symthaea-fabrication-kernel) | Tests call `OperatorCommandTracker`/`GatewayConsensusTracker`/`IncidentLedger` with no `use` (all are root-exported), plus `run_standard_fault_matrix` exists but is never re-exported at the crate root. | trivial | high |
| Test Sub-Crates (symthaea-vocal-tract) | Examples import a `Checkpoint*` API defined **nowhere in the workspace**; plus 3 undeclared deps (`hound` optional-but-used-unconditionally, `postcard`, `blake3`). Zero tests ran. | medium | high |
| Test Sub-Crates (symthaea-wisdom) | 24 of 28 `src/` files are literal `// placeholder`; lib.rs declares only 3 modules; 2 test targets import 24 and 41 nonexistent symbols. Lib compiles; the crate is hollow. | large | high |

### stale-test (6)

| Job | Root cause (one line) | Fix | Conf. |
|---|---|---|---|
| Test Sub-Crates (symthaea-broca) | `examples/epistemic_gate_strength.rs` sets `GatingConfig::enable_hard_mask`, a field that **never existed in any commit** — and which CLAUDE.md says the project deliberately does not implement. | trivial | high |
| Test Sub-Crates (symthaea-spore) | `test_generation_produces_text` asserts `num_tokens > 0`, but `18c4434f79` moved the no-checkpoint routing threshold 0.5→1.1, so it now always takes `generate_structured()`, which hardcodes `num_tokens: 0` while emitting real text. | trivial | high |
| Test Sub-Crates (symthaea-neuromodulators) | 3 tachyphylaxis tests pin level via `reuptake_rate: 0.0`, but clearance is now Michaelis-Menten and never reads that field; level decays, counter resets, `high_exposure_cycles == 0`. 309 pass / 3 fail. | small | high |
| Test Sub-Crates (symthaea-fabrication-kernel)* | *(listed above — E0433 is a test-local omission, E0432 a genuine missing re-export)* | — | — |
| Test Sub-Crates (symthaea-manipulator) | `consciousness_proofs.rs:133` asserts effort is non-increasing Green≥Yellow≥Orange, but `admittance.rs` deliberately sets compliance_gain Green 0.2 < Yellow 0.45 < Orange 0.75. Only the Green≥Yellow step fails. | small | **medium** |
| Test Sub-Crates (symthaea-psych-bench) | `regression_against_baseline` compares against a 131-day-old snapshot; the gate is direction-blind (improvements flagged Critical) and the baseline is saturated with `mean=1.0, ci=[1.0,1.0]`. 607 pass / 24 "critical". | medium | **medium** |

### lint-only (1)

| Job | Root cause (one line) | Fix | Conf. |
|---|---|---|---|
| Clippy | 25 occurrences, 4 lint kinds, **all in `symthaea-alife`**: `manual_is_multiple_of` ×17, `doc_lazy_continuation` ×6, `should_implement_trait` ×1, `useless_conversion` ×1. Toolchain-pin-driven (1.93→1.95→1.96), not a source regression. | small | high |

### mixed runtime failure (1)

| Job | Root cause (one line) | Fix | Conf. |
|---|---|---|---|
| Test Feature Matrix (web_research/pathology) — `web_research` leg | `cargo test -p symthaea --lib --features "web_research_module school_learning" -- --test-threads=1`: **6613 passed, 18 failed** in 5677 s. Not one cause — see breakdown below. | medium | **medium** |

*This was the job with no supplied diagnosis. Diagnosed here from its log plus local source.* Breakdown of the 18:

- **8 × `language::nix_codegen`** (`test_eval_catches_typos_that_parse_misses`, `test_repair_returns_within_max_iterations`, `self_improve_smoke`, `render_rag_draft_includes_options_as_comments`, `generate_with_rag_uses_idiom_for_known_prompt`, `generate_with_rag_falls_back_for_unknown_service`, `rag_fast_cold_latency_is_sub_second`, and the `result.parses` assertions): `parses` is computed by shelling out to `nix-instantiate` (`nix_codegen.rs:1260/1347/1426`). This job installs no Nix. **Sibling tests in the same file already guard with `[skip] nix-instantiate not available` (lines ~2650/2677/2730); these 8 do not.** Fix: extend the existing guard. High confidence, but see Limits.
- **10 others, apparently independent**: `nix_scorer::golden_with_only_dynamic_attrpaths_fails_closed` (pure Rust — `golden_unscorable` not firing on `services.redis.servers..enable`; **not** nix-related, verified `nix_scorer.rs` never shells out); `semantic_repair` ×2 (`syn` `unwrap()` on `Err("unexpected end of input, expected lifetime")` and on `None`); `cfc_code_sequencer` ×2 (dimension asserts, left: 24 / left: 14041); `compose_codegen::postgres_with_stack_intent_classification` (got `SingleService`); `coding_prediction_error` (got `"off_by_one"`); `type_causal_model::test_skeleton_iterator_chain` (`.collect()` absent); `school::code_learning` (11 predictions, wanted 12+); `coding_agent::test_end_to_end_fibonacci_generation` ("Should have generated code") — this last one is plausibly P0-#4 (the 1024-token truncation) surfacing, **not confirmed**.

---

## 2. Shared causes

**22 failures are not 5 causes. They are 3 genuinely shared causes covering 6 jobs, one shared *class* covering 3 more, and 13 independent problems.** I am not going to manufacture more tidiness than the evidence supports.

### Real shared causes — one fix turns ≥2 jobs green

| # | Cause | Jobs | One fix? |
|---|---|---|---|
| **S1** | **Stale export branch.** The branch received hand-copied *lint scripts* (`b7791d0431`, +314/−0, scripts only) without the *source fixes* those lints were written against. Both fixes exist in the monorepo and are ancestors of the commit the branch claims to mirror (`d7e8a5e20e`). | Embodiment Safety Composition, Orphan Module Check | **Yes** — one fresh full `export-to-standalone.sh` run from ≥ `d7e8a5e20e`. |
| **S2** | **Cross-repo `../../../../` path deps stripped at export while `src/` uses them unconditionally.** The library, not a test, fails to compile in the public repo. 14+ crates carry such deps; only these two use them unconditionally, which makes this a latent trap with no local signal. | symthaea-auv, symthaea-orbital | **Yes** — one export-pipeline decision (publish/vendor, feature-gate, or exclude). |
| **S3** | **Hardcoded CI sub-crate matrix out of sync with the workspace.** Byte-identical `package ID specification … did not match any packages`, two different reasons (crate absent from the export vs. deliberately `exclude`d). | symthaea-evidence-plane, symthaea-perception | **Yes** — derive the matrix from `cargo metadata --no-deps`. Deleting two lines also works but leaves the class alive. |

### Shared class, **not** a shared fix

**C1 — patch-series orphan API** (`symthaea-wisdom`, `symthaea-vocal-tract`, `symthaea-fabrication-kernel`). Same provenance (patch sets landing files whose supporting implementation was never authored — the same defect as Symtropy's 57 placeholders and the `symthaea-humanoid` P0-#6 incident) and the same symptom shape (E0432/E0433 in non-lib targets). But the fixes are **trivial / medium / large** respectively: fabrication-kernel just needs 3 `use` lines and one root re-export; vocal-tract's `Checkpoint*` API exists nowhere in the workspace; wisdom is 24 placeholder modules. **Do not batch these.**

### Genuinely independent (13 jobs)

Clippy, Muse, WASM Spore, Hardened Nix, Genesis temporal_unified, both Feature Interactions legs, broca, spore, neuromodulators, manipulator, psych-bench, and the web_research leg. Each needs its own fix. Within the web_research leg alone there are ~11 distinct causes.

---

## 3. Defect vs. bookkeeping

The project's prior heuristic — "red CI means a test is stale, not that the code is broken" — **does not hold for this run.** Neither category dominates.

| Category | Jobs | Share |
|---|---|---|
| **Config / manifest / CI drift** (bookkeeping) | 10 | 45% |
| **Stale test / test-vs-API drift** | 6 | 27% |
| **Real missing or broken non-test code** | 4 | 18% |
| **Lint-only** | 1 | 5% |
| **Mixed runtime (≈11 causes in one job)** | 1 | 5% |
| **Infra / environment** | **0** | 0% |

### Real defects in shipped code (4 jobs, plus 5 riding inside other jobs)

Primary:
1. **`symthaea` lib is uncompilable in a valid feature combination** — `safety-agents,ssm_language` hits two real gating holes. Only this leg strips `default-mind`, so it is the only place this is visible.
2. **`symthaea-therapeutic` does not compile at all** — missing `model_registry::ModelExecutionReceipt`, never existed on any branch. The crate's own `lib.rs:13-43` documents this and explicitly forbids the cheap fix (deleting the field would trade a fail-closed clinical abstention for an unsafe success).
3. **`symthaea-muse`'s `studio` bin is unbuildable in committed monorepo code** — two missing `pub mod` lines.
4. **`symthaea-wisdom` is a hollow crate** — 24/28 source files are `// placeholder`.

Riding inside jobs classified otherwise, and worth separate tickets:
- **`symthaea-neuromodulators`**: `reuptake_rate` is dead in the clearance path while its doc comment still claims otherwise; the neighbouring negative-control test now passes **vacuously** (the counter never accumulates for any input) — a probe that could not catch a real regression.
- **`symthaea-spore`**: `generate_structured()` reports `num_tokens: 0` for text that genuinely contains words. Defensibly a real bug, not just a stale assertion.
- **`symthaea-psych-bench`**: the regression gate is direction-blind (4 confirmed *improvements* reported as Critical) and the baseline contains zero-width CIs, so any value below exactly 1.0 trips by construction. **Separately: `ArcChain/chain_3_accuracy` 1.0000 → 0.0167 and `chain_4` → 0.1500, with chain_3 scoring *below* chain_4.** That may be a real capability regression and must not be regenerated away.
- **`language::nix_scorer`**: `golden_unscorable` fails to fire on a dynamic attrpath — pure-Rust logic, unrelated to the nix binary.
- **`coding_agent::test_end_to_end_fibonacci_generation`**: "Should have generated code" — possibly P0-#4's 1024-token truncation surfacing. Unconfirmed.

### Not defects

- **Clippy**: all 25 are style/doc formatting with zero behavior change. `manual_is_multiple_of` on `x % 2 == 0` parity checks; `doc_lazy_continuation` indentation; one redundant `.into_iter()`. Fired because the toolchain pin advanced, not because `symthaea-alife` regressed.
- **Embodiment Safety Composition / Orphan Module Check**: both gate scripts pass cleanly against the current monorepo tree. There is no defect in monorepo source; the branch is simply stale.
- **The prior note "25+ real style lints in `symthaea-vision-manifold` alone" is stale and should be struck.** That crate *was* compiled under this exact `-D warnings` run and produced zero diagnostics; its lints were fixed by `0b76c8cdc3` / `8f0412f170`. The "25" is a coincidence of count, not of location.

---

## 4. Recommended order of attack

Green-count estimates are per-fix and assume nothing else changes. They are honest, not optimistic — several jobs abort before running everything they contain, so clearing the first error can expose a second.

### Tier 1 — cheap, no decision required (est. **+8 green**, ~half a day)

| # | Action | Greens | Note |
|---|---|---|---|
| 1 | Delete or derive the two bogus sub-crate matrix entries (`ci.yml:1255`, `1355`). Durable version: generate from `cargo metadata --no-deps`. | **+2** | Pure bookkeeping. Kills the whole class. |
| 2 | `symthaea-muse/src/lib.rs`: add 2 feature-gated `pub mod` lines. **The exact patch already exists uncommitted in the local monorepo working tree.** | **+1** | Step's 2nd command (`cargo check --bins`) never ran; may expose more. |
| 3 | `symthaea-fabrication-kernel`: add 3 `use` lines (copy from `tests/federated_release_pipeline.rs:8-13`) + one root re-export of `run_standard_fault_matrix`. | **+1** | |
| 4 | `symthaea-broca`: delete the dead `enable_hard_mask` arm from `examples/epistemic_gate_strength.rs`. Do **not** add the field — the project deliberately has no hard mask. | **+1** | |
| 5 | `symthaea-spore`: fix `test_generation_produces_text`, or better, make `generate_structured()` report a real token count. | **+1** | Prefer the latter — the current value is wrong for any caller. |
| 6 | `symthaea-neuromodulators`: repin the 3 tests under the Michaelis-Menten model (`mm_v_max: 0.0`, or re-`produce()` each cycle). Also fix the false `reuptake_rate` doc comment and the vacuous negative control. | **+1** | |
| 7 | `symthaea` lib: two localized `cfg` edits (gate `causal_reasoning_bridge`; move `#[cfg(feature="sentinel")]` onto the argument at `cycle.rs:409` — the pattern already exists correctly at `safety_supervisor.rs:44-48`). | **+1** | Real lib bug. |

### Tier 2 — cheap but each carries a trap or a small judgement (est. **+3**)

| # | Action | Greens | Trap |
|---|---|---|---|
| 8 | Genesis `temporal_unified`: fix `required-features` **and** the matrix feature string **together**. | **+1** | Fixing only `required-features` makes cargo silently *skip* the example and fail on `grep -q "PASS"` with a misleading message. Also: rewrite the step to `tee` its output — the real error is currently unloggable. |
| 9 | `Clippy`: `cargo clippy --fix` clears 24 of 25; the last (`AgentIdAllocator::next`) needs rename-vs-`#[allow]`. | **+1?** | The job aborted at the first failing crate — only 7 of the workspace's crates were ever linted. **Expect a tail.** Treat +1 as provisional. |
| 10 | Hardened Nix: fix `src`/`root` to not escape the flake root, and `sourceRoot`. | **+1** | 2 of the step's 3 `nix build` commands never ran (`bash -e`). Their status is unknown. |

### Tier 3 — needs a decision, not a fix

| # | Decision | Greens | Why it is a decision |
|---|---|---|---|
| **D1** | **Fresh full re-export vs. continue hand-patching the branch.** | **+2** | A clean re-export fixes Embodiment Safety + Orphan Module for free, but drops ~12 hand-applied triage commits on this branch — including one self-labelled *"EXPERIMENT (to be reverted): serial test-threads for the mesh leg only"* — and may surface new legs. This is the highest-leverage item and also the riskiest. Decide before doing anything else in Tier 1, since it may moot some of it. |
| **D2** | **Export pipeline for cross-repo path deps** (auv, orbital). Publish/vendor `positioning` + `orbital-mechanics`, feature-gate every consumer, or drop both from the matrix and document them monorepo-only. | **+2** | Currently the script strips a dep it knows the source requires, guaranteeing a red job. Whatever is chosen should also close the latent trap for the other 12+ crates with escaping path deps. |
| **D3** | **WASM Spore size budget.** Raise 512 KB → ~700 KB and correct the crate's now-false "~500KB" description, **or** actually reduce: reorder `wasm-opt` to run *after* `wasm-bindgen` (it currently optimizes a file that is then re-emitted and never re-optimized), add a wasm size profile, and make `nixward` opt-in rather than implied by the `wasm` feature. | **+1** | Product call: is 681 KB acceptable in a browser? |
| **D4** | **manipulator: stale premise or real safety-cascade bug?** Cheap decisive check exists — run the test and read its printed per-tier Motor Gain table. Green `motor_gain ≈ 1.0` ⇒ stale test; near 0 ⇒ real bug. | **+1** | Do the check before writing any fix. Medium confidence either way today. |
| **D5** | **psych-bench: is `ArcChain` a real regression?** Fix direction-blindness and add a zero-width-CI guard first; then a human must look at chain_3 1.0000 → 0.0167 **before** anyone regenerates `baselines/v0.9.0.json`. | **+1** | Regenerating first would silently bless a possible capability loss. Note the test currently *skips* locally (the baseline file is tracked in git but absent from the working tree) — that divergence should be fixed too. |
| **D6** | **vocal-tract: delete the orphaned checkpoint examples or author the subsystem?** | **+1** | Deleting is mechanically trivial but is a product decision about a landed patch series. Also needs 3 dep fixes (`hound` `required-features`, `postcard`/`blake3` added or their consumers removed). |
| **D7** | **wisdom: implement or quarantine?** Check `docs/ops/*Patch_Sets*` for recoverable `git format-patch` files **first** — the Symtropy pass recovered 16 documents / 5,854 lines that way after initially declaring them unrecoverable. | **+1** | Quarantining the 5 test files behind a feature makes CI green in minutes but hides a genuinely unimplemented crate. |
| **D8** | **therapeutic `ModelExecutionReceipt`.** The crate documents this as *"a separately authorized clinical-safety task, not a cleanup item"* and forbids the stub. | **+1** | Needs authorization, not engineering time. |

### web_research leg — handle last (**+0 until all 18 pass**)

Guarding the 8 `nix_codegen` tests with the file's own existing `nix-instantiate` availability check removes 8/18 for near-zero effort, but the job stays red until the other 10 are individually triaged. Note also this leg runs `--test-threads=1` and took **95 minutes**; it is the single most expensive job in the run.

### Honest cumulative estimate

- Tier 1 alone: **~8 of 22 green.**
- Tier 1 + Tier 2: **~11 of 22**, with Clippy provisional.
- Reaching 22 requires resolving **8 decisions**, two of which (D7 wisdom, D8 therapeutic) are multi-day-to-multi-week work, and one (D5 psych-bench) may uncover a real capability regression that is more important than the red job.

---

## 5. Limits

**Nothing was executed locally. No fix was verified. No file was edited, committed, or re-run.** All work here was read-only: `gh` API queries, downloading job logs, and `grep`/`sed` over monorepo source.

**Provenance of the diagnoses.** 21 of the 22 came from per-job analyses supplied to me, which I did not independently re-derive. I diagnosed the 22nd — `Test Feature Matrix (web_research/pathology)`, which had no supplied entry — from its log plus local source inspection. One supplied entry was a self-flagged non-job ("ignore this entry; included only in error") and is excluded.

**33 cancelled jobs are a bigger blind spot than the 22 failures.** They comprise the *entire* `Test Feature Combinations` matrix (25 legs), the *entire* `Test CI-safe` matrix (7 legs), and `Stress Tests (nightly)`. None ran. **The true failure count is ≥22, possibly considerably higher.** In particular, the two legs the roadmap previously named — `Test CI-safe (science-infra)` and `Test Feature Combinations (mesh)` — were cancelled here, so their status is unchanged and unknown.

**Six jobs abort before running everything they contain**, so blast radius exceeds what the logs show:
- Clippy linted only 7 crates before aborting; the main `symthaea` crate was never reached.
- Hardened Nix ran 1 of 3 `nix build` commands.
- Muse ran 1 of 2 commands.
- Genesis `temporal_unified`'s compiler error was **never emitted at all** — `bash -e` kills the step before `echo "$output"`. That diagnosis rests entirely on dependency resolution (`cargo tree` with and without the features), not on reading the error. Confidence is high but the evidence is indirect.
- vocal-tract reported 4 failing targets of 15 examples; cargo cancelled the rest. Most of the 15 are probably broken.
- wisdom reported 2 failing test targets, but 5 of its 7 test files reference the missing subsystem.

**Low-confidence diagnoses (2):**
- **manipulator** — medium. Cannot exclude a genuine regression that drove Green's effective motor gain toward zero, which would reclassify this from stale-test to a real safety-cascade bug. Not reproduced locally (host load 37.5 on 12 cores, 16 concurrent sessions; building it pulls symthaea-core + symthaea-fep). The decisive check is cheap and named in D4.
- **psych-bench** — medium *precisely because* the stale-baseline mechanism and a possible real ARC regression are entangled. Confidence would rise if someone confirmed whether `ArcChain` regressed for a code reason.

**Inference, not proof:** the attribution of the 8 `nix_codegen` failures to a missing `nix-instantiate` binary. No "No such file or directory" string appears anywhere in the log — because the *guarded* sibling tests passed, and cargo captures passing tests' stdout. What I verified: `try_nix_parse`/`try_nix_eval` shell out to `nix-instantiate` (`nix_codegen.rs:1260/1347/1426`); the failing tests call them with no availability guard while siblings in the same file do guard; and the job's only system-package step is `sudo apt-get update` (it installs no Nix). I rate this high-confidence, but it is not directly observed.

**Monorepo/branch divergence.** The tested tip `b69c7ea365` has drifted from its own export snapshot (~12 hand-applied triage commits since 2026-07-27), so local greps do not always reflect the tested tree. Two diagnoses hit this explicitly: `symthaea-wisdom`'s `src/` has 4 files locally vs. 28 in the tested tree, and the Embodiment/Orphan diagnoses were verified against the branch via the GitHub contents API rather than locally. Where a claim depends on the tested tree specifically, I have said so.

**Log-reading gotcha, worth recording for the next pass.** These logs carry ANSI escapes in **caret notation** (literal `^` + `[` bytes, from `CARGO_TERM_COLOR=always` plus GitHub's re-encoding). The commonly-used `sed 's/\\x1b\\[[0-9;]*m//g'` is a **silent no-op** on them, and a subsequent `grep "error:"` returns zero hits — which reads as "no errors found". The working filter is:

```
perl -pe 's/(?:\\e|\\^\\[)\\[[0-9;]*[A-Za-z]//g'
```

Two independent diagnosers hit this and flagged it; it is the most likely cause of a future triage pass concluding a job "has no visible error".",
    "jobs": [
      {
        "job": "Test Sub-Crates (symthaea-vocal-tract)",
        "root_cause": "First error chronologically: `error[E0433]: cannot find module or crate `hound` --> crates/domains/symthaea-vocal-tract/examples/f1_probe.rs:32:26`. Dominant error (3 of 4 failing targets): `error[E0432]: unresolved imports `symthaea_vocal_tract::CheckpointOperationalTrustMetrics`, `...::CheckpointOperationalTrustRequirements`, `...::MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES`, `...::apply_series21_public_verifiability`, ... --> examples/checkpoint_series21_hybrid_gossip_verifier.rs:12:5` — "no `CheckpointOperationalTrustMetrics` in the root".",
        "classification": "missing-symbol",
        "confidence": "high",
        "fix_difficulty": "medium",
        "shared_with": "Test Sub-Crates (symthaea-wisdom) and Test Sub-Crates (symthaea-fabrication-kernel) — identical root cause class: patch-series-landed non-lib targets importing an API surface the crate never exported (wisdom's tests/archive_startup_invariants.rs + tests/runtime_release_invariants.rs import ~24 `symthaea_wisdom::Evidence*/Trust*/Archive*/Action*` symbols that don't exist; fabrication-kernel's tests/ reference `OperatorCommandTracker`/`GatewayConsensusTracker`/`IncidentLedger`). The undeclared-dependency half is also shared by Test Sub-Crates (symthaea-orbital) (missing `orbital_mechanics`) and Test Sub-Crates (symthaea-auv) (missing `positioning`).",
        "fix_sketch": "Two independent fixes. (1) Undeclared deps: `hound` is an OPTIONAL dep behind feature `hound` (default = []), and examples/f1_probe.rs uses it unconditionally — add an `[[example]] name = "f1_probe" required-features = ["hound"]` block. `postcard` and `blake3` are not in Cargo.toml at all (no [dev-dependencies] section exists) — add them or delete their consumers. (2) The Checkpoint API: grep confirms `CheckpointOperationalTrustMetrics`, `CheckpointKeyring` etc. are defined NOWHERE in the entire symthaea workspace — they appear only inside these example files. All 15 examples import from `symthaea_vocal_tract::`, but lib.rs only re-exports controller/encoder/fep/metrics/pipeline/types. Either delete the orphaned checkpoint examples (mechanically trivial, but a product decision) or author the missing subsystem (large).",
        "notes": "Zero tests executed in this job — `cargo test -p symthaea-vocal-tract` compiles examples/ before running anything, so example-compile failure aborts the whole crate's test coverage (log has 0 `test result:` lines). The crate lib itself compiled fine. Provenance: the checkpoint examples arrived via commit 12ff3e5c88 (2026-07-20) "feat(vocal-tract): apply symthaea vocal tract patch series" — matching the known monorepo-wide patch-tooling defect where patch sets land files whose supporting implementation was never authored. Only 4 targets are reported failing because cargo aborted early ("build failed, waiting for other jobs to finish"); realistically most of the 15 examples are broken. NOTE ON METHOD: the prescribed `sed 's/\\x1b\\[[0-9;]*m//g'` does NOT work on these logs — the escapes are LITERAL `^` + `[` two-character sequences (verified via od -c), not real ESC bytes, so that sed is a silent no-op and grep for "error:" returns nothing. Correct strip: `perl -pe 's/(?:\\e|\\^\\[)\\[[0-9;]*[A-Za-z]//g'`."
      },
      {
        "job": "Test Sub-Crates (symthaea-wisdom-adjacent placeholder)",
        "root_cause": "NOT ONE OF MY ASSIGNED JOBS — ignore this entry; included only in error. See the four real entries.",
        "classification": "unclear",
        "confidence": "low",
        "fix_difficulty": "unknown",
        "shared_with": "none",
        "notes": "Disregard."
      },
      {
        "job": "Test Sub-Crates (symthaea-broca)",
        "root_cause": "`error[E0609]: no field `enable_hard_mask` on type `GatingConfig` --> crates/domains/symthaea-broca/examples/epistemic_gate_strength.rs:161:14` — `cfg_hard.enable_hard_mask = true;`, with rustc noting available fields are `unknown_factual_penalty`, `unknown_hedging_boost`, `uncertain_factual_penalty`, `uncertain_hedging_boost`, `coherence_drift_threshold` "... and 33 others".",
        "classification": "stale-test",
        "confidence": "high",
        "fix_difficulty": "trivial",
        "shared_with": "Pattern-level (not identical symbol) with Test Sub-Crates (symthaea-vocal-tract), (symthaea-wisdom), (symthaea-fabrication-kernel): a non-lib target authored against an API that does not exist in the crate, blocking the whole job. The specific missing field is unique to broca.",
        "fix_sketch": "Delete the `cfg_hard` / hard-mask comparison arm from examples/epistemic_gate_strength.rs (lines ~161 and its downstream uses). Do NOT add the field: CLAUDE.md documents by design that broca's epistemic gating is "a strong probabilistic deterrent, not a hard/physical block ... verified no -inf/hard-mask suppression in gating.rs" — so the arm is measuring a capability the project deliberately does not implement.",
        "notes": "`git log -S enable_hard_mask -- symthaea/crates/domains/symthaea-broca/src` returns EMPTY — the field never existed in src/ at any point in history. It appears in exactly one place repo-wide: this example, introduced by commit b2052043d9 (2026-07-08, "feat(broca): epistemic-gate strength experiment"). So this is not an API removal that stranded a test; the example was committed broken. It went unnoticed because the workflow file itself was structurally invalid (per MASTER_ROADMAP P0-#2, the `replace()` bug meant zero jobs could run) until the 2026-07-27 repair. Zero tests executed in this job — the log contains no `test result:` line at all; a single broken example target blocks the entire crate's suite. Verified present at the exact commit under test (export of monorepo 6a2fdb112e)."
      },
      {
        "job": "Test Sub-Crates (symthaea-neuromodulators)",
        "root_cause": "`thread 'tests::test_tachyphylaxis_triggers_after_20_cycles' panicked at crates/domains/symthaea-neuromodulators/src/lib.rs:3499:9: assertion `left == right` failed  left: 0  right: 20` (`assert_eq!(t.high_exposure_cycles, 20)`). Two sibling failures share the cause: `src/lib.rs:3530:9: "Should enter withdrawal"` and `src/lib.rs:3613:9: "DA should be tolerant after sustained high"`. Result: 309 passed; 3 failed.",
        "classification": "stale-test",
        "confidence": "high",
        "fix_difficulty": "small",
        "shared_with": "none — this is the only job in the run failing on tachyphylaxis/neuromodulator dynamics. (Test Sub-Crates (symthaea-manipulator) and (symthaea-spore) are also runtime test failures but in unrelated suites: `--test consciousness_proofs` and `--lib` respectively.)",
        "fix_sketch": "The tests pin the transmitter level high via `reuptake_rate: 0.0`, but `Transmitter::reuptake()` (src/transmitter.rs:169) no longer reads `reuptake_rate` at all — clearance is now Michaelis-Menten: `clearance = mm_v_max * |delta| / (mm_k_m + |delta|)` with defaults mm_v_max=0.15, mm_k_m=0.4. So the level decays anyway. Arithmetic confirms the exact observed value: from level=0.8/baseline=0.5, high_thresh = baseline + tolerance_threshold = 0.7; cycle 1 → level 0.7357 (>0.7, counter=1); cycle 2 → level 0.6801 (<0.7, else-branch resets counter to 0); it never recovers, so after 20 cycles high_exposure_cycles == 0. Fix the tests to pin level under the current model — set `mm_v_max: 0.0` in the struct literal, or re-`produce()` each cycle to hold level above threshold.",
        "notes": "Judgement call between stale-test and real-bug, and there IS a genuine defect adjacent to it worth flagging separately: `reuptake_rate` is now dead in the clearance path while the doc comment above it still claims "At low deviation: linear (backward-compatible with old reuptake_rate)" — that claim is false, the field is never read. The per-transmitter values still set in src/lib.rs (0.08, 0.06, 0.08, ...) and `adapt()`/`reset()`/`reuptake_rate_for_test()` are all now inert config-drift. I classified stale-test because the tachyphylaxis mechanism itself is intact (it fires correctly when the level is genuinely held high); only the tests' lever for holding it high was silently removed. Secondary observation: the neighbouring negative-control test `assert_eq!(t.high_exposure_cycles, 0, "Should not accumulate below threshold")` now passes vacuously — the counter never accumulates for ANY input — i.e. a degenerate probe that would not catch a real regression. Verified the implementation at the exact commit under test."
      },
      {
        "job": "Test Sub-Crates (symthaea-psych-bench)",
        "root_cause": "`thread 'regression_against_baseline' panicked at crates/domains/symthaea-psych-bench/tests/full_battery.rs:532:5: Critical performance regression detected (24 critical)!` — preceded in stdout by the harness's own `WARNING: Baseline "baseline" is 131 days old (generated 2026-03-20T22:20:33Z). Consider regenerating.` Report: 634 metrics total, 607 pass, 3 warning, 24 critical. Everything else in the crate is green (1361 lib tests pass).",
        "classification": "stale-test",
        "confidence": "medium",
        "fix_difficulty": "medium",
        "shared_with": "none — no other job in the run compares against a committed benchmark baseline.",
        "fix_sketch": "Do NOT simply regenerate baselines/v0.9.0.json — that would silently bless a possible real capability drop. Two defects to fix first. (1) The gate is DIRECTION-BLIND: RegressionSnapshot::compare (src/harness/snapshot.rs:227-268) flags Critical whenever `current.mean < baseline.ci_lower`, with no higher_is_better metadata — so genuine IMPROVEMENTS are reported as critical regressions (WCST/total_errors 19.5→16.7, WCST/non_perseverative_errors 9.7→8.1, TowerOfLondon/avg_excess_moves 1.41→0.44, TowerOfLondon/rt_ticks 12.64→11.11 are all better-is-lower metrics). Add per-metric direction. (2) The baseline is SATURATED with zero-width CIs: I dumped the committed blob and many ARC metrics are mean=1.0000 with ci=[1.0000,1.0000], so any value below exactly 1.0 trips Critical by construction. Guard against zero-width CIs. Then re-baseline, and separately investigate ArcChain.",
        "notes": "Baseline provenance: the test loads baselines/v0.9.0.json (full_battery.rs:361). That file is tracked in git but ABSENT from the local working tree, so locally the test takes its `if !baseline_path.exists() { return; }` early-skip path and silently passes — it only actually runs on a clean CI checkout. That divergence is itself worth fixing. The baseline is transparently degenerate: ArcNoise/accuracy_50pct = 1.0000 (perfect accuracy at 50% noise), noise_resilience = 1.0000, accuracy_drop = 0.0000; ArcFewShot 1shot..5shot all 1.0000 with fewshot_gain = 0.0 and learning_rate = 0.0; ArcScaling 1.0000 at every dimension 128d-1024d with dimension_efficiency_slope = 0.0 — i.e. dimension has no effect. These are ceiling artifacts, and the current values (ArcNoise 0.8187/0.8375/0.8625/0.8750/0.8000, degrading with noise) are more physically plausible than the baseline. This is exactly the 'suspiciously tight CI is a red flag' pattern already recorded in this project's memory. CAVEAT — one finding should NOT be dismissed as baseline drift: Reasoning::ArcChain/chain_3_accuracy 1.0000 → 0.0167 (-98.3%) and chain_4_accuracy 1.0000 → 0.1500, i.e. near-total failure in absolute terms, and chain_3 scoring BELOW chain_4 is internally odd. That needs a human look before anyone regenerates the snapshot. Confidence is medium (not high) precisely because the stale-baseline mechanism and a possible real ARC regression are entangled here."
      },
      {
        "job": "Test Sub-Crates (symthaea-wisdom)",
        "root_cause": "error[E0432]: unresolved imports `symthaea_wisdom::ActionExecutionCoordinator`, `symthaea_wisdom::EvidenceLedger`, `symthaea_wisdom::EvidenceSigner`, `symthaea_wisdom::TrustRegistry`, ... (24 symbols) --> crates/domains/symthaea-wisdom/tests/archive_startup_invariants.rs:7:5, with rustc noting `no `ActorId` in the root` / `no `ActionExecutor` in the root`. Second target: same E0432 for 41 symbols at crates/domains/symthaea-wisdom/tests/runtime_release_invariants.rs:7:5. UNDERLYING CAUSE: 24 of the 28 files in crates/domains/symthaea-wisdom/src/ are literal 15-byte `// placeholder` stubs (evidence.rs, execution.rs, runtime.rs, release.rs, deployment.rs, archive*.rs, authority_*.rs, postgres_sync.rs, ...), and src/lib.rs declares only `pub mod autopoiesis; pub mod harmonics; pub mod meta_cognition;`.",
        "classification": "missing-symbol",
        "confidence": "high",
        "fix_difficulty": "large",
        "fix_sketch": "The crate lib compiles fine — only integration tests fail. Real fix: implement (or recover from the crate's own PATCH_SERIES.md / SERIES_VI_INTEGRATION.md / SERIES_VIII_INTEGRATION.md patch sets) the 24 placeholder modules and declare/re-export them from lib.rs. Precedent for recovery exists: the Symtropy pass recovered 16 documents/5,854 lines from `docs/ops/..._Patch_Sets_...` git format-patches after initially declaring the content unrecoverable — check for an analogous patch directory before writing anything from scratch. Trivial stopgap to get CI green instead: quarantine the 5 test files that reference the unbuilt subsystem (archive_startup_invariants, runtime_release_invariants, authority_checkpoint_invariants, postgres_sync_live, production_deployment_invariants) behind a non-default feature or #[ignore], but that hides a genuinely unimplemented crate rather than fixing it.",
        "shared_with": "Test Sub-Crates (symthaea-vocal-tract) and Test Sub-Crates (symthaea-fabrication-kernel) fail from the SAME placeholder-integration defect: vocal-tract has 13 placeholder files of 38 in src/ and fails with the identical E0432 shape (`unresolved imports symthaea_vocal_tract::CheckpointOperationalTrustMetrics, ...`); fabrication-kernel has 1 placeholder of 128 and fails E0433 `cannot find type OperatorCommandTracker` / `GatewayConsensusTracker`. Not shared with the other 3 jobs in my assignment.",
        "notes": "CI reported only 2 failing test targets, but 5 of the 7 test files reference the missing subsystem — cargo cancels pending units after the first errors, so the true blast radius is larger than the log shows. This is the same repo-wide `// placeholder` patch-tooling defect already documented in MASTER_ROADMAP (Symtropy: '57 files repo-wide are literally // placeholder'; symthaea-humanoid P0-#6: contact.rs/hierarchical.rs were one-line placeholders with live consumers). NOTE a monorepo/standalone divergence I hit while checking: my local branch review/symthaea-bridges-security has only 4 files in that src/ dir (the 24 placeholders are absent entirely), while the exported/tested tree has all 28. Diagnosis above is stated against the tested tree, not local."
      },
      {
        "job": "Test Sub-Crates (symthaea-auv)",
        "root_cause": "error[E0432]: unresolved import `positioning` --> crates/domains/symthaea-auv/src/navigation_bridge.rs:9:5 — `use of unresolved module or unlinked crate `positioning``. Three more at src/navigation_estimator.rs:86:9 (`pub use positioning::Measurement;`), src/training.rs:15:5, plus error[E0433] at src/navigation_estimator.rs:46:20. This is a LIB failure, not a test failure.",
        "classification": "config-drift",
        "confidence": "high",
        "fix_difficulty": "small",
        "fix_sketch": "Root cause verified directly: the standalone export comments the dependency out. Monorepo Cargo.toml has `positioning = { path = "../../../../mycelix-workspace/mycelix-position/lib/positioning" }` (a path that escapes the symthaea repo into the monorepo), and the exported manifest at the tested commit reads `# [standalone-stripped] positioning = { path = ... }` while src/ still uses it unconditionally. So the crate is structurally unbuildable in the public repo though it builds fine in the monorepo. Fix options: (a) make `positioning` an optional dep and cfg-gate the three usages so the crate degrades gracefully when stripped; (b) publish/vendor `positioning` so the standalone can depend on it by version; (c) teach export-to-standalone.sh to also exclude crates whose sources depend on a stripped dep, rather than emitting a manifest it knows cannot compile. (a) or (c) is the durable fix — silently stripping a dep that the source requires guarantees a red job.",
        "shared_with": "Test Sub-Crates (symthaea-orbital) fails from the identical mechanism — error[E0433] `cannot find module or crate `orbital_mechanics``, from the stripped cross-repo dep `orbital-mechanics = { path = "../../../../mycelix-workspace/crates/orbital-mechanics" }`. Not shared with the other 3 jobs in my assignment.",
        "notes": "14+ crates in crates/domains/ carry out-of-repo `../../../../` path deps (broca, cell-foundry, exoskeleton, helicopter, multirotor, vehicle, quadruped, engineering, sensors, ...). Most of those PASSED, so stripping alone is not sufficient to break a crate — it only breaks when the source uses the stripped crate unconditionally rather than behind an optional feature. That makes this a latent trap: any future unconditional `use` of a stripped dep in those crates turns their CI leg red with no local signal, since the monorepo build never exercises the stripped manifest."
      },
      {
        "job": "Test Sub-Crates (symthaea-evidence-plane)",
        "root_cause": "error: package ID specification `symthaea-evidence-plane` did not match any packages — emitted by `cargo test -p symthaea-evidence-plane` immediately after dependency download, before any compilation. There is no rustc error at all; the package simply is not in the workspace being tested.",
        "classification": "config-drift",
        "confidence": "high",
        "fix_difficulty": "trivial",
        "fix_sketch": "The crate does not exist anywhere in the tested tree (verified: 404 for both crates/core/symthaea-evidence-plane and crates/domains/symthaea-evidence-plane at the tested commit), yet .github/workflows/ci.yml line 1255 lists it in the hardcoded sub-crate matrix. Timeline confirms the inconsistency: the crate was added to the monorepo in commit 1f242dbd55, which is NOT an ancestor of the export source commit 6a2fdb112e (the monorepo's own ci.yml at 6a2fdb112e correctly does not list it) — so the branch's ci.yml is newer than the exported crate tree. Fix: either re-export from a commit that includes the crate, or drop the matrix entry until it lands. Durable fix: generate the sub-crate matrix from `cargo metadata --no-deps` instead of hand-maintaining a hardcoded list, which makes this whole failure class impossible.",
        "shared_with": "Test Sub-Crates (symthaea-perception) fails with the identical error string (`package ID specification symthaea-perception did not match any packages`), but via a different sub-mechanism worth distinguishing: that crate DOES exist in the exported tree and is deliberately in the workspace `exclude` list (commented 'ARCHIVED 2026-07-16 ... Excluded so the workspace stops compiling a dead island') while still being listed in the CI matrix. Same class (matrix names a package the workspace does not contain), same trivial fix. Not shared with the other 3 jobs in my assignment.",
        "notes": "Both of these are pure CI-matrix bookkeeping, not code defects — no source is broken. They are exactly the kind of debt MASTER_ROADMAP P0-#2 predicted would surface now that the workflow file validates and jobs actually run for the first time. Note the tested head commit b69c7ea is not the export commit itself but a later fix commit on branch export/6a2fdb112e-20260727-180925, so the branch has drifted from its own export snapshot; I could not pin exactly which commit introduced the matrix entry on the standalone side."
      },
      {
        "job": "Test Sub-Crates (symthaea-manipulator)",
        "root_cause": "thread 'proof1_consciousness_cascade_curve' panicked at crates/domains/symthaea-manipulator/tests/consciousness_proofs.rs:133:5: "Cognitive effort must be non-increasing across tiers: Green 0.004204355 >= Yellow 0.0872845 >= Orange 0.08369347". This is a runtime assertion failure, not a compile error — the crate builds and 109 lib tests pass; only the consciousness_proofs integration target fails (2 passed, 1 failed).",
        "classification": "stale-test",
        "confidence": "medium",
        "fix_difficulty": "small",
        "fix_sketch": "The assertion's own comment states its premise: 'Green/Yellow/Orange all pass the same controller output through a shrinking gain, so mean effort must be non-increasing.' That open-loop premise is contradicted by the crate's current design. src/admittance.rs:39-41 sets compliance_gain Green 0.2 < Yellow 0.45 < Orange 0.75, and admittance.rs's OWN unit tests assert exactly that increasing order plus 'Orange (less confident) should yield more than Green' — i.e. the crate deliberately makes lower-confidence tiers more compliant, which adds tier-scaled effort in the opposite direction. Only the Green>=Yellow step fails (Yellow>=Orange still holds); Green's 0.0042 is ~20x BELOW Yellow, consistent with a high-authority closed loop settling to its target quickly while degraded tiers keep commanding against persistent error. Fix: restate the invariant against the commanded/cognitive component alone rather than total control_effort (separating the admittance contribution), or assert monotonic motor_gain directly instead of inferring it from integrated effort.",
        "shared_with": "none — the other three runtime-failure jobs in this run (symthaea-psych-bench --test full_battery, symthaea-neuromodulators --lib, symthaea-spore --lib) also fail at runtime rather than compile, but I did not pull their assertion messages, so I cannot claim a shared cause and am not asserting one.",
        "notes": "NOT flaky: the sweep is fully deterministic (fixed phi list, fixed seed 42, fixed step count), so this will reproduce identically on re-run — do not retry-to-green. I did NOT reproduce locally: host load was 37.5 on 12 cores with 16 concurrent Claude sessions, and building this crate pulls symthaea-core/symthaea-fep, so per the project's concurrency rules I did not launch it — hence medium rather than high confidence. The residual doubt is real: I cannot fully exclude a genuine regression that drove Green's effective motor gain toward zero, which would make this real-bug rather than stale-test. The decisive check is cheap once load allows — run the test and read its printed per-phi table (it prints Motor Gain per tier); if Green's motor_gain prints ~1.0 the test premise is simply stale, if it prints near 0 there is a real safety-cascade bug. Timeline context: admittance landed 2026-07-07 (03370ffe1a), BEFORE the test's last edit 2026-07-10 (298d2419be, which already carved Red out of this same invariant), so this most likely broke later and stayed invisible while CI was structurally unable to run (MASTER_ROADMAP P0-#2)."
      },
      {
        "job": "Test Sub-Crates (symthaea-orbital)",
        "root_cause": "error[E0433]: cannot find module or crate `orbital_mechanics` in this scope --> crates/domains/symthaea-orbital/src/scenarios.rs:11:5 (13 errors total, also simulator.rs:2-4, trajectory_planning.rs:29-30/94-95, types.rs:1; rustc help: "if you wanted to use a crate named `orbital_mechanics`, use `cargo add orbital_mechanics`")",
        "classification": "config-drift",
        "confidence": "high",
        "fix_difficulty": "medium",
        "fix_sketch": "symthaea-orbital/Cargo.toml declares `orbital-mechanics = { path = "../../../../mycelix-workspace/crates/orbital-mechanics" }`, a path dep that escapes the symthaea/ tree. scripts/export-to-standalone.sh's strip_external_path_deps() (~line 477) comments that line out on export, but src/ still `use`s orbital_mechanics unconditionally, so the lib cannot compile in the standalone repo. Three options: (a) publish/vendor orbital-mechanics so the standalone can depend on it from crates.io, (b) put every orbital_mechanics-using module behind an off-by-default cargo feature so the stripped standalone build still compiles, or (c) drop symthaea-orbital (and symthaea-auv) from the Test Sub-Crates matrix and document them as monorepo-only. Not a trivial fix -- it needs a real decision about the export pipeline.",
        "shared_with": "Test Sub-Crates (symthaea-auv) -- identical mechanism: `positioning = { path = "../../../../mycelix-workspace/mycelix-position/lib/positioning" }` is stripped the same way, producing `error[E0432]: unresolved import `positioning`` / `error[E0433]: cannot find module or crate `positioning``. One export-pipeline decision fixes both.",
        "notes": "Verified against the local monorepo: the dep line exists in symthaea/crates/domains/symthaea-orbital/Cargo.toml with an explicit comment about the nalgebra 0.32-vs-0.34 boundary, and export-to-standalone.sh has a resolution-based (not dot-counting) strip pass for exactly this shape of path. This is drift introduced when the Sol Atlas orbital work wired the real astrodynamics crate in -- CLAUDE.md still describes the two as "unconnected". Not a stale test and not a lint: the LIBRARY fails to compile."
      },
      {
        "job": "Test Sub-Crates (symthaea-fabrication-kernel)",
        "root_cause": "error[E0433]: cannot find type `OperatorCommandTracker` in this scope --> crates/domains/symthaea-fabrication-kernel/tests/durable_gateway_pipeline.rs:223:9 ("use of undeclared type"), plus GatewayConsensusTracker at :224:9 and IncidentLedger at :225:9; second failing target: error[E0432]: unresolved import `symthaea_fabrication_kernel::run_standard_fault_matrix` --> tests/operational_release_pipeline.rs:14:52 ("no `run_standard_fault_matrix` in the root")",
        "classification": "stale-test",
        "confidence": "high",
        "fix_difficulty": "trivial",
        "fix_sketch": "Two independent test-side fixes; the lib itself compiles clean. (1) tests/durable_gateway_pipeline.rs passes three new args to FabricationGatewayState::genesis(...) but has no `use` for those types at all (verified: grep count 0 in that file). All three ARE exported at crate root -- src/lib.rs:376 `pub use operator_command_tracker::{AppliedOperatorCommand, OperatorCommandTracker, ...}`, plus lib.rs:286 and :335 for the other two -- so just add the three `use symthaea_fabrication_kernel::{...}` lines (the sibling test tests/federated_release_pipeline.rs:8-13 already imports them correctly and can be copied). (2) run_standard_fault_matrix exists as `pub fn` at src/fault_injection.rs:134 under `pub mod fault_injection` (lib.rs:35) but is never re-exported at the crate root -- either add it to a root `pub use fault_injection::{...}` or change the test to import `symthaea_fabrication_kernel::fault_injection::run_standard_fault_matrix`.",
        "shared_with": "Test Sub-Crates (symthaea-wisdom) is the same CLASS (integration tests importing names absent from the crate root, E0432) but almost certainly a different root cause -- wisdom's unresolved list is ~24 and ~41 symbols across two test targets, which looks like a whole module tree missing/cfg-gated rather than a couple of forgotten re-exports. Do not assume one fix covers both. Test Sub-Crates (symthaea-vocal-tract) also shows a partial instance of the same class alongside a separate missing `hound` crate.",
        "notes": "Note the two errors are different flavours despite both being test-side: E0433 "use of undeclared type" means the test never imported the name (test-local omission), while E0432 "no X in the root" means the crate genuinely does not re-export it. Both are trivial but need different edits. Symbols verified present in local source at src/operator_command_tracker.rs:46, src/fault_injection.rs:134."
      },
      {
        "job": "Test Sub-Crates (symthaea-spore)",
        "root_cause": "thread 'broca::tests::test_generation_produces_text' panicked at crates/domains/symthaea-spore/src/broca.rs:2577:9: assertion failed: result.num_tokens > 0  (test result: FAILED. 272 passed; 1 failed)",
        "classification": "stale-test",
        "confidence": "high",
        "fix_difficulty": "trivial",
        "fix_sketch": "Traced to commit 18c4434f79 ("feat: BrocaLite always-on language..."), which changed BrocaLite::generate()'s no-checkpoint routing threshold from 0.5 to 1.1 (broca.rs:2125, `let threshold = if self.checkpoint_loaded { 0.6 } else { 1.1 };`). The test builds ThoughtChannels with consciousness 0.7, which previously (0.7 >= 0.5) took generate_autoregressive() and got `num_tokens: generated_ids.len()` (broca.rs:2462); it now always takes generate_structured(), which hardcodes `num_tokens: 0` (broca.rs:2183) while still producing non-empty text -- which is why the preceding `assert!(!result.text.is_empty())` on line 2576 passes. Either (a) update the test to assert on text/word content rather than num_tokens (or force the autoregressive path), or (b) make generate_structured() report a real token count instead of hardcoding 0 -- (b) is arguably the better fix since a caller reading num_tokens today sees 0 for text that genuinely contains words.",
        "shared_with": "none",
        "notes": "Deterministic, NOT flaky: BrocaLite::new(42) sets checkpoint_loaded:false (broca.rs:1959), the test sets SamplingStrategy::Greedy, and nothing loads a checkpoint -- so with threshold 1.1 every consciousness value routes to the structured path on every machine. This test therefore also fails locally; it was committed broken rather than being CI-environment-specific. Defensible alternative classification is real-bug (num_tokens:0 on a path that emits real text is a genuinely wrong field value), but the routing change is clearly deliberate per its own code comment ("Without checkpoint: ALWAYS use structured -- random autoregressive is word soup"), so the test is what lags."
      },
      {
        "job": "Test Sub-Crates (symthaea-perception)",
        "root_cause": "error: package ID specification `symthaea-perception` did not match any packages (from `cargo test -p symthaea-perception`; no compile errors at all -- dependency resolution succeeded, the package simply is not in the workspace graph)",
        "classification": "config-drift",
        "confidence": "high",
        "fix_difficulty": "trivial",
        "fix_sketch": "crates/domains/symthaea-perception is deliberately listed in the root symthaea/Cargo.toml `exclude` array (added 2026-07-16, vision review P2.2, with the comment "ARCHIVED... zero consumers... Excluded so the workspace stops compiling a dead island"), but .github/workflows/ci.yml:1355 still lists `- symthaea-perception` in the Test Sub-Crates matrix. An excluded crate is never a valid `-p` target even though the directory is still present and `crates/domains/*` is a members glob. Fix: delete line 1355 from the matrix (correct, matches the archival decision), or un-exclude the crate if it is meant to be live again.",
        "shared_with": "Test Sub-Crates (symthaea-evidence-plane) fails with the byte-identical error string, but for a DIFFERENT underlying reason and needs a different fix: symthaea-evidence-plane IS a legitimate workspace member locally (crates/core/symthaea-evidence-plane, matched by the `crates/core/*` glob, and referenced at Cargo.toml:207) -- it landed 2026-07-28 (commit 1f242dbd55) and most likely has not reached the standalone export yet. Same symptom, do not apply the same fix.",
        "notes": "Confirmed by reading both sides locally: the exclude entry with its dated archival comment in symthaea/Cargo.toml, and the matrix entry in .github/workflows/ci.yml. Also worth noting the exclude block carries a warning that an explicit "." in `members` silently disables `exclude` for glob-matched members -- not the issue here (exclude is working, which is exactly why -p fails), but relevant if anyone tries to "fix" this by touching members."
      },
      {
        "job": "Embodiment Safety Composition",
        "root_cause": "scripts/check-embodiment-safety-composition.sh: "FAIL: symthaea-gravcraft (crates/domains/symthaea-gravcraft/src/embodiment.rs) / derives a safety tier from Phi but never references: safety_override moral_safety/apply_moral_gate / => a SafetyAgent override or an ethics verdict has no route to this platform's actuators. Compose it: max(phi_level, safety_override, moral_safety)" (exit 1). Not a compile error — this is a shell source-lint gate.",
        "classification": "config-drift",
        "confidence": "high",
        "fix_difficulty": "trivial",
        "shared_with": "Orphan Module Check — identical root cause (stale export branch: gate scripts copied forward, their source fixes not)",
        "fix_sketch": "The fix already exists in the monorepo: commit d49a18aaa9 "fix(gravcraft): route SafetyAgent override + moral verdict to the actuators; enforce in CI" (2026-07-29 12:22 +0200) added safety_override/moral_safety fields, apply_moral_gate(), set_safety_override()/clear_safety_override() to symthaea-gravcraft/src/embodiment.rs. It just never reached this PR branch. Correct fix: re-run symthaea/scripts/export-to-standalone.sh from a current monorepo commit (>= d7e8a5e20e) to produce a fresh full git-archive export, instead of hand-copying individual files onto the 2026-07-27 snapshot. Cheap stopgap: mirror that one file onto the branch.",
        "notes": "VERIFIED, not inferred. The run checked out branch export/6a2fdb112e-20260727-180925 @ b69c7ea365, whose tip commit message says "Mirrored from monorepo commit d7e8a5e20e" — but the tree is stale. I fetched the branch's copy of the file via the GitHub contents API: it contains exactly one relevant line, `self.current_safety = MotorSafetyLevel::from_phi(phi);`, with no safety_override/moral_safety anywhere. The monorepo version at d7e8a5e20e has 18 such references, and d49a18aaa9 IS an ancestor of d7e8a5e20e (git merge-base --is-ancestor confirms). The gate script itself only landed in the monorepo on 2026-07-29 10:54 (commit 5fb63c99de "test(symthaea): encode two audit findings as executable checks") and was manually copied onto this branch at 18:03 by commit b7791d0431, whose own message admits: "this was a manual incremental-copy gap, not a bug in export-to-standalone.sh itself". That commit added ONLY the two scripts (+112/-0 and +202/-0) and no source. So the audit *lint* was propagated while the *fix* was not. I ran `bash scripts/check-embodiment-safety-composition.sh` against the local monorepo tree (read-only, pure filesystem scan, no build): exit 0, "OK: all 16 Phi-gating embodiment platforms compose safety_override + moral gate." There is no real defect in monorepo source. Note the script's own header documents that it strips comments before grepping precisely because gravcraft's doc comment *describing* the missing gate defeated the first version of the lint."
      },
      {
        "job": "Orphan Module Check",
        "root_cause": "scripts/check-orphan-modules.sh: first failure line is "ORPHAN: crates/domains/symthaea-acoustics/src/acoustic_two_port.rs  (1 lines — no 'mod acoustic_two_port;' anywhere in crates/domains/symthaea-acoustics/src)", followed by 117 more ORPHAN lines plus "STRANDED SUBTREE: crates/domains/symthaea-music-theory/src/evidence_calibration.rs is an empty module root beside evidence_calibration/ (5699 lines unreachable)" (exit 1). Every one of the 118 orphans is a one-line `// placeholder` file. Not a compile error — shell source-lint gate.",
        "classification": "config-drift",
        "confidence": "high",
        "fix_difficulty": "trivial",
        "shared_with": "Embodiment Safety Composition — identical root cause (stale export branch: gate scripts copied forward, their source fixes not)",
        "fix_sketch": "The fix already exists in the monorepo: commit 53ea9e9a48 "chore: delete 148 orphaned `// placeholder` files, de-quarantine 9 crates" (2026-07-29 11:11 +0200) removed all of them and shrank QUARANTINE 28 -> 19. Same remedy as the sibling job: re-run symthaea/scripts/export-to-standalone.sh from a current monorepo commit (>= d7e8a5e20e) for a clean full git-archive export rather than continuing to hand-copy files onto the 2026-07-27 snapshot.",
        "notes": "VERIFIED, and the mismatch is unusually precise — the branch got the *policy* but not the *cleanup*. I downloaded the branch's scripts/check-orphan-modules.sh (202 lines) and its QUARANTINE array holds exactly 19 entries, i.e. the POST-de-quarantine list from 53ea9e9a48, with acoustics/canvas/coding-theory/hal/legal-reasoning/manipulator/statistics/therapeutic/wisdom already removed. But the branch's *tree* is pre-cleanup: `gh api contents/crates/domains/symthaea-hal/src/arming.rs?ref=b69c7ea365` returns literally `// placeholder`, while that path is deleted in the monorepo at d7e8a5e20e. So those 9 crates were un-shielded on a tree that still contains the very files they were shielded for — which is why the report is dominated by acoustics/canvas/coding-theory/hal/legal-reasoning/manipulator. 53ea9e9a48 IS an ancestor of d7e8a5e20e. I ran `bash scripts/check-orphan-modules.sh` against the local monorepo tree (read-only filesystem scan): exit 0, "OK: no orphan modules or broken examples in 202 crate(s) checked, 18 quarantined", zero ORPHAN lines. No real defect in monorepo source. Caveat worth passing on: a fresh full re-export will fix these two jobs but may surface other legs, because this branch has accumulated ~12 hand-applied CI-triage commits since 2026-07-27 (including one self-labeled "EXPERIMENT (to be reverted): serial test-threads for the mesh leg only") that a clean re-export would drop."
      },
      {
        "job": "WASM Compatibility (Spore)",
        "root_cause": "`WASM size: 665KB (681742 bytes)` → `ERROR: WASM binary exceeds 500KB budget` — the hard gate `if [ "$WASM_SIZE" -gt 512000 ]; then ... exit 1` at .github/workflows/ci.yml:1207 ("Size budget check" step). There is NO compile error anywhere in this job: `cargo build --release --target wasm32-unknown-unknown --features wasm -p symthaea-spore`, `wasm-opt`, and `wasm-bindgen` all completed successfully; the job dies purely on the artifact-size policy check.",
        "classification": "config-drift",
        "confidence": "high",
        "fix_difficulty": "small",
        "shared_with": "none — the other job I was assigned (Hardened Nix Regressions) fails for a completely unrelated Nix path-resolution reason. Both are, however, instances of the same meta-pattern: long-accumulated debt that only became visible now that the workflow file finally validates (MASTER_ROADMAP P0-#2).",
        "fix_sketch": "Two independent options. (a) Cheap/honest: raise the gate (e.g. 700000) or convert it to a warning, and update symthaea-spore/Cargo.toml's now-false description "WASM consciousness kernel (~500KB)". (b) Actually reduce size — there is real untapped headroom: the pipeline runs wasm-opt BEFORE wasm-bindgen (ci.yml:1186-1199 optimizes target/.../symthaea_spore.wasm into /tmp, then wasm-bindgen re-emits the measured symthaea_spore_bg.wasm which is never re-optimized). Conventional order is wasm-bindgen first, then `wasm-opt -Oz` on the *_bg.wasm. Also `-O2` (not `-Oz`/`-Os`) and workspace `[profile.release] opt-level = 3, lto = "thin"` (Cargo.toml:2289-2293) are speed-tuned, not size-tuned; a wasm-specific size profile (opt-level="z", lto="fat", panic="abort") is the standard lever. Third lever: the `wasm` feature pulls `dep:nixward` (crates/domains/symthaea-spore/Cargo.toml:24) — the ~45K LOC NixOS crate — into every browser build; making that its own opt-in feature is likely the single biggest win.",
        "notes": "Not a fresh regression and not flaky: the size is byte-identical (681742) across runs 30301729186 (Jul 27), 30380739822 (Jul 28) and 30496274683 — deterministic and stable, 33% over budget. This job has failed on all 12 most-recent ci.yml runs; it has apparently never been green. Blame: `symthaea-nix` (renamed `nixward` on 2026-07-27, commit 20ff17ffbc) was wired into spore's `wasm` feature on 2026-06-16 (commit 4a212afc76) — the most plausible point where the artifact outgrew the budget, though I did not build locally to confirm attribution. Classified config-drift rather than real-bug because nothing is functionally broken (the smoke/E2E steps were never reached, so their status is unknown) — the manifest/feature set simply drifted away from a budget written when the crate was smaller. If you prefer to treat unchecked binary bloat as a defect, 'real-bug' is a defensible reclassification; the evidence is the same either way."
      },
      {
        "job": "Hardened Nix Regressions",
        "root_cause": "`error: path '/nix/store/' is not in the Nix store`, raised while forcing `evalApi.drvPath` (nixpkgs lib/customisation.nix:415) from `nodes.api.systemd.services.eval-api.serviceConfig`. Origin: /srv/luminous-dynamics/symthaea/nix/tests/eval-api-security.nix line 26 `src = ../../../.;` (and line 31 `root = toString ../../../.`). From `<flakeroot>/nix/tests/`, three levels up escapes the flake root; inside the store copy `/nix/store/<hash>-source/nix/tests/../../..` resolves literally to `/nix/store/` — verbatim the string in the error. Fails on the FIRST of the step's three commands (`nix build .#checks.x86_64-linux.eval-api-security`, ci.yml:646); `bash -e` means eval-service-module and service-module-smoke never ran and their status is unknown.",
        "classification": "config-drift",
        "confidence": "high",
        "fix_difficulty": "medium",
        "shared_with": "none — the WASM (Spore) job fails on an artifact size budget, an unrelated cause.",
        "fix_sketch": "The file bakes in monorepo geometry that does not survive the standalone export. Two things must change together: (1) `src`/`root` must not escape the flake root — in the standalone repo the flake IS symthaea/, so this should be `./.` (or `../..`), not `../../../.`; (2) `sourceRoot = "source/symthaea"` (line 39) is equally invalid there — I confirmed via `gh api repos/Luminous-Dynamics/symthaea/contents` at b69c7ea that the standalone root contains flake.nix, nix/, crates/, src/, Cargo.toml directly, with NO `symthaea/` subdir and none of the `mycelix-identity`/`mycelix-finance`/`mycelix-workspace`/`mycelix-position` siblings the filter (lines 33-38) whitelists. So the check needs either a monorepo-vs-standalone-aware src derivation, or to be excluded from the exported flake's `checks` (flake.nix:1097) until it is. Only eval-api-security.nix has the escaping path — the other two checks in the step are untouched by this fix.",
        "notes": "Pure config/packaging drift, not a code defect and not an infra/env failure: Nix installed cleanly, KVM was enabled, flake inputs fetched fine (two `##[warning]Failed to restore/save` lines are GitHub's Actions cache service 400ing — cosmetic, unrelated). This is an evaluation-time error; nothing was ever built. The test dates to 2026-04-02 (commit cf6d6a1054) and this job has failed on all recent ci.yml runs — consistent with the check never having been exercised via `nix build .#checks...` in either repo, since the flake-root escape would bite in the monorepo too (a flake only copies symthaea/ into the store there as well). Worth flagging to whoever owns the export tooling: this is the same monorepo→standalone class of bug as the stale symthaea-nix-web crate and dead [[example]] entry fixed on 2026-07-27."
      },
      {
        "job": "Clippy",
        "root_cause": "error: method `next` can be confused for the standard trait method `std::iter::Iterator::next` --> crates/domains/symthaea-alife/src/agent_id.rs:42:5 (first of 25; job ends with "could not compile `symthaea-alife` (lib) due to 25 previous errors")",
        "classification": "lint-only",
        "confidence": "high",
        "fix_difficulty": "small",
        "fix_sketch": "Command is `cargo clippy -p symthaea --lib --bins --features "$CI_FEATURES" -- -D warnings` on toolchain 1.96.0. 24 of the 25 are mechanically auto-fixable with `cargo clippy --fix` on the same invocation: (a) 17x `manual_is_multiple_of` -- rewrite `x % 2 == 0` / `tick % self.config.shuffle_epoch_ticks == 0` as `x.is_multiple_of(2)` (ma001l.rs 8, ma001r.rs 7 -- wait, per-file split is ma001l.rs 8, ma001r.rs 7, organism.rs 5, ma001.rs 3, population.rs 1, agent_id.rs 1 across ALL lints); (b) 6x `doc_lazy_continuation` -- indent the wrapped doc-comment continuation lines by 3 spaces (organism.rs, population.rs:316, and others); (c) 1x `useless_conversion` at ma001.rs:659:58 -- drop the `.into_iter()` inside `partners.into_iter().zip(values.into_iter())`. The 1 remaining lint (`should_implement_trait` on `AgentIdAllocator::next`, agent_id.rs:42) needs a judgement call: either rename to `next_id()` (about 16 in-crate call sites in ma001.rs/ma001r.rs/population.rs/events.rs/encounter.rs/agent_id.rs plus one external consumer, crates/domains/symthaea-futures-symtropy/src/ecological.rs), or slap `#[allow(clippy::should_implement_trait)]` on the method. NOTE: the build aborted at the first failing crate, so only 7 workspace crates were ever linted (symthaea-core, symthaea-fep, symthaea-futures-core, symthaea-types, symthaea-vision-manifold, symthaea-earth-system, symthaea-alife). The main `symthaea` crate itself and the rest of its workspace dependency tree were never reached -- expect a further tail after this crate is cleared.",
        "notes": "DISTINCT LINTS: 4 lint kinds, 25 total occurrences, ALL in ONE crate (symthaea-alife). Breakdown: manual_is_multiple_of x17, doc_lazy_continuation x6, should_implement_trait x1, useless_conversion x1. PRIOR NOTE CORRECTED: the claim of "25+ real style lints in symthaea-vision-manifold alone" does NOT hold at this commit. symthaea-vision-manifold WAS compiled under this exact `-D warnings` run (log line 2424, `Checking symthaea-vision-manifold v0.1.0`) and produced zero diagnostics -- the only lint locations in the entire log are crates/domains/symthaea-alife/src/{ma001l.rs:8, ma001r.rs:7, organism.rs:5, ma001.rs:3, population.rs:1, agent_id.rs:1}. That crate's lints were fixed by monorepo commits 0b76c8cdc3 ("PR #31 Format Check + Clippy CI legs", touching camera.rs/checkpoint.rs/encoder.rs/manifold.rs) and 8f0412f170. The "25" figure coincidentally matches but is a different crate -- the note is stale, not merely misattributed. REAL DEFECT ASSESSMENT: none. All 25 are pure style/doc-formatting. The 17 `manual_is_multiple_of` hits are `x % 2 == 0` parity checks and modulo-tick schedules -- semantically identical to the suggested `is_multiple_of`, zero behavior change. `doc_lazy_continuation` is doc-comment indentation only. `useless_conversion` is a redundant `.into_iter()` inside a `.zip()`, no behavior change. `should_implement_trait` on `AgentIdAllocator::next` is a naming-ambiguity warning; the method is a correct monotonic id allocator that deliberately never returns Option, so implementing Iterator would be wrong -- an `#[allow]` is defensible here. ROOT CAUSE CLASS: toolchain-version-driven. rust-toolchain.toml pins 1.96.0 and CI installs 1.96.0; `manual_is_multiple_of` is a relatively recent warn-by-default clippy lint, so these fired when the pin advanced (CI pin history in MASTER_ROADMAP: 1.93 -> 1.95 -> 1.96), not because the alife source regressed. Log-reading gotcha for future runs: `gh run view --log-failed` on this repo emits ANSI escapes in CARET NOTATION (literal `^` `[` bytes, because CARGO_TERM_COLOR=always plus GitHub's own re-encoding), so the documented `sed 's/\\x1b\\[[0-9;]*m//g' `strips NOTHING and grep for "error:" returns zero hits. The working filter is `sed 's/\\^\\[\\[[0-9;]*m//g'`.",
        "shared_with": "none"
      },
      {
        "job": "Muse (tests, studio, wasm UI)",
        "root_cause": "error[E0433]: cannot find `teaching_corpus` in `symthaea_muse` --> crates/domains/symthaea-muse/src/bin/muse_studio.rs:1629:37 (`let corpus = symthaea_muse::teaching_corpus::corpus()`). 6 identical-class E0433s total: `teaching_corpus` at muse_studio.rs:1629, :2246, :2577, :2583 and `symbolic_import` at :2425, :2427. Failing step: `cargo test -p symthaea-muse --features studio --bin muse_studio`.",
        "classification": "missing-symbol",
        "confidence": "high",
        "fix_difficulty": "trivial",
        "shared_with": "Orphan Module Check (partially — it flags these same two files as ORPHAN, but also 116 unrelated orphans in symthaea-wisdom/symthaea-therapeutic/symthaea-muse-ui/symthaea-statistics, mostly 1-line placeholder stubs, so that job would still fail after this fix)",
        "fix_sketch": "Add the two missing module declarations to crates/domains/symthaea-muse/src/lib.rs, alphabetically among the existing decls:

  #[cfg(feature = "theory")]
  pub mod symbolic_import;
  ... (before `pub mod synth;`)
  #[cfg(feature = "theory")]
  pub mod teaching_corpus;
  ... (before `pub mod temporal_confirmatory;`)

The `theory` gate is correct because Cargo.toml has `studio = ["theory", ...]`. This exact 4-line patch already exists UNCOMMITTED in the local monorepo working tree (`git diff HEAD -- symthaea/crates/domains/symthaea-muse/src/lib.rs`) — it just needs committing and re-exporting. Verify with `cargo test -p symthaea-muse --features studio --bin muse_studio`.",
        "notes": "Verified against the exact tested SHA b69c7ea365 (branch export/6a2fdb112e-20260727-180925), not just locally: fetched lib.rs at that ref via the GitHub API — zero matches for `teaching_corpus`/`symbolic_import`. Both module FILES do exist at that ref (teaching_corpus.rs 9218 bytes / 274 lines, symbolic_import.rs 14274 bytes / 368 lines), so this is orphan-module wiring, not deleted code. All four referenced functions exist with matching signatures (teaching_corpus.rs:88 `pub fn corpus()`, :95 `pub fn audition_path()`, symbolic_import.rs:27 `pub fn parse_symbolic()`, :315 `pub fn analyze()`) — so nothing is stale and no API drift is involved. NOT export/sync drift: the monorepo's own HEAD lib.rs is equally missing the declarations (`git show HEAD:...` confirms), so the committed code is genuinely unbuildable for the `studio` bin. The earlier steps of this job all passed (508 default-feature lib tests ok, 732 theory lib tests ok), so this is the sole cause. The step's second command (`cargo check -p symthaea-muse --features studio --bins`) never ran — it may surface further issues once the first is fixed.",
        "line": 1629,
        "file": "crates/domains/symthaea-muse/src/lib.rs"
      },
      {
        "job": "Feature Interactions (safety-agents,ssm_language)",
        "root_cause": "error[E0432]: unresolved import `symthaea_causal_reasoning::counterfactual` --> src/knowledge/causal_reasoning_bridge.rs:22:32 ("could not find `counterfactual` in `symthaea_causal_reasoning`"; note: "the item is gated behind the `counterfactual` feature" at crates/core/symthaea-causal-reasoning/src/lib.rs:18). A SECOND, independent error follows in the same job: error[E0061]: this method takes 5 arguments but 6 arguments were supplied --> src/cognitive_loop/cycle.rs:409:56, method defined at src/cognitive_loop/safety_supervisor.rs:28.",
        "classification": "config-drift",
        "confidence": "high",
        "fix_difficulty": "small",
        "fix_sketch": "Two localized edits, both pure feature-gating.
(1) E0432: `src/knowledge/mod.rs:35` declares `pub mod causal_reasoning_bridge;` unconditionally, and `causal_reasoning_bridge.rs:22` imports `symthaea_causal_reasoning::counterfactual` unconditionally, but that module is `#[cfg(feature = "counterfactual")]` and only the symthaea-level `reasoning_engine` feature forwards it (Cargo.toml:825 `reasoning_engine = ["magi_loop", "symthaea-causal-reasoning/counterfactual"]`). Fix: gate the import plus the `to_identification_dag()` converter it feeds behind `#[cfg(feature = "reasoning_engine")]`, or gate the whole `pub mod causal_reasoning_bridge;` declaration in src/knowledge/mod.rs:35.
(2) E0061: `SafetySupervisor::assess` (safety_supervisor.rs:28) declares its 6th parameter as `#[cfg(feature = "sentinel")] collective_immune: Option<&CollectiveImmuneState>`, but the call site in cycle.rs:409 places the cfg INSIDE a block expression (`{ #[cfg(feature="sentinel")] { Some(&self.collective_immune_state) } #[cfg(not(...))] { None } }`), so the argument is always present. Fix: move `#[cfg(feature = "sentinel")]` onto the argument expression itself at the call site — the same crate already does this correctly at safety_supervisor.rs:44-48 when forwarding to `compute_enforcement`.",
        "notes": "Not a stale test and not lint-only — this is real library code that has never been compiled in this feature combination. Both bugs are masked by the default feature set: `default = ["default-mind"]` and `default-mind` (Cargo.toml:1001) enables BOTH `reasoning_engine` and `sentinel`. This CI leg is the only one that strips them, because the `ssm_language` case in .github/workflows/ci.yml:1133-1138 forces `cargo check -p symthaea --no-default-features --features "safety-agents,ssm_language"` (needed to avoid the broca_lite/ssm_language `compile_error!` guard). The counterfactual import was introduced by e75adccf08 (2026-07-10, "CausalDAG converter — AGW Phase 5.3 un-islanding"). Verified against local source at /srv/luminous-dynamics/symthaea; `safety-agents = []` and `sentinel = []` are independent, non-implying features (Cargo.toml:858, 714).",
        "shared_with": "none — checked. The sibling matrix leg `safety-agents,reasoning_engine` would still hit the E0061 (sentinel off) but is not in my assignment. `Genesis Mission Benchmarks (temporal_unified, safety-agents)` is NOT a sharer: it runs `cargo run --example ... --features "safety-agents,unstable-examples" --release` WITH default features (ci.yml:1438), so it has sentinel+reasoning_engine; its exit-101 has a different, unlogged cause (cargo output captured into a shell variable). `Clippy` shows neither symbol. `Feature Interactions (therapeutic,reasoning_engine)` fails for a completely different reason."
      },
      {
        "job": "Feature Interactions (therapeutic,reasoning_engine)",
        "root_cause": "error[E0432]: unresolved import `crate::model_registry` --> crates/domains/symthaea-therapeutic/src/uncertainty.rs:10:12 — "could not find `model_registry` in the crate root" (`use crate::model_registry::ModelExecutionReceipt;`). Fails in the dependency crate `symthaea-therapeutic`, before the `symthaea` lib is ever reached.",
        "classification": "missing-symbol",
        "confidence": "high",
        "fix_difficulty": "medium",
        "fix_sketch": "There is no `model_registry.rs` in crates/domains/symthaea-therapeutic/src/ (confirmed by directory listing). The crate's own lib.rs:13-43 carries an explicit doc block titled "THIS CRATE DOES NOT COMPILE (as of 2026-07-29)" stating that "neither the module nor the type has ever existed in any commit on any branch", since 30b5d9ab97 (2026-07-21, "integrate new domain patchsets") when uncertainty.rs arrived. It also explicitly forbids the cheap fix: "Do not 'fix' this by deleting the field" — `EstimateEnvelope::model_receipt` gates `uncertainty::AbstentionReason::ProvenanceRequired`, so removing it would trade a fail-closed clinical-safety abstention for an unsafe success. The sanctioned fix per that doc is to design a real `ModelExecutionReceipt` (model identity + immutable version/hash, execution timestamp, non-leaking input-classification references, inference mode, uncertainty/calibration data, runtime identity, integrity binding, permitting policy version) and add the `model_registry` module — described in-tree as "a separately authorized clinical-safety task, not a cleanup item". A mechanical stub would make CI green in minutes but is the thing the crate warns against; hence medium, not trivial.",
        "notes": "Known, documented, deliberate breakage rather than a fresh regression — but it IS a genuine hard compile failure, not a stale test or a lint. `therapeutic` is the only feature that pulls this crate in (Cargo.toml:785 `therapeutic = ["dep:symthaea-clinical", "dep:symthaea-therapeutic", "ssm_language", ...]`), which is why this is the only leg in the matrix that surfaces it. Note the CI case-statement matches the literal matrix string, so ",therapeutic,reasoning_engine," does not match the `*,ssm_language,*` arm even though `therapeutic` transitively enables `ssm_language`; this leg therefore runs the default `cargo check --features` branch. Verified against local source at /srv/luminous-dynamics/symthaea.",
        "shared_with": "none — no other failing job in this run compiles `symthaea-therapeutic`. Explicitly checked: `Clippy`'s failed log contains no reference to model_registry/therapeutic, and the `Test Sub-Crates` matrix does not include symthaea-therapeutic. In particular this does NOT share a cause with `Feature Interactions (safety-agents,ssm_language)`, whose two errors are feature-gating holes in the main crate."
      },
      {
        "job": "Genesis Mission Benchmarks (temporal_unified, safety-agents)",
        "root_cause": "Not visible in the log — the workflow step swallows it. `.github/workflows/ci.yml:1436` runs `output=$(cargo run --example benchmark_genesis_temporal_unified --features "safety-agents,unstable-examples" --release 2>&1)` under `shell: bash -e`, so the non-zero cargo exit kills the step before `echo "$output"` on the next line; the entire 432-line job log ends at `##[error]Process completed with exit code 101`. Root cause established independently by dependency resolution: `examples/benchmark_genesis_temporal_unified.rs:19-35` uses `symthaea_physics::` (7 sites) and `symthaea_nuclear_forensics::` (2 sites), but those are optional `[dependencies]` gated behind the `physics` / `nuclear-forensics` features (Cargo.toml:357, 422, 750, 861) and are NOT dev-dependencies. `Cargo.toml:2096-2097` declares `required-features = ["safety-agents", "unstable-examples"]` and `ci.yml:1412-1413` passes `features: safety-agents` — neither enables them. Verified with `cargo tree -p symthaea --features "safety-agents,unstable-examples" -e normal,dev`: graph contains `symthaea-cell-foundry` and `symthaea-materials` (unconditional dev-deps, Cargo.toml:1021,1033) but NOT `symthaea-physics` / `symthaea-nuclear-forensics`. Positive control `--features "physics,nuclear-forensics,unstable-examples"` pulls both in. The example is therefore structurally uncompilable under CI's feature set (E0433, use of undeclared crate or module).",
        "classification": "config-drift",
        "confidence": "high",
        "fix_difficulty": "trivial",
        "shared_with": "none — this is the only failing entry of the 8-job genesis-benchmarks matrix. The 7 siblings all pass, including `Genesis Mission Benchmarks (safety, safety-agents)` which uses the identical feature string, so neither the library nor the `safety-agents` feature is implicated. No other failing job in run 30496274683 shares this cause.",
        "fix_sketch": "Both the manifest and the CI matrix must change together. (1) Cargo.toml:2097 -> `required-features = ["safety-agents", "physics", "nuclear-forensics", "unstable-examples"]`; (2) ci.yml:1413 -> `features: safety-agents,physics,nuclear-forensics` for the `temporal_unified` matrix entry. Alternative single-change fix: promote `symthaea-physics` and `symthaea-nuclear-forensics` to unconditional `[dev-dependencies]` (Cargo.toml:1020-1040) exactly as `symthaea-cell-foundry` and `symthaea-materials` already are, which is why those two resolve today. IMPORTANT: fixing only `required-features` is a trap — cargo would then silently SKIP the example under `--features safety-agents`, printing nothing, and the `grep -q "PASS"` guard at ci.yml:1439 would fail with the misleading `Benchmark did not print PASS`. Separately worth fixing: rewrite the step as `cargo run ... 2>&1 | tee /tmp/out` (with `set -o pipefail`) or `echo "$output"` before the failure, so future failures in this job are diagnosable at all.",
        "notes": "Secondary latent risk, not the current cause: line 102 of the example asserts `ratio < 5.0 && ratio > 0.1` on a wall-clock timing ratio (long-horizon vs short-horizon predict_at, 500 iters). That is inherently flaky on shared GitHub runners and will likely become the next failure mode once the feature set is corrected and the example actually builds. Also note `unstable-examples` is self-documented at Cargo.toml:844 as a 'quarantine gate for examples with stale APIs (never enabled by default)', so this example lives in a bucket that is expected to rot — but the specific defect here is a feature/manifest mismatch, not a stale API. Timing evidence consistent with a late compile failure: step ran 00:52:55 -> 01:01:59 (9m04s), i.e. full release build of the lib, then the example compiles last and errors; exit 101 is cargo's build-failure code. READ ONLY — nothing was edited, committed, or re-run."
      }
    ]
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Diagnose"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Catalogue"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "subcrates-1",
      "phaseIndex": 1,
      "phaseTitle": "Diagnose",
      "agentId": "ab9bb6e645b9d2847",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785409999937,
      "queuedAt": 1785409999850,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Diagnose why these 4 CI jobs failed on run 30496274683 of Luminous-Dynamics/symthaea.

JOBS (exact names, use verbatim):
  Test Sub-Crates (symthaea-psych-bench)
  Test Sub-Crates (symthaea-neuromodulators)
  Test Sub-Crates (symthaea-broca)
  Test Sub-Crates (symthaea-vocal-tract)

For each: pull the failed-step log, find the FIRST real error, and classify it.
These are sub-crate test jobs, so th…",
      "lastProgressAt": 1785410637656,
      "tokens": 180973,
      "toolCalls": 51,
      "durationMs": 637719,
      "resultPreview": "{"jobs":[{"job":"Test Sub-Crates (symthaea-vocal-tract)","root_cause":"First error chronologically: `error[E0433]: cannot find module or crate `hound` --> crates/domains/symthaea-vocal-tract/examples/f1_probe.rs:32:26`. Dominant error (3 of 4 failing targets): `error[E0432]: unresolved imports `symthaea_vocal_tract::CheckpointOperationalTrustMetrics`, `...::CheckpointOperationalTrustRequirements`,…"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "subcrates-2",
      "phaseIndex": 1,
      "phaseTitle": "Diagnose",
      "agentId": "a5f676282138faa7f",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785409999943,
      "queuedAt": 1785409999850,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Diagnose why these 4 CI jobs failed on run 30496274683 of Luminous-Dynamics/symthaea.

JOBS (exact names, use verbatim):
  Test Sub-Crates (symthaea-wisdom)
  Test Sub-Crates (symthaea-auv)
  Test Sub-Crates (symthaea-evidence-plane)
  Test Sub-Crates (symthaea-manipulator)

For each: pull the failed-step log, find the FIRST real error, and classify it.
These are sub-crate test jobs, so the likely…",
      "lastProgressAt": 1785410618781,
      "tokens": 167188,
      "toolCalls": 42,
      "durationMs": 618838,
      "resultPreview": "{"jobs":[{"job":"Test Sub-Crates (symthaea-wisdom)","root_cause":"error[E0432]: unresolved imports `symthaea_wisdom::ActionExecutionCoordinator`, `symthaea_wisdom::EvidenceLedger`, `symthaea_wisdom::EvidenceSigner`, `symthaea_wisdom::TrustRegistry`, ... (24 symbols) --> crates/domains/symthaea-wisdom/tests/archive_startup_invariants.rs:7:5, with rustc noting `no `ActorId` in the root` / `no `Actio…"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "subcrates-3",
      "phaseIndex": 1,
      "phaseTitle": "Diagnose",
      "agentId": "a4b6b3cfed600992a",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785409999944,
      "queuedAt": 1785409999851,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Diagnose why these 4 CI jobs failed on run 30496274683 of Luminous-Dynamics/symthaea.

JOBS (exact names, use verbatim):
  Test Sub-Crates (symthaea-orbital)
  Test Sub-Crates (symthaea-fabrication-kernel)
  Test Sub-Crates (symthaea-spore)
  Test Sub-Crates (symthaea-perception)

For each: pull the failed-step log, find the FIRST real error, and classify it.
These are sub-crate test jobs, so the …",
      "lastProgressAt": 1785410378620,
      "tokens": 157183,
      "toolCalls": 31,
      "durationMs": 378675,
      "resultPreview": "{"jobs":[{"job":"Test Sub-Crates (symthaea-orbital)","root_cause":"error[E0433]: cannot find module or crate `orbital_mechanics` in this scope --> crates/domains/symthaea-orbital/src/scenarios.rs:11:5 (13 errors total, also simulator.rs:2-4, trajectory_planning.rs:29-30/94-95, types.rs:1; rustc help: \\"if you wanted to use a crate named `orbital_mechanics`, use `cargo add orbital_mechanics`\\")","c…"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "job-1",
      "phaseIndex": 1,
      "phaseTitle": "Diagnose",
      "agentId": "a420c76ad2f712dd7",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785409999946,
      "queuedAt": 1785409999852,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Diagnose why these CI job(s) failed on run 30496274683 of Luminous-Dynamics/symthaea.

JOBS (exact names, use verbatim):
  Embodiment Safety Composition
  Orphan Module Check






HOW TO GET A JOB'S FAILURE (follow exactly):

  JID=$(gh run view 30496274683 --repo Luminous-Dynamics/symthaea --json jobs \\
        -q '.jobs[] | select(.name=="<EXACT JOB NAME>") | .databaseId')
  gh run view --repo …",
      "lastProgressAt": 1785410401153,
      "tokens": 159293,
      "toolCalls": 35,
      "durationMs": 401207,
      "resultPreview": "{"jobs":[{"job":"Embodiment Safety Composition","root_cause":"scripts/check-embodiment-safety-composition.sh: \\"FAIL: symthaea-gravcraft (crates/domains/symthaea-gravcraft/src/embodiment.rs) / derives a safety tier from Phi but never references: safety_override moral_safety/apply_moral_gate / => a SafetyAgent override or an ethics verdict has no route to this platform's actuators. Compose it: max(…"
    },
    {
      "type": "workflow_agent",
      "index": 5,
      "label": "job-2",
      "phaseIndex": 1,
      "phaseTitle": "Diagnose",
      "agentId": "a7ec873cab6a627e2",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785409999947,
      "queuedAt": 1785409999852,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Diagnose why these CI job(s) failed on run 30496274683 of Luminous-Dynamics/symthaea.

JOBS (exact names, use verbatim):
  WASM Compatibility (Spore)
  Hardened Nix Regressions



NOTE: these may be infra/env failures (missing toolchain target, nix build) rather than code defects. Distinguish carefully.


HOW TO GET A JOB'S FAILURE (follow exactly):

  JID=$(gh run view 30496274683 --repo Luminous…",
      "lastProgressAt": 1785410393534,
      "tokens": 151316,
      "toolCalls": 24,
      "durationMs": 393587,
      "resultPreview": "{"jobs":[{"job":"WASM Compatibility (Spore)","root_cause":"`WASM size: 665KB (681742 bytes)` → `ERROR: WASM binary exceeds 500KB budget` — the hard gate `if [ \\"$WASM_SIZE\\" -gt 512000 ]; then ... exit 1` at .github/workflows/ci.yml:1207 (\\"Size budget check\\" step). There is NO compile error anywhere in this job: `cargo build --release --target wasm32-unknown-unknown --features wasm -p symthaea-s…"
    },
    {
      "type": "workflow_agent",
      "index": 6,
      "label": "job-3",
      "phaseIndex": 1,
      "phaseTitle": "Diagnose",
      "agentId": "ae5ad2c72ed297e9c",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785409999948,
      "queuedAt": 1785409999852,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Diagnose why these CI job(s) failed on run 30496274683 of Luminous-Dynamics/symthaea.

JOBS (exact names, use verbatim):
  Clippy

NOTE: Clippy runs with -D warnings. Report HOW MANY distinct lints and in which crates, and whether any indicate a real defect rather than style. A prior note claimed 25+ lints in symthaea-vision-manifold alone — verify or correct that.




HOW TO GET A JOB'S FAILURE (…",
      "lastProgressAt": 1785410242441,
      "tokens": 139812,
      "toolCalls": 23,
      "durationMs": 242492,
      "resultPreview": "{"jobs":[{"job":"Clippy","root_cause":"error: method `next` can be confused for the standard trait method `std::iter::Iterator::next` --> crates/domains/symthaea-alife/src/agent_id.rs:42:5 (first of 25; job ends with \\"could not compile `symthaea-alife` (lib) due to 25 previous errors\\")","classification":"lint-only","confidence":"high","fix_difficulty":"small","fix_sketch":"Command is `cargo clip…"
    },
    {
      "type": "workflow_agent",
      "index": 7,
      "label": "job-4",
      "phaseIndex": 1,
      "phaseTitle": "Diagnose",
      "agentId": "a39c4cb7784d63b86",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785409999950,
      "queuedAt": 1785409999853,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Diagnose why these CI job(s) failed on run 30496274683 of Luminous-Dynamics/symthaea.

JOBS (exact names, use verbatim):
  Muse (tests, studio, wasm UI)






HOW TO GET A JOB'S FAILURE (follow exactly):

  JID=$(gh run view 30496274683 --repo Luminous-Dynamics/symthaea --json jobs \\
        -q '.jobs[] | select(.name=="<EXACT JOB NAME>") | .databaseId')
  gh run view --repo Luminous-Dynamics/symt…",
      "lastProgressAt": 1785410215879,
      "tokens": 151971,
      "toolCalls": 23,
      "durationMs": 215929,
      "resultPreview": "{"jobs":[{"job":"Muse (tests, studio, wasm UI)","root_cause":"error[E0433]: cannot find `teaching_corpus` in `symthaea_muse` --> crates/domains/symthaea-muse/src/bin/muse_studio.rs:1629:37 (`let corpus = symthaea_muse::teaching_corpus::corpus()`). 6 identical-class E0433s total: `teaching_corpus` at muse_studio.rs:1629, :2246, :2577, :2583 and `symbolic_import` at :2425, :2427. Failing step: `carg…"
    },
    {
      "type": "workflow_agent",
      "index": 8,
      "label": "job-5",
      "phaseIndex": 1,
      "phaseTitle": "Diagnose",
      "agentId": "a3484f768b683a29a",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785409999951,
      "queuedAt": 1785409999853,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Diagnose why these CI job(s) failed on run 30496274683 of Luminous-Dynamics/symthaea.

JOBS (exact names, use verbatim):
  Feature Interactions (safety-agents,ssm_language)
  Feature Interactions (therapeutic,reasoning_engine)






HOW TO GET A JOB'S FAILURE (follow exactly):

  JID=$(gh run view 30496274683 --repo Luminous-Dynamics/symthaea --json jobs \\
        -q '.jobs[] | select(.name=="<EXA…",
      "lastProgressAt": 1785410356067,
      "tokens": 155715,
      "toolCalls": 34,
      "durationMs": 356116,
      "resultPreview": "{"jobs":[{"job":"Feature Interactions (safety-agents,ssm_language)","root_cause":"error[E0432]: unresolved import `symthaea_causal_reasoning::counterfactual` --> src/knowledge/causal_reasoning_bridge.rs:22:32 (\\"could not find `counterfactual` in `symthaea_causal_reasoning`\\"; note: \\"the item is gated behind the `counterfactual` feature\\" at crates/core/symthaea-causal-reasoning/src/lib.rs:18). A…"
    },
    {
      "type": "workflow_agent",
      "index": 9,
      "label": "job-6",
      "phaseIndex": 1,
      "phaseTitle": "Diagnose",
      "agentId": "a1aaeb53b215c8fca",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785409999952,
      "queuedAt": 1785409999853,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Diagnose why these CI job(s) failed on run 30496274683 of Luminous-Dynamics/symthaea.

JOBS (exact names, use verbatim):
  Genesis Mission Benchmarks (temporal_unified, safety-agents)






HOW TO GET A JOB'S FAILURE (follow exactly):

  JID=$(gh run view 30496274683 --repo Luminous-Dynamics/symthaea --json jobs \\
        -q '.jobs[] | select(.name=="<EXACT JOB NAME>") | .databaseId')
  gh run vie…",
      "lastProgressAt": 1785410303483,
      "tokens": 144894,
      "toolCalls": 25,
      "durationMs": 303531,
      "resultPreview": "{"jobs":[{"job":"Genesis Mission Benchmarks (temporal_unified, safety-agents)","root_cause":"Not visible in the log — the workflow step swallows it. `.github/workflows/ci.yml:1436` runs `output=$(cargo run --example benchmark_genesis_temporal_unified --features \\"safety-agents,unstable-examples\\" --release 2>&1)` under `shell: bash -e`, so the non-zero cargo exit kills the step before `echo \\"$out…"
    },
    {
      "type": "workflow_agent",
      "index": 10,
      "label": "catalogue",
      "phaseIndex": 2,
      "phaseTitle": "Catalogue",
      "agentId": "a97ba2a37f45b25bb",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785410637710,
      "queuedAt": 1785410637705,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "gh pr view 31 --repo Luminous-Dynamics/symthaea --json numb…",
      "promptPreview": "Produce the CI failure catalogue for run 30496274683 of Luminous-Dynamics/symthaea (PR #32,
branch export/6a2fdb112e-20260727-180925).

RUN TOTALS (verified): 100 success, 22 failure, 33 cancelled, 2 skipped.
Note: `Workspace Target Integrity` PASSES.

PER-JOB DIAGNOSES:
[
 {
  "job": "Test Sub-Crates (symthaea-vocal-tract)",
  "root_cause": "First error chronologically: `error[E0433]: cannot find…",
      "lastProgressAt": 1785411068979,
      "tokens": 161972,
      "toolCalls": 16,
      "durationMs": 431269,
      "resultPreview": "# CI Failure Catalogue — run [30496274683](https://github.com/Luminous-Dynamics/symthaea/actions/runs/30496274683)

**Repo:** Luminous-Dynamics/symthaea · **PR:** [#32](https://github.com/Luminous-Dynamics/symthaea/pull/32) (OPEN) · **Branch:** `export/6a2fdb112e-20260727-180925` @ `b69c7ea365`
**Totals:** 100 success · **22 failure** · 33 cancelled · 2 skipped (157 jobs)

## Record corrections (M…"
    }
  ],
  "totalTokens": 1570317,
  "totalToolCalls": 304


---

# Addendum — `ArcChain` root-caused (2026-07-30, same session)

The catalogue's D5 asked a human to look before regenerating the baseline. Done. **It is a real
defect, it reproduces locally and deterministically, and it is not baseline drift.**

## Reproduction

New guard `crates/domains/symthaea-psych-bench/tests/arc_chain_monotonicity.rs`, run locally:

```
chain_2=0.9333  chain_3=0.0167  chain_4=0.1500
test accuracy_is_non_increasing_in_chain_length ... FAILED
test no_condition_is_anti_correlated ............. FAILED
```

`chain_3 = 0.0167` reproduces the CI value **exactly** — 1/60, i.e. the benchmark picks the
distractor 59 times out of 60. Scoring is 2-AFC, so chance is 0.50 and this is not weak
performance: it is **systematic anti-correlation**. A defect that consistent cannot be capability
drift.

## Why nothing caught it

`test_degradation_with_length` (`arc_chain.rs`) asserts nothing about degradation or length:

```rust
let sim_2 = result.metrics["chain_2_similarity"].mean;
let sim_4 = result.metrics["chain_4_similarity"].mean;
assert!(sim_2.is_finite() && sim_4.is_finite());   // read, never compared
assert!(deg.is_finite(), "degradation should be finite");
```

It passes for any values, including a total inversion — a probe that cannot fail for the reason it
exists. Same shape as the vacuous negative control this catalogue found in `symthaea-neuromodulators`.
Two independent instances in one run makes this a pattern worth sweeping for.

## Mechanism (confirmed by direct test, not inferred)

`arc_dataset::fair_distractor_grid` is **chain-blind**. It returns the first of five fixed
candidates that differs from the true output:

```rust
let candidates = [reflect_x(input), reflect_y(input),
                  color_replace(input,0,1), color_replace(input,1,2), input.to_vec()];
candidates.into_iter().find(|g| g != true_output)
```

In practice that is almost always `reflect_x(input)` — asserted directly in
`distractor_is_the_chains_own_first_step`, which **passes**.

Three of the six chains in `get_chains` *begin* with `ReflectX` (indices 0, 2, 4 — one per length
group). For those, the 2-AFC "wrong answer" is **the chain's own first step**: a partially-correct
grid. An HDC prediction that under-applies the composed rule then sits closer to the partial
transform than to the true full output, so the benchmark scores it wrong for being
*insufficiently* wrong. The distractor is not fair, despite the function name.

## Honest limit on the attribution

This explains the *direction* and the below-chance floor, but **not yet the full group-level
pattern**. Each length group contains exactly one `ReflectX`-initial chain, so a single bad chain
per group should cap a group near 0.5 — yet group 0 scores 0.93 and group 1 scores 0.017. Both
chains in group 1 must be failing. Per-chain (not per-group) instrumentation is needed to finish
the attribution; the benchmark currently only aggregates by length.

## Recommendation

**Do not regenerate `baselines/v0.9.0.json`.** The baseline being saturated at `mean=1.0,
ci=[1.0,1.0]` is a second, separate defect, and regenerating would bake this one in as the new
"expected" behaviour.

The distractor fix changes benchmark semantics and therefore every recorded ArcChain score, so it
is **left as an explicit decision, not taken unilaterally**. The two failing guards are committed
deliberately: they encode the defect precisely and will go green when it is fixed. The psych-bench
CI job was already red on `regression_against_baseline`, so this adds no new red job.

---

# Addendum 2 — `ArcChain` RESOLVED, and Addendum 1's mechanism was wrong (2026-07-31)

Fixed in `0473e6f6cb`. **Addendum 1's stated mechanism is refuted.** It is left above unedited,
because the way it was wrong is the useful part.

## What Addendum 1 claimed

That the distractor collided with the chain's own first step, so the three chains beginning with
`ReflectX` were scored against a partially-correct answer. That the distractor *is*
`reflect_x(input)` was confirmed by a passing test — but confirming the distractor's **identity**
is not confirming it is the **cause**, and Addendum 1 slid from one to the other. Its own "honest
limit" section flagged that the group-level pattern was unexplained; that flag was the real signal
and should have blocked the mechanism claim rather than sitting beside it.

## What the data actually shows

Per-chain metrics — which did not exist, and whose absence is precisely why Addendum 1 could not
finish the diagnosis — were added and run:

| chain | transforms | before | after fix |
|---|---|---|---|
| c0 (len 2) | **ReflectX**, TranslateRight | **1.0000** | 1.0000 |
| c1 (len 2) | ColorReplace, ReflectY | 0.8667 | 1.0000 |
| c2 (len 3) | **ReflectX**, TranslateDown, ColorReplace | 0.0333 | 0.7000 |
| c3 (len 3) | TranslateRight, ReflectY, Rotate90 | 0.0000 | 0.5333 |
| c4 (len 4) | **ReflectX**, ColorReplace, TranslateRight, ReflectY | 0.2667 | 0.8667 |
| c5 (len 4) | TranslateDown, Rotate90, ColorReplace, ReflectX | 0.0333 | 0.5667 |

`c0` begins with `ReflectX` and scored a **perfect 1.0000** — the exact case the prefix theory
predicted would fail worst. The theory is dead. The real split is chain **length**.

## The actual mechanism: a distance confound

`apply_rule` is near-identity. Measured similarity of the prediction to the **input** is ~0.78
across every chain, while similarity to the **true output** is only ~0.55 for 3–4 step chains
(0.63 for 2-step). The distractor is a single-transform variation of the input, so its distance
from the input is **constant**, while the target's distance **grows with every chain step**.

A near-identity prediction therefore matches the input-adjacent distractor reliably — and more
reliably the longer the chain. That produces scores far *below* chance rather than at it, which is
why the failure looked like anti-knowledge rather than absent knowledge.

The fix makes the distractor the output of a **different chain of the same length**, so both
options are equidistant from the input by construction.

## Two things this does not do

**It does not make the benchmark score well — it makes it score honestly.** Several chains now sit
near 0.50, the correct reading of "rule composition does not generalise here." The old numbers
implied strong anti-knowledge, which was an artifact.

**It does not restore length-monotonicity, and that claim was withdrawn rather than rescued.** The
chains are not difficulty-matched: the two weakest are exactly the two containing `Rotate90`
regardless of length, and 4-step `c4` (0.867) outscores both 3-step chains. So the length groups
are confounded by transform composition, and the monotonicity guard added on 2026-07-30 was
**deleted as wrong to assert** — not silenced to get green. Measuring degradation-with-length
needs nested chains (chain_3 = chain_2 + 1 step); that would alter every recorded value and has
not been done.

## Still open

**`baselines/v0.9.0.json` is still saturated at `mean=1.0, ci=[1.0,1.0]`** — a second, independent
defect that this work did not touch. Regenerating it now would at least no longer bake in the
distractor artifact, but it would bake in the zero-width CIs.

**Six other ARC benchmarks still use `fair_distractor_grid`** (arc_strict, arc_noise,
arc_staircase, arc_scaling, arc_fewshot, arc_dataset). They were deliberately left alone —
changing the shared function would silently move all six. They carry the same confound wherever
their target sits more than one transform from the input, which is worth checking before trusting
any of their absolute numbers.
