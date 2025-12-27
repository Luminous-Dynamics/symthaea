# Symthaea v1.3: Deterministic Semantics, Safe Actions, and Path to Cognition

**Status**: Draft  
**Scope**: Align implementation with v1.2 intent, add deterministic HDC, safe ActionIR execution, and concrete milestones toward a cognitive system.

---

## Current State (v1.2 + recent changes)

- **HDC**: Bit-packed HV16 available; random per-run concepts still used in code; hash-based projection added but not wired end-to-end.
- **Action layer**: ActionIR with policy/sandbox validation; `SimpleExecutor` supports simulated mode (default) and an env-gated real mode (`SYMTHAEA_ALLOW_REAL_EXEC=1`).
- **Nix intents**: Install/search/remove/upgrade with `--profile` parsing and unsafe flags rejected; integration tests cover ActionIR validation + simulated execution.
- **Cognition**: LTC/consciousness graph untrained; semantics not grounded; memory persistence limited.
- **Safety**: Structured validation exists; no full rollback, no telemetry/audit.
- **Tests/CI**: Passing targeted tests; legacy warnings remain.

---

## Target Architecture (v1.3 focus)

1) **Deterministic Semantics**
   - Use hash-based projection (`project_to_hv`) for all concepts/queries.
   - Persist concept store to disk (hash → HV16) for stability across runs.
   - Introduce minimal semantic primitives (NSM-inspired) as fixed seeds.
   - Binding/bundling as core algebra; example:
   ```rust
   use symthaea::hdc::{project_to_hv, HV16};

   let red = project_to_hv(b"red");
   let car = project_to_hv(b"car");
   let red_car = red.bind(&car); // unique composite
   let desc = HV16::bundle(&[red_car, project_to_hv(b"fast")]); // add properties
   ```

2) **Safe Action Execution**
   - Keep ActionIR as the only execution surface.
   - Validation: policy + sandbox; reject unknown flags; allowlist programs.
   - Execution modes:
     - `Simulated` (default) – no process spawning.
     - `Real` (env/feature gated) – only allowlisted commands; return stdout/stderr.
   - Add optional rollback for writes (backup/restore or delete).
   - Telemetry: log validated actions + outcomes for audit.

3) **Intent Parsing Hardening**
   - Structured parsing for Nix intents: install/search/remove/upgrade with flags (`--profile`, `--file`), reject unknown/unsafe flags.
   - Table-driven tests: query → ActionIR → validation → execution outcome.
   - Future: add “describe/configure” intents as NoOp with guidance text.

4) **Cognition Path (planned)**
   - Replace random semantics in LTC/consciousness graph with persisted HVs.
   - Memory: persist concept map + item memory + graph; stabilize pause/resume.
   - Control loop: perception → planning (ActionIR) → validation/execution → feedback → memory update.
   - Add minimal learning signal (success/fail → threshold tuning; store action/response pairs).

---

## Milestones & Work Packages

**M1: Semantic Determinism**
- Wire hash projection into the main pipeline; remove random per-run concept creation.
- Add persistent concept store (hash → HV16) with load/save.
- Tests: determinism across runs; binding/bundling properties.

**M2: Action Layer Hardening**
- Finish flag parsing (known flags only) and add allowlist checks.
- Add rollback for writes (backup/restore) and telemetry.
- Gate real execution via env/feature; tests for allowed/denied commands.

**M3: Cognition Loop Skeleton**
- Persist graph + semantics; stable pause/resume.
- Minimal perception→planning→execution→feedback loop with logging.
- Store action/outcome pairs; basic retrieval for repetition/consistency.

**M4: Cleanup & CI**
- Remove legacy warnings, unused imports/vars, trivial assertions.
- Define fast test suite for CI (unit + integration on ActionIR/Nix parsing/semantic determinism).

---

## API Notes (v1.3)

- **HDC**: Prefer `project_to_hv` and HV16 algebra (bind/bundle). Avoid random concept generation.
- **ActionIR**: Only surface for execution. Validate before execute. Use `SimpleExecutor::from_env()` to respect real/ simulated toggle.
- **NixUnderstanding**: Emits `NixAction { action, description, confidence }`. Handles install/search/remove/upgrade with optional `--profile`. Unknown flags → NoOp.

---

## Next Implementation Steps (suggested)

1) Wire hash-based semantics into the main flow (replace random concepts) and add concept store persistence.
2) Add rollback + telemetry to `SimpleExecutor`; add allowlist enforcement for real mode.
3) Expand intent tests to cover flags/edge cases and ensure rejection paths.
4) Trim warning noise for cleaner CI output.
