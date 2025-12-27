# Symthaea v1.3: Cognitive HDC System Design

**Purpose**: Replace v1.2 with a single, coherent design that aligns the “revolutionary improvements” and session learnings into a buildable path. This is not a diff; it is a fresh, integrated specification.

---

## Executive Summary

Symthaea v1.3 is a cognitive architecture that combines:
- **Deterministic Hyperdimensional Computing (HDC)** with semantic primitives (NSM-inspired) for stable, composable meaning.
- **ActionIR + Policy/Sandbox** as the only execution surface, with simulated by default and gated real execution.
- **Memory & Persistence** spanning semantic store, episodic traces, and consciousness graph.
- **Cognition Loop** that ties perception → semantic encoding → planning (ActionIR) → validated execution → feedback → learning.

Goals for v1.3:
1) Make semantics stable and persisted (no random per-run vectors).
2) Make actions safe and auditable (ActionIR-only, policy/sandbox, optional rollback).
3) Deliver a minimal cognitive loop with measurable behavior (task success, retrieval, learning signals).
4) Reduce hallucination/claims gap: clearly mark implemented vs. aspirational components.

---

## Guiding Principles

1) **Determinism over randomness**: hash-based HVs, persisted stores, reproducible tests.
2) **Safety first**: ActionIR validation + sandbox + policy; real execution opt-in; telemetry and rollback.
3) **Compositional semantics**: binding/bundling algebra over HVs; semantic primitives as anchors.
4) **Auditability**: trace every action from intent to validated execution and outcome.
5) **Incremental cognition**: build a small, measurable loop before adding complexity.

---

## Architectural Stack

### 1) Semantics (HDC + Primitives)
- **HV16 bit-packed hypervectors**; primary API: `project_to_hv(bytes)` (BLAKE3) → deterministic 2048-bit vector.
- **Semantic primitives**: seed a fixed set of NSM-like concepts (`I, you, do, happen, good, bad, want, know, move, place, time, number`) as stable anchors.
- **Algebra**:
  - Binding: `A.bind(&B)` → unique composite.
  - Bundling: `HV16::bundle(&[...])` → prototype.
- **Persistence**: concept store (hash → HV16) serialized to disk; item memory for episodes; load/save on start/stop.
- **Example**:
  ```rust
  use symthaea::hdc::{project_to_hv, HV16};
  let red = project_to_hv(b"red");
  let car = project_to_hv(b"car");
  let red_car = red.bind(&car);             // “red car”
  let fast = project_to_hv(b"fast");
  let desc = HV16::bundle(&[red_car, fast]); // “fast red car”
  ```

### 2) Cognition Loop
- **Perception**: inputs → HV encoding (hash-based) → working memory.
- **Planning**: intent parsing → ActionIR plan (structured, no shell strings).
- **Validation**: policy + sandbox path validation, flag allowlist; unknown flags → NoOp.
- **Execution**:
  - `Simulated` (default): no process spawn, returns simulated outcome.
  - `Real` (opt-in via env/feature): whitelisted programs only; returns stdout/stderr.
- **Feedback**: log outcome, update episodic memory, adjust confidence/thresholds.
- **Learning (v1.3 minimal)**: store (intent, ActionIR, outcome); reuse on repeats; simple success/fail counters.

### 3) Action & Safety Layer
- **ActionIR** variants: Read/Write/Delete/Create/List, RunCommand, Sequence, NoOp.
- **PolicyBundle**: allowlists for programs; filesystem patterns; budgets; max write size.
- **SandboxRoot**: canonicalize paths; reject escapes/symlinks; relative patterns resolved to sandbox.
- **Rollback (planned)**: backup on write/delete; restore on failure; tag irreversible actions.
- **Telemetry**: log validated actions + outcomes (simulated or real).
- **Execution gating**: `SYMTHAEA_ALLOW_REAL_EXEC=1` enables `ExecutionMode::Real`; otherwise simulated.

### 4) Memory & Persistence
- **Concept store**: hash→HV16 map persisted; deterministic across runs.
- **Episodic traces**: ActionIR + outcome + timestamp; stored for retrieval.
- **Consciousness graph**: nodes reference HV components; persist graph with semantic store; resume restores both.
- **Sleep/consolidation** (planned v1.3+): bundle recent traces; prune by importance; recombine (REM) for novelty.

### 5) Consciousness/Reasoning (Minimal v1.3)
- **Replace randomness**: use persisted HVs in graph nodes; remove per-run random vectors.
- **Metrics**: track graph size, self-loops, simple φ estimate (if retained), and retrieval success.
- **Behavioral check**: measurable tasks (e.g., repeated Nix intent with cache hit, safe execution) as “cognitive” signals.

---

## Implementation Plan (v1.3)

**P1: Semantics Determinism**
- Wire `project_to_hv` into the main pipeline; remove random concept creation.
- Add concept store persistence (disk) and load on init; tests for determinism across runs.

**P2: Action/Safety Hardening**
- Enforce flag allowlist; unknown/unsafe flags → NoOp with message.
- Add rollback scaffolding (backup on write/delete) and telemetry logging.
- Gate real execution via env/feature; add tests for allowed/denied programs/args.

**P3: Minimal Cognition Loop**
- Persist graph + semantics together; ensure pause/resume stability.
- Record (intent, ActionIR, outcome) and reuse for identical queries (cache).
- Add a simple success/fail adaptation: boost confidence on repeated successes; degrade on failures.

**P4: Cleanup & CI**
- Remove legacy warnings (unused imports/vars, trivial comparisons).
- Define a fast CI suite: action/safety tests, intent parsing tests, semantic determinism tests.

---

## Intended Behavior (Examples)

1) **Install flow (simulated default)**  
Input: “install ripgrep --profile dev”  
→ Parse intent → ActionIR::RunCommand(program="nix", args=["profile","install","--profile","dev","nixpkgs#ripgrep"])  
→ Validate against policy/sandbox → Simulated executor returns `SimulatedCommand`.

2) **Search flow (real mode, opt-in)**  
Env: `SYMTHAEA_ALLOW_REAL_EXEC=1`  
Input: “search vim”  
→ ActionIR validated → Real executor runs whitelisted `nix search` → returns stdout/stderr.

3) **Memory reuse**  
Second “install ripgrep --profile dev” uses stored ActionIR/outcome to respond faster; records success count.

---

## Risks & Mitigations

- **Hallucinated capabilities**: All non-implemented features marked aspirational; demo text must reflect reality.
- **Unsafe execution**: Real mode gated by env/feature; allowlist enforcement; sandbox/canonicalization.
- **State loss**: Persist concept store and graph together; tests for round-trip integrity.
- **Warning noise**: Clean test warnings to surface real regressions.

---

## Deliverables Checklist (v1.3)

- [ ] Concept store persistence + deterministic semantics wired into main flow.
- [ ] ActionIR validation: flag allowlist, NoOp on unknown flags, rollback hooks.
- [ ] Executor gating: simulated default, real mode opt-in with allowlist tests.
- [ ] Memory cache of intents → ActionIR → outcomes; reused on repeat queries.
- [ ] Fast CI suite defined; legacy warnings removed.
- [ ] Documentation updated (README/demo + this design) to reflect actual capabilities.

---

## Notes on Scope vs. Aspirations

- Swarm/libp2p, advanced perception, TTS, rich consciousness metrics remain out-of-scope for v1.3; keep references as future work only.
- LTC/consciousness dynamics remain heuristic until semantics and memory are stable; focus on measurable task performance first.
