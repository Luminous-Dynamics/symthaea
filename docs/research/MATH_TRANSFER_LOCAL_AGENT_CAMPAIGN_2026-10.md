# Math-Transfer Campaign: Local Agent Runbook

**Status:** research plan and local tooling only  
**Prepared:** 2026-10-19  
**Core repositories:** `Luminous-Dynamics/symthaea`, `Luminous-Dynamics/symtropy`, `Luminous-Dynamics/mycelix`  
**Applied engineering consumer:** `Luminous-Dynamics/sol-atlas-leptos`  
**Source corpus:** `openai/math` (pinned by commit before analysis)  
**Default operating mode:** local, branch-isolated, evidence-first; remote publication requires human approval

## Mission

Turn mathematical results into specific, measured improvements in Symthaea, Symtropy, and Mycelix. Do not stop at paper summaries. For every candidate result, establish:

1. the exact claim and its assumptions;
2. the formal proof scope, if a formalization exists;
3. whether the current code satisfies those assumptions;
4. the smallest software contract that could transfer the result;
5. a frozen reference corpus and independent oracle;
6. negative controls that would catch semantic mistakes;
7. a baseline benchmark and an honest statement of what the experiment does **not** establish.

A theorem is not a product capability. A proof artifact is not automatically a proof about production code. A passing test is scoped evidence, not universal validation.

## Safety and execution contract

The local agent MUST:

- inspect repository state before editing; never discard, overwrite, reset, clean, or stash pre-existing user changes without explicit permission;
- create one isolated worktree or clone per repository and per independent workstream;
- record `git status --short --branch`, base SHA, tested SHA, toolchain versions, exact commands and exit codes;
- stop and report if the target tree is dirty or its provenance cannot be established;
- run local checks without waiting for GitHub Actions, while recording hosted CI only as supplementary exact-head evidence;
- distinguish `PASS_WITH_SCOPE`, `FAIL`, `BLOCKED_WITH_EVIDENCE`, and `NOT_TRANSFERRED`;
- keep source fixtures immutable and versioned; do not silently normalize malformed input or loosen tolerances to make a test pass;
- never report queued, skipped, cancelled, stale-head, unrelated, or partial workflows as passing;
- never fabricate Lean theorems, paper claims, citations, benchmark results, tool output, hashes, or receipts;
- make no remote push, PR update, issue comment, merge, release, or branch-protection change without human approval;
- keep experimental scripts out of production paths. Prefer Rust for durable project tooling; a small independent oracle may use another language only if that independence is explicitly useful and it remains research-only.

## Starting anchors — revalidate before relying on them

These are inventory hints, not permission to overwrite active work. Re-query PR heads and workflow evidence before using any revision as current.

| Workstream | Anchor | Evidence and boundary |
|---|---|---|
| Symthaea finite-categorical EIG reference | [PR #8106](https://github.com/Luminous-Dynamics/symthaea/pull/8106) | Draft frozen reference: 35 ordered cases (23 valid, 12 invalid), plus semantic mutation controls. This is not proof of production correspondence. |
| Symthaea independent EIG qualifier | [PR #8107](https://github.com/Luminous-Dynamics/symthaea/pull/8107) | Draft, explicitly marked NEVER MERGE. The recorded exact-head qualifier passed at [run 39219906244](https://github.com/Luminous-Dynamics/symthaea/actions/runs/39219906244). Scope is the independent reference oracle, not state purity, production integration, calibration, or a scientific FEP theorem. |
| Symthaea math-transfer roadmap | [PR #8104](https://github.com/Luminous-Dynamics/symthaea/pull/8104) | Draft research map. Check it for existing work and avoid duplicates; roadmap prose itself is not proof. |
| Symtropy graph-entropy reference | [PR #2086](https://github.com/Luminous-Dynamics/symtropy/pull/2086) | Dedicated [graph entropy corpus run](https://github.com/Luminous-Dynamics/symtropy/actions/runs/39220370229) passed at its recorded head. The broad [CI run](https://github.com/Luminous-Dynamics/symtropy/actions/runs/39220370247) failed at formatting before later checks. Keep these outcomes distinct. |
| OpenAI mathematics collection | [openai/math](https://github.com/openai/math) | Pin the exact commit and record per-paper formalization scope. The collection contains manuscripts at different verification stages; do not treat the catalog as a blanket guarantee that every result is formalized. |

## Core paradigm shifts

### 1. From paper summaries to theorem-to-capability transfer

Use a fixed pipeline:

`source result → assumptions → software preconditions → contract → reference fixtures → independent checks → benchmark → evidence receipt`

If an arrow cannot be justified, stop at that boundary and state why.

### 2. From single scores to typed uncertainty

Keep distinct: entropy proxies, expected information gain, realized information, hypothetical posterior beliefs, canonical live belief, accepted evidence, confidence/calibration, and proof of truth. Do not let one stand in for another without an explicit mathematical derivation and contract.

### 3. From happy-path tests to falsification

Every reference needs valid cases, invalid cases, edge cases, metamorphic properties and semantic mutation controls. A deliberately wrong implementation should fail the tests; explain which test catches each mutation.

### 4. From local invariants to compositional guarantees

Prove a bounded subsystem first. Compose results only when interfaces expose matching assumptions, units, state transitions, conservation/accounting boundaries and failure semantics. Never generalize finite graph entropy to whole-system thermodynamics or graph routing to arbitrary dynamical systems without a valid theorem.

### 5. From benchmark anecdotes to reproducible evidence

Freeze data and split boundaries, record baseline and candidate, run ablations, preserve negative results, and report uncertainty. A benchmark suite should be deterministic when possible and should never silently use a different source, branch or head.

## Prioritized campaign

### P0 — Establish trustworthy local baseline

1. Run `scripts/math-campaign-preflight.sh` against the local review-clone root.
2. Review all generated reports and protect dirty or unrecognized working trees.
3. Confirm Rust/Cargo, Nix, Lean/Lake/Elan and Git availability; record versions.
4. Revalidate the active PR heads and current files for Symthaea #8104/#8106/#8107 and Symtropy #2086.
5. In an isolated Symtropy worktree, run `cargo fmt --all -- --check` and record the exact formatter diff. Fix only formatting in a separate focused commit; rerun formatter and the targeted graph-entropy integration test locally, followed by relevant clippy and workspace tests. Do not change mathematical expectations to satisfy formatting or CI.
6. Run the frozen Symthaea EIG reference against its independent oracle at the exact source SHA, and record the production-integration boundary as not established unless separately tested.

**Deliverable:** local baseline manifest and a short list of verified blockers, not speculative refactoring.

### P1 — Symthaea: inference, information theory and dynamical reliability

Prioritize in this order, subject to repository inventory:

- Finite-categorical expected information gain: retain the reference contract; test zero predictive mass, normalized probabilities, KL direction, expectation over observations, and sign in a minimization objective.
- Relative entropy / divergence contracts: domain constraints, zero support, finite/non-finite values, asymmetry, identity and data-processing properties where assumptions hold.
- Calibration and uncertainty: proper scoring rules, reliability curves, Brier/log loss, distribution shift and abstention cost; never use calibration as a proxy for proof or factual truth.
- Stability and bounded dynamics: identify state variables, norms, discretization, assumptions and valid operating region before considering Lyapunov-style certificates. Compare predicted and observed trajectories, stability boundaries and numerical error.
- Formal proof retrieval: evaluate whether assumption-aware retrieval improves held-out proof completion without weakening kernel checking, adding `sorry`/axioms, or accepting changed theorem statements.

Potential transfer only after code inspection: improve the action selector's information term, bounded belief updates, proof-backed configuration constraints or runtime monitoring. No production change before benchmark evidence.

### P2 — Symtropy: flows, conservation, stability and constrained optimization

- Graph entropy/dissipation: preserve the exact functional and its assumptions. Test sign, zero cases, edge reversal, vertex relabeling, malformed conductance, non-finite state and invalid edge endpoints.
- Conservation/accounting ledgers: prove debit/credit identity at a defined subsystem boundary; stress retries, partial failure, unit conversion, duplicate events and order changes. Do not claim whole-world conservation from a ledger invariant.
- Constrained allocation: establish feasibility, budget bounds, scale behavior, symmetry/permutation invariance and infeasibility reporting before optimizing objective quality.
- Graph metrics and routing: compare approximations against exact shortest paths on bounded graph families; report distortion, top-k agreement, p50/p95 runtime and memory. Do not substitute an approximation for a hard feasibility or safety constraint.
- Dynamical stability and dimensional analysis: test unit consistency, limiting cases, analytical solutions where available, step-size convergence and parameter sensitivity. Compare only within the validated model regime.

Potential transfer only after code inspection: a certified allocation kernel, graph-flow invariant, stability monitor, or bounded approximation with an explicit error contract.

### P3 — Mycelix: trust, Byzantine safety and civic coordination

Inspect existing DNA/zome, integrity validation and governance contracts before designing anything new. The first candidate should be narrow and assumption-explicit:

- Quorum intersection and Byzantine safety: parameterize validator count (n), Byzantine bound (f), quorum thresholds and membership view; enumerate small cases, derive the intersection condition, test boundary values and demonstrate counterexamples outside the assumptions.
- Evidence aggregation: test duplicate evidence, conflicting claims, Sybil-like weighting, dependence/correlation, source revocation, missing data and adversarially selected subsets. Do not interpret reputation as truth.
- Allocation and mechanism design: formalize feasibility, budget conservation, strategy assumptions, fairness definitions and manipulability; compare against simple baselines and publish trade-offs rather than a single unsupported “fairness” score.
- Governance under uncertainty: model quorum failure, abstention, appeal/reversal, stale membership, network partition and recovery. Safety/liveness claims must state synchrony and fault assumptions.

Potential transfer only after code inspection: a pure Rust reference model for a bounded rule plus property-based or exhaustive small-state tests and cross-implementation fixtures. Do not alter consensus-critical production validation in the discovery pass.

### P4 — Cross-repository research compiler

Prototype a local research index that stores:
- stable paper ID, repository commit and file/blob hashes;
- theorem statement and assumptions extracted with source location;
- formalization path, dependencies and exact claim covered;
- candidate code sites and required preconditions;
- fixture IDs, test commands, baseline/candidate metrics and evidence receipts;
- rejected transfer candidates with the reason they were rejected.

Use a plain TSV/JSON index first. Do not introduce a new service or database until evidence shows the need. Every cross-repo claim must link to canonical source data rather than copy a mutable fixture.

## Benchmarks and initial gates

These are starting thresholds, not claims of achieved performance. Tighten them only with a documented numerical or domain justification; never loosen a gate just to pass.

| Area | Minimum correctness gates | Capability / performance measure |
|---|---|---|
| EIG reference | 100% valid corpus passes; 100% invalid corpus rejects; independent formula implementations agree within a justified tolerance; zero-mass outcomes get no fabricated posterior; semantic mutants detected | Decision-quality uplift over entropy-proxy baseline on held-out synthetic tasks; report effect size and uncertainty |
| Probability / divergence | domain checks; support-boundary tests; symmetry only where mathematically expected; metamorphic properties; non-finite inputs rejected or explicitly handled | calibrated proper-score improvement on held-out distribution shifts, with abstention coverage/cost |
| Graph entropy | all valid and invalid reference cases pass; sign and zero semantics preserved; relabel and orientation controls pass | numeric residual / drift, scaling behavior, runtime and memory versus baseline |
| Allocation / ledgers | zero infeasible outputs accepted; budget/debit-credit invariants hold; duplicate/retry controls; order invariance where specified | welfare/fairness trade-offs, feasibility rate, worst-case regret and runtime |
| Graph approximation | compare only on graph families meeting theorem assumptions; exact solver retained as oracle | distortion, top-k agreement, p50/p95 runtime and memory over graph size/density |
| Stability / physics | dimensional consistency; explicit operating bounds; analytical/known-solution tests; convergence studies | stability boundary error, trajectory residual and numerical convergence rate |
| Mycelix quorum | exhaustive small-state counterexamples and proof or sound bounded verification under stated (n,f,q); fail closed outside assumptions | safety violations (target zero in explored state space), liveness under explicitly stated network assumptions, adversarial cost |
| Proof retrieval | no weakened theorem, no new unapproved axioms, no proof holes; all candidate proofs compile in pinned environment | start with +10 percentage-point absolute held-out proof-closure target, with unchanged checker and blind split; report sample size and confidence interval |

Do not aggregate these into one score. A hard safety failure cannot be compensated by throughput or average benchmark improvements.

## Working procedure for each candidate

1. **Inventory:** locate prior implementation, issue, PR, tests and known blockers. Record exact commit SHA before study.
2. **Source:** pin the exact `openai/math` commit and source files. Record blob SHAs and paper/catalog IDs.
3. **Interpretation:** write a short claim card: theorem, assumptions, definitions, formal proof scope, and what is not proved.
4. **Mapping:** name the exact code contract and preconditions. If the production representation does not satisfy an assumption, do not transfer.
5. **Reference:** freeze versioned fixtures. Include edge and invalid cases, a second oracle where possible, metamorphic tests and named semantic mutations.
6. **Baseline:** run current code and a simple/reference implementation on identical fixtures. Define metrics before inspecting candidate results.
7. **Implementation:** one narrow branch/worktree; minimum useful change; add tests. Do not refactor unrelated code.
8. **Verification:** run local formatting, unit/integration/property tests, clippy, relevant workspace build and Lean checks when applicable. Record every command and exit code.
9. **Review:** inspect full diff and blame/ownership boundaries; verify clean state and exact SHA. Confirm each claim matches collected evidence.
10. **Report:** write a receipt with results and non-claims. Ask before pushing or opening any implementation PR.

## Evidence receipt schema

Every candidate must emit a JSON receipt matching this shape. Unknown values are null or explicit blockers, never guessed.

```json
{
  "schema": "luminous-math-transfer-receipt-v1",
  "status": "PASS_WITH_SCOPE | FAIL | BLOCKED_WITH_EVIDENCE | NOT_TRANSFERRED",
  "repository": "owner/repo",
  "base_sha": "full SHA",
  "tested_sha": "full SHA",
  "source_math_repository": "openai/math",
  "source_math_commit": "full SHA or null",
  "source_files": [{"path": "...", "blob_sha": "...", "claim_scope": "..."}],
  "fixtures": [{"path": "...", "sha256": "...", "case_count": 0}],
  "toolchain": {"rustc": "...", "cargo": "...", "lean": "...", "nix": "..."},
  "commands": [{"command": "...", "exit_code": 0, "output_sha256": "..."}],
  "metrics": [{"name": "...", "value": "...", "unit": "...", "limit": "..."}],
  "mutation_controls": [{"name": "...", "detected": true}],
  "limitations": ["..."],
  "not_established": ["..."],
  "timestamp_utc": "..."
}
```

The schema is a template, not a completed result. A passing corpus does not imply correctness of production code unless correspondence is separately established.

## Output layout

Keep working reports outside canonical source fixtures until the team decides a result is ready for repository ownership:

```text
math-transfer-campaign/
  README.md
  inventory/
    summary.md
    symthaea.md
    symtropy.md
    mycelix.md
  index.tsv
  findings/
    information-theory.md
    graph-entropy-and-flows.md
    dynamical-stability.md
    optimization-and-allocation.md
    trust-quorums-and-governance.md
  receipts/
    <repo>-<tested-sha>-<workstream>.json
```

The local preflight script creates the inventory reports. Its output includes local paths, branches and repository status; inspect and redact before sharing outside the trusted working environment.

**Local check at campaign PR head `59256ccb07e86264c4ed17de96c7db8ccf12b4c0`:** `bash -n scripts/math-campaign-preflight.sh` passed. The script was executed against a temporary root of symlinks to isolated Symthaea, Symtropy and Mycelix worktrees; it generated the summary and all three repository reports and made no repository changes. The output reports were kept outside version control because they contain local paths and branch/worktree state. This inventory run did not run project tests, benchmarks or proof checkers.

## Expected final report

Begin with a one-screen summary, then include detail links:

- **Changes:** exact paths and commits; no generic “math improved” claim.
- **Best verified result:** exact tested SHA and evidence URL.
- **Checks:** full command, environment, exit code; label local and remote separately.
- **Benchmark deltas:** baseline versus candidate, sample size, uncertainty, ablations and negative results.
- **Assumptions and non-claims:** theorem boundaries and unsupported transfers.
- **Blockers:** exact evidence.
- **Next three tasks:** ordered by expected capability gain, feasibility and risk.
- **Remote actions:** explicitly state whether any push, PR or merge occurred.

## Execution update — 2026-10-19

This campaign is producing scoped, versioned contracts and exact-head qualification rather than waiting for broad repository runs. These are research artifacts; no PR listed below has been merged.

- **Symthaea finite-categorical EIG:** [reference PR #8106](https://github.com/Luminous-Dynamics/symthaea/pull/8106) freezes 35 cases (23 valid / 12 invalid); the separate **NEVER MERGE** [qualifier PR #8107](https://github.com/Luminous-Dynamics/symthaea/pull/8107) passed exact-head qualification at [run 39219906244](https://github.com/Luminous-Dynamics/symthaea/actions/runs/39219906244). Formula disagreement was about (1.11\times10^{-16}), six metamorphic comparisons passed, and all invalid cases rejected. It validates the finite-categorical reference oracle only; state purity, production correspondence, scientific calibration and a general FEP theorem are not established.
- **Symtropy graph entropy:** [PR #2086](https://github.com/Luminous-Dynamics/symtropy/pull/2086) passed its dedicated corpus workflow [run 39251968643](https://github.com/Luminous-Dynamics/symtropy/actions/runs/39251968643) and broad CI [run 39251969032](https://github.com/Luminous-Dynamics/symtropy/actions/runs/39251969032) at exact head `283dcc3fbec2becadaa501b442d03424243b6245`. Scope is the frozen finite-graph functional and tests, not whole-system thermodynamics.
- **Symtropy energy accounting:** [PR #2113](https://github.com/Luminous-Dynamics/symtropy/pull/2113) now has a complete exact-head pass at `8ccbef01b6ce0bd214423a7e52ba2403734fd4f4`: [reference workflow](https://github.com/Luminous-Dynamics/symtropy/actions/runs/39254295333) and [broad CI](https://github.com/Luminous-Dynamics/symtropy/actions/runs/39254295577). Its 24-case reference oracle, formatting, locked metadata and all 212 thermodynamics library tests passed, followed by clean postflight. The prior postflight failure from a repository-local Cargo `target/` was resolved by moving build artifacts under `/tmp`. It still does not establish that production callers enumerate every physical energy boundary or that a model is physically valid.
- **Mycelix trust posterior:** [PR #5221](https://github.com/Luminous-Dynamics/mycelix/pull/5221) passed exact-head qualification at [run 39251973242](https://github.com/Luminous-Dynamics/mycelix/actions/runs/39251973242), head `efefb2ae8bca3c49de159e59c12dec34144061f1`. It freezes 22 cases (13 valid / 9 invalid) and a 3,456-case property grid. This is a reference/calculation qualification, not production correspondence, calibration, Sybil resistance or correlated-source resistance.
- **Mycelix TCDM transition candidate:** [PR #5223](https://github.com/Luminous-Dynamics/mycelix/pull/5223) now contains a 30-case transition contract (17 valid / 13 invalid), a 100,000-step invariant sequence, five semantic mutation controls and an exact-head source-mirror parity gate for three trust copies and two MATL bridge copies. The independent arithmetic recomputation matched all frozen decisions. The PR was rebased onto current main and is now at exact head `ea3143543b7e9f5e63da452d8b03c4fa9ef996f5`; dedicated [workflow run 39289565506](https://github.com/Luminous-Dynamics/mycelix/actions/runs/39289565506) remains queued, so no Rust PASS is claimed. The independent D6U runtime executor is failing at the same head and needs its own investigation; the generic CI job was skipped for the draft PR, not passed.
- **Sol-Atlas regional repair continuity:** [PR #170](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/170) adds a Rust-only synthetic evaluator comparing centralized, local and distributed repair cells against matched demand, explicit disruptions, finite stock, calibration, staffing, power, routes and capacity windows. A frozen JSON fixture encodes the expected contrast: central stock arrives after the central route fails, a local cell can serve in window 0, and the distributed reserve may serve in window 1. This is not Johannesburg/Gauteng data. New review passes distinguish delayed parts, unknown inventory routes and supply unavailability; they also bound both fit evaluations and estimated stock-lot inspections. Its latest exact-head [reference run 39288824360](https://github.com/Luminous-Dynamics/sol-atlas-leptos/actions/runs/39288824360) is pending and the [broad check 39288824377](https://github.com/Luminous-Dynamics/sol-atlas-leptos/actions/runs/39288824377) is queued.
- **Mycelix production-hardening lead:** [issue #5222](https://github.com/Luminous-Dynamics/mycelix/issues/5222) records timestamp regression, invalid interaction weights, confidence semantics and duplicated source ownership. At the inspected base, all three `trust.rs` blobs match and both MATL bridge blobs match; resolve runtime ownership before applying the candidate semantics to production.
- **Operating rule:** work proceeds on independent source/fixture/review tasks while runners are queued. A queued, pending, skipped, cancelled, stale-head or partial workflow is not a pass; always record the SHA and scope that actually passed.

The key cross-project finding is that “more uncertainty” or “more precision” is not itself a capability. An epistemic score needs action-dependent semantics; a trust posterior needs calibration and adversarial evaluation; a conservation ledger needs a complete boundary; a repair plan needs time-indexed resources and stock. Each new contract states what it measures and what it cannot prove.

## Source links

- [OpenAI math repository](https://github.com/openai/math)
- [Manuscript map](https://github.com/openai/math/blob/main/CONTENTS.md)
- [Formalization catalogue](https://github.com/openai/math/blob/main/lean/formalization.yaml)
- [Symthaea EIG source](https://github.com/Luminous-Dynamics/symthaea/pull/8106)
- [Symthaea EIG independent qualifier](https://github.com/Luminous-Dynamics/symthaea/pull/8107)
- [Symthaea transfer roadmap](https://github.com/Luminous-Dynamics/symthaea/pull/8104)
- [Symtropy graph entropy transfer](https://github.com/Luminous-Dynamics/symtropy/pull/2086)
