# PITB SorryDB track v0

**Status:** benchmark design and receipt contract only. No SorryDB task has been run by PITB; there are no PITB SorryDB scores, Lean receipts, or SOTA claims.

## Why this is a separate track

The research-math transfer benchmark asks whether an insight learned from one mathematical family transfers to an unseen family. SorryDB asks whether a system can complete open proof obligations in the original Lean project where they occur. These are different estimands, task populations, and verification paths. **Never combine their scores or present one as a substitute for the other.**

The fixed external cohort is the January 2026 snapshot SorryDB_2601, specifically data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json. The SorryDB project documents a 2,601-task snapshot and a 1,000-task evaluation split; the paper describes a source pool spanning 78 real-world formalization projects. The dataset records the repository remote, branch, commit, Lean version, and sorry source coordinates needed to reconstruct a task. The independent verifier checks candidate proofs in the original project. See the links below.

PITB must not silently rededuplicate, resample, refresh, remove difficult projects, or replace this frozen cohort. A later nightly snapshot is a different temporal cohort, not an updated version of the January leaderboard.

## External reference points (not PITB results)

The SorryDB leaderboard currently reports these rows for SorryDB_2601:

| Approach | Reported result | Protocol class |
|---|---:|---|
| Aleph Prover | 57.2% Pass@1 | Specialized prover |
| Gemini Flash 3 (Agentic) | 30.3% Pass@1 | Iterative agent |
| Gemini Flash 3 (Self-Correcting) | 27.9% Pass@1 | Iterative |
| Claude Opus 4.5 (Self-Correcting) | 27.1% Pass@1 | Iterative |
| Gemini Pro 3 | 11.0% Pass@1; 20.5% Pass@32 | General-purpose |
| Gemini Flash 3 | 10.8% Pass@1; 20.5% Pass@32 | General-purpose |
| Claude Opus 4.5 | 7.8% Pass@1; 15.4% Pass@32 | General-purpose |
| GPT-5.2 | 6.2% Pass@1; 13.2% Pass@32 | General-purpose |

The live leaderboard's **63.2% combined performance** is the fraction of tasks solved by at least one approach currently listed on that board. It is an ensemble union, not an individual system's pass rate, not a Pass@k score, and not a PITB result. Do not claim a system beats 63.2% by comparing its own pass rate to that union.

**Important version distinction:** the published ICML 2026 paper's Table 2 reports a **35.7% combined union** for the set of methods evaluated in that paper. The current hosted leaderboard reports 63.2% and includes additional/current entries such as Aleph Prover (57.2% Pass@1), which is not a row in the paper's Table 2. Both are unions on the 1,000-task SorryDB_2601 slice, but they refer to different sets of approaches and publication states. PITB records the live-board 63.2% as the current external reference and 35.7% as the paper's historical study-set union; neither is a single-system baseline.

The paper describes iterative/self-correcting systems as running for up to 16 iterations, while general-purpose and specialized approaches may be evaluated by Pass@1 or Pass@32. In particular, the 30.3% Gemini Flash 3 agentic result is a single agent run per task with iterative tool/model feedback, not a single raw model completion. The fill-in-the-sorry unit can be a hole inside a larger theorem rather than a complete theorem from scratch. Comparisons must therefore publish iteration, feedback, context, and total resource limits—not only the Pass@k label.

Pass@1 and Pass@32 are also not interchangeable. The agentic 30.3% row is Pass@1; the general-purpose 15.4% Claude row is Pass@32 (with 7.8% Pass@1). Compare like with like and publish the attempt policy and resource envelope. A 32-candidate ceiling is not a licence to spend 32 times the compute without reporting it.

## Experimental question

Does an evidence-backed proof-engineering stack improve **verified real-project proof completion per unit of compute**, without changing the intended theorem, weakening the checker, or making the result harder to reproduce?

The primary causal comparison is within PITB: same exact tasks, same underlying model, same prompt family, same Lean/toolchain environment, and matched total resource limits. The public SorryDB leaderboard is an external reference, not a substitute for those matched arms.

### Registered method arms

| Arm | Intervention | Purpose |
|---|---|---|
| S0 | Same-model direct baseline; complete original project context; no PITB insight or retrieval | Reference treatment |
| S1 | S0 + intent-aware premise retrieval/query rewriting | Retrieval contribution |
| S2 | S0 + mechanically captured proof-state/context evidence, including local instances | Context-evidence contribution |
| S3 | S0 + reviewed typed proof insights and explicit preconditions/failure boundaries | Transfer contribution |
| S4 | S0 + S1 + S2 + S3, with evidence-gated routing and repair | Full PITB system |

A deterministic tactics baseline and published external approaches should be reported as additional references when their exact configurations can be reproduced. Do not compare different foundation models as if they were a single-component ablation. Every arm is bound to a config digest; all arms share a frozen task list and resource caps.

## Task reconstruction and proof acceptance

For every row:

1. Resolve the SorryDB record by its exact task ID and pin the source repository commit, Lean toolchain, package/lock files, and source coordinates.
2. Build the original project at that commit. The pretty-printed debug_info.goal is a debugging aid, not the authoritative proof-state input; reconstruct the actual goal from Lean.
3. Capture the elaborated target type and proof state. Record the ordered local context and local-instance registry; do not flatten a Lean context into an unordered bag of hypotheses.
4. Propose one or more proof candidates. The benchmark interface is a proof string replacing the designated sorry, consistent with SorryDB's strategy contract.
5. Apply the candidate in an isolated checkout. For the qualified primary score, the patch must replace only the designated sorry span. The theorem declaration, imports, unrelated source files, project commit, and build configuration must remain fixed.
6. Run the project's pinned Lean build/check in a clean verifier process. A successful candidate must close the target with no remaining goals and no sorryAx/admitted proof.
7. Recompute the target declaration's elaborated type digest after patching and require exact equality with the pre-patch type digest under the same pinned serializer/toolchain.
8. Record #print axioms (or an equivalent machine-readable axiom audit). Separate kernel-only proofs from any explicitly approved compiler-trust tier; never silently pool them. Unreviewed new axioms or trusted escape hatches disqualify the candidate from the primary integrity-qualified score.
9. Preserve the complete candidate, source patch, build output, verifier receipt, toolchain identity, resource use, and content hashes.

The statement digest is a within-pinned-toolchain identity check, not a universal cross-version canonical theorem identity. Store the serializer ID and Lean version beside it. Compilation alone does not prove that the source obligation was the intended mathematics, so retain source correspondence and mutation checks as separate qualifications.

## Outcomes and denominators

Use one record per task × method arm × replicate, with one ordered attempt list. The analyzer—not a model or hand-edited CSV—derives task success from the attempt receipts.

- verified_success: independent checker accepted the proof and all integrity conditions passed.
- verified_failure: an attempted candidate was checked and did not close the obligation.
- timeout: a known operational non-solve under the registered budget.
- runner_unavailable: a known operational non-solve; report separately as infrastructure failure as well as in the operational denominator.
- not_attempted: remains visible in assignment accounting and is not silently dropped.
- unknown: evidence is genuinely indeterminate; it blocks confirmatory qualification and remains in the denominator accounting.

Report raw assigned, attempted, verified, failed, timeout, unavailable, not-attempted, unknown, and excluded counts. Do not silently convert missing evidence to success or delete it. Do not silently convert an indeterminate record to a known failure. A run with any unknown outcome is diagnostic-only until reconciled.

Use repository-clustered uncertainty because tasks from one Lean project share dependencies, style, authorship and build environment. Report both task-weighted micro averages and repository-weighted macro averages. Family-level bootstrap from the OpenAI-math transfer track is not the correct resampling unit here.

## Metrics

**Primary:** exact-task Pass@1 under a fixed per-task resource envelope, plus the paired absolute difference against S0.

**Secondary:** Pass@8 and Pass@32; success-versus-cost curves; verified successes per dollar and per 1,000 generated tokens; wall-clock time to verified proof; API/tool calls; Lean checker/build time; proof-size and edit-size distributions; per-project coverage; and failure taxonomy.

**Integrity:** target-type digest invariance; patch-scope compliance; zero admitted proofs; axiom/trust-tier distribution; independent-verifier replay rate; reproducibility from a clean pinned checkout; and mutation-suite false-accept rate.

Always report total resource consumption and cost per verified success, not only the best pass@k point. A system that finds one proof after an unbounded number of calls is not a lower-cost improvement.

## Qualification targets

These are proposed thresholds, not measured results:

| Gate | Proposed target |
|---|---|
| Receipt completeness | 100% of assigned task × arm rows accounted for; no silent exclusions |
| Exact replay | 100% of claimed successes independently rechecked from pinned source commits |
| Semantic integrity | 0 theorem-type changes, 0 out-of-span source edits, 0 admitted proofs, 0 unapproved new axioms |
| Mutation controls | 0 false accepts in the registered statement/patch/axiom mutation suite |
| Matched PITB utility | At least +5 percentage points absolute Pass@1 over S0, with a repository-clustered 95% interval excluding zero |
| Efficiency | At least 10% lower cost per verified success than S0 at matched solve rate, or a strictly better success-cost frontier |
| Industry stretch | At least 60% Pass@1 on the frozen SorryDB_2601 cohort, exceeding the published 57.2% Aleph Prover Pass@1 reference under a genuinely matched protocol |
| Forward transfer | Replicate the direction of improvement on a separately frozen post-January-2026 SorryDB cohort, with no tuning on that cohort |

The industry stretch is intentionally hard. Aleph Prover is a specialized/proprietary reference, and a result is not a fair head-to-head unless the run budgets, project reconstruction, checker, and accepted trust tier are matched. If that cannot be established, report the 57.2% as a published reference—not as a controlled baseline.

The +5pp and cost gates are causal PITB claims; the 60% target is an external performance stretch. Passing one does not imply passing the others.

## Paradigm-shift hypotheses

1. **Proof completion as a transactional source transformation.** The system proposes a replacement for a designated hole; a separate verifier checks type identity, patch scope, project reproducibility and axioms before the result can count. Proof search remains untrusted.
2. **Context as an elaboration resource, not a text prompt.** Local declarations, binder metadata, instance candidates, auxiliary declarations and dependency closure are represented explicitly. Test support subsets and context interventions while preserving the target.
3. **Retrieval as route reconstruction.** Combine intent-aware query rewriting with formal dependency traversal, then measure complete premise-route coverage and downstream verified completion—not only Recall@100.
4. **Typed transfer with refusal.** A proof insight carries formal preconditions, state transition, evidence anchors and failure boundaries. The system should abstain when a target violates the contract instead of forcing a high-similarity but invalid analogy.
5. **Compute as a sequential decision.** Choose the next retrieval, context projection, repair or candidate attempt by expected marginal verified success per unit cost. Stop when another attempt is not justified by calibrated evidence.
6. **A continual benchmark, not a one-time contest.** Treat SorryDB_2601 as the frozen historical comparison and post-snapshot tasks as a separate forward-generalization cohort. Record task birth time, repository commit and first-seen time to make temporal leakage auditable.

The novel PITB claim should therefore be: **more verified real-project completions, or the same completions at lower cost, because transferable strategies and context evidence guide a better proof-engineering policy.** It should not be an unsupported claim that PITB's raw pass rate already exceeds industry systems.

## Current blockers and next executable milestones

1. Pin the exact upstream SorryDB_2601_1000_evaluation_split.json bytes, SHA-256, Git blob SHA, and source commit. **Completed as source inspection and hash verification; the data file is not copied into this repository.**
2. Implement a Rust importer that rejects changed bytes, duplicate IDs, malformed repository pins, path traversal, invalid source spans, and mismatched task/repository counts. **Implemented and wired into CI; not yet compiled or executed on the runner.**
3. Establish a three-project, three-toolchain smoke cohort and source verifier, including exact source/manifest blob pins and a real LocalInstances API compatibility calibration. **Candidate and source verifier implemented; build and Lean capture not yet observed.**
4. Implement a Rust task × arm × replicate scorekeeper deriving Pass@k, repository-clustered intervals, cost curves, and denominator accounting from receipts.
5. Run matched S0–S4 pilot arms on a development-only subset. Freeze prompts/configs before opening any confirmatory rows.
6. Run the full pinned cohort, then separately evaluate a temporally later SorryDB snapshot. Never use the January public test slice as evidence of contamination-free generalization.

The current PITB v2 Lean exporter/context evidence is not yet compiled, and its exact-head CI has not been observed. This SorryDB contract is design/infrastructure work only; it does not close those existing validation gaps.

## Upstream references

- SorryDB leaderboard and January 2026 snapshot: https://sorrydb.org/
- SorryDB ICML 2026 paper: https://proceedings.mlr.press/v306/letson26a.html
- SorryDB source and frozen split: https://github.com/SorryDB/SorryDB
- Database record format: https://github.com/SorryDB/SorryDB/blob/master/doc/DATABASE.md
- Strategy input/output contract and runner guidance: https://github.com/SorryDB/SorryDB/blob/master/doc/STRATEGIES.md
- Legendre's pinned Mathlib retrieval comparison: https://www.legendre-leaderboard.com/
- LeanGraphSearch results and route-coverage definition: https://pluslabnlp.github.io/LeanGraphSearch/
- Lean 4 reference on local instance synthesis: https://lean-lang.org/doc/reference/latest/Type-Classes/Instance-Synthesis/


## Exact source identity and importer

The frozen evaluation file is pinned to SorryDB commit \`74b3b63f8ba890d675f362924b21ac55db410383\` and Git blob \`af3b663429dc4689ee728d2fb5c7d8d5034476f7\`. Its exact UTF-8 payload is 1,599,271 bytes with SHA-256 \`26bfd11099f6b8db7f80d49bdea5d9f8ea0d23ba65d44d93f67ed42ba1530062\`. The Git blob identity was independently recomputed from the retrieved payload and matched the blob SHA-1 published by the pinned Git tree. The audit found 1,000 records, 78 distinct repository remotes, and no duplicate task IDs.

Two rows use a zero-based start column; therefore the importer and receipt schema allow column zero while retaining one-based line numbers and strictly ordered source spans.

Use the Rust importer after obtaining the file from the pinned checkout:

\`\`\`bash
cargo run -p symthaea-proof-insights --bin import_sorrydb_snapshot -- --self-test

cargo run -p symthaea-proof-insights --bin import_sorrydb_snapshot -- \\
  --input /path/to/SorryDB_2601_1000_evaluation_split.json \\
  --output /tmp/pitb-sorrydb-task-manifest.json
\`\`\`

The importer rejects altered source bytes, duplicate task IDs, non-canonical repository remotes, malformed commit/toolchain identifiers, unsafe paths, malformed spans, and unexpected task/project counts. It emits a deterministic manifest with source pins, task ordinals, project/repository identity, source coordinates, and hashed debugging-goal strings. The debugging goal is not treated as authoritative proof-state evidence.

The source byte hash and manifest are provenance artifacts only. They do not establish that a source repository can be rebuilt or that any proof obligation has been solved. Lean reconstruction and independent checking remain a separate milestone.

## Toolchain stratification and smoke candidates

The frozen 1,000-task cohort spans 14 Lean toolchain tags, ranging from v4.18.0-rc1 through v4.27.0-rc1. Its largest groups are v4.27.0-rc1 (318), v4.26.0 (218), v4.24.0 (145), and v4.26.0-rc2 (97). Report environment readiness by toolchain and repository separately from theorem-proving success. A source checkout, toolchain, dependency, or untouched-project build failure is not a model proof failure; preserve it in operational accounting and do not silently remove it from the cohort.

The first three source-inspected smoke candidates are recorded in smoke-cohort-candidates.v0.json: LeanCourse25/v4.24.0, PrimeNumberTheoremAnd/v4.26.0, and clap-lean/v4.26.0-rc2. The third task was selected because the SorryDB debug-context preview contains inst✝ : Fact (Nat.Prime p), providing a direct challenge for the new local-instance evidence path. This textual hint is only for selection. The actual LocalInstances registry must be read from the real Lean metavariable state.

All three exact sorry source spans and project manifest blob pins have been inspected. None has been rebuilt, reconstructed as real Lean evidence, or solved. This is a small integration smoke cohort, not a statistical accuracy sample.


## Snapshot status at 2026-10-31

The raw source hash, source Git blob identity, 1,000-task count, 78-repository count, duplicate-ID check, and source-field audit have been independently inspected. The exact three smoke task coordinates and repository configuration blob identities are pinned in smoke-cohort-candidates.v0.json.

The Rust importer and source verifier are committed. The focused workflow is designed to compile/test/lint the isolated PITB package, import the exact frozen snapshot twice and compare outputs, verify the three checkouts, build the original pinned projects, and compile the LocalInstances calibration. GitHub currently reports no workflow runs or commit status entries for the latest PR head. Therefore none of those compile/build/capture steps is claimed as passed. No real SorryDB proof has been attempted and no PITB success rate exists.


## Bounded speculative proof ledger

PITB's R0–R4 runtime comparison is recorded in a branch ledger. Branches may return retrieval routes, partial lemmas, repair plans, counterexamples, or complete proof candidates, but all remain untrusted. A proof candidate is represented by a SHA-256 of the exact candidate bytes. Promotion must bind that digest to the source branch, the sorry-only patch digest, the original target-type digest, a checker identity/receipt, and a distinct independent replay identity/receipt. The checker and replay must attest the same candidate bytes. Changing the target statement, promoting a different branch artifact, adding unapproved axioms, or escaping the sorry span blocks success.

Resource accounting separates elapsed wall-clock from summed branch wall-work. Parallel branches may accumulate 180 branch-ms while completing in 120 elapsed-ms; that is not an accounting contradiction. Token count, model calls, Lean CPU and cost remain additive. The Rust `analyze_sorrydb_branch_ledger` binary validates these joins and budget invariants and has a synthetic negative-control suite. The fixture is contract-only and does not record an executed branch or proof.


## Structural expression identity (2026-11-01)

PITB now has a separate v1 structural Lean-expression wire contract and an independent Rust verifier. It binds target expressions to exact source/task, Lean runtime-artifact, imported OLean-closure, and Lake environment metadata, maps FVarIds to ordered local-context indices, preserves the local-instance registry, and keeps transparent-let dependencies separate from nondependent-have value provenance. The contract deliberately uses an environment-scoped structural digest rather than claiming a cross-version semantic theorem ID. See docs/research/pitb-lean-expr-wire-v1-2026-11-01.md.

The checked-in example is synthetic. The Rust verifier has not yet been compiled in this environment, and the Lean v1 exporter has not yet been implemented or run; no real capture or proof result is claimed.


## Live elaboration snapshot arm (R5)

R5 adds the Level-2 live proof-state snapshot baseline to the existing R0–R4 runtime comparison. It is R4 with the post-elaboration proof state captured once; branches share the immutable environment and receive independent mutable metavariable-context copies. This isolates the benefit of avoiding repeated import and theorem-body elaboration from the benefit of PITB's retrieval, context and strategy interventions.

Every R5 branch ledger must bind a separate snapshot receipt SHA-256 and pass `analyze_sorrydb_branch_ledger --validate LEDGER --snapshot-receipt RECEIPT`. The Rust verifier hashes the exact receipt bytes, parses the strict receipt structure, recomputes the full live-state digest over environment + goal metadata + target + LocalContext + MetavarContext, and joins task ID, repository commit, proof-state identity and started branch IDs against the ledger. v0 requires mutable tactic/typeclass caches to be discarded or recreated per branch; cache sharing is not allowed unless a separate dependency-invalidation qualification exists. The checked-in R4 fixture remains a no-snapshot synthetic contract fixture.


### R5 snapshot receipt v0

The separate receipt contract is `snapshot-branch-receipt-v0.schema.json`, with a synthetic no-run fixture at `fixtures/snapshot-branch-receipt.synthetic.json` and focused workflow `.github/workflows/pitb-sorrydb-snapshot-contract.yml`. It binds task/source identity, structural proof-state identity, Lean runtime and imported OLean manifests, snapshot server/RPC build identity, capture stage, shared immutable Environment, per-branch mutable MetavarContext copies, private workspaces, cache policy and measured branch/resource metrics.

The receipt is required in addition to the branch ledger's `snapshot_receipt_sha256`. A hash-shaped link is not proof that a receipt exists; the independent schema validator must accept the receipt bytes and the run's evidence must be retained. The synthetic fixture remains explicitly not attempted. No live snapshot implementation or measured R5 result is claimed.


The standalone R5 receipt schema and no-run fixture are:
- `snapshot-branch-receipt-v0.schema.json`
- `fixtures/snapshot-branch-receipt.synthetic.json`
- `.github/workflows/pitb-sorrydb-snapshot-contract.yml`

The LeanLean repository-compression cohort has a separate per-repository outcome contract under `leanlean-track/`; none of its scores are mixed into SorryDB or family-transfer reports.


The R5 receipt's `proof_state_sha256` is recomputed over five component digests: environment, metavariable goal metadata, target expression, ordered local context and full MetavarContext. It is deliberately stronger than the static Lean Expr wire v1 proof-state digest; R5 must bind the mutable metavariable context as well.
