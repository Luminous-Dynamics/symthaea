# PITB research note: LeanLean repository-scale proof compression (2026-11-01)

**Status:** external benchmark incorporated into the capability portfolio; PITB evaluation design only. No LeanLean task has been downloaded, rebuilt, scored, or modified by PITB. No compression result is claimed.

## Why this benchmark belongs in PITB

SorryDB measures closing a designated proof hole inside a real project. LeanLean measures an adjacent but distinct job: making an entire real Lean codebase smaller while retaining the validity of protected declarations. Its result depends on mathematical simplification, dependency restructuring, declaration deletion/addition, and syntax-level compression—not merely filling a sorry.

The official LeanLean benchmark uses 64 real-world Lean repositories, up to 174k lines of code, and a 12-hour per-repository budget. Its paper reports an average compression score of 48.3% for Opus 5, while the smaller Gemini 3.8 Flash and GPT-5.6 Luna configurations report below 25%. The authors find that most savings arise from proof-dependency restructuring; only Opus 5 and Muse Spark 1.3 obtain substantial gains from syntactic optimization. Source: https://www.sri.inf.ethz.ch/publications/rognvaldsson2026leanlean.

The released benchmark card reports 13.8M stripped tokens and 620 protected declarations, with 16 repositories in each of four stripped-token bands (≤10k, 10k–50k, 50k–250k, >250k). The official repository documents the pipeline:
1. Download repositories at pinned commits and build them.
2. Run lean-strip offline and verify reproduction with the Palomar Comparator.
3. Run agents in isolated containers for 12 hours with 8 CPUs and 64 GiB.
4. Rebuild the result, run the comparator, and compute token-reduction metrics.
5. Attribute savings across syntax, automation, dead code, structural and rewriting categories.

Sources: https://github.com/eth-sri/lean-lean and https://huggingface.co/datasets/eth-sri/lean-lean.

## Keep the leaderboard distinct

PITB should publish three different formal Lean cohorts, not one merged pass percentage:

| Cohort | Unit of work | Primary question |
|---|---|---|
| Research-family transfer | Frozen theorem family | Does a typed strategy learned from one family transfer to an unseen family? |
| SorryDB_2601 | One sorry hole in its original project | Can the system complete real repository obligations under a bounded budget? |
| LeanLean | One pinned repository with protected declarations | Can the system reduce repository proof/code size while preserving protected mathematics? |

A higher LeanLean compression score does not imply higher SorryDB completion or better unseen-family transfer. A lower SorryDB cost does not imply a smaller maintained Lean library. Keep denominators, uncertainty, budgets, checkers, and primary endpoints separate.

The current public LeanLean result is a contextual paper baseline, not a guaranteed matched PITB head-to-head. The benchmark repository includes currently updated model and harness configs that may postdate the paper table. For any reproduction, pin one named dataset snapshot, exact source commits, comparator bundle, prompt, model, effort, harness and resource limits. Never silently substitute the latest dataset or model config while citing the paper's table.

## PITB hypothesis: compression by proof-dependency surgery

PITB's specific contribution should be to predict and validate which declarations are genuinely necessary for a protected theorem, then propose a minimum-change dependency reorganization.

The system should build on three separate evidence graphs:

- **Typing dependency graph:** declarations required for the reconstructed local context to typecheck.
- **Assigned-value provenance graph:** declarations contributing to the construction of assigned values; a nondependent have value is provenance, not automatically a typing prerequisite.
- **Elaborated proof support:** declarations actually used by a particular proof term.

For a repository task, extend these to a declaration-level graph across modules: imports, theorem/definition dependencies, instances, simp lemmas, coercions, notation, syntax, macros, elaborators, and generated declarations. LeanLean's repository context means a theorem's explicit proof dependencies are not the entire operational dependency closure.

The proposer may suggest:
- removing declarations that are unreachable from protected theorem obligations;
- replacing long proof chains with a smaller set of proved intermediate lemmas;
- factoring repeated proof patterns into shared reusable lemmas;
- switching to a stronger, already available abstraction or canonical theorem;
- simplifying tactics or syntax after structural dependency changes;
- splitting or merging files only when import and elaboration behavior remains valid.

Every proposal remains untrusted. Rebuild the entire candidate repository and run the protected-theorem comparator from a clean checkout. Any missing protected theorem, changed target type, unapproved axiom/trust change, or comparator failure makes the candidate ineligible regardless of its compression score.

## Causal method arms

A first PITB evaluation should use the same model, model effort, agent harness, repository set, time/CPU/RAM limits, comparator version and system prompt family across arms:

| Arm | Intervention | What it isolates |
|---|---|---|
| L0 | Matched direct repository-editing baseline | Reference |
| L1 | L0 + repository declaration/dependency graph | Value of explicit dependency structure |
| L2 | L1 + support-antichain and alternative-route analysis for protected declarations | Value of minimal support prediction |
| L3 | L2 + typed, target-preserving repair operations | Value of constrained structural edits |
| L4 | L3 + typed proof-insight transfer from the separate training-family corpus | Value of transferable mathematical strategies |
| L5 | L4 + adaptive search/repair budget allocation | Value of marginal verified progress per unit cost |

Do not tune on the public evaluation repositories and then describe the result as a clean generalization test. The public snapshot is an external benchmark; for confirmatory claims, hold out entire repositories before the run or prepare a separately governed future cohort. Record pre-exposure and tuning history. A public-data/pretraining audit is a limitation, not a box to claim away.

A no-op baseline and the official lean-strip preprocessing reproduction are required controls. Compare PITB to the official paper baseline only under a protocol that can actually be matched; when model/harness versions differ, label the published paper values as contextual.

## Metrics

**Primary metric:** official LeanLean compression score with an exact pinned comparator.

**Required co-primary integrity:** 100% of protected declarations pass the pinned comparator and clean repository rebuild. Failed or invalid submissions receive no compression credit.

**Secondary metrics:**
- total stripped-token reduction and reduction by compression category;
- number of deleted, added, changed and shared declarations;
- protected-theorem coverage and per-repository pass/fail;
- raw project size, stripped size and final size;
- build, comparator, elaboration and tactic CPU time;
- wall time, peak memory, model calls, generated tokens, tool calls and spend;
- cost per comparator-preserving token removed;
- diff/patch size, edit reversals and invalid-edit attempts;
- repository-weighted macro average as well as token/repository micro aggregates;
- p50/p95 time-to-first valid compression and budget-versus-score curves.

Do not report only the mean compression score. The repository sizes span four bands; a strong result on compact projects can hide a failure on massive libraries. Break out all four bands, and publish task-level results with the official aggregate.

## Proposed qualification targets

These are research targets, not measured results:

| Goal | Proposed target |
|---|---:|
| Reproduction | 64/64 pinned source/comparator inputs reconstructed; baseline comparator replay |
| Correctness | 100% protected theorem preservation on any credited run |
| Average compression | ≥50% on the exact named 64-repository cohort (paper reference: 48.3%) |
| Massive-repository robustness | ≥25% average compression on the >250k stripped-token band, with no protected theorem regression |
| Efficiency | ≥20% lower cost per comparator-preserving token removed than the strongest matched baseline |
| Dependency contribution | Demonstrate a positive, paired L1→L2 improvement in dependency-attributed compression, not merely syntactic shortening |
| Repair integrity | 100% exact protected target identity; zero unapproved trust/axiom changes |
| Replication | Same-direction gain on a separate, temporally later or independently governed repository cohort |

The 50% target is deliberately modestly above the published 48.3% aggregate, not a claim that PITB should immediately beat Opus 5. The strongest first win could be a better cost/compression frontier while matching the comparator score, especially on the massive repository band.

## Paradigm shift: from code shrinkage to verified dependency optimization

The deeper research objective is not minimum bytes. It is **minimum verified dependency cost**.

A smaller repository is only useful if it remains maintainable and preserves all protected semantics. PITB should therefore estimate the cost of a proof dependency by its impact on:
- required imported modules and compiled artifacts;
- theorem/proof elaboration cost;
- typeclass and simp search behavior;
- duplicated argument structure;
- future theorem reuse and maintenance complexity;
- auditability and proof provenance.

This creates a multi-objective frontier rather than a compression-only optimizer. A candidate that removes 30% of tokens but adds a fragile global instance or a 20× elaboration regression should not dominate a slightly larger, structurally cleaner result.

The Pareto dimensions should include compression, comparator preservation, build/verification cost, proof elaboration latency, dependency centrality, and edit/maintenance complexity. Use a reviewed weighted utility only for internal scheduling; publish all raw dimensions and Pareto sets.

## Work plan

1. Pin the official LeanLean repository commit, dataset snapshot and comparator bundle digests. Confirm source license constraints for every upstream repository; the dataset license is mixed-upstream, so do not redistribute repository trees or proof artifacts without per-project permission.
2. Reproduce preprocessing on a four-repository smoke subset, then verify all 64 source and comparator identities before any agent run.
3. Import the official dependency graph and protected theorem manifest into a read-only PITB adapter; do not rewrite LeanLean's evaluation harness.
4. Build target-preserving declaration graph and support-antichain receipts for protected theorem dependencies.
5. Implement L1 and L2 interventions first; compare them under matched budgets before adding typed repair and strategy transfer.
6. Audit all proposed edits with clean rebuild, comparator, target identity, axioms/trust changes, import/instance/simp behavior, and resource accounting.
7. Only after the adapter and no-op baseline reproduce the official pipeline should PITB execute an agent arm.

## Sources

- Official SRI paper and benchmark summary: https://www.sri.inf.ethz.ch/publications/rognvaldsson2026leanlean
- Official benchmark implementation and evaluation protocol: https://github.com/eth-sri/lean-lean
- Published 64-project dataset metadata: https://huggingface.co/datasets/eth-sri/lean-lean
- PITB support antichain research: data/benchmarks/proof-insight-transfer-v0 (separate cohort and endpoint)
