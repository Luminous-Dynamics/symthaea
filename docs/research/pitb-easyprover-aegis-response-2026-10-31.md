# PITB architecture response to EasyProver and Aegis

**Date:** 2026-10-31  
**Status:** research and ablation design; no PITB EasyProver/Aegis run or result.

## New evidence

The 28 October 2026 EasyProver preprint reports 669/672 PutnamBench proofs, all six IMO 2026 formalizations, 597/600 NTP4VC verification conditions, and all 189 VERINA tasks resolved by proof or refutation. Those outcomes are from separate benchmark cohorts and cannot be compared directly with SorryDB Pass@1.

The architectural contribution is more instructive than the headline scores: recursive agents in isolated workspaces, asynchronous result returns, parallel tactic probes, admission control for agents and Lean queries, a stateless elastic elaboration service, and a lock-protected promotion step followed by independent checking. The authors report that some tasks needed more than 300 child agents. Their 2,000-task Aegis sample produced 1,054 proofs (52.7%) under 30 minutes per statement.

## PITB shift: bounded speculative proof engineering

Treat proof search as a speculative DAG, not one long chat transcript:
- Immutable task envelope: SorryDB task ID, repository commit, target source span, Lean/Lake pins, goal digest, LocalContext digest, LocalInstances digest and budget.
- Private branch workspaces and isolated contexts. Branches may propose a helper proof, retrieval route, context projection, typed repair, or counterexample/refutation.
- Asynchronous typed returns. Parents receive useful subproof artifacts, counterexamples, failed preconditions, dependency receipts and resource summaries, not every child diagnostic.
- Evidence-aware branch selection. Choose follow-on actions by expected marginal verified success per unit cost; stop low-value branches.
- Separate admission controls for agent fan-out and expensive Lean queries. Begin with 1/2/4/8 workers and apply backpressure before the elaborator is saturated.
- Shared immutable build caches and warm elaboration workers where safe; never share mutable project workspaces across speculative branches.
- Single transactional promotion gate. Under a lock, promote one candidate, verify exact task/source/statement identity, ensure only the target sorry span changed, check kernel proof and axiom policy, and replay from a clean pinned checkout. Revert on any failure.

A successful tactic probe is a search signal, not a benchmark success. Only the independent promotion checker may emit a verified-success receipt.

## Registered runtime ablation

| Arm | Design |
|---|---|
| R0 | Serial single-context agent |
| R1 | Independent restarts with isolated contexts |
| R2 | Recursive delegation with private workspaces and asynchronous returns |
| R3 | R2 plus parallel tactic probes and separate agent/Lean resource caps |
| R4 | R3 plus PITB context evidence, typed insights, support selection and repair routing |

Hold model, prompts, task set, Lean toolchain, checker, trust tier and total resource cap constant. Evaluate at several common resource caps. Report Pass@1/8/32 separately and use repository-clustered intervals.

Required secondary measures: useful-child-return rate, duplicate-work rate, branch survival/termination cause, tactic-probe success rate, total Lean CPU time, elaborator queue time, warm-cache hit rate, p50/p95 time to first verified proof, tokens, and cost per independently verified proof. The architecture only wins if it improves verified outcomes or cost at fixed reliability.

## Aegis is separate

Aegis is a generated implementation-level verification benchmark, not a SorryDB extension. It uses safe Rust implementations, differential validation against Aeneas-translated Lean models, frozen generated specifications and kernel-checked proof/refutation labels. Before using it, pin the source release, license, artifacts, statement set and checker semantics. Report proof, refutation, unresolved, invalid-specification and abstention as separate outcomes. Also audit translation fidelity, generated-spec quality and mutation controls: proving a proposition about a translated program does not alone certify that the proposition captures the intended user requirement.

## Targets

These are hypotheses, not results:
1. Zero branch workspace contamination and exact immutable task/proof-state identity.
2. Better p95 time-to-first-verified-proof at matched total budget, with no reliability loss.
3. Higher useful-child returns and verified solutions per 1,000 branch tokens.
4. Lower cost per verified proof through isolated useful results and efficient probe filtering.
5. 100% independent replays of promoted proofs, with no theorem drift, sorryAx leakage or unapproved axioms.
6. Forward replication on a newer SorryDB snapshot; separate Aegis evaluation only after its labels and artifacts are pinned.

## Execution order

1. Finish SorryDB source preflight and the pinned three-project smoke.
2. Implement the minimal branch/workspace ledger with a two-worker cap and a single promotion gate.
3. Establish R0/R1 on a development-only subset.
4. Add asynchronous R2, then small-scale R3.
5. Add PITB evidence routing as R4 and run matched budgets.
6. Evaluate a temporally later SorryDB snapshot, then prepare an independent Aegis adapter.

The paradigm shift to test is whether **evidence-aware, isolated, asynchronous search spends less on dead ends and returns more verifiable proof value per unit cost**. This is a falsifiable claim, not a promise that more agents automatically improve results.

## Source

Ganguly et al., *Scaling Lean Theorem Proving for Real-World Code Verification*, arXiv:2610.38174 (28 October 2026): https://arxiv.org/abs/2610.38174
