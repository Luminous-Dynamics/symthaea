# Proof Insight Transfer: temporal, repository-realistic challenge track

**Date:** 2026-10-20  
**Status:** proposed external replication protocol; no temporal challenge run or transfer result is claimed.  
**Relationship to PITB v0:** this is a separately reported challenge track. It does not replace the existing frozen OpenAI/math family split, its primary contrast, or its acceptance gates.

## Why a second evaluation regime is needed

The PITB v0 70/15/15 family split is a good contamination control for a static corpus, but a random family split cannot reproduce the real work loop: the available lemma library grows over time, APIs and project conventions are local, and proof availability is historically asymmetric. A system may appear to transfer insights because its candidate strategy was extracted from a proof or dependent declaration that would not yet have existed at the moment of the target.

A temporal track asks a stronger causal question:

> Given only the definitions, declarations, proofs, and insight records that existed before a target theorem was completed, does exposing a compatible typed proof strategy improve the rate of exact target proofs over equally budgeted retrieval and proof-context controls?

This separates transfer from hindsight. It also measures whether a strategy survives the shift from isolated mathematics to real dependency-rich verification.

## Evidence from adjacent benchmarks

- **Dalek Lean Bench** provides an applicable chronological design: its public repository documents a 249-theorem proof timeline and 146 extracted theorem–proof pairs. For a target at timeline position *i*, it retains earlier proofs and replaces the target and later proofs with `sorry`, reflecting the context available when the target was originally developed. This is a strong *evaluation pattern*, not a published PITB result. [Repository and protocol](https://github.com/Beneficial-AI-Foundation/dalek-lean-bench).
- **VeriSoftBench** evaluates 500 theorem tasks from 23 real-world Lean repositories. Its project page reports a best result of 41% with curated dependency context versus 35% with the raw repository, demonstrating that context selection is itself a large variable. PITB must use a matched context budget and should not confuse an insight treatment with a context-curation advantage. [Benchmark and comparisons](https://utopia-group.github.io/VeriSoftBench/).
- **LeanSearch v2** reports strong global-premise retrieval results: nDCG@10 0.62 on its single-query retrieval benchmark and 46.1% retrieval of known premise groups within ten candidates on its 69-query research-Mathlib benchmark; its controlled downstream experiment reported 20% proof success versus 16% for the next-best retriever and 4% without retrieval. These are paper-specific results and must not be numerically compared as though they were obtained on the temporal track. They justify retaining a strong, pinned premise-retrieval arm. [Paper and artifact link](https://arxiv.org/abs/2605.13137).

## Track design

### T1 — chronological replay on a frozen repository

Use a separately pinned Dalek Lean Bench snapshot as the first implementation target. Record the upstream commit, timeline digest, each target's source commit and theorem locator, the toolchain/Mathlib/Aeneas revisions, and hashes of the generated repository state.

For a target at historical position *i*:
1. Build a clean worktree from the exact historical state and verify the commit and working-tree digest.
2. Retain only proofs and declarations available by that point. Replace the target proof and all later timeline proof bodies with the benchmark's registered holes; do not remove definitions needed to parse the repository.
3. Build and run a negative-control check proving that later-only declarations cannot be referenced successfully from the target.
4. Construct the retrieval index and insight library only from material available at or before position *i*. Record every retrieved item's source commit and provenance.
5. Run all arms under a matched per-target cap for model calls, tokens, wall-clock, and cost. Save stdout/stderr, tool events, build/test output, model identifiers, and exact receipt hashes.
6. Restore the clean worktree before the next target. The evaluation harness itself must be deterministic even when the solver is stochastic.

Don't use a generated full-proof-recovery snapshot as the only setting: its context lets targets see declarations from the final library. Run the chronological setting as the primary temporal score, and use full-recovery only as a secondary, separately labelled diagnostic.

### T2 — matched experimental arms

| Arm | Context/action | Question answered |
|---|---|---|
| T0 | No retrieval beyond the mandatory target-local context | Basic proof-search baseline |
| T1 | Lexical/AST retrieval from historically available artifacts | Benefit of simple retrieval |
| T2 | Pinned strong global-premise/dependency-aware retrieval | Whether PITB beats a serious retrieval baseline |
| T3 | Typed insight contract only; no source proof text | Incremental value of abstract strategy |
| T4 | Same typed insight plus the exact source proof/context used by the registered proof-context arm | Whether strategy and proof context are complementary |
| T5 | HDC/vector or graph-reranking ablation, same candidate/token cap | Representation ablation |
| T6 | Matched negative-control insights with deliberately violated preconditions | Whether the compatibility gate prevents harmful transfers |

The primary contrast for this track must be preregistered before running any eligible temporal test target: T3 versus whichever of T1/T2 wins on an earlier validation time window. T4 and T5 are secondary. T6 is a safety gate, not a competing primary score. The current PITB v0 primary contrast remains unchanged.

### Chronological data partition and contamination controls

- Partition by time blocks, not random theorem rows: earlier block for insight authoring/training, the next block for design decisions and baseline selection, the final untouched block for confirmation. The final blocks must be defined from the pinned timeline before running any model.
- Keep same-PR, same-file, renamed-theorem, statement-equivalent, and dependency-descendant clusters together when they could reveal the same mathematical result.
- Hash exact theorem statements and normalized syntax/AST fingerprints; use dependency-graph overlap to flag likely leakage. Human reviewers may adjudicate potential duplicates without seeing model outcomes.
- The target's original proof is sealed from the system. It is used only by the external checker/score harness to define the hidden target and the chronological boundary.
- Any missing historical state, non-buildable upstream commit, or unresolved proof-hiding mutation becomes an explicit excluded/unknown row with a reason. Never silently drop difficult or unavailable targets.
- Use only repository-licensed source and follow every upstream license and attribution requirement. The challenge package must store source citations/digests instead of copying restricted source text unless redistribution is explicitly permitted.

## Outcomes and proposed project targets

The sole positive outcome is the exact target theorem passing the pinned Lean checker at its original position **and** passing the registered semantic/dependency/axiom policy. A declaration that compiles in a later, richer environment is not a success for the temporal track.

| Metric | Proposed acceptance/measurement |
|---|---|
| Primary treatment utility | T3 improves exact-target semantic-gate success by at least +10 percentage points over the earlier-window-selected T1/T2 baseline; the family/project-clustered 95% interval is wholly above zero |
| Statistical adequacy | Simulate the complete decision rule using validation-derived project/family sizes, report cluster-ICC sensitivity and Monte Carlo confidence bounds; no fewer than 20 independent clusters even to consider confirmation |
| Future-proof leakage | 100% of frozen future-proof-use mutations and hidden-later-declaration controls must be detected |
| Negative-transfer controls | 100% rejection of the frozen finite set with wrong object, polarity/direction, quantifier scope, missing prerequisite or unavailable dependency |
| Positive controls | Report compatible-positive recall and the false-rejection rate. A reject-everything system must fail this gate |
| Reproduction | Clean replay at the same pins reproduces every timeline target boundary, hidden-proof manifest and target input digest |
| Cost-effectiveness | Report verified solves per dollar and per wall-clock hour, tokens per verified solve, median/p95 latency and model/tool calls by arm |
| Explanation quality | Blinded annotation of whether each typed insight correctly states source move, preconditions, target obligation, and failure boundary; report reviewer agreement separately from proof outcomes |
| External validity | Replicate on a second, independently pinned repository before claiming general state-of-the-art transfer |

The thresholds above are **proposed project gates**, not established industry standards and not measured results. Freeze the eligible sample, annotator rubric, target blocks, exact retrieval implementations and budgets before a confirmatory run.

## What would count as a real paradigm shift?

1. **Causal strategy transfer rather than proof similarity.** If T3 wins against T1/T2 with equal budget, a proof insight has demonstrated incremental utility beyond returning similar premises or showing the proof.
2. **Historically valid knowledge.** A strategy may be used only if its source evidence existed at the target time. This lets the benchmark detect temporal leakage that a conventional random split cannot reveal.
3. **Proof obligations as the interface.** A strategy is an executable contract over object/domain, relation direction, preconditions, obligation change and dependency effects. It is a typed hypothesis, never a trusted correctness claim.
4. **Safety is measured, not implied.** The system must identify when analogies fail and abstain on incompatible contracts; verified proof success and rejection-quality are distinct outcomes.
5. **Useful work beats raw pass rate.** Publish incremental verified solves and their uncertainty per unit cost, including failed attempts and timeouts, not just a headline pass percentage.

## Implementation sequence

1. Pin a clean Dalek Lean Bench commit, license metadata, Lean toolchain and timeline CSV digest; first add a dry-run verifier that proves each target's pre/post timeline boundary without making model calls.
2. Reproduce a small chronological slice with zero model calls, exact proof-hiding checks, and deterministic receipts. Verify that removing the target proof does not change the theorem statement or unrelated build context.
3. Add a source-index adapter and per-target historical-availability filter. Assert fail-closed behavior for future commits, missing commit metadata, unparseable targets and stale indexes.
4. Add T0–T6 orchestration behind an explicit track identifier. All arms consume the same target manifest and per-target budgets; only the declared context/treatment varies.
5. Freeze the early/validation/final time blocks, baseline selector and power report. Run a low-cost pilot only after the harness's synthetic future-proof controls pass.
6. Report all excluded/unknown targets and all arm denominators. No broad SOTA claim until the registered gate passes and a second independent repository reproduces it.

## References

- [Dalek Lean Bench repository](https://github.com/Beneficial-AI-Foundation/dalek-lean-bench) — chronological proof replay and full-proof-recovery distinction.
- [VeriSoftBench project page](https://utopia-group.github.io/VeriSoftBench/) — repository-scale formal verification and context-curation baseline.
- [LeanSearch v2](https://arxiv.org/abs/2605.13137) — global premise retrieval and controlled downstream comparison.
- [PITB v0 evaluation protocol](../../data/benchmarks/proof-insight-transfer-v0/evaluation-protocol.md) — frozen primary experiment, arm definitions, and admissible claims.
