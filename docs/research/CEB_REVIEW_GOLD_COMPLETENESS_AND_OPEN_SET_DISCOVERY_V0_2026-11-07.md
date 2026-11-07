# CEB Review Gold-Completeness and Open-Set Discovery Protocol V0 — 2026-11-07

**Status:** Public research/design protocol. Does not create a task, gold label, hidden holdout or evaluation result.  
**Owner:** Existing CEB-REVIEW / CEB-EVAL / causal-diagnosis ownership; #10419, #10723, #10367, #8311, #9431 and #10680/#10681.  
**Primary CEB metrics:** unchanged — `defect_recall_at_5`, `actionable_precision`, `reviewer_burden`.  
**Execution boundary:** candidate-controlled execution remains blocked until #10064; independent pristine regrade remains blocked until #10369.

## 1. The unresolved measurement problem

Every defect benchmark has a discovery process. If the "gold" set is only the union of issues already found by human reviewers, agents, static tools or a finite group of model reviewers, then a genuine defect missed by every gold producer is invisible. A candidate that discovers it cannot be credited by fixed-gold recall unless the benchmark has a separate, defensible route for independent adjudication.

Two contemporary benchmarks disclose this limitation:

- **CodeReviewBench** uses 30 merged PRs and 95 human-authored golden bugs. Its maintainers explicitly state that bugs shipped because no human noticed them are absent; the benchmark measures recovery of human-flagged defects, not exhaustive defect discovery. Its strongest current result is DeepSeek V4 Pro at 44.2% recall / 43.6% precision, with a 95% bootstrap interval of 32.3–56.3 recall points over 30 PRs. [Leaderboard and per-model scorecard](https://www.codereviewbench.com/leaderboard) · [maintainer discussion of gold coverage](https://www.codereviewbench.com/blog/ai-code-review-recall-2026).
- **ReviewBench** builds gold from finite heterogeneous producers and explicitly states that genuine findings missed by all producers remain absent. Its augmented metrics classify unmatched candidate findings, but augmented recall has an agent-specific denominator and is not a common ranking metric. [Methodology and threats to validity](https://github.com/review-bench/ReviewBench/blob/main/docs/METHODOLOGY.md).

This is not a criticism of those benchmarks: finite gold construction is necessary and useful. It is a reason for CEB to report **gold coverage uncertainty** separately from candidate performance.

## 2. Paradigm shift: gold is an evidence frontier, not an oracle by declaration

CEB should distinguish three nested sets:

1. **Frozen target set:** historically or independently established defects used for the strict, comparable target-recall calculation. Its membership and denominator are frozen before candidate review.
2. **Adjudicated open-set discoveries:** candidate findings not in the frozen target set that are subsequently confirmed by blinded independent reviewers against the exact source revision and frozen requirements. These remain separate from the frozen target score.
3. **Unobserved defect remainder:** the set of potential defects not found by any producer or audit. It is unknown and must remain explicitly unknown unless a defensible coverage-bound method is available.

Do not call the frozen target set exhaustive merely because it has multiple producers. Do not automatically promote a candidate finding into gold. Do not infer no defects from an empty review. Do not use candidate-dependent novel true positives to expand a cross-agent recall denominator.

## 3. Two-channel scoring and adjudication

### Channel A — Fixed-gold performance

- Freeze target identity, causal mechanism, source tree, introducing change, relevant requirement, repaired companion, severity and provenance before model evaluation.
- Report the existing CEB `defect_recall_at_5` and `actionable_precision` under their original definitions.
- Keep target hit, target miss and unresolved target-label dispositions distinct.
- Never change the frozen target set because one candidate finds or misses a result.

### Channel B — Open-set discovery

- Capture all candidate findings, including findings that do not match a frozen target.
- Deduplicate by semantic defect identity, not lexical similarity alone.
- Have two independent blinded experts adjudicate every critical/high-severity unmatched finding and a predeclared stratified sample of other unmatched findings.
- Reviewers see the exact source revision, applicable requirements and evidence anchors, but not the candidate/model identity or its proposed match where blinding is feasible.
- Dispositions: independently confirmed new defect; valid non-defect suggestion; duplicate; unsupported/noise; out of scope; unresolved.
- A confirmed new defect is a discovery result for that candidate and becomes a candidate for future task development only after separate task admission and custody review. It does not retroactively enter the current frozen denominator.
- Report adjudicated novel-defect count, precision among sampled unmatched findings, audit sample sizes and uncertainty. Do not call an unreviewed unmatched finding a true positive.

### Channel C — Gold-completeness audit

Independently sample PRs where:
- all candidate reviewers produced no findings;
- every candidate reviewer agreed on "clean";
- no original human review comment exists;
- the original human-review set is sparse relative to change size/risk;
- static/test/author-follow-up evidence indicates a possible latent issue;
- only one producer family contributes a large share of gold findings.

A blind audit team should investigate these PRs without seeing the target labels or candidate output. Use source history, frozen specifications, tests/property contracts, static-analysis traces and controlled counterexamples where authorized. This audit estimates missed-gold prevalence in the sampled strata; it does not establish exhaustive gold completeness for every PR.

## 4. Producer independence and common-mode blind spots

For every gold producer, record:
- model/provider/version or human reviewer role;
- prompt, harness, tools, retrieval context and allowed evidence;
- whether the producer saw human comments, target/fix commits, reference patches or another producer's output;
- repository/task/source-family overlap;
- whether a producer was later used as candidate evaluator or judge.

Build a producer-dependence graph. Shared model families, training data, prompts, static-analysis rules, repositories and source artifacts are dependence edges. A model ensemble is not independent just because it has multiple names.

Use leave-one-producer-family-out sensitivity analysis:
1. recompute which gold findings remain discoverable when one source family is removed;
2. report the fraction of gold targets uniquely contributed by each source family;
3. re-score candidates against each declared sensitivity gold set as a robustness diagnostic;
4. do not choose the most flattering gold composition after seeing scores.

A naïve capture–recapture estimate of unseen defects is **not permitted**. The independence and homogeneous-capture assumptions are implausible when reviewers share context, model families and defect salience. Capture–recapture may be explored only as a sensitivity analysis with explicit dependence models and no claim that it estimates the true total defect count.

## 5. Blind adjudication and evaluator controls

- Separate original defect provenance, candidate finding-to-target matching, open-set validity adjudication and gold-completeness audit.
- A matcher can propose identity; it cannot establish truth.
- LLM judge agreement is not independent corroboration when judges share model/provider/prompt/data lineage.
- Audit false matches and missed matches across accepted, rejected and unresolved strata.
- Preserve per-judge outputs and reviewer dissent.
- Human adjudicators must be blind to candidate identity, arm, first-pass labels and gold producer where feasible.
- If full blinding is impossible, record why and stratify results.
- An unresolved finding is neither a false positive nor a true positive until evidence resolves it.
- Reuse #8311 oracle qualification, #9431 correlated-judge work and #10680/#10681 verifier-of-verifiers qualification. Do not create another oracle authority or judge registry.

## 6. Source-state and causal checks

For a candidate finding claimed to be a new defect:
- bind the claim to the exact source revision and frozen requirement;
- reconstruct trigger → path/state → unsafe operation → violated invariant;
- examine the parent, introducing and repaired source states where available;
- test nearest neighboring mechanisms and alternative explanations;
- classify source-only inference separately from compiler/runtime/intervention observation;
- do not infer that a later fix confirms the candidate explanation unless the fixing change is causally relevant and independently reviewed.

This protocol composes with the P/I/F/N/M/IRR causal-diagnosis preregistration. The open-set audit must not use model chain-of-thought; it evaluates evidence and concise falsifiable claims.

## 7. Sampling and uncertainty rules

- Sample by PR/repository and risk stratum, not by individual comment alone.
- Oversample zero-finding PRs and single-producer gold strata for completeness audits; preserve inclusion probabilities and report weighted estimates.
- Keep the development audit, calibration sample and sealed confirmation sample disjoint.
- Select audit sample sizes using a preregistered precision target and pilot-derived prevalence/cluster design effect.
- Use repository-clustered confidence intervals or a hierarchical model. Do not treat multiple comments on one PR as independent samples.
- Report the number of PRs, findings, repositories, defect families, human reviewers and unresolved labels.
- Report both macro-by-PR and micro-by-finding summaries.
- Do not publish a "gold completeness %" without a defined target population, audit sampling design, adjudication sensitivity and uncertainty interval.
- Keep the unobserved remainder unknown outside the audit's population and confidence bounds.

## 8. Integration with existing CEB measures

No new primary metric is introduced.

- `defect_recall_at_5` remains fixed-gold top-five target detection.
- `actionable_precision` remains blinded actionability among submitted findings under the existing frozen rubric. Open-set candidate findings enter this numerator only after independent adjudication; target status remains separately recorded.
- `reviewer_burden` includes audit and adjudication time when comparing assisted workflows, not just candidate review generation time.

Supplementary reports (novel confirmed defects, sampled unmatched-finding precision, no-finding audit miss rate, producer-family dependence and gold-set sensitivity) are diagnostic analyses, not additional CEB primary metric IDs.

## 9. Failure controls

| Control | What it catches |
|---|---|
| Candidate discovers a valid defect absent from frozen gold | Fixed-gold incompleteness without corrupting denominator |
| Candidate finding is plausible but not independently verified | Candidate-generated truth |
| All candidate reviewers agree on a clean PR but blind audit finds a defect | Common-mode omission |
| Gold finding comes only from one model family | Producer dependence |
| Same-family judge agrees with reference generator | False independence |
| One producer removed and gold recall changes sharply | Gold composition sensitivity |
| Candidate writes more comments and expands its own novel-TP denominator | Agent-dependent augmented-recall gaming |
| Empty review / no comments | Incorrect inference of clean status |
| Repaired companion has other defects | Target-specific control misread as whole-PR safety |
| Incomplete evidence for a new defect | Forced binary labels instead of unresolved |

## 10. Runner-independent development plan

1. Apply the protocol to the 26 public synthetic ERG scenarios; label which source state and evidence class are needed to decide each case.
2. Create an example where every visible reviewer misses a defect and a blind auditor discovers it; keep it synthetic and source-only.
3. Create an example where a candidate proposes a non-target finding that is real but requires separate expert adjudication.
4. Draft the producer-dependence graph and leave-one-family-out report template.
5. Define stratified audit inclusion probabilities and the pilot-derived sample-size procedure.
6. Map output fields to the existing adjudication card without adding a new task/receipt schema.
7. Obtain #8311/#9431/#10680/#10681 and statistical owner review before considering task admission.

All examples remain public development materials, permanently ineligible as sealed tasks.

## 11. Claim boundary

This protocol does not establish exhaustive benchmark gold, estimate the total number of defects in any repository, qualify an evaluator, admit tasks or produce model scores. A blind audit can estimate a sampled population's missed-defect prevalence only within its declared sampling design and uncertainty. Unobserved defects outside that scope remain unknown. Candidate-controlled execution is prohibited until #10064 qualifies isolation; independent pristine regrade remains blocked until #10369 qualifies signer trust and runner attestation.
