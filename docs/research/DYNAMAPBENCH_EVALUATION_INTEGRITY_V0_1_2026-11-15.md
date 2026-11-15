# DynaMapBench evaluation integrity protocol — 2026-11-15

**Status:** proposed evaluation protocol. No benchmark runs or comparative scores are claimed.

## 1. Why evaluator integrity must be a first-class research track

A development-intelligence benchmark can be wrong even when its candidate system is excellent. Invalid tasks, hidden future information, correlated gold labels, contaminated agent environments, evaluator leakage and non-independent grading can dominate the reported score.

SWE-Bench Pro V2's September 22, 2026 update is a useful example of the level of scrutiny expected for serious engineering evaluation:
- its public split was reduced from 731 to 642 tasks after 89 were judged invalid;
- 69 task instructions contradicted their grading tests and were corrected through blinded expert review;
- the agent phase disabled web tools to reduce access to fixing commits;
- every diff was regraded in a pristine image; and
- the release applied a two-sided task gate: the reference patch must pass and the empty patch must fail.

These are not cosmetic details. They change the meaning and credibility of the score. See the [SWE-Bench Pro V2 release and locked protocol](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full).

## 2. DynaMapBench should have two independent gates

### 2.1 Task validity gate

Every task must have:
- exact repository commit, root tree and relevant blob identities;
- a time-consistent pre-task information cutoff;
- a task statement independently judged to be answerable from permitted evidence;
- a reference outcome that satisfies the acceptance contract;
- a negative/empty control that does not satisfy the same contract;
- an explicit configuration world and execution environment;
- an independently curated relevant/forbidden/unknown evidence set; and
- a documented reason for exclusion when the task fails validation.

A reference patch passing is not enough: the empty patch must fail. A task that passes without any intervention is not a discriminating engineering task.

### 2.2 Evaluator integrity gate

The evaluator must be separately built, pinned and administered:
- candidate context cannot read gold labels or grader credentials;
- network access is disabled unless the benchmark explicitly evaluates network-dependent behavior;
- task environments are immutable and reset for every candidate;
- every output is regraded in a clean environment;
- evaluator and corpus digests are pinned before the run;
- scorer identity and version are included in the result receipt;
- candidate and evaluator do not share parser/reducer code where an independent oracle is required; and
- an auditor can reproduce the result from the pinned artifacts.

The independent temporal replay crate added in D2 reduces common-mode code risk, but because it is authored within the same repository it does not satisfy independent administration or blind label custody.

## 3. Time-consistent repository evaluation

Use the ATime-Consistent design principle: freeze the repository and all derived code knowledge at T0, then derive tasks only from changes in the later interval (T0,T1]. The candidate must not receive post-T0 source, issue comments, PR metadata, test output, model context or labels unless the task explicitly asks a retrospective question with a later knowledge cutoff.

SWE-Time adds an important complement: retrospective regression analysis and prospective intended-change reasoning should both be tested. DynaMapBench D2 must keep actor availability and ledger-recorded cutoffs distinct from the repository's effective/valid time.

References:
- [ATime-Consistent (2026)](https://arxiv.org/abs/2603.26137)
- [SWE-Time (2026)](https://arxiv.org/abs/2611.12083)

## 4. No single benchmark score

DynaMapBench must publish separate scorecards for:
- source identity and extractor correctness;
- temporal reconstruction and actor-knowledge leakage;
- retrieval coverage and budget efficiency;
- evidence acquisition and EVSI regret;
- change-impact precision/recall;
- causal intervention prediction and calibration;
- act/defer and hard-gate compliance;
- longitudinal maintenance outcomes; and
- evaluator/task validity.

A critical false acceptance, unauthorized action or future-evidence leak must remain visible as a hard failure; it cannot be offset by a high retrieval or task-completion score.

## 5. Result receipt

Every benchmark run should produce a signed or independently witnessed result receipt containing:
- benchmark ID and exact version;
- corpus digest and task-family split;
- candidate commit/build identity;
- evaluator commit/build identity;
- model and agent harness identity;
- execution environment and configuration-world digest;
- input cutoff and source snapshot identity;
- task count, valid/invalid/excluded counts and reasons;
- each score, denominator and uncertainty interval;
- cost, latency, tokens, compute and reviewer time;
- negative-control results;
- artifact digests for candidate outputs and clean regrades; and
- a claim ceiling describing what the run does not establish.

For security-relevant work, add an external transparency witness. Sigstore Rekor's transparency-log model is useful precedent: append-only claims are auditable through verifiable data structures and independent consistency witnesses. A local SHA-256 chain is not a substitute for external witnessing or source authentication. See [Sigstore Rekor logging overview](https://docs.sigstore.dev/logging/overview/) and the [Rekor transparency-log specification](https://github.com/sigstore/architecture-docs/blob/main/rekor-spec.md).

## 6. Prospective acceptance targets

These are DynaMapBench research targets, not industry standards or achieved scores:
- zero acceptance of planted future-evidence, wrong-repository and unauthorized-source controls;
- zero tasks admitted when reference/empty-patch validity gates fail;
- 100% replay equivalence across frozen deterministic D2 fixtures;
- at least 95% independently adjudicated source/retrieval precision and recall;
- benchmark evaluator agreement of at least 99% against a separate oracle on deterministic synthetic tasks;
- publish exact denominator and confidence intervals for every zero-error claim; and
- no aggregate score that hides critical safety, temporal or authority failures.

A zero-error result on a small negative-control set does not prove zero risk. Report the denominator and a confidence bound, and expand adversarial coverage.

## 7. Immediate work items

1. Extend DynaMapBench manifest with reference-pass/empty-fail receipts and immutable clean-regrade identity.
2. Add time-consistent T0/T1 task construction and block future PR/issue/comment leakage.
3. Move gold custody and evaluator administration outside candidate-readable repository artifacts.
4. Add independent evaluator mutation tests and scorer self-tests.
5. Add a signed/witnessed run receipt and verifiable external checkpoint.
6. Keep temporal, acquisition, causal, abstention and longitudinal scorecards separate.
