# Connectome multi-rater measurement protocol V1

**Status:** executable measurement contract; synthetic fixture only; no biological outcome corpus has been evaluated.  
**Tracking:** [SCI-CONNECTOMICS-004 / issue #9186](https://github.com/Luminous-Dynamics/symthaea/issues/9186).

## Purpose

The previous post-hoc evaluator stores one reviewer decision and one adjudication per candidate. That collapses disagreement and cannot distinguish an ambiguous candidate from a confident reviewer error. This protocol retains each independent vote and measures agreement, adjudication uncertainty, reviewer confidence calibration, and panel effort.

The new `connectome-multi-rater-audit` binary is isolated from `connectome-query-impact`. It consumes the exact proposal input, exact frozen ranking report, and a separate evaluation-only outcome file. It does not rank, propose, select, or mutate graph annotations.

A directly relevant comparator is Lee et al., *Beyond Agreement: Standardizing Crowdsourced Synapse Annotations through Proofreading in EM Connectomics* (bioRxiv preprint, 2025; updated 2026): five raters (one expert plus four calibrated annotators) reviewed model-proposed synapse candidates, and the study used a probabilistic Dawid–Skene model to retain consensus uncertainty. The authors report that calibration improved individual agreement with the expert, while chance-corrected agreement gains were modest and dataset-dependent. This supports retaining raw votes and uncertainty; it does not imply that this project has reproduced the method or achieved its results. https://doi.org/10.1101/2025.09.26.678851

## Data contract

For every candidate in the frozen proposal pool, record at least two adjudicator votes and at least two reviewer reviews.

- Adjudication labels: `correct`, `incorrect`, `indeterminate`.
- Reviewer actions: `accept`, `reject`, `abstain`.
- IDs are pseudonymous and stable within a study. Reviewer and adjudicator ID sets must be disjoint.
- Reviewer confidence is the probability-like self-assessment that their accept/reject action is the correct action. It is in [0,1] for accept/reject and null for abstain.
- Reviewer time is measured elapsed minutes for that review event. It is not model-estimated effort.
- The outcome file binds to exact proposal bytes, exact ranking-report bytes, and an independently versioned adjudication protocol SHA-256.

The executable validator checks full candidate-pool coverage, duplicate/orphan IDs, role separation, vote counts, finite positive time, confidence/action consistency, and exact byte digests. The JSON Schema is a companion contract; executable validation remains authoritative for cross-file checks.

## Consensus and agreement semantics

A candidate consensus is emitted only when one category has a **strict majority** of all votes. A tie or a plurality without a strict majority remains unresolved. `indeterminate` is a valid adjudicator category, not a synonym for missing data.

The audit reports separate nominal Krippendorff alpha estimates for reviewer decisions and adjudicator labels using the finite-sample coincidence formulation. For candidate i with n_i ratings and d_i disagreeing unordered pairs, observed disagreement is Do = sum_i(2*d_i/(n_i-1))/N, where N is the total number of ratings. Expected disagreement is De = 2*D/(N*(N-1)), where D is the number of disagreeing unordered pairs in the pooled category counts. Alpha is 1 - Do/De and is not estimable when there are no within-candidate pairs or expected disagreement is zero. This calculation and its denominators are emitted explicitly.

The tool also reports:
- per-candidate raw vote counts, strict-majority labels, majority fractions, reviewer consensus correctness when determinable, and total/mean panel review minutes;
- pooled and per-reviewer accept/reject/abstain counts, actionable decision accuracy, accepted-decision precision, false-accept rate among incorrect proposals, correct-proposal acceptance rate, abstention rate, and time;
- confidence Brier score and 10-bin expected calibration error for accept/reject reviews whose independent adjudication consensus is determinate.

Confidence scores are not evaluated for abstentions or unresolved/indeterminate gold. Indeterminate and unresolved candidates are not counted as correct or incorrect. Reviewer agreement does not establish biological truth; adjudicator agreement does not either. The independent adjudication panel remains a separate reference process.

## Blinded collection protocol

1. Freeze the proposal pool, ranking report, candidate evidence, query bundle, task instructions, review UI build, and protocol SHA before any reviewer sees a candidate.
2. Blind reviewers to ranking policy, model score, candidate priority, other reviewers' decisions, and adjudicator outcomes. Randomize candidate order within reviewer and balance order across reviewers.
3. Assign at least three independent adjudicators per candidate where feasible. Adjudicators should be blinded to the proposed policy and reviewer decisions. Two adjudicators are the schema minimum, not the recommended target: a 1–1 split is correctly unresolved.
4. Calibrate annotators on a separate training set with explicit criteria and anchor examples. Never leak calibration answers into the lockbox.
5. Preserve every raw vote, abstention, confidence, review duration, reviewer ID, adjudicator ID, evidence digest, and protocol version. Do not overwrite disagreements with a majority label.
6. Route unresolved/indeterminate cases to a documented adjudication/arbitration procedure. Keep original votes and append the arbitration result as a new, separately identified record in a future schema revision; do not back-edit source votes.
7. Report the cost to construct the full blinded corpus separately from policy review budgets. Panel minutes are the sum of independent reviewer events, not wall-clock elapsed time.
8. Use a separate prospective randomized or blocked queue trial to estimate causal effects of policy order, learning, fatigue, and reviewer specialization. This retrospective audit is not that trial.

## Benchmark targets and kill criteria

- Report reviewer and adjudicator alpha with uncertainty intervals at the candidate/volume dependence level once an adequately sized biological corpus exists. Do not set a high reviewer-alpha threshold as a success condition: difficult scientific cases may have genuine ambiguity.
- For the adjudication panel, target a predeclared adjudication protocol with at least 90% of candidates either reaching a strict-majority determinate label or explicitly routed to arbitration; publish the unresolved fraction rather than dropping it.
- On determinate held-out items, report Brier score and ECE against a constant/base-rate confidence baseline. Require non-inferior or improved calibration before using confidence to control abstention or automation.
- Do not authorize automatic graph writes from reviewer consensus. V1 remains review-only, and high-impact edits require independent provenance and validation.
- If reviewer identity or outcome leakage reaches the ranker, if the held-out candidate pool is filtered after votes are inspected, or if unresolved cases are silently removed, invalidate the benchmark run.

No SOTA win is claimed. A probabilistic latent-rater model (for example, Dawid–Skene) is a future candidate only after this raw-vote contract and independent anchor cases exist. The current implementation deliberately uses transparent strict-majority summaries and nominal alpha rather than pretending a latent-class model can manufacture ground truth.

## Run

```bash
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome_multi_rater_audit -- \
  --proposals frozen-proposals.json \
  --ranking-report frozen-ranking-report.json \
  --outcomes blinded-multi-rater-outcomes.json \
  --output multi-rater-audit-report.json
```

The example file contains synthetic placeholder digests; replace them with exact file SHA-256 values before use.
