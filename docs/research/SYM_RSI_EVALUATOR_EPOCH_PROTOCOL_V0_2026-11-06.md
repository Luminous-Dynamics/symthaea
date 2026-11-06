# Symthaea RSI Evaluator Epoch Protocol V0

**Status:** design-only; no promotion authority or evaluator service is implemented by this document.  
**Date:** 2026-11-06  
**Owner:** SYM-RSI-EVAL-EVOL-001 (#10708)  
**Authority:** independent evaluator service in #10573

## 1. Purpose and non-goals

This protocol specifies how Symthaea may improve evaluators without allowing a candidate and its judge to co-adapt into a shared failure mode.

It adds an epoch boundary around evaluator changes:

- evaluator criteria are immutable during candidate search inside an epoch;
- evaluator challengers are compared against independently governed anchors at epoch boundaries;
- a challenger cannot grade or authorize its own adoption;
- historical results retain the exact evaluator version that produced them;
- all epochs retain an evaluator-independent external confirmation lane for capability comparisons.

This is **not** a new benchmark receipt, evidence ledger, signer envelope, REUSE implementation, promotion API, scheduler, or evaluator service. Existing owners remain authoritative:

| Concern | Existing owner |
|---|---|
| Independent candidate promotion and decision-only feedback | #10573 |
| Candidate / source-tree / build / lineage identity | #10569 |
| Canonical run receipts, accounting and attestation/regrade | #10314 / #10533 / #10370 |
| Execution isolation and trust roots | #10064 / #10369 |
| Paired, cluster-aware inference and population uncertainty | #8438 / #9433 / #10513 |
| R4 same-start improver experiment | #10497 |
| D/H/A/M causal decomposition | #10674 |
| Audit-resistant algorithmic RSI | #10691 |

## 2. Threat model

The protocol addresses:

1. **Reciprocal reward hacking:** an agent changes to satisfy an evaluator while an evaluator changes to favor its agent.
2. **Construct drift:** a judge starts rewarding style, verbosity, tool count, confidence or proxy completion instead of the declared capability.
3. **Anchor overfitting:** an evaluator challenger is optimized against the same anchor examples later used to approve it.
4. **Shared blind spots:** multiple learned judges agree because they share training data, architecture, rubric assumptions or a shortcut.
5. **Nonstationary score laundering:** scores from different metric/evaluator versions are concatenated as though they were comparable.
6. **Retrospective decision rewriting:** a new evaluator silently changes the meaning of an old promotion.
7. **Authority confusion:** the candidate, evaluator-improver or development harness gains access to anchor custody, promotion keys or detailed hidden results.

A hash or signature proves identity/integrity of an artifact; it does not prove evaluator validity, anchor independence, construct validity or truthful measurement.

## 3. Roles and trust separation

Four logical roles must remain distinct. They may be implemented by more than one process or organization, but source identity, credentials, mutable state and access permissions must be independently controlled.

### 3.1 Candidate improver

Can inspect development receipts and detailed development results. Can propose target-system changes. Cannot read hidden anchor labels, evaluator-selection outcomes, promotion certificates or the evaluator service's private state.

### 3.2 Evaluator improver

Can propose evaluator implementations, rubric revisions, calibration methods or additional diagnostic dimensions. Cannot:
- modify candidate artifacts or claim their own proposal as an independent grade;
- access hidden anchor labels or test-generation secrets;
- change trusted task-universe membership, anchor custody, trust roots, promotion policy or error allocation;
- write to the accepted evaluator registry.

### 3.3 Independent evaluator authority

Controls anchor custody, task-family sampling, objective graders, expert adjudication, evaluator-challenger comparisons, evaluator-version admission and epoch creation. It also operates #10573's independent promotion boundary.

### 3.4 External confirmation authority

Maintains a fixed evaluator-independent capability measure across evaluator epochs. Examples include deterministic executable tests, proof checkers, physical outcome measurements, externally adjudicated outcomes or a stable, independently governed benchmark. It does not accept the co-evolved evaluator's output as ground truth.

**Authority rule:** no candidate or evaluator-improver process may update the accepted evaluator version, promotion policy, anchor manifest, REUSE error state or external confirmation criteria.

## 4. Anchor hierarchy and independence

Every evaluator challenger must be judged on multiple anchor classes.

### 4.1 Objective anchors

Examples: deterministic executable tests, type/format invariants, theorem checkers, known numerical solutions, constraint satisfaction, source/build identity and reproducible simulation outcomes.

### 4.2 Expert-adjudicated anchors

Examples: research quality, paper review, semantic correctness or engineering judgments where no complete executable oracle exists. Use independent expert annotations, documented adjudication, reviewer disagreement and explicit abstention. An LLM judge may assist triage but cannot silently replace ground truth.

### 4.3 Adversarial and construct-validity anchors

Include style substitution, verbosity inflation, reward hacking, shortcut behavior, distribution shift, false citations, impossible optima, invalid score ranges, subgroup regressions, evaluator disagreement and tasks where abstention is correct.

### 4.4 Split discipline

Keep separate:
- evaluator development examples;
- evaluator-selection anchors;
- evaluator confirmatory anchors;
- candidate development/selection evidence;
- candidate promotion evidence under #10573;
- fixed evaluator-independent external confirmation.

Disjointness is required at the true independent unit (e.g. repository/project/task family/source population), not merely at row or seed level. Correlated task rows, repeated judgments or retries do not become independent samples by receiving different IDs.

Anchor membership and labels remain outside candidate/evaluator-improver workspaces. Public repositories may contain non-sensitive contracts, protocols and synthetic negative controls, but not secret answer keys or hidden task-generation material.

## 5. Epoch state machine

The only allowed high-level state transitions are:

    Draft -> Frozen -> Active -> Closed -> ChallengerReview
                                      |                 |
                                      |                 +-> Reject -> NextEpoch uses incumbent evaluator
                                      |                 |
                                      |                 +-> Admit -> NewEpoch freezes challenger
                                      |
                                      +-> Archive; never reopen or rewrite

### Draft

The independent authority prepares the evaluator build, criterion, anchor commitments, error-budget allocations, non-inferiority margins and evaluation plan.

### Frozen

All versioned fields are content-bound. No in-epoch changes are allowed to evaluator source/build, rubric, metric range, normalizer, task universe, grader, split, decision alphabet or allocation.

### Active

Candidate proposals are assessed only under the frozen evaluator. The proposer receives only the predeclared coarse feedback permitted by #10573. Detailed scores and certificates remain private.

### Closed

The epoch stops accepting candidate evaluations. The complete attempt population and resource accounting are frozen.

### ChallengerReview

A new evaluator candidate is compared with the incumbent using separately reserved evaluator-anchor samples and a fixed adjudication protocol. This is not another candidate-promotion round.

### Next epoch

An admitted evaluator is frozen as the next epoch's incumbent. The previous evaluator and every old result remain immutable. Rejected challengers remain archived as rejected attempts.

An evaluator replacement must never occur in the middle of an active candidate round.

## 6. Error-budget composition across epochs

This protocol must not accidentally reset statistical error budgets at an evaluator change.

Before any adaptive search, the independent authority declares two global budgets:

- alpha_candidate_global: maximum family-wise false-promotion budget for candidate improvements across the approved sequence;
- alpha_evaluator_global: maximum false-admission budget for evaluator-challenger decisions across the approved sequence.

Each epoch and each evaluator-challenger comparison receives a preallocated portion:

    sum_e alpha_candidate_epoch[e] <= alpha_candidate_global
    sum_j alpha_evaluator_challenge[j] <= alpha_evaluator_global

These equations are necessary accounting constraints, **not by themselves a proof** that a particular adaptive procedure is valid. The exact REUSE theorem, indexing, independence assumptions, candidate count, history semantics and feedback restrictions must be reviewed for the implemented service. The evaluator-admission procedure requires its own valid inference plan.

Important boundaries:

- No new branch, restart, experiment ID, evaluator version or changed task name may reset either global budget.
- If the allocation underflows or its history is unknown, fail closed.
- The evaluator-challenger sample must be separate from the candidate promotion sample when required by the proof.
- A theorem proved for a stationary evaluator within one epoch must not be advertised as a theorem over evaluator changes without a separately reviewed composition.
- The external confirmation lane remains the cross-epoch capability comparison authority.

## 7. Evaluator challenger admission

The challenger and incumbent grade the same independently controlled anchor population. Report each dimension separately; do not hide false acceptance or subgroup regression inside an average.

Required evaluator-quality measurements:

- ground-truth agreement and calibrated probabilities;
- sensitivity, specificity, precision/recall and false-accept/false-reject rates;
- Brier score and log score for probabilistic outputs;
- reliability diagrams and calibration by subgroup/domain;
- ECE only when sample size supports meaningful bin estimates;
- adjudication rate and inter-rater/evaluator disagreement;
- construct validity against objective/expert outcome anchors;
- adversarial shortcut and collusion attack success;
- robustness by task family, difficulty and distribution shift;
- coverage and abstention precision/risk;
- evaluation cost, latency, determinism and reproducibility.

An evaluator challenger is admissible only when all preregistered conditions hold:

1. Non-inferiority on every protected objective and construct-validity anchor family under declared margins.
2. A predeclared, independently confirmed improvement in at least one evaluator-quality or resource-cost dimension.
3. No critical security, leakage, subgroup, shortcut or false-accept regression.
4. Replication on a second independent evaluator-confirmation sample.
5. Human review of unresolved expert-grounded disagreement.
6. Exact source/build identity, dependency set, rubric digest and anchor-protocol identity are bound to the challenger.
7. The independent authority signs the admission decision; the evaluator-improver cannot do so.

A better mean score cannot compensate for a material increase in false acceptance or a protected-construct regression.

## 8. Cross-epoch score semantics

Raw scores from different evaluator epochs are different measurements unless shown otherwise.

- Retain evaluator ID, criterion version, task population, normalizer, threshold, grader build and evidence identity with every result.
- Never overwrite old scores or historical decisions after evaluator replacement.
- If a bridge is used, both evaluators grade a shared independently controlled anchor population; publish raw differences, calibration, uncertainty and failure modes.
- Do not assume a monotonic mapping between evaluator epochs.
- Do not merge bridge-normalized scores into one capability trend without a validated measurement-invariance argument.
- Report per-epoch candidate results and the fixed external confirmation result separately.
- A changed evaluator may explain a score discontinuity; it does not establish that candidate capability changed.

### Selective invalidation, not historical erasure

RQGM describes discarding utility records produced by a replaced evaluator and recomputing the active archive ranking, because old evaluator scores must not influence the new evaluator's search trajectory. Symthaea should preserve the same statistical intent while retaining stronger auditability:

1. Keep the original raw evaluation record, evaluator ID, source/build digest, task identity and decision immutable in the audit history.
2. On evaluator replacement, mark every evaluator-dependent derived value (fitness, rank, posterior, clade productivity contribution, selection cache or aggregate) from the displaced evaluator as **stale for active selection**.
3. Recompute active evaluator-dependent values under the new evaluator from newly generated scores or bridge evidence. Never silently reinterpret an old evaluator's score as a score from the new evaluator.
4. Preserve evaluator-independent artifacts (source/build identity, deterministic test outputs, proof-checker results, execution/resource receipts) only when their semantics truly do not depend on the displaced evaluator and their provenance remains valid.
5. Keep historical decisions as they were made. A new evaluator may disagree with an old decision; report the disagreement as a new observation, never rewrite the old decision.
6. Record the invalidation/recomputation dependency set so an auditor can determine exactly which derived values were removed from selection and which independent facts were retained.

This is **selective invalidation**: old utility evidence stops affecting current search, but the historical evidence itself remains available for audit and reproduction. It prevents both stale-ranking bias and silent deletion of unfavorable history.

## 9. Experimental comparison

Run these arms under matched task access, candidate budget, total tokens/compute, evaluator calls, human adjudication budget and stopping rules:

| Arm | Candidate | Evaluator | Purpose |
|---|---|---|---|
| A | Frozen | Frozen | Stationary baseline |
| B | Evolving | Frozen | Agent/harness RSI alone |
| C | Frozen | Evolving | Evaluator quality/cost alone |
| D | Evolving | Epochally evolving | Controlled co-evolution |
| E | Evolving | Unrestricted co-evolution | Negative control only; never authorizes promotion |

Use at least three evaluator regimes:
1. deterministic executable/software grading;
2. research/report/paper-review quality with expert anchors;
3. mathematical proof/formal reasoning grading with proof-checker and expert anchors.

Primary outcomes are the independent external capability vector and false-promotion/false-evaluator-admission behavior. Secondary outcomes include evaluator quality, evaluation cost, adjudication burden, epoch length, bridge uncertainty, candidate yield per resource and G0-G4 transfer.

## 10. Adversarial qualification controls

The evaluator protocol must reject or expose:

- candidate style/verbosity changes that raise judge score without changing correctness;
- evaluator challenger that improves average agreement but fails a protected subgroup;
- candidate-visible hidden task IDs, anchor labels or evaluator-selection outcomes;
- evaluator patch that edits task harness, task split or anchor membership;
- mismatched candidate/evaluator source or build provenance;
- changed score bounds or unreachable declared optimum;
- challenger tuned on the same anchor used for admission;
- retrospective rewriting of prior scores or decisions;
- two co-evolved judges that share a failure mode;
- expert disagreement or abstention silently relabelled as success;
- candidate/evaluator collusion that defeats a learned judge but fails objective/external confirmation.

## 11. Illustrative epoch declaration

The following TOML is a **protocol example only**, not a new receipt schema and not an implemented service contract. Digests and references are placeholders; do not place secret labels, credentials, private keys, raw test seeds or hidden task content in a public manifest.

    protocol_version = "sym-rsi-evaluator-epoch-v0"
    epoch_id = "epoch-0001"
    parent_epoch_id = ""

    [incumbent_evaluator]
    evaluator_id = "evaluator-v1"
    source_revision = "<immutable-source-revision>"
    build_digest = "<verified-executable-digest>"
    criterion_digest = "<frozen-rubric-and-metric-digest>"
    normalizer_digest = "<normalizer-digest>"

    [task_population]
    universe_commitment = "<task-universe-commitment>"
    independent_unit = "task-family"
    development_split_ref = "<authorized-development-manifest>"
    selection_anchor_ref = "<custodied-selection-anchor-reference>"
    confirmation_anchor_ref = "<custodied-confirmation-anchor-reference>"
    external_confirmation_ref = "<evaluator-independent-reference>"

    [authority]
    promotion_service_ref = "issue-10573"
    source_build_provenance_ref = "issue-10569"
    trust_policy_ref = "<independent-trust-policy-reference>"

    [error_budget]
    candidate_global_alpha = 0.05
    candidate_epoch_alpha = 0.01
    evaluator_global_alpha = 0.01
    evaluator_challenge_alpha = 0.002

    [feedback]
    proposer_decision_alphabet = ["promoted", "not_promoted", "insufficient_evidence"]
    detailed_candidate_scores_visible = false
    anchor_labels_visible = false

Values above are illustrative only. The independent authority must set budgets once for the complete campaign and allocate them according to the reviewed proof; copying the example values does not qualify an experiment.

## 12. No-runner implementation tranche

Before #10064 and #10573 are qualified, permissible work is design-only:

- define the evaluator-epoch state machine and role/authority boundaries;
- map epoch metadata to existing canonical evidence/attestation ownership;
- define anchor class and split custody contracts without publishing hidden labels;
- define deterministic static validation rules for declarations;
- specify cross-epoch score-comparability and error-budget semantics;
- author synthetic, non-executable adversarial examples.

Do not execute untrusted candidate code, evaluator patches, repository build scripts or hidden graders on an unqualified runner. Do not connect evaluator admission to promotion/deployment authority.

## 13. Acceptance checklist

- [ ] Every epoch is frozen and content-bound before candidate search.
- [ ] Candidate, evaluator-improver, evaluator authority and external confirmer have separate roles and access.
- [ ] Anchor populations are independent at the task-family/project/source level.
- [ ] Objective, expert-adjudicated and adversarial anchors all exist.
- [ ] Evaluator admission has preregistered non-inferiority, superiority and false-acceptance criteria.
- [ ] No detailed evaluator score is revealed to the candidate during adaptive search.
- [ ] Global candidate/evaluator error budgets cannot reset across epochs, restarts or branches.
- [ ] Cross-epoch scores are versioned; original decisions are immutable.
- [ ] A fixed evaluator-independent confirmation lane measures actual candidate capability.
- [ ] At least two evaluator versions and the frozen/evaluator-only/co-evolving controls are compared.
- [ ] Collusion, shortcut, leakage, subgroup and score-range negative controls are demonstrated.
- [ ] Exact-head static checks and independent review pass before any service implementation is considered qualified.

## 14. Claim boundary

A better evaluator is not itself a better agent. A co-evolved agent and evaluator agreeing with each other is not evidence of correctness. Changing the utility makes scores nonstationary. Until the independent authority, anchor custody, epoch error composition, provenance and external confirmation are qualified, this document is a design protocol—not a guarantee, a working promotion service, or evidence of RSI/SOTA improvement.

## References

- RQGM, *The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators* (2026): https://arxiv.org/abs/2606.26294
- REUSE, *Which Self-Improvements Should We Trust?* (2026): https://arxiv.org/abs/2609.33180
- RSI-Env (2026): https://arxiv.org/abs/2611.01736
- RSIBench protocol: https://harness-rsibench.com/docs/
- FIRE-Bench (ICML 2026): https://proceedings.mlr.press/v306/wang26d.html


## 15. Conservative cross-epoch composition: conditional union-bound proof obligations

The phrase "no cross-epoch guarantee" should not be read as saying composition is impossible. A conservative composition may be possible under stronger epoch-separation conditions. The following is a **sufficient-condition sketch**, not a proof that the REUSE theorem, evaluator-admission procedure or a future service already satisfies these conditions.

Let \(H_{e-1}\) denote the complete history before candidate epoch \(e\): previous candidate proposals and coarse decisions, accepted evaluator versions, evaluator-admission outcomes, development information, candidate-generator state, and all other variables that can shape the next candidate or evaluator.

Let \(F_e\) be the event that epoch \(e\) makes at least one false candidate promotion. Suppose:

1. The independent authority freezes evaluator \(V_e\), its rubric, task population, grader, normalizer, decision alphabet and error allocation **before** accessing candidate-confirmation sample \(S_e\).
2. \(S_e\) is fresh and independent of \(H_{e-1}\), evaluator-challenger anchor data, development/selection inputs and all candidate/evaluator-improver randomness, except for the permitted within-epoch decision-only REUSE feedback.
3. The within-epoch service enforces the exact REUSE assumptions and can establish the conditional guarantee
   \[
   \Pr(F_e\mid H_{e-1})\leq \alpha_{c,e}
   \]
   almost surely for every admissible prior history. A marginal guarantee averaged over histories is not sufficient for this step.
4. Every attempted epoch is included in the campaign accounting. Restarts, branches, evaluator IDs, experiment IDs and rejected/failed epochs cannot reset or erase the allocation.
5. Evaluator-admission event \(G_j\) for challenger \(j\) is governed by its own fresh anchor sample, disjoint from candidate-confirmation samples, and has a separately established conditional false-admission bound
   \[
   \Pr(G_j\mid K_{j-1})\leq \alpha_{v,j}
   \]
   for the full prior evaluator-admission history \(K_{j-1}\).

Then, by the tower property and the union bound:

\[
\Pr\left(\bigcup_e F_e\right)
\leq \sum_e \Pr(F_e)
= \sum_e \mathbb{E}\left[\Pr(F_e\mid H_{e-1})\right]
\leq \sum_e \alpha_{c,e}.
\]

Similarly:

\[
\Pr\left(\bigcup_j G_j\right)\leq \sum_j \alpha_{v,j}.
\]

Consequently, the probability of either kind of event is bounded by the sum of the two global allocations:
\[
\Pr\left(\bigcup_e F_e\;\cup\;\bigcup_j G_j\right)
\leq \sum_e \alpha_{c,e}+\sum_j\alpha_{v,j}.
\]

This is deliberately conservative and may spend more error budget than a joint proof. It makes no claim that the two event families are independent.

### What this sketch does and does not establish

- It shows why a fresh, independent candidate-confirmation sample per evaluator epoch is the straightforward safe route.
- It shows why an evaluator challenger must be chosen using a separate anchor sample and frozen before the next candidate-confirmation sample is touched.
- It explains why simply splitting alpha across labels while reusing the same adaptive holdout or admitting an evaluator based on candidate-promotion outcomes does **not** meet the sufficient conditions.
- It does not prove the within-epoch conditional REUSE guarantee; that must be established by #10573's implementation and theorem review.
- It does not prove evaluator-admission validity; #10708 must preregister and qualify that inference separately.
- It does not prove measurement invariance or make raw scores comparable across epochs. The fixed external confirmation lane remains necessary.
- It may be operationally expensive because each epoch needs fresh candidate-confirmation evidence. Reusing the same benchmark population across evaluator epochs requires a different proof and must not be assumed safe.

### Required review questions

Before adopting this composition:
1. Does the exact REUSE theorem establish a bound conditional on arbitrary admissible prior history, or only under a narrower experiment model?
2. Does candidate-generator state contain any information derived from the next epoch's confirmation sample beyond allowed decisions?
3. Can evaluator selection, calibration, rubric revision or anchor curation leak information from \(S_e\)?
4. Are candidate and evaluator familywise error budgets both frozen at the start of the campaign and reconciled on every restart?
5. Are task-family/project clusters the actual independent units, with no hidden overlap across epoch samples?
6. Does the evaluator challenger procedure have its own valid conditional error control?

Until every question is answered with implementation evidence and an independent statistical review, this is a research proof sketch only. It must not be described as a REUSE cross-epoch theorem or a qualified false-promotion guarantee.
