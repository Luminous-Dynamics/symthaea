# Evidence-Governed Substrate Adaptation

**Status:** Research architecture and qualification contract. No integrated production result is claimed.

**Related implementation:** HLB/WSP research PR #10751.

**Related work:** FEP action-value audit #8391; production diagnostic adapter #11122; open-world adequacy and robust decision #11181; anytime-valid sequential repair verification #11278.

## 1. Research proposition

The Holographic Liquid Substrate should not be allowed to modify itself merely because a loss decreased, an entropy score increased, or a candidate passed a narrow test. Adaptation should be a sequence of evidence-qualified transitions.

The proposed loop is:

\[
\text{Monitor} \rightarrow \text{Diagnose} \rightarrow
\text{Program} \rightarrow \text{Evaluate} \rightarrow
\text{Verify} \rightarrow \text{Promote}
\]

Each transition has a distinct authority boundary:

- **Open-world adequacy monitor:** detects support failures and persistent predictive misspecification. It distinguishes insufficient evidence from no detected inadequacy and detected inadequacy. No detected inadequacy is not proof of correctness.
- **FEP / decision-focused diagnostics:** selects observations that can improve downstream decisions. Model EIG, decision EIG, EVSI, pragmatic value, observation cost, safety risk, and novelty remain separate.
- **Sequential diagnostic planner:** can select complementary probe sequences when no single probe has positive one-step EVSI.
- **WSP:** proposes typed, bounded parameter programs; it does not directly mutate the live substrate.
- **SWI:** validates exact base-model, architecture, manifest, parameter-group, and program digests, then materializes a detached candidate transactionally.
- **Sequential statistical verifier:** compares a frozen candidate to its baseline on paired, fresh confirmatory episodes with anytime-valid bounds.
- **Independent signed verifier:** evaluates the complete task, negative-control, capability-preservation, and rollback evidence. It alone may authorize candidate acceptance.

The implementation currently covers several of these components as isolated or partially connected primitives. Production predictive-observation integration and end-to-end promotion remain open.

## 2. Why the control planes must remain separate

### 2.1 Information is not decision value

For model hypothesis \(M\), action-conditioned observation \(O\), and probe \(a\):

\[
I(M;O\mid a)
\]

measures model information gain. If the task-optimal decision \(D^*\) is a deterministic function of \(M\), then:

\[
I(D^*;O\mid a) \le I(M;O\mid a)
\]

A probe may reveal a great deal about an irrelevant model distinction while leaving the optimal action unchanged.

Expected value of sample information instead measures reduction in minimum expected downstream loss:

\[
EVSI(a) = R(q) -
\mathbb{E}_{o\sim p(o\mid a)}
\left[R(q(\cdot\mid o,a))\right]
\]

Probe selection should compare this benefit against explicit observation cost and safety constraints. No-probe remains a valid action. Multi-step planning is necessary when complementary observations have zero one-step value.

### 2.2 In-family uncertainty is not model adequacy

Bayesian posterior concentration is conditional on the supplied generative family. It does not demonstrate that the family itself is adequate.

Prequential evaluation must score predictions frozen before each observation. Zero-support outcomes remain hard adequacy failures; they are not repaired by epsilon smoothing. The open-world gate permits detached candidate work only when there is no detected inadequacy under preregistered thresholds. This is a permission to investigate a candidate, not a statement that the model is true.

### 2.3 Statistical improvement is not independent acceptance

A candidate can improve an aggregate score and still damage a protected behavior. The statistical gate establishes a narrow paired-loss claim; it does not certify safety, correctness, or capability preservation.

The full verifier must additionally check protected task families, healthy controls, structural invariants, rollback, and evidence provenance. The signed verifier receipt must bind the exact candidate, base model, protocol, precondition gate, confirmatory shard, result manifest, and statistical report digests.

## 3. Anytime-valid sequential repair evidence

For paired episode \(i\), let baseline and candidate losses be \(L_i^{(b)},L_i^{(c)}\in[0,1]\), and define improvement:

\[
X_i=L_i^{(b)}-L_i^{(c)} \in [-1,1]
\]

Positive values favor the candidate. Let \(\delta\) be a preregistered minimum practically meaningful mean improvement.

For candidate ordinal \(j\) and interim look \(n\), allocate:

\[
\alpha_{j,n} =
\frac{\alpha_{\mathrm{campaign}}}
{j(j+1)n(n+1)}
\]

Both reciprocal-product series sum to one, so the sum of all candidate/look allocations is at most the campaign-wide error budget. A two-sided Hoeffding radius is:

\[
r_{j,n} =
\sqrt{\frac{2}{n}
\ln\left(\frac{2}{\alpha_{j,n}}\right)}
\]

The report uses the clipped interval:

\[
\left[
\max(-1,\bar X_n-r_{j,n}),
\min(1,\bar X_n+r_{j,n})
\right]
\]

Qualification states:

- **Benefit established:** lower bound exceeds \(\delta\).
- **Benefit not established:** upper bound is below \(\delta\).
- **Collecting:** neither condition holds and the budget remains.
- **Inconclusive at budget:** the maximum confirmatory sample is exhausted without either conclusion.

The campaign object owns the common alpha budget, frozen protocol digest, candidate ordinal sequence, and no-reuse registries for candidate-model and confirmatory-shard digests. Each candidate verifier also binds its precondition-gate digest.

### Restartable evidence state

The campaign exposes a deterministic checkpoint containing its fixed alpha/configuration, frozen protocol digest, next candidate ordinal, and sorted no-reuse registries for candidate and confirmatory-shard digests. Restore validates the checkpoint version, digest, registry cardinalities, and duplicate entries before another candidate can be assigned.

Each in-progress candidate can also be checkpointed with its immutable binding, frozen test configuration, and ordered paired observations. Restore rebuilds the running mean, duplicate-episode set, and observation hash by replaying every observation. It rejects altered checkpoint contents, a report digest that does not reproduce, malformed losses, duplicate episodes, and evidence appearing after a terminal decision.

The checkpoint digest is an integrity check, not authentication. Durable storage must authenticate checkpoints and preserve access controls; the independent verifier still validates signed receipts and recomputes the raw confirmatory stream. Checkpointing preserves the campaign's no-reuse registry across restarts only when the authenticated checkpoint is actually restored rather than replaced with a fresh campaign.

### Assumptions and limitations

The guarantee is conditional on bounded paired losses, a frozen candidate, a fixed evaluation protocol, and a valid confirmatory sampling process. Candidate-specific shards must be fresh and not reused to select later edits. The campaign ledger prevents duplicate shard digests within a campaign and its checkpoint preserves that registry across authenticated restarts. It still cannot prove that a caller has truthfully described the underlying data split or that two different digests do not encode overlapping observations.

The current implementation uses conservative Hoeffding bounds. It does not yet implement task-family guardrail sequences, a campaign-level episode-content overlap oracle, empirical-Bernstein confidence sequences, or betting-based e-process comparisons. These are tracked in #11278.

## 4. Evidence chain

A candidate evaluation should be reproducible from:

1. Base model and candidate model digests.
2. WeightProgram digest, typed target manifest, and edit-budget claims.
3. Open-world adequacy report and gate-decision digest.
4. Frozen evaluation-protocol digest and candidate-specific confirmatory-shard digest.
5. Ordered paired episode IDs and baseline/candidate loss values.
6. Sequential confidence-sequence report and digest.
7. Protected-behavior, negative-control, rollback, and independent-verifier result manifests.
8. Signed acceptance or rejection receipt.

A missing or mismatched link blocks acceptance. A queued workflow, incomplete experiment, or missing exact-head receipt is not evidence of success.

## 5. Research and industry comparisons

The programme must compare against:
- fixed-horizon paired tests;
- naïve repeated peeking on a reused holdout (negative control);
- fixed-look multiplicity-corrected paired tests;
- anytime-valid Hoeffding confidence sequences;
- empirical-Bernstein or betting-based confidence sequences;
- random matched-sparsity and analytic repair;
- no-edit and rollback baselines.

The WSP-Scope 2026 study reports substantial but uneven gains across 16 editing trajectories and two model scales, with six trajectories retaining original weights when validation was inconclusive. WeightLoop (2026) studies experimental-feedback-driven persistent editing. These make candidate search, evidence reuse, cross-task transfer, and reliable abstention central benchmark dimensions, not implementation details.

## 6. Benchmark targets

These are proposed targets, not measured Symthaea results:

| Dimension | Target |
|---|---|
| Family-wise false qualification | At most 5% under the preregistered null and adaptive candidate ordering |
| Power | At least 90% for the declared minimum detectable paired effect at the fixed evaluation budget |
| No-effect behavior | No candidate accepted when the true improvement is below the practical threshold, within the family-wise error guarantee |
| Evidence binding | 100% exact digest agreement across candidate, protocol, shard, precondition, and result receipts |
| Data isolation | Zero confirmatory-shard reuse within a campaign |
| Capability preservation | No protected task family exceeds its preregistered non-inferiority margin |
| Decision efficiency | Fewer probes or lower regret than matched EIG, greedy, always-inspect, and RL baselines where decision-relevant information is available |
| Compute efficiency | Report latency, memory, allocations and energy at matched outcome quality |

Report confidence intervals, power, episodes-to-decision, candidate count, observation cost, subgroup regression, false qualification, and inconclusive rate. Synthetic fixture results must be labeled as such and must not be presented as deployment error rates.

## 7. Claim ladder

1. **Algebraic correctness:** formulas and numerical invariants pass independent reference tests.
2. **Component correctness:** each isolated kernel passes exact-head tests and negative controls.
3. **Protocol correctness:** manifests, splits, digests, and receipts are independently checked.
4. **Synthetic capability:** known-truth fixtures demonstrate intended control behavior.
5. **Integrated capability:** a live adapter uses actual predictive observations and candidate models.
6. **Held-out advantage:** preregistered experiments show improvement over matched strong baselines.
7. **Independent replication:** another implementation reproduces the result.

No level implies a later one. No SOTA, safety, consciousness, or self-improvement claim follows from passing an earlier level.
