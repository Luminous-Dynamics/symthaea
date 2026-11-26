# SDCB observational-equivalence and intervention-separation oracle v1

**Status:** bounded exact reference; eight Rust tests authored, not yet compiled or executed in this environment.  
**Implementation:** \`scripts/research/sdcb_identifiability_oracle.rs\`  
**Fixture:** \`SDCB_IDENTIFIABILITY_ORACLE_V1.example.json\`  
**Expected result:** \`SDCB_IDENTIFIABILITY_ORACLE_V1.expected.json\`

## 1. Why this increment exists

The common-cause oracle (#13804) can choose a calibration measurement under a finite shared-state model. The joint evidence ledger (#13805) can update scientific, shared-cause, and device-local state after a qualified receipt. A remaining gap is more fundamental:

**Can the available observations distinguish the scientific explanation from an instrument/common-cause explanation at all?**

A model may be confidently wrong when two competing causal/measurement worlds generate the same observable distribution. More sensors, more repeated readings, a larger predictor, or a more sophisticated posterior approximation cannot recover information that is absent from the experiment.

This is not a claim that identifiability analysis or intervention design is new. The contribution is a small, exact, machine-checkable gate connecting paired-world identifiability to SDCB's action selection and abstention.

## 2. Exact paired-world definition

Let \(W_A\) and \(W_B\) be two fully specified candidate worlds. For each candidate action \(a\), each world supplies a categorical outcome distribution:

\[
P_A(y\mid a),\qquad P_B(y\mid a).
\]

The total-variation separation is:

\[
\operatorname{TV}(a;W_A,W_B)
=\frac12\sum_y|P_A(y\mid a)-P_B(y\mid a)|.
\]

For the declared observational action set \(\mathcal O\), the two worlds are observationally equivalent when:

\[
\max_{a\in\mathcal O}\operatorname{TV}(a;W_A,W_B)
\le \epsilon_{\mathrm{id}}.
\]

An intervention \(i\in\mathcal I\) breaks this equivalence in the supplied candidate set when:

\[
\max_{i\in\mathcal I}\operatorname{TV}(i;W_A,W_B)
> \epsilon_{\mathrm{id}}.
\]

The result is deliberately scoped to the declared world pair, outcome model, and candidate action set. It is not a proof that no unlisted observation can ever distinguish the worlds.

## 3. Decision value and abstention

Let \(\pi=P(W_A)\), and let \(u_\bot\) be the declared utility for abstaining. For action \(a\), the exact Bayes-optimal expected utility is:

\[
V(a)=\sum_y\max\left\{
\pi P_A(y\mid a),
(1-\pi)P_B(y\mid a),
u_\bot\left[\pi P_A(y\mid a)+(1-\pi)P_B(y\mid a)\right]
\right\}.
\]

The no-experiment baseline is:

\[
V_0=\max\{\pi,1-\pi,u_\bot\}.
\]

The oracle selects an action only when its cost is within the hard budget and its expected utility strictly exceeds \(V_0\). It reports structural intervention separation independently of affordability. Thus:
- *identifiable in principle* does not imply *affordable*;
- *affordable* does not imply *physically valid*;
- *a verified intervention receipt* does not prove that the declared likelihood model is scientifically adequate.

## 4. Frozen analytical fixture

The example has two observationally equivalent explanations:

- \(W_A\): scientific effect present, shared offset zero;
- \(W_B\): scientific effect absent, shared offset one.

Two differently named science devices have the same outcome distributions under both worlds. Their total-variation separation is zero. With equal world priors and abstention utility 0.60, the best action is to abstain, utility 0.60.

A candidate intervention isolates/resets the declared common cause and then repeats the science measurement. Its likelihood rows are \([0.95,0.05]\) in \(W_A\) and \([0.05,0.95]\) in \(W_B\). The exact results are:
- intervention total variation: 0.90;
- forced binary Bayes accuracy: 0.95;
- optimal expected utility with abstention: 0.95;
- gain over the no-experiment baseline: 0.35.

These are constructed finite distributions, not physical-sensor measurements and not evidence that a real intervention achieves 95% accuracy.

## 5. Fail-closed input contract

The reference rejects:
- blank or identical world identifiers;
- invalid prior or abstention utility;
- missing observation candidates;
- blank or duplicated action identifiers;
- zero-cost actions;
- non-finite, negative, out-of-range, or non-normalized likelihood rows;
- mismatched outcome dimensions.

The oracle has no external dependencies and uses deterministic tie-breaking (higher expected utility, then lower cost, then lexical action identifier).

## 6. Integration path

Use this as a pre-decision qualification layer:

1. Symthaea proposes a pair of explanations that could account for the accepted evidence.
2. The evaluator constructs paired-world likelihoods for the registered observational and intervention candidates.
3. The identifiability oracle reports observational equivalence and candidate intervention separation.
4. SDCB evaluates resource-feasible actions; if none creates decision value, it abstains or requests a genuinely new intervention class.
5. The executor performs only an authorized experiment; a separately governed verifier authenticates the receipt.
6. EGBT updates beliefs only after evidence qualification. The receipt does not certify physical validity or causal correctness.

Near-term next step: connect the oracle's paired-world representation to #13804's scenario likelihoods, then extend it to exact two-stage contingent action trees and multiple paired worlds. Do not silently replace a finite exact reference with an approximate HDC/CfC policy.

## 7. Limits

This v1 reference:
- evaluates one categorical action at a time;
- compares two supplied worlds;
- assumes each likelihood row is already correctly specified;
- does not represent repeated-measure dependence or sequential state transitions;
- does not authenticate, execute, or physically validate interventions;
- does not infer causal structure from observational data;
- does not yet support continuous latent factors, Gaussian processes, or low-rank covariance models.

Those are explicit follow-on work, not implied capabilities. The paired-world check is an exact control for downstream learned or approximate selectors.
