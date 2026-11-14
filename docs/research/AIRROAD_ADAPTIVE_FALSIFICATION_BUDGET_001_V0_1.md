# AIRROAD-ADAPTIVE-FALSIFICATION-BUDGET-001 — detection power under adaptive excitation

**Status:** V0.1 exact finite-action power-ceiling and hard-set sequence reference  
**Tracking issue:** [#12008](https://github.com/Luminous-Dynamics/symthaea/issues/12008)  
**Composes with:** #11868/#11887 active excitation, #11997 independent holdout falsification, #11914/#11986/#11990 shared nuisance and complex statistical uncertainty

> This reference tells us when a design cannot support a requested model-adequacy power target. It does not construct a detector or prove that a positive power ceiling is attainable.

## 1. Research basis and SOTA boundary

Choudhary (2026-11-05), [Adaptive Experimentation Can Hide Model Error: Detection Limits and Falsification Budget](https://arxiv.org/abs/2611.07529), derives an own-design power ceiling for adaptive adequacy tests. It distinguishes information that the selected measurements never acquired from information that a detector failed to use. The paper also reports mixed results: the frozen policy failed on fresh continuous-null panels; a later revised policy had no resolved advantage at a longer horizon and trailed class-discrimination EIG in one measured replay. AIRROAD treats this as a rigorous warning and benchmark comparator, not a recipe that is already proven to dominate.

Niu, Shen and Yong (2026), [Active model discrimination: A multi-stage approach with adaptive partitions](https://doi.org/10.1016/j.automatica.2026.113067), Automatica 190, article 113067, provides an adjacent bounded-uncertainty multi-stage model-discrimination baseline.

Existing owners remain distinct:
- #11868 ranks candidate port/frequency experiments by model disagreement.
- #11887 composes adaptive experiment decision trees under bounded cost.
- #11997 owns one exact independent holdout and model-falsification witness.
- This issue owns the power ceiling and sequence-level budget adequacy accounting.

## 2. Statistical own-design power ceiling

For finite action space \(\mathcal A\), finite outcome space \(\mathcal Y\), stationary truth \(P(\cdot\mid a)\), and a working null family \(\mathcal M_0=\{p_\theta:\theta\in\Theta\}\), let \(\nu(a)\) be the fixed design's action proportions.

Define:
\[
D(\nu)=\inf_{\theta\in\Theta}
\sum_{a\in\mathcal A}\nu(a)
\operatorname{KL}\left(P(\cdot\mid a)\,\|\,p_\theta(\cdot\mid a)\right).
\]

For a test with false-rejection probability at most \(\alpha\) under every null parameter, the data-processing inequality gives a necessary condition for power at least \(1-\beta>\alpha\):
\[
T D(\nu)\geq \operatorname{kl}(1-\beta\|\alpha),
\]
where \(\operatorname{kl}\) is binary relative entropy.

This is a power **ceiling / necessary resource condition**. It is not a lower bound on the power of a practical detector. A value above the threshold does not prove the test will attain the target. The theorem's assumptions and scope must remain attached to every report.

### Exact two-action Bernoulli fixture

Actions are \(A\) and \(B\). The off-model truth has:
\[
P(Y=1\mid A)=P(Y=1\mid B)=1/5.
\]

The finite working family has two null parameters:
\[
\theta_1:\quad p_{\theta_1}(1\mid A)=1/5,\quad
p_{\theta_1}(1\mid B)=4/5,
\]
\[
\theta_2:\quad p_{\theta_2}(1\mid A)=4/5,\quad
p_{\theta_2}(1\mid B)=1/5.
\]

An A-only design is indistinguishable from \(\theta_1\), so \(D(1,0)=0\). A B-only design is indistinguishable from \(\theta_2\), so \(D(0,1)=0\). No valid size-\(\alpha\) test can exceed power \(\alpha\) under either one-action design, however many observations it collects.

For a balanced design \(\nu=(1/2,1/2)\), each null has the same worst-member divergence:
\[
D(1/2,1/2)
=\frac{1}{2}\operatorname{KL}(\operatorname{Ber}(1/5)\|
  \operatorname{Ber}(4/5))
=\frac{3}{10}\ln 4.
\]

For \(\alpha=0.05\) and target power \(1-\beta=0.8\):
\[
\operatorname{kl}(0.8\|0.05)\approx1.9064420542,
\quad D\approx0.4158883083.
\]
Therefore the necessary horizon is \(T\geq4.58402416\), so the first integer horizon not ruled out by this necessary condition is 5. Four samples are insufficient by the ceiling; five samples are not a guarantee of 80% power.

The fixture's truth is a simulator oracle. The value \(D(\nu)\) is not directly observable from real measurements unless justified bounds or a specified truth set make it computable.

## 3. Deterministic joint-sequence separation under shared calibration

Statistical power and hard-set model falsification are different claims. V0.1 also freezes a deterministic counterexample where **each single action leaves both hypotheses feasible**, but the pair of observations rejects both when the calibration nuisance is correctly shared.

Let the observations be \(y_A=y_B=0\), measurement error be exactly zero, and the one campaign-shared calibration offset be:
\[
\xi_{\mathrm{cal}}\in[-1,0].
\]

Two candidate model predictions are:
\[
H_1(A)=0,\quad H_1(B)=1,
\qquad
H_2(A)=1,\quad H_2(B)=0.
\]

Observation model:
\[
y(a)=H_k(a)+\xi_{\mathrm{cal}}.
\]

- Action A alone: H1 is feasible with \(\xi=0\); H2 is feasible with \(\xi=-1\).
- Action B alone: H1 is feasible with \(\xi=-1\); H2 is feasible with \(\xi=0\).
- Joint A+B: H1 would require the same \(\xi\) to equal both 0 and -1; H2 would require both -1 and 0. Both model intersections are empty.

If the implementation incorrectly gives each frequency/action an independent calibration coordinate, both models survive the two observations. The shared coordinate is therefore essential to the joint falsification result.

This is a minimal algebraic fixture, not a claim about a physical two-port system. It tests why sequential experiment design must carry shared nuisance coordinates across all actions rather than evaluate each point independently.

## 4. Required V0.1 verifier

Use Python standard library only:
- exact rational Bernoulli probabilities and action proportions;
- exact derivation of \(D(\nu)=0\) for A-only/B-only policies;
- exact symbolic expression \(D(1/2,1/2)=3/10\ln 4\), with high-precision Decimal evaluation for the binary-KL threshold and integer-horizon calculation;
- verify the fixed-design necessary condition without presenting it as achievable power;
- exact interval/set intersection for the shared-calibration sequence;
- show that each singleton action leaves both candidates feasible;
- show that the combined action sequence leaves no candidate feasible under one shared calibration variable;
- show that an incorrectly reset, action-local calibration variable makes both candidates survive;
- reject promotion of a power ceiling to a valid test, a positive ceiling to guaranteed power, or a surviving set to a proven true model.

## 5. Benchmark ladder

Under matched total observations, input energy, operating limits and uncertainty:
1. fixed logarithmic sweep;
2. D-optimal/Fisher-information parameter design;
3. one-step model-discrimination EIG;
4. multi-stage active model discrimination with adaptive partitions (#11887 comparator);
5. random/exploration baseline;
6. adaptive parameter design without a falsification reserve;
7. preregistered falsification reserve;
8. sequence-level shared-nuisance set separation.

For statistical campaigns, report type-I error, empirical power on fresh off-model worlds, own-design power ceiling, detector capture loss, expected/worst-case cost and horizon dependence. For hard-set campaigns, report candidate intersection, separation margin, shared nuisance feasibility and actions required. Do not merge these into a single "information score."

## 6. Claim ceiling

V0.1 validates a finite-action necessary power condition and a deterministic shared-nuisance sequence counterexample. It does not prove an adaptive policy optimal, a fixed falsification fraction sufficient, arbitrary continuous-null statistical validity, real-world model adequacy, IQC feasibility, robust stability or airworthiness.
