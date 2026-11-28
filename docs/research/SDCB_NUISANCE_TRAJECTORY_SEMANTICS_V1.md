# SDCB persistent versus rectangular nuisance trajectory semantics

**Status:** exact analytical two-stage counterexample; Rust verifier authored but not compiled or CI-qualified.

Machine-readable fixture: `docs/research/SDCB_NUISANCE_TRAJECTORY_SEMANTICS_V1.json`.

## Why this matters

A nuisance parameter can be:
- persistent and shared over a complete episode;
- reselected or drifted at each stage;
- selected as a function of public history;
- or governed by a Markov/finite-state transition.

Those are different trajectory families, even when each stage uses the same marginal parameter set. A test valid for one is not automatically valid for another.

This distinction is central to metrology: shared reference bias/common-cause effects, device-local drift, temperature-dependent response and per-sample noise must not be collapsed into one undifferentiated variance.

## Frozen two-stage challenge

Nuisance values are \(p\in\{1/5,4/5\}\). Observations are binary, and the terminal test uses rejection probability \(5/16\) when \(X_1\ne X_2\), otherwise zero. The target size is \(\alpha=1/10\).

### Persistent shared nuisance

Choose one p for the complete episode; conditional on p, X1 and X2 are iid Bernoulli(p).

For either p:
\[
P(X_1\ne X_2)=2p(1-p)=8/25.
\]
The terminal size is therefore:
\[
(8/25)(5/16)=1/10.
\]

The frozen test is exactly size-alpha for the declared persistent family.

### Rectangular precommitted nuisance

Choose p1 and p2 independently from the same set before each observation. For p1=1/5 and p2=4/5:
\[
P(X_1\ne X_2)
=p_1(1-p_2)+(1-p_1)p_2
=17/25.
\]
The same test now has false rejection:
\[
(17/25)(5/16)=17/80 > 1/10.
\]

The valid terminal randomization cap for this larger family is:
\[
\alpha/(17/25)=5/34.
\]

### History-adaptive rectangular nuisance

Now permit p2 to depend predictably on X1. Set p1=1/5, p2=4/5 after X1=0, and p2=1/5 after X1=1. Then:
\[
P(X_1\ne X_2)
=(4/5)(4/5)+(1/5)(4/5)=4/5.
\]
The same nominal test has false rejection \(1/4\), and the valid randomization cap becomes \(1/8\).

The verifier enumerates all eight deterministic non-anticipating second-stage selectors over the two p1 choices and two conditional p2 choices.

## Exact results

| Trajectory family | Worst discordance | False rejection with 5/16 test | Validity at alpha=0.1 | Valid randomization cap |
|---|---:|---:|---|---:|
| Persistent shared | 8/25 | 1/10 | valid | 5/16 |
| Rectangular precommitted | 17/25 | 17/80 | invalid | 5/34 |
| Rectangular history-adaptive | 4/5 | 1/4 | invalid | 1/8 |

This is a counterexample to silently transferring a guarantee from one uncertainty contract to another. It is not an assertion that a physical sensor nuisance is always rectangular or adversarial.

## Implementation

- `scripts/research/sdcb_nuisance_trajectory_semantics.rs` — exact rational evaluator for the three trajectory families.
- `scripts/research/run_sdcb_nuisance_trajectory_semantics_tests.sh` — standalone Rust test/executable harness.
- Eight authored tests cover persistent size, precommitted rectangular failure, history-adaptive failure, enumeration of all selectors, corrected randomization caps, predictable adaptation and family non-interchangeability.

## Required application to sensor qualification

Every observation-law receipt should explicitly identify:
- nuisance scope (episode/device/channel/window/stage/observation);
- static, bounded-drift, Markov or history-dependent dynamics;
- shared-reference/common-cause coupling;
- whether actions or public history affect nuisance evolution;
- calibration/reset semantics and epoch boundaries;
- the complete joint trajectory law or hard support.

Compose with #13904's dynamic state-space uncertainty, #13805's joint evidence transition, #14185/#14218's minimax test authority, and #14222's optional-stopping induction. Do not implement another state estimator or policy planner.

## Validation boundary

The result is an exact finite two-stage counterexample. It does not identify real sensor drift, validate a calibration chain, or establish SOTA/industry superiority. Those require the actual reference ancestry, time-series and environmental evidence for the instrument under test.
