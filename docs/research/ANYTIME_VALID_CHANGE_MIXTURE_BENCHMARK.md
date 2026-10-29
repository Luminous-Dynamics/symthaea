# Anytime-valid unknown-change-time Bernoulli mixture

Status: research prototype; exact-head Rust qualification is pending. This document distinguishes the mathematical guarantee from synthetic performance observations.

## Why this direction

The current active-evidence work has a composite-band betting monitor and a finite-horizon CUSUM reference. The composite monitor accumulates evidence from the start of the stream, which can dilute a later change with an earlier stable segment. The finite-horizon CUSUM is more powerful in the present Bernoulli fixture, but its threshold is calibrated for a fixed horizon. This branch explores a third point on the trade-off frontier: one anytime-valid e-process that mixes over unknown change time, direction, shift magnitude, and a small grid of timescales.

This is not a claim to have invented e-detectors. Shin, Ramdas, and Rinaldo's framework builds CUSUM/Shiryaev–Roberts-style e-detectors from sums of e-processes started at consecutive times, with nonasymptotic false-alarm control and delay results ([2024 paper, DOI 10.51387/23-NEJSDS51](https://doi.org/10.51387/23-NEJSDS51)). FOCuS develops efficient online likelihood-ratio tests across unknown change locations in one-parameter exponential-family streams ([Ward et al., 2024, DOI 10.1007/s11222-024-10416-6](https://doi.org/10.1007/s11222-024-10416-6)). mixFOCuS studies the communication-versus-power trade-off for mixed-type distributed streams ([Yang et al., 2025, DOI 10.1111/jtsa.12834](https://doi.org/10.1111/jtsa.12834)). Our research contribution is an auditable bounded-conditional-null Bernoulli implementation with cross-language trace checks and directly comparable Symthaea baselines.

## Formal contract

Assume each outcome $X_t\in\{0,1\}$ has a predictable conditional event probability $p_t\in[\ell,u]=[0.48,0.52]$. The conditional rate may depend on the full observed history; the null is not IID.

For an upward alternative $q>u$, define the Bernoulli likelihood-ratio increment against the least-favourable edge $u$:

\[
L_t(q)=\begin{cases}
q/u,& X_t=1,\\
(1-q)/(1-u),& X_t=0.
\end{cases}
\]

For every accepted $p_t\le u$, $\mathbb E[L_t(q)\mid\mathcal F_{t-1}]\le1$. For a downward alternative $q<\ell$, use the same ratio against $\ell$; its conditional expectation is at most one whenever $p_t\ge\ell$. Each component therefore preserves the nonnegative-supermartingale property under the declared conditional null.

For a hazard $h\in(0,1)$, mix all possible change starts with $\pi_k=h(1-h)^{k-1}$. Let $A_t$ be active suffix wealth. The component recurrence is:

\[
A_t=L_t(q)(A_{t-1}+\pi_t),\qquad E_t=A_t+(1-h)^t.
\]

The final term is the mass of change times that have not occurred yet. It ensures $E_0=1$ and keeps the component a nonnegative supermartingale. A fixed equal-weight mixture across both directions, five alternatives per direction and three hazards remains an e-process. Ville's inequality then controls the probability of crossing $1/\alpha$ at any time by $\alpha$, under this exact null contract.

The implementation uses 30 components:
- upward alternatives: $q\in\{0.55,0.60,0.70,0.80,0.90\}$, null edge $u=0.52$;
- downward alternatives: $q\in\{0.45,0.40,0.30,0.20,0.10\}$, null edge $\ell=0.48$;
- change-time hazards: $h\in\{0.04,0.01,0.0025\}$, with geometric prior mean start times 25, 100 and 400 observations;
- log-space recursion; threshold (\log E_t\ge-\log(0.05/12)\).

**Scope limit:** the guarantee only applies when every predictable conditional event rate remains within the declared band. It does not automatically apply to arbitrary high-dimensional feature-distribution shift, and it cannot make an incorrectly chosen null band scientifically valid.

## Deterministic qualification suite

The Rust CLI emits 14 deterministic trace probes:
1. stationary $p=0.50$ and both accepted-null edges $p=0.48,0.52$;
2. a selection-bias control at $p=0.74$, deliberately outside the null band;
3. alternating in-band endpoints and a predictable feedback schedule that selects the next rate based on the current e-value before observing the next outcome;
4. upward and downward step shifts, change after observation 100;
5. all-one and all-zero extreme paths.

Run the probe and independent reference checker:

```bash
cargo run --locked --quiet --manifest-path tools/active-evidence-shift-monitor/Cargo.toml -- --anytime-mixture-probes-csv > /tmp/anytime-mixture-probes.csv
python3 tools/active-evidence-shift-monitor/reference_check.py --anytime-mixture-self-test
python3 tools/active-evidence-shift-monitor/reference_check.py --anytime-mixture-probes /tmp/anytime-mixture-probes.csv
```

The Python reference independently reproduces first-alarm steps and final log-e-values for every trace, tests the one-step conditional expectation over every binary history through depth 8 at five null rates, verifies prior mass normalization, and checks extreme-path threshold crossings. Workflow controls run the CSV twice and compare bytes, reject a mutated alarm step, and include the artifact hash in the provenance receipt.

## Exploratory Monte Carlo results

A vectorized NumPy prototype ran 1,000 trials per scenario at horizon 1,000, with a step change after observation 100. These figures are an exploratory pilot, not a Rust workflow artifact or the source of the theoretical guarantee.

| Post-change rate | Post-change detection | 95% Wilson interval | Median delay, among detections | P90 delay, among detections |
|---:|---:|---:|---:|---:|
| 0.55 | 2.7% | 1.9–3.9% | 674 | 854 |
| 0.60 | 81.0% | 78.5–83.3% | 510 | 790 |
| 0.70 | 99.9% | 99.4–100% | 133 | 225 |
| 0.80 | 100% | 99.6–100% | 58 | 89 |
| 0.45 | 3.1% | 2.2–4.4% | 666 | 839 |
| 0.40 | 81.1% | 78.6–83.4% | 537 | 785 |

The optional reproducible runner is `tools/active-evidence-shift-monitor/anytime_mixture_monte_carlo.py`. It records trial count, horizon, seeds, PRNG and NumPy version, and exports Wilson confidence intervals. With zero observed alarms in a 1,000-run null sample, the estimate is zero but the true risk is not; do not replace the mathematical e-process guarantee with a finite simulation estimate.

In the previous exact-DP pilot, the never-reset composite-band betting process detected p1=.55 and .60 shifts at 1.769% and 73.527%. The new unknown-start mixture improves on those in this Monte Carlo pilot because evidence can accrue from a later suffix rather than always from the beginning of the stream. The conservative family-wise-bound, finite-horizon CUSUM reference reported higher one-sided power (4.884% and 90.407%) in the same synthetic setup, while its threshold is tied to the fixed 1,000-observation horizon. This is a trade-off, not a universal win: the new component is intended for a time-uniform guarantee under the stated conditional null and hedging over unknown change times/rates.

## Next benchmark gates

1. Exact-head Rust tests, formatting and independent Python probe parity must pass.
2. Freeze $\alpha=0.05/12$, alternatives, hazards, and scenarios before evaluation; no post-hoc tuning to the test traces.
3. Benchmark false alarms with enough null-run exposure to estimate stringent error rates; report confidence upper bounds when no alarms are observed.
4. Compare to a faithful e-detector implementation, FOCuS/mixFOCuS where their assumptions fit, the current finite-horizon CUSUM, ADWIN, EWMA and Page–Hinkley.
5. Evaluate source-held-out real streams, dependence, gradual/recurring shifts, selection-mechanism changes, sparse multi-stream shifts, runtime and memory. Report detection probability plus conditional median/P90 delays; do not let missed detections disappear from the latency metrics.
6. No SOTA or production claim until independent reproduction on real, source-distinct streams.
