# Dependent Conditional-Risk Null Validation V1

Status: deterministic Monte Carlo research benchmark. It does not change the live Symthaea controller. The exact-head Rust test and the full qualification runner must pass before its empirical outcomes count as a qualified result.

## Why this is a stronger test

The existing monitor benchmarks used short, deterministic score fixtures to verify recurrence, checkpointing, alpha allocation and paired controls. That is useful for software correctness, but not a meaningful stress test of the statistical premise behind a risk e-process. Here the observations are serially dependent: a hidden Markov environment switches between normal and high-risk regimes, and that regime persists for multiple timesteps.

The benchmark's null is designed to satisfy the key conditional-risk assumption in every latent state:
- normal regime: Bernoulli loss probability 0.13;
- high regime: Bernoulli loss probability 0.15;
- risk bound: (epsilon=0.15);
- normal-to-high regime switch probability 0.02 per step, high-to-normal probability 0.05.

Because both latent regimes have loss probabilities at or below (epsilon), any posterior mixture over the hidden state also has conditional loss probability at or below (epsilon), despite temporal dependence. This is the premise the e-process needs. It is stronger than saying only that the pooled average loss happens to be low.

After timestep 133 in the shifted control, probabilities change to 0.30 in normal and 0.48 in high regime. This violates the declared risk bound and is the power control.

## Four policies on the same loss path

Each trial runs four monitors against the same dependent loss trace and the same overall per-strategy (delta=0.05):

- persistent e-process with aggressive bet fractions [0.25, 0.50, 0.75];
- episode-reset, alpha-spent e-process with the same aggressive bets;
- persistent e-process with conservative bet fractions [0.03, 0.08, 0.15];
- episode-reset, alpha-spent e-process with the conservative bets.

Episode-reset monitors restart every 20 steps. They use (delta_j = delta/[j(j+1)]) for episode index (j), so total allocated false-alarm probability across episodes remains bounded by (delta) for each strategy. Persistent monitors keep one mixture process across all 400 steps and use the threshold (log(1/delta)).

The bet-mixture comparison is important: a fixed mixture containing only large bets may be valid but have poor power for weak, high-variance Bernoulli evidence. A mixture including small bets can accumulate weaker signals; however, resetting every 20 steps discards that accumulated evidence, and its rising alpha-spending threshold can suppress detection. The benchmark measures this trade-off instead of changing the false-alarm budget to make one policy win.

## Experiment and outputs

- 3,000 trials per condition, 400 timesteps per trial, and 20 reset episodes.
- Null and shifted trial (i) share the exact same seed, regime path and pre-shift loss sequence; only the post-shift loss probability changes.
- Raw output contains one row per trial with loss/regime traces encoded as 400-bit hexadecimal strings, loss counts, high-regime exposure, regime transitions, alarm flags, first-detection steps and reset-episode alarm counts.
- Summary output reports empirical alarm rates, 95% Wilson intervals, false-alarm counts under the null, missed-shift counts under the alternative, detection delay, regime exposure and mean number of reset episodes alarmed.
- The qualification script checks trace length and contents, recovers loss/regime counts from the hexadecimal traces, verifies paired pre-shift identity and exact summary arithmetic, reruns byte-identically and adds source/output SHA-256 hashes to the receipt.

The Wilson intervals quantify Monte Carlo uncertainty for this deterministic pseudo-random experiment; they are descriptive estimates, not formal proof of the Ville/alpha-spending guarantee.

## Acceptance gates

- Null false-alarm rates for all four strategies stay at or below the target (delta=0.05) in the frozen trial panel.
- Under the shifted control, conservative-bet persistent detection is at least 90%, aggressive-bet persistent detection at least 75%, and aggressive-bet reset detection at least 50%.
- Raw traces and summary arithmetic agree, every trace has exactly 400 observations, and paired null/shift trials share identical regimes and the first 133 loss outcomes.
- Repeated executions produce byte-identical artifacts and a complete provenance receipt.

These are regression gates for a deliberately designed fixture, not universal estimates of detector performance. The result must be reproduced under new hidden-regime transition parameters and held-out seeds before generalizing.

## Relation to the literature and remaining gap

The statistical framing comes from Timans et al., [On Continuous Monitoring of Risk Violations under Unknown Shift, UAI 2025](https://proceedings.mlr.press/v286/timans25a.html), which studies testing-by-betting for risk violations under dynamic shift. This benchmark is a smaller fixed-mixture conditional-risk null test; it is not a reproduction of their broader methods.

The next external benchmark should use a pinned dynamics model and an independently measured physical or high-fidelity safety loss, with held-out disturbance families and a matched nominal false-alarm budget. It should also compare against conformal OOD monitoring plus reachability fallback, e-value mixtures tuned without evaluation data, and state/control-dependent reachable-set methods. No SOTA or physical-safety claim follows from this synthetic benchmark alone.
