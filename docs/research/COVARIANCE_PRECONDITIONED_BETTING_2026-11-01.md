# Covariance-preconditioned predictable betting for projective e-closure

**Status:** draft research extension of #9751/#9712. This is a model-assisted intersection process, not a replacement for the arbitrary-dependence closure in #9695.

## 1. Statistical construction

For a fixed intersection S, let A_t=S intersect I_t denote its active coordinates. Assume a common filtration F_t, predictable active sets, coordinatewise null means theta_i,t <= 0, and a predictable global conditional-mgf envelope V_t whose active principal submatrix is V_A,t:

```
E[ exp(lambda^T (Z_A,t - theta_A,t)) | F_(t-1) ]
    <= exp(0.5 * lambda^T V_A,t lambda)
```

for every predictable nonnegative lambda in the supported domain.

The current-round factor is

```
X_S,t = exp(lambda_S,t^T Z_A,t
            - 0.5 * lambda_S,t^T V_A,t lambda_S,t)
```

where lambda_S,t is selected before Z_A,t is observed. Conditional expectation is at most

```
exp(lambda_S,t^T theta_A,t) <= 1
```

so the product is an intersection test supermartingale/e-process. The strategy used to choose lambda changes power, not this validity argument.

## 2. Covariance-preconditioned update

After recording the current factor, update the next-round bet by

```
lambda_(t+1) =
    clip_[0,L](
      lambda_t + eta * D_A^(-1) * (Z_A,t - V_A * lambda_t)
    ),
D_A = diag(V_A)
```

This is a diagonal covariance-preconditioned projected-gradient step for the one-step log factor. It moves toward a nonnegative direction with positive historical score residuals while accounting for the dependence penalty V_A*lambda_t.

The update is intentionally separated from the factor:
1. call `bets(active)` to freeze the pre-update vector;
2. compute the factor using that vector and the current score;
3. call `observe(scores)` to update the bet for the next round.

The state machine rejects same-round lookahead, active-set retraction, duplicate bet reads, missing bet reads, dimension mismatch and non-finite input. Nested activation is allowed. Inactive coordinates are ignored by the update, and their bet stays zero until activation.

The learning rate eta and cap L affect power only. They must be fixed ex ante for each expert. A mixture over several fixed eta values is valid; selecting the pointwise largest expert is not.

## 3. Projectivity and scope

For each fixed intersection S, the bet state is intersection-specific and sees only S intersect I_t. V_A,t is always the principal submatrix of one globally specified envelope. Extending the universe with dormant coordinates therefore leaves earlier factors unchanged. A later activation can affect subsequent bets within that same intersection, but cannot rewrite its prefix.

The code's PSD check is only a numerical screen. It does not prove that V dominates the true conditional mgf. Under Gaussian observations, V_A-Sigma_A being PSD is sufficient. For non-Gaussian observations, the conditional mgf envelope must be separately justified.

The method is not valid under arbitrary dependence without an envelope. The #9695 affine projective merger remains the model-robust fallback.

## 4. Fixed-prior learning-rate mixture

For eta_j in a finite set fixed before observing the data, each predictable nonnegative-bet process M_t^(j) is an e-process under the same mgf envelope. With fixed prior weights pi_j summing to one,

```
Mbar_t = sum_j pi_j M_t^(j)
```

is also an e-process. This lets the strategy adapt to the useful learning rate without selecting a single best rate after observing evidence. The current runner uses a uniform prior over eta = {0.025, 0.05, 0.10, 0.20}. The separate eta=.05 result is included as a diagnostic, not as a selected confirmatory method.

## 5. Projectively consistent covariance envelope

The first experiment used the dimension-specific upper envelope
`Sigma(0.5) + 1.2 I`. It dominates the registered equicorrelation family for d=5, but its diagonal correction changes with d. It must therefore **not** be used to claim projectivity under future universe extension.

The corrected global kernel is

```
V_d = (1-rho_min) I_d + rho_max 11^T
```

for a declared nonnegative equicorrelation interval
`0 <= rho_min <= rho <= rho_max <= 1`. In the benchmark interval [.2,.8],

```
V_d = 0.8 I_d + 0.8 11^T.
```

For every rho in [.2,.8],

```
V_d - Sigma_d(rho)
  = (rho - 0.2) I_d + (0.8 - rho) 11^T >= 0.
```

This is a sum of positive-semidefinite matrices. Every principal submatrix is the same kernel at the smaller dimension, so universe extension does not rewrite any earlier covariance penalty. The result is more conservative in some joint directions and less conservative in others than the d=5 dimension-specific envelope.

An independent numerical check tested dimensions 1–15, 101 rho grid points per interval, and intervals [.2,.8], [.1,.6] and [0,1]. All covariance differences were PSD within numerical tolerance, and every smaller-dimensional matrix matched the corresponding leading principal submatrix exactly. The checked-in tests now include the same extension-invariance property.

### Gaussian static-correlation diagnostic with global projective V

A paired exploratory run used 20,000 streams per cell, d=5, 60 rounds, alpha=.05, eta=.05, cap=1.5, warmup=5 and global V=.8I+.8J. The coordinate-local vector and shared-scalar methods used the same V and the same simulated stream in each cell. The table reports crossing probability by 60 rounds, with Wilson 95% intervals.

| rho | Method | All-null | Sparse mean (0,0,0,0,.8) | Dense mean (.45 x 5) |
|---:|---|---:|---:|---:|
| .2 | Preconditioned | 0.050% [0.027, 0.092] | 95.160% [94.854, 95.449] | 87.050% [86.578, 87.508] |
| .2 | Coordinate-local | 0.100% [0.065, 0.154] | 96.685% [96.428, 96.924] | 22.900% [22.323, 23.488] |
| .2 | Shared scalar | 0.000% [0, 0.019] | 0.060% [0.034, 0.105] | 21.705% [21.139, 22.282] |
| .5 | Preconditioned | 0.465% [0.380, 0.569] | 95.435% [95.137, 95.716] | 81.515% [80.971, 82.047] |
| .5 | Coordinate-local | 0.135% [0.093, 0.196] | 96.225% [95.952, 96.480] | 27.820% [27.203, 28.445] |
| .5 | Shared scalar | 0.025% [0.011, 0.059] | 0.790% [0.676, 0.922] | 26.065% [25.461, 26.678] |
| .8 | Preconditioned | 1.570% [1.407, 1.752] | 95.715% [95.425, 95.987] | 77.655% [77.072, 78.227] |
| .8 | Coordinate-local | 0.260% [0.198, 0.341] | 95.895% [95.611, 96.161] | 29.795% [29.165, 30.433] |
| .8 | Shared scalar | 0.200% [0.147, 0.272] | 2.015% [1.829, 2.219] | 28.075% [27.456, 28.702] |

The preconditioned method substantially improves dense-alternative crossing relative to these two simple betting baselines while retaining high sparse-alternative crossing. Its sampled all-null crossing rate increases with rho but remains below alpha. This is an intersection-level diagnostic, not a proof of calibration or a closed-FDR/SupFDR result. Different methods use different predictable bet paths, so the power comparison is descriptive under the registered Gaussian law, not a dominance theorem.

### Fixed-prior strategy portfolio under the same global V

A second independent local run used the runner's SHA-derived cell seeds and the same 20,000-stream / 60-round / d=5 configuration. It compared the fixed eta=.05 expert, a uniform fixed-prior mixture over eta={.025,.05,.10,.20}, a fixed 50/50 mixture of coordinate-local and eta-mixture e-processes, and an equal-weight mixture of coordinate-wise e-processes (valid under arbitrary cross-coordinate dependence when each marginal factor is valid).

| rho | Method | All-null | Sparse | Dense |
|---:|---|---:|---:|---:|
| .2 | eta=.05 | 0.045% [0.024, 0.086] | 94.815% [94.499, 95.114] | 86.690% [86.212, 87.154] |
| .2 | fixed eta mixture | 0.035% [0.017, 0.072] | 91.675% [91.284, 92.050] | 82.685% [82.154, 83.203] |
| .2 | 50/50 vector + eta mixture | 0.040% [0.020, 0.079] | 96.560% [96.298, 96.804] | 74.090% [73.478, 74.693] |
| .2 | coordinate e-process mixture | 0.005% [0.001, 0.028] | 93.360% [93.007, 93.697] | 54.090% [53.399, 54.780] |
| .5 | eta=.05 | 0.385% [0.308, 0.481] | 95.510% [95.214, 95.788] | 81.320% [80.774, 81.854] |
| .5 | fixed eta mixture | 0.250% [0.190, 0.329] | 92.595% [92.224, 92.950] | 77.330% [76.745, 77.905] |
| .5 | 50/50 vector + eta mixture | 0.145% [0.101, 0.208] | 97.120% [96.879, 97.343] | 69.690% [69.049, 70.323] |
| .5 | coordinate e-process mixture | 0.005% [0.001, 0.028] | 93.530% [93.181, 93.863] | 45.845% [45.155, 46.537] |
| .8 | eta=.05 | 1.465% [1.308, 1.641] | 96.020% [95.740, 96.282] | 77.685% [77.103, 78.257] |
| .8 | fixed eta mixture | 1.330% [1.180, 1.498] | 93.500% [93.150, 93.833] | 74.355% [73.745, 74.955] |
| .8 | 50/50 vector + eta mixture | 0.705% [0.598, 0.831] | 97.540% [97.316, 97.746] | 67.530% [66.878, 68.175] |
| .8 | coordinate e-process mixture | 0.020% [0.008, 0.051] | 93.605% [93.257, 93.936] | 38.515% [37.843, 39.192] |

This comparison highlights the trade-off: a fixed prior over eta avoids selecting the best learning rate on the same data; the 50/50 vector/eta portfolio improves sparse crossing but pays dense-signal power; the coordinate mixture is much more conservative and avoids modeling cross-coordinate dependence. These are single-intersection diagnostics only. They do not establish closed-FDR or SupFDR validity by themselves; the proof contract must hold for each component and their fixed mixture.

The simulation used independent local NumPy code and seeds 9776, 9806 and 9836 for rho=.2,.5,.8. It was not produced by the checked-in benchmark runner and not by CI. The runner now uses the global projective kernel and emits Wilson intervals so the result can be reproduced in the repository environment.

## 6. Earlier fixed-dimension envelope experiments (historical)

The next tables preserve the earlier experiment for comparison only. Those results use `Sigma(0.5)+1.2I) at d=5. That envelope is valid for the fixed d=5 covariance family, but changing d changes its diagonal correction, so it is **not** a dimension-extension/projectivity result. Do not use the earlier tables as evidence for future-universe projectivity.


The following results were generated by local NumPy experiments implementing the above factor/update equations. They are not repository-runner or CI results. They are single-intersection crossing diagnostics over 60 rounds with alpha=.05, d=5, cap L=1.5, and the fixed envelope V=Sigma(.5)+1.2I covering equicorrelation rho in [.2,.8]. Wilson intervals are reported.

### Gaussian noise: fixed eta=.05, 20,000 streams per cell

| True rho | All-null crossing | Sparse mean (0,0,0,0,.8) | Dense mean (.45 x 5) |
|---:|---:|---:|---:|
| .2 | 0.045% [0.024, 0.086] | 89.630% [89.200, 90.045] | 92.025% [91.641, 92.392] |
| .5 | 0.460% [0.375, 0.564] | 89.620% [89.190, 90.035] | 87.410% [86.943, 87.863] |
| .8 | 1.805% [1.630, 1.999] | 89.620% [89.190, 90.035] | 84.045% [83.531, 84.546] |

### Gaussian noise: fixed-prior eta mixture, 20,000 streams per cell

| True rho | All-null crossing | Sparse mean (0,0,0,0,.8) | Dense mean (.45 x 5) |
|---:|---:|---:|---:|
| .2 | 0.020% [0.008, 0.051] | 83.525% [83.004, 84.033] | 88.225% [87.771, 88.664] |
| .5 | 0.380% [0.304, 0.475] | 83.665% [83.146, 84.171] | 83.700% [83.182, 84.205] |
| .8 | 1.750% [1.577, 1.941] | 84.025% [83.511, 84.526] | 80.975% [80.425, 81.513] |

The fixed eta mixture sacrifices some power versus the best single eta in a given cell but avoids selecting eta on the same data. A fixed 50/50 mixture of the earlier coordinate-local vector process and the eta=.05 process also performed well on the same Gaussian stream: at rho=.5, sparse crossing was 90.935% and dense crossing 80.910%. These are diagnostic comparisons, not a multiple-testing power result.

### Non-Gaussian and time-varying stress

Use factor noise

```
Z_i,t - theta_i,t = sqrt(rho_t)*G_0,t + sqrt(1-rho_t)*G_i,t
```

where G_0,t and G_i,t are independent Rademacher signs. Its covariance is Sigma(rho_t), and log(cosh(x)) <= x^2/2 gives the same conditional sub-Gaussian mgf proxy. A deterministic switch rho_t=.2 for the first 30 rounds and .8 for the next 30 remains inside the same envelope.

In 10,000-stream local experiments with eta=.05:
- Rademacher factor, static rho=.2/.5/.8: all-null crossings 0.01% / 0.46% / 1.90%; sparse power 88.85% / 89.60% / 89.06%; dense power 92.04% / 87.77% / 84.19%.
- Rademacher factor, rho switching .2 to .8 at round 30: all-null crossings 0.75%, sparse power 89.07%, dense power 89.58%.
- Gaussian factor, same switch: all-null crossings 0.66%, sparse power 89.51%, dense power 89.35%.

For the fixed-prior eta mixture under Rademacher factor noise, 10,000-stream all-null crossing rates at rho=.2/.5/.8 were 0.02% / 0.36% / 1.80%; sparse power 82.49% / 83.51% / 82.50%; dense power 88.51% / 84.10% / 81.21%. Under the .2-to-.8 switch, rates were 0.43% / 82.91% / 86.41% for all-null / sparse / dense.

These simulated crossing rates are not the validity proof. They do not establish closed-FDR, stopped-FDR, FWER or SupFDR. A proper confirmatory run must use frozen seeds, at least 50,000 streams per cell, and the matched guarantee protocol in #9717.

## 7. Reproducible runner

Run:

```bash
python validation/benchmark_covariance_preconditioned_betting.py \
  --seed 9756 --streams 20000 --dimension 5 --horizon 60 \
  --output covariance-preconditioned-results.json
```

The runner includes Gaussian and Rademacher-factor noise; static rho=.2/.5/.8 and a predictable .2-to-.8 switch; null, sparse, dense and heterogeneous alternatives; coordinate-local, scalar, coordinate-mixture and preconditioned experts; fixed-prior learning-rate averaging; and Wilson intervals.

Tests are authored but have not been executed from a repository checkout in this environment. No CI pass is claimed.

## 8. Prior art and benchmark relationship

Predictable adaptive betting is established. The `seqcomp` adaptive-betting vignette describes aGRAPA and ONS-m-style bet selection for bounded score-difference martingales and emphasizes that temporal structure changes which betting rule is best. Those procedures are useful bet-selection comparators, but their null and increment assumptions are not automatically identical to this multivariate conditional-mgf setting.

The 2026 UAI online e-closure/compound-e-value work reports arbitrary-dependence online FDR improvements and O(log t) decision algorithms for a single-pass streaming problem. Our dynamic target additionally requires the prefix/projectivity property when existing hypotheses continue accruing evidence after later arrivals. Compare on a separate matched leaderboard; do not claim superiority from these intersection-level simulations.

## 9. Files

- validation/projective_vector_betting.py
- validation/test_projective_vector_betting.py
- validation/benchmark_covariance_preconditioned_betting.py
- validation/test_benchmark_covariance_preconditioned_betting.py
