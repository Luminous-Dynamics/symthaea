# HIST-CAUSAL-001 v0.11: persistent-cluster confounding and assignment-unit validity

## Question

What happens when the experimental unit is a family/campaign, but the data contains many participants and repeated measurements sharing a persistent latent regime?

## Synthetic design

- 64 independent clusters, 32 randomized to treatment and 32 control.
- 16 participants nested within every cluster.
- Three repeated probe measurements per participant.
- A persistent Gaussian cluster intercept with SD 1.0 is shared by all observations in a cluster.
- A participant intercept with SD 0.5 is shared across that participant's repeated probes.
- Measurement noise SD 1.0.
- True treatment effect 0 for the null run.
- 1,000 independent Monte Carlo experiments.
- 999 cluster-level randomization permutations per experiment.
- Fixed seed 20261103.

The actual randomization unit is the cluster. Participant/probe observations are nested and correlated.

## Results under the true null

| Analysis | False-positive rate at 5% | Approx. 95% Wilson interval |
|---|---:|---:|
| Invalid participant-level Welch test | 68.2% | 65.25–71.01% |
| Welch test on 64 cluster means | 5.0% | 3.81–6.53% |
| Randomization test permuting 32/64 clusters | 4.9% | 3.73–6.42% |

The naive analysis counts 3,072 repeated observations as if they were independent. Under this deliberately strong but plausible intracluster dependence, it rejects the true null in 68.2% of simulations. Cluster-mean Welch and cluster-level randomization are close to the nominal target.

Power sensitivity under the same design:

| True treatment effect | Participant-level invalid | Cluster Welch | Cluster randomization |
|---:|---:|---:|---:|
| 0.0 | 68.2% | 5.0% | 4.9% |
| 0.2 | 76.8% | 12.6% | 12.5% |
| 0.4 | 90.0% | 35.2% | 34.5% |
| 0.6 | 97.1% | 64.3% | 64.2% |

The invalid test appears more powerful only because it is badly anti-conservative. It cannot be compared as a legitimate baseline.

## Interpretation and limits

The randomization test permutes assignment at the actual cluster unit. Under the sharp null and the declared random assignment, it remains valid in the presence of arbitrary within-cluster dependence. It does not protect against cross-cluster interference, treatment-dependent attrition, assignment violations, or non-exchangeable cluster recruitment.

The simulation uses a Gaussian random-intercept model. It is a stress test, not a universal guarantee and not an empirical result.

## Integration

This is a separate persistence/cluster stratum for Symtropy SIM-REL-017 and SIM-REL-014. Do not pool these results with independent-observation OOD tests. Preserve cluster IDs, assignment unit and intervention carryover in every benchmark receipt. It complements the v0.10 Berger–Boos nuisance-grid proof fixture but does not itself solve unknown continuous nuisance parameters.

Next: persistent non-Gaussian latent regimes; cluster-level bootstrap comparison; interference across clusters; site/batch randomization; missingness/attrition; and cluster-aware multi-model catalogue adequacy.
