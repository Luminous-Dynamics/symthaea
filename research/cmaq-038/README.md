# CMAQ-038 — Multi-resolution predictive-rank spectrum monitor

## Research question

Can a pre-registered, multi-resolution basis over the entire finite predictive-rank grid detect higher-order distribution changes that are nearly invisible to the hand-selected Order and Dispersion features in CMAQ-031 / CMAQ-033, while retaining an anytime-valid null alarm argument under the same fixed-reference exchangeability assumptions?

This is a comparative monitor, not a new conformal theorem. It does not claim arbitrary-shift detection or a guarantee that a non-alarm means the environment is unchanged.

## Implemented monitor

For calibration size n, the predictive-rank support has m=n+1 buckets. CMAQ-038 constructs a deterministic, pruned dyadic Haar contrast tree over those buckets. Padding to the next power of two is allowed, but contrasts with an empty side are discarded.

Each contrast is a function h in [-1/2,1/2]. At observation t, for rank R_t, the predictive null law is

    pi[t,j] = (1 + N[t-1,j]) / (n + t),

where N[t-1,j] is the number of previous deployment ranks in bucket j. The monitor computes

    Z[t,h] = h(R_t) - sum_j pi[t,j] h(j).

For each contrast, it maintains two fixed-bet components, with betting directions +Z and -Z:

    E[t,h,+/-] = 1 + (1/2) * (+/- Z[t,h]).

Since h is in [-1/2,1/2], Z is in [-1,1], so every factor is at least 1/2. The mixture has fixed normalized weights: equal mass per active resolution level, then equal mass per contrast at that level and per direction. The monitor quarantines permanently when the mixture wealth reaches 1/alpha.

The pruned tree contains exactly m-1=n contrasts and 2n directional components. For n=99, that is 99 contrasts and 198 components.

The current implementation computes a single integer prefix sum over predictable bucket weights for each observation. Each Haar contrast then obtains its conditional mean with two prefix differences. This is O(m) preprocessing plus O(m) component updates per rank, rather than rescanning all m buckets for every one of the O(m) features.

## Trust and claim boundary

The monitor consumes an already-qualified one-based rank. It does not:
- authenticate source traces, aggregate scores, signatures, signer authority or tie-breaking;
- establish common-cause closure or physical independence;
- validate that the score/rank producer used the same fixed reference;
- establish conditional coverage or power against every distributional alternative;
- detect hidden mechanism shifts that leave the monitored score distribution unchanged;
- certify physical recoverability, actuator behavior or safety.

CMAQ-031 / CMAQ-032 and the governed evidence-ingestion authority remain responsible for score custody, rank construction, receipt checks, ties, cluster identity and the fixed-reference binding. CMAQ-036 closes only declared dependence keys; it does not prove independence when keys are disjoint.

The anytime type-I argument is conditional on the exact predictive-rank null for an immutable reference, with valid rank construction and tie handling. The finite feature mixture preserves that argument; it does not remove the null assumptions.

## Paired executable probe

Run from the repository root:

    cargo +1.100.0 run --release --manifest-path research/cmaq-038/Cargo.toml --locked --bin paired_probe -- 10000 500 20261127 > cmaq-038-results.csv

Arguments are replication count, horizon and seed. Defaults are 1,000, 500 and 20,261,127. CSV is written to stdout; configuration and paired-discordance counts are written to stderr.

The executable compares:
1. cmaq031_fixed_four: the actual CMAQ-031 four-feature fixed-bet construction, using its public centered_rank_payoff implementation, equal mixture and lambda=1/2;
2. cmaq038_haar_spectrum: the new complete multi-resolution rank-grid feature bank.

The two methods receive exactly the same ranks within each replication.

### Probe scenarios

- predictive_rank_null: exact Pólya-urn draws from (1+N[t-1,j])/(n+t), so the generator matches the fixed-reference predictive rank law rather than drawing a naive IID uniform rank sequence.
- trimodal_rank_grid: draws ranks 11, 51 and 91 with probabilities 31.25%, 37.5% and 31.25%. This alternative has idealized mean rank 51 and mean absolute deviation 25 rank units around its center; it deliberately alters higher-order shape.

This alternative is a constructed rank-space challenge. It is not the raw-score generator used in CMAQ-035, not a reproduction of the Kuang–Xia PRM paper, and not the CCTM comparison. Report it as an exploratory feature-coverage experiment only.

Metrics are detection/alarm rate with Wilson intervals and conditional median / p90 / p95 delay. A delay is blank when a method does not alarm. Under the null, alarm rate—not paired timing wins—is the relevant measure. The simulation PRNG is deterministic and non-cryptographic.

## First exact-head smoke result

Exact-head GitHub Actions run 20 passed formatting, all 10 deterministic tests, strict Clippy, the paired smoke benchmark and artifact preservation at commit `5cf164d7f209b7d9dc21b59f6119a4271608393f`.

The preserved 64-replication, horizon-200 smoke artifact reported:

| Scenario | CMAQ-031 fixed-four | CMAQ-038 spectrum |
|---|---:|---:|
| Predictive-rank null alarms | 0/64 | 0/64 |
| Trimodal rank-grid alarms | 0/64 | 64/64 |
| Trimodal median delay | — | 91 |
| Trimodal p90 / p95 delay | — | 97 / 97 |

Both null arms' 95% Wilson intervals were 0–5.66%; 64 replications are not enough to establish compliance with a 5% anytime alarm budget. The trimodal result is a small deterministic smoke probe, not an estimate of general power or evidence of superiority. The longer registered experiment remains 10,000 replications per scenario, followed by the full CMAQ-037 matrix.

Artifact: `cmaq-038-rank-spectrum-smoke`, GitHub Actions run `46004264438`, artifact digest `sha256:790392f6c191a50aeed9bd8b1644cc125dfa5d409acf0f599e532fb455e647a0`.

## Deterministic tests

The library tests cover:
- every pair of rank buckets separated by at least one feature in the complete dyadic basis;
- exact count of 2n directional components;
- normalized mixture weights and unit initial wealth;
- conditional centering against the predictive rank law after nontrivial prior counts;
- nonnegative betting factors over the complete rank support;
- transactional rejection of invalid rank values;
- permanent quarantine after a deterministic trimodal stress sequence;
- invalid calibration / alpha / support-size rejection.

## Qualification

The dedicated GitHub workflow runs Rust 1.100.0 formatting, tests and strict Clippy, followed by a small paired smoke benchmark. The 10,000-replication run is a separately reproducible experiment, not a CI gate.

No exact-head CI pass, SOTA advantage, physical safety result, or industry superiority is claimed until independently observed and supported by artifacts. The local authoring environment may not have the Rust toolchain; in that case the workflow is the software qualification path.

## Required next comparisons

CMAQ-038 should be added as an exploratory arm—not substituted for any published comparator—in the CMAQ-037 matrix. The registered comparison should include:
- exact CMAQ-031 fixed-four baseline;
- CMAQ-033 ONS order/dispersion monitor;
- CMAQ-038 multi-resolution Haar spectrum;
- faithful CCTM implementation checked against the official reference code and paper vectors;
- conditional CTM / WATCH or other methods only where estimands and assumptions match.

Expand over effective independent calibration block counts 19, 49, 99, 199, 499 and 999; alpha 0.01, 0.05 and 0.10; horizons 100, 500 and 2,000; continuous and tied scores; and pre-registered location, scale, tail, multimodal, gradual, transient and autocorrelation alternatives. Run at least 10,000 null replications per registered cell and enough alternative replications to estimate power intervals.

Report worst-family detection power, false-alarm probability, p50/p90/p95 delay, feature-count-dependent runtime, memory, and score-ingestion-to-quarantine latency. Dependence-violating cells must be labelled separately; never present them as satisfying the exchangeability null.

The primary scientific question is not whether more features always detect shifts faster. It is the detection-power / null-alarm / compute frontier, including cases where the larger mixture dilutes mass and therefore loses to a smaller, targeted basis.
