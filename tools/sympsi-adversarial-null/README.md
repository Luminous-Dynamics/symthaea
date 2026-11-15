# SYMPsi adversarial-null benchmark

**Status:** synthetic-only measurement qualification. This package does not run an AI participant, ingest human/field data, or test whether psi exists.

## Why a second benchmark?

The original SYMPsi harness qualifies an IID fixed-sample binomial analysis at 25% (four-choice) and 20% (five-choice). That is necessary but not sufficient. A model can have a stable label preference, a target schedule can drift, and time-varying conditions can make target and response *marginals* align even when each response is generated independently of its corresponding target.

This benchmark compares three approaches:
1. **Naive fixed 1/k binomial:** the existing fixed-sample threshold from PR #12244.
2. **Global conditional target randomization:** hold the entire response stream fixed and randomize target labels across the whole campaign, conditional on target/response category margins.
3. **Block-conditional target randomization:** hold response streams fixed and randomize target labels only within preregistered exchangeability blocks.

The conditional statistic uses the exact mean and variance of the match count under label permutation, with a one-sided normal approximation. The resulting FPR is measured by outer Monte Carlo campaigns. This is a diagnostic approximation, not yet a production confirmatory test; it must itself pass calibration and independent review.

For a controlled null, the preregistered Monte Carlo gate compares the **observed** rejection-rate estimate with the nominal alpha using a four-standard-error tolerance (plus one campaign of discreteness). The 99% Wilson interval is reported separately for inspection; it is not compared with a second four-SE allowance. This avoids counting Monte Carlo uncertainty twice. Passing means only that this deterministic simulation estimate is not unexpectedly far from the declared null probability; it does not prove a population FPR bound.

## Frozen null worlds

Each profile uses its original planned sample size and exact-binomial critical value. The manifest fixes 10,000 campaigns per scenario.

Each campaign seed is SHA-256-derived from the master seed, profile ID, scenario ID and campaign index. Target and response generation then use separate domain-separated synthetic RNG streams. Campaign seeds must not be obtained by incrementing a PRNG seed by the generator's own step constant; that creates overlapping streams and invalidates Monte Carlo independence.

| Scenario | Target/response construction | Expected diagnostic |
|---|---|---|
| uniform_iid | Independent uniform targets and responses | All three procedures control FPR |
| static_marginal_alignment | Independent targets and responses from the same biased fixed distribution | Naive 1/k rule inflates; global conditional test controls |
| target_imbalance_uniform_response | Biased target schedule; uniform independent responses | All three control FPR |
| persistent_response_preference | Uniform independent targets; response labels have an autocorrelated preference | Target randomization preserves response sequence; conditional method should control |
| blockwise_shared_drift | Independent target and response draws within each block from matching drifting distributions; response process is persistent | Naive and global pooled tests inflate; block-conditional test controls |

The two aligned-marginal scenarios are deliberately non-psi null worlds: no response is generated from its paired target. They show how hit rates can exceed 1/k through confounding or shared environment alone.

## Reproduce

From repository root:

    cargo fmt --manifest-path tools/sympsi-adversarial-null/Cargo.toml --all -- --check
    cargo test --manifest-path tools/sympsi-adversarial-null/Cargo.toml
    cargo clippy --manifest-path tools/sympsi-adversarial-null/Cargo.toml --all-targets -- -D warnings

    GIT_HEAD="$(git rev-parse HEAD)"
    cargo run --manifest-path tools/sympsi-adversarial-null/Cargo.toml -- \
      --git-head "$GIT_HEAD" \
      --manifest tools/sympsi-adversarial-null/fixtures/manifest-v0.1.json \
      --out /tmp/sympsi-adversarial-null-receipt.json

Receipt output is written only to a new path. The receipt binds the exact Git head, raw manifest digest, profile, scenario, Monte Carlo campaigns, FPRs, Wilson intervals and expected dispositions.

## Interpretation boundary

- This benchmark tests the measurement and analysis pipeline under specified synthetic mechanisms, not psi or AIA.
- SplitMix64 is deterministic synthetic fixture randomness only. It is not cryptographic and cannot choose real experimental targets.
- The global conditional method assumes target labels are exchangeable across the whole campaign under the null.
- The block-conditional method assumes exchangeability only within the preregistered blocks. If the target generator uses other structure, that structure must be modeled in a new scenario and declared before analysis.
- The conditional normal approximation must not be treated as a validated confirmatory test solely because this synthetic benchmark passes.
- Failures and inflated naive rates are successful diagnostic outcomes when the declared expected behaviour is observed.
- Real target signature verification, beacon freshness, response chronology and independent timestamp witnesses remain separate gates in SCI-PSI-003.

The goal is not a larger hit rate. The goal is a measurement system that can distinguish target information from label preference, schedule imbalance, temporal drift and other non-psi explanations.
