# HARDSEC-021 — Atom-Safe Randomized PIT for Sequential CCTM

**Status:** Rust randomized-PIT CCTM candidate, versioned schemas and synthetic known-answer fixture authored; no physical acquisition or randomization guarantee claimed  
**Issue:** [#9485](https://github.com/Luminous-Dynamics/symthaea/issues/9485)  
**Stacked on:** [HARDSEC-020 context-assignment auditor](https://github.com/Luminous-Dynamics/symthaea/pull/9484) and [HARDSEC-019 CCTM](https://github.com/Luminous-Dynamics/symthaea/pull/9478)

## 1. Problem: score atoms break the ordinary PIT shortcut

The ordinary probability integral transform (F(X)) is uniform under an i.i.d. null when the score distribution is continuous. If the score distribution has atoms (repeated integers, quantized scores or rounded outputs), (F(X)) is not generally uniform. Hardware telemetry and score pipelines can produce such ties.

HARDSEC-019's original deterministic CCTM used a fixed empirical CDF (hat F(X)), which leaves score continuity/atom treatment as a material assumption. HARDSEC-021 introduces an explicit randomized PIT and reports atom diagnostics rather than silently assuming a continuous score distribution.

## 2. Randomized PIT construction

Let (Xsim P), with CDF (F), and let (Usimmathrm{Uniform}(0,1)) be independent of (X). Define:

[
G(X,U)=F(X^-)+U{F(X)-F(X^-)}.
]

This randomized probability integral transform is uniform on ([0,1]) for arbitrary distributions, including discrete distributions with atoms, provided (U) is independent and uniform.

For a fixed empirical reference CDF (hat F_0), HARDSEC-021 uses:

[
hat G_0(x,u)=hat F_0(x^-)+u{hat F_0(x)-hat F_0(x^-)}.
]

The implementation obtains (L=#\{r_i<x\}), (H=#\{r_i\leq x\}), and computes:

[
hat G_0(x,u)=\frac{L+u(H-L)}{n}.
]

### Why the DKW correction is retained

On the DKW event (sup_x|hat F_0(x)-F(x)|\leq\epsilon), the same sup-norm bound applies to left limits. The randomized interpolation is a convex combination of the two endpoints, so:

[
|hat G_0(x,u)-G(x,u)|\leq\epsilon.
]

Thus the same conservative DKW half-width can be used in the CCTM betting correction, conditional on the DKW event and the independence assumptions. This is a derived extension of the paper method, not a claim that the authors' current software implements randomized PIT.

## 3. Randomizer source and fail-closed contract

Each stream record now carries:
- unique `randomizer_id`;
- one `pit_randomizer_uniform` draw in ([0,1));
- the plan and stream bind a separate `pit_randomizer_source_sha256`;
- a predeclared policy `independent_uniform_after_score_acquisition_v1`.

The randomizer must be generated after the score is acquired, by an independently operated CSPRNG. The auditor checks range, unique draw IDs, exact stream binding and randomizer-source digest. It cannot prove that draws are uniform or independent of scores, context assignment or the full past. The receipt therefore keeps `randomizer_independence_status=assumption_not_verified`.

The monitor also requires the truth-free HARDSEC-020 context-assignment receipt. Invalid sensors, unqualified PMU acquisition and abstained/unseen workload contexts do not update any wealth process.

## 4. DKW-CCTM and assumptions

HARDSEC-021 retains the published DKW-corrected CCTM:
- per-context (epsilon_c=\sqrt{\ln(2/\delta_c)/(2n_c)});
- (C_c=1/(0.5+\sqrt{1+k^2}\epsilon_c));
- the paper's corrected betting function and predictable ONS update;
- Bonferroni alpha/delta allocation across the finite predeclared contexts.

The synthetic plan uses total (alpha=0.04), total (delta=0.01), and two contexts, giving (alpha_c=0.02) and (delta_c=0.005). The conservative unconditional bound is (alpha+delta=0.05) only if the paper's assumptions hold.

The following remain explicitly unverified:
- i.i.d. null score sequence within each context;
- i.i.d. reference scores within each context;
- context assignment exogeneity (audited by HARDSEC-020, but not authenticated);
- randomizer independence and uniformity;
- fixed scoring-function identity/authenticity;
- actual independence of reference-group IDs.

Randomized PIT fixes ties/atoms. It does **not** fix serial dependence, nonstationarity, attacker-controlled dependence or invalid context assignment.

## 5. Executable and fixture

- Rust executable: `examples/hardsec_context_cctm_randomized_pit.rs`
- Harness feature: `hardsec-context-cctm-rpit-monitor`
- Plan schema: `schemas/hardsec-context-cctm-rpit-plan-v1.schema.json`
- Stream schema: `schemas/hardsec-context-cctm-rpit-stream-v1.schema.json`
- Fixtures: `tests/fixtures/hardsec/context-cctm-rpit-v1/plan.json` and `stream.json`

The fixture has two contexts, 600 reference scores each, 19 windows, unique per-window randomizer IDs, and one declared randomizer source. It reuses the HARDSEC-020 truth-free context receipt. The synthetic CCTM crossing remains at sequence index 16; three records abstain.

The receipt reports:
- per-context reference atom counts and duplicate occurrences;
- per-context test-stream score tie counts;
- each randomized PIT value, randomizer ID and the reference lower/tie counts;
- the randomizer-source artifact digest and unverified independence assumption;
- the CCTM's DKW/alpha allocations and theorem assumptions.

Twenty-eight authored Rust tests cover the randomized-PIT algebra, atom interpolation, DKW/ONS mechanics, context receipt binding, invalid randomizer ranges, duplicate randomizer IDs, randomizer artifact mismatch and prior CCTM failure cases.

## 6. Run

```bash
cargo test --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-context-cctm-rpit-monitor \
  --example hardsec_context_cctm_randomized_pit

cargo run --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-context-cctm-rpit-monitor \
  --example hardsec_context_cctm_randomized_pit -- \
  --plan tests/fixtures/hardsec/context-cctm-rpit-v1/plan.json \
  --stream tests/fixtures/hardsec/context-cctm-rpit-v1/stream.json \
  --context-receipt /path/to/hardsec-020-runtime-receipt.json \
  --receipt /tmp/hardsec-021-rpit-cctm-receipt.json
```

## 7. Required benchmark comparisons

Compare:
1. deterministic ECDF CCTM (HARDSEC-019);
2. randomized-PIT CCTM (HARDSEC-021);
3. authors' pinned Python CCTM implementation;
4. static threshold, EWMA and Page-Hinkley/CUSUM controls.

Required score distributions:
- continuous scores;
- discrete 2-, 4-, 16- and 256-level quantization;
- heavy-tailed continuous scores;
- reference/sample ties with varying atom masses;
- score rounding and clipping;
- repeated-value, perfectly dependent null streams;
- independent vs reused, delayed, score-dependent and non-uniform randomizer draws.

Report false alarms over fixed monitoring horizons, detection delay, attack-effect recall, coverage/abstention, atom rates, calibration reference support, CPU and memory overhead. A randomized PIT does not validate an autocorrelated stream; dependence negative controls remain mandatory.

### Aspirational targets

- No additional loss of null calibration when moving from continuous to discrete/quantized scores under the stated i.i.d./randomizer assumptions.
- Identical or better matched-horizon false-alarm control than deterministic ECDF CCTM on continuous scores.
- Zero CCTM updates for missing/duplicate/unqualified randomizer records.
- Incremental randomization and CCTM p95 overhead below 1% of the full monitoring pipeline.
- At least 20% lower p95 detection delay than the strongest reproduced baseline at the same empirical false-alert budget.

These are research targets, not achieved performance or a theorem under arbitrary dependence.

## 8. Claim ceiling

This implementation is a derived randomized-PIT extension, not a verbatim implementation of the authors' current Python package. A separate proof audit and comparison against paper equations and pinned Python source are still required. Input digests do not authenticate the CSPRNG, randomizer independence, group identities or hardware. No physical detector result, SOTA superiority, cross-platform transfer or hardware certification is claimed.
