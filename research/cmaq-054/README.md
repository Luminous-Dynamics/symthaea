# CMAQ-054 — Evidence-adaptive multi-scale Fourier kernel family

## Purpose

CMAQ-050 provides one continuous-covariate product-Fourier paired monitor. CMAQ-051 combines two fixed representation families. CMAQ-054 generalizes that idea to a frozen, registered family of product-Fourier experts spanning response and covariate frequency scales.

This direction is motivated by Sreedhar & Li, *Anytime-Valid Two-Sample Testing over Kernel Families* (ICML 2026 Workshop). That work already studies anytime-valid mixtures over kernel families and evidence-driven online reweighting. CMAQ-054 does not claim that mixture principle as new. Its engineering target is a resource-bounded conditional paired monitor whose inventory, alpha allocation, exact prior weights and transactional update semantics are bound into CMAQ-053's compositional risk plan.

## Conditional null and expert process

Every expert k freezes:
- response Fourier map phi_k;
- shared-covariate Fourier map psi_k;
- a convex target S_k;
- response/covariate reference and target digests;
- a positive rational prior weight w_k.

For a paired sample (Z_t, X_t, Y_t), it observes

    r_(t,k) = 0.5 * psi_k(Z_t) tensor [phi_k(Y_t) - phi_k(X_t)].

Unit-norm Fourier maps give ||r_(t,k)||_2 <= 1. Under the common conditional paired-distribution null, each member has a conditionally centered bounded observation and feeds a CMAQ-046 e-process.

The prior weights are immutable and satisfy sum_k w_k=1. The family wealth is

    E_t = sum_k w_k E_(t,k),

which remains a nonnegative supermartingale under the common null even when expert processes are dependent. Its evidence-posterior is

    q_(t,k) = w_k E_(t,k) / E_t.

The posterior describes which registered expert currently explains the evidence. It does not change prior weights, maps, targets, alpha or null assumptions. Rejection occurs at E_t >= 1/alpha_test from the CMAQ-053 composed risk plan. Each child receives the exact threshold alpha_test*w_k; therefore an individual expert cannot be stopped by its own threshold before the corresponding mixture contribution alone has reached the family rejection boundary.

## Implementation contract

- Bounded expert count and total feature dimension.
- Fixed response and covariate Fourier maps per expert, with immutable map/target/reference bindings.
- Fixed positive rational priors normalized exactly to one.
- Requires the portfolio inventory digest to equal the representation_inventory_sha256 in the immutable CMAQ-053 risk plan.
- Child alpha values are computed from alpha_test times prior weight and reduced exactly before conversion to the CMAQ-046 u32 interface. Unrepresentable thresholds fail closed.
- One global monotone pair sequence and pair-ID replay inventory.
- All feature transforms, child updates, mixture log-sum-exp and posterior weights are staged before a single family commit.
- Child monitors keep only bounded O(d) state; pair IDs are stored once at the family level.
- Stable log-domain mixture arithmetic and bounded expert/feature counts.
- The family stops permanently at its composed portfolio threshold.

The e-value and posterior are necessarily evaluated in floating-point log space. The authoritative fleet/reference/representation alpha allocations and prior normalization remain exact rational quantities. Numerical tolerance and overflow behavior are covered by tests; this is not a formal verified-numerics implementation.

## Tests and deterministic probe

The Rust crate has no third-party dependencies and tests:
- exact rational prior normalization and member-alpha derivation;
- duplicate expert IDs, dimension mismatches and inventory-binding mismatch;
- initial wealth normalization and posterior sum;
- deterministic scale-selective alternative where one registered response frequency is deliberately blind and another resolves the change;
- null/no-difference paired streams;
- pair replay and sequence rejection;
- fail-closed input errors and all-expert transactional updates;
- bounded feature and expert inventories.

Run:

    cargo +1.100.0 fmt --manifest-path research/cmaq-054/Cargo.toml --all -- --check
    cargo +1.100.0 test --manifest-path research/cmaq-054/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-054/Cargo.toml --locked --all-targets -- -D warnings
    cargo +1.100.0 run --release --manifest-path research/cmaq-054/Cargo.toml --locked --bin kernel_family_probe

The probe is a deterministic wiring and scale-separation check. It is not a Type-I calibration study, an SCPT reproduction, or an SOTA benchmark.

## Required frontier comparison

Run identical conditional-pair streams and covariates across:
1. each fixed expert alone;
2. CMAQ-050 single product-Fourier monitor;
3. CMAQ-051 context/product portfolio;
4. CMAQ-054 fixed-prior kernel-family mixture;
5. faithful ICML 2026 kernel-family betting implementation;
6. faithful SCPT (Zheng & Zhang, arXiv:2611.22886);
7. pooled mean-difference and single-kernel baselines.

Primary alternatives include smooth low-frequency effects, sharp/high-frequency effects, sparse covariate regions, oscillating conditional effects, equal-marginal conditional changes, conditional scale/tail/skew changes and local covariate changes. Negative controls include covariate-only shift with unchanged conditional response law, equal-pair nulls, common-cause/duplicate-pair attacks and alternatives deliberately orthogonal to the registered finite feature family.

Primary cells should use at least 10,000 independent full-horizon null paths and 5,000 alternative paths. Report full-horizon ever-alarm and confidence intervals, unconditional detection and miss rate, H+1-restricted delay, p50/p90/p95 delay, worst expert/family, posterior-weight dynamics, p99 latency, memory and total feature count. Paired comparisons must resample effective common-cause blocks, not overlapping rows.

## Claim boundary

The finite registered Fourier family is not a universal characteristic-kernel test. All members must test the same conditional null, use predictable shared covariates and have valid targets. A fixed prior mixture preserves validity; selecting a kernel after observing outcomes and then reporting that selected member at the original alpha does not. The family cannot add experts or retune feature maps after observing the monitored stream. No advantage over ICML 2026 or SCPT is claimed until faithful head-to-head reproduction succeeds.