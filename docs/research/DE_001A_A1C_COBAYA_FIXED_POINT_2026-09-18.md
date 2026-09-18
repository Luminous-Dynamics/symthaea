# DE-001A1C — Released Cobaya fixed-point contract

**Status:** preregistered contract only; no A1C likelihood execution is reported here.

**Scientific authority:** `NONE`

## Purpose

A1C is the second implementation lane for the already-frozen DESI DR2 all-tracer BAO point. It is deliberately narrower than an optimizer reproduction.

A1C asks only:

> When the exact A0-authenticated DESI mean/covariance and the exact A1P-authenticated `(Omega_m, h*r_d)` point are evaluated once through the released Cobaya 3.6.2 DESI DR2 BAO likelihood, does the resulting BAO chi-square reproduce the frozen reference within the already-preregistered tolerance?

It does not sample, minimize, tune parameters, install data, access the network, or make a cosmological claim.

## Frozen released-likelihood identity

- Cobaya: `3.6.2`
- source commit: `899f30a49f85de610dac321e91a1af50018e56aa`
- source-distribution SHA-256: `8f1061d6347427f08380e1e0c0b766d695d3978b5439fb0b1cc1a7002152d9c8`
- likelihood alias: `bao.desi_dr2`
- class: `cobaya.likelihoods.bao.desi_dr2.desi_bao_all.desi_bao_all`
- likelihood YAML SHA-256: `fd7e9bf2dcf5ffee90a9a30b18227f4337d6d5c1978782c63513cbe0d8280daa`
- BAO data repository commit: `bb0c1c9009dc76d1391300e169e8df38fd1096db`

The pinned Cobaya source aliases `bao.desi_dr2` to the released all-tracer class. That class inherits Cobaya's generic BAO implementation, which obtains the required DESI BAO predictions from provider-supplied angular-diameter distances, Hubble values, and `rdrag`.

## Exact subject parity with A1R

A1C is not permitted to choose a new point or tolerance. The contract checker requires bit-exact equality with the frozen A1R manifest for:

- model identity;
- DESI best-fit artifact SHA-256;
- mean/covariance roles, sizes, SHA-256 values, and 13-row count;
- `Omega_m = 0.29717787`;
- `h*r_d = 101.54786 Mpc`;
- reference `chi2_BAO = 10.282299`;
- absolute reproduction tolerance `0.01`.

Any difference is `INVALID` contract provenance, not a scientific negative result.

## Gauge-normalized background provider

BAO-only ratios at fixed flat-LambdaCDM `Omega_m` depend on the product `H0*r_d = 100*(h*r_d)`, not on a separately inferred `H0` and `r_d`.

A1C therefore freezes a computational gauge:

- `rdrag_gauge_mpc = 100.0`;
- `H0 = 100 * (h*r_d) / rdrag_gauge_mpc`.

At the frozen point this makes the numerical `H0` value equal to `101.54786 km/s/Mpc`. This is **not a physical H0 inference** and **not a claim that rdrag is 100 Mpc**. It is only a normalization that lets the released Cobaya BAO interface receive `D_A(z)`, `H(z)`, and `rdrag` while preserving the exact BAO ratios implied by the two-parameter subject.

The provider must use `c = 299792.458 km/s`, matching the constant frozen in A1R and the pinned Cobaya source.

## Forbidden operations

The A1C execution contract requires all of the following:

- exactly one likelihood evaluation;
- no sampler;
- no minimizer;
- no optimization;
- no parameter mutation;
- no CAMB use;
- no network access;
- no Cobaya data installation;
- no mutation of the supplied package/data path.

A future implementation that violates any of these rules is not A1C.

## Independence scope

A1C and A1R intentionally share the same DESI compressed measurement and covariance, the same flat-LambdaCDM model family, and the same frozen point.

The useful implementation diversity is narrower:

- A1R uses Rust background geometry and its own Gaussian covariance evaluation;
- A1C will use a separately implemented Python background provider and Cobaya's released observable mapping / Gaussian BAO likelihood.

Therefore A1C is not independent data reduction and not independent model-family evidence. Its purpose is software-path agreement.

## Execution prerequisites

The contract explicitly does **not** authorize execution. A1C may become executable only after both prerequisites have qualified:

1. DE-001A1Q evidence-bundle PASS for the A1R fixed-point chain;
2. DE-001A numerical-environment qualification PASS.

Only then should a new execution subject bind the environment receipt, A0 receipt, A1P/A1Q lineage, A1C contract hash, Python evaluator hash, exact Cobaya package identity, one-call output, and postflight immutability.

## Next gate

After a qualified A1C result exists, DE-001A1X should compare A1R and A1C without averaging them away. Prediction-vector and chi-square disagreement must remain a first-class result.

Even an A1X PASS has reproduction authority only. It does not establish LambdaCDM, dynamic dark energy, or an observational anomaly.
