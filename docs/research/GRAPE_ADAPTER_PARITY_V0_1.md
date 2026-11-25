# GRAPE adapter parity v0.1

Issue: #13609  
Implementation: benchmark::grape_adapter_parity  
Status: Rust NLMS/RLS and four-level context primitives with authored tests; exact-head validation pending.

## Scope

The pinned upstream source is cheonwoolee/GRAPE at commit cce2d27a092abd9e397d7739e4beb6dfe0ae145e, Apache-2.0. This module reproduces selected online-adapter numerical primitives; it is not yet a complete port of the global Ridge pipeline or the full runner.

## Feature adapter

- NLMS: w <- w + eta * error * x / (||x||^2 + epsilon).
- RLS: px=P*x, denominator=lambda+xᵀPx, gain=px/denominator, w <- w+gain*error, and P <- (P-gain*(xᵀP))/lambda.
- RLS covariance is symmetrized after each update and validated with Cholesky positive-definiteness checks.
- Both updates execute transactionally against a clone; invalid dimensions, non-finite values, singular denominators or invalid covariance preserve the old state.
- RLS covariance is O(n_a^2) storage and O(n_a^2) update cost, with an additional O(n_a^3) Cholesky validation in this safety-oriented reference implementation. That validation cost must be measured before high-throughput use.

## Context hierarchy

Implements the upstream global / equipment / chamber / equipment×chamber correction:
- split equipment/chamber at the last underscore;
- update global EMA and all applicable local EMAs;
- reliability per level is n/(n+n_min);
- correction is the normalized weighted average of available levels;
- unseen contexts use only the global estimate and cannot alias a different equipment/chamber key.

## Temporal adapter

Implements the global temporal residual EMA and training-residual percentile threshold:
- linear-interpolated percentile over absolute training residuals;
- emit zero correction while |EMA| < threshold;
- otherwise emit the residual EMA.

The full GrapeModel lifecycle still needs integration: ridge model outputs, feature ranking, training initialization, base/target clipping, and exact update decomposition over chronological splits.

## Authored tests

Eight tests currently cover:
1. hand-computed scalar RLS weight/covariance;
2. RLS covariance symmetry and positive definiteness under 200 updates;
3. NLMS scalar equation;
4. failed RLS update atomicity;
5. four-level context reliability weighting;
6. unseen equipment/chamber global fallback;
7. context-count overflow atomicity;
8. temporal percentile, EMA and activation.

## Required next comparison

First compare these numerical primitives against the pinned Python implementation using static known-answer vectors. Then integrate the exact global Ridge/feature-selection and runner lifecycle for the official-code-faithful track. Keep the delayed K1-gated path separate and compare it at equal qualified-label budgets.

No paper-level reproduction or manufacturing performance improvement is claimed by this module alone.
