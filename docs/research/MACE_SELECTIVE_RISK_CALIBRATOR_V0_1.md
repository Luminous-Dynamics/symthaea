# MACE selective evidence risk calibrator v0.1

**Status:** standalone offline calibrator and four-way assessment kernel implemented with synthetic fixtures. Dedicated Rust validation is pending.

## Primary endpoint

The primary endpoint is the probability that a new independent calibration cluster contains at least one accepted decision whose evaluator-only bounded loss exceeds a preregistered unacceptable-loss threshold. Each cluster contributes at most one Bernoulli unsafe indicator at a given group and threshold. Repeated turns, assertions, evidence spans and ablation rows do not inflate the number of trials.

This is deliberately stricter than per-row selective risk. Mean bounded loss and coverage are descriptive, not separately certified per-row guarantees.

## Exact bound and post-selection control

For k unsafe clusters among n clusters with at least one accepted case, the one-sided Clopper-Pearson upper bound solves P_p(X <= k) = delta_adjusted for X ~ Binomial(n,p). The calibrator applies Bonferroni over every declared group × threshold hypothesis. The simultaneous family permits maximum-coverage threshold selection on the calibration split without unadjusted post-selection claims.

With zero unsafe clusters, a single unadjusted 95% bound is about 9.5% at 30 clusters and falls below 5% only at 59 clusters. Multiple groups and thresholds reduce adjusted delta and require more clusters. The 30-cluster floor is a minimum support gate, not a promise that a 5% risk target is certifiable.

## Hierarchical groups and fallback

Groups form an explicit parent tree. Parent data include examples assigned to descendants. A leaf may fall back only when its own accepted-cluster support is insufficient, fallback is explicitly enabled, and the declared parent has a usable certificate. The leaf certificate records the parent group and exact parent certificate digest. No fallback is used to conceal demonstrated risk exceedance or distribution shift.

## Four-way decision rule

1. Hard evidence, authority, scope, lineage and lifecycle gates fail: abstain or escalate, regardless of score.
2. Distribution shift: suspend the certificate and acquire or abstain.
3. No certificate, insufficient support or risk above target: acquire or abstain.
4. Score below the certified threshold: retrieve more, acquire, or abstain.
5. Only a matching usable group certificate, adequate score and passing hard gates permit accept-and-act.

## Guarantee assumptions and limits

- Calibration and evaluation splits are disjoint at the true independent cluster boundary.
- Cluster-level unsafe indicators are independent and identically distributed, or the sampling design otherwise justifies the exact binomial bound.
- The score, loss, policy, thresholds, group tree and hard-gate definition are frozen and hashed.
- Candidate code cannot observe evaluator losses, outcomes or calibration labels.
- Bonferroni covers the declared group × threshold family only; post-hoc groups are outside the guarantee.
- Detected shift suspends the certificate. This kernel does not prove exchangeability or detect every shift.
- Digests provide integrity, not issuer identity, signatures or independent evaluator custody.

No guarantee transfers automatically to a new policy, score, hierarchy, source distribution, language, model revision or deployment population. Confidence scores, HDC similarity, bootstrap intervals and retrieval relevance are not substitutes for the declared bound.

## Synthetic tests

Eleven deterministic tests cover the exact zero-violation bound and sample-size consequence, multiplicity conservatism, safe threshold selection, hard-gate dominance, risk-exceeds-target status, explicit parent fallback, shift and exchangeability suspension, hierarchy cycle/digest rejection, retrieve-more behavior, certificate-gated acceptance and tamper rejection.

## Run

```bash
cargo test --manifest-path research/mace-selective-risk-calibrator-v0-1/Cargo.toml
cargo clippy --manifest-path research/mace-selective-risk-calibrator-v0-1/Cargo.toml --all-targets -- -D warnings
```

The manifest, group-certificate and decision-assessment digests are recomputable integrity commitments, not signatures. This is an offline calibration kernel, not a production policy. Review its endpoint and assumptions before describing any MACE result as calibrated or certified.