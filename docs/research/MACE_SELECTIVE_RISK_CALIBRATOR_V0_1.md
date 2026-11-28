# MACE selective evidence risk calibrator v0.1

**Status:** standalone offline calibrator and four-way assessment kernel implemented with synthetic fixtures. The cluster-split firewall is now mandatory. Dedicated Rust validation is pending.

## Primary endpoint

The primary endpoint is the probability that a new independent calibration cluster contains at least one accepted decision whose evaluator-only bounded loss exceeds a preregistered unacceptable-loss threshold. Each cluster contributes one Bernoulli observation at a given group and threshold: it is selected when at least one case passes, and unsafe when any selected case exceeds the loss limit. The fixed-sequence guarantee is applied to these independent cluster-level observations; repeated turns, assertions, evidence spans and ablation rows do not inflate the number of trials.

This is deliberately stricter than per-row selective risk. Mean bounded loss and coverage are descriptive, not separately certified per-row guarantees.

## Calibration/evaluation split firewall

Calibration and evaluation inventories must be supplied separately. Each contains sorted, unique HMAC-SHA256 commitments for independent clusters, episodes and observations; raw identities are never accepted by the inventory contract. Both inventories must use the same commitment namespace, benchmark snapshot, task definition, group hierarchy, score definition, loss definition and independence-unit schema. The audit rejects any cluster, episode or observation overlap, including cases where only the episode or record overlaps. Evaluation outcome labels must be sealed and inaccessible to the candidate. The audit receipt binds both inventory digests and the complete preregistered policy/threshold configuration into the calibration manifest. Each calibration row must match the inventory's cluster, episode and observation commitments exactly. Cluster tokens are intended to be HMAC-SHA256 commitments produced by a trusted preprocessor with a key held outside the candidate. Synthetic tests use deterministic SHA-256 fixture tokens only. These checks validate declared inventories; they cannot prove the upstream identity mapping or custodian behavior without an independently authenticated preprocessor and evaluator.

## Exact bound and post-selection control

For k unsafe clusters among n clusters with at least one accepted case, the one-sided Clopper-Pearson upper bound solves P_p(X <= k) = delta_adjusted for X ~ Binomial(n,p). The default calibrator uses fixed-sequence exact Clopper–Pearson tests over a preregistered descending sufficiency-score grid (equivalently, ascending uncertainty). It allocates family-wise delta across declared groups, then stops each group's threshold scan at the first uncertified prefix. This retains a finite-sample high-probability risk guarantee without a threshold-count penalty. A simultaneous Bonferroni mode remains available when recovering later safe regions in a non-monotone risk curve is more important than power.

With zero unsafe clusters, a single unadjusted 95% bound is about 9.5% at 30 clusters and falls below 5% only at 59 clusters. Multiple groups and thresholds reduce adjusted delta and require more clusters. The 30-cluster floor is a minimum support gate, not a promise that a 5% risk target is certifiable.

## Hierarchical groups and fallback

Groups form an explicit parent tree. Parent data include examples assigned to descendants. A leaf may fall back only when its own accepted-cluster support is insufficient, fallback is explicitly enabled, and the declared parent has a usable certificate. The leaf certificate records the parent group and exact parent certificate digest. No fallback is used to conceal demonstrated risk exceedance or distribution shift.

## Four-way decision rule

1. Hard evidence, authority, scope, lineage and lifecycle gates fail: abstain or escalate, regardless of score.
2. Distribution shift: suspend the certificate and acquire or abstain.
3. No certificate, insufficient support or risk above target: acquire or abstain.
4. Score below the certified threshold: retrieve more, acquire, or abstain.
5. Only a matching usable group certificate, its integrity-verified calibration manifest, adequate score and passing hard gates permit accept-and-act.

## Guarantee assumptions and limits

- Calibration and evaluation splits are disjoint at cluster, episode and observation levels in the supplied HMAC-commitment inventories.
- Cluster-level unsafe indicators are independent and identically distributed, or the sampling design otherwise justifies the exact binomial bound.
- The score, loss, policy, thresholds, group tree and hard-gate definition are frozen and hashed.
- Decision-time assessment requires exact policy, sufficiency-score-definition and group-hierarchy digests to match the certificate; a numerically similar score or reused group name is not sufficient.
- Candidate code cannot observe evaluator losses, outcomes or calibration labels.
- Bonferroni covers the declared group × threshold family only; post-hoc groups are outside the guarantee.
- Detected shift suspends the certificate. This kernel does not prove exchangeability or detect every shift.
- Digests provide integrity, not issuer identity, signatures or independent evaluator custody.

No guarantee transfers automatically to a new policy, score, hierarchy, source distribution, language, model revision or deployment population. Confidence scores, HDC similarity, bootstrap intervals and retrieval relevance are not substitutes for the declared bound.

## Synthetic tests

Twenty-one deterministic tests cover the exact zero- and nonzero-violation bounds against independent Beta-quantile references, multiplicity conservatism, safe threshold selection, hard-gate dominance, risk-exceeds-target status, explicit parent fallback, shift and exchangeability suspension, hierarchy cycle/digest rejection, retrieve-more behavior, certificate-gated acceptance, manifest linkage/tamper rejection, three-level split overlap, label sealing, candidate isolation and inventory-to-row commitment equality.

## Run

```bash
cargo test --manifest-path research/mace-selective-risk-calibrator-v0-1/Cargo.toml
cargo clippy --manifest-path research/mace-selective-risk-calibrator-v0-1/Cargo.toml --all-targets -- -D warnings
```

The split inventory, split audit receipt, manifest, group-certificate and decision-assessment digests are recomputable integrity commitments, not signatures. This is an offline calibration kernel, not a production policy. Review its endpoint and assumptions before describing any MACE result as calibrated or certified.

**Coverage semantics:** `coverage` is independent-cluster acceptance probability, matching the cluster-risk endpoint. `decision_coverage` is separately reported as the equal-cluster-weighted fraction of individual decisions accepted. This prevents decision rows from being mistaken for independent risk trials.
