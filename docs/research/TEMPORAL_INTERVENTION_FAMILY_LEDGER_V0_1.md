# Temporal intervention family ledger adapter v0.1

**Tracking:** NEURO-SILICON-014 / #9929  
**Implementation:** \`crates/domains/symthaea-architecture-engineering/src/temporal_intervention_family.rs\`  
**Composes:** NAS anytime family/look ledger #9464 / PR #9465; temporal audit #9561 / PR #9881; controller convergence #9882; blocked randomization #9670.  
**Status:** Rust reference implementation and tests authored; no Rust build/test/CI or statistical benchmark execution claimed.

## 1. Purpose

PR #9881 provides two distinct inference paths for paired temporal interventions:

1. exact fixed-sample sign-flip randomization inference under the Fisher sharp no-effect null;
2. an anytime-valid fixed-fraction paired-sign e-process for independently randomized fair-coin pairs under the same sharp null.

Both paths initially accepted a caller-supplied comparison-family size. A free integer does not establish that every policy contrast and every outcome axis was declared before outcomes were inspected.

The repository already has a generic frozen family manifest and append-only confidence-look ledger in #9465. This adapter reuses it. It does not reimplement family-size allocation, per-look alpha spending, cumulative replicate-prefix recomputation, receipt chaining, JSON replay or generic ledger verification.

## 2. Canonical family definition

Each preregistered treatment/control contrast declares:

- stable contrast ID;
- task digest;
- candidate model digest;
- evaluator digest;
- treatment-policy digest;
- control-policy digest;
- assignment-manifest digest.

A contrast must have distinct treatment and control policy digests. All identifiers and digests are validated, and contrast declarations must be strictly sorted and unique by contrast ID.

Each contrast expands to exactly three streams:

- wrong-belief exposure;
- abstention exposure;
- missing-output exposure.

Every stream declares bounded support [0,1] and decision threshold 0.5. The stream ID embeds the contrast ID, a canonical domain-separated SHA-256 identity digest over all contrast fields, and the axis name.

The generic family manifest sorts and hashes the complete stream set. With K contrasts, the family size is exactly 3K; callers cannot independently supply a smaller family-size integer. At family alpha 0.05 and K=2, each of six streams receives stream alpha 0.05/6.

The family manifest digest must be committed to an external preregistration record before outcomes are exposed. A hash does not prove chronology or custody; use a trusted timestamp, signed commit, transparency log or immutable external record where required.

## 3. Paired outcome encoding

For each evaluator-validated randomized pair and each exposure axis, the adapter consumes the measured treatment and control exposure values, episode duration, treatment orientation, assignment scheme and treatment probability. It requires independent fair-coin assignment with treatment probability 0.5, a positive episode duration, and both exposure values no greater than that duration. It derives:

\[
d_i = Y_i^{control} - Y_i^{treatment}.
\]

A caller cannot supply a precomputed benefit sign. The encoded value is derived from the two measured arm exposures. Encode:

| Paired benefit | Bounded observation | Interpretation |
|---|---:|---|
| d_i > 0 | 1.0 | Treatment reduced exposure |
| d_i = 0 | 0.5 | Tie |
| d_i < 0 | 0.0 | Treatment increased exposure |

Under the Fisher sharp no-effect null, if treatment orientation is independently randomized with probability 0.5 within every pair, the non-tie sign is symmetric and the encoded observation has conditional expectation 0.5. Ties deterministically contribute 0.5.

The generic anytime bounded-mean ledger evaluates the mean sign score against threshold 0.5. Its Hoeffding alpha-spending confidence sequence is a conservative, time-uniform comparator to #9881's specialized paired-sign martingale. It is **not** the same statistic as the fixed-fraction e-process, and its power/decision semantics must be reported separately.

The per-stream confidence sequence requires independent bounded replicate observations. Independence is supplied by independent randomized pair assignments only under the declared design and no-interference assumptions. Bonferroni across the complete stream family does not require independence between exposure axes or policy contrasts.

## 4. Pair identity and replay binding

The generic look ledger hashes numerical replicate batches. The temporal wrapper additionally records, per stream/look:

- pair IDs;
- evaluator context digests;
- source-family digests;
- assignment-receipt digests;
- evaluator-realized signed exposure benefits;
- derived stream identity and look index;
- pair-batch digest;
- corresponding generic confidence-receipt digest.

The verifier reconstructs the bounded sign observations from the pair records and checks exact equality with the generic ledger batch. It then replays the generic ledger and validates the outer evidence-ledger digest.

Append operates transactionally on a clone and commits the new state only after the resulting wrapper verifies. Duplicate pair IDs, context digests or assignment-receipt digests within one stream are rejected across all looks.

The pair binding is still a commitment, not proof that the assignment was genuinely randomized. The upstream evaluator must verify assignment signatures, randomization receipts, source-family design, candidate/evaluator isolation and the no-interference assumptions.

## 5. What is not supported

- Blocked/fixed-count assignment and cluster-level assignment are not accepted by the independent fair-coin sign interpretation. Use a design-specific assignment-space method such as #9670.
- Interference, cross-pair carryover, shared mutable episode state or common exogenous coupling can invalidate independent pair signs; randomize/analyze at the correct independent cluster level.
- A confidence interval crossing 0.5 is inconclusive, not evidence of equivalence.
- A sign-score confidence sequence does not estimate effect magnitude. Keep exposure means, quantiles, CVaR and resource costs as separate estimands.
- The ledger cannot prove external preregistration chronology, signer identity, randomization quality or physical measurement validity.
- The Hoeffding stream is conservative and may be less powerful than #9881's e-process or the induced terminal-test sequence in #9820.

## 6. Qualification tests authored

Fifteen Rust tests cover:

1. exactly three sorted streams per contrast;
2. sorted/unique contrast declarations;
3. invalid SHA-256 and identical-policy rejection;
4. contrast metadata mutation invalidating the frozen family;
5. exact sign encoding for positive/negative/tied outcomes;
6. family-size derivation and full replay verification;
7. strong positive sign stream reaching the declared threshold;
8. sequential look-index and cumulative-prefix progression;
9. duplicate pair/context/assignment-receipt rejection without ledger mutation;
10. undeclared contrast and empty-batch rejection;
11. pair-sign and generic-ledger tampering detection;
12. missing pair binding and outer-digest tampering detection;
13. JSON round-trip with full replay;
14. exact three-axis family cardinality.

These tests are authored but have not been compiled or executed.

## 7. Benchmark plan

Compare on identical evaluator-validated pair sequences:

- #9881 fixed-fraction paired-sign e-process;
- this adapter's Hoeffding alpha-spending confidence sequence;
- #9820's induced fixed-horizon test when a valid terminal test is defined;
- naive repeatedly inspected fixed-look tests as invalid negative controls.

Report per contrast and axis:

- familywise false-crossing probability;
- time-to-crossing and inconclusive rate;
- paired sign rate and realized exposure magnitude;
- calibration and prediction error separately from statistical test results;
- memory/compute/verification and measured energy as separate resource axes.

Use exact finite null enumeration for small horizons and at least 100,000 independently generated study families for exploratory simulation; use held-out seeds and Monte Carlo uncertainty for frozen confirmatory simulations. Simulated design validity does not replace verifying real assignment receipts.

## 8. Claim ceiling

A PASS may establish that temporal contrast declarations map correctly into the existing generic family ledger, that family size is derived from the complete declared contrast/axis set, and that pair sign batches replay to the exact corresponding confidence receipts.

It does not establish randomization quality, independence, no interference, external preregistration chronology, real-world causal generalization, tail-risk improvement, energy efficiency, production readiness or SOTA superiority.
