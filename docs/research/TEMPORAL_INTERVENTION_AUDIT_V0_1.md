# Temporal Intervention Audit v0.1

**Tracking:** NEURO-SILICON-011 / #9561  
**Related:** NEURO-SILICON-010 / #9557, #9872; joint temporal controller #9666; higher-order interaction evaluator #9876  
**Implementation:** `research/temporal-intervention-audit`  
**Status:** Rust research code and tests authored; no Rust build, test, CI or benchmark execution claimed.

## 1. Why this work is distinct

The temporal controller predicts the value of observing, retaining, refreshing, computing or verifying. The joint oracle in #9666 measures exact feasible alternatives for one small episode, and #9876 decomposes interactions within that episode. #9557/#9561 specify that temporal-value estimates must be tested against randomized retain/evict counterfactuals.

What is still missing is a multi-episode inferential layer: when repeated matched interventions are run across independently randomized episodes, does the action actually reduce evaluator-measured exposure, and how much evidence supports that conclusion?

This package implements that layer. It does not replace the single-episode subset oracle, pairwise/higher-order decomposition, continuous-time reference, adaptive resource-price controller, or temporal calibration diagnostics.

## 2. Experimental unit and data contract

Each preregistered pair contains two runs from the same episode context and source family. Exactly one run receives the treatment policy and the other receives the control policy. The treatment is independently assigned to the first or second run using a fair coin.

Each pair binds:

- unique pair and run IDs;
- shared context and source-family commitments for both arms;
- task, candidate model, evaluator, treatment policy and control policy digests;
- common randomization-manifest digest;
- unique assignment-receipt digest;
- assignment orientation and the declared 0.5 treatment probability;
- episode duration;
- candidate-predicted benefit by exposure axis;
- evaluator-measured outcome by arm.

The audit requires the full preregistered pair-ID set. This catches omitted pairs at the interface boundary; it does not prove that the preregistration manifest itself is authentic. The upstream runner must validate signed receipts, assignment generation, process isolation and the expected pair count.

## 3. Outcomes and failure semantics

Wrong-belief, abstention and missing-output exposure remain separate. The audit never sums them into an opaque scalar.

- Completed arm: use evaluator-measured exposure components and require their sum to be no greater than the episode duration.
- Explicit abstention: assign the full episode duration to abstention exposure.
- Missing response, timeout or invalid response: assign the full episode duration to missing-output exposure.

This intent-to-treat convention prevents a failed or missing run from disappearing from the denominator. A future protocol may define finer partial-exposure attribution, but it must be identical across treatment and control and fixed before confirmatory evaluation.

## 4. Exact randomization inference

For each axis, treatment benefit is control exposure minus treatment exposure. With independently randomized treatment orientation inside each pair, the sharp null of no treatment effect implies a sign-flip randomization distribution conditional on the pairwise outcome magnitudes.

The engine enumerates all \(2^n\) sign assignments. It uses Gray-code order to update the signed sum in constant time per assignment. It reports:

- candidate-predicted total and mean benefit;
- evaluator-realized total and mean benefit;
- signed prediction error and mean absolute prediction error;
- wins, losses and ties;
- exact one-sided p-value for positive treatment benefit;
- exact two-sided p-value;
- Bonferroni-adjusted one-sided p-value upper display and familywise decision.

The Bonferroni family size must include the three co-primary exposure axes and every other treatment contrast in the preregistered comparison family. This crate does not implement post-hoc selection correction for an unknown policy search.

The p-value is finite-sample exact only under the verified assignment mechanism and sharp null. It is not a confidence interval for heterogeneous treatment effects. Do not describe it as proving a constant effect, temporal stationarity, conditional coverage or generalization to unseen source families.

## 5. Relation to conformal risk and drift monitoring

The repository already has two adjacent but different lines:

- #9351/#9354: finite-state conformal risk recertification under finite exchangeability. Its guarantee must not be transferred to non-exchangeable drifting streams without a separate argument.
- #8795: a frozen-reference mixture likelihood-ratio drift monitor, with its own stated nominal-distribution assumptions and latched alarm. It is not a generic arbitrary-dependence detector.

This package does not reimplement either. Randomized intervention inference answers a different question: whether a preregistered action changes realized outcomes under a known randomization design. Future integration should require a fresh drift/readiness receipt before exporting action-value calibration, and use the existing monitor rather than a second ad hoc Page-Hinkley/CUSUM implementation.

## 6. Benchmark protocol

### Pilot

- 8–12 matched randomized pairs across development source families;
- verify assignment generation, pairing, complete manifest intake and failure accounting;
- use the pilot only to debug the protocol and estimate variance.

### Confirmatory

- freeze one treatment/control contrast, candidate model digest, task, resource ceilings and exposure horizon;
- split by generated process/source family before any candidate model or calibration fitting;
- independently randomize treatment orientation for every pair;
- preregister the familywise alpha and total comparison-family size;
- use a power analysis on development-only paired differences to determine the confirmatory pair count; do not assume 20 pairs is sufficient;
- keep all failures, timeouts, abstentions and missing outcomes in the manifest;
- retain a disjoint confirmatory family if the same study selects among multiple policies.

### Report

Publish separately for wrong-belief, abstention and missing-output exposure:

- mean and distribution of paired benefit;
- prediction bias and MAE;
- win/loss/tie count;
- exact randomization p-value and adjusted familywise decision;
- source-family and regime-stratified results;
- missing/invalid/timeout counts;
- resource costs and latency from a separately verified measurement path;
- sensitivity to pair construction and assignment protocol.

Do not report only the three exposure means. Include source-family worst cases and all planned strata. Resource effects should remain separate vector axes, with measured energy only when instrumented.

## 7. Qualification targets (targets, not results)

- 100% of preregistered pair IDs accounted for;
- zero accepted context/source-family mismatch;
- zero accepted non-fair or non-independent assignment receipts;
- zero dropped timeout, missing or invalid outcome pairs;
- exact p-value parity with an independent brute-force reference for all small test cases;
- calibration error and realized effect reported on disjoint source-family partitions;
- no claim of treatment benefit unless the predeclared familywise rule passes;
- no SOTA or energy claim without matched external baseline and physical measurement.

## 8. Next integration steps

1. Connect #9561's retain/evict paired interventions to this audit using evaluator-validated receipts.
2. Compare the multi-episode effect with #9667/#9876's single-episode exact subset and interaction estimates; disagreement is a result to investigate, not something to average away.
3. Feed only the finalized, authorized aggregate calibration receipt to the next candidate episode.
4. Bind that receipt to the candidate model/policy digest and the latest #8795 drift/readiness state.
5. Add matched resource-vector cost comparisons once the #9667/#9828 controller branches are consolidated.
6. Extend above 20 pairs only with a preregistered Monte Carlo randomization engine and exact small-n parity tests.

## 9. Claim ceiling

The implementation is an evaluator-side reference engine. Tests written in source do not establish that Rust compiles or that the tests pass. A valid exact p-value requires a real independently randomized design and verified assignment receipts. A statistically significant contrast is local to the experimental protocol. No population-wide causal effect, temporal competence, calibration guarantee, hardware efficiency, production readiness or SOTA superiority follows from this PR alone.



## 10. Anytime-valid sequential paired inference

### Why a second inference path exists

The exact fixed-sample sign-flip p-value is valid for its preregistered sample, but repeatedly inspecting a fresh fixed-sample p-value and stopping at a favorable look does not preserve its nominal type-I error. The sequential module adds a separate time-uniform path for a particularly clear design: independently randomized fair-coin treatment orientation inside every matched pair.

This is not a generic optional-stopping fix for arbitrary randomization designs. It is not a replacement for the exact fixed-sample permutation test, the general online e-process work (#8679), test induction (#9820), or finite-filtration continuation oracle (#9855).

### Betting construction

For one frozen policy contrast and one exposure axis, let each informative pair produce a sign Sᵢ ∈ {−1,+1}, where +1 means control exposure exceeds treatment exposure. Zero differences are ties and do not change the process. Under Fisher's sharp no-effect null and fair-coin orientation, each nonzero sign is conditionally symmetric given the paired outcomes and prior history.

Use the fixed betting fraction λ = 1/2:

**Eₜ = ∏ᵢ(1 + 0.5 Sᵢ) = (3/2)ᵂᵗ × (1/2)ᴸᵗ,**

where Wₜ and Lₜ count beneficial and harmful signs. A tie contributes one. Each factor is nonnegative and has conditional expectation one under the sharp null, so Eₜ is a nonnegative martingale. Ville's inequality therefore controls the probability that it ever crosses a fixed boundary, regardless of the stopping time.

For a predeclared family of M contrasts/axes and familywise level α, use the Bonferroni boundary M/α for each process. Then:

**P(there exists t and j such that Eⱼ,ₜ ≥ M/α) ≤ α.**

The engine computes Eₜ as an exact integer numerator/denominator, not floating point. It reports current crossing and latches the first crossing look, since a martingale can later fall below its boundary. The integer representation is bounded to at most 64 pairs.

### Required assumptions and excluded designs

- Every pair's treatment orientation is independently randomized with probability exactly 0.5, conditional on the pre-assignment history.
- The sharp null is no treatment effect for every randomized unit. This is not a test of a weak average-effect null.
- Pair construction is valid; treatment does not interfere with other randomized units or create unmodelled carryover. If interference exists, randomize and analyze at the independent cluster level.
- Fixed-count/blocked assignment is not supported by this sign process. Use a design-specific assignment-space enumerator such as the blocked randomization work in #9670 instead.
- Assignment receipts and randomization generation must be verified upstream. The module only checks the supplied design labels/digests.
- All outcomes, including timeout, invalid, missing output and abstention, remain in the exposure record.
- The append-only runner must persist each sequential state transition. The in-memory state alone does not resist rollback.

The implementation deliberately ignores effect magnitude and uses a single fixed betting fraction. It is a transparent baseline, not a power-optimal test. A later comparison may test predeclared mixtures of betting fractions or design-induced test sequences, but must preserve exact small-case references and the same randomization assumptions.

### Qualification tests and metrics

The Rust module adds thirteen authored tests: rational wealth under beneficial/harmful/tied signs; familywise threshold; latched first crossing; exact integer arithmetic at the 64-pair cap; non-fair assignment and design mismatch rejection; sequential-gap/replay/context failure controls; separate failure exposure; and exact dynamic-programming enumeration of crossing probability through a 20-look finite null tree.

An independent exhaustive sign-sequence calculation gives 4,186 crossings among 1,048,576 sequences by look 20 at familywise α = 0.05 and M = 3. This is 0.399% for one axis, below its 1.667% allocated bound. The authored Rust dynamic-programming test pins this count and verifies the bound at every prefix. This is a finite mathematical regression for the specified design, not evidence about real Symthaea effects.

Compare the sequential process with:
1. the fixed-sample exact sign-flip p-value after the preregistered final look;
2. a naive repeatedly inspected fixed-sample p-value as an invalid negative control;
3. a predeclared betting mixture or a general bounded-value e-process where its null assumptions match;
4. design-specific blocked inference when assignment is blocked.

Report false-crossing probability, time to crossing, probability of no crossing by the declared maximum, paired effect magnitude, and all three exposure axes. Keep statistical evidence separate from the realized effect-size estimate and resource costs.

### Literature and repository composition

Koning & van Meer (2026), *Anytime validity is free: inducing sequential tests*, JRSSB 88(4), DOI [10.1093/jrsssb/qkag050](https://doi.org/10.1093/jrsssb/qkag050), show how to induce an anytime-valid sequence from a valid terminal test using conditional expectations. The implementation here uses a different forward betting construction specialized to independently randomized paired signs; it does not claim the paper's terminal-power-matching result.

Compose with #8679 for general bounded value/risk e-processes, #9820 for finite fixed-horizon test induction, #9855 for the finite-filtration stopping/continuation oracle, and #9670 for blocked randomization designs. Do not merge their distinct assumptions into one generic "anytime-valid" label.
