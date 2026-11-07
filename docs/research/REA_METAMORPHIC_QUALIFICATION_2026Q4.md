# REA typed-claim metamorphic qualification (2026 Q4)

**Parent implementation:** [#10740 — independent typed-claim replay](https://github.com/Luminous-Dynamics/symthaea/pull/10740)  
**Research contract:** [#10738 — metamorphic laws](https://github.com/Luminous-Dynamics/symthaea/issues/10738)  
**Producer:** [#10665 — typed claim evaluator v0](https://github.com/Luminous-Dynamics/symthaea/pull/10665)

## Purpose

The finite expected-disposition corpus checks known examples. Metamorphic tests check whether evaluator/replay behavior changes correctly under transformations whose semantic effect is known. They are a second axis of qualification, not a replacement for the independently authored oracle.

The key distinction is between:
- **representation-preserving transformations**, which must preserve semantic identity and replay;
- **transport-only transformations**, which must not change the bound claim;
- **claim/premise mutations**, which must invalidate an old derivation;
- **information-loss transformations**, which may only preserve or weaken a conclusion, never upgrade it.

## Laws already present in the replay conformance suite

The current replay integration tests cover recursive JSON object-key permutation, JCS numeric equivalence, raw Evidence whitespace, evidence transport-order permutations with fixed claim order, ordered-premise binding, unreferenced-record rejection, duplicate-record rejection, missing-premise downgrade, unsupported-predicate closure under evidence-count sweeps, contradiction cardinality sweeps, request/result mutation, confidence/authority non-promotion and bounded CLI behavior.

The additional tests in this PR extend two less-covered boundaries.

## New law 1 — scope-information monotonicity

**Property:** Removing a scope qualifier from a claim cannot make a more-specific observation support a broader claim.

Test construction:
1. Admit an Evidence record with an exact Android platform and device identifier.
2. Evaluate a claim scoped to both values; it may be supported within that exact scope.
3. Remove the platform, the environment identifier, and then both from the requested claim while keeping the Evidence observation unchanged.
4. Each broadened claim must become underdetermined, carry an explicit scope-unverifiable premise assessment, and reject replay of the original positive result.

This is deliberately conservative. An observation about one known platform/device is not automatically an observation about all platforms/devices. V0 does not infer temporal validity or cross-platform generalization.

## New law 2 — confidence/authority non-amplification

**Property:** Confidence and authority labels cannot turn an observation into a different semantic predicate.

For each of the twelve unimplemented stronger predicates, the test crosses three confidence descriptors (observed, derived, inferred) with five authority descriptors (shipped artifact, controlled replay, historical reference, external service, analyst inference). Every combination must remain `unverified`, every premise must be `not_evaluated`, and the selected rule must remain `no-registered-predicate-rule/v0`.

This intentionally tests a 180-cell finite matrix. It does not claim that confidence or authority are meaningless; it proves only that those metadata labels do not silently satisfy proof obligations that v0 does not implement.

## New law 3 — future temporal fields are not silently ignored

**Property:** A caller cannot append an unrecognized `as_of` instant or `freshness_policy` and receive the old v0 result.

The request parser must reject the unknown fields. Temporal validity remains under #10600 / #10760 until a separately versioned temporal contract and verifier exist. This is an explicit compatibility boundary, not an improvised interpretation of timestamps.

## Qualification scorecard

Report per relation and fault family:

| Relation | Target | Failure meaning |
|---|---:|---|
| JSON key permutation | 100% replay equivalence | Canonicalization or raw parser inconsistency |
| Evidence transport permutation | 100% replay equivalence for fixed request order | Replay depends on incidental transport order |
| Claim premise permutation | 100% mismatch against old result | Claim request digest/order binding failure |
| Unreferenced / duplicate Evidence | 100% rejection | Premise-set substitution or double-counting |
| Scope qualifier removal | 100% underdetermined or stricter | Scope broadening by information loss |
| Assumption addition | 100% unverified | Unchecked assumptions silently accepted |
| Confidence/authority sweep | 100% no stronger predicate disposition | Metadata promoted into semantic authority |
| Contradictory evidence cardinality | 100% conflicted | Majority-vote or count-based truth laundering |
| Unsupported predicate evidence sweep | 100% unverified | Evidence count bypasses missing verifier |
| Temporal-field injection in v0 | 100% invalid request | Unversioned temporal semantics accepted |

The finite metamorphic target is zero violations across all listed transformations. Report the denominator for every law; do not aggregate into one score.

## Paradigm shift — qualify the measurement instrument, not only the model

Recent evaluation science makes a second change necessary. NIST AI 800-3 explicitly separates fixed-benchmark accuracy from generalized accuracy over a population of similar items and demonstrates generalized linear mixed models for estimating item difficulty, variance components and uncertainty across 22 API-access models and three benchmarks. A 2026 Nature paper on general scales uses demand profiles to explain and predict performance on unseen tasks/benchmarks, illustrating that task labels alone are weak descriptions of item difficulty.

REA should therefore maintain two different scorecards:

1. **Finite conformance:** deterministic properties and independently authored expected dispositions. Target exact agreement and zero false accepts on the frozen suite.
2. **Generalized agent capability:** sampled task instances annotated with demand profiles, with uncertainty and task/item variance modeled explicitly. Do not extrapolate a result on this fixed set to all artifact investigations.

For REA, candidate task-demand dimensions include:
- number of inferential steps between raw observation and requested proposition;
- scope dimensionality and cross-scope mismatch;
- evidence ambiguity, incompleteness and source quality;
- contradiction count and dependency/common-mode structure;
- serialization/canonicalization and parser adversariality;
- temporal drift, supersession and revocation burden;
- binary complexity, obfuscation, and auxiliary-tool requirements;
- resource pressure, evidence volume and concurrency;
- authorization and policy constraints;
- novelty/contamination risk and reference-oracle accessibility.

These dimensions should be measured/validated, not simply assigned as intuition-based difficulty scores. Separate fixed artifact family, task instance, run and model effects where data volume and assumptions support a hierarchical model. Keep item-level hidden ground truth and source adjudication independent from model output.

## External comparison — do not conflate the tracks

### AgentRE-Bench v3.1

The November 5, 2026 report covers six models, 20 Windows PE binaries and five attempts per binary (600 episodes). Muse Wonder reports 87.37/100 and GPT-6 Astra 87.04/100; the simultaneous 95% interval for their difference includes zero. The report separately reports valid-answer coverage and conditional answer quality, and states that independent human adjudication is outstanding. This is a useful task-performance baseline, not a semantic claim-soundness benchmark.

Reference: https://www.agentre-bench.ai/results/v3-1/

### SRE-Bench

The October 2026 release describes 19 private programs, 262 binary instances and 1,572 deterministic tasks across five RE domains, four source languages and 44 anti-analysis mechanisms. Its public-facing GPT-6 Astra row covers 181 of 262 eligible instances because refusals/safeguards reduce graded coverage; the 149 fully solved instances are 82.3% of the *graded* subset, not 82.3% of the full eligible set. Unconstrained pass@1 and pass@4 are separate settings. REA should retain the eligible, graded, refusal and completion denominators.

Reference: https://daplab.cs.columbia.edu/benchmarks/sre-bench/

### NIST AI 800-3 and TEVV-Athlon

NIST AI 800-3 distinguishes benchmark accuracy from generalized accuracy and recommends statistical methods whose assumptions and uncertainty are explicit. The TEVV-Athlon framework supplies a context-specific assessment method organized around Events and Tools generating data for Blocks tied to measurement concepts. REA's benchmark manifest should map every metric to a defined measurement construct, event, tool, data artifact and analysis procedure.

References:
- https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models
- https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems

### SCITT RFC 9943

SCITT offers signed-statement transparency and registration receipts, not statement truth. RFC 9943 §9.2 says an issuer can make false statements and registration only shows the statement was produced by that issuer. That is the exact non-implication REA must preserve when mapping transparency receipts to claims.

Reference: https://www.rfc-editor.org/rfc/rfc9943.html

## Next research/engineering steps

1. Keep v0 authority fixed; do not add semantic predicates merely to improve benchmark scores.
2. Extend metamorphic tests only when a transformation has a precise expected relation and owner.
3. Compose #10600 temporal validity, #10410 provider lineage, #10630 repository continuity and #2270 cyber claim semantics through their source-owned APIs.
4. Use hidden held-out tasks and independent human review for generalized agent comparisons.
5. Publish fixed-suite accuracy, generalized accuracy, coverage, abstention, false-entailment, replay mismatch and grader agreement separately.

## Claim ceiling

These tests qualify representation invariance, exact premise binding and no-upgrade behavior for the supported v0 observation rule. They do not prove that the evidence universe is complete, that a provider is honest, that sources are independent, that an artifact is safe, or that a result generalizes to unseen binaries or platforms.
