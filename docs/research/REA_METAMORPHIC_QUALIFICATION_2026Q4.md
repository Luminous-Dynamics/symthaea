# REA typed-claim metamorphic qualification (2026 Q4)

**Parent implementation:** [#10958 — precision-aware independent claim replay](https://github.com/Luminous-Dynamics/symthaea/pull/10958)  
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

The original metamorphic suite extends scope-information monotonicity, confidence/authority non-amplification, and temporal schema closure. The precision-aware refresh carries those laws onto the replay result that includes JCS integer-precision warnings, and adds population-completeness and lexical-warning invariants.

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

For each of the twelve unimplemented stronger predicates, the test crosses three confidence descriptors (observed, derived, inferred) with five authority descriptors (shipped artifact, controlled replay, historical reference, external service, analyst inference). In the precision-aware refresh, each Evidence record also carries an unsafe-integer warning. Every combination must remain `unverified`, every premise must be `not_evaluated`, the warning must be retained, and the selected rule must remain `no-registered-predicate-rule/v0`.

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

The November 5, 2026 v3.1 report covers six models, 20 Windows PE binaries and five attempts per binary (600 episodes). Muse Wonder reports 87.37/100 and GPT-6 Astra 87.04/100; the simultaneous 95% interval for their difference is -3.8203 to +4.4689 and includes zero. Both submit 100/100 valid answers. The report separately reports full-suite Overall, valid-answer coverage and conditional answer quality, and states that independent human adjudication is outstanding. These are fixed-suite results, not generalized capability estimates. This is a useful task-performance baseline, not a semantic claim-soundness benchmark.

Reference: https://www.agentre-bench.ai/results/v3-1/

### SRE-Bench

The October 2026 release describes 19 private programs, 262 binary instances and 1,572 deterministic tasks across five RE domains, four source languages and 44 anti-analysis mechanisms. Its public-facing GPT-6 Astra row is graded on 181/262 eligible instances; it fully solves 149/181 graded instances (82.3%), which is 56.9% of the full eligible cohort. Unconstrained pass@1 (88.0%) and pass@4 (99.2%) are separate settings and must not be conflated with public-facing coverage. REA should retain the eligible, graded, refusal and completion denominators.

Reference: https://daplab.cs.columbia.edu/benchmarks/sre-bench/

### NIST AI 800-3 and TEVV-Athlon

NIST AI 800-3 distinguishes benchmark accuracy from generalized accuracy and recommends statistical methods whose assumptions and uncertainty are explicit. The TEVV-Athlon framework supplies a context-specific assessment method organized around Events and Tools generating data for Blocks tied to measurement concepts. REA's benchmark manifest should map every metric to a defined measurement construct, event, tool, data artifact and analysis procedure.

References:
- https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models
- https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems

### SCITT RFC 9943

SCITT offers signed-statement transparency and registration receipts, not statement truth. RFC 9943 §9.2 says an issuer can make false statements and registration only shows the statement was produced by that issuer. That is the exact non-implication REA must preserve when mapping transparency receipts to claims.

Reference: https://www.rfc-editor.org/rfc/rfc9943.html


### SLSA v1.2, TUF 1.0.37 and SCITT — complementary assurance layers

SLSA v1.2 is the current approved specification and separates Build and Source tracks. Build provenance describes the builder, process and inputs so consumers can compare actual provenance with expected provenance. Source provenance describes how a source revision was created and change-managed. Neither track should be promoted into a semantic statement that the resulting program is correct or safe. REA should ingest these as typed, scoped evidence and preserve the source-owned claim ceiling.

TUF's current specification repository identifies version 1.0.37 (dated October 8, 2026). Its trusted metadata roles and update procedure address specific update-system attacks, including rollback, freeze and mix-and-match. TUF metadata can support update freshness/integrity obligations when the exact trusted root, versions, expirations and role thresholds are verified. It does not establish that the target artifact is semantically benign or that a downstream provider claim is true.

SCITT RFC 9943 provides signed-statement registration, receipts and transparent audit. Section 9.2 explicitly warns that issuers may make false statements; Section 9.3 notes that issuers may selectively withhold statements. Complete enumeration of a transparency service's registered set is not the same as complete issuer participation.

REA's role is not to replace these specifications. It is to preserve their exact verified properties as premises and prevent invalid implication from a lower-level property to a stronger proposition.

References:
- [SLSA v1.2 specification and tracks](https://slsa.dev/spec/v1.2/tracks)
- [SLSA v1.2 provenance](https://slsa.dev/spec/v1.2/provenance)
- [TUF specification (1.0.37)](https://github.com/theupdateframework/specification/blob/master/tuf-spec.md)
- [SCITT RFC 9943](https://www.rfc-editor.org/rfc/rfc9943.html)

### Temporal retrieval research

TEMPO (EMNLP 2026) provides a useful adjacent benchmark design: 1,730 multi-domain queries across 13 domains, decomposed into 3,976 retrieval steps, with Temporal Coverage@k and Temporal Precision@k to test whether answers cover the required periods. REA should adapt the *measurement principle*, not the dataset: a time-scoped claim should be scored on whether it retrieves and reasons over every required interval, not simply whether it finds a plausible recent document.

Reference: [TEMPO: Realistic Multi-Domain Benchmark for Temporal Reasoning-Intensive Retrieval](https://aclanthology.org/2026.emnlp-main.291/).

## Next research/engineering steps

1. Keep v0 authority fixed; do not add semantic predicates merely to improve benchmark scores.
2. Extend metamorphic tests only when a transformation has a precise expected relation and owner.
3. Compose #10600 temporal validity, #10410 provider lineage, #10630 repository continuity and #2270 cyber claim semantics through their source-owned APIs.
4. Use hidden held-out tasks and independent human review for generalized agent comparisons.
5. Publish fixed-suite accuracy, generalized accuracy, coverage, abstention, false-entailment, replay mismatch and grader agreement separately.

## Claim ceiling

These tests qualify representation invariance, exact premise binding and no-upgrade behavior for the supported v0 observation rule. They do not prove that the evidence universe is complete, that a provider is honest, that sources are independent, that an artifact is safe, or that a result generalizes to unseen binaries or platforms.

## Population completeness and selective disclosure: no-upgrade law (2026-11-07)

A set of valid observations can still be an incomplete view of the population it purports to describe. The v0 claim evaluator must not infer whole-archive or issuer-history completeness from self-reported terminal pagination, page counts, “all issued statements submitted” flags, or a complete view of entries already registered in a transparency log.

Keep these claims distinct:

1. **Complete within declared query/snapshot:** every item in an independently identified, versioned query or snapshot was enumerated and reconciled.
2. **Complete within captured artifact:** every member/byte-range in the exact admitted artifact was enumerated under the qualified reader.
3. **Complete within registered transparency view:** every entry in the named tree/checkpoint/log scope was enumerated and verified.
4. **Issuer participation completeness:** an independent obligation or audit establishes that the issuer submitted every relevant statement it created or observed.
5. **Observed subset / unknown population:** the records establish only the returned subset; the real-world universe or issuer participation remains unqualified.

The no-upgrade property added to the replay conformance suite binds three distinct provider observations: a declared query reports terminal pagination and matching counts; an issuer reports that all statements were submitted but supplies no independent issuer-history audit; and a transparency view reports every registered entry enumerated. The claim predicate is `whole_archive_completeness`. All three Evidence records are independently identity-verified and replayed, but the disposition must remain `unverified` because evaluator v0 has no predicate-specific completeness rule. Provider count, `confidence=observed`, `authority=external-service`, and a terminal cursor do not change that ceiling.

Processing completeness is a separate obligation from enumeration and acquisition. A provider may enumerate and acquire all three items while successfully interpreting only two and recording the third as unsupported or failed. The evaluator must preserve that failed/unsupported item; it must not equate a terminal cursor, zero acquisition failures, process exit code zero, or a provider-supplied all-processed flag with successful downstream processing. A future completeness rule must account for every acquired item as success, failure or unsupported, and separately state whether the claim requires all items to be successfully interpreted or merely accounted for.

The conformance suite now includes a named incomplete-processing metamorphic case: the provider reports three enumerated/acquired items, two processed results, one unsupported-format failure, and a self-reported all-processed flag. The exact provider observation may be supported as a report. The stronger whole-population completeness claim remains unverified under v0, and replay must reproduce that ceiling.

This is not a rejection of the narrower observations. A provider may establish what it reported about a specific query or log view. The forbidden inference is from “complete set of registered/returned records” to “complete set of records the issuer created” or “complete real-world population”. Completeness requires separate proof obligations for population definition, enumeration coverage, acquisition coverage, processing coverage, issuer participation and temporal coverage.

RFC 9943 (SCITT) is directly relevant: a Transparency Service records signed statements that issuers submit, but an issuer may selectively submit or withhold statements. Therefore complete enumeration of a transparency log can establish a complete view of registered statements at a named checkpoint; it cannot by itself establish complete issuer participation or statement accuracy. See [RFC 9943](https://www.rfc-editor.org/rfc/rfc9943.html).

The metamorphic target is **zero completeness uplift** from provider self-assertion, duplicate/syndicated providers, pagination claims, registered-log completeness or record count. Any future completeness predicate must bind an authoritative population definition, stable snapshot/query, page/cursor/sequence coverage, successful acquisition and processing accounting, independent issuer participation where the claim requires it, and a target-time interval. Until that proof contract exists, v0 remains `unverified`.

## Precision-aware metamorphic laws (2026-11-07)

The parent replay result now contains \`numeric_precision_warnings\`. Those warnings are diagnostics about the parsed integer representation that JCS will map through ECMAScript binary64; they are not new proof premises or a stronger semantic predicate.

### Law 4 — precision-warning representation invariance

Recursively reorder JSON object members in the raw claim, raw Evidence and serialized result while keeping arrays, values and exact Evidence IDs unchanged. The replay must preserve:
- the same precision-warning source, Evidence ID, JSON Pointer and exact integer value;
- the same claim-request digest and derivation digest;
- the same disposition and ReplayMatch outcome.

This tests the composition of JCS canonicalization, precision-warning sorting and the independent replay implementation. It does not claim that the original integer lexeme is retained after parsing; exact raw-token provenance remains the separate #10952 contract.

### Law 5 — precision diagnostics never grant claim authority

Cross the same twelve unimplemented predicates with the 3 × 5 confidence/authority descriptor matrix, while every Evidence record contains an integer outside the ECMAScript safe-integer range. The warnings must appear in the result and derivation, but all 180 combinations must remain \`unverified\` with \`not_evaluated\` premises and \`no-registered-predicate-rule/v0\`.

A warning that an observation may have lost integer precision is not an exact numeric proof, a producer signature, a completeness witness, or a reason to promote an unsupported predicate.

### Extended adversarial cases — population and process completeness

The suite also distinguishes a complete list of registered/returned items from a complete issuer-created or real-world population, and distinguishes enumeration/acquisition from downstream processing. It covers:
- issuer and transparency-service self-report that all statements were submitted;
- terminal pagination with a missing page number;
- all items enumerated and acquired, but one item unsupported/failed during processing.

The narrow \`provider_reported_observation\` predicate may support exactly what the provider reported. The stronger \`whole_archive_completeness\` predicate remains \`unverified\` until its own proof contract establishes population definition, enumeration, acquisition, processing and any required issuer participation.

## Current stack and status

- Precision-aware producer: [#10927](https://github.com/Luminous-Dynamics/symthaea/pull/10927).
- Precision-aware independent replay: [#10958](https://github.com/Luminous-Dynamics/symthaea/pull/10958).
- Original metamorphic contract: [#10738](https://github.com/Luminous-Dynamics/symthaea/issues/10738); the older base-specific test PR [#10847](https://github.com/Luminous-Dynamics/symthaea/pull/10847) is superseded by the current-stack refresh.
- Raw lexeme source audit: [#10952](https://github.com/Luminous-Dynamics/symthaea/pull/10952), with fuzz qualification [#10986](https://github.com/Luminous-Dynamics/symthaea/pull/10986).
- Numeric-intent profile remains design-only under [#11011](https://github.com/Luminous-Dynamics/symthaea/pull/11011); do not infer a field's domain type from a precision warning.

The current exact-head parent (#10958) passed 57 replay conformance tests, 29 Evidence-core tests, 15 claim tests, two pinned upstream differential tests, formatting and warnings-denied Clippy. The seven tests added by this refresh are authored but are not yet represented as passing until the new exact head completes its focused workflow.

