# REA claim verification: industry comparison and qualification targets (2026 Q4)

**Status:** research and benchmark protocol, not a claim of achieved SOTA.

## Executive finding

REA claim evaluation occupies a gap between several mature but orthogonal assurance layers:

- SLSA provenance describes the build subject, build definition, resolved dependencies, builder and execution details.
- Sigstore Rekor and Certificate Transparency provide signed, auditable log inclusion and append-only/consistency evidence.
- IETF SCITT provides a generic architecture for signed supply-chain statements and transparency receipts.
- TUF protects update metadata against rollback, freeze and mix-and-match attacks.
- REA Evidence binds a normalized provider observation to an artifact and operation.
- The REA typed-claim layer asks whether the exact requested proposition is supported, refuted, conflicted, inapplicable, underdetermined or unverified under the exact Evidence references and scope.
- The independent replay verifier checks whether a separate implementation reproduces the producer's serialized result.

These layers should compose without substituting for one another. A signed statement may be false. A valid log inclusion proves registration, not statement accuracy. A valid build provenance record does not establish runtime benignness. A REA Evidence semantic ID commits to a structured observation but does not authenticate its producer. A claim result must never upgrade any of these facts by inference alone.

## 1. Industry capability comparison

| System / standard | Strongly qualifies | Does not establish by itself | REA relationship |
|---|---|---|---|
| SLSA v1.2 Build + Source tracks | Build subject, build definition, resolved dependencies, builder identity, run details, and source revision/change-management provenance | Source-code correctness, runtime safety, signer authorization outside the chosen trust policy, or truth of downstream semantic claims | Consume track-specific provenance as evidence; require expected signer/builder policy and preserve the relevant claim ceiling |
| Sigstore Rekor | Signed-entry registration, inclusion proof, signed checkpoint, and auditable append-only log history | That the signed statement is true, complete, benign or authorized for a specific target | Compose log evidence with claim evaluation; a log entry is not a positive semantic claim |
| Certificate Transparency RFC 9162 | Merkle inclusion, consistency between tree heads, and monitoring/auditing of log behavior | Semantic correctness of logged objects or universal agreement without monitoring / gossip | Reuse the Merkle model and explicit checkpoint/witness obligations |
| SCITT RFC 9943 | Interoperable signed-statement transparency and registration receipts | Statement accuracy; SCITT explicitly permits false statements and subsequent supersession | Exact signed statement predicate and receipt become separate evidence; claim truth remains independently evaluated |
| TUF | Threshold metadata signatures, version/expiry checks and client-side rollback/freeze defense | Whether the bytes named by trusted metadata are semantically safe or correct | Use TUF for update trust/currentness, not as a general truth engine |
| in-toto v1 | Authorized supply-chain steps, signed link metadata, material/product rules, layout expiry and inspection steps | Whether opaque target files are benign or whether every project-specific semantic claim is entailed | Consume layout/link evidence as workflow and custody premises; independently evaluate artifact semantics |
| GUAC | Aggregation and normalization of SBOMs, attestations, vulnerabilities and dependency relationships into a queryable graph | Whether an observation is true, whether providers are independent, or whether a proposition is supported under exact scope | Reuse graph/identity integration; REA claim calculus should qualify entailment, provenance ceilings, lineage unknowns and independent replay |
| CycloneDX VEX | Contextual vulnerability exploitability status for a product/component | General artifact correctness, provenance authenticity, or truth outside the declared product/version context | Import VEX as a scoped issuer statement; retain issuer, product/version, status, justification, timestamp and trust policy as separate premises |
| OpenSSF Scorecard / OSPS Baseline | Automated project security-practice checks and maturity-oriented control coverage | Runtime safety, artifact semantics, provider independence, or proof that a specific delivered binary matches the project | Use as project/process posture evidence, never as a shortcut to an artifact-level claim |

| REA Evidence v1 | Versioned structured provider observation, schema validation, semantic identity and JCS commitment | Producer authentication, trusted time, source independence, archive completeness, or the truth of the normalized observation | Existing source contract; do not mutate identity semantics |
| REA claim evaluator v0 | Exact provider-reported normalized result under matching subject/operation/represented environment scope | General semantic entailment, authenticity, temporal validity, independence or security verdicts | Narrow semantic composition rule; all stronger predicates fail closed |
| REA independent replay verifier | Reproduction of the serialized v0 disposition and derivation from raw request and verified Evidence | Independent cryptographic primitives, producer authenticity, or underlying artifact truth | Independent semantic implementation; shared qualified Evidence/JCS primitives are disclosed |

### Standards and implementation references

- [SLSA specification v1.2](https://slsa.dev/spec/v1.2/) and [v1.2 provenance overview](https://slsa.dev/spec/v1.2/provenance)
- [Sigstore Rekor overview](https://docs.sigstore.dev/logging/overview/)
- [Sigstore security model](https://docs.sigstore.dev/about/security/)
- [IETF RFC 9943 — SCITT architecture](https://www.rfc-editor.org/rfc/rfc9943.html)
- [IETF RFC 9162 — Certificate Transparency v2](https://www.rfc-editor.org/rfc/rfc9162.html)
- [The Update Framework specification v1.0.36 (5 August 2026)](https://theupdateframework.github.io/specification/v1.0.36/)
- [in-toto v1 specification](https://github.com/in-toto/specification/blob/master/in-toto-spec.md)
- [GUAC — Graph for Understanding Artifact Composition](https://github.com/guacsec/guac)
- [CycloneDX VEX capability](https://cyclonedx.org/capabilities/vex/)
- [OpenSSF Scorecard](https://openssf.org/projects/scorecard/)
- [OpenSSF Project Security Baseline v2026.08.28](https://baseline.openssf.org/)
- [IETF RFC 8785 — JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html)

in-toto is the adjacent integrity layer for declared supply-chain steps: layouts define authorized functionaries and material/product expectations, and link metadata records performed steps. It intentionally treats target files as opaque; project-specific inspection is an extension point rather than a built-in proof of semantic truth. GUAC is the closest graph-aggregation analogue: it ingests supply-chain metadata and maps entity relationships, but graph completeness and edge truth remain separate qualification questions. CycloneDX VEX is a contextual vulnerability-exploitability statement for a product/component, not a generic security verdict. OpenSSF Scorecard and the OSPS Baseline measure project-level practices/maturity, not the behavior of one exact binary. REA should interoperate with these instead of recreating them, while preserving source, scope, authority and non-implications in each imported claim.

Important detail: SLSA v1.2 is the current approved specification and has separate Build and Source tracks. Build provenance traces an artifact to its source/build process; Source provenance records how revisions were created and change-managed. REA adapters should bind a track-specific predicate and preserve the consumer's expected signer/builder policy. Sigstore's own security model requires independent Rekor monitoring; a log's signed append-only property is not a semantic truth oracle. SCITT RFC 9943 is a June 2026 IETF Standards Track publication for single-issuer signed-statement transparency; its receipts prove registration and auditability, not the truth of the issuer's statement. TUF 1.0.36 (5 August 2026) addresses update metadata, rollback and freeze protections. These systems supply distinct evidence families and must not be collapsed into one trust score. The current SLSA working draft additionally proposes Build Environment and Dependency tracks, but these are not approved v1.2 requirements. REA should model those draft tracks as experimental adapters until they reach an approved specification stage.

## 2. Research benchmarks and what they teach us

### Binary analysis and claim-level exactness — AgentRE-Bench v3.1

The November 5, 2026 v3.1 report evaluates six models on 20 Windows PE artifacts, five attempts per artifact, for 600 episodes. Muse Wonder records 87.37/100 Overall and GPT-6 Astra 87.04/100, with both returning 100/100 valid answers. The simultaneous 95% interval for Muse minus Astra is -3.8203 to +4.4689 and includes zero, so the report does not establish which model would score higher on repeat, nor practical equivalence. DeepSeek V4.1 Flash records 78.31/100 with 98/100 valid answers and two budget exits scored zero. The report separates full-suite Overall, valid-answer coverage and conditional answer quality. Its exact-claim review reproduced all 600 score objects, but the report also says independent human adjudication has not occurred. It identifies 46 source-supported mappings penalized by fixed-rubric fact/field-placement rules and reports no full-credit episode. These are fixed-suite results for the v3.1 artifact set, not evidence of generalization to unfamiliar binaries.

**REA consequence:** score at least four things separately:

1. factual/behavioral correctness against hidden ground truth;
2. allowed-fact and claim-to-field/schema correctness;
3. completion, coverage, abstention, refusal and budget exits;
4. grader reproducibility and independent adjudication.

A schema-valid answer is not automatically correct. A correct fact placed in an unauthorized field is a distinct interface error from hallucination. Conditional quality over answered examples must not hide coverage failures.

Reference: [AgentRE-Bench v3.1 results and method](https://www.agentre-bench.ai/results/v3-1/).

### Realistic binary reverse engineering — SRE-Bench

SRE-Bench (updated October 2026) is the closest hard external benchmark for agentic reverse engineering at realistic scale: 19 private programs, 262 binary instances, 1,572 deterministic graded tasks, five domains, four source languages, and 44 in-house anti-analysis mechanisms. It was built with more than 5,000 expert hours, and runs instances in isolated environments without exposing source or reference solutions. [Official benchmark](https://daplab.cs.columbia.edu/benchmarks/sre-bench/).

Its published public-facing result for GPT-6 Astra is 5.55 passed tasks per binary out of six across 181 graded instances; it fully solved 149/181 graded instances (82.3%), with 7 zero-score instances. However, only 181 of the 262 eligible instances were graded in that run. The 82.3% number is conditional on graded instances; relative to the full 262-instance eligible cohort, 149 complete solves are 56.9%. The benchmark also reports a separate unconstrained Astra pass@1 of 88.0% and pass@4 of 99.2%; those must not be mixed with the public-facing score or used as a direct comparison to AgentRE-Bench. [Official result details](https://daplab.cs.columbia.edu/benchmarks/sre-bench/).

**REA consequence:** reuse SRE-Bench's deterministic hidden-ground-truth and realistic-domain design where its access rules permit, but do not try to beat its leaderboard with a different denominator. Add a separate REA-groundedness overlay: was each recovered fact actually entailed by source-bound observations, exact artifact/scope references and independently replayed derivations? Keep specialist RE capability and epistemic grounding as two independent axes.

### Source-groundedness and parametric leakage — RDQA

RDQA (EMNLP 2026) tests whether multimodal agents actually retrieve and inspect the requested source rather than produce correct answers from parametric memory. It has 1,350 questions across document, video and audio sources and 20 real-world scenarios. The paper introduces Parametric Leakage Rate (PLR), the fraction of correct answers produced without genuine evidence retrieval. It reports 24.0% adjusted accuracy for the best generic agent, 41.1% for its process-aware RDQA-Skill baseline, and PLR as high as 96.0% among evaluated systems. These are RDQA-specific measures, not REA measurements, but the mechanism generalizes directly to reverse engineering: a model may guess a likely binary behavior without inspecting the exact artifact. [ACL Anthology paper](https://aclanthology.org/2026.emnlp-main.1855/).

**REA consequence:** add an REA-specific parametric leakage rate, with a preregistered denominator and strict source-evidence requirements. For every correct raw answer, test whether an accepted Evidence chain binds the exact artifact, operation, occurrence/byte range and scope. Correct-but-ungrounded answers receive no evidence-grounded credit; report raw accuracy and PLR separately. Use paired high-entropy artifact mutations to break prior-based guessing, plus irrelevant-byte controls to test justified invariance. Do not claim comparability with RDQA PLR unless its exact published scoring method is adopted.

### Continuous engineering — SWE-Milestone and SWE-CI

SWE-Milestone reconstructs real release histories into milestone DAGs and evaluates agents in an unbroken sequence, preserving earlier mistakes and regression effects. Its paper reports that some isolated-task results above 80% fell to at most 38% in continuous settings across its evaluated models/frameworks. Its current public leaderboard is a different versioned evaluation and must not be conflated with the paper's study.

SWE-CI evaluates 100 repository-maintenance tasks whose histories span an average of 233 days and 71 commits. Both reinforce that a patch passing at one revision is not evidence that a repository claim remains valid after later changes.

ChainSWE (EMNLP 2026) adds sequential multi-bug maintenance: 304 chronological issues across 54 Python projects, where later fixes depend on the shared evolving codebase. Its evaluation reports performance drops of up to 70% as issue-chain length increases. This is adjacent evidence, not a REA score: ChainSWE measures code-maintenance success, while REA must additionally measure whether earlier evidence claims become stale, remain historically replayable, or are selectively invalidated as upstream code and dependencies change. [ChainSWE paper](https://aclanthology.org/2026.emnlp-main.371/).

DORA's current software-delivery model complements agent benchmarks with five operational metrics: change lead time, deployment frequency, failed-deployment recovery time, change-failure rate, and deployment rework rate. DORA groups the first three as throughput and the last two as instability. REA should report its operational qualification similarly: replay and selective-revalidation latency as throughput; false matches, missed drift, rework after invalidation, and recovery from failed qualification as instability. Keep these disaggregated rather than collapsing them into one scalar.

**REA consequence:** a claim derivation is valid for its exact source revision, dependency closure and rule version. Add a continuity track for selective revalidation after changes; reuse #10630 rather than creating another repository graph.

References:
- [SWE-Milestone paper (ICML 2026)](https://proceedings.mlr.press/v306/deng26d.html)
- [SWE-Milestone project and current leaderboard](https://swe-milestone.com/)
- [SWE-CI paper](https://arxiv.org/abs/2603.03823)
- [ChainSWE (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.371/)

### Lifecycle and temporal reasoning — Vera-Life-Eval, TRACE and TKE-Bench

Vera-Life-Eval contains 8,500 questions and 41,203 versioned evidence pieces across current, past, stale-only, update, conflict and multi-hop conditions. Its lifecycle-faithfulness definition requires semantic support, target-time validity, non-supersession and provenance.

TRACE provides more than 16,000 model-specific chronological tests across Modify and Insert settings, with seven evaluation dimensions and explicit probing of history/future preservation.

TKE-Bench studies temporal knowledge accessibility and the ChronoExpert paper reports catastrophic temporal interference in the evaluated editing baselines.

**REA consequence:** query-time target time and system knowledge time must be distinct. A receipt timestamp is not source valid time; a current retrieval does not make its source current. Reuse #10600 for lifecycle/as-of semantics and keep v0 platform-version/currentness claims underdetermined when the source contract cannot bind them.

References:
- [Vera-Life-Eval / Vera-Life (ACL Anthology, 2026)](https://aclanthology.org/2026.groundlm-1.18/)
- [TRACE (EMNLP Findings 2026)](https://aclanthology.org/2026.findings-emnlp.1045/)
- [TKE-Bench / ChronoExpert (EMNLP Findings 2026)](https://aclanthology.org/2026.findings-emnlp.1823/)

### Numeric semantics and precision intent — RFC 7493 and RFC 8785

RFC 8785 defines a canonical representation by constraining JSON numbers to IEEE-754 binary64 and using ECMAScript-compatible serialization. It explicitly shows rounding during canonical serialization and recommends strings for values requiring greater integer or decimal precision. RFC 7493 separately says interoperable receivers cannot be expected to treat integers outside ±(2^53−1) as exact, and recommends JSON strings when exact interchange is required. [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html) · [RFC 7493](https://www.rfc-editor.org/rfc/rfc7493.html).

This produces four different objects that REA must not conflate:

1. **Raw source lexeme:** exact byte spelling and JSON Pointer, bound to the raw input SHA-256.
2. **JCS semantic number:** the binary64 interpretation used for deterministic canonicalization and semantic identity.
3. **Domain numeric intent:** whether a path denotes a binary64 measurement, bounded exact integer, exact integer string, fixed decimal string, or opaque number.
4. **Measurement semantics:** unit, scale, calibration, range and rounding policy.

The producer/replay pair (#10927/#10958) now discloses unsafe integer precision where the parsed representation retains the integer. The standalone raw-token audit (#10952) can preserve integer lexemes before that information is lost. Neither layer determines whether a number is a money amount, a byte count, a timestamp, a sensor measurement or an opaque field.

**REA consequence:** keep Evidence v1 and JCS identities unchanged; add a separate, versioned numeric-intent profile at schema-defined JSON Pointers. Proposed types are \`binary64\`, \`safe_integer\`, \`exact_integer_string\`, \`fixed_decimal_string\` and \`opaque_numeric\`. Bind the profile digest, path, raw-source digest, unit/scale and rounding policy into a typed validation result. Never infer exactness from the token's magnitude or lexical form.

**Benchmark gates:** 100% type-profile conformance on a frozen corpus; zero silent binary64 coercions for exact-integer/decimal-string paths; 100% RFC 8785 parity for binary64 paths; zero profile/unit/scale mismatches accepted; and explicit lexical-coverage states for raw-token-audited, parsed-value-only and unavailable. Keep numeric correctness, measurement validity and provider truth as distinct claims.

This is a research extension tracked by [#10989 — schema-bound numeric intent](https://github.com/Luminous-Dynamics/symthaea/issues/10989), not authority added to claim evaluator v0.

### Statistical validity — NIST AI 800-3 and TEVV-Athlon

NIST AI 800-3 distinguishes accuracy on a fixed benchmark from generalized accuracy over a broader population of similar tasks. It warns that common analyses can hide assumptions or produce invalid uncertainty estimates; it demonstrates generalized linear mixed models and variance decomposition on 22 API-access models and three benchmarks.

NIST's TEVV-Athlon Framework is a four-stage method for constructing a context-specific assessment from TEVV objectives; it structures that assessment around Events and Tools that produce data for Blocks linked to measurement concepts. The initial NIST AI 200-2 public draft comment period ran through October 6, 2026. NIST AI 800-3 (February 2026) separately argues that benchmark generalization and uncertainty depend on explicit statistical assumptions, and demonstrates hierarchical modelling of model capability and item difficulty.

NIST AI 800-3 (February 2026) formalizes the distinction between fixed-benchmark accuracy and generalized accuracy over a population of similar items. Its study of 22 API-access models across three benchmarks demonstrates generalized linear mixed models, item-difficulty estimation and variance decomposition as ways to make assumptions and uncertainty explicit. [NIST publication](https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models).

**REA consequence:** report fixed-corpus conformance and generalized performance as separate estimands. For model/agent comparisons, use paired tasks, item/domain random effects where assumptions are defensible, variance decomposition, and confidence intervals. Keep every unsupported, abstained, missing, invalid and budget-exhausted attempt in the denominator.

References:
- [NIST AI 800-3 (February 2026)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-3.pdf)
- [NIST TEVV-Athlon framework](https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems)
- [NIST ARIA evaluation planning manual](https://www.nist.gov/publications/aria-evaluation-planning-manual-elements-aria-style-ai-evaluations)

## 3. Paradigm shifts worth pursuing

### A. From evidence aggregation to typed non-implication

Evidence does not vote. Every claim predicate has its own proof obligations. A large number of weaker, correlated observations cannot become a stronger proposition by majority vote or confidence arithmetic.

**Research test:** construct paired non-implication controls where a valid premise is necessary but insufficient: valid checksum vs authenticity, signature validity vs signer trust, log inclusion vs statement accuracy, provider observation vs artifact truth, one member receipt vs package completeness.

### B. From self-attested output to independent replay

The evaluator must not be its own oracle. Separate producer, replay verifier and expected-disposition source. The replay checker should reject a result with a valid-looking digest but incorrect disposition, missing field, substituted premise, changed assertion or changed scope.

**Research test:** freeze independent expected labels before running production evaluation; require 100% replay agreement on the finite supported subset and zero false accepts for all seeded mutations.

### C. From snapshot claims to lifecycle claims

A claim should say what was true within which target-time interval and what the system knew at which knowledge-time cutoff. Corrections and supersessions change present applicability without erasing historical receipts.

**Research test:** current/past/stale-only/update/conflict/multi-hop suites; separately score world-valid-at-T and known-to-system-by-K.

### D. From model scores to measurement-qualified evidence

Report benchmark accuracy, generalization assumptions, sampling frame, item difficulty, task/domain variance, evaluator agreement, exposure and sequential selection. No composite scalar should hide governance failures.

**Research test:** paired, preregistered baselines; independent expected dispositions; fixed-suite and held-out tracks; blind adversarial controls.

### E. From generic verification to a claim ceiling

A verifier must publish exactly what it proves and what it does not. No type, receipt, score, confidence, framework mapping or schema validity can increase authority by itself.

**Research test:** mechanically mutate source category, claim kind, signer, builder, log, scope, time, and dependency lineage. The stronger claim must remain unsupported unless its own obligation set is satisfied.

## 4. Proposed REA benchmark matrix and target gates

Targets below are **proposed release gates, not achieved results**.

| Track | Primary metrics | Qualification target |
|---|---|---|
| Request/result schema | required-field accuracy, unknown-field rejection, duplicate-key rejection, schema mutation detection | 100% exact disposition for finite schema corpus; 0 duplicate/unknown/omitted-field acceptance |
| Claim entailment | disposition accuracy per predicate/fault family, false entailment, false refutation | 100% finite oracle agreement; zero false supported/refuted decisions on explicit non-implication controls |
| Scope binding | subject, build/member, operation, platform, environment and version leakage | 0 wrong-scope positive results |
| Contradiction | conflict preservation, source-family grouping, contradiction omission | 100% preservation of seeded same-scope contradiction; no majority-vote resolution |
| Unknown/abstention | underdetermined/inapplicable/unverified accuracy, selective risk vs coverage | 100% expected disposition on finite unknown/inapplicable suite; explicit missing/unrepresented obligations |
| Replay | exact result reproduction, request/digest mutation detection, deterministic replay | 100% supported-subset replay; 0 acceptance of a forged result with incorrect semantics |
| Temporal | target-time accuracy, stale/superseded detection, valid-time vs knowledge-time confusion | 100% finite as-of lifecycle suite once #10600-owned evidence is integrated |
| Continuity | stale-claim invalidation, selective revalidation precision/recall, regression preservation | 100% exact finite change-impact oracle; minimize unnecessary re-runs without retaining stale claims |
| Provider lineage | false-independence uplift, common-mode detection, unknown-lineage treatment | 0 independence credit from known duplicates or unknown ancestry |
| Agent task use | valid claim coverage, source locator precision, unsupported-claim rate, abstention quality, field/schema placement | report per dimension; no aggregate may compensate for a false entailment or authority bypass |
| Operations | p50/p95/p99 latency, peak RSS, bytes processed, allocations, energy per replay, result size | publish by input size/fault family with exact hardware, build, concurrency and run count |

## 5. Baseline protocol

Keep baselines on identical evidence, scope, answer schema, time cutoff, context budget and tool budget:

1. Raw provider output with ordinary JSON/schema validation only.
2. Evidence ID verification without typed claims.
3. Typed claim evaluator v0.
4. Typed claim evaluator plus independent replay.
5. Evaluator plus source-lineage, lifecycle and repository-continuity inputs when their owner contracts are qualified.
6. A strong model/agent synthesis baseline using the same Evidence records but without claim gating.
7. Model/agent synthesis with typed claim gating and explicit abstention.

Ablate one dimension at a time: exact artifact scope, operation, platform, temporal scope, source lineage, contradictions, assumptions, replay, and result-schema checks.

Do not compare a synthetic deterministic control result to a model benchmark leaderboard. Do not import a benchmark as a baseline until source revision, protocol, license, task rights and scoring method are pinned.

## 6. Immediate engineering sequence

1. Qualify producer evaluator v0 at exact head (#10665).
2. Qualify the independent replay crate and conformance cases (#10679 / #10673).
3. Freeze a public seed corpus and independently authored expected dispositions under #10629; keep a blinded held-out set.
4. Add result schema conformance tests, RFC 8785 vectors and request/result mutation matrix.
5. Compose #2270 cyber claim semantics, #10600 lifecycle validity and #10630 engineering continuity only through their own owner APIs and evidence contracts.
6. Add benchmark-run receipt and per-fault-family scorecard using the existing canonical qualification receipt infrastructure; do not invent another receipt envelope.
7. Only after the finite oracle suite passes, run matched model/agent evaluations and report uncertainty, coverage and task transfer.

## 7. Claim ceiling

This work establishes only a scoped, reproducible evaluation of a provider-reported observation from semantically verified REA Evidence. It does not establish general truth, producer identity, trustworthiness, artifact safety, temporal validity, source independence, completeness, platform acceptance, or security-control effectiveness. Those remain separate predicates with separate source-owned proof obligations.

## 8. Independent replay CLI

The standalone package now includes `rea-claim-replay`, a read-only command for CI and reviewer workflows:

```sh
cargo run --locked --manifest-path tools/rea-claim-replay-verifier/Cargo.toml -- \
  claim.json result.json evidence-1.json evidence-2.json
```

Evidence paths are optional when the claim intentionally contains no premises. The CLI emits a JSON replay receipt on stdout and uses these exit codes: 0 = ReplayMatch; 1 = well-formed result but semantic ReplayMismatch; 2 = invalid input, bounded-read/resource failure, IO or canonicalization error.

The command reads at most 1 MiB for the claim, 4 MiB for the result, 64 MiB per Evidence record and 128 MiB total Evidence bytes. It uses bounded `Read::take(limit + 1)` reads rather than unbounded `fs::read`, then invokes the same independently implemented replay path used by the conformance tests. It does not write, extract, execute or mutate supplied artifacts.

A matching replay receipt remains a reproducibility result, not a signature, trusted timestamp, provider-authentication result or proof of underlying artifact truth.

### Important qualifications when composing established standards

**SLSA is a build-platform claim, not a complete semantic inventory.** In the approved SLSA Provenance v1.2 Build Track, `builder.id` determines the trusted build-platform boundary and the accuracy/completeness guarantee that consumers may rely on. `resolvedDependencies` is an unordered set and its completeness is best-effort through Build L3. Therefore, absence of a dependency from a provenance record is not a proof that the build did not consume it. The REA claim model should retain the SLSA build level and signer-builder trust policy as explicit scope/assumptions and must not infer a complete dependency closure from a well-formed provenance object. [SLSA v1.2 Build provenance](https://slsa.dev/spec/v1.2/build-provenance).

**Sigstore is more than the Rekor database.** A full Sigstore verification flow checks the artifact signature and digest, verifies that the certificate identity matches the expected identity and validates the certificate against the Sigstore trust root, then checks transparency-log inclusion. Rekor provides auditable registration and inclusion evidence; the security model also calls out monitoring and the trust placed in the log/checkpoint. This can establish an identity-bound signing event for the artifact bytes under the chosen policy. It still does not prove those bytes are benign, that the signer is authorized for every product namespace, or that a semantic REA claim about the artifact is correct. [Sigstore overview](https://docs.sigstore.dev/); [Rekor security model](https://docs.sigstore.dev/about/security/).

**SCITT makes the statement-truth boundary explicit.** RFC 9943 says an issuer may make false statements and that registration proves the statement was produced by an issuer; relying parties still determine accuracy. This is not a weakness unique to SCITT: transparent registration is intentionally orthogonal to semantic truth. REA should link a SCITT receipt to the signed statement digest, verify issuer identity/policy separately, and then evaluate the statement's predicate against artifact- and time-scoped Evidence. [RFC 9943 §9.2](https://www.rfc-editor.org/rfc/rfc9943.html#section-9.2).

**TUF is a metadata-update security framework.** Its threat model includes rollback, freeze, fast-forward, mix-and-match, threshold-key compromise and wrong-software installation. A TUF-qualified target record is strong evidence about what a trusted repository authorized as a target under a metadata version/expiry policy. It is not a substitute for inspecting the target's internal semantics or proving runtime behavior. [TUF specification](https://github.com/theupdateframework/specification/blob/master/tuf-spec.md).

## 8. Implementation update — 2026-11-06

The producer implementation is [#10665](https://github.com/Luminous-Dynamics/symthaea/pull/10665). Its focused REA workflow passed on exact head `e2160007bcd8eb03c283465c08ba07e34372b75f` after the raw Evidence move/clone reduction, named provider metadata record, bundle-adapter Clippy cleanup and test-slice borrowing fixes. This is focused compatibility/claim-evaluator qualification, not full-workspace qualification or general semantic truth.

The independent replay implementation is the current refreshed stack [#10740](https://github.com/Luminous-Dynamics/symthaea/pull/10740), tracked by [#10673](https://github.com/Luminous-Dynamics/symthaea/issues/10673). It includes:
- a frozen public seed oracle with 23 explicit expected-disposition/replay-outcome cases, authored independently of the production evaluator;
- a standalone, bounded, read-only `rea-claim-replay` CLI with exit 0 for match, 1 for semantic mismatch, and 2 for invalid input / IO / resource failure;
- separate request/result parsing, recursive duplicate-key rejection, required-nullable-field checks and closed result schemas;
- verification of raw REA Evidence semantic IDs and rejection of duplicate, malformed, missing-inconsistent, unrequested and unreferenced Evidence identifiers;
- complete JCS-canonical result comparison, recomputed request/derivation hashes and rejection of self-digested forged dispositions;
- explicit fail-closed treatment of unsupported predicates, assumptions and scope dimensions absent from REA Evidence v1;
- 1 MiB claim, 4 MiB result, 128 MiB aggregate Evidence, 64 MiB per Evidence and 4,096-record limits;
- a claim-authorized Evidence projection after full verification, so raw_result and unrelated metadata do not remain in the replay inference map;
- disclosed reuse of the shared REA Evidence identity verifier and RFC 8785 canonicalization primitive. Independence is at the claim scope/disposition/replay logic level, not a second cryptographic implementation or canonicalizer.

The replay exact-head workflow has already passed the parent formatter, upstream Evidence differential suite and warnings-denied parent Clippy on the refreshed stack. The independent replay formatter has been iterated on exact output; the current replay head must still complete the independent conformance and Clippy steps before a pass is claimed. Do not interpret a queued or skipped job as success.

The current-parent independent fuzz workspace is [#10742](https://github.com/Luminous-Dynamics/symthaea/pull/10742), with research contract [#10726](https://github.com/Luminous-Dynamics/symthaea/issues/10726). It keeps cargo-fuzz out of production dependencies and has three targets: raw request parsing, raw result parsing, and bounded framed end-to-end replay. It uses pinned nightly/cargo-fuzz, an inspectable public seed corpus, deterministic framing tests, 1 MiB per-input cap, 8 Evidence records per fuzz frame, five-second per-input timeout and 512 MiB runner RSS ceiling. Its corrected exact-head workflow passed the deterministic frame parser suite and all three bounded campaigns. This is bounded no-crash evidence for that exact run, not a parser security certification.

The older replay draft [#10710](https://github.com/Luminous-Dynamics/symthaea/pull/10710) was closed as superseded after its branch diverged from the parent; its work was carried to #10725.

### Additional finite-oracle gates

The 23-case frozen public seed covers provider-observation support/refutation, JCS numeric equivalence and near-miss, same-scope conflict preservation, wrong subject/operation, unrepresented platform version, missing Evidence, unimplemented authenticity, assumptions, forged disposition with recomputed digest, changed falsification condition, duplicate raw JSON keys, omitted nullable fields, unknown fields, version drift, duplicate/unrequested Evidence inputs, malformed/duplicate/unrequested/conflicting result Evidence references, and CLI exit distinctions.

The public seed is a conformance floor, not a broad population estimate. A blinded holdout, external adjudication and continuous-maintenance evaluation remain necessary before claims about real-world agent reliability.

## 10. Fuzz qualification: smoke versus comparative evidence

The current-parent fuzz implementation is [#10759](https://github.com/Luminous-Dynamics/symthaea/pull/10759). It targets raw claim requests, raw claim results, and framed end-to-end replay inputs. Its exact-head workflow completed three 46-second campaigns at pinned nightly/cargo-fuzz versions:

| Target | Executions | Mean executions/s | Peak RSS |
|---|---:|---:|---:|
| Claim request parser | 1,852,807 | 40,278 | 338 MiB |
| Claim result parser | 417,569 | 9,077 | 336 MiB |
| Framed end-to-end replay | 2,873,120 | 62,459 | 319 MiB |

The deterministic frame parser unit tests passed. All three campaigns completed without a recorded crash or timeout under the 512 MiB RSS cap. The three production replay source files (Cargo.toml, src/lib.rs, src/main.rs) had identical Git blob SHAs in the #10740 and #10759 trees at comparison time. The run therefore exercised the same replay implementation; #10740's subsequent changes were in its conformance tests.

These numbers are a **bounded CI smoke receipt**, not a competitive fuzzing benchmark. Execution rates are not directly comparable across target types, different runner classes, compiler flags, sanitizer configurations, or corpus states. libFuzzer cov / ft values are instrumentation counters, not source-line coverage or proof of semantic completeness.

### Industry methodology and next target

Google FuzzBench exists because fuzzer evaluation has historically lacked a consistent methodology. Its published design uses multiple real-world benchmarks, repeated trials, raw data and statistical comparisons; its sample report uses 10 fuzzers, 24 benchmarks, 20 trials and 24-hour campaigns. OSS-Fuzz similarly separates fuzz execution from coverage inspection and supports multiple engines and sanitizers. REA should adopt those principles without pretending a 46-second CI run is equivalent.

A practical REA qualification ladder:

1. **PR smoke:** one 45-second run per target, bounded to 1 MiB inputs, five-second per-input timeout and 512 MiB RSS. This detects immediate harness regressions and crashes cheaply.
2. **Scheduled comparison:** at least 10 independent trials per target, same pinned runner image/CPU class, compiler/toolchain, sanitizer set, seed corpus digest and CPU budget. Start with five minutes per trial and archive every trial's raw data.
3. **Release qualification:** 20 independent trials per target, longer time budget, both seeded regression and evolved corpus; compare coverage/feature growth distributions and crash discovery, not just mean executions/s.
4. **Sanitizer diversity:** run address and undefined-behavior sanitizers where the target/toolchain permits; consider memory sanitizer only when dependencies are instrumented compatibly.
5. **Semantic mutation suite:** add known fault injections (wrong disposition, wrong request digest, duplicate keys, premise deletion, scope mutation, unreferenced evidence) and require each fault to be detected. Fuzzing parser bytes alone cannot establish that the semantic oracle is strong.

Record runner image, CPU model/quotas, compiler flags, sanitizer, target SHA, corpus digest, trial number, seed, duration, executions, instrumentation counters, peak RSS, crashes, minimized reproducers and artifact digests. Compare repeated trials using medians/distributions and uncertainty, with a per-fault-family view. Keep raw data so future comparisons can be reanalyzed.

**REA-specific improvement over generic fuzzing:** the benchmark's highest-value output is not raw input throughput; it is the set of new parser states and semantic invariants exercised, the time to discover a minimized counterexample, and whether every injected epistemic fault is rejected. A high-throughput fuzzer that never reaches a valid claim/result/Evidence path can be less useful than a slower target with a grammar-aware seed corpus.

References:
- [FuzzBench paper](https://research.google/pubs/fuzzbench-an-open-fuzzer-benchmarking-platform-and-service/)
- [FuzzBench benchmark methodology and sample report](https://github.com/google/fuzzbench)
- [OSS-Fuzz documentation](https://google.github.io/oss-fuzz/)
- [OSS-Fuzz project setup and coverage guidance](https://google.github.io/oss-fuzz/getting-started/new-project-guide/)

## 9. New qualification paradigm — metamorphic semantic laws

Issue [#10738](https://github.com/Luminous-Dynamics/symthaea/issues/10738) extends fixed examples with transformations whose expected relation is known even when no new answer label is authored.

Initial laws implemented or being qualified in #10740:
- recursive JSON object-key permutation is representation-only; Evidence ID and replay must remain equivalent under JCS;
- reordering raw Evidence transport records is representation-only when the claim's exact ordered Evidence ID list is unchanged;
- reordering the claim's Evidence ID list is semantic input mutation and the old result must not replay;
- duplicate or unreferenced records are invalid, never extra votes;
- premise weakening, unsupported assumptions and absent scope dimensions must never produce an epistemic upgrade;
- adding a same-scope contradiction preserves conflict rather than majority-voting it away;
- unsupported predicate families remain non-positive regardless of evidence count.

The target is 100% preservation of the finite metamorphic laws and zero unexpected upgrades. These tests qualify the transformations they exercise; they do not prove general semantic correctness.


## 8. Quantitative baseline refresh and fair comparison protocol (2026-11-06)

### Lifecycle evidence is now a measurable SOTA axis

Vera-Life-Eval reports 8,500 questions and 41,203 versioned evidence pieces across six lifecycle settings: current, past, stale-only, update, conflict and multi-hop. Its Vera-Life controller reports LifeEM 0.623 versus 0.374 for the best prompt-only baseline on the shared Qwen3-8B generator; timestamp-only and text-only regimes report 0.584 and 0.504 respectively. This is a strong published lifecycle-reasoning baseline, but it is not a directly comparable score to REA's deterministic claim conformance suite.

REA should adopt its lifecycle categories and build a source-disjoint, versioned test track after composing #10600. Proposed target: zero supported claims on stale-only or superseded evidence when the target-time contract invalidates them; exact expected dispositions on the finite lifecycle oracle; and separate current/past/conflict/unknown coverage, not one LifeEM-like scalar.

Reference: [Vera-Life-Eval (GroundLM 2026)](https://aclanthology.org/2026.groundlm-1.18/).

### Continuous engineering is a different estimand from one-shot repair

SWE-Milestone's ICML 2026 study reports evaluated systems above 80% on isolated tasks but no more than 38% in continuous milestone settings, across 12 frontier models and four agent frameworks. SWE-CI provides 100 repository-maintenance tasks with average histories of 233 days and 71 commits. These benchmarks justify treating exact-head evidence as revision-local: a qualification receipt at revision R must not imply the claim remains current at R+1.

REA's continuity target should be measured as:
- stale-claim detection recall after relevant source/dependency changes;
- selective revalidation precision (how many re-runs were actually necessary);
- stale claim false-retention rate;
- regression and recovery rate over a milestone sequence;
- time/cost to re-establish a qualified claim.

Reuse #10630's source/repository continuity model rather than introduce another revision graph.

References: [SWE-Milestone (ICML 2026)](https://proceedings.mlr.press/v306/deng26d.html); [SWE-CI](https://arxiv.org/abs/2603.03823).

### AgentRE v3.1 is a fixed-suite diagnostic, not a transferable truth score

The November 5 v3.1 report has six models × 20 Windows PE artifacts × five attempts = 600 episodes. Muse Wonder scored 87.37 Overall and GPT-6 Astra 87.04; the simultaneous 95% interval for the difference is -3.8203 to +4.4689 and includes zero. Both supplied 100/100 valid answers. DeepSeek V4.1 Flash scored 78.31 with 98/100 valid answers. The review reproduced all 600 score objects, but human adjudication remains outstanding; 46 source-supported mappings still incur field-placement/allowed-fact penalties, and no episode received full credit.

REA should reproduce the task/attempt denominator and keep at least these axes distinct: artifact fact accuracy, claim-evidence entailment, allowed-fact and schema placement, coverage/nonanswers, and independent adjudicator agreement. A correct answer placed in the wrong field is an interface/schema error; a claim unsupported by the binary is a factual error. Do not combine them without a published weighting rule.

Reference: [AgentRE-Bench v3.1 report, updated 2026-11-05](https://www.agentre-bench.ai/results/v3-1/).

### Statistical target: estimate generalization, not just fixed-corpus accuracy

NIST AI 800-3 warns that simple averages and standard errors can imply invalid uncertainty when benchmark assumptions are implicit, system performance is conflated, or task populations are heterogeneous. Its demonstration uses a generalized linear mixed model across 22 API-access models and three benchmarks to separate system and item/task variation.

For a later REA agent benchmark, preregister:
- fixed-suite performance and target-population performance as different estimands;
- random effects for artifact family, task and source lineage where sampling supports them;
- paired comparisons on identical evidence, tool budgets, model settings and target-time cutoffs;
- uncertainty intervals and multiplicity control for multiple model comparisons;
- full denominator accounting for abstentions, invalid outputs, refusals, timeouts and budget exits.

Do not claim a 100% finite-oracle pass is a 100% generalization rate. The finite oracle is a software correctness gate; statistical generalization is a separate empirical question.

References: [NIST AI 800-3 (February 2026)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-3.pdf); [NIST TEVV-Athlon framework](https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems).

### Standards versioning and semantics

Pin exact standard revisions in every adapter and qualification receipt. The TUF versioned v1.0.36 specification page is dated 2026-08-05; the specification repository's current master metadata labels a 1.0.37 text dated 2026-10-08, while the public version selector still lists v1.0.33 as its latest entry. This is a documentation/version-index discrepancy, not a reason to invent a single "latest" version. For reproducible integration, record the exact versioned text URL and content digest and distinguish a published version from a repository working draft.

RFC 8785 is an Informational RFC and requires the I-JSON-compatible subset: no duplicate object property names, Unicode preservation, and numbers expressible as IEEE-754 binary64. This is why REA rejects duplicate keys and uses ECMAScript-compatible JCS for semantic commitments. For values requiring exact precision beyond binary64, encode the value as a string under an application schema rather than relying on an interoperable JSON number.

SCITT RFC 9943 explicitly says an issuer may make false statements; a receipt demonstrates registration in the transparency service, not the accuracy of the statement. This is a hard non-implication the REA typed-claim layer must preserve.

References: [TUF v1.0.36](https://theupdateframework.github.io/specification/v1.0.36/); [TUF specification repository](https://github.com/theupdateframework/specification); [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html); [RFC 9943 §9.2](https://www.rfc-editor.org/rfc/rfc9943.html#section-9.2).

### Fair SOTA comparison: REA should be a composition layer, not a replacement leaderboard

The existing tools measure orthogonal properties. REA should compare against them in matched workflows:
1. Native SLSA/TUF/Sigstore/SCITT verification with their own policy gates.
2. REA Evidence admission alone.
3. Producer claim evaluation.
4. Independent replay.
5. Lifecycle + source-lineage + revision-continuity composition where the owner APIs are qualified.
6. Model/agent synthesis with the same evidence but no claim gating.
7. Model/agent synthesis with claim gating, independent replay and explicit abstention.

The improvement claim should be: lower false-entailment and stale-evidence error at a measured cost, while retaining useful coverage—not "beats SLSA" or "beats AgentRE" on unrelated scores.


## 9. New source-grounding and reverse-engineering frontiers (2026-11-06)

### RDQA: correctness is not evidence retrieval

The October 2026 EMNLP paper *Can Your Agent Read the Source Faithfully?* introduces RDQA: 1,350 questions over documents, video and audio, with 20 real-world scenarios and nine capability families. It defines Parametric Leakage Rate (PLR) to catch correct answers generated without genuine retrieval. Across 12 agent systems, the best generic agent has 24.0% adjusted accuracy and a process-aware RDQA-Skill baseline reaches 41.1%; PLR reaches as high as 96.0% in the evaluated systems. These are RDQA results, not REA performance numbers, but the failure mode is exactly relevant: answer correctness and evidence-grounded correctness are different estimands.

**REA consequence: add a source-grounding / parametric-leakage track.** For each hidden artifact, include unique per-instance facts that cannot be reliably guessed from domain priors. The benchmark should:
- randomize high-entropy values, offsets, identifiers, constants and behavior markers across artifacts;
- use paired counterfactual artifacts that differ in one controlled fact while keeping surrounding task language fixed;
- require the agent's answer and claim result to bind the exact source artifact/member digest and occurrence;
- instrument actual source-byte reads / tool access and preserve extraction or observation digests, not merely model-reported tool calls;
- score both correct recovery and whether the conclusion changes appropriately under the counterfactual mutation;
- measure source-grounded accuracy, unsupported-correct-answer rate / PLR, evidence coverage, false entailment, abstention and mutation sensitivity separately.

A model that guesses the right value but cannot bind it to an observation gets zero source-grounded credit. A model that calls a tool but cites the wrong member/occurrence also gets zero. Tool-call count is not evidence; instrumented access plus exact observation identity and claim scope is.

The key metamorphic law is **counterfactual sensitivity**: when an artifact's relevant bytes are changed and the expected fact changes, the system's evidence-backed answer must change accordingly or abstain. When unrelated bytes are changed, the claim may remain invariant only if its exact scope/subject contract permits that transformation. Both sides need explicit expected labels.

Reference: [Gao et al., EMNLP 2026, RDQA / source-grounded multimodal retrieval](https://aclanthology.org/2026.emnlp-main.1855/).

### SRE-Bench and AgentRE-Bench measure different reverse-engineering regimes

The October 2026 SRE-Bench update reports 19 private programs, 262 binary instances and 1,572 deterministically graded tasks across firmware, malware, games, file formats and protocols, four source languages and 44 anti-analysis mechanisms. Public-facing GPT-6 Astra scores 5.55 of six tasks on 181 graded instances, fully solving 149 (82.3%) of those graded instances. That denominator matters: model-side cyber safeguards left 81 of the 262 eligible instances ungraded/refused. In the separate unconstrained evaluation, GPT-6 Astra reaches pass@4 99.2% but pass@1 88.0%, showing that candidate selection/reliability remains an issue.

AgentRE-Bench v3.1 is a different fixed Windows PE benchmark: six models, 20 artifacts, five attempts each (600 episodes). Its current 87.37 vs 87.04 scores are not comparable with SRE-Bench's out-of-six task score. The two should be reported as separate protocols, not blended into a synthetic leaderboard.

**REA consequence:** build two separate transfer tracks:
1. **Deterministic artifact-operation track:** ZIP/APK/IPA/member evidence admission, integrity, signatures, parsers and exact-byte controls. Aim for complete finite-oracle correctness and zero false accepts.
2. **Agent reverse-engineering track:** protected held-out artifacts, source-free tasks, exact observation lineage, counterfactual mutation pairs, source-grounding/PLR, per-domain coverage, refusal/budget accounting and independent human/automated adjudication.

REA should be the evidence/claim layer beneath either track, not claim to beat the binary-analysis specialists at disassembly or decompilation by itself.

References: [DAPLab SRE-Bench (updated October 2026)](https://daplab.cs.columbia.edu/benchmarks/sre-bench/); [AgentRE-Bench v3.1, updated November 5, 2026](https://www.agentre-bench.ai/results/v3-1/).

### New proposed benchmark gates for source-grounded RE

| Metric | Proposed target | What it prevents |
|---|---:|---|
| Source-grounded claim accuracy | 100% on finite deterministic hidden-value controls | Correct-looking but ungrounded answers |
| Parametric leakage rate on controlled facts | 0% on unique high-entropy facts | Guessing from priors |
| Counterfactual mutation sensitivity | 100% on relevant one-field mutations | Evidence-insensitive answers |
| Irrelevant-mutation invariance | 100% where the scope is unchanged | Overreacting to unrelated bytes |
| Source/member/occurrence binding | 100% | Citing the wrong artifact or duplicate-name member |
| False supported claims on unsupported controls | 0 | Plausible but non-entailed conclusions |
| Evidence access attribution | 100% of accepted source claims have instrumented acquisition/observation lineage | Self-reported tool-use as proof |
| Adjudication agreement | Report inter-rater/automatic-human agreement and unresolved cases | Treating a reproducible but flawed grader as ground truth |

These are proposed release gates, not measured REA results. Do not copy RDQA's PLR numerical definition without inspecting and pinning its scoring protocol; use the concept but define a REA-specific denominator and adjudication plan before collecting scores.
