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
| REA Evidence v1 | Versioned structured provider observation, schema validation, semantic identity and JCS commitment | Producer authentication, trusted time, source independence, archive completeness, or the truth of the normalized observation | Existing source contract; do not mutate identity semantics |
| REA claim evaluator v0 | Exact provider-reported normalized result under matching subject/operation/represented environment scope | General semantic entailment, authenticity, temporal validity, independence or security verdicts | Narrow semantic composition rule; all stronger predicates fail closed |
| REA independent replay verifier | Reproduction of the serialized v0 disposition and derivation from raw request and verified Evidence | Independent cryptographic primitives, producer authenticity, or underlying artifact truth | Independent semantic implementation; shared qualified Evidence/JCS primitives are disclosed |

### Standards and implementation references

- [SLSA specification v1.2](https://slsa.dev/spec/v1.2/) and [v1.2 provenance overview](https://slsa.dev/spec/v1.2/provenance)
- [Sigstore Rekor overview](https://docs.sigstore.dev/logging/overview/)
- [Sigstore security model](https://docs.sigstore.dev/about/security/)
- [IETF RFC 9943 — SCITT architecture](https://www.rfc-editor.org/rfc/rfc9943.html)
- [IETF RFC 9162 — Certificate Transparency v2](https://www.rfc-editor.org/rfc/rfc9162.html)
- [The Update Framework specification](https://github.com/theupdateframework/specification/blob/master/tuf-spec.md)
- [IETF RFC 8785 — JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html)

Important detail: SLSA v1.2 is the current approved specification and includes Build and Source tracks. The Build Track describes artifact provenance and expected build process/inputs; the Source Track describes how source revisions were created and change-managed. The v1.1 provenance page is now marked retired. REA adapters should pin the track-specific v1.2 predicate and preserve the expected signer/builder policy boundary. Rekor's log entries are signed records and its log requires independent monitoring. SCITT explicitly distinguishes registration of a signed statement from the statement's accuracy. These are precisely the boundaries the REA claim layer must preserve. 

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

### Continuous engineering — SWE-Milestone and SWE-CI

SWE-Milestone reconstructs real release histories into milestone DAGs and evaluates agents in an unbroken sequence, preserving earlier mistakes and regression effects. Its paper reports that some isolated-task results above 80% fell to at most 38% in continuous settings across its evaluated models/frameworks. Its current public leaderboard is a different versioned evaluation and must not be conflated with the paper's study.

SWE-CI evaluates 100 repository-maintenance tasks whose histories span an average of 233 days and 71 commits. Both reinforce that a patch passing at one revision is not evidence that a repository claim remains valid after later changes.

**REA consequence:** a claim derivation is valid for its exact source revision, dependency closure and rule version. Add a continuity track for selective revalidation after changes; reuse #10630 rather than creating another repository graph.

References:
- [SWE-Milestone paper (ICML 2026)](https://proceedings.mlr.press/v306/deng26d.html)
- [SWE-Milestone project and current leaderboard](https://swe-milestone.com/)
- [SWE-CI paper](https://arxiv.org/abs/2603.03823)

### Lifecycle and temporal reasoning — Vera-Life-Eval, TRACE and TKE-Bench

Vera-Life-Eval contains 8,500 questions and 41,203 versioned evidence pieces across current, past, stale-only, update, conflict and multi-hop conditions. Its lifecycle-faithfulness definition requires semantic support, target-time validity, non-supersession and provenance.

TRACE provides more than 16,000 model-specific chronological tests across Modify and Insert settings, with seven evaluation dimensions and explicit probing of history/future preservation.

TKE-Bench studies temporal knowledge accessibility and the ChronoExpert paper reports catastrophic temporal interference in the evaluated editing baselines.

**REA consequence:** query-time target time and system knowledge time must be distinct. A receipt timestamp is not source valid time; a current retrieval does not make its source current. Reuse #10600 for lifecycle/as-of semantics and keep v0 platform-version/currentness claims underdetermined when the source contract cannot bind them.

References:
- [Vera-Life-Eval / Vera-Life (ACL Anthology, 2026)](https://aclanthology.org/2026.groundlm-1.18/)
- [TRACE (EMNLP Findings 2026)](https://aclanthology.org/2026.findings-emnlp.1045/)
- [TKE-Bench / ChronoExpert (EMNLP Findings 2026)](https://aclanthology.org/2026.findings-emnlp.1823/)

### Statistical validity — NIST AI 800-3 and TEVV-Athlon

NIST AI 800-3 distinguishes accuracy on a fixed benchmark from generalized accuracy over a broader population of similar tasks. It warns that common analyses can hide assumptions or produce invalid uncertainty estimates; it demonstrates generalized linear mixed models and variance decomposition on 22 API-access models and three benchmarks.

NIST's TEVV-Athlon Framework is a four-stage method for constructing a context-specific assessment from TEVV objectives; it structures that assessment around Events and Tools that produce data for Blocks linked to measurement concepts. The initial NIST AI 200-2 public draft comment period ran through October 6, 2026. NIST AI 800-3 (February 2026) separately argues that benchmark generalization and uncertainty depend on explicit statistical assumptions, and demonstrates hierarchical modelling of model capability and item difficulty.

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

**SLSA is a build-platform claim, not a complete semantic inventory.** In SLSA Provenance v1.1, `builder.id` determines the trusted build-platform boundary and the accuracy/completeness guarantee that consumers may rely on. `resolvedDependencies` is an unordered set and its completeness is best-effort through Build L3. Therefore, absence of a dependency from a provenance record is not a proof that the build did not consume it. The REA claim model should retain the SLSA build level and builder trust policy as explicit scope/assumptions and must not infer a complete dependency closure from a well-formed provenance object. [SLSA v1.1 provenance](https://slsa.dev/spec/v1.1/provenance).

**Sigstore is more than the Rekor database.** A full Sigstore verification flow checks the artifact signature and digest, verifies that the certificate identity matches the expected identity and validates the certificate against the Sigstore trust root, then checks transparency-log inclusion. Rekor provides auditable registration and inclusion evidence; the security model also calls out monitoring and the trust placed in the log/checkpoint. This can establish an identity-bound signing event for the artifact bytes under the chosen policy. It still does not prove those bytes are benign, that the signer is authorized for every product namespace, or that a semantic REA claim about the artifact is correct. [Sigstore overview](https://docs.sigstore.dev/); [Rekor security model](https://docs.sigstore.dev/about/security/).

**SCITT makes the statement-truth boundary explicit.** RFC 9943 says an issuer may make false statements and that registration proves the statement was produced by an issuer; relying parties still determine accuracy. This is not a weakness unique to SCITT: transparent registration is intentionally orthogonal to semantic truth. REA should link a SCITT receipt to the signed statement digest, verify issuer identity/policy separately, and then evaluate the statement's predicate against artifact- and time-scoped Evidence. [RFC 9943 §9.2](https://www.rfc-editor.org/rfc/rfc9943.html#section-9.2).

**TUF is a metadata-update security framework.** Its threat model includes rollback, freeze, fast-forward, mix-and-match, threshold-key compromise and wrong-software installation. A TUF-qualified target record is strong evidence about what a trusted repository authorized as a target under a metadata version/expiry policy. It is not a substitute for inspecting the target's internal semantics or proving runtime behavior. [TUF specification](https://github.com/theupdateframework/specification/blob/master/tuf-spec.md).

## 8. Implementation update — 2026-11-06

The producer implementation is [#10665](https://github.com/Luminous-Dynamics/symthaea/pull/10665), whose focused REA workflow passed at exact head `08947cc924386d75190a0ca3a9660b5a0425cd31` (40 tests across the focused harnesses). That is a focused compatibility/claim-evaluator result, not full-workspace qualification or general semantic truth.

The independent verifier is in refreshed stacked draft [#10710](https://github.com/Luminous-Dynamics/symthaea/pull/10710), tracked by [#10673](https://github.com/Luminous-Dynamics/symthaea/issues/10673). It now includes:
- a frozen public seed oracle at `tools/rea-claim-replay-verifier/tests/fixtures/rea-claim-replay-v0-oracle.json`, with 23 explicit expected-disposition/replay-outcome cases, authored independently of the production evaluator;
- a standalone, bounded, read-only `rea-claim-replay` CLI: exit 0 for replay match, 1 for semantic mismatch, 2 for invalid input / IO / resource failure;
- duplicate-key rejection and schema validation for raw claim and result JSON;
- verification of semantic Evidence IDs from raw Evidence inputs;
- rejection of duplicate, malformed, missing-inconsistent, unrequested and unreferenced Evidence identifiers in the claim/result/replay set;
- complete-result comparison under RFC 8785 canonical bytes, not a self-reported derivation hash;
- explicit fail-closed treatment of unsupported predicates, assumptions and scope dimensions absent from REA Evidence v1;
- a pinned Rust 1.99.1 toolchain, isolated Cargo manifest, generated lockfile artifact, formatting, conformance tests and warnings-denied Clippy gates.

The independent implementation deliberately reuses the REA Evidence identity verifier and JCS canonicalization primitive. It is independent at the claim scope/disposition/replay implementation level, not a second cryptographic primitive or canonicalizer. The new verifier's exact-head workflow is queued/in progress in this draft; no replay tests or Clippy result are claimed as passing yet.

### Additional finite-oracle gates

The public seed corpus now explicitly covers:
- provider-observation support and refutation;
- JCS numeric equivalence and numeric near-miss;
- same-scope contradiction preservation;
- wrong subject/operation and unrepresented platform version;
- missing Evidence and unimplemented authenticity;
- unsupported assumptions;
- forged disposition even after recomputing an unkeyed derivation digest;
- changed falsification condition with an old result;
- duplicate raw JSON keys, omitted nullable fields, unknown fields and version drift;
- duplicate/unrequested Evidence inputs;
- malformed, duplicate, unrequested and conflicting result Evidence references;
- CLI output and stable semantic-match/mismatch exit distinctions.

The frozen public seed is a conformance floor, not a broad population estimate. A hidden holdout and external adjudication are still required before making claims about real-world agent reliability.
