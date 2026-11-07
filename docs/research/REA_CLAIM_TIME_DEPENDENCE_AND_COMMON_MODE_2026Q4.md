# REA claim time-dependence and common-mode qualification (2026 Q4)

**Status:** research/design proposal for a future, separately versioned claim-rule family. This document does not change REA Evidence v1, claim evaluator v0, or the independent replay contract.

**Ownership:** integration proposal for [#10760](https://github.com/Luminous-Dynamics/symthaea/issues/10760), composing [#10600](https://github.com/Luminous-Dynamics/symthaea/issues/10600) temporal validity, [#10410](https://github.com/Luminous-Dynamics/symthaea/issues/10410) provider lineage, [#10630](https://github.com/Luminous-Dynamics/symthaea/issues/10630) evidence continuity, [#10629](https://github.com/Luminous-Dynamics/symthaea/issues/10629) typed claim entailment, and [#10740](https://github.com/Luminous-Dynamics/symthaea/pull/10740) independent replay.

---

## 1. Executive design decision

The next claim-evaluation generation needs two additional dimensions:

1. **Temporal applicability:** whether an observation is valid for the requested event, acquisition, evaluation time and freshness policy.
2. **Dependence-aware corroboration:** whether two observations are independent under an explicit failure-domain and lineage model, or merely distinct reports derived from common causes.

Neither dimension changes what an Evidence record says. They determine whether that Evidence is applicable to a particular claim and what kinds of inference are authorized.

A timestamp is not automatically trusted time. A distinct provider name is not automatically an independent source. A transparency receipt proves a registration event under a log policy, not the truth or event-time of the statement. Unknown clocks and unknown ancestry stay unknown.

The implementation must not turn these dimensions into a universal confidence or trust score. They belong in typed, versioned policy and derivation inputs.

## 2. Temporal model: separate the clocks

A single timestamp cannot describe the lifecycle of a piece of evidence. Keep these events distinct:

| Time dimension | Meaning | Typical source | What it does not establish alone |
|---|---|---|---|
| Event time | When the underlying behavior/state allegedly occurred | Subject system, event log, measurement | When anyone observed it |
| Acquisition time | When the provider acquired data from the subject | Device/session log, collector | That the provider clock is correct |
| Observation time | When analysis was performed or the state was measured | Provider execution record | When the statement was issued |
| Statement time | When the issuer created the claim/Evidence | Signed statement metadata | That the event happened then |
| Ingestion time | When REA received the record | Receiving service | That the upstream clock is trustworthy |
| Transparency time | When a log registered a signed statement | CT/Rekor/SCITT receipt | The underlying event time or truth |
| Evaluation time | When the relying party evaluates the claim | Verifier clock | That the claim was true at another time |

A claim request should state the temporal question it asks. The Evidence side should say which time dimensions are actually present and how each was established. Missing times must not be silently filled from the verifier's current clock.

### 2.1 Candidate versioned structures

A future claim rule may reference a TemporalScope with:

- an as-of instant or half-open interval;
- the temporal relation required by the predicate (at, during, before, after, overlaps, latest-known-at);
- a declared maximum observation age, when freshness is meaningful;
- allowed clock skew and time-resolution limits;
- required timestamp authority / time-source policy;
- a revocation and supersession rule;
- an explicit statement of which clock dimensions are mandatory.

A linked TemporalObservation may contain:

- event-time interval, if known;
- acquisition start/finish, if known;
- observation start/finish, if known;
- statement creation time;
- ingestion time;
- transparency receipt and checkpoint references;
- clock identity, synchronization evidence, uncertainty bounds and source lineage;
- whether a field is provider-reported, collector-observed, hardware-attested, or externally timestamped.

A separate FreshnessPolicy should define max age, tolerated skew, handling of gaps, expiry semantics and the clock authority accepted for the specific claim family. Policies must be versioned and content-addressed. A freshness policy is not itself evidence that an event occurred.

### 2.2 Temporal dispositions

Keep temporal status distinct from claim truth. Candidate typed statuses:

| Status | Meaning |
|---|---|
| CurrentWithinPolicy | Required temporal conditions are established for the requested as-of scope |
| Stale | The policy's freshness/expiry bound is exceeded |
| NotYetValid | The record's validity window has not started |
| RevokedOrSuperseded | A relevant revocation/supersession rule invalidates use of this statement |
| TimeUnverifiable | Required clock/time evidence is absent or untrusted |
| ClockConflict | Required time sources disagree outside allowed uncertainty |
| TemporallyInapplicable | The record belongs to a different event/time scope |
| TemporalUnknown | The available evidence cannot resolve the relation |

These are applicability results. A record can be current and still be false. A record can be authentic and still be stale. A stale record does not prove the opposite of what it once reported.

### 2.3 Time trust ladder

Do not treat all clocks as equivalent:

1. Provider-reported wall clock: an assertion by the provider.
2. Collector-observed receipt time: a time at the REA boundary, subject to the collector's own trust policy.
3. Signed timestamp from a trusted authority: a statement-time commitment under an authority policy.
4. Transparency-log inclusion/checkpoint: evidence that a statement was registered by a log by a checkpoint, under the log's assumptions and monitoring policy.
5. Hardware- or platform-attested time: only as strong as the attestation chain, hardware trust and verification policy.

A higher rung does not automatically validate lower-rung event timestamps. For example, a log inclusion receipt can bound when a provider statement was registered; it does not prove when the underlying artifact was observed or the reported event occurred.

## 3. Common-mode model: independence is relational

### 3.1 Why provider count is unsafe

Four reports may be one observation copied four times. Different tools may share the same parser, database, model, signature rules, acquisition session or upstream report. Two operations from one suite may instead use independent acquisition channels.

The relevant unit is not the provider name. It is the declared ancestry of the specific observation, under a predicate-specific failure-domain model.

### 3.2 Candidate lineage objects

Keep provider lineage as a linked, independently verifiable object. Do not change the REA Evidence semantic identity projection without an upstream contract revision.

A lineage profile may capture, where known:

- provider executable/container digest and version;
- operation and configuration/profile digest;
- parser, decoder, disassembler, decompiler and native library ancestry;
- analysis engine and plugin versions;
- signature/rules database digest and update revision;
- model family, model-weight digest and calibration/training sources when disclosed;
- acquisition session, transport path and target/runtime environment;
- upstream statement/Evidence identifiers;
- operator/manual interpretation and reviewer lineage;
- unknown or proprietary dependencies as explicit unknown nodes.

Each node and edge must be scoped and supported by evidence. A name or version string is not a digest. An edge is not true merely because a provider declares it; retain its source and verification state.

### 3.3 Candidate relationship vocabulary

Examples include:

- USES_ENGINE
- USES_PARSER
- USES_RULESET
- USES_MODEL
- CALIBRATED_FROM
- ACQUIRED_THROUGH
- DERIVED_FROM
- REFORMATS
- REPLAYS_FROM
- SAME_ACQUISITION
- SHARES_UPSTREAM_REPORT
- RUNS_IN_ENVIRONMENT

These relationships should have issuer, source Evidence, validity interval and verification status. Unknown lineage is an explicit result, not an empty graph that can be mistaken for independence.

### 3.4 Independence outcomes

Candidate classifications for a pair/set of observations:

- SharedFailureDomain
- IndependentWithinDeclaredGraph
- LineageUnknown
- LineageIncomplete
- NotApplicable

The words “independent within declared graph” are intentionally bounded. They mean no relevant shared failure-domain edge was found in the declared and sufficiently complete graph for this predicate. They do not mean metaphysical or universal independence.

No global independence scalar. No simple “N tools agree” vote. A future rule may reason about distinct failure domains only after it states which failure domains matter for that predicate.

## 4. Composition rules and non-implications

The future evaluator should enforce these boundaries:

1. A provider timestamp alone cannot establish event time or freshness.
2. A trusted registration time cannot be substituted for observation time.
3. A valid signature cannot establish that the statement is accurate.
4. A current statement can still be false.
5. A stale statement is not evidence of the opposite proposition.
6. Different provider names do not establish independence.
7. Removing lineage information cannot increase independence.
8. A record derived from another record cannot count as an independent acquisition.
9. Additional evidence sharing the same failure domain cannot resolve a conflict by quantity.
10. A missing observation is not a negative observation.
11. A new dependency or policy event should invalidate only claims in its affected dependency/time closure.
12. An unrelated event should preserve claims whose scope and inputs are unchanged.

Unsupported time or lineage must lower the permitted conclusion or produce an explicit unresolved result. It must never be silently upgraded.

## 5. Adversarial benchmark design

Keep temporal and common-mode tests in separate strata, then combine them in a smaller cross-product.

### 5.1 Temporal strata

| Family | Positive control | Adversarial mutation | Expected relation |
|---|---|---|---|
| Freshness | In-policy observation at requested time | Evaluate after expiry | Current becomes Stale |
| Future time | Trusted source within skew | Provider reports future observation | TimeUnverifiable or ClockConflict |
| Backdating | Statement and log time agree | Provider backdates statement | Log time does not rewrite event time |
| Clock skew | Independent clocks within bound | Offset exceeds policy | ClockConflict |
| Revocation | Valid credential at event time | Credential later revoked | Apply explicit status-time rule; never assume timeless validity |
| Supersession | Latest statement selected | Reuse superseded statement | RevokedOrSuperseded or policy-defined historical validity |
| Transparency | Inclusion and checkpoint verify | No independent monitoring/consistency | Registration is established; split-view assurance remains limited |
| TUF metadata | Current version and unexpired metadata | Rollback/freeze/expired metadata | Reject current-update claim |
| Missing time | No required time dimension | Attempt to infer from ingestion time | TimeUnverifiable |
| Scope transition | Same artifact/time scope | Device/SDK/platform changes | Prior claim only survives if its scope permits |

### 5.2 Common-mode strata

| Family | Setup | Expected classification |
|---|---|---|
| Same parser | Different provider names, same parser digest | SharedFailureDomain |
| Shared engine | Different wrappers, same native engine | SharedFailureDomain |
| Shared database | Different engines, same rules/signature DB | Dependence recorded for relevant predicate |
| Shared model | Different API wrappers, same model family/backend | Independence not assumed |
| Reformatting | Provider B summarizes A | DerivedFrom, not independent |
| Same acquisition | Multiple reports from one session | Shared acquisition domain |
| Independent acquisition | Separate channels and roots | IndependentWithinDeclaredGraph only if graph is sufficiently complete |
| Unknown ancestry | Proprietary/unreported dependency | LineageUnknown |
| Hidden common cause | Inject same false fact through a shared source | Must not produce independent corroboration |
| Independent counterevidence | One independent source contradicts one shared cluster | Preserve contradiction and lineage |

### 5.3 Cross-product tests

Combine:
- stale evidence from a shared parser;
- a current but provider-reported timestamp from an unknown clock;
- two provider names whose reports share a model;
- an independent observation with a mismatched platform;
- a superseded statement that is still present in an append-only log;
- a policy update that invalidates only one branch of the dependency graph.

The expected result must distinguish temporal applicability, lineage independence and semantic disposition. Do not compress them into one score.

## 6. Metrics and qualification targets

Report each metric separately:

| Metric | Definition | Initial target |
|---|---|---|
| False temporal-upgrade rate | Invalid/missing/untrusted time control accepted as current | 0 seeded false upgrades |
| Stale acceptance rate | Expired/stale control accepted for a current claim | 0 seeded false accepts |
| Revocation/supersession recall | Invalidated statements correctly rejected under policy | 100% on finite deterministic suite |
| False-independence rate | Shared-domain or unknown-lineage pair classified as independent | 0 seeded false independence |
| Lineage edge precision/recall | Correct typed relationship recovery on labeled graph | Report per edge family; no hidden aggregate |
| Unknown calibration | Unknown ancestry/time kept unknown when evidence is absent | 100% on missingness controls |
| Selective invalidation precision | Only claims affected by an event are invalidated | 100% on finite deterministic mutation matrix |
| Selective invalidation recall | Every affected claim is invalidated | 100% on finite deterministic mutation matrix |
| Replay determinism | Same exact request, Evidence, policy and graph versions yield same derivation | 100% |
| Revalidation latency | p50/p95 from change event to qualified new state | Establish baseline, then reduce without weakening gates |
| Rework and recovery | Repeated/reversed qualification after a failed check | Report separately |

For every zero-failure result, report the denominator and a confidence bound. Zero failures in a small fixture set is not zero underlying risk. Use blinded holdouts, independent adjudication and hierarchical statistics when generalizing beyond a fixed corpus.

## 7. Industry alignment and boundaries

- **SLSA v1.2:** approved Build and Source tracks. Build provenance binds artifact to build process and inputs; Source provenance addresses revision creation/change management. The current working draft adds Build Environment and Dependency tracks, but these remain draft proposals. REA should consume signed provenance under explicit signer-builder policy; absence from best-effort resolvedDependencies is not a general proof of non-use.
- **Sigstore/Rekor and Certificate Transparency:** signed identity and transparent registration/inclusion/consistency. They require monitoring/gossip or equivalent defenses for split views and do not establish semantic truth.
- **SCITT RFC 9943:** transparency and accountability for signed statements. Issuers may make false statements; relying parties assess accuracy and supersession.
- **TUF 1.0.37:** version, threshold, expiry, rollback and freeze protections for update metadata. It does not prove artifact behavior or semantic safety.
- **NIST AI 800-3 and TEVV-Athlon:** statistical uncertainty/generalization and context-specific assessment design.
- **DORA:** disaggregated throughput and instability measures; useful precedent for avoiding one composite operational score.
- **SWE-Milestone and SWE-CI:** continuous software evolution and maintenance histories; REA adds the lifecycle of evidence validity and claim invalidation across those changes.

## 8. Proposed execution sequence

1. Freeze a schema-neutral temporal/lineage vocabulary and threat model.
2. Produce 20–30 design-only adversarial cases with independent expected applicability/lineage labels.
3. Map each proposed field to an existing owner and existing REA objects; do not duplicate the graph or receipt.
4. Add deterministic no-upgrade tests against current v0, showing that unsupported time/lineage cannot upgrade existing predicates.
5. Wait for the owning temporal and lineage objects before defining a v1 wire schema.
6. Bind temporal policy digest, lineage graph digest, trust roots, rule version and exact Evidence IDs into future replay.
7. Add a blinded holdout and external adjudication before performance claims.

## 9. Claim ceiling

This design qualifies declared temporal policies and lineage relationships over exercised fixtures. It does not prove a source clock is truthful, a proprietary dependency declaration is complete, distinct failure domains are truly independent, or the underlying real-world proposition is true.

---

## References

- [SLSA specification v1.2](https://slsa.dev/spec/v1.2/)
- [SLSA v1.2 Build provenance](https://slsa.dev/spec/v1.2/build-provenance)
- [SLSA v1.2 tracks](https://slsa.dev/spec/v1.2/tracks)
- [SLSA working draft tracks](https://slsa.dev/spec/draft/tracks)
- [Sigstore security model](https://docs.sigstore.dev/about/security/)
- [IETF RFC 9943 — SCITT architecture](https://www.rfc-editor.org/rfc/rfc9943.html)
- [IETF RFC 9162 — Certificate Transparency v2](https://www.rfc-editor.org/rfc/rfc9162.html)
- [The Update Framework 1.0.36](https://theupdateframework.github.io/specification/v1.0.36/)
- [NIST AI 800-3 (February 2026)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-3.pdf)
- [NIST TEVV-Athlon Framework](https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems)
- [DORA software delivery performance metrics](https://dora.dev/guides/dora-metrics/)
- [SWE-Milestone paper (ICML 2026)](https://proceedings.mlr.press/v306/deng26d.html)
- [SWE-CI paper](https://arxiv.org/abs/2603.03823)

## 8. Population completeness and selective disclosure

Temporal applicability and provider lineage still do not establish that the evidence population is complete. A complete, append-only transparency view can be a complete view of what was registered while remaining incomplete as a history of what an issuer created or observed.

RFC 9943 §9.3 explicitly allows issuers to refuse registration or selectively submit only some statements. Its §9.2 also distinguishes statement registration from statement accuracy and permits supersession/correction. Therefore:

- complete transparency-log enumeration means all entries in the named log/tree/checkpoint scope were retrieved and verified;
- it does not mean every statement made by the issuer was registered;
- an issuer-signed declaration that it submitted everything is still an issuer assertion unless a separately qualified participation/audit policy establishes that obligation and its fulfillment;
- provider-reported terminal cursors, page counts, or complete=true fields do not independently establish population completeness.

### Separate completeness dimensions

Keep six questions distinct:

1. Population definition: who is authorized to define the universe, query, filters, interval and version?
2. Enumeration coverage: were all pages/cursors/sequences/partitions in that declared scope traversed, with terminal closure and no gaps?
3. Acquisition coverage: was every enumerated item actually acquired from the same snapshot/session?
4. Processing coverage: was every acquired item processed, or explicitly retained as partial, unsupported, failed or unresolved?
5. Issuer/source participation: is there independent evidence that the source submitted every item it created or observed?
6. Temporal coverage: do retention, clock, expiry, supersession and interval gaps leave holes?

These do not collapse into a completeness boolean. CompleteWithinDeclaredQuery is weaker than complete-for-population; CompleteWithinRegisteredTransparencyView is weaker than complete-issuer-history. CompleteWithinCapturedArtifact is a byte/artifact claim and says nothing about the universe of artifacts outside it.

### Composition boundary

Reuse the qualified-expectation authority principle in #453 as a precedent, #10629 for claim-specific completeness obligations, #6283 for sampling-frame/measurement validity, #10410 for common-mode lineage, and #10630 for selective revalidation after population or source changes. Do not copy a telemetry schema into REA or create a new generic coverage graph.

Unknown population definition, omitted pages, untrusted terminal markers, failed acquisitions, unprocessed items, selective issuer submission, or undisclosed source filters must remain explicit. Absence from a partial view cannot be promoted to a negative conclusion.

### Required no-upgrade properties

- A provider's all_pages_seen flag cannot establish enumeration completeness without an independently bound query/snapshot and cursor/sequence closure.
- A complete transparency-log view cannot establish complete issuer participation absent an independent participation guarantee or audit.
- Adding more providers that expose the same incomplete upstream collection cannot create independent completeness.
- A self-reported population total cannot authorize negative inference.
- Missing, unsupported, failed and partial records remain in the denominator; do not silently drop them.
- An unrelated population change must not invalidate claims whose declared scope and dependencies are unaffected.

### Benchmark additions

Measure false-completeness rate by dimension, omission-detection recall by omission family, false-incompleteness rate, population-definition authority accuracy, pagination/gap detection, acquisition/processing accounting, issuer-participation uncertainty, selective invalidation precision/recall, and latency/cost by population and page count. Require zero false CompleteWithin outcomes on seeded missing-page, omitted-item, selective-publication and incomplete-processing controls; publish denominators and statistical upper bounds.

Claim ceiling: even a perfectly replayed transparency ledger only establishes its registered view under the chosen log policy. It does not prove the issuer registered every statement it created, nor that a complete declared query equals the real-world population.
