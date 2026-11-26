# MACE Independent Decision-Sufficiency Evaluator v1

**Status:** Service design and qualification contract. This document is a specification, not a claim that the independent evaluator, lineage resolver, calibration pipeline, or production transport has been implemented.

## 1. Role and non-goals

The evaluator answers one narrow question:

> Given this exact independently qualified evidence set and this consumer-owned DecisionNeed, is the evidence sufficient to support the declared decision within the declared risk, regret, freshness, and resource limits?

It is not a retrieval backend, evidence authority, policy author, action authorizer, executor, or outcome observer. It must not rewrite the consumer's DecisionNeed, increase its risk tolerance, widen scope, substitute an action set, or turn an abstention into permission.

The runtime chain is:

`acquire \rightarrow qualify \rightarrow resolve\ lineage \rightarrow evaluate\ sufficiency
\rightarrow authorize \rightarrow execute \rightarrow observe\ outcome`

Each arrow crosses an explicit contract. A positive result in one stage does not imply a positive result in the next.

## 2. Separation of duties

The production evaluator must be separately operated from:
- the candidate-memory store and retrieval backend;
- the evidence-qualification issuer;
- the action authorizer and executor;
- the outcome observer used for held-out regret evaluation.

At minimum it needs separate process identity, credentials, signing key, trust profile, storage, deployment/release path, audit stream, and rollback domain. Its trust profile must be provisioned out-of-band by the consumer and protected by a monotonic version floor or external witness. Merely using a different verifier label or key ID is not organizational independence.

The evaluator must not possess action-execution credentials. It may return `sufficient`, `insufficient`, `indeterminate`, or `abstained`; only the consumer's separate authorization path can make an action executable.

## 3. Request contract

The consumer sends:
1. The exact DSSE/in-toto evidence-qualification attestation and qualified evidence projection.
2. The DecisionNeed document loaded by the consumer from its authorized policy registry.
3. The exact decision/use binding: run, invocation, domain, subject scope, purpose, query, environment, tool profile, candidate-set digest, qualified-set digest, qualification receipt, lifecycle snapshot, and trusted current time.
4. Current rights, revocation, correction, and erasure observations or verifiable references, where the domain requires them.

The evaluator independently recomputes the DecisionNeed digest and verifies that its complete contract is immutable for this request. It does not accept thresholds from the candidate records or from an evaluator-authored replacement manifest.

A request is rejected or returned as indeterminate when required payloads, lineage, lifecycle state, calibration, or measurement dependencies cannot be resolved. Missing data is not interpreted as proof of independence or safety.

## 4. Independent source-lineage and root-family resolution

The evaluator must resolve each candidate's exact payload and provenance from immutable, content-addressed sources. It must verify payload digests before using content.

For each evidence item it traces:
- stable source identity and source-content digest;
- parent evidence identifiers;
- derivation/transformation receipts;
- upstream source and transformation receipts;
- correction, revocation, deletion, and erasure state;
- shared origin, publisher, upstream dataset, extraction pipeline, model, or other common-cause relationships relevant to the evidence class.

The resolver builds a directed provenance graph and rejects cycles. It must not treat a missing parent, unknown ancestor, unavailable receipt, or unresolvable source identity as an independent root. Unknown ancestry contributes **zero** to the independent-root count.

A root family is a connected source-dependency/common-cause family for the DecisionNeed's declared evidence class. Two documents, URLs, chunks, mirrors, vendors, or model summaries are not independent merely because their IDs differ. If they trace to a shared upstream origin or shared causal dependency, they collapse into one family for that requirement.

For each requirement, the evaluator returns:
- `evidence_class_sha256`, exactly equal to the consumer DecisionNeed requirement;
- supporting `basis_receipt_sha256` values;
- a sorted list of unique `root_family_sha256` identities;
- one `provenance_receipt_sha256` per root family, identifying the immutable lineage/common-cause resolution receipt.

The `provenance_receipt_sha256` is SHA-256 over the exact UTF-8 bytes of the stored DSSE envelope JSON for the corresponding source-lineage receipt. It is a content address, not a statement that the referenced receipt has been fetched or independently verified by the consumer. The resolver must preserve the original bytes and make them retrievable for audit/replay.

A requirement may be marked `met` only when its evidence class matches, its supporting basis is valid, and the number of distinct resolved root families meets or exceeds the consumer's `minimum_independent_root_families`. The consumer-side verifier checks the signed claim, exact evidence class, unique root-family identities, lineage-receipt commitments, and DecisionNeed-specific cardinality. A later independent provenance verifier must fetch and verify each lineage receipt and independently reproduce the graph/correlation result before deployment may treat the lineage claim as verified.

## 5. Sufficiency, uncertainty, and regret

The evaluator uses only the utility profile, action set, risk limit, confidence floor, regret tolerance, and sufficiency policy pinned by the DecisionNeed.

### Risk and calibration

Risk estimates must be calibrated against a held-out, domain-representative set with predeclared splits. Publish reliability diagrams, calibration error, Brier/log scores where applicable, one-sided confidence bounds, subgroup results, and out-of-distribution performance. A point estimate is not a risk upper bound.

The evaluator must return `indeterminate` or `abstained` when the calibration receipt is stale, the evaluated case is out of supported distribution, the sample size cannot support the declared bound, or a material subgroup has not been evaluated. It must not infer high confidence from the amount of retrieved text.

### Decision regret

Regret is evaluated in the exact utility units and action space declared by the DecisionNeed. The target is an upper confidence bound on regret, not a post-hoc narrative that the selected action was reasonable.

Use paired cases and counterfactual action outcomes where available. Separate:
- observed outcomes from counterfactual estimates;
- randomized/intervention evidence from observational estimates;
- measured ground truth from simulator-derived estimates;
- training/calibration data from held-out evaluation data.

Do not allow the chosen action or post-action information to leak into the pre-decision evidence set. If the counterfactual is not identifiable, report indeterminate rather than fabricating a regret bound. The evaluator's regret interval and estimator version must be bound by the sufficiency policy and calibration receipt.

## 6. Independent resource measurement

Cost, latency, context tokens, tool calls, and maximum evidence age must be measured for the exact request and qualified evidence set. Measurements should originate from an independent gateway, runtime meter, or signed execution telemetry—not from a self-reported field in the memory record.

The evaluator records raw measurement receipt digests and the aggregation procedure. Cost includes ingestion, retrieval, reranking, lineage resolution, evaluation, and retries; do not report only the final model call. Latency includes queueing and all external calls. Token accounting includes the context actually delivered to the decision model.

If a measurement is missing, inconsistent, or unverifiable, a sufficient result is forbidden.

## 7. Output and failure semantics

The evaluator signs the DSSE/in-toto DecisionSufficiency statement consumed by the MACE v1 verifier. The signed predicate binds:
- exact candidate and qualified evidence set;
- qualification receipt and lifecycle snapshot;
- complete DecisionNeed digest;
- exact requirement/evidence-class/root-family claims;
- risk, confidence, calibration, regret, and resource observations;
- evaluator profile, key, validity interval, and receipt status.

Only `sufficient` can pass the consumer sufficiency gate. `insufficient`, `indeterminate`, and `abstained` are useful, valid outcomes—not protocol errors. Signature or binding failures are hard rejects.

The evaluator should return a stable reason code and retain the diagnostic trace. It must not silently drop an unavailable requirement or return a partial evidence set as sufficient.

## 8. Adversarial qualification suite

Before the service is considered deployable, its independent test suite must include:
- two mirrors of the same upstream source presented as different vendors;
- copied documents and chunked records presented as multiple roots;
- a transformed summary with an omitted parent edge;
- cyclic derivation graphs and conflicting parent receipts;
- unknown, unavailable, revoked, corrected, or erased ancestors;
- a common upstream dataset hidden behind separate publishers;
- source identities that collide while payloads differ, and identical payloads with distinct verified origins;
- a provenance receipt whose bytes change after its digest is computed;
- poisoned calibration, post-selection, and test-set leakage;
- overconfident predictions under distribution shift;
- manipulated clocks, stale revocation state, and profile rollback;
- resource telemetry that omits retries, queueing, or delivered context;
- correlated evaluator and qualifier keys or shared operator credentials;
- replay of a valid receipt against a changed DecisionNeed, scope, action set, utility profile, or qualified evidence set.

Every adversarial case needs a paired negative control that proves the detector does not reject the safe case indiscriminately.

## 9. Operational independence evidence

A production qualification package must include:
- separate workload identities and access-policy exports;
- key-generation and custody evidence, rotation/revocation drills, and external profile-version witness;
- deployment and storage separation records;
- immutable audit log checkpoints;
- source resolver implementation digest and dependency inventory;
- lineage-receipt fixture corpus with expected family partitions;
- calibration dataset provenance, frozen split manifests, and evaluator version;
- independently collected resource telemetry;
- held-out paired-regret report and uncertainty analysis;
- outage, rollback, replay, erasure, and recovery drills.

A signed evaluator receipt is not proof that this separation exists. The separation package is itself evidence that must be independently reviewed.

## 10. Evaluation gates

The evaluator is not ready for consequential use until it demonstrates:
1. **Lineage exactness:** 100% of known shared-root/common-cause adversarial fixtures collapse to the expected single family; unknown ancestry never increases family count.
2. **Requirement soundness:** zero false-`met` outcomes in the frozen adversarial corpus.
3. **Calibration:** pre-registered one-sided risk bounds achieve their stated coverage on held-out data, including worst-supported subgroup and drift slices.
4. **Regret:** the upper bound has demonstrated coverage on held-out paired outcomes; cases without identifiable counterfactuals abstain.
5. **Measurement:** independent telemetry reconciles to raw gateway/runtime records within a predeclared tolerance.
6. **Lifecycle:** revoked, corrected, deleted, and erased evidence cannot support a current sufficient verdict.
7. **Economics:** report the regret/utility versus all-in cost/latency/token Pareto frontier, not accuracy alone.
8. **Reproducibility:** a second operator can reproduce the verdict from immutable request, evidence, resolver, calibration, and measurement receipts.

The primary safety metric is the **false-sufficient rate**: the fraction of cases where the system emits sufficient despite at least one unmet DecisionNeed requirement or a violated declared risk/regret/resource limit. Its target is zero in the exhaustive protocol conformance and adversarial fixture suite; empirical field estimates must include confidence bounds and clearly state their sample size.

## 11. Explicit non-claims

This document specifies the production evaluator. It does not mean the service, resolver, lineage receipt verifier, calibration pipeline, telemetry meter, or action authorizer exists. The consumer-side verifier currently checks the sufficiency authority's signed claims and the structural root-family/evidence-class contract. Until the source-lineage receipts are independently fetched, verified, and replayed, source-family independence remains an evaluator attestation rather than an end-to-end verified fact.
