# MACE Decision Sufficiency Runtime v1

**Status:** Consumer-side verification and use-binding contract. This module verifies a separately issued sufficiency attestation. It does not implement the evaluator, prove source truth, authorize actions, or observe outcomes.

## 1. Why this is a separate stage

MACE now distinguishes the following statements:

1. **Acquisition:** the retrieval backend reports which candidate records and operations it produced.
2. **Qualification:** an independently governed qualifier determines which records are admissible for a stated scope and purpose.
3. **Decision sufficiency:** a separately governed evaluator determines whether the exact qualified set is enough for one declared DecisionNeed under explicit risk, regret, calibration and resource constraints.
4. **Action authorization:** a live authorization service separately permits a particular action.
5. **Outcome:** an independent observation records what actually happened.

A non-empty qualified evidence batch is not a sufficiency result. A sufficient result is not an action permit. An observed outcome is not proof that the prior decision was justified.

## 2. Consumer-owned DecisionNeed

DecisionNeed v1 is a typed manifest loaded from an authorized consumer policy registry. It must not be supplied by candidate memory or chosen by the qualifier/evaluator response.

The manifest binds:
- decision identity and domain;
- subject/effective scope and purpose;
- allowed action-set commitment;
- utility-profile commitment;
- risk limit and minimum confidence level;
- regret tolerance in micro-utility units;
- sufficiency policy and calibration profile/receipt digests;
- maximum evidence age;
- context-token, tool-call, monetary-cost and latency budgets;
- sorted requirement IDs, evidence-class commitments, minimum independent-root-family counts and per-requirement maximum evidence ages.

Its domain-separated digest commits to all of those fields. The consumer passes the decoded manifest into the sufficiency verifier; duplicated request values must exactly match it. A changed action set, utility model, calibration receipt, risk threshold, required evidence family or resource budget creates a different DecisionNeed digest.

The runtime parser rejects duplicate JSON keys, unknown fields, malformed digests, noncanonical requirement ordering, zero independent-root requirements, and requirement age limits that exceed the manifest-wide maximum.

## 3. Independent evaluator attestation

The evaluator returns a DSSE v1.0.2 envelope carrying an in-toto Statement v1. The Statement subject names and hashes the exact *qualified evidence set*, not the original candidate set. Its closed MACE predicate binds:
- run, invocation, domain, subject scope, environment and tool profile;
- query, original candidate set, exact qualified-evidence set and original qualification receipt;
- DecisionNeed digest and purpose;
- sufficiency policy, calibration profile and calibration receipt;
- status and every required requirement disposition, exact evidence-class digest and supporting receipt digests;
- for each claimed-met requirement, sorted independent-root-family digests paired with transitive-lineage/common-cause provenance-receipt digests;
- risk upper bound, risk limit and confidence;
- regret upper bound and declared regret tolerance;
- observed cost, latency, context tokens, tool calls and maximum evidence age;
- evaluator trust-profile digest, signer key ID, issue time and expiry.

The verifier requires the evaluator role identifier to be distinct from the qualification role, and rejects a shared trust-profile digest or signer key ID. The evaluator trust profile must be freshly provisioned and pinned outside candidate memory and its rollback domain. A cached old evaluator verifier can miss revocation and is not a current-at-use guarantee.

A requirement marked **met** must name the exact evidence-class digest in the DecisionNeed, include supporting basis receipts, and carry at least the DecisionNeed minimum number of distinct root-family commitments. Each root-family commitment is paired with a provenance receipt that identifies the evaluator's transitive lineage and common-cause analysis. The consumer rejects wrong evidence classes, duplicate/noncanonical family identities, missing lineage commitments, and an insufficient family count.

The evaluator must resolve the provenance receipts independently: trace derivation edges to source roots, collapse shared roots and common-cause dependencies, reject unknown ancestry as independent, and compute root families from source lineage rather than document count, chunk count, vendor count, or copied URLs. The signed digests make the claim auditable and replayable; they do not make the provenance computation true by themselves. The consumer-side verifier currently checks the committed digests and family cardinality but does not independently fetch/replay each lineage receipt or recompute the source graph.

The evaluator attestation is a signed claim from a separate authority. This verifier does not implement the evaluator's risk model, reproduce its calibration data, independently measure the resource figures, or mathematically prove its regret bound. Those are requirements on the evaluator service and independent measurement pipeline.

## 4. Conditions for a sufficient verdict

The verifier accepts a sufficient status only if:
- all required requirement IDs exactly equal the consumer-owned DecisionNeed set;
- every requirement is marked met;
- every requirement has at least one basis receipt digest;
- the risk upper bound is present and no greater than the manifest risk limit;
- confidence is at least the manifest confidence floor;
- the calibration profile and calibration receipt exactly match the manifest;
- the regret upper bound is present and no greater than the manifest regret tolerance;
- observed cost, latency, context tokens, tool calls and maximum evidence age are present and within the corresponding manifest budgets;
- the evaluator signature is valid under the evaluator's separately pinned profile;
- the receipt is valid at the trusted current time and bound to the exact qualified evidence set and DecisionNeed.

Insufficient, indeterminate and abstained are valid evaluator outcomes when their receipt is correctly signed and bound. An unmet or unknown requirement may carry no basis receipts and no root-family commitments; a requirement marked met may not. They never become sufficient by having a non-empty evidence set. A sufficient status with missing measurements, wrong evidence class, too few independent roots or a breached limit is rejected.

## 5. Revalidation at every use

The consumer must revalidate both authorities together at each subsequent evidence read:
1. reload the currently pinned qualification verifier and revalidate the original qualification signature and full use binding;
2. reload the currently pinned evaluator verifier and revalidate the sufficiency signature and full DecisionNeed binding;
3. sample a trusted current clock after obtaining current authority state;
4. reject if either receipt expires, either signer/profile is revoked or rotated, the scope/purpose/DecisionNeed changes, or the qualified evidence-set digest changes.

The sufficiency verifier therefore cannot revive an expired or revoked qualification. It is not sufficient to cache a previously verified boolean or a stale verifier object.

## 6. What is not implemented

This increment implements the receipt and consumer verification boundary only. It does not implement:
- the independent decision-sufficiency evaluator or its key custody;
- the utility/risk model or calibration process;
- independent telemetry that measures cost, latency, tokens, tool calls and source age;
- proof of regret bounds from held-out outcomes;
- a monotonic external trust-profile witness or rollback-resistant clock across restart;
- live rights/currentness checks, action authorization or consume-once execution;
- decision/action execution receipts, independent world observations, outcome receipts or policy promotion;
- a production sidecar network call to a separately operated evaluator.

The existing qualification and sufficiency verifiers therefore establish that two distinct pinned authorities made correctly bound signed statements. They do not prove those authorities are organizationally independent or that their underlying judgments are true. Production deployment requires separate credentials, processes, storage, trust roots and independently observable source/measurement pipelines.

## 7. Research and benchmark connection

The DecisionNeed manifest makes the internal MACE Decision Memory Benchmark measurable. For each matched task, publish:
- decision regret and confidence intervals;
- all-in cost, context tokens, tool calls and latency;
- evidence-family independence and ablation influence;
- sufficiency calibration and abstention quality;
- worst-family and lifecycle/security regressions.

Compare the Pareto frontier of regret against full resource cost. The evaluator-only oracle remains a diagnostic upper bound and must never be exposed to the query planner. External AML Cycle 2, LongMemEval-V2, AMA-Bench, WorldMemArena, FAMA and TimlMemBench protocols remain separate and unchanged.

The long-term differentiator is not one more recall metric. It is evidence-efficient decision-making whose admissibility, sufficiency assumptions, regret tolerance, cost and current authority can be independently inspected.

## 8. Relevant MACE ownership

- #13455 owns the independent qualification boundary.
- #13284 owns decision-sufficiency frontiers and EVSI-guided acquisition.
- #13264 owns verified transitive evidence dependencies.
- #13141 owns paired access/intervention evaluation.
- #13319 owns adversarial memory lifecycle and integrity evaluation.
- #13517 owns sealed paired LongMemEval-V2 evaluation.

This runtime should be integrated with those owners, not treated as a replacement for their evaluators.
