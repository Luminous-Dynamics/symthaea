# MACE retrieval observation semantics

**Status:** additive correction to the research LongMemEval-V2 sidecar. This is not an independent qualification service, a decision-sufficiency evaluator, or an external benchmark result.

## The semantic boundary

A non-empty retrieval result proves only that the adapter returned candidates. It does not prove that the candidates are admissible, current, authoritative, useful to the reader, sufficient for the DecisionNeed, or causally beneficial to the downstream decision.

The state progression must remain explicit:

```
CandidatesReturned
      |
      v
Independent qualification
  Admissible / Rejected / Indeterminate
      |
      v
Decision-sufficiency assessment
  Sufficient / Insufficient / Unknown
      |
      v
Decision + independently observed outcome
```

Each stage requires a separately bound receipt. A candidate backend must not issue its own qualification receipt, and a qualification receipt must not be mistaken for an action authorization or an outcome receipt.

## Changes in this increment

- New retrievals with one or more candidate evidence records are labelled `candidates_returned`, not `evidence_sufficient`.
- The legacy `evidence_sufficient` spelling remains decodable for replay. It retains the same v1 digest tag, but its meaning is narrowed to the historical observation that candidates were returned. It is not a decision-sufficiency assertion.
- Three or more candidate-return observations can be consolidated into a `RetrievalStrategyNoteV1` with `Exploratory` status.
- `applicable_notes()` exposes only `PolicyEligible` notes. This increment has no API that can mint that status, so exploratory notes cannot silently become retrieval policy.
- `exploratory_notes()` exposes diagnostic notes separately.
- The strategy-note digest domain is advanced to v2 because status now participates in the commitment.
- Rust and Python share a fixed candidate-return digest vector.

## Still required before policy learning

1. A separately operated evidence qualifier with externally provisioned trust roots, exact use binding, transitive provenance closure, rights, and current-at-use lifecycle checks.
2. A decision-sufficiency receipt bound to the exact qualified evidence, DecisionNeed, fixed policy, and evaluator profile.
3. A sealed/independently operated evaluator that measures paired counterfactual outcomes on held-out task clusters.
4. A policy-eligibility transition that verifies the held-out qualification receipt, prespecified uplift, family guardrails, and cost accounting.
5. A versioned end-to-end protocol migration that preserves old trace replay without assigning historical v1 labels stronger semantics.

Until those exist, a retrieval hit is an acquisition observation only. It must not be counted as an improvement, sufficiency, or learned policy success.

## Claim boundary

This code change has authored Rust and Python regression tests, including a cross-language digest vector and legacy replay coverage. The Rust toolchain is unavailable in the current execution environment; those Rust tests have not been compiled or executed here. No GitHub Actions pass, LongMemEval-V2 result, security qualification, or SOTA improvement is claimed.
