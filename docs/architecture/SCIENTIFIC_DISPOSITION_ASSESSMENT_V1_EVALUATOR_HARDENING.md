# Scientific Disposition Assessment v1 — evaluator-closure hardening

**Status:** semantic hardening companion to `SCIENTIFIC_DISPOSITION_ASSESSMENT_V1.md`.

This note closes a second loophole before implementation: exact policy identity is insufficient if the runtime evaluator can consult hidden ambient state or if the executed implementation is not bound to the declared policy semantics.

## 1. Core theorem

```text
policy document identity
    != policy implementation identity
    != verified policy execution
    != disposition assessment
```

and:

```text
declared inputs
    + hidden ambient state
    -> not replay-closed
```

A future disposition must therefore be derived by an evaluator whose complete authority-relevant input surface is explicit and whose executed semantics are bound to the registered policy profile.

## 2. Closed evaluation surface

A deterministic disposition evaluator should consume only declared immutable inputs such as:

```text
ScientificEvidenceViewSnapshot
qualified eligibility receipts
argument/adjudication graph snapshot
dependency graph snapshot
target-compatibility/triangulation snapshot
falsifier outcomes
DispositionPolicyProfile
information cutoff / scientific-use profile
```

It should not silently consult:

```text
wall-clock time
network state
unbound database rows
current web search
mutable process-global configuration
hidden model state
unstated user preferences
randomness
```

If such information is scientifically required, it must first be converted into an explicit bound input artifact with provenance/currentness semantics.

## 3. Policy profile and evaluator implementation are distinct

A declarative policy may state rules such as:

```text
qualified unresolved undercutting defeater -> block support inference
triggered preregistered falsifier under profile F -> refuted within F scope
incomplete evidence view -> no full-support disposition
```

An evaluator implementation executes those rules.

The assessment should therefore retain both:

```text
disposition_policy_profile_id
evaluator_implementation_artifact_id
```

and, where required, an exact execution-capsule/receipt identity.

Same policy profile implemented by two evaluators is not assumed equivalent without qualification.

Same evaluator binary run under two policy profiles is not the same assessment lineage.

## 4. No opaque model-as-policy authority

A learned model may assist disposition analysis, but:

```text
LLM output
classifier probability
embedding similarity
HDC score
```

must not directly become `ScientificDispositionAssessment` authority.

If a learned component participates, the policy must explicitly define its role, admissible output contract, model artifact identity, training/evaluation provenance where relevant, uncertainty/abstention behavior, and whether deterministic replay is expected.

The learned output remains an input/evidence object unless a separately qualified policy grants it a bounded inferential role.

## 5. Predicate-level reason trace

A replayable evaluator should retain not only the final disposition and graph edges but the policy predicates that materially determined the result.

Conceptually:

```text
PolicyEvaluationTraceV1 {
    policy_profile_id,
    evaluator_implementation_id,
    input_snapshot_ids,
    evaluated_predicates,
    predicate_results,
    precedence/resolution steps,
    unresolved predicates,
    primary_disposition,
}
```

Examples of predicate results might include:

```text
EvidenceViewComplete = false
QualifiedSupportExists = true
QualifiedOppositionExists = true
ActiveUndercuttingDefeaterExists = false
TriggeredFalsifierExists = false
TargetCompatibilityEstablished = true
RequiredReplicationDiversityEstablished = false
```

The exact vocabulary remains policy/domain-owned.

The trace must preserve predicates that lose to higher-precedence rules; otherwise the primary disposition can hide important scientific context.

## 6. Precedence is policy, not implementation accident

If two rules conflict, their precedence must be declared by the immutable policy profile or yield an unresolved disposition.

The evaluator may not resolve conflict using:

```text
match-arm order
map iteration order
first row returned
latest insertion
thread scheduling
floating-point tie accident
```

unless that exact behavior is intentionally part of the qualified policy contract.

## 7. Unknown must remain explicit

A predicate may have more than Boolean truth values.

A useful generic pattern may include:

```text
Satisfied
NotSatisfied
Unknown
NotApplicable
BlockedByMissingQualification
```

The evaluator must not coerce `Unknown` into either positive or negative evidence unless the exact policy explicitly defines that conservative treatment.

This is especially important for incomplete dependency inventories, unavailable historical source snapshots, untestable causal assumptions, and unresolved lifecycle branches.

## 8. External stochastic or computational procedures

If a disposition policy intentionally invokes a stochastic or expensive external procedure, replay closure requires an explicit execution artifact.

Conceptually:

```text
policy input snapshot
+ implementation artifact
+ execution capsule
+ RNG seed/state if applicable
+ output artifact
    -> qualified evaluator result
```

The shared kernel should not pretend a stochastic evaluation is deterministic merely because the policy name is stable.

## 9. Currentness includes evaluator qualification

A disposition can become stale not only because evidence changes but because evaluator qualification changes.

A future currentness check may need to bind:

```text
policy profile generation
implementation qualification generation
execution environment/capsule identity
```

when those dimensions are authority-relevant.

A security defect or semantic bug discovered in evaluator v1 may invalidate current use of assessments produced by v1 without deleting their historical existence.

That invalidation belongs to external adjudication/current eligibility, consistent with #769.

## 10. Policy upgrades do not rewrite old assessments

Suppose:

```text
policy-v1 -> SupportedWithinScope
policy-v2 -> Contested
```

for the same evidence snapshot.

Both assessments remain historically valid records of what their exact policies derived.

The Atlas does not rewrite the v1 assessment into v2.

Instead:

```text
Assessment A1 --used--> policy-v1
Assessment A2 --used--> policy-v2
```

and current-view policy selection is a separate explicit choice.

## 11. Differential qualification

A future implementation should use differential tests where possible:

```text
same exact input snapshot
+ reference policy interpreter
+ optimized evaluator implementation
    -> identical predicate trace + disposition
```

for a qualification corpus.

This can detect semantic drift between policy specification and implementation without claiming formal equivalence for all inputs.

## 12. Adversarial qualification cases

A future focused qualification should prove at least:

1. evaluator cannot access wall-clock/current network state through the pure disposition path;
2. changing evaluator implementation identity creates a new assessment lineage;
3. same declared policy with deliberately divergent evaluator behavior is detected by parity qualification;
4. rule precedence is determined by policy semantics, not source-code/insertion order;
5. losing predicates remain present in the reason trace;
6. `Unknown` cannot silently become `Satisfied`;
7. a learned model output cannot mint disposition authority without explicit bounded policy semantics;
8. stochastic evaluation binds exact seed/state and execution receipt;
9. later evaluator invalidation changes current-use eligibility without deleting historical assessments;
10. policy-v1 and policy-v2 assessments remain distinct even when they share proposition/evidence inputs.

## 13. Important non-claims

This note does not choose a policy language, require every domain to use one evaluator implementation, define universal rule precedence, ban learned models from scientific reasoning, or establish formal verification of evaluator semantics.

It only ensures that the future Theory Atlas cannot claim replayable disposition reasoning while allowing an undeclared implementation or ambient state to decide what the policy means at runtime.
