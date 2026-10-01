# GIS Action Provenance Contract

## Purpose

Extend the epistemic provenance chain from:

Evidence → Frame → Conclusion

to:

Evidence → Frame → Conclusion → Action.

A frame revision is not a declaration that a conclusion was false and is not, by itself, a declaration that a previously executed action was wrong. It creates a provenance obligation: future actions whose epistemic prerequisites are affected must be re-evaluated according to their risk and authorization policy.

## Contract

For an action A:

- A declares the conclusions it depends on.
- Each dependency declares why it matters: inference, causal, ontology, evidence, or assumption.
- A declares a risk tier.
- Historical authorization records the frame, conclusions, evidence, policy, and decision that existed when the action was authorized.
- A later frame revision may create a re-evaluation witness without rewriting that historical record.

The core invariant is:

> Historical authorization is evidence of what authorized an action then; it is not automatically current authorization for executing the action again now.

## Revision propagation

The deterministic path is:

    Frame F1
      ↓ revision
    Frame F2
      ↓ impact mask
    affected conclusions
      ↓ typed action dependencies
    affected actions
      ↓ risk policy
    Ready | RequiresReevaluation | Deferred

Typed dependency reasons prevent graph proximity from being treated as semantic dependence.

An ontology-only revision should affect ontology-dependent actions without automatically gating an unrelated causal action. A causal-model revision should affect causal interventions. An evidence-boundary revision should affect actions whose support depends on that boundary.

The implementation now exposes a direct bridge that first asks the conclusion graph to reopen affected claims, then applies typed action dependencies and risk policy. Typed conclusion edges propagate only when their dependency kind is affected by the revision. Legacy untyped conclusion dependencies remain conservative because their semantic basis is unavailable. The action record also exposes a fail-closed prerequisite check for High/Critical actions. The caller must supply the authoritative current conclusion-ID set; action-provided identifiers are not treated as proof that a prerequisite exists. This check is a separate policy operation and must be invoked by the eventual execution boundary.

## Historical execution

Executed actions are immutable historical facts. A later revision may establish that a future repetition requires fresh authorization, but it must not rewrite:

- the frame that existed at execution time;
- the evidence available at execution time;
- the conclusion state used at execution time;
- the authorization policy;
- the recorded decision.

This keeps correction compatible with auditability.

## Risk policy

Informational and low-risk actions may remain executable under policy when the changed provenance does not require current support.

High-risk and critical actions should require current epistemic support when an affected prerequisite changes. Missing or stale provenance should fail closed for those actions.

This is intentionally a policy boundary rather than a confidence threshold.

## Re-evaluation witness

A gated action should retain an explicit witness containing:

- prior frame;
- revised frame;
- affected conclusion IDs;
- typed dependency reasons;
- the policy reason for re-evaluation.

This makes the gate explainable without trusting a free-form model rationale as authorization.

## Adversarial requirements

The current module includes focused tests for high-risk gating, ontology-only typed propagation through the conclusion/action bridge, and missing-prerequisite deferral. These tests are source-level additions; they have not yet been run in a repository build. The full conformance suite should deterministically test:

1. ontology-only revision gates ontology-dependent high-risk actions;
2. causal revision gates causal interventions;
3. evidence-boundary revision gates evidence-dependent actions;
4. unrelated actions remain unaffected;
5. superseded conclusions never resurrect old action authorization;
6. missing/stale provenance fails closed for high-risk actions;
7. executed history remains immutable;
8. cycles terminate deterministically;
9. re-evaluation is distinct from falsification;
10. historical authorization is distinct from current authorization.

## Research alignment

Recent work independently converges on several of these separations. Typed provenance research distinguishes stored material from supported claims and uses protected decision witnesses. Recent agent-authorization work separates per-action authorization from standing capability and mutable runtime state. Work on tool-using agents likewise separates action induction from execution authorization and warns against promoting historical context into current authority.

GIS should therefore treat action provenance as a dependency-bearing temporal contract, not another scalar confidence score.
