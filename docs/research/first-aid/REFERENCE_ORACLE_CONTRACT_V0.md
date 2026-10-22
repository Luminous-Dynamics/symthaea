# Independent Reference Oracle Contract v0.1

**Status:** proposed software-process specification. It is not a medical protocol, clinical ground truth, clinical validation, or deployment authorization.

**Parent work:** [First-Aid workflow safety benchmark, PR #8151](https://github.com/Luminous-Dynamics/symthaea/pull/8151). Follow-up tracker: [issue #8367](https://github.com/Luminous-Dynamics/symthaea/issues/8367).

## 1. Why this specification exists

The production JSON trace checker and finite-state model are useful differential checks, but they are authored in one repository and maintained under one set of assumptions. Matching outputs cannot rule out a requirement error shared by both models. This file freezes an explicit semantic contract before a separately structured reference implementation is added.

The reference implementation must **not** import production checker types or call production transition functions. It should parse JSON independently, convert only after validation to its own minimal event vocabulary, and implement the rules below using a different state representation or transition structure. It must produce its own ordered violation list. Comparing a second program that copies the production function line-for-line is not useful independence.

Even this structural separation is not enough to claim independent ground truth. Unless a qualified reviewer independently derives or adjudicates the rules and expected outcomes, this remains same-team implementation-diversity evidence. The receipt must record that limitation.

## 2. Claim ceiling

The permitted claim is:

> The identified software revision conforms to this versioned synthetic workflow-process contract for the exact cases executed and reported.

Do **not** infer first-aid advice correctness, clinical safety, human-factors benefit, emergency-service integration, patient outcomes, regulatory compliance, or field readiness. Scenarios use software lifecycle events only; they must contain no patient records or unreviewed treatment instructions.

## 3. Input and validation contract

The input represents an ordered event trace. The trace-level initial connectivity applies until changed by an event. Object-member order is not semantic; event-array order is semantic and must never be canonicalized by sorting.

Reject a trace rather than score it when any of the following is true:

- The schema version, required root fields, enum values, or required event fields are unknown or invalid.
- Unknown root/event members are present under the strict v0 contract.
- Trace identity is blank or duplicated within a corpus, there are no events, or event sequence numbers are not strictly increasing.
- An event carries data that is not permitted for its event kind (for example, digest status on an event other than bundle verification, or connectivity on an event other than connectivity change).
- A required bundle identity is absent/blank, bundle state is missing on load/verification, the digest result is missing on verification, resource state/presentation is incomplete, sensitive-data/consent attributes are incomplete, or a connectivity change omits its new state.
- An event or trace is outside the declared contract rather than explicitly supported.

Unknown values must not silently become valid defaults. Parsing failure is distinct from a detected process violation: rejection is not an empty violation list and must be reported separately.

## 4. Abstract state

The independent oracle may use another representation, but it must preserve these semantic facts:

- Whether a session is active.
- The currently loaded bundle identity, if any.
- The identity of the bundle currently verified for presentation, if any.
- A monotonic set of revoked bundle identities.
- Whether uncertainty remains pending, and whether that pending uncertainty has been explicitly preserved for handoff.
- Current connectivity, updated only by ordered connectivity-change events.

Resource freshness/presentation and sensitive-data/consent events are event-local decisions in v0. The contract does not pretend to model clocks, real connectivity, cryptographic key validity, or the semantics of dynamically changing consent. Those are separate future contracts, not facts that this simplified model may assume.

## 5. Transition rules and required violation output

Process events in input order. Append each violation when its triggering event is encountered. Preserve duplicates: two separate offending events produce two occurrences. Do not sort or deduplicate the output. Within a single render event, use the explicit precedence below.

| Event | Required semantic rule |
|---|---|
| session_started | Mark the session active; clear loaded and verified bundle identity. Do not erase revocation history. If uncertainty is pending and not preserved, append uncertainty_lost_at_session_boundary; the uncertainty is not silently discarded by starting another session. |
| bundle_loaded(id, state) | Set loaded identity to id; clear prior verified identity. Loading alone grants no presentation authority. |
| bundle_verified(id, state, digest) | Success requires an active session, identity equal to currently loaded identity, state valid, digest exactly true, and identity not revoked. Otherwise clear verified identity and append invalid_bundle_verification. On success, set verified identity to id. |
| protocol_rendered(id, state?) | First append render_without_active_session if no session is active. Then, if id is revoked or this render event labels the state revoked, append render_after_revocation. Otherwise append render_without_verified_bundle unless the currently verified identity exactly equals id. The revoked branch suppresses only the generic missing-verification violation for this render; it does not suppress the inactive-session violation. |
| bundle_revoked(id) | Add id to the monotonic revoked set. If this identity is currently loaded, clear its verified identity immediately. No later session/load event can erase revocation history. |
| session_interrupted | Mark session inactive and clear verified identity. |
| session_resumed | Mark session active and clear verified identity even if no interruption event appears in the trace. Resume never restores previous verification. |
| uncertainty_detected | Mark uncertainty pending and clear its preserved flag; a newer uncertainty event requires fresh preservation. |
| uncertainty_preserved | Set preserved only if uncertainty is pending. Without pending uncertainty, the event creates no authorization or proof. |
| unknown_normalized | Always append unknown_normalized. |
| handoff_emitted | If uncertainty is pending and not preserved, append uncertainty_lost_at_handoff. Then consume the pending/preserved uncertainty state for this handoff. |
| resource_presented(state, presentation) | If state is stale or unknown and presentation is confirmed, append stale_resource_claimed_confirmed. An unconfirmed stale/unknown resource is a benign control for this rule. |
| connectivity_changed(state) | Replace current connectivity with the supplied event-time value. Do not retroactively change preceding events. |
| network_required_for_core | Append offline_core_network_dependency if current connectivity is offline; append intermittent_core_network_dependency if it is intermittent; append nothing if online. Only this explicitly network-dependent core action is covered by the v0 rule. |
| sensitive_data_egress(payload, consent) | Append sensitive_data_egress_without_consent exactly when payload is sensitive and consent scope is not true. Non-sensitive egress is not flagged by this invariant alone. |

### Violation alphabet v0

The oracle's only declared process violation codes are:

- invalid_bundle_verification
- render_without_active_session
- render_after_revocation
- render_without_verified_bundle
- unknown_normalized
- uncertainty_lost_at_handoff
- uncertainty_lost_at_session_boundary
- stale_resource_claimed_confirmed
- offline_core_network_dependency
- intermittent_core_network_dependency
- sensitive_data_egress_without_consent

Adding, removing, renaming, or changing precedence for a code requires a contract version review and explicit fixture/receipt migration; it is not a silent benchmark update.

## 6. Conformance test design

The initial independent implementation must compare its output, in order and with multiplicity, against the frozen reference expectation for the 12 existing trace fixtures, then against the declared mutation/control inventory. It must add focused cases for:

1. render outside a session while the bundle is revoked (both applicable violations and the specified order);
2. a valid bundle verified, then a different bundle loaded, then an attempted render of the old identity;
3. a revoked identity after a new session starts;
4. two uncertainty detections separated by a preservation event;
5. uncertainty detection, session boundary, and later handoff;
6. offline → online and online → intermittent changes with network-required events before/between/after;
7. repeated renders or network-required events to prove duplicate violation instances are retained;
8. stale/unknown resources paired with both confirmed and unconfirmed controls;
9. all four sensitivity/consent combinations;
10. malformed, unknown-field, missing-metadata, duplicate-sequence, and wrong-digest corpus inputs.

For each new case, explicitly state whether the expected result is **evaluate** with an ordered list or **reject** with a reason class. Do not encode malformed-input rejection as a no-violation success.

## 7. Blinded evaluation in a public repository

A committed public "holdout" is not secret from anyone who can read it. Calling a public corpus held out only because it has different IDs does not prevent benchmark tuning to its expected labels.

Use a **commit-then-reveal protocol** if a genuine blind test is needed:

1. A reviewer who is not tuning the candidate generates the evaluation traces and independently adjudicates expected outcomes.
2. Before the evaluation build is frozen, publish a commitment manifest containing SHA-256 digests of the exact raw corpus bytes, expected-outcome file, generator/version/config, and adjudication protocol. Do not publish the underlying held-back cases yet.
3. Keep the committed bytes with the independent evaluator or a restricted evaluation service; record custody, access, and any changes. No patient or identifiable health data are needed.
4. Freeze the candidate source SHA, executable/toolchain identity and evaluation protocol digest before the evaluator receives the candidate.
5. Run the blind suite once for the registered score. Preserve the signed/logged result and all counterexamples. After the result is locked, release the corpus and expected outcomes where licensing and governance permit.
6. After reveal, those cases become public regression tests and must no longer be described as unseen. Create a new committed holdout for a later evaluation.

If no independent custodian or restricted evaluation path is available, use the honest label **public held-out-by-generation split**, not **blind holdout**. It may still be useful for distributional tests but does not prevent adaptive benchmark overfitting.

## 8. Independence and adjudication record

For every reference-oracle release, record:

- specification version and exact raw-file digest;
- author(s), reviewer(s), and role separation;
- whether reviewers saw the production implementation or its current expected labels before adjudication;
- disagreements, resolution method, and unresolved cases;
- generation seed and generator/source digest for synthetic cases;
- corpus split, commitment publication time, reveal time, and custody record if blinded;
- parser/schema rejection totals separated from semantic violation results;
- exact candidate source/build/toolchain identity and all per-case outputs.

At least one reviewer should be independent of implementation tuning. If that is not achieved, the report must say so rather than infer independence from a different programming language or file.

## 9. What passing does—and does not—mean

A pass means exact conformance to the frozen synthetic process contract for the exact evaluation inputs and revision. It is an engineering verification result.

It is not evidence that the selected medical protocol is correct, that users understand it under stress, that actual emergency services are reachable, that a clinical intervention improves outcomes, or that deployment is legally authorized. Those claims require separate qualified clinical content review, human-factors research, privacy/governance review, and jurisdiction-specific regulatory assessment.
