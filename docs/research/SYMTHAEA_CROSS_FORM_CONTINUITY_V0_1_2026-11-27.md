# Symthaea Cross-Form Semantic Continuity v0.1

**Date:** 2026-11-27  
**Status:** renderer-independent transition planner and regression fixtures authored; Bevy integration pending

## 1. Research objective

A polyembodied agent should be able to change its visual form without changing the meaning or authority of the underlying event. A constellation can become a puppet or mandala, but semantic identity, event lineage, privacy class and current consent bounds must remain intact.

This differs from ordinary video identity consistency. A video avatar must keep one character visually coherent across chunks. Symthaea must preserve semantic identity across different visual grammars while permitting the user to switch form, reduce motion, lower intensity or revoke permission.

## 2. Transition kinds

- **Initial** — no previous frame exists.
- **NoChange** — same event, content digest, form, authority, intensity and motion setting.
- **HardCut** — authority/privacy/intensity tightens, source permission expires, either scene is static/zero intensity, or another safety boundary requires immediate replacement.
- **SameFormUpdate** — same visual form, new semantic cue, with compatible live policy.
- **CrossFormSameCue** — same exact canonical-cue digest, new form.
- **CrossFormNewCue** — a new event/content digest and new form under compatible policy.

Same-cue morphing is proven by digest equality, not by similarity of labels or model claims. A different event can be animated as a new cue, but it is never described as the same cue.

## 3. Continuity descriptor

The renderer-independent descriptor stores only:
- session and clock-domain ID;
- event ID and contiguous event sequence;
- grant ID/version;
- canonical-cue digest;
- form and motion mode;
- privacy class;
- effective authorized intensity;
- source observation time;
- effective-permit expiry.

It contains no raw evidence-reference identifiers, source semantic targets or media. It is derived from an immutable scene frame.

## 4. Permission and safety rules

The planner is a pure transition planner, not an authorization system. A returned cross-fade plan is valid only while the host independently rechecks both the source and target permits against the live policy on each renderer update. A descriptor is not a permit and is not proof of current authorization.

Hard-cut conditions:
- source expired;
- grant identity/version changes;
- target effective permit expires earlier than the source's;
- authorized intensity is reduced;
- privacy class changes;
- source or target is static/reduced-motion;
- either effective intensity is zero.

Cross-fade duration is clamped to the configured same-form or cross-form duration, policy maximum, remaining source effective-permit lifetime and remaining target effective-permit lifetime.

Default proposals are 90 ms for same-form updates, 180 ms for cross-form changes and 250 ms maximum. These are design defaults, not human-perception findings or measured device timings. If a source expires during a transition, its opacity is immediately forced to zero and the target may take over only if its own permit is still live. At target expiry, the plan returns an error and the renderer must clear output.

Privacy-class change is always a hard cut. Do not interpolate from public to local-sensitive, restricted, shared or remote presentation. A user privacy downgrade wins over visual smoothness.

A user pause, cancel, revocation, missing clock, missing policy or changed opaque permit must override the transition plan and clear/replace output. The host must recheck policy for both frames; the planner itself cannot observe asynchronous revocation.

## 5. Implementation

Implementation: crates/domains/symthaea-soma/src/embodiment_transition.rs

The planner returns source and target opacity weights at host-supplied monotonic time. It never subtracts timestamps from different clock domains.

The planner is not yet wired into the Bevy renderer. The current Bevy adapter still replaces a scene when a new queue revision is accepted. The integration work is tracked separately in issue #13965.

## 6. Regression coverage

Eighteen Rust tests are authored for:
- initial frame, no-change and same-form updates;
- same-cue cross-form morph versus new-cue cross-form change;
- exact cue digest continuity and same-event payload mismatch;
- session and clock mismatch;
- sequence progression and observation-time regression;
- grant change hard cut;
- numeric expiry shortening/hard cut;
- privacy-class change hard cut;
- reduced intensity, zero intensity and static motion hard cuts;
- duration bounded by both frame lifetimes;
- source expiry during transition;
- target expiry and clock regression.

Tests are authored, not compiled or executed. The planner has no Bevy runtime, GPU, animation or user-study result.

## 7. Proposed benchmark track

Evaluate the transition planner separately from image/video generation:
- 100% exact canonical-cue digest preservation for same-cue cross-form transitions;
- 100% correct hard cut for grant/version, privacy-class, reduced-intensity, reduced-motion and expiry tightening;
- zero source opacity after source effective expiry;
- zero target output at/after target effective expiry;
- 100% transition duration within both source and target permit lifetimes;
- zero transition execution after policy recheck fails;
- <= 90 ms same-form update and <= 180 ms cross-form transition on the frozen local renderer profile (design target, not a perceptual threshold);
- >= 95% human-rated semantic interpretation agreement across forms on held-out events;
- report p50/p95/p99 transition timing, stale-frame exposure, frame drops, CPU/GPU/energy, and failure/revocation latency.

User study dimensions must include comprehensibility, distraction, cognitive load, accessibility, privacy perception and ability to interrupt. Do not optimize for anthropomorphic attachment.

## 8. Next integration

1. Wire SceneContinuityDescriptor::from_frame and plan_transition into an isolated Bevy scene manager.
2. Require a live source permit and target permit; recheck both on every frame.
3. Keep source and target entities separately tagged; apply planner opacity weights only after policy checks.
4. On revocation, missing clock, privacy change, intensity downgrade, pause or cancel, remove source immediately; do not wait for a fade.
5. Add screenshot/state replay, frame-time and expiry fault-injection tests.
6. Run a consented comprehension study before making claims about better continuity or user experience.
