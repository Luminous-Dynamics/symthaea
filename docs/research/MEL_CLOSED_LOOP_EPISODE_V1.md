# MEL-CLOSED-LOOP-001 — Forecast-to-observed-effect episode v1

**Status:** composition-envelope draft. The JSON Schema and three synthetic examples are authored and locally schema-validated. No Rust reducer, live adapter, hardware test, detector study or musician study is claimed by this first increment.

**Parent / integration:** [#13480](https://github.com/Luminous-Dynamics/symthaea/issues/13480). This work is stacked on [PR #13348](https://github.com/Luminous-Dynamics/symthaea/pull/13348) and composes [#13349](https://github.com/Luminous-Dynamics/symthaea/issues/13349), [#10382](https://github.com/Luminous-Dynamics/symthaea/issues/10382), [#13313](https://github.com/Luminous-Dynamics/symthaea/issues/13313), [#10190](https://github.com/Luminous-Dynamics/symthaea/issues/10190), [#10265](https://github.com/Luminous-Dynamics/symthaea/issues/10265), [#13254](https://github.com/Luminous-Dynamics/symthaea/issues/13254) and [#10206](https://github.com/Luminous-Dynamics/symthaea/issues/10206).

## Why this envelope exists

Existing owners qualify distinct facts: a forecast was committed before its predicted output; a musician issued a scoped disposition for that exact forecast revision; a turn-taking decision authorized or withheld a bounded action; an effect reached a particular application/output stage; an independent recorder captured and closed the target interval; and a separately qualified event observer reported present, absent, ambiguous or not estimable.

This v1 document is a **join envelope**, not a second implementation of any upstream ledger, capture adapter, detector, clock mapper or provenance service.

## Files

- `schemas/mel-closed-loop-episode-v1.schema.json` — closed Draft 2020-12 composition manifest.
- `fixtures/mel-closed-loop-episode-present.synthetic.json` — complete capture with a detector-reported event.
- `fixtures/mel-closed-loop-episode-absent.synthetic.json` — complete empty window after WAIT/ignore; no producer realization record is required (`effect: null`).
- `fixtures/mel-closed-loop-episode-unobserved.synthetic.json` — sequence gap, dropped frames and XRUN; observation is explicitly unobserved.

All hashes, clocks, identities, event records and receipt references in these fixtures are synthetic sentinels. They are not real artifacts, signatures or qualifications.

## Binding rule

The episode verifier must resolve every receipt to its owning verifier and exact source. A syntactically valid receipt reference is not proof that the referenced receipt exists, authenticates, is current or passed. Repeated join keys let an independent reducer compare identities rather than trust a producer summary.

Required joins:
1. Forecast ID/revision and immutable execution identity across forecast, disposition, decision and effect.
2. Disposition actor/scope and sequence against the authenticated authority receipt.
3. Decision sequence/action against the action-effect ledger.
4. Capture ID, payload digest and target window against the independently revalidated closure receipt.
5. Event-manifest capture ID and payload digest against the exact capture.
6. Detector build/config/scope against a current qualification receipt.
7. Producer, capture-recorder and event-observer principal/role separation.
8. Clock-domain identity and time-window coverage, retaining mapping uncertainty and expiry from owner receipts.

The top-level reported closure and detector result are **untrusted summaries**. The reducer must replay upstream closure and observer receipts and derive its own outcome.

## Outcome semantics

| Layer | Values | Meaning |
|---|---|---|
| Musician disposition | accept / steer / veto / branch / ignore / clarify / abstain | What the collaborator allowed, changed or withheld |
| Turn-taking decision | WAIT / RESPOND / CONTINUE / YIELD | What the live policy decided |
| Effect stage | not requested / authorized / queue applied / enqueued / callback observed / CPAL predicted / loopback observed / not applied / unknown | Highest independently supported execution stage |
| Detector observation | present / absent / ambiguous / not estimable / unobserved | What the detector reports for the exact capture and scope |

`present` and `absent` are eligible only when the full target window has a revalidated, complete capture closure and detector qualification is valid for the exact build/config/scope. An incomplete capture is `unobserved`, even if an empty manifest was emitted. A complete empty window supports **detector-reported absence**, not ground-truth absence; detector false negatives remain measured in a separate held-out qualification study.

A veto is an action-authority fact, not an acoustic observation. Already-admitted callback frames or device buffers cannot be claimed physically recalled. Queue invalidation is not a claim that hardware was silenced.

## Reducer requirements (next increment)

The Rust reducer should fail closed on ID/revision/sequence mismatches; stale, expired or superseded forecasts; RESPOND/CONTINUE without valid authorization; effects scheduled after a veto boundary; producer/capture/observer identity collapse; receipt or payload substitution; mismatched capture/window/clock identities; incomplete frame/sequence accounting, missing watermarks, gaps, drops, XRUNs, restarts, ring overflow or unknown clock mapping; present/absent without eligible closure and detector qualification; `absent` represented as ground truth; unknown execution treated as retry permission; and model/adapter/config/runtime drift across paired counterfactual arms.

The reducer must preserve failed, pending and unobserved attempts. A steer or branch cannot rewrite an earlier forecast.

## Adversarial controls

The fixtures provide the positive/negative shapes. The reducer suite must add digest substitution, one-frame truncation, undrained final buffer, sequence gap, ring overflow, XRUN, restart, stale clock, revoked detector qualification, deliberate detector miss, late veto at every output stage, stale forecast, branch cross-binding and same-label/different-model-identity controls.

The key detector negative control is **complete capture + deliberately injected event + detector-reported absent**. It exposes false-negative behavior; capture completeness cannot certify detector correctness.

## Authoring-time validation

- Draft 2020-12 schema meta-validation: passed with local `jsonschema` 4.26.0.
- Three positive synthetic fixtures (`present`, `absent`, `unobserved`): 3/3 passed.
- Five mutated negative shapes (ineligible capture, unqualified detector, missing event manifest, production identity without model digest, and steer without successor forecast): 5/5 rejected.
- The complete-empty fixture uses `effect: null`; no producer realization record is invented.

This is contract-authoring validation only. It is not a Rust compilation/test/Clippy run, cryptographic receipt verification, runtime test or hardware qualification.

## Scope and claim boundary

JSON Schema validates wire shape and encoded conditional constraints only. It does not check chronology, digest recomputation, receipt authenticity, identity separation, capture completeness, detector correctness, causal contingency or physical audio. Those belong to the reducer and existing upstream verifiers.

This increment does not claim live Melothaea performance, acoustic latency, detector sensitivity, musician preference, causal collaboration gains or SOTA superiority. Exact-head Rust qualification is required when the reducer is added; no Python production substitute is introduced.