# MACE-CoS Guard v0.1

**Status:** research reference; not production enforcement, security certification, or an external benchmark result.

## Research question

Can a validator reject information-flow, scope/purpose/audience, target-binding and action-authority failures that occur only after individually admitted agent skills are composed?

MACE-CoS moves from checking each skill in isolation to checking the complete ordered composition, every derived artifact and each final sink.

## Current research context

- **CoSA / CoSA-Bench** (October 2026) reports 288 tasks across recruiting, finance, IT and business. The authors report that skill-pool additions capped at 5% increase violations from 35.02% to 58.45%; their harness guard reduces violations from 65.28% to 6.67% on a reported OpenClaw recruiting slice, with benign completion moving from 66.67% to 65.00%.
- **SkillFusion** studies offline reconciliation of conflicting, conditional and redundant directives.
- **SkillAlign** tests different exposure interfaces—full instructions, hints, summaries, workflows and no exposure—and finds that exposure affects both task utility and context cost.
- **SkillShift-Bench** evaluates skill executability under contextual, perceptual and execution shifts.

Those results belong to their own papers and protocols. They are not MACE-CoS measurements.

## Prototype invariants

- A four-level classification lattice is monotone across derived artifacts.
- Scope, purpose, audience and target remain bound across handoffs.
- Expired, revoked, stale, out-of-scope and over-budget inputs fail closed.
- Skills may require capabilities but cannot mint capabilities.
- Selected-skill conflicts are checked composition-wide.
- External actions require a separate structural authorization binding to principal, scope, action, target, argument digest, capability, expiry and nonce.
- Plan, request context, registry and source manifest have deterministic content digests.

## Run

Requires Go 1.23+ and uses only the standard library:

```sh
go test ./...
go run . < example-request.json
```

Exit 0 means the structured request passed this reference contract; exit 1 means policy rejection; exit 2 means malformed input.

## Important limits

This does not interpret arbitrary skill prose, authenticate issuer/source identity, verify signatures, establish that an upstream verifier-status field is truthful, execute actions, inspect model hidden-state or side-channel information flows, or reproduce CoSA-Bench. A digest establishes content identity, not provenance or authority. A production design needs a trusted broker that revalidates policy and authorization immediately before every side effect.

See parent issue [#9516](https://github.com/Luminous-Dynamics/symthaea/issues/9516), composition work [#11557](https://github.com/Luminous-Dynamics/symthaea/issues/11557), and the benchmark matrix documented there.
