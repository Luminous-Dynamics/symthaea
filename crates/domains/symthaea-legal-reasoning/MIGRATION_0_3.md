# Migrating to 0.3

Version 0.3 is additive. The legacy `Rule`, `try_derive`, deontic string API,
and basic Hohfeld enum remain available.

## Move formal propositions to explicit literals

Prefer `Literal::Positive(Atom)` and `Literal::Negative(Atom)` over encoding
negation in names such as `"not_liable"`.

## Give every rule a stable identity

Create `FormalRule` values with `RuleId`, choose `RuleKind`, and assemble them in
a `RulePack`. Bind source provisions before approving a production pack.

## Declare priority instead of relying on ordering

Convert specificity, authority, temporal, or procedural decisions into explicit
`Superiority` edges with a recorded `PriorityBasis`. Cycles and unknown rule
references fail construction.

## Treat direct resolution as four-valued

Do not convert `LegalStatus` to a Boolean without an explicit policy:

- `Supported` means undefeated support only for the query;
- `Refuted` means undefeated support only for its opposite;
- `Both` means unresolved support survives on both sides;
- `Undetermined` means neither side is established.

## Preserve legal-time dimensions

Use `TemporalDimensions` when decision time and event applicability can differ.
Do not select a revision by vector position when `unique_governing_revision`
returns an overlap.

## Record lifecycle events explicitly

Use `ActionEvent` and `WaiverEvent` only after upstream evidence and legal
classification have produced formal events. The lifecycle engine replays those
records; it does not authenticate them.

## Bind results externally

Wrap important results in `EvidenceEnvelope` and hash the returned canonical
bytes with the cryptographic profile selected by the surrounding system.

Lifecycle event IDs provide identity, not hidden same-day ordering. Supply finer-grained formal time upstream when legal sequence matters.
