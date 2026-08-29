# Symthaea Reality Ledger v1

## Purpose

Reality Ledger v1 exists to support nested simulation, dreams, replay, digital
worlds and physical embodiment without allowing provenance collapse.

The central invariant is:

> What Symthaea perceived, inferred, imagined, replayed, imported, created or
> physically sensed must remain distinguishable after storage and recall.

This is a provenance contract, not a metaphysical theory and not evidence of
subjective experience.

## Reality layers

- `PhysicalGrounded`: independently identified physical sensor/instrument plane.
- `DigitalCommitted`: authoritative persistent digital/simulated world.
- `Counterfactual`: hypothetical branch of a parent world.
- `Replay`: historical/reconstructed world content.
- `Dream`: internally generated imagination world.
- `Imported`: externally authored/imported world without grounding established here.
- `Unknown`: unresolved provenance; never silently upgraded.

A `DigitalCommitted` world is authoritative only inside its declared digital
lineage. It must not be described as physically grounded merely because it is
persistent.

## World lineage

Every derived world has an explicit parent reference and generation depth.
Nested worlds are entered through `RealityContextStack`. A child must name the
currently inhabited world as its parent, use the next generation depth, remain
below a configured maximum nesting depth, and may not cycle back into an active
world identity.

This makes nested simulations structurally closer to:

```
physical / committed parent
        |
        +-- counterfactual A
        |       |
        |       +-- counterfactual A.1
        |
        +-- replay R
        |
        +-- dream D
```

rather than a flat memory pool where all events are indistinguishable.

## Append-only event provenance

`RealityLedger` is a BLAKE3-linked append-only record sequence. Each record
binds:

- record identity and sequence;
- full world descriptor;
- record kind;
- evidence source;
- optional host revision and frame/tick;
- evidence/artifact digest;
- previous-record digest.

The raw payload is intentionally external. This keeps the ledger host-neutral
while cryptographically binding the evidence artifact selected by the host.

An empty ledger is not verified evidence.

## Memory admission

Memory admission is provenance-preserving rather than reward based.

A direct physical sensor record may enter `PhysicalWorldBound` memory, but that
label does not make the sensor infallible. A derived computation over a physical
or committed digital world is marked `CommittedWorldDerived`, not direct
observation.

Counterfactual and dream records become `HypotheticalOnly`. They may be recalled
as imagination, but must not claim that the event occurred in the parent/current
world. Replays remain `ReplayOnly`; imports remain `ImportedUnverified`; unknown
provenance remains `UnknownHold`.

## Counterfactual materialization

A counterfactual is never relabeled as committed reality.

`CounterfactualCommitReceipt` instead records that an externally authorized
mutation of the parent `DigitalCommitted` world reproduced the selected
counterfactual state. It requires:

- source layer exactly `Counterfactual`;
- target layer exactly `DigitalCommitted`;
- source parent exactly the target world/lineage;
- non-empty before/source/after state digests;
- non-empty external authority receipt digest;
- exact `source_state_digest == target_after_state_digest`.

Dreams cannot bypass this gate. A future dream-to-art workflow must first create
a reviewed counterfactual/proposal or another explicit materialization contract.

## Relationship to existing Symthaea systems

Reality Ledger should compose with, not replace:

- `symthaea-dream`: generates dream/counterfactual content;
- `symthaea-chronicle`: temporal reasoning within a world/history;
- `symthaea-futures-ledger`: forecasting-study receipts and calibration;
- Internal World: beliefs/self-models and future causal generative self;
- Art Studio / Four Ghost: proposal/preview/commit creative worlds;
- Symtropy: a concrete `DigitalCommitted` Bevy host;
- robotics/sensors: possible `PhysicalGrounded` hosts after sensor provenance is qualified.

The intended dependency direction is that these systems emit or consume reality
provenance; the reality ledger does not become an omniscient world model.

## Authority boundary

Reality Ledger v1 has no mechanism for:

- active-policy mutation;
- tool authorization;
- physical actuator authority;
- aesthetic preference/reward;
- deciding whether Symthaea is conscious;
- deciding whether a simulation is metaphysically real.

It records provenance and checks that stronger claims are not made than the
recorded source supports.

## Next host milestone

After v1 qualifies, add a Symtropy adapter that assigns every committed Bevy
studio world a `WorldDescriptor`, assigns every four-ghost branch a
`Counterfactual` child descriptor, emits records for real GPU observations, and
requires `CounterfactualCommitReceipt` when a selected ghost is materialized.

The first host test should prove that a ghost-world event recalled after commit
still points to the ghost lineage rather than being rewritten as a historical
event in the parent world.
