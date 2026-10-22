# SPORE-FED-001B — Deterministic Federation Witness

**Status:** executable semantic witness  
**Issue:** #6311  
**Parent:** #6309  
**Claim ceiling:** deterministic software/network semantics only

## Purpose

This witness is the semantic oracle for the Spore federation showcase. It deliberately does **not** implement another transport protocol. It models the state transitions that the showcase visualizer must be able to observe from either deterministic semantic replay or an adapter around the existing authenticated swarm/native witnesses.

Executable source: `crates/domains/symthaea-spore/examples/spore_federation_witness.rs`

Run locally with:

```text
cargo run -p symthaea-spore --example spore_federation_witness
```

Qualify with:

```text
cargo test -p symthaea-spore --example spore_federation_witness
```

No CI result is implied by this document.

## Semantic contract

| State | Meaning | Can capability sharing mutate it? |
|---|---|---|
| node identity | stable sovereign node identity | **No** |
| identity generation | explicit identity lifecycle | **No** |
| capability generation | version of locally declared capability state | **Only on local change** |
| capability set | transferable knowledge/availability | **Yes, explicitly** |
| authority epoch | local authorization generation | **No** |
| event lane | observation / analysis / recommendation / authorization / execution | **No implicit promotion** |

> Capability transfer changes recipient capability state, never recipient identity or authority.

## Deterministic scenarios

The example exercises:

- capability transfer without identity transfer;
- authorization-boundary rejection;
- partition with a bounded queue;
- explicit delivery delay;
- capability-generation change;
- stale-generation rejection;
- post-heal delivery;
- deterministic replay (`run() == run()`);
- absence of any encoded physical-link proof.

## Expected dispositions

The baseline run must contain all of these dispositions:

- **Applied** — local capability declaration and valid capability application;
- **Queued** — capability traffic crossing a closed link;
- **Rejected(authorization-boundary)** — analysis-shaped input cannot mint authority;
- **Rejected(stale-capability-generation)** — an event from an older capability generation is not silently current.

The example also keeps node identities and authority epochs unchanged while capabilities move between nodes.

## Why this is stronger than a static demo

A static animation can make an architectural claim look true. The witness instead gives the visual layer a deterministic semantic oracle:

`configuration + event sequence → disposition ledger`

The public visualization can therefore render the same event vocabulary for replay-only runs, native local federation runs, and future DTN/interplanetary profiles.

The visualizer should never infer success from animation timing. It should consume explicit dispositions.

## Native federation boundary

The next adapter layer should translate existing swarm/native events into this semantic vocabulary rather than reproduce transport logic.

The showcase must preserve the existing transport claim ceilings:

- local acceptance is not remote application commit;
- reliable protocol acknowledgement is not domain execution;
- datagram receipt is not durable state;
- authenticated transport is not authorization.

This keeps the showcase honest while still making the existing networking work visible.

## Claim ceiling

This witness establishes only deterministic behavior of the executable software model.

It does **not** establish:

- physical interplanetary communications;
- deployment at planetary scale;
- consciousness or sentience;
- governance legitimacy;
- social or economic outcomes;
- safety of arbitrary real-world autonomous action.

Those claims require independent evidence and are outside this qualification.