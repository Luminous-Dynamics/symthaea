# Symthaea AI Observatory

This crate contains a small, deterministic comparison kernel for **pre-registered operational predictions** about external-AI observations. It is research infrastructure, not a model adapter, scientific authority, or consciousness detector.

## Current scope

- Validates a versioned prediction registry and computes a deterministic BLAKE3 content digest.
- Compares a registered categorical prediction with a supplied observation.
- Fails closed when evidence is missing, not held out, post hoc, unobservable at the required tier, tied to a different condition/intervention, or marked confounded.
- Identifies theory pairs whose predictions do or do not differ on the same observable and condition.
- Reports missing, duplicate, unexpected, and incomplete trial manifests.
- Includes deterministic known-answer tests for these cases.

## Use

Run the crate's unit tests from the workspace root:

```sh
cargo test -p symthaea-ai-observatory
```

That command is a run instruction, not a claim that tests passed in this change.

## Result semantics

`Supported` means only that the observed operational outcome equals the exact registered categorical prediction under the provided inputs. It does **not** establish:
- consciousness or its absence;
- phenomenal experience or suffering;
- truth of an entire scientific theory;
- causality without a valid intervention/identification design;
- independent replication based solely on distinct IDs;
- integrity of event chronology, signatures, or artifact custody.

The module validates sequence ordering and digest syntax, but caller-supplied sequence numbers do not authenticate chronology. BLAKE3 digests identify bytes; they do not prove who produced the bytes or that they were captured honestly. The current kernel also does not collect internal telemetry, invoke provider APIs, calculate statistical evidence from continuous outcomes, or integrate with a trusted external event log.

## Architecture boundary

The generic `symthaea-evidence-plane` remains responsible for declared-vs-measured mechanism-counter integrity. This crate owns only the domain-specific semantics for comparing frozen theory predictions and operational outcomes. It does not fork generic provenance, authorization, or evidence-ledger infrastructure.

See [AI-OBS-011 issue #7255](https://github.com/Luminous-Dynamics/symthaea/issues/7255) and [protocol contract audit](https://github.com/Luminous-Dynamics/symthaea/pull/7258).
