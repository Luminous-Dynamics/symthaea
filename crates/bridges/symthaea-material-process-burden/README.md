# Symthaea Material Process-Burden Evidence

This crate exposes two explicit process-burden quantities for Tier-1 material screening:

- `maximum_process_temperature` in kelvin;
- `synthesis_step_count` as a count.

It intentionally does **not** collapse them into one manufacturability score.

Every record is bound to one exact material/process/conditions context and one explicit evidence basis. Modeled routes remain analytical evidence; documented experimental, pilot and industrial routes remain experiment evidence without being promoted into universal manufacturability validation.

The normalized capture binds exact source-document SHA-256, source metadata, process identity and values. The adapter verifies the source bytes before emitting evidence.

A successful receipt contains both predictions separately so a screening policy must choose the manufacturability quantity it actually intends to optimize or constrain.

Neither metric by itself establishes yield, purity, throughput, equipment availability, cost, scale-up success, worker safety, product quality, supply availability or deployment fitness.

## Network-free CLI

`material-process-burden <process-data.json> <source-document> <candidate-id> <material-id> <process-id> [explicit-mapping-note]`

Without a mapping note, exact-material-ID mode is required.
