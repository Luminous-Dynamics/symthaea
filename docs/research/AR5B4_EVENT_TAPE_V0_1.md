# AR-5B4 Event Tape v0.1

**Status:** implementation candidate; not yet compiled or executed on an exact repository head.  
**Module:** `symthaea-architecture-engineering::event_tape`  
**Tracking:** [Issue #9362](https://github.com/Luminous-Dynamics/symthaea/issues/9362)

## Purpose

The event tape is the evaluator-owned record of what a search policy selected, what the evaluator/verifier returned, what the policy was allowed to observe, and what resources the action consumed. It provides a common event contract for mechanism-matched policy comparisons and a replay boundary for policy adapters.

It is not a benchmark result, a digital signature, proof of runner identity, or evidence that the evaluator itself is honest.

## Frozen identities

The header binds the campaign, policy ID and algorithm ID, canonical JSON policy configuration and its SHA-256 digest, task recipe, comparison profile, evaluator, verifier, proposal source, initial canonical direction IDs, execution mode, worker count, and total normalized budget ceiling.

A mechanism-matched tape must include a shared proposal-stream digest. A best-practical-system tape may use a different proposal source, but must identify that source. These comparison modes must not be merged into one result table.

## Event contents

Every event contains a unique event ID, policy ID, worker and round, optional batch ID, the last event sequence visible to the policy, the digest of that exact visible prefix, an optional retry reference, the selected action intent, the evaluator/verifier outcome, and a complete resource receipt.

Candidate evaluations bind candidate, parent, semantic-delta, task-recipe, comparison-profile, and evaluator digests. A verified result requires a finite score and evidence-artifact digest. Rejected, inconclusive, crashed, and timed-out actions remain in the tape and remain charged.

Direction proposals bind a descriptor digest and semantic delta. Only the verifier may admit a canonical direction ID. A duplicate must map to an already registered ID and cannot mint another arm. An admitted ID must be genuinely new to the current registry.

## Resource accounting

Each action records host CPU milliseconds, accelerator milliseconds, peak memory bytes, energy millijoules, input/output tokens, tool calls, evaluator calls, verifier calls, and normalized budget charge in microunits.

A reading is either `Measured(value)` or `Unknown(reason)`. Measured zero is valid; unknown is never converted to zero. `validate_integrity` preserves structurally valid but unmetered tapes for diagnosis. `validate` is the qualification gate and rejects any unknown measurement or unknown charge. Retries are separate events and consume separate charges. The running normalized charge may not exceed the header budget ceiling.

## Sequential and batched visibility

Sequential events must use worker 0, have no batch ID, advance one round per event, and see the entire completed prefix.

Batched events use a common batch ID and round. All events in the batch must have the same pre-batch visibility cutoff and the same visible-history digest. Each worker may contribute at most one action to a batch, and sibling results cannot become visible until a later batch. Worker count and claim caps are part of the frozen header.

## Integrity and replay

The tape is append-only at the API boundary. Each event hash binds its sequence, prior event hash, and serialized payload. Validation checks sequence continuity, the hash chain, the policy/config identity, visible-prefix digest, action/outcome agreement, canonical direction registration, retry identity, resource receipts, worker/batch semantics, claim caps, and budget ceiling.

A `PolicyReplay` adapter receives only the event prefix declared visible for a decision plus that decision's worker/round context. It must reproduce whether the policy selected a registered direction or requested a new direction. Candidate-generation and evaluator correctness remain separate responsibilities.

The SHA-256 chain is tamper-evident only when the tape is retained and compared against a trusted digest. It does not authenticate who produced the tape. Signing, remote append-only storage, independent evaluator deployment, and cross-policy campaign reconciliation remain follow-on work.

## Qualification limits

Passing the tape validator establishes structural integrity and accounting completeness under the declared contract. It does not establish that a scientific claim is novel, that the evaluator's labels are correct, or that one policy is superior. Confirmatory comparisons still require paired landscapes, independent templates, pre-registered endpoints, sealed evaluation, external baseline reproduction, independent replay, and held-out transfer.
