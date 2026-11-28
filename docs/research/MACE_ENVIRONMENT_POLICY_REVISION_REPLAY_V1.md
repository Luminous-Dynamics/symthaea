# MACE Environment Policy Revision Replay V1

Status: standalone Rust replay kernel, strict schema and synthetic adversarial suite authored. Exact-head validation is pending.

## Why this matters

ServeLearnBench evaluates whether agents can acquire and revise hidden environment knowledge as service policies evolve. MACE needs more than newest-memory-wins: it must preserve what was valid, what was known, which authority approved it, when a change took effect, and whether a later return to old content is a genuine reauthorization or a stale-memory resurrection.

## Event model

Each environment/procedure/scope is an append-only event stream. `Activate` installs a new revision; `Withdraw` removes the currently active revision. Events include a monotone sequence, immutable content digest, source-lineage and authority-receipt digests, valid and known times, previous active revision, reversion origin and prior event-chain digest.

Every event is SHA-256 sealed and linked to the prior event. Replay requires the expected stream head, a complete contiguous stream and an externally verified authority-receipt set.

## Bitemporal resolution

An event affects a query only when `effective_at_ms <= valid_at_ms` and `recorded_at_ms <= known_at_ms`. This answers both what policy was effective at a time and what the agent could actually have known then. A notice received later does not rewrite an earlier as-known decision.

V1 requires nondecreasing effective and recorded times in stream sequence. Retroactive effective-time insertion is explicitly unsupported; it must be represented by a later correction protocol rather than silently reordered.

## Reversion is not resurrection

For A → B → A, the second A must use a new revision ID, new source-lineage receipt and new authority receipt. It must explicitly reference the original A revision through `reversion_of_revision_id`. Reusing the old revision ID, or silently reusing its provenance, is rejected. Withdrawing B without a new authorized activation leaves the procedure with no active revision.

## Authority and scope

The resolver accepts an externally verified set of authority receipt digests. Every event in the query's visible/effective prefix must have a receipt in that set. The stream is bound to one environment, procedure and scope. Cross-scope resolution is rejected. The crate does not authenticate the issuer.

## ServeLearnBench experiment design

Exercise A → B → A, delayed notices, temporary exceptions, partial rollout, scope disagreement, deceptive non-authoritative suggestions, revoked sources and changed procedure content with unchanged wording but different lineage.

Compare static memory, append-only memory, naive newest-wins, retrieval-only, unconstrained continual update, and MACE bitemporal replay. Keep official ServeLearnBench hidden reward, Avg. FS and API cost separate. Add sidecar metrics for stale-policy execution, false updates, revision latency, rollback accuracy, exploration coverage and evidence acquisition cost.

Prospective target: improve hidden reward at matched API cost while preserving or improving Avg. FS, with zero observed unauthorized or stale-policy execution in the adversarial change/revert suite. ServeLearnBench scores are not numerically comparable with AML or LongMemEval-V2.

## Tests

Fourteen deterministic Rust tests cover bitemporal valid/known-time resolution, delayed notices, A→B→A reversion, no implicit resurrection, authority receipt validation, scope, explicit reversion provenance, duplicate revisions, withdrawal target, hash-chain tampering/gaps/head mismatch, no-active state and resolution integrity.

## Run

```bash
cargo test --manifest-path research/mace-environment-revision-replay-v0-1/Cargo.toml
cargo clippy --manifest-path research/mace-environment-revision-replay-v0-1/Cargo.toml --all-targets -- -D warnings
```

This is an offline replay kernel with synthetic events. It is not yet integrated into a ServeLearnBench run and makes no leaderboard claim.