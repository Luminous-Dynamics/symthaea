# temporal-replay-oracle

A standalone, independent reference evaluator for EGDDM D2 temporal replay.

## Why this crate is separate

This crate does not import, link, or call development-map-validator. It separately:
- parses the D2 event-ledger and replay-query contracts;
- recomputes the event hash chain;
- checks source/recording/actor-availability ordering and transition preconditions;
- derives actor-knowledge, ledger-as-recorded and retrospective-restated projections;
- compares normalized results against hand-derived expected-state fixtures.

The goal is to reduce common-mode bugs between the candidate reducer and the evaluator. It is still authored by the same project and is not an external independent audit; a separate reviewer and separately administered gold custody remain necessary.

## Run

From this crate directory:

    cargo test

    cargo run -- \
      ../../docs/development-map/fixtures/temporal-event-ledger-v0.1.json \
      ../../docs/development-map/fixtures/temporal-replay-actor-knowledge-v0.1.json \
      ../../docs/development-map/fixtures/temporal-event-ledger-v0.1.expected.json \
      temporal-replay-actor-knowledge-v0.1.json

For the ledger-as-recorded and restated fixtures, substitute the relevant query filename for the second argument and the final query key.

With two arguments, the executable prints its independently computed normalized projection. With four arguments, it compares that result to the expected fixture and exits nonzero on mismatch.

## Authority and status

This is a reference evaluator over caller-supplied JSON. It does not authenticate the originating source, actor availability receipt, trusted clock, Git object existence, or completeness of the event stream. It cannot by itself qualify CI/test outcomes. The expected-state fixtures are hand-derived.

Six Rust unit-test definitions cover three known-answer temporal views, event-identity tampering, repository-identity tampering and duplicate terminal transitions. They have not been compiled or executed. Do not treat them as passing.

The candidate and oracle intentionally share the v0.1 data contract, but not implementation code. Future qualification should add an independently maintained evaluator, blind gold-label custody, mutation tests and externally witnessed input digests.
