# symthaea-assurance-trust-store-current-state

Resolve the currently active monotonic trust-store segment before readiness evaluation.

Before any recovery, the current segment must begin at logical store revision 1. After an accepted recovery, the current segment instead begins at the exact first replacement checkpoint recorded in the one-shot recovery acceptance ledger.

Within the active counter epoch the resolver requires contiguous logical revisions, exact predecessor digests, strictly increasing counters, non-decreasing anchor/policy state, stable counter epoch, and non-regressing time.

This keeps the original no-epoch-change checkpoint-chain theorem strict while still permitting explicitly reviewed hardware/store replacement through the separate recovery-continuity path.

This crate only resolves assurance state and never grants physical authority.
