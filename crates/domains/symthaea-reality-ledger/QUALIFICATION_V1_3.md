# Symthaea Reality Ledger v1.3 — Persistent World Lifecycle Qualification

Status: **fresh qualification required**.

v1.3 extends the host-neutral Reality Ledger with snapshot, suspend, resume,
archive, revisit and snapshot-bound fork continuity evidence. It does not itself
serialize or mutate a host world and does not grant lifecycle authority.

## Mechanical gates

Run from one frozen source HEAD/TREE in the repository's real Nix environment:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-reality-ledger --all-targets
cargo test -p symthaea-reality-ledger
cargo clippy -p symthaea-reality-ledger --all-targets -- -D warnings
```

Retain exact HEAD/TREE, Cargo.lock, flake.lock, rust-toolchain.toml, rustc/cargo,
architecture, Nix shell identity and relevant build flags.

## Required lifecycle invariants

A qualifying run must preserve all of the following:

1. snapshot digests bind the complete `WorldDescriptor`, genesis digest, typed
   host state, ledger head, host artifact, frame and previous-snapshot link;
2. snapshot successors reference the exact prior snapshot digest and cannot
   regress frame coordinates when both frames are known;
3. `Suspend` is `Active -> Suspended`, `Resume` is `Suspended -> Active`, and
   `Archive` is `Suspended -> Archived`;
4. every suspend/resume/archive receipt requires an external authority receipt;
5. lifecycle receipts reference the exact snapshot and exact typed state;
6. equal-looking state bytes in a different semantic domain or algorithm fail
   continuity checks;
7. ordered `WorldLifecycleTimeline` replay must reproduce its terminal state;
8. an archived timeline cannot accept a later resume transition;
9. revisit proof requires a closed prior presence and a distinct open resumed
   presence for the exact same world and agent;
10. prior exit state == snapshot state == resumed entry state as a typed digest;
11. snapshot-bound forks begin from the exact source state and exact parent;
12. counterfactual forks use `CounterfactualOf`, committed forks use
    `SpawnedFrom`, and child identities cannot reuse the parent identity;
13. persisting a fork requires an external persist-authority receipt, while an
    explicitly ephemeral counterfactual fork may remain authority-poor;
14. no lifecycle or revisit receipt is interpreted as evidence of subjective
    continuity.

## Host-adapter boundary

A host such as Symtropy must separately prove that its snapshot artifact digest
actually refers to the persisted bytes used for restoration. Host serialization,
asset restoration, GPU state and deterministic simulation continuation are not
qualified merely because this host-neutral crate passes.

For a restore/revisit study, the host must also demonstrate that the restored
semantic state equals the snapshot's typed state before opening the new
presence session and that the lifecycle timeline is `Active` at re-entry.

## Claim boundary

A PASS supports only that the Reality Ledger can represent and verify persistent
world-state continuity, ordered authority-gated lifecycle transitions, and
snapshot-bound forks. It does not establish consciousness, subjective
continuity, physical grounding, perfect host determinism, or autonomous
lifecycle authority.
