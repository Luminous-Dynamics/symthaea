# Symthaea Reality Ledger v1.2 qualification

Reality Ledger v1.2 is a new evidence lineage. The earlier v1 qualification receipt
must not be inherited automatically because the public v1 construction head and the
integrated compilation-aligned head were not byte-identical.

## Mechanical gates

Run under the project Nix development shell:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-reality-ledger --all-targets
cargo test -p symthaea-reality-ledger
cargo clippy -p symthaea-reality-ledger --all-targets -- -D warnings
```

Record exact source HEAD/TREE, `Cargo.lock`, `flake.lock`, toolchain identity,
architecture and relevant build flags.

## Required semantic gates

1. Typed digest domains prevent equal-looking values from different state serializers
   from satisfying a materialization equality gate.
2. Claim grounding remains independent of world layer:
   - derived computation cannot claim direct physical observation;
   - direct digital-world observation remains digital provenance.
3. Checkpoint structural validation detects a substituted ledger head. Cryptographic
   signature verification remains external and must not be claimed by this crate.
4. Seeded-stochastic genesis requires an explicit seed.
5. Presence sessions with authority-bearing capabilities require an external authority
   receipt digest.
6. WorldGraph rejects missing parents, inconsistent generation depth and cycles while
   preserving siblings after context exit.
7. WorldObservationBundle rejects mixed world, lineage, revision, frame, state digest,
   camera or fidelity planes and rejects missing required planes.
8. TypedCounterfactualCommitReceipt requires exact typed source-state == committed
   after-state and never reclassifies the counterfactual source world.

## Integration gates before a Symtropy-world claim

Do not claim a live Symtropy adapter PASS from this host-neutral crate alone. A separate
adapter qualification must establish that the committed Bevy world, each four-ghost
preview, GPU color/depth/object-ID receipts and materialization receipts are mapped to
one consistent Reality Ledger world/revision/frame/state identity.

## Scientific boundary

A PASS establishes provenance mechanics. It does not establish physical truth,
consciousness, subjective experience, aesthetic quality, object permanence beyond the
measured contract, or mutation authority.
