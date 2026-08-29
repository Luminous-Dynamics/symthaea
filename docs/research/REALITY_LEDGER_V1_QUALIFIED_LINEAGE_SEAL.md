# Reality Ledger v1 qualified lineage seal

Date: 2026-08-29

## Authoritative qualification receipt supplied from the integrated workspace

The qualified Reality Ledger v1 execution root reported after Nix qualification is:

- qualified integrated HEAD: `224e78e4c5a1bc533ec1065549f2a9a5bf732958`
- qualified integrated TREE: `cf2a74769786181ffde4b5ed87117b11b342085d`
- public patch-source base: `eea6e6d8d562b55a57c852dd125cbc7b676f49a0`
- public construction head before compile alignment: `f3e113e4ed691b7192b93e54a4ca0345f400b7a5`
- archive SHA-256: `9553ddb9bc0a9916d022a156b1380dcdaef3c5c48b63871ba84654d20c9c5f4d`
- `cargo check -p symthaea-reality-ledger --all-targets`: PASS
- `cargo test -p symthaea-reality-ledger`: 15/15 PASS
- `cargo clippy -p symthaea-reality-ledger --all-targets -- -D warnings`: PASS

## Important reproducibility boundary

The integrated qualification HEAD `224e78e4...` is not currently resolvable in the
public standalone GitHub repository. Therefore this repository must not claim that
`f3e113e4...` is byte-identical to the qualified root, and the compile-alignment delta
must not be reconstructed from guesswork.

Reality Ledger v1.2 is intentionally based on the last public construction head
`f3e113e4...` and MUST receive a fresh Nix qualification after application to the
integrated workspace. A v1.2 qualification receipt should publish the exact final
source HEAD/TREE used for the new evidence lineage.

This file is a lineage seal, not a substitute patch for the unpublished compilation
alignment delta.
