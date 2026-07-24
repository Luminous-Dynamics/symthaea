# Series 21 authoring checklist

## Before editing

- Obtain and checksum the exact Series 20 source archive.
- Confirm its advertised final Git tree independently.
- Initialize a clean Git repository and record the baseline commit/tree.
- Run existing Series 20 tests before adding code.

## Per patch

- Keep one authority boundary per commit.
- Add independent audit recomputation, not builder self-check only.
- Reject unknown fields on every persisted model.
- Use `u64`/fixed-width numeric identities in persistence.
- Reauthenticate external authority at each state mutation.
- Never let a hash stand in for signer trust or legal authority.

## Final verification

```text
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo check -p symthaea-music-theory --examples
```

Run focused lanes for `resumption`, `segment`, `recovery`, `publication`, `witness_policy`, `gossip`, `quarantine`, and `schema`, then the normal Nix lane.

## Reproducibility

- Replay every numbered mail patch against the untouched base.
- Require authored and replayed Git trees to match byte-for-byte.
- Produce deterministic source and patch archives.
- Hash every distributed artifact; keep the outer archive checksum externally.
