# Series 31 authoring checklist

- Begin from the exact Series 21 final tree with the Series 30 execution backlog accepted as planning input.
- Verify every prerequisite archive and exact tree identity.
- Convert each plan into one intentional patch.
- Record exact commit, tree, test, report, and review identities.
- Run formatting, all-target/all-feature Cargo, Clippy, Nix, independent-verifier, transaction, compatibility, and deterministic packaging lanes.
- Preserve failures and blockers rather than editing them out of history.
- Add no new lifecycle authority semantics.
- Publish only claims supported by retained evidence.
