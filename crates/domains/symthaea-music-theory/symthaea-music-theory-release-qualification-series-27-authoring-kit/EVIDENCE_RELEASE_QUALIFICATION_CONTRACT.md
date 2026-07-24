# Release Qualification and API Stabilization Contract

A Series 27 release candidate may claim qualification only when:

1. Every public API, schema role, command, artifact format, verifier role, and deployment assumption has an explicit stability tier.
2. Series 21 historical data remains honestly verifiable and incompatible mutation paths fail explicitly.
3. Stable canonical roles pass a frozen cross-implementation conformance corpus.
4. Complete public lifecycle scenarios pass through supported APIs and commands across persistence restarts.
5. The real mail series replays cleanly from the exact Series 21 tree and reproduces one qualified final tree.
6. Supported serialization lanes produce identical stable bytes.
7. Worst-case-valid artifacts remain within recorded resource budgets and hostile inputs remain bounded.
8. Long-history, restart, crash, transaction-race, privacy, supply-chain, and mutation-surface gates pass.
9. The release claim matrix exposes failed, unsupported, not-run, and waived cells without aggregate masking.
10. Source, patch, conformance, test, benchmark, API, claim, and evidence artifacts reproduce byte-for-byte.
11. The release-candidate freeze invalidates stale evidence after any source or dependency change.
12. A clean third-party offline verification run succeeds from the public evidence bundle.

This contract does not claim universal production reliability, support for unexecuted platforms, or permanent architectural finality.
