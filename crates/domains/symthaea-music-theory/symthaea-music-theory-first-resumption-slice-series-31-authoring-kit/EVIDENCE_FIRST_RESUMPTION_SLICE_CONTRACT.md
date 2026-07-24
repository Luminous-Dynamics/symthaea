# First Resumption Vertical-Slice Qualification Contract

Series 31 may claim the first lifecycle slice is qualified only when:

1. A native Series 21 closure fixture verifies.
2. A content-derived trust segment binds the exact closure and selected branch.
3. Caller-owned resumption policy cannot be weakened by bundle data.
4. Recovery-authority and witness quorums authenticate one exact plan.
5. Publisher delegation and allowance are fresh, post-closure, and segment scoped.
6. All mutable preconditions are reauthenticated at commit time.
7. The catalog append, allowance consumption, segment update, ordinal advance, and receipt commit atomically.
8. Two first-mutation attempts cannot both commit.
9. Replay, staleness, policy substitution, crash, rollback, and race corpora pass.
10. Rust and an independent implementation agree on frozen canonical vectors.
11. Public API and CLI scenarios reproduce the same exact result.
12. The qualification report links exact source, tree, tests, vectors, and limitations.

This contract does not qualify reopening, recursive recovery, retirement, production deployment, or scientific correctness of publication content.
