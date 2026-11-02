# Exact logarithmic prefix-budget index: research design for #9992

## 1. Problem in the current writer

The reference writer uses clone-then-validate and scans all candidates and tokens after each event. This is valuable as a correctness oracle, but a full update costs O(n + T) for n candidates and T escrow tokens. The independent verifier should remain linear; the online writer is the part to optimize.

The key invariant is not a global token scan. It is a maximum-prefix condition over candidate balances.

## 2. Suffix-range formulation

Let

    b_i = I_i - O_i
    P_0 = 0
    P_k = sum_{i=1}^k b_i.

The forward-only prefix budget is

    P_k <= 0 for every k,
    equivalently max(P_0,...,P_n) = 0.

A reservation of amount a from donor i increments O_i, so it changes every prefix containing i:

    P_k <- P_k - a, for k >= i.

A credit of amount a into recipient j increments I_j, so:

    P_k <- P_k + a, for k >= j.

Both are suffix range-adds. Therefore a lazy segment tree over the prefix array supports each reservation or credit in O(log n) and exposes the global maximum at its root in O(1). Since P_0=0 remains in the tree, every valid state has root.max exactly zero.

The tree is independent of reserve policy. FixedRate, CappedOvershoot and future bounded policies only determine the withdrawal amount a; the prefix index only enforces where capital may move.

## 3. Dynamic order and append

For a fixed candidate universe, maintain prefix leaves 0..n. Candidate IDs may be sparse, so map immutable global IDs to stable order positions.

When appending candidate n+1, initialize

    P_(n+1) = P_n,

not zero. The new candidate initially has no balance, so the new terminal prefix equals the old terminal prefix. An implicit treap supports worst-case logarithmic split/merge/append; a capacity-doubling segment tree is simpler but only amortized for resizing. Do not advertise worst-case O(log n) append if using a rebuild-on-growth implementation.

## 4. Exact signed arithmetic

The present ledger uses u128 totals. Silently converting prefix balances to i128 would narrow the supported range. A custom checked signed-magnitude value is sufficient for valid states:

    SignedAmount { negative: bool, magnitude: u128 }

Each prefix balance is a difference of two prefix totals, each bounded by the existing checked u128 incoming/outgoing totals. Under the accepted prefix invariant, P_k is in [-u128::MAX, 0]. The lazy tag for a node is the difference between two valid prefix values and is also bounded in magnitude by u128::MAX. Implement checked add/subtract, normalization of zero, and total ordering. No wrapping, saturating arithmetic or lossy casts.

The event path should preflight candidate arithmetic, total-in/total-out, token remainder and token identity. Apply the suffix update with a bounded undo journal of the touched O(log n) nodes; if any invariant fails, restore exactly those nodes. Keep the existing clone-and-full-scan writer as a reference implementation until differential tests are mature.

## 5. Per-event complexity target

| Operation | Target |
|---|---:|
| Candidate lookup | O(log n) with BTreeMap, or O(1) with dense stable slots |
| Donor cap / own-increment check | O(1) |
| Prefix budget reserve update | O(log n) |
| Prefix budget credit update | O(log n) |
| Global prefix maximum check | O(1) |
| Source-scoped token lookup | O(log T) with BTreeMap |
| Token balance / escrow-total check | O(1) local + aggregate counter |
| Append-only receipt event | O(1) amortized |
| Independent verifier | O(n + T + events), deliberately linear |

The total writer event should be O(log n + log T), excluding output serialization and optional full audit scans. A snapshot that materializes all candidates/tokens remains O(n+T); do not conflate snapshot complexity with event-update complexity.

## 6. Python exact-arithmetic differential screen

The committed standard-library harness models the prefix tree with exact Fractions and compares it with a brute-force prefix vector after each reserve/credit event. It checks:

- root maximum equals the brute-force maximum and remains exactly zero;
- every prefix budget;
- donor and own-increment caps;
- nonnegative and nondecreasing weighted scores;
- exact escrow conservation;
- source-local token ordinals.

Recorded run: 10 traces, 64 candidates, 10 rounds, 132 reserve/credit events, 2,434 segment-tree nodes visited in total, and at most 20 nodes visited by one suffix update. Every final root maximum was zero.

This validates the data-structure transformation at the Python model level only. The node-visit count is an algorithmic screen, not a Rust latency or throughput result. The Python brute-force comparison is intentionally O(n) per event.

## 7. Rust implementation sequence

1. Add an independent Rust SignedAmount type with exhaustive sign/carry/borrow tests around 0, u64::MAX, i128::MAX, i128::MAX+1 and u128::MAX.
2. Add a fixed-size lazy max segment tree for a declared global candidate universe.
3. Differentially apply identical randomized reserve/credit events to the current linear writer, optimized tree writer and independent verifier.
4. Test first/middle/last positions, partial credits, split tokens, overflow, duplicate IDs, wrong destinations and rejected-event rollback.
5. Add dynamic append only after fixed-universe equivalence; use an implicit treap if worst-case logarithmic insertion is a requirement.
6. Benchmark n=10^2, 10^3, 10^4, 10^5 and 10^6 with token counts proportional to n. Report evidence-only, reserve, credit and append workloads separately.
7. Keep the linear verifier independent and run it against every optimized receipt in randomized differential tests.

## 8. Comparison to current methods

Xu, Fischer & Ramdas (UAI 2026) report O(log t) time for online e-closure / compound-e-value decisions. That is the right algorithmic reference point, but not an apples-to-apples runtime comparison: a rejection-decision routine is not the same workload as a provenance-preserving ledger that performs token accounting and receipt validation.

Our performance claim should be narrower and testable: preserve the same exact prefix invariant and independent replay while moving writer event updates from O(n+T) scans to O(log n + log T) data-structure operations. Only compiled Rust measurements on pinned hardware may substantiate speedup.

## Claim ceiling

This work is an algorithmic optimization plan plus a Python structure check. It does not improve statistical power, alter the compound null-mass proof, validate e-processes, authenticate filtrations/arrivals, or establish superiority over SOTA.
