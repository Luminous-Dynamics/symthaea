# Merkle-authenticated projective closure score receipts

**Status:** research prototype, stacked on PR #9748. It proves an exact closure-separator score relative to a trusted snapshot root. It does not prove that source evidence is true, that the active roster is complete, or that a solver found an optimal rejection set.

## 1. What is being proved

For exact rational:

\`\`\`
b_i = alpha * w_i * (M_i - 1)
r = |R| > 0
Q(R) = sum_i max(0, 1{i in R}/r - b_i)
\`\`\`

define:

\`\`\`
C = sum_i max(0, -b_i)
Q(R) = C
       + count{i in R : b_i < 1/r}/r
       - sum{i in R : 0 <= b_i < 1/r} b_i.
\`\`\`

Equivalently, the final sum is the difference between selected-key prefix sums below 1/r and below zero.

The authenticated tree is ordered by exact (b_i, hypothesis_id). Each node commits to its rational key, stable hypothesis ID, selected bit, child hashes, and subtree aggregates:
- subtree size;
- selected count;
- selected-key sum;
- total baseline contribution sum max(0,-b_i);
- a digest of selected IDs in deterministic in-order (b_i, hypothesis_id) order.

A deterministic median-balanced tree gives logarithmic depth. Strict threshold comparisons exclude every key with b_i == threshold, including duplicate-evidence ties.

## 2. Receipt and independent verification

The receipt contains:
- tree root hash;
- source manifest root;
- snapshot sequence, previous snapshot hash and configuration;
- alpha, weight digest and evidence version;
- root aggregate, rejection count and selected-ID projection digest;
- one authenticated prefix proof at 1/r;
- one authenticated prefix proof at 0;
- claimed exact rational closure score.

Each prefix proof reveals the root-to-boundary search path. Every path node includes its own entry and sibling child hash/aggregate. The verifier reconstructs the path bottom-up, recomputes each aggregate and node hash, checks the traversed child commitment at every level, and compares the result with the caller-pinned root. It then recomputes:

\`\`\`
Q(R) = C + count_prefix(1/r)/r
          - (sum_prefix(1/r) - sum_prefix(0)).
\`\`\`

The proof size and verification work are O(log n) tree nodes for each prefix, plus exact rational arithmetic and SHA-256. Two prefix proofs are included in each receipt. Snapshot construction in this prototype sorts the entries in O(n log n) and builds a balanced tree in O(n); it is not a dynamic authenticated update structure yet.

## 3. Trust boundary and rollback semantics

A Merkle proof proves correctness **relative to a committed root**. It does not prove:
- the raw e-values or weights are statistically valid;
- the evidence was generated under the declared filtration;
- no hypothesis was omitted from the active roster;
- the source manifest itself is truthful;
- the solver's proposed set is utility-optimal;
- a later snapshot is globally canonical.

The upstream ingestion/manifest verifier must independently attest that the evidence snapshot, weight vector, alpha and active hypothesis roster correspond to the committed manifest root. The tree root and this attestation must be signed/anchored together. This prototype binds the manifest root into the snapshot header but does not implement that external attestation.

The snapshot includes a sequence number and previous snapshot hash, but the verifier does not maintain a monotonic trusted checkpoint. An external append-only transparency log or signed monotonic state is required to prevent rollback and equivocation/forks.

The selected-ID projection digest is computed in tree order (b_i, hypothesis_id). It commits to the selected membership projection for that fixed evidence snapshot; it is not a standalone canonical sorted-ID set hash independent of b_i.

## 4. Files

- validation/projective_closure_receipt.py — deterministic balanced Merkle tree, strict prefix proof generation, exact score receipt and standalone verifier.
- validation/test_projective_closure_receipt.py — randomized exact-rational differential checks, ties, duplicate keys/IDs, tampered path/aggregate/threshold/score, snapshot/manifest/config mismatch and chain metadata.
- validation/benchmark_projective_closure_receipt.py — build, proof generation, proof bytes, verification and O(n) exact scan benchmark.
- validation/test_benchmark_projective_closure_receipt.py — benchmark correctness checks.

## 5. Benchmark

Run:

\`\`\`bash
python validation/benchmark_projective_closure_receipt.py \\
  --sizes 1000 10000 100000 --repeats 3 \\
  --output closure-receipt-results.json
\`\`\`

The benchmark reports snapshot build time, proof generation time, verifier time, proof bytes, proof path nodes and an independent exact O(n) scan for comparison. It intentionally distinguishes the O(n log n) snapshot build from O(log n) proof verification. A static snapshot receipt and dynamic per-event update are separate costs.

The checked-in tests and benchmark runner have been authored but not executed from a connected repository checkout in this environment. No CI pass is claimed.

## 6. Next steps

1. Run randomized proof verification and mutation controls in the repository's pinned environment.
2. Benchmark at n=10^3, 10^4, 10^5, and 10^6; report build vs query vs verification separately.
3. Integrate a receipt verifier with #9748 while retaining the O(n) exact scan as an independent fallback.
4. Add a real source-manifest attestation binding raw evidence bytes, e-values, weights, roster, filtration metadata and snapshot root.
5. Add a monotonic transparency checkpoint and explicit fork/rollback negative controls.
6. Investigate persistent authenticated AVL/B-tree updates to avoid rebuilding the full balanced tree for every evidence snapshot.
7. Extend receipts to proof-carrying capacity/utility solver decisions only after separating feasibility proof from optimality certificates.

The target is not blockchain for statistics. The useful shift is that a computationally efficient statistical decision can carry a compact proof of the exact mathematical condition it claims to satisfy, while keeping the source-truth and theorem assumptions explicit.
