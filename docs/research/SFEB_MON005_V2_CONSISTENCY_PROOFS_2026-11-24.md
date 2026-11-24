# SFEB MON-005 v2 — RFC 9162 consistency proofs and suffix-only witnessing

**Date:** 2026-11-24  
**Status:** executable Node.js reference model; locally tested 45/45  
**PR:** https://github.com/Luminous-Dynamics/symthaea/pull/13518  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/13515

## Why v2 matters

MON-005 v1 recomputed the full record prefix at every witness checkpoint. This was conservative and easy to audit, but costs O(n) work and data transfer at every checkpoint. It is the wrong scaling path for long-running evidence logs.

V2 implements the RFC 9162 Merkle Tree Hash consistency-proof algorithm. A witness that already trusts a checkpoint of size m can verify that a new root of size n is an append-only extension using a compact proof, without receiving the old m records. The witness receives only the appended suffix, checks its tenant/sequence/previous-hash chain from the trusted terminal record hash, verifies the consistency proof from the old root to the new root, and verifies inclusion of each appended record in the new root.

RFC 9162 bounds a consistency proof to at most ceil(log2(n)) + 1 nodes. This is the decisive scaling change: proof verification is logarithmic in tree size rather than replaying the entire prior record list.

## Executed evidence

Exact source: `benchmarks/frontend/monitoring/mon005-witnessed-evidence-v2.mjs.gz`  
Report: `benchmarks/frontend/monitoring/mon005-witnessed-evidence-report-v2.json`  
README: `benchmarks/frontend/monitoring/mon005-witnessed-evidence-README-v2.md`

Node.js v22.16.0:

- **45/45 deterministic cases passed.**
- **496 inclusion proofs** verified across all leaf positions in tree sizes 1–31.
- **2,016 consistency proofs** verified: every old-tree size m for every new-tree size n in [2,64], with 1 <= m < n.
- Each consistency proof is checked against the bound ceil(log2(n)) + 1.
- Negative controls reject wrong previous roots, changed suffix content, altered inclusion proofs, empty/truncated/extended consistency paths, invalid size parameters, wrong tenant/sequence/prior hash, rollback and history rewrite.
- Four test-only Ed25519 witnesses sign two checkpoints (12 and 18 records); a 3-of-4 quorum tolerates one unavailable witness and rejects two signatures.
- Witness update receives the suffix only (six records) plus inclusion and consistency proofs; it does not receive the historical 12-record prefix.
- Raw source and gzip stdin syntax/execution outputs match byte-for-byte.

## RFC algorithm and interoperability

RFC 9162 (Experimental, December 2021) defines the Certificate Transparency Merkle Tree Hash: leaf hashes are domain separated from internal nodes, inclusion proofs support membership verification, and consistency proofs prove that a later tree is an append-only extension of an earlier one. Its tree hash uses `HASH(0x00 || leaf_data)` for leaves and `HASH(0x01 || left || right)` for internal nodes, with the tree recursively split at the largest power of two smaller than the node count.

This implementation follows that tree shape and the RFC consistency proof generation/verification procedure. Its leaf data is the 32-byte digest of the canonical SFEB record, not the raw record bytes; the record's own digest binds tenant, sequence, predecessor and payload. Test coverage exhaustively checks the finite 2..64 size range, but this is not an independent cross-implementation interoperability certification. Next add RFC test vectors or differential tests against a mature CT library.

RFC 9162 also describes split-view risk: a misbehaving log may present inconsistent views to different clients. Consistency proofs alone do not stop this if different clients are shown different histories. Independent monitors, gossip and external witnesses are still needed. The 3-of-4 signature scheme is intended as a separate SFEB checkpoint-witness layer, not a replacement for a public transparency ecosystem.

## 3-of-4 witness assumption

Any two 3-of-4 quorums intersect in at least two witnesses. With at most one Byzantine witness, the intersection includes at least one honest witness. If honest witnesses durably refuse to sign two different heads for the same `(tenant, epoch)` using atomic state, two conflicting quorums cannot both be formed. This depends on independent trust and failure domains and durable atomic per-tenant/epoch state. Four objects in one Node process do not establish operational independence or a distributed BFT theorem.

## Remaining hardening

1. Replace the local strict-JSON subset with a conformant RFC 8785 JSON Canonicalization Scheme implementation and cross-language vectors (Rust, Go and JavaScript).
2. Differential-test RFC 9162 inclusion and consistency proof algorithms against independent implementations and official fixtures.
3. Implement independent Rust witness services with durable monotonic state, atomic unique/CAS enforcement for `(tenant_id, epoch)`, independent keys, rotation and revocation.
4. Race conflicting checkpoint heads across multiple processes; test crash/restart, backup rollback, signer equivocation and one compromised witness.
5. Implement efficient proof generation/range retrieval at a real log backend; v2's generator still accepts a full local event array, even though the witness receives only a suffix and proof bundle.
6. Anchor the SFEB checkpoint digest in an established transparency service such as Rekor/Tessera or a CT-compatible log, and add independent gossip/monitoring.
7. Integrate one real idempotent/conditional external sink, verify the objective fence at the sink and independently read the postcondition.

## Industry benchmark design

SFEB should not compete with DBOS or Temporal for durable workflow replay. Compare:

- PostgreSQL native transactional outbox;
- DBOS Go `RunAsTransaction` on one datasource (same-database co-commit baseline);
- DBOS Rust steps (separate at-least-once baseline);
- Temporal native activities/history/retries;
- each relevant native arm augmented with SFEB evidence, sink authority and witnessed checkpoint;
- existing transparency-log anchoring, with and without SFEB's external-action evidence semantics.

Matched arms must share source traces, tenant policy, idempotency keys, sink semantics, durability configuration, hardware, worker counts and failure schedule. Measure action p50/p95/p99 separately from asynchronous proof/quorum/anchor p50/p95/p99. Also report inclusion/consistency proof bytes, signature CPU, checkpoint cadence, witness egress, storage growth, recovery, unavailable-witness degradation and cost per independently verified and externally anchored action.

Proposed targets (not achieved): zero accepted unauthorized or cross-tenant actions over >=1M independent adversarial attempts per release candidate; zero conflicting quorum heads under stated assumptions; 100% of `externally_anchored` results independently verified; no success on unknown sink outcomes; <=10% p95/p99 overhead versus matched native external-action arms. Publish raw denominators and traces; do not compare the old SQLite smoke result to DBOS provider sizing numbers.

## References

- RFC 9162: https://www.rfc-editor.org/rfc/rfc9162.html
- RFC 8785: https://www.rfc-editor.org/rfc/rfc8785.html
- Sigstore Rekor security model: https://docs.sigstore.dev/about/security/
- Trillian/Tessera project documentation: https://google.github.io/trillian/
