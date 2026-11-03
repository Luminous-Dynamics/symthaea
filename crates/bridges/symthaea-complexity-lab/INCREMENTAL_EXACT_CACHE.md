# ETH-019 — content-addressed incremental exact inference

**Status:** Python reference prototype executed and committed alongside its deterministic results. Rust integration, Lean proof, public benchmark comparison and measured speedup remain open.

## Positioning against industry

Incremental exact model counting is established prior art. PBCount2 (Yang & Meel, AAAI 2025) provides exact projected and incremental pseudo-Boolean counting using LOW-MD ordering and caching. Its evaluation reports solving at least 1.18× as many incremental benchmarks as the previous best PB counter, and its paper notes that it disables preprocessing in incremental mode to preserve cache IDs and variable/constraint mappings. The authors identify inprocessing or extra transformation metadata as future directions.

- Paper: https://doi.org/10.1609/aaai.v39i11.33240
- PDF: https://ojs.aaai.org/index.php/AAAI/article/download/33240/35395

Repeated exact inference and message reuse in trees/junction trees is also established. This work does not claim novelty for incremental inference, caching, content addressing or message passing. It tests a narrower capability: cache exact bucket-elimination messages by canonical input content and algebra identity, authenticate a prior independent-checker receipt, and propagate only the consequences of changed message content across several exact output algebras.

## Architecture

Treat intermediate bucket messages as immutable exact artifacts.

A message key binds:
1. schema/protocol version;
2. exact algebra identity and semantics version;
3. variable cardinalities;
4. eliminated variable;
5. the **multiset** of canonical input-factor/message digests (duplicates are significant);
6. checker identity.

A factor's content digest binds its sorted scope and all exact table cells. The key intentionally does not bind factor IDs or input-list order, so semantically identical inputs reuse the same message. The bucket inputs are sorted by digest while preserving duplicates.

When a factor changes:
- the earliest affected bucket receives a different key and is recomputed;
- the new output message is independently checked and authenticated;
- downstream buckets whose actual input-message content is unchanged can still hit;
- disconnected components remain reusable;
- a changed scope/domain/algebra/checker version changes keys and cannot silently hit.

The content-addressed key is a correctness boundary, not a heuristic invalidation guess. Same-scope factor splitting/merging and insertion/removal of exact unit factors can now preserve the normalized bucket key. A downstream message may be reusable even if an upstream key changed, but only when the newly computed upstream output has exactly the same canonical digest. The normalizer identity is included in every key so a future normalization rule change cannot silently reuse old receipts.

## Reuse trust model

A fresh message is computed by the producer and checked by a distinct reference checker path before being admitted to the cache. The receipt binds exact message key, output digest, algebra and checker ID. The prototype uses HMAC-SHA256 with a protected test key so that a cache payload plus recomputed public digest/receipt fields cannot be forged without the key.

HMAC authenticates stored bytes; it does not prove the checker was sound or the key uncompromised. Production should use protected signing keys/KMS or hardware-backed attestation, a separately built verifier, pinned checker binary hashes, key rotation/revocation, cache quotas and auditable provenance. HMAC test credentials are not production credentials.

## Executed reference

Files:
- `reference/incremental_exact_cache.py`: standalone producer, separate bucket checker, content-addressed cache, receipt validation, two exact algebras, direct assignment oracle and mutation suite. Run with `python3 reference/incremental_exact_cache.py`.
- `reference/incremental_exact_cache_results.json`: deterministic results.

Seed: `20261103`.

| Check | Result |
|---|---:|
| Unchanged warm evaluation | 6/6 message hits |
| One factor edit in one of two disconnected triangles | 3 dependent misses; 3 unaffected-component hits |
| Result agreement for that edit | Incremental = cold = direct enumeration |
| Reordering original factor list | 6/6 message hits |
| Factor-scope edit | Recomputed safely; exact agreement |
| Forged/rehashed payload without valid HMAC | Rejected and recomputed |
| Cache copied under wrong auth key | Entries rejected and recomputed |
| Cross-algebra cache namespace | 0 cross-algebra hits |
| Randomized mixed-cardinality updates | 64 graphs × 2 algebras = 128/128 exact matches |
| Randomized edits: cache hits / misses | 509 / 259 |
| Randomized edits: invalidated message steps | 253 |

Across the 128 randomized edits, the cached producer executed 1,964 additions and 3,804 multiplications versus 7,098 additions and 13,136 multiplications for cold recomputation. This is a reduction of roughly 72.3% and 71.0% in **producer operation counts** on this specific fixture/seed. It is not wall-time speedup: Python object/digest/HMAC costs, fresh-message checker replay, serialization, cache memory and public benchmark behaviour are not included in that comparison.

The checker is a separate implementation path in the same Python reference file, not yet a separately compiled/audited binary. The finite suite is not a Lean proof.

## Exact supported algebra catalogue

- Natural counting over unbounded Python integers.
- MaxCount with score, weighted multiplicity and distinct assignment support.

The MaxCount reference assumes every original factor table cell represents exactly one local assignment. A future API that accepts pre-aggregated factors with overlapping hidden assignment provenance must carry provenance or reject distinct-support outputs.

The cache namespace includes algebra ID, but an algebra ID string is not proof of its laws. Production acceptance must use a closed, formally qualified algebra catalogue until arbitrary algebra definitions can provide verifiable law evidence.

## Benchmarks and SOTA comparison protocol

Primary baseline: PBCount2 on exactly matched projected/unprojected PB instances and update sequences. Also compare cold recomputation, unchecked in-memory caching, receipt-checked caching, preprocessing-disabled ID-stable caching, and content-addressed inprocessing with transformation metadata. For graphical-model inputs, compare junction-tree/message-reuse systems.

Per update, report:
- solved updates and exact result agreement;
- producer CPU/wall time, checker time, HMAC/serialization time;
- cache hits, misses, rejected entries and invalidated messages;
- algebra additions/multiplications avoided;
- cache bytes, peak RSS and retained stale-entry growth;
- trace/certificate bytes and checker identity;
- median, p90/p99 latency and full update-sequence cactus plots.

Frozen update strata:
- local factor-cell/weight changes;
- factor addition/removal;
- scope change;
- variable-cardinality change;
- variable/factor reordering;
- preprocessing and inprocessing rewrites;
- reverted edits and repeated edits;
- disconnected and articulation-separated components;
- tampered, stale, cross-algebra and cross-checker cache objects.

Do not compare different projections or different output contracts. PBCount2 is a counter, not a MaxCount/Pareto sampler. Use shared representable semantics only.

## Qualification gates — aspirational, not achieved

- 100% incremental/cold/direct agreement for every supported algebra on tractable frozen cases.
- Zero stale/corrupt/unauthenticated message reuses.
- 100% correct invalidation on changes to factor content, scope, domains, algebra semantics or checker identity.
- Held-out public PBCount2-compatible incremental sequences.
- At least 2× lower median per-update time than cold recomputation on a preregistered low-change slice **including checking and authentication costs**.
- Bounded cache memory and explicit eviction/garbage-collection semantics.
- Publish hit-rate, invalidation and failed-case curves; no aggregate-only claim.

## Next implementation steps

1. Port the canonical key and message cache to Rust; use cryptographic content digests and canonical encoding, not `DefaultHasher`.
2. Make the producer and checker separate crates/binaries; formalize the bucket-message preservation theorem.
3. Integrate ETH-016 certificate receipts and ETH-017 algebra-aware order plans.
4. Add preprocessing transformation provenance so normalization can preserve reuse when semantics are unchanged.
5. Benchmark against PBCount2 on matched public incremental sequences, reporting all producer/checker/authentication costs.
