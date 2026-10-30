# Broca proof-candidate admission boundary

**Status:** reference adapter; Rust tests authored but not yet compiled or executed.

## Why this boundary exists

A lexical retriever is a candidate generator, not an authority source. BM25 can rank a matching passage, but lexical relevance cannot establish that the passage:

- belongs to the exact immutable source revision;
- was known at the requested transaction-time coordinate;
- was world-valid at the requested valid-time coordinate;
- is not retracted or superseded at the query's knowledge time;
- is authorized for the requesting principal and scope;
- is not contradicted by another admissible source;
- has an exact token cost under the pinned renderer and tokenizer.

Treating a high BM25 score as proof admission would collapse retrieval relevance, provenance, authorization, temporal validity and proof sufficiency into one ungrounded trust decision.

## Two-phase architecture

1. **Candidate retrieval.** \`EvidenceIndex::search_bm25\` provides deterministic ranked text hits and the content digest already checked at ingestion. Rank is a diagnostic ordering signal only.
2. **Trusted admission.** A bitemporal source resolver, authorization adapter, contradiction resolver and pinned token renderer produce one \`ProofCandidateAdmission\` per retrieved evidence ID. The envelope carries source revision ID, source family, scope, index snapshot \`known_at\`, source \`recorded_at\`, \`retracted_at\`, valid-time interval, principal authorization decision/receipt digest, contradiction edges and exact rendered block token cost.
3. **Admission join.** \`admit_lexical_hits_for_proof_budget\` joins each hit to exactly one admission envelope. It rejects missing/unused/duplicate admissions, duplicate HDC IDs, malformed source pointers, cross-scope metadata, non-finite BM25 scores, content-digest mismatches and mismatched snapshot coordinates. It never estimates missing metadata.
4. **Exact proof selection.** The existing OR-of-AND proof optimizer performs the final temporal, scope, authorization, contradiction, independent-source-family and token-budget checks. Ineligible candidates remain in the request so the solver can return explicit rejection reasons.
5. **Verified reader hand-off.** The \`memory_verified_reader_context\` adapter independently verifies the selected proof packet and exposes only the exact packet excerpts to the reader. Provenance and selected claim-proof edges remain separate control-plane metadata.

## Non-negotiable invariants

- \`index_snapshot_known_at_unix_ms == request.known_at_unix_ms\`. A current index cannot silently answer a historical knowledge query. The caller must materialize/rebuild the EvidenceIndex for the exact historical snapshot before retrieval.
- \`admission.scope_id == request.scope_id\`. Cross-scope candidates are refused before candidate construction.
- \`recorded_at\`, \`retracted_at\`, \`valid_from\`, and \`valid_until\` are copied from the trusted admission envelope, never inferred from BM25 rank or a chat timestamp.
- \`authorization_accepted\` must be set only after a trusted authorization adapter evaluates the receipt/policy. A receipt SHA-256 is an integrity reference, not a signature or proof of authority.
- \`token_cost\` must come from the exact pinned packet renderer and tokenizer, including locator/envelope overhead. Character count, BM25 score or a generic tokenizer estimate is not a substitute.
- The source-span digest is recomputed from the exact UTF-8 text returned by the EvidenceIndex. The current adapter uses the full text payload (\`0..len\`); subspan extraction requires a distinct trusted source-span resolver.
- \`conflicts_with\` must come from a trusted contradiction resolver and be symmetric; the proof request validator checks the graph before solving.
- Search rank is one-based and preserved as diagnostic metadata. It is never used as the proof sufficiency objective.

## Historical index limitation

The current \`EvidenceIndex\` is an in-memory index whose ordinary BM25 method filters using its present lifecycle state. Passing a \`known_at\` value into the new adapter does **not** make a present-day index historically correct. The adapter therefore requires an explicit snapshot-time binding and documents that the caller must provide a matching historical index materialization. A future persistence integration must rebuild or query the evidence index from the append-only bitemporal source ledger at the exact \`known_at\` coordinate, with deterministic state-digest parity.

This is deliberately fail-closed rather than pretending that a current tombstone/lifecycle map can reconstruct a former state.

## Qualification tests

The reference adapter includes tests for:

- exact hit/admission join and successful solver admission;
- index snapshot \`known_at\` mismatch;
- missing and unused admission envelopes;
- duplicate evidence IDs and HDC IDs;
- non-finite BM25 score rejection;
- corrupted returned-text digest rejection;
- cross-scope admission rejection;
- preserving unauthorized evidence as a solver rejection;
- future, retracted and expired evidence rejection by the solver;
- refusing to replace a pre-populated candidate list silently.

These tests establish adapter conformance only. They are not LoCoMo, LongMemEval, MemDerive, SciValidityBench or vendor-comparison scores.

## Future work

1. Add a deterministic EvidenceIndex snapshot digest and a signed admission receipt that binds the index digest, bitemporal cutoff, principal, policy, tokenizer and renderer.
2. Add a historical EvidenceIndex materializer driven by the append-only source ledger, with exact index rebuild parity tests.
3. Replace externally supplied conflict edges with a versioned contradiction graph whose false-positive/false-negative rates are benchmarked.
4. Add exact renderer-tokenization receipts and per-query accounting.
5. Integrate the admission join, exact proof solver and verified reader hand-off into the actual LongMemEval/AML generation adapter.
6. Benchmark candidate recall separately from proof admissibility and end-to-end answer accuracy.

No source authenticity, cryptographic authorization, natural-language entailment, historical index replay, production deployment or SOTA result is claimed by this adapter.
