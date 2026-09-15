# HLS validity evidence-chain contract

The retained validity-memory evidence lineage is intentionally layered rather than collapsed into one summary artifact.

The qualified ordering is:

`v1 base evidence -> independent v1 verification -> v2 falsification evidence -> independent v2/base-link verification -> v3 seed-level summaries`.

The independent chain verifier is required to establish all of the following before v3 derivation is trusted:

- complete canonical SHA-256 chain verification for both retained v1 and v2 JSONL files;
- exact record grammar and observation counts for the requested frozen protocol;
- exact subject SHA and protocol in both headers;
- complete, clean, unchanged-subject footers with no scientific claim;
- v1 primary/score-path parity flag;
- v2 source-bundle verification and unchanged-source postflight flags;
- v2 fresh isolated v1 build and v1 hash-chain verification flags;
- exact equality between the v2 base commitment and the retained v1 file SHA-256, terminal record digest, record count, and observation counts.

V3 remains a deterministic descriptive transform of the verified per-seed v2 evidence. It does not replace v1 or v2 and does not make the summary artifact authoritative over the raw records.

A valid hash chain establishes internal integrity and lineage consistency, not a favorable scientific outcome. The qualified workflow therefore retains the raw artifacts and receipt together and continues to state `scientific_claim = none` until results are separately interpreted after the frozen experiment executes.
