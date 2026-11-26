# MACE Source-Lineage Receipt Verifier V1

**Status:** Implemented consumer-side DSSE receipt verification contract and schemas on `mace/source-lineage-receipt-verifier`. Not a production source resolver or raw provenance-graph replay engine. Rust tests and build have not yet been executed.

## 1. Purpose

The DecisionSufficiency evaluator makes a requirement-level claim that multiple pieces of evidence belong to distinct independent source-root families. The consumer must not accept that claim merely because the evaluator signed a root count.

MACE therefore separates three signed roles:

1. **Evidence qualifier:** determines whether evidence is admissible.
2. **Decision-sufficiency evaluator:** assesses whether the exact qualified set supports the consumer's declared DecisionNeed.
3. **Source-lineage resolver:** independently resolves source ancestry, transformations, and relevant common-cause dependencies, then attests one root-family result per requirement.

The consumer checks that these authorities use different pinned trust profiles and different signer key IDs. Separate verifier labels alone are not sufficient operational separation; production still requires separate operators/process identities, credentials, storage and rollback domains.

## 2. Receipt and subject binding

Each source-lineage receipt is a DSSE v1.0.2 envelope carrying an in-toto Statement v1. Its subject names and hashes exactly one root-family identity:

`urn:luminous:mace:source-root-family:v1:<root_family_sha256>`

The closed predicate binds:
- run and query-invocation identity;
- exact qualified-evidence-set digest;
- exact consumer DecisionNeed digest;
- requirement ID and evidence-class digest;
- root-family identity;
- source-graph and transitive-lineage-closure digests;
- source-node count, dependency-edge count and unresolved-ancestor count;
- the resolver trust-profile digest, key ID, issuance time and expiry;
- status `independent`.

A receipt marked independent requires a non-zero source-node count and zero unresolved ancestors. Missing or unknown ancestry must never increase the independent-root count. A resolver's signed statement is still a claim about its graph analysis; the consumer does not infer source independence from a signature alone.

## 3. Exact receipt commitment

The DecisionSufficiency requirement contains one `provenance_receipt_sha256` per root-family commitment. This verifier computes SHA-256 over the exact UTF-8 bytes of the DSSE envelope JSON string supplied by the caller. It does not parse and reserialize the JSON before hashing.

Consequently, whitespace, key ordering or base64 representation changes alter the commitment even if the decoded statement is semantically equivalent. The evaluator must preserve and make retrievable the exact envelope bytes it committed to. This prevents a caller from silently substituting another receipt after sufficiency evaluation.

## 4. Verification and use-time protocol

`verify_source_lineage_receipts_v1` takes:
- the already qualified candidate evidence batch;
- the current qualification-use binding and current qualification verifier;
- the verified DecisionSufficiency result and its use binding;
- the current evaluator verifier;
- the consumer-owned DecisionNeed;
- the exact source-lineage envelope set;
- an out-of-band pinned resolver profile digest and resolver verifier.

It first revalidates qualification and sufficiency against their current profiles and a shared trusted timestamp. It then:
1. derives the exact set of requirement/root/envelope-digest tuples from the verified sufficiency predicate;
2. requires exactly one supplied envelope per committed tuple;
3. rejects missing, unexpected, duplicate or reused receipt digests;
4. checks exact raw envelope-byte digest;
5. checks the in-toto subject, predicate type and closed predicate shape;
6. checks exact run, invocation, qualified set, DecisionNeed, requirement, evidence class and root-family binding;
7. verifies the DSSE PAE signature under the separately pinned source-lineage role;
8. rejects a profile or key ID shared with qualification or sufficiency.

The returned `VerifiedSourceLineageSetV1` retains the original PAE/signatures and receipt digest. Its `validate_at_use` method revalidates qualification, sufficiency and all lineage signatures together using one current-time value and freshly supplied verifier profiles. No cached boolean is sufficient.

## 5. What this verifier proves—and does not prove

It establishes that:
- the exact source-lineage envelope bytes committed to by the sufficiency evaluator were provided;
- those envelopes are signed under the third pinned resolver profile;
- each envelope is bound to the same qualified evidence set and DecisionNeed;
- the resolver's signed claim concerns the exact requirement, evidence class and root-family identity;
- the three cryptographic authorities are separated by pinned profile digest and signer key ID.

It does **not** establish that:
- the resolver actually fetched the source bytes named by its graph;
- the source graph is complete or acyclic;
- every derivation parent and common-cause edge was found;
- two sources are genuinely independent in the real world;
- the source graph or closure digest corresponds to a correctly computed graph;
- the resolver is organizationally independent from the evaluator or qualifier.

These require an independently deployed source resolver and a separate graph replay verifier that fetches content-addressed source nodes and edge receipts, validates every digest, detects cycles/dangling/unknown ancestors, recomputes root-family partitions and common-cause collapse, and compares its result with the resolver receipt. That work remains future production scope.

## 6. Tests and adversarial requirements

The module includes deterministic DSSE fixtures and tests for:
- a valid resolver role and PAE signature;
- rejection of a wrong resolver role;
- exact envelope byte changes affecting the committed digest.

Required next tests include:
- missing and unexpected lineage envelopes;
- one receipt reused across two root families;
- evaluator/qualifier key reuse;
- a receipt for another qualified set, DecisionNeed, requirement or evidence class;
- subject/root-family rebinding;
- unresolved ancestors and non-independent status;
- expired receipt, revoked signer, profile rollback and rotation;
- malformed duplicate-key payload, unknown predicate fields, oversized payload and invalid digests;
- source graph replay with cycles, dangling edges, hidden common roots, omitted parents and deliberately false independence claims.

The module is not ready for consequential use until the raw source graph replay verifier and independent resolver deployment pass those cases.

## 7. Schemas

- `data/schemas/mace_source_lineage_receipt_v1.schema.json`
- `data/schemas/mace_source_lineage_statement_v1.schema.json`

The schemas describe the predicate and in-toto statement. Runtime code enforces exact field sets, digest and timestamp semantics, profile and signer verification, and DecisionNeed-specific bindings. JSON syntax parsing is not a substitute for a draft-2020-12 schema validator.
