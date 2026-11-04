# Evidence-qualified connectome correspondence protocol V0.1

**Receipt schema:** `research/connectomics/connectome-correspondence-receipt-v0.1.schema.json`  
**World manifest schema:** `research/connectomics/connectome-world-artifact-manifest-v0.1.schema.json`  
**Research issue:** [#10238](https://github.com/Luminous-Dynamics/symthaea/issues/10238)

## 1. Purpose and non-goals

This protocol defines a typed partial-correspondence receipt between two versioned connectome worlds whose segmentation, mesh, skeleton, neuron-root and synapse object IDs may differ.

It is **not** a universal graph-isomorphism test, nor a claim that a topological match establishes biological identity. It is an evidence-qualified relation between objects in two named source releases.

V0.1 is a schema and synthetic contract only. It does not yet implement a biological correspondence engine or an executable receipt verifier.

## 2. Identity and correspondence vocabulary

The following relations are not interchangeable:

- **Artifact identity:** exact byte identity of a source artifact or annotation release.
- **Segment lineage:** source-authoritative edit-history relation between segmentation objects across releases. It can be one-to-many or many-to-one.
- **Neuron correspondence:** an evidence-backed relation between biological neuron identities. It is not inferred solely from segment IDs, degree sequences or abstract graph isomorphism.
- **Synapse correspondence:** an independent relation between synapse candidates and their pre/post partner evidence. It is not inherited automatically from neuron correspondence.
- **Structural representation identity:** equality under one exact, named, versioned representation canonicalizer. Connectome-universe canonicalizer V0.1 remains ID-preserving.
- **Query-state equivalence:** the same frozen analysis query yields equal outputs across different graph representations. This does not imply the structures are identical.

A single pair of worlds can be query-equivalent while their segmentation graphs are not representation-identical. A biological neuron can remain the same lineage while its segment partition and IDs change. Two unrelated neurons can have isomorphic abstract topology.

## 3. Required inputs

Each world manifest pins:
- world, dataset, source-volume and release identifiers;
- a complete object inventory for the object classes in scope: SEGMENT, NEURON, SYNAPSE and ARTIFACT;
- exact segmentation SHA-256 and optional mesh / skeleton / synapse annotation SHA-256;
- coordinate-frame identifier and voxel spacing in physical nanometres;
- object inventory with object type, object ID, optional lineage ID and optional stable anchor ID.

The correspondence receipt pins:
- exact source and target manifest byte digests;
- validated physical registration transform and residual / uncertainty bounds;
- correspondence policy, engine, verifier and schema digests;
- mapping statements, evidence references and independent support status;
- query comparisons under frozen query / analysis digests, reported separately from mapping statements.

A source or target manifest digest is the SHA-256 of the exact sidecar bytes, not a producer-authored string. The verifier must recompute it and compare the manifest's embedded world identity and object inventory with the receipt.

## 4. Partial mapping and cardinality

Every mapping statement contains source IDs, target IDs, relation type, cardinality, support status, evidence list and rationale code.

Cardinality rules:
- `ONE_TO_ONE`: exactly 1 source and 1 target.
- `ONE_TO_MANY`: exactly 1 source and at least 2 targets.
- `MANY_TO_ONE`: at least 2 sources and exactly 1 target.
- `MANY_TO_MANY`: at least 2 sources and at least 2 targets.
- `UNMAPPED`: one or more sources and exactly zero targets.

The cardinality label must be recomputed from source/target array lengths and must match exactly. The schema and semantic verifier both reject mismatches. For V0.1, every source object in the source manifest inventory must occur in at least one mapping; objects with no target correspondence must be declared by an explicit UNMAPPED row. Multiple mapping rows may reference the same source only when they represent distinct typed relations or explicit unresolved alternatives; the verifier does not infer identity by picking one.

Do not force split / merge lineage into bijections. A partial mapping can leave objects unresolved; every unmapped or unresolved source object is counted. Do not silently discard ambiguous alternatives.

Every source and target object ID must exist in the corresponding pinned world manifest. Every target object may be consumed by at most one mapping row; many-to-one must be represented inside one row, not simulated with repeated target claims. Unmapped target objects are allowed and should be reported separately in a future target-coverage report. The semantic verifier rejects target reuse across mapping rows. A `NEURON_CORRESPONDENCE` mapping must reference neuron objects; a `SYNAPSE_CORRESPONDENCE` mapping must reference synapse objects; `SEGMENT_LINEAGE` must reference segment objects; `ARTIFACT_IDENTITY` must reference artifact objects. A source/target relation with mismatched object types is rejected.

A `SEGMENT_LINEAGE` relation does not itself assert neuron identity. A neuron correspondence needs independent identity evidence (for example a stable source anchor or expert adjudication) under the declared evidence policy.

## 5. Evidence and status semantics

Evidence kinds are typed and each points to an immutable artifact digest and locator. Authority is recorded but not trusted merely because a field says SOURCE_DERIVED or INDEPENDENTLY_VERIFIED.

- `CLAIMED_VERIFIED`: a producer's declared status only. It is not an authenticated verification result. A future verifier may promote a mapping only after the referenced artifacts and authority chain have been authenticated, all mapping constraints hold, and the policy's required independent evidence is satisfied.
- `AMBIGUOUS`: multiple correspondences survive the admissible evidence; preserve the alternatives and do not choose a best match silently.
- `NOT_ESTIMABLE`: required source artifacts, registration support or independent anchors are absent or insufficient.
- `REJECTED`: evidence contradicts the mapping or a hard constraint fails.

The V0.1 schema does not authenticate evidence. Until that verifier exists, synthetic fixture statuses are expected outcomes of the test protocol, not validated biological claims.

## 6. Physical registration

Coordinate transforms are explicit and directional. V0.1 fixes column-vector convention, nanometre coordinates, and source-to-target multiplication: `p_target = T_source_to_target * p_source`. The 4×4 matrix is affine, and its final row must be `[0, 0, 0, 1]` within `1e-9`; its upper-left 3×3 determinant must be finite and have absolute value greater than `1e-12`. The source-to-target transform, coordinate-frame policy, physical voxel spacing, maximum observed residual and uncertainty bound must be present when status is VERIFIED.

- Use physical units (nm / µm), never voxel-index distances without anisotropy conversion.
- A transform with unknown or unacceptable uncertainty cannot support identity.
- Registration similarity is supporting evidence, not proof of identity.
- Reflections, axis swaps and anisotropic scale changes must be explicitly allowed or rejected by the pinned coordinate-frame policy.

V0.1 carries a 4×4 transform in the receipt schema and the above convention is normative. The future semantic verifier must check the matrix, recompute physical residuals from actual anchor coordinates and the pinned coordinate-frame policy, and authenticate the evidence artifacts. It must not trust a caller's `max_residual_nm` value alone. The V0.1 schema/fixture validator only checks matrix form; it does not authenticate the transform or certify registration.

## 7. Query equivalence

Query comparisons are separate from mappings. Each binds:
- query definition digest;
- analysis implementation digest;
- source output digest;
- target output digest;
- `SAME`, `DIFFERENT` or `NOT_ESTIMABLE`.

A SAME query output does not turn an ambiguous neuron mapping into verified identity. A query whose output depends on the unresolved mapping must remain NOT_ESTIMABLE or carry all admitted outcomes.

## 8. Required synthetic controls

1. Valid segment split lineage (one source segment to two target segments), with neuron correspondence separately represented.
2. Valid segment merge lineage (two source segments to one target segment), without collapsing neuron identity.
3. Isomorphic graph shapes but different source anchors: neuron identity remains AMBIGUOUS.
4. Query outputs SAME across different representations: record query-state equivalence only.
5. Same object IDs but different source volume/release: artifact identity does not transfer automatically.
6. Synapse presence matched but pre/post partners swapped: synapse correspondence rejected / ambiguous.
7. One-to-many mapping labelled one-to-one: reject cardinality mismatch.
8. Mapping references an object absent from source/target manifest: reject.
9. Duplicate mapping IDs, duplicate object IDs, cyclic lineage, stale transaction receipt or sidecar hash mismatch: reject.
10. Physical transform with axis reflection / wrong handedness / voxel anisotropy mismatch: reject or NOT_ESTIMABLE under pinned policy.
11. Multiple plausible neuron correspondences: preserve ambiguity and query-consequence union; no top-score collapse.
12. Rollback preserves historical lineage event while current object state points to the post-rollback release.

## 9. Metrics and qualification targets

Measure separately:
- artifact manifest byte-integrity pass rate;
- object inventory coverage and orphan mapping count;
- lineage precision / recall by split, merge, continuation and rollback;
- cardinality-constraint violation rate;
- neuron identity false-positive rate (critical metric);
- neuron identity precision / recall on independent adjudicated pairs;
- synapse correspondence and pre/post polarity error rates;
- ambiguous / NOT_ESTIMABLE rate by evidence-support stratum;
- physical registration residual and uncertainty coverage;
- query-state equivalence, query-state disagreement and query NOT_ESTIMABLE rates;
- runtime and memory by object count and ambiguity set size.

Proposed targets are qualification goals, not results:
- zero false identity assertions in exhaustive synthetic negative controls;
- 100% preservation of one-to-many / many-to-one alternatives in known-answer controls;
- 100% rejection of object membership, cardinality and stale-manifest mutations;
- at least 99% lineage precision and 95% lineage recall on sufficiently powered, independently adjudicated relation strata, with cluster-aware lower bounds;
- zero query-equivalence promotion when the output depends on unresolved correspondence;
- independent receipt replay yields deterministic output for the same pinned evidence.

## 10. Claim ceiling

A V0.1 receipt schema and synthetic fixtures demonstrate only that the contract can represent typed partial correspondence and prevent category errors. They do not validate source evidence, authenticate edit lineage, establish neuron identity, or prove anatomical equivalence. Biological correspondence results require a later independent verifier and adjudicated real-world evaluation.

## 11. V0.1 executable contract and claim ceiling

The schema is structural. A semantic verifier must independently check sidecar raw-byte hashes, embedded world identity, object inventories, mapping IDs, evidence IDs, object membership, relation/object-type compatibility, cardinality, explicit source coverage, summary counters and query-output relationships.

The mapping status enum uses `CLAIMED_VERIFIED` rather than `VERIFIED` to make its untrusted nature explicit. The expected output of the first verifier is `COUNTERS_RECOMPUTED` with `trusted_promotion=false`; it reports support-status labels under `support_status_claim_counts`. It must not treat `support_status=VERIFIED`, `authority=INDEPENDENTLY_VERIFIED`, `registration.status=VERIFIED`, or `disposition=CORRESPONDENCE_VERIFIED` as authenticated truth. It has no evidence-artifact adapter, signer trust-root, source edit-history verifier or physical-anchor residual verifier. It therefore reports claimed statuses separately and keeps the actual receipt disposition NOT_ESTIMABLE unless a hard structural contradiction requires REJECTED or an explicitly preserved ambiguous alternative is present.

A mapping row count is not a population metric. Mapping precision/recall targets require an independently adjudicated object-pair reference set, clustered by source volume / neuron / acquisition lineage. A source object repeated in alternative mappings must not inflate the denominator.

The V0.1 schema includes an explicit ARTIFACT object type so artifact identity is not falsely represented as neuron or segment identity.
