# HCP-MMP1 neuromaps Lineage-B Evidence Verifier v1

Status: **archival-verification mechanism qualification only; no real Lineage-B evidence bundle is qualified by this profile**

This profile defines a non-executing verifier for evidence produced by the candidate Lineage-B transform introduced in PR #523.

Its core authority rule is:

`internal self-hash != retained evidence authority`

A bundle is admissible only when its stored content digest matches a SHA-256 root supplied from outside the bundle and the bundle re-binds to the exact method, run, semantic namespace, **generator implementation**, Workbench identity, output artifacts, scientific-input commitment, and independence contract.

## Separation of powers

- the Lineage-B derivation mechanism may produce candidate evidence;
- this archival verifier may validate retained evidence;
- it does not execute Workbench or replay the transform;
- mechanism-execution proof remains a separate Evidence Plane / replay theorem;
- FMQ-010 cross-lineage comparison remains a separate theorem.

No one layer may mint the authority of another.

## Required external roots

Verification requires the expected evidence `content_digest` plus the exact method manifest, run manifest, canonical HCP-MMP1 area-order file and left/right semantic outputs.

The verifier recomputes and cross-checks method/run/area SHA-256 roots, the generator implementation map and aggregate, scientific-input commitment, Workbench roots, output hashes and semantic provenance.

## Generator implementation rebinding

The evidence bundle carries a closed-world map of exact SHA-256 roots for the three Symthaea modules that produced Lineage B:

- `hcpmmp_neuromaps_common.py`;
- `hcpmmp_neuromaps_gifti.py`;
- `derive_hcpmmp1_neuromaps_lineage_b.py`.

The verifier recomputes the aggregate implementation digest from that map, feeds that exact digest into reconstruction of the scientific-input commitment, and requires both semantic outputs to carry the same `generator_implementation_digest`.

Changing a per-module root, an aggregate root, or an output's generator root cannot regain validity merely by recomputing the outer evidence self-hash.

This proves implementation identity consistency, **not** that those implementation bytes are scientifically correct. Code qualification/review remains a separate theorem.

## Semantic rebinding

For each semantic output the verifier requires the exact schema/space/hemisphere, 10,242 vertices, only canonical hemisphere-prefixed HCP-MMP1 labels or `null`, complete 180-area coverage, lineage/hemisphere source id, scientific-input source digest, exact generator id/version, exact generator implementation digest and exact terms reference.

## Independence boundary

The complete independence object is closed-world and must equal the method manifest's independence contract plus:

`independence_established = false`

`status = requires_external_provenance_review`

Unknown authority fields are rejected even after evidence re-hashing.

## Qualification gates

The dependency-free synthetic suite covers ten contracts:

1. valid retained-root bundle;
2. wrong external retained root;
3. unknown independence authority after re-hashing;
4. noncanonical semantic substitution after re-hashing;
5. semantic source/scientific-commitment substitution;
6. run-manifest rebinding;
7. Workbench-root substitution;
8. output-digest mismatch;
9. self-hash corruption;
10. generator per-module map tamper after outer re-hashing.

The dedicated CI additionally requires Python compilation, the complete verifier suite, CLI availability, no Workbench execution/replay or network acquisition surface, and explicit generator-provenance authority ratchets.

## Non-goals

Successful archival verification does not establish authorized source acquisition, actual Workbench execution, independent execution provenance, scientific correctness of the generator code, atlas correctness, FMQ-010 satisfaction, empirical neural alignment, consciousness evidence, or benchmark de-quarantine authority.
