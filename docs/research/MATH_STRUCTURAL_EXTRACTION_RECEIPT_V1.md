# MATH-RET-CONVERGENCE-001A — Structural Extraction Receipt v1

Status: contract / qualification support only  
Authority: `MeasurementOnly`

## Purpose

This contract turns the representation-extraction requirements from
`MATH-RET-INTEGRATION-001` (#4367) into a fail-closed machine-checkable receipt.
It does **not** extract either representation and does not qualify the frozen Q0
subjects by itself.

The eventual extraction is allowed to call itself `RelocationOnly` only if it
preserves the frozen structural-HDC and canonical-AST representations exactly.
Any representation-rule change is a new lineage.

## Frozen predecessor subjects

The contract binds these exact Git objects:

| Subject | Commit | Blob |
| --- | --- | --- |
| structural HDC example | `d52e08164ce44201dea3fa56b78dfb93dd35cfd3` | `35ab3f0d51dc7298d80836f5efec2878a73bb75d` |
| canonical AST example | `1f38f8d0a217df7fd01715602058234c85cfe58c` | `1b1253b976a0a918d149b0acbb0a85e65f9f1f39` |
| development fixture | `1f38f8d0a217df7fd01715602058234c85cfe58c` | `491ffb02d0540974345534f266b9976a6597e548` |
| preregistered holdout source | `b7619f80440d12d57bef5e24c3377b158ec5714d` | `ccd0715b1030b88b7ed1d82f52bfbb8ffd70eaac` |

Frozen representation identities remain:

- `symthaea-math-structural-hdc-v1`
- `canonical-ast-sparse-v1`

The prerequisite execution gate is exact subject
`694ef48e53296c2c6fe39a9c07927085b2f2754f`, workflow run `35459503395`, and
must conclude `success` before a relocation receipt can validate.

## Preparation environment

At preparation base `a884f7770f7df98a9a955881cf03e813d709d7de`, the
core dependencies used by the frozen examples and the package/lock boundary are
bit-identical to the old representation lineage. Those object IDs are recorded
in `math_structural_extraction_subject_v1.json`.

This is evidence about migration risk at preparation time, not a promise that
future `main` cannot change. If the environment changes before extraction, the
extraction still has to satisfy the behavioral compatibility receipt; the old
representation identities may not be silently redefined.

## Relocation-only law

A valid receipt requires all of the following:

1. prerequisite Q0 execution qualification actually concluded `success`;
2. the exact frozen predecessor source/fixture blobs are named;
3. HDC encoder ID remains `symthaea-math-structural-hdc-v1`;
4. sparse encoder ID remains `canonical-ast-sparse-v1`;
5. `representation_rule_changed == false`;
6. there is exactly one reusable implementation of each representation;
7. the old examples delegate to the reusable library implementation instead of
   retaining independent copies;
8. every frozen development case has identical predecessor/extracted HDC-vector,
   sparse-map, and ranking digests;
9. aggregate HDC pairwise similarity, sparse cosine, and tie-order transcripts
   are identical;
10. the holdout remains score/ranking blind during relocation;
11. the blind holdout representation digest is identical before/after;
12. at least one evidence reference binds the receipt to executable output.

Any failed equality makes the extraction fail. It is not permission to update a
predecessor expected value.

## Canonical bytes for compatibility digests

The extraction implementation must document and freeze deterministic byte
encodings before producing receipt hashes. At minimum:

- HDC vectors: dimension/order-preserving canonical bit/word serialization;
- sparse maps: UTF-8 key order sorted lexicographically, exact finite numeric
  counts serialized canonically;
- candidate rankings: ordered source/candidate IDs including deterministic tie
  order;
- pairwise transcripts: stable case ordering plus exact similarity values under
  the predecessor scoring implementation.

Do not hash debug formatting whose representation can change with Rust or
library formatting behavior.

## Holdout firewall

The extraction PR needs a compatibility witness for the preregistered holdout,
but it must not become the first scientific evaluation of that holdout.
Therefore v1 permits only a **blind representation digest**:

```text
frozen holdout source
  -> predecessor representation serialization
  -> domain-separated aggregate digest

frozen holdout source
  -> extracted representation serialization
  -> same domain-separated aggregate digest

require predecessor_digest == extracted_digest
```

Forbidden during extraction:

- similarity-score emission;
- positive-vs-negative ranking;
- top-k evaluation;
- accuracy/MRR/Recall computation;
- representation tuning from holdout behavior;
- changing labels/cases after observing any derived holdout result.

A later explicitly authorized holdout-evaluation tranche can score the frozen
holdout. This receipt does not authorize that evaluation.

## Authority separation

The receipt is `MeasurementOnly`. It proves relocation compatibility if all
checks pass. It cannot establish:

- theorem truth or proof validity;
- epistemic confidence;
- HDC advantage;
- canonical-AST superiority;
- proof-search improvement;
- production readiness;
- novelty.

The semantic validator rejects fields that attempt to smuggle several of these
authorities into the receipt.

## Files

- `data/benchmarks/math_structural_extraction_subject_v1.json` — frozen source
  identities and preparation environment.
- `.github/schemas/math-structural-extraction-receipt-v1.schema.json` — receipt
  interchange schema.
- `.github/scripts/validate-math-structural-extraction-receipt.py` — stdlib-only
  semantic validator and negative self-tests.
- `.github/workflows/math-structural-extraction-receipt.yml` — contract lane;
  verifies old Git objects, validator behavior, preparation-environment identity,
  and the holdout firewall.

## Required next sequence

This contract deliberately does not skip the existing gates:

```text
#4528 exact frozen representation execution
        ↓ success required
reconstruct on then-current main
        ↓
extract one reusable HDC + one reusable AST implementation
        ↓
produce MATH structural extraction receipt v1
        ↓ receipt PASS
real canonical sparse S backend
        ↓ qualified
real structural HDC H backend
        ↓ qualified
controlled evaluation / later fusion
```

The synthetic guarded runtime capsule and representation lineage remain separate
until these gates are satisfied.
