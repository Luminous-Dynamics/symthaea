# MATH-RET-SOURCE-COVERAGE-001A — Common Representation Coverage Contract

Status: draft qualification contract. Authority: `MeasurementOnly`.

## Problem

A shared `candidate_set_sha256` is necessary but not sufficient for a controlled retrieval experiment.

Two index manifests can commit to the same frozen candidate universe while their builders silently differ after parsing, normalization, encoding, or serialization. For example, an ExactNormalForm builder could omit unsupported formulas while Syntax/HDC keeps them. The resulting S/H/N/F comparison would then change both representation **and population**.

This contract freezes the missing theorem:

```text
frozen candidate set
    == sources accepted by the shared parser
    == sources accepted by the frozen normalizer
    == sources materialized for every required representation target
```

under `CommonIntersectionRequired`.

## Artifact

`math-retrieval-source-coverage-v1` binds:

- exact candidate-set digest/count;
- corpus snapshot, knowledge boundary, and candidate-eligibility policy;
- source-object contract;
- parser contract + implementation + canonical `FolFormulaExt` serialization;
- normalizer contract + implementation + canonical ExactNormalForm serialization;
- every required representation target;
- every candidate source in canonical digest order;
- the parsed and normalized object identities for that source;
- one ready representation object per target, including exact serialized-byte count.

Each representation target freezes:

```text
target_id
channel
representation_family
representation_sha256
item_serialization_sha256
input_stage
max_serialized_item_bytes
```

`input_stage` makes the data dependency explicit:

```text
SourceObject          -> lexical syntax representation
ParsedFolFormulaExt   -> structural syntax representation
ExactNormalForm       -> normal-form representation
```

## Common-intersection law

A qualified artifact permits no per-arm missingness.

For every source in the candidate set:

```text
parse_status == Parsed
normalization_status == Normalized

for every required target:
    status == Ready
    serialized_bytes <= frozen target byte ceiling
```

If a source is unsupported by a required parser, normalizer, encoder, or serialization ceiling, it cannot remain in a `CommonIntersectionRequired` candidate set. The candidate eligibility policy must be rebuilt/frozen before evaluation rather than silently dropping the source in one arm.

An unsupported formula is **not false mathematics**. It is a representation/coverage limitation and has no truth authority.

## Exact candidate binding

The validator consumes:

1. the coverage artifact;
2. the exact MATH-RET-CANDIDATE-001A candidate-set file;
3. the exact math-search experiment manifest.

It requires the SHA-256 of the candidate-set bytes to equal `candidate_universe.candidate_set_sha256`, then requires coverage rows to equal the candidate list exactly and in exactly the same order.

It also binds the coverage artifact back to the experiment shared contract for:

```text
corpus_snapshot_sha256
knowledge_boundary_sha256
source_object_contract_sha256
normalization_contract_sha256
normalization_implementation_sha256
```

## Representation coverage

Targets are sorted by `target_id`; rows are sorted by `SourceObjectDigest`; each row's representation entries must exactly match the frozen target order.

This forbids:

- silent target omission;
- duplicate sources;
- duplicate targets;
- per-arm population shrinkage;
- replacing one target with another under the same row;
- serialized representation objects above the frozen byte ceiling;
- binding coverage to a different candidate-set file with the same claimed count.

The representation object digest is a content identity for the materialized representation item. It is not a relevance score and does not authorize retrieval.

## Intended Q0/Q1 targets

The first full experiment is expected to declare at least the representation targets corresponding to:

```text
L  Syntax / Lexical / SourceObject
S  Syntax / CanonicalSparse / ParsedFolFormulaExt
H  Syntax / HDC / ParsedFolFormulaExt
N  ExactNormalForm / CanonicalSparse / ExactNormalForm
```

F reuses the qualified S and N populations. SHUF derives from H; PERM derives from its frozen parent representation. Those controls must not gain a different source population.

The contract is generic and does not hard-code these IDs or encoder hashes; those exact identities belong to the experiment lineage.

## What this contract does not prove

A PASS does not prove:

- that an index artifact actually contains the declared representation objects;
- that ranking/scoring is correct;
- that a retriever is useful;
- that HDC is superior;
- that normal forms are useful;
- mathematical equivalence or theorem truth;
- proof success or novelty.

The next build-receipt tranche must bind these ready representation objects into the exact index artifact. Runtime membership/trace/payload qualification remains independently enforced by MATH-RET-RUNTIME-001A..001E.

## Required adversarial canaries

The validator self-test rejects at least:

- omitted candidate row;
- reordered candidate rows;
- duplicate source identity;
- omitted representation target from one source;
- non-ready representation row;
- representation byte size above the frozen target ceiling;
- candidate-set digest substitution;
- normalizer identity substitution;
- non-canonical target order.

## Integration discipline

This contract is intentionally a child of the guarded-runtime stack but does not modify #4338, #4352, #4357, or #4365.

It also does not copy the example-local structural encoders from #4087/#4112. Actual representation object production remains downstream of the convergence gate in #4367. Once the frozen encoders are extracted without behavior drift, this coverage contract becomes the population-equality gate before building real S/H/N indices.
