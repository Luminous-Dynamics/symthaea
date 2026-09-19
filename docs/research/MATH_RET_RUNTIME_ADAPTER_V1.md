# MATH-RET-RUNTIME-001B — Frozen Contract Adapter Qualification

Status: staged child qualification of MATH-RET-RUNTIME-001A.

## Question

Can the typed owner-independent Rust execution seam produce a deterministic
runtime receipt and canonical payload materialization that are accepted by the
already-frozen MATH-RET graph, trace, and payload-audit validators without a
post-hoc reinterpretation of ranking or packing?

This is an interoperability/evidence question, not a retrieval-quality claim.

## Boundary

The pipeline is:

```text
normative validator fixtures
        |
        v
real content-addressed graph files
        |
        v
MATH-RET-001D v1.1 graph qualification
        |
        v
runtime config containing exact qualified digests
        |
        v
typed Rust RetrievalExecutor (arm S)
        |
        +-- exact MATH-RET-TRACE-001A JSON
        |
        +-- exact materialized canonical payload bytes
        |
        v
stdlib content-addressing adapter
        |
        +-- hash exact trace bytes
        +-- hash exact payload bytes
        +-- assemble MATH-RET-PAYLOAD-001A paths/hashes
        |
        v
frozen trace validator
        |
        v
frozen payload-audit validator
```

Python is not allowed to choose/reorder retrieved identities or redo packing.
Those decisions come from the Rust seam. Python supplies only repository
fixture construction and content-addressing/path fields needed by the frozen
file-oriented contracts.

## Full graph fixture

The adapter starts from the normative MATH-EXP v2.1 fixture and constructs the
complete preregistered Q0 arm set:

```text
A     baseline
L     lexical control
R     random-retrieval control
S     canonical sparse Syntax
H     HDC Syntax
N     canonical sparse ExactNormalForm
F     conventional Syntax + ExactNormalForm fusion
SHUF  shuffled-HDC control
PERM  permuted-association control
```

All retrieval arms share one exact candidate universe and one exact context
packer. The F arm reuses the exact S and N indices, so the graph does not invent
special fusion-only representations merely to satisfy the test.

The fixture contains:

- one experiment manifest;
- one shared context packer;
- seven single-channel index manifests;
- one fusion v1.1 policy;
- eight retrieval bindings;
- one graph bundle resolving every artifact by path + exact SHA-256.

Every artifact is written as canonical compact JSON with a trailing newline and
its digest is computed from those exact bytes.

## Runtime arm

The first adapter qualification targets arm **S** only.

That is intentional. A single-index execution is sufficient to test the new
interoperability boundary:

```text
qualified graph identity
 -> typed Rust request
 -> exactly one retrieval query
 -> ranked source-object identities
 -> shared canonical materializer
 -> rank-preserving whole-item packing
 -> atomic runtime evidence
 -> frozen trace validator
 -> actual payload-byte audit validator
```

Fusion already has independent Rust/Python replay canaries. An F-arm adapter is
a later one-variable extension after this base bridge qualifies.

## Rust fixture emitter

`src/bin/emit_fixture.rs` is a std-only binary using the public
`RetrievalExecutor` API. It receives exact graph/budget identities through a
simple `key=value` config produced only after graph qualification.

The backend returns exactly three source-object identities. It cannot provide
payload bytes. One `CanonicalSourceMaterializer` resolves those identities to
canonical UTF-8 payloads.

The binary writes:

- `trace.json` in the exact frozen MATH-RET-TRACE-001A shape;
- canonical payload files;
- `payload-plan.tsv`, a non-authoritative bridge containing role/rank/source
  identity/path/claimed byte count from the typed execution.

The trace's normalized compute value is rendered from fixed-point microunits to
a canonical decimal string without floating point.

## Stdlib adapter

`qualify_contract_adapter.py` performs only the steps that require file-byte
identity:

1. generate/write the complete contract-valid graph fixture;
2. run MATH-RET-001D v1.1 and save its exact report bytes;
3. provide those exact digests/budgets to the Rust fixture emitter;
4. run MATH-RET-TRACE-001A on the Rust-emitted trace;
5. read the exact materialized payload files;
6. independently check each byte count from the Rust payload plan;
7. compute each payload SHA-256 and the exact trace SHA-256;
8. assemble MATH-RET-PAYLOAD-001A;
9. run the frozen payload-audit validator;
10. write a compact qualification summary of the exact evidence identities.

The adapter never supplies retrieval scores, changes order, skips candidates,
or recalculates the packer's decisions.

## Dedicated workflow

`.github/workflows/math-retrieval-runtime-adapter.yml` uses Rust 1.96.0 and:

```text
cargo fmt/check/test --locked  (requalify parent seam)
python py_compile             (adapter syntax)
graph v1.1 self-test
trace v1 self-test
payload-audit v1 self-test
full contract-adapter qualification
upload exact generated evidence directory
```

The evidence artifact is retained for 30 days and keyed by the exact Git head.

## Promotion law

A passing workflow demonstrates only:

- the Rust seam compiles/tests on the pinned runner;
- the complete generated graph satisfies frozen contract semantics;
- the Rust-emitted trace satisfies the frozen trace validator;
- the materialized payload evidence satisfies the frozen payload-audit
  validator;
- those artifacts all refer to one exact content-addressed lineage.

It does not establish retrieval relevance, HDC advantage, normal-form advantage,
fusion benefit, theorem truth, or novelty.

## Next gate

After this adapter qualifies, add an **F-arm fixture** using the same graph,
materializer, packer, and evidence adapter. The only new runtime intervention
should be the already-frozen two-channel fusion path.

Only after both S and F runtime receipts qualify should a real structural
Syntax/HDC/normal-form index implementation be connected to
`QualifiedRetrievalBackend`.
