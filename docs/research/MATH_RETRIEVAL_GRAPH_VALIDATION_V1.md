# MATH-RET-001D — Cross-Contract Retrieval Fairness Closure v1

Status: contract and validator for content-addressed retrieval graph qualification.

Authority: `MeasurementOnly`.

## Purpose

MATH-RET-001A/B/C freeze individual retrieval components. MATH-EXP-001 v2.1 binds an arm to a retrieval-binding digest. This tranche closes the remaining gap: proving that the **actual referenced files** form one fair retrieval experiment graph.

The key distinction is:

```text
manifest string says sha256:X
        !=
file bytes actually hash to sha256:X
        !=
referenced artifacts are causally fair across arms
```

MATH-RET-001D checks all three layers.

## Bundle as resolver, not authority

A digest cannot safely discover a repository path by itself. The bundle therefore maps exact content digests to repository-relative paths:

```json
{
  "version": "math-retrieval-graph-bundle-v1",
  "bundle_id": "...",
  "authority": "MeasurementOnly",
  "artifacts": [
    {
      "kind": "ExperimentManifest",
      "path": "...",
      "sha256": "sha256:..."
    }
  ]
}
```

The path is only a deterministic resolver. Scientific identity remains the SHA-256 of the exact UTF-8 JSON bytes.

The validator rejects:

- absolute paths;
- `.` / `..` traversal;
- paths escaping the selected repository root after resolution;
- missing/non-file paths;
- duplicate paths;
- duplicate digest aliases;
- byte/digest mismatches.

Exactly one `ExperimentManifest` is required.

## Delegated semantic validation

MATH-RET-001D does not reimplement the five predecessor contracts.

After byte verification, it delegates documents to the exact sibling validators:

```text
ExperimentManifest -> validate-math-search-experiment-v2.1.py
RetrievalBinding   -> validate-math-retrieval-binding.py
RetrievalIndex     -> validate-math-retrieval-index.py
FusionPolicy       -> validate-math-retrieval-fusion.py
ContextPacker      -> validate-math-retrieval-context-packer.py
```

The cross-validator is responsible only for graph resolution and invariants that exist **between** otherwise valid documents.

## Reachability law

Every artifact in a qualification bundle must be reachable from the experiment root.

```text
experiment
   -> retrieval binding
       -> index
       -> context packer

or

experiment
   -> retrieval binding
       -> syntax index
       -> exact-normal-form index
       -> fusion policy
       -> context packer
```

An unreferenced artifact causes failure.

This prevents a qualification package from containing alternate unused indices or policies that could later be selected opportunistically.

## Shared candidate universe

Every retrieval index participating in the experiment must share the exact candidate-universe signature:

```text
corpus_snapshot_sha256
knowledge_boundary_sha256
candidate_eligibility_policy_sha256
candidate_set_sha256
candidate_count
exclude_query_source_item
exclude_solution_artifacts
source_identity_kind
dedup_policy
canonical_candidate_order
```

Therefore representation and scoring may vary while **what is eligible to be retrieved** cannot.

Each candidate universe must also bind the experiment's exact corpus snapshot and knowledge boundary.

## Exact-normal-form binding

Every `ExactNormalForm` index must bind exactly the experiment root's:

```text
normalization_contract_sha256
normalization_implementation_sha256
```

A normal-form arm cannot silently use a newer or different normalizer than another arm.

## Single-index arm binding

For a single-index arm, the validator requires agreement among experiment arm, binding and index for:

- arm ID;
- representation family;
- representation channel;
- representation digest;
- control transformation;
- candidate universe;
- root corpus / knowledge boundary;
- exact normalizer identity where applicable.

The index must support at least the experiment's `retrieved_items_max` top-k ceiling.

`RandomRetrieval` is required to expose no mathematical representation channel.

`LexicalRetrieval` is represented as a lexical syntax index with no additional control transform.

## Fusion arm binding

For fusion arms, the validator resolves:

```text
Syntax index
ExactNormalForm index
Fusion policy
Shared context packer
```

It requires:

- exact binding/fusion input digest agreement;
- Syntax and ExactNormalForm channel roles;
- identical candidate universes across both input indices;
- each channel input quota <= its index's supported top-k;
- exact root/fusion output budget equality;
- exact fusion-output / packer payload-serialization equality;
- exact experiment/fusion-policy digest agreement.

`FusionHDC` requires both component indices to use HDC representation.

`FusionConventional` permits lexical or canonical-sparse Syntax, and requires canonical-sparse ExactNormalForm.

## Derived fusion representation identity

A single arm-level representation digest for a fusion arm is meaningful only if it is derivable from its components.

v1 freezes this identity function:

```text
bytes =
  "symthaea-fusion-representation-v1\n" ||
  "Syntax="          || syntax_representation_sha256 || "\n" ||
  "ExactNormalForm=" || normal_representation_sha256 || "\n" ||
  "FusionPolicy="    || fusion_policy_sha256         || "\n"

fusion_representation_sha256 = SHA256(bytes)
```

The experiment arm must carry exactly this derived digest.

This converts a previously opaque fusion identity into a mechanically reconstructable identity without introducing another artifact layer.

## Shared downstream context

All retrieval arms must bind the same exact context-packer digest.

The packer must bind the experiment's exact source-object contract and close exactly to the root retrieval budget:

```text
packer.max_output_items
    == experiment.retrieved_items_max

packer.max_output_bytes
    == experiment.retrieval_context_bytes_max

packer.max_output_item_bytes
    == experiment.retrieved_item_bytes_max
```

This preserves the MATH-RET-001C isolation law:

```text
representation may change WHICH source objects are selected
representation may not change WHAT mathematical bytes are supplied downstream
```

## Validation report

Successful validation emits one canonical compact JSON report:

```text
math-retrieval-graph-validation-report-v1
```

It records:

- experiment ID and exact digest;
- bundle digest;
- retrieval-arm count;
- artifact count;
- shared candidate-set digest/count;
- shared context-packer digest;
- normalization contract/implementation identities;
- closed retrieval budget;
- `all_checks_passed = true`.

The report remains `MeasurementOnly`.

Exact Git head, toolchain, command line and workflow/operator identity remain separate execution-lineage evidence and should be bound alongside the report bytes/digest.

## Commands

Pure cross-contract adversarial self-test:

```bash
python3 .github/scripts/validate-math-retrieval-graph.py --self-test
```

Actual bundle qualification:

```bash
python3 .github/scripts/validate-math-retrieval-graph.py \
  path/to/retrieval-graph-bundle.json \
  --repo-root . \
  --report retrieval-graph-report.json
```

The actual qualification path recomputes every listed SHA-256 from file bytes and delegates every artifact through its predecessor semantic validator.

## Self-test state

The exact source used for this tranche passed its pure cross-contract adversarial self-test before commit.

Canaries include:

- candidate-set drift between fusion inputs;
- root/context-packer budget inflation;
- exact-normalizer implementation drift;
- downstream payload-serialization drift.

Focused repository qualification is still required to exercise on-disk resolution and all delegated validators together.

## Deliberate nonclaims

MATH-RET-001D does **not** establish that:

- an index artifact was built from the declared candidate universe correctly;
- an approximate index actually meets its declared exact-recall floor;
- retrieval outputs obey the manifest at runtime;
- a retrieved neighbor is mathematically useful or equivalent;
- HDC beats conventional structure;
- exact normalization improves retrieval;
- fusion improves retrieval;
- any theorem is true, proved or novel.

It establishes that the content-addressed **manifest graph** is internally resolved and fair enough for later runtime evidence to be interpretable.

## Next gate

After focused execution of all contract validators, the next tranche should produce a runtime **retrieval trace/evidence receipt** per query containing only evidence-plane metadata:

- query/source-object digest;
- exact arm/binding/index/fusion/packer digests;
- candidate-set digest;
- ranked selected source-object digests;
- consumed items/bytes/compute;
- control seed/artifact where applicable;
- no theorem-truth authority.

That receipt will let experiment reports prove that runtime behavior actually respected the qualified graph, rather than merely possessing valid manifests.
