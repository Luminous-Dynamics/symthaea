# MATH-RET-001C — Retrieval Binding + Canonical Context Packer v1

Status: contract layer connecting experiment arms to exact retrieval artifacts while keeping downstream payload identical across retrieval representations.

Authority: `MeasurementOnly`.

## Why two contracts are needed

MATH-EXP-001 v2.1 gives each retrieval arm one content-addressed edge:

`retrieval_binding_sha256`

MATH-RET-001A defines a single retrieval index.

MATH-RET-001B defines a two-channel fusion policy.

MATH-RET-001C resolves the dependency graph and freezes the final downstream payload policy.

```text
experiment arm
      |
      v
retrieval binding
      |
      +-- SingleIndex
      |      +-- one exact index manifest
      |
      +-- Fusion
             +-- syntax index manifest
             +-- exact-normal-form index manifest
             +-- fusion policy
      |
      v
shared context packer
      |
      v
canonical source-object context
```

## Retrieval binding

Schema:

`.github/schemas/math-retrieval-binding-v1.schema.json`

Validator:

`.github/scripts/validate-math-retrieval-binding.py`

Every binding carries:

- binding identity;
- `MeasurementOnly` authority;
- exact experiment arm ID;
- binding mode;
- exact context-packer digest.

### SingleIndex mode

Requires exactly:

`index_manifest_sha256`

and forbids syntax/normal-form/fusion fields.

### Fusion mode

Requires exactly:

```text
syntax_index_manifest_sha256
normal_form_index_manifest_sha256
fusion_policy_sha256
```

and forbids the ambiguous singular index field.

The syntax and normal-form index digests must differ.

This manifest is wiring only. It does not establish that referenced artifacts exist, authenticate themselves, or satisfy cross-contract fairness. That belongs to the next cross-manifest validation gate.

## Canonical context packer

Schema:

`.github/schemas/math-retrieval-context-packer-v1.schema.json`

Validator:

`.github/scripts/validate-math-retrieval-context-packer.py`

The context packer isolates **retrieval selection** from **downstream representation**.

The governing law is:

```text
retrieval representation may change which source objects are selected
retrieval representation may NOT silently change what downstream receives
```

Every selected source identity is re-fetched under one frozen source-fetch policy and serialized under one frozen canonical source-object serialization.

Required payload:

`CanonicalSourceObject`

## Metadata non-leakage

The following are mandatory `false` for downstream visibility:

```text
retrieval_score_visible_to_downstream
retrieval_rank_visible_to_downstream
retrieval_channel_visible_to_downstream
representation_bytes_visible_to_downstream
normal_form_visible_to_downstream
provenance_sidecar_visible_to_downstream
```

The provenance sidecar itself is required, but it remains evidence-plane metadata rather than downstream cognitive input.

This prevents a treatment from gaining information through annotations such as:

- “HDC ranked this first”;
- an exact-normal-form string;
- embedding bytes;
- channel labels;
- retrieval scores;
- hidden provenance cues.

If any of those are later intentionally exposed, that is a separate experimental intervention.

## Deterministic whole-item packing

The packer consumes:

`RankedSourceObjectDigests`

and freezes:

```text
dedup_key = SourceObjectDigest
order_policy = PreserveRetrievedRank
byte_accounting = CanonicalUtf8Bytes
partial_item_policy = RejectWholeItem
nonfitting_item_policy = StopBeforeFirstNonFittingItem
unused_budget_reallocation = false
deterministic = true
```

Why stop at the first non-fitting item instead of skipping it and searching later entries?

Because skipping a large high-ranked source to cherry-pick smaller lower-ranked items changes the effective opportunity set as a function of payload length. Stopping is conservative, deterministic, and easy to compare across representations.

The packer never issues a deeper retrieval query to recover unused budget.

## Packer budget

Every packer binds:

```text
max_output_items
max_output_bytes
max_output_item_bytes
```

The upcoming cross-contract validator should require these to equal the corresponding MATH-EXP root ceilings:

```text
max_output_items      == experiment.retrieved_items_max
max_output_bytes      == experiment.retrieval_context_bytes_max
max_output_item_bytes == experiment.retrieved_item_bytes_max
```

Using equality rather than merely `<=` removes another avoidable degree of freedom.

All comparable retrieval arms should bind the **same exact context-packer digest**.

## Fusion interaction

MATH-RET-001B already requires fused output to resolve to canonical source objects.

The shared context packer remains the final downstream renderer. The fusion policy determines ranked source identity; the packer determines canonical downstream bytes.

Thus:

```text
syntax/HDC/normal-form bytes -> ranking only
canonical source bytes       -> downstream only
```

## Required next cross-contract gate

MATH-RET-001D should consume actual files and verify their SHA-256 digests rather than trusting manifest strings.

For every retrieval arm it should check:

1. `sha256(binding bytes) == experiment.arm.retrieval_binding_sha256`;
2. `binding.arm_id == experiment.arm_id`;
3. referenced index/fusion/packer bytes match their declared digests;
4. every index candidate universe matches the experiment corpus and knowledge boundary;
5. comparable indices share exact candidate-set digest/count/eligibility policy;
6. experiment representation channels/family/digest agree with the bound index or fusion pair;
7. exact-normal-form indices bind the experiment's frozen normalizer contract/implementation;
8. every binding uses the same context-packer digest;
9. context-packer budget equals the experiment root retrieval ceilings;
10. fusion-policy input index digests equal the binding's syntax/normal-form index digests;
11. fusion global ceilings do not exceed the experiment/packer ceilings;
12. downstream payload serialization is identical across all arms.

That produces a machine-checkable fairness closure over the whole retrieval dependency graph.

## Validator qualification state

The two standalone validators include adversarial self-tests in source, but this tranche does **not** claim they have executed yet. Focused execution should run them explicitly before promotion.

## Nonclaims

Valid binding and packer manifests do not establish that:

- referenced files exist;
- content digests are correct;
- indices were built correctly;
- the candidate universes actually match;
- fusion is beneficial;
- HDC or normalization is beneficial;
- downstream proof/search improves;
- any theorem is true or novel.

They freeze the last local contracts needed for a cross-manifest fairness proof.
