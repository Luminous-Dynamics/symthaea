# Dense Visual Encoder Evidence v1

Status: implementation contract for VIS-002.

## Purpose

VIS-002 defines how learned or deterministic dense visual encoders may contribute evidence to Symthaea without becoming the visual world model themselves.

The governing rule is:

```text
model output is inferred sensory evidence, not observation and not truth
```

A model label such as "DINO", "SigLIP", "JEPA", or a local checkpoint name is not sufficient evidence about what bytes actually executed.

## Input boundary

`DenseVisualInput` binds together:

- tightly packed source pixels;
- frame width and height;
- channel count;
- the exact `VisualObservationRef` owned by the capture boundary.

Construction rejects zero dimensions, unsupported channels, byte-length mismatch, arithmetic overflow, and inputs above the contract memory ceiling.

Anonymous pixels without a `VisualObservationRef` are not accepted by this provenance-bearing interface.

## Provider interface

`DenseVisualEncoder` is provider-neutral. Implementations may later use ONNX, Candle, Torch, external services, or deterministic reference encoders.

Successful implementations return a validated `DenseFeatureMap` with row-major layout:

```text
[grid_rows, grid_cols, feature_dim]
```

The output contract validates dimensions, checked tensor size, the configured memory ceiling, and finite feature values.

## Execution receipt

Every result carries `DenseEncoderReceipt` with:

- the backend class that actually executed;
- a provider label;
- model artifact identity;
- preprocessing artifact identity;
- dense-output semantics.

Requested configuration is not substituted for actual execution identity.

## Artifact identity

`ArtifactIdentity` distinguishes:

- `Pinned`: exact bytes are content-addressed by BLAKE3 or SHA-256;
- `Unpinned`: a label/revision is known but exact bytes are not proven;
- `BuiltIn`: the component is compiled into the running binary but is not independently content-addressed by this receipt;
- `Unknown`: no useful artifact identity is available.

`model_and_preprocessing_pinned()` is true only when both artifacts carry exact digests.

Even that predicate is deliberately weaker than "reproducible execution": hardware, runtime build, execution provider, precision mode, kernel implementation, and accelerator determinism are outside this receipt and require later qualification if they matter to a claim.

## Epistemic contract

`DenseFeatureMap::new(...)` constructs its provenance as:

```text
source observation -> VisualEvidence::Inferred -> dense feature map
```

Confidence `1.0` does not change the origin to `Observed`.

Dense representations may later contribute evidence to object hypotheses, scene understanding, prediction, memory, or world-state updates, but those systems must preserve this lineage.

## Memory limits

The initial contract caps:

- source raster admission at 256 MiB;
- dense outputs at 16,777,216 f32 elements (64 MiB).

These are conservative allocation ceilings for the evidence boundary, not performance targets or recommended tensor sizes.

## Dense semantics

The initial semantics vocabulary includes:

- `SpatialTokens` — 2-D tokens/patch embeddings with spatial correspondence;
- `PyramidLevel` — one dense feature-pyramid level;
- `ProviderDefined` — provider-specific dense semantics that have not yet been normalized.

A provider-specific result must not be interpreted as a standardized spatial representation merely because its tensor is three-dimensional.

## Serialization boundary

Invariant-bearing identities, receipts, and dense feature maps are Serialize-only in this tranche. Validating deserialization is intentionally deferred so wire input cannot bypass constructors.

Leaf vocabularies such as digest values and backend enums may use ordinary serde because they do not independently constitute a trusted execution receipt.

## Integration sequence

After VIS-002 qualifies:

1. add a deterministic adapter around the existing Patch-HDC path as a control where useful;
2. implement one optional learned dense backend behind this interface;
3. content-address exact model and preprocessing artifacts;
4. benchmark frozen dense features against the Patch-HDC baseline on controlled perception tasks;
5. add an explicit dense-feature-to-HDC evidence adapter rather than replacing VisionManifold;
6. use Symtropy hidden ground truth to measure whether learned features improve geometry, persistence, and prediction;
7. only then consider enabling learned features in persistent object/world beliefs.

## Benchmark rule

A learned encoder is useful only if it improves a declared downstream measurement under a frozen protocol. Model size, benchmark reputation, or qualitative visual appeal are not substitutes for a Symthaea-specific causal comparison.

Recommended early comparisons include:

- object correspondence under appearance change;
- occlusion/reappearance identity;
- lighting and viewpoint shifts;
- spatial segmentation consistency;
- sample efficiency of downstream HDC binding;
- calibration of semantic/object hypotheses;
- compute and memory cost per useful improvement.

## Authority / nonclaims

VIS-002 grants no motor, camera-motion, targeting, manipulation, navigation, robotics, or other actuation authority. It does not establish exact runtime reproducibility, learned-model calibration, semantic correctness, object identity, geometry correctness, or world-model fidelity.
