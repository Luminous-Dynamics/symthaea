# Structured Foveation Evidence v2

Status: VIS-001R refinement stacked on VIS-000R.

## Purpose

Make semantic visual evidence trustworthy enough to feed later object memory/world-state work by binding **what pixels were observed** and **what semantic computation actually executed**.

This refines the first VIS-001 adapter, which accepted a source identity at structuring time. v2 requires provenance to enter at the capture/frame boundary instead.

## End-to-end provenance path

```text
capture owner
  -> VisualStreamRef + VisualObservationRef
  -> FoveationManager::on_observed_frame(frame, observation)
  -> saliency item freezes observation
  -> FoveationRequest freezes observation
  -> background VentralPipeline
  -> FoveationResult preserves observation + execution receipt
  -> StructuredFoveationEvidence
  -> VisualEvidence::Inferred(exact observation)
```

No post-hoc source ID is accepted by the structuring adapter.

## Capture-bound admission

`on_observed_frame` requires the legacy framebuffer metadata to exactly match the typed observation:

- `frame.frame_id == observation.frame_id()`
- `frame.timestamp_us == observation.captured_at_us()`

Mismatch fails before manager state changes.

The compatibility `on_frame` path remains available, but it explicitly clears typed provenance. Results from that path may still be used by legacy consumers, but `StructuredFoveationEvidence::from_result` refuses to convert them into provenance-bearing structured evidence.

## Stale-frame invariant

Pending saliency belongs to one exact framebuffer.

Whenever the current frame is replaced, undispatched pending saliency is cleared. Dispatch also verifies as defense in depth that:

- current frame id matches queued frame id;
- current frame timestamp matches queued timestamp;
- current typed observation matches the queued observation.

If any check fails, the pending request is discarded rather than cropping new pixels under old identity.

Already-dispatched requests retain their frozen observation and may complete later.

## Semantic execution receipt

Every `FoveationResult` now records three separate facts:

1. `requested_routing` — configuration intent;
2. `operation` — what semantic operation actually ran (`Ocr`, `Embedding`, `Caption`, `Fallback`, `Unknown`);
3. `kind` — what backend actually produced the result.

Initial backend kinds are:

- `HashStubV1`
- `SemanticVisionOnnxUnpinned`
- `SemanticVisionDeterministicStub`
- `ErrorFallbackRandom`
- `Unknown`

This split is required because the archived SemanticVision path can initialize successfully while SigLIP has no loaded ONNX session and therefore returns a deterministic stub embedding. A high-level "real pipeline" label is not evidence that learned inference executed.

It also exposes the current routing limitation honestly: the feature-gated SemanticVision integration currently calls `embed_image` even when another route was requested. The receipt therefore records `operation = Embedding` rather than copying the requested route into execution truth.

## Model identity boundary

`SemanticVisionOnnxUnpinned` means only that an ONNX session actually backed the embedding operation. It does **not** establish exact model artifact identity.

`VentralExecutionReceipt::exact_model_artifact_pinned()` is therefore false for every v1 receipt.

A future VIS-002 model-evidence tranche must bind at least model family/version, exact weights/artifact digest, preprocessing contract, projection identity, and relevant runtime configuration before reproducible learned-model identity can be claimed.

## Structured evidence rule

`StructuredFoveationEvidence::from_result(result)` succeeds only when:

- `result.source_observation` exists;
- its frame id matches `result.source_frame_id`;
- its capture time matches `result.source_timestamp_us`;
- VIS-000 confidence/origin/lineage validation succeeds.

The resulting semantic claim is always `VisualOrigin::Inferred`, including at confidence `1.0`.

## Authority boundary

This is perception evidence only. Neither capture provenance nor semantic execution receipts grant camera motion, motor, robotics, targeting, actuation, execution, or other physical authority.

## Next gates before persistent world state

1. Qualify VIS-000R stream/clock semantics.
2. Qualify this VIS-001R exact head.
3. Reconcile any exact Cargo-generated root lockfile delta rather than hand-authoring lock bytes.
4. VIS-002: exact learned-model artifact/preprocessing/projection receipts.
5. VIS-003: honest region/box geometry beyond centroid-only tracking.
6. Only then integrate semantic/entity evidence into VIS-004 persistent visual beliefs.

## Nonclaims

VIS-001R does not establish recognition accuracy, model calibration, exact learned-model reproducibility, object identity, segmentation, world position, cross-camera synchronization, or embodied safety. It makes the evidence path explicit enough for those later claims to be tested rather than assumed.
