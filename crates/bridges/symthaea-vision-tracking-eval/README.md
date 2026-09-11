# symthaea-vision-tracking-eval

Adapter from `symthaea-vision-manifold` centroid tracks into
`symthaea-tracking-eval`.

The current vision tracker stores persistent object centroids in patch/grid
coordinates. This adapter converts those centroids into normalized image-plane
points and converts MOT-style ground-truth boxes into normalized **box centers**.

That enables honest measurement of:

- false positive points
- false negative points
- track identity switches
- tracks that never match ground truth
- false-positive behavior on negative-only sequences

It deliberately does **not** invent bounding boxes for Symthaea tracks or claim
box-IoU/HOTA/IDF1 compatibility.

## Important boundary

```text
image/grid centroid -> normalized image point
```

not:

```text
image/grid centroid -> world position / range / bearing
```

World-space geometry belongs behind an explicit calibrated camera/depth model with
its own uncertainty and model-assurance evidence.

## Standalone MOT-style evaluation

The crate includes an example that runs the real `VisionManifold` over `img1/`
frames, parses MOT-style `gt/gt.txt` boxes as center points, preserves empty frames,
and reports the point-tracking metrics:

```bash
cargo run -p symthaea-vision-tracking-eval --example mot_point_eval -- \
  data/mot-sample/train/MOT17-02/ --frames 500 --gate 0.08
```

Optional arguments:

- `--frames N` limits the number of frames
- `--gate F` sets the reviewed normalized point-match distance
- `--target N` selects the square input size passed to the vision manifold

The evaluator emits two BLAKE3 receipts:

- a **dataset digest** over the exact ground-truth bytes plus every processed encoded image frame
- a **run receipt** over that dataset digest plus the evaluated frame count, frame limit, match gate, target/grid geometry, enabled object/temporal binding, object-memory capacity, and assumed frame rate

The sequence directory pathname is intentionally not part of the digest, so relocating an
identical dataset does not create a different evidence identity.

A sequence with no `gt/gt.txt` is treated as background-only by the evaluator.
That is valid release evidence **only when the dataset is independently established
to contain no relevant tracked objects**. A missing or accidentally omitted
annotation file must not be reinterpreted as negative ground truth.

The output is explicitly labelled centroid/point tracking; it is not HOTA, IDF1,
or MOTA.

## Verification

```bash
cargo test -p symthaea-vision-tracking-eval
```
