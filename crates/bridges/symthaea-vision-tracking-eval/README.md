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

## Verification

```bash
cargo test -p symthaea-vision-tracking-eval
```
