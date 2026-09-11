# symthaea-tracking-eval

Deterministic evaluation for centroid/point trackers.

This crate exists because the current Symthaea vision tracker exposes persistent
object centroids rather than benchmark-grade bounding boxes. It therefore avoids
mislabeling centroid metrics as HOTA, IDF1, or MOTA.

## Metrics

For explicit frame records, including empty/negative frames, it reports:

- true positives
- false positives
- false negatives
- precision / recall / F1 when defined
- identity switches
- unique prediction tracks
- tracks that never match ground truth
- frames containing false positives
- negative frames containing false positives
- clean-negative-frame fraction
- normalized match distance

Matching uses a deterministic maximum-cardinality bipartite assignment within a
reviewed normalized-distance gate. Candidate ordering is nearest-first, but the
reported assignment is **not** claimed to be a globally minimum-distance solution.

## Why empty frames matter

Long negative-only sequences are first-class inputs. A tracker that performs well
on balanced object/no-object clips can still be unusable if it invents persistent
tracks during hours of ordinary background imagery.

The report therefore explicitly identifies negative-only sequences and false tracks.

## What this is not

This is not a standards-complete MOT benchmark. Full HOTA/IDF1 evaluation should
arrive only after Symthaea emits the necessary detection geometry and sequence data
without approximating those standards from centroids.

## Verification

```bash
cargo test -p symthaea-tracking-eval
```
