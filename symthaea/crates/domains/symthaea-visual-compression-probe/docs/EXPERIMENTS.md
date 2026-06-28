# Experiment Plan

## Experiment 1: Reconstruction baseline

Question:

> How much visual detail survives when only the top-k block coefficients are stored?

Procedure:

```bash
svcp benchmark fixtures/tiny_pump_scan.pgm --block 8 --keep 4
svcp benchmark fixtures/tiny_pump_scan.pgm --block 8 --keep 8
svcp benchmark fixtures/tiny_pump_scan.pgm --block 8 --keep 16
```

Metrics:

- MSE
- PSNR
- coefficient density
- text-to-raw ratio

Interpretation:

This is the ordinary codec-like baseline. It should not be oversold.

## Experiment 2: Query without decode

Question:

> Can packets be ranked by structural similarity without reconstructing pixels?

Procedure:

1. Encode a directory of `.pgm` scans into `.svmp` packets.
2. Run `svcp query <query.svmp> <packet-dir> --top 5`.
3. Inspect whether similar pump/fracture/flow patterns rise to the top.

Metrics:

- HDC similarity
- topology similarity
- combined similarity
- human relevance of top-k results

## Experiment 3: False-green evidence

Question:

> Can a packet retain useful visual evidence when machine telemetry says healthy?

Procedure:

1. Capture a pump scan before repair.
2. Capture a pump scan after repair.
3. Encode both.
4. Compare packet metrics and topology changes.

Desired result:

The packet should preserve enough structure to say: “the leak shape changed” even without saving full imagery.

## Experiment 4: Chronicle artifact suitability

Question:

> Is the packet small, stable, and interpretable enough to attach to a durable repair event?

Check:

- deterministic encoding
- readable packet diff
- stable HDC signature for near-identical images
- topology delta for structural changes
- no false certainty in metadata
