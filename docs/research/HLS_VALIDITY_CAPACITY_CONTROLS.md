# HLS validity-memory orthogonal capacity controls v0

This note freezes two control experiments before the exploratory capacity sweep is interpreted.

The purpose is to resolve a confound discovered during source audit: the original span-length axis changes both archive write segmentation and the expected density of semantic changes. The controls below vary those factors separately while holding the null-model load coordinates fixed.

## Fixed common coordinates

Every `research_v0` control case uses:

- dimension `D = 4096`;
- relation keys `K = 8`;
- candidate values `C = 8`;
- checkpoint horizon `H = 256`.

Therefore every case represents

`K * H = 2048`

key-checkpoint facts, with

`rho = K * H / D = 0.5`.

The analytic null model consequently predicts the same target/distractor score variance for every control case. Any systematic difference across a control axis is evidence that structure omitted by the null model matters.

## Forced semantic schedule

Candidate assignments are deterministic but adjacent **semantic runs are guaranteed to differ**.

For each key:

1. choose the first candidate from a deterministic seed-derived value;
2. for every later semantic run, choose a nonzero modular jump in `[1, C-1]`;
3. add that jump modulo `C`.

Thus every semantic boundary is an actual semantic mutation. This is different from the exploratory #2768 span schedule, where adjacent independently sampled spans can occasionally repeat a value.

## Control A — semantic mutation density

The archive writes **one span per checkpoint** in every case:

`write_segment_length = 1`.

Therefore write count is constant:

`K * H = 2048 writes`.

Only semantic run length changes:

`1, 2, 4, 8, 16, 32, 64 checkpoints`.

Since every semantic boundary is forced to change candidate identity, total semantic changes are exactly

`K * (H / semantic_run_length - 1)`.

This axis therefore changes semantic mutation density while holding write count, represented facts, dimension, candidate count, and temporal horizon fixed.

## Control B — pure write segmentation

The checkpoint-level semantic history is fixed:

`semantic_run_length = 32`.

Every seed/key uses the same forced-change semantic sequence in every segmentation case.

Only the write partition changes:

`write_segment_length = 1, 2, 4, 8, 16, 32`.

All segment lengths divide the semantic run exactly, so subdividing a semantic run never introduces a semantic boundary.

The semantic-change count is constant:

`8 * (256 / 32 - 1) = 56`.

The number of archive writes varies:

`2048, 1024, 512, 256, 128, 64`.

If these cases differ materially, that effect is attributable to write segmentation/numerical accumulation rather than semantic history or null-model load.

## Mechanical equivalence check

The public smoke protocol contains two cases with identical:

- seed;
- `D/K/C/H`;
- semantic-run length;
- forced semantic history;

but different write partitions (`1` versus `2` checkpoints per write).

The contract requires:

- identical correct counts and query counts;
- equal semantic-change counts;
- different write counts;
- mean and minimum cleanup margins to agree within `1e-9`.

This is a representation-equivalence check, not a requirement that retrieval be correct.

## Untouched seeds

`research_v0` uses eight seeds not used by the exploratory #2768 capacity sweep:

`32001..=32008`.

The plan contains 13 cases:

- seven semantic-density cases;
- six pure-segmentation cases.

Total observations:

`13 * 8 = 104`.

Seeds or axis values may not be changed after outcomes are observed while retaining the `research_v0` name.

## Reported fields

Every observation records:

- correct and total queries;
- accuracy;
- mean and smallest cleanup margin;
- spans written;
- guaranteed semantic-change count;
- represented key-checkpoint facts;
- facts per dimension;
- candidate-score evaluations;
- pre-result null target-score variance;
- pre-result null distractor-score variance.

## Interpretation

### If semantic-density cases differ while segmentation cases do not

The natural interpretation is that repeated semantic associations/correlated value structure materially affect finite-dimensional interference beyond the independent-fact null model.

### If segmentation cases differ

The analytic span representation or floating-point accumulation order has a measurable effect even when checkpoint-level semantics are identical. That numerical/representation effect must be understood before using span length as a cognitive-memory variable.

### If neither control differs

The original span-length effect, if any, is less likely to be caused by these two factors independently and may reflect their interaction or ordinary seed variance.

### If both differ

Both repeated semantic structure and write representation matter; a stronger theory should model them separately rather than repairing the original `rho` law post hoc.

## Scientific boundary

These controls do not establish historical HLS superiority or a universal memory-capacity law. They exist to make the next theory less wrong before historical memory is integrated into learned HLS.

The intended sequence remains:

`capacity sweep -> pre-result null model -> orthogonal controls -> theory revision/confirmatory threshold -> historical HLS integration`.
