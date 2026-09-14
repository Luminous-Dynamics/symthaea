# GEOM-002B — Coarse-Graining Sensitivity Envelope

Status: stacked implementation candidate on GEOM-002A. Promotion is blocked until GEOM-001, GEOM-002A, and this tranche all qualify at their exact heads.

## Question

GEOM-002B asks whether a reported coarse-scale causal advantage survives alternative preregistered coarse-grainings with the same state-count profile.

A result that appears only under one hand-selected partition is partition-sensitive evidence, not a robust multiscale result.

## Comparability contract

Every variant:

- uses the same finest TPM;
- uses the same `CausalAdvantageRule`;
- has a unique non-empty label;
- contains the same number of scale transitions;
- yields the same number of states at each scale index.

The actual state assignments may differ. If scale counts or state-count profiles differ, GEOM-002B fails closed instead of comparing incomparable objects.

## Reported envelope

For each scale index, record min/max ranges across variants for:

- effective information;
- determinism;
- degeneracy;
- causal advantage versus the finest scale.

Also report:

- how many variants peak at each scale;
- `peak_scale_stability = max_peak_count / variant_count`;
- how many variants pass the preregistered causal-advantage rule;
- `criterion_met_fraction`.

No averaging is required to hide disagreement. Wide ranges and unstable peaks are themselves evidence about sensitivity.

## Deterministic controls

### Replication control

Two identical coarse-grainings with different labels must produce zero-width metric ranges, identical peak scale, and peak-scale stability 1.0.

### Partition-sensitivity control

Use the GEOM-002A asymmetric four-state TPM.

Variant A: `[0,0,0,1]`

- aligns the partition with the three degenerate fine states;
- derives a deterministic two-state flip TPM;
- produces 1 bit of macro EI;
- passes the 0.1-bit causal-advantage criterion.

Variant B: `[0,0,1,1]`

- uses the same 4→2 state-count reduction but a different partition;
- produces substantially lower macro EI;
- does not beat the finest scale.

The expected sensitivity result is therefore:

- one variant peaks fine and one peaks coarse;
- peak-scale stability = 0.5;
- criterion-met fraction = 0.5;
- macro EI range width > 0.5 bits.

This is a deliberate demonstration that a valid causal-emergence result can still depend strongly on the chosen partition.

## Interpretation boundary

Allowed:

> Across the preregistered partition family, the peak causal scale was stable in X% of variants and the decision criterion passed in Y%.

Not allowed from GEOM-002B alone:

- selecting only the best partition and ignoring alternatives;
- treating peak-scale instability as consciousness evidence;
- treating causal-advantage robustness as proof of downward causation;
- collapsing sensitivity into a single consciousness score;
- inferring any gravity/consciousness physical relationship.

## Next stage

GEOM-002C may jointly report this causal sensitivity envelope with GEOM-001 Fisher-Rao trajectory observables. The two measurement families must remain separate fields with separate nulls and may not be merged into an omnibus consciousness or gravity score.
