# SYM-RSI-SEM-004 — Preregistration Amendment 2

Status: **FROZEN BEFORE SEM-004 IMPLEMENTATION OR MEASUREMENT**

Parent preregistration:

`2a11924821831137a58253ec6013ba40a63f0f3d`

Ambiguity-band calibration amendment:

`bdd779afedc7603978009278e4a661b9d568b443`

## Purpose

This amendment freezes the exact deterministic SEM-004 corpus generator before any SEM-004 similarity, calibration threshold, held-out endpoint, or disposition is computed.

It changes no primary success criterion and no retrieval rule. It only removes generator degrees of freedom left intentionally open by the parent protocol.

## Arithmetic contract

All generated scalar values are `f32`.

Each arithmetic stage in the implementation must reproduce Rust `f32` semantics. Integer mixing uses wrapping `usize` arithmetic, followed by exact integer modulo before conversion to `f32`.

Frozen dimensions:

`D = {4, 8, 16}`

Nominal semantic support remains `[-1, 1]`.

## Candidate-memory generator

Per dimension generate candidate indices `0..96`.

For candidate index `i`:

`family = i / 2`

`side = i % 2`

First compute a family center for coordinate `j`:

`mixed = 0x51ED`
`      + D * 211`
`      + family * 73`
`      + j * 43`
`      + j * j * 11`

using wrapping integer arithmetic.

Then:

`center_j = (mixed % 1401) / 1000.0 - 0.700`

The sibling-separation coordinate is:

`k = (family * 7 + D * 3) % D`

Apply:

- side `0`: `context[k] -= 0.100`
- side `1`: `context[k] += 0.100`

All other coordinates equal the family center.

The candidate generator domain/version is frozen as:

`semantic-gated-relative-generator.v1`

Every candidate must be inside nominal encoder support. Failure is an integrity error.

## Labelled clean calibration queries

Per dimension generate `32` queries, one from each family `0..31`.

The labelled true source is the even sibling:

`source_index = 2 * family`

Start from the exact source candidate context.

Perturbation magnitude:

`m = 0.025 + 0.005 * (family % 8)`

Perturbation coordinate:

`k = (source_index * 11 + 17) % D`

Direction:

- if `(family + D + 17) % 2 == 0`, add `m`;
- otherwise subtract `m`.

Clamp the perturbed coordinate to `[-0.95, 0.95]`.

The labelled source identity is fixed by construction before semantic encoding.

## Ambiguous calibration queries

Per dimension generate `16` queries from sibling families `0..15`.

For each family, the intended parents are candidate indices:

`left = 2 * family`

`right = 2 * family + 1`

The ambiguous calibration query is the coordinate-wise midpoint:

`query_j = (left_j + right_j) * 0.5`

Both parent identities are fixed before semantic encoding.

## Unrelated calibration-control queries

Per dimension generate `32` queries for indices `0..32` from a distinct unrelated family.

For unrelated-control index `i` and coordinate `j`:

`mixed = 0xA53F`
`      + D * 257`
`      + i * 113`
`      + j * 67`
`      + j * j * 17`

using wrapping integer arithmetic.

Then:

`query_j = (mixed % 1401) / 1000.0 - 0.700`

These queries have no designated candidate source.

## Clean held-out queries

Per dimension generate `32` queries, one from each family `0..31`.

The labelled true source is the odd sibling, ensuring it is exact-distinct from the clean calibration source identity:

`source_index = 2 * family + 1`

Start from the exact source context.

Perturbation magnitude:

`m = 0.0275 + 0.005 * ((family * 3 + 1) % 8)`

Perturbation coordinate:

`k = (source_index * 13 + 29) % D`

Direction:

- if `(family + D + 29) % 2 == 0`, add `m`;
- otherwise subtract `m`.

Clamp the perturbed coordinate to `[-0.95, 0.95]`.

The true source identity is fixed before semantic encoding.

## Ambiguous held-out queries

Per dimension generate `24` queries from sibling families `24..48`.

For local index `r in 0..24`:

`family = 24 + r`

Parents:

`left = 2 * family`

`right = 2 * family + 1`

Held-out ambiguous query:

`query_j = (left_j + right_j) * 0.5`

Both intended parent identities are fixed before semantic encoding.

## Unrelated held-out queries

Per dimension generate `24` queries using the same unrelated equation as the calibration controls but with indices `64..88`.

For local index `r in 0..24`:

`i = 64 + r`

Then use the frozen unrelated equation from above.

This makes unrelated calibration and unrelated held-out query identities deterministically disjoint by index family before exact-identity verification.

## OOD held-out queries

Per dimension generate `16` OOD queries from candidate source indices `80..96`.

For local offset `r in 0..16`:

`source_index = 80 + r`

Start from the exact candidate source context.

Select coordinate:

`k = (r * 5 + D) % D`

Replace that coordinate with:

- if the source coordinate is non-negative: `1.35 + abs(source[k])`;
- otherwise: `-1.35 - abs(source[k])`.

Every OOD query must therefore have at least one coordinate outside nominal support and a distinct exact identity.

## Exact-disjointness requirement

Before any SEM-004 calibration or held-out evaluation, compute exact BLAKE3 `ContextIdentity` for every generated context and require zero overlap across:

- candidate bank;
- clean calibration queries;
- ambiguous calibration queries;
- unrelated calibration controls;
- clean held-out queries;
- ambiguous held-out queries;
- unrelated held-out queries;
- OOD held-out queries.

Also require zero overlap with all six frozen SEM-003 corpus-canary sets bound by subject:

`22e246a0a9bcc0260c3b77bc384b9bc5cf17b0c9`

Any overlap is an integrity failure. Do not regenerate constants after observing such a failure; amend the protocol explicitly before measurement if a frozen collision genuinely exists.

## Calibration quantiles restated

Nearest-rank `q95` remains:

`rank = ceil(N * 0.95)`

clamped to `[1, N]`, selecting `sorted[rank - 1]`.

Per dimension:

- `G_D = q95(unrelated calibration top-1 similarities)` from `32` controls;
- `C_D = q95(clean calibration true-source gaps)` from `32` labelled queries;
- `A_D = q95(ambiguous calibration weaker-parent gaps)` from `16` ambiguous calibration queries;
- `R_D = max(C_D, A_D)`.

No held-out observation may affect these values.

## Candidate ordering and ties

For diagnostics requiring ordered candidates, sort by:

1. similarity descending;
2. exact BLAKE3 context digest ascending;
3. legacy fast hash ascending as a final deterministic tie-break.

Candidate-set membership itself depends only on the frozen gate/band inequalities, not on ordering.

## Corpus receipt requirements

The future SEM-004 receipt must bind:

- generator version `semantic-gated-relative-generator.v1`;
- every partition count;
- every partition digest;
- exact-disjointness result;
- SEM-003 corpus-canary identity used for cross-experiment overlap checking;
- per-dimension `G_D`, `C_D`, `A_D`, and `R_D` bit patterns.

## Scientific boundary

This amendment freezes synthetic data generation only.

It does not assert that SEM-004 will pass, that the new rule is superior, or that any semantic candidate is true. A negative SEM-004 result must remain a negative result rather than trigger silent generator or threshold retuning.
