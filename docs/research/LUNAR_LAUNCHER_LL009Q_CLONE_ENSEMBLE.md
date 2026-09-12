# LL-009Q — Site01 clone-ensemble terrain horizon evidence

## Purpose

LL-009Q closes a scientific semantics gap in the Site01 terrain chain without
pretending that an RMS map is a deterministic terrain bound.

NASA PGDA product 78 publishes:

- the nominal Site01 / Connecting Ridge 5 m LDEM;
- a per-pixel total-Z uncertainty raster whose published semantics are RMS error;
- 100 Site01 statistical clone rasters.

Barker et al. describe the clone construction explicitly. For each realization,
an observationally filtered simulated terrain produces a Z-error map, and that
Z-error map is added back to the nominal LDEM to form a clone. The 100 clones
form a statistical ensemble with approximately the same error properties as the
data.

LL-009Q therefore treats the published ensemble as **empirical ensemble
evidence**. It does not convert one RMS value into an unqualified physical upper
bound.

Primary sources:

- NASA PGDA product 78:
  `https://pgda.gsfc.nasa.gov/products/78`
- Barker et al. (2021), accepted manuscript:
  `https://ntrs.nasa.gov/api/citations/20205009660/downloads/Mike%20Barker_%20Paper_%20Improved%20Lola.pdf`

## Source-family contract

`configs/lunar_transport/ll009q_site01_clone_family_v1.json` declares the
published Site01 clone family compactly:

- members 1 through 100;
- four-digit member tokens;
- filenames
  `Site01_final_adj_5mpp_0001_err.tif` through
  `Site01_final_adj_5mpp_0100_err.tif`;
- NASA PGDA HTTPS source directory;
- safe local artifact directory;
- role `terrain_error_realization_m`;
- interpretation `additive_z_error_to_nominal_ldem`;
- no invented SHA-256 values.

`generate_ll009q_clone_plan.py` expands that compact declaration into an exact
`ll009n.nasa-acquisition-plan.v1` document. The expanded plan can be handed
unchanged to the LL-009N acquisition/offline-verification machinery.

This avoids checking a repetitive 100-entry source list into the repository
while still making the generated list byte-deterministic and hashable.

The generator fails on:

- member-count/range disagreement;
- insufficient index width;
- non-HTTPS sources;
- non-allowlisted hosts;
- path traversal;
- malformed SHA-256 pins;
- duplicate expanded source IDs, URLs or destinations.

The checked-in Site01 family deliberately leaves `expected_sha256_by_index`
empty. A NASA filename or page link does not establish exact source-byte
identity. Hashes become evidence only after the actual source bytes have been
acquired.

## Exact ensemble lock

A real promoted Q run requires a clone-only LL-009N source lock produced from
the exact expanded plan.

The Q materializer checks that:

1. the supplied clone plan is byte-equivalent to a fresh deterministic expansion
   of the checked-in family declaration;
2. the clone lock binds the exact clone-plan SHA-256;
3. the lock contains exactly the expected members in stable numeric order;
4. every local clone artifact re-hashes to its locked SHA-256 and byte count;
5. there are no missing or extra members.

Thus `0001..0100` is one exact ensemble lineage. A 99-member subset is a
different evidence object and does not satisfy the 100-member Site01 policy.

## Nominal and site-state binding

The ensemble does not stand alone.

A Q run also binds:

- the LL-009N nominal Site01 source lock;
- the exact nominal elevation SHA-256;
- the LL-009P site-anchor receipt;
- the exact LL-009L raster configuration defining the admitted near-field layer.

The LL-009P receipt supplies the raster-supported site row/column and nominal
site elevation.

The LL-009L site block must equal the P receipt exactly, and the L near-field
elevation source must bind the same nominal LDEM hash as the N source lock.

## Same-realization observer policy

This is an important detail.

For an additive clone realization `e_j(x)`, Q reconstructs terrain member `j`
as

```text
z_j(x) = z_nominal(x) + e_j(x)
```

and reconstructs the observer elevation from the **same realization at the site
pixel**:

```text
z_site,j = z_site,nominal + e_j(site)
```

The site is therefore not held artificially fixed while surrounding terrain is
perturbed.

This preserves the correlated relative-height information represented by one
published clone realization.

The checked-in policy names this rule:

```text
observer_policy = same_realization_site_pixel
```

A future full-clone-surface dataset may use another explicit interpretation, but
an `unknown` clone interpretation must fail closed rather than being inferred
from a filename.

## All-pixel horizon scan

LL-009Q intentionally does **not** reuse LL-009L's top-K candidate reduction.

For each exact clone member it:

1. opens the exact nominal LDEM and exact clone raster;
2. requires single-band, shape, CRS, affine and pixel-scale alignment;
3. reuses the exact LL-009L radial band and azimuth-bin policy;
4. reconstructs each admitted terrain elevation;
5. reconstructs the same-realization site elevation;
6. transforms pixel centers onto the explicit lunar sphere;
7. computes Moon-centered terrain and observer vectors;
8. computes azimuth/elevation in the site-local basis;
9. scans every admitted source pixel and retains the maximum elevation in each
   azimuth bin.

The raster-cell scan is vectorized block-by-block, but the evidence semantics
are equivalent to evaluating every admitted cell in every clone.

This matters because the terrain cell that dominates one ensemble member may
not be among the nominal LDEM's retained top-K cells.

## Empirical summaries

For each azimuth bin the exact 100-member horizon array is summarized with:

- observed minimum;
- observed maximum;
- nearest-rank empirical quantiles declared in the hashed Q policy.

The checked-in quantiles are:

- 0.05;
- 0.50;
- 0.95;
- 0.99;
- 1.00.

The estimator is the empirical CDF inverse / nearest-rank estimator:

```text
Q(q) = sorted_values[ceil(q * n) - 1]
```

for `q` in `(0, 1]`.

The estimator choice is evidence metadata, not an implementation detail.

## Finite-ensemble semantics

The per-bin maximum means only:

> the largest horizon elevation observed among these exact published ensemble
> members.

It does **not** mean:

- a deterministic physical upper bound;
- a confidence level for an unspecified parent distribution;
- a guarantee that a 101st plausible realization cannot exceed it;
- proof that the real terrain lies inside the clone envelope.

Likewise, an empirical 95th percentile is a percentile of the finite published
ensemble unless another statistical model is explicitly introduced and bound.

LL-009O should therefore classify a valid Q receipt as
`empirical_ensemble`, not `hard_upper_bound`.

## Spatial-support boundary

Q operates on the raster support represented by the nominal LDEM plus clone
fields.

It deliberately reports:

```text
spatial_support_class = sample_points_only
```

until LL-009R closes the different question of unresolved terrain between
raster support points.

This keeps two independent uncertainties separate:

1. **vertical/model uncertainty at raster support**, addressed by O/Q;
2. **continuous terrain between support points**, addressed by R.

Neither is allowed to borrow confidence from the other.

## Far field

The first Q lane is the Site01 5 m near-field ensemble.

It does not assign clone semantics to the large-area Product 90
`LDEM_80S_*_ADJ_ERR.TIF` products. Their uncertainty methodology remains a
separate evidence question.

A downstream full horizon may eventually combine:

- Site01 near-field ensemble horizons;
- independently qualified far-field horizons;
- LL-009R spatial-support margins/classes.

Until then, Q is a near-field empirical terrain-uncertainty receipt.

## Local synthetic executed evidence

The LL-009Q source-plan self-test has been executed locally and verifies:

- deterministic expansion;
- exactly ordered member naming;
- member-count mismatch rejection;
- non-NASA host rejection;
- path traversal rejection.

The LL-009Q ensemble materializer self-test has been executed locally with
Rasterio 1.5.0 and verifies:

- exact plan/family/lock binding;
- exact nominal-source binding;
- LL-009P/L site-state agreement;
- additive clone reconstruction;
- same-realization site elevation;
- all-admitted-pixel horizon scanning;
- deterministic nearest-rank quantiles;
- deterministic replay;
- shifted clone-grid rejection.

These are **logic tests only**. They contain no NASA clone bytes and establish
no real Site01 horizon result.

Promoted execution must use the pinned evidence GIS lineage already declared by
LL-009L, exact NASA source locks, and all downstream claim gates.

## Resulting evidence chain

```text
NASA PGDA product 78
      │
      ├── nominal Site01 LDEM + RMS raster
      │          ↓
      │       LL-009N
      │          ↓
      │       LL-009P
      │
      └── 100 Site01 clone error realizations
                 ↓
       LL-009Q family expansion
                 ↓
       LL-009N exact clone source lock
                 ↓
       LL-009Q all-pixel clone horizons
                 ↓
       empirical bin quantiles / observed max
                 ↓
       LL-009O claim semantics
                 ↓
       LL-009R spatial-support gate
                 ↓
       LL-009K/J/I downstream horizon,
       visibility and site-viability reasoning
```

## Non-claims

Passing LL-009Q does not establish:

- deterministic physical horizon completeness;
- unresolved/subpixel terrain closure;
- far-field uncertainty closure;
- a distributional tail guarantee beyond the finite clone ensemble;
- illumination availability;
- Earth/relay communications availability;
- thermal or geotechnical suitability;
- construction or operations authority;
- transportation architecture viability.
