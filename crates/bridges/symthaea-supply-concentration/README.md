# Symthaea Supply-Concentration Evidence

This crate provides one narrow Tier-1 supply-resilience input: elemental producer/jurisdiction concentration measured with the Herfindahl-Hirschman Index (HHI).

It does **not** claim that concentration alone is total supply resilience.

## Why stage is explicit

Mining, refining, processing, and manufacturing concentration are different markets and can have very different geographic distributions. A dataset therefore carries exactly one explicit supply-chain stage.

Changing the stage changes the dataset identity and the prediction method provenance.

## HHI convention

For producer/jurisdiction shares `s_i` expressed as fractions in `[0,1]`:

`HHI_normalized = sum(s_i^2)`

This crate reports the normalized `[0,1]` form. Multiply by `10,000` to obtain the conventional HHI point scale used when shares are expressed as percentages.

The source table must provide complete normalized shares for each element: shares must sum to `1` within a small numerical tolerance. A grouped `Other` jurisdiction can be supplied by the normalization process when the source does not enumerate every producer individually.

## Time/data basis is explicit

Every dataset declares one of:

- observed year;
- estimated year;
- projected year + scenario.

A projection is therefore not silently treated as an observation.

## Candidate aggregation policy

Element-level HHI does not by itself define a compound/material-level metric. This crate exposes two explicitly different policies:

### `maximum_element_hhi`

Use the highest HHI among any element in the formula.

This is a conservative bottleneck screen. Trace/light elements can dominate the result even when their mass fraction is small.

### `mass_weighted_mean_hhi`

Weight each elemental HHI by that element's formula mass fraction.

This represents average mass exposure, but can dilute a small-mass bottleneck element.

Neither policy is universally correct. A benchmark or discovery cohort must fix the aggregation policy before comparing candidates.

## Evidence boundary

The normalized JSON dataset contains:

- source title/version/URI;
- SHA-256 of the exact source document;
- normalization timestamp and extraction note;
- supply-chain stage;
- observed/estimated/projected basis;
- element → producer/jurisdiction share tables.

The calculation additionally requires the exact source-document bytes and verifies their SHA-256 before producing evidence.

The exact normalized JSON bytes receive their own digest, and the canonical normalized dataset receives a domain-separated digest.

## Fail-closed composition handling

Formula parsing reuses `symthaea-process-discovery`'s simple Hill-shaped parser. Atomic masses come from `symthaea-bandgap`.

Every element present in the formula must also have a supply-share record. Missing supply data fails closed instead of being ignored.

Parentheses, hydrates, fractional occupancy, isotope notation, disorder, and non-stoichiometry remain outside this v0 parser scope.

## Discovery evidence semantics

A successful result emits a `symthaea-discovery::Prediction` with:

- metric: `supply_concentration_hhi`;
- unit: `score`;
- fidelity: `Analytical`;
- supporting evidence: `Dataset`;
- explicit stage, temporal basis, and aggregation policy in model provenance;
- exact source/capture/dataset digests;
- per-element HHI and formula mass fraction.

No calibrated uncertainty model is claimed; epistemic uncertainty is marked fully unknown.

## Normalized dataset example

This fixture is illustrative only:

    {
      "source_title": "Example supply publication",
      "source_version": "2026",
      "source_uri": "https://example.org/supply",
      "source_document_sha256": "<sha256 of exact source bytes>",
      "normalized_at_utc": "2026-09-12T18:00:00Z",
      "extraction_note": "Normalized production shares; Other closes each row to 1.0.",
      "stage": "refining",
      "basis": {"observed": {"year": 2025}},
      "records": [
        {
          "symbol": "Li",
          "producers": [
            {"jurisdiction": "A", "share": 0.60},
            {"jurisdiction": "B", "share": 0.25},
            {"jurisdiction": "Other", "share": 0.15}
          ]
        }
      ]
    }

## Network-free CLI

    supply-concentration <supply-data.json> <source-document> <candidate-id> <formula> <max|mass-weighted>

The CLI performs no network access and does not include host-local paths in the receipt.

## Deliberate non-claims

A successful concentration receipt does not establish:

- absolute resource availability;
- reserves/resources;
- future mine/refinery capacity;
- probability of disruption;
- substitution difficulty;
- recycling availability;
- trade restrictions;
- political stability;
- co-product/by-product dependence;
- demand growth or deficit risk;
- environmental/social impact;
- total supply resilience.

Those should remain independent evidence dimensions or explicitly composed higher-level analyses.
