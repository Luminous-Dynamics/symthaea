# symthaea-materials-project-stability

Network-free Materials Project thermodynamic-stability evidence adapter for Symthaea's Tier-1 energy-material screening.

The adapter consumes an immutable JSON export from Materials Project `SummaryDoc` records and turns one explicitly bound MP material into an `energy_above_hull` discovery prediction.

## Why this is the first Tier-1 external adapter

`energy_above_hull` has a concrete thermodynamic interpretation and current Materials Project `SummaryDoc` exposes it in eV/atom. It is useful for screening thermodynamic stability, while remaining clearly **computed evidence** rather than experimental synthesis proof.

## Network boundary

This crate performs no HTTP requests and accepts no API key.

Acquisition is intentionally separate:

1. use the official Materials Project client outside the scientific evaluation path;
2. export the returned `SummaryDoc` objects to JSON;
3. record the client/model versions and exact requested fields in metadata;
4. pass those files into this adapter;
5. retain the raw JSON SHA-256 in every downstream receipt.

This prevents a live database response from silently changing during a supposedly reproducible Symthaea run.

## Required SummaryDoc fields

The capture metadata must state that these fields were requested:

- `material_id`
- `formula_pretty`
- `energy_above_hull`
- `formation_energy_per_atom`
- `deprecated`
- `last_updated`
- `warnings`

Unknown fields in exported SummaryDoc objects are ignored by the normalized parser but remain bound by the exact raw JSON SHA-256.

## Example external acquisition

An external Python acquisition step can use the official `mp-api` client, for example:

```python
import json
from mp_api.client import MPRester

fields = [
    "material_id",
    "formula_pretty",
    "energy_above_hull",
    "formation_energy_per_atom",
    "deprecated",
    "last_updated",
    "warnings",
]

with MPRester("YOUR_API_KEY") as mpr:
    docs = mpr.materials.summary.search(
        material_ids=["mp-149"],
        fields=fields,
    )

with open("summary-docs.json", "w", encoding="utf-8") as handle:
    json.dump([doc.model_dump(mode="json") for doc in docs], handle, sort_keys=True)
```

The acquisition environment should separately record exact `mp-api` and `emmet` versions. This Rust crate does not infer or trust package versions from the JSON payload.

A matching metadata file has this shape:

```json
{
  "source_name": "Materials Project",
  "api_route": "/materials/summary",
  "mp_api_version": "<exact-version>",
  "emmet_version": "<exact-version>",
  "retrieved_at_utc": "<external timestamp>",
  "query_description": "exact MP material ids requested for Tier-1 stability evidence",
  "requested_fields": [
    "material_id",
    "formula_pretty",
    "energy_above_hull",
    "formation_energy_per_atom",
    "deprecated",
    "last_updated",
    "warnings"
  ]
}
```

The timestamp is provenance text, not trusted chronology proof.

## Candidate/material identity boundary

Thermodynamic stability is structure/material-entry specific. A composition-only generated candidate must not silently inherit the stability of a convenient polymorph.

Bindings therefore use one of two explicit modes:

- `ExactMaterialId`: candidate ID must exactly equal the MP material ID;
- `ExplicitMapping`: a differently named candidate may be mapped only with a non-empty human-reviewable note explaining the identity assertion.

The second mode records an assertion; it does not independently prove structural identity.

## Evidence semantics

A successful binding emits:

- metric: `energy_above_hull`;
- unit: `eV/atom`;
- fidelity: `FirstPrinciples`;
- supporting evidence kind: `Dataset`;
- exact raw capture SHA-256;
- normalized capture SHA-256;
- MP material ID/formula/last-updated metadata;
- source warnings;
- client/model versions;
- explicit binding assumption.

This distinction is deliberate. The value is produced from first-principles thermodynamic calculations/phase-diagram processing and distributed through a database capture; it is not an experimental measurement.

## Uncertainty boundary

The adapter does not invent a calibrated interval.

The generic discovery uncertainty record is emitted as:

- epistemic = `1.0` — the generic contract's fully-unknown endpoint;
- aleatoric = `0.0` — no stochastic noise model is attached to this deterministic captured database value;
- interval = `None`.

The receipt explicitly states that `aleatoric = 0` is not a claim that nature or experiment is noiseless.

## Deprecated entries

Deprecated Materials Project records may be captured for audit/history, but they cannot satisfy the stability dimension. Binding one fails closed.

## CLI

```text
materials-project-stability-evidence \
  <summary-docs.json> \
  <capture-metadata.json> \
  <candidate-id> \
  <material-id> \
  [explicit-mapping-note]
```

Omit the mapping note only when candidate ID and MP material ID are exactly identical. Host-local file paths are not emitted into the evidence receipt.

## Deliberate non-claims

This adapter does not establish:

- experimental thermodynamic stability;
- synthesizability or kinetic accessibility;
- phase purity;
- device performance;
- material novelty;
- safety;
- manufacturability;
- calibrated uncertainty;
- deployment fitness or physical authority.

It supplies one explicit Tier-1 evidence dimension. The other dimensions must remain incomplete until their own qualified evidence adapters exist.
