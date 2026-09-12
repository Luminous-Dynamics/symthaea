# Symthaea Critical-Material Burden Evidence

This crate produces one narrow Tier-1 material-screening metric:

`critical_material_mass_fraction`

It answers only:

> What fraction of this exact formula mass is contributed by elements appearing on this exact externally supplied designation list?

It does **not** define which elements are globally or permanently critical, and it does not estimate total supply, geopolitical, environmental, economic, recycling, toxicity, or manufacturability risk.

## Evidence boundary

The changing critical-element designation lives outside the Rust binary. A normalized JSON list contains:

- source title;
- source version/date;
- source URI;
- SHA-256 of the authoritative source document;
- normalization timestamp and extraction note;
- element symbols.

The calculation API also requires the original source-document bytes. Their SHA-256 must match the declared source digest before a prediction can be emitted.

The normalized JSON itself is independently content-addressed, and the canonical designation list receives a separate domain-separated digest. This keeps three identities distinct:

1. authoritative source document;
2. normalized designation capture;
3. candidate-specific calculation receipt.

## Calculation

For formula element `i` with stoichiometric count `n_i` and standard atomic mass `m_i`:

`mass_i = n_i * m_i`

The reported value is:

`sum(mass_i for designated critical elements) / sum(mass_i for all formula elements)`

Element masses come from `symthaea-bandgap`'s periodic-table data. Formula parsing reuses the fail-closed simple formula parser from `symthaea-process-discovery`.

The parser deliberately rejects unsupported chemistry rather than guessing. Parentheses, hydrates, fractional occupancy, disorder, isotopic notation, and non-stoichiometric formulas require a future richer composition adapter.

A value of `0.0` means only that no element in the exact formula appears on the exact supplied list. It is **not** evidence of zero supply risk.

## Discovery evidence semantics

A successful calculation emits a `symthaea-discovery::Prediction` with:

- metric: `critical_material_mass_fraction`;
- unit: `fraction`;
- fidelity: `Analytical`;
- supporting evidence kind: `Dataset`;
- explicit source-document SHA-256;
- exact normalized-list identity;
- a per-element mass ledger.

No calibrated uncertainty model currently exists for applicability of an external critical-elements designation. The generic prediction therefore marks epistemic uncertainty as fully unknown and attaches no confidence interval.

## Normalized list example

The following is only a schema example, not an authoritative list:

    {
      "source_title": "Example authoritative critical-elements publication",
      "source_version": "2026-09-12",
      "source_uri": "https://example.org/source",
      "source_document_sha256": "<sha256 of exact source bytes>",
      "normalized_at_utc": "2026-09-12T18:00:00Z",
      "extraction_note": "Manually normalized from table 1; second reviewer checked symbols.",
      "elements": ["Co", "Li", "Ni"]
    }

Do not treat that example element list as scientific or policy data.

## Network-free CLI

    critical-material-burden <critical-elements.json> <source-document> <candidate-id> <formula>

The command:

1. reads the exact normalized list bytes;
2. reads the exact authoritative source-document bytes;
3. verifies the declared source SHA-256;
4. parses the candidate formula;
5. computes the transparent mass ledger;
6. emits a JSON receipt.

Host-local paths are not included in the receipt.

## Deliberate non-claims

A successful receipt does not establish:

- total material criticality;
- supply security or resilience;
- market concentration;
- resource abundance;
- mine/refinery capacity;
- geopolitical risk;
- toxicity or environmental impact;
- recyclability;
- synthesizability or manufacturability;
- device performance;
- scientific validation;
- deployment fitness.

Those are separate evidence dimensions and should remain separate.
