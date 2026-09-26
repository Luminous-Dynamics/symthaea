# TE-BENCH-001D — Exact workbook parser and normalization architecture

Status: architecture-only design subject

Date: 2026-09-26

Parents / related work:

- TE-BENCH-001A #6088 — benchmark custody architecture
- TE-BENCH-001B #6093 — frozen source manifest
- TE-BENCH-001C #6095 — exact source capture receipt
- TE-CORE-001A #6074 — thermoelectric evidence architecture
- TE-CORE-001C #6086 — independent synthetic reference, still qualification-gated

## Purpose

Define the deterministic boundary between an exact externally captured workbook artifact and a later normalized thermoelectric source-row corpus.

This subject exists because:

```text
exact workbook bytes
!= workbook successfully opened
!= cell values read
!= source row reconstructed
!= units understood
!= material state identified
!= component-property observation
!= benchmark admission
```

The first external artifact under this design is ESTM:

```text
KRICT-DATA/SIMD
commit: 72d597cb6fdf6b79f6ca86ff759f9136de5d66ef
path: dataset/estm.xlsx
Git blob: 3ec7c111d861cc2fe2446198f68a884d8f468f3b
size: 680110 bytes
```

That identity is source custody only. This document does not parse the workbook and creates no observation authority.

## Core theorem

A parser may prove only a narrower proposition:

```text
exact workbook artifact
+ exact parser/runtime/profile
+ deterministic sheet/cell traversal
+ explicit raw-to-normalized transformations
+ source-location custody
-> replayable parsed source records
```

It still does **not** prove:

- the paper's scientific claim is correct;
- a material formula maps to one unique phase;
- a dopant/carrier/specimen state is known;
- multiple property rows are mutually compatible;
- a reported zT can be independently reconstructed;
- a row belongs in a particular train/test split;
- the source is independent of another source;
- any candidate is prospective or novel.

## Layer separation

Do not expose one generic `ThermoelectricRow` that silently mixes parser, subject and scientific semantics.

Use an explicit progression such as:

```text
ExternalArtifactIdentity
    ↓
WorkbookParseReceipt
    ↓
RawWorkbookCell / RawWorkbookRow
    ↓
NormalizedSourceField / NormalizedSourceRow
    ↓
SubjectResolutionCandidate
    ↓
ConditionedPropertyCandidate
    ↓
[separate scientific/custody admission]
```

Each arrow is a separate theorem and may fail independently.

## Exact parser identity

`WorkbookParserProfileV1` should bind at minimum:

- exact source artifact identity;
- exact parser implementation artifact/version;
- exact language/runtime version;
- exact workbook library/package identity and dependency closure where feasible;
- parser configuration;
- formula handling policy;
- date/time conversion policy;
- locale/decimal/thousands-separator policy;
- Unicode normalization policy;
- blank-cell policy;
- spreadsheet-error policy;
- merged-cell policy;
- hidden row/column/sheet policy;
- sheet ordering policy;
- row/column traversal policy;
- numeric coercion policy;
- string trimming policy;
- unit-token normalization profile;
- deterministic output serialization profile.

Changing any claim-relevant parser/profile element creates a new parse lineage.

## `WorkbookParseReceiptV1`

Bind:

- source capture receipt identity;
- source Git blob and expected byte count;
- observed input-byte digest if/when locally materialized;
- parser profile identity;
- ordered sheet inventory;
- each sheet's dimensions as observed by the parser;
- hidden/visible state;
- merged-region inventory when present;
- formula-cell census;
- error-cell census;
- non-empty cell census;
- parse warnings/refusals;
- deterministic receipt digest.

A successful workbook open is not enough.

```text
openpyxl/library opened file
!= source semantics parsed
```

## Raw cell custody

Every normalized field must remain traceable to raw workbook coordinates.

Conceptually:

```text
RawWorkbookCellRefV1 {
  source_artifact_ref,
  sheet_name,
  sheet_index,
  row_index,
  column_index,
  a1_coordinate,
  raw_cell_type,
  raw_value_representation,
  formula_text_if_any,
  cached_value_state_if_any,
  style_or_number_format_ref_if_material,
}
```

Do not store only the final normalized number.

For every value later used scientifically, preserve enough information to answer:

> Which exact cell(s) in which exact workbook bytes produced this value?

## Formula policy

Spreadsheet formula semantics are especially dangerous because a library may expose a cached result, formula text, or both.

Preserve:

```text
formula expression
!= cached workbook value
!= freshly recalculated value
```

The first profile should default to one of these explicit dispositions:

- `LiteralValueCell`;
- `FormulaWithTrustedSourceCachedValue`;
- `FormulaCachedValueUnavailable`;
- `FormulaRecalculationRequired`;
- `FormulaUnsupported`.

Do not silently execute spreadsheet formulas inside the scientific parser unless an exact spreadsheet-engine execution profile is separately qualified.

If the workbook has no claim-relevant formulas, prove that with a census rather than assuming it.

## Merged cells and structural presentation

Merged headers often encode semantic hierarchy.

A parser must not fill forward merged-cell values silently and then pretend they were ordinary source cells.

If a normalized column name is derived from stacked/merged headers, bind all contributing header cells and the header-composition rule.

Likewise:

```text
visual table region
!= logical dataset table
```

until its boundaries are explicitly frozen.

## Hidden content

Hidden rows, columns or worksheets cannot be silently dropped or admitted.

Record them and assign one of:

```text
IncludedByProfile
ExcludedPresentationOnly
ExcludedAuxiliary
ExcludedWithReason
UnknownPurposeReviewRequired
```

A later discovery that a hidden sheet contains claim-relevant data changes the parser/benchmark lineage.

## Raw row identity

A `RawWorkbookRowV1` should bind:

- exact sheet;
- source row number;
- ordered source cell refs;
- header mapping profile;
- row parse status;
- exact original textual/numeric representations;
- deterministic row identity.

Row identity must not depend on normalized composition alone.

```text
same formula + same temperature
from two workbook rows
!= automatically duplicate evidence
```

They may represent different specimens, papers, methods, or repeated observations.

## Normalization must be field-specific

No universal string-to-number cleanup function should have scientific authority.

Separate normalization profiles for fields such as:

- chemical formula / material label;
- temperature;
- Seebeck coefficient;
- electrical conductivity;
- thermal conductivity;
- power factor;
- zT;
- source DOI/reference;
- measurement direction;
- carrier/doping metadata;
- specimen/process metadata.

Each profile should preserve raw text and normalized representation side by side.

## Units

A numeric value without a source unit is not ordinary normalized scientific evidence.

Preserve:

```text
raw numeric token
+ raw unit token/header context
+ unit interpretation profile
-> normalized quantity candidate
```

not:

```text
raw number
-> assume SI
```

For unit conversions, bind:

- source unit token;
- canonical unit;
- exact scale/offset operation;
- unit parser/profile identity;
- uncertainty if unit is ambiguous.

Do not infer units from typical literature conventions when the exact source representation is unresolved.

## Thermoelectric quantity classes

The parser/normalizer must not collapse the following:

```text
SeebeckCoefficient
ElectricalConductivity
ElectricalConductivityOverTau
ThermalConductivityTotal
ThermalConductivityElectronic
ThermalConductivityLattice
PowerFactorReported
PowerFactorRecomputed
ZTReported
ZTRecomputed
```

The same source column label may require `AmbiguousPropertyKind` rather than a guessed mapping.

## Reported versus recomputed quantities

Preserve source-reported values separately from later derived values.

```text
source-reported power factor
!= recomputed S^2 sigma

source-reported zT
!= independently recomputed zT
```

A later consistency validator may compare them, but the parser must never overwrite one with the other.

## Temperature and carrier state

Temperature must bind source value, normalized value and unit.

Carrier state is more difficult. A material row containing only a nominal chemical formula and temperature does not prove:

- carrier concentration;
- carrier type;
- chemical potential;
- dopant site;
- compensation state;
- Hall-measured state.

Represent absent claim-critical state as `Unknown`, not as a default.

## Composition parsing boundary

A chemical formula parser may normalize syntax, element ordering or stoichiometry only under a declared profile.

It may not prove:

- crystal structure;
- phase purity;
- site occupancy;
- dopant location;
- vacancy state;
- specimen identity;
- equivalence between nominal and measured composition.

Conceptually:

```text
ParsedCompositionCandidate
!= CanonicalThermoelectricSubject
```

Subject resolution belongs to a later alias/lineage stage.

## Source reference / DOI custody

If the workbook contains references, preserve exact source tokens and normalization lineage.

A DOI normalization may remove URL wrappers/case differences, but:

```text
same DOI
!= same specimen
!= same measurement condition
!= duplicate row
```

A paper can contain many specimens and many measurements.

## Missingness

Missing data is scientific information.

At minimum distinguish:

```text
CellBlank
ExplicitNA
ExplicitNotMeasured
BelowDetectionOrResolution
ParserCouldNotInterpret
SourceAmbiguous
NotApplicable
UnknownReason
```

Do not map all missingness to numeric zero or one null class if the source differentiates them.

## Numeric precision

Preserve source lexical precision before converting to floating point.

For example:

```text
"0.20"
!= source representation "0.2"
```

for provenance, even if the normalized numeric value is equal.

When scientific calculations later require floating point, bind conversion profile and avoid implying source precision greater than reported.

## Duplicate and alias boundary

Parser-level exact duplicate detection may identify byte/row equality, but semantic duplicates require later subject/source analysis.

Keep separate:

```text
ExactRowDuplicate
NormalizedFieldDuplicate
SamePaperCandidate
SameMaterialCandidate
SameSpecimenCandidate
SemanticAliasUnknown
```

Do not deduplicate on formula + temperature alone.

## Error/refusal vocabulary

Suggested parse dispositions:

```text
ParsedWithoutWarning
ParsedWithNonScientificWarning
AmbiguousHeader
AmbiguousUnit
UnsupportedCellType
UnsupportedFormula
ConflictingMergedHeader
MissingRequiredField
InvalidNumericToken
ScientificFieldKindUnknown
SourceReferenceMalformed
ParserInvariantViolation
```

A rejected row remains part of the parse census.

## Independent validation

The eventual `TE-BENCH-001D` executable train should not test against ESTM first.

Use a deliberately small synthetic workbook containing hostile spreadsheet features:

1. clean literal table;
2. merged multi-row header;
3. hidden auxiliary sheet;
4. hidden claim-relevant row;
5. formula cell with cached value;
6. formula without cached value;
7. spreadsheet error cell;
8. blank vs explicit `N/A`;
9. unit in header only;
10. conflicting row-level unit;
11. Unicode minus sign;
12. scientific notation;
13. decimal comma / locale trap;
14. numeric-looking string;
15. leading/trailing whitespace;
16. duplicate rows from same source;
17. same composition/temperature from different DOI;
18. same DOI but different specimen label;
19. `sigma/tau` mislabeled as `sigma` hostile case;
20. total kappa mislabeled lattice kappa hostile case;
21. reported PF inconsistent with recomputation;
22. reported zT inconsistent with components;
23. missing carrier state;
24. formula normalization collision;
25. row-order permutation with stable semantic output but different raw row identity.

The independent reference validator must derive the expected normalized representation from workbook facts, not case IDs.

## Determinism / metamorphic tests

Require at least:

- same exact workbook + same parser profile -> byte-identical normalized output;
- changed source workbook blob -> new parse lineage;
- changed parser/library version -> new parse lineage unless explicitly proven irrelevant under profile;
- reordered sheets -> raw structural identity changes even if normalized records compare equal;
- changed header unit -> normalized quantity identity changes;
- changed cell lexical value with numerically equal float -> raw provenance changes;
- changed formula text with same cached value -> raw provenance changes;
- changed hidden-row policy -> parser profile identity changes;
- same normalized row from different source coordinates -> distinct raw-row ancestry;
- failed rows cannot disappear from the census.

## Suggested implementation train

```text
TE-BENCH-001D0
this architecture

TE-BENCH-001D1
synthetic hostile XLSX fixture

TE-BENCH-001D2
independent stdlib/minimal reference expectations
+ exact workbook parser harness under pinned runtime

TE-BENCH-001D3
ESTM parse receipt over exact blob 3ec7c111...

TE-BENCH-001E
subject / alias / leakage graph

TE-BENCH-001F
condition-bearing observation corpus

TE-BENCH-001G
independent custody / split validator

only then:
retrospective model scoring
```

The production parser should remain a source-ingestion component. It must not live inside TE-CORE scientific authority simply because it reads thermoelectric values.

## Qualification rule

Every executable parser/reference subject receives its own exact-head qualification.

A parser qualification may establish deterministic extraction/normalization under the exact source/profile. It cannot establish scientific correctness of the source, subject equivalence, measurement quality or benchmark validity by itself.

## Claim ceiling

TE-BENCH-001D may establish that exact workbook bytes were deterministically parsed into source-located, typed, unit-aware normalized records under an exact parser profile. It does not establish material identity, phase/doping/carrier state, component compatibility, thermoelectric performance, model accuracy, synthesis, device efficiency, or discovery.
