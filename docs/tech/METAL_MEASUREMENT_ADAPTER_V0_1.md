# NIST AM Bench source-bound adapter contract v0.1

## Scope and paradigm shift

Move manufacturing prediction work from “the synthetic validator accepted a scenario” toward an auditable chain:

source DOI → exact source artifact → preserved measurements → explicit material/process lineage → frozen evaluation split → independently recomputed metric.

The adapter establishes source and data-contract integrity before any model comparison. It does not claim that source parsing means a model is accurate or that a challenge profile qualifies a physical part.

## Selected profile

**Historical challenge:** the original AMB2025-02 submission deadline was August 29, 2025. This protocol treats the publication as a retrospective public evaluation profile, not an open challenge entry.

NIST AMB2025-02 describes macroscale quasi-static tensile testing of PBF-L IN718 in the as-built condition. The official README says eight miniature tensile specimens were excised from the same parent specimen, AMB2022-718-AMMT-B7-P4; testing followed ASTM E8 at a strain rate of 1×10^-3 s^-1 and a 3 mm gauge length. The requested target is average tensile properties across those eight specimens, with processing and microstructure data from AMB2022-01 used as calibration inputs.

Related AMB2022-01 source publications listed by NIST include the 3D-build modeling-description data (DOI 10.18434/mds2-2607), in-situ thermography/processing scripts (10.18434/mds2-2715), microstructure measurements (10.18434/mds2-2692), residual strain/stress/deflection measurements (10.18434/mds2-2711), and serial-sectioning/X-ray CT measurements (10.18434/mds2-2767). The 2025 challenge README explicitly links the latter DOI for EBSD-related calibration data. These are cataloged as candidate related sources only; the official submission template and artifact-level fields still determine which source objects are actually needed.

Primary links:
- DOI and citation instructions: https://doi.org/10.18434/mds2-3735
- NIST PDR discovery link: https://data.nist.gov/od/id/mds2-3735
- Official calibration archive: https://data.nist.gov/od/ds/mds2-3735/calibration%20data.zip
- Official prediction submission template: https://data.nist.gov/od/ds/mds2-3735/prediction%20submission%20template.xlsx
- NIST fair-use, licensing and citation policy: https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications
- NIST AMB2025-02 challenge README: https://www.nist.gov/system/files/documents/2025/03/05/readme%20AMB2025-02%20Macroscale%20Quasi-Static%20Tensile%20Tests%20of%20PBF-L%20IN718.pdf
- Upstream AMB2022-01 microstructure dataset DOI: https://doi.org/10.18434/mds2-2767
- NIST direct-link and citation guidance: https://www.nist.gov/ambench/direct-am-bench-data-links-and-referencing-guidance

The challenge is not eight independent builds. The source records one parent specimen from which the eight target specimens were cut. Randomly assigning sibling specimens to train/test would inflate independent evidence. This profile therefore binds outcome records from the target parent lineage to one held-out group. Any later benchmark must state exactly which source-derived features were available at fit time and which measured outcome labels were withheld.

## Source and rights gate

The committed profile is metadata-only. It contains no dataset bytes and has no invented artifact hashes. NIST's general fair-use policy states that data/works created by NIST employees and not covered by the Standard Reference Data Act are generally not subject to U.S. copyright under 17 U.S.C. §105, while warning that third-party/extramural material and foreign rights may differ. That general policy is useful context but does not, by itself, resolve the authorship/rights status of every file in this particular archive. The exact PDR record and archive/template attribution statements still need to be reviewed before populating artifact-specific permissions. Artifact-specific terms, local use and redistribution permissions therefore remain unreviewed. Before files may be used:
1. Inspect the actual DOI landing page attribution/citation instructions and the terms applying to each downloadable artifact.
2. Record source object ID/revision, access date, artifact URL/path, terms URL, media type, byte count and SHA-256 from the exact lawfully obtained bytes. Record local-use and repository-bundling permissions per artifact as well as in the dataset policy.
3. Set terms and local-use flags only when the evidence supports those choices. Repository bundling has its own separate permission flag.
4. Keep artifact files external to the repository unless bundling is explicitly permitted and reviewed.

A DOI, a public PDR record or a source link is not itself a license grant.

## Record contract

Every observation carries:
- unique record ID, declared source DOI, source-object ID, upstream row ID and exact source-field identity;
- a source-object entry in the manifest and a declared artifact digest for that object;
- consistent variable identity/raw unit for each repeated source field and a consistent normalized unit for each normalized variable;
- specimen ID and parent/lineage group ID;
- variable ID, raw value and raw unit;
- optional normalized value/unit plus a conversion note, while preserving raw value/unit;
- reported measurement uncertainty and its unit, if supplied;
- censoring semantics and explicit finite lower/upper bounds for left-, right- and interval-censored records; never substitute a point estimate for a censored value;
- explicit unknown status and explanation;
- record role (predictor vs target outcome) and split (calibration vs held out).

Unknown metadata must remain explicit. The validator rejects placeholder IDs such as “unknown” where a stable record/lineage identity is necessary to establish split integrity. It never coerces censored or missing measurements to zero. It rejects undeclared source DOIs, duplicate record IDs, unjustified normalization, unexplained unknown censoring and calibration target outcomes from the held-out target lineage.

## Benchmark protocol and proposed gates

The first stage is data integrity—not model competition. CLI record validation verifies artifact bytes before accepting observation rows. The `inspect-csv` command inventories the exact hashed CSV's parsed record count, headers and per-column empty/nonempty counts. A strict quote-syntax guard rejects unterminated quoted fields, quotes inside unquoted fields and junk after a closing quote; row widths and ambiguous headers are checked by the CSV parser/inventory. Inspection parses the rechecked byte buffer and binds the inventory to its exact SHA-256 to avoid a verify-then-reopen time-of-check/time-of-use gap.

For archive-based sources, `inspect-zip` verifies the manifest-bound ZIP bytes and inventories its central-directory members without extracting them. It rejects absolute/traversal paths, backslash/colon ambiguity, repeated or dot path components, Windows-reserved/trailing-dot/trailing-space names, symlink entries, duplicate or case-colliding names, and archives above the member-count limit. The receipt binds the exact outer archive digest, member paths, declared compressed/uncompressed sizes and CRC-32 metadata. ZIP CRC-32 is not treated as a cryptographic content digest; selected members must still be lawfully accessed, extracted through a safe path, and individually SHA-256 verified before semantic parsing. ZIP inventory proves structure/path safety only, not licensing or measurement truth.

The new `reconcile-csv` operation adds an explicit source mapping rather than guessing semantics from header names. Its mapping contract declares the exact source-native row-ID header, exact source header, canonical field ID, source unit/type label, metadata/predictor/target role and missing-cell policy. Reconciliation requires unique nonblank source-native row IDs, exact matching headers, unique canonical field IDs, and a declared missing-value policy. **Every other source header must be mapped or explicitly excluded with a nonblank reason**; any unmapped/unexcluded column fails closed. The receipt binds the source artifact digest, mapping digest, header digest, ordered source-row-ID digest, parsed row count, per-mapped-field nonempty/missing counts and per-excluded-column empty/nonempty counts. A field must declare `text` or `finite_number`; numeric fields are lexically parsed and reject malformed or non-finite values. Dataset-specific missing tokens (for example, a sentinel such as `NA`) must be explicitly declared and are counted separately. This establishes row/field and basic numeric-lexical integrity only: the tool does not validate unit dimensions, convert units, infer lineage, or prove model accuracy.

**Source integrity**
- 100% of declared local artifacts match byte count and SHA-256.
- Archive inventory rejects path traversal, symlinks, ambiguous member names and out-of-budget member counts without extracting files.
- Exact source DOI, artifact URL, citation instructions, parser/schema version and source revision are bound in a receipt.
- No artifact is imported before artifact-specific terms and local-use permissions are reviewed.

**Parser integrity**
- The structural CSV inventory rejects blank or case/whitespace-colliding headers, inconsistent row widths and invalid quote syntax, and accounts for empty versus nonempty cells by column.
- Source mapping is explicit and exact; every parsed source row has a unique stable row ID. Every non-row-ID header is mapped or excluded with a reason; zero unaccounted columns.
- Fields explicitly marked `finite_number` reject malformed and non-finite values. Missing sentinels are exact, case-sensitive and declared per field; zero silent sentinel inference.
- Missing cells/tokens follow a declared per-field policy and are never imputed silently.
- 100% of duplicate IDs, undeclared source identities, unsafe file paths, missing stable lineage, inconsistent source-field units/variable mappings, unjustified conversion, invalid/non-finite values, fabricated censored point values and malformed censoring bounds are rejected.
- Independent count reconciliation: every input row is either represented or listed with a reason for exclusion; no silent drops.
- Raw source values and units survive every normalization.

**Split integrity**
- Zero held-out target-lineage outcomes enter calibration.
- The split file is frozen and hashed before fitting or inspecting held-out metrics.
- Any sibling specimens from the same parent lineage remain grouped; no row-level random split is described as independent-build validation.
- Baselines use calibration data only. If the available calibration/target structure supports only a single target group, report a one-profile challenge result—not a population-generalization estimate.

**Prediction quality (not yet assessed)**
- Compare at least a transparent baseline and the candidate model on the exact declared target properties.
- Report absolute error per target, normalized error only with a declared denominator, uncertainty/interval coverage only where a calibrated interval method is supplied, and residuals by known process/orientation groups.
- Independently recompute every metric from the bound prediction and actual-value files.
- Do not set a numerical win target until the target columns, units, aggregation rule, permitted training inputs and source's official submission/measurement template have been reconciled.

## Limits

A valid manifest or artifact digest does not validate a prediction. A score on this dataset does not establish process capability, stock certification, service-life prediction, material allowable, finished-part conformity, safety, production yield, cost savings or applicability to other alloys/processes. Those require distinct evidence profiles and qualification paths.
