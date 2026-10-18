# NIST AM Bench source-bound adapter contract v0.1

## Scope and paradigm shift

Move manufacturing prediction work from “the synthetic validator accepted a scenario” toward an auditable chain:

source DOI → exact source artifact → preserved measurements → explicit material/process lineage → frozen evaluation split → independently recomputed metric.

The adapter establishes source and data-contract integrity before any model comparison. It does not claim that source parsing means a model is accurate or that a challenge profile qualifies a physical part.

## Selected profile

NIST AMB2025-02 describes macroscale quasi-static tensile testing of PBF-L IN718 in the as-built condition. The official README says eight miniature tensile specimens were excised from the same parent specimen, AMB2022-718-AMMT-B7-P4; testing followed ASTM E8 at a strain rate of 1×10^-3 s^-1 and a 3 mm gauge length. The requested target is average tensile properties across those eight specimens, with processing and microstructure data from AMB2022-01 used as calibration inputs.

Primary links:
- DOI and citation instructions: https://doi.org/10.18434/mds2-3735
- NIST PDR discovery link: https://data.nist.gov/od/id/mds2-3735
- NIST AMB2025-02 challenge README: https://www.nist.gov/system/files/documents/2025/03/05/readme%20AMB2025-02%20Macroscale%20Quasi-Static%20Tensile%20Tests%20of%20PBF-L%20IN718.pdf
- Upstream AMB2022-01 microstructure dataset DOI: https://doi.org/10.18434/mds2-2767
- NIST direct-link and citation guidance: https://www.nist.gov/ambench/direct-am-bench-data-links-and-referencing-guidance

The challenge is not eight independent builds. The source records one parent specimen from which the eight target specimens were cut. Randomly assigning sibling specimens to train/test would inflate independent evidence. This profile therefore binds outcome records from the target parent lineage to one held-out group. Any later benchmark must state exactly which source-derived features were available at fit time and which measured outcome labels were withheld.

## Source and rights gate

The committed profile is metadata-only. It contains no dataset bytes and has no invented artifact hashes. Artifact-specific terms, local use and redistribution permissions remain unreviewed. Before files may be used:
1. Inspect the actual DOI landing page attribution/citation instructions and the terms applying to each downloadable artifact.
2. Record access date, artifact URL/path, media type, byte count and SHA-256 from the exact lawfully obtained bytes.
3. Set terms and local-use flags only when the evidence supports those choices. Repository bundling has its own separate permission flag.
4. Keep artifact files external to the repository unless bundling is explicitly permitted and reviewed.

A DOI, a public PDR record or a source link is not itself a license grant.

## Record contract

Every observation carries:
- unique record ID and declared source DOI;
- specimen ID and parent/lineage group ID;
- variable ID, raw value and raw unit;
- optional normalized value/unit plus a conversion note, while preserving raw value/unit;
- reported measurement uncertainty and its unit, if supplied;
- censoring semantics, including explicit unknown status and explanation;
- record role (predictor vs target outcome) and split (calibration vs held out).

Unknown metadata must remain explicit. The validator rejects placeholder IDs such as “unknown” where a stable record/lineage identity is necessary to establish split integrity. It never coerces censored or missing measurements to zero. It rejects undeclared source DOIs, duplicate record IDs, unjustified normalization, unexplained unknown censoring and calibration target outcomes from the held-out target lineage.

## Benchmark protocol and proposed gates

The first stage is data integrity—not model competition.

**Source integrity**
- 100% of declared local artifacts match byte count and SHA-256.
- Exact source DOI, artifact URL, citation instructions, parser/schema version and source revision are bound in a receipt.
- No artifact is imported before artifact-specific terms and local-use permissions are reviewed.

**Parser integrity**
- 100% of duplicate IDs, undeclared source identities, unsafe file paths, missing stable lineage, invalid units, unjustified conversion, invalid/non-finite values and malformed censoring controls are rejected.
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
