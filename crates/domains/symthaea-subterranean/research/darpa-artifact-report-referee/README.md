# DARPA Artifact Report Referee v0.1

**Status:** standalone reference implementation and synthetic tests authored. No external dataset bytes are vendored.

## Purpose

Evaluate candidate artifact reports against a separately held ground-truth catalogue. The evaluator is intentionally separate from the perception/planning path. Two distinct evaluation modes are provided:

1. A generic offline evaluator that uses configurable tolerance and one-to-one maximum-cardinality matching.
2. A historical DARPA Systems Final stream evaluator that replays the documented report-order, retry, duplicate and budget semantics.

## Data intake and provenance

The source DARPA TSV parser accepts `type<TAB>x<TAB>y<TAB>z`, including the optional trailing empty tab field. It creates deterministic IDs using the source file identifier and 1-based line number. The source-native type label is preserved exactly.

The source manifest `DARPA_GROUND_TRUTH_SOURCE_MANIFEST_V0_1.json` pins the source tree commit and per-file Git blob IDs, byte lengths and row counts. It does not include source dataset binaries. Data redistribution rights remain explicitly unresolved.

## Coordinate contract

Every truth and candidate record has an exact coordinate-frame ID. The evaluation profile must name the expected frame; frame mismatches reject. Coordinates must already be in metres in that exact frame. This evaluator never guesses CRS, axis order, units or transformations.

The historical Systems Final function expects the DARPA reference frame `darpa` (the competition frame aligned at the Starting Gate). Any transform must be a separate, versioned and uncertainty-carrying operation before candidate admission.

## Type labels and the official crosswalk

Two type vocabularies are kept separate:

- Ground-truth CSV labels: e.g. `cellphone`, `fireextinguisher`.
- Official Final Event report strings from the DARPA Artifacts Specification / ICD: e.g. `Cell Phone`, `Fire Extinguisher`.

The explicit `darpa_final_2021_crosswalk_v0_1()` maps source strings to official report strings. The crosswalk carries a version ID and is validated for duplicate source names, duplicate official names and unsupported official values. No implicit lowercase, whitespace or spelling normalization is performed.

## Evaluation modes

### Generic evaluator — `evaluate_reports`

Inputs:
- ground-truth ID, source label, exact frame ID and x/y/z;
- candidate report ID, type label, exact frame ID, x/y/z and monotonic submission sequence;
- an explicit Euclidean 3-D tolerance in metres and optional maximum report count.

Reports carry unique monotonic submission sequence numbers; duplicate sequence values reject as ambiguous. Reports are selected in submission order, then matched one-to-one to ground truth. It uses deterministic maximum-cardinality bipartite matching so a greedy nearest-neighbour mistake cannot reduce the total match count and duplicate reports cannot claim one truth record twice. This mode is a generic offline evaluator; it is not the historical DARPA command-post state machine.

### Historical Systems Final replay — `evaluate_systems_final_stream`

Pinned rule source: *DARPA Subterranean Challenge — Competition Rules, Final Event, Revision 1, 25 May 2021*, Sections 7.3, 8.3.5, 10.1 and 10.3; artifact report strings are from *Artifacts Specification, Final Event, 1 April 2021* and the corresponding ICD.

The Systems Final rules define:
- Preliminary run: 20 artifacts, maximum 25 artifact reports.
- Prize run: 40 artifacts, maximum 45 artifact reports.
- One point for an exact official artifact report string and Euclidean 3-D error **less than or equal to 5 m**.
- An invalid report (wrong type or insufficient accuracy) can be corrected by another report.
- Once an artifact has been successfully reported, later duplicate reports are invalid and still count against the report allowance.
- Reports beyond the allowance are rejected.
- Report time is not a direct score factor, but report times participate in final ranking tie-breakers.

The evaluator requires unique `submission_sequence` values and replays reports in that order. The first valid report wins per ground-truth item; invalid attempts leave the item available for retry; reports beyond the round allowance are excluded. It emits points, invalid reports, missed artifacts, processed report count, budget-rejected report IDs and the crosswalk version.

**Deliberate limits:** it does not reproduce the DARPA transport/API, real-time score receipts, time-based tie-break ranking, course-exploration validation, judge decisions or the Virtual Competition's different duplicate-report handling. If multiple unreported ground-truth items of the same class lie within 5 m of one report, v0.1 chooses the nearest (then lexical ID) as a deterministic local tie-break; the rules document does not specify this ambiguous-neighbour case. Do not describe this kernel as a full official command-post clone.

## Benchmarks and tests

The source-authored suite includes:
- exact type and 3-D Euclidean threshold;
- vertical error;
- duplicate reports and one-to-one matching;
- a nearest-neighbour trap where maximum-cardinality assignment matters;
- submission-order report budgets;
- class and distance false positives/negatives;
- input-order invariance;
- invalid and non-finite records;
- frame mismatch;
- TSV parsing with trailing tab fields and raw class preservation;
- Systems Preliminary retry-after-invalid / duplicate-after-success;
- Prize round 45-report cap;
- exact official report string crosswalk.

These are synthetic conformance fixtures. They do not prove performance on a live robot or re-create the DARPA official score logs.

## Run

```sh
cargo fmt --all -- --check
cargo test --offline --all-targets
cargo clippy --offline --all-targets -- -D warnings
```

No external crate dependencies are required.

## Claim ceiling

A passing synthetic test suite establishes only the declared evaluator contract. It does not validate perception, localization, map quality, operator workflow, communication, mission survival, structural/geotechnical safety, mine access or planetary transfer.
