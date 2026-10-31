# Connectomics Intake Audit

Small, standalone Rust gate for the ConnectomeBench2 V2 intake process described in the Symthaea neuroscience research plan.

## Scope

This tool validates the declared manifest, checks task/label/input-mode consistency, rejects known EM-view leakage fields, requires the correct single-/dual-segment geometry preprocessing profile, hashes every declared local artifact with SHA-256, checks byte sizes and split declarations, and writes a deterministic JSON report.

It does **not** parse Parquet schemas, prove label correctness, validate the upstream commit contents, prove samples are independent, or measure model accuracy. A passing report only means the manifest and local bytes satisfy these checks.

## Build

This directory is a standalone Cargo workspace and is intentionally not a member of the main Symthaea workspace.

```sh
cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml
cargo build --release --manifest-path tools/connectomics-intake-audit/Cargo.toml
```

For reproducible CI, generate and commit this crate's `Cargo.lock`, then add `--locked` to both commands.

## Query-specific stopping snapshot audit

The `connectome-stopping-audit` binary validates a JSONL series against the initial versioned snapshot contract in `docs/neuroscience/CONNECTOME_STOPPING_SNAPSHOT.schema.json`. It checks immutable dataset/query/profile identity, strictly increasing snapshot sequence, parent graph digest continuity, nondecreasing edit counts, operation-class totals, invariant names, query metric names/units, and digest formatting. It also emits a descriptive tail-range stability signal and walk-forward predictions for declared graph invariants and scientific-query metrics, comparing a fitted asymptotic curve with persistence and linear-trend baselines.

```sh
cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome_stopping_audit -- \
  --input snapshot-series.jsonl \
  --tail-window 3 \
  --relative-tolerance 0.01 \
  --absolute-tolerance 1e-9 \
  --forecast-min-training-snapshots 4 \
  --output stopping-audit.json
```

Each JSONL record is one snapshot object conforming to `CONNECTOME_STOPPING_SNAPSHOT.schema.json`. Order records from oldest to newest. The first snapshot has sequence index 0 and no previous graph digest; each later snapshot names the immediately preceding snapshot's graph digest. Cumulative operation-class counts must sum to `cumulative_committed`. An illustrative three-snapshot fixture is at `docs/neuroscience/examples/CONNECTOME_STOPPING_SNAPSHOTS.example.jsonl`; it intentionally shows graph invariants looking stable while a query metric changes, and uses fake data/digests only.

**Important:** the tool emits report schema version 2 and always recommends `CONTINUE_REVIEW`; its status is `VALIDATED_NOT_STOPPED` only when consistency checks pass. Snapshots may optionally carry `synaptic_completeness`: the exact total synapse-population count and the count whose pre- and postsynaptic locations both belong to fully proofread cells, bound to the population definition, synapse-ID set and fully-proofread-cell-set digests. The report emits the latest fraction and a separate tail-stability/status (`AVAILABLE`, `INCOMPLETE_TAIL`, `ZERO_DENOMINATOR_NOT_ESTIMABLE`, `INSUFFICIENT_TAIL_HISTORY`, or `NOT_PROVIDED`). Missing values and zero denominators are not imputed; the completeness metric is never inferred from graph invariants. A synthetic false-stop fixture at `docs/neuroscience/examples/CONNECTOME_STOPPING_FALSE_STOP_COUNTEREXAMPLE.example.jsonl` deliberately shows stable invariants alongside a stable 20% synaptic completeness and a changing rare path query; the recommendation remains `CONTINUE_REVIEW`. The asymptotic curve is fitted using prior snapshots only and compared with last-value persistence and a linear-trend baseline; summaries report held-out MAE for all baselines and relative improvement on the exact same target snapshots where the asymptotic fit was estimable, avoiding a misleading comparison across different subsets. This comparison conditions on the target snapshot's cumulative edit count, available only retrospectively, so it does not yet forecast elapsed time or edits required to reach a quality threshold. No calibrated forecast intervals are implemented. Tail stability is descriptive, and neither stability nor curve fit proves biological correctness. The schema contains no independently calibrated residual high-impact-error risk, and global graph invariants can stabilize while local scientific errors persist. A valid automatic stopping gate requires separate risk evidence and held-out evaluation; this binary does not make that decision. The six-snapshot synthetic forecast fixture is `docs/neuroscience/examples/CONNECTOME_WALK_FORWARD_SNAPSHOTS.example.jsonl`; the three-snapshot counterexample remains `docs/neuroscience/examples/CONNECTOME_STOPPING_SNAPSHOTS.example.jsonl`. The separate false-stop fixture, `docs/neuroscience/examples/CONNECTOME_STOPPING_FALSE_STOP_COUNTEREXAMPLE.example.jsonl`, keeps graph invariants near their asymptote while synaptic completeness is stable at 20% and the critical path count changes 12→1. The stopping audit currently has 21 authored Rust regression tests by source count; they are not a test-pass claim.


## Run

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml -- \
  --manifest path/to/intake-manifest.json \
  --root path/to/local-dataset-artifacts \
  --output intake-audit-report.json
```

Every `input_files[].path` is relative to `--root`. Absolute paths, parent-directory traversal, symlink escapes, missing files, byte-count mismatches, and SHA-256 mismatches fail closed. Keep dataset license/source metadata with the manifest.

## Manifest contract

- Dataset ID is `jeffbbrown2/ConnectomeBench2`; dataset revision must be `v2` (not `main`). Record the exact dataset-revision commit SHA in the manifest.
- The upstream code commit is pinned to `5679b39aa2641d8d0afeed88b441e696f87a66df`. This pin was reviewed against `src/neurips_training/data.py::route_row` and `scripts/data_generation/neurips/postprocess_giga_parquet.py::derive_task_routing`. Change the pin only after auditing the replacement source commit and updating the code/schema contracts together.
- `false_split_correction` uses `false_split_correction_label = same_neuron` and dual-segment `geometry` for merge correction.
- `false_merge_identification` uses `false_merge_identification_label = !same_neuron` and single-segment `geometry_single` for split correction.
- Allowed model inputs are `geometry`, `geometry_single`, `em_xy`, `em_xz`, and `em_yz`. `em_best`, labels, IDs, routing, and metadata are not accepted as model inputs.
- Single-segment geometry declares `geometry_preprocessing_profile=single_segment_zero_mask_channels` to mirror the reference loader's defensive mask-zeroing step. Dual geometry declares `dual_segment_native`.
- If EM slices are inputs, the preprocessing profile must declare A/B fusion and zeroing B for single-segment input, or native dual masks for dual-segment input.

## ConnectomeBench2 V2 metadata lineage and split audit

The `connectome-dataset-audit` binary audits the metadata-only sidecars before training or scoring. It consumes JSONL exported from the three V2 Parquet sidecars, so no image payloads are read and no Python runtime is added to the Rust toolchain.

From the ConnectomeBench2 V2 dataset root, export only the identity, labels, route, mask, modality, and metadata columns with DuckDB:

```sh
duckdb -c ".read /path/to/symthaea/tools/connectomics-intake-audit/export_connectomebench2_metadata.sql"
```

Then run the Rust auditor from the Symthaea checkout:

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome-dataset-audit -- \\
  --input /path/to/connectomebench2_metadata.jsonl \\
  --dataset-revision-commit-sha <resolved-v2-commit-sha> \\
  --cube-edge-nm 50000 \\
  --output connectomebench2-lineage-audit.json \\
  --cohort-manifest-output connectomebench2-modality-cohorts.json
```

The `--dataset-revision-commit-sha` argument is required and must be the exact 40-character commit SHA resolved from the Hugging Face `v2` revision used to download the dataset; do not pass the literal string `v2`. The report records both this dataset commit and the pinned upstream code commit. The SHA is operator-supplied provenance: the offline auditor does not contact Hugging Face or independently prove that the local files came from that commit.

The export SQL reads `metadata/train.parquet`, `metadata/val.parquet`, and `metadata/test.parquet`, and deliberately excludes geometry and EM bytes. The auditor fails closed on malformed/unknown sidecar fields, bad sample-hash composition, duplicate combined IDs, sample-type/strategy mismatches, incorrect `same_neuron` or derived label values, mask-flag drift, task-routing drift, missing source split-group keys, operation-group reuse across splits, and reuse of a physical 50 µm cube across partitions even when its rows originate from different sample banks. It reconstructs the upstream grouping key by archive bank: `operation_id` for edit/adjacent-control rows, `source_operation_id` for junction controls, and the pre/post root plus synapse coordinate for synapse controls. It normalizes the source-side `val` spelling to `validation`.

Cross-split root identity is reported as a warning, not a hard failure. A neuron may span multiple 50 µm cubes; such overlap creates a dependence and possible leakage risk that needs a neuron-grouped sensitivity evaluation, but the overlap alone does not prove that train/test samples are duplicates. Missing spatial coordinates are reported and never assigned fabricated cube IDs; a present but malformed/non-finite coordinate fails closed instead of bypassing the spatial check. A PASS is limited to the checks above; it does not prove biological independence, ground-truth correctness, or model performance.

The report also includes a deterministic `modality_coverage` map for `overall`, task route, sample type, species, split, source archive, sample-type × species × split, and route × species × split strata. Each row-count group reports geometry availability, any EM slot, missing EM, per-view `em_xy`/`em_xz`/`em_yz` counts, rows with at least one non-label-conditioned EM view, rows with the full three-view EM set, `em_best` presence and EM-best-only rows, fused geometry+usable-EM eligibility, and corresponding fractions. `em_best` is counted separately and never contributes to usable EM eligibility; it is not an allowed single-segment model input. Route strata count one row in each declared route, so counts across routes can exceed the overall row count. This is a metadata-sidecar eligibility summary only: it does not resolve payload bytes, certify readable images, establish a probability-sampling design, or show that EM missingness is random. Paired model comparisons must still use the same sample IDs, labels, source revision and exact modality transformations.

The synthetic fixture `docs/neuroscience/examples/CONNECTOMEBENCH2_V2_METADATA_LINEAGE.example.jsonl` has one fake row per sample type. It is not a substitute for running the audit on the exact downloaded V2 revision. The V2 revision commit SHA still must be recorded in the intake manifest; this offline binary does not resolve the Hugging Face branch to a commit or prove that the downloaded bytes came from that branch.

### Content-addressed modality cohorts

The optional `--cohort-manifest-output` writes a deterministic sample-ID manifest **only when the metadata audit status is PASS**. The schema is `docs/neuroscience/CONNECTOME_MODALITY_COHORT_MANIFEST.schema.json`. A synthetic example (with placeholder source revision/digests) is at `docs/neuroscience/examples/CONNECTOME_MODALITY_COHORT_MANIFEST.example.json`; it is not a biological dataset audit output. It contains:
- `geometry_all`;
- `em_legal_any_view` (at least one of `em_xy`, `em_xz`, `em_yz`);
- `em_legal_full_triplet` (all three legal EM views);
- `fused_geometry_legal_em`;
- `paired_geometry_only` and `paired_em_legal_views`, which have exactly identical sorted sample IDs across the broad geometry+EM intersection;
- three additional task-specific paired pairs: `paired_false_split_correction_geometry_only` / `paired_false_split_correction_em_legal_views` (requires dual geometry), `paired_false_merge_identification_geometry_only` / `paired_false_merge_identification_em_legal_views` (requires single geometry), and `paired_split_mask_generation_geometry_only` / `paired_split_mask_generation_em_legal_views` (requires single geometry);
- `geometry_without_legal_em`;
- `em_best_only_excluded`.

Each membership has a sorted unique ID list, count, SHA-256, explicit eligibility rule and `NOT_ESTIMABLE` status for empty cohorts. The cohort-set digest binds the exact dataset revision SHA, source sidecar JSONL SHA-256, policy version, cohort names/rules/counts and membership digests. The route-specific pairs prevent a dual-segment false-split correction cohort from being pooled with single-segment false-merge identification or split-mask-generation cohorts. The geometry and EM members of each task-specific pair have exactly identical sorted IDs. The manifest records eligibility, but does not prove that image payloads exist or are readable. Paired comparisons must still verify the exact task label, A/B mask fusion, preprocessing transforms and actual Parquet bytes in the next qualification tier. The output path must differ from `--output`.

## Frozen prediction scorer

A second binary, `proofreading-score`, evaluates **frozen** predictions and independently adjudicated labels. It is deliberately separate from the runtime model/ranker. It currently consumes JSONL so metrics can be unit-tested without downloading raw EM or decoding Parquet.

Build and test:
```sh
cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin proofreading-score -- \
  --input predictions.jsonl \
  --mapping-reference https://example.org/reviewed-label-map \
  --dataset-revision-commit-sha <40-character-hub-commit-sha> \
  --policy-id specialist-v1 \
  --prediction-profile-sha256 <64-character-config-sha256> \
  --evaluation-partition development \
  --review-time-basis synthetic-proxy \
  --bootstrap-replicates 500 \
  --output score-report.json
```

The input JSONL contains one row per unique source operation:
```json
{"sample_id":"source-op-001","group":"mouse:merge","cluster_id":"mouse-cube-017","label_is_error":true,"p_error":0.93,"review_priority_score":0.94,"review_minutes":2.0,"impact_weight":4.0}
{"sample_id":"source-op-002","group":"fly:split","cluster_id":"fly-cube-204","label_is_error":false,"p_error":0.12,"review_priority_score":0.07,"review_minutes":1.5,"impact_weight":0.0}
```

- `cluster_id` is the declared dependency unit for uncertainty estimation (e.g. source volume/spatial cube/neuron); choose the unit that matches the sampling and intended generalization claim. Multiple rows from one cluster are resampled together. A cluster ID is not proof that clusters are independent. For a domain-shift run, one cluster ID may not span two domain IDs; IDs must be globally unique or domain-qualified. An OOD audit with only in-domain or only held-out rows is `OOD_NOT_ESTIMABLE`, not a scored positive result.
- `label_is_error` is an **evaluation-only** adjudicated label, normalized so `true` means a genuine reconstruction error requiring correction. `mapping_reference` must point to the reviewed task/label contract.
- `p_error` is the model's error probability; the scorer reports balanced accuracy at 0.5, AUROC, tie-aware average precision, Brier score, log loss, and group-level metrics.
- `review_priority_score` is the frozen policy ranking for which candidate an expert should review next. The evaluator measures error yield and impact-weight capture at fixed item and review-minute budgets.
- `impact_weight` is ground-truth evaluator metadata. It must never be a runtime input to the ranker or the candidate model.
- `review_minutes` may be prospectively measured or a synthetic proxy. The required `--review-time-basis` makes that distinction explicit; synthetic proxies cannot support a real-world time-saving claim.
- The schema-version-5 report includes dataset revision commit SHA, pinned upstream code commit, policy ID, frozen profile digest, evaluation partition, input SHA-256, deterministic cluster-bootstrap 95% intervals for balanced accuracy, review-budget yield, and risk–coverage checkpoints. Review-budget yield and risk–coverage are reported overall and per `group`, at the same item/minute budgets and coverage targets. Three matched-budget control policies are also scored on the same frozen rows: `uncertainty_only` (least confident first), `probability_only` (highest predicted error probability first), and `deterministic_random` (SHA-256 keyed order from the frozen input digest, group ID, and sample ID). Control metrics are computed overall and per group, and the random baseline is deterministic for replay.

For every control and budget, the report also estimates the paired difference **submitted priority minus control** using a deterministic cluster bootstrap over the same resampled clusters for both policies. It reports 95% intervals separately for impact-weighted error recall, ordinary error recall, and precision. Item budgets are recalculated at the same requested fraction in each bootstrap replicate; minute budgets use the fixed supplied minute cap. Each metric reports its valid replicate count and is marked not estimable unless at least 80% of requested replicates are valid and at least two dependency clusters exist. Intervals are conditional on the submitted frozen rankings and the deterministic-random seed; they do not account for re-training, policy selection on the lockbox, or incorrect cluster assignments.

Per-group deltas help stop strong pooled scores from hiding a weak species × error-family or a queue policy that fails to beat simple alternatives. Duplicate sample IDs fail closed before scoring; aggregate multiple image views at the source-operation level. Balanced-accuracy intervals and paired review-budget delta intervals are provided; AUROC, average precision, Brier score, and log loss currently have point estimates only. The scorer does not validate Parquet labels or certify cluster independence. It always emits `SCORED_NOT_PROMOTED`; model promotion still requires the locked split protocol and independent grouped statistical analysis.

## Counterfactual query-impact review prioritization

The `connectome-query-impact` binary evaluates proposed weighted-edge mutations independently against one immutable directed weighted graph and a preregistered set of scientific queries. It reports each query's baseline value, counterfactual value, delta, normalized absolute delta, and weighted contribution. The candidate score is

```text
weighted_query_sensitivity =
    sum(query_weight * min(abs(query_delta) / normalization_scale, 1))
    / sum(query_weight)

probability_weighted_query_sensitivity_per_review_minute =
    estimated_error_probability * weighted_query_sensitivity / review_minutes
```

Supported query metrics in schema v1 are directed edge weight, weighted in-degree, weighted out-degree, directed path existence, weighted feed-forward-loop counts, and weighted directed three-cycle counts. Motif counts sum products of the three directed edge weights; directed cycles are divided by three to remove rotation duplicates. Motif queries are limited to frozen subgraphs containing at most 4,096 directed edges. Supported candidate mutations are adding/removing positive edge weight. Duplicate graph edges must be aggregated by the producer. The query bundle's weights and scales are scientific-policy choices and must be frozen before evaluation. Candidate error probabilities must come from an independently calibrated model/policy; this tool does not train or calibrate them.

Run the checked-in synthetic example:

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome-query-impact -- \\
  --input docs/neuroscience/examples/CONNECTOME_QUERY_IMPACT_SYNTHETIC.example.json \\
  --output query-impact-report.json
```

The report includes canonical SHA-256 digests of the baseline and each counterfactual graph. The digest encodes sorted directed edge triples as big-endian source ID, target ID, and IEEE-754 weight bits, making it independent of input edge order. The caller-supplied `source_graph_sha256` remains separately recorded and is not silently substituted for the canonical graph digest. Each candidate's evidence SHA-256 is retained in its result.

A query may optionally include `reference_value`, an evaluation-only oracle result from an independently corrected/validated graph. The report then records baseline and counterfactual absolute query error and signed normalized error reduction (positive means closer to the reference; negative means worse). Oracle values do **not** enter the candidate priority formula; they must be withheld from proposal generation, model input, and queue ranking. When no query reference values exist, the reduction is explicitly not estimable. This is the first metric that can distinguish a large but harmful graph perturbation from a correction that improves a held-out scientific answer.

The input also requires a preregistered `review_budgets_minutes` array. For each budget, the report replays six deterministic policies on the same candidate pool: query-sensitivity × error probability per minute, error probability per minute, uncertainty per minute, local weighted-degree change per minute, query sensitivity alone per minute, and deterministic random. The local-degree baseline sums absolute changes in weighted in-/out-degree across affected nodes; it is a simple structural comparator, not a global centrality method.

For each policy/budget pair, candidates are visited in policy order. A candidate consumes its declared review minutes when selected; its mutation is applied to an in-memory graph only if it is valid on the evolving graph. A conflicting/incompatible candidate is still counted as reviewed but is not applied. The final weighted query error is compared with the starting graph only for queries with evaluation-only oracle values. This sequential replay avoids pretending conflicting candidates are additive.

The replay is a **retrospective policy simulation**, not a model of human acceptance: it assumes every selected candidate whose mutation remains valid is accepted, and it does not model reviewer rejection, deliberation, variable review time, or new evidence acquired during review. Actual expert-minute savings require prospective review records. No annotation file or authoritative graph is changed; every candidate remains `HUMAN_REVIEW_ONLY`. If no positive-weight oracle values exist, query-error reduction is explicitly not estimable.

The synthetic fixture deliberately includes an alternate-path control, candidates that conflict on a shared edge, and oracle values that distinguish beneficial from harmful changes. It is executable negative-control data, not a biological result.

## Post-hoc reviewer outcome audit

The `connectome-review-outcome-audit` binary is intentionally separate from the query-impact ranker. It consumes three files: the exact original proposal JSON, the frozen ranking report emitted by `connectome-query-impact`, and a separate evaluation-only reviewer-outcome file. It verifies the SHA-256 of the proposal bytes against the ranking report, verifies the canonical baseline graph digest, and checks that the outcome file is bound to both exact artifacts. The outcomes file also pins the adjudication protocol by SHA-256.

The outcome file must contain exactly one record for every candidate in the frozen proposal pool:
- `adjudication`: `correct`, `incorrect`, or `indeterminate`, from an independent adjudicator judging whether the proposed graph mutation is actually the correct correction;
- `reviewer_decision`: the recorded expert decision `accept`, `reject`, or `abstain`;
- `actual_review_minutes`: measured reviewer time, not a model estimate.

Run sequence:

```sh
# 1. Freeze proposal inputs and produce the ranking report.
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome-query-impact -- \\
  --input proposals.json --output frozen-ranking.json

# 2. Bind the separate outcome file to the exact frozen artifacts.
sha256sum proposals.json frozen-ranking.json adjudication-protocol.md

# 3. Populate docs/neuroscience/examples/CONNECTOME_REVIEW_OUTCOMES.example.json
#    with those digests and one independently adjudicated/reviewed record per candidate.

# 4. Evaluate the frozen policies using measured review time and human decisions.
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome-review-outcome-audit -- \\
  --proposals proposals.json \
  --ranking-report frozen-ranking.json \\
  --outcomes reviewer-outcomes.json \\
  --output post-hoc-review-evaluation.json
```

For each frozen policy/minute budget, the evaluator selects candidates in the saved ranking order using `actual_review_minutes`. It applies only proposals the reviewer accepted, in sequence on the evolving graph. An accepted but incorrect proposal is deliberately applied so its query harm is counted. An accepted proposal that has become stale/invalid due to an earlier edit is counted as a stale conflict and is not applied. Rejected and abstained correct proposals are counted as missed correction opportunities.

The report exposes:
- reviewer accept/reject/abstain outcomes crossed with independent correct/incorrect/indeterminate adjudication;
- accepted-decision precision and applied-correction precision, each excluding indeterminate adjudications from its denominator;
- candidate-pool correct-proposal review, acceptance, and actual-application recall (these are **not whole-connectome recall**);
- incorrect-proposal acceptance rate, actual review minutes, stale conflicts, and before/after oracle query error.

This is still a retrospective replay, not a randomized prospective causal estimate. All candidates in the frozen pool must be independently adjudicated and have review outcomes/time records; the same pool is then replayed under every frozen policy. Reviewer learning/order effects are not modeled. The evaluator does not modify authoritative graph or annotation files.


For a defensible offline comparison, the frozen candidate pool should be reviewed in a blinded shadow-evaluation process: reviewers should not see which policy ranked a candidate, and an independent adjudication protocol should determine proposal correctness. This incurs annotation effort to create the outcome corpus; that corpus-construction cost must be reported separately and must not be presented as policy review savings. Policy budgets then replay subsets of the fully adjudicated pool using each candidate's measured review time.

**Information-flow boundary:** never provide the reviewer-outcome file to `connectome-query-impact` or candidate generation. The ranker is a separate binary that does not define or parse this outcome schema. The post-hoc evaluator is the only tool that consumes adjudication/reviewer labels. The synthetic outcome template contains zero placeholders for the two artifact hashes; replace them with SHA-256 digests from the exact files before running it.

## Exit codes

- `0`: all declared checks pass.
- `1`: manifest/data integrity gate failed; a JSON report is written when possible.
- `2`: command/input parsing or report-writing error.

The tool deliberately does not auto-correct labels, task names, or file hashes. Fix the source or have the mapping reviewed, update the manifest, then rerun.


## Candidate-generation coverage audit (SCI-CONNECTOMICS-006)

The new `connectome-candidate-coverage-audit` binary addresses a distinct blind spot: ranking metrics only describe the candidate pool that was generated. They cannot identify eligible proofreading errors the generator never proposed.

This evaluator binds the exact proposal bytes, ranking-report bytes, sampling protocol, and seed commitment. It draws a deterministic stratified simple random sample without replacement of primary sampling units (PSUs) from a complete declared search frame. Every selected PSU must attest to exhaustive expert search, including selected units with zero errors. Each independently verified error is mapped to zero or more IDs from the frozen proposal pool; an empty mapping is a candidate-generation miss.

The complete `error_families` vocabulary must be unique and preregistered in the audit input; every sampled error must belong to it. Declared families with no sampled observations remain visible with zero estimated count and undefined recall when the denominator is zero.

The report exposes sampled per-family counts and keeps declared families with no observed sample errors in the report. For non-census strata, it withholds normal-approximation intervals not only for samples below 30 PSUs but also when the observed total or ratio-residual sample variance is zero; a zero sample variance is not evidence that rare errors are absent from unsampled PSUs.

The report estimates:
- finite-frame total, covered, and uncovered error counts using stratified Horvitz–Thompson totals;
- candidate-generation recall as a ratio of covered to total estimated errors;
- impact-weighted candidate-generation recall using predeclared scientific query impact weights;
- missed error count/impact by observed error family;
- separate independent audit-panel person-minutes;
- first-order Taylor-linearized ratio design variance, with normal-approximation intervals withheld unless every non-census stratum has at least 30 sampled PSUs.

Run sequence:

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome-query-impact -- \
  --input proposals.json --output frozen-ranking.json

sha256sum proposals.json frozen-ranking.json \
  docs/neuroscience/CONNECTOME_CANDIDATE_COVERAGE_AUDIT_PROTOCOL_V1.md

# Populate the coverage file from a committed, complete PSU frame, then reveal
# the preregistered seed and include every selected PSU, even if no errors were found.
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-candidate-coverage-audit -- \
  --proposals proposals.json \
  --ranking-report frozen-ranking.json \
  --sampling-protocol docs/neuroscience/CONNECTOME_CANDIDATE_COVERAGE_AUDIT_PROTOCOL_V1.md \
  --spatial-frame spatial-frame.json \
  --spatial-frame-report spatial-frame-report.json \
  --audit independent-volume-search-audit.json \
  --output candidate-coverage-report.json
```

**Scope and information-flow limits:** estimates apply only to the complete declared finite search frame, not arbitrary unsampled tissue or the whole connectome. Exhaustive search and proposal-to-error mappings need independent protocol enforcement. The candidate generator/ranker must never receive sampled locations, seed, error identities, outcomes, or impact weights. This audit is not interchangeable with #9187, which samples the unreviewed tail *inside* the candidate pool. Neither audit establishes causal policy efficiency; that requires prospective randomized/block-randomized queue assignment. No biological sample or performance gain is claimed.


## Query-scoped reliability certificate verifier

The `connectome-query-cert-audit` binary checks certificate cross-field semantics and provenance references. It rejects a certified state with any non-passing gate or non-estimated uncertainty, rejects conditional certificates that hide a failed critical gate, requires every dependency to reference a bound source revision digest, and requires explicit revision-change evidence for invalidated certificates.

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-query-cert-audit -- \
  --certificate docs/neuroscience/examples/CONNECTOME_QUERY_RELIABILITY_CERTIFICATE.example.json
```

This verifies internal consistency only; it does not independently fetch artifacts or prove biological correctness. The checked-in example is synthetic and intentionally conditional.


## Spatial PSU frame audit

The `connectome-spatial-frame-audit` binary verifies exact, disjoint 3D lattice tiling of a declared integer voxel bounding box, checks eligible/excluded dispositions and exports the eligible `population_frame` using the exact digest contract consumed by `connectome-candidate-coverage-audit`.

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-spatial-frame-audit -- \
  --frame docs/neuroscience/examples/CONNECTOME_SPATIAL_FRAME.example.json \
  --eligibility-mask docs/neuroscience/examples/CONNECTOME_SPATIAL_FRAME_ELIGIBILITY_MASK.hex \
  --output spatial-frame-report.json
```

Then pass the exact report bytes to candidate coverage with `--spatial-frame-report spatial-frame-report.json`. The coverage audit rejects any report whose SHA-256, complete lattice digest, eligible population digest, or exact PSU/stratum export does not match the audit input. A frame pass proves tiling only inside the declared box; source-volume and eligibility-mask completeness must be independently verified.

The verifier permits NOT_APPLICABLE only for a non-critical gate with a documented scope note. A critical NOT_APPLICABLE gate blocks full certification. Every uncertainty component must declare assumptions; estimated components need a point estimate, and non-estimable components cannot carry one or an interval.


## Joint correction portfolio evaluator

The `connectome-portfolio-audit` binary performs exact subset optimization for up to 20 interacting graph-edit candidates on small, frozen evaluation graphs. It compares probability-weighted query sensitivity, probability per minute, uncertainty per minute, and deterministic-random rankings (seeded from a reference-free ranking-view digest) against an evaluator-only exact optimum. The ranking function receives a reference-free query view, so held-out query values are not present in its input types. The random baseline is also seeded from the reference-free ranking-view digest, not the full evaluator-input digest. It reports pairwise interactions, regret, and reverse-application-order sensitivity. See `docs/neuroscience/CONNECTOME_CORRECTION_PORTFOLIO_AUDIT_PROTOCOL_V1.md`.

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome-portfolio-audit -- \
  --input docs/neuroscience/examples/CONNECTOME_CORRECTION_PORTFOLIO.example.json \
  --output correction-portfolio-report.json
```

The exact optimum is bounded to the submitted candidate pool, budget, graph, query bundle, and canonical candidate-ID application order. Reference query values are evaluation-only and are never consumed by the candidate ranking policies.


## Hidden co-linear merge boundary baseline

The experimental `connectome-colinear-merge-audit` binary is a synthetic ordered-centreline change-point baseline. It compares a no-proposal control, radius-only and multifeature piecewise-linear scores, emits review-only boundary hypotheses, and optionally evaluates them against a separate truth file bound to the exact input SHA-256:

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome-colinear-merge-audit -- \
  --input docs/neuroscience/examples/CONNECTOME_COLINEAR_PROFILE_INPUT.example.json \
  --truth docs/neuroscience/examples/CONNECTOME_COLINEAR_BOUNDARY_TRUTH.example.json \
  --tolerance-nm 500 --output colinear-merge-report.json
```

The input contains no labels; the evaluator truth is parsed only after proposal generation. This is a 1D feature-profile baseline, not a detector trained on or operating directly on EM imagery. The default is the V1 sample-index baseline. To enable the optional physical-arclength mode, add both `resample_interval_nm` and `half_window_nm` to the input's `detector` object. For example, a 25 nm grid with a 400 nm half-window uses 16 grid samples per half-window. The report retains V1 outputs and adds separate radius-only/multifeature results for two physical resampling policies: piecewise-linear interpolation and nearest-source-observation (ties select the earlier observation). Linear interpolation can smear abrupt transitions; nearest-observation preserves discontinuities but localizes them only to the source sampling resolution. The report records each policy and per-track uncovered terminal tail. Both settings must be held constant across source sampling-density comparisons; allow localization tolerance to include the source sampling interval; the score threshold is still heuristic and not calibrated. See `docs/neuroscience/CONNECTOME_COLINEAR_MERGE_AUDIT_PROTOCOL_V1.md` and schema `docs/neuroscience/CONNECTOME_COLINEAR_PROFILE_INPUT.schema.json`.


## Query-scoped certificate dependency DAG

The `connectome-query-cert-audit` verifier now requires `query.dependency_roots` and explicit `depends_on` edges for every dependency record. It rejects unresolved references, duplicate identities/edges, reachable cycles, and dependency records outside every declared root's transitive closure. Its audit report emits root/node/edge counts and a canonical dependency-graph SHA-256 that remains stable under JSON node/edge reordering. See `docs/neuroscience/CONNECTOME_QUERY_RELIABILITY_CERTIFICATE_V1.md`. Passing this check validates the declared graph, not the scientific completeness of the query's root manifest.


## Acquisition–proofreading co-design

Research track [#9282](https://github.com/Luminous-Dynamics/symthaea/issues/9282) investigates choosing between targeted re-imaging, specialist re-tracing, and human review by expected query-error reduction per total cost. It includes independent probability sampling of untouched regions to protect against active-selection bias. No microscope-control implementation is included.


## Finite reconstruction query robustness audit

The `connectome-query-ensemble-audit` binary evaluates the same scientific graph query across a finite, explicit set of provenance-bound reconstruction scenarios. V1 reports per-query min/max values, decision stability, per-scenario outcomes, canonical graph/scenario-set digests, and shared evidence-lineage groups.

It is intentionally an **unweighted finite scenario evaluator**. It does not produce posterior probabilities, confidence intervals, or evidence that the submitted scenarios exhaust all biologically plausible reconstructions.

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-query-ensemble-audit -- \
  --input docs/neuroscience/examples/CONNECTOME_QUERY_ENSEMBLE_AUDIT.example.json \
  --output query-ensemble-report.json

cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-query-ensemble-audit
```

See `docs/neuroscience/CONNECTOME_QUERY_ENSEMBLE_AUDIT.schema.json` and `CONNECTOME_QUERY_ENSEMBLE_AUDIT_PROTOCOL_V1.md` for the exact input and scope contract. The checked-in fixture contains a query that changes across plausible graphs and two decision-stable controls. All results are synthetic.

## Domain-shift calibration and OOD referral audit

The frozen proofreading-score evaluator accepts an optional, complete domain tuple on every prediction row: domain_id, shift_axis, and is_ood_domain. Keep task/error-family group separate from the domain identity. A single run evaluates one declared shift axis; run separate inputs for species, region, source volume/archive, acquisition protocol, segmentation lineage, modality availability, or morphology regime.

Optional ood_score values are the frozen novelty model output, independent of p_error. When present they must be present on every row and the run must supply --ood-detector-id plus --ood-detector-profile-sha256; both are recorded in the report. Detector identity/provenance without a complete score column is rejected. Any domain-tagged run must also supply --domain-split-manifest PATH. The scorer hashes and parses that file and verifies dataset revision, exact evaluation-partition membership, sample/cluster/domain IDs, shift axis, held-out classification, seen-domain training/calibration membership and absence of held-out train/calibration leakage. On confirmatory/lockbox evaluation, --ood-threshold-selection-input accepts a separate development predictions JSONL. The scorer validates it against the same split manifest and detector profile, then selects the threshold that maximizes development held-out-domain recall under a fixed 5% false-OOD-rate ceiling. It rejects use of development as both selection and evaluation. The report includes per-domain classification/reliability metrics, ten fixed-width p_error calibration bins, a 15-bin adaptive confidence-versus-correctness ECE/MCE, a 12-bin equal-mass confidence reliability curve (ties are not split), with per-bin cluster-bootstrap intervals for observed accuracy and mean confidence, high-confidence false-accept rates at 0.90 and 0.95, low-confidence missed-error mass, and novelty AUROC/AUPRC. A development-derived threshold adds confirmatory/lockbox false-OOD rate and held-out-domain recall. If no threshold satisfies the 5% development constraint, no operating threshold is applied.

    cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin proofreading-score -- --input docs/neuroscience/examples/CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT.example.jsonl --mapping-reference docs/neuroscience/CONNECTOMEBENCH2_V2_INTAKE_MANIFEST.schema.json --dataset-revision-commit-sha 1111111111111111111111111111111111111111 --policy-id domain-shift-fixture-v1 --prediction-profile-sha256 EXACT_64_CHARACTER_PROFILE_DIGEST --evaluation-partition development --review-time-basis synthetic-proxy --require-domain-shift-audit --domain-split-manifest docs/neuroscience/examples/CONNECTOME_PROOFREADING_DOMAIN_SPLIT.example.json --ood-detector-id mahalanobis-diagonal-v1 --ood-detector-profile-sha256 EXACT_64_CHARACTER_OOD_DETECTOR_DIGEST --output domain-shift-development-report.json

See docs/neuroscience/CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT_PROTOCOL_V1.md and the input schema. The domain split and prediction fixtures are synthetic; the all-ones dataset revision is a placeholder identity, not a real ConnectomeBench2 revision. For real evaluation, replace it with the exact V2 commit in both the CLI and manifest. The fixture is synthetic. Fixed-bin and adaptive-confidence ECE/MCE, reliability-curve ECE, Brier, and OOD AUROC/AUPRC now include deterministic cluster-bootstrap 95% percentile intervals; resampling occurs at the declared cluster_id unit and invalid one-class OOD replicates are excluded. Intervals are withheld below two clusters or below 80% valid replicates. This does not prove the caller's cluster_id is a valid independent sampling unit, and two clusters are not adequate promotion support. Real-data lockbox evaluation and repeated-seed interval-coverage simulations remain mandatory.
