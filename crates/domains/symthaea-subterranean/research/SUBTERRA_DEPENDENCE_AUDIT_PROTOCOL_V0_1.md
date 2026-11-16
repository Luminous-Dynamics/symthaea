# Subterra dependence-aware split protocol v0.1

**Date:** 2026-11-16  
**Status:** Research protocol and reference auditor authored; not executed  
**Purpose:** prevent hierarchical, repeated, or common-cause observations from leaking across Training, Calibration and Evaluation.

## 1. Why row-level splitting fails

Subterranean science datasets are naturally hierarchical: site, campaign, instrument calibration session, environmental episode, acquisition, processing run, derived observation and adjudication campaign. A mesh, point cloud, feature table and image may all descend from one sensor acquisition. Multiple sites may also share a calibration session, storm/dust episode, processing run or reference-label panel.

Treating these rows as independent can make held-out evaluation look much stronger than deployment generalization. A 15 October 2026 geoscience short communication documents leakage from nested geological observations and reports F1 inflation exceeding 20 percentage points in its case studies (Scharf, Daggitt & Kirkland, *Earth and Planetary Science Letters*, DOI 10.1016/j.epsl.2026.120250). A 3 November 2026 review of small experimental datasets likewise emphasizes grouped/blocked validation for measurements nested within specimens, tools and batches (DOI 10.3390/info17111095).

These results motivate strict grouping; they do not establish Subterra's performance.

## 2. Grouping contract

The reference auditor receives one row per raw/derived evidence record with:

- stable case ID;
- physical site ID;
- one or more physical acquisition-lineage IDs;
- fixed split assignment;
- zero or more declared shared-cause groups.

The auditor automatically joins rows that share:
- a site ID;
- a physical acquisition-lineage ID.

It additionally joins rows sharing one or more declared groups:
- instrument calibration session;
- environmental episode;
- processing lineage/run;
- reference-label/adjudication campaign;
- another explicitly named common-cause episode.

The grouping graph uses transitive closure. If A and B share an environmental episode, and B and C share a processing run, A/B/C form one dependency component and must remain in a single split.

Multiple derived rows from one or more physical acquisitions are allowed and retained. Every contributing acquisition ID joins the derived record to its source components, so fused products cannot hide shared physical inputs. This is more expressive than rejecting every repeated acquisition ID, while still preventing derived products from masquerading as independent acquisitions.

## 3. Split semantics

The auditor rejects:
- one site assigned to multiple splits;
- one acquisition lineage assigned to multiple splits;
- one declared common-cause group assigned to multiple splits;
- duplicate case identities;
- blank identity/group fields;
- duplicate shared-cause membership within a case.

The component is a *split isolation unit*, not an inferred statistical model. The algorithm makes no numeric correlation claim. It does not prove components are independent: undeclared confounders, shared instrument drift not represented as an episode, spatial autocorrelation and selection bias remain possible.

Use episode/campaign IDs, not global instrument or software identifiers, unless the experiment specifically intends leave-one-device-out or leave-one-version-out generalization. Record both descriptive metadata and the grouping decision separately.

## 4. Relationship to likelihood calibration

The current `subterra-likelihood-calibration` v0.1 reference deliberately accepts one case per site/action and a unique acquisition-lineage ID. The dependence auditor is a preflight over richer raw evidence manifests; it does not mutate or auto-convert those records to calibration cases. An integration must preserve source IDs, aggregate derived products by their true physical acquisition, and ensure every dependency component is assigned wholly to one split before the calibration estimator is called.

For a new-site claim, hold out entire sites. For device transfer, separately hold out instrument identities or calibration episodes. For temporal transfer, separately hold out environmental/time episodes. Do not indiscriminately join a global device model or software version into all splits: that would answer a different generalization question and could collapse the entire dataset into one component.

## 5. Required benchmarks

| Benchmark | Measurement | Proposed acceptance |
|---|---|---|
| HierarchyLeakageBench | Case, site, acquisition and declared-cause overlap between splits | Zero dependency-component overlap |
| DerivedArtifactInflationBench | Number of raw/derived rows versus unique acquisitions/components | No reported sample count without all three denominators |
| TransitiveCauseBench | Shared-cause chain A↔B↔C | One split component; no chain may cross splits |
| CalibrationEpisodeHoldoutBench | Leave-one-calibration-episode-out | Report performance and coverage separately |
| EnvironmentEpisodeHoldoutBench | Leave-one-environmental-episode-out | Report performance degradation and abstention |
| DeviceTransferBench | Leave-one-device or calibration-family-out | Separate transfer result; never mix with random row split |
| LabelCampaignHoldoutBench | Hold out adjudication campaign | Quantify label-source shift independently |
| GroupedUncertaintyBench | Site/component-clustered intervals | Report cluster count and interval method; no iid-row interval claim |

Proposed target: zero detected shared-group leakage in the frozen manifest. This is a finite-manifest gate, not proof that all confounders were declared.

## 6. Comparison plan

Publish both:
1. ordinary row-random split as a deliberately leakage-prone diagnostic only;
2. grouped split by site/acquisition/common-cause component as the qualifying result.

Add leave-one-site-out, leave-one-environmental-episode-out, leave-one-calibration-session-out, leave-one-device-out and leave-one-processing-lineage-out when there are enough independent groups. These answer different deployment questions and must not be collapsed into one score.

For each split, publish:
- row count;
- unique site count;
- unique acquisition count;
- declared shared-cause membership count;
- dependency-component count and size distribution;
- groups held out;
- Brier/NLL, score coverage and confidence intervals;
- baseline, configuration, dataset manifest digest, code revision and failure cases.

A site-cluster bootstrap is only meaningful when there are enough independent sites. Where sites share an environmental or calibration episode, bootstrap the larger dependency component or use a hierarchical method. Do not treat a thousand derived frames from three caves as a thousand independent replicates.

## 7. Paired clustered uncertainty

The crate includes a deterministic paired percentile bootstrap for one aggregate baseline/candidate score pair per dependency component. Components receive equal weight; raw rows inside a component do not multiply their influence. The report records the method ID, component count, replicate count, seed, confidence level, mean scores, mean paired difference and percentile interval.

The implementation requires at least 20 dependency components and 200 resamples, rejects duplicate component IDs and non-finite values, sorts IDs before resampling, and uses a fixed-seed SplitMix64 generator for reproducibility. For loss metrics, negative candidate-minus-baseline values favor the candidate.

This is a small-sample descriptive mechanism, not a universal confidence guarantee. Prefer 20–30 or more genuinely distinct components where feasible, and publish the component count. It does not repair selection bias, nonstationarity, incorrect grouping or undeclared confounders. It must not be described as an iid-row bootstrap.

## 8. Symtropy manifest consumer

The auditor now consumes the Symtropy v0.1 observation-lineage JSON sidecar through strict typed deserialization. Its semantic validator rejects duplicate record IDs, duplicate acquisition/group memberships, dangling source IDs, source-graph cycles, raw records with parent sources, derived records without parents, and derived products whose acquisition set is not exactly the union of the parent acquisition sets.

The evaluator supplies a separate `record_id → split` map. The converter requires exactly one split assignment for every record and no unknown assignment keys, then turns source records into the same LineageDataset types used by the typed API. This keeps physical evidence producers from asserting their own evaluation partitions.

A six-record fixture is mirrored from Symtropy commit `2ca4a4897480ce3849be72cb2bdca975390471d4` (fixture blob `348f4e8e3f4e817f033ea6cc89d43bb24a07c2f3`; schema blob `06c7dcb36975fbedc4810f121e84a670768dc5e0`). The fixture demonstrates two raw acquisitions fused into one derived product, plus a shared calibration/environment episode across sites. The tests verify the expected 5-record common-cause component and a separate evaluation singleton.

The consumer now cross-checks the manifest against schema-conforming site-evidence JSON projections. Every manifest record must resolve to one observation under the same site ID; observation IDs must be globally unique; every site observation must be represented exactly once. Negative controls cover missing records, duplicate site/observation IDs, cross-site binding, unsupported schema version, and unmapped site observations. This does not authenticate provenance references or prove the declared common-cause groups are complete.

## 9. Implementation status and limits

Standalone package:
`subterra-dependence-audit/`

- Deterministic union-find grouping by site, acquisition and declared shared causes.
- Transitive dependency components.
- Fail-closed split-crossing checks.
- Exact component/group counts and maximum component size.
- 47 authored tests, including 10 paired cluster-bootstrap cases, 3 multi-acquisition grouping cases, 11 strict lineage-manifest consumer cases and 8 site-observation cross-reference cases.

The package is AUTHORED, not compiled or executed in the current environment. It does not estimate correlations or effective sample size, detect undeclared causes, authenticate IDs, or integrate with the likelihood fitter. Those are follow-on work, not implied capabilities.
