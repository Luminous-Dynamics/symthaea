# TRANSPORT-BENCH-EXT-001A — External transport benchmark registry v1

Issue: #6257  
Parent: #6255 / #6244  
Planning snapshot: 2026-09-27  
Schema: `transport-bench-ext-001a-registry-v1`  
Canonical compact JSON SHA-256: `87cbe6e38b84aaeecfa188997774b64aec209b314f158b2f7867ee9f34b29665`

## Purpose

Freeze a bounded, rights-aware external-benchmark registry before transport adapters execute or download third-party data.

This subject is architecture/evidence metadata only. It is not a legal opinion, benchmark endorsement, benchmark completeness claim, or permission to use any third-party material.

## Core theorem

```text
external benchmark exists
!= rights admitted for this consumer
!= benchmark result obtained
!= physical capability
!= field safety
!= regulatory approval
!= service reliability
!= human benefit
```

and:

```text
benchmark score A
+ benchmark score B
!= one universal transport score
```

unless a separate comparison theorem explicitly establishes compatible task, metric, population/domain, version, and aggregation semantics.

## Exact-use rule

Before an external dataset, evaluator, standard corpus, or simulator is consumed, the adapter must bind an exact benchmark generation and separately admit the intended use.

Unknown or unclear terms remain `UnknownNeedsReview`. No unknown rights state can be interpreted as permissive.

An open-source adapter does not make third-party benchmark data or derived models open source.

## Version rule

Friendly benchmark names are not identities.

Changing any claim-critical element creates a new benchmark subject generation, including as applicable:

- release/version/date;
- task/track;
- train/validation/test split;
- evaluator/toolkit;
- metric or aggregation;
- allowed external-data policy;
- scenario/ODD applicability;
- license/terms;
- leaderboard/submission policy.

No floating `latest` identity may support a qualified result.

## Evidence classes

The V1 registry admits only the closed-world classes frozen in the JSON corpus. These classes describe the proposition being evaluated; they are not a trust ranking.

Examples:

```text
OfflinePerceptionBenchmark
!= ClosedLoopSimulationBenchmark
!= SyntheticToHilDomainGapBenchmark
!= OperationalOutcomeComparison
```

## Exposure and leakage

Benchmark exposure remains explicit:

```text
public before subject freeze
development split used
validation split used
test labels available
leaderboard feedback consumed
pretraining exposure unknown
post-freeze held-out
```

A public benchmark may remain useful comparative evidence while no longer supporting pristine blind-generalization language.

## V1 registry

The canonical JSON contains exactly B01–B23 in order.

### Road / automated driving

B01–B13 cover Waymo perception/motion/E2E, Waymax, nuScenes, Argoverse 2, nuPlan, CARLA Leaderboard 2.1, CommonRoad, ASAM OpenSCENARIO, Safety Pool, MLPerf Automotive v0.5, and Waymo Safety Impact as a separate operational-outcome reference.

Important frozen boundary:

```text
Waymo Open Dataset benchmark
!= Waymo Driver operational performance
```

Waymo's dataset is an unlabeled mixture of manual and autonomous operation and its current license is non-commercial. Waymax is also governed by a non-commercial license and explicitly restricts real-world vehicle development/validation use.

CARLA 2.1 is simulation evidence. MLPerf Automotive v0.5 is compute latency/performance evidence.

### Rail

B14 records RAIL-BENCH as a railway perception/odometry research benchmark only.

```text
RAIL-BENCH PASS
!= automated train operation
```

### Fixed-wing

B15 records IDF-DS as a public fixed-wing telemetry dataset/reference subject. Any Symthaea benchmark task built on it requires its own frozen train/validation/test and evaluator contract.

### Maritime

B16/B17 preserve ISO/AWI 25927 and 25930 as under-development standards work items, not published standards or leaderboards.

B18 deliberately leaves the exact COLREG/AIS dataset deferred until provenance, task semantics, terms, and evaluator are selected.

B19 records an explicit V1 benchmark gap for broad end-to-end underwater autonomy.

### Space / orbital

B20/B21 record SPEED/SPEED+ as bounded spacecraft-pose and synthetic-to-HIL domain-gap benchmarks. B22 records NASA/JPL RPOD facilities/processes as external physical-test precedents rather than a downloadable universal leaderboard.

### Launch / ascent

B23 records an explicit V1 benchmark gap. Internal analytic/reference fixtures and separately admitted public mission data are preferable to inventing a weak universal external benchmark.

## Anti-Goodhart rule

A benchmark may reveal a useful weakness or comparative advantage but cannot become the engineering objective by itself.

Required downstream behavior:

```text
benchmark improvement
-> preserve exact task-specific result
-> test cross-benchmark transfer
-> test internal hostile cases
-> later compare with independent physical observations where applicable
```

Do not tune a transport system to one leaderboard and then describe the result as broad transport competence.

## Claim ceiling

A qualified V1 source establishes only a frozen representation of benchmark identities, task classes, planning-time version facts, conservative rights/use states, adapter dispositions, and claim ceilings.

It establishes no benchmark performance, no legal permission, no physical capability, no transport safety, no regulatory compliance, no field reliability, no commercial readiness, and no execution authority.
