# TRANSPORT-BENCH-EXT-001B — benchmark coverage extension

Issue: #6266  
Parent architecture: #6255  
Parent frozen registry: #6257 / PR #6262  
Snapshot date: 2026-09-27

## Purpose

Extend the frozen external-transport benchmark registry to mobile robotics, off-road/heavy terrestrial systems, rotorcraft/aerial estimation, planetary surface mobility, fixed-corridor transport, and novel modes without mutating V1.

This generation is an additive registry extension. It does not make V1 retroactively complete and does not claim benchmark performance.

## Parent binding

- V1 source head: `9e00c54f5f5c319df2c25d4af4a2381a21ffc4e9`
- V1 tree: `ff87b6915904e7d4a82862c0e4e101dc0d51e099`
- V1 schema: `transport-bench-ext-001a-registry-v1`
- V1 compact JSON SHA-256: `87cbe6e38b84aaeecfa188997774b64aec209b314f158b2f7867ee9f34b29665`

## New theorem: benchmark count is not evidence independence

```text
benchmark A
+ benchmark B
+ benchmark C
!= three independent witnesses
```

when they share materially relevant roots such as:

- source dataset / annotation lineage;
- simulator or scenario generator;
- sensor assumptions;
- evaluator implementation or metric family;
- model/pretraining exposure;
- community/leaderboard tuning feedback.

Every V2 entry therefore carries explicit `common_mode_roots`.

## V2 additions

### B24 — BARN Challenge 2026

Use as a low-speed mobile-navigation benchmark with a hidden simulation evaluation and standardized physical finals.

Planning-time facts independently checked from the official 2026 challenge page:
- 300 public pre-generated BARN environments;
- 50 hidden evaluation environments;
- standardized Clearpath Jackal + 2D LiDAR task;
- simulation qualifier and physical finals;
- dynamic-obstacle extension in 2026.

The physical final is meaningful external physical evidence for the exact standardized challenge, but:

```text
BARN physical-final success
!= warehouse logistics capability
!= payload handling
!= fleet reliability
!= deployment authority
```

Rights remain `UnknownNeedsReview` until the exact challenge/data/code terms are admitted.

### B25 — TartanGround

Current project page reports 63 environments, 878 trajectories and about 1.44 million samples, with RGB/depth/flow/LiDAR/IMU/semantic/occupancy modalities.

Dataset: CC BY 4.0. Toolkit: MIT.

Use for perception, occupancy, SLAM and navigation subproblems only.

### B26 — RELLIS-3D

Admit as off-road perception evidence only after exact version and terms review.

```text
semantic segmentation
!= traversability
!= traction/dynamics
!= heavy-equipment autonomy
```

### B27 — EuRoC MAV

Bind DOI `10.3929/ethz-b-000690084`.

The official ETH dataset provides stereo images, synchronized IMU and motion/structure ground truth, with documented synchronization and dynamic-motion ground-truth limitations.

ETH Research Collection currently labels the dataset `In Copyright - Non-Commercial Use Permitted`.

```text
VIO / SLAM score
!= flight-control competence
!= airworthiness
```

### B28 — TartanAir V2

Current project page identifies the V2 dataset as CC BY 4.0 and the toolkit as MIT.

Use only for synthetic perception/SLAM/navigation evidence.

### B29 — JPL Mars Yard III

Treat as an `ExternalPhysicalTestPrecedent`, not a public leaderboard.

JPL describes a >2000 m² planetary-rover proving ground with variable slopes and representative terrain used for autonomy, wheeled mobility, slope climbing and related testing.

Matching a JPL test pattern does not create JPL/NASA qualification.

### B30 — ERNEST 2026 field campaign

Treat as an external field-test precedent.

NASA/JPL reported a March 2026 campaign in which ERNEST traveled about 16 miles over 37 hours of drive time across rugged terrain and varied lighting after high-fidelity virtual training.

The public report is not a standardized downloadable benchmark or Symthaea capability evidence.

### B31 — fixed-corridor transport

Retain `BenchmarkGap`.

Use exact analytic fixtures, component/control standards, and benign physical rigs until a defensible cross-system external benchmark exists.

### B32 — heavy mobile equipment

Retain `BenchmarkGap` for end-to-end autonomous construction/mining/agricultural mobile-equipment competence.

Off-road perception datasets may support bounded subclaims only.

### B33 — novel transport modes

Airships, hovercraft/amphibious systems, maglev and novel guideways require mode-specific audits. Lack of a public benchmark is not evidence against engineering value.

## Source references used for this planning freeze

- BARN Challenge 2026:
  `https://people.cs.gmu.edu/~xiao/Research/BARN_Challenge/BARN_Challenge26.html`
- TartanGround:
  `https://tartanair.org/tartanground/`
- TartanAir V2:
  `https://tartanair.org/`
- EuRoC MAV:
  `https://projects.asl.ethz.ch/datasets/euroc-mav/`
  and DOI `10.3929/ethz-b-000690084`
- JPL Mars Yard III:
  `https://www-robotics.jpl.nasa.gov/how-we-do-it/facilities/marsyard-iii/`
- ERNEST 2026:
  `https://www.jpl.nasa.gov/news/nasa-testing-advanced-capabilities-for-moon-mars-rovers/`

RELLIS-3D rights/version remain deliberately unresolved in this generation pending exact source review.

## Claim ceiling

A qualified V2 extension can establish only a faithful additive registry of benchmark/reference identities, conservative rights states, common-mode lineage, applicability, gaps, and claim ceilings.

It establishes no benchmark performance, benchmark independence by count, legal permission beyond the exact admitted terms, physical transport capability, safety, certification, mission readiness, commercial readiness, or operation authority.
