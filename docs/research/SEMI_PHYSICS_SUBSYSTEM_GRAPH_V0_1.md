# SEMI-PHYSICS-SUBSYSTEM-GRAPH-001 — process subsystem graph manifest v0.1

**Issue:** #14029  
**Implementation:** `symthaea-semiconductor::benchmark::process_subsystem_graph`  
**Status:** Rust manifest validator with 13 authored unit tests; exact-head validation pending.

## Research motivation

Lin et al., *Physics-structured graph neural networks for interpretable fault detection in semiconductor manufacturing processes*, Digital Chemical Engineering (September 2026), DOI 10.1016/j.dche.2026.100337, organize sensor channels into physically meaningful process subsystems, combine them with categorical context, and evaluate a graph-attention classifier. The study compares against optimized GBDT/LightGBM-style baselines and also evaluates TabM and TabR. Its production data cover 18 months, with the latest 20% of each eligible tool chronology held out and a minimum 8,000 observations per tool. The paper reports that TabM/TabR are useful complementary models but do not consistently replace optimized LightGBM under severe class imbalance.

This benchmark should reproduce the paper's process-grouping and split before exploring new graph learning. The article's attention maps are model attribution, not causal-effect evidence.

Source: https://www.sciencedirect.com/science/article/pii/S2772508126000505

## Manifest contract

The v0.1 manifest keeps four entity classes distinct:

1. **Subsystem nodes:** subsystem identity, process stage, subsystem kind, source artifact digest and engineering evidence digest.
2. **Sensor channels:** sensor identity, unit, clock domain, feature-schema digest and source artifact digest.
3. **Sensor memberships:** explicit sensor-to-subsystem mapping, transformation digest and mapping evidence digest.
4. **Context dimensions:** typed categorical context (tool, chamber, recipe revision, product family, lot, material, process step, facility or other), vocabulary digest and source artifact digest.

Inter-subsystem relations are explicitly typed:
- material/process flow;
- control dependency;
- measurement dependency;
- thermal coupling;
- electrical coupling.

Directed material/process, control and measurement dependencies must be acyclic. Thermal and electrical coupling relations are symmetric and may form cycles. The validator rejects a repeated undirected coupling pair regardless of endpoint order; its relation digest is endpoint-order invariant for symmetric relations.

There is no untyped catch-all relation. Each edge has a relation ID and engineering evidence digest. The graph does not infer missing relations or physical causality.

## Fail-closed validation

The validator checks:
- versioned schemas and non-empty identifiers;
- lowercase SHA-256 structure and graph/relation digest replay;
- unique subsystem, sensor, membership, relation and context IDs;
- valid sensor and subsystem membership endpoints;
- unique sensor/subsystem membership pairs;
- graph/sensor feature-schema digest equality;
- unique context feature IDs, kept distinct from sensor channel IDs;
- relation endpoint existence and typed relation semantics;
- duplicate relation pair/kind rejection;
- cycle-free directed dependency subgraph;
- canonical graph digest independent of collection ordering.

If the owner-declared graph inventory is incomplete, report status is `Unknown`, not `Complete`. Structural validity does not turn an incomplete inventory into a complete model of a fab.

## Why this is separate from a causal graph

The manifest records engineering-declared grouping and relationships that a graph model can consume. It is not:
- a learned causal DAG;
- a proof that a thermal/control/material relation caused an observed quality change;
- a source of intervention effects;
- an authorization to change recipe, setpoint or tool state.

Use #13673/#13616 for falsifiable causal hypotheses, negative controls and intervention design. Use #13883 for sequential multi-signal shift diagnosis. This graph manifest supplies versioned process structure and feature lineage to those systems; it does not replace their identification tests.

## Reproduction benchmark

Before claiming an improvement:
- reproduce the paper's 18-month window, tool threshold (>8,000 observations), label semantics, sensor grouping and per-tool latest-20% temporal split;
- freeze per-tool inclusion and class prevalence;
- fit preprocessing/feature selection only on the training chronology;
- compare optimized LightGBM, FNN, TabM, TabR/retrieval and the physics-structured GNN;
- use identical sensor inputs, splits, missing-data policy and label budget;
- compare against flat MLP/Transformer, subsystem aggregation only, edge-deletion and edge-permutation controls;
- add leave-tool-out, leave-process-family-out, new-recipe and low-label onboarding as separate transfer tracks.

Report macro-F1, balanced accuracy, MCC, PR-AUC, ROC-AUC, recall at fixed FPR, false alarms per 1,000 runs, per-tool and worst-tool performance, calibration, selective risk/coverage, inference latency and memory. Under severe class imbalance, accuracy is not the primary metric.

A graph win requires matched-parameter ablations showing the declared topology contributes beyond subsystem pooling and the encoder alone. Attention weights remain model attribution.

## Evidence composition

The intended chain is:
`qualified sensor/label lineage -> frozen feature schema -> subsystem graph version -> model snapshot -> prediction receipt`

A sensor reassignment, unit change, clock remap, transformation change or graph-topology change produces a new graph/schema digest and requires downstream snapshot requalification (#13461). Only numeric-bound K1-qualified labels (#13467) can enter training/online updates. Model family dependencies remain explicit through #13465.

## Proposed targets (not achieved)

- Faithfully reproduce the article's temporal split and baseline metrics before improving them.
- Aim for 3–5 percentage-point macro-F1 or PR-AUC gain over optimized LightGBM on a preregistered process family, while holding worst-tool recall and false-alarm burden within declared limits.
- 100% rejection of malformed/dangling/duplicate graph relations, directed cycles and feature-schema mismatch.
- 0 label promotion from UNKNOWN/BLOCKED/stale/unavailable metrology evidence.
- No accepted model gain from random wafer splits or training/test leakage.
- Report uncertainty at both wafer and independent tool/process-family levels.

The performance targets are research hypotheses only; the paper's production dataset is not currently in this repository and no local reproduction is claimed.

## Claim ceiling

Synthetic manifest tests verify software contracts, not the real fab topology. A valid graph manifest does not establish causal structure, process safety, production yield gains, cross-fab generalization or recipe authority.
