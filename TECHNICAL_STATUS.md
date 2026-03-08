# Symthaea Technical Status

Honest per-capability assessment of Symthaea's current state as of 2026-03-04.

## Status Levels

| Level | Meaning |
|-------|---------|
| **REAL** | Working code with tests, actively used in the cognitive loop |
| **STRUCTURAL** | Code exists, compiles, has some tests, but not fully wired or exercised end-to-end |
| **STUB** | Interface exists but minimal/placeholder implementation |
| **PLANNED** | Described in docs but no meaningful code |

## Confidence Levels

| Level | Meaning |
|-------|---------|
| **HIGH** | 10+ tests, benchmarked or soak-tested |
| **MEDIUM** | Some tests, compiles clean |
| **LOW** | Compiles but limited testing |

---

## Assessment

| # | Capability | Status | Confidence | Evidence | Notes |
|---|-----------|--------|------------|----------|-------|
| 1 | **Core Pipeline: HDC Encoding** | REAL | HIGH | 29 tests in `hdc_ltc_unified.rs`, 23 in validation, 8 in `text_encoder.rs`; 2,403 LOC unified neuron; benchmarked at 97ns/warm-word, 379us/sentence | 16,384D hypervectors, O(1) closed-form temporal jumps. Used every cycle. Profiled via `cycle_profiler` example. |
| 1 | **Core Pipeline: CfC Temporal Prediction** | REAL | HIGH | 13 tests in `prediction.rs`, 29 in `hdc_ltc_unified.rs`; release cycle 4.3ms (234Hz) | LTC/CfC step is ~2% of cycle time. Cincinnati-form neurons with closed-form solution. |
| 1 | **Core Pipeline: Predictive Coding Loop** | REAL | HIGH | 6 tests in `predictive_encoder.rs`, 54 in `cycle.rs`, 178 in integration tests; 977 LOC `cycle.rs` | 8-phase pipeline (perception to output). Rayon-parallel post-processing. Soak-tested via `soak_moral_anomaly` example. |
| 2 | **Consciousness: IIT/Phi** | REAL | HIGH | 20 in `integrated_information.rs`, 132 in `tiered_phi/tests.rs`, 12+9+7 in tiered core/advanced/streaming | Tiered Phi (proxy for exact Phi, which is intractable >12 nodes). Spearman rho=0.50 across 15 topologies. 4D hypercube Phi champion validated. |
| 2 | **Consciousness: Global Workspace Theory** | REAL | HIGH | 14 tests in `global_workspace.rs`, 7 in `gwt_integration.rs`, 48 in `narrative_gwt_integration.rs`; handler dispatch verified in integration + soak tests | GWT broadcasting wired into cognitive loop via `UnifiedGlobalWorkspace`. `register_handler()` API + dispatch to registered handlers. Urgency-gated submission (Critical=always, Normal=every 2nd, Cruise=every 4th). Coalition tracking, ignition detection, attentional blink. |
| 2 | **Consciousness: Equation V2** | REAL | HIGH | 43 tests in consciousness engine tests, 18+25 in equation crate (`engine.rs`+`tests.rs`), 65 calibration tests | Master equation combines Phi, GWT workspace, HOT, substrate feasibility. Calibration system with normative z-scores. |
| 3 | **Active Inference / FEP** | REAL | MEDIUM | 10 in `hierarchical_free_energy.rs`, 14 in `predictive_processing.rs`, 4 in `vocal_tract_fep.rs` (426 LOC); FEP manager wired in `cycle_phase_dynamics.rs` | FEP inference step runs every cycle. Outputs: `fep_action_idx`, `fep_pragmatic_value`, `fep_accuracy`, `fep_complexity`, `fep_surprise`, `fep_td_error`. Prediction error drives learning rate boost. Hierarchical and predictive processing modules supplement the core FEP path. |
| 4 | **Moral Algebra: Classification** | REAL | HIGH | 28 in `moral_algebra.rs`, 10 in `moral_parser.rs`, 8 in `moral_prototypes.rs`, 8 in `moral_text_encoder.rs` | 91.1% classification accuracy. Ablation study shows per-category classifiers contribute +33.6pp. Cached `standard_obligations` for 9.5x speedup. |
| 4 | **Moral Algebra: Topology** | REAL | HIGH | 34 tests in `moral_topology.rs`, 8 proptest in `proptest_moral_topology.rs` | Beta-0 Betti numbers, unity, completeness, circularity metrics. NaN-safe guards. Soak-tested. |
| 4 | **Moral Algebra: Ethics Engine** | REAL | HIGH | 12 tests in `ethics_engine.rs` (982 LOC) | Wired into cognitive loop. Computes moral topology telemetry each cycle (beta_0, unity, completeness, circularity in CycleMetadata). |
| 5 | **Voice: Vocal Tract** | REAL | HIGH | 36 in `pipeline.rs`, 37 in `controller.rs`, 7+6+9+6 in other crate files; 7,694 LOC crate + 16,662 LOC main voice module | 5-formant LF glottal model, LTC controller (avg 4.4 Hz vowel error), cascade AllPoleResonator, CMU Dict G2P (135K words), diphthong trajectories. MCD 4.03 dB. |
| 5 | **Voice: Vocoder** | REAL | HIGH | 32 tests in `vocoder.rs` (2,188 LOC), 3 in `neural_vocoder.rs` | SourceType-aware excitation (7 manner types), OU jitter/shimmer, aspiration noise. WAV output. Neural vocoder is a 274-LOC early-stage addition. |
| 5 | **Voice: Prosody & G2P** | REAL | MEDIUM | 18 in `formant_targets.rs`, 6 in `articulatory_synthesizer.rs`, 9 in `orchestrator.rs` | Intonation contours (statement/question/exclamation), duration model, stress/position lengthening, syllabification. CMU Dict-based G2P. |
| 5 | **Voice: Cognitive Loop Integration** | STRUCTURAL | LOW | Voice accessors in `behavior.rs` provide `VoiceCognitiveState` | Voice pipeline accessible from loop but not driven by it per-cycle. Used via REPL voice and benchmarks, not automatic speech output. |
| 6 | **Swarm: Iroh P2P Bridge** | STRUCTURAL | MEDIUM | 2 tests in `iroh/mod.rs`, 4 in `mind/tests/iroh.rs`, 14 in `service.rs`; `start_iroh()` on ContinuousMind, `enable_p2p()` on Symthaea facade | Iroh integration behind `swarm` feature gate. Bridge actor spawns via tokio. Not exercised in production cognitive loop — no messages flow. |
| 6 | **Swarm: Discovery Layer** | REAL | HIGH | 14 integration tests in `discovery_integration.rs`; 1,000-cycle soak test (`soak_discovery_pipeline`); 0 false rejections | **CapabilityCard** (BLAKE3-hashed self-description from live state), **ReputationBridge** (interaction-gated vouch with phi threshold), **TopologicalHandshake** (weighted substrate+Jaccard+phi compatibility). `capability_card()` on CognitiveLoopService + Symthaea facade. JSON roundtrip verified. Self-compat 1.0, cross-compat 0.97. |
| 6 | **Swarm: Federated Learning** | STRUCTURAL | MEDIUM | 17 in `federated_cfc.rs`, 10 in `hybrid_bft.rs`, 3 in `projection_federated.rs`; 27,604 LOC total swarm, 318 total tests | Byzantine fault tolerance validated to 34%. Federated CfC weight sharing and projection averaging implemented. No live multi-node deployment. |
| 6 | **Swarm: Mesh Network** | STRUCTURAL | HIGH | 82 in `mesh/mod.rs`, 22 in `dual_layer.rs`, 25 in `lora_fragment.rs`, 21 in `mesh_receiver.rs` | LoRa fragment protocol, dual-layer mesh, sensor integration. Compiles clean. No hardware testing. |
| 6 | **Swarm: Holochain Cortex** | STRUCTURAL | LOW | `find_by_capability()` searches LRU cache, sorts by reputation. 1 test. | Cache-only search — no DHT backend (blocked by rmp-serde conflict). Separate Mycelix hApps have full Holochain coverage. |
| 7 | **Physics Bridge** | REAL | HIGH | 75 tests across 8 files (crate); 7 tests in `physics_integration.rs` (210 LOC); 4,303 LOC crate | Feature-gated (`physics-bridge`). `query_cycle()` wired in `cycle_phase_dynamics.rs:479` — queries physics catalog every Nth cycle (default 10), stride-sampled blend into CfC input (default weight 0.1). `PhysicsBridgeTelemetry` populated in output phase. Config: `physics_bridge_blend_weight`, `physics_bridge_query_interval`. |
| 8 | **Memory: Episodic Replay** | REAL | HIGH | 31 tests in `episodic_replay.rs` (1,778 LOC) | Wired into cognitive loop via `cycle_phases_memory.rs`. Dream replay integration exists. |
| 8 | **Memory: Semantic + Coordinator** | REAL | MEDIUM | 7 in `semantic.rs`, 12 in `coordinator.rs`, 9 in `hippocampus.rs`, 5 in `conversation.rs`, 4 in `coherence.rs`; 5,180 LOC total crate | Memory coordinator wired into cognitive loop constructor. Hippocampal model with semantic indexing. |
| 8 | **Memory: Dream Engine** | STRUCTURAL | MEDIUM | Dream replay referenced in `cycle_phases_dream.rs`, `cycle.rs`, constructor | Dream replay mechanism exists and is called in the cognitive loop, but `cycle_phases_dream.rs` has 0 direct tests. Tested indirectly through integration tests. |
| 8 | **Memory: Resonator Codebook** | STRUCTURAL | HIGH | 19 in `resonator.rs`, 32 in `sdm.rs`, 19 in `long_term_memory.rs` | Resonator networks and SDM implemented in symthaea-core. Used for similarity search and pattern completion. Not directly driving the cognitive loop's memory subsystem. |
| 9 | **Substrate Independence: Framework** | REAL | HIGH | 17 in `substrate_independence.rs`, 9 in `substrate_validation.rs`, 9 in `substrate_composition.rs` (35 total); 840+580 LOC | 8 substrate types, 9-dimensional requirements, honest validation with evidence levels and feasibility gaps. |
| 9 | **Substrate Independence: Loop Integration** | REAL | HIGH | 20 tests in `substrate_manager.rs` (636 LOC); referenced in 8 cognitive loop files | `SubstrateManager` computes feasibility, validation overlays, tau modulation. Wired into consciousness equation via `effective_feasibility`. Default `SiliconDigital` substrate. **However**: consciousness engine tests all hardcode `substrate_feasibility: 1.0`, meaning the dynamic path exists but tests don't exercise substrate variation. |
| 10 | **Neuromodulation: Bath System** | REAL | HIGH | 7 tests in `cycle_neuromod_phase.rs`, neuromod accessors with 7+ fields in telemetry | Dopamine, serotonin, norepinephrine, acetylcholine bath with effective levels in CycleMetadata. Wired into cognitive loop per-cycle. |
| 10 | **Neuromodulation: Circadian** | REAL | MEDIUM | 12 tests in `chronobiology.rs` (278 LOC) | Biorhythm with CircadianPhase (Day/Night/etc). Used in loop: dream probability modulated by night phase, circadian_phase appears in CycleMetadata. |
| 10 | **Neuromodulation: Virtual Body / Homeostasis** | STRUCTURAL | MEDIUM | 7 tests in `virtual_body.rs` (352 LOC) | Virtual body state exists and is constructed in the loop, but homeostatic feedback is limited. No full interoceptive loop. |
| 11 | **Reasoning Engine: 7-Step Cycle** | REAL | MEDIUM | 12 in `reasoning_engine/mod.rs`, 6 in `types.rs`, 5 in `telemetry.rs`, 3 in `narrative.rs`; 2,118 LOC | Feature-gated (`reasoning_engine`). Wired in `cycle_phase_dynamics.rs` with `ReasoningContext`. 7-step cycle with Phi/gating/planning. Compiles clean under feature. |
| 11 | **Reasoning Engine: Planning** | STRUCTURAL | MEDIUM | 7 in `mcts.rs`, 5 in `dream_integration.rs`, 4 in `snapshot.rs`, 5 in `types.rs` (temporal_planning) | MCTS-based temporal planning exists. Dream integration for plan rehearsal. Not the primary decision-making path. |
| 12 | **Social Cognition: Theory of Mind** | STRUCTURAL | LOW | 0 tests in `social_coherence.rs` (72 LOC), 6 in `phi_dyad.rs` (341 LOC) | `SocialCoherenceTier` holds Phi-Dyad calculator and partner model. HV ring buffers for dyad computation. Wired in loop but social_coherence.rs itself is a thin struct with no standalone tests. |
| 12 | **Social Cognition: Empathy** | STRUCTURAL | LOW | 6 in `empathic_unification.rs`, 4 in `affective_consciousness.rs` | Empathic unification and affective consciousness modules exist. Referenced in primitive tier and cycle subsystems. Not deeply exercised. |
| 13 | **Vision: Foveation** | STRUCTURAL | HIGH | 7+13+13+8+17 = 58 tests; 1,811 LOC (`symthaea-foveation` crate) | Foveation crop, channel processing, ventral stream, manager. New crate, not yet wired into cognitive loop (no references in `src/cognitive_loop/`). |
| 13 | **Vision: Vision Manifold** | STRUCTURAL | HIGH | 41+62+18+11+12+11+8+8 = 171 tests; 6,908 LOC (`symthaea-vision-manifold` crate) | Encoder, manifold, attention, predictive, bridge, camera, training. Extensive test suite. Not wired into cognitive loop. |
| 14 | **Broca / Language: Projection** | STRUCTURAL | HIGH | 40 in `projection.rs`, 33 in `liquid_mamba.rs`, 20 in `mamba.rs`, 20 in `tokenizer.rs`, 22 in `evaluation.rs`, 21 in `temporal_projection.rs`, 13 in `training.rs`; 14,589 LOC | HDC-SSM projection, Liquid Mamba, tokenizer, training pipeline. Referenced once in `cycle.rs` (SSM projection parameter). Has dedicated training binary. Not driving text generation in the cognitive loop. |
| 14 | **Broca / Language: Temporal Prediction** | STRUCTURAL | HIGH | 21 in `temporal_projection.rs` (1,920 LOC), 9 in `checkpoint.rs`, 8 in `controller.rs` | Temporal projection with checkpointing. Standalone training and evaluation. |
| 15 | **Genesis: Genomics** | STRUCTURAL | HIGH | ~105 tests across 10 files; part of 20,797 LOC across 5 crates | DNA assembly, damage modeling, error correction, FEP agent, quality metrics. 12 integration tests. Not connected to cognitive loop. |
| 15 | **Genesis: Population** | STRUCTURAL | HIGH | ~175 tests across 11 files | Breeding strategy, diversity, effective population, genetics, governance, inbreeding. Self-contained simulation framework. |
| 15 | **Genesis: Cell Foundry** | STRUCTURAL | HIGH | ~163 tests across 16 files | iPSC, IVG, SCNT, multi-scale predictor, lab controller. Ethics gate built in. |
| 15 | **Genesis: Ectogenesis** | STRUCTURAL | HIGH | ~102 tests across 13 files | Artificial womb, consent proxy, microbiome, hormones, monitoring. |
| 15 | **Genesis: Nurture** | STRUCTURAL | HIGH | ~166 tests across 15 files | Bowlby attachment, co-regulation, milestones, sleep, language acquisition. Nurture bridge in cognitive loop (`nurture_bridge.rs`, 6 tests) connects attachment to neuromod bath. |
| 16 | **Neural Bridge** | STRUCTURAL | MEDIUM | 9 in `neural_bridge.rs`, 3 in `neural_bridge_v2.rs`, 4 in `consciousness_probe.rs`; 1,160 LOC | External model integration (ONNX, probe weights). Feature-gated (`neural-bridge`). Optional field in cognitive loop constructor. Loads probe weights from `models/neural_bridge/` if present. |

---

## Summary Statistics

| Category | Total Tests (approx) | Status Distribution |
|----------|---------------------|-------------------|
| Core Pipeline (1) | ~135 | All REAL |
| Consciousness (2) | ~310 | All REAL |
| Active Inference (3) | ~28 | REAL |
| Moral Algebra (4) | ~100 | All REAL |
| Voice Pipeline (5) | ~177 | Mostly REAL, loop integration STRUCTURAL |
| Swarm / P2P (6) | ~332 | Discovery layer REAL (14 integration + soak); P2P/mesh STRUCTURAL |
| Physics Bridge (7) | ~82 | REAL (feature-gated, cycle-integrated) |
| Memory (8) | ~138 | Mostly REAL |
| Substrate Independence (9) | ~55 | REAL (framework + loop integration) |
| Neuromodulation (10) | ~26 | Mostly REAL |
| Reasoning Engine (11) | ~47 | REAL (feature-gated) |
| Social Cognition (12) | ~16 | STRUCTURAL |
| Vision (13) | ~229 | STRUCTURAL (not in loop) |
| Broca / Language (14) | ~208 | STRUCTURAL (not driving loop) |
| Genesis Pipeline (15) | ~734 | STRUCTURAL (standalone) |
| Neural Bridge (16) | ~16 | STRUCTURAL |

## Key Observations

1. **The cognitive loop is real.** The core pipeline (HDC encode, CfC evolve, predict, learn) runs at 234Hz in release mode with extensive tests and profiling. This is the backbone that works.

2. **Consciousness metrics are real but proxy-based.** IIT/Phi is computed via tiered approximation (exact Phi is intractable beyond 12 nodes). GWT broadcasting and the master consciousness equation are wired and tested. The numbers are meaningful within the model but should not be interpreted as measuring actual phenomenal consciousness.

3. **Many subsystems are well-tested but disconnected.** Vision manifold (171 tests), Broca (208 tests), Genesis (734 tests), and Physics Bridge (75 tests) are substantial, well-tested codebases that compile clean but do not feed into the cognitive loop. They are standalone capability modules awaiting integration.

4. **Substrate independence has the plumbing but tests cheat.** The SubstrateManager is wired into the loop and computes dynamic feasibility, but all consciousness engine tests hardcode `substrate_feasibility: 1.0`. The honest confidence path exists but is not validated end-to-end.

5. **Social cognition is thin.** Theory of Mind is represented as a PhiDyad calculation on HV ring buffers (72 LOC, 0 standalone tests). This is the weakest wired subsystem.

6. **Voice pipeline is the most complete peripheral system.** MCD 4.03 dB, CMU Dict G2P, LTC-trained formant controller, cascade filters, prosody -- all with substantial tests. But it operates via REPL/benchmarks, not as automatic cognitive loop output.

7. **Swarm has a real discovery layer but no live networking.** Capability cards, reputation bridge, and topological handshake are tested (14 integration tests, 1,000-cycle soak). Federated learning, BFT, and mesh are structural (27K+ LOC, 318 tests). No multi-node deployment has occurred.

8. **Physics bridge is now cycle-integrated.** Feature-gated (`physics-bridge`), queries physics catalog every Nth cycle and blends into CfC input. Telemetry populated. 82 tests total.

---

*Last updated: 2026-03-04. Run `cargo test --lib` to verify current test counts.*
