# AR-5A — Genotype / phenotype / evaluation influence audit v0.1

**Status:** source-audited, narrow implementation in progress; focused Rust tests are authored but not yet executed.  
**Parent:** AR-5 #9197  
**Source baseline reviewed:** `707efe6147eb9f083977aef91721a8424db0d9d7`  
**Claim ceiling:** this audit identifies encoded fields and their use under one evaluator profile. It does not establish that neuroevolution discovers better architectures.

## Why this audit exists

A search algorithm only searches the design variables that its decoder and evaluator actually connect. A genome can be large while most bits are neutral, and a parameter can decode into a configuration while never being consumed by the experiment's execution path.

AR-5 therefore requires a genotype-to-phenotype-to-measurement influence map before treating evolutionary improvement as architecture discovery.

## 1. Encoded-bit census

The neuroevolution `NeuralGenome` is stored as a 16,384-bit `BinaryHV`.

The checked-in decoders read these ranges:

| Region | Bit range (half-open) | Bits | Use |
|---|---:|---:|---|
| Neural configuration | `[0, 380)` | 380 | CfC/HDC configuration, topology flags, network initialization seed |
| Reserved gap | `[380, 400)` | 20 | No decoder reads these positions |
| Cognitive threshold configuration | `[400, 634)` | 234 | Threshold phenotype and static threshold-consistency fitness |
| Unreferenced tail | `[634, 16384)` | 15,750 | No decoder reads these positions |
| **Total read by either decoder** | | **614** | 3.75% of stored bit positions |

Thus 15,770 of 16,384 positions (96.25%) are not read by the two current decoders. This is a source-level bit-range census, not a claim that all 614 read positions independently change behavior. Several encodings are many-to-one, and some decoded values are inactive under a given execution profile.

Before this change, mutation, crossover and Hamming distance operated on the full 16,384-bit vector. Neutral bits could therefore:
- consume mutation/crossover variation without changing decoded fields;
- make two behaviorally identical candidates appear genetically distant;
- inflate the diversity signal used by speciation.

The AR-5A implementation now canonicalizes unencoded positions to zero, limits crossover to encoded bytes, and computes Hamming distance only over decoded bit ranges. This is still an **encoded-locus distance**, not a semantic architecture distance; it intentionally retains distinctions such as initialization seed and bit patterns that may decode to equivalent values.

## 2. Neural-locus influence inventory

The current source mapping is:

| Locus | Decoded destination | Current use in neuroevolution evaluation profile |
|---|---|---|
| `tau_base` | `UnifiedConfig.tau_base` | Consumed by continuous-time dynamics |
| `backbone_tau` | `UnifiedConfig.backbone_tau` | Consumed by state-dependent time-constant calculation |
| `activation` | `UnifiedConfig.activation` | Consumed by neuron equilibrium/activation |
| `learning_rate` | `UnifiedConfig.learning_rate` | Used by explicit update methods; current neuroevolution evaluator does not call those methods |
| `momentum` | `UnifiedConfig.momentum` | Used by explicit update methods; current neuroevolution evaluator does not call those methods |
| `weight_decay` | `UnifiedConfig.weight_decay` | Used by explicit update methods; current neuroevolution evaluator does not call those methods |
| `gating_steepness` | `UnifiedConfig.gating_steepness` | Consumed by closed-form gating |
| Fourier count/frequencies | `UnifiedConfig.fourier_frequencies` | Consumed by time-varying Fourier basis injection |
| Layer count/sizes | `UnifiedNetworkConfig.layer_sizes` | Determines instantiated network topology |
| Skip connection flag | `UnifiedNetworkConfig.skip_connections` | Consumed by layer-input construction |
| Layer binding flag | `UnifiedNetworkConfig.use_layer_binding` | Consumed by layer-input construction |
| `weight_seed` | `NeuralPhenotype.weight_seed` | **Previously decoded but not used by spawn/reset; fixed in this tranche** |

The learning parameters are not intrinsically useless: they affect explicit Hebbian/STDP/contrastive/adaptive update methods in the core neuron implementation. The narrower finding is that the present neuroevolution fitness loop evolves the network with `evolve_closed_form`, feeds observations to the FEP agent, and does not invoke those update methods. Consequently, those genes are dormant under this exact evaluation profile. A future training/update profile may activate them and must report that as a different treatment.

## 3. Initialization-seed mismatch repaired

Before AR-5A:
- `NeuralGenome::decode()` extracted a 64-bit `weight_seed`.
- `NeuralOrganism::spawn_with_dim()` initialized `HdcLtcUnifiedNetwork` from the caller's root `GenesisSeed`, not the decoded `weight_seed`.
- `reset_evaluation()` initialized the network from an organism-ID-derived phrase.

The locus therefore did not control initialization, and resetting the same genotype could change its network representation based on organism ID rather than the genome's initialization variable.

AR-5A changes this path:
1. derive a domain-separated network seed from the root seed phrase plus the decoded `weight_seed`;
2. retain the root seed in the organism;
3. reconstruct the same genotype-conditioned initialization on reset;
4. test that changing only `weight_seed` changes the initialized network;
5. test that the same genotype/root seed reproduces initialization across organism IDs and resets.

This is a seed/representation-identity repair. A change in `weight_seed` is **not** counted as a new architecture topology or new computational paradigm.

## 4. Threshold-locus influence ceiling

The threshold decoder reads bits `[400, 634)`. In the current neuroevolution path, `compute_fitness()` calls `evaluate_threshold_fitness()` and stores its static consistency score as `threshold_fitness`.

That means these loci influence the current selection score through a hand-authored consistency heuristic. This does not demonstrate that the corresponding threshold values are wired into a live cognitive-loop execution during this evaluator profile. Future claims must distinguish:
- decoded threshold value;
- static consistency score;
- live runtime consumer;
- independently measured capability effect.

The safety thresholds remain outside the genome by design; this audit does not alter that boundary.

## 5. Known decoder aliases

The current decoder uses mappings such as:
- activation index modulo 5;
- layer count modulo 5;
- layer sizes modulo 32;
- Fourier count capped at 8.

Consequently, multiple bit patterns can map to the same decoded value. Active-bit Hamming distance can still overstate phenotype distance. The next semantic-distance tranche should compare canonical decoded fields and, eventually, measured structural behavior—not pretend raw bit distance is architecture distance.

## 6. Separate fitness-objective audit

A separate source audit should reconcile `FitnessWeights`, the composite score, Pareto dominance, and crowding distance. Their declared objective sets and weighting paths need to agree before evolutionary results are interpreted as optimizing a stable multi-objective contract. That work is intentionally not changed in this seed/inactive-bit repair.

## 7. Validation plan and current status

Authored regressions in this tranche:
- random/genesis constructors canonicalize unencoded bits;
- mutation/crossover do not preserve or create unencoded variation;
- Hamming distance ignores reserved/unencoded bits;
- changing only `weight_seed` changes network initialization;
- identical genotype/root seed reproduces the initialization across IDs and resets.

**Not yet executed:** local `cargo test`, `cargo fmt`, or exact-head hosted tests. The current authoring environment has no Rust toolchain. CI must remain the authoritative executable evidence, and queued/skipped jobs are not passes.

## 8. What this unlocks for architecture discovery

The corrected substrate makes the search process more honest:
- candidates differ through declared loci rather than random tail noise;
- network initialization becomes a reproducible, explicit treatment variable;
- unused parameters are reported as profile-inactive instead of receiving assumed credit;
- future architecture-only distance can exclude seed variables and collapse decoder aliases;
- new operators can be admitted only after a counterexample demonstrates that the frozen operator grammar cannot express a required behavior.

The desired next step is a canonical typed candidate representation that separates:
`ArchitectureID`, `RepresentationSeed`, `EvaluationSeed`, `CandidateConfigDigest`, and `EvaluatorProfileID`.

That is the bridge from fixed-genome optimization toward genuine search over architecture structure and runtime regimes.
