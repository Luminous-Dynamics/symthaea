# NEUROALIGN-001 — public-main source inventory and pilot selection

**Audit date:** 2026-11-27  
**Audited subject:** `Luminous-Dynamics/symthaea@1099539fa43a4578948ab152a622ac1bca04efdb`  
**Status:** source audit only; no model weights or neuroscience datasets were downloaded, no Rust commands were run, and no benchmark scores were produced.  
**Parent:** [Issue #14054](https://github.com/Luminous-Dynamics/symthaea/issues/14054)

## Executive decision

Symthaea has two credible, distinct evaluation surfaces:

1. **BGE-M3 transformer layer activations** — the strongest current candidate for a first Brain-Score Language neural/behavioural predictivity pilot. It exposes real pretrained transformer layers, but is feature-gated and currently lacks a pinned model-weight revision/digest and a benchmark adapter.
2. **Patch-HDC + CfC VisionManifold** — the most native HDC/continuous-time candidate for controlled visual dynamics and computation-through-dynamics tasks. It is a seeded patch-feature encoder, not a pretrained visual backbone. It should first be compared with simple image features, Gabor filters, and matched CNN/ViT and temporal baselines.

**Do not start with NLB'21 as the first pilot.** NLB's primary evaluation involves modelling neural population spiking activity, including co-smoothing held-out neurons. Symthaea currently has no qualified NLB-compatible neural population decoder/latent-variable evaluation adapter. Build and validate that interface after reproducing the official NLB baseline.

## Source inventory

| Surface | Main source | What is actually present | Availability / qualification status |
|---|---|---|---|
| BGE-M3 layer extractor | `src/perception/layer_extractor.rs`, `src/perception/bge_m3.rs` | Candle implementation loading tokenizer/config/weights and exposing embedding output plus each transformer layer; declared 24 layers × 1,024 hidden dimensions | Behind `neural-bridge`; downloads model artifacts from Hugging Face; not run or weight-pinned in this audit |
| Unified embedding API | `src/perception/modern_embeddings.rs` | BGE-M3 backend provides per-layer activations; `UnifiedEmbedder::auto()` selects BGE-M3 | ModernBERT, Nomic, GTE and E5 entries currently fall back to BGE-M3 rather than loading distinct backends; `embed_batch` is sequential |
| HDC activation projection | `src/perception/neural_bridge.rs`, `src/perception/neural_bridge_v2.rs` | Linear projection from encoder features to 16,384D HDC; optional epistemic metadata | `NeuralBridgeV2::load_default()` requires `models/neural_bridge/probe_weights_bge_m3.npy`; `encoder_only()` creates a zero-filled projection, so its projected HDC output must not be treated as a trained representation |
| Patch-HDC visual encoder | `crates/domains/symthaea-vision-manifold/src/encoder.rs` | Deterministically seeded position/feature/level hypervector bases; patch features include base features, with configurable motion, YCbCr colour, opponent-colour and optional depth channels | Root package feature `vision-manifold` is optional and not part of the default feature set; no pretrained visual backbone |
| Continuous-time visual state | `crates/domains/symthaea-vision-manifold/src/manifold.rs` | CfC-style temporal manifold, prediction, surprise tracking and optional predictive hierarchy/object binding | Good candidate for temporal-dynamics qualification; not yet a neural recording decoder |
| Model integrity | `src/perception/model_integrity.rs`, `src/perception/layer_extractor.rs` | SHA-256 utilities and verified-mmap wrapper exist | Layer extractor currently calls the wrapper with `expected_hash = None`; the wrapper allows loading while warning that weights are unverified. Experiment receipts must pin exact downloaded weight digests independently |
| Perception registry | `src/perception/model_status.rs` | Tracks NotAttempted / Loading / Loaded / Unavailable / Degraded | Status reporting is useful, but registry state is not proof of model identity or neural validity |

### Important source-level distinctions

- A layer extractor implemented in source is not evidence it has run successfully on the current machine.
- BGE-M3's 24 layer outputs are pretrained transformer activations; the HDC projection is a separate learned artifact and must have its own data, training, and checkpoint lineage.
- The BGE-M3 model and layer extractor are language encoders. They must not be entered into vision benchmarks as visual models.
- VisionManifold's HDC basis construction is deterministic given seeds, but determinism is not the same as pretrained biological feature learning.
- Current layer extraction returns pooled activations, with optional token activations. Preserve exact pooling, tokenization, attention-mask and layer semantics in benchmark receipts.
- Current BGE-M3 layer loading passes no expected SHA-256 to the integrity wrapper. Pin the exact upstream revision and compute/record local weight digests before any confirmatory comparison.

## External benchmark selection

### 1. Brain-Score Language — first neural/behavioural pilot candidate

Official project and documentation:
- https://github.com/brain-score/language
- https://brain-score-language.readthedocs.io/en/latest/

Why it fits: the platform exposes language benchmarks that evaluate computational models against human behavioural and internal brain measurements. BGE-M3 provides a concrete language representation with intermediate activations.

Required work before scoring:
- Confirm the chosen benchmark's stimulus format, language, task and model API contract.
- Implement an `ArtificialSubject` adapter (or the current officially supported equivalent) with a faithful mapping from benchmark stimuli to BGE-M3 inputs and requested layer responses.
- Record BGE-M3 repository revision, tokenizer/config digests, weight-file digests, layer index, pooling, device and precision.
- Reproduce one official reference model's published score using the exact benchmark version before evaluating Symthaea.
- Compare final-layer and preregistered intermediate-layer probes; never select the best layer on confirmatory data.
- Include a text/statistical baseline and the benchmark's strongest compatible published models.

**Claim ceiling:** a successful result would establish benchmark-specific neural/behavioural predictivity, not that Symthaea has a human-like mind or is conscious.

### 2. Brain-Score Vision — conditional HDC visual pilot

Official project: https://www.brain-score.org/

The Patch-HDC encoder can be exposed as a visual model candidate, but only for benchmarks where its input/output interface and visual scale are appropriate. Start with a declared low-level or early-visual hypothesis; do not claim ventral-stream equivalence based on patch encoding alone.

Required arms:
- pixel/statistical baseline;
- Gabor / oriented-energy baseline;
- Patch-HDC without temporal dynamics;
- Patch-HDC + CfC;
- matched CNN / ViT reference;
- optional HDC ablations for position, colour, opponent colour, motion and object binding.

Use benchmark-native ceilings and per-area scores. Do not average across unrelated brain areas to mask failures.

### 3. Computation-through-Dynamics Benchmark (CtDB) — native temporal pilot

Paper/resources:
- https://pubmed.ncbi.nlm.nih.gov/39975240/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11839132/

Compose with [#12368](https://github.com/Luminous-Dynamics/symthaea/issues/12368), which already owns HDC-LTC/CfC update-law qualification and temporal benchmark baselines.

Use the official task definitions where compatible. Measure the computation performed by the learned dynamics, not only trajectory reconstruction. Include simple numerical references and matched recurrent models. A synthetic CtDB result is dynamical-mechanism evidence, not biological neural correspondence.

### 4. Neural Latents Benchmark '21 — later neural-population lane

Official resources:
- https://neurallatents.github.io/
- https://github.com/neurallatents/nlb_tools
- https://neurallatents.github.io/datasets.html

The EvalAI challenge stopped accepting submissions in January 2026 and the test data were released, so local evaluation is possible but increases the risk of test-set overfitting. The official suite includes seven dataset variants across four tasks/brain areas; the principal modelling objective includes co-smoothing unseen neurons.

Before using it:
- Reproduce the official codepack baseline first.
- Implement a dataset-native neural latent / decoder interface, not a text or image HDC adapter.
- Split and report by dataset/task/subject as prescribed.
- Keep the now-public test split out of all model and hyperparameter selection; report multiple seeds and the full exposure history.

### 5. FALCON — later robust neural decoding lane

Official resources:
- https://snel-repo.github.io/falcon/
- https://pubmed.ncbi.nlm.nih.gov/39345641/

Proceed only after Symthaea has a valid sensorimotor decoding interface and a clear training / held-out / recalibration protocol. Preserve task-specific metrics (e.g. movement prediction versus communication decoding) rather than merging them into one score.

## Proposed benchmark matrix

| Arm | Representation / dynamics | Main question |
|---|---|---|
| A | Linear / ridge / GLM | How much can a simple control explain? |
| B | Gabor / handcrafted visual features | Is the visual task solved by low-level features? |
| C | BGE-M3 raw layer activations | What neural/behavioural alignment exists before HDC projection? |
| D | BGE-M3 + verified trained HDC probe | Does HDC preserve or improve useful structure? |
| E | Patch-HDC only | Does the HDC visual code add value over hand-crafted baselines? |
| F | Patch-HDC + CfC | Does temporal dynamics add value? |
| G | Matched GRU/LSTM | Does CfC earn its complexity? |
| H | Suitable SSM / Neural CDE / GRU-ODE-Bayes | Does continuous-time modelling improve irregular-time tasks? |
| I | Strong official benchmark model | Where is the current external reference? |
| J | Oracle / noise ceiling, if defined | What is the task's upper bound? (not a deployable competitor) |

All comparisons must be task-compatible. Do not compare an image model against language data, or a neural spike decoder against an embedding benchmark.

## Preregistered metric groups

Keep metrics separate; do not invent a single “neuroscience score.”

1. **Neural/behavioural predictivity:** benchmark-native score, cross-validated neural prediction, behavioural likelihood/accuracy, noise-ceiling-normalized result where the benchmark defines it.
2. **Dynamics:** task computation accuracy, trajectory error, temporal generalisation, response to controlled perturbations, long-horizon drift and irregular-sampling robustness.
3. **Generalisation:** held-out subjects/sessions, held-out stimulus/task families, OOD transfer, worst-group outcomes.
4. **Uncertainty:** calibration, coverage, selective risk, abstention and performance under data shift.
5. **Efficiency:** parameter count, weight bytes, peak RSS/VRAM, training/inference compute, p50/p95/p99 latency and measured energy at a stated boundary.
6. **Evidence quality:** exact source/model/data/preprocessing/seed identities, replay success, invalid/missing execution counts, and independent reproduction.

## Promotion gates

- Reproduce the official baseline under the same benchmark version before a Symthaea comparison.
- Freeze model revision, stimulus adapter, layer/pooling, task split, score, preprocessing and calibration manifest before confirmatory evaluation.
- Use held-out subjects/sessions and task families; trial/time-bin bootstrap alone is not an independent-subject analysis.
- Require a practically meaningful paired improvement with an uncertainty interval excluding no improvement before claiming superiority. Set the practical-effect margin using a development-only pilot and preregister it before holdout scoring.
- For efficiency claims, establish non-inferior task quality under a preregistered margin and then compare measured resource use.
- If a linear/Gabor/GRU or official baseline wins, use that model for the task. A failed HDC/CfC arm is still a valuable result.
- Publish negative and inconclusive results. No result in this programme is an AI-consciousness determination.

## Immediate execution order

1. **BGE-M3 reproducibility:** pin upstream revision and weight digests; capture one known input's token IDs and layer outputs; test all-layer extraction and deterministic replay on the target runtime.
2. **Brain-Score Language adapter:** select one compatible benchmark, reproduce its official baseline, then score BGE-M3 with preregistered layers.
3. **VisionManifold export:** add a read-only feature/trajectory export with complete model/config/input/timestamp receipts; use it in #12368/CtDB-compatible experiments.
4. **NLB compatibility audit:** reproduce the official codepack baseline before designing any Symthaea neural-population adapter.
5. **HDC probe qualification:** only after probe provenance is pinned, test raw BGE-M3 versus trained HDC-projected activations with identical splits and parameter-selection rules.

## Existing work to compose, not duplicate

- [#14054 NEUROALIGN-001](https://github.com/Luminous-Dynamics/symthaea/issues/14054) — programme owner.
- [#8971 consciousness theory-neutral adversarial benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/8971) and [PR #8972](https://github.com/Luminous-Dynamics/symthaea/pull/8972) — theory cards and Cogitate public-data audit.
- [#12368 HDC-LTC dynamics arbitration](https://github.com/Luminous-Dynamics/symthaea/issues/12368) — continuous-time update law and temporal baseline ownership.
- [#8391 FEP epistemic-value qualification](https://github.com/Luminous-Dynamics/symthaea/issues/8391), [#11122 EVSI integration](https://github.com/Luminous-Dynamics/symthaea/issues/11122), [PR #8394](https://github.com/Luminous-Dynamics/symthaea/pull/8394) — action-conditioned information value.
- [#3041 Psych-Bench calibration freeze](https://github.com/Luminous-Dynamics/symthaea/issues/3041), [PR #3043](https://github.com/Luminous-Dynamics/symthaea/pull/3043), [PR #3044](https://github.com/Luminous-Dynamics/symthaea/pull/3044), [PR #3045](https://github.com/Luminous-Dynamics/symthaea/pull/3045) — calibration and target provenance.

## External research references

- Friston et al. (2026), *Active inference and artificial reasoning*, Nature Communications, DOI: https://doi.org/10.1038/s41467-026-77209-5
- Cogitate Consortium (2025), *Adversarial testing of global neuronal workspace and integrated information theories of consciousness*, Nature, DOI: https://doi.org/10.1038/s41586-025-08888-1
- Brain-Score Language: https://github.com/brain-score/language
- Neural Latents Benchmark: https://neurallatents.github.io/
- Computation-through-Dynamics: https://pubmed.ncbi.nlm.nih.gov/39975240/
- FALCON: https://snel-repo.github.io/

---

**Current result:** a source-grounded candidate inventory and ranked pilot plan. **Not yet a result:** model-loading success, benchmark adapter, neural-data download, measured score, SOTA comparison, or biological/consciousness claim.
