# HDC Task × Association × Dimension Map v0.1

**Status:** literature-grounded experiment plan; the task-to-representation recommendations are hypotheses to test, not Symthaea benchmark results. Do not interpret a proposed dimension range as a universal optimum.

## The central distinction

Choose the *association structure* and *similarity geometry* for the task first, then find the smallest dimension that meets the declared quality/reliability target. Increasing dimension reduces some random interference, but it does not repair an unsuitable encoding, data leakage, poor temporal dynamics, or a mismatched similarity geometry.

Three choices must be reported independently:

1. **Representation type:** packed binary/BSC, bipolar MAP, continuous-f32, and (as a later research arm) complex-valued FHRR.
2. **Association/encoding family:** role–filler key/value binding, bundle/multiset, positional/permutation or n-gram sequences, correlated level vectors for numeric features, random projection, or stateful LTC/CfC dynamics.
3. **Width:** semantic hypervector dimension and, for dynamical experiments, local evolving state dimension. Do not silently treat these as the same axis.

## Task-to-representation hypotheses

| Workload / association | Preferred first arm | Initial dimension sweep | Why this is the first hypothesis | What must decide |
|---|---|---:|---|---|
| Symbolic facts, attribute lookup, compositional records | BSC binary or bipolar MAP; role–filler/key–value binding, then bundle fields | 1K, 2K, 4K, 8K, 16K, 32K | Binding naturally expresses key/value or role/filler associations; packed binary offers exact XOR binding and compact storage. | Unbinding/field-retrieval accuracy, false positives at full corpus scale, fan-in tolerance, memory and p95 lookup latency. |
| Small/tabular classification and few-shot learning | Bipolar/binary prototypes; key–value encodings with correlated level vectors for numeric features; compare random projection | 1K, 2K, 4K, 8K, 16K (32K only if needed) | A 2025 comparative review across seven common HDC datasets and 121 UCI tasks found key–value encodings strong for tabular features; random projection often traded some structure for speed. This is a starting point, not a result for Symthaea. | Held-out accuracy/F1, calibration, training data efficiency, update cost, and SVM/LDA/ridge or appropriate conventional baselines. |
| Cognitive retrieval / factorizing object attributes | Key–value binding of object, location, time, size; correlated level codes for numeric/spatial features; iterative factorization as a separate algorithm | 2K, 4K, 8K, 16K, 32K | The geometry of attribute codes matters: the latest studied universal encoding reports a trade-off between near-orthogonality for decoding/factorization and correlated codes for statistical learning. | Factor recovery, spurious attractor rate, attribute error versus codebook correlation length and dimension. |
| Noisy associative cleanup / nearest-neighbour recall | Packed binary BSC or bipolar MAP, with cleanup memory and explicit corruption sweep | 2K, 4K, 8K, 16K, 32K, 64K | Distributed vectors can support robust similarity lookup, but capacity depends on memory size, corruption, bundling fan-in and the acceptable false-match probability. | Recall@1/Recall@k, false acceptance rate over the whole index, corruption tolerance, bytes per item, lookup throughput and index scale. |
| Ordered sequences, n-grams, event-symbol chains, short text | Binary/bipolar positional or n-gram encoding; compare bundled-sequence similarity with binding-based sequences | 2K, 4K, 8K, 16K, 32K | Permutation/position makes order explicit; n-grams preserve local context. Binding-based and bundling-based sequences have different similarity and retrieval behaviour and should not be collapsed into one “sequence” score. | Exact recall, sequence similarity ranking, insertion/deletion/substitution robustness, sequence length and order sensitivity. |
| Sensor/activity/speech-feature classification | Numeric-level plus role–filler encoding or random projection; static HDC prototype baseline first | 1K, 2K, 4K, 8K, 16K, 32K | Structured numeric codes can preserve nearby-value relationships; projection can be faster. For temporal tasks, benchmark static aggregation separately from temporal order-aware methods. | Subject/session-held-out accuracy, early-exit performance, drift robustness, encoding cost and end-to-end latency. |
| Irregularly sampled prediction, temporal association and control | ContinuousHV f32 with existing HDC-LTC/CfC; compare bounded iterative and fused evolution | Semantic: 16K, 32K, 64K, 128K, 256K. Local evolving state: 1K, 4K, 16K, 32K | Continuous state and liquid time constants directly target evolving state and variable timing. A large semantic address space does not prove every semantic coordinate must evolve at every timestep. | Refined-reference trajectory error, tau/gate/equilibrium distortion, held-out forecast/control quality, long-horizon recovery, p95/p99 latency, state traffic and energy. 128K/256K remain exploratory. |
| Robotics and edge-world state | Hybrid: keyed object/relationship associations in HDC plus a bounded local continuous state for motion/control | Semantic: 8K, 16K, 32K, 64K. Local dynamics: 1K, 4K, 16K | Retrieval and symbolic composition may benefit from distributed role–filler codes; control needs low-latency temporal evolution. Treat these as separable components unless end-to-end evidence shows otherwise. | Task success, safety constraint violations, state-estimation error, worst-case latency/deadline misses, memory and transfer cost. |
| Relational graphs, multi-hop and hierarchical structures | Edge/role key–value associations plus explicit composition depth; compare flat bundling against hierarchical/segmented memory | 4K, 8K, 16K, 32K, 64K | More dimensions may reduce interference, but deeper structure also accumulates unbinding noise. Better decomposition or segmented memory may beat simply increasing width. | Exact relation recovery by hop/depth, distractor rejection, query cost and total memory, plus negative controls for invalid paths. |
| Online adaptation and changing environments | Binary/bipolar prototype updates or regenerative classifiers for fast adaptation; continuous state only where temporal evolution is required | 1K, 4K, 8K, 16K, 32K | Lightweight online updates are an HDC strength, but fast adaptation can overwrite old capabilities; larger width alone does not solve retention. | Learning curve at fixed sample budget, adaptation delay, retained-old-task performance, drift detection, update cost and memory growth. |
| Approximate/analogue/in-memory implementation | Keep as a separate precision/hardware track, usually beginning with the validated BSC/MAP kernel or a declared ContinuousHV quantization | Hardware-feasible widths plus matched software widths | Hardware precision, ADC/read noise, array tiling and transfer costs can change which representation is best. Do not compare approximate hardware with exact digital runs without quality and boundary alignment. | Task quality, calibration/error distribution, whole-system energy, kernel-to-system speedup, write/verify costs and reproducibility. |

## Important nuance: similarity geometry can matter as much as D

For categorical symbols, near-orthogonal random codes are a reasonable default. For continuous quantities such as speed, location, time and temperature, independently random codes throw away useful neighbourhood structure; correlated level encodings intentionally make nearby values more similar. But high correlation can make factor retrieval ambiguous. The 2026 study on optimal hyperdimensional representation explicitly studies this learning-versus-decoding tension using MNIST and controlled synthetic examples; it motivates a sweep of correlation length, not a universal encoder winner.

Therefore every comparison must sweep at least one encoding-geometry control where applicable:

- Random/near-orthogonal categorical codebook.
- Correlated level codebook for numeric attributes, with correlation length swept independently of D.
- Random-projection control matched to the same D.
- For ordered input, positional/permutation or n-gram code against unordered bundle.
- For continuous dynamics, full-width evolution against matched local-state projections and the manifold/residual proposal.

## Dimension selection protocol

Do not simply choose whichever D gives the best observed test result.

1. Freeze the tasks, retrieval index scale, sequence lengths, bundle fan-in, corruption/noise model and deployment memory/deadline requirements.
2. Use a coarse power-of-two sweep; begin with 1K–16K for most static tasks and 16K–64K for capacity-sensitive tests. Spend 128K/256K experiments primarily on active ContinuousHV dynamics or tasks where a lower width demonstrably fails a declared criterion.
3. Sweep representation/encoding and dimension as separate factors. Do not change precision, encoding, training algorithm and D simultaneously in one comparison.
4. Select D and hyperparameters on training/validation data only. Freeze that choice before held-out testing; for time series, use chronological or subject/session-separated splits.
5. Choose the **smallest D that clears the predeclared task and reliability gates**, then report the complete quality–latency–memory–energy Pareto frontier. Larger D wins only if its extra quality/reliability justifies its added cost.
6. Compare with fit-for-purpose baselines, including conventional classifiers on classification tasks and a conventional time-series/control baseline for temporal tasks.
7. Record every failed or losing cell. An unevaluated cell is unknown, not a pass; do not hide a negative result by changing a threshold after seeing it.

## Minimum per-cell receipt

Record repository commit and dirty state; representation and precision; semantic and local dynamical widths; binding/bundling/sequence encoding; numeric correlation scale; seed and corpus hash; train/validation/test split; codebook and prototype counts; fan-in, sequence length, noise and query ambiguity; quality and failure counts; false-positive rate at declared index size; p50/p95/p99; updates or queries per second; resident and peak temporary bytes; bytes moved/allocations; energy boundary and instrument uncertainty where measurable.

For dynamical cells additionally record dt profile, tau/gate/equilibrium distortion, per-step and terminal trajectory error against a refinement-checked reference, long-horizon boundedness/recovery, and held-out task quality.

## Research basis and limits

- Vergés et al., *Classification using hyperdimensional computing: a review with comparative analysis*, Artificial Intelligence Review (2025): compares seven established HDC datasets and 121 UCI classification datasets; found key–value encoding strong on tabular data and notes the need for consistent benchmarking. https://doi.org/10.1007/s10462-025-11181-2
- Poduval et al., *Optimal hyperdimensional representation for learning and cognitive computation*, Frontiers in Artificial Intelligence (2026): investigates how dimension and code correlations influence classification, retrieval and factorization. Its results motivate the correlation-geometry sweep; its simplified experiments do not establish a universal best dimension for Symthaea. https://doi.org/10.3389/frai.2026.1690492
- Kleyko et al., *Hyperdimensional computing: a framework for stochastic computation and symbolic AI*, Journal of Big Data (2024): reviews binding, bundling, permutation, records, sequences and associative memory. https://doi.org/10.1186/s40537-024-01010-8
- Hasani et al., *Liquid Time-constant Networks*, AAAI (2021): grounds LTC as a time-continuous recurrent model for time-series prediction and varying time constants. https://doi.org/10.1609/aaai.v35i9.16936

The literature provides useful hypotheses and baseline tasks, but does not establish which dimension is best for Symthaea’s current implementation. That requires the controlled measurements above.
