# HLB-MEMORY-001 — Holographic Memory Engine (HME)
## Architecture proposal and qualification plan

**Status:** Research RFC / design proposal. No engine implementation, migration, performance result, or production-readiness claim is made here.

**Date:** 2026-11-07  
**Repository baseline:** Symthaea `main` at `e327835a6d8ac815135c92dd582785fbe2928833`  
**Implementation tracker:** [Issue #10915](https://github.com/Luminous-Dynamics/symthaea/issues/10915)

---

## 1. Executive decision

Symthaea should investigate a purpose-built **Holographic Memory Engine (HME)**: a Rust memory/storage/index layer designed around the actual data model of its HDC/CfC/LTC cognitive architecture.

The recommendation is **not** to build a general-purpose DBMS from scratch. Instead:

1. Keep live HDC-LTC state in preallocated, resident memory. A cognitive time step must not depend on database, disk, network, or vector-index I/O.
2. Separate live dynamical state, associative long-term memory, and durable checkpoint/replay state. They are different workloads with different consistency and latency contracts.
3. Design HDC-aware immutable vector segments, dimension/representation metadata, query planning and exact reranking.
4. Use a well-understood embedded storage primitive for the first persistent implementation. Compare current SQLite, Fjall, redb and RocksDB; do not select one by reputation.
5. Make the memory index rebuildable from exact source records and transformation manifests. A search index is never the source of truth.
6. Integrate the existing 2K–256K dynamical-HDC programme, the state-continuous resolution-transition work, and the existing tiered memory architecture. Do not create a second dimension controller, memory manager, temporal ledger, provenance graph, or hardware evidence pipeline.

The most important research hypothesis is that **semantic width and evolution width need not be the same**. Symthaea may benefit from a high-dimensional semantic address space while evolving a smaller local dynamical state, provided the projection, residual/novelty pathway, and liquid trajectory are explicitly qualified. This is an experiment, not an assumption.

## 2. Canonical owners and scope boundaries

HME composes with the following existing work:

| Existing owner | Responsibility HME consumes |
|---|---|
| [#7919 — MEMORY-SUBSTRATE-001](https://github.com/Luminous-Dynamics/symthaea/issues/7919) | General tiered RAM/persistent-storage architecture, query/update/recovery planes, integrity and repairability |
| [#7749 — NEURO-SILICON-002](https://github.com/Luminous-Dynamics/symthaea/issues/7749) | Dynamical HDC-LTC 16K–256K, semantic-width versus evolution-width matrix, full-width/compressed controls |
| [#6596 — resolution extensibility](https://github.com/Luminous-Dynamics/symthaea/pull/6596) | Open power-of-two resolution metadata; canonical versus exploratory versus custom tiers; operator evidence |
| [#6560 / #6561](https://github.com/Luminous-Dynamics/symthaea/issues/6560) | Explicit state-continuous conversion and trajectory-resolution evaluation |
| [#7772 — HDC-CAPACITY-001](https://github.com/Luminous-Dynamics/symthaea/issues/7772) | Fan-in, false-match curves, query multiplicity and capacity |
| [#10693 — HLB-002](https://github.com/Luminous-Dynamics/symthaea/issues/10693) | HDC/CfC neuron correctness, optimizer and HoloLiquid v2 contracts |
| [#7864 / #7866](https://github.com/Luminous-Dynamics/symthaea/issues/7864) | End-to-end HDC retrieval and data-locality benchmark / host baseline |
| [#10896](https://github.com/Luminous-Dynamics/symthaea/issues/10896) | External vector-search backend bake-off; now explicitly separate from internal HME architecture |
| [#3348](https://github.com/Luminous-Dynamics/symthaea/issues/3348) | Manifold-compressed liquid dynamics and projection controls |
| [#10901 / #10906](https://github.com/Luminous-Dynamics/symthaea/issues/10901) | FPGA and memristive near-data-compute experiments |

HME must not create:
- a competing HDC algebra or a new similarity definition;
- a second HDC resolution controller;
- a generic cognitive memory manager duplicating #7919;
- a canonical proposition graph, temporal ledger, rights registry, or evidence receipt;
- a replacement for PostgreSQL, SQLite, Holochain, or DuckDB;
- a new generic benchmark runner or hardware qualification pipeline.

### 2.1 What HME owns

HME owns the boundary between:
- HDC-native representation layout and storage;
- resident working-set management for memory items and indexes;
- immutable memory segments and their manifest;
- HDC-specific candidate generation and exact reranking;
- append-only memory updates and background consolidation;
- deterministic checkpoint/replay of memory-engine state;
- resource admission and observability specific to these workloads.

It does **not** own the semantics of whether an external proposition is legally authoritative, true, rights-qualified, or temporally applicable. Those remain with the existing Mycelix/Symthaea evidence contracts.

## 3. Repository audit: current architecture and concrete scaling pressure

The following observations come from source inspection of the current public repository. They are not measurements from a local build or runtime.

### 3.1 Resolution tiers are not yet one coherent production architecture

- The global HDC configuration currently describes 4K, 16K, 32K and 65,536 dimensions, with 16,384 as the default.
- The HDC-LTC bridge's adaptive configuration defaults to a 2,048–16,384 range, stepping by 2,048.
- The open resolution-extension work #6596 classifies 1K–64K as canonical and 128K/256K as exploratory. It introduces an open power-of-two resolution type and machine-readable operator-conformance evidence; it does not make 128K/256K production defaults.
- #7749 already asks the more important dynamical question: at 16K, 32K, 64K, 128K and 256K, which execution profile offers the best trajectory-quality, latency, state-memory and data-movement trade-off?

A valid power-of-two dimension is not evidence that a task benefits from it, that the allocation is affordable, or that changing to it preserves the liquid trajectory.

### 3.2 The current adaptive bridge reconstructs the network

The current `HdcLtcBridge::maybe_resize()` changes `hdc_dim`, rebuilds the input and output projections, and constructs a new `HdcLtcUnifiedNetwork` at the destination dimension.

That is a network reconstruction, not a proof-carrying conversion of the live liquid state. It does not itself establish preservation of:
- current neuron states;
- learned role/weight vectors;
- input masks;
- tau modulators;
- gate parameters;
- optimizer momentum;
- temporal counters and history;
- historical snapshots.

#6560/#6561 already own the state-continuity and trajectory-conversion problem. HME must consume their typed conversion contracts and must not silently reset or resize state.

### 3.3 A unified neuron is materially larger than one HDC vector

The inspected `HdcLtcUnifiedNeuron` owns eight `ContinuousHV`-sized arrays:
1. state;
2. weight hypervector;
3. input mask;
4. tau modulator;
5. gate weight;
6. gate bias;
7. weight momentum;
8. input momentum.

For continuous f32, one vector costs (4D) bytes. At 256K, that is 1 MiB per vector and 8 MiB for those eight arrays per neuron before allocator overhead, configuration, snapshots, projection matrices, and temporary buffers.

At 100 neurons, those eight arrays alone represent about 800 MiB of logical payload. At 1,000 neurons, about 7.8 GiB. These figures illustrate a working-set pressure point; they are not measured peak RSS.

### 3.4 The bridge projection matrices can dominate

The current bridge defaults to input and output widths of 256. It maintains:
- input projection: (input_dim 	imes D) f32 values;
- output projection: (D 	imes output_dim) f32 values.

With both widths at 256, their combined logical size is:

[
4D(256+256) 	ext{bytes}
]

At D=256K this is 512 MiB, before the neuron state, scratch snapshots, and other working buffers. This is one reason that simply increasing the vector resolution is not a free capacity upgrade.

### 3.5 LiquidHolocell is not the unified HDC-LTC neuron

The inspected `LiquidHolocell` is a simpler scalar-tau memory primitive. Its current dilation:
- upsamples by tiling/permuting the existing values;
- downsamples by bundling chunks and explicitly loses information.

Static similarity or an apparently preserved first segment does not establish preservation of the resulting dynamical trajectory. HLB-002 and #6560/#6561 must qualify this independently.

### 3.6 Existing memory infrastructure must remain a baseline

Symthaea already has:
- `rusqlite`/SQLite persistence for local knowledge, conversation/memory and other local records;
- episodic memory in `symthaea-memory`;
- BinaryHV LSH;
- optional DuckDB for analytical work;
- exact in-memory HDC operations and SIMD paths.

The HME must demonstrate why any new segment format, storage engine or index is superior for a declared workload before it becomes a production dependency.

## 4. Three memory roles that must never be conflated

### 4.1 Live dynamical state

The live state is the mutable state required to evolve the cognitive system now.

Properties:
- bounded update latency;
- explicit time and update counters;
- preallocated or predictably allocated buffers;
- no synchronous disk or network I/O in a time step;
- exact ownership of current state and trainable parameters;
- snapshot boundaries under a declared policy.

This belongs in the hot in-memory state arena. Persistent storage is an observer/checkpoint path, not a dependency of the per-step update.

### 4.2 Associative long-term memory

Associative memory contains episodes, concepts, learned relations, sensory patterns and reusable representations. It is queried, updated, reinforced, decayed, consolidated, indexed and retrieved.

Properties:
- exact HDC representation and similarity contract;
- item identity and representation profile;
- retrieval under interference and fan-in;
- query-scale false-match risk;
- optional approximate candidate indexing followed by exact reranking;
- provenance, temporal and access metadata where an item references source evidence.

This is where an HDC-aware segment layout and query planner are most likely to add value.

### 4.3 Durable checkpoint and replay state

Checkpoint/replay state preserves the information needed to resume or audit an execution.

A checkpoint is not an episode and is not a semantic vector. It may include:
- neuron states;
- learned parameters and optimizer momentum;
- time/update counters;
- projection/model/architecture fingerprints;
- memory-engine manifest generation;
- the update-log sequence included in the snapshot.

A checkpoint policy should be bounded and lifecycle-aware. Persisting every solver operation by default would create unnecessary write amplification and potentially interfere with real-time operation.

## 5. Proposed HME architecture

<escape>
                         Cognitive loop
                               |
                    +----------+----------+
                    |                     |
              Live state arena      Memory query API
                    |                     |
           HDC-LTC evolve path      Eligibility + exact rerank
                    |                     |
             Snapshot boundary     Associative query/index plane
                    |                     |
                    +----------+----------+
                               |
                    Append-only update log
                               |
                  Immutable segment builder
                               |
                    Atomic manifest publish
                               |
                    Recovery / replay plane
</escape>

### Plane A — live liquid-state arena

Responsibilities:
- aligned, preallocated slabs for active state and parameter vectors;
- explicit representation type and validated resolution;
- separate mutable state from immutable model/role snapshots;
- checked resource admission before allocation;
- explicit memory budgets for resident state and temporary scratch;
- optional fast/medium/slow placement only where the active cognitive architecture specifies distinct state roles;
- bounded checkpoint handoff to the persistence plane.

Design variants to benchmark:
- structure-of-arrays (SoA);
- array-of-structures (AoS);
- grouped/tiled SoA by neuron and parameter role;
- hot/cold split for active state versus infrequently used parameters.

Do not assume SoA or AoS is categorically faster. Access order in the actual HDC-LTC update, SIMD width, cache and working-set size must decide.

### Plane B — associative query/index plane

Responsibilities:
- store immutable, resolution-homogeneous segments with stable item IDs;
- expose representation-aware distance and algebra contracts;
- use exact scan for small/hot working sets;
- use an index only when the index is measured to help;
- apply hard eligibility/access constraints before evidence is presented;
- perform exact HDC reranking against the canonical stored representation;
- return IDs, exact scores and lineage references rather than copying entire candidate stores unnecessarily.

Representation families must remain distinct:

| Representation | Likely exact operation | Candidate index baseline |
|---|---|---|
| Packed binary/BSC | XOR + popcount / Hamming | Existing BinaryHV LSH; exact SIMD scan |
| Bipolar/discrete | Declared algebra-specific similarity | Exact scan first; qualify separate ANN |
| Continuous f32 | Declared dot/cosine/distance function | Exact SIMD scan; test HNSW only if the distance and dimension are supported |
| Quantized | Original-space similarity approximation | Versioned quantization index + exact original-space rerank |
| Manifold/residual | Transform-specific retrieval | Explicit latent candidate index + residual/novelty path |

A binary LSH index is not a continuous-HDC index. Cosine similarity is not automatically the correct metric for every continuous HDC state. The actual algebra contract must choose the operator.

### Plane C — append-only update and consolidation plane

Proposed lifecycle:
1. assign a stable memory item/update ID;
2. validate representation profile, dimensions and payload length;
3. attach source/event, provenance, rights and transform references when applicable;
4. append a checksummed immutable update record;
5. acknowledge only after the declared durability level is satisfied;
6. asynchronously build or update immutable index segments;
7. verify new segment checksums and manifest references;
8. atomically publish the new manifest generation;
9. retain the previous verified generation until the replacement is known to be usable;
10. compact old segments/deltas under a bounded background budget.

A failed append must not advance the durable update watermark. A failed compaction must not replace the last valid manifest. Duplicate updates must be idempotent under a stable update ID.

This should begin with an embedded storage primitive and immutable blob/segment files, not a custom WAL/transaction implementation.

### Plane D — recovery, continuity and invalidation

The recovery plane should qualify:
- manifest checksums and generation;
- segment identity, digest, format and representation;
- last acknowledged update sequence;
- restart/replay idempotence;
- missing/corrupt segment behavior;
- snapshot/state compatibility;
- rollback to the last validated manifest;
- exact versus approximate conversion lineage;
- invalidation following source/version/rights changes.

Unknown, incomplete and corrupt states must be explicit. They must never become empty memory, a valid empty result, or a silently accepted stale state.

## 6. The most important paradigm shift: semantic width ≠ evolution width

A 256K-dimensional semantic representation does not necessarily imply evolving 256K f32 coordinates on every cognitive cycle.

Use two independent axes:

[
D_s = 	ext{semantic representation width}
]

[
D_e = 	ext{local dynamical evolution width}
]

Candidate semantic widths:
- 16K, 32K, 64K, 128K, 256K.

Candidate local evolution widths:
- 1K, 4K, 16K, 32K.

Candidate architecture:
- high-dimensional semantic addressing and associative memory;
- lower-dimensional fast local liquid state where task evidence permits;
- explicit lift/projection into semantic space;
- explicit residual or novelty pathway that preserves information not represented by the compact state;
- retrieval and reasoning remain bound to the canonical semantic/source representations.

Required controls:
1. full-width liquid evolution;
2. direct low-width evolution;
3. matched-width fixed-seed random projection;
4. deterministic orthogonal/nested projection where mathematically applicable;
5. manifold projection with explicit residual;
6. learned task-specific projection only after non-learned controls.

No hidden full-width recurrent computation or full-vector residual transfer on every step is allowed in a run claiming compressed dynamical operation.

Primary measurements:
- task utility;
- trajectory divergence;
- tau/gate/equilibrium error;
- reconstruction error;
- novelty/residual recall;
- bytes moved per step;
- projection/lift overhead;
- resident and peak temporary memory;
- p50/p95/p99 update latency;
- adaptation and snapshot continuity.

This work belongs to #7749 and #3348. HME consumes the resulting representation and transition contracts; it does not build another projection engine.

## 7. Storage-engine decision: custom HME, not custom DBMS

The first design should use an existing proven storage primitive for durable metadata/deltas and add a narrow HDC-aware segment/index layer.

### Candidate storage engines

| Candidate | Strengths | Role in HME evaluation | Main caution |
|---|---|---|---|
| SQLite/rusqlite | Already integrated; mature transactional persistence; local/offline | Baseline and metadata/checkpoint store | Blob-heavy random access and high write rates need measurement |
| Fjall | Embeddable Rust LSM; range/prefix access, keyspaces, snapshots/transactions depending on version | Append-heavy delta and immutable segment manifest candidate | LSM compaction and operational behavior must be tested |
| redb | Embedded Rust copy-on-write B+Tree | Metadata/manifests and read-snapshot candidate | Large blob/vector workload must be measured |
| RocksDB | Mature LSM architecture, WAL/memtable/SST/compaction | Strong storage-engine reference | Native dependency, configuration surface and operational complexity |
| Custom HME segment format | HDC-specific layout, dimension tags, exact SIMD scan and HDC index | The actual custom engineering differentiator | Must not reinvent transaction/recovery primitives without evidence |

Fjall documents itself as an embeddable Rust LSM key-value engine with range/prefix search, keyspaces, optional serializable transactions, blob separation and background maintenance. <Link url="https://docs.rs/fjall/latest/fjall/" title="Fjall documentation"/>. Its documented scope is a storage engine, not a relational DBMS. <Cite refs={["turn269309search2","turn269309search8"]}/>

RocksDB's architecture provides a mature reference for WAL, memtable, sorted immutable files and compaction. <Link url="https://github.com/facebook/rocksdb/wiki/RocksDB-Overview" title="RocksDB architecture"/>. redb is a Rust embedded ACID B+Tree reference. <Link url="https://docs.rs/redb/latest/redb/" title="redb documentation"/>.

### HME segment format — proposed, not implemented

Each immutable segment should have:
- magic and format version;
- segment ID and generation;
- representation family and exact dimension;
- scalar datatype/packing/endianness;
- declared distance/algebra profile;
- item count and payload byte length;
- stable item IDs;
- offset table or compact fixed/variable record directory;
- payload digest/checksum;
- metadata/lineage sidecar reference;
- optional index family and index-version metadata;
- build/transform identity and source frontier reference.

Payloads should be stored without an implicit dimension conversion. Any quantized, projected, manifold or approximate representation must carry an explicit transform ID, parent representation digest and error/quality contract.

The first implementation should prefer immutable segment files plus a small manifest and append-only delta, using an established embedded engine for manifest and update durability. A bespoke page/WAL engine is not a P0 deliverable.

### Query execution plan

1. validate the query representation and dimension;
2. bind a snapshot/manifest generation;
3. resolve hard constraints and access eligibility;
4. select candidate segments by representation/dimension and metadata filters;
5. choose exact scan or qualified candidate index using corpus size and workload;
6. generate candidate IDs;
7. fetch original representation bytes;
8. exact rerank using the canonical HDC similarity function;
9. return ranked IDs, exact score, representation profile and evidence references;
10. record the manifest/generation/segment frontier used by the query.

Never rank vectors of incompatible representation families or dimensions. A similarity index cannot certify identity, truth, corroboration, provenance or rights.

## 8. External vector databases: challengers, not the holographic brain

HME should compare external products, but they solve a different problem: vector candidate retrieval.

- **Qdrant:** documents dense vectors up to 65,535 dimensions, so it can technically store 256K? No: its documented maximum is 65,535, meaning it covers 16K/32K/64K but not 128K/256K. This makes it a candidate for selected 16K–64K continuous representations, not a 256K backend. <Cite refs={["turn269309search0","turn269309search6"]}/>
- **pgvector:** its HNSW index supports `vector` up to 2,000 dimensions, `halfvec` up to 4,000 and `bit` up to 64,000; binary HDC may fit only when the representation genuinely is binary and Hamming/Jaccard is the intended algebra. Continuous 16K–256K HDC does not fit ordinary pgvector ANN HNSW as a direct dense vector. <Cite refs={["turn269309search3","turn269309search5"]}/>
- **Vespa:** a potential advanced hybrid/ranking system, but not a persistence substitute for live liquid state. Benchmark only if lexical/vector/tensor fusion and multi-phase ranking are required. <Link url="https://docs.vespa.ai/" title="Vespa documentation"/>.
- **SQLite:** remains the on-node persistence baseline; HME should demonstrate why an additional store is worth its maintenance burden.
- **DuckDB + Parquet:** remains offline analysis/evaluation, not the live associative-memory authority.

Qdrant also requires careful payload-index design for filterable search; its own documentation recommends creating the relevant payload indexes before ingestion so the search planner can make use of them. <Cite refs={["turn269309search4","turn269309search9"]}/>

An external vector DB does not own the live neuron state, learned parameter continuity, or exact checkpoint/replay contract. It may be a useful candidate index for some long-term memories.

## 9. Capacity: dimension alone is not memory capacity

HDC capacity must be measured against:
- vector dimension;
- number of bound/bundled associations;
- fan-in and correlated/repeated properties;
- distractor population and semantic similarity;
- number of queries/comparisons;
- cleanup/decoding strategy;
- representation drift or corruption.

Single-query false-match probability and all-pairs false-match probability are not the same application risk. If (p) is per-comparison false-match probability and (M) comparisons are made, a conservative union-bound accounting is (P(	ext{any false match}) le M p). This is an accounting bound, not a measured rare-event rate.

Reuse #7772's deterministic generator, held-out thresholds, correlated distractors and query-multiplicity accounting. Do not invent a second capacity oracle.

## 10. Frozen benchmark matrix

### 10.1 Resolution and data representation

| Dimension | Continuous-f32 bytes/HV | Packed-binary bytes/HV | Status in this research |
|---:|---:|---:|---|
| 2K | 8 KiB | 256 B | Candidate |
| 4K | 16 KiB | 512 B | Candidate |
| 8K | 32 KiB | 1 KiB | Candidate |
| 16K | 64 KiB | 2 KiB | Canonical baseline |
| 32K | 128 KiB | 4 KiB | Canonical comparison |
| 64K | 256 KiB | 8 KiB | Canonical upper tier |
| 128K | 512 KiB | 16 KiB | Exploratory |
| 256K | 1 MiB | 32 KiB | Exploratory |

These are logical payload sizes only.

### 10.2 Associative workload scale

- stored items: 64, 256, 1K, 4K, 16K, 64K, 256K; 1M where feasible;
- fan-in: controlled odd cardinalities plus task-specific fan-in;
- query multiplicity: 1, batches 4/16/64, all-pairs and clustering;
- positive and negative retrievals;
- correlated, repeated and adversarially similar distractors;
- read-only, insert, mixed update and rights-revocation lifecycle.

### 10.3 Dynamical trajectory workload

- full-width 16K/32K/64K/128K/256K;
- lower evolution widths 1K/4K/16K/32K;
- fixed and irregular dt;
- long horizons, abrupt inputs, recovery;
- high→low, low→high and high→low→high conversions;
- online adaptation and momentum preservation;
- snapshot/restore after transitions.

### 10.4 Storage and recovery workload

- cold start and warm resident;
- append, segment build, manifest publish, compaction;
- process crash at each durable boundary;
- torn/corrupt manifest and segment;
- duplicate/out-of-order update;
- snapshot restore and replay;
- rights revocation and descendant invalidation;
- local disk, mmap/NVMe, external vector index.

## 11. Metrics and SOTA/industry comparison

Do not collapse correctness, retrieval, dynamics and cost into one score.

| Dimension | Required metric |
|---|---|
| Exact semantics | 100% exact result agreement with independent HDC oracle |
| Associative capacity | Retrieval accuracy, target/distractor margin, false-match rate by fan-in and query count |
| Dynamic fidelity | State error, tau/gate error, trajectory divergence, irregular-dt error and hysteresis |
| Memory efficiency | Bytes/item, resident set, peak temporary bytes, allocations and bytes moved |
| Retrieval performance | p50/p95/p99 latency, throughput, exact/ANN recall and eligible-candidate recall |
| Update performance | Append throughput, materialization/compaction lag, index rebuild time |
| Durability | Acknowledged-update recovery, checksum detection, restart/replay parity |
| Energy | Joules per correct retrieval/update, only with a declared direct measurement boundary |
| Operational | Setup, upgrades, recovery time, dependency burden and maintenance complexity |

### Baselines to reproduce

- current exact in-memory HDC scan;
- current binary LSH and SIMD path;
- current SQLite persistence and episodic memory;
- Fjall/redb/RocksDB as embedded storage primitives;
- Qdrant/pgvector/Vespa only for dimension- and metric-compatible vector workloads;
- full-width vs semantic/evolution-width separation under #7749;
- hardware profiles from #7864/#10901/#10906 only with measured end-to-end transfers and workload quality.

### Proposed project gates (not achieved)

- 100% deterministic oracle agreement on frozen exact fixtures;
- zero silent corruption or acknowledged-update loss in the declared crash model;
- zero hidden state resets during dimension transitions;
- zero post-revocation payload returns;
- zero unqualified resolution promotion;
- at equal task quality, target ≥1.5× end-to-end useful retrieval throughput **or** ≥30% lower directly measured joules per correct retrieval, with p95 no more than 10% worse for that declared workload.

These are proposed stretch gates and must be preregistered before candidate results.

## 12. Related SOTA and hardware research

A 2026 Nature Communications paper demonstrated HDC-related in-memory computing on an analogue memristive crossbar SoC for 21-language identification. The paper reports 95.24% accuracy on its physical system and a VMM-based representation/classification path at 1,024 dimensions. It is evidence that algorithm/hardware co-design can change the resource trade-off, but it does not demonstrate 256K dynamical HDC, Symthaea's liquid-state continuity, or a general-purpose memory engine. <Cite refs={["turn269309search1"]}/>

This motivates HME's near-data and storage-layout research, but the first HME milestones should be software/CPU exactness, local storage and replay. Hardware-specific approximate or analogue profiles remain separate.

## 13. Implementation sequence

1. Inventory the live cognitive loop's ownership of neuron state, episodic records, SQLite persistence, LSH, snapshot structures and serialization.
2. Reuse #7919's general memory architecture and #7749's resolution/task matrix. Do not create duplicate controllers or runners.
3. Generate deterministic memory-footprint and capacity fixtures using #7772 and exact byte formulas.
4. Prototype read-only exact host search over immutable resolution-homogeneous segments; compare against the current in-memory/SQLite baseline.
5. Compare SQLite, Fjall, redb and RocksDB as persistence primitives under identical append/snapshot/recovery workloads.
6. Add append-only delta and atomic manifest generation after the segment format and crash model are reviewed.
7. Integrate explicit state/parameter/snapshot conversion metadata from #6560/#6561; do not implement a hidden resize/reset path.
8. Add external vector candidates only where representation dimensions and distance semantics match.
9. Connect to FPGA/memristive work only after software exactness and end-to-end locality evidence.
10. Promote only the smallest architecture that clears correctness, recovery, task-quality and Pareto gates.

## 14. Risks and nonclaims

- A valid 256K representation is not evidence of useful capacity or a better trajectory.
- An in-memory exact scan may outperform an ANN index for small, hot working sets.
- A custom HME may create less value than improving current SQLite/segment layout; that is an acceptable negative result.
- The live liquid state must never depend on database availability.
- Static vector similarity cannot qualify dynamic trajectory conversion.
- An external vector index is not an authority store or proof of source evidence.
- No custom engine is implemented by this RFC.
- No local performance, compilation, test, hardware, energy or SOTA result is claimed.
- No consciousness claim follows from an HME implementation or a large-dimensional holographic representation.

The programme succeeds if it identifies the smallest reproducible memory substrate that improves Symthaea's actual associative/dynamical workloads without weakening state continuity, exactness, recovery or evidence boundaries.

## 8A. Exact similarity semantics must be build- and stride-invariant

A source audit identified a correctness hazard in the current `ContinuousHV::similarity()` API:

- With the `simd` feature, it dispatches to `similarity_simd()` over the full vector.
- Without the `simd` feature, it reads the global cognitive `STRIDE`; when stride is not 1, it accumulates dot products and norms only at indices selected by `step_by(stride)`.

Therefore, an API named `similarity()` is not necessarily a full-dimensional exact oracle in every build/runtime configuration. The HME must not silently use the adaptive/stride path to qualify exact stored-memory retrieval.

The exact query contract must use a scalar full-dimension reference over every coordinate, with independently checked SIMD kernels as implementations of that same contract. A stride-based or otherwise sampled similarity path may remain useful for adaptive cognitive operation, but it must be explicitly typed as approximate, have its own quality/recall curve, and must never emit an Exact qualification status. Exact retrieval semantics must not depend on a global runtime stride or compile-time feature selection.

Also reuse [#7955 — metadata-first scans and payload cache residency](https://github.com/Luminous-Dynamics/symthaea/issues/7955), plus #7934/#7922. HME should compare metadata-first SoA and interleaved layouts using the existing cold-open methodology rather than create a duplicate layout/residency experiment.
### Related HDC core contract

Exact/stride-sampled similarity semantics are tracked separately in [#10918 — HDC-SEMANTICS-002](https://github.com/Luminous-Dynamics/symthaea/issues/10918). HME must consume the canonical full-dimension exact reference and must not create a competing similarity implementation. The existing stride-based path is treated as an explicitly approximate/adaptive mode.

## 15. Industry benchmark crosswalk and progressive exact retrieval

The companion [HME industry benchmark crosswalk](HME_INDUSTRY_BENCHMARK_CROSSWALK_2026-11-07.md) maps HME claims to three separate benchmark layers: HDC representation/dynamics, agent-memory utility/governance, and storage/retrieval systems. It reuses #8427, #9145, #9075, #9177, #9139, #7922, #7934, #7955, #7864/#7866 and #10896; it does not create a competing benchmark runner.

A new research child, [#10945 — HME-EXACT-PRUNE-001](https://github.com/Luminous-Dynamics/symthaea/issues/10945), investigates progressive exact scans using admissible partial-score bounds. The goal is to stop reading a candidate only after proving it cannot enter top-k, while preserving full-scan oracle IDs, ordering and scores. This is distinct from ANN: all candidates remain logically accounted for, and approximate indexes may reorder evaluation but may not omit candidates. The Binary/Hamming and Continuous/cosine bounds have separate proof and numeric-safety requirements. No speedup is claimed; a full scan remains the fallback and canonical oracle.


## 16. Bounded query execution, explicit approximation profiles and current research children

Three further child programmes extend HME without creating a new manager or benchmark system:

- [#10945 — progressive exact scans with admissible early-abandon bounds](https://github.com/Luminous-Dynamics/symthaea/issues/10945). Full candidate accounting; prune only by a proven conservative bound; optional proof-carrying receipts.
- [#10973 — bounded query execution and truthful completion states](https://github.com/Luminous-Dynamics/symthaea/issues/10973). Separates ExactComplete, ApproximateComplete, Incomplete, BudgetExhausted and UnavailableOrInvalid. A best-so-far result is not a complete top-k. Composes #5471 value-of-computation and #9177 proof-packet budgets.
- [#10974 — phase-balanced, uncertainty-aware approximate similarity](https://github.com/Luminous-Dynamics/symthaea/issues/10974). Characterizes fixed-origin `step_by(stride)` aliasing and evaluates seeded stratified / phase-rotating / adaptive-refinement profiles. Legacy cognition throttling remains unchanged until qualified migration.

Two deterministic machine-readable fixtures are checked into this branch:
- `docs/research/fixtures/hme_query_sampling_and_completion_v0.json`
- `docs/research/fixtures/hme_progressive_exact_bounds_v0.json`

The small sampling fixture contains counterexamples where fixed-origin stride 4 maps full cosine 1.0 to its zero-norm fallback 0.0, and maps full cosine 0.0 to sampled cosine 1.0. The progressive-bound fixture checks 19,200 small continuous real-arithmetic prefix cases and 768 four-bit binary cases; no mathematical under-bound was observed within the declared enumeration tolerance. It also demonstrates an ordinary binary64 evaluation of the ContinuousHV upper-bound expression returning 0.4999999999999999 when the true cosine is 0.5, sufficient to falsely prune at a 0.49999999999999994 threshold. This is a concrete reason not to treat an ordinary floating-point value as a conservative bound.

These are small synthetic contract fixtures and independent arithmetic sanity checks, not Rust tests, production floating-point proofs, benchmark runs or performance claims.

## 17. Similarity numerical contract details

The source audit behind [#10918](https://github.com/Luminous-Dynamics/symthaea/issues/10918) additionally found:
- Full-coordinate coverage is not bitwise-deterministic scoring: AVX2+FMA, scalar fallback and unrolled scalar paths use different accumulation orders.
- `ContinuousHV::from_vec` accepts arbitrary f32 values even though the representation is described as [-1,1]; NaN/Inf/out-of-profile and overflow semantics must be explicit at HME segment admission.
- The current cosine implementation returns 0.0 when the product-norm denominator is below 1e-10; a future checked profile must version any changed near-zero behavior.
- Higher-level helpers may sanitize non-finite results while `ContinuousHV::similarity` itself does not; HME must not inherit caller-specific silent sanitization.

HME's exact score profile must bind full-coordinate coverage, accumulation semantics, numeric domain, zero/near-zero behavior, non-finite rejection and tie ordering. A similarity tolerance is not permission to change top-k ordering near ties.


## 18. Certified compressed exact retrieval

New child: [#10978 — HME-CERT-COMPRESS-001](https://github.com/Luminous-Dynamics/symthaea/issues/10978).

Explore a two-tier ContinuousHV representation:
- compact approximate code + certified residual/error envelope in the fast tier;
- original exact ContinuousHV payload in a colder immutable segment.

For x = x_hat + r with a verified ||r||₂ ≤ ε, Cauchy–Schwarz yields a conservative real-arithmetic upper bound for q·x. With a certified interval for ||x||, derive a cosine upper interval; if the norm lower bound is zero or numeric uncertainty overlaps the kth threshold, load the original payload rather than prune.

The intended first benefit is lower hot-memory residency and fewer cold-payload reads, not necessarily lower total bytes. Every exact result still reranks survivors from original payloads. This must remain distinct from ANN and from lossy quantized output.

The mathematical formula is not a floating-point certificate: outward rounding, metadata provenance, transform/codebook digest and independent validation of the residual envelope are mandatory. A checksum alone cannot prove that epsilon is conservative.


## 19. Machine-readable bounded-query receipt

The HME branch now includes:
- `docs/research/schemas/hme_query_receipt_v0.schema.json` — structural JSON Schema for a versioned query receipt.
- `docs/research/fixtures/hme_query_receipt_states_v0.json` — full synthetic receipts for ExactComplete (full scan and safe-pruned), ApproximateComplete, Incomplete/BudgetExhausted and UnavailableOrInvalid.

The schema validates structure and status-dependent fields. It does not prove candidate-count arithmetic, manifest consistency, residual bounds or pruning validity. An independent semantic verifier must reconcile counts against the pinned generation and validate any proof/certificate claims before an ExactComplete or mathematical-proof status can be trusted.


A companion [blockwise residual fixture](fixtures/hme_certified_compression_block_bounds_v0.json) illustrates why per-block certificates may be tighter than a single global residual radius. In its 4D synthetic example, the query-weighted block residual dot bound is 2.0 versus 10.1 under the global bound; the corresponding real-arithmetic cosine upper bounds are approximately 0.01048 and 0.05291. This is a geometry sanity check only; it does not validate a production quantizer or floating-point implementation.


## 20. HDC-LTC checkpoint completeness

New child: [#10979 — HME-DYN-CHECKPOINT-001](https://github.com/Luminous-Dynamics/symthaea/issues/10979).

The current HdcLtcUnifiedNetwork has a separate NetworkStateSnapshot intentionally limited to dynamic state, layer outputs and evolution clocks; its own documentation says weights and momentum are excluded. This is useful for in-memory evolution restore but cannot, alone, resume a trained learner. HME therefore separates EvolutionSnapshot, LearnerCheckpoint, ReplayCheckpoint and WeightsOnlyArtifact, and requires training-continuation parity after restore.


A machine-readable source-derived field matrix is available at `docs/research/fixtures/hme_dyn_checkpoint_field_matrix_v0.json`. It distinguishes the actual NetworkStateSnapshot contents from the full neuron/network fields required by LearnerCheckpoint and ReplayCheckpoint. It is a field inventory, not a stable wire schema or executed persistence test.


## 21. HDC-LTC bridge projection co-design

New child: [#10981 — HLB-PROJ-CO-DESIGN-001](https://github.com/Luminous-Dynamics/symthaea/issues/10981).

HdcLtcBridge retains dense input/output projection matrices. For 256-wide input/output at 256K dimensions, they represent 512 MiB of logical f32 payload. The output projection's training update is an outer product, hence a rank-one algebraic delta per training step. The new experiment compares current dense matrices, seeded regeneration, structured fast transforms, exact rank-one factor logs and explicitly bounded factor consolidation.

The analytical fixture `docs/research/fixtures/hme_bridge_projection_footprint_v0.json` includes dense footprint, MAC counts and exact factor storage at several D/rank values. It does not report RSS or measured performance.


The field matrix fixture now also inventories all 12 HdcLtcBridge fields. HdcLtcBridge does not derive Serialize/Deserialize and contains trained output-projection state plus adaptive-dimension counters and seed lineage. The complete learner checkpoint must wrap this bridge-level state around the inner-network checkpoint; scratch buffers are reconstructed empty.


## 22. Structured bridge projections: explicit Hadamard/FWHT comparison

Direct prior art: Salahudeen, Maalouf and Elfadel, “Deterministic orthogonal projections for hyperdimensional regression: A Hadamard submatrix approach,” *Intelligent Systems with Applications* 31 (September 2026), DOI 10.1016/j.iswa.2026.200707. The publisher abstract reports up to 9× lower selected hyperspace dimensions and up to 19× wall-clock speedup on six regression datasets, while also describing mixed, task-dependent predictive performance, strongest results with a closed-form solver and no universal compression. This is published prior art, not a Symthaea result. See #10981 for the reproduction and task-transfer controls.

A concrete fast-transform hypothesis for HdcLtcBridge is:
- Input expansion 256→D: pad into D, apply a versioned sign/permutation, then normalized FWHT. This creates a compact orthogonal expansion without retaining a dense D×256 matrix; the transform remains a changed representation profile.
- Output reduction D→256: independently seeded sign/permutation + FWHT, then select 256 coordinates (or compare a Hadamard-submatrix readout). A structured fixed base can be paired with the exact rank-one trainable adapter tracked in #10981.

The FWHT arithmetic model is O(D log D), compared with the dense O(256D) projection. The transform is not assumed equivalent to the existing random dense map, and static regression improvement does not imply HDC-LTC trajectory fidelity. Include random/structured/adversarial inputs, irregular dt, sequence prediction, online learning, and high→low→high transitions. The transform seed, sign/permutation generation, normalization, selected coordinate indices and implementation version are part of the representation/checkpoint identity.
