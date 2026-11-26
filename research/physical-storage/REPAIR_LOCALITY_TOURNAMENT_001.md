# Physical Storage Read/Repair Locality Tournament — v0.1

**Programme:** PHYSICAL-STORAGE-REPAIR-LOCALITY-001  
**Tracking:** [Symthaea #13778](https://github.com/Luminous-Dynamics/symthaea/issues/13778)  
**Classical ECC baselines:** [Symthaea #13750](https://github.com/Luminous-Dynamics/symthaea/issues/13750)  
**Spatial layout:** [Symthaea #13662](https://github.com/Luminous-Dynamics/symthaea/issues/13662)  
**Bootstrap survivability:** [Symthaea #13774](https://github.com/Luminous-Dynamics/symthaea/issues/13774)  
**Physical reader/channel:** [Symtropy #7477](https://github.com/Luminous-Dynamics/symtropy/issues/7477) and [#7534](https://github.com/Luminous-Dynamics/symtropy/issues/7534)

## 1. Purpose

Exact recoverability is necessary but not sufficient for a useful preservation service. A code may recover a missing shard by reading a large portion of a stripe, moving a reader through many regions, revisiting calibration references, and keeping the archive in a degraded state for a long time.

The benchmark measures **repair work per independently verified byte restored**, while preserving code correctness, full-fault tolerance, total storage overhead and bootstrap validity.

This protocol separates three concepts:

- **Code guarantee:** whether a declared erasure pattern is mathematically recoverable.
- **Repair plan:** which available source fragments/regions a candidate chooses to read.
- **Physical access cost:** how those reads map to real bytes, optical fields of view, seeks, layer changes, energy and elapsed time.

A candidate may produce a low-cost repair plan only after an independent code oracle verifies that the selected helpers actually reconstruct the requested symbols. A cost-minimizing helper list is not proof of recoverability.

## 2. 2026 research baselines

### LESS — FAST '26

Cheng et al., "LESS is More for I/O-Efficient Repairs in Erasure-Coded Storage", USENIX FAST 2026, pp. 631–641. https://www.usenix.org/conference/fast26/presentation/cheng

LESS layers multiple extended sub-stripes formed from Reed–Solomon coding and exposes a configurable trade-off between accessed bytes and I/O seeks. Its HDFS evaluation reports improved single-block repair and full-node recovery over I/O-optimal erasure-code baselines. This is the closest published reference for the access/seek trade-off we need to benchmark.

### ROE — IEEE INFOCOM 2026

"Repair-Oriented Encoding for Erasure Codes with Localities", IEEE INFOCOM 2026, DOI 10.1109/INFOCOM59046.2026.11571631. https://doi.org/10.1109/INFOCOM59046.2026.11571631

ROE jointly optimizes cross-rack traffic during encoding and repair cost after encoding, using placement/locality-aware replication. Its reported measurements are for distributed storage clusters, not an optical archive.

### ZJC — Journal of Systems Architecture 2026

"ZJC: Constructing fully local repair in erasure codes for distributed cloud storage", DOI 10.1016/j.sysarc.2026.103735. https://doi.org/10.1016/j.sysarc.2026.103735

The paper introduces overlapping local parity groups with two recovery paths and describes two-helper local repair without global parity in its evaluated distributed-storage model. Its storage, helper and availability semantics must be reproduced before it can be compared to the physical-memory layout.

### HALO — Future Generation Computer Systems 2026

"HALO: A scalable framework for hotness-aware coding and transformation-efficient placement", DOI 10.1016/j.future.2026.108562. https://doi.org/10.1016/j.future.2026.108562

HALO combines hotness-aware redundancy, non-uniform LRC and graph-guided placement under skewed access traces. It is a useful candidate for a service compiler with different read-hotness classes; its HDFS/cluster results are not transferable physical measurements.

### Survey

"A Survey of the Past, Present, and Future of Erasure Coding for Storage Systems", ACM, DOI 10.1145/3708994. https://doi.org/10.1145/3708994

The survey organizes LRC, regenerating codes, piggybacking, XOR codes, sub-disk codes and algorithmic repair/read/update techniques. Use it to expand the comparator register, not as a performance baseline.

## 3. Core research hypothesis

The design space should jointly choose:

- code and minimum-distance guarantee;
- local-group structure and sub-stripe size;
- physical placement of data/local/global parity;
- hotness and requested-object profile;
- access topology and reader capabilities;
- spare/control/bootstrap placement;
- repair policy and background re-verification schedule.

The objective is not minimum bytes read at any cost. Report a Pareto frontier across:
- net payload efficiency;
- full-fault recovery guarantee;
- bytes accessed and physical read operations;
- seeks/FOV/layer transitions;
- latency and energy;
- degraded-state exposure;
- common-cause independence.

A candidate that improves single-failure repair by weakening two-failure protection must be shown as a different service contract, not ranked as a free improvement.

## 4. Candidate classes

Use separate comparison arms:
1. Custom MDS [6,4,3] with conventional full-stripe reconstruction.
2. Reed–Solomon with sub-stripe/layered repair (LESS-style).
3. LRC with disjoint local groups.
4. Overlapping local groups / two-path local repair (ZJC-style).
5. Regenerating or repair-bandwidth-efficient code.
6. Placement-aware repair topology (ROE-style).
7. Hotness-aware non-uniform LRC and placement (HALO-style).
8. Fixed row-major / spatial interleaver without repair optimizer.
9. Exhaustive optimal repair plan on finite tiny cases.

External code candidates require immutable source/version, license, code matrix, code rate, field/basis, sub-packetization, local-group assignment, supported fault sets, helper selection, known-answer vectors and build/toolchain pins. Paper labels alone are not enough.

## 5. Fairness and storage budget

Every arm gets identical payload bytes, request profile, physical grid, service requirements, total capacity budget, environmental assumptions, fault schedules and scorer.

Report the following denominators separately:
- user payload bytes;
- local parity bytes;
- global parity bytes;
- bootstrap/control bytes;
- spares and reserved tiles;
- replicated/cached content.

A local code may use more parity than MDS. Do not present its lower repair cost without accounting for additional storage overhead and its actual minimum distance. Compare at:
- equal net payload capacity;
- equal total parity/control overhead;
- equal full-fault tolerance;
where those conditions can be achieved. If a code cannot satisfy all three simultaneously, publish multiple matched strata and state the trade-off.

## 6. Media-specific cost models

### Abstract tile/region model

For every repair plan, report:
- requested bytes repaired;
- source bytes read;
- unique source tiles and regions visited;
- helper count;
- local/global parity sources accessed;
- layer/plane transitions;
- number of reader positioning operations;
- access amplification = bytes read / requested bytes repaired.

The v0.1 scorer handles only this abstract model and synthetic byte/region counters.

### Optical glass model (future)

A physical optical reader adapter may later add:
- field-of-view count and image pixels read;
- stage travel, focus/NA changes, registration work and layer switching;
- detector readout bytes, retries and re-calibration;
- end-to-end measured time and energy at a declared system boundary.

A 3D tile coordinate difference is not automatically a physical seek cost. It needs a qualified optical reader and motion/access model.

### HDD, tape, SSD and DNA

Use media-specific models:
- HDD: head seeks, rotational latency, track/zone behaviour and transfer.
- Tape: mount, locate/rewind, streaming, shoe-shining and library staging.
- SSD: page/block granularity, controller/FTL behaviour, read amplification and interface transfer.
- DNA: oligo selection, unique-molecule coverage, sequencing reads and targeted resequencing. Do not force these into a spatial-tile metric.

Never sum unlike access units into one unvalidated "seek" count.

## 7. Workload and fault matrix

- pristine full and selective reads;
- each single shard/tile loss;
- all two-erasure patterns inside the custom [6,4,3] bound;
- local-group member loss;
- local parity loss;
- two losses in one local group;
- simultaneous losses across two overlapping groups;
- adjacent region/cuboid and layer damage;
- whole stripe and whole layer loss;
- a second fault arriving while a first repair is in progress;
- hot object / cold archive read mixes;
- calibration/bootstrap loss (consume #13774, do not duplicate the bootstrap state machine);
- reader unavailable, registration drift and stale calibration;
- shared or unknown failure domains.

For every case, keep the candidate-visible observations separate from scorer-only original bytes/digests and fault ground truth.

## 8. Metrics

Report separately:
- exact requested bytes and independently accepted bytes;
- repair outcome, uncorrectable, wrong-codeword and abstention counts;
- bytes read / bytes repaired;
- unique helpers, regions and tiles read;
- read operations and physical seeks/FOV/layer transitions where defined;
- local vs global parity accessed;
- time-to-first-recovered-byte, p50/p95/p99 repair latency;
- energy and network transfer at declared boundaries;
- parity/local-group/bootstrap/spare overhead;
- full-fault tolerance after repair and degraded-state duration;
- second-fault exposure and common-cause cut sets.

No single weighted metric may hide failures in exact recovery, fault tolerance or evidence.

## 9. Anti-gaming and correctness

- The scorer owns the original payload and expected digest.
- The repair planner cannot score its own helper selection.
- The independent algebra oracle verifies all helpers can reconstruct the target for the declared erasure set.
- A syntactically valid but incorrect reconstructed codeword must fail digest acceptance.
- A partial object is not a completed repair until full object integrity and relevant manifest relations pass.
- The old copy is retained until destination bytes, metadata and capsule lineage pass independent verification.
- Failed/aborted/timeout repairs remain in denominators; no silent removal of adverse runs.
- Synthetic seeds are not physical samples; deterministic masks are not probabilities.

## 10. Prospective targets

Finite software targets:
- 100% exact byte reconstruction on supported positive cases.
- Zero false repair-success on wrong helper set, wrong code version, stale layout, corrupted metadata and digest-mismatch controls.
- Exact exhaustive-oracle agreement on all declared tiny instances.
- Deterministic replay from exact inputs, parameters and scorer version.
- Publish read amplification, helper count, access-operation count and redundancy overhead separately.

Research challenge targets after matched baselines:
- At equal net payload rate and full-fault tolerance, target at least 30% lower bytes read for a single-object/single-stripe repair than full-stripe RS.
- At equal bytes read, target at least 20% fewer physical positioning operations where a calibrated physical access model defines them.
- No reduction in two-erasure / declared compound-fault recoverability.
- Physical energy and latency targets only after calibrated reader/device baselines exist.

These percentages are hypotheses, not achievements or copied industry results.

## 11. Ownership and evidence ceiling

- Symthaea #13750 owns code/error-correction tournament.
- Symthaea #13662/#13726 own spatial shard placement and search.
- Symthaea #13774 owns bootstrap/control-plane loss semantics.
- Symtropy #7477/#7534 own the physical optical channel and measured reader observations.
- Symtropy #7482 owns service-contract compilation.
- Mycelix #5762 owns operational/common-cause dependency closure.

The v0.1 abstract model cannot establish physical seeking, actual microscope scan cost, glass/tape/HDD/SSD performance, production recovery, retention or SOTA superiority. Physical claims require instrument-bound measurements and matched, preregistered trials.
