# NEURO-SILICON-003A: Fused Exact HDC Retrieval Tile v0.1

**Status:** architecture/RTL proposal plus an analytical lower-bound model; not yet simulated, synthesized, board-measured, or fabricated.  
**Tracking:** [issue #7908](https://github.com/Luminous-Dynamics/symthaea/issues/7908), child of [end-to-end retrieval benchmark #7864](https://github.com/Luminous-Dynamics/symthaea/issues/7864).  
**RTL draft:** `crates/domains/symthaea-silicon/rtl/hdc_nearest_distance_reduce.sv`  
**Directed testbench draft:** `crates/domains/symthaea-silicon/rtl/tests/hdc_nearest_distance_reduce_tb.sv`  
**Model:** `crates/domains/symthaea-silicon/schemas/synthetic_retrieval_roofline_v0_1.json`

## 1. The chip we should try to build first

Do not begin with a custom general-purpose CPU or GPU. Symthaea has a more bounded accelerator target: **an exact, memory-local hypervector retrieval engine**. It scans a persistent store of packed `BinaryHV` vectors, computes Hamming distances, reduces candidates locally, and returns a small record rather than streaming every distance back to the host.

The initial vertical slice is deliberately smaller than the whole tile: a streaming nearest-distance reducer. It accepts a candidate stream `(distance, stable_item_id, last)`, chooses minimum distance, breaks equal-distance ties by the smallest stable item ID, and emits exactly one result at the end of a non-empty query. That piece can be verified independently while the existing XOR and Hamming streaming work continues.

The proposed physical hierarchy is:

```text
Host / Symthaea Rust oracle
  -> query encoder (existing semantic owner)
  -> query FIFO + DMA / bus interface
  -> exact 16,384-bit query buffer
  -> banked vector-store reader
  -> XOR -> per-lane popcount -> exact accumulator
  -> deterministic nearest-match reducer
  -> result FIFO (ID + distance; not all N distances)
  -> host adapter / receipt
```

The query encoding and BinaryHV semantics remain owned by the canonical Rust code. This RTL does not duplicate them. The accelerator starts at the already-encoded, source-bound query and ends at the exact stable-ID result. A later integrated benchmark adds query encoding and workload setup if that is the product boundary being claimed.

## 2. Why this architecture is worth testing

At 16,384 bits, one packed vector occupies **2,048 bytes (2 KiB)**.

| Store size | Vector data only | Practical first question |
|---:|---:|---|
| 64 | 128 KiB | Can a small on-chip bank prove the full streaming path? |
| 256 | 512 KiB | Does local residency beat an optimized CPU/SIMD scan? |
| 1,024 | 2 MiB | Which memory tier becomes the bottleneck? |
| 4,096 | 8 MiB | Does external-memory streaming dominate arithmetic? |
| 16,384 | 32 MiB | Does the design need hierarchy, filtering, or a different accelerator boundary? |

Those are capacity calculations, not measured FPGA capacities. Metadata, alignment, ECC and double buffering add to the actual memory footprint. Do not assume the complete codebook fits in BRAM or ASIC SRAM.

A 15-bit unsigned distance can represent every value from 0 through 16,384 inclusive. The reducer assumes the upstream exact Hamming engine supplies only values in that legal range; the RTL integration stage must preserve that contract and independently test boundary values.

## 3. Paradigm shifts

### A. From faster primitives to less data movement

A high-throughput popcount unit is not the product. The product is a correct retrieval result delivered at lower end-to-end cost. Keep the codebook resident where possible, move the query to it, and perform the final minimum/top-k reduction close to the datapath. Count cold codebook loading separately from warm query service.

### B. Treat memory bandwidth as part of the compute architecture

Choose lane count and width together with the physical read ports/banks that feed them. If the memory cannot supply each lane every cycle, extra arithmetic lanes become idle silicon. A wide datapath can lose to a narrower one once BRAM packing, routing, arbitration and clock closure are accounted for.

### C. Optimize work per byte, not just parallel arithmetic

Test batched queries explicitly. When the same resident codebook serves several queries, its bytes can be reused across multiple distance calculations. The sweep must report per-query latency and batch throughput separately; batching that improves mean throughput but violates tail-latency constraints is not an unconditional win.

### D. Fuse reduction, not semantics

The output should be an ID and distance (or later a bounded top-k list), not N distances that the host must sort again. Fusing the reduction removes traffic without changing exact Hamming algebra. The tie policy must remain explicit and identical across Rust, simulator and hardware.

### E. Let conventional silicon win

For small stores, CPU/SIMD may be faster and simpler. The FPGA candidate should only be retained where a frozen workload demonstrates a useful Pareto improvement. Do not assume a custom accelerator deserves a place in the product.

### F. Keep approximate computing on a separate branch of the evidence tree

Approximate HDC, reduced precision, analog or memristive arrays can become later candidates only after representation identity, per-task quality floors, drift and recalibration requirements are versioned. They must not alter the exact digital baseline silently.

## 4. Existing work to compose — not duplicate

- [#5804](https://github.com/Luminous-Dynamics/symthaea/issues/5804) and [#5805](https://github.com/Luminous-Dynamics/symthaea/issues/5805): workload ownership and conventional CPU/GPU/FPGA comparisons.
- [#5808](https://github.com/Luminous-Dynamics/symthaea/issues/5808) and [#5813](https://github.com/Luminous-Dynamics/symthaea/pull/5813): canonical semantics and frozen Rust-generated oracle corpus.
- [#7713](https://github.com/Luminous-Dynamics/symthaea/issues/7713), [#7728](https://github.com/Luminous-Dynamics/symthaea/issues/7728), [#7731](https://github.com/Luminous-Dynamics/symthaea/issues/7731), [#7718](https://github.com/Luminous-Dynamics/symthaea/pull/7718), and [#7734](https://github.com/Luminous-Dynamics/symthaea/pull/7734): XOR, streaming lanes and Hamming differential work. Their existence is not a pass claim; the issue train itself notes the skipped-run limitation.
- [#5818](https://github.com/Luminous-Dynamics/symthaea/issues/5818): source-bound HDL simulation, formal/equivalence and synthesis qualification.
- [#5800](https://github.com/Luminous-Dynamics/symthaea/issues/5800): FPGA implementation/board evidence through physical design and external fabrication.
- [#7864](https://github.com/Luminous-Dynamics/symthaea/issues/7864) and [#7866](https://github.com/Luminous-Dynamics/symthaea/pull/7866): end-to-end workload, transfer and host-baseline measurement boundary.

This work adds the top-1 reducer and its model. It does not establish that pending XOR/Hamming RTL is qualified, and does not start another generic HDL generator.

## 5. Reducer interface contract

The RTL draft uses a ready/valid stream:

- `s_valid_i && s_ready_o`: one candidate is accepted.
- `s_last_i`: the accepted candidate closes the current non-empty query.
- `m_valid_o && m_ready_i`: the one-result record is consumed.
- Comparison order: `(distance ascending, stable_item_id ascending)`.
- While `m_valid_o && !m_ready_i`, output valid and both payload fields must remain stable; the input is backpressured.
- The module serializes queries and may insert a bubble after the result handshake. Do not hide this behavior in the benchmark; optimize it only if the measured workload justifies higher-throughput framing.
- The enclosing stream protocol is responsible for rejecting missing/early/invalid framing. A query that never presents `last` must not be reported as complete.

Directed testbench coverage includes non-sorted candidates, repeated minimum distances, smallest-ID tie-breaking, a single-candidate maximum distance of 16,384, a zero-distance tie, output backpressure and a timeout. These are **authored tests, not executed tests** in this change.

## 6. Analytical lower-bound model (not a synthetic hardware result)

Machine-readable screening scenarios are in `synthetic_retrieval_roofline_v0_1.json`. It computes:

`ideal_cycles = ceil(16,384 / lane_width_bits) * ceil(store_vectors / parallel_distance_lanes)`

and reports the cycle-model latency at a **hypothetical 100 MHz**. It assumes one accepted beat per lane per cycle, zero memory latency/stalls, continuous feed to every lane, and enough independent memory banks/ports. Those assumptions form an optimistic lower bound, not a predicted or measured implementation.

Example: a 1,024-vector codebook occupies 2 MiB. With a 512-bit datapath, there are 32 beats per vector:

| Parallel distance lanes | Ideal cycles/query | Ideal latency at 100 MHz |
|---:|---:|---:|
| 1 | 32,768 | 327.68 µs |
| 4 | 8,192 | 81.92 µs |
| 8 | 4,096 | 40.96 µs |

At an explicitly hypothetical decimal 8 GB/s, transferring the entire 2 MiB codebook takes an ideal 262.144 µs, while a 2 KiB query takes 0.256 µs. Real protocol/driver overhead and memory timing are excluded. An eight-lane 512-bit engine accepting one beat per lane every 10 ns would require **51.2 GB/s** of aggregate datapath feed; four lanes require 25.6 GB/s. That bandwidth demand shows why the lane-count sweep must include memory banking/capacity rather than extrapolating arithmetic throughput alone.

Interpretation: if the accelerator reloads the full store for each query, transfer can erase the benefit; a persistent codebook radically changes that comparison, but incurs build/load/residency costs. If multiple queries reuse a codebook, measure batching and amortization rather than assuming the warm path describes all deployments.

The analytical JSON is intended to prune the search space and choose measurements. It includes **no** empirical speedup, power, energy, FPGA resource count or achievable frequency.

## 7. Validation and promotion ladder

1. **Source review:** exact interface and parameter widths reviewed; independent expected outputs pinned.
2. **Simulation:** run the authored testbench with a named simulator/version and capture the exact source SHA, command, exit code, stdout/stderr hashes, warnings and test corpus identity.
3. **Independent differential/property tests:** compare reduction against a software minimum over deterministic permutations and random fixtures; test tie order, one/zero/max values, reset, stalls and consecutive queries. The software oracle must not call code generated from the RTL.
4. **Formal checks where supported:** stability while stalled, no result without a closing accepted candidate, one output per query, output payload corresponds to an accepted candidate, and stable tie order.
5. **Synthesis:** use an established HDL/EDA flow. Record top, parameters, tool versions, constraints, warnings, LUT/FF/BRAM/DSP counts, timing reports and artifact hashes. A Yosys or vendor synthesis pass alone is not FPGA hardware evidence.
6. **Board observation:** pin board/FPGA identity, bitstream SHA-256, clocks, actual memory interface/topology, firmware/host revision and raw measurements. Separate on-device kernel time from end-to-end latency, but publish both.
7. **ASIC exploration:** only if the measured FPGA/resource profile supports it. A named mature-node flow may use open SKY130/GF180 libraries for a candidate study; OpenROAD documents prior tapeouts using these process ecosystems, while the open SKY130 PDK itself warns that it is an experimental preview and not intended for production settings. A design-flow result is not a fabricated/tested chip.

### Required baselines and outputs

- P0 canonical Rust CPU oracle and P1 optimized CPU/SIMD on the same frozen vectors, tie contract and query mix.
- P2 operation-by-operation offload with every copy/synchronization.
- P3 persistent-codebook accelerator returning only final result.
- P4 batched persistent-codebook mode, with batch-build and queueing cost included.
- GPU only on a real, comparable device; simulator/CPU fallback is not GPU evidence.

Report correctness, p50/p95/p99, single-query latency, batch throughput, bytes transferred/query, cold load, store build/update cost, actual resident memory tier, stalls, task quality, resource use and measured energy independently.

## 8. Proposed performance gates (not achieved)

- **Semantic gate:** 100% canonical result agreement across admitted fixtures; zero malformed/truncated frame acceptance and zero output duplication/loss/reordering.
- **Useful FPGA gate:** at equal task quality, >=1.5x end-to-end throughput **or** >=30% lower directly measured joules per correct query, with p95 latency no more than 10% worse for the declared workload. Preregister the scenario and measurement boundary before viewing candidate results.
- **Repeatability:** aim for at least 30 measured batches per configuration where practical; preserve raw observations and uncertainty. Very long endurance/durability tests need their own declared protocol.
- **Area/timing gate:** only counts/reports tied to exact target, tool/version and constraints. Do not make a universal LUT or MHz target before choosing an FPGA family.
- **ASIC gate:** measurable benefit over viable CPU/FPGA options must justify one-time engineering, memory macro and interface risk, external fabrication, packaging, test and lifecycle cost. ASIC timing must be qualified at declared corners; FPGA timing is not a proxy.
- **Negative results:** retain candidates that lose. If data movement dominates, change the memory system/algorithm; do not add lanes reflexively.

## 9. Prior art and research context

- [NysX, arXiv:2512.08089](https://arxiv.org/abs/2512.08089) reports an end-to-end HDC graph-classification accelerator on a Zynq UltraScale+ FPGA and emphasizes streaming, lookup behavior, local data movement and workload-level comparisons. Its reported speedup/energy figures are workload- and platform-specific; they are neither a direct Symthaea baseline nor a target that this design has achieved.
- [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) documents an RTL-to-GDS flow and prior SKY130/GF180 tapeouts. That makes physical-design exploration a credible external tool path, not evidence this branch is ready for a tapeout.
- [SkyWater open PDK](https://github.com/google/skywater-pdk) states that the published SKY130 PDK is an experimental preview and warns that it is not intended for production use. The first silicon candidate should therefore be a bounded research/test chip, not a production promise.

## 10. Explicit nonclaims

This change does not claim the RTL is syntactically/elaboration-qualified, simulated, formally proven, synthesized, timing-closed, FPGA-observed, ASIC-ready, taped out or fabricated. It does not demonstrate faster Symthaea retrieval, lower energy, higher task quality, neuromorphic advantage, or a physical frequency of 100 MHz. The only generated performance values are analytical outputs from the stated lower-bound formula.
