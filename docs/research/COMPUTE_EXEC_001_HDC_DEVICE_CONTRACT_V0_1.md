# COMPUTE-EXEC-001 — HDC host/device execution contract v0.1

**Status:** proposed protocol and machine-readable schema; not an implemented driver or hardware qualification.  
**Owner:** Symthaea silicon integration.  
**Issue:** [#12953](https://github.com/Luminous-Dynamics/symthaea/issues/12953).

## 1. Executive decision

Make the accelerator boundary a versioned, capability-bound command protocol. The host submits *semantic work*, not raw register pokes or arbitrary pointers. The device receives only registered buffer handles and bounded byte ranges. Completion is accepted only for the exact command, device epoch and buffer generation that were submitted.

The first protocol profile admits three exact operations:

1. **HDC_BIND_XOR_EXACT** — XOR two canonical 16,384-bit BinaryHV vectors.
2. **HDC_HAMMING_EXACT** — return an exact integer Hamming distance in the inclusive range 0..16,384.
3. **HDC_TOP1_EXACT** — scan a declared candidate set and return the minimum Hamming distance, breaking ties by the lowest stable item ID.

No CfC, floating-point, approximate, analog, photonic, arbitrary-kernel or general DMA command is admitted in v0.1. Those require separate semantics and qualification.

### Why this is the next paradigm shift

Traditional accelerator interfaces expose a device-specific command queue and leave correctness, placement, memory lifetime, reset and fallback behavior scattered across drivers and applications. This contract makes those concerns explicit:

- **Semantic portability:** the same operation identity can be implemented by CPU, SIMD, Vulkan/WebGPU or FPGA without changing its mathematical meaning.
- **Data locality as part of the command:** the command names resident buffers and permitted ranges, not just an arithmetic kernel.
- **Failure is a first-class outcome:** reset, timeout and uncertain completion do not masquerade as successful work.
- **Qualification follows the artifact:** the operation, implementation, device profile, fixture and result remain linked.
- **Safe fallback:** exact CPU fallback is a declared execution path, not a silent algorithm substitution.

This is an integration contract for existing owners, not a new scheduler, GPU framework or evidence ontology.

## 2. Ownership and composition

| Concern | Canonical owner | This contract's relationship |
|---|---|---|
| BinaryHV algebra and oracle | #5808 / #5813 | Consumes exact semantics; does not redefine them |
| Conventional workload baselines | #5805 | Uses its frozen workload identities and boundaries |
| HDL simulation/formal/synthesis | #5818 | Supplies qualified implementation artifacts |
| FPGA-to-MPW evidence ladder | #5800 | Supplies stage/evidence distinctions |
| Semantic GPU execution | #6968 | Reuses operation identity and receipt concepts |
| Hamming/top-1 reducer RTL | #7908 / #7909 | Defines device-side result semantics |
| End-to-end retrieval/locality | #7864 / #7866 | Supplies benchmark boundary and promotion targets |
| Physical FPGA qualification | #10901 | Supplies exact board/SOM and board observation |
| RAM/persistence/recovery | #7919 | Owns durable storage and memory-tier policy |
| Cross-paradigm comparison | #9236 | Owns architecture portfolio and routing-regret comparisons |
| Carrier and module | #10886 / #10888 | Supplies board/module physical interface constraints |

No new HDC algebra, GPU Fabric, evidence ontology, persistence layer, DMA framework or board authority is introduced.

## 3. Canonical representation

The only representation admitted in v0.1 is:

| Property | Required value |
|---|---|
| Representation ID | luminous.binaryhv.bipolar.16384.v1 |
| Logical dimension | 16,384 bits |
| Payload length | 2,048 bytes |
| Packing | bit 0 is the least-significant bit of byte 0; byte sequence is canonical |
| Bind | bitwise XOR |
| Distance | popcount(A XOR B), integer 0..16,384 |
| Top-1 tie break | lowest stable item ID, independent of input ordering |

The canonical Rust BinaryHV implementation remains authoritative. The hardware mapping must not reinterpret bit polarity, byte order, padding, or tie semantics.

An even-N bundle or a continuous-f32 HDC-LTC profile is intentionally excluded until its canonical tie/numeric and trajectory semantics have been frozen for hardware use.

## 4. Protocol objects

Three independent machine-readable schemas accompany this document:

- device capability manifest;
- host command descriptor;
- device completion record.

They describe protocol shape and basic field constraints. They do not replace semantic validation, IOMMU setup, cryptographic attestation, HDL proof or hardware testing.

### 4.1 Device capability manifest

The host must obtain a capability manifest before dispatch. It binds:

- stable device identity and a per-initialization device epoch;
- backend and execution class (software reference, simulator, synthesis estimate, FPGA hardware, ASIC hardware);
- implementation artifact identity and hardware profile digest where available;
- admitted operation IDs and representation IDs;
- maximum aggregate command payload, required alignment and in-flight command bound;
- DMA isolation profile;
- cache-coherency behavior;
- the evidence stage and provenance references supporting those declarations.

A manifest is a claim by the backend until independently verified. It is not a hardware attestation merely because it contains hashes. A physical device with no qualified DMA isolation must not be admitted for host-memory DMA.

### 4.2 Command descriptor

A command binds:

- protocol schema/version;
- command ID, device ID, device epoch and monotonic sequence within that epoch;
- canonical operation ID and semantic contract identity;
- exact representation identity and dimensions;
- workload/input manifest digest;
- implementation/kernel artifact digest when known;
- timeout budget;
- operation parameters;
- a list of opaque buffer handles, generations, offsets, lengths and access rights.

**Raw host pointers, process virtual addresses and unconstrained physical addresses are forbidden.** The host driver resolves opaque handles inside a registered DMA domain. Offset and length arithmetic must use checked operations before any mapping or device submission.

The schema defines buffer descriptors generically. The host semantic validator must enforce operation-specific role/length/access rules and checked range arithmetic:

| Operation | Required buffers | Required size rules |
|---|---|---|
| XOR exact | input_a, input_b, output | each exactly 2,048 bytes |
| Hamming exact | input_a, input_b, result | inputs exactly 2,048 bytes; result holds one unsigned 16-bit distance |
| Top-1 exact | query, candidate_vectors, candidate_ids, result | query 2,048 bytes; candidate payload = candidate_count × 2,048; ID payload = candidate_count × 8; result at least 16 bytes |

Top-1's 16-byte result record is little-endian: winner stable ID (u64), Hamming distance (u16), then six reserved zero bytes. The host rejects non-zero reserved bytes. Candidate IDs are stable application IDs, not array positions.

For all operations, max_transfer_bytes is interpreted as the checked aggregate length of every buffer in one command. Offset alignment is enforced independently of payload length, allowing the exact 2-byte Hamming-distance result while retaining an 8-byte-aligned start address. The validator requires all input buffers to be readable and the operation's output buffer to be writable; output/input range overlap is rejected.

### 4.3 Completion record

The device completion binds the command ID, device ID, epoch, sequence, semantic operation ID, canonical semantic-contract digest and implementation digest. A successful completion must also echo an output_buffer object containing the exact output role, opaque handle ID, generation, offset and length from the submitted command. The host compares every field before returning ownership or accepting the result. A matching command ID alone is insufficient. The host also compares the echoed operation, canonical semantic-contract digest and implementation digest with the pending command; a device/profile change while work is in flight invalidates the completion.

A successful completion includes a bounded result object and output digest. A digest is an integrity aid, not a digital signature or proof of correct computation. The pure state machine returns `CompletedOutputAvailable` only after the matching completion has returned buffer ownership to the host. This is **not** a verified-result status: the caller must read the actual output bytes, recompute the digest, check the operation-specific result encoding and compare against the independently generated software oracle before admitting the answer to the application. Physical authenticity requires a separately qualified trust chain.

The pure host validator defines the operation contract IDs for semantic version 1.0.0 as symthaea.binaryhv.bind-xor-exact, symthaea.binaryhv.hamming-exact and symthaea.binaryhv.top1-exact.

The host pins each semantic digest to the following canonical UTF-8 descriptors. The bytes are exactly the lines shown, separated by LF (0x0A), with one final LF; no BOM, whitespace rewriting or CRLF conversion is permitted. SHA-256 is over those exact bytes.

```text
luminous.hdc.semantic-contract
version=1.0.0
operation=hdc.bind_xor.exact.v1
representation=luminous.binaryhv.bipolar.16384.v1
dimension_bits=16384
payload_bytes=2048
bit_order=lsb0
byte_order=little
result=output_byte[i]=input_a_byte[i]^input_b_byte[i]; i=0..2047
```

SHA-256 of the descriptor bytes above: `156f87ae7d96cdc89cf09d84723c2f65aa80f2c0fcfc193df06f8545ed332c67`

```text
luminous.hdc.semantic-contract
version=1.0.0
operation=hdc.hamming.exact.v1
representation=luminous.binaryhv.bipolar.16384.v1
dimension_bits=16384
payload_bytes=2048
bit_order=lsb0
byte_order=little
result=popcount(input_a XOR input_b)
distance_range=0..16384
result_encoding=unsigned_u16_little_endian
```

SHA-256 of the descriptor bytes above: `c7e4cdcf7ee0ae962e1d4d033a125312a82e65a1aac7ce2f8d4f4d03b132712e`

```text
luminous.hdc.semantic-contract
version=1.0.0
operation=hdc.top1.exact.v1
representation=luminous.binaryhv.bipolar.16384.v1
dimension_bits=16384
payload_bytes=2048
bit_order=lsb0
byte_order=little
candidate_id=stable_unsigned_u64
score=popcount(query XOR candidate)
winner=argmin(distance, stable_candidate_id)
tie_break=lowest_stable_candidate_id
result_layout=16_bytes_little_endian:u64_winner_id,u16_distance,u48_reserved_zero
```

SHA-256 of the descriptor bytes above: `04b106f50f40963f0d2f88f9d779b83d9a55bd22d2e151f45f5211fa05c3f9cd`

A change to any descriptor byte requires a new semantic version/digest and an explicit protocol admission update. The host validator rejects a valid-looking SHA-256 value that is not the pinned canonical digest. Successful completions echo that digest so the host can bind the returned result to the pending semantic command.

## 5. DMA, memory ownership and isolation

### 5.1 Buffer lifecycle

A buffer transitions through these host-owned states:

1. **Unregistered** — no device access is possible.
2. **Registered** — handle and generation are assigned; access rights and permitted range are fixed.
3. **Host-owned** — host may write/read according to the registered rights.
4. **Device-owned** — host does not mutate or recycle the submitted range; required cache clean/invalidate and memory barriers have been performed.
5. **Completed** — matching completion observed; ownership returns to host after the required invalidation/barrier.
6. **Quarantined** — device loss, reset or ambiguous completion means the range cannot be reused until the driver proves the old DMA context is quiescent.

No buffer may be freed, remapped or reused while a device could still write it.

### 5.2 Isolation profiles

- **IOMMU domain:** preferred; device DMA is restricted to the command's registered mappings.
- **Dedicated bounded DMA pool:** weaker fallback for boards without an available IOMMU. All device-visible memory is isolated in a dedicated pool, with bounds and ownership checked by the driver. This profile must be disclosed and tested.
- **None / unknown:** physical DMA execution is not admitted.

### K26 / Zynq UltraScale+ platform note

The AMD Zynq UltraScale+ MPSoC used by the K26 family includes an SMMU capable of restricting DMA-capable masters to assigned address spaces. This is a potential implementation of the protocol's IOMMU-domain profile, not automatic qualification. AMD's 2026.1 UG1137 documents SMMU behavior; UG1085 describes SMMU translation/protection. AMD's deployment guidance for Zynq UltraScale+ notes that PetaLinux BSPs disable the SMMU by default and that programmable-logic master integration has additional configuration and limitations.

For a K26 board, the qualification record must therefore identify the exact SOM, carrier, boot firmware/BSP, SMMU configuration, PL DMA master/Stream ID, IOMMU domain attachment and the actual negative out-of-domain DMA test. A SoC block's presence or a device-tree node is not enough. If the configured PL path cannot be independently qualified, use the weaker dedicated bounded DMA pool and label it accordingly.

References:
- AMD [UG1137 2026.1 — System Memory Management Unit](https://docs.amd.com/r/en-US/ug1137-zynq-ultrascale-mpsoc-swdev/System-Memory-Management-Unit)
- AMD [UG1085 — System Memory Management Unit](https://docs.amd.com/r/en-US/ug1085-zynq-ultrascale-trm/System-Memory-Management-Unit)
- AMD [Zynq UltraScale+ MPSoC SMMU integration guidance](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/428114096/Zynq+UltraScale+MPSoC+SMMU)

Do not assume the selected K26/KV260, ECP5 board, SoM, firmware or carrier has an enabled/usable IOMMU path or coherent DMA. Verify exact platform documentation and then test the actual board.

### 5.3 Cache and ordering

Cache behavior must be explicit per hardware profile: coherent, explicit flush/invalidate, or unknown. Unknown is not treated as coherent. Every transfer must have explicit ownership and ordering barriers. The protocol does not infer coherence from the CPU architecture name.

## 6. Device epochs, command sequencing and ambiguous outcomes

A new device initialization/reset creates a new 128-bit device epoch. Commands carry that epoch and a strictly increasing sequence number. Completions from an earlier epoch are rejected even if their command ID matches.

The pure Rust host state machine reserves every referenced handle atomically before the driver is called. It enforces strictly increasing sequence numbers per epoch, bounded in-flight commands and unique command IDs. A timeout, failed/cancelled device command or ambiguous outcome blocks **all new submissions in that device epoch**, not only reuse of the affected buffers. Recovery has two independent gates: (1) establish a distinct new device epoch through externally completed reset/reinitialization; and (2) obtain independent confirmation that the old DMA context is quiescent. New dispatch remains blocked until both gates are satisfied, regardless of whether the quarantined buffers themselves have already been individually released. It does not perform MMIO, map DMA, pin memory, flush caches or establish physical quiescence.

The host-side execution state is:

**Host-owned → Validated → Device-owned → Completed**

Failure paths are distinct:

- Rejected before execution: the device guarantees it did not acquire DMA ownership; the host may reclaim the buffers.
- Successful completion: only the exact command/epoch/sequence and echoed output handle/generation/range return the buffers to host ownership.
- Device failure, cancellation, timeout or **OUTCOME_UNKNOWN**: buffers enter quarantine because protocol v0.1 carries no independent DMA-quiescence assertion.
- Reset: invalidate all prior-epoch commands, quarantine outstanding buffers, reinitialize the device and reconcile before reuse.

Quarantined buffers are released only after an explicit confirm_dma_quiescent(old_epoch) call from a driver/platform layer that has independently established the old DMA context cannot write again. A confirmation received before reset releases eligible buffers but does not resume dispatch; a reset received before confirmation rotates the epoch but also does not resume dispatch. Only after both events are recorded does the ledger admit new commands. Calling either pure state-machine method is not itself proof of physical quiescence or successful reset.

A timeout is not proof that a command did not execute. Do not blindly resubmit a command into the same output range. Exact HDC operations are mathematically pure, but concurrent DMA writes, buffer reuse and command duplication can still corrupt memory. A replay is permitted only after the previous DMA context is quiescent and the input/output generations are reconciled.

## 7. Exact fallback and backend eligibility

The command names a semantic operation, not a preferred chip. A backend is eligible only when its capability manifest admits the exact operation, representation, dimensions, limits and evidence stage required by the caller.

The initial backend comparison uses:

- canonical Rust scalar reference;
- optimized CPU/SIMD;
- GPU Fabric backend only when actual execution is observed;
- RTL simulation (correctness evidence only);
- synthesis estimate (resource evidence only);
- identified FPGA hardware (physical execution only).

If an FPGA is unavailable, over capacity, reset, stale, unsupported or unqualified, the host may execute the **same exact semantic operation** on the canonical CPU/SIMD path. The result records the actual backend. It must never report an FPGA execution when a CPU fallback ran.

An approximate, analogue or reduced-dimensional result is a different representation/operation profile. It cannot satisfy an exact request or inherit an exact qualification.

## 8. Mycelix, Symtropy and Sol-Atlas boundaries

- **Mycelix** may record device, firmware, board, lot, repair and operation provenance. A provenance record is not proof that a measurement is true, and a device is not authorized to mutate a journal/payment/actuator merely because it returned a result.
- **Symtropy** may model thermal, power, reliability and physical resource behavior. Modelled values remain distinct from board measurements.
- **Sol-Atlas** may track mineral, component, manufacturing, repair and lifecycle provenance. Carrier assembly with imported FPGA/SoC remains imported-silicon dependent.
- **Symthaea** owns semantic work requests and canonical software references. It does not self-certify hardware correctness.

## 9. Qualification matrix

### Descriptor and host validation

Positive fixtures:
- valid XOR, Hamming and Top-1 command shapes;
- one candidate;
- stable ID ties with candidate order shuffled;
- distinct sequence numbers within one epoch;
- exact capability match.

Negative controls:
- unknown operation/version;
- wrong representation/dimension/byte length;
- arithmetic overflow in candidate_count × vector_bytes;
- offset/length outside registered buffer;
- zero length, misalignment, stale buffer generation;
- read/write permission mismatch;
- duplicate sequence or command ID;
- wrong device epoch;
- unsupported IOMMU/coherency profile;
- unknown capability or artifact mismatch.

### Runtime and memory safety

- device completion after timeout;
- duplicate, late or cross-epoch completion;
- device reset during in-flight DMA;
- queue full / backpressure;
- cancellation race;
- input/output aliasing;
- non-coherent cache stale data;
- partial output and malformed reserved bytes;
- device loss with uncertain DMA quiescence.

Acceptance requires invalid descriptors to be rejected before DMA, no stale or mismatched-output completion accepted, no new submissions after an uncertain outcome in the same epoch, no mutation outside registered output ranges, and no buffer reuse before quiescence. Include completion output-handle/generation/range substitution, output/input overlap, duplicate roles, non-canonical decimal u64 strings, and u64 overflow as negative controls.

### Functional parity

For every admitted command, compare output byte-for-byte (or exact integer distance) against the canonical Rust oracle. Include all-zero/all-one, alternating, one-hot, boundary-bit, fixed-seed random and adversarial fixtures. Top-1 ties must use lowest stable ID, not lowest input slot.

## 10. Benchmark and competitive targets

Reuse #7864 and #10901. Do not create a second benchmark runner.

Run the same frozen vectors, query IDs, candidate order, task contract and measurement boundary across the canonical scalar CPU, optimized CPU/SIMD, any real GPU backend and FPGA candidate. Report separately:

- exact result parity and task success;
- end-to-end p50/p95/p99, throughput and cold-start/setup;
- host/device bytes, registration, copying, queueing, synchronization and result transfer;
- resident memory, capacity, bank utilization and bytes per accepted query;
- energy/query only with a calibrated meter and declared full-system boundary;
- device utilization, stalls, reset/recovery and thermal behavior;
- FPGA LUT/FF/BRAM/URAM/DSP and timing only from a named synthesis/implementation target.

Keep kernel-only numbers diagnostic; the product decision is made on end-to-end successful work.

The existing project candidate gate remains: equal exact outputs and task quality, with either ≥1.5× end-to-end throughput or ≥30% lower measured joules/query, while p95 latency is no more than 10% worse in the declared serving profile. These are proposed gates, not measured results or universal industry standards.

Include negative controls where CPU/SIMD wins: small batches, warm cache, transfer-bound requests, tiny codebooks and unsupported device profiles.

## 11. Evidence and claim ceiling

| Evidence available | Allowed claim | Not established |
|---|---|---|
| Schema/fixture | Protocol shape is specified | Driver, DMA or device behavior |
| Host validator tests | Descriptor logic behaves as tested | RTL correctness or IOMMU isolation |
| RTL simulation | Simulated design matches tested oracle fixtures | Synthesis, FPGA, power or speed |
| Formal properties | Named properties hold under the proof model | Whole-system correctness |
| Synthesis | Resource/netlist report for exact flow | Routed timing or board behavior |
| FPGA implementation | Bitstream/implementation for exact board target | Observed board operation |
| FPGA board run | Physical function under captured conditions | ASIC behavior or commercial reliability |
| Calibrated end-to-end board benchmark | Measured workload result within stated boundary | General superiority outside that profile |

The first PR implements only the machine-readable contract and fixtures. No driver, RTL integration, simulator result, formal proof, synthesis result, FPGA measurement or performance superiority is claimed.

## 12. Proposed PR sequence

A. Protocol schemas, capability manifest, examples and negative-case catalogue.  
B. Pure host-side validator, operation-specific range/permission checks, output-binding echo and deterministic state-machine tests; no MMIO/DMA.  
C. Simulator adapter and command/completion differential tests.  
D. Exact FPGA adapter on the #10901 board profile with declared isolation/coherency.  
E. End-to-end benchmark and receipt integration with #6968/#7864.

