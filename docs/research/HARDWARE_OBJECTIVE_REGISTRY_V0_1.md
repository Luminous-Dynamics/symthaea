# Hardware Objective Registry v0.1

**Status:** Typed registry, hardware-profile manifest, and fifteen Rust test declarations are committed on `research/hardware-objective-registry-v0.1`. Compile/format/test execution and real dataset import remain outstanding.

**Tracking:** [NATS-Bench-MCU v2 integration #9411](https://github.com/Luminous-Dynamics/symthaea/issues/9411) and [Pareto-regret VoC #9389](https://github.com/Luminous-Dynamics/symthaea/issues/9389).

## 1. Purpose

Multi-objective NAS needs a machine-checkable definition of every objective and every measurement context. A score such as "latency = 2.0" is meaningless without its unit, direction, normalization, device, firmware, runtime, compiler, quantization, and measurement protocol. A search budget also needs its own acquisition-cost unit and cost model; energy per inference is an objective value, not the cost of querying the energy objective during search.

The registry therefore separates:

- objective value units (e.g. milliseconds per inference, millijoules per inference, KiB, accuracy percent);
- direction (maximize or minimize);
- fixed normalization range;
- objective source (static metadata, trained evaluation, imported benchmark, or physical device measurement);
- measurement protocol digest;
- hardware profile digest, when device-bound;
- measurement acquisition-cost unit and cost-model digest.

It normalizes values into maximize-is-better utility for the Pareto kernel. Out-of-range values are rejected instead of clipped; clipping would conceal distribution shift and distort Pareto geometry.

## 2. Hardware profile manifest

A profile binds:

- profile and device identity;
- board ID and board revision;
- firmware SHA-256;
- runtime SHA-256;
- compiler/toolchain SHA-256;
- quantization-specification SHA-256;
- measurement-protocol SHA-256;
- instrument model and instrument ID;
- sampling frequency.

The profile digest is calculated over the complete canonical profile with the digest field cleared. The objective registry contains the profile manifests themselves, not just free-floating hash strings, and includes them in its own canonical digest. Registry validation checks each profile digest, rejects duplicate profile IDs/digests, requires every physical or imported physical objective to reference a profile in the registry, and requires the profile's protocol digest to match the objective's protocol digest.

A digest establishes identity, not truth. Acquisition still has to verify that firmware and instrument logs correspond to the claimed hardware.

## 3. Typed measurement outcomes

A candidate-objective record has one of four states:

- **Measured:** finite raw value, replicate ID, sample count, source-artifact SHA-256, and positive acquisition cost in the registry's declared resource unit.
- **Infeasible:** explicit reason code and source-artifact SHA-256 (e.g. flash budget exceeded).
- **NotMeasured:** explicit reason code because no observation was supplied or attempted.
- **Failed:** explicit reason code, retryability and evidence SHA-256 because an attempted measurement failed.

These states are never converted into each other. In particular, an infeasible architecture is not assigned zero latency, zero energy, maximum penalty, or a fabricated resource value. A not-measured objective is unknown, not bad. A failed measurement is not a hardware infeasibility; it may be retryable.

## 4. Normalization

For frozen raw bounds (l < u), define:

- maximize: (z = (x-l)/(u-l));
- minimize: (z = 1-(x-l)/(u-l)).

The accepted range is exactly [l,u] up to a 1e-12 floating tolerance. Out-of-range values fail rather than clamp. If a benchmark requires wider future support, publish a new registry version and re-qualify; do not silently alter the scale after observing evaluation outcomes.

## 5. NATS-Bench-MCU v2.0 mapping

Official dataset: [Zenodo DOI 10.5281/zenodo.21644470](https://doi.org/10.5281/zenodo.21644470), published 28 July 2026, CC BY 4.0. It measures NATS-Bench TSS architectures on Nordic nRF5340 with a Nordic Power Profiler Kit II at 100 kHz.

The camera-ready v2 deposit reports:
- 15,625 architecture indices;
- 14,281 successful hardware profiles;
- 1,344 flash-infeasible architectures;
- 1,024 KiB application-core flash and 277.6 KiB fixed firmware overhead, leaving a 746 KiB deployable INT8 flatbuffer budget;
- compact publication exports for hardware metrics, NATS-Bench accuracies, failures, coverage and INT8-ranking summaries;
- 16 raw archive chunks totalling about 36.2 GB and a correction overlay for 117 post-hoc corrections.

The public GitHub README still contains earlier v1 numbers (14,279/1,346 and an 800 KiB precheck). The official v2 Zenodo record is authoritative. Never combine v1 descriptions and v2 records without an explicit migration.

Recommended imported objectives (each receives its own stable ID and frozen raw scale):

| Objective ID | Source | Value unit | Direction | Notes |
|---|---|---|---|---|
| `int8_accuracy_cifar10` | ImportedBenchmark | percent | maximize | Use the INT8-deployable accuracy value, not FP32 as a silent substitute |
| `int8_accuracy_cifar100` | ImportedBenchmark | percent | maximize | Keep dataset/task in the ID or task key |
| `latency_nrf5340` | ImportedPhysicalDeviceMeasurement | seconds/inference | minimize | Bind exact nRF5340 board/firmware/runtime/PPK2 profile |
| `energy_nrf5340` | ImportedPhysicalDeviceMeasurement | millijoules/inference | minimize | Preserve per-device source data and replicate aggregation |
| `power_nrf5340` | ImportedPhysicalDeviceMeasurement | milliwatts | minimize | Do not infer from rounded energy if direct source data exists |
| `current_nrf5340` | ImportedPhysicalDeviceMeasurement | microamperes | minimize | Preserve PPK2 provenance |
| `flash_rom_nrf5340` | ImportedPhysicalDeviceMeasurement | KiB | minimize | Use a separate hard feasibility cap |
| `sram_nrf5340` | ImportedPhysicalDeviceMeasurement | KiB | minimize | Use measured/allocated memory definition from v2 README |

Exact CSV column names and units must be read from the v2 export README before importing. The list above is the intended semantic mapping, not a claim that the file has already been acquired or parsed.

Acquisition-cost units should initially use a clearly named replay budget such as `objective-query` or a predeclared device-profiling time unit. Do not use the objective's own energy/latency as its measurement cost. If original campaign wall-clock cost is unavailable, report query-count-budget results separately from real profiling-cost claims.

## 6. Import and verification

1. Acquire the exact v2 export archive from Zenodo. Check its publisher-provided MD5 as a transport integrity check, then compute our own SHA-256 over the downloaded archive and extracted files.
2. Extract into quarantined staging. Read the v2 README and headers from the archive, not the stale GitHub README.
3. Deterministically import the hardware metrics, accuracy and failures tables; preserve original row/file identity and exact architecture indices.
4. Verify the measured and infeasible sets are disjoint and their union is exactly 0..15624. Verify the v2 counts (14,281 and 1,344).
5. Check unique candidate indices, valid units, finite values, allowed device/profile IDs, duplicate rows, coverage and INT8 accuracy joins.
6. Bind the correction overlay and its provenance. Reproduce corrected failure labels and corrected records without modifying source files in place.
7. Independently re-import and compare all normalized objective records and Pareto summaries.

Do not impute latency, current, power or energy for flash-infeasible candidates. Their objective vector is partially observed with explicit hard infeasibility status. The Pareto/decision layer must first apply feasibility, then compare objective values among feasible candidates.

## 7. Tests and boundaries

Sixteen authored Rust tests currently cover profile digest binding, missing profile rejection, registry order invariance, registry tampering, maximize/minimize normalization, range rejection, physical device/protocol mismatch, imported-profile existence and protocol matching, acquisition-cost unit mismatch, positive cost requirements and distinct measured/infeasible/not-measured/failed outcomes.

The code has not been compiled or executed in the current environment; `cargo`, `rustc` and `rustfmt` are unavailable. The Zenodo export has not been downloaded here, so there are no acquired-file hashes or import measurements yet.

This registry is a semantic and provenance gate, not a data-source truth oracle. A second importer/verifier and raw-artifact audit remain mandatory.
