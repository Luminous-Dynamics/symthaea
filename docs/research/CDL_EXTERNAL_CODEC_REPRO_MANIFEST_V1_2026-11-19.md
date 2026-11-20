# CDL-Bench external codec reproduction manifest v1

**Date:** 2026-11-19  
**Status:** Repository artifacts inspected and commits pinned; external toolchains are not installed in the current execution environment; no DeXOR or TerseTS comparison has been run.

This document pins the first reproducible external baseline cohort for the F64C/F64S benchmark. Paper-reported numbers remain contextual until the listed artifacts and datasets have actually been built and executed under the manifest.

## 1. DeXOR

| Field | Pinned value |
|---|---|
| Repository | `https://github.com/SuDIS-ZJU/DeXOR` |
| Branch at audit | `master` |
| Commit | `06128b3df57cc1e8c53681231e235e22eb508c8d` |
| README blob | `f481cbef7fbbe80e26eb5317dfd2b39e5ac60031` |
| `pom.xml` blob | `21fbdf4cdfef6302a70f509808b8f9623b566468` |
| License blob | `888d1154b5e2341ced3432cf413a50307866cec8` |
| Required runtime per upstream README | Java 8 |
| Build | `mvn clean install` |
| Expected artifact | `target/OL-TSC-1.0-jar-with-dependencies.jar` |
| Input shape | CSV `<timestamp,value>` |
| Primary dataset | `datasets/Overall`, 22 CSV files |
| Dataset tree SHA (Git object) | `9793cb72b735a052b362fe40db64538cb2c77357` |

### Supported methods

The upstream README lists `Gorilla`, `Chimp`, `Chimp128`, `Elf`, `ElfPlus`, `Camel`, `DeXOR`, `ALP`, `Elf*` and `SElf*`. It also states that the current application supports `double` only. The first reproduction run should execute:

```sh
java -jar target/OL-TSC-1.0-jar-with-dependencies.jar \
  -in ./datasets/Overall \
  -out ./storage \
  -log ./results \
  -m DeXOR Gorilla Chimp Chimp128 Elf ElfPlus ALP 'Elf*' 'SElf*'
```

`Camel` is not included in the default exact-lossless comparison. Although the DeXOR README describes it as lossless under its restricted decimal-place use case, that is not sufficient for the arbitrary IEEE-754 input corpus in CDL-Bench. It may be tested in a separate restricted-precision stratum only after bitwise round-trip validation over every included input.

### TSFile caveat

The checked-in `pom.xml` currently comments out the upstream dependency:

```xml
<dependency>
  <groupId>org.apache.tsfile</groupId>
  <artifactId>tsfile</artifactId>
  <version>2.2.0-SNAPSHOT</version>
</dependency>
```

The README says the TSFile implementation is in the repository's `tsfile/java/tsfile` subtree and instructs users to build it and add the dependency manually. The pinned script therefore runs the documented CSV `Overall` path and does not silently change the pom or claim the TSFile path is reproduced. A separate TSFile reproduction must record the exact TSFile commit/build artifact, dependency resolution and any source modification.

### Required capture

The pinned shell script verifies the checkout commit, Java major version, Maven presence, 22-file dataset count and writes SHA-256 hashes of every source CSV. It records Java/Maven versions and runner output. The reproduction report must additionally preserve the complete command line, method configuration, JVM flags, operating system, CPU, memory, warm-up/repeat policy, and exact source data order.

## 2. TerseTS

| Field | Pinned value |
|---|---|
| Repository | `https://github.com/cmcuza/TerseTS` |
| Branch at audit | `main` |
| Commit | `64abd7767f8fc6ac1ef4e932f2027a74cf8fb4a0` |
| README blob | `29a744bdca86dd6358a65f2fc188f476deb53b2d` |
| Build file blob | `8724a04a976da6472784efdab6e83c0bdfdd8ad5` |
| Method registry blob | `a03e256ca7079575c8523a960a82f4f7d102a195` |
| Configuration blob | `0cbbebd787b725124b7fb4abbc8982ac4204ebf1` |
| License | Apache-2.0 |
| Build system | Zig; native library has C API and language bindings |
| Project warning | README explicitly labels the current version beta and not production ready |

### Exact-lossless shortlist for the first comparison

Source inspection supports the following initial whitelist. Even these methods must pass CDL-Bench's own bitwise tests at the pinned revision before a result is admitted.

| Method | First exact-lossless stratum? | Basis / caveat |
|---|---:|---|
| `Uncompressed` | Yes | Raw IEEE-754 8-byte bit-cast representation |
| `Chimp64` | Yes | Source explicitly claims bitwise losslessness and contains bitwise round-trip tests for NaN payloads, infinities and signed zero |
| `Chimp128` | Yes | Source explicitly claims bitwise losslessness and contains corresponding bitwise tests |
| `Elf` | Yes | Source explicitly claims bitwise losslessness and contains bitwise round-trip tests for NaN payloads, infinities and signed zero |

### Conditional candidates, not pre-approved

| Method | Why conditional |
|---|---|
| `RunLengthEncoding` | Its encoder detects runs using `value == current_value`. IEEE-754 `+0.0 == -0.0` although their u64 encodings differ. A run crossing that sign change can be decoded as the first sign's bit pattern. It must be excluded from any corpus containing that adjacency unless fixed upstream or pre-screened under a narrowly declared input contract. |
| `BitPackedDeltaEncoding` | It rejects non-finite values and values with magnitude above (10^{15}). It also classifies scaled values with numeric `==` and converts through i64, so signed-zero preservation is not established. Keep it in a restricted finite-domain experiment only after CDL-Bench bitwise validation of every sample. |

A source-level test-name `testLosslessMethod` is not enough to admit a codec to the bit-exact leaderboard. In this pinned TerseTS revision, the generic helper computes an absolute difference, casts it to f32 and checks that it is zero; that is weaker than comparing all 64 bits. The stronger `expectLosslessRoundTrip` helper does compare bit-cast u64 values, and Chimp64/Chimp128/Elf include explicit tests using it. CDL-Bench must use its own u64 bit comparison for every run.

### Exclusions

- `Camel`: TerseTS source explicitly calls it a lossy value representation and warns that binary floating-point reconstruction may produce a neighbouring f64 bit pattern even when decimal precision is retained. Do not include it in the bit-exact leaderboard.
- `BitPackedQuantization`, `SerfQT`, `BitPackedBUFF`, `MacaqueS`, `MacaqueV`, functional approximation, line simplification, histogram approximation and DFT methods: excluded until a method-specific proof/test shows exact u64 recovery. A method name or a generic decompressor is not sufficient.
- Any bounded-error or lossy method belongs in a separate protocol with its explicit error metric, pointwise/aggregate guarantee and identical tolerance—not in the exact-lossless table.

### Build

The pinned `build.zig` creates a library from `src/capi.zig`, includes the PocketFFT C source, enables libc, and defines a `test` step. The current script records the Zig version and runs:

```sh
zig build test -Dtarget=x86_64-linux -Doptimize=ReleaseFast
zig build -Dtarget=x86_64-linux -Doptimize=ReleaseFast
```

The expected Linux shared library path is `zig-out/lib/libtersets.so`. The helper then compiles `scripts/research/tersets_bitwise_gate.c` against `bindings/c/tersets.h` and the pinned shared library with `cc -std=c11 -O2 -Wall -Wextra -Werror`. It runs the strict bitwise gate: Uncompressed/Chimp64/Chimp128/Elf must preserve every u64 bit on IEEE-754 edge and signed-zero fixtures; RLE and BitPackedDelta are reported separately as conditional probes. Before adapter development, retain full Zig and C build logs and capture the library SHA-256. The upstream project does not pin a Zig release in the inspected root files; the compiler version must therefore be fixed in the reproduction record, not inferred.

## 3. Apache Arrow / Parquet ALP benchmark

This is a separate ALP reproduction path from the ALP implementation bundled in DeXOR. Use the upstream Parquet/Arrow implementation and its official benchmark harness.

| Field | Pinned value |
|---|---|
| Repository | https://github.com/alamb/alp_benchmark |
| Branch at audit | main |
| Repository commit | 601e2b1849d86069cb3036950ab9db1ab74d08b4 |
| README blob | a54465b454f4367f13072af3ffd0b4da67f40093 |
| benchmark.sh blob | b371590c1b00bf452effa3089e3843fa65eb2caa |
| download_data.sh blob | 36b5d3f3da9eb015dedabf360b6dcc7427c6cdb3 |
| Cargo.toml blob | f854e1a782628d0a799562bde5d188c3377c8494 |
| Arrow-rs / Parquet dependency revision | f9794b4f4ac9fa896ed507b6d7c6e7556db041a9 |
| Dataset | 30 CWI f64 binary files |
| Dataset archive | 6.7 GiB |
| Dataset archive SHA-256 | 1070817918b9e2b2cc7003995927bd04fe7b942045383913d3f40437eda29831 |
| Encoding | Raw little-endian IEEE-754 binary64, 8 bytes/value, no header |
| Upstream runner | benchmark.sh (downloads/verifies/extracts corpus, prints environment, then cargo run --release --bin benchmark) |

### Reproduction command

```sh
git clone https://github.com/alamb/alp_benchmark.git
cd alp_benchmark
git checkout 601e2b1849d86069cb3036950ab9db1ab74d08b4
./benchmark.sh
```

The script pins the Arrow-rs ALP implementation revision in its Cargo manifest and the corpus archive digest in download_data.sh. It defaults to RUSTFLAGS=-C target-cpu=native, and records CPU model, architecture, SIMD, logical CPUs, OS/kernel, CPU governor, Rust/LLVM/Cargo versions, benchmark commit, worktree cleanliness and dataset hash. To use a pre-provisioned corpus, set ALP_DATASET_DIR; the upstream runner still checks for all 30 expected files.

### Fairness and interpretation

- ALP is best suited to decimal-like floating-point columns. Keep arbitrary full-range bit patterns, NaN-payload tests and sparse multi-field records as separate strata; ALP's own project overview warns that wide exponent/significance data is outside its target distribution.
- Run the official CWI corpus unchanged first. Then adapt the same raw production-derived numeric columns from CDL-Bench datasets into Parquet columns without reordering or dropping values.
- Compare ALP's own Parquet output with complete CDL frames and PLAIN+Zstd / BYTE_STREAM_SPLIT+Zstd controls. Count Parquet footer, page metadata, indexes, exceptions and any required dictionaries. Report random-access latency as well as complete file size.
- Match the host CPU, SIMD build, thread/core budget, memory ceiling and warm-up/repetition protocol across CDL and ALP. The ALP runner's target-cpu=native flag must be matched by the CDL build when reporting SIMD-enabled throughput.
- The public ALP benchmark README itself still contains a prepublication checklist. Pinning and running it reproduces that benchmark harness and commit; it does not turn its own published numbers into independently audited results.

**Execution status at this audit:** repository scripts, dependency revision and corpus digest were inspected and pinned. The 6.7 GiB archive was not downloaded, and no ALP benchmark was built or run here because the environment lacks Rust/Cargo. No ALP reproduction result is claimed.

## 3. Reproduction helper

The repository includes:

`scripts/research/cdl_external_codec_repro.sh`

It never clones, fetches or checks out external repositories. Provide already checked-out source trees at the exact pinned commits:

```sh
DEXOR_DIR=/path/to/DeXOR \
TERSETS_DIR=/path/to/TerseTS \
bash scripts/research/cdl_external_codec_repro.sh
```

Both environment variables are optional; unset paths are explicitly skipped. Output defaults to `target/cdl-external` and can be changed with `CDL_EXTERNAL_OUT`.

The DeXOR CSV workflow requires Java 8 and Maven. The TerseTS workflow requires Zig and a C compiler. If a required toolchain is unavailable, the script fails closed rather than substituting another version.

## 5. Matched benchmark rules

1. The same ordered f64 bit patterns and chunk boundaries go to every method.
2. All exact-lossless methods must reconstruct every u64 bit pattern, including NaN payloads, signed zero, infinities and subnormals. Reject a method/result if even one bit differs.
3. Compare each method's actual compressed payload/frame size as exposed by that implementation. If a format includes a method byte, checksum or dictionary, count it. If the implementation's result omits a decoder dictionary or shared model, report both payload-only and total deployment costs.
4. Keep DeXOR's own dataset-oriented CSV run separate from the CDL synthetic fixtures; then add a paired common-corpus run only where input parsing and all method constraints match.
5. The DeXOR repository's `Overall` CSV files must be hash-recorded; no silent reformatting or numeric parse/re-serialization before comparison.
6. Report per-file ratios, geometric mean, compression/decompression throughput, peak memory, CPU cycles or time, and random-access/chunk constraints.
7. Publish all negative results and method restrictions, especially restricted decimal precision and unsupported precision/types.
8. Do not call TerseTS a production benchmark. Its upstream README marks it beta; label the harness accordingly.
9. No results are considered reproduced until artifact commit, toolchain version, input hashes, parameters, commands and bitwise verification logs are attached.

## 6. Current execution status

The repository and artifact inspection is complete. The external helper passes `bash -n` and its unset-environment dry run correctly reports DeXOR/TerseTS as skipped; the TerseTS C gate passes `cc -std=c11 -O2 -Wall -Wextra -Werror -fsyntax-only` against the inspected API header shape. The current environment has Java 21.0.12.1 and C/C++ compilers, but does not have Maven, Zig or Rust/cargo. The ALP benchmark's 6.7 GiB CWI archive has not been downloaded. Therefore:
- DeXOR has **not** been built or run.
- TerseTS has **not** been built or run.
- The upstream Arrow/Parquet ALP benchmark has **not** been built or run.
- No external codec comparison numbers are claimed.
- The setup script is authored and pinned, but its runtime workflow remains pending an environment with the required toolchains.

This distinction is intentional: source inspection and a pinned runner are reproducibility infrastructure, not reproduction evidence.


## 4. tparquet: end-to-end Pco/PCODEC + Zstd + ALP storage benchmark

The tparquet benchmark is now a mandatory storage comparator because it measures complete Parquet files on public market-data previews and selects PCODEC/Pco or Zstd per numeric column. Pin the ALP-inclusive result snapshot:

| Field | Pinned value |
|---|---|
| Repository | https://github.com/aperiodic-io/tparquet |
| Result/source commit | `d1a40f6fa5508c99ee95e9609f3c38b111e21cf6` |
| Results file blob | `c27c74dc5da5dd5330fdb3f9675492a7fad9b00e` |
| Benchmark report | `bench/RESULTS.md` |
| Published run date | 2026-11-07 |
| Host description | 4-vCPU cloud VM |
| Datasets | 31 datasets, 151,499,256 rows, four market-data providers |
| Reported all-dataset sizes | ZSTD(9) 957.1 MB; tuned ZSTD(9) 700.7 MB; tuned+ALP 649.5 MB; tparquet 478.6 MB |
| Reported all-dataset throughput | tparquet 102 MB/s write, 924 MB/s read; ALP 169/1,231; tuned ZSTD 141/1,129; default ZSTD 97/916 |

The run reports bit-exact values and smaller tparquet files for all 31 datasets. It chooses PCODEC or Zstd per numeric column from an initial batch; strings/flags remain on the general-purpose path. The ALP comparator applies ALP to all floating-point columns in addition to tuned Parquet encodings.

The source snapshot and results can be inspected at the pinned commit URL. The benchmark depends on public provider previews and downloads about 1.2 GB. The top-level `make bench` command in the repository runs the harness and rewrites the result report. Follow the benchmark's own README and verify the exact dataset/checksum list before execution.

### Fair comparison rules

- Run the same source samples and preserve every input row, key, null and timestamp.
- Compare complete Parquet/CDL file bytes, including footers, dictionaries, row groups, page metadata, indexes and checksums. Report value-payload size separately.
- Include Parquet default Zstd, tuned Zstd, tuned+ALP, Pco/PCODEC per column, and tparquet's actual per-column selector. A fixed ALP-only arm is not a substitute for best-of-ALP/Zstd.
- Measure read/write throughput, compression/decompression CPU, peak RSS, random single-row/field access, scans, and concurrent operation. Keep writer and reader implementations/cores comparable.
- Record whether a Parquet implementation supports ALP 2.14 preview and do not assume every reader can decode it. An unsupported reader must fail, not silently return incorrect values.
- This report is an external published benchmark, not an independently rerun CDL-Bench result.

## 5. OpenTelemetry Arrow / OTAP Dataflow Engine: transport and processing baseline

The project has advanced from the experimental adapter in OTEP-0156 to the incubating Rust OpenTelemetry Arrow Dataflow Engine. Pin the audited snapshot:

| Field | Pinned value |
|---|---|
| Repository | https://github.com/open-telemetry/otel-arrow |
| Main commit at audit | `bd5909819727f7a16a7985d97e510d35a9302e66` (2026-11-20) |
| README blob | `4e763fda422dbe40cc0b611c2b6428c95067a1ad` |
| Benchmark guide blob | `b7e419dee0410cbe2eea2ca7612adbb6ff574a7d` |
| Architecture | Arrow record batches / Arrow IPC over gRPC; OTLP interoperability; Rust dataflow engine |
| Status | Incubating / experimental |

Benchmark coverage in the pinned guide includes ~100k records/sec single-core normal load, 95% filtering (100k logs/sec input and ~5k output), backpressure, batch sizes 10..8192, saturation, 1/2/4/8/16-core scaling, pass-through, idle memory and binary size. The stated scaling target is >0.90 efficiency. The dashboards track throughput, CPU, network bytes, egress bytes per log, RAM and dropped-message percentage.

This is a transport/dataflow comparator, not a file compression comparator. Use it in a separate OTLP/OTAP track with end-to-end conversion, wire bytes, processing, memory, backpressure, retries/acks and drops included. Do not compare its message bytes directly with tparquet file bytes.

Reproduction URLs:
- Benchmark guide: https://github.com/open-telemetry/otel-arrow/blob/bd5909819727f7a16a7985d97e510d35a9302e66/docs/benchmarks.md
- Continuous benchmark dashboard: https://open-telemetry.github.io/otel-arrow/benchmarks/continuous/
- Filter comparison: https://open-telemetry.github.io/otel-arrow/benchmarks/nightly/filter/
- Saturation: https://open-telemetry.github.io/otel-arrow/benchmarks/nightly/saturation/
- Scaling efficiency: https://open-telemetry.github.io/otel-arrow/benchmarks/nightly/scaling-efficiency/

**CDL benchmark target, not current achievement:** match 100k records/sec per core in a comparable normal-load scenario, then target >0.90 scaling efficiency to 16 cores. Include pass-through and transformed-path tests, and report dropped records (must be zero for lossless transport), CPU, RAM and bytes on wire.
