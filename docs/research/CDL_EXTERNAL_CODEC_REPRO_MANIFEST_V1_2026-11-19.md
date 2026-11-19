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

## 4. Matched benchmark rules

1. The same ordered f64 bit patterns and chunk boundaries go to every method.
2. All exact-lossless methods must reconstruct every u64 bit pattern, including NaN payloads, signed zero, infinities and subnormals. Reject a method/result if even one bit differs.
3. Compare each method's actual compressed payload/frame size as exposed by that implementation. If a format includes a method byte, checksum or dictionary, count it. If the implementation's result omits a decoder dictionary or shared model, report both payload-only and total deployment costs.
4. Keep DeXOR's own dataset-oriented CSV run separate from the CDL synthetic fixtures; then add a paired common-corpus run only where input parsing and all method constraints match.
5. The DeXOR repository's `Overall` CSV files must be hash-recorded; no silent reformatting or numeric parse/re-serialization before comparison.
6. Report per-file ratios, geometric mean, compression/decompression throughput, peak memory, CPU cycles or time, and random-access/chunk constraints.
7. Publish all negative results and method restrictions, especially restricted decimal precision and unsupported precision/types.
8. Do not call TerseTS a production benchmark. Its upstream README marks it beta; label the harness accordingly.
9. No results are considered reproduced until artifact commit, toolchain version, input hashes, parameters, commands and bitwise verification logs are attached.

## 5. Current execution status

The repository and artifact inspection is complete. The helper passes `bash -n` and its unset-environment dry run correctly reports both external runs as skipped; the C gate passes `cc -std=c11 -O2 -Wall -Wextra -Werror -fsyntax-only` against the inspected API header shape. The current environment has Java 21.0.12.1 but does not have Maven, Zig or Rust/cargo. Therefore:
- DeXOR has **not** been built or run.
- TerseTS has **not** been built or run.
- No external codec comparison numbers are claimed.
- The setup script is authored and pinned, but its runtime workflow remains pending an environment with the required toolchains.

This distinction is intentional: source inspection and a pinned runner are reproducibility infrastructure, not reproduction evidence.
