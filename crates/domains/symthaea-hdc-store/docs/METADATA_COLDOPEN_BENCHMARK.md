# Metadata-first mmap layout experiment

**Tracking:** [HDC-STORE-COLDOPEN-001 #7955](https://github.com/Luminous-Dynamics/symthaea/issues/7955)  
**Status:** experimental synthetic microbenchmark; it does not change or open the production store format.

## Run

From the repository root:

```sh
SYMTHAEA_SOURCE_SHA="$(git rev-parse HEAD)" \
RUSTC_VERSION="$(rustc --version)" \
HOST_PROFILE="$(uname -m) / <CPU model>" \
STORAGE_PROFILE="<filesystem and physical device model>" \
cargo run -p symthaea-hdc-store --release --example metadata_coldopen_bench -- \
  --vectors 16384 --rounds 30 --seed 7 --csv /tmp/metadata-coldopen.csv
```

The benchmark generates two new files under a unique temporary directory and deletes them on exit. It refuses to reuse an existing benchmark workspace or overwrite an existing CSV. Add `--keep-files` to preserve the generated synthetic files for inspection. The benchmark refuses to overwrite both the CSV and its `.meta.txt` sidecar; use fresh paths for each run. Injected run identity appears in console output and the sidecar.

The maximum corpus is 1,048,576 vectors, but that creates multiple gigabytes of test files. Begin with the default 16,384 vectors and scale deliberately.

## What is compared

Both generated files contain identical, deterministic 2,048-byte payloads and a stable ID sequence.

- **v2-like interleaved:** records have a 2,080-byte stride, with a status byte and eight-byte ID at record start, plus payload beginning 32 bytes into each record.
- **metadata-first SoA candidate:** 16-byte metadata records are contiguous, followed by padding to the host's page size and then the packed payload array.

This is a synthetic layout experiment, not a parser for production HdcStore files. A successful result does not make the candidate format compatible, crash-safe or production-ready.

## Measurements and their limitations

On Linux the example collects:
- process minor and major page-fault counter deltas from `/proc/self/stat`;
- before/after mapped-page residency snapshots from `mincore`, counted separately over metadata-probe and payload ranges;
- elapsed metadata-scan time, file size and a checksum that must match for the two layouts.

Before each run the benchmark requests `POSIX_FADV_DONTNEED` for the file where available. This is only an advisory cache-discard request. The residency count before scanning is reported so a run that begins with resident payload pages cannot silently be treated as cold. Residency snapshots can also race with kernel reclaim.

**Do not interpret page-faults/residency as physical media reads.** A major fault is not a direct block-I/O count; a minor fault can occur without disk I/O; the filesystem may use cache or readahead; and neither this program nor the OS hint guarantees cold cache. Physical I/O claims require separate block-layer/device instrumentation. Energy and write amplification are not measured.

The benchmark intentionally reads only status and ID bytes during its timed scan. The v2-like metadata resides in the same virtual pages as payload bytes, whereas the SoA metadata array is physically separated in the file. This isolates the cache-locality hypothesis; it does not test exact similarity search, store mutation, corruption handling, recovery or metadata-index overhead.

## Interpreting results

Use at least 30 rounds per layout for a published result, keep the raw CSV plus `.meta.txt` sidecar, and record source SHA, Rust version, OS/kernel, page size, filesystem, device model, power mode and whether the residency precondition was actually cold-ish. The harness records injected source/compiler/host/storage labels but cannot reliably discover every hardware/filesystem detail for you. Compare:
1. pre-scan payload pages resident;
2. post-scan payload pages resident;
3. minor/major page-fault deltas;
4. metadata scan p50/p95 and dispersion;
5. total file size and same-checksum validation.

A strong result is one where the pre-scan residency is low and the SoA run consistently brings materially fewer payload pages into residency while both layouts validate the identical metadata. If both files are already resident, treat the measurement as a warm-cache workload. Don't infer SSD reads from the virtual-memory counters.

## Proposed next gate (not yet achieved)

Only if the synthetic experiment shows a robust reduction in payload-page residency on a named host should work proceed to an actual versioned HdcStore candidate. That follow-on needs format versioning, checksummed metadata, exact live-content equivalence, interrupted-write/recovery tests, export/import and migration tests, and end-to-end retrieval benchmarks. Format v2 remains unchanged.

## Operating-system references

- Linux `mincore(2)` reports a snapshot of pages resident in memory, which can change after the call: [man7 mincore(2)](https://man7.org/linux/man-pages/man2/mincore.2.html).
- Linux `POSIX_FADV_DONTNEED` attempts to free cached pages but is advisory, and partial-page requests can be ignored: [man7 posix_fadvise(2)](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html).

These are why each sample reports pre-scan residency and the hint's return code rather than assuming a cold cache.
