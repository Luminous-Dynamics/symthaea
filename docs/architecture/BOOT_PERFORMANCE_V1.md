# Spore Boot Performance v1

Status: draft measurement contract

## Goal

Spore should make boot more legible and beautiful without materially delaying a usable graphical session. Performance claims require distributions across repeated boots, not a single fast observation.

## Two measurement layers

### 1. Deterministic headless renderer benchmark

`spore-boot-bench` measures CPU simulation and rasterization without DRM, systemd, or a compositor.

Default case:

```bash
cargo run -p symthaea-quicken-fb --bin spore-boot-bench --release
```

Standard resolution matrix:

```bash
cargo run -p symthaea-quicken-fb --bin spore-boot-bench --release -- --matrix
```

The benchmark fixes:

- scene seed;
- 30 Hz simulation timestep;
- warmup frame count;
- measured frame count;
- growth input.

It reports `grow`, `render`, and total CPU-frame timing distributions plus final branch count and XRGB8888 surface size.

It intentionally does **not** claim to measure DRM/KMS performance.

### 2. Live renderer receipt

When `services.symthaea-boot.performance.enable = true`, `quicken-fb` records timings in memory and writes one receipt on exit. No per-frame file I/O is performed.

Receipt fields include:

- DRM open time;
- first completed frame latency from process start;
- total measured frames;
- 30 Hz frame-work deadline misses;
- final branch count;
- `grow()` distribution;
- CPU `render()` distribution;
- DRM `blit_from()` distribution;
- total measured frame-work distribution;
- post-loop DRM release time.

The receipt is diagnostic evidence, never boot authority.

## Percentiles

Every timing series reports:

- count;
- min;
- integer mean;
- p50;
- p95;
- p99;
- max.

Percentile indexing is deterministic and shared between the headless and live paths.

## Whole-boot evidence

`scripts/measure-spore-boot.sh` captures a privacy-minimized evidence directory with:

- `systemd-analyze time`;
- critical chain;
- blame view;
- selected monotonic systemd properties for Spore, display manager, and graphical target;
- kernel/NixOS version when available;
- Spore performance/handoff receipts when present.

It deliberately excludes journal contents, hostname, user names, network identifiers, serials, process lists, environments, and command lines.

## Comparative protocol

For a serious Spore ON/OFF comparison:

1. use otherwise identical NixOS generations/specialisations;
2. alternate conditions rather than running all OFF then all ON;
3. collect at least enough boots to characterize variance (20+ per condition is a useful qualification target, not a statistical guarantee);
4. compare median, MAD or another robust dispersion metric, p90/p95, and outliers;
5. record cold/warm cache policy and power state;
6. do not discard slow boots without a documented external cause.

An example alternating block is:

```text
OFF
ON
ON
OFF
```

Repeat the block rather than relying on one ordering.

## Initial budgets, not claims

Until hardware evidence exists, use these only as engineering budgets:

- renderer first frame: should feel immediate once DRM is available;
- frame work: comfortably below the 33.3 ms 30 Hz budget at target resolutions;
- normal frame deadline misses: near zero;
- post-DRM release: target tens of milliseconds, investigate >100 ms;
- renderer service stop: hard bound currently 1000 ms;
- total Spore-enabled time-to-session delta: target statistically indistinguishable from OFF apart from a small bounded display handoff.

Do not publish these as achieved numbers until measured.

## Optimization order

Only optimize measured bottlenecks. Candidate hypotheses include:

1. repeated child-existence scans in mycelium growth;
2. full-screen CPU repaint of mature/static geometry;
3. full intermediate XRGB8888 buffer plus framebuffer copy;
4. mapping/copy behavior in `blit_from()`;
5. 500 microsecond sleep-loop wakeup frequency.

The benchmark must tell us which matter before code complexity is added.

## Rename sequencing

Do not mix the `quicken-fb` -> `spore-boot` rename into performance qualification. First obtain stable baseline receipts and CI gates under the existing binary name, then perform a mechanical compatibility-preserving rename in a separate PR so performance changes and naming changes remain independently reviewable.
