# symthaea-logparse

**Status:** Phase 1 spike scaffold (Apr 13, 2026). Not production.

Part of the MSP / IT-ops wedge. See `memory/project_msp_wedge.md` for the full
three-product ladder, kill criteria, and strategic context.

## What this crate is

A Phase 1 spike to answer one question:

> Can Symthaea's HDC encoder cluster real IT-ops logs (Windows Event Log +
> syslog) into useful semantic categories?

If the answer is yes (cluster purity ≥ 0.50 on a labeled Evtx corpus), the
"autonomous Tier-2 engineer" thesis survives and Phase 2 (fleshing out
`symthaea-support` + wiring `reasoning_engine` to a typed `SystemStateGraph`)
is justified.

If the answer is no, the thesis dies cheaply here instead of after months of
product work.

## What this crate is NOT

- Not a production log collector. Evtx parsing runs offline on staged corpora.
- Not wired to `symthaea-core`'s real HDC encoder yet — uses a local reference
  implementation with matching dimensionality (16,384) and binding operators.
  The `hdc-encoder` feature is a placeholder for the swap.
- Not a clustering library. The benchmark example uses a nearest-centroid
  baseline. Real HDBSCAN integration is a TODO.

## Layout

| File | Purpose |
|---|---|
| `src/event.rs` | Normalized `LogEvent` all sources flatten into |
| `src/evtx_source.rs` | Windows Event Log parsing (via `evtx` crate) |
| `src/syslog_source.rs` | RFC5424/RFC3164 syslog parsing (via `syslog-loose`) |
| `src/encoder.rs` | HDC encoder (6 role-filler bindings → 16,384D bipolar HV) |
| `src/cluster.rs` | Cluster purity metric + nearest-centroid baseline |
| `examples/cluster_evtx.rs` | Phase 1 benchmark runner |

## Running the spike

```bash
# Stage a labeled corpus at /tmp/evtx-corpus with labels.csv
cargo run -p symthaea-logparse --example cluster_evtx -- /tmp/evtx-corpus/
```

Expected `labels.csv` format:

```csv
filename,label
lateral_movement_01.evtx,lateral_movement
ransomware_03.evtx,ransomware
benign_baseline.evtx,benign
```

## Kill criterion

Purity < 0.50 on a corpus of ≥5 distinct incident classes → thesis fails,
stop work on the MSP wedge.

## Next session TODO

1. Stage the DFIR.training Evtx corpus (free, public, labeled)
2. Swap nearest-centroid for real HDBSCAN
3. Per-provider purity breakdown (which event classes separate cleanly)
4. Wire real `symthaea-core` HDC encoder behind the `hdc-encoder` feature
5. If green: open Phase 2 tracking issue for `symthaea-support` fleshing
