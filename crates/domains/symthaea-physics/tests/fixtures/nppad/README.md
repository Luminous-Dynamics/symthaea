# NPPAD fixtures (external validation data)

Source: [thu-inet/NuclearPowerPlantAccidentData](https://github.com/thu-inet/NuclearPowerPlantAccidentData)
("NPPAD"), MIT licensed. Simulated with PCTRAN, a widely-used PWR training simulator.
Citation: Zhao et al., *An open time-series simulated dataset covering various accidents
for nuclear power plants*, Scientific Data (2022), https://doi.org/10.1038/s41597-022-01879-1

## What's here

Seven CSV files pulled from `Operation_csv_data/` on 2026-07-07, **truncated to the
first 400 rows** (or fewer if the source file is shorter) — 10-second sampling, so
≤4000 s per file. The full files (up to ~700 rows) are available in the upstream repo;
truncation keeps these fixtures small while covering the interesting pre/post-scram
window for every case.

| File | Accident | Severity | Ground truth (from upstream `*Transient Report.txt`) |
|---|---|---|---|
| `Normal_1.csv` | Normal operation | — | Steady state, no events. Used for detector calibration + false-alarm check. |
| `LOCA_1.csv` | Hot-leg LOCA | 1% of 100 cm² | Malfunction at t=0.5s. Automatic reactor scram at **t=2032.5s**. |
| `LOCA_50.csv` | Hot-leg LOCA | 50% of 100 cm² | Malfunction at t=0.5s. Automatic reactor scram at **t=127.5s**. |
| `RW_1.csv` | Rod withdrawal | 1% | Malfunction at t=0.5s. Automatic reactor scram (high flux) at **t=403.5s**. |
| `TT_1.csv` | Turbine trip | — | Trip at t=0.5s. Handled by relief valves; **no reactor scram** in this window. |
| `SGATR_1.csv` | SG-A tube rupture | 1% of 1 full rupture | Malfunction at t=0.5s. **No scram** in this window (mild severity). |
| `ATWS_1.csv` | Anticipated Transient Without Scram | — | Malfunction at t=0.5s. **By construction, no scram ever fires** — this is the case for which automatic protection does not act, making it the most direct test of independent advisory monitoring. |

All files share the identical 97-column header (verified by hash before truncation).

## Why these seven

Deliberately spans: a slow accident (LOCA/1, scram at ~34 minutes) vs a fast one
(LOCA/50, ~2 minutes), a reactivity insertion (RW), a secondary-side event with no
scram (TT), a radiological pathway (SGATR), and the one case where the plant's own
automatic protection is defined to fail (ATWS) — the strongest argument for why
advisory monitoring has independent value even in a fully protected plant.

## Honesty notes

- This validates the *detection method* (calibrated HDC free-energy + kσ/persistence)
  against real PWR-simulator dynamics, not the synthetic fault signatures in
  `fission_bench.rs` — the channel selection and normalization here are independent,
  see `src/nppad_validation.rs`.
- Only one `Normal` run exists upstream; calibration and the false-alarm check both
  come from splitting this single continuous run, not independent baselines.
- PCTRAN output is a deterministic ODE/lookup-table simulation — it has none of the
  instrument noise a real plant would show. A detector tuned here may need a wider
  margin against real sensor noise.
