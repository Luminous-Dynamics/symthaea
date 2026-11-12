# Plasma Energy Audit — Offline CLI and fixture guide

**Implementation:** [PLASMA-ENERGY-AUDIT-001 #11751](https://github.com/Luminous-Dynamics/symthaea/issues/11751)  
**Parent:** [PLASMA-ENERGY-001 #11718](https://github.com/Luminous-Dynamics/symthaea/issues/11718)  
**Date:** 2026-11-12  
**Claim ceiling:** source metadata and declared synthetic accounting only; no solver, thermal model, physical result or technology superiority.

## 1. Command

From the Symthaea workspace root:

```sh
cargo run -p xtask -- plasma-energy-audit
```

Emit structured JSON:

```sh
cargo run -p xtask -- plasma-energy-audit --json
```

Paths may be overridden:

```sh
cargo run -p xtask -- plasma-energy-audit \
  --plasma-sources data/research/plasma_energy_reference_cases_v0_1.json \
  --hts-sources data/research/hts_fusion_magnet_reference_cases_v0_1.json \
  --thermal-fixtures data/research/thermal_store_accounting_fixtures_v0_1.json
```

The command is offline and uses existing `xtask` dependencies. It does not invoke WarpX, Zapdos, PROCESS, NekRS, SAM, THEA, a plasma power supply or a thermal solver.

## 2. Source-register audit

The source audit requires:
- an HTTP(S) source URL with a non-empty authority (scheme-only values such as `https://` are rejected);
- a non-empty programme and claim ceiling;
- declared, unique evidence classes;
- at least one uniquely identified source case and negative control;
- every case's evidence class to appear in the declared set;
- a source URL using HTTP(S);
- an explicit `luminous_reproduction=false` field on every case;
- an expected disposition for each negative control;
- every `FUSION_DIRECT_ENERGY_CONVERSION` and `FUSION_FUEL_ENERGY_PARTITION` case has `particle_energy_origin`, `fusion_born_output_demonstrated=false`, and `comparison_status`.

This is metadata integrity only. The DEC-specific gate records that fusion-born electrical output was not demonstrated/attributed in the source case. This does not assert that a physical fusion-born contribution was exactly zero; unknown contributions remain unquantified. It prevents externally heated plasma recovery or a laboratory fundamental-unit experiment being silently promoted to demonstrated fusion-born energy recovery. It does not independently check paper contents, reproduce reported values, or establish that a source supports every interpretation in its case description.

Current frozen input:
- plasma-energy register: 28 cases / 28 controls, including five direct-energy-conversion cases, one D–³He/DT fuel-mixture model, four miniature/modular low-temperature plasma device references, and thermal-processing/ETES comparison references;
- HTS fusion magnet register: 6 cases / 12 controls.

## 3. Synthetic thermal-store fixture

The fixture is `data/research/thermal_store_accounting_fixtures_v0_1.json`. It uses integer joules and an explicitly synthetic evidence class.

Each scenario has:
- a unique scenario ID and technology label;
- a functional heat-service boundary (service identity/type, outlet temperature, optional pressure/flow, delivery duration, trace digest);
- an exact wall-plug energy-boundary contract identity, including which conversion and auxiliary systems are inside the meter perimeter;
- a metering mode;
- wall-plug meter inputs with unique input and meter-boundary identities;
- explicit primary-process and auxiliary meter roles, or one exclusive total meter;
- stored-energy increase, charge conversion losses, auxiliary dissipation;
- energy after the storage hold and hold losses;
- stored energy withdrawn, useful delivered heat and discharge losses;
- expected disposition.

The wall-plug boundary contract ID is mandatory and participates in comparability. Two scenarios can each close their own accounting yet remain NOT_COMPARABLE when their input metering perimeters differ. The branch-meter rule is intentionally strict: an auxiliary meter must be present. A true zero auxiliary draw must be represented by an explicit meter observation of zero. An absent meter or unknown energy is not equivalent to zero.

### Accounting identities

`Primary-process electrical input = stored-energy increase + charge conversion loss`

`Auxiliary electrical input = declared auxiliary dissipation`

`Stored-energy increase = energy after hold + storage hold loss`

`Energy withdrawn = useful heat delivered + discharge loss`

`Total wall-plug input = useful heat delivered + charge loss + auxiliary dissipation + hold loss + discharge loss`

An exclusive total meter can be used instead of branch meters, but cannot be summed with branch meters. Duplicate meter boundaries are rejected.

Only after all gates pass is the derived fraction computed:

`wall-plug-to-useful-heat = useful heat delivered / total wall-plug input`

It is not electricity-to-electricity round-trip efficiency, and it is not a thermodynamic exergy efficiency.

## 4. Dispositions

| Disposition | Meaning |
|---|---|
| `PASS` | Declared identities, boundary, meters and energy equations are complete and close |
| `BLOCKED` | Required evidence or service boundary is unknown/incomplete |
| `REJECTED` | Identity, meter scope, unit or accounting invariant is violated |
| `COMPARABLE` | Both scenarios pass and their entire service contracts match exactly |
| `NOT_COMPARABLE` | Valid individual scenarios do not deliver the same declared service |

A `PASS` on a synthetic case does not reproduce an experiment. A `COMPARABLE` disposition means only that the frozen service and wall-plug energy-boundary contracts match; it does not establish technical merit or commercial competitiveness.

## 5. Frozen synthetic reference results

The current fixture deliberately contains:
- a resistance-heated synthetic baseline;
- a plasma-heated synthetic candidate;
- a separate 600°C service candidate;
- unknown auxiliary input;
- an absent auxiliary meter;
- total/branch meter overlap;
- duplicate meter boundary;
- charge, hold and discharge balance defects;
- an incomplete heat-service boundary;
- unknown energy unit;
- a valid exclusive-total-meter scenario.

The same-service reference rows currently compute:
- resistance fixture: 732,804 ppm (73.2804%);
- plasma fixture: 675,609 ppm (67.5609%).

These values are invented test inputs used to exercise arithmetic and comparability. They are **not** measured or predicted technology performance, are not an estimate of plasma efficiency, and must never enter an industry leaderboard.

## 6. Test commands and claim limits

When a Rust-capable environment is available, run:

```sh
cargo fmt --check
cargo test -p xtask plasma_energy_audit
cargo clippy -p xtask --all-targets -- -D warnings
```

The draft PR authors 19 Rust tests, including tests against the frozen source and thermal fixtures, duplicate-identity controls and DEC energy-origin refusal tests. At the time this guide was written, the available execution environment did not contain `cargo` or `rustc`; therefore no Rust test, format, compile or Clippy pass is claimed.

The evaluator does not:
- solve plasma, MHD, fusion, heat-transfer or electrochemical equations;
- estimate thermal material properties or degradation;
- validate a laboratory paper's numerical results;
- authorize a heater, high-voltage/RF system, quench test or physical experiment;
- determine that plasma, resistance or another process is superior;
- replace Symtropy's deterministic run receipts or external solver capability profiles.

The next validation step is to execute the exact-head Rust tests on a Rust-capable host, then proceed to source-specific numerical reproduction only after raw source data, units, boundaries, rights and specialist-solver capability are admitted.


## D–³He fuel-mixture and direct-conversion boundary

The plasma source register includes a 2026 numerical ARC-class tokamak fuel-mixture study comparing D–T with 50% D / 50% ³He. It reports a 525 MW model comparison with source-specific temperatures/densities, average energy per reaction and neutron fractions. A derived arithmetic check records the modeled charged-energy fractions and their ratio.

This is not a DEC experiment. The D–³He primary reaction produces charged alpha/proton products, but the D–³He fuel mixture still produces neutrons through D–D and secondary D–T reactions. The fuel-partition record is explicitly origin-tagged and keeps `fusion_born_output_demonstrated=false` because it contains no direct-converter experiment. Its much lower modelled neutron share is bought with a much higher temperature/density point and unresolved ignition/stability requirements in the source model. The charged-energy fraction must not be promoted to a converter efficiency or plant net electricity.


## Miniaturized and modular plasma device source cases

The source register now adds four external laboratory references under `LOW_TEMPERATURE_PLASMA_MINIATURIZATION` and `LOW_TEMPERATURE_PLASMA_NUMBERING_UP`:

- The 2026 low-cost helium APPJ reference reports a 100 g / 15 × 10 × 2 cm power supply, 54 mm glass tube, 2 L/min helium flow, less than USD 45 component cost and source-specific in-vitro results. The source explicitly leaves long-term electrode wear/drift and practical treatment-area limits unresolved.
- The 2025 membrane-stabilized microfluidic DBD chip reports 7.31–11.32 nm gold nanoparticle sizes with 1.8 nm standard deviation. This does not establish industrial throughput or whole-system energy performance.
- The 43-element LTCC CAP array demonstrates a 10 × 10 cm treatment region, with a source-specific 91% CFU decrease on a named biofilm coupon under its conditions.
- The 2023 RF microplasma array demonstrates five individually ignitable/quenchable cells; expansion to tens or hundreds of cells is discussed by the authors but not a demonstrated scale-out result.

The new negative controls prevent low parts cost from becoming lifecycle qualification, particle-size precision from becoming industrial throughput, one biofilm endpoint from becoming universal efficacy, and a five-cell demonstration from becoming a hundred-cell array claim.

The programme is [MICROPLASMA-001 #11780](https://github.com/Luminous-Dynamics/symthaea/issues/11780).
