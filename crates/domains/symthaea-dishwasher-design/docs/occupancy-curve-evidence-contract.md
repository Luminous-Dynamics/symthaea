# Evidence contract for measured dishwasher occupancy curves

Status: research implementation contract, 3 November 2026. The contract is designed to prevent unmeasured or quality-failing data from silently becoming a scheduler cost curve. It does not authenticate instruments, certify cleaning, or establish statistical confidence by itself.

## 1. Unit of observation

One physical run is one complete execution of one exact machine/programme/load configuration at a declared occupancy and soil/surface protocol.

Never treat:
- multiple image frames from one wash as independent physical replicates;
- repeated sensor samples within one cycle as separate runs;
- a manufacturer's label as an internal partial-load measurement;
- one mixed-soil load as equivalent to the separate soil-family curves;
- a cleaning image score as a rinse-residue or sanitization assay.

The run archive is one row per independent cycle, plus immutable links to time-series, images, assay records and calibration certificates.

## 2. Fixed curve scope

A single `CurveScope` freezes:
- exact machine SKU/model and regional variant;
- exact programme and options;
- soil family and soil batch;
- soil age at wash start;
- surface material and fixed load-set identity;
- detergent product/batch and dose protocol;
- incoming water batch, hardness as CaCO3 and inlet temperature;
- ambient temperature and relative humidity;
- frozen test protocol revision;
- SHA-256 digest of the raw-data archive.

A change to any scope field requires a separate curve or an explicitly designed multi-factor model. Do not pool data across detergent batches, machine SKUs, soil ages, hardness or ambient conditions and call it one occupancy curve.

## 3. Per-run minimum fields

The repository template is `experiments/measurement-template.csv`. It records:
- globally unique run ID and independent replicate ID;
- protocol, timestamp, machine, programme, occupancy and load set;
- soil batch / age, material, detergent batch / dose;
- water batch / hardness / inlet temperature;
- ambient temperature / relative humidity;
- branch pressure / flow and pump electrical measurements;
- separated pump, heater, controls, drying, standby and total real wall energy;
- fresh and recirculated water volumes and elapsed duration;
- initial and residual soil mass;
- cleaning pass + immutable ground-truth reference;
- rinse pass + immutable ground-truth reference;
- drying endpoint and cross-contamination endpoint where required;
- calibration IDs, a calibration-bundle SHA-256 and raw-run digest.

Use a real-power/energy analyser. Voltage multiplied by current is not a valid substitute for real energy on non-unity-power-factor motor/drive loads. Missing data are empty, never zero.

## 4. Rust construction path

The intended API is:

```rust
let curve = MeasuredOccupancyCurve::from_runs(
    scope,
    vec![
        (1, one_item_physical_runs),
        (4, four_item_physical_runs),
        (8, eight_item_physical_runs),
    ],
)?;
```

The public constructor accepts run-level observations. Each run carries:
- unique run ID;
- cleaning and final-rinse pass calls, grounded in independent assay/coupon evidence;
- total wall real energy, fresh water and total process duration;
- cleaning and rinse evidence references;
- calibration-bundle SHA-256.

The API derives minimum, mean and maximum from the run values. It rejects blank or duplicate IDs, missing evidence references, malformed calibration digests, non-finite/negative resource observations and any cleaning/rinse failure. Run IDs must be unique across the entire curve, not merely within one occupancy.

The private summary constructor cannot be called by downstream consumers to inject hand-entered averages through the supported API.

## 5. Replicates and statistics

Five distinct physical runs per occupancy is the minimum **pilot** threshold copied from issue #10136. It is not a statistical-power guarantee and is not an acceptable confirmatory sample size by default.

Use pilot variance and a preregistered minimum effect size to calculate confirmatory sample size. Randomize run order, block by day/soil preparation where appropriate, and hold out complete days or batches for model evaluation.

The API's min/mean/max are descriptive observed ranges. They are **not** confidence intervals, prediction intervals or a bootstrap guarantee. Interpolation between measured occupancies is diagnostic only; out-of-range estimates are rejected. Confirm every target occupancy physically and evaluate any interpolation model on held-out occupancy points.

## 6. Quality failures and denominators

A curve eligible for scheduling must have zero cleaning and rinse failures at every occupancy point. A failed run must not be quietly removed and replaced by a better run. Preserve every failure in the raw archive, report failure rate and resources consumed, and mark that occupancy ineligible until the process is corrected and independently retested.

For system-level reporting, retain:
- all attempted runs;
- all completed-but-failed runs;
- all rewash / unprocessed items;
- all consumed energy, water and detergent.

Do not calculate resource per successful clean by dropping the resources of failed cycles.

Drying, hygiene and sanitization are separate endpoints. A curve that passes cleaning/rinse does not imply those other claims.

## 7. Provenance limits

Run IDs and SHA-256 digests protect identity and detect content changes when independently recomputed. They do not establish that:
- a run was physically independent;
- the meter was calibrated correctly;
- the assay was performed correctly;
- the evidence reference actually contains the claimed result;
- the data were not fabricated before hashing.

Keep instrument certificates, lab notebook, operator, randomization record, raw meter export, coupon image/assay, machine firmware and analysis commit in the external research archive. Perform an independent audit before using a curve for product or market decisions.

## 8. Scheduler integration gate

Do not replace illustrative linear `ModeProfile` coefficients until:
1. two exact machine/programme combinations have occupancy curves;
2. occupancy includes at least 1, 2, 4, 6, 8, 10 and full capacity where feasible;
3. cleaning/rinse endpoints pass at every occupancy included in an eligible curve;
4. held-out occupancy predictions meet preregistered error and uncertainty limits;
5. measured time, energy and fresh-water curves include warm-up, standby, flush, drying and upstream hot-water energy where applicable.

Issue #10136 is the physical measurement programme. The current Rust API is only a guarded data/model interface; it contains no measured dishwasher data.
