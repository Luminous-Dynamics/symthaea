# Tire Monitoring Dataset

A large-scale multi-scenario tire monitoring dataset with synchronized
pressure, temperature, and speed measurements, collected over 18 months
of real-world driving under diverse weather and road surface conditions.

## Citation

To be updated upon publication.

## File

`tire_monitoring_dataset.csv` — 210,128 samples
(196,260 normal / 13,868 anomalous)

## Fields

| Field | Description |
|---|---|
| `Unit_ID` | Acquisition segment identifier. Samples with the same `Unit_ID` come from one continuous driving run and are kept within a single subset when splitting the data at the segment level. |
| `Time_Step` | Sampling index within a segment, corresponding to the 15-s reporting interval. Absolute timestamps are not included in the released data. |
| `Pressure` | In-cavity tire gauge pressure (kPa) |
| `Temperature` | In-cavity gas temperature (°C) |
| `Speed` | Vehicle speed (km/h), computed from the sensor-measured wheel rotation period |
| `Label` | 0 = normal, 1 = anomalous |
| `Collection_Method` | `Experimental` — controlled deflation experiment on closed test roads (219 segments)<br>`Natural` — natural leakage event recorded during everyday driving on public roads (12 segments)<br>`Normal` — normal acquisition segment from everyday driving on public roads, containing no anomaly |

## Descriptive Statistics

| Variable | Unit | Min | Max | Mean | Std Dev |
|---|---|---|---|---|---|
| Tire pressure | kPa | 197.4 | 289.3 | 252.9 | 11.6 |
| Tire temperature | °C | 1.6 | 45.9 | 25.2 | 6.1 |
| Vehicle speed | km/h | 2.8 | 144.9 | 85.6 | 31.1 |

## Data Acquisition

- Sensing terminal: in-tire MEMS sensing unit (Model S4.1M) integrating an
  Infineon SP49 sensing chip and a Silicon Labs BG22 Bluetooth main
  control chip.
- Tires: 205/55 R16 passenger car tires (Hankook Ventus H432).
- Each record corresponds to one 15-s data packet transmitted via
  Bluetooth Low Energy. Pressure and temperature are smoothed on the
  sensor by a 5-s sliding-average filter applied once per reporting
  cycle, so adjacent samples are temporally non-overlapping.
- Short single-sample gaps caused by wireless transmission loss are
  filled by linear interpolation. Gaps longer than 60 s are treated as
  segment boundaries and are not interpolated.

## Notes

- The dataset contains 231 anomaly segments in total
  (219 controlled deflation experiments + 12 natural leakage events).
- In each controlled deflation experiment, the tire starts in a properly
  inflated state and slow deflation is manually triggered during driving,
  with a target deflation rate of approximately 2.2–2.7 kPa/min.
  Samples from the first reporting instant after the trigger are labeled
  as anomalous; samples acquired before the trigger are retained as
  normal. A single `Experimental` segment therefore contains both
  `Label` 0 and `Label` 1.
- The natural leakage events show leakage rates of 2.28–2.48 kPa/min,
  and the controlled deflation experiments show 2.23–2.64 kPa/min.
- The safety threshold is 200 kPa. Experiments are manually terminated
  once the tire pressure reaches this threshold; due to response
  latency, the pressure at termination may fall slightly below 200 kPa.
- This dataset targets slow and progressive pressure anomalies (e.g.,
  gradual leakage). Instantaneous events such as blowouts are not
  included.
- Recommended usage: split the data at the segment level (by `Unit_ID`)
  so that samples from the same segment do not appear in more than one
  subset. The associated paper uses an 8:1:1 train/validation/test split,
  applied separately to anomaly segments and normal segments.

## License

Released under [CC BY 4.0](LICENSE).
