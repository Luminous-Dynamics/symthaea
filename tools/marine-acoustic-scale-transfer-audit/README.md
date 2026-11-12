# Marine acoustic scale-transfer audit

A standalone Rust audit for a narrow, versioned model-scale to full-scale cavitation-noise comparison.

## Run

```sh
cargo test --manifest-path tools/marine-acoustic-scale-transfer-audit/Cargo.toml
cargo run --manifest-path tools/marine-acoustic-scale-transfer-audit/Cargo.toml --   docs/research/marine/fixtures/synthetic-acoustic-scale-transfer-v0.1.0.json
```

Exit code 0 means the declared measured pair is within its stated uncertainty. Exit code 2 means incomplete, structurally invalid, or outside the declared uncertainty. Exit code 64 means invocation or input loading failed.

The committed fixture is synthetic and must return `INCOMPLETE`. It is a contract and arithmetic test vector, not model-test or sea-trial evidence.

## What v0.1 calculates

For the explicitly selected ITTC 18th-committee scale-law family:

- Frequency ratio (ITTC Eq. 15): `(n_s / n_m) * sqrt(sigma_s / sigma_m)`.
- SPL correction (ITTC Eq. 16): exponent profile selected by bandwidth convention and acoustic-efficiency regime.
- Per-band residual: observed full-scale source level minus extrapolated model-scale source level.
- Combined expanded uncertainty: conservative linear bound by default; root-sum-square only when an independence evidence reference is supplied.

The four supported Eq. 16 profiles are fixed in code. The caller cannot replace exponents while retaining a known profile ID.

## Eligibility boundary

The audit requires distinct model-RPM strata, revision identity, positive scale inputs, explicit facility transfer-function and background corrections, wake/cavitation/water-quality evidence digests, full-scale source-level basis, propagation separation, and explicit per-band uncertainty.

Version 0.1 accepts exact frequency-bound scaling only. It does not interpolate or re-bin spectra. A band is eligible only when its model signal is above background.

The model-scale and full-scale comparison must use source levels. Received levels or unlike radiated-noise quantities are not interchangeable. The synthetic fixture remains incomplete even when every reference resolves.

## Limitations

- The executable audits a declared transfer calculation; it does not prove the physical truth of the declared evidence or authenticate certificates.
- The ITTC 4.2–7.1 dB full-scale prediction uncertainty reported in Revision 03 is based on questionnaire estimates, not a universal per-project acceptance tolerance. This tool does not hard-code it.
- Cavitation-pattern similarity, Reynolds effects, wake fidelity, nuclei/gas content, facility reverberation, background subtraction and transfer-function characterization require independent engineering review and experimental evidence.
- No CFD solver, ecological impact model, vessel certification or causal improvement claim is implemented.
- Exact-head Rust tests/format/clippy and physical validation must be reported separately.

## External benchmark targets

The first multi-facility target is the [ITTC Round Robin Test — Cavitation and Noise](https://ittc.info/benchmark-repository/round-robin-test-cavitation-and-noise/). The ITTC page was updated 2 July 2026 and gives a 31 January 2027 results deadline. The no-rudder case is mandatory; the rudder case is optional. The page provides geometry, a result template, a test presentation and agreement materials.

This is a **model-scale inter-facility reproducibility benchmark**, not by itself a model-to-full-scale validation dataset. Use it to assess facility/operator spread, cavitation observations, spectrum processing and declared uncertainty before claiming transfer accuracy. ITTC's general benchmark repository explicitly warns that it does not take responsibility for data property or link maintenance; raw-data availability and reuse rights must still pass the project's source-admission process.

Full-scale transfer comparators to investigate separately:
- Lee et al. (2024), *Prediction of cavity inception speed and underwater radiated noise of a full-scale marine propeller based on a cavitation tunnel model test*, DOI 10.1016/j.oceaneng.2024.118456.
- Gaggero et al. (2025), *Propeller design by optimisation for underwater radiated noise reduction: a full scale application*, DOI 10.1016/j.oceaneng.2025.121412.
- Khraisat et al. (2026), *Cavitation underwater radiated noise methodology applied to a propeller operating in-behind condition*, DOI 10.1016/j.oceaneng.2025.124084. The published abstract reports agreement for some pressure-pulse/cavity metrics but significant discrepancies in URN predictions, including sensitivity to FW-H surfaces/interfaces and tunnel blockage. This is a valuable falsification case, not a universally validated solver baseline.

No dataset is admitted merely because a paper or benchmark page exists. Obtain the raw bytes, verify their digest, review reuse rights, bind units/measurement definitions and uncertainty, and freeze calibration/hold-out groups before evaluation (see #8146).
