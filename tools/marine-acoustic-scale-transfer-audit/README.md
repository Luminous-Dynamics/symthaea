# Marine acoustic scale-transfer audit

A standalone Rust audit for a narrow, versioned model-scale to full-scale cavitation-noise comparison.

## Run

```sh
cargo test --manifest-path tools/marine-acoustic-scale-transfer-audit/Cargo.toml
cargo run --manifest-path tools/marine-acoustic-scale-transfer-audit/Cargo.toml --   docs/research/marine/fixtures/synthetic-acoustic-scale-transfer-v0.1.0.json
```

Exit code 0 means the numerical comparison is complete and within the declared uncertainty, **but independent physical review is still required**. Exit code 2 means incomplete, structurally invalid, or outside the declared uncertainty. Exit code 64 means invocation or input loading failed.

The committed fixture is synthetic and must return `INCOMPLETE` with `NOT_EVALUATED_SYNTHETIC` physical-validation status. It is a contract and arithmetic test vector, not model-test or sea-trial evidence.

## What v0.1 calculates

For the explicitly selected ITTC 18th-committee scale-law family:

- Frequency ratio (ITTC Eq. 15): `(n_s / n_m) * sqrt(sigma_s / sigma_m)`.
- SPL correction (ITTC Eq. 16): exponent profile selected by bandwidth convention and acoustic-efficiency regime.
- Per-band residual: observed full-scale source level minus extrapolated model-scale source level.
- Per-band background-noise correction in the energy domain, followed by the per-band facility transfer-function correction to source level.
- Combined expanded uncertainty: conservative linear bound by default; root-sum-square only when an independence evidence reference is supplied. Each input is explicitly an expanded uncertainty at the band’s declared confidence level.

The four supported Eq. 16 profiles are fixed in code. The caller cannot replace exponents while retaining a known profile ID.

## Eligibility boundary

The audit requires distinct model-RPM strata, revision identity, positive scale inputs, calibration/facility-transfer evidence digests, per-condition background traces, wake/cavitation/water-quality evidence, declared model/full-scale similarity states, full-scale source-level basis, propagation separation, and explicit per-band uncertainty. The transfer function is supplied as a per-band dB value and applied by the audit after background subtraction.

The output deliberately separates:
- `structural_disposition`: whether declared inputs are complete enough to calculate;
- `comparison_disposition`: whether the per-band residuals fall inside the declared combined uncertainty;
- `physical_validation_disposition`: always `NOT_EVALUATED_SYNTHETIC` or `REQUIRES_INDEPENDENT_PHYSICAL_REVIEW`.

The CLI never labels a physical campaign as validated. Evidence hashes are only syntactic digest identifiers; this executable does not authenticate certificates, verify raw trace bytes, or independently prove declared model/full-scale similarity.

Background subtraction uses the paired signal-plus-background and background SPL values for each band:
- below 3 dB margin: unusable for Eq. 16 scaling;
- 3 dB to below 10 dB: energy-domain subtraction is required;
- 10 dB or above: subtraction is not required (but may be declared and calculated).
The supplied margin is cross-checked against the two measured levels. See ITTC Rev. 03, §2.5.3.

Version 0.1 accepts exact frequency-bound scaling only. It does not interpolate or re-bin spectra. Proportional bandwidth requires a power spectrum; power spectral density is not silently substituted. Cavitation patterns SHEET, CLOUD and BUBBLE are eligible only when model/full-scale pattern, wake and water-quality comparisons are explicitly MATCHED. TIP_VORTEX and MIXED remain INCOMPLETE because the additional Reynolds-number scale correction is not implemented; ITTC says the proposed tip-vortex acoustic corrections need further validation.

The model-scale and full-scale comparison must use source levels normalized to the same 1 m reference distance. For the model, the audit first subtracts background (where required), then applies the measured facility transfer-function value to obtain source level. The full-scale value must already have propagation separated and be reduced to source level. Accordingly, v0.1 asserts both Eq. 16 reference distances are 1 m (so r_m/r_s = 1) and refuses raw hydrophone SPL/received-level inputs. This avoids applying a second, inconsistent range correction. Received levels or unlike radiated-noise quantities are not interchangeable. The synthetic fixture remains incomplete even when every reference resolves.

## Limitations

- The executable audits a declared transfer calculation; it does not prove the physical truth of the declared evidence or authenticate certificates. Evidence digests are format-checked identifiers, not certificate-chain verification or raw-trace verification. Numerical agreement is not physical validation.
- v0.1 consumes evidence digests and explicit correction declarations; it does not yet resolve the calibration/timebase registries or the shared acoustic/propulsion timebase receipt from #11641. That integration is a prerequisite before the audit is used in a real campaign.
- The ITTC 4.2–7.1 dB full-scale prediction uncertainty reported in Revision 03 is based on questionnaire estimates, not a universal per-project acceptance tolerance. This tool does not hard-code it.
- The audit checks declared similarity dispositions and digests; it does not establish similarity from the evidence itself. Cavitation-pattern similarity, Reynolds effects, wake fidelity, nuclei/gas content, facility reverberation, background subtraction and transfer-function characterization still require independent engineering review and experimental evidence.
- The audit does not yet resolve the calibration/timebase registries or consume the shared acoustic/propulsion timebase receipt from #11641. That integration, plus raw-trace digest verification, is a prerequisite for real campaign use.
- No CFD solver, ecological impact model, vessel certification or causal improvement claim is implemented.
- The standalone workflow generates one Cargo.lock per exact-head run, uses `--locked` for Clippy/tests/audit, and preserves the lockfile as a workflow artifact. Commit that generated lockfile before treating dependency resolution as repository-frozen.
- Exact-head Rust tests/format/clippy and physical validation must be reported separately.

## External benchmark targets

The first multi-facility target is the [ITTC Round Robin Test — Cavitation and Noise](https://ittc.info/benchmark-repository/round-robin-test-cavitation-and-noise/). The ITTC page was updated 2 July 2026 and gives a 31 January 2027 results deadline. The no-rudder case is mandatory; the rudder case is optional. The page provides geometry, a result template, a test presentation and agreement materials.

This is a **model-scale inter-facility reproducibility benchmark**, not by itself a model-to-full-scale validation dataset. Use it to assess facility/operator spread, cavitation observations, spectrum processing and declared uncertainty before claiming transfer accuracy. ITTC's general benchmark repository explicitly warns that it does not take responsibility for data property or link maintenance; raw-data availability and reuse rights must still pass the project's source-admission process.

Full-scale transfer comparators to investigate separately:
- Lee et al. (2024), *Prediction of cavity inception speed and underwater radiated noise of a full-scale marine propeller based on a cavitation tunnel model test*, DOI 10.1016/j.oceaneng.2024.118456.
- Gaggero et al. (2025), *Propeller design by optimisation for underwater radiated noise reduction: a full scale application*, DOI 10.1016/j.oceaneng.2025.121412.
- Khraisat et al. (2026), *Cavitation underwater radiated noise methodology applied to a propeller operating in-behind condition*, DOI 10.1016/j.oceaneng.2025.124084. The published abstract reports agreement for some pressure-pulse/cavity metrics but significant discrepancies in URN predictions, including sensitivity to FW-H surfaces/interfaces and tunnel blockage. This is a valuable falsification case, not a universally validated solver baseline.

No dataset is admitted merely because a paper or benchmark page exists. Obtain the raw bytes, verify their digest, review reuse rights, bind units/measurement definitions and uncertainty, and freeze calibration/hold-out groups before evaluation (see #8146).
