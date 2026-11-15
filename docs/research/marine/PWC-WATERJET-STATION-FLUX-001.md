# PWC-WATERJET-STATION-FLUX-001
## Discrete axial energy-flux integration for propulsion-test stations 1 and 7

**Status:** v0.1.0 synthetic reference implementation  
**Parent work:** PWC-WATERJET-INSTALLED-ENERGY-001 #12232  
**Pump-loop precursor:** PWC-WATERJET-ENERGY-REFERENCE-001 #12231  
**Physical scenario owner:** PWC-WATERJET-DYNAMICS-001 #12135

> This is a synthetic arithmetic/continuity reference only. It is not a waterjet solver, test-standard certification, installed propulsion model, PWC performance result, or safety case.

## 1. Research shift: compute the measured field, not a convenient average

The propulsion-test energy flux is a spatial integral across the capture and discharge stations. A single average speed can lose the local (u^3)-weighted kinetic-energy contribution, especially for distorted inlet profiles or non-uniform nozzle flow.

The v0.1 auditor therefore integrates every declared cell in each section. It preserves:
- local axial velocity (u_\xi);
- signed axial normal projection (n_\xi);
- static pressure (p);
- elevation (z_s);
- differential cell area (dA);
- common density, pressure-reference datum and elevation datum.

This is a discrete quadrature of the ITTC equation, not an approximation that replaces the measured profile with a bulk mean.

## 2. Method and source applicability

Primary source: ITTC 7.5-02-05-03.1, *Waterjet Propulsive Performance Prediction – Propulsion Test and Extrapolation*, Effective Date 2011, Revision 02.

The procedure:
- covers self-propulsion experiments and model-to-full-scale extrapolation;
- is generally applicable to waterjet vehicles with flush intakes, with adaptation needed for other arrangements;
- calls the propulsion method the momentum-flux method;
- stresses that overall powering uncertainty is sensitive to flow-rate measurement error;
- requires the capture-area velocity distribution and its geometry, with the capture area not reducible to an assumed nozzle mean;
- defines effective jet-system power from the net energy-flux difference at stations 7 and 1.

Official source: [ITTC 7.5-02-05-03.1 (2011 Rev.02)](https://www.ittc.info/media/8067/75-02-05-031.pdf)

The ITTC procedure uses station 7 to define the system boundary. It permits station 6 only when the parallel-nozzle-outflow assumption justifies treating the vena contracta diameter as equal to the nozzle discharge diameter. This v0.1 auditor always uses station 7; it has no station-6 shortcut mode.

## 3. Exact calculation contract

ITTC equation (11) defines total axial energy flux at station (s):

\[
E_{s\xi}=\int_{A_s}\rho\left(\frac{1}{2}u_\xi^2+\frac{p}{\rho}-g z_s\right)(u_\xi n_\xi)\,dA
\]

The implemented equivalent integrand is:

\[
\left(\frac{1}{2}\rho u_\xi^2+p-\rho g z_s\right)
(u_\xi n_\xi)\,dA
\]

For quadrature cells (i=1..N_s):

\[
Q_s=\sum_i u_{\xi,i}n_{\xi,i}A_i
\]

\[
E_{s\xi}=\sum_i
\left(\frac{1}{2}\rho u_{\xi,i}^{2}+p_i-\rho g z_i\right)
u_{\xi,i}n_{\xi,i}A_i
\]

and the effective jet-system power is:

\[
P_{JSE}=E_{7\xi}-E_{1\xi}
\]

The sign convention is explicit: (n_\xi) is the signed projection of each cell's surface normal onto the declared local axial direction. The same coordinate convention is used at both stations. A reversed normal is not silently converted to its absolute value.

### Shared reference and time

Both stations must declare the same pressure-reference ID and elevation-datum ID. Pressure may be absolute or gauge, but its reference must be explicit and common. Elevation values must use a common vertical datum. Every observation must fall in the declared measurement window and be available by the evaluation time. The record also declares a maximum observation-time skew across density, both station planes and every cell channel. If the observed skew exceeds that limit, the comparison is INCOMPARABLE rather than treating different transient states as one station-to-station power balance.

### Continuity gate

Station flow is integrated independently at station 1 and station 7. Before reporting (P_{JSE}), their nominal flow difference and conservative uncertainty interval must fit the predeclared absolute-plus-relative tolerance:

\[
|Q_7-Q_1| \leq \epsilon_{abs}+\epsilon_{rel}\max(|Q_1|,|Q_7|)
\]

If nominal flow is mismatched, the comparison is INCOMPARABLE. If the nominal passes but the uncertainty interval crosses the tolerance, the result is INCOMPLETE. If the declared direction produces non-positive flow, it is INCOMPARABLE. No absolute-value repair or hidden flow normalization is permitted.

## 4. Deterministic non-uniform known-answer fixture

All data are synthetic. Density is (1000\,\mathrm{kg/m^3}), pressure is (101325\,\mathrm{Pa}), elevation is zero, and (n_\xi=1).

Station 1:
- cell A: (u_\xi=1\,\mathrm{m/s}), (A=0.5\,\mathrm{m^2});
- cell B: (u_\xi=3\,\mathrm{m/s}), (A=0.5\,\mathrm{m^2});
- integrated (Q_1=2\,\mathrm{m^3/s});
- energy flux (E_{1\xi}=209650\,\mathrm{W}).

Station 7:
- cell A: (u_\xi=8\,\mathrm{m/s}), (A=0.125\,\mathrm{m^2});
- cell B: (u_\xi=12\,\mathrm{m/s}), (A=1/12\,\mathrm{m^2});
- integrated (Q_7=2\,\mathrm{m^3/s});
- energy flux (E_{7\xi}=306650\,\mathrm{W}).

Therefore:

\[
P_{JSE}=306650-209650=97000\,\mathrm{W}
\]

This known-answer case deliberately has non-uniform velocities to catch implementations that collapse the profile into one mean velocity before evaluating kinetic-energy flux.

## 5. Fail-closed adversarial controls

- **Flow mismatch:** station 7 area is perturbed so integrated (Q_7\ne Q_1). Disposition is INCOMPARABLE; (P_{JSE}) is null.
- **Reversed normal:** station-7 (n_\xi) is set to (-1). Signed flow becomes negative and the result is INCOMPARABLE; no absolute-value correction is applied.
- **Pressure-reference mismatch:** INVALID.
- **Elevation-datum mismatch:** INVALID.
- **Future/unavailable measurement:** INVALID.
- **Duplicate cell identity, non-positive area, invalid (n_\xi), bad units, or reversed uncertainty bounds:** INVALID.
- **Non-positive power difference or an uncertainty interval that crosses zero:** BLOCKED or INCOMPLETE rather than clipped.

The report deliberately provides no NVR, ideal efficiency, interaction efficiency, ducting efficiency, overall efficiency, or hull-waterjet efficiency. Those quantities require additional distinct inputs/tests and are not established by stations 1 and 7 alone.

## 6. SOTA and engineering benchmark opportunity

Recent adjacent research (not PWC validation) reinforces why field-resolved measurements matter:

- Jiao et al., *Ocean Engineering* 368, article 128342, published 15 November 2026, report experiments and DES/cavitation-model comparisons for hub-vortex/cavitation pressure signals at 300–1000 rpm. Their abstract reports a 5.1–5.5 Hz vortex feature, speed-dependent spectral power-law exponents, high-frequency energy above 100 Hz rising from 3.0% to 6.4%, and a skewness transition at high speed. This is a pump-rig diagnostic comparator, not a PWC performance model or universal cavitation threshold. [DOI 10.1016/j.oceaneng.2026.128342](https://doi.org/10.1016/j.oceaneng.2026.128342)
- Wang et al., *Ocean Engineering* 356 (30 May 2026), use a coupled 6-DOF/rotating-waterjet CFD framework for an amphibious vehicle and report manoeuvre-induced intake/operating-point asymmetry. This is an adjacent vehicle class, not validated PWC data. [DOI 10.1016/j.oceaneng.2026.125403](https://doi.org/10.1016/j.oceaneng.2026.125403)
- ITTC's 2011 propulsion method remains a documented calculation baseline, while the 2024 ITTC waterjet cavitation procedure treats self-propulsion, waterjet-system and pump tests as complementary. These should not be collapsed into one nominal pump-efficiency score.

The opportunity is an open, repeatable cross-scale evidence chain: calibrated pump loop → measured inlet/capture profile → station energy-flux integration → installed-system tests → transient manoeuvres → independently validated PWC trials. Each stage carries configuration, timing, calibration, uncertainty and data-rights provenance.

## 7. Proposed benchmark goals (not achieved)

- 100% agreement on analytic station-energy-flux known-answer vectors, including non-uniform profiles.
- Exact agreement between integrated flow and an independent calibrated flow measurement within a predeclared uncertainty budget.
- Zero (P_{JSE}) output when station flow continuity fails or cannot be established.
- Zero station-6 substitution unless a versioned, explicit parallel-outflow/vena-contracta justification is present.
- 100% shared pressure reference, elevation datum, timebase and configuration identity across station records.
- No overall efficiency output without matched pump-loop, inlet/system and propulsion-test inputs.
- All failed, censored, missing, out-of-range and domain-shifted measurement windows remain in the denominator.

No numerical accuracy target should be selected until actual sensor repeatability, capture-area geometry, sampling, velocity-field measurement method, covariance and facility uncertainty are known.

## 8. Claim ceiling and next tests

A passing v0.1 run demonstrates only the implementation's synthetic quadrature, sign, datum, availability and flow-continuity semantics. It does not demonstrate a real PWC's energy use, thrust, pump map, installation loss, cavitation, acoustic behaviour, manoeuvrability, safety, regulatory compliance, or model-to-full-scale similarity.

Next physical work, only after access/rights and facility capability are confirmed:
1. characterize the capture-area geometry and measure the upstream velocity distribution with an appropriate method (e.g., LDV/V3V where feasible);
2. calibrate the nozzle flow-rate instrument in situ and quantify bias/uncertainty;
3. capture pressure, elevation, local axial velocity and station geometry on one synchronized timebase;
4. separately validate pump-loop and installed-system tests;
5. run repeatable matched operating points before manoeuvre transients;
6. reserve independent conditions for validation and preserve the full uncertainty budget.
