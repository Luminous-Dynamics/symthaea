# PWC-WATERJET-PERFORMANCE-COMPOSITION-001
## Matched test composition for waterjet powering efficiency

**Status:** v0.1.0 synthetic composition implementation  
**Issue:** #12266  
**Parent:** PWC-WATERJET-INSTALLED-ENERGY-001 #12232  
**Pump-loop reference:** #12231 / PR #12233  
**Propulsion-test station flux:** PR #12264

> The current fixtures are synthetic. A successful run validates only the composition equations, stage matching and fail-closed semantics. It does not qualify a real PWC or waterjet system.

## 1. Why this is a distinct research increment

A technically correct pump-loop result and a technically correct propulsion-test energy-flux result do not automatically make a valid overall-efficiency result. The test conditions, pump revision, scale state, flow rate, craft speed, NVR, water state, test method, and relevant interaction terms must be compatible.

The current programme therefore needs a composition contract that distinguishes:
- the uninstalled/free-stream pump characteristic;
- the inlet and nozzle losses;
- the installed pump penalty caused by non-uniform inflow;
- ideal jet/nozzle efficiency;
- hull–waterjet interaction from the propulsion test.

The aim is not to multiply a series of point estimates; it is to preserve the derivation and evidence lineage of each term.

## 2. Primary method references (equations rechecked 15 November 2026)

### ITTC 7.5-02-05-03.1 (2011 Rev.02)

The propulsion test procedure defines:
- overall efficiency: \(\eta_{OA}=\eta_0\eta_{INT}\);
- interaction efficiency: \(\eta_{INT}=(1-t)\eta_{eI}/\eta_{mI}\);
- free-stream efficiency: \(\eta_0=\eta_I\eta_P\eta_{duct}\);
- system head: \(H_{JS}=P_{JSE}/Q_J\);
- ideal efficiency: \(\eta_I=2/(1+NVR)\).

NVR is a result obtained from the propulsion test. This implementation consumes the documented NVR value and its source/method reference; it does not infer a velocity-ratio formula from a product brochure or unsupported calculation shortcut.

Official source: [ITTC 7.5-02-05-03.1 (2011 Rev.02)](https://www.ittc.info/media/8067/75-02-05-031.pdf), especially pp. 8–9, Eqs. (19)–(23).

### ITTC 7.5-02-05-03.2 (2017 Rev.02)

The waterjet-system procedure defines:
- pump efficiency: \(\eta_P=P_{PE}/P_D\);
- ducting efficiency: \(\eta_{duct}=P_{JSE}/P_{PE}\);
- effective jet-system power: \(P_{JSE}=E_7-E_1\);
- effective pump power: \(P_{PE}=P_{JSE}+\zeta_{13}E_0+\zeta_{57}E_7=\rho g Q_J H_{JS}\);
- inlet-duct loss: \(\zeta_{13}=(E_1-E_3)/E_0\);
- free-stream pump efficiency: \(\eta_{P0}=\rho g Q_J H_{JS}/P_D\);
- installed pump efficiency: \(\eta_P=\eta_{P0}\eta_{inst}\).

Important method boundaries:
- an inlet-duct test can measure \(\zeta_{13}\), but does not itself determine nozzle loss \(\zeta_{57}\);
- \(\zeta_{57}\) requires CFD or a waterjet-system test;
- \(\eta_{inst}\) is not obtained in a pump-loop test; a waterjet-system test is needed;
- a waterjet-system test can replace separate inlet and pump-loop tests, or supplement them.

Official source: [ITTC 7.5-02-05-03.2 (2017 Rev.02)](https://www.ittc.info/media/11852/75-02-05-032.pdf), especially pp. 8–9, Eqs. (1)–(12).

## 3. Two supported evidence paths

### Path A — composite independent tests

Requires:
1. propulsion test with NVR, flow rate and interaction terms;
2. station-1/station-7 energy-flux record providing \(P_{JSE}\), \(E_1\), \(E_7\), and independently integrated flows;
3. inlet-duct test providing \(E_0\), \(E_1\), \(E_3\), from which \(\zeta_{13}\) is calculated;
4. pump-loop test providing \(\eta_{P0}\);
5. waterjet-system test providing \(\zeta_{57}\), \(\eta_{inst}\), \(P_D\), \(H_{JS}\), and \(P_{PE}\).

The auditor calculates:
\[
\zeta_{13}=\frac{E_1-E_3}{E_0}
\]
\[
P_{PE}=P_{JSE}+\zeta_{13}E_0+\zeta_{57}E_7
\]
and cross-checks against:
\[
P_{PE}=\rho g Q_J H_{JS}
\]
It separately checks that \(\eta_P=\eta_{P0}\eta_{inst}\) agrees with \(P_{PE}/P_D\) within declared tolerances.

### Path B — integrated waterjet-system test

A waterjet-system test may replace the separate inlet and pump-loop tests. In this path, the system test provides compatible \(P_D\), \(P_{PE}\), and station-flux data. The auditor calculates:
\[
\eta_P=P_{PE}/P_D,\qquad
\eta_{duct}=P_{JSE}/P_{PE}
\]
and composes with propulsion-test \(NVR\) and interaction terms.

It intentionally leaves \(\eta_{P0}\), \(\eta_{inst}\), \(\zeta_{13}\), and \(\zeta_{57}\) absent. It does not invent a pump-loop comparison or installation efficiency when the separate test was not performed.

## 4. Match conditions before combining measurements

Every stage record binds:
- exact craft/hull/pump configuration digests;
- model/full-scale and Reynolds/scale context ID;
- craft speed, flow rate, NVR, water density and temperature;
- stage-specific source record digest, method ID, calibration reference, timebase reference and measurement window.

The common tolerance object explicitly controls flow, craft speed, NVR, density, temperature, power and efficiency comparison. Configuration hashes and scale-context IDs must match exactly. Numeric condition measurements must agree within the declared tolerances and their uncertainty intervals must overlap within that tolerance.

Independent tests may have different measurement dates and separate calibration/timebase references. What must match is their declared operating condition and configuration—not an artificial requirement that different facilities ran at the same wall-clock time.

A mismatch yields INCOMPARABLE and prevents a combined efficiency result.

## 5. Known-answer synthetic benchmark

The known-answer fixture uses:
- \(Q_J=0.02\,\mathrm{m^3/s}\);
- \(\rho=998\,\mathrm{kg/m^3}\);
- \(H_{JS}=7.1523181271\,\mathrm{m}\);
- \(P_{JSE}=1,000\,\mathrm{W}\);
- \(E_0=4,000\,\mathrm{W}\), \(E_1=3,000\,\mathrm{W}\), \(E_3=2,800\,\mathrm{W}\), \(E_7=4,000\,\mathrm{W}\);
- \(\zeta_{13}=0.05\), \(\zeta_{57}=0.05\);
- \(P_{PE}=1,400\,\mathrm{W}\), \(P_D=2,000\,\mathrm{W}\);
- \(\eta_{P0}=0.8\), \(\eta_{inst}=0.875\);
- \(NVR=2\), \(1-t=0.9\), \(\eta_{eI}=0.8\), \(\eta_{mI}=1\).

Expected derived results:
- \(\eta_P=0.7\);
- \(\eta_{duct}=5/7\approx0.7142857143\);
- \(\eta_I=2/3\);
- \(\eta_{INT}=0.72\);
- \(\eta_0=1/3\);
- \(\eta_{OA}=0.24\).

The integrated-system-path fixture omits inlet/pump-loop records and must still compute \(\eta_{OA}\), while correctly leaving \(\eta_{P0}\) and \(\eta_{inst}\) null.

## 6. Adversarial cases and claim ceiling

The synthetic fixture set includes:
- pump revision mismatch -> INCOMPARABLE;
- zero NVR -> INCOMPLETE, no ideal/overall efficiency;
- nominal pump efficiency above unity -> BLOCKED and never clipped;
- independent and integrated path fixtures to prevent fabricated intermediate terms.

A future extension should add mismatched NVR/scale/temperature, zeta negative/absent, power-balance discrepancy, uncertainty overlap, revoked calibration, and rights/trace-reference failures. These should remain separately visible, not averaged into one score.

A valid synthetic result is arithmetic-only. It does not establish measured PWC efficiency, pump-map accuracy, hull-waterjet integration, full-scale transfer, cavitation, handling, safety, emissions, ISO/SAMSA or class compliance, or product superiority.
