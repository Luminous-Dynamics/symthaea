# AIRROAD-PDCM-MIMO-001 — V0.1 cross-port hidden-dynamics reference

**Status:** exact-rational synthetic MIMO negative control  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11761  
**Base impedance track:** https://github.com/Luminous-Dynamics/symthaea/pull/11745

## 1. Paradigm shift: stop treating a converter network as a set of independent scalar ports

The asymmetric #11749 example has unequal branches, but it remains a one-common-source SISO aggregate. This new fixture is intentionally different: it preserves a full two-port return-ratio matrix with off-diagonal coupling.

Cvetanović et al. formally compare terminated port-level SISO and device-level MIMO impedance assessment. They explain that a single SISO port-pair can give an incorrect stability prediction when unstable port-level dynamics are hidden from that pair; their MIMO method captures the multi-port coupling. The paper also discusses meshed networks and multi-port converters, and validates its analysis using HIL and a laboratory prototype. [IEEE TPEL paper, DOI 10.1109/TPEL.2024.3403236](https://doi.org/10.1109/TPEL.2024.3403236).

This is not a claim that Luminous invented MIMO impedance analysis. The benchmark asks whether our evidence and model interfaces preserve all the information needed to use it correctly.

## 2. Exact cross-port loop

The frozen model is:

```
L(s) = [ 0          1/(s - 1) ]
       [ 1/(s + 2)  0          ]
```

The diagonal projections are both exactly zero:

```
L11(s) = 0
L22(s) = 0
```

A diagonal-only SISO test sees (1+L_{11}=1) and (1+L_{22}=1), so it reports no diagonal loop interaction. It discards the actual off-diagonal loop.

Under the declared negative-feedback convention, the complete matrix condition is:

```
det(I + L(s)) = 1 - 1/((s - 1)(s + 2))
```

Clearing the declared channel denominators gives:

```
(s - 1)(s + 2) - 1 = s^2 + s - 3
```

The exact roots are:

```
r_positive = (-1 + sqrt(13))/2  ≈  1.3027756377 s^-1
r_negative = (-1 - sqrt(13))/2  ≈ -2.3027756377 s^-1
```

One positive real closed-loop pole exists. The full matrix therefore reveals an unstable coupled mode that diagonal-only SISO projections miss.

The open-loop matrix has one declared RHP channel pole at (s=+1) and one LHP pole at (s=-2). The pole count and channel realization identity must be preserved alongside determinant samples. A determinant trace by itself is not a generalized-Nyquist certificate.

## 3. Exact frequency points

At (omega=0):

```
L12 = -1
L21 = 1/2
det(I + L) = 3/2
```

At (omega=1) rad/s:

```
L12 = (-1 - j)/2
L21 = (2 - j)/5
det(I + L) = (13 + j)/10
```

These complex values are recomputed from the exact model in both the Rust implementation and an independent Python Fraction verifier.

## 4. Sensitivity: make the benchmark useful for design, not only detection

Let (k_{12}) and (k_{21}) be the two cross-port residues. The positive root of:

```
s^2 - (p12 + p21)s + p12*p21 - k12*k21 = 0
```

has local sensitivity:

```
dr_positive/dk12 = k21/sqrt(discriminant)
dr_positive/dk21 = k12/sqrt(discriminant)
```

For this fixture, both are (1/sqrt{13}) per residue unit. This is a small exact reference for the next participation/sensitivity layer; it is not an aircraft control recommendation.

## 5. Implementation and negative controls

- Rust reference: `symthaea-circuits::dc_bus_mimo_reference`.
- Machine-readable fixture: `docs/research/fixtures/airroad-pdcm-mimo-hidden-dynamics-v0.1.json`.
- Independent oracle: `tools/airroad-pdcm-impedance-qualifier/verify_mimo_hidden_dynamics.py`.

The implementation preserves the full 2x2 matrix, evaluates (det(I+L(jomega))), carries the active open-loop RHP pole count, computes the cleared characteristic polynomial where both cross channels are present, and reports the positive real root and cross-coupling sensitivities.

The negative controls remove one or both cross channels. When only one cross channel remains, (det(I+L)equiv1), but the active channel's open-loop pole metadata must still be retained. This prevents a determinant-only result from erasing a realization pole.

## 6. Industry/SOTA comparison

- PLECS 5.0 already provides AC, loop-gain and frequency-response tools. Its standalone frequency-response method drives a model to an operating point, injects sequential sinusoidal perturbations and extracts response at discrete frequencies. Those functions are established commercial capabilities, not our novelty. [PLECS frequency response](https://docs.plexim.com/plecs/5.0/analysis/frequency-response/).
- Qu et al. (2025) report a six-converter DC microgrid, generalized-Nyquist analysis, participation/sensitivity curves and RT-LAB HIL. This is a relevant scaled-system baseline. [DOI 10.1007/s43236-024-00938-7](https://doi.org/10.1007/s43236-024-00938-7).
- Tang et al. (2026) report N-machine coupled AC/DC instability propagation, two cascade modes and HIL validation on two- and three-machine systems. The next benchmark tier should therefore include DC-side instability propagation through network coupling, rather than treating each converter independently. [DOI 10.1016/j.epsr.2026.113503](https://doi.org/10.1016/j.epsr.2026.113503).
- NASA AREAL is a 200 kW / 600–700 VDC reconfigurable testbed for architecture and system-interaction studies, with up to 1,000 V peak operation. NASA NEAT provides MW-scale powertrain testing and simulated altitude. Luminous should pursue low-energy and partner PHIL evidence rather than claim to replace these facilities. [AREAL](https://www.nasa.gov/eap-labs-and-testbeds/areal/) · [NEAT](https://www.nasa.gov/eap-labs-and-testbeds/neat/).

The defensible differentiator is the evidence layer: exact port topology and directions, full matrix terms, open-loop pole/realization metadata, calibrated frequency response, uncertainty and validity domain, configuration digests, independent state-space comparison, and mission/fault consequence.

## 7. Standards boundary

The current DLA ASSIST entry lists MIL-STD-704 as active, with Revision F Notice 4 dated 24 June 2026. Its stated scope is aircraft electric power characteristics at utilization-equipment input terminals; it explicitly excludes electromagnetic interference and voltage spikes. It is therefore a useful interface/power-quality benchmark input, not a substitute for network dynamic-stability qualification. [DLA ASSIST document record](https://quicksearch.dla.mil/qsDocDetails.aspx?ident_number=35901).

## 8. Proposed benchmark targets

- 100% exact agreement for the frozen 2x2 rational matrix and frequency points.
- Zero diagonal-only SISO passes when required cross-port data are nonzero and omitted.
- Zero loss of active open-loop RHP pole counts when determinant cancellations occur.
- 100% exact agreement between the 2x2 matrix characteristic and the equivalent state-space reference on matched fixtures.
- Correct local sensitivity signs and values on exact known-answer cases.
- Later holdouts: 3+ ports, nonreciprocal and asymmetric matrices, frequency-coupled dq models, hard uncertainty, and matched HIL/PHIL transfer matrices.
- Report margin, port/frequency coverage, unsupported-domain abstention and runtime separately. Do not claim superiority over PLECS or published generalized-Nyquist methods on synthetic data.

## 9. Claim ceiling

This is a synthetic abstract 2x2 return-ratio counterexample. It is not a general generalized-Nyquist solver, a measured converter model, aircraft network validation, HIL/PHIL correlation, flight-safety analysis, airworthiness evidence or certification.
