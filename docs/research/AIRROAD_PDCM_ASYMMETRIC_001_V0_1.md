# AIRROAD-PDCM-ASYMMETRIC-001 — V0.1 exact asymmetric two-converter benchmark

**Status:** synthetic exact-rational analytical reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11749  
**Base interaction implementation:** https://github.com/Luminous-Dynamics/symthaea/pull/11745

## Why this case matters

The symmetric shared-source example proves that two individually stable branches can interact unstably. This asymmetric case removes identical-branch symmetry: the feeder resistances, input capacitances, operating voltages and constant-power loads differ.

Both branch input subsystems remain locally stable against a stiff common bus, but the shared source has a positive real pole. The example is still one common source port and two parallel loads, so it is a SISO aggregate return-ratio case—not general MIMO stability.

## Frozen exact parameters and equilibrium

| Quantity | Value |
|---|---:|
| Source voltage (V_s) | 17/8 V |
| Source resistance (R_s) | 1 Ω |
| Bus capacitance (C_0) | 1 F |
| Feeder resistance (R_1) / (R_2) | 1 Ω / 2 Ω |
| Input capacitance (C_1) / (C_2) | 1 F / 2 F |
| CPL (P_1) / (P_2) | 1/2 W / 5/32 W |

The equilibrium witness is (V_0=3/2) V, (V_1=1) V and (V_2=5/4) V.

Currents are (I_s=5/8) A, (I_1=1/2) A and (I_2=1/8) A. Exact power closure:

```
85/64 W source input
= 21/32 W load
+ 25/64 W source resistance loss
+ 1/4 W feeder 1 loss
+ 1/32 W feeder 2 loss
```

## Isolated versus coupled stability

With the bus voltage held stiff, the branch poles are:
- branch 1: -1/2 s^-1;
- branch 2: -1/5 s^-1.

Both are stable.

The full Jacobian for state order [V0,V1,V2] is:

```
[ -5/2    1     1/2 ]
[  1     -1/2    0  ]
[  1/4    0    -1/5 ]
```

The exact characteristic polynomial is:

```
lambda^3 + (16/5)lambda^2 + (29/40)lambda - 1/80
```

Its constant term is negative. The polynomial is negative at zero and positive for sufficiently large positive real lambda, so it has at least one positive real root. The coupled system is unstable, despite both stiff-bus branch poles being negative.

## SISO source/load return-ratio cross-check

The exact small-signal port functions are:

```
Y1(s) = (s - 1/2)/(s + 1/2)
Y2(s) = (s - 1/20)/(2s + 2/5)
Zs(s) = 1/(s + 1)
L(s) = Zs(s)*(Y1(s) + Y2(s))
```

The closed-loop characteristic condition (1+L(s)=0) produces:

```
2s^3 + (32/5)s^2 + (29/20)s - 1/40
```

Dividing by 2 gives the same state-space characteristic polynomial. The frozen fixture also includes exact complex return-ratio values at (omega=0) and (1) rad/s.

This is an exact SISO cross-check for a single shared bus with unequal branches. It is not a generalized-Nyquist theorem or a multi-source MIMO result.

## Deliverables

- Machine-readable exact-rational fixture: `docs/research/fixtures/airroad-pdcm-asymmetric-two-converter-v0.1.json`.
- Independent standard-library Fraction verifier: `tools/airroad-pdcm-impedance-qualifier/verify_asymmetric_fixture.py`.
- Rust unit test on the existing `dc_bus_interaction` API.
- Focused workflow that runs both exact-rational fixture verifiers and direct zero-dependency Rust tests/Clippy.

## Claim ceiling

Synthetic asymmetric RC/CPL algebra only. The case does not model converter control-loop states, line inductance, switching, measured complex impedance, protection, battery electrochemistry, thermal response, PHIL/HIL, aircraft performance, airworthiness or certification.
