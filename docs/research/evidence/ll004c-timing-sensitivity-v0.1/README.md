# LL-004C Timing-Sensitivity Oracle — Evidence Record

- **Status:** Phase-0 analytic reference; independent reproduction executed
- **Parent:** Symthaea #1595 (LL-004 moving-target rendezvous)
- **Composes with:** independent moving-target oracle PR #1596
- **Evidence class:** synthetic analytic reference
- **Runtime dependencies of the submitted script:** Python standard library only
- **Scope:** exact constant-velocity pod/receiver motion with bounded phase offsets

## Why this increment exists

The moving-target encounter oracle already demonstrates an exact nominal intercept and a qualitative one-second target-phase shift. This increment turns that observation into a quantitative timing-sensitivity reference with a frozen sweep, an exact closed-form oracle, and fail-closed coverage controls.

The swept variable is the receiver's epoch offset relative to the pod release epoch. It does not alter the receiver's physical trajectory or invent a new ephemeris. It evaluates the same declared linear track at (t + delta).

## Closed-form reference

Synthetic SI fixture:

```text
pod:    r_p(t) = (t, 0, 0) m        v_p = (1, 0, 0) m/s
target: r_t(t) = (10, 1 - 0.1 t, 0) m   v_t = (0, -0.1, 0) m/s
```

For target phase offset (delta), the relative position is

[
r_{rel}(t,delta) = (t-10,;-0.1(t-10)-0.1delta,;0).
]

Minimizing its squared norm gives

[
t_{TCA}(delta) = 10 - rac{delta}{101},
qquad
d_{min}(delta) = rac{|delta|}{sqrt{101}},
qquad
v_{rel} = sqrt{1.01};	ext{m/s}.
]

This exact answer is independent of numerical integration or an orbital propagator. It is a narrow mathematical oracle, not an orbital flight model.

## Frozen sweep

Phase offsets: -2, -1, 0, +1, +2 seconds. Encounter window: [0, 20] seconds. Track coverage: [-5, 25] seconds. The wider source track ensures the complete shifted encounter window is covered; extrapolation is prohibited.

| Target phase offset (s) | TCA (s) | Miss distance (m) |
|---:|---:|---:|
| -2 | 10.01980198019802 | 0.1990074380419978 |
| -1 | 10.009900990099009 | 0.09950371902099892 |
| 0 | 10.000000000000000 | 0 |
| +1 | 9.990099009900990 | 0.09950371902099908 |
| +2 | 9.980198019801980 | 0.19900743804199789 |

Derived summary:

- Central TCA sensitivity: **-0.00990099009900991 s/s**.
- Worst miss over the declared ±2 s phase envelope: **0.19900743804199789 m**.
- Maximum miss-growth ratio over the sampled nonzero offsets: **0.09950371902099908 m/s**.

## Verification performed

The submitted source contains a runnable `self_test()` with five golden cases and negative controls. The checked-in script itself has **not** yet been executed as a file in the authoring container.

Instead, an independently re-created implementation of the checked-in algorithm and closed-form equations was executed in the local analysis runtime. It passed:
- all five exact TCA and miss-distance cases;
- central finite-difference TCA sensitivity;
- maximum miss over the declared phase envelope;
- track-coverage rejection;
- missing nominal-offset rejection;
- duplicate/non-increasing epoch rejection.

Output: `Independent LL-004C closed-form reproduction PASS`.

The exact expected JSON report stored beside this record has SHA-256:
`ebc08a7653b2e54cfb74e8034b0b353ef90626dc95f830a2c54dd60a829d4948`.

The source is bound by Git blob `4a4366ee0f0d899c9425b1df6b4c9e5b65a8fc5e`. This is a Git object ID, not a SHA-256 content digest.

## Claim ceiling and next gate

This result qualifies only a closed-form constant-velocity timing-sensitivity fixture. It does **not** validate the production Rust encounter solver, gravitational propagation, an actual lunar/cislunar ephemeris, NRHO/Gateway rendezvous, covariance realism, navigation, terminal guidance, capture safety, or flight performance.

Next gate: run the submitted file's `self_test()` on exact PR head, add parity fixtures to the production LL-004 implementation, and independently verify the same timing offsets using the moving-target oracle's propagated dynamics. After that, use provenance-bound external target tracks and compare frame/trajectory outputs against mature independent tools (Orekit, GMAT, and compatible Brahe tasks). No operational tolerance or capture envelope should be inferred from this synthetic fixture.
