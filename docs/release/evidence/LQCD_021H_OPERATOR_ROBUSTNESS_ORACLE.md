# LQCD-021H — independent paired operator-robustness oracle

Independent standard-library execution for the operator-convention robustness semantics tracked by #2717. This subject uses synthetic paired measurements only and imports no Symthaea operator implementation.

Exact executed subject SHA-256:

`224ac60208f3cda3e0ab4295daf4506af34cbe815e1298edad3c207e946a4d30`

Canonical result SHA-256:

`0c20ee6fa5c52214f75486b724f272f29e8d370346479364d21447c5bc98242e`

## Paired theorem

Every comparison is keyed by the same synthetic `ConfigId`, vector, and temporal extent for two named operator conventions:

- `exhaustive`;
- `bresenham`.

The analysis never treats the two measurements as independent Monte Carlo samples. Eight contiguous blocks of eight configurations are resampled as paired units.

Breaking the per-ConfigId operator pairing fails closed.

## Axis null control

The axis-aligned `(1,0,0)` family is constructed as an exact null control.

Both raw complex Wilson values and the late-time plateau estimate must be bit-identical under the two conventions. The executed fixture obtains exact zero raw and plateau differences.

This tests the semantics expected when the two spatial transporters collapse to the same path.

## Off-axis behavior

Off-axis raw complex Wilson loops are deliberately different. Their imaginary components are retained as diagnostic data and may not be silently discarded before validation.

The primary robustness question is late-time behavior, not raw-loop identity.

The synthetic robust fixture produces late plateau differences:

- `(1,1,0)`: `0.000855910244`;
- `(1,1,1)`: `0.000928145522`;
- `(2,1,0)`: `0.001000270149`.

The synthetic policy freezes a late-time window `T=4..6` and a declared robustness domain consisting of an absolute synthetic tolerance plus two paired jackknife standard errors. All robust-fixture off-axis differences remain inside that domain.

## Sensitivity control

A second fixture injects a genuine `+0.012` late-time ground-state shift into the bounded operator for `(1,1,1)`.

The resulting paired late difference is `0.012928145522`, and the oracle returns:

`OperatorSensitivityDetected`.

This negative control ensures the robustness machinery does not simply bless any two operators that share configurations.

## Typed outcomes

The oracle exercises:

- `OperatorRobustWithinDeclaredDomain`;
- `OperatorSensitivityDetected`;
- `Inconclusive`.

A truncated paired sample becomes `Inconclusive`.

## Scientific boundary

This subject qualifies paired-analysis and operator-sensitivity semantics only. It does not establish real exhaustive-vs-Bresenham equivalence on beta=6.0 gauge configurations, historical EHK transporter identity, continuum rotational restoration, or final benchmark agreement.
