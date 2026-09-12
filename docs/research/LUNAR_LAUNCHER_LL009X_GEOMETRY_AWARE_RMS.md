# LL-009X — Geometry-aware distribution-free RMS horizon optimization

LL-009X tightens LL-009W without adding a probability distribution, independence assumption, covariance model, or new source semantics.

W assigns every admitted pixel the same tail budget `alpha/N`, producing `k = sqrt(N/alpha)` and the safe radial-height envelope `z_i + k s_i`. X observes that the scientific quantity downstream is not maximum height error—it is the **skyline angle**. Pixels far below the skyline can tolerate much more height error than a nearly blocking ridge before either changes visibility.

## Radial geometry

For observer radius `R_s`, terrain radial unit vector `u_i`, nominal terrain radius `R_i`, `c = u_i·u_s`, and `q = sqrt(1-c²)`, positive radial height error `delta` produces

`tan h_i(delta) = ((R_i + delta)c - R_s) / ((R_i + delta)q)`.

For `q > 0`,

`d/dR [(R c - R_s)/(R q)] = R_s/(R² q) > 0`,

so apparent elevation is strictly monotone with positive radial height. The local tangent component is `R * projection_tangent(u_i)`, so changing radial height cannot change the support point's azimuth. Each point therefore stays in its original horizon bin.

For candidate horizon `H`, the radial increment required to reach it is

`delta_i(H) = R_s / (c - q tan H) - R_i`

when the denominator is positive. At or above the point's radial-asymptotic elevation, no finite positive radial increment can breach `H`.

## Binwise familywise risk

Markov on squared RMS error gives

`P(e_i > delta_i(H)) <= min(1, s_i² / delta_i(H)²)`.

X allocates the exact W whole-sky budget by exact admitted support count:

`alpha_b = alpha * N_b/N`.

This is not a new risk policy. Under W's uniform `alpha/N` allocation, the same `N_b` pixels already consume exactly `alpha N_b/N`. Therefore W's envelope is a feasible solution inside every X bin.

X uses deterministic bisection to find the smallest `H_b` whose summed per-pixel Markov bound is no larger than `alpha_b`. Every emitted X bin is then required to satisfy

`H_X,b <= H_W,b + tolerance`.

If that domination invariant fails, X fails closed.

## Replayability and scale

X re-hashes the exact W/L sources and independently reconstructs the admitted population. It must reproduce W's exact total `N`. Per-bin optimization records are spooled to bounded scratch files rather than retaining millions of Python objects in memory. Each bin emits both an input-population digest and a digest over its final `(row, col, delta, Markov contribution)` records.

## Local adversarial campaign

The synthetic Rasterio 1.5.0 campaign passed deterministic replay, exact W population reproduction, whole-sky risk-budget conservation, per-bin risk compliance, radial azimuth invariance, closed-form threshold geometry, W-domination, and W receipt tamper rejection.

A deliberately high/RMS ridge surrounded by deep low-risk terrain produced a maximum improvement of about **8.73 degrees** versus W's uniform per-pixel envelope while preserving the same whole-sky alpha and the same no-Gaussian/no-independence assumptions.

That number is synthetic logic evidence, not a prediction for Site01.

## Boundary

X remains conditional on LL-009V/W's RMS second-moment model. It covers represented raster support points only; LL-009R remains authoritative for unresolved terrain. X uses W's nominal observer; LL-009U or a successor must repeat the geometry-aware calculation for each exact Q member-specific observer state before the result can become a full Site01 statistical horizon.
