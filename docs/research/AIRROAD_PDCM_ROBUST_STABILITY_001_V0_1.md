# AIRROAD-PDCM-ROBUST-STABILITY-001 — V0.1 symmetric hard-bound certificate

**Status:** exact-rational theorem and executable reference contract  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11748  
**Base model:** https://github.com/Luminous-Dynamics/symthaea/pull/11745

## 1. Purpose

This increment classifies robust small-signal stability over deterministic parameter intervals for one narrow model: two identical constant-power converter inputs coupled through identical resistive feeders to a single Thevenin source and a common bus capacitor.

It does not generalize to asymmetric branches, multiple buses, line inductance, converter control loops or arbitrary MIMO networks.

## 2. Exact symmetric model

Let source conductance be (G_s>0), branch feeder conductance (G>0), branch CPL negative incremental conductance (h=P/\bar V^2>0), common bus capacitance (C_0>0), and identical input capacitance (C>0).

The linearized Jacobian is:

```
J = [ -(Gs + 2G)/C0    G/C       G/C
           G/C       (-G+h)/C      0
           G/C          0       (-G+h)/C ]
```

The differential branch mode is (lambda_d=(-G+h)/C). The symmetric common-mode block is:

```
Jc = [ -(Gs+2G)/C0    2G/C0
          G/C        (-G+h)/C ]
```

Its determinant is:

```
det(Jc) = [Gs*G - (Gs + 2G)*h] / (C0*C)
```

Define:

```
h_crit(Gs,G) = Gs*G/(Gs+2G)
```

For positive parameters, (h_crit<G). Therefore:
- (h<h_crit): common mode has negative trace and positive determinant; differential mode is also stable. The complete symmetric model is Hurwitz stable.
- (h=h_crit): the common-mode determinant is zero. This is a boundary, not robust PASS.
- (h>h_crit): common-mode determinant is negative, so one common-mode pole is positive and one negative.

## 3. Monotonic endpoint theorem

```
d(h_crit)/d(Gs) = 2*G^2/(Gs+2G)^2 > 0
d(h_crit)/d(G)  = Gs^2/(Gs+2G)^2 > 0
```

For independently supplied hard intervals:

```
Gs in [Gs_lower, Gs_upper]
G  in [G_lower, G_upper]
h  in [h_lower, h_upper]
```

the threshold extrema are exactly:

```
h_crit_min = h_crit(Gs_lower, G_lower)
h_crit_max = h_crit(Gs_upper, G_upper)
```

The robust decision is:
- robustly Hurwitz stable if (h_upper<h_crit_min);
- robustly unstable with a positive real pole if (h_lower>h_crit_max);
- otherwise boundary/mixed/not established.

No grid search, Monte Carlo or probability conversion is needed for this model family. This endpoint theorem is stronger than sampling the parameter box, because it covers the entire admitted rectangular set.

## 4. Numerical implementation boundary

The Rust API validates finite, positive, ordered conductance intervals and evaluates the endpoint formula without forming the potentially overflowing product (Gs*G). It expands the threshold range by a binary64 guard and requires strict separation. This guard is conservative screening, **not** a formally directed-rounded interval enclosure. The exact-rational fixture is the independent reference for the frozen cases; broader numerical proof still belongs to the exact-rational / interval qualification owners.

The result must not be called robustly stable if any interval touches the guarded critical boundary. Statistical confidence intervals, datasheet typical values, and unsupported fitted ranges are not hard bounds.

## 5. Frozen benchmark corpus

Machine-readable cases live at `docs/research/fixtures/airroad-pdcm-robust-stability-v0.1.json`.

The five exact-rational cases cover:
1. a hard set fully below the threshold (robust stable);
2. a hard set fully above the threshold (robust unstable);
3. a set spanning the possible threshold range (not established);
4. exact boundary touch (not established);
5. widening a previously robust-stable interval until it overlaps the boundary (not established).

The independent Python standard-library Fraction verifier recomputes the threshold extrema, strict dispositions, exact rational margins, monotonic derivatives and conductance-scaling metamorphic relation.

## 6. Benchmarks and next model

- 100% exact classification of the frozen rational interval corpus.
- Zero robust-stable outcomes at or across the boundary.
- Zero probability assumptions or grid-sampling claims of coverage.
- Exact preservation of normalized margin under common positive scaling of (G_s,G,h).
- Widening an input hard interval must not make a previously ambiguous result more certain.
- Explicit binary64 guard and abstention rate reported separately from exact rational results.

This theorem is not a general robust impedance theorem. The next stage is a separately qualified robust SISO/MIMO method for asymmetric and frequency-dependent source/load port models, using the calibrated impedance and hard-bound uncertainty owners.

## 7. Claim ceiling

A passing software/rational suite establishes only the theorem and arithmetic for this symmetric RC/CPL model over the declared rectangular conductance set. It does not establish real converter stability, aircraft DC-network stability, fault tolerance, PHIL/HIL correlation, flight safety, airworthiness or certification.
