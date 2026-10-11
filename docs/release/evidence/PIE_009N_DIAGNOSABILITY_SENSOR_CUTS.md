# PIE-009N diagnosability / sensor-cut oracle evidence

Date: 2026-09-12

## Scope

This evidence note records the independent synthetic reference semantics in `scripts/pie-009n-diagnosability-oracle.py`.

The oracle asks a decision-specific question: does the active sensor set distinguish every hidden world that implies `PROCEED` from every hidden world that implies `BLOCK`? It does not require full hidden-state identification when that extra distinction is irrelevant to the decision.

## Independent execution

The final candidate self-test was executed locally with Python 3 on 2026-09-12 and returned:

`ok`

No Symthaea module is imported by the oracle.

## Reference semantics

The oracle freezes these rules:

- unknown state remains unknown; no probability distribution is invented;
- an explicitly failed/uninformative sensor may return `UNKNOWN`, which never creates discrimination by itself;
- diagnosability is decision-specific: worlds in the same decision class need not be mutually distinguishable;
- every cross-decision world pair must be separated by at least one active sensing path;
- removing sensing information cannot create a valid distinction that was not already present;
- inclusion-minimal sufficient sensor sets are reported rather than every feasible superset;
- inclusion-minimal sensor-loss cut sets are reported relative to a declared diagnosable baseline;
- common-mode groups are explicit and may be analyzed as group-loss cut sets;
- an undiagnosable baseline fails closed rather than producing misleading cut sets;
- malformed/duplicate/incomplete sensor mappings fail closed.

## Executed synthetic fixture

The fixture contains four hidden worlds spanning generator health and intertie state. The decision `PROCEED` is allowed only for `generator healthy + tie closed`.

There are two independent generator-health sensing paths (`gen_current`, `gen_vibration`) and two independent tie-state paths (`tie_aux`, `tie_optical`), plus an uninformative `UNKNOWN` diagnostic channel.

The self-test demonstrates:

1. the complete baseline is diagnosable;
2. the inclusion-minimal sufficient sensor sets are exactly one generator-health path plus one tie-state path;
3. the `UNKNOWN` channel cannot manufacture certainty;
4. losing any one of the four useful sensors still leaves the decision diagnosable;
5. losing both generator-health sensors is a minimum sensor cut;
6. losing both tie-state sensors is a minimum sensor cut;
7. the corresponding independent common-mode group pairs are minimum group cut sets;
8. malformed duplicate/incomplete sensor models fail closed;
9. cut-set analysis refuses an already-undiagnosable baseline.

## Important distinction

`decision diagnosable` does **not** mean `full physical state identified`.

For black-start/recovery, the required proof is that the surviving instrumentation can distinguish states that require different safe actions. Additional state identification may be useful, but it is not silently treated as necessary for every decision.

## Limitations

This is structural symbolic observability only. It does not model measurement noise, calibration drift, correlated numeric uncertainty, sensor dynamics, physical redundancy qualification, Bayesian inference, Kalman filtering, fault probabilities, electrical transients, cyber compromise, or hardware authority.

A later production layer should compose this with PIE-009L belief-state planning, PIE-009M proposal verification, PIE-009J startup-energy accounting, and actual evidence-bearing sensor/interface models rather than enlarging this oracle into a second plant simulator.

## Promotion boundary

`independent diagnosability oracle -> production sensor/evidence types -> cross-check against PIE-009L beliefs -> proof-carrying proposal verifier -> local authority/safety kernel`

Tracks PIE-009N, #1968, #1976, #1924, #1932, #1647 and master #1604.
