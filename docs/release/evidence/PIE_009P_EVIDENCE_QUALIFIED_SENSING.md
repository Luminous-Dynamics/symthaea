# PIE-009P evidence-qualified sensing oracle

Date: 2026-09-13

## Scope

This evidence note records the independent synthetic semantics in `scripts/pie-009p-evidence-qualified-sensing-oracle.py`.

The oracle is deterministic and implementation-independent. It does not claim real lunar/Mars sensor accuracy, calibration lifetime, failure probability, cybersecurity, plant qualification, or execution authority.

## Independent execution

The final candidate self-test was executed locally with Python 3 and returned:

`ok`

No Symthaea module is imported by the oracle.

## Frozen semantics

A measurement may contribute to a safety-critical conclusion only if all declared evidence qualifiers pass:

- measurement age is within the sensor's explicit freshness bound;
- calibration remains valid at the current step;
- self-test is passing;
- provenance is on the sensor's explicit allowlist;
- measurement topology version matches the current topology version.

Qualified `UNKNOWN` readings remain unknown and provide no directional support.

Fusion requirements are explicit rather than implicit. A policy may require minimum directional votes plus minimum diversity across declared sensor failure domains, power domains, and data domains. Two agreeing sensors on the same vulnerable path therefore do not automatically count as independent evidence.

If qualified directional readings disagree, the oracle emits `CONFLICT`. It does not majority-vote through contradictory evidence.

## Executed synthetic fixtures

The self-test demonstrates:

1. two independent qualified generator-health paths plus two independent tie-state paths support `PROCEED`;
2. a stale generator path removes enough independent evidence to make the generator conclusion `INDETERMINATE`;
3. failed self-test disqualifies a tie-state path;
4. expired calibration disqualifies a generator path;
5. unapproved provenance disqualifies a tie-state path;
6. topology-version mismatch disqualifies a measurement;
7. contradictory qualified generator readings produce `CONFLICT` and block the restart decision;
8. a qualified `UNKNOWN` reading does not count as a directional vote;
9. agreeing same-domain generator paths fail explicit diversity requirements;
10. the exact freshness boundary is accepted while one step beyond is rejected;
11. evidence-bundle digesting is deterministic and metadata-sensitive;
12. duplicate registries/observations fail closed.

## Important limitations

The oracle treats calibration validity, self-test state, provenance allowlists, and domain declarations as trusted inputs. It does not yet model the productive dependency chain needed to maintain calibration standards, verify reference artifacts, detect maliciously forged telemetry, quantify numeric uncertainty, or resolve a qualified sensor conflict.

Those are separate concerns and should remain separate rather than turning this reference into a monolithic sensor stack.

## Promotion boundary

The intended path is:

`independent sensing-evidence oracle -> production evidence types -> PIE-009N diagnosability composition -> PIE-009O active diagnosis composition -> PIE-009M proposal verification -> authority/control boundary`

Tracks #2144, #2030, #2036, #1968, #1976, #1647 and master #1604.
