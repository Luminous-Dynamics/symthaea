# PIE-009Q calibration-chain / metrology-closure oracle

Date: 2026-09-13

## Scope

This evidence note records the independent synthetic semantics in `scripts/pie-009q-metrology-closure-oracle.py`.

The oracle is structural only. It does not claim real calibration uncertainty, SI traceability, sensor drift, artifact aging, laboratory qualification, certification, or lunar/Mars metrology performance.

## Independent execution

The final candidate self-test was executed locally with Python 3 and returned:

`ok`

No Symthaea module is imported by the oracle.

## Frozen semantics

Calibration capability is modeled as a dependency closure problem with two distinct seed sets:

- `operational` includes presently available local and imported capabilities;
- `local` includes only capabilities declared locally renewable.

A sensor calibration target is classified:

- `LocallyRenewable` when its calibration capability is reachable from the local seed set;
- `ImportDependent` when reachable operationally but not locally;
- `Unavailable` when no declared operational route exists.

Unsupported dependency cycles do not self-bootstrap. Direct self-dependencies, unknown nodes, duplicate recipe IDs, and duplicate sensor targets fail closed.

## Executed synthetic fixtures

The self-test demonstrates:

1. a temperature sensor calibration route supported by a local bench + local standard is `LocallyRenewable`;
2. a pressure sensor calibration route requiring an imported reference is `ImportDependent`;
3. an optical sensor whose fixture is unavailable is `Unavailable`;
4. an unsupported two-node calibration cycle remains unreachable;
5. adding an explicit local production route for the pressure reference promotes the pressure sensor to `LocallyRenewable`;
6. removing a local standard removes, rather than improves, local calibration closure;
7. malformed direct self-calibration and unknown prerequisites fail closed.

## Important limitations

This oracle assumes declared recipes are valid and does not prove that a reference artifact, fixture, procedure, environmental chamber, or calibration interval is physically adequate. It also does not model calibration scheduling, uncertainty propagation, reference degradation, inter-laboratory comparison, human/operator competence, or cyber authenticity.

Those belong in later evidence layers.

## Promotion boundary

The intended path is:

`structural metrology closure -> evidence-bearing real calibration recipes -> lifecycle/aging semantics -> PIE-009P sensing qualification -> PIE-009N/O diagnosability/diagnosis`

Tracks #2150, #2144, #2030, #2036, #1647 and master #1604.
