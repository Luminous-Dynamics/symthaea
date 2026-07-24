# Deployment Configuration Contract

A simulator, SIL, HIL, or physical run must bind one immutable deployment
manifest before authority is issued. The manifest identifies:

- airframe and software revision;
- backend kind;
- calibration, scenario, qualification campaign, hardware contract, and claim-ledger digests;
- required compile/runtime features;
- module version bindings; and
- the maximum claim level the deployment is allowed to request.

`DeploymentManifest::digest_fnv1a64` is a deterministic content identifier only.
It is not collision-resistant and does not authenticate the deployment. A
physical-hardware manifest must carry an external cryptographic authenticity
reference, and the integrating system must verify that signature before arming.

Any mismatch produces a `Rejected` binding report. There is no best-effort
fallback to a nearby calibration, latest scenario, simulation backend, or
unversioned module.
