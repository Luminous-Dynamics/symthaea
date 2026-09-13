# First-instruction simulation deployment

This crate strengthens `symthaea.simulation.deployment.v1` with an exact cgroup-v2 resource policy and first-instruction worker execution. It does not replace or mutate deployment-v1.

The live authority chain is:

`ActiveAdmission + BoundSimulationDeployment + ActiveWorkerQualification + exact package/image + live CgroupV2Lease -> BoundFirstInstructionDeployment -> FirstInstructionDeploymentInvocation`.

Persisted evidence is audit-only and cannot recreate admission, worker qualification, cgroup, process, or deployment authority.

The first-instruction path preserves the base deployment frame and wall/stdout/stderr watchdog envelope exactly. It does not currently reproduce the legacy pre-exec rlimit envelope, and therefore records `legacy_preexec_rlimits_applied=false` rather than claiming cgroup/rlimit equivalence.

A canonical release receipt is intentionally outside this crate. Release must remain owned by `symthaea-sim-release` and recheck both package admission and worker qualification immediately before minting the existing release receipt.
