# MACE-CoS Assurance Protocol v0.3

Dependency-free Go reference for:
- the task-success × control-compliance outcome matrix;
- evidence closure against an externally pinned mandatory-control inventory;
- deterministic compilation of composable SUT/scenario profiles into inventories.

A task success with any failed or unproven mandatory control is not a clean success. Evidence incompleteness is distinct from confirmed violation. An indeterminate task verdict sits outside the 2×2 matrix.

## Inventory closure

The preregistration authority freezes a typed control inventory. The harness supplies its SHA-256 pin independently of the agent/run record. Missing mandatory evidence becomes proof-incomplete; unknown/duplicate IDs, unknown verdicts and inventory substitution reject. Mandatory N/A requires both prior eligibility and a run-specific rationale.

## Profile compiler

The matrix package defines four SUT layers (foundation model, interaction/access, execution and scheduling, underlying infrastructure) and six scenario classes (tool-using, long-term memory, DevOps/development, embodied, multi-agent, autonomous/iterative). A versioned profile catalogue maps criterion obligations to one or more layers and scenarios.

The compiler selects rows when infrastructure-profile and scenario scopes both intersect the declared profile. It composes duplicates conservatively: mandatory is OR; N/A permission is AND. The composite declaration also binds the exact request traversal order and one configuration digest for every selected infrastructure and scenario profile. It hashes the complete catalogue after canonicalizing row and scope order, then binds that digest, the selected profile, traversal and constituent configs to the inventory identity. Even an edit to a currently unselected catalogue row changes the digest; changing traversal or any constituent configuration changes identity too. This does not prove catalogue completeness. Catalogue ownership and independent review remain external governance requirements.

## Verify

Requires Go 1.23 or newer and no third-party Go modules.

Run `./verify.sh` from this directory. The script checks gofmt, unit tests, the race detector and go vet.

## Claim boundary

This PR contains the run classifier, inventory closure and profile compilation reference. It is not a trusted observer, sandbox, runtime hook implementation, effect broker, cryptographic witness or external benchmark result. A digest is not a signature. Runtime completeness and catalogue completeness are not established.

Related work:
- [#11760 — repeated-run task × control assurance protocol](https://github.com/Luminous-Dynamics/symthaea/issues/11760)
- [#11763 — intention-to-treat skill effect estimator](https://github.com/Luminous-Dynamics/symthaea/issues/11763)
- [#11741 — isolated runtime provenance hooks](https://github.com/Luminous-Dynamics/symthaea/issues/11741)
- [#11802 — profile-to-obligation inventory compiler research](https://github.com/Luminous-Dynamics/symthaea/issues/11802)
