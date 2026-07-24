# Supply-chain admission

A signed binary alone is insufficient evidence for a body-coupled actuator release. The Series 42 admission manifest binds:

- final artifact digest;
- source-tree digest;
- lockfile digest;
- compiler and toolchain digest;
- hermetic build-recipe digest;
- SBOM digest;
- every dependency's source locator and content digest;
- license, review, vulnerability and withdrawal status;
- build-script review;
- at least two matching reproducible builds.

The verifier is fail-closed. Git dependencies, unreviewed dependencies, unapproved licenses, known findings, withdrawn packages, stale manifests, or an unreviewed build script are rejected according to policy.
