# BASALT Cargo build-script environment fixture

Two dependency-free Cargo packages read BASALT_BUILD_INPUT from the Cargo invocation environment. The hidden case omits rerun-if-env-changed; the positive control declares it. The runner supplies baseline and mutated values explicitly and stores only SHA-256 value commitments in evidence.
