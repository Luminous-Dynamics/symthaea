# MACE formal effect/fence model

This directory contains a finite TLA+ specification for the destination effect-or-fence protocol.

## What is modeled

- Two independent idempotency keys and two possible intents per key.
- Requests can be sent and remain delayed before destination delivery.
- Effect delivery and fence acquisition serialize through one destination transaction slot.
- The effect path stages both a receipt and a business row; it is externally visible only after one atomic commit.
- The fence path stages a receipt but never a business row.
- Rollback/process crash clears uncommitted staging; it does not create a terminal result.
- A delayed dispatch arriving after a fence is consumed/rejected without replacing the fence.
- Reconciliation distinguishes effect committed, fence committed and not-found-unfenced. Only a committed fence yields a non-execution result.

The model is deliberately a safety model. It does not claim liveness, fairness, throughput, multi-host consensus, physical durability, authorization issuance or arbitrary Byzantine endpoint behavior. Status codes and the exact Go implementation's wire parsing are outside this abstraction.

## Reproducible TLC run

The model is authored for the stable TLA+ tools release v1.7.4 and Java 11+. The upstream release publishes SHA-1 bee4a54f3ee3d4afc347c3240ec2d9e93b075104 for tla2tools.jar. The checker runner requires the user to supply the jar; it does not silently download an unpinned binary.

Commands:

    curl -fL https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar -o tla2tools.jar
    echo 'bee4a54f3ee3d4afc347c3240ec2d9e93b075104  tla2tools.jar' | sha1sum -c -
    TLA2TOOLS_JAR=$PWD/tla2tools.jar ./run-tlc.sh

Run with one TLC worker for a deterministic single-worker state exploration. Do not turn on symmetry or state constraints without recording the exact model configuration and generated-state count.

## Evidence status

The Go finite-state model remains executed and passes its bounded 68-state / 200-transition exploration. This TLA+ module/config are newly authored formal artifacts. **TLC has not yet been run on this module** in the current authoring environment: Java 21 is present, but no TLA+ tools JAR is installed and outbound binary retrieval is unavailable. Do not report this as TLA+/TLC verified until the pinned runner completes with a zero-error result and the TLC version, state count and trace digest are recorded.
