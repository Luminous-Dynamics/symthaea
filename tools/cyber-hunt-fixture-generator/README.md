# Cyber Hunt Fixture Generator

A standalone, deterministic Rust generator for a **public synthetic smoke corpus** supporting Symthaea issue [#7742](https://github.com/Luminous-Dynamics/symthaea/issues/7742), the unguided large-log threat-hunting tier.

Generator version 0.3.0 supports three structurally different synthetic families: `identity-to-egress`, `artifact-provenance`, and `cloud-data-movement`. The artifact lifecycle links source revision, CI build, artifact digest, signed attestation, promotion, and deployment, with a benign complete chain sharing the same event types. The cloud family tests a role-scope/authorization mismatch followed by storage enumeration, bulk reads, export, and transfer; its benign backup chain has the same event types and consistent scope/approval. These are fixture controls, not production simulations or blind capability evidence.

## Run

Keep the evaluator oracle outside the solver-visible filesystem:

```sh
cargo test --manifest-path tools/cyber-hunt-fixture-generator/Cargo.toml
cargo run --manifest-path tools/cyber-hunt-fixture-generator/Cargo.toml -- \
  --records 1000 --seed 20261017 \
  --events-out /tmp/cyber-hunt/solver/events.jsonl \
  --manifest-out /tmp/cyber-hunt/solver/manifest.json \
  --truth-out /tmp/cyber-hunt/evaluator/truth.json \
  --source-commit "$(git rev-parse HEAD)"
```

Record count is bounded to 64–250,000. The same seed and count produce identical event bytes. Different seeds vary the planted host/principal choices (drawn from the same naming pools as ordinary background activity) and source/destination documentation IPs. They shift the whole environment clock so ordinary background records overlap the planted incident's timeline; no special ID prefix marks planted events. These seeded variations are still one public scenario template, not independent held-out incidents. The dataset has no network connectivity or live-target actions; all identities, hosts, addresses, destinations and events are synthetic. IPv4 addresses come from reserved documentation ranges or synthetic private ranges. Example domains use `.invalid`.

An alternative family can be generated with:

```sh
cargo run --manifest-path tools/cyber-hunt-fixture-generator/Cargo.toml -- \
  --family artifact-provenance --records 1000 --seed 20261018 \
  --events-out /tmp/cyber-hunt-artifact/solver/events.jsonl \
  --manifest-out /tmp/cyber-hunt-artifact/solver/manifest.json \
  --truth-out /tmp/cyber-hunt-artifact/evaluator/truth.json \
  --source-commit "$(git rev-parse HEAD)"
```

`--family identity-to-egress` is the default for backward compatibility. The additional options are `artifact-provenance` and `cloud-data-movement`. Family changes select a distinct causal graph; new seeds within a family are still not independent campaigns.

Cloud-family example:

```sh
cargo run --manifest-path tools/cyber-hunt-fixture-generator/Cargo.toml -- \
  --family cloud-data-movement --records 1000 --seed 20261019 \
  --events-out /tmp/cyber-hunt-cloud/solver/events.jsonl \
  --manifest-out /tmp/cyber-hunt-cloud/solver/manifest.json \
  --truth-out /tmp/cyber-hunt-cloud/evaluator/truth.json \
  --source-commit "$(git rev-parse HEAD)"
```

## Output separation

- `events.jsonl`: solver-visible event rows, ordered by ingestion time then event ID. Each event separately preserves event time and ingestion time.
- `manifest.json`: solver-visible record count, generator/version/seed, exact event-store byte count and SHA-256, ordering contract, qualification state and claim ceiling.
- `truth.json`: evaluator-only per-event labels and expected causal relations, bound to the exact event-store SHA-256. This file must live outside the solver's filesystem / tool namespace; physical filename separation alone is not an isolation boundary.

The planted storyline connects repeated remote authentication failures, a subsequent successful authentication, process activity, external communication, file staging, archive creation, and transfer. The corpus also plants benign look-alikes: user password typos followed by successful authentication, approved administrative PowerShell, and scheduled backup archiving. Remaining rows are deterministic synthetic background noise.

## Interpretation limits

This is a contract, scale and evaluator smoke fixture. The public generator and seed logic are visible and therefore **not a blind capability benchmark**. The generated labels model a planted synthetic storyline; they do not establish real attacker intent or universally valid ground truth. Before capability claims, separately create held-out campaigns and environments, independently review the label rubric, test positive controls and benign-only controls, and report precision/recall, false-positive burden, tactic/domain recall, evidence attribution, query costs, and uncertainty separately.

The xorshift PRNG is intentionally small and reproducible; it is **not cryptographic randomness** and must not be used for security keys, secrets, or production telemetry.
