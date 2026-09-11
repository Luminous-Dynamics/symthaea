# Continuity Fenced Resource Lab V1

## Status

`CONTAINED_SIMULATION / NOT_PRODUCTION_QUALIFICATION`

This evidence note freezes the first contained resource-boundary lab for the #1528 actuation-fencing theorem.

It grants no production execution authority, does not qualify NETCONF/Redfish/gNOI/storage/BMC/Spore adapters, and does not prove any external device enforces fencing.

## Purpose

The lab tests one deliberately small theorem using only a caller-supplied SQLite file:

```text
stale or replayed authority
must be rejected by the same transactional boundary
that checks the fence and mutates the simulated resource
```

The resource boundary stores:

- exact resource ID;
- exact backend ID;
- exact enforcement-profile ID;
- current monotonic fence generation;
- current permit/deny disposition;
- sticky emergency-stop state;
- simulated resource value;
- durably consumed token IDs.

A permit token binds the exact resource/backend/profile identities, fence generation, disposition, fresh challenge and token identity.

## Atomic mutation boundary

`actuate` executes inside one SQLite `BEGIN IMMEDIATE` transaction.

Inside that same transaction it:

1. reads the current resource/fence state;
2. rejects wrong resource/backend/profile identity;
3. rejects stale generation;
4. rejects emergency-stop / deny state;
5. rejects a previously consumed token;
6. mutates the simulated resource value;
7. inserts the consumed-token record;
8. commits both together.

The lab therefore does not model the unsafe pattern:

```text
check fence
commit check
later mutate resource somewhere else
```

It models one transactional check-and-mutate boundary.

## Adversarial scenarios

`scripts/test-continuity-fenced-resource-lab.py` executes the lab as separate subprocesses and requires:

1. **stale holder rejection** — generation `N` is rejected after the resource advances to `N+1`;
2. **one-use consumption** — a successfully consumed permit cannot mutate twice;
3. **durable reopen** — committed resource value and token-consumption state survive closing/reopening the SQLite database;
4. **crash-before-commit atomicity** — the actuator process exits with code `91` after executing the mutation statements but before `COMMIT`; reopening the database must show neither the value change nor token consumption, after which the exact same current permit may execute once;
5. **emergency-stop dominance** — a newer emergency-stop generation makes older permits stale;
6. **deny cannot mutate** — the deny token itself cannot perform a resource mutation;
7. **sticky emergency stop** — V1 refuses to issue a later permit after emergency stop;
8. **cross-resource rejection** — a valid token for another resource is rejected even when backend/profile identities and generation are otherwise compatible.

## Executed local result

Before repository commit, the exact lab/harness pair was syntax-checked and executed in a local Linux/Python environment.

Observed summary:

```text
status=PASS
final_generation=4
final_value=12
consumed_count=2
```

The executed property set was:

- `stale_generation_rejected`;
- `one_use_replay_rejected`;
- `committed_state_survives_reopen`;
- `crash_before_commit_is_atomic`;
- `emergency_stop_dominates`;
- `deny_token_cannot_mutate`;
- `cross_resource_token_rejected`.

Local execution is not repository qualification; the dedicated exact-head workflow is the repository evidence boundary.

## Dedicated workflow

`.github/workflows/continuity-fenced-resource-lab.yml` uses:

- exact PR-head checkout;
- immutable `actions/checkout` SHA;
- read-only repository permissions;
- Python syntax checks;
- recorded Python/SQLite versions;
- execution of the full subprocess harness.

## Relationship to #1549 / #1550 / #1528

This lab is downstream experimental evidence, not a replacement for the closed-world qualification protocol.

It should eventually contribute evidence to the nine-obligation campaign only after:

- the parent Rust continuity stack compiles/qualifies;
- the campaign-bound evidence wrapper exists and matches #1578's independent oracle;
- #1550 defines verifier-owned admission;
- the lab campaign binds exact harness/environment/toolchain identities.

Even then, qualification of this SQLite simulation does not imply that a switch, BMC, storage controller, hypervisor, database or Spore privileged helper enforces the same property.

Each real adapter needs its own backend/resource campaign under #1528.

## Non-claims

This lab does not establish:

- cryptographic authentication of permits;
- current owner/verifier authority;
- a production execution capability;
- hardware monotonic storage;
- resistance to malicious privileged modification of the SQLite file;
- multi-host consensus correctness;
- NETCONF/gNMI/Redfish/gNOI behavior;
- production crash consistency outside SQLite's transactional model;
- any permission to mutate real infrastructure.
