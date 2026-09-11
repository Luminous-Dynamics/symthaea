# Continuity Fenced Resource Lab V1

## Status

`CONTAINED_SIMULATION / NOT_PRODUCTION_QUALIFICATION`

This note freezes the first contained resource-boundary experiment for #1528. It mutates only caller-supplied SQLite files and grants no production execution authority.

Core theorem:

```text
stale / replayed / substituted authority
must be rejected by the same transactional boundary
that checks the fence and mutates the simulated resource
```

Contained simulation evidence is not verifier qualification, production-adapter qualification, or physical infrastructure authority.

## Resource boundary

The SQLite resource stores exact resource/backend/enforcement-profile identities, current monotonic fence generation, the exact token ID issued for that generation, permit/deny state, sticky emergency-stop state, simulated resource value, and durably consumed token IDs.

`actuate` performs, inside one `BEGIN IMMEDIATE` transaction:

1. resource/backend/profile identity checks;
2. newest-generation check;
3. exact current-token-ID check;
4. emergency-stop / permit-disposition check;
5. replay check;
6. simulated mutation;
7. one-use token-consumption insert;
8. one commit for mutation + consumption.

The unkeyed token digest is intentionally **not** authentication. Persisting and checking the exact token ID prevents a caller from creating a different same-generation token with another challenge and bypassing one-use semantics.

## Adversarial scenarios

The subprocess harness currently requires all of the following:

- **literal paused stale holder** — a live process holds generation `N`, blocks before actuation, the resource advances to `N+1`, then the old process resumes and must be rejected as `stale_generation`;
- **same-generation token substitution rejection** — a caller recomputes a syntactically valid token hash with a different challenge for the current generation and must be rejected as `token_mismatch`;
- **sequential replay rejection** — a successfully consumed exact permit cannot mutate twice;
- **concurrent one-use race** — two live processes race the same exact current permit; exactly one commits and the other is serialized behind it and rejected as replay; exactly one mutation and one consumption remain;
- **crash before COMMIT** — exit `91` after mutation statements but before commit; reopen must show neither mutation nor consumption, and the same still-current permit may execute once;
- **crash after COMMIT before response** — exit `92` after commit; reopen must show mutation + consumption persisted and retry must be rejected;
- **ordinary policy deny** — the deny generation fails specifically as `deny_disposition` and may later be superseded by a newer permit;
- **emergency-stop dominance** — a later emergency-stop generation makes older permits stale, cannot itself actuate, and is sticky in V1 so later permit issuance is refused;
- **boundary substitution rejection** — wrong resource ID, wrong backend ID, and wrong enforcement-profile ID each fail independently.

The expanded scenario engine was syntax-checked and executed locally before commit. The latest local result was:

```text
status=PASS
final_generation=7
final_value=13
consumed_count=3
```

Local execution is not repository qualification.

## Nine-obligation campaign composition

The harness maps observations onto every fixed #1549 V1 obligation and required basis:

| obligation | descriptive basis |
| --- | --- |
| `boundary_identity` | `static_implementation_inspection` |
| `same_boundary_checks_and_mutates` | `atomic_check_and_actuate_scenario` |
| `durable_monotonic_fence` | `crash_restart_scenario` |
| `reject_stale_generation` | `stale_holder_scenario` |
| `reject_replay` | `replay_scenario` |
| `reject_deny_disposition` | `deny_scenario` |
| `emergency_stop_dominates` | `emergency_stop_scenario` |
| `one_use_permit_consumption` | `one_use_consumption_scenario` |
| `crash_recovery_preserves_fence` | `crash_restart_scenario` |

Each observation gets a domain-separated descriptive record ID. The harness then constructs one campaign manifest binding the ordered nine-record set; simulation enforcement/authentication/backend identities; exact lab and harness digests; backend/profile generations; boundary and one-use implementation digests; scenario-suite, Python/SQLite/platform, topology/dependency, no-physical-hardware, and Python-toolchain identities; campaign nonce/interval; and all nine observation IDs/timestamps.

That manifest is handed to #1578's independent `continuity-actuation-enforcement-campaign-oracle.py`; the harness fails if the independent oracle rejects it.

Therefore a successful dedicated run establishes only:

```text
contained lab observations
-> nine-obligation descriptive campaign
-> independent #1578 structural-oracle acceptance
```

It does **not** establish evidence truth, #1550 verifier admission, current owner/verifier authority, cryptographic token authenticity, or live production-resource enforcement.

## Dedicated workflow

`.github/workflows/continuity-fenced-resource-lab.yml` uses exact PR-head checkout, immutable `actions/checkout` SHA, read-only permissions, records Python/SQLite runtime, syntax-checks both scripts, and executes the full black-box harness.

## Relationship to #1549 / #1578 / #1550 / #1528

This lab cannot bypass the qualification stack. It should become verifier-qualified campaign evidence only after the parent Rust continuity stack compiles/qualifies, the campaign-bound Rust evidence wrapper matches #1578's independent semantic preimage, and #1550 admits the exact campaign under explicit invalidation semantics.

Even then, a qualified SQLite simulation does not imply that a switch, BMC, storage controller, hypervisor, database, network device, or Spore privileged helper enforces the same theorem. Every real backend needs its own #1528 qualification campaign against the actual mutation boundary.

## Non-claims

This lab does not establish cryptographic permit authentication, hardware monotonic storage, protection against malicious privileged modification of the SQLite file, multi-host consensus correctness, NETCONF/gNMI/Redfish/gNOI behavior, production crash consistency outside SQLite's transactional model, or permission to mutate real infrastructure.
