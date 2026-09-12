# ADR-019 — RSK Verified Schema to Type-State Adapter

- **Status:** Accepted for reference architecture; production admission denied
- **Change Class:** A
- **Scope:** Replicator Safety Kernel semantic provenance / validator configuration
- **Related:** #1668, #1678, #1679, #1938, #1969, #1973, #2035, #2038

## Context

The RSK structural validator intentionally exposes reference constructors that accept a schema ID plus a rule table. This keeps the semantic core testable in isolation.

After adding resource schema v0.2 runtime-ID binding, static composition review showed that production use still needs a stronger rule:

```text
correct digest + arbitrary rule table != verified schema-derived validator
```

Otherwise a caller could pair a known-good schema ID with locally modified required flags, ranges, bit classes, or numeric mappings.

## Decision

Production validator configuration must be a deterministic image of exact verified canonical schema bytes.

The reference tooling therefore adds a schema-to-rule adapter that accepts only the schema object itself and derives:

- capability bit width and assignable/reserved/retired rules;
- resource v0.2 numeric IDs, required flags and min/max bounds;
- exact schema ID by validating/hashing the same canonical bytes.

There is no override-rule input.

## Capability rule

Capability rule output is sorted by numeric bit index and is derived from canonical entries plus canonical reserved-bit declarations.

Unknown bits remain unknown; the adapter does not invent permissive defaults.

## Resource rule

Resource rule output requires resource schema v0.2 or later explicitly admitted equivalent.

Historical v0.1 resource schemas cannot produce a production-target runtime rule table because they did not commit runtime numeric IDs.

Rules are sorted by committed `numeric_id` and derive structural bounds only from the canonical v0.2 dimension bytes.

## Production type-state direction

Reference `Resolved*::new(id, rules)` constructors remain useful for isolated tests but must not become production authority constructors.

Future integration should expose an opaque schema-derived/verified validator type created only from `VerifiedSchemaRegistrySnapshot` resolution.

## Restart rule

Verified validator type state is reconstructed after restart from reverified registry evidence and exact canonical schema bytes. Serialized `verified=true` or cached rule tables cannot recreate authority by themselves.

## Consequences

### Positive

- eliminates side-loaded rule-table substitution;
- makes Python/Rust adapter parity testable;
- ensures resource numeric mapping comes only from digested v0.2 bytes;
- keeps structural validation independent from replication grant authority;
- preserves isolated reference constructors without promoting them to trust roots.

### Costs

- one additional deterministic adapter boundary must be implemented in Rust;
- production validator types need provenance-bearing construction paths;
- restart/replay logic must rebuild trusted validators from verified evidence.

## Rejected alternatives

### Trust any rule table carrying a known schema ID

Rejected because the rule table itself could disagree with the bytes that produced the digest.

### Put mutable rule overrides in the registry

Rejected because runtime meaning would then depend on metadata outside the schema identity.

### Remove reference constructors entirely now

Rejected because the uncompiled Rust stack still benefits from isolated structural testing. The safer incremental path is to keep them explicitly non-production and build a stronger production type state above them.

## Validation status

The Python deterministic adapter and adversarial tests are authored in the stacked candidate branch. GitHub exact-head execution is still pending the repository-wide Actions queue.

Rust production integration is not implemented and remains blocked by #1926 plus the broader verified-registry/evidence gates.

## Production status

**DENIED / NOT YET ELIGIBLE**.
