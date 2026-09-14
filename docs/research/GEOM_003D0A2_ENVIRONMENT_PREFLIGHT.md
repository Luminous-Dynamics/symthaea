# GEOM-003D0A2 — Pre-construction environment preflight

Status: implementation candidate. This tranche was defined without executing or inspecting a target GWT lesion outcome.

Parent authority: #3157
Implementation issue: #3161
Exact predecessor: `ca43e28cbeeea9c7b1b1b048da18716041ca923c`

## Purpose

A reproducible GEOM campaign must reject ambient state **before** constructing the target cognitive loop. This is especially important for `SYMTHAEA_THRESHOLD_OVERRIDES_PATH`: the real constructor reads that environment-selected file once and can replace compile-time cognitive thresholds with a promoted phenotype. A check after construction would only discover contamination after it had already changed the agent.

D0A2 therefore lives in a standalone bridge crate and accepts a `CognitiveLoopConfig` before `CognitiveLoopService::new` is called.

## Primary-campaign contract

The preflight requires:

- fixed UTC hour finite and in `[0,24)`;
- explicit non-empty `genesis_phrase`;
- `async_training == false`;
- `enable_online_learning == false`;
- absence of `SYMTHAEA_THRESHOLD_OVERRIDES_PATH`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MYCELIX_CONDUCTOR_URL`, and `MYCELIX_APP_ID`;
- an existing, readable, empty campaign persistence directory;
- explicit absolute `aesthetic_memory_path` strictly inside that directory;
- any configured `memory_db_path` or `epistemic_auditor_db_path` also strictly inside the directory;
- no parent-directory (`..`) traversal in accepted persistence paths.

The first GEOM campaign intentionally uses compile-time thresholds only. A future campaign may study a promoted threshold phenotype, but that requires a new preregistration that hash-pins the exact file before execution.

## Privacy and evidence

The report records only whether a forbidden environment variable was present. Secret values are never copied into the report.

The caller-supplied environment-snapshot form is itself fail-closed: it must contain every forbidden variable exactly once. This prevents a malformed synthetic snapshot from omitting a channel and being mistaken for a clean environment.

The preflight does not mutate the process environment, delete historical files, create persistence state, construct the cognitive loop, or execute an intervention.

## Persistence semantics

For the primary campaign, an explicit aesthetic path is required even when a build happens not to use the `creative` feature. This keeps the environmental contract invariant under later feature-set changes and prevents the known `.claude/aesthetic_memory.json` fallback from silently returning if `creative` is enabled.

`memory_db_path=None` is permitted because the cognitive-loop configuration defines that case as in-memory session state. `epistemic_auditor_db_path=None` is permitted because it disables the auditor persistence path. If either path is supplied, it is isolated under the campaign root.

The campaign root must be empty *at preflight time*. Later creation of declared files belongs to the run lineage and must be captured by the environment manifest/postflight authority.

## Required controls

The candidate tests:

- clean preflight pass;
- each forbidden environment variable independently;
- malformed/incomplete environment snapshot;
- invalid fixed UTC values;
- missing genesis phrase;
- async training and online learning rejection;
- missing, non-directory, and non-empty campaign roots;
- missing/relative aesthetic path;
- parent traversal and out-of-root path escape;
- accepted optional auditor path inside the root.

D0A2 is necessary but not sufficient for D0/D1. The exact-sham reproducibility gate, pre/post environment manifest, measurement qualification, and frozen statistical authority remain mandatory.
