# WCARE-45 — Child-verifier execution promotion protocol v1

Status: `PREREGISTERED_PREFLIGHT`
Authority: `MeasurementOnly`
Protocol version: `wcare45-child-verifier-promotion-v1`

## Purpose

WCARE-44 Stage A can recompute the authenticated-replication candidate graph and validate attribution/contract identity, but deliberately cannot establish that WCARE-42 or WCARE-43 actually executed.

WCARE-45 freezes the Stage-B boundary. It distinguishes exact ancestry convergence from tree integration and executable qualification.

## Exact convergence subject

The first WCARE-45 integration commit is:

`37a532d12b7a57f3332a8c4551400a2d3a407af1`

with exact parents:

- WCARE-44: `972cac01ff6f2b9839ac42bc9b25c073b988ec73`
- WCARE-42: `4bb84790bab3dc6a6d2e30d0ef081950a3954717`
- WCARE-43: `9e4bfd621e3c48ba6335f95b3dd1fd7b36dccf25`

Its tree is intentionally the exact WCARE-44 tree. Therefore:

`exact child ancestry present != child files imported != child verifier executed`

This is a feature, not a shortcut: Stage B must not infer executable qualification from Git ancestry.

## Current fail-closed preflight

The v1 preflight verifies:

1. the exact convergence commit exists;
2. all three exact parent lineages are ancestors of the evaluated HEAD;
3. the convergence commit has the exact expected ordered parent set;
4. the evaluated tree contains the Stage-A WCARE-44 review unit;
5. WCARE-42/WCARE-43 executable prerequisites are not silently inferred from ancestry;
6. final promotion booleans remain false.

At the initial freeze, the integration tree intentionally does not contain the WCARE-42/WCARE-43 child verifier trees and WCARE-42 still lacks a standalone Cargo.lock on its source branch. Therefore the truthful classification is:

`CHILD_EXECUTION_INDETERMINATE`

not promotion.

## Future Stage-B requirements

A later promotion-capable tranche must explicitly integrate and bind the exact child implementation bytes, then re-execute them.

### WCARE-42

Every exact WCARE-40 builder provenance/relation subject required by WCARE-44 must receive one accepted, non-synthetic WCARE-42 result from the exact preregistered verifier. The standalone Cargo.lock must exist and the qualifier must run under `--locked`. Missing, duplicate, untrusted, rejected, synthetic, or indeterminate required subjects block builder-authentication promotion.

### WCARE-43

The exact preregistered verifier/integrity lineage must execute. Final temporal promotion requires a non-synthetic `ESTABLISHED` result over the exact WCARE-40 plan/result and WCARE-41 authentication-plan subject, with strict `T_commit < T_replica` for every qualifying replica. Synthetic fixtures can never be promoted.

### Conjunction

Only after both child execution lineages are independently established and the exact WCARE-44 candidate conjunction is satisfied may a future WCARE-45 result set builder authentication, temporal preregistration, and authenticated-preregistered replication true.

## Promotion monotonicity

A downstream Stage-B verifier may strengthen a claim only by satisfying an explicit missing prerequisite. It may not reinterpret an absent child tree, missing lockfile, queued CI run, synthetic fixture, candidate observation, or ancestry edge as executable evidence.

## Safety boundary

This protocol grants no runtime authority. Even successful future Stage-B promotion does not establish subject correctness, consciousness, phenomenal experience, suffering, moral patienthood, objective moral truth, binding consent, veto/self-preservation authority, or solved alignment.
