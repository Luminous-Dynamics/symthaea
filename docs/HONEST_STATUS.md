# Symthaea Current Status

<!-- repo-status: claims.toml -->

This page is the human-readable status entry point for the repository.

The previous contents of this file were a **February 5, 2026 snapshot**. They included release-era counts and broad labels such as "Production Ready (100%)" that no longer describe the current research repository precisely. That snapshot remains available in Git history, but it is not current qualification evidence.

## Current authority

Repository status is now split deliberately:

- `repo.toml` — machine-readable repository maturity, Repository Conformance level, policy, and promotion boundary.
- `claims.toml` — machine-readable registry for claims that have been migrated into the new integrity system.
- `docs/CLAIMS.md` — deterministic human-readable rendering of `claims.toml`.
- `README.md` — current project overview, limitations, quick start, and evidence table.

The claims migration is currently **partial**. Absence from `claims.toml` does not establish that a claim is false or unsupported; it means that claim has not yet been migrated into this registry.

## Current maturity and conformance boundary

Symthaea is an **experimental/research cognitive architecture**. The repository contract marks it as:

- canonical repository: yes;
- maturity: research;
- production ready: no;
- externally security audited: no;
- Repository Conformance level: RC-1 candidate for this exact source tree;
- canonical-branch protected promotion: not asserted by this profile.

RC-1 here means the repository has machine-readable claim scope, evidence, limitations, contained evidence paths, deterministic human rendering, and adversarially tested integrity tooling once the exact-head hosted lane passes. It is a repository-process statement, not a product-safety, scientific-validity, or production-readiness certification.

Because protected promotion is not asserted, RC-3 is not claimed.

Individual cryptographic primitives, benchmarks, experiments, or subcrates may have stronger local evidence. Those local results do not automatically promote the entire repository to production-ready or externally validated status.

## Claim scope

Every migrated claim has an explicit semantic scope. Evidence about one scope must not silently promote another.

For example:

```text
benchmark-profile evidence
    !=
whole-repository qualification
```

and:

```text
consciousness-related internal indicator
    !=
proof of consciousness
```

## Scientific and safety boundaries

The current project does **not** claim that:

- Symthaea proves consciousness or sentience;
- internal consciousness-relevant indicators establish subjective experience;
- an internal benchmark is equivalent to independent replication;
- a working cryptographic primitive is equivalent to legal, medical, financial, or regulatory compliance.

Where historical measurements are retracted or superseded, that status should remain explicit rather than being silently removed.

## Historical snapshots

Older status reports remain useful as historical evidence about what was believed, measured, or implemented at a particular commit. They are not self-refreshing authority. A historical status statement transfers forward only when current evidence and the current repository contract support it.
