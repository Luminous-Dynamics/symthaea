# Patch 0015: feat add release delta and risk report

**Series:** 29

## Objective

Summarize exactly what changed between maintained releases and what evidence supports the change.

## Intended changes

- Generate API, schema, canonical-vector, dependency, command, test, benchmark, and documentation deltas.
- Link each corrective change to regression and review records.
- Separate fixes, hardening, compatibility changes, and new experimental work.

## Required evidence

- Unexplained canonical or schema delta blocks release.
- Known risks and waivers remain visible.
- Report reproduces from retained inputs.

## Non-claims

- Does not replace detailed changelogs or advisories.
- Does not imply unchanged code is risk free.
