# Patch 0009: security run recurring dependency and advisory gates

**Series:** 29

## Objective

Continuously re-evaluate the supply chain as advisories and releases change.

## Intended changes

- Run lockfile, license, advisory, abandoned-dependency, duplicate-version, and optional-feature scans on a defined cadence.
- Record tool and database versions.
- Require explicit disposition for newly discovered risks.

## Required evidence

- A newly prohibited advisory blocks the next release.
- Historical scan results remain reproducible from retained databases where possible.
- Feature-specific dependencies are included.

## Non-claims

- Does not guarantee advisory databases are complete.
- Does not automatically update dependencies without tests.
