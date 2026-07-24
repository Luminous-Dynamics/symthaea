# Patch 0012: security inventory and block all mutation bypasses

**Series:** 26

## Objective

Ensure no older direct API, CLI, route, job, or helper can bypass segment, freeze, cycle, or retirement state.

## Intended changes

- Generate a mutation-surface inventory from code and configuration.
- Route every mutation through typed transition gates and compare-and-commit.
- Deprecate or make private legacy direct mutation helpers.

## Required tests

- Compile-time and runtime inventory tests cover every mutation surface.
- Post-freeze and post-retirement bypass attempts fail.
- Queued and cached operations revalidate at commit.

## Non-claims

- Does not control unmodeled third-party binaries.
- Does not remove read-only verification APIs.
