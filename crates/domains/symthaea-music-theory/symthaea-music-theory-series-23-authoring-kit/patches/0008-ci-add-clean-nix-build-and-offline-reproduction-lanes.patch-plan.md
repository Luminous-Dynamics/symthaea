# Patch 0008: Add clean Nix build and offline reproduction lanes

**Series:** 23

## Objective

Prove that the release does not depend on mutable local state or undeclared network access.

## Intended changes

- Build from a clean source export through the project Nix entrypoint.
- Run a second build with network access disabled after dependencies are realized.
- Compare declared outputs and evidence artifacts.

## Required tests

- Dirty workspace inputs do not enter the derivation.
- An undeclared runtime fetch fails.
- Two clean builds produce identical public artifacts.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
