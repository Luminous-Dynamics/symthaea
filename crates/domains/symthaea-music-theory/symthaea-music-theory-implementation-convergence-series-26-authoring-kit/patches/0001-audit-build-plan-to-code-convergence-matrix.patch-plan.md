# Patch 0001: audit build plan to code convergence matrix

**Series:** 26

## Objective

Map every Series 22–25 plan item to an exact Rust module, persisted role, public API, command, fixture, and release claim before implementation begins.

## Intended changes

- Inventory all 99 planned patches from grounded Series 22–25 and classify them as model, policy, verifier, transaction, persistence, tooling, test, or documentation work.
- Record dependencies, duplicate concepts, incompatible names, and places where one shared primitive should replace multiple planned implementations.
- Require every planned public claim to name executable evidence and its production path.

## Required tests

- The matrix fails when a plan item has no implementation owner or verification evidence.
- Duplicated schema roles and competing state transitions are identified before code lands.
- No documentation-only item can satisfy an implementation requirement.

## Non-claims

- Does not claim the plans are already correct.
- Does not preserve plan wording when implementation evidence demands revision.
