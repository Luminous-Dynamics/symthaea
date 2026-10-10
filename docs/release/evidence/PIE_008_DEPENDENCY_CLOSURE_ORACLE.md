# PIE-007/008 Dependency-Closure Oracle Evidence

Status: independent structural reference; synthetic execution evidence only.

## Purpose

`scripts/pie-dependency-closure-oracle.py` freezes the structural distinction between:

- capabilities that are locally reproducible;
- capabilities that are operationally reachable only because declared imports are available;
- capabilities that remain unavailable.

The oracle is intentionally independent of Symthaea code.

## Executed self-test

Executed locally on 2026-09-11 with Python 3. The script returned `ok`.

Synthetic fixtures verified:

- local metal can close a simple frame-production route;
- a motor that requires an imported bearing remains import-dependent;
- a downstream machine inherits that import dependence;
- removing the bearing makes the motor/machine unavailable;
- adding a complete local bearing route closes the downstream chain locally;
- an unsupported `A <-> B` dependency cycle does not bootstrap itself;
- an all-local alternative route dominates an import-dependent route for closure classification;
- importing a finished machine makes it operationally available but not locally reproducible;
- removing a structurally important import reports downstream target and capability loss;
- undeclared/missing dependencies fail closed as unreachable.

## Semantics

Production recipes form an AND/OR graph:

- all dependencies of one recipe are required (AND);
- any complete recipe may produce an output (OR);
- local closure is computed from local primitive capabilities only;
- operational reachability is computed with local primitives plus declared imports;
- unsupported cycles remain unresolved by fixed-point construction.

## Import leverage boundary

Current import leverage is structural only: it counts capabilities/targets lost when one import is removed. It does **not** yet weight mass, cost, production rate, mission importance, lifetime, inventory, or economic value.

Those quantitative extensions belong in later PIE-008/009 work.

## Non-claims

Structural closure does not prove that a recipe is physically feasible, has enough throughput, can be maintained, has sufficient inventory, or is safe/qualified. No lunar or Martian real-world dependency data are asserted in this oracle.

Tracks #1618 and master #1604.
