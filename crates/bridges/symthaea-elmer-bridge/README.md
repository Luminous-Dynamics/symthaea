# symthaea-elmer-bridge

Generic, closure-bound Elmer FEM/multiphysics adapter boundary for Symthaea engineering.

ENG-FEM-001A intentionally implements **case identity and result parsing before process execution**.

Current guarantees:

- the top-level SIF is the closure `Primary` artifact;
- the exact Elmer executable bytes and reported identity are closure-bound;
- meshes, material/property data, plugins, configuration, and environment influences can be represented by the shared `symthaea-solver-closure` graph;
- the adapter-owned `SaveScalars` output schema is itself bound as a closure `Configuration` artifact;
- scalar output parsing is strict, finite, column-count checked, and independent of arbitrary solver-log text;
- output filenames cannot escape the future case working directory;
- dry-run results remain non-evidence;
- real execution fails closed until the shared `CommandSolver` can bind working directory and ambient environment under ENG-EXEC-001 (#5666).

This crate must not grow a private subprocess runner. Once ENG-EXEC-001 lands, real Elmer execution should reuse the existing hardened timeout/output-drain machinery in `symthaea-sim-bridge` and bind its exact execution context into the shared solver closure.

`case valid != solver executed != converged != model validated != physically measured`
