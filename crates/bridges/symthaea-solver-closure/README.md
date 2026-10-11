# symthaea-solver-closure

Generic, evidence-oriented transitive input-closure semantics for external engineering solvers.

This crate exists because a digest of a top-level netlist/case/model is **not** equivalent to a digest of every byte and ambient configuration source that can influence a numerical result.

It deliberately does not execute solvers, resolve arbitrary paths, download model libraries, or grant shell/process authority. Adapters must first construct an explicit closure from already-admitted artifacts and fail closed when a dependency is unresolved.

Core invariant:

`top-level input digest != complete solver input closure`

Initial consumers are expected to include ngspice, Elmer, Gmsh/OpenFOAM, CAD/meshing adapters, and later acoustics solvers.
