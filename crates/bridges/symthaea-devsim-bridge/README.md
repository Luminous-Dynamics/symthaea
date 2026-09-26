# symthaea-devsim-bridge

Semiconductor-specific typed boundary for the bounded DEVSIM reference path in ENG-SEMI-REF-001C.

This crate deliberately keeps semiconductor geometry, regions, doping assumptions, contacts, mesh policy, model profile, temperature, bias sweep, solver settings, and requested observables out of the generic scalar `SimulationRequest` surface.

REF-001C-1 owns only typed request semantics, fail-closed validation, and deterministic request identity.

It does **not** render DEVSIM input yet, execute Python or DEVSIM, parse solver output, establish convergence, qualify mesh independence, compare against the analytical oracle, or grant fabrication/process authority.

Core boundary:

`typed request accepted != DEVSIM input rendered != solver executed != physics validated != physical device validated`
