# symthaea-energy-discovery

Energy-domain semantics for the generic `symthaea-discovery` contracts.

This crate answers **what an energy candidate is being asked to do**. It deliberately does not implement optimizers, solvers, material models, economic models, or deployment authority.

## Why application context is first-class

There is no universal "best battery" or "best generator". A technology that is attractive for an electric vehicle may be poor for twelve-hour grid storage, while a low-energy-density storage technology may be excellent for a stationary microgrid.

`EnergyDiscoveryProfile` therefore keeps objectives separate and application-dependent. It never hides them behind one weighted score. Existing Pareto machinery can operate on the resulting objective set.

## Initial applications

- electric vehicles
- grid frequency response
- grid storage with an explicit minimum duration
- microgrids with an explicit minimum duration
- remote power with an explicit minimum duration
- spacecraft power/storage
- bulk electricity generation
- industrial heat

Duration requirements are hard service constraints only when the application itself states a minimum duration. Other thresholds remain caller-owned rather than being invented by this crate.

## Evidence and authority boundary

This crate creates descriptive candidate/profile records only. Numerical values arrive later as `Prediction`s from validated models, simulations, experiments, or field evidence. The crate cannot run experiments, invoke solvers, procure equipment, select a winner for deployment, or authorize physical action.
