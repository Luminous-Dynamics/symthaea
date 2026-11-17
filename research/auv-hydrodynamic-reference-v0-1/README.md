# AUV Hydrodynamic Reference v0.1

Owner: Symthaea #8083 (AUV-HYDRO-CAL-001)  
Research parent: Symthaea #8082 (HYDRO-CO-DESIGN-001)  
Symtropy demo integration: Symtropy #5702 / #5727  
Status: standalone analytical Rust reference; runner qualification pending.

## Purpose

Provide independent hand-checkable equations for:
- quadratic drag from vehicle velocity relative to water;
- body/world frame conversion for water-current velocity;
- hydrostatic buoyancy and weight with explicit sign;
- translational and rotational effective inertia including both dry-body and added terms;
- mechanical power across a force/velocity boundary.

The crate does not import or call the production symthaea-auv hydrodynamics helpers. The example parameters reproduce the current illustrative defaults to make their implications visible. They are not declared physically calibrated.

## Relative-flow drag

For each declared translational axis:

    u_relative_body = u_vehicle_body - R_world_to_body * u_water_world
    F_drag = -0.5 * rho * Cd * A * abs(u_relative) * u_relative

A world-frame force perturbation is not the same thing as a water-flow velocity. A physical current model must alter the relative-flow velocity (or a more complete fluid model), not only add a fixed force to the vehicle.

## Hydrostatics

The example uses positive-up buoyancy:

    B = rho * g * V_displaced
    W = m * g
    F_net_up = B - W

For the current illustrative values rho=998 kg/m3, volume=0.05 m3, mass=50 kg and g=9.81 m/s2:
- B = 489.519 N upward;
- W = 490.5 N downward;
- F_net_up = -0.981 N (slightly negative buoyancy).

This means the current default comment "slightly positively buoyant" does not match its numeric sign under the module's documented convention.

## Effective inertia

A diagonal six-DOF approximation must add dry rigid-body mass/inertia to the corresponding fluid added-mass terms:

    M_translation_effective = m_dry + M_added_translation
    I_rotation_effective = I_dry + I_added_rotation

A rotational inertia term containing only added inertia is not a complete rigid-body rotational mass model.

## Mechanical power boundary

At a force/velocity boundary:

    P_mechanical = F dot v

This is not automatically propeller shaft power or electrical input. Shaft/electrical power requires independently specified propulsor, motor and drive characteristics and their efficiencies. A sum of force-squared terms divided by an efficiency is not dimensionally watts.

## Known-answer fixture

The illustrative surge defaults rho=998 kg/m3, Cd=50 and A=0.03 m2 produce:
- 748.5 N drag magnitude at 1 m/s;
- 46.78125 N at 0.25 m/s;
- four times the drag magnitude when speed doubles at fixed coefficient and area.

These are arithmetic consequences of the declared equation and coefficients, not a claim that the coefficients or forces match a real AUV.

## Run

From this directory:

    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings
    cargo test --all-targets
    cargo run

No third-party dependencies are required.

## Tests

The suite covers exact analytical drag values, quadratic scaling, sign reversal, zero relative flow, world-to-body current rotation, hydrostatic sign, neutral buoyancy, dry-plus-added effective inertia, zero dry-inertia rejection, force-dot-velocity power, and malformed profile/state controls.

## Claim ceiling

Passing these tests proves only that this independent reference matches the equations written here. It does not validate the production AUV simulator, CFD, free-surface effects, viscous flow fidelity, cavitation, thruster/propeller maps, actual electrical energy, endurance, seaworthiness or physical operating safety.
