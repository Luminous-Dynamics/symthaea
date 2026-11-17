use auv_hydrodynamic_reference::{
    HydroReferenceProfile, evaluate_hydrostatics, evaluate_relative_flow_drag,
    effective_inertia, mechanical_power_w, FlowCase,
};

fn main() {
    let profile = HydroReferenceProfile {
        fluid_density_kg_m3: 998.0,
        drag_coefficient_dimensionless: [50.0, 100.0, 100.0],
        reference_area_m2: [0.03, 0.15, 0.15],
        dry_mass_kg: 50.0,
        displaced_volume_m3: 0.05,
        dry_inertia_kg_m2: [1.0, 2.0, 3.0],
        added_mass_kg: [10.0, 30.0, 30.0],
        added_inertia_kg_m2: [0.5, 5.0, 5.0],
        gravitational_acceleration_m_s2: 9.81,
    };
    let flow = FlowCase {
        vehicle_velocity_body_m_s: [1.0, 0.0, 0.0],
        water_velocity_world_m_s: [0.0; 3],
        body_to_world_wxyz: [1.0, 0.0, 0.0, 0.0],
    };

    let drag = evaluate_relative_flow_drag(&profile, &flow).expect("valid reference flow");
    let hydrostatics = evaluate_hydrostatics(&profile).expect("valid reference hydrostatics");
    let inertia = effective_inertia(&profile).expect("valid reference inertia");
    let power = mechanical_power_w([10.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        .expect("valid force/velocity boundary");

    println!("suite=AUV-HYDRODYNAMIC-REFERENCE-001");
    println!("relative_velocity_m_s={:?}", drag.relative_velocity_body_m_s);
    println!("surge_drag_n={}", drag.drag_force_body_n[0]);
    println!("net_hydrostatic_up_n={}", hydrostatics.net_force_up_n);
    println!("effective_translation_mass_kg={:?}", inertia.translation_kg);
    println!("effective_rotation_inertia_kg_m2={:?}", inertia.rotation_kg_m2);
    println!("mechanical_power_w={power}");
    println!("note=analytical references only; no CFD or physical calibration claim");
}
