use marine_autonomy_vector_bench::{
    assess_case, AutonomyCase, AutonomyProfile, EvidenceClass, Interval, ResourceBudget,
};

fn row(id: &str, stock: f64, use_per_day: f64) -> ResourceBudget {
    let unit = if id == "food_protein" { "kg" } else { "L" };
    ResourceBudget {
        dimension_id: id.to_owned(),
        quantity_unit: Some(unit.to_owned()),
        rate_unit: Some(format!("{unit}/day")),
        usable_stock: Some(Interval::new(stock, stock)),
        net_depletion_per_day: Some(Interval::new(use_per_day, use_per_day)),
        configuration_digest: Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned()),
        operating_case_id: Some("synthetic-offshore-v1".to_owned()),
        conditions_digest: Some("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned()),
        evidence_ref: Some(format!("synthetic:{id}")),
        evidence_class: Some(EvidenceClass::Synthetic),
    }
}

fn main() {
    let case = AutonomyCase {
        profile: AutonomyProfile {
            profile_id: "synthetic-liveaboard-30d".to_owned(),
            profile_manifest_digest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            target_days: 30.0,
            required_dimension_ids: vec![
                "potable_water".to_owned(),
                "food_protein".to_owned(),
                "blackwater_headroom".to_owned(),
            ],
        },
        configuration_digest: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
        operating_case_id: "synthetic-offshore-v1".to_owned(),
        conditions_digest: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
        resources: vec![
            row("potable_water", 180.0, 20.0),
            row("food_protein", 600.0, 20.0),
            row("blackwater_headroom", 240.0, 20.0),
        ],
    };

    let report = assess_case(&case);
    println!("{report:#?}");
}
