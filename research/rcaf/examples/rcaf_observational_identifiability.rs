use cetb_rcaf::observational_identifiability::{
    assess_observational_identifiability, construct_observationally_equivalent_pair,
};
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let clean = (0..12)
        .map(|time| {
            (0..4)
                .map(|roi| ((time as f64 + 1.0) * (roi as f64 + 1.0) * 0.17).sin())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let spatial = [0.5, -0.5, 0.5, -0.5];
    let temporal = (0..12)
        .map(|time| (time as f64 * 0.23).sin())
        .collect::<Vec<_>>();
    let pair =
        construct_observationally_equivalent_pair(&clean, &spatial, &temporal, 0.3)?;
    let equivalent = assess_observational_identifiability(
        &pair.target_change_world,
        &pair.artifact_only_world,
        0.0,
    )?;

    let mut distinct_world = pair.artifact_only_world.clone();
    distinct_world[0][0] += 0.01;
    let distinct = assess_observational_identifiability(
        &pair.target_change_world,
        &distinct_world,
        1e-6,
    )?;

    let rendered = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"rcaf.observational-identifiability/0.1.0\",\n",
            "  \"source_status\": \"synthetic_equivalence_fixture\",\n",
            "  \"equivalence_case\": {{\n",
            "    \"resolution_tolerance\": {:.12},\n",
            "    \"max_absolute_difference\": {:.12},\n",
            "    \"root_mean_square_difference\": {:.12},\n",
            "    \"compared_values\": {},\n",
            "    \"observationally_equivalent_at_resolution\": {},\n",
            "    \"disposition\": \"{}\"\n",
            "  }},\n",
            "  \"distinct_control\": {{\n",
            "    \"resolution_tolerance\": {:.12},\n",
            "    \"max_absolute_difference\": {:.12},\n",
            "    \"root_mean_square_difference\": {:.12},\n",
            "    \"compared_values\": {},\n",
            "    \"observationally_equivalent_at_resolution\": {},\n",
            "    \"disposition\": \"{}\"\n",
            "  }},\n",
            "  \"claim_limits\": [\n",
            "    \"exact constructive equivalence only; no empirical data analysed\",\n",
            "    \"the declared tolerance must be set from instrument/data resolution before analysis\",\n",
            "    \"observational distinction is not causal or biological identification\",\n",
            "    \"the correct outcome for equivalent worlds is abstention\"\n",
            "  ]\n",
            "}}\n"
        ),
        equivalent.resolution_tolerance,
        equivalent.max_absolute_difference,
        equivalent.root_mean_square_difference,
        equivalent.compared_values,
        equivalent.observationally_equivalent_at_resolution,
        equivalent.disposition.as_str(),
        distinct.resolution_tolerance,
        distinct.max_absolute_difference,
        distinct.root_mean_square_difference,
        distinct.compared_values,
        distinct.observationally_equivalent_at_resolution,
        distinct.disposition.as_str(),
    );

    if let Some(path) = std::env::args_os().nth(1) {
        fs::write(path, rendered)?;
    } else {
        io::stdout().write_all(rendered.as_bytes())?;
    }
    Ok(())
}
