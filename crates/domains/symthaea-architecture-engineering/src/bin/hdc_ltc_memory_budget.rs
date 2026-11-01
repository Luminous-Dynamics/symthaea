use std::env;
use std::process::ExitCode;

use symthaea_architecture_engineering::edge_intelligence_benchmark::{
    assess_hdc_ltc_sram_budget, estimate_hdc_ltc_memory,
};

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("hdc-ltc-memory-budget: {message}");
    ExitCode::FAILURE
}

fn parse_u64(value: &str, field: &str) -> Result<u64, String> {
    value.parse::<u64>().map_err(|_| format!("{field} must be an unsigned integer"))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 4 {
        return fail(
            "usage: hdc_ltc_memory_budget <dimension> <layer-sizes-comma-separated> <sram-bytes> <protected-reserve-bytes>",
        );
    }

    let dimension = match args[0].parse::<usize>() {
        Ok(value) => value,
        Err(_) => return fail("dimension must be an unsigned integer"),
    };
    let layer_sizes = match args[1]
        .split(',')
        .map(|part| part.parse::<usize>().map_err(|_| "layer sizes must be positive integers"))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    let sram = match parse_u64(&args[2], "sram-bytes") {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    let reserve = match parse_u64(&args[3], "protected-reserve-bytes") {
        Ok(value) => value,
        Err(error) => return fail(error),
    };

    let estimate = match estimate_hdc_ltc_memory(dimension, &layer_sizes) {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    let assessment = match assess_hdc_ltc_sram_budget(estimate, sram, reserve) {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    match serde_json::to_string_pretty(&assessment) {
        Ok(json) => println!("{json}"),
        Err(error) => return fail(format!("cannot serialize assessment: {error}")),
    }

    if assessment.fits_vector_payload_lower_bound {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "INFEASIBLE: vector payload lower bound exceeds available SRAM; this excludes temporaries, allocator/runtime, firmware and stacks"
        );
        ExitCode::from(2)
    }
}
