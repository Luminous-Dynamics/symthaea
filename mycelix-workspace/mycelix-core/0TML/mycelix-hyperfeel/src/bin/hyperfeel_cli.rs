use std::io::{self, Read};

use mycelix_hyperfeel::{HyperGradient, ModelUpdate};

fn main() {
    let mut input = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut input) {
        eprintln!("failed to read stdin: {err}");
        std::process::exit(1);
    }

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: hyperfeel_cli <encode|aggregate>");
        std::process::exit(1);
    }

    let cmd = args[1].as_str();
    match cmd {
        "encode" => encode(&input),
        "aggregate" => aggregate(&input),
        _ => {
            eprintln!("unknown command: {cmd}");
            std::process::exit(1);
        }
    }
}

fn encode(input: &str) {
    let update: ModelUpdate = match serde_json::from_str(input) {
        Ok(u) => u,
        Err(err) => {
            eprintln!("invalid JSON for ModelUpdate: {err}");
            std::process::exit(1);
        }
    };

    let hg = match HyperGradient::from_gradient(&update.gradient) {
        Ok(h) => h,
        Err(err) => {
            eprintln!("failed to encode gradient: {err}");
            std::process::exit(1);
        }
    };

    match serde_json::to_string(&hg) {
        Ok(json) => println!("{json}"),
        Err(err) => {
            eprintln!("failed to serialize HyperGradient: {err}");
            std::process::exit(1);
        }
    }
}

fn aggregate(input: &str) {
    let list: Vec<HyperGradient> = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("invalid JSON for HyperGradient list: {err}");
            std::process::exit(1);
        }
    };

    let aggregated = match HyperGradient::aggregate(&list) {
        Some(h) => h,
        None => {
            eprintln!("no hypergradients provided");
            std::process::exit(1);
        }
    };

    match serde_json::to_string(&aggregated) {
        Ok(json) => println!("{json}"),
        Err(err) => {
            eprintln!("failed to serialize aggregated HyperGradient: {err}");
            std::process::exit(1);
        }
    }
}

