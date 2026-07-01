use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod rhn_sweep;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    RhnSweep {
        #[arg(long, default_value = "1024")]
        dims: String,
        #[arg(long, default_value = "32")]
        objects: String,
        #[arg(long, default_value = "1")]
        seeds: String,
        #[arg(long, default_value = "8")]
        branching: String,
        #[arg(long, default_value = "100")]
        split_thresholds: String,
        #[arg(long, default_value = "2")]
        redundancy_ks: String,
        #[arg(long, default_value = "3")]
        fanouts: String,
        #[arg(long, default_value = "LeafOnly")]
        policies: String,
        #[arg(long, default_value = "reports/rhn_v011_sweep")]
        out: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::RhnSweep {
            dims,
            objects,
            seeds,
            branching,
            split_thresholds,
            redundancy_ks,
            fanouts,
            policies,
            out,
        } => {
            let dims = parse_list(&dims)?;
            let objects = parse_list(&objects)?;
            let seeds = parse_list(&seeds)?;
            let branching = parse_list(&branching)?;
            let split_thresholds = parse_list(&split_thresholds)?;
            let redundancy_ks = parse_list(&redundancy_ks)?;
            let fanouts = parse_list(&fanouts)?;
            let policies = policies.split(',').map(|s| s.to_string()).collect();

            rhn_sweep::run_sweep(
                dims,
                objects,
                seeds,
                branching,
                split_thresholds,
                redundancy_ks,
                fanouts,
                policies,
                out,
            )?;
        }
    }
    Ok(())
}

fn parse_list(s: &str) -> anyhow::Result<Vec<usize>> {
    s.split(',')
        .map(|item| item.parse::<usize>().map_err(anyhow::Error::from))
        .collect()
}
