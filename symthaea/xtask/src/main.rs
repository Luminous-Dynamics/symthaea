use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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
            fs::create_dir_all(&out)?;
            println!("Sweep initiated to {:?}", out);
        }
    }
    Ok(())
}
