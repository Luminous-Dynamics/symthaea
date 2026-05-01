// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::Result;
use clap::Parser;
use mycelix_bridge_codegen::{scan_zome_dir, generate_leptos_bridge};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the zome coordinator source directory
    #[arg(short, long)]
    src: PathBuf,

    /// Output path for the generated bridge file
    #[arg(short, long)]
    out: PathBuf,

    /// Role name (e.g. "finance", "civic")
    #[arg(short, long)]
    role: String,

    /// Zome name
    #[arg(short, long)]
    zome: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Scanning {} for zome functions...", args.src.display());
    let fns = scan_zome_dir(&args.src, &args.role, &args.zome)?;
    
    println!("Found {} functions. Generating bridge...", fns.len());
    let generated = generate_leptos_bridge(&fns);
    
    std::fs::create_dir_all(args.out.parent().unwrap())?;
    std::fs::write(&args.out, generated)?;
    
    println!("Successfully generated bridge at {}", args.out.display());
    Ok(())
}
