// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generate a visual review matrix using the exact Spore boot renderer.
//!
//! This is an operator/development tool only. It creates representative factual
//! state receipts, composes their real BootGenomes, and renders each case through
//! the same `EcologyRenderer` path used by live DRM/KMS output.

use std::fs;
use std::path::PathBuf;

use symthaea_boot_ecology::{
    BootEcologyComposer, BootStateReceipt, GenerationHealth, GenerationTransition,
    MorphologyLineage, PreviousTermination, StorageState,
};
use symthaea_quicken_fb::preview::render_preview;

#[derive(Debug, Clone)]
struct Args {
    out: PathBuf,
    width: u32,
    height: u32,
    fps: u16,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("spore-boot-preview-matrix: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out).map_err(|e| format!("create {}: {e}", args.out.display()))?;

    let cases = preview_cases();
    let mut manifest_cases = Vec::with_capacity(cases.len());

    for (name, receipt, lineage) in cases {
        let case_dir = args.out.join(name);
        fs::create_dir_all(&case_dir)
            .map_err(|e| format!("create {}: {e}", case_dir.display()))?;
        fs::write(
            case_dir.join("boot-state.json"),
            serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        fs::write(
            case_dir.join("lineage.json"),
            serde_json::to_vec_pretty(&lineage).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        let genome = BootEcologyComposer::compose(&receipt, &lineage);
        let summary = render_preview(
            genome.clone(),
            case_dir.join("frames"),
            args.width,
            args.height,
            args.fps,
        )
        .map_err(|e| format!("render {name}: {e}"))?;

        manifest_cases.push(serde_json::json!({
            "name": name,
            "family": format!("{:?}", genome.family),
            "accent_family": genome.accent_family.map(|family| format!("{family:?}")),
            "cue": format!("{:?}", genome.cue),
            "seed": genome.seed_hex(),
            "duration_ms": summary.duration_ms,
            "frame_count": summary.frame_count,
            "receipt": format!("{name}/boot-state.json"),
            "lineage": format!("{name}/lineage.json"),
            "frames": format!("{name}/frames"),
        }));

        eprintln!(
            "{name:24} family={:?} cue={:?} frames={}",
            genome.family, genome.cue, summary.frame_count
        );
    }

    let manifest = serde_json::json!({
        "schema": "spore-boot-preview-matrix-v1",
        "renderer": "symthaea-quicken-fb/ecology_renderer",
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "cases": manifest_cases,
    });
    fs::write(
        args.out.join("matrix-manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    println!("{}", args.out.display());
    Ok(())
}

fn preview_cases() -> Vec<(&'static str, BootStateReceipt, MorphologyLineage)> {
    let seed = [0x5Au8; 32];
    let mut cases = Vec::new();

    cases.push((
        "01-first-germination",
        BootStateReceipt::first_boot(seed),
        MorphologyLineage::default(),
    ));

    // Several healthy boots deliberately vary only the boot counter. This
    // demonstrates that one machine keeps a recognizable identity while its
    // morphology can select different healthy visual grammars over time.
    for (index, counter) in [12u64, 13, 14, 15].into_iter().enumerate() {
        let mut receipt = healthy_receipt(seed, counter);
        receipt.previous_uptime_secs = 6 * 60 * 60 + counter * 97;
        let mut lineage = mature_lineage();
        lineage.successful_boots = 80 + counter;
        cases.push((
            match index {
                0 => "02-healthy-return-a",
                1 => "03-healthy-return-b",
                2 => "04-healthy-return-c",
                _ => "05-healthy-return-d",
            },
            receipt,
            lineage,
        ));
    }

    let mut long_uptime = healthy_receipt(seed, 88);
    long_uptime.previous_uptime_secs = 21 * 86_400;
    let mut old_lineage = mature_lineage();
    old_lineage.successful_boots = 630;
    old_lineage.maturity = 0.82;
    cases.push(("06-mature-long-uptime", long_uptime, old_lineage));

    let mut update = healthy_receipt(seed, 89);
    update.generation_transition = GenerationTransition::Updated {
        from: "generation-328".into(),
        to: "generation-329".into(),
    };
    update.generation_health = GenerationHealth::Unknown;
    cases.push(("07-generation-growth-ring", update, mature_lineage()));

    let mut rollback = healthy_receipt(seed, 90);
    rollback.generation_transition = GenerationTransition::RolledBack {
        attempted: "generation-330".into(),
        restored: "generation-329".into(),
    };
    rollback.generation_health = GenerationHealth::KnownGood;
    let mut rollback_lineage = mature_lineage();
    rollback_lineage.last_known_good_generation = Some("generation-329".into());
    cases.push(("08-rollback-restoration", rollback, rollback_lineage));

    let mut interrupted = healthy_receipt(seed, 91);
    interrupted.previous_termination = PreviousTermination::PowerLoss;
    interrupted.generation_health = GenerationHealth::Recovery;
    interrupted.storage_state = StorageState::JournalReplayed;
    cases.push(("09-interrupted-kintsugi", interrupted, mature_lineage()));

    let mut repaired = healthy_receipt(seed, 92);
    repaired.storage_state = StorageState::Repaired { repairs: 2 };
    repaired.generation_health = GenerationHealth::Recovery;
    cases.push(("10-storage-repair", repaired, mature_lineage()));

    let mut suspend = healthy_receipt(seed, 93);
    suspend.previous_termination = PreviousTermination::Suspend;
    suspend.previous_uptime_secs = 11 * 60 * 60;
    cases.push(("11-suspend-relight", suspend, mature_lineage()));

    let mut hibernate = healthy_receipt(seed, 94);
    hibernate.previous_termination = PreviousTermination::Hibernate;
    hibernate.previous_uptime_secs = 3 * 86_400;
    cases.push(("12-hibernate-crystal-thaw", hibernate, mature_lineage()));

    let mut hardware = healthy_receipt(seed, 95);
    hardware.previous_hardware_fingerprint = Some("hardware-a".into());
    hardware.hardware_fingerprint = Some("hardware-b".into());
    cases.push(("13-hardware-budding", hardware, mature_lineage()));

    let mut mesh = healthy_receipt(seed, 96);
    mesh.mesh_enabled = true;
    mesh.mesh_peers_last_seen = 9;
    cases.push(("14-mesh-return", mesh, mature_lineage()));

    let mut thermal = healthy_receipt(seed, 97);
    thermal.previous_termination = PreviousTermination::ThermalEmergency;
    thermal.thermal_events = 1;
    thermal.generation_health = GenerationHealth::Recovery;
    cases.push(("15-thermal-recovery", thermal, mature_lineage()));

    let mut memory_pressure = healthy_receipt(seed, 98);
    memory_pressure.oom_events = 3;
    cases.push(("16-memory-pressure", memory_pressure, mature_lineage()));

    cases
}

fn healthy_receipt(seed: [u8; 32], boot_counter: u64) -> BootStateReceipt {
    BootStateReceipt {
        schema_version: symthaea_boot_ecology::BOOT_ECOLOGY_SCHEMA_VERSION,
        machine_visual_seed: seed,
        boot_counter,
        previous_termination: PreviousTermination::CleanReboot,
        previous_uptime_secs: 8 * 60 * 60,
        generation_transition: GenerationTransition::Same,
        generation_health: GenerationHealth::KnownGood,
        storage_state: StorageState::Clean,
        oom_events: 0,
        thermal_events: 0,
        hardware_fingerprint: Some("hardware-stable".into()),
        previous_hardware_fingerprint: Some("hardware-stable".into()),
        mesh_enabled: false,
        mesh_peers_last_seen: 0,
    }
}

fn mature_lineage() -> MorphologyLineage {
    MorphologyLineage {
        schema_version: symthaea_boot_ecology::BOOT_ECOLOGY_SCHEMA_VERSION,
        successful_boots: 80,
        recovery_marks: 2,
        maturity: 0.48,
        last_genome_seed: Some([0x33; 32]),
        last_known_good_generation: Some("generation-329".into()),
    }
}

fn parse_args() -> Result<Args, String> {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    if raw.iter().any(|arg| matches!(arg.as_str(), "-h" | "--help")) {
        print_usage();
        std::process::exit(0);
    }

    let mut out = PathBuf::from("spore-boot-preview-matrix");
    let mut width = 480u32;
    let mut height = 270u32;
    // Low default rate keeps the full matrix review-sized. Individual cases can
    // still be rendered at 30fps through `quicken-fb preview` after selection.
    let mut fps = 2u16;

    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--out" => {
                i += 1;
                out = PathBuf::from(raw.get(i).ok_or("--out requires a value")?);
            }
            "--width" => {
                i += 1;
                width = parse_number(raw.get(i), "--width")?;
            }
            "--height" => {
                i += 1;
                height = parse_number(raw.get(i), "--height")?;
            }
            "--fps" => {
                i += 1;
                fps = parse_number(raw.get(i), "--fps")?;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
        i += 1;
    }

    if width == 0 || height == 0 {
        return Err("preview dimensions must be non-zero".into());
    }
    if fps == 0 || fps > 120 {
        return Err("--fps must be between 1 and 120".into());
    }

    Ok(Args {
        out,
        width,
        height,
        fps,
    })
}

fn parse_number<T>(value: Option<&String>, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse::<T>()
        .map_err(|e| format!("invalid {flag}: {e}"))
}

fn print_usage() {
    eprintln!(
        "Generate exact Spore boot visual matrix\n\n\
         Usage:\n\
           cargo run -p symthaea-quicken-fb --bin spore_boot_preview_matrix -- \\\n             [--out DIR] [--width 480] [--height 270] [--fps 2]\n\n\
         The low default FPS is for visual review. Re-render selected cases with\n\
         quicken-fb preview at 30fps for motion review."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_covers_distinct_lifecycle_states() {
        let cases = preview_cases();
        assert!(cases.len() >= 16);
        assert!(cases.iter().any(|(_, receipt, _)| {
            matches!(receipt.previous_termination, PreviousTermination::Suspend)
        }));
        assert!(cases.iter().any(|(_, receipt, _)| {
            matches!(
                receipt.generation_transition,
                GenerationTransition::RolledBack { .. }
            )
        }));
        assert!(cases.iter().any(|(_, receipt, _)| receipt.hardware_changed()));
        assert!(cases.iter().any(|(_, receipt, _)| receipt.mesh_enabled));
    }

    #[test]
    fn matrix_genomes_are_not_all_the_same_family() {
        let mut families = std::collections::BTreeSet::new();
        for (_, receipt, lineage) in preview_cases() {
            families.insert(format!(
                "{:?}",
                BootEcologyComposer::compose(&receipt, &lineage).family
            ));
        }
        assert!(families.len() >= 6, "only got {families:?}");
    }
}
