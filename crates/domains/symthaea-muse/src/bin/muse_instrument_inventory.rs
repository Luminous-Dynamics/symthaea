//! Emit a machine-readable inventory of Muse's declared instrument mappings
//! and the assets available on this machine.
//!
//! Example:
//!   cargo run -p symthaea-muse --bin muse_instrument_inventory -- \
//!     --vcsl-root data/samples/vcsl \
//!     --vsco2-root data/samples/vsco2-ce \
//!     --soundfont /path/to/soundfont.sf2 \
//!     --fluidsynth /path/to/fluidsynth
//!
//! Runtime render probes are opt-in and require --soundfont. Use
//! --hash-indexed-samples to content-fingerprint WAV files accepted by Muse's
//! filename indexer; this adds disk I/O. Missing sample roots are unavailable
//! evidence. The root path/size manifest remains lightweight and does not hash
//! WAV content. A nonzero exit means the inventory could not be produced
//! faithfully (for example, an unreadable root or missing SoundFont).

use std::env;
use std::error::Error;
use std::path::PathBuf;

use symthaea_muse::vcsl::capability_inventory_with_options;

fn main() {
    if let Err(error) = run() {
        eprintln!("muse_instrument_inventory: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let default_vcsl = env::var_os("SYMTHAEA_VCSL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../../data/samples/vcsl"));
    let default_vsco2 = env::var_os("SYMTHAEA_VSCO2_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_vcsl.join("../vsco2-ce"));

    let mut vcsl_root = default_vcsl;
    let mut vsco2_root = default_vsco2;
    let mut soundfont_path = None;
    let mut fluidsynth_path = None;
    let mut hash_indexed_samples = false;
    let mut args = env::args().skip(1);

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--vcsl-root" => {
                vcsl_root = PathBuf::from(
                    args.next()
                        .ok_or("--vcsl-root requires a directory path")?,
                );
            }
            "--vsco2-root" => {
                vsco2_root = PathBuf::from(
                    args.next()
                        .ok_or("--vsco2-root requires a directory path")?,
                );
            }
            "--soundfont" => {
                soundfont_path = Some(PathBuf::from(
                    args.next()
                        .ok_or("--soundfont requires a file path")?,
                ));
            }
            "--fluidsynth" => {
                fluidsynth_path = Some(PathBuf::from(
                    args.next()
                        .ok_or("--fluidsynth requires an executable path or command name")?,
                ));
            }
            "--hash-indexed-samples" => {
                hash_indexed_samples = true;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: muse_instrument_inventory [--vcsl-root PATH] \
                     [--vsco2-root PATH] [--soundfont FILE] [--fluidsynth PATH] [--hash-indexed-samples]\n\
                     Emits JSON to stdout. Missing sample roots are unavailable. \
                     --fluidsynth enables per-program offline render probes and \
                     requires --soundfont. --hash-indexed-samples hashes indexed WAV bytes and adds disk I/O."
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let inventory = capability_inventory_with_options(
        &vcsl_root,
        &vsco2_root,
        soundfont_path.as_deref(),
        fluidsynth_path.as_deref(),
        hash_indexed_samples,
    )?;
    println!("{}", serde_json::to_string_pretty(&inventory)?);
    Ok(())
}
