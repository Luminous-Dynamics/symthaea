//! Emit a machine-readable inventory of Muse's declared instrument mappings
//! and the assets available on this machine.
//!
//! Example:
//!   cargo run -p symthaea-muse --bin muse_instrument_inventory -- \
//!     --vcsl-root data/samples/vcsl \
//!     --vsco2-root data/samples/vsco2-ce \
//!     --soundfont /path/to/soundfont.sf2
//!
//! Missing optional sample roots are recorded as unavailable evidence. The
//! path/size manifest digest does not hash WAV content; an optional SoundFont
//! file is content-hashed. A nonzero exit means the inventory could not be
//! produced faithfully (for example, an unreadable root or missing SoundFont).

use std::env;
use std::error::Error;
use std::path::PathBuf;

use symthaea_muse::vcsl::capability_inventory;

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
            "--help" | "-h" => {
                println!(
                    "Usage: muse_instrument_inventory [--vcsl-root PATH] \
                     [--vsco2-root PATH] [--soundfont FILE]\n\
                     Emits deterministic JSON to stdout. Missing sample roots \
                     are reported as unavailable."
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let inventory = capability_inventory(
        &vcsl_root,
        &vsco2_root,
        soundfont_path.as_deref(),
    )?;
    println!("{}", serde_json::to_string_pretty(&inventory)?);
    Ok(())
}
