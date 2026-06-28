use std::env;
use std::fs;
use std::path::Path;

use symthaea_visual_compression_probe::{Result, VisualMemoryPacket};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let output_dir = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/tmp/svcp-corpus");
    fs::create_dir_all(output_dir)?;

    let fixtures = [
        (
            "tiny_pump_scan",
            include_bytes!("../fixtures/tiny_pump_scan.pgm").as_slice(),
        ),
        (
            "tiny_pump_scan_after",
            include_bytes!("../fixtures/tiny_pump_scan_after.pgm").as_slice(),
        ),
        (
            "tiny_crack_scan",
            include_bytes!("../fixtures/tiny_crack_scan.pgm").as_slice(),
        ),
    ];

    for (name, bytes) in fixtures {
        let image = symthaea_visual_compression_probe::parse_pgm(bytes)?;
        let packet = VisualMemoryPacket::encode(&image, 8, 10)?;
        let path = Path::new(output_dir).join(format!("{name}.svmp"));
        packet.write_text(&path)?;
        println!("wrote {}", path.display());
    }

    println!("query example:");
    println!(
        "  cargo run -p symthaea-visual-compression-probe --bin svcp -- query {}/tiny_pump_scan.svmp {} --top 3",
        output_dir, output_dir
    );
    Ok(())
}
