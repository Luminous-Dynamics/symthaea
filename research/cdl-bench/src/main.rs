use cdl_bench::{
    compress, crc32, decompress, generate_e0, predictor_state_bytes, E0Kind, ModelKind,
};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

const FRAME_HEADER_BYTES: usize = 32;
const DEFAULT_E0_BYTES: usize = 64 * 1024;
const DEFAULT_MAX_DECODE_BYTES: usize = 512 * 1024 * 1024;

fn usage() -> &'static str {
    "Usage:
  cdl-bench encode [--model order0|order1] INPUT OUTPUT
  cdl-bench decode [--max-bytes N] INPUT OUTPUT
  cdl-bench e0 [--bytes N] [--seed N]

The E0 command checks round-trip integrity and compares compressed lengths with
installed zstd, brotli, xz, and gzip commands when available."
}

fn parse_usize(value: Option<&String>, flag: &str) -> Result<usize, Box<dyn Error>> {
    value
        .ok_or_else(|| format!("missing value after {flag}"))?
        .parse::<usize>()
        .map_err(|e| format!("invalid value for {flag}: {e}").into())
}

fn parse_u64(value: Option<&String>, flag: &str) -> Result<u64, Box<dyn Error>> {
    value
        .ok_or_else(|| format!("missing value after {flag}"))?
        .parse::<u64>()
        .map_err(|e| format!("invalid value for {flag}: {e}").into())
}

fn encode_command(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut index = 1;
    let mut model = ModelKind::Order0;
    if args.get(index).map(String::as_str) == Some("--model") {
        model = ModelKind::parse(
            args.get(index + 1)
                .ok_or("missing model after --model")?,
        )
        .ok_or("model must be order0 or order1")?;
        index += 2;
    }
    if args.len() != index + 2 {
        return Err(usage().into());
    }

    let input_path = &args[index];
    let output_path = &args[index + 1];
    let input = fs::read(input_path)?;
    let started = Instant::now();
    let frame = compress(&input, model)?;
    let elapsed = started.elapsed();
    fs::write(output_path, &frame)?;

    let executable_bytes = std::env::current_exe()
        .ok()
        .and_then(|path| fs::metadata(path).ok())
        .map(|meta| meta.len());

    println!("codec=cdl-bench-v1");
    println!("model={}", if model == ModelKind::Order0 { "order0" } else { "order1" });
    println!("source_bytes={}", input.len());
    println!("payload_bytes={}", frame.len().saturating_sub(FRAME_HEADER_BYTES));
    println!("frame_header_bytes={FRAME_HEADER_BYTES}");
    println!("frame_bytes={}", frame.len());
    println!("predictor_state_bytes={}", predictor_state_bytes(model));
    match executable_bytes {
        Some(bytes) => println!("runner_binary_bytes={bytes}"),
        None => println!("runner_binary_bytes=unavailable"),
    }
    println!("source_crc32={:08x}", crc32(&input));
    println!("elapsed_ms={}", elapsed.as_millis());
    if input.is_empty() {
        println!("frame_to_source_ratio=n/a");
    } else {
        println!(
            "frame_to_source_ratio={:.6}",
            frame.len() as f64 / input.len() as f64
        );
    }
    println!("note=runner binary size excludes shared libraries and OS runtime");
    Ok(())
}

fn decode_command(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut index = 1;
    let mut max_bytes = DEFAULT_MAX_DECODE_BYTES;
    if args.get(index).map(String::as_str) == Some("--max-bytes") {
        max_bytes = parse_usize(args.get(index + 1), "--max-bytes")?;
        index += 2;
    }
    if args.len() != index + 2 {
        return Err(usage().into());
    }

    let frame = fs::read(&args[index])?;
    let started = Instant::now();
    let decoded = decompress(&frame, max_bytes)?;
    let elapsed = started.elapsed();
    fs::write(&args[index + 1], &decoded)?;
    println!("decoded_bytes={}", decoded.len());
    println!("source_crc32={:08x}", crc32(&decoded));
    println!("elapsed_ms={}", elapsed.as_millis());
    println!("round_trip_output_written=true");
    Ok(())
}

fn external_version(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines().next().map(str::trim).filter(|line| !line.is_empty()).map(str::to_owned)
}

fn external_compressed_len(command: &str, args: &[&str], input: &[u8]) -> Option<usize> {
    let mut child = Command::new(command)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    {
        let mut stdin = child.stdin.take()?;
        stdin.write_all(input).ok()?;
    }

    let output = child.wait_with_output().ok()?;
    if output.status.success() {
        Some(output.stdout.len())
    } else {
        None
    }
}

fn report_external_versions() {
    let tools: [(&str, &[&str]); 4] = [
        ("zstd", &["--version"]),
        ("brotli", &["--version"]),
        ("xz", &["--version"]),
        ("gzip", &["--version"]),
    ];
    for (name, args) in tools {
        match external_version(name, args) {
            Some(version) => println!("# external_{name}={version}"),
            None => println!("# external_{name}=not-installed-or-unavailable"),
        }
    }
}

fn e0_command(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut bytes = DEFAULT_E0_BYTES;
    let mut seed = 0xCD1_B_EA_5Eu64;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--bytes" => {
                bytes = parse_usize(args.get(index + 1), "--bytes")?;
                index += 2;
            }
            "--seed" => {
                seed = parse_u64(args.get(index + 1), "--seed")?;
                index += 2;
            }
            other => return Err(format!("unknown E0 option: {other}\n{}", usage()).into()),
        }
    }
    if bytes > 64 * 1024 * 1024 {
        return Err("E0 input limit is 64 MiB per generated stream".into());
    }

    println!("# benchmark=CDL-Bench-E0-v1");
    println!("# bytes={bytes}");
    println!("# seed={seed}");
    report_external_versions();
    println!("source,raw_bytes,cdl_order0_frame,cdl_order1_frame,zstd,brotli,xz,gzip,round_trip");

    for kind in E0Kind::ALL {
        let source = generate_e0(kind, bytes, seed);
        let frame0 = compress(&source, ModelKind::Order0)?;
        let frame1 = compress(&source, ModelKind::Order1)?;
        let decoded0 = decompress(&frame0, bytes)?;
        let decoded1 = decompress(&frame1, bytes)?;
        if decoded0 != source || decoded1 != source {
            return Err(format!("round-trip failure for {}", kind.name()).into());
        }

        let zstd = external_compressed_len("zstd", &["-q", "-c"], &source);
        let brotli = external_compressed_len("brotli", &["-q", "-c"], &source);
        let xz = external_compressed_len("xz", &["-c"], &source);
        let gzip = external_compressed_len("gzip", &["-c"], &source);

        let display = |value: Option<usize>| value.map_or_else(|| "NA".to_owned(), |v| v.to_string());
        println!(
            "{},{},{},{},{},{},{},{},exact",
            kind.name(),
            source.len(),
            frame0.len(),
            frame1.len(),
            display(zstd),
            display(brotli),
            display(xz),
            display(gzip),
        );
    }
    println!("# note=CDL frame sizes include the 32-byte header; external sizes include each tool's native framing");
    println!("# note=model state and executable sizes are not added to the per-stream frame column; see README for cold-start accounting");
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("encode") => encode_command(&args),
        Some("decode") => decode_command(&args),
        Some("e0") => e0_command(&args),
        _ => Err(usage().into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
