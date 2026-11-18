use cdl_bench::{
    compress, compress_bounded_i16, compress_i16, crc32, decompress, decompress_bounded_i16,
    decompress_i16, generate_e0, generate_fsm, generate_scm, predictor_state_bytes,
    BoundedPredictor, E0Kind, ModelKind, NumericTransform, ScmIntervention,
};
use std::error::Error;
use std::fs;
use std::io::{Read, Write};
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
  cdl-bench causal [--steps N] [--seed N]
  cdl-bench numeric [--values N] [--seed N]
  cdl-bench bounded [--values N] [--seed N] [--tolerance N]

The E0, causal and numeric commands check exact round-trip integrity.
The bounded command uses a separate per-sample error guarantee."
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

    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut output = Vec::new();
        stdout.read_to_end(&mut output).map(|_| output)
    });

    let write_result = {
        let mut stdin = child.stdin.take()?;
        stdin.write_all(input)
    };
    let status = child.wait().ok()?;
    let output = reader.join().ok()?.ok()?;
    if write_result.is_ok() && status.success() {
        Some(output.len())
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


fn causal_command(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut steps = 10_000usize;
    let mut seed = 0xCA55_A1u64;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--steps" => {
                steps = parse_usize(args.get(index + 1), "--steps")?;
                index += 2;
            }
            "--seed" => {
                seed = parse_u64(args.get(index + 1), "--seed")?;
                index += 2;
            }
            other => return Err(format!("unknown causal option: {other}\n{}", usage()).into()),
        }
    }
    if steps < 4 {
        return Err("causal generator requires at least four steps for its intervention window".into());
    }
    if steps > 1_000_000 {
        return Err("causal generator limit is 1,000,000 steps per trace".into());
    }

    println!("# benchmark=CDL-Bench-E1-E2-v1");
    println!("# steps={steps}");
    println!("# seed={seed}");
    println!("source,raw_bytes,cdl_order0_frame,cdl_order1_frame,zstd,round_trip");

    let fsm = generate_fsm(steps, 8, 4, seed)?;
    let scm_observational = generate_scm(steps, seed, None)?;
    let scm_interventional = generate_scm(
        steps,
        seed,
        Some(ScmIntervention {
            variable: 1,
            start_step: steps / 4,
            end_step: (steps * 3) / 4,
            value: 20_000,
        }),
    )?;

    let (e3_training, e3_held_out) = e3_mechanism_splits();
    let mut sources: Vec<(String, Vec<u8>)> = vec![
        ("fsm-observations".to_owned(), fsm.observations.clone()),
        ("scm-observational".to_owned(), scm_observational.stream.clone()),
        ("scm-do-x1".to_owned(), scm_interventional.stream.clone()),
    ];
    for (index, config) in e3_training.iter().enumerate() {
        let trace = generate_scm_with_config(steps, seed + index as u64 + 100, *config, None)?;
        sources.push((format!("e3-train-{index}"), trace.stream));
    }
    for (index, config) in e3_held_out.iter().enumerate() {
        let trace = generate_scm_with_config(steps, seed + index as u64 + 200, *config, None)?;
        sources.push((format!("e3-heldout-{index}"), trace.stream));
    }

    for (name, source) in &sources {
        let source = source.as_slice();
        let frame0 = compress(source, ModelKind::Order0)?;
        let frame1 = compress(source, ModelKind::Order1)?;
        if decompress(&frame0, source.len())? != source
            || decompress(&frame1, source.len())? != source
        {
            return Err(format!("round-trip failure for {name}").into());
        }
        let zstd = external_compressed_len("zstd", &["-q", "-c"], source)
            .map_or_else(|| "NA".to_owned(), |value| value.to_string());
        println!(
            "{name},{},{},{},{zstd},exact",
            source.len(),
            frame0.len(),
            frame1.len()
        );
    }

    // The latent states are evaluator-only ground truth; print only counts here.
    println!("# fsm_latent_state_count={}", fsm.state_count);
    println!("# fsm_observation_alphabet_size={}", fsm.alphabet_size);
    println!("# scm_variables=5");
    println!("# scm_intervention=do(x1=20000) on [steps/4,3*steps/4)");
    println!("# e3_training_configs={}", e3_training.len());
    println!("# e3_heldout_combinations={}", e3_held_out.len());
    println!("# note=E3 held-out tuples reuse mechanism component values but combine them in unseen configurations");
    println!("# note=these are deterministic generator and codec smoke measurements, not predictive transfer results");
    Ok(())
}


fn numeric_command(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut count = 10_000usize;
    let mut seed = 0x16_16u64;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--values" => {
                count = parse_usize(args.get(index + 1), "--values")?;
                index += 2;
            }
            "--seed" => {
                seed = parse_u64(args.get(index + 1), "--seed")?;
                index += 2;
            }
            other => return Err(format!("unknown numeric option: {other}\n{}", usage()).into()),
        }
    }
    if count > 1_000_000 {
        return Err("numeric generator limit is 1,000,000 values per sequence".into());
    }

    let random_bytes = generate_e0(E0Kind::Incompressible, count.saturating_mul(2), seed);
    let random_values: Vec<i16> = random_bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect();

    let constant = vec![512i16; count];
    let ramp: Vec<i16> = (0..count)
        .map(|i| ((i * 13) % 60_000) as i32 - 30_000)
        .map(|value| value as i16)
        .collect();
    let mut walk = Vec::with_capacity(count);
    let mut current = 0i32;
    for byte in random_bytes.iter().take(count) {
        current = (current + *byte as i32 - 128).clamp(i16::MIN as i32, i16::MAX as i32);
        walk.push(current as i16);
    }

    let sources = [
        ("constant", constant),
        ("ramp", ramp),
        ("random-walk", walk),
        ("uniform-random", random_values),
    ];

    println!("# benchmark=CDL-Bench-numeric-i16-v1");
    println!("# values={count}");
    println!("# seed={seed}");
    println!("source,transform,model,raw_bytes,frame_bytes,zstd_raw_bytes,ratio,round_trip");

    for (name, values) in sources {
        let mut raw = Vec::with_capacity(values.len() * 2);
        for value in &values {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        let zstd = external_compressed_len("zstd", &["-q", "-c"], &raw)
            .map_or_else(|| "NA".to_owned(), |value| value.to_string());

        for transform in [
            NumericTransform::Raw,
            NumericTransform::Delta,
            NumericTransform::DeltaOfDelta,
            NumericTransform::XorPrevious,
        ] {
            for model in [ModelKind::Order0, ModelKind::Order1] {
                let frame = compress_i16(&values, transform, model)?;
                let decoded = decompress_i16(&frame, values.len())?;
                if decoded != values {
                    return Err(format!("numeric round-trip failure for {name}/{transform:?}/{model:?}").into());
                }
                let ratio = if raw.is_empty() {
                    "n/a".to_owned()
                } else {
                    format!("{:.6}", frame.len() as f64 / raw.len() as f64)
                };
                let model_name = if model == ModelKind::Order0 { "order0" } else { "order1" };
                println!(
                    "{name},{:?},{model_name},{},{},{zstd},{ratio},exact",
                    transform,
                    raw.len(),
                    frame.len()
                );
            }
        }
    }
    println!("# note=frame lengths include both the 32-byte numeric wrapper and inner CDL frame");
    println!("# note=Zstandard encodes the raw little-endian i16 bytes; Pco/Sprintz/TSCom-Bench/SEP are external follow-up baselines");
    Ok(())
}


fn bounded_command(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut count = 10_000usize;
    let mut seed = 0xB0_0D_16u64;
    let mut tolerance = 5u16;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--values" => {
                count = parse_usize(args.get(index + 1), "--values")?;
                index += 2;
            }
            "--seed" => {
                seed = parse_u64(args.get(index + 1), "--seed")?;
                index += 2;
            }
            "--tolerance" => {
                let parsed = parse_u64(args.get(index + 1), "--tolerance")?;
                tolerance = u16::try_from(parsed)
                    .map_err(|_| "tolerance must be in 0..=32767")?;
                index += 2;
            }
            other => return Err(format!("unknown bounded option: {other}\n{}", usage()).into()),
        }
    }
    if count > 1_000_000 {
        return Err("bounded generator limit is 1,000,000 values per sequence".into());
    }
    if tolerance > 32767 {
        return Err("tolerance must be in 0..=32767".into());
    }

    let noise = generate_e0(E0Kind::IidSkewed, count, seed);
    let constant = vec![8192i16; count];
    let ramp: Vec<i16> = (0..count)
        .map(|i| (i as i32 / 8 - 16_000).clamp(i16::MIN as i32, i16::MAX as i32) as i16)
        .collect();
    let mut random_walk = Vec::with_capacity(count);
    let mut walk = 0i32;
    for byte in &noise {
        walk = (walk + *byte as i32 - 128).clamp(i16::MIN as i32, i16::MAX as i32);
        random_walk.push(walk as i16);
    }
    let demand_like: Vec<i16> = (0..count)
        .map(|i| {
            let slow = ((i / 240) % 80) as i32 * 7;
            let daily = ((i % 1440) / 24) as i32 * 2;
            let pulse = if i % 997 < 12 { 900 } else { 0 };
            (12_000 + slow + daily + pulse + (noise[i] as i32 - 128) / 2)
                .clamp(i16::MIN as i32, i16::MAX as i32) as i16
        })
        .collect();

    let sources = [
        ("constant", constant),
        ("ramp", ramp),
        ("random-walk", random_walk),
        ("demand-like-synthetic", demand_like),
    ];

    println!("# benchmark=CDL-Bench-bounded-i16-v1");
    println!("# values={count}");
    println!("# seed={seed}");
    println!("# tolerance={tolerance}");
    println!("source,predictor,model,raw_bytes,frame_bytes,zstd_raw_bytes,ratio,max_abs_error,guarantee");

    for (name, values) in sources {
        let mut raw = Vec::with_capacity(values.len() * 2);
        for value in &values {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        let zstd = external_compressed_len("zstd", &["-q", "-c"], &raw)
            .map_or_else(|| "NA".to_owned(), |value| value.to_string());

        for predictor in [BoundedPredictor::Previous, BoundedPredictor::Linear] {
            for model in [ModelKind::Order0, ModelKind::Order1] {
                let frame = compress_bounded_i16(&values, tolerance, predictor, model)?;
                let reconstructed = decompress_bounded_i16(&frame, values.len())?;
                let max_error = values
                    .iter()
                    .zip(&reconstructed)
                    .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
                    .max()
                    .unwrap_or(0);
                if max_error > tolerance as u32 {
                    return Err(format!("bounded-error guarantee failed for {name}").into());
                }
                let ratio = if raw.is_empty() {
                    "n/a".to_owned()
                } else {
                    format!("{:.6}", frame.len() as f64 / raw.len() as f64)
                };
                let predictor_name = match predictor {
                    BoundedPredictor::Previous => "previous",
                    BoundedPredictor::Linear => "linear",
                };
                let model_name = if model == ModelKind::Order0 { "order0" } else { "order1" };
                println!(
                    "{name},{predictor_name},{model_name},{},{},{zstd},{ratio},{max_error},PASS",
                    raw.len(),
                    frame.len()
                );
            }
        }
    }
    println!("# note=bounded-error frames are lossy and are not comparable to the exact-lossless leaderboard");
    println!("# note=the tolerance applies to every sample; frame includes predictor, model, tolerance and reconstructed CRC");
    println!("# note=synthetic demand-like sequence is not a real electricity or ridership dataset");
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("encode") => encode_command(&args),
        Some("decode") => decode_command(&args),
        Some("e0") => e0_command(&args),
        Some("causal") => causal_command(&args),
        Some("numeric") => numeric_command(&args),
        Some("bounded") => bounded_command(&args),
        _ => Err(usage().into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
