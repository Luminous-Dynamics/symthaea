use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use symthaea_visual_compression_probe::{
    EncodingParams, GrayImage, PROBE_EXPERIMENT_VERSION, ProbeError, Result, VisualMemoryPacket,
    benchmark_image, edge_energy, image_hash64, mse, packet_manifest_header, packet_manifest_row,
    packet_similarity, psnr, visual_summary,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_help();
        return Ok(());
    }
    match args[1].as_str() {
        "inspect" => inspect(&args[2..]),
        "encode" => encode(&args[2..]),
        "decode" => decode(&args[2..]),
        "compare" => compare(&args[2..]),
        "metrics" => metrics(&args[2..]),
        "fingerprint" => fingerprint(&args[2..]),
        "benchmark" => benchmark(&args[2..]),
        "query" => query(&args[2..]),
        "validate" => validate(&args[2..]),
        "diff" => diff(&args[2..]),
        "sweep" => sweep(&args[2..]),
        "index" => index(&args[2..]),
        "corpus-benchmark" => corpus_benchmark(&args[2..]),
        "summary" => summary(&args[2..]),
        "batch-encode" => batch_encode(&args[2..]),
        "matrix" => matrix(&args[2..]),
        "self-test" => self_test(&args[2..]),
        "pipeline" => pipeline(&args[2..]),
        "doctor" => doctor(&args[2..]),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(ProbeError::InvalidArgs(format!("unknown command: {other}"))),
    }
}

fn inspect(args: &[String]) -> Result<()> {
    if args.len() != 1 {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp inspect <input.pgm>".into(),
        ));
    }
    let image = GrayImage::read_pgm(&args[0])?;
    let packet = VisualMemoryPacket::encode(&image, 8, 10)?;
    println!("image: {}x{}", image.width, image.height);
    println!("image_hash64={:016x}", image_hash64(&image));
    println!("edge_energy={:.8}", edge_energy(&image));
    println!("blocks: {}", packet.blocks.len());
    println!("packet_hash64={:016x}", packet.stable_hash64());
    println!("topology samples: {}", packet.topology.len());
    for sample in packet.topology.iter().step_by(3) {
        println!(
            "  threshold={:.3} beta0={} beta1={}",
            sample.threshold, sample.beta0, sample.beta1
        );
    }
    Ok(())
}

fn encode(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp encode <input.pgm> <output.svmp> [--block N] [--keep K]".into(),
        ));
    }
    let input = &args[0];
    let output = &args[1];
    let block = flag_usize(args, "--block", 8)?;
    let keep = flag_usize(args, "--keep", 10)?;
    let image = GrayImage::read_pgm(input)?;
    let packet = VisualMemoryPacket::encode(&image, block, keep)?;
    packet.validate()?;
    packet.write_text(output)?;
    println!("encoded {}x{} image", image.width, image.height);
    println!("block_size={block} keep_coeffs={keep}");
    println!(
        "blocks={} topology_samples={}",
        packet.blocks.len(),
        packet.topology.len()
    );
    println!("image_hash64={:016x}", image_hash64(&image));
    println!("packet_hash64={:016x}", packet.stable_hash64());
    Ok(())
}

fn decode(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp decode <input.svmp> <output.pgm>".into(),
        ));
    }
    let packet = VisualMemoryPacket::read_text(&args[0])?;
    packet.validate()?;
    let image = packet.decode()?;
    image.write_pgm(&args[1])?;
    println!("decoded {}x{} image", image.width, image.height);
    println!("packet_hash64={:016x}", packet.stable_hash64());
    println!("decoded_image_hash64={:016x}", image_hash64(&image));
    Ok(())
}

fn compare(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp compare <a.pgm> <b.pgm>".into(),
        ));
    }
    let a = GrayImage::read_pgm(&args[0])?;
    let b = GrayImage::read_pgm(&args[1])?;
    println!("mse={:.8}", mse(&a, &b)?);
    println!("psnr={:.3} dB", psnr(&a, &b)?);
    println!("a_hash64={:016x}", image_hash64(&a));
    println!("b_hash64={:016x}", image_hash64(&b));
    println!("a_edge_energy={:.8}", edge_energy(&a));
    println!("b_edge_energy={:.8}", edge_energy(&b));
    let pa = VisualMemoryPacket::encode(&a, 8, 10)?;
    let pb = VisualMemoryPacket::encode(&b, 8, 10)?;
    println!("{}", packet_similarity(&pa, &pb).to_pretty_text());
    Ok(())
}

fn metrics(args: &[String]) -> Result<()> {
    if args.is_empty() {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp metrics <input.svmp> [--json]".into(),
        ));
    }
    let packet = VisualMemoryPacket::read_text(&args[0])?;
    packet.validate()?;
    let metrics = packet.metrics();
    if has_flag(args, "--json") {
        println!(
            "{{\"packet_hash64\":\"{:016x}\",\"metrics\":{}}}",
            packet.stable_hash64(),
            metrics.to_json()
        );
    } else {
        println!("packet_hash64={:016x}", packet.stable_hash64());
        println!("{}", metrics.to_pretty_text());
    }
    Ok(())
}

fn fingerprint(args: &[String]) -> Result<()> {
    if args.is_empty() {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp fingerprint <input.pgm|input.svmp> [--json] [--block N] [--keep K]".into(),
        ));
    }
    let input = &args[0];
    let packet = if input.ends_with(".svmp") {
        let packet = VisualMemoryPacket::read_text(input)?;
        packet.validate()?;
        packet
    } else {
        let block = flag_usize(args, "--block", 8)?;
        let keep = flag_usize(args, "--keep", 10)?;
        let image = GrayImage::read_pgm(input)?;
        VisualMemoryPacket::encode(&image, block, keep)?
    };

    if has_flag(args, "--json") {
        print!(
            "{{\"packet_hash64\":\"{:016x}\",\"metrics\":{},\"topology\":[",
            packet.stable_hash64(),
            packet.metrics().to_json()
        );
        for (i, sample) in packet.topology.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!(
                "{{\"threshold\":{:.6},\"beta0\":{},\"beta1\":{}}}",
                sample.threshold, sample.beta0, sample.beta1
            );
        }
        println!("]}}");
    } else {
        println!("packet_hash64={:016x}", packet.stable_hash64());
        println!("{}", packet.metrics().to_pretty_text());
        println!("topology:");
        for sample in &packet.topology {
            println!(
                "  threshold={:.3} beta0={} beta1={}",
                sample.threshold, sample.beta0, sample.beta1
            );
        }
    }
    Ok(())
}

fn benchmark(args: &[String]) -> Result<()> {
    if args.is_empty() {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp benchmark <input.pgm> [--block N] [--keep K] [--json]".into(),
        ));
    }
    let block = flag_usize(args, "--block", 8)?;
    let keep = flag_usize(args, "--keep", 10)?;
    let image = GrayImage::read_pgm(&args[0])?;
    let report = benchmark_image(&image, block, keep)?;
    if has_flag(args, "--json") {
        println!(
            "{{\"experiment_version\":\"{}\",\"image_hash64\":\"{:016x}\",\"edge_energy\":{:.8},\"report\":{}}}",
            PROBE_EXPERIMENT_VERSION,
            image_hash64(&image),
            edge_energy(&image),
            report.to_json()
        );
    } else {
        println!("experiment_version={PROBE_EXPERIMENT_VERSION}");
        println!("image_hash64={:016x}", image_hash64(&image));
        println!("edge_energy={:.8}", edge_energy(&image));
        println!("{}", report.to_pretty_text());
    }
    Ok(())
}

fn query(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp query <query.svmp> <packet-dir> [--top N] [--json]".into(),
        ));
    }
    let query = VisualMemoryPacket::read_text(&args[0])?;
    query.validate()?;
    let dir = Path::new(&args[1]);
    let top = flag_usize(args, "--top", 5)?;
    let mut results = Vec::new();
    for path in svmp_paths(dir)? {
        let Ok(packet) = VisualMemoryPacket::read_text(&path) else {
            continue;
        };
        if packet.validate().is_err() {
            continue;
        }
        let similarity = packet_similarity(&query, &packet);
        results.push((path, packet.stable_hash64(), similarity));
    }
    results.sort_by(|a, b| {
        b.2.combined_similarity
            .partial_cmp(&a.2.combined_similarity)
            .unwrap_or(Ordering::Equal)
            .then_with(|| display_path(&a.0).cmp(&display_path(&b.0)))
    });
    results.truncate(top);

    if has_flag(args, "--json") {
        print!(
            "{{\"query_hash64\":\"{:016x}\",\"results\":[",
            query.stable_hash64()
        );
        for (i, (path, hash, sim)) in results.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!(
                "{{\"path\":\"{}\",\"packet_hash64\":\"{:016x}\",\"similarity\":{}}}",
                json_escape(&display_path(path)),
                hash,
                sim.to_json()
            );
        }
        println!("]}}");
    } else {
        println!("query_hash64={:016x}", query.stable_hash64());
        for (path, hash, sim) in results {
            println!("{}", display_path(&path));
            println!("packet_hash64={hash:016x}");
            println!("{}", sim.to_pretty_text());
        }
    }
    Ok(())
}

fn validate(args: &[String]) -> Result<()> {
    if args.is_empty() {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp validate <input.svmp> [--json]".into(),
        ));
    }
    let packet = VisualMemoryPacket::read_text(&args[0])?;
    let result = packet.validate();
    if has_flag(args, "--json") {
        match result {
            Ok(()) => println!(
                "{{\"ok\":true,\"packet_hash64\":\"{:016x}\",\"metrics\":{}}}",
                packet.stable_hash64(),
                packet.metrics().to_json()
            ),
            Err(err) => println!(
                "{{\"ok\":false,\"error\":\"{}\"}}",
                json_escape(&err.to_string())
            ),
        }
    } else {
        result?;
        println!("ok");
        println!("packet_hash64={:016x}", packet.stable_hash64());
        println!("{}", packet.metrics().to_pretty_text());
    }
    Ok(())
}

fn diff(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp diff <a.svmp> <b.svmp> [--json]".into(),
        ));
    }
    let a = VisualMemoryPacket::read_text(&args[0])?;
    let b = VisualMemoryPacket::read_text(&args[1])?;
    a.validate()?;
    b.validate()?;
    let sim = packet_similarity(&a, &b);
    let ma = a.metrics();
    let mb = b.metrics();
    if has_flag(args, "--json") {
        println!(
            "{{\"a_hash64\":\"{:016x}\",\"b_hash64\":\"{:016x}\",\"similarity\":{},\"delta\":{{\"stored_coefficients\":{},\"prototype_text_bytes\":{},\"topology_samples\":{}}}}}",
            a.stable_hash64(),
            b.stable_hash64(),
            sim.to_json(),
            mb.stored_coefficients as isize - ma.stored_coefficients as isize,
            mb.prototype_text_bytes as isize - ma.prototype_text_bytes as isize,
            mb.topology_samples as isize - ma.topology_samples as isize,
        );
    } else {
        println!("a_hash64={:016x}", a.stable_hash64());
        println!("b_hash64={:016x}", b.stable_hash64());
        println!("{}", sim.to_pretty_text());
        println!(
            "delta_stored_coefficients={}",
            mb.stored_coefficients as isize - ma.stored_coefficients as isize
        );
        println!(
            "delta_prototype_text_bytes={}",
            mb.prototype_text_bytes as isize - ma.prototype_text_bytes as isize
        );
        println!(
            "delta_topology_samples={}",
            mb.topology_samples as isize - ma.topology_samples as isize
        );
    }
    Ok(())
}

fn sweep(args: &[String]) -> Result<()> {
    if args.is_empty() {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp sweep <input.pgm> [--blocks 4,8,16] [--keeps 2,4,8,12] [--csv|--json]"
                .into(),
        ));
    }
    let image = GrayImage::read_pgm(&args[0])?;
    let blocks = flag_list_usize(args, "--blocks", &[4, 8, 16])?;
    let keeps = flag_list_usize(args, "--keeps", &[2, 4, 8, 12])?;
    let mut rows = Vec::new();
    for block in blocks {
        for keep in &keeps {
            if block == 0 || *keep == 0 || *keep > block * block {
                continue;
            }
            let report = benchmark_image(&image, block, *keep)?;
            rows.push((block, *keep, report));
        }
    }
    if has_flag(args, "--json") {
        print!(
            "{{\"image_hash64\":\"{:016x}\",\"rows\":[",
            image_hash64(&image)
        );
        for (i, (block, keep, report)) in rows.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!(
                "{{\"block\":{},\"keep\":{},\"report\":{}}}",
                block,
                keep,
                report.to_json()
            );
        }
        println!("]}}");
    } else {
        println!(
            "block,keep,stored_coefficients,coefficient_density,prototype_text_bytes,text_to_raw_ratio,mse,psnr_db,combined_similarity"
        );
        for (block, keep, report) in rows {
            println!(
                "{},{},{},{:.8},{},{:.8},{:.8},{},{}",
                block,
                keep,
                report.metrics.stored_coefficients,
                report.metrics.coefficient_density,
                report.metrics.prototype_text_bytes,
                report.metrics.text_to_raw_ratio,
                report.mse,
                csv_float(report.psnr_db),
                csv_float(report.self_similarity.combined_similarity)
            );
        }
    }
    Ok(())
}

fn index(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp index <packet-dir> <output.tsv|-> [--json]".into(),
        ));
    }
    let dir = Path::new(&args[0]);
    let output = &args[1];
    let mut rows = Vec::new();
    for path in svmp_paths(dir)? {
        let packet = match VisualMemoryPacket::read_text(&path) {
            Ok(packet) => packet,
            Err(_) => continue,
        };
        if packet.validate().is_err() {
            continue;
        }
        rows.push((path, packet.stable_hash64(), packet.metrics()));
    }
    if has_flag(args, "--json") {
        let mut out = String::from("{\"entries\":[");
        for (i, (path, hash, metrics)) in rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"path\":\"{}\",\"packet_hash64\":\"{:016x}\",\"metrics\":{}}}",
                json_escape(&display_path(path)),
                hash,
                metrics.to_json()
            ));
        }
        out.push_str("]}\n");
        write_or_print(output, &out)?;
    } else {
        let mut out = String::from(
            "path\tpacket_hash64\twidth\theight\tblock_size\tkeep_coeffs\tblocks\tstored_coefficients\tcoefficient_density\tprototype_text_bytes\ttext_to_raw_ratio\n",
        );
        for (path, hash, metrics) in rows {
            out.push_str(&format!(
                "{}\t{:016x}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{:.8}\n",
                display_path(&path),
                hash,
                metrics.width,
                metrics.height,
                metrics.block_size,
                metrics.keep_coeffs,
                metrics.blocks,
                metrics.stored_coefficients,
                metrics.coefficient_density,
                metrics.prototype_text_bytes,
                metrics.text_to_raw_ratio
            ));
        }
        write_or_print(output, &out)?;
    }
    Ok(())
}

fn corpus_benchmark(args: &[String]) -> Result<()> {
    if args.is_empty() {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp corpus-benchmark <image-dir> [--block N] [--keep K] [--json]".into(),
        ));
    }
    let dir = Path::new(&args[0]);
    let block = flag_usize(args, "--block", 8)?;
    let keep = flag_usize(args, "--keep", 10)?;
    let paths = pgm_paths(dir)?;
    if has_flag(args, "--json") {
        print!("{{\"block\":{},\"keep\":{},\"images\":[", block, keep);
        for (i, path) in paths.iter().enumerate() {
            let image = GrayImage::read_pgm(path)?;
            let report = benchmark_image(&image, block, keep)?;
            if i > 0 {
                print!(",");
            }
            print!(
                "{{\"path\":\"{}\",\"image_hash64\":\"{:016x}\",\"edge_energy\":{:.8},\"report\":{}}}",
                json_escape(&display_path(path)),
                image_hash64(&image),
                edge_energy(&image),
                report.to_json()
            );
        }
        println!("]}}");
    } else {
        println!(
            "path,image_hash64,edge_energy,block,keep,stored_coefficients,text_to_raw_ratio,mse,psnr_db,combined_similarity"
        );
        for path in paths {
            let image = GrayImage::read_pgm(&path)?;
            let report = benchmark_image(&image, block, keep)?;
            println!(
                "{},{:016x},{:.8},{},{},{},{:.8},{:.8},{},{}",
                csv_escape(&display_path(&path)),
                image_hash64(&image),
                edge_energy(&image),
                block,
                keep,
                report.metrics.stored_coefficients,
                report.metrics.text_to_raw_ratio,
                report.mse,
                csv_float(report.psnr_db),
                csv_float(report.self_similarity.combined_similarity)
            );
        }
    }
    Ok(())
}

fn summary(args: &[String]) -> Result<()> {
    if args.is_empty() {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp summary <input.pgm> [--block N] [--keep K] [--topology-levels N] [--json]"
                .into(),
        ));
    }
    let block = flag_usize(args, "--block", 8)?;
    let keep = flag_usize(args, "--keep", 10)?;
    let topology_levels = flag_usize(args, "--topology-levels", 16)?;
    let params = EncodingParams::new(block, keep, topology_levels)?;
    let image = GrayImage::read_pgm(&args[0])?;
    let summary = visual_summary(&image, params)?;
    if has_flag(args, "--json") {
        println!("{}", summary.to_json());
    } else {
        println!("{}", summary.to_pretty_text());
    }
    Ok(())
}

fn batch_encode(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp batch-encode <image-dir> <packet-dir> [--block N] [--keep K] [--topology-levels N] [--manifest OUT] [--json]".into(),
        ));
    }
    let image_dir = Path::new(&args[0]);
    let packet_dir = Path::new(&args[1]);
    let block = flag_usize(args, "--block", 8)?;
    let keep = flag_usize(args, "--keep", 10)?;
    let topology_levels = flag_usize(args, "--topology-levels", 16)?;
    let manifest = flag_string(args, "--manifest");
    let params = EncodingParams::new(block, keep, topology_levels)?;
    fs::create_dir_all(packet_dir)?;
    let mut rows: Vec<(PathBuf, u64, usize)> = Vec::new();
    let mut manifest_text = String::new();
    manifest_text.push_str(packet_manifest_header());
    manifest_text.push('\n');
    for path in pgm_paths(image_dir)? {
        let image = GrayImage::read_pgm(&path)?;
        let packet = VisualMemoryPacket::encode_with_params(&image, params)?;
        packet.validate()?;
        let stem = path.file_stem().and_then(|x| x.to_str()).unwrap_or("image");
        let output = packet_dir.join(format!("{stem}.svmp"));
        packet.write_text(&output)?;
        manifest_text.push_str(&packet_manifest_row(&display_path(&output), &packet));
        manifest_text.push('\n');
        rows.push((output, packet.stable_hash64(), packet.blocks.len()));
    }
    if let Some(path) = manifest {
        write_or_print(&path, &manifest_text)?;
    }
    if has_flag(args, "--json") {
        print!("{{\"params\":{},\"packets\":[", params.to_json());
        for (i, (path, hash, blocks)) in rows.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!(
                "{{\"path\":\"{}\",\"packet_hash64\":\"{:016x}\",\"blocks\":{}}}",
                json_escape(&display_path(path)),
                hash,
                blocks
            );
        }
        println!("]}}");
    } else {
        println!("encoded_packets={}", rows.len());
        println!("{}", params.to_pretty_text());
        for (path, hash, blocks) in rows {
            println!("{}\t{:016x}\tblocks={}", display_path(&path), hash, blocks);
        }
    }
    Ok(())
}

fn matrix(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp matrix <packet-dir> <output.csv|-> [--json]".into(),
        ));
    }
    let dir = Path::new(&args[0]);
    let output = &args[1];
    let mut packets = Vec::new();
    for path in svmp_paths(dir)? {
        let packet = VisualMemoryPacket::read_text(&path)?;
        packet.validate()?;
        packets.push((path, packet));
    }
    if has_flag(args, "--json") {
        let mut out = String::from("{\"packets\":[");
        for (i, (path, packet)) in packets.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"path\":\"{}\",\"packet_hash64\":\"{:016x}\"}}",
                json_escape(&display_path(path)),
                packet.stable_hash64()
            ));
        }
        out.push_str("],\"pairs\":[");
        let mut first = true;
        for i in 0..packets.len() {
            for j in 0..packets.len() {
                if !first {
                    out.push(',');
                }
                first = false;
                let sim = packet_similarity(&packets[i].1, &packets[j].1);
                out.push_str(&format!(
                    "{{\"i\":{},\"j\":{},\"similarity\":{}}}",
                    i,
                    j,
                    sim.to_json()
                ));
            }
        }
        out.push_str("]}\n");
        write_or_print(output, &out)?;
    } else {
        let mut out = String::from("path");
        for (path, _) in &packets {
            out.push(',');
            out.push_str(&csv_escape(&display_path(path)));
        }
        out.push('\n');
        for (row_path, row_packet) in &packets {
            out.push_str(&csv_escape(&display_path(row_path)));
            for (_, col_packet) in &packets {
                let sim = packet_similarity(row_packet, col_packet);
                out.push_str(&format!(",{:.8}", sim.combined_similarity));
            }
            out.push('\n');
        }
        write_or_print(output, &out)?;
    }
    Ok(())
}

fn self_test(args: &[String]) -> Result<()> {
    let fixture_dir_buf;
    let fixture_dir: &Path = if args.is_empty() || args[0].starts_with("--") {
        fixture_dir_buf =
            PathBuf::from("crates/domains/symthaea-visual-compression-probe/fixtures");
        &fixture_dir_buf
    } else {
        Path::new(&args[0])
    };
    let block = flag_usize(args, "--block", 8)?;
    let keep = flag_usize(args, "--keep", 10)?;
    let topology_levels = flag_usize(args, "--topology-levels", 16)?;
    let params = EncodingParams::new(block, keep, topology_levels)?;
    let paths = pgm_paths(fixture_dir)?;
    if paths.is_empty() {
        return Err(ProbeError::InvalidArgs(format!(
            "no .pgm fixtures found in {}",
            fixture_dir.to_string_lossy()
        )));
    }

    let mut packets: Vec<(PathBuf, VisualMemoryPacket, f32)> = Vec::new();
    let mut min_psnr = f32::INFINITY;
    let mut max_recon_mse = 0.0f32;
    let mut min_self_similarity = f32::INFINITY;
    let mut total_edge_energy = 0.0f32;

    for path in &paths {
        let image = GrayImage::read_pgm(path)?;
        let packet = VisualMemoryPacket::encode_with_params(&image, params)?;
        packet.validate()?;
        let decoded = packet.decode()?;
        let recon_mse = mse(&image, &decoded)?;
        let recon_psnr = psnr(&image, &decoded)?;
        let sim = packet_similarity(&packet, &packet);
        min_psnr = min_psnr.min(recon_psnr);
        max_recon_mse = max_recon_mse.max(recon_mse);
        min_self_similarity = min_self_similarity.min(sim.combined_similarity);
        total_edge_energy += edge_energy(&image);
        packets.push((path.clone(), packet, recon_psnr));
    }

    let mut max_offdiag_similarity = 0.0f32;
    let mut pair_count = 0usize;
    for i in 0..packets.len() {
        for j in 0..packets.len() {
            if i == j {
                continue;
            }
            let sim = packet_similarity(&packets[i].1, &packets[j].1);
            max_offdiag_similarity = max_offdiag_similarity.max(sim.combined_similarity);
            pair_count += 1;
        }
    }

    if has_flag(args, "--json") {
        print!(
            "{{\"ok\":true,\"experiment_version\":\"{}\",\"fixture_dir\":\"{}\",\"params\":{},\"images\":{},\"min_psnr_db\":{},\"max_reconstruction_mse\":{:.8},\"min_self_similarity\":{:.8},\"max_offdiag_similarity\":{:.8},\"mean_edge_energy\":{:.8},\"packets\":[",
            PROBE_EXPERIMENT_VERSION,
            json_escape(&fixture_dir.to_string_lossy()),
            params.to_json(),
            packets.len(),
            json_float(min_psnr),
            max_recon_mse,
            min_self_similarity,
            max_offdiag_similarity,
            total_edge_energy / packets.len() as f32,
        );
        for (i, (path, packet, recon_psnr)) in packets.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!(
                "{{\"path\":\"{}\",\"packet_hash64\":\"{:016x}\",\"psnr_db\":{}}}",
                json_escape(&display_path(path)),
                packet.stable_hash64(),
                json_float(*recon_psnr)
            );
        }
        println!("]}}");
    } else {
        println!("self_test=ok");
        println!("experiment_version={PROBE_EXPERIMENT_VERSION}");
        println!("fixture_dir={}", fixture_dir.to_string_lossy());
        println!("{}", params.to_pretty_text());
        println!("images={}", packets.len());
        println!("min_psnr_db={}", csv_float(min_psnr));
        println!("max_reconstruction_mse={max_recon_mse:.8}");
        println!("min_self_similarity={min_self_similarity:.8}");
        println!("max_offdiag_similarity={max_offdiag_similarity:.8}");
        println!("pair_count={pair_count}");
        println!(
            "mean_edge_energy={:.8}",
            total_edge_energy / packets.len() as f32
        );
        for (path, packet, recon_psnr) in packets {
            println!(
                "{}\t{:016x}\tpsnr_db={}",
                display_path(&path),
                packet.stable_hash64(),
                csv_float(recon_psnr)
            );
        }
    }
    Ok(())
}

fn pipeline(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        return Err(ProbeError::InvalidArgs(
            "usage: svcp pipeline <image-dir> <work-dir> [--block N] [--keep K] [--topology-levels N] [--json]".into(),
        ));
    }
    let image_dir = Path::new(&args[0]);
    let work_dir = Path::new(&args[1]);
    let packet_dir = work_dir.join("packets");
    let block = flag_usize(args, "--block", 8)?;
    let keep = flag_usize(args, "--keep", 10)?;
    let topology_levels = flag_usize(args, "--topology-levels", 16)?;
    let params = EncodingParams::new(block, keep, topology_levels)?;
    fs::create_dir_all(&packet_dir)?;

    let manifest_path = work_dir.join("manifest.tsv");
    let benchmark_path = work_dir.join("benchmark.csv");
    let matrix_path = work_dir.join("similarity.csv");
    let summaries_path = work_dir.join("summaries.jsonl");

    let mut manifest_text = String::new();
    manifest_text.push_str(packet_manifest_header());
    manifest_text.push('\n');
    let mut benchmark_text = String::from(
        "path,image_hash64,edge_energy,block,keep,stored_coefficients,text_to_raw_ratio,mse,psnr_db,combined_similarity\n",
    );
    let mut summaries_text = String::new();
    let mut packets: Vec<(PathBuf, VisualMemoryPacket)> = Vec::new();

    for path in pgm_paths(image_dir)? {
        let image = GrayImage::read_pgm(&path)?;
        let packet = VisualMemoryPacket::encode_with_params(&image, params)?;
        packet.validate()?;
        let report = benchmark_image(&image, block, keep)?;
        let summary = visual_summary(&image, params)?;
        let stem = path.file_stem().and_then(|x| x.to_str()).unwrap_or("image");
        let packet_path = packet_dir.join(format!("{stem}.svmp"));
        packet.write_text(&packet_path)?;
        manifest_text.push_str(&packet_manifest_row(&display_path(&packet_path), &packet));
        manifest_text.push('\n');
        benchmark_text.push_str(&format!(
            "{},{:016x},{:.8},{},{},{},{:.8},{:.8},{},{}\n",
            csv_escape(&display_path(&path)),
            image_hash64(&image),
            edge_energy(&image),
            block,
            keep,
            report.metrics.stored_coefficients,
            report.metrics.text_to_raw_ratio,
            report.mse,
            csv_float(report.psnr_db),
            csv_float(report.self_similarity.combined_similarity)
        ));
        summaries_text.push_str(&summary.to_json());
        summaries_text.push('\n');
        packets.push((packet_path, packet));
    }

    let mut matrix_text = String::from("path");
    for (path, _) in &packets {
        matrix_text.push(',');
        matrix_text.push_str(&csv_escape(&display_path(path)));
    }
    matrix_text.push('\n');
    for (row_path, row_packet) in &packets {
        matrix_text.push_str(&csv_escape(&display_path(row_path)));
        for (_, col_packet) in &packets {
            let sim = packet_similarity(row_packet, col_packet);
            matrix_text.push_str(&format!(",{:.8}", sim.combined_similarity));
        }
        matrix_text.push('\n');
    }

    fs::create_dir_all(work_dir)?;
    fs::write(&manifest_path, &manifest_text)?;
    fs::write(&benchmark_path, &benchmark_text)?;
    fs::write(&matrix_path, &matrix_text)?;
    fs::write(&summaries_path, &summaries_text)?;

    if has_flag(args, "--json") {
        println!(
            "{{\"ok\":true,\"work_dir\":\"{}\",\"params\":{},\"packet_count\":{},\"manifest\":\"{}\",\"benchmark\":\"{}\",\"matrix\":\"{}\",\"summaries\":\"{}\"}}",
            json_escape(&work_dir.to_string_lossy()),
            params.to_json(),
            packets.len(),
            json_escape(&manifest_path.to_string_lossy()),
            json_escape(&benchmark_path.to_string_lossy()),
            json_escape(&matrix_path.to_string_lossy()),
            json_escape(&summaries_path.to_string_lossy()),
        );
    } else {
        println!("pipeline=ok");
        println!("work_dir={}", work_dir.to_string_lossy());
        println!("{}", params.to_pretty_text());
        println!("packet_count={}", packets.len());
        println!("manifest={}", manifest_path.to_string_lossy());
        println!("benchmark={}", benchmark_path.to_string_lossy());
        println!("matrix={}", matrix_path.to_string_lossy());
        println!("summaries={}", summaries_path.to_string_lossy());
    }
    Ok(())
}

fn doctor(args: &[String]) -> Result<()> {
    if has_flag(args, "--json") {
        println!(
            "{{\"crate\":\"symthaea-visual-compression-probe\",\"version\":\"{}\",\"recommended_test\":\"cargo test -p symthaea-visual-compression-probe\",\"avoid\":\"cargo test symthaea-visual-compression-probe\",\"reason\":\"without -p, Cargo treats the argument as a test-name filter and may compile unrelated workspace test targets\"}}",
            PROBE_EXPERIMENT_VERSION
        );
    } else {
        println!("symthaea-visual-compression-probe {PROBE_EXPERIMENT_VERSION}");
        println!("recommended_test=cargo test -p symthaea-visual-compression-probe");
        println!(
            "recommended_smoke=cargo run -p symthaea-visual-compression-probe --bin svcp -- self-test crates/domains/symthaea-visual-compression-probe/fixtures --json"
        );
        println!("avoid=cargo test symthaea-visual-compression-probe");
        println!(
            "why=without -p, Cargo treats the argument as a test-name filter and may compile unrelated workspace test targets"
        );
        println!(
            "note=workspace warnings about ignored package-local profiles and unused patches are upstream workspace hygiene, not failures in this probe crate"
        );
    }
    Ok(())
}

fn pgm_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|x| x.to_str())
            .unwrap_or_default();
        if ext.eq_ignore_ascii_case("pgm") {
            paths.push(path);
        }
    }
    paths.sort_by_key(|path| display_path(path));
    Ok(paths)
}

fn svmp_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|x| x.to_str())
            .unwrap_or_default();
        if ext.eq_ignore_ascii_case("svmp") {
            paths.push(path);
        }
    }
    paths.sort_by_key(|path| display_path(path));
    Ok(paths)
}

fn write_or_print(output: &str, contents: &str) -> Result<()> {
    if output == "-" {
        print!("{contents}");
    } else {
        fs::write(output, contents)?;
    }
    Ok(())
}

fn display_path(path: &PathBuf) -> String {
    path.to_string_lossy().to_string()
}

fn json_escape(input: &str) -> String {
    let mut out = String::new();
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn csv_escape(input: &str) -> String {
    if input.contains(',') || input.contains('"') || input.contains('\n') || input.contains('\r') {
        format!("\"{}\"", input.replace('"', "\"\""))
    } else {
        input.to_string()
    }
}

fn json_float(value: f32) -> String {
    if value.is_finite() {
        format!("{value:.8}")
    } else if value.is_infinite() && value.is_sign_positive() {
        "null".to_string()
    } else if value.is_infinite() && value.is_sign_negative() {
        "null".to_string()
    } else {
        "null".to_string()
    }
}

fn csv_float(value: f32) -> String {
    if value.is_finite() {
        format!("{value:.8}")
    } else if value.is_infinite() && value.is_sign_positive() {
        "inf".to_string()
    } else if value.is_infinite() && value.is_sign_negative() {
        "-inf".to_string()
    } else {
        "nan".to_string()
    }
}

fn flag_usize(args: &[String], flag: &str, default: usize) -> Result<usize> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag {
            let value = args
                .get(i + 1)
                .ok_or_else(|| ProbeError::InvalidArgs(format!("missing value for {flag}")))?;
            return value
                .parse::<usize>()
                .map_err(|_| ProbeError::InvalidArgs(format!("bad integer for {flag}: {value}")));
        }
        i += 1;
    }
    Ok(default)
}

fn flag_list_usize(args: &[String], flag: &str, default: &[usize]) -> Result<Vec<usize>> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag {
            let value = args
                .get(i + 1)
                .ok_or_else(|| ProbeError::InvalidArgs(format!("missing value for {flag}")))?;
            let mut parsed = Vec::new();
            for part in value.split(',') {
                if part.trim().is_empty() {
                    continue;
                }
                parsed.push(part.trim().parse::<usize>().map_err(|_| {
                    ProbeError::InvalidArgs(format!("bad integer in {flag}: {part}"))
                })?);
            }
            if parsed.is_empty() {
                return Err(ProbeError::InvalidArgs(format!("empty list for {flag}")));
            }
            return Ok(parsed);
        }
        i += 1;
    }
    Ok(default.to_vec())
}

fn flag_string(args: &[String], flag: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag {
            return args.get(i + 1).cloned();
        }
        i += 1;
    }
    None
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn print_help() {
    println!(
        "symthaea-visual-compression-probe {PROBE_EXPERIMENT_VERSION}\n\n\
Usage:\n\
  svcp inspect <input.pgm>\n\
  svcp encode <input.pgm> <output.svmp> [--block N] [--keep K]\n\
  svcp decode <input.svmp> <output.pgm>\n\
  svcp compare <a.pgm> <b.pgm>\n\
  svcp metrics <input.svmp> [--json]\n\
  svcp fingerprint <input.pgm|input.svmp> [--json] [--block N] [--keep K]\n\
  svcp benchmark <input.pgm> [--block N] [--keep K] [--json]\n\
  svcp query <query.svmp> <packet-dir> [--top N] [--json]\n\
  svcp validate <input.svmp> [--json]\n\
  svcp diff <a.svmp> <b.svmp> [--json]\n\
  svcp sweep <input.pgm> [--blocks 4,8,16] [--keeps 2,4,8,12] [--csv|--json]\n\
  svcp index <packet-dir> <output.tsv|-> [--json]\n\
  svcp corpus-benchmark <image-dir> [--block N] [--keep K] [--json]\n\
  svcp summary <input.pgm> [--block N] [--keep K] [--topology-levels N] [--json]\n\
  svcp batch-encode <image-dir> <packet-dir> [--block N] [--keep K] [--topology-levels N] [--manifest OUT] [--json]\n\
  svcp matrix <packet-dir> <output.csv|-> [--json]\n\
  svcp self-test [fixture-dir] [--block N] [--keep K] [--topology-levels N] [--json]\n\
  svcp pipeline <image-dir> <work-dir> [--block N] [--keep K] [--topology-levels N] [--json]\n\
  svcp doctor [--json]\n\n\
Notes:\n\
  - P2 and P5 grayscale PGM are supported.\n\
  - .svmp is a readable prototype packet, not a compact production format.\n\
  - Query ranks packets by HDC + topology similarity without decoding pixels.\n\
  - Sweep/corpus-benchmark are for claim discipline: compare parameters before claiming wins."
    );
}
