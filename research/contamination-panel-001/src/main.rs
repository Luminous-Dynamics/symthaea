use contamination_panel_bench::{
    ADJUDICATION_REPORT_SCHEMA, AdjudicationRecordV1, PublicChallengePacketV1,
    audit_blinded_adjudications, generate_challenge_bundle, GeneratorConfigV1,
};
use std::{
    env,
    error::Error,
    fs,
    io::{self, BufRead},
    path::Path,
    process,
};

fn decode_hex(input: &str) -> Result<Vec<u8>, String> {
    if input.len() % 2 != 0 {
        return Err("seed hex must contain an even number of characters".to_owned());
    }
    let mut output = Vec::with_capacity(input.len() / 2);
    for pair in input.as_bytes().chunks_exact(2) {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("seed_hex contains a non-hexadecimal character".to_owned()),
    }
}

fn write_bytes(path: &str, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, contents)?;
    Ok(())
}

fn generate(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 7 {
        return Err(
            "usage: contamination-panel-bench generate <generation-id> <pairs-per-tier> <seed-hex> <public-packet.json> <sealed-key.json> <manifest.json>"
                .into(),
        );
    }
    let pairs_per_tier = args[2].parse::<u32>()?;
    let seed = decode_hex(&args[3]).map_err(io::Error::other)?;
    let config = GeneratorConfigV1 {
        generation_id: args[1].clone(),
        pairs_per_tier,
    };
    let bundle = generate_challenge_bundle(&config, &seed).map_err(io::Error::other)?;

    write_bytes(&args[4], &bundle.public_packet_bytes)?;
    write_bytes(&args[5], &bundle.sealed_key_bytes)?;
    write_bytes(&args[6], &bundle.manifest_bytes)?;

    println!("public_packet_sha256={}", bundle.public_packet_sha256);
    println!("sealed_key_sha256={}", bundle.sealed_key_sha256);
    println!("seed_commitment_sha256={}", bundle.seed_commitment_sha256);
    println!("WARNING: keep the sealed key and seed outside reviewer/model access.");
    Ok(())
}

fn audit_adjudications(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 4 {
        return Err(
            "usage: contamination-panel-bench audit-adjudications <public-packet.json> <adjudications.jsonl> <report.json>"
                .into(),
        );
    }
    let packet_bytes = fs::read(&args[1])?;
    let packet: PublicChallengePacketV1 = serde_json::from_slice(&packet_bytes)?;
    let file = fs::File::open(&args[2])?;
    let mut records = Vec::new();
    for (line_number, line) in io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: AdjudicationRecordV1 = serde_json::from_str(&line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid adjudication JSON at line {}: {error}", line_number + 1),
            )
        })?;
        records.push(record);
    }

    let report = audit_blinded_adjudications(&packet, &records).map_err(io::Error::other)?;
    if report.schema != ADJUDICATION_REPORT_SCHEMA {
        return Err("internal report schema mismatch".into());
    }
    let report_bytes = serde_json::to_vec_pretty(&report)?;
    write_bytes(&args[3], &report_bytes)?;
    println!(
        "valid={} pairs={} disputed={} insufficient_evidence={}",
        report.valid,
        report.pair_count,
        report.disputed_pair_count,
        report.insufficient_evidence_pair_count
    );
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("generate") => generate(&args),
        Some("audit-adjudications") => audit_adjudications(&args),
        _ => Err(
            "commands: generate | audit-adjudications\nRun a command without the required arguments to see its usage."
                .into(),
        ),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(2);
    }
}
