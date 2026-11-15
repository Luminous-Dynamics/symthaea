use std::env;
use std::fs;

use sympsi_beacon_verifier::verify_quicknet_beacon;

fn main() {
    if let Err(error) = run() {
        eprintln!("sympsi-beacon-verifier: {error:?}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    if arguments.len() == 0 || arguments.len() > 2 {
        return Err(
            "usage: sympsi-beacon-verifier <round> <signature-hex-file> (exactly 48 bytes)"
                .into(),
        );
    }

    let requested_round = arguments
        .next()
        .ok_or_else(|| "missing requested round".to_string())?
        .parse::<u64>()
        .map_err(|error| format!("invalid round: {error}"))?;
    let signature_path = arguments
        .next()
        .ok_or_else(|| "missing signature hex file".to_string())?;
    let signature_text = fs::read_to_string(&signature_path)
        .map_err(|error| format!("cannot read signature file: {error}"))?;
    let compact: String = signature_text.chars().filter(|ch| !ch.is_whitespace()).collect();
    if compact.len() != 96 {
        return Err("signature file must contain exactly 96 hex characters".into());
    }
    let decoded = decode_hex(&compact)?;
    let signature: [u8; 48] = decoded
        .try_into()
        .map_err(|_| "signature must decode to exactly 48 bytes".to_string())?;

    let verified = verify_quicknet_beacon(requested_round, requested_round, &signature)
        .map_err(|error| format!("verification failed: {error:?}"))?;
    println!("verification=VALID_SIGNATURE_FOR_PINNED_QUICKNET_TRUST_ROOT");
    println!("round={}", verified.round());
    println!("scheduled_unix_seconds={}", verified.scheduled_unix_seconds());
    println!("chain_hash={}", encode_hex(verified.chain_hash()));
    println!("randomness_public={}", encode_hex(verified.randomness()));
    println!("freshness=NOT_ESTABLISHED");
    println!("response_commitment_chronology=NOT_ESTABLISHED");
    Ok(())
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if !value.len().is_multiple_of(2) {
        return Err("hex input has an odd number of characters".into());
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let hi = hex_nibble(pair[0])?;
        let lo = hex_nibble(pair[1])?;
        bytes.push((hi << 4) | lo);
    }
    Ok(bytes)
}

fn hex_nibble(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("signature contains a non-hex character".into()),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}
