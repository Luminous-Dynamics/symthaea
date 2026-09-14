// CORE-ID-001 explicit cross-language canonical lexical identity oracle.

use std::env;
use std::fs;
use std::path::Path;

const PROFILE_ID: &str = "symthaea.core.canonical-lexical-id.v1";

fn is_explicit_edge_whitespace(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0009..=0x000D
            | 0x0020
            | 0x0085
            | 0x00A0
            | 0x1680
            | 0x2000..=0x200A
            | 0x2028
            | 0x2029
            | 0x202F
            | 0x205F
            | 0x3000
    )
}

fn is_forbidden_control(ch: char) -> bool {
    let cp = ch as u32;
    cp < 0x20 || cp == 0x7F
}

fn admit_utf8_bytes(data: &[u8], max_utf8_bytes: usize) -> bool {
    if max_utf8_bytes == 0 || data.is_empty() || data.len() > max_utf8_bytes {
        return false;
    }
    let value = match std::str::from_utf8(data) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let first = match value.chars().next() {
        Some(ch) => ch,
        None => return false,
    };
    let last = match value.chars().next_back() {
        Some(ch) => ch,
        None => return false,
    };
    if is_explicit_edge_whitespace(first) || is_explicit_edge_whitespace(last) {
        return false;
    }
    !value.chars().any(is_forbidden_control)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn decode_hex(text: &str) -> Result<Vec<u8>, String> {
    let bytes = text.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return Err("hex input must have even length".to_string());
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let hi = hex_nibble(pair[0]).ok_or_else(|| "invalid hex digit".to_string())?;
        let lo = hex_nibble(pair[1]).ok_or_else(|| "invalid hex digit".to_string())?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn run_vectors(path: &Path) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut out = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = raw.split('\t').collect();
        if parts.len() != 4 {
            return Err(format!("vector line {}: expected 4 fields", index + 1));
        }
        let name = parts[0];
        let max_utf8_bytes = parts[1]
            .parse::<usize>()
            .map_err(|_| format!("vector {name}: invalid byte limit"))?;
        let data = decode_hex(parts[2])?;
        let actual = if admit_utf8_bytes(&data, max_utf8_bytes) {
            "accept"
        } else {
            "reject"
        };
        let expected = parts[3];
        if expected != "accept" && expected != "reject" {
            return Err(format!("vector {name}: invalid expected status"));
        }
        if actual != expected {
            return Err(format!(
                "vector {name}: expected {expected}, got {actual}"
            ));
        }
        out.push(format!("{name}\t{actual}"));
    }
    if out.is_empty() {
        return Err("vector corpus must not be empty".to_string());
    }
    Ok(out)
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_string())
    }
}

fn self_test() -> Result<(), String> {
    require(
        PROFILE_ID == "symthaea.core.canonical-lexical-id.v1",
        "profile id mismatch",
    )?;

    let ascii_4096 = vec![b'a'; 4096];
    let ascii_4097 = vec![b'a'; 4097];
    require(admit_utf8_bytes(&ascii_4096, 4096), "4096-byte boundary")?;
    require(
        !admit_utf8_bytes(&ascii_4097, 4096),
        "4097-byte overflow",
    )?;

    let ascii_256 = vec![b'a'; 256];
    let ascii_257 = vec![b'a'; 257];
    require(admit_utf8_bytes(&ascii_256, 256), "256-byte boundary")?;
    require(!admit_utf8_bytes(&ascii_257, 256), "257-byte overflow")?;

    let multibyte_4096 = "é".repeat(2048);
    let multibyte_4098 = "é".repeat(2049);
    require(
        multibyte_4096.len() == 4096 && admit_utf8_bytes(multibyte_4096.as_bytes(), 4096),
        "multibyte exact boundary",
    )?;
    require(
        !admit_utf8_bytes(multibyte_4098.as_bytes(), 4096),
        "multibyte overflow",
    )?;

    require(
        admit_utf8_bytes("p\u{00A0}1".as_bytes(), 4096),
        "interior NBSP must remain admitted",
    )?;
    require(
        admit_utf8_bytes("\u{200B}x".as_bytes(), 4096),
        "zero-width space is not V1 edge whitespace",
    )?;
    require(
        admit_utf8_bytes("\u{FEFF}x".as_bytes(), 4096),
        "BOM is not V1 edge whitespace",
    )?;
    require(
        !admit_utf8_bytes("\u{00A0}x".as_bytes(), 4096),
        "leading NBSP must be rejected",
    )?;
    require(
        !admit_utf8_bytes("x\u{3000}".as_bytes(), 4096),
        "trailing ideographic space must be rejected",
    )?;

    require(
        "Process-A".as_bytes() != "process-a".as_bytes(),
        "case must remain byte-distinct",
    )?;
    require(
        "é".as_bytes() != "e\u{0301}".as_bytes(),
        "normalization must not be implicit",
    )?;

    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    match args.as_slice() {
        [_, flag] if flag == "--self-test" => {
            self_test()?;
            println!("ok");
            Ok(())
        }
        [_, flag, path] if flag == "--vectors" => {
            self_test()?;
            for line in run_vectors(Path::new(path))? {
                println!("{line}");
            }
            Ok(())
        }
        _ => Err("choose --vectors <path> or --self-test".to_string()),
    }
}
