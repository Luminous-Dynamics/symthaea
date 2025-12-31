//! Kokoro TTS Tokenizer
//!
//! Converts text to phoneme tokens using Misaki (official Kokoro G2P) as primary,
//! with espeak-ng as fallback. Misaki produces the exact IPA format that Kokoro expects.
//!
//! Performance optimization: Uses a persistent Misaki server via socket to avoid
//! the 7-11 second model loading overhead on each tokenization.

use std::collections::HashMap;
use std::process::Command;
use std::path::PathBuf;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::net::TcpStream;
use std::time::Duration;

/// Kokoro vocabulary - maps IPA characters to token IDs
fn build_vocab() -> HashMap<char, i64> {
    let mut vocab = HashMap::new();

    // Punctuation and special
    vocab.insert('$', 0);   // BOS/EOS
    vocab.insert(';', 1);
    vocab.insert(':', 2);
    vocab.insert(',', 3);
    vocab.insert('.', 4);
    vocab.insert('!', 5);
    vocab.insert('?', 6);
    vocab.insert('—', 9);   // em dash
    vocab.insert('…', 10);  // ellipsis
    vocab.insert('"', 11);
    vocab.insert('(', 12);
    vocab.insert(')', 13);
    vocab.insert('"', 14);  // left double quote
    vocab.insert('"', 15);  // right double quote
    vocab.insert(' ', 16);

    // IPA combining/modifiers
    vocab.insert('\u{0303}', 17);  // combining tilde
    vocab.insert('ʣ', 18);
    vocab.insert('ʥ', 19);
    vocab.insert('ʦ', 20);
    vocab.insert('ʨ', 21);
    vocab.insert('ᵝ', 22);
    vocab.insert('ꭧ', 23);

    // Uppercase (tone markers in some dialects)
    vocab.insert('A', 24);
    vocab.insert('I', 25);
    vocab.insert('O', 31);
    vocab.insert('Q', 33);
    vocab.insert('S', 35);
    vocab.insert('T', 36);
    vocab.insert('W', 39);
    vocab.insert('Y', 41);
    vocab.insert('ᵊ', 42);  // modifier schwa

    // Basic Latin lowercase
    vocab.insert('a', 43);
    vocab.insert('b', 44);
    vocab.insert('c', 45);
    vocab.insert('d', 46);
    vocab.insert('e', 47);
    vocab.insert('f', 48);
    vocab.insert('h', 50);
    vocab.insert('i', 51);
    vocab.insert('j', 52);
    vocab.insert('k', 53);
    vocab.insert('l', 54);
    vocab.insert('m', 55);
    vocab.insert('n', 56);
    vocab.insert('o', 57);
    vocab.insert('p', 58);
    vocab.insert('q', 59);
    vocab.insert('r', 60);
    vocab.insert('s', 61);
    vocab.insert('t', 62);
    vocab.insert('u', 63);
    vocab.insert('v', 64);
    vocab.insert('w', 65);
    vocab.insert('x', 66);
    vocab.insert('y', 67);
    vocab.insert('z', 68);

    // IPA vowels
    vocab.insert('ɑ', 69);  // open back unrounded
    vocab.insert('ɐ', 70);  // near-open central
    vocab.insert('ɒ', 71);  // open back rounded
    vocab.insert('æ', 72);  // near-open front unrounded
    vocab.insert('β', 75);  // voiced bilabial fricative
    vocab.insert('ɔ', 76);  // open-mid back rounded
    vocab.insert('ɕ', 77);  // voiceless alveolo-palatal fricative
    vocab.insert('ç', 78);  // voiceless palatal fricative
    vocab.insert('ɖ', 80);  // voiced retroflex plosive
    vocab.insert('ð', 81);  // voiced dental fricative (th in "the")
    vocab.insert('ʤ', 82);  // voiced postalveolar affricate (j in "judge")
    vocab.insert('ə', 83);  // schwa
    vocab.insert('ɚ', 85);  // r-colored schwa
    vocab.insert('ɛ', 86);  // open-mid front unrounded
    vocab.insert('ɜ', 87);  // open-mid central unrounded
    vocab.insert('ɟ', 90);  // voiced palatal plosive
    vocab.insert('ɡ', 92);  // voiced velar plosive
    vocab.insert('ɥ', 99);  // labial-palatal approximant
    vocab.insert('ɨ', 101); // close central unrounded
    vocab.insert('ɪ', 102); // near-close front unrounded
    vocab.insert('ʝ', 103); // voiced palatal fricative
    vocab.insert('ɯ', 110); // close back unrounded
    vocab.insert('ɰ', 111); // velar approximant
    vocab.insert('ŋ', 112); // velar nasal (ng)
    vocab.insert('ɳ', 113); // retroflex nasal
    vocab.insert('ɲ', 114); // palatal nasal
    vocab.insert('ɴ', 115); // uvular nasal
    vocab.insert('ø', 116); // close-mid front rounded
    vocab.insert('ɸ', 118); // voiceless bilabial fricative
    vocab.insert('θ', 119); // voiceless dental fricative (th in "think")
    vocab.insert('œ', 120); // open-mid front rounded
    vocab.insert('ɹ', 123); // alveolar approximant (English r)
    vocab.insert('ɾ', 125); // alveolar tap
    vocab.insert('ɻ', 126); // retroflex approximant
    vocab.insert('ʁ', 128); // voiced uvular fricative
    vocab.insert('ɽ', 129); // retroflex flap
    vocab.insert('ʂ', 130); // voiceless retroflex fricative
    vocab.insert('ʃ', 131); // voiceless postalveolar fricative (sh)
    vocab.insert('ʈ', 132); // voiceless retroflex plosive
    vocab.insert('ʧ', 133); // voiceless postalveolar affricate (ch)
    vocab.insert('ʊ', 135); // near-close back rounded
    vocab.insert('ʋ', 136); // labiodental approximant
    vocab.insert('ʌ', 138); // open-mid back unrounded
    vocab.insert('ɣ', 139); // voiced velar fricative
    vocab.insert('ɤ', 140); // close-mid back unrounded
    vocab.insert('χ', 142); // voiceless uvular fricative
    vocab.insert('ʎ', 143); // palatal lateral approximant
    vocab.insert('ʒ', 147); // voiced postalveolar fricative (zh)
    vocab.insert('ʔ', 148); // glottal stop

    // Stress and length markers
    vocab.insert('ˈ', 156); // primary stress
    vocab.insert('ˌ', 157); // secondary stress
    vocab.insert('ː', 158); // length mark
    vocab.insert('ʰ', 162); // aspiration
    vocab.insert('ʲ', 164); // palatalization

    // Tone markers
    vocab.insert('↓', 169);
    vocab.insert('→', 171);
    vocab.insert('↗', 172);
    vocab.insert('↘', 173);
    vocab.insert('ᵻ', 177);

    vocab
}

const MISAKI_SOCKET_PATH: &str = "/tmp/misaki_server.sock";
const MISAKI_TCP_PORT: u16 = 19877;

/// Get path to Misaki tokenizer script
fn get_misaki_script_path() -> PathBuf {
    // Check XDG data home first, then fall back to ~/.local/share
    let data_home = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join(".local/share")
        });

    data_home.join("symthaea/bin/misaki_tokenize.py")
}

/// Parse JSON response from Misaki server
fn parse_misaki_response(response: &str) -> Result<(Vec<i64>, String), String> {
    let json: serde_json::Value = serde_json::from_str(response.trim())
        .map_err(|e| format!("Failed to parse Misaki JSON: {} - response: {}", e, response))?;

    if let Some(error) = json.get("error") {
        return Err(format!("Misaki error: {}", error));
    }

    let phonemes = json["phonemes"]
        .as_str()
        .ok_or("Missing phonemes in Misaki output")?
        .to_string();

    let tokens: Vec<i64> = json["tokens"]
        .as_array()
        .ok_or("Missing tokens in Misaki output")?
        .iter()
        .filter_map(|v| v.as_i64())
        .collect();

    Ok((tokens, phonemes))
}

/// Tokenize using the persistent Misaki server (Unix socket)
fn tokenize_with_misaki_server_unix(text: &str) -> Result<(Vec<i64>, String), String> {
    let mut stream = UnixStream::connect(MISAKI_SOCKET_PATH)
        .map_err(|e| format!("Failed to connect to Misaki server: {}", e))?;

    stream.set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| format!("Failed to set timeout: {}", e))?;

    // Send text with newline delimiter
    stream.write_all(format!("{}\n", text).as_bytes())
        .map_err(|e| format!("Failed to send to Misaki server: {}", e))?;

    // Read response
    let mut response = String::new();
    stream.read_to_string(&mut response)
        .map_err(|e| format!("Failed to read from Misaki server: {}", e))?;

    parse_misaki_response(&response)
}

/// Tokenize using the persistent Misaki server (TCP fallback)
fn tokenize_with_misaki_server_tcp(text: &str) -> Result<(Vec<i64>, String), String> {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", MISAKI_TCP_PORT))
        .map_err(|e| format!("Failed to connect to Misaki TCP server: {}", e))?;

    stream.set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| format!("Failed to set timeout: {}", e))?;

    // Send text with newline delimiter
    stream.write_all(format!("{}\n", text).as_bytes())
        .map_err(|e| format!("Failed to send to Misaki server: {}", e))?;

    // Read response
    let mut response = String::new();
    stream.read_to_string(&mut response)
        .map_err(|e| format!("Failed to read from Misaki server: {}", e))?;

    parse_misaki_response(&response)
}

/// Tokenize using Misaki subprocess (fallback - SLOW, ~7-11 seconds)
fn tokenize_with_misaki_subprocess(text: &str) -> Result<(Vec<i64>, String), String> {
    let script_path = get_misaki_script_path();

    if !script_path.exists() {
        return Err(format!("Misaki script not found at {:?}", script_path));
    }

    // Call the Python wrapper script
    let output = Command::new("python3")
        .arg(&script_path)
        .arg(text)
        .output()
        .map_err(|e| format!("Failed to run Misaki: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Misaki failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_misaki_response(&stdout)
}

/// Tokenize using Misaki (official Kokoro G2P)
/// Returns (tokens, phonemes) on success
///
/// Tries connection methods in order of speed:
/// 1. Unix socket (fastest, ~5ms)
/// 2. TCP socket (fast, ~10ms)
/// 3. Subprocess (slow, ~7-11 seconds - model reloads each time)
fn tokenize_with_misaki(text: &str) -> Result<(Vec<i64>, String), String> {
    // Try Unix socket first (fastest)
    if let Ok(result) = tokenize_with_misaki_server_unix(text) {
        return Ok(result);
    }

    // Try TCP socket (fallback)
    if let Ok(result) = tokenize_with_misaki_server_tcp(text) {
        return Ok(result);
    }

    // Fall back to subprocess (slow but always works)
    tracing::warn!("Misaki server not running, falling back to slow subprocess mode");
    tracing::warn!("Start the server for 1000x faster tokenization: python3 ~/.local/share/symthaea/bin/misaki_server.py");
    tokenize_with_misaki_subprocess(text)
}

/// Convert text to IPA phonemes using espeak-ng (fallback)
fn text_to_ipa_espeak(text: &str) -> Result<String, String> {
    // Use American English voice for consistency
    let output = Command::new("espeak-ng")
        .args([
            "--ipa", "-q",  // IPA output, quiet
            "-v", "en-us",  // American English
            text
        ])
        .output()
        .map_err(|e| format!("espeak-ng failed: {}", e))?;

    if !output.status.success() {
        return Err(format!("espeak-ng error: {}", String::from_utf8_lossy(&output.stderr)));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Tokenize using espeak-ng (fallback when Misaki unavailable)
fn tokenize_with_espeak(text: &str) -> Result<(Vec<i64>, String), String> {
    let vocab = build_vocab();
    let ipa = text_to_ipa_espeak(text)?;

    // Start with BOS token
    let mut tokens = vec![0i64];

    // Map each IPA character to token ID
    for ch in ipa.chars() {
        if let Some(&token_id) = vocab.get(&ch) {
            tokens.push(token_id);
        }
        // Skip unknown characters
    }

    // End with EOS token
    tokens.push(0);

    Ok((tokens, ipa))
}

/// Tokenize text to Kokoro token IDs
/// Uses Misaki (official Kokoro G2P) as primary, espeak-ng as fallback
pub fn tokenize(text: &str) -> Result<Vec<i64>, String> {
    // Try Misaki first (produces correct IPA for Kokoro)
    match tokenize_with_misaki(text) {
        Ok((tokens, _phonemes)) => {
            tracing::debug!("Tokenized with Misaki: {} tokens", tokens.len());
            Ok(tokens)
        }
        Err(e) => {
            tracing::warn!("Misaki unavailable ({}), falling back to espeak-ng", e);
            let (tokens, _ipa) = tokenize_with_espeak(text)?;
            Ok(tokens)
        }
    }
}

/// Tokenize with detailed debug output
pub fn tokenize_debug(text: &str) -> Result<(Vec<i64>, String), String> {
    // Try Misaki first
    match tokenize_with_misaki(text) {
        Ok((tokens, phonemes)) => {
            tracing::info!("Misaki phonemes: {}", phonemes);
            Ok((tokens, phonemes))
        }
        Err(e) => {
            tracing::warn!("Misaki unavailable ({}), falling back to espeak-ng", e);
            tokenize_with_espeak(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_ipa_espeak() {
        let result = text_to_ipa_espeak("Hello world").unwrap();
        assert!(!result.is_empty());
        // Should contain IPA characters
        assert!(result.contains('h') || result.contains('ə') || result.contains('l'));
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello").unwrap();
        // Should have BOS, phoneme tokens, EOS
        assert!(tokens.len() >= 3);
        assert_eq!(tokens[0], 0);  // BOS
        assert_eq!(tokens[tokens.len() - 1], 0);  // EOS
    }

    #[test]
    fn test_vocab_has_common_phonemes() {
        let vocab = build_vocab();
        // Check common IPA symbols exist
        assert!(vocab.contains_key(&'ə'));  // schwa
        assert!(vocab.contains_key(&'ŋ'));  // eng (ng)
        assert!(vocab.contains_key(&'ʃ'));  // sh
        assert!(vocab.contains_key(&'θ'));  // th (think)
        assert!(vocab.contains_key(&'ð'));  // th (the)
    }
}
