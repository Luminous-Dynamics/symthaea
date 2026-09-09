//! HAK-015 dependency-free Rust reference for `hak.canonical-json.v1`.
//!
//! This is intentionally a narrow evidence-metadata canonicalizer, not a
//! general-purpose JSON implementation. It exists as an independent runtime
//! check against the Python reference implementation.

use std::collections::HashSet;

const PROFILE_ID: &str = "hak.canonical-json.v1";
const MAX_SAFE_INTEGER: i128 = 9_007_199_254_740_991;

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, pos: 0 } }

    fn parse(mut self) -> Result<Value, String> {
        let value = self.parse_value()?;
        self.skip_ws();
        if self.pos != self.bytes.len() {
            return Err(format!("trailing data at byte {}", self.pos));
        }
        Ok(value)
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b' ' | b'\n' | b'\r' | b'\t') {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some(b'n') => { self.expect_bytes(b"null")?; Ok(Value::Null) }
            Some(b't') => { self.expect_bytes(b"true")?; Ok(Value::Bool(true)) }
            Some(b'f') => { self.expect_bytes(b"false")?; Ok(Value::Bool(false)) }
            Some(b'"') => Ok(Value::Str(self.parse_string()?)),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-' | b'0'..=b'9') => self.parse_integer(),
            Some(other) => Err(format!("unexpected byte 0x{other:02x} at {}", self.pos)),
            None => Err("unexpected end of JSON".to_string()),
        }
    }

    fn expect_bytes(&mut self, expected: &[u8]) -> Result<(), String> {
        if self.bytes.get(self.pos..self.pos + expected.len()) == Some(expected) {
            self.pos += expected.len();
            Ok(())
        } else {
            Err(format!("expected {:?} at byte {}", String::from_utf8_lossy(expected), self.pos))
        }
    }

    fn parse_integer(&mut self) -> Result<Value, String> {
        self.skip_ws();
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') { self.pos += 1; }
        let first = *self.bytes.get(self.pos).ok_or_else(|| "incomplete number".to_string())?;
        match first {
            b'0' => {
                self.pos += 1;
                if matches!(self.bytes.get(self.pos).copied(), Some(b'0'..=b'9')) {
                    return Err("leading zero in JSON number".to_string());
                }
            }
            b'1'..=b'9' => {
                self.pos += 1;
                while matches!(self.bytes.get(self.pos).copied(), Some(b'0'..=b'9')) { self.pos += 1; }
            }
            _ => return Err("invalid JSON integer".to_string()),
        }
        if matches!(self.bytes.get(self.pos).copied(), Some(b'.' | b'e' | b'E')) {
            return Err("floating-point/exponent numbers are not permitted".to_string());
        }
        let token = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| e.to_string())?;
        let parsed: i128 = token.parse().map_err(|_| "integer parse overflow".to_string())?;
        if parsed < -MAX_SAFE_INTEGER || parsed > MAX_SAFE_INTEGER {
            return Err("integer outside HAK v1 safe range".to_string());
        }
        Ok(Value::Int(parsed as i64))
    }

    fn parse_array(&mut self) -> Result<Value, String> {
        self.skip_ws();
        self.pos += 1;
        let mut values = Vec::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b']') { self.pos += 1; return Ok(Value::Array(values)); }
        loop {
            values.push(self.parse_value()?);
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => { self.pos += 1; }
                Some(b']') => { self.pos += 1; break; }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.pos)),
            }
        }
        Ok(Value::Array(values))
    }

    fn parse_object(&mut self) -> Result<Value, String> {
        self.skip_ws();
        self.pos += 1;
        let mut values = Vec::new();
        let mut seen = HashSet::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b'}') { self.pos += 1; return Ok(Value::Object(values)); }
        loop {
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b'"') { return Err(format!("object key must be string at byte {}", self.pos)); }
            let key = self.parse_string()?;
            if !seen.insert(key.clone()) { return Err(format!("duplicate object key: {key:?}")); }
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b':') { return Err(format!("expected ':' at byte {}", self.pos)); }
            self.pos += 1;
            let value = self.parse_value()?;
            values.push((key, value));
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => { self.pos += 1; }
                Some(b'}') => { self.pos += 1; break; }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.pos)),
            }
        }
        Ok(Value::Object(values))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.skip_ws();
        if self.bytes.get(self.pos) != Some(&b'"') { return Err("expected string".to_string()); }
        self.pos += 1;
        let mut out = String::new();
        loop {
            let b = *self.bytes.get(self.pos).ok_or_else(|| "unterminated string".to_string())?;
            match b {
                b'"' => { self.pos += 1; break; }
                b'\\' => {
                    self.pos += 1;
                    let esc = *self.bytes.get(self.pos).ok_or_else(|| "incomplete escape".to_string())?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'), b'\\' => out.push('\\'), b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'), b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'), b'r' => out.push('\r'), b't' => out.push('\t'),
                        b'u' => {
                            let first = self.parse_hex_u16()?;
                            if (0xD800..=0xDBFF).contains(&first) {
                                if self.bytes.get(self.pos..self.pos + 2) != Some(&b"\\u"[..]) {
                                    return Err("high surrogate without low surrogate".to_string());
                                }
                                self.pos += 2;
                                let second = self.parse_hex_u16()?;
                                if !(0xDC00..=0xDFFF).contains(&second) {
                                    return Err("invalid low surrogate".to_string());
                                }
                                let scalar = 0x10000 + (((first as u32 - 0xD800) << 10) | (second as u32 - 0xDC00));
                                out.push(char::from_u32(scalar).ok_or_else(|| "invalid surrogate pair".to_string())?);
                            } else if (0xDC00..=0xDFFF).contains(&first) {
                                return Err("lone low surrogate".to_string());
                            } else {
                                out.push(char::from_u32(first as u32).ok_or_else(|| "invalid Unicode scalar".to_string())?);
                            }
                        }
                        _ => return Err(format!("invalid escape at byte {}", self.pos - 1)),
                    }
                }
                0x00..=0x1f => return Err("unescaped control character in string".to_string()),
                _ => {
                    let tail = std::str::from_utf8(&self.bytes[self.pos..]).map_err(|e| format!("invalid UTF-8: {e}"))?;
                    let ch = tail.chars().next().ok_or_else(|| "invalid UTF-8 string".to_string())?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
        Ok(out)
    }

    fn parse_hex_u16(&mut self) -> Result<u16, String> {
        let end = self.pos + 4;
        let bytes = self.bytes.get(self.pos..end).ok_or_else(|| "short Unicode escape".to_string())?;
        let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
        if !text.bytes().all(|b| b.is_ascii_hexdigit()) { return Err("invalid Unicode escape".to_string()); }
        self.pos = end;
        u16::from_str_radix(text, 16).map_err(|e| e.to_string())
    }
}

fn parse_strict_bytes(raw: &[u8]) -> Result<Value, String> {
    std::str::from_utf8(raw).map_err(|e| format!("invalid UTF-8: {e}"))?;
    Parser::new(raw).parse()
}

fn parse_strict_json(raw: &str) -> Result<Value, String> { parse_strict_bytes(raw.as_bytes()) }

fn quote_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{000c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) <= 0x1f => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn utf16_key(value: &str) -> Vec<u16> { value.encode_utf16().collect() }

fn canonical_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Int(n) => n.to_string(),
        Value::Str(s) => quote_string(s),
        Value::Array(items) => format!("[{}]", items.iter().map(canonical_string).collect::<Vec<_>>().join(",")),
        Value::Object(fields) => {
            let mut sorted: Vec<_> = fields.iter().collect();
            sorted.sort_by(|(ka, _), (kb, _)| utf16_key(ka).cmp(&utf16_key(kb)));
            let body = sorted.iter().map(|(k, v)| format!("{}:{}", quote_string(k), canonical_string(v))).collect::<Vec<_>>().join(",");
            format!("{{{body}}}")
        }
    }
}

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let mut h = [0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let mut msg = data.to_vec();
    let bit_len = (msg.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let j = i * 4;
            w[i] = u32::from_be_bytes([chunk[j], chunk[j+1], chunk[j+2], chunk[j+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh) = (h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh=g; g=f; f=e; e=d.wrapping_add(t1); d=c; c=b; b=a; a=t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a,b,c,d,e,f,g,hh]) { *slot = slot.wrapping_add(value); }
    }
    h.iter().map(|v| format!("{v:08x}")).collect::<String>()
}

fn hak_sha256(domain: &str, value: &Value) -> String {
    assert!(!domain.is_empty() && !domain.contains('\0'));
    let canonical = canonical_string(value);
    let mut preimage = Vec::new();
    preimage.extend_from_slice(PROFILE_ID.as_bytes());
    preimage.push(0);
    preimage.extend_from_slice(domain.as_bytes());
    preimage.push(0);
    preimage.extend_from_slice(canonical.as_bytes());
    format!("sha256:{}", sha256_hex(&preimage))
}

fn object_get<'a>(value: &'a Value, key: &str) -> &'a Value {
    match value {
        Value::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v).unwrap_or_else(|| panic!("missing key {key}")),
        _ => panic!("expected object"),
    }
}
fn as_str(value: &Value) -> &str { if let Value::Str(v)=value { v } else { panic!("expected string") } }
fn as_array(value: &Value) -> &[Value] { if let Value::Array(v)=value { v } else { panic!("expected array") } }
fn as_bool(value: &Value) -> bool { if let Value::Bool(v)=value { *v } else { panic!("expected bool") } }
fn as_int(value: &Value) -> i64 { if let Value::Int(v)=value { *v } else { panic!("expected int") } }

#[cfg(test)]
mod tests {
    use super::*;
    const GOLDEN: &str = include_str!("../docs/architecture/hak/golden/hak-canonical-json-v1.vectors.json");
    const PROFILE: &str = include_str!("../docs/architecture/hak/canonical-json-v1.profile.json");

    #[test]
    fn sha256_reference_vector() {
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn machine_readable_profile_matches_rust_contract() {
        let profile = parse_strict_json(PROFILE).expect("profile must parse");
        assert_eq!(as_str(object_get(&profile, "schema_version")), "hak.canonical-json-profile.v1");
        assert_eq!(as_str(object_get(&profile, "profile_id")), PROFILE_ID);
        let base = object_get(&profile, "base_standard");
        assert_eq!(as_str(object_get(base, "name")), "RFC 8785");
        assert_eq!(as_str(object_get(base, "relationship")), "CompatibleSubset");

        let input = object_get(&profile, "input_contract");
        assert_eq!(as_str(object_get(input, "encoding")), "UTF-8");
        assert_eq!(as_str(object_get(input, "duplicate_object_names")), "Reject");
        assert_eq!(as_str(object_get(input, "unicode_strings")), "UnicodeScalarValuesOnly");
        let numbers = object_get(input, "numbers");
        assert_eq!(as_str(object_get(numbers, "profile")), "SafeIntegerOnly");
        assert_eq!(as_int(object_get(numbers, "minimum")), -(MAX_SAFE_INTEGER as i64));
        assert_eq!(as_int(object_get(numbers, "maximum")), MAX_SAFE_INTEGER as i64);
        assert!(!as_bool(object_get(numbers, "floating_point_allowed")));
        assert_eq!(as_int(object_get(numbers, "negative_zero_canonicalizes_to")), 0);

        let serialization = object_get(&profile, "serialization");
        assert_eq!(as_str(object_get(serialization, "object_key_order")), "RFC8785Utf16CodeUnits");
        assert!(as_bool(object_get(serialization, "object_sorting_recursive")));
        assert_eq!(as_str(object_get(serialization, "array_order")), "PreserveExactly");
        assert_eq!(as_str(object_get(serialization, "whitespace")), "None");
        assert_eq!(as_str(object_get(serialization, "string_escaping")), "RFC8785Compatible");
        assert_eq!(as_str(object_get(serialization, "unicode_normalization")), "None");
        assert_eq!(as_str(object_get(serialization, "output_encoding")), "UTF-8");

        let digest = object_get(&profile, "digest_contract");
        assert_eq!(as_str(object_get(digest, "algorithm")), "SHA-256");
        assert_eq!(as_str(object_get(digest, "preimage")), "UTF8(profile_id) || 0x00 || UTF8(domain) || 0x00 || canonical_bytes");
        assert!(as_bool(object_get(digest, "domain_must_be_nonempty")));
        assert!(as_bool(object_get(digest, "domain_must_not_contain_nul")));

        let migration = object_get(&profile, "migration");
        assert!(!as_bool(object_get(migration, "reinterpret_existing_digests")));
        assert!(as_bool(object_get(migration, "historical_python_sort_keys_digests_remain_historical")));
    }

    #[test]
    fn shared_positive_vectors_match_bytes_and_digests() {
        let corpus = parse_strict_json(GOLDEN).expect("golden corpus must parse");
        assert_eq!(as_str(object_get(&corpus, "profile_id")), PROFILE_ID);
        for vector in as_array(object_get(&corpus, "positive")) {
            let id = as_str(object_get(vector, "id"));
            let raw = as_str(object_get(vector, "raw_json"));
            let expected = as_str(object_get(vector, "canonical_utf8"));
            let domain = as_str(object_get(vector, "digest_domain"));
            let expected_digest = as_str(object_get(vector, "hak_sha256"));
            let parsed = parse_strict_json(raw).unwrap_or_else(|e| panic!("{id}: {e}"));
            assert_eq!(canonical_string(&parsed), expected, "canonical bytes differ for {id}");
            assert_eq!(hak_sha256(domain, &parsed), expected_digest, "digest differs for {id}");
        }
    }

    #[test]
    fn shared_negative_vectors_are_rejected() {
        let corpus = parse_strict_json(GOLDEN).expect("golden corpus must parse");
        for vector in as_array(object_get(&corpus, "negative")) {
            let id = as_str(object_get(vector, "id"));
            let raw = as_str(object_get(vector, "raw_json"));
            assert!(parse_strict_json(raw).is_err(), "negative vector unexpectedly accepted: {id}");
        }
    }

    #[test]
    fn invalid_utf8_is_rejected_before_json_interpretation() {
        let raw = b"{\"x\":\"\xff\"}";
        assert!(parse_strict_bytes(raw).is_err());
    }

    #[test]
    fn utf16_order_places_astral_emoji_before_fb33() {
        let value = parse_strict_json("{\"דּ\":1,\"😀\":2}").unwrap();
        assert_eq!(canonical_string(&value), "{\"😀\":2,\"דּ\":1}");
    }

    #[test]
    fn unicode_is_not_normalized() {
        let value = parse_strict_json("{\"é\":1,\"é\":2}").unwrap();
        assert_eq!(canonical_string(&value), "{\"é\":2,\"é\":1}");
    }
}
