//! Bounded CSV/TSV intake with byte-accurate cell locators.
//!
//! This adapter preserves exact source lexemes and emits CSV_CELL observations.
//! It does not infer units, resolve null sentinels, authenticate sources, or
//! establish the scientific validity of the measured values.

use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

pub const PARSER_ID: &str = "symthaea.delimited-text";
pub const PARSER_VERSION: &str = "0.1.0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delimiter {
    Comma,
    Tab,
}

impl Delimiter {
    fn byte(self) -> u8 {
        match self {
            Self::Comma => b',',
            Self::Tab => b'\t',
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Comma => "comma",
            Self::Tab => "tab",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseLimits {
    pub max_input_bytes: u64,
    pub max_elapsed_ms: u64,
    pub max_observations: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawReceipt {
    pub expected_byte_length: u64,
    pub expected_raw_sha256: String,
    pub acquisition_content_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseConfig {
    pub delimiter: Delimiter,
    pub limits: ParseLimits,
}

impl ParseConfig {
    pub fn config_sha256(&self) -> String {
        let canonical = format!(
            "parser={PARSER_ID};version={PARSER_VERSION};delimiter={};max_input_bytes={};max_elapsed_ms={};max_observations={}",
            self.delimiter.label(),
            self.limits.max_input_bytes,
            self.limits.max_elapsed_ms,
            self.limits.max_observations,
        );
        sha256(canonical.as_bytes())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Text,
    Integer,
    Decimal,
    Boolean,
    Unknown,
}

impl ValueKind {
    fn as_schema_value(&self) -> &'static str {
        match self {
            Self::Text => "TEXT",
            Self::Integer => "INTEGER",
            Self::Decimal => "DECIMAL",
            Self::Boolean => "BOOLEAN",
            Self::Unknown => "UNKNOWN",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsvCellLocator {
    /// Zero-based data-record index. Header is not counted.
    pub record_index: u64,
    pub column_index: u64,
    pub column_name: String,
    /// Inclusive zero-based byte offset into the immutable raw artifact.
    pub byte_start: u64,
    /// Exclusive zero-based byte offset into the immutable raw artifact.
    pub byte_end: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub observation_id: String,
    pub locator: CsvCellLocator,
    /// Exact UTF-8 lexeme from the source bytes, including quotes when present.
    pub raw_lexeme: String,
    /// Decoded CSV field value. It does not replace raw_lexeme.
    pub decoded_value: String,
    pub value_kind: ValueKind,
    /// v0 never infers units from adjacent headers or values.
    pub unit_lexeme: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Disposition {
    Parsed,
    Malformed,
    LimitExceeded,
    IntegrityFailed,
}

impl Disposition {
    pub fn as_schema_value(&self) -> &'static str {
        match self {
            Self::Parsed => "PARSED",
            Self::Malformed => "MALFORMED",
            Self::LimitExceeded => "LIMIT_EXCEEDED",
            Self::IntegrityFailed => "INTEGRITY_FAILED",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseIssue {
    pub reason_code: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ParseUsage {
    /// Logical input bytes admitted to parser work; never exceeds max_input_bytes.
    /// The parser receives an already captured byte slice, not a live stream.
    pub bytes_read: u64,
    pub elapsed_ms: u64,
    pub observations_emitted: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseResult {
    pub disposition: Disposition,
    pub raw_sha256: Option<String>,
    pub config_sha256: String,
    pub derived_payload_sha256: Option<String>,
    pub observations: Vec<Observation>,
    pub issues: Vec<ParseIssue>,
    pub limits: ParseLimits,
    pub usage: ParseUsage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Field {
    byte_start: usize,
    byte_end: usize,
    raw_lexeme: String,
    decoded_value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CsvError {
    Malformed(String),
    MalformedCode(&'static str, String),
    LimitExceeded(String),
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn result_with_failure(
    disposition: Disposition,
    config: &ParseConfig,
    raw_sha256: Option<String>,
    bytes_read: u64,
    elapsed_ms: u64,
    reason_code: &str,
    message: String,
) -> ParseResult {
    ParseResult {
        disposition,
        raw_sha256,
        config_sha256: config.config_sha256(),
        derived_payload_sha256: None,
        observations: Vec::new(),
        issues: vec![ParseIssue {
            reason_code: reason_code.to_string(),
            message,
        }],
        limits: config.limits.clone(),
        usage: ParseUsage {
            bytes_read,
            elapsed_ms,
            observations_emitted: 0,
        },
    }
}

/// Parse comma-separated values with strict RFC4180-style quoting and LF/CRLF rows.
pub fn parse_csv(raw: &[u8], receipt: &RawReceipt, limits: ParseLimits) -> ParseResult {
    parse_delimited(
        raw,
        receipt,
        ParseConfig {
            delimiter: Delimiter::Comma,
            limits,
        },
    )
}

/// Parse tab-separated values with the same quoting and provenance rules as CSV.
pub fn parse_tsv(raw: &[u8], receipt: &RawReceipt, limits: ParseLimits) -> ParseResult {
    parse_delimited(
        raw,
        receipt,
        ParseConfig {
            delimiter: Delimiter::Tab,
            limits,
        },
    )
}

pub fn parse_delimited(raw: &[u8], receipt: &RawReceipt, config: ParseConfig) -> ParseResult {
    let started = Instant::now();
    let byte_length = raw.len() as u64;

    // Length and resource gates run before hashing, decoding or scanning source bytes.
    if byte_length > config.limits.max_input_bytes {
        return result_with_failure(
            Disposition::LimitExceeded,
            &config,
            None,
            0,
            started.elapsed().as_millis() as u64,
            "INPUT_BYTE_LIMIT",
            format!(
                "Input has {byte_length} bytes; configured maximum is {}.",
                config.limits.max_input_bytes
            ),
        );
    }
    if config.limits.max_elapsed_ms == 0 {
        return result_with_failure(
            Disposition::LimitExceeded,
            &config,
            None,
            0,
            started.elapsed().as_millis() as u64,
            "ELAPSED_TIME_LIMIT",
            "Elapsed-time budget is zero; parsing was not started.".to_string(),
        );
    }

    let raw_hash = sha256(raw);
    if receipt.expected_byte_length != byte_length
        || !is_sha256(&receipt.expected_raw_sha256)
        || !is_sha256(&receipt.acquisition_content_sha256)
        || receipt.expected_raw_sha256 != raw_hash
        || receipt.acquisition_content_sha256 != raw_hash
    {
        return result_with_failure(
            Disposition::IntegrityFailed,
            &config,
            Some(raw_hash),
            byte_length,
            started.elapsed().as_millis() as u64,
            "RAW_RECEIPT_MISMATCH",
            "Raw artifact length, artifact digest, or acquisition receipt digest does not match the supplied bytes.".to_string(),
        );
    }

    if std::str::from_utf8(raw).is_err() {
        return result_with_failure(
            Disposition::Malformed,
            &config,
            Some(raw_hash),
            byte_length,
            started.elapsed().as_millis() as u64,
            "INVALID_UTF8",
            "Delimited-text v0 accepts strict UTF-8 only.".to_string(),
        );
    }

    let observations = match parse_observations(raw, &config, started) {
        Ok(observations) => observations,
        Err(CsvError::Malformed(message)) => {
            return result_with_failure(
                Disposition::Malformed,
                &config,
                Some(raw_hash),
                byte_length,
                started.elapsed().as_millis() as u64,
                "MALFORMED_DELIMITED_TEXT",
                message,
            );
        }
        Err(CsvError::MalformedCode(reason_code, message)) => {
            return result_with_failure(
                Disposition::Malformed,
                &config,
                Some(raw_hash),
                byte_length,
                started.elapsed().as_millis() as u64,
                reason_code,
                message,
            );
        }
        Err(CsvError::LimitExceeded(message)) => {
            return result_with_failure(
                Disposition::LimitExceeded,
                &config,
                Some(raw_hash),
                byte_length,
                started.elapsed().as_millis() as u64,
                "PARSER_RESOURCE_LIMIT",
                message,
            );
        }
    };

    let elapsed_ms = started.elapsed().as_millis() as u64;
    if started.elapsed() > Duration::from_millis(config.limits.max_elapsed_ms) {
        return result_with_failure(
            Disposition::LimitExceeded,
            &config,
            Some(raw_hash),
            byte_length,
            elapsed_ms,
            "ELAPSED_TIME_LIMIT",
            format!(
                "Parsing took {elapsed_ms} ms; configured maximum is {} ms.",
                config.limits.max_elapsed_ms
            ),
        );
    }

    let payload_sha256 = derived_payload_sha256(&observations);
    ParseResult {
        disposition: Disposition::Parsed,
        raw_sha256: Some(raw_hash),
        config_sha256: config.config_sha256(),
        derived_payload_sha256: Some(payload_sha256),
        usage: ParseUsage {
            bytes_read: byte_length,
            elapsed_ms,
            observations_emitted: observations.len() as u64,
        },
        limits: config.limits,
        observations,
        issues: Vec::new(),
    }
}

fn parse_observations(
    raw: &[u8],
    config: &ParseConfig,
    started: Instant,
) -> Result<Vec<Observation>, CsvError> {
    let delimiter = config.delimiter.byte();
    let header_limit = usize::try_from(config.limits.max_observations.max(1)).unwrap_or(usize::MAX);
    let Some((header, mut cursor)) = parse_one_record(raw, 0, delimiter, header_limit)? else {
        return Err(CsvError::MalformedCode(
            "MISSING_HEADER",
            "Delimited-text artifact is empty and has no header record.".to_string(),
        ));
    };

    if header.is_empty() || header.iter().any(|field| field.decoded_value.trim().is_empty()) {
        return Err(CsvError::MalformedCode(
            "INVALID_HEADER",
            "Header names must be non-empty.".to_string(),
        ));
    }

    let mut seen_headers = std::collections::BTreeSet::new();
    for field in &header {
        if !seen_headers.insert(field.decoded_value.clone()) {
            return Err(CsvError::MalformedCode(
                "DUPLICATE_HEADER",
                format!("Duplicate header name {:?}.", field.decoded_value),
            ));
        }
    }

    let mut observations = Vec::new();
    let mut record_index = 0u64;
    while cursor < raw.len() {
        if started.elapsed() > Duration::from_millis(config.limits.max_elapsed_ms) {
            return Err(CsvError::LimitExceeded(
                "Elapsed-time limit reached while building cell observations.".to_string(),
            ));
        }

        let remaining = config
            .limits
            .max_observations
            .saturating_sub(observations.len() as u64);
        let field_limit = usize::try_from(remaining).unwrap_or(usize::MAX);
        let Some((fields, next_cursor)) = parse_one_record(raw, cursor, delimiter, field_limit)? else {
            break;
        };
        if next_cursor <= cursor {
            return Err(CsvError::Malformed("Parser made no forward progress.".to_string()));
        }
        if fields.len() != header.len() {
            return Err(CsvError::MalformedCode(
                "ROW_WIDTH_MISMATCH",
                format!(
                    "Data record {record_index} has {} fields; header has {}.",
                    fields.len(),
                    header.len()
                ),
            ));
        }

        for (column_index, field) in fields.iter().enumerate() {
            let column_name = header[column_index].decoded_value.clone();
            observations.push(Observation {
                observation_id: format!("csv-r{record_index}-c{column_index}"),
                locator: CsvCellLocator {
                    record_index,
                    column_index: column_index as u64,
                    column_name,
                    byte_start: field.byte_start as u64,
                    byte_end: field.byte_end as u64,
                },
                raw_lexeme: field.raw_lexeme.clone(),
                decoded_value: field.decoded_value.clone(),
                value_kind: classify_value(&field.decoded_value),
                unit_lexeme: None,
            });
        }

        cursor = next_cursor;
        record_index += 1;
    }

    Ok(observations)
}

fn parse_one_record(
    raw: &[u8],
    start_cursor: usize,
    delimiter: u8,
    max_fields: usize,
) -> Result<Option<(Vec<Field>, usize)>, CsvError> {
    if start_cursor >= raw.len() {
        return Ok(None);
    }

    let mut fields = Vec::new();
    let mut field_start = start_cursor;
    let mut index = start_cursor;
    let mut in_quotes = false;
    let mut quoted_field = false;
    let mut quote_closed = false;

    while index < raw.len() {
        let byte = raw[index];
        if in_quotes {
            if byte == b'"' {
                if index + 1 < raw.len() && raw[index + 1] == b'"' {
                    index += 2;
                    continue;
                }
                in_quotes = false;
                quote_closed = true;
                index += 1;
                if index < raw.len()
                    && raw[index] != delimiter
                    && raw[index] != b'\n'
                    && raw[index] != b'\r'
                {
                    return Err(CsvError::Malformed(
                        "Unexpected bytes after the closing quote of a field.".to_string(),
                    ));
                }
                continue;
            }
            index += 1;
            continue;
        }

        if quote_closed && byte != delimiter && byte != b'\n' && byte != b'\r' {
            return Err(CsvError::Malformed(
                "Unexpected bytes after the closing quote of a field.".to_string(),
            ));
        }

        match byte {
            b'"' if index == field_start && !quoted_field => {
                in_quotes = true;
                quoted_field = true;
                index += 1;
            }
            b'"' => {
                return Err(CsvError::Malformed(
                    "Quote found inside an unquoted field.".to_string(),
                ));
            }
            value if value == delimiter => {
                push_field(&mut fields, raw, field_start, index, quoted_field, max_fields)?;
                index += 1;
                field_start = index;
                quoted_field = false;
                quote_closed = false;
            }
            b'\n' => {
                push_field(&mut fields, raw, field_start, index, quoted_field, max_fields)?;
                return Ok(Some((fields, index + 1)));
            }
            b'\r' => {
                if index + 1 >= raw.len() || raw[index + 1] != b'\n' {
                    return Err(CsvError::Malformed(
                        "Bare carriage return is not an accepted record separator.".to_string(),
                    ));
                }
                push_field(&mut fields, raw, field_start, index, quoted_field, max_fields)?;
                return Ok(Some((fields, index + 2)));
            }
            _ => index += 1,
        }
    }

    if in_quotes {
        return Err(CsvError::Malformed("Unterminated quoted field.".to_string()));
    }

    push_field(&mut fields, raw, field_start, raw.len(), quoted_field, max_fields)?;
    Ok(Some((fields, raw.len())))
}

fn push_field(
    fields: &mut Vec<Field>,
    raw: &[u8],
    start: usize,
    end: usize,
    quoted: bool,
    max_fields: usize,
) -> Result<(), CsvError> {
    if fields.len() >= max_fields {
        return Err(CsvError::LimitExceeded(
            "Maximum cell-observation count would be exceeded.".to_string(),
        ));
    }
    fields.push(build_field(raw, start, end, quoted)?);
    Ok(())
}

fn build_field(
    raw: &[u8],
    start: usize,
    end: usize,
    quoted: bool,
) -> Result<Field, CsvError> {
    let bytes = raw.get(start..end).ok_or_else(|| {
        CsvError::Malformed("Field byte span is outside the raw artifact.".to_string())
    })?;
    let raw_lexeme = std::str::from_utf8(bytes)
        .map_err(|_| CsvError::Malformed("Field is not valid UTF-8.".to_string()))?
        .to_string();

    let decoded_value = if quoted {
        if bytes.len() < 2 || bytes.first() != Some(&b'"') || bytes.last() != Some(&b'"') {
            return Err(CsvError::Malformed("Quoted field has invalid delimiters.".to_string()));
        }
        let inner = &bytes[1..bytes.len() - 1];
        let mut decoded = Vec::with_capacity(inner.len());
        let mut index = 0usize;
        while index < inner.len() {
            if inner[index] == b'"' {
                if index + 1 >= inner.len() || inner[index + 1] != b'"' {
                    return Err(CsvError::Malformed("Unescaped quote inside quoted field.".to_string()));
                }
                decoded.push(b'"');
                index += 2;
            } else {
                decoded.push(inner[index]);
                index += 1;
            }
        }
        String::from_utf8(decoded)
            .map_err(|_| CsvError::Malformed("Decoded field is not valid UTF-8.".to_string()))?
    } else {
        raw_lexeme.clone()
    };

    Ok(Field {
        byte_start: start,
        byte_end: end,
        raw_lexeme,
        decoded_value,
    })
}

fn classify_value(value: &str) -> ValueKind {
    if value.is_empty() {
        return ValueKind::Unknown;
    }
    if value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("false") {
        return ValueKind::Boolean;
    }
    if is_numeric_lexeme(value) {
        if value.contains('.') || value.contains('e') || value.contains('E') {
            return ValueKind::Decimal;
        }
        return ValueKind::Integer;
    }
    ValueKind::Text
}

fn is_numeric_lexeme(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut index = 0usize;
    if bytes[index] == b'+' || bytes[index] == b'-' {
        index += 1;
        if index == bytes.len() {
            return false;
        }
    }

    let mut integer_digits = 0usize;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        integer_digits += 1;
        index += 1;
    }

    let mut fractional_digits = 0usize;
    if index < bytes.len() && bytes[index] == b'.' {
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            fractional_digits += 1;
            index += 1;
        }
        if fractional_digits == 0 {
            return false;
        }
    }
    if integer_digits == 0 {
        return false;
    }

    if index < bytes.len() && (bytes[index] == b'e' || bytes[index] == b'E') {
        index += 1;
        if index < bytes.len() && (bytes[index] == b'+' || bytes[index] == b'-') {
            index += 1;
        }
        let exponent_start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if exponent_start == index {
            return false;
        }
    }
    index == bytes.len()
}

fn derived_payload_sha256(observations: &[Observation]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-delimited-observation-payload-v0.1.0\0");
    for observation in observations {
        hash_string(&mut hasher, &observation.observation_id);
        hasher.update(observation.locator.record_index.to_le_bytes());
        hasher.update(observation.locator.column_index.to_le_bytes());
        hash_string(&mut hasher, &observation.locator.column_name);
        hasher.update(observation.locator.byte_start.to_le_bytes());
        hasher.update(observation.locator.byte_end.to_le_bytes());
        hash_string(&mut hasher, &observation.raw_lexeme);
        hash_string(&mut hasher, &observation.decoded_value);
        hash_string(&mut hasher, observation.value_kind.as_schema_value());
        match &observation.unit_lexeme {
            Some(unit) => {
                hasher.update([1]);
                hash_string(&mut hasher, unit);
            }
            None => hasher.update([0]),
        }
    }
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> ParseLimits {
        ParseLimits {
            max_input_bytes: 4096,
            max_elapsed_ms: 2_000,
            max_observations: 64,
        }
    }

    fn receipt(raw: &[u8]) -> RawReceipt {
        let hash = sha256(raw);
        RawReceipt {
            expected_byte_length: raw.len() as u64,
            expected_raw_sha256: hash.clone(),
            acquisition_content_sha256: hash,
        }
    }

    #[test]
    fn exact_csv_cell_locator_matches_v0_fixture() {
        let raw = b"sample_id,temperature_C\nS001,21.50\n";
        let result = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(result.disposition, Disposition::Parsed);
        assert_eq!(result.observations.len(), 2);
        let temperature = &result.observations[1];
        assert_eq!(temperature.locator.record_index, 0);
        assert_eq!(temperature.locator.column_index, 1);
        assert_eq!(temperature.locator.column_name, "temperature_C");
        assert_eq!(temperature.locator.byte_start, 29);
        assert_eq!(temperature.locator.byte_end, 34);
        assert_eq!(temperature.raw_lexeme, "21.50");
        assert_eq!(temperature.decoded_value, "21.50");
        assert_eq!(temperature.value_kind, ValueKind::Decimal);
        assert_eq!(temperature.unit_lexeme, None);
    }

    #[test]
    fn preserves_quoted_commas_and_exact_lexeme() {
        let raw = b"id,description\n1,\"cold, filtered\"\n";
        let result = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(result.disposition, Disposition::Parsed);
        let cell = &result.observations[1];
        assert_eq!(cell.raw_lexeme, "\"cold, filtered\"");
        assert_eq!(cell.decoded_value, "cold, filtered");
        assert_eq!(cell.value_kind, ValueKind::Text);
    }

    #[test]
    fn handles_escaped_quotes_and_embedded_newlines() {
        let raw = b"id,note\n1,\"line one\nsaid \"\"yes\"\"\"\n";
        let result = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(result.disposition, Disposition::Parsed);
        let cell = &result.observations[1];
        assert_eq!(cell.decoded_value, "line one\nsaid \"yes\"");
        assert!(cell.raw_lexeme.contains("\"\"yes\"\""));
        assert!(cell.locator.byte_end > cell.locator.byte_start);
    }

    #[test]
    fn accepts_crlf_and_tsv() {
        let raw = b"sample\tvalue\r\nS1\t-2.50e-2\r\n";
        let result = parse_tsv(raw, &receipt(raw), limits());
        assert_eq!(result.disposition, Disposition::Parsed);
        assert_eq!(result.observations[1].decoded_value, "-2.50e-2");
        assert_eq!(result.observations[1].value_kind, ValueKind::Decimal);
    }

    #[test]
    fn duplicate_headers_are_malformed() {
        let raw = b"id,value,value\n1,2,3\n";
        let result = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(result.disposition, Disposition::Malformed);
        assert!(result.observations.is_empty());
        assert_eq!(result.issues[0].reason_code, "DUPLICATE_HEADER");
        assert_eq!(result.derived_payload_sha256, None);
    }

    #[test]
    fn row_width_mismatch_is_malformed_without_partial_observations() {
        let raw = b"id,value\n1\n";
        let result = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(result.disposition, Disposition::Malformed);
        assert!(result.observations.is_empty());
        assert_eq!(result.issues[0].reason_code, "ROW_WIDTH_MISMATCH");
    }

    #[test]
    fn invalid_utf8_is_malformed() {
        let raw = b"id,value\n1,\xff\n";
        let result = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(result.disposition, Disposition::Malformed);
        assert_eq!(result.issues[0].reason_code, "INVALID_UTF8");
        assert!(result.observations.is_empty());
    }

    #[test]
    fn quote_errors_fail_closed() {
        let raw = b"id,value\n1,\"unterminated\n";
        let result = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(result.disposition, Disposition::Malformed);
        assert_eq!(result.issues[0].reason_code, "MALFORMED_DELIMITED_TEXT");
        assert!(result.observations.is_empty());
    }

    #[test]
    fn integrity_mismatch_emits_no_observations() {
        let raw = b"id,value\n1,2\n";
        let mut wrong = receipt(raw);
        wrong.acquisition_content_sha256 = "0".repeat(64);
        let result = parse_csv(raw, &wrong, limits());
        assert_eq!(result.disposition, Disposition::IntegrityFailed);
        assert!(result.observations.is_empty());
        assert_eq!(result.derived_payload_sha256, None);
    }

    #[test]
    fn input_size_limit_is_checked_before_hash_or_parse() {
        let raw = b"id,value\n1,2\n";
        let mut configured = limits();
        configured.max_input_bytes = 2;
        let result = parse_csv(raw, &receipt(raw), configured);
        assert_eq!(result.disposition, Disposition::LimitExceeded);
        assert_eq!(result.usage.bytes_read, 0);
        assert!(result.observations.is_empty());
    }

    #[test]
    fn observation_limit_rejects_the_entire_payload() {
        let raw = b"id,value\n1,2\n";
        let mut configured = limits();
        configured.max_observations = 1;
        let result = parse_csv(raw, &receipt(raw), configured);
        assert_eq!(result.disposition, Disposition::LimitExceeded);
        assert!(result.observations.is_empty());
        assert_eq!(result.usage.observations_emitted, 0);
    }

    #[test]
    fn crlf_rows_also_obey_observation_limit() {
        let raw = b"id,value\r\n1,2\r\n";
        let mut configured = limits();
        configured.max_observations = 1;
        let result = parse_csv(raw, &receipt(raw), configured);
        assert_eq!(result.disposition, Disposition::LimitExceeded);
        assert!(result.observations.is_empty());
        assert_eq!(result.usage.observations_emitted, 0);
    }

    #[test]
    fn zero_elapsed_budget_rejects_before_scan() {
        let raw = b"id,value\n1,2\n";
        let mut configured = limits();
        configured.max_elapsed_ms = 0;
        let result = parse_csv(raw, &receipt(raw), configured);
        assert_eq!(result.disposition, Disposition::LimitExceeded);
        assert_eq!(result.usage.bytes_read, 0);
    }

    #[test]
    fn empty_cells_are_unknown_not_implicitly_null() {
        let raw = b"id,value\n1,\n";
        let result = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(result.disposition, Disposition::Parsed);
        assert_eq!(result.observations[1].raw_lexeme, "");
        assert_eq!(result.observations[1].value_kind, ValueKind::Unknown);
    }

    #[test]
    fn null_like_sentinels_are_not_coerced() {
        let raw = b"id,value\n1,NA\n2,null\n";
        let result = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(result.disposition, Disposition::Parsed);
        assert_eq!(result.observations[1].value_kind, ValueKind::Text);
        assert_eq!(result.observations[3].value_kind, ValueKind::Text);
    }

    #[test]
    fn derived_payload_digest_is_deterministic() {
        let raw = b"id,value\n1,21.50\n";
        let first = parse_csv(raw, &receipt(raw), limits());
        let second = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(first.derived_payload_sha256, second.derived_payload_sha256);
        assert_eq!(first.observations, second.observations);
    }

    #[test]
    fn schema_payload_usage_matches_emitted_observations() {
        let raw = b"id,value\n1,2\n";
        let result = parse_csv(raw, &receipt(raw), limits());
        assert_eq!(result.usage.bytes_read, raw.len() as u64);
        assert_eq!(result.usage.observations_emitted, result.observations.len() as u64);
        assert!(result.usage.bytes_read <= result.limits.max_input_bytes);
        assert!(result.usage.observations_emitted <= result.limits.max_observations);
        assert!(result.derived_payload_sha256.as_ref().is_some_and(|hash| is_sha256(hash)));
    }
}
