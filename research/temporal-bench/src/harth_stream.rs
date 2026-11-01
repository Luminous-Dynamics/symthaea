//! Strict streaming ingestion for the UCI HARTH accelerometer dataset.
//!
//! This parser preserves participant/source-family identity, all six raw
//! channels, labels, timestamp order, and sampling gaps before windowing.
//! It does not authenticate the source archive; bind its byte digest through
//! real_dataset_manifest and keep participant files in disjoint split roles.

use std::error::Error;
use std::fmt;
use std::io::BufRead;

use super::digest::CanonicalHasher;

const NOMINAL_SAMPLE_PERIOD_NS: i128 = 20_000_000;
const GAP_THRESHOLD_NS: i128 = NOMINAL_SAMPLE_PERIOD_NS * 3 / 2;

const EXPECTED_HEADERS: [&str; 8] = [
    "timestamp",
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
    "label",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthCsvError {
    pub line: u64,
    pub field: String,
    pub message: String,
}

impl HarthCsvError {
    fn new(line: u64, field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { line, field: field.into(), message: message.into() }
    }
}

impl fmt::Display for HarthCsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {} field {}: {}", self.line, self.field, self.message)
    }
}
impl Error for HarthCsvError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum HarthActivity {
    Walking = 1,
    Running = 2,
    Shuffling = 3,
    StairsAscending = 4,
    StairsDescending = 5,
    Standing = 6,
    Sitting = 7,
    Lying = 8,
    CyclingSitting = 13,
    CyclingStanding = 14,
    CyclingSittingInactive = 130,
    CyclingStandingInactive = 140,
}

impl HarthActivity {
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::Walking),
            2 => Some(Self::Running),
            3 => Some(Self::Shuffling),
            4 => Some(Self::StairsAscending),
            5 => Some(Self::StairsDescending),
            6 => Some(Self::Standing),
            7 => Some(Self::Sitting),
            8 => Some(Self::Lying),
            13 => Some(Self::CyclingSitting),
            14 => Some(Self::CyclingStanding),
            130 => Some(Self::CyclingSittingInactive),
            140 => Some(Self::CyclingStandingInactive),
            _ => None,
        }
    }

    pub const fn code(self) -> u16 {
        self as u16
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarthRow {
    /// Naive calendar nanoseconds used only for within-record ordering and gaps.
    /// No UTC offset is inferred from the source timestamp.
    pub timestamp_ns: i128,
    /// back_x, back_y, back_z, thigh_x, thigh_y, thigh_z in g.
    pub acceleration_g: [f32; 6],
    pub activity: HarthActivity,
    /// Zero-based data-row index, excluding the header and optional export index.
    pub row_index: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthStreamSummary {
    pub source_family_sha256: [u8; 32],
    pub row_count: u64,
    pub first_timestamp_ns: i128,
    pub last_timestamp_ns: i128,
    pub minimum_delta_ns: Option<i128>,
    pub maximum_delta_ns: Option<i128>,
    pub gap_count: u64,
    pub maximum_gap_ns: Option<i128>,
    pub has_export_index_column: bool,
    /// Digest of normalized parsed rows, not the original CSV bytes.
    pub normalized_rows_sha256: [u8; 32],
}

fn parse_csv_record(line: &str, line_number: u64) -> Result<Vec<String>, HarthCsvError> {
    let content = line.trim_end_matches(|character| character == '\r' || character == '\n');
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut characters = content.chars().peekable();
    let mut in_quotes = false;
    let mut quote_closed = false;

    while let Some(character) = characters.next() {
        if in_quotes {
            if character == '"' {
                if characters.peek() == Some(&'"') {
                    characters.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                    quote_closed = true;
                }
            } else {
                field.push(character);
            }
            continue;
        }

        if quote_closed {
            match character {
                ',' => {
                    fields.push(std::mem::take(&mut field));
                    quote_closed = false;
                }
                value if value.is_ascii_whitespace() => {}
                _ => {
                    return Err(HarthCsvError::new(
                        line_number,
                        "csv",
                        "unexpected data after a closing quote",
                    ));
                }
            }
            continue;
        }

        match character {
            ',' => fields.push(std::mem::take(&mut field)),
            '"' if field.is_empty() => in_quotes = true,
            '"' => {
                return Err(HarthCsvError::new(
                    line_number,
                    "csv",
                    "quote is only valid at the start of a field",
                ));
            }
            value => field.push(value),
        }
    }

    if in_quotes {
        return Err(HarthCsvError::new(
            line_number,
            "csv",
            "unterminated quoted field; multiline fields are not supported",
        ));
    }
    fields.push(field);
    Ok(fields)
}

fn normalize_header(header: &str) -> String {
    header.trim().to_ascii_lowercase()
}

fn is_optional_export_index_header(header: &str) -> bool {
    matches!(
        normalize_header(header).as_str(),
        "" | "index" | "unnamed: 0"
    )
}

fn parse_digits(bytes: &[u8], start: usize, end: usize) -> Option<u32> {
    let slice = bytes.get(start..end)?;
    if slice.is_empty() || !slice.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut value = 0_u32;
    for digit in slice {
        value = value.checked_mul(10)?.checked_add(u32::from(*digit - b'0'))?;
    }
    Some(value)
}

fn is_leap_year(year: u32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: u32, month: u32) -> Option<u32> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if is_leap_year(year) { 29 } else { 28 }),
        _ => None,
    }
}

/// Parse the timezone-naive HARTH timestamp forms YYYY-MM-DD HH:MM:SS[.fffffffff]
/// and YYYY-MM-DDTHH:MM:SS[.fffffffff]. Offsets are deliberately rejected.
fn parse_timestamp_ns(value: &str, line: u64) -> Result<i128, HarthCsvError> {
    let bytes = value.as_bytes();
    if bytes.len() < 19
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || !matches!(bytes.get(10), Some(&b' ') | Some(&b'T'))
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
    {
        return Err(HarthCsvError::new(
            line,
            "timestamp",
            "expected YYYY-MM-DD HH:MM:SS with optional fractional seconds",
        ));
    }

    let year = parse_digits(bytes, 0, 4)
        .ok_or_else(|| HarthCsvError::new(line, "timestamp", "invalid year"))?;
    let month = parse_digits(bytes, 5, 7)
        .ok_or_else(|| HarthCsvError::new(line, "timestamp", "invalid month"))?;
    let day = parse_digits(bytes, 8, 10)
        .ok_or_else(|| HarthCsvError::new(line, "timestamp", "invalid day"))?;
    let hour = parse_digits(bytes, 11, 13)
        .ok_or_else(|| HarthCsvError::new(line, "timestamp", "invalid hour"))?;
    let minute = parse_digits(bytes, 14, 16)
        .ok_or_else(|| HarthCsvError::new(line, "timestamp", "invalid minute"))?;
    let second = parse_digits(bytes, 17, 19)
        .ok_or_else(|| HarthCsvError::new(line, "timestamp", "invalid second"))?;

    if !(1900..=9999).contains(&year)
        || days_in_month(year, month).is_none_or(|maximum| day == 0 || day > maximum)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(HarthCsvError::new(
            line,
            "timestamp",
            "calendar or clock component is out of range",
        ));
    }

    let mut fractional_ns = 0_u32;
    if bytes.len() > 19 {
        if bytes[19] != b'.' || bytes.len() > 29 {
            return Err(HarthCsvError::new(
                line,
                "timestamp",
                "timezone suffixes and fractional precision above 9 digits are unsupported",
            ));
        }
        let fraction = &bytes[20..];
        if fraction.is_empty() || !fraction.iter().all(u8::is_ascii_digit) {
            return Err(HarthCsvError::new(
                line,
                "timestamp",
                "fractional seconds must contain 1 to 9 digits",
            ));
        }
        let parsed = parse_digits(bytes, 20, bytes.len())
            .ok_or_else(|| HarthCsvError::new(line, "timestamp", "invalid fractional seconds"))?;
        fractional_ns = parsed;
        for _ in fraction.len()..9 {
            fractional_ns = fractional_ns.saturating_mul(10);
        }
    }

    // Gregorian civil-date conversion. The time key is intentionally naive:
    // HARTH provides date/time text but this parser does not invent a timezone.
    let year_i64 = i64::from(year);
    let month_i64 = i64::from(month);
    let adjusted_year = year_i64 - i64::from(month <= 2);
    let era = if adjusted_year >= 0 {
        adjusted_year
    } else {
        adjusted_year - 399
    } / 400;
    let year_of_era = adjusted_year - era * 400;
    let adjusted_month = month_i64 + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days_since_epoch = era * 146_097 + day_of_era - 719_468;

    let seconds_since_epoch = i128::from(days_since_epoch) * 86_400
        + i128::from(hour) * 3_600
        + i128::from(minute) * 60
        + i128::from(second);
    Ok(seconds_since_epoch * 1_000_000_000 + i128::from(fractional_ns))
}

fn parse_float(value: &str, line: u64, field: &str) -> Result<f32, HarthCsvError> {
    let parsed = value.trim().parse::<f32>().map_err(|_| {
        HarthCsvError::new(line, field, "expected a finite decimal acceleration value")
    })?;
    if !parsed.is_finite() {
        return Err(HarthCsvError::new(
            line,
            field,
            "NaN and infinity are not valid sensor observations",
        ));
    }
    Ok(parsed)
}

fn parse_activity(value: &str, line: u64) -> Result<HarthActivity, HarthCsvError> {
    let code = value.trim().parse::<u16>().map_err(|_| {
        HarthCsvError::new(line, "label", "expected an unsigned integer HARTH activity code")
    })?;
    HarthActivity::from_code(code).ok_or_else(|| {
        HarthCsvError::new(line, "label", format!("unknown HARTH activity code {code}"))
    })
}

fn add_row_to_digest(
    hasher: &mut CanonicalHasher,
    row: &HarthRow,
) {
    hasher.bytes(&row.timestamp_ns.to_le_bytes());
    for channel in row.acceleration_g {
        hasher.u32(channel.to_bits());
    }
    hasher.u16(row.activity.code());
}

/// Stream one HARTH CSV recording with O(one input record) parser memory.
///
/// source_family_sha256 must be the pinned subject/recording identity from the
/// real-data manifest. The callback receives each parsed row in source order.
pub fn parse_harth_csv<R, F>(
    mut reader: R,
    source_family_sha256: [u8; 32],
    mut consume: F,
) -> Result<HarthStreamSummary, HarthCsvError>
where
    R: BufRead,
    F: FnMut(HarthRow) -> Result<(), String>,
{
    if source_family_sha256 == [0; 32] {
        return Err(HarthCsvError::new(
            0,
            "source_family_sha256",
            "source-family identity must be nonzero and pinned before parsing",
        ));
    }

    let mut line_buffer = String::new();
    let mut line_number = 1_u64;
    if reader.read_line(&mut line_buffer).map_err(|error| {
        HarthCsvError::new(line_number, "io", error.to_string())
    })? == 0
    {
        return Err(HarthCsvError::new(1, "header", "recording is empty"));
    }

    let mut headers = parse_csv_record(&line_buffer, line_number)?;
    if let Some(first) = headers.first_mut() {
        if first.starts_with('\u{feff}') {
            *first = first.trim_start_matches('\u{feff}').to_owned();
        }
    }
    let has_export_index_column = headers.len() == 9
        && headers.first().is_some_and(|header| is_optional_export_index_header(header));
    let column_offset = usize::from(has_export_index_column);
    let expected_len = EXPECTED_HEADERS.len() + column_offset;
    if headers.len() != expected_len {
        return Err(HarthCsvError::new(
            1,
            "header",
            format!(
                "expected 8 canonical columns or 9 with one recognized leading export index; got {}",
                headers.len()
            ),
        ));
    }
    for (column, expected) in EXPECTED_HEADERS.iter().enumerate() {
        let actual = normalize_header(&headers[column + column_offset]);
        if actual != *expected {
            return Err(HarthCsvError::new(
                1,
                "header",
                format!("column {} must be {expected}, got {actual}", column + 1 + column_offset),
            ));
        }
    }

    let mut expected_export_index: Option<u64> = None;
    let mut first_timestamp_ns = None;
    let mut last_timestamp_ns = None;
    let mut minimum_delta_ns: Option<i128> = None;
    let mut maximum_delta_ns: Option<i128> = None;
    let mut gap_count = 0_u64;
    let mut maximum_gap_ns: Option<i128> = None;
    let mut row_count = 0_u64;
    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.harth-normalized-rows.v1");
    hasher.bytes(&source_family_sha256);

    loop {
        line_buffer.clear();
        let bytes_read = reader.read_line(&mut line_buffer).map_err(|error| {
            HarthCsvError::new(line_number.saturating_add(1), "io", error.to_string())
        })?;
        if bytes_read == 0 {
            break;
        }
        line_number = line_number.saturating_add(1);
        let fields = parse_csv_record(&line_buffer, line_number)?;
        if fields.len() != expected_len {
            return Err(HarthCsvError::new(
                line_number,
                "csv",
                format!("expected {expected_len} fields, got {}", fields.len()),
            ));
        }

        if has_export_index_column {
            let index = fields[0].trim().parse::<u64>().map_err(|_| {
                HarthCsvError::new(line_number, "export_index", "expected a sequential unsigned integer")
            })?;
            match expected_export_index {
                None if index == 0 || index == 1 => {}
                None => {
                    return Err(HarthCsvError::new(
                        line_number,
                        "export_index",
                        "first export index must be 0 or 1",
                    ));
                }
                Some(expected) if index == expected => {}
                Some(expected) => {
                    return Err(HarthCsvError::new(
                        line_number,
                        "export_index",
                        format!("expected sequential index {expected}, got {index}"),
                    ));
                }
            }
            expected_export_index = Some(index.checked_add(1).ok_or_else(|| {
                HarthCsvError::new(line_number, "export_index", "index overflow")
            })?);
        }

        let timestamp_ns = parse_timestamp_ns(
            fields[0 + column_offset].trim(),
            line_number,
        )?;
        let channel_names = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"];
        let mut acceleration_g = [0.0_f32; 6];
        for index in 0..6 {
            acceleration_g[index] = parse_float(
                &fields[index + 1 + column_offset],
                line_number,
                channel_names[index],
            )?;
        }
        let activity = parse_activity(&fields[7 + column_offset], line_number)?;

        if let Some(previous) = last_timestamp_ns {
            let delta = timestamp_ns.checked_sub(previous).ok_or_else(|| {
                HarthCsvError::new(line_number, "timestamp", "timestamp arithmetic overflow")
            })?;
            if delta <= 0 {
                return Err(HarthCsvError::new(
                    line_number,
                    "timestamp",
                    "timestamps must be strictly increasing within each source-family file",
                ));
            }
            minimum_delta_ns = Some(minimum_delta_ns.map_or(delta, |minimum| minimum.min(delta)));
            maximum_delta_ns = Some(maximum_delta_ns.map_or(delta, |maximum| maximum.max(delta)));
            if delta > GAP_THRESHOLD_NS {
                gap_count = gap_count.checked_add(1).ok_or_else(|| {
                    HarthCsvError::new(line_number, "timestamp", "gap counter overflow")
                })?;
                maximum_gap_ns = Some(maximum_gap_ns.map_or(delta, |maximum| maximum.max(delta)));
            }
        } else {
            first_timestamp_ns = Some(timestamp_ns);
        }

        let row = HarthRow {
            timestamp_ns,
            acceleration_g,
            activity,
            row_index: row_count,
        };
        add_row_to_digest(&mut hasher, &row);
        consume(row).map_err(|message| HarthCsvError::new(line_number, "consumer", message))?;
        row_count = row_count.checked_add(1).ok_or_else(|| {
            HarthCsvError::new(line_number, "row_count", "row counter overflow")
        })?;
        last_timestamp_ns = Some(timestamp_ns);
    }

    if row_count == 0 {
        return Err(HarthCsvError::new(1, "data", "recording contains a header but no rows"));
    }

    Ok(HarthStreamSummary {
        source_family_sha256,
        row_count,
        first_timestamp_ns: first_timestamp_ns.ok_or_else(|| {
            HarthCsvError::new(1, "timestamp", "missing first timestamp after non-empty row validation")
        })?,
        last_timestamp_ns: last_timestamp_ns.ok_or_else(|| {
            HarthCsvError::new(1, "timestamp", "missing last timestamp after non-empty row validation")
        })?,
        minimum_delta_ns,
        maximum_delta_ns,
        gap_count,
        maximum_gap_ns,
        has_export_index_column,
        normalized_rows_sha256: hasher.finish(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const FAMILY: [u8; 32] = [7; 32];

    fn parse(input: &str) -> Result<(HarthStreamSummary, Vec<HarthRow>), HarthCsvError> {
        let mut rows = Vec::new();
        let summary = parse_harth_csv(Cursor::new(input.as_bytes()), FAMILY, |row| {
            rows.push(row);
            Ok(())
        })?;
        Ok((summary, rows))
    }

    fn header() -> &'static str {
        "timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z,label\n"
    }

    fn row(timestamp: &str, label: &str) -> String {
        format!("{timestamp},0.1,-0.2,1.0,0.3,-0.4,0.9,{label}\n")
    }

    #[test]
    fn canonical_harth_record_streams_all_channels_and_activity() {
        let input = format!(
            "{}{}{}",
            header(),
            row("2024-02-29 12:00:00.000000", "1"),
            row("2024-02-29 12:00:00.020000", "140")
        );
        let (summary, rows) = parse(&input).unwrap();
        assert_eq!(summary.row_count, 2);
        assert_eq!(summary.minimum_delta_ns, Some(20_000_000));
        assert_eq!(summary.maximum_delta_ns, Some(20_000_000));
        assert_eq!(summary.gap_count, 0);
        assert_eq!(rows[0].acceleration_g, [0.1, -0.2, 1.0, 0.3, -0.4, 0.9]);
        assert_eq!(rows[1].activity, HarthActivity::CyclingStandingInactive);
        assert_eq!(rows[0].row_index, 0);
    }

    #[test]
    fn optional_leading_export_index_is_checked_and_not_treated_as_a_sensor() {
        let input = format!(
            "Unnamed: 0,{}0,{}1,{}",
            header(),
            row("2024-01-01 00:00:00.000000", "6"),
            row("2024-01-01 00:00:00.020000", "7")
        );
        let (summary, rows) = parse(&input).unwrap();
        assert!(summary.has_export_index_column);
        assert_eq!(summary.row_count, 2);
        assert_eq!(rows[0].acceleration_g[0], 0.1);
    }

    #[test]
    fn csv_quotes_and_crlf_are_supported_for_timestamp_field() {
        let input = concat!(
            "\u{feff}timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z,label\r\n",
            "\"2024-01-01 00:00:00.000000\",0,0,1,0,0,1,6\r\n",
            "\"2024-01-01 00:00:00.020000\",0,0,1,0,0,1,7\r\n"
        );
        let (summary, rows) = parse(input).unwrap();
        assert_eq!(summary.row_count, 2);
        assert_eq!(rows[1].activity, HarthActivity::Sitting);
    }

    #[test]
    fn timestamps_must_be_strictly_increasing() {
        let input = format!(
            "{}{}{}",
            header(),
            row("2024-01-01 00:00:00.020000", "1"),
            row("2024-01-01 00:00:00.020000", "1")
        );
        let error = parse(&input).unwrap_err();
        assert_eq!(error.field, "timestamp");
    }

    #[test]
    fn timestamp_calendar_and_timezone_ambiguity_fail_closed() {
        let invalid_day = format!(
            "{}{}",
            header(),
            row("2023-02-29 12:00:00.000000", "1")
        );
        assert_eq!(parse(&invalid_day).unwrap_err().field, "timestamp");

        let timezone = format!(
            "{}{}",
            header(),
            row("2024-01-01T00:00:00+02:00", "1")
        );
        assert_eq!(parse(&timezone).unwrap_err().field, "timestamp");
    }

    #[test]
    fn gaps_are_reported_not_silently_resampled() {
        let input = format!(
            "{}{}{}{}",
            header(),
            row("2024-01-01 00:00:00.000000", "1"),
            row("2024-01-01 00:00:00.020000", "1"),
            row("2024-01-01 00:00:00.060000", "1")
        );
        let (summary, _) = parse(&input).unwrap();
        assert_eq!(summary.gap_count, 1);
        assert_eq!(summary.maximum_gap_ns, Some(40_000_000));
    }

    #[test]
    fn non_finite_acceleration_is_rejected() {
        let input = format!(
            "{}2024-01-01 00:00:00.000000,NaN,0,1,0,0,1,1\n",
            header()
        );
        let error = parse(&input).unwrap_err();
        assert_eq!(error.field, "back_x");
    }

    #[test]
    fn unknown_activity_code_is_rejected() {
        let input = format!(
            "{}{}",
            header(),
            row("2024-01-01 00:00:00.000000", "99")
        );
        let error = parse(&input).unwrap_err();
        assert_eq!(error.field, "label");
    }

    #[test]
    fn optional_export_index_must_be_contiguous() {
        let input = format!(
            "index,{}0,{}2,{}",
            header(),
            row("2024-01-01 00:00:00.000000", "1"),
            row("2024-01-01 00:00:00.020000", "2")
        );
        let error = parse(&input).unwrap_err();
        assert_eq!(error.field, "export_index");
    }

    #[test]
    fn normalized_digest_is_deterministic_and_bound_to_source_family() {
        let input = format!(
            "{}{}{}",
            header(),
            row("2024-01-01 00:00:00.000000", "1"),
            row("2024-01-01 00:00:00.020000", "2")
        );
        let (first, _) = parse(&input).unwrap();
        let (second, _) = parse(&input).unwrap();
        assert_eq!(first.normalized_rows_sha256, second.normalized_rows_sha256);

        let other_family = parse_harth_csv(
            Cursor::new(input.as_bytes()),
            [8; 32],
            |_| Ok(()),
        )
        .unwrap();
        assert_ne!(first.normalized_rows_sha256, other_family.normalized_rows_sha256);
    }

    #[test]
    fn all_twelve_public_harth_activity_codes_are_explicit() {
        for code in [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140] {
            assert_eq!(HarthActivity::from_code(code).map(HarthActivity::code), Some(code));
        }
        assert_eq!(HarthActivity::from_code(0), None);
        assert_eq!(HarthActivity::from_code(99), None);
    }

    #[test]
    fn zero_source_family_identity_is_rejected_before_reading() {
        let error = parse_harth_csv(
            Cursor::new(header().as_bytes()),
            [0; 32],
            |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(error.field, "source_family_sha256");
    }
}
