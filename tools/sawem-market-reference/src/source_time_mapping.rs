//! Source-clock provenance and explicit UTC conversion for field-data replay.
//!
//! Raw civil timestamps are never treated as UTC by implication. Each record
//! must name a source clock convention whose signed UTC offset is supported by
//! a source reference and digest. Offsets are record-level evidence: callers
//! must resolve daylight-saving transitions and source-specific clock changes
//! before mapping. This module does not guess an IANA timezone or repair skew.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CivilDateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceClockConvention<'a> {
    pub clock_id: &'a str,
    /// Exact label stated by the source, such as "UTC+1" or "UTC+2".
    pub source_clock_label: &'a str,
    /// Signed local offset from UTC in minutes.
    pub utc_offset_minutes: i16,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceTimestampRecord<'a> {
    pub record_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub clock_id: &'a str,
    pub local_timestamp: CivilDateTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedSourceTimestamp<'a> {
    pub record_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub clock_id: &'a str,
    pub source_clock_label: &'a str,
    pub utc_offset_minutes: i16,
    pub local_timestamp: CivilDateTime,
    pub timestamp_utc_seconds: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceClockComparison<'a> {
    pub local_timestamp: CivilDateTime,
    pub left_clock_id: &'a str,
    pub left_source_clock_label: &'a str,
    pub left_timestamp_utc_seconds: i64,
    pub right_clock_id: &'a str,
    pub right_source_clock_label: &'a str,
    pub right_timestamp_utc_seconds: i64,
    /// right UTC instant minus left UTC instant for the same civil wall time.
    pub right_minus_left_utc_seconds: i64,
    pub same_instant: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceTimeError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidUtcOffset,
    InvalidCivilDate,
    InvalidCivilTime,
    UnknownClockConvention,
    DuplicateClockId,
    DuplicateRecordId,
    ArithmeticOverflow,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i32, month: u8) -> Option<u8> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 if valid_leap_year(year) => Some(29),
        2 => Some(28),
        _ => None,
    }
}

fn validate_civil_timestamp(value: CivilDateTime) -> Result<(), SourceTimeError> {
    if !(1..=9999).contains(&value.year) {
        return Err(SourceTimeError::InvalidCivilDate);
    }
    let Some(max_day) = days_in_month(value.year, value.month) else {
        return Err(SourceTimeError::InvalidCivilDate);
    };
    if value.day == 0 || value.day > max_day {
        return Err(SourceTimeError::InvalidCivilDate);
    }
    if value.hour >= 24 || value.minute >= 60 || value.second >= 60 {
        return Err(SourceTimeError::InvalidCivilTime);
    }
    Ok(())
}

/// Convert a proleptic-Gregorian civil date to days since 1970-01-01.
/// The algorithm is integer-only and supports the validated year range.
fn days_from_civil(value: CivilDateTime) -> Result<i64, SourceTimeError> {
    let year = i64::from(value.year);
    let month = i64::from(value.month);
    let day = i64::from(value.day);
    let adjusted_year = year - i64::from(value.month <= 2);
    let era = if adjusted_year >= 0 {
        adjusted_year
    } else {
        adjusted_year - 399
    } / 400;
    let year_of_era = adjusted_year - era * 400;
    let adjusted_month = month + if value.month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era.checked_mul(146_097)
        .and_then(|days| days.checked_add(day_of_era))
        .and_then(|days| days.checked_sub(719_468))
        .ok_or(SourceTimeError::ArithmeticOverflow)
}

fn validate_clock(clock: SourceClockConvention<'_>) -> Result<(), SourceTimeError> {
    if !nonempty(clock.clock_id)
        || !nonempty(clock.source_clock_label)
        || !nonempty(clock.source_ref)
    {
        return Err(SourceTimeError::InvalidEvidenceReference);
    }
    if !valid_sha256(clock.source_sha256) {
        return Err(SourceTimeError::InvalidSourceDigest);
    }
    if !(-840..=840).contains(&clock.utc_offset_minutes) {
        return Err(SourceTimeError::InvalidUtcOffset);
    }
    Ok(())
}

/// Map one source-local timestamp to UTC using its explicit, source-bound
/// offset. Positive offsets are east of UTC: UTC = local time - offset.
pub fn map_source_timestamp<'a>(
    record: SourceTimestampRecord<'a>,
    clock: SourceClockConvention<'a>,
) -> Result<MappedSourceTimestamp<'a>, SourceTimeError> {
    validate_clock(clock)?;
    if !nonempty(record.record_id) || !nonempty(record.source_ref) || !nonempty(record.clock_id) {
        return Err(SourceTimeError::InvalidEvidenceReference);
    }
    if !valid_sha256(record.source_sha256) {
        return Err(SourceTimeError::InvalidSourceDigest);
    }
    if record.clock_id != clock.clock_id {
        return Err(SourceTimeError::UnknownClockConvention);
    }
    validate_civil_timestamp(record.local_timestamp)?;

    let days = days_from_civil(record.local_timestamp)?;
    let local_seconds = days
        .checked_mul(86_400)
        .and_then(|seconds| {
            seconds.checked_add(i64::from(record.local_timestamp.hour) * 3_600)
        })
        .and_then(|seconds| {
            seconds.checked_add(i64::from(record.local_timestamp.minute) * 60)
        })
        .and_then(|seconds| seconds.checked_add(i64::from(record.local_timestamp.second)))
        .ok_or(SourceTimeError::ArithmeticOverflow)?;
    let offset_seconds = i64::from(clock.utc_offset_minutes)
        .checked_mul(60)
        .ok_or(SourceTimeError::ArithmeticOverflow)?;
    let utc_seconds = local_seconds
        .checked_sub(offset_seconds)
        .ok_or(SourceTimeError::ArithmeticOverflow)?;

    Ok(MappedSourceTimestamp {
        record_id: record.record_id,
        source_ref: record.source_ref,
        source_sha256: record.source_sha256,
        clock_id: clock.clock_id,
        source_clock_label: clock.source_clock_label,
        utc_offset_minutes: clock.utc_offset_minutes,
        local_timestamp: record.local_timestamp,
        timestamp_utc_seconds: utc_seconds,
    })
}

/// Compare the same wall-clock label under two source-declared offsets.
/// A nonzero result is reported as a clock discrepancy; no correction is
/// applied and neither clock is assumed to be authoritative.
pub fn compare_source_clocks<'a>(
    local_timestamp: CivilDateTime,
    left: SourceClockConvention<'a>,
    right: SourceClockConvention<'a>,
) -> Result<SourceClockComparison<'a>, SourceTimeError> {
    let left_record = SourceTimestampRecord {
        record_id: "clock-comparison/left",
        source_ref: left.source_ref,
        source_sha256: left.source_sha256,
        clock_id: left.clock_id,
        local_timestamp,
    };
    let right_record = SourceTimestampRecord {
        record_id: "clock-comparison/right",
        source_ref: right.source_ref,
        source_sha256: right.source_sha256,
        clock_id: right.clock_id,
        local_timestamp,
    };
    let mapped_left = map_source_timestamp(left_record, left)?;
    let mapped_right = map_source_timestamp(right_record, right)?;
    let delta = mapped_right
        .timestamp_utc_seconds
        .checked_sub(mapped_left.timestamp_utc_seconds)
        .ok_or(SourceTimeError::ArithmeticOverflow)?;

    Ok(SourceClockComparison {
        local_timestamp,
        left_clock_id: left.clock_id,
        left_source_clock_label: left.source_clock_label,
        left_timestamp_utc_seconds: mapped_left.timestamp_utc_seconds,
        right_clock_id: right.clock_id,
        right_source_clock_label: right.source_clock_label,
        right_timestamp_utc_seconds: mapped_right.timestamp_utc_seconds,
        right_minus_left_utc_seconds: delta,
        same_instant: delta == 0,
    })
}

/// Map an ordered source file. It does not sort, interpolate, deduplicate by
/// timestamp, or infer missing timezone information.
pub fn map_source_timestamp_series<'a>(
    records: &[SourceTimestampRecord<'a>],
    clocks: &[SourceClockConvention<'a>],
) -> Result<Vec<MappedSourceTimestamp<'a>>, SourceTimeError> {
    let mut mapped = Vec::with_capacity(records.len());

    for (clock_index, clock) in clocks.iter().copied().enumerate() {
        validate_clock(clock)?;
        if clocks[..clock_index]
            .iter()
            .any(|previous| previous.clock_id == clock.clock_id)
        {
            return Err(SourceTimeError::DuplicateClockId);
        }
    }

    for (index, record) in records.iter().copied().enumerate() {
        if records[..index]
            .iter()
            .any(|previous| previous.record_id == record.record_id)
        {
            return Err(SourceTimeError::DuplicateRecordId);
        }
        let clock = clocks
            .iter()
            .copied()
            .find(|clock| clock.clock_id == record.clock_id)
            .ok_or(SourceTimeError::UnknownClockConvention)?;
        mapped.push(map_source_timestamp(record, clock)?);
    }

    Ok(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn civil(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> CivilDateTime {
        CivilDateTime {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }

    fn clock(id: &'static str, label: &'static str, offset: i16) -> SourceClockConvention<'static> {
        SourceClockConvention {
            clock_id: id,
            source_clock_label: label,
            utc_offset_minutes: offset,
            source_ref: id,
            source_sha256: SOURCE_SHA,
        }
    }

    fn record(id: &'static str, clock_id: &'static str, local_timestamp: CivilDateTime) -> SourceTimestampRecord<'static> {
        SourceTimestampRecord {
            record_id: id,
            source_ref: id,
            source_sha256: SOURCE_SHA,
            clock_id,
            local_timestamp,
        }
    }

    #[test]
    fn maps_source_declared_positive_offset_to_utc_across_date_boundary() {
        let mapped = map_source_timestamp(
            record("schedule/row-1", "schedule-utc-plus-1", civil(2024, 1, 1, 0, 30, 0)),
            clock("schedule-utc-plus-1", "UTC+1", 60),
        )
        .unwrap();
        assert_eq!(mapped.timestamp_utc_seconds, 1_704_067_800);
        assert_eq!(mapped.utc_offset_minutes, 60);
    }

    #[test]
    fn same_wall_clock_label_under_utc_plus_one_and_utc_plus_two_is_not_same_instant() {
        let comparison = compare_source_clocks(
            civil(2024, 5, 28, 12, 0, 0),
            clock("schedule", "UTC+1", 60),
            clock("measurement", "UTC+2", 120),
        )
        .unwrap();
        assert_eq!(comparison.right_minus_left_utc_seconds, -3_600);
        assert!(!comparison.same_instant);
        assert_eq!(
            comparison.left_timestamp_utc_seconds - comparison.right_timestamp_utc_seconds,
            3_600
        );
    }

    #[test]
    fn leap_days_are_valid_only_in_gregorian_leap_years() {
        let valid = map_source_timestamp(
            record("leap-valid", "utc", civil(2024, 2, 29, 0, 0, 0)),
            clock("utc", "UTC", 0),
        );
        assert!(valid.is_ok());
        assert_eq!(
            map_source_timestamp(
                record("leap-invalid", "utc", civil(2023, 2, 29, 0, 0, 0)),
                clock("utc", "UTC", 0)
            ),
            Err(SourceTimeError::InvalidCivilDate)
        );
    }

    #[test]
    fn daylight_saving_transition_is_not_inferred_from_a_zone_label() {
        let winter = clock("source-winter", "Europe/Berlin winter UTC+1", 60);
        let summer = clock("source-summer", "Europe/Berlin summer UTC+2", 120);
        let winter_map = map_source_timestamp(
            record("winter-row", "source-winter", civil(2024, 1, 15, 12, 0, 0)),
            winter,
        )
        .unwrap();
        let summer_map = map_source_timestamp(
            record("summer-row", "source-summer", civil(2024, 7, 15, 12, 0, 0)),
            summer,
        )
        .unwrap();
        assert_eq!(winter_map.timestamp_utc_seconds, 1_705_320_000);
        assert_eq!(summer_map.timestamp_utc_seconds, 1_720_440_000);
    }

    #[test]
    fn invalid_offsets_dates_and_source_digests_fail_closed() {
        assert_eq!(
            map_source_timestamp(
                record("bad-offset", "bad", civil(2024, 1, 1, 0, 0, 0)),
                clock("bad", "UTC+15", 900)
            ),
            Err(SourceTimeError::InvalidUtcOffset)
        );
        assert_eq!(
            map_source_timestamp(
                record("bad-time", "utc", civil(2024, 1, 1, 24, 0, 0)),
                clock("utc", "UTC", 0)
            ),
            Err(SourceTimeError::InvalidCivilTime)
        );
        let bad_digest = SourceClockConvention {
            source_sha256: "not-a-digest",
            ..clock("bad-digest", "UTC", 0)
        };
        assert_eq!(
            map_source_timestamp(
                record("bad-digest", "bad-digest", civil(2024, 1, 1, 0, 0, 0)),
                bad_digest
            ),
            Err(SourceTimeError::InvalidSourceDigest)
        );
    }

    #[test]
    fn timestamp_series_requires_unique_clock_and_record_ids() {
        let clocks = [clock("utc", "UTC", 0), clock("utc", "UTC+1", 60)];
        assert_eq!(
            map_source_timestamp_series(&[], &clocks),
            Err(SourceTimeError::DuplicateClockId)
        );
        let unique_clock = [clock("utc", "UTC", 0)];
        let records = [
            record("same", "utc", civil(2024, 1, 1, 0, 0, 0)),
            record("same", "utc", civil(2024, 1, 1, 0, 1, 0)),
        ];
        assert_eq!(
            map_source_timestamp_series(&records, &unique_clock),
            Err(SourceTimeError::DuplicateRecordId)
        );
    }

    #[test]
    fn unknown_clock_convention_is_not_replaced_with_a_default() {
        assert_eq!(
            map_source_timestamp_series(
                &[record("row", "missing-clock", civil(2024, 1, 1, 0, 0, 0))],
                &[]
            ),
            Err(SourceTimeError::UnknownClockConvention)
        );
    }
}
