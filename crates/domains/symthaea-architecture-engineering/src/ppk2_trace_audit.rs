//! Raw PPK2 sample CSV validation and timestamp-aware energy reconstruction.
//! Input is exported from ppk2_samples.parquet by the upstream NATS-Bench-MCU helper.
//! This verifies the supplied CSV bytes, not acquisition authenticity or the original Parquet.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt;

use csv::{ReaderBuilder, StringRecord, Trim};
use serde::Serialize;
use sha2::{Digest, Sha256};

const SCHEMA: &str = "symthaea.nas.ppk2-trace-audit";
const VERSION: &str = "0.1.0";
const DOMAIN: &[u8] = b"symthaea.nas.ppk2-trace-audit.v0.1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ppk2TraceAuditConfig {
    pub voltage_v: f64,
    pub expected_sample_rate_hz: f64,
    pub sample_rate_tolerance_fraction: f64,
    pub maximum_gap_ratio: f64,
}

impl Default for Ppk2TraceAuditConfig {
    fn default() -> Self {
        Self {
            voltage_v: 3.3,
            expected_sample_rate_hz: 100_000.0,
            sample_rate_tolerance_fraction: 0.20,
            maximum_gap_ratio: 2.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ppk2TraceAuditError {
    pub field: String,
    pub message: String,
}

impl Ppk2TraceAuditError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for Ppk2TraceAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for Ppk2TraceAuditError {}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Ppk2SegmentMetrics {
    pub segment_id: String,
    pub sample_count: usize,
    pub first_time_ms: f64,
    pub last_time_ms: f64,
    pub duration_ms: f64,
    pub minimum_current_ua: f64,
    pub maximum_current_ua: f64,
    pub time_weighted_mean_current_ua: f64,
    pub mean_power_uw: f64,
    pub energy_uj: f64,
    pub median_interval_ms: f64,
    pub maximum_interval_ms: f64,
    pub median_sample_rate_hz: f64,
    pub maximum_gap_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Ppk2TraceAuditReceipt {
    pub schema: String,
    pub version: String,
    pub sample_csv_sha256: String,
    pub sample_csv_bytes: usize,
    pub voltage_v: f64,
    pub expected_sample_rate_hz: f64,
    pub sample_rate_tolerance_fraction: f64,
    pub maximum_gap_ratio: f64,
    pub segment_count: usize,
    pub total_sample_count: usize,
    pub mean_segment_current_ua: f64,
    pub mean_segment_power_uw: f64,
    pub mean_segment_energy_uj: f64,
    pub minimum_segment_energy_uj: f64,
    pub maximum_segment_energy_uj: f64,
    pub mean_segment_duration_ms: f64,
    pub median_interval_ms: f64,
    pub median_sample_rate_hz: f64,
    pub maximum_observed_gap_ratio: f64,
    pub segments: Vec<Ppk2SegmentMetrics>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Ppk2TraceAuditResult {
    pub receipt: Ppk2TraceAuditReceipt,
}

#[derive(Debug, Clone)]
struct Sample {
    row_number: usize,
    time_ms: f64,
    current_ua: f64,
}

fn finite(raw: &str, field: &str, row: usize) -> Result<f64, Ppk2TraceAuditError> {
    let value = raw.parse::<f64>().map_err(|_| {
        Ppk2TraceAuditError::new(
            format!("samples.row[{row}].{field}"),
            format!("expected a finite number, got {raw:?}"),
        )
    })?;
    if !value.is_finite() {
        return Err(Ppk2TraceAuditError::new(
            format!("samples.row[{row}].{field}"),
            "NaN and infinity are forbidden",
        ));
    }
    Ok(value)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() { return None; }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

fn validate_config(c: Ppk2TraceAuditConfig) -> Result<(), Ppk2TraceAuditError> {
    if !c.voltage_v.is_finite() || c.voltage_v <= 0.0 {
        return Err(Ppk2TraceAuditError::new("config.voltage_v", "must be finite and positive"));
    }
    if !c.expected_sample_rate_hz.is_finite() || c.expected_sample_rate_hz <= 0.0 {
        return Err(Ppk2TraceAuditError::new("config.expected_sample_rate_hz", "must be finite and positive"));
    }
    if !c.sample_rate_tolerance_fraction.is_finite()
        || !(0.0..=1.0).contains(&c.sample_rate_tolerance_fraction)
    {
        return Err(Ppk2TraceAuditError::new("config.sample_rate_tolerance_fraction", "must be in [0, 1]"));
    }
    if !c.maximum_gap_ratio.is_finite() || c.maximum_gap_ratio < 1.0 {
        return Err(Ppk2TraceAuditError::new("config.maximum_gap_ratio", "must be finite and at least 1"));
    }
    Ok(())
}

fn audit_segment(
    id: &str,
    samples: &[Sample],
    c: Ppk2TraceAuditConfig,
) -> Result<Ppk2SegmentMetrics, Ppk2TraceAuditError> {
    if samples.len() < 2 {
        return Err(Ppk2TraceAuditError::new(
            format!("segments.{id}.sample_count"),
            "at least two samples are required",
        ));
    }

    let mut intervals = Vec::with_capacity(samples.len() - 1);
    let mut integral_ua_ms = 0.0;
    let mut min_current = f64::INFINITY;
    let mut max_current = f64::NEG_INFINITY;

    for sample in samples {
        min_current = min_current.min(sample.current_ua);
        max_current = max_current.max(sample.current_ua);
    }
    for pair in samples.windows(2) {
        let dt = pair[1].time_ms - pair[0].time_ms;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Ppk2TraceAuditError::new(
                format!("segments.{id}.row[{}].time_ms", pair[1].row_number),
                "timestamps must be strictly increasing within each segment",
            ));
        }
        intervals.push(dt);
        integral_ua_ms += (pair[0].current_ua + pair[1].current_ua) * 0.5 * dt;
    }

    let duration_ms = samples.last().unwrap().time_ms - samples.first().unwrap().time_ms;
    let mut sorted_intervals = intervals.clone();
    let median_interval_ms = median(&mut sorted_intervals).ok_or_else(|| {
        Ppk2TraceAuditError::new(format!("segments.{id}.intervals"), "no intervals")
    })?;
    let maximum_interval_ms = intervals.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0);
    let gap_ratio = maximum_interval_ms / median_interval_ms;
    if gap_ratio > c.maximum_gap_ratio {
        return Err(Ppk2TraceAuditError::new(
            format!("segments.{id}.maximum_gap_ratio"),
            format!("maximum interval ratio {gap_ratio:.6} exceeds limit {:.6}", c.maximum_gap_ratio),
        ));
    }

    let median_sample_rate_hz = 1_000.0 / median_interval_ms;
    let rate_error = (median_sample_rate_hz - c.expected_sample_rate_hz).abs()
        / c.expected_sample_rate_hz;
    if rate_error > c.sample_rate_tolerance_fraction {
        return Err(Ppk2TraceAuditError::new(
            format!("segments.{id}.sample_rate_hz"),
            format!("observed {median_sample_rate_hz:.3} Hz differs from expected {:.3} Hz by {:.3}%",
                c.expected_sample_rate_hz, rate_error * 100.0),
        ));
    }

    // µA × ms × V = nJ; divide by 1,000 to obtain µJ.
    let energy_uj = integral_ua_ms * c.voltage_v / 1_000.0;
    let mean_current_ua = integral_ua_ms / duration_ms;
    let mean_power_uw = mean_current_ua * c.voltage_v;
    if !energy_uj.is_finite() || !mean_current_ua.is_finite() || !mean_power_uw.is_finite() {
        return Err(Ppk2TraceAuditError::new(format!("segments.{id}.derived_metrics"), "non-finite integration result"));
    }

    Ok(Ppk2SegmentMetrics {
        segment_id: id.to_owned(),
        sample_count: samples.len(),
        first_time_ms: samples.first().unwrap().time_ms,
        last_time_ms: samples.last().unwrap().time_ms,
        duration_ms,
        minimum_current_ua: min_current,
        maximum_current_ua: max_current,
        time_weighted_mean_current_ua: mean_current_ua,
        mean_power_uw,
        energy_uj,
        median_interval_ms,
        maximum_interval_ms,
        median_sample_rate_hz,
        maximum_gap_ratio: gap_ratio,
    })
}

fn receipt_digest(r: &Ppk2TraceAuditReceipt) -> Result<String, Ppk2TraceAuditError> {
    let mut canonical = r.clone();
    canonical.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|e| {
        Ppk2TraceAuditError::new("receipt", format!("cannot serialize: {e}"))
    })?;
    let mut h = Sha256::new();
    h.update(DOMAIN);
    h.update(bytes);
    Ok(format!("{:x}", h.finalize()))
}

/// Revalidate a serialized trace receipt before downstream scoring or uncertainty analysis.
/// This checks the receipt's canonical digest and all aggregate/segment consistency relations.
/// It does not authenticate the original acquisition; source CSV/Parquet provenance is separate.
pub fn validate_ppk2_trace_audit_receipt(
    receipt: &Ppk2TraceAuditReceipt,
) -> Result<(), Ppk2TraceAuditError> {
    if receipt.schema != SCHEMA || receipt.version != VERSION {
        return Err(Ppk2TraceAuditError::new("schema_version", "unsupported trace receipt schema/version"));
    }
    let is_digest = |value: &str| {
        value.len() == 64
            && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    };
    if !is_digest(&receipt.sample_csv_sha256) || !is_digest(&receipt.receipt_sha256) {
        return Err(Ppk2TraceAuditError::new("receipt.digest", "source and receipt digests must be lowercase SHA-256 hex"));
    }
    if receipt.sample_csv_bytes == 0
        || receipt.segment_count == 0
        || receipt.segment_count != receipt.segments.len()
        || receipt.total_sample_count == 0
    {
        return Err(Ppk2TraceAuditError::new("receipt.inventory", "invalid byte, segment, or sample inventory"));
    }
    validate_config(Ppk2TraceAuditConfig {
        voltage_v: receipt.voltage_v,
        expected_sample_rate_hz: receipt.expected_sample_rate_hz,
        sample_rate_tolerance_fraction: receipt.sample_rate_tolerance_fraction,
        maximum_gap_ratio: receipt.maximum_gap_ratio,
    })?;

    let mut ids = HashSet::with_capacity(receipt.segments.len());
    let mut total_samples = 0usize;
    for segment in &receipt.segments {
        if segment.segment_id.trim().is_empty() || !ids.insert(segment.segment_id.as_str()) {
            return Err(Ppk2TraceAuditError::new("receipt.segments.segment_id", "segment IDs must be non-empty and unique"));
        }
        if segment.sample_count < 2 {
            return Err(Ppk2TraceAuditError::new(
                format!("receipt.segments.{}.sample_count", segment.segment_id),
                "each segment must contain at least two samples",
            ));
        }
        total_samples += segment.sample_count;
        for (field, value) in [
            ("first_time_ms", segment.first_time_ms),
            ("last_time_ms", segment.last_time_ms),
            ("duration_ms", segment.duration_ms),
            ("minimum_current_ua", segment.minimum_current_ua),
            ("maximum_current_ua", segment.maximum_current_ua),
            ("time_weighted_mean_current_ua", segment.time_weighted_mean_current_ua),
            ("mean_power_uw", segment.mean_power_uw),
            ("energy_uj", segment.energy_uj),
            ("median_interval_ms", segment.median_interval_ms),
            ("maximum_interval_ms", segment.maximum_interval_ms),
            ("median_sample_rate_hz", segment.median_sample_rate_hz),
            ("maximum_gap_ratio", segment.maximum_gap_ratio),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Ppk2TraceAuditError::new(
                    format!("receipt.segments.{}.{}", segment.segment_id, field),
                    "must be finite and positive",
                ));
            }
        }
        if segment.last_time_ms <= segment.first_time_ms
            || (segment.last_time_ms - segment.first_time_ms - segment.duration_ms).abs() > 1e-9
            || segment.minimum_current_ua > segment.maximum_current_ua
            || segment.maximum_gap_ratio > receipt.maximum_gap_ratio
        {
            return Err(Ppk2TraceAuditError::new(
                format!("receipt.segments.{}.consistency", segment.segment_id),
                "segment timing, current range, or gap ratio is inconsistent",
            ));
        }
    }
    if total_samples != receipt.total_sample_count {
        return Err(Ppk2TraceAuditError::new("receipt.total_sample_count", "does not equal segment sample inventory"));
    }

    let approx_eq = |a: f64, b: f64| {
        (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
    };
    let expected_current = mean(receipt.segments.iter().map(|m| m.time_weighted_mean_current_ua)).unwrap();
    let expected_power = mean(receipt.segments.iter().map(|m| m.mean_power_uw)).unwrap();
    let expected_energy = mean(receipt.segments.iter().map(|m| m.energy_uj)).unwrap();
    let expected_duration = mean(receipt.segments.iter().map(|m| m.duration_ms)).unwrap();
    let expected_min_energy = receipt.segments.iter().map(|m| m.energy_uj).min_by(f64::total_cmp).unwrap();
    let expected_max_energy = receipt.segments.iter().map(|m| m.energy_uj).max_by(f64::total_cmp).unwrap();
    if !approx_eq(receipt.mean_segment_current_ua, expected_current)
        || !approx_eq(receipt.mean_segment_power_uw, expected_power)
        || !approx_eq(receipt.mean_segment_energy_uj, expected_energy)
        || !approx_eq(receipt.mean_segment_duration_ms, expected_duration)
        || !approx_eq(receipt.minimum_segment_energy_uj, expected_min_energy)
        || !approx_eq(receipt.maximum_segment_energy_uj, expected_max_energy)
    {
        return Err(Ppk2TraceAuditError::new("receipt.aggregates", "aggregate metrics do not match segment inventory"));
    }
    if receipt_digest(receipt)? != receipt.receipt_sha256 {
        return Err(Ppk2TraceAuditError::new("receipt.receipt_sha256", "does not match canonical receipt"));
    }
    Ok(())
}

/// Audit a CSV exported from the raw PPK2 Parquet trace.
/// Required columns: segment_id, time_ms, current_uA.
pub fn audit_ppk2_trace_csv(
    input: &str,
    config: Ppk2TraceAuditConfig,
) -> Result<Ppk2TraceAuditResult, Ppk2TraceAuditError> {
    validate_config(config)?;
    let mut reader = ReaderBuilder::new().has_headers(true).flexible(false).trim(Trim::All)
        .from_reader(input.as_bytes());
    let headers = reader.headers().map_err(|e| {
        Ppk2TraceAuditError::new("samples.header", format!("cannot read header: {e}"))
    })?.clone();

    let mut seen = HashSet::with_capacity(headers.len());
    for header in headers.iter() {
        if header.trim().is_empty() || !seen.insert(header.to_owned()) {
            return Err(Ppk2TraceAuditError::new("samples.header", "empty or duplicate column name"));
        }
    }
    let locate = |name: &str| headers.iter().position(|h| h == name).ok_or_else(|| {
        Ppk2TraceAuditError::new("samples.header", format!("required column {name:?} is missing"))
    });
    let segment_col = locate("segment_id")?;
    let time_col = locate("time_ms")?;
    let current_col = locate("current_uA")?;

    let mut segments = BTreeMap::<String, Vec<Sample>>::new();
    for (index, record) in reader.records().enumerate() {
        let row_number = index + 2;
        let row = record.map_err(|e| Ppk2TraceAuditError::new(
            format!("samples.row[{row_number}]"), format!("malformed CSV: {e}"),
        ))?;
        let id = row.get(segment_col).unwrap_or("").to_owned();
        if id.trim().is_empty() {
            return Err(Ppk2TraceAuditError::new(format!("samples.row[{row_number}].segment_id"), "must be non-empty"));
        }
        let time_ms = finite(row.get(time_col).unwrap_or(""), "time_ms", row_number)?;
        let current_ua = finite(row.get(current_col).unwrap_or(""), "current_uA", row_number)?;
        if current_ua < 0.0 {
            return Err(Ppk2TraceAuditError::new(format!("samples.row[{row_number}].current_uA"), "negative current is forbidden"));
        }
        segments.entry(id).or_default().push(Sample { row_number, time_ms, current_ua });
    }
    if segments.is_empty() {
        return Err(Ppk2TraceAuditError::new("samples.rows", "trace has no sample records"));
    }

    let mut metrics = Vec::with_capacity(segments.len());
    let mut all_intervals = Vec::new();
    let mut samples_total = 0usize;
    let mut max_gap_ratio = 0.0_f64;
    for (id, segment) in &segments {
        let m = audit_segment(id, segment, config)?;
        for pair in segment.windows(2) {
            all_intervals.push(pair[1].time_ms - pair[0].time_ms);
        }
        max_gap_ratio = max_gap_ratio.max(m.maximum_gap_ratio);
        samples_total += m.sample_count;
        metrics.push(m);
    }

    let mut intervals_copy = all_intervals;
    let median_interval_ms = median(&mut intervals_copy).ok_or_else(|| {
        Ppk2TraceAuditError::new("samples.intervals", "no timestamp intervals")
    })?;
    let mut receipt = Ppk2TraceAuditReceipt {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        sample_csv_sha256: format!("{:x}", Sha256::digest(input.as_bytes())),
        sample_csv_bytes: input.len(),
        voltage_v: config.voltage_v,
        expected_sample_rate_hz: config.expected_sample_rate_hz,
        sample_rate_tolerance_fraction: config.sample_rate_tolerance_fraction,
        maximum_gap_ratio: config.maximum_gap_ratio,
        segment_count: metrics.len(),
        total_sample_count: samples_total,
        mean_segment_current_ua: mean(metrics.iter().map(|m| m.time_weighted_mean_current_ua)).unwrap(),
        mean_segment_power_uw: mean(metrics.iter().map(|m| m.mean_power_uw)).unwrap(),
        mean_segment_energy_uj: mean(metrics.iter().map(|m| m.energy_uj)).unwrap(),
        minimum_segment_energy_uj: metrics.iter().map(|m| m.energy_uj).min_by(f64::total_cmp).unwrap(),
        maximum_segment_energy_uj: metrics.iter().map(|m| m.energy_uj).max_by(f64::total_cmp).unwrap(),
        mean_segment_duration_ms: mean(metrics.iter().map(|m| m.duration_ms)).unwrap(),
        median_interval_ms,
        median_sample_rate_hz: 1_000.0 / median_interval_ms,
        maximum_observed_gap_ratio: max_gap_ratio,
        segments: metrics,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_digest(&receipt)?;
    Ok(Ppk2TraceAuditResult { receipt })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Ppk2TraceAuditConfig {
        Ppk2TraceAuditConfig {
            voltage_v: 3.3,
            expected_sample_rate_hz: 100_000.0,
            sample_rate_tolerance_fraction: 0.01,
            maximum_gap_ratio: 1.5,
        }
    }

    #[test]
    fn constant_current_energy_is_exact() {
        let input = concat!("segment_id,time_ms,current_uA\n","0,0.00,1000\n","0,0.01,1000\n","0,0.02,1000\n");
        let r = audit_ppk2_trace_csv(input, cfg()).unwrap();
        assert!((r.receipt.segments[0].energy_uj - 0.066).abs() < 1e-12);
        assert!((r.receipt.segments[0].mean_power_uw - 3300.0).abs() < 1e-12);
    }

    #[test]
    fn ramp_energy_uses_trapezoid_rule() {
        let input = concat!("segment_id,time_ms,current_uA\n","0,0.00,0\n","0,0.01,500\n","0,0.02,1000\n");
        let r = audit_ppk2_trace_csv(input, cfg()).unwrap();
        assert!((r.receipt.segments[0].energy_uj - 0.033).abs() < 1e-12);
    }

    #[test]
    fn irregular_timestamps_use_actual_deltas() {
        let input = concat!("segment_id,time_ms,current_uA\n","0,0.00,1000\n","0,0.01,1000\n","0,0.019,1000\n","0,0.029,1000\n");
        let r = audit_ppk2_trace_csv(input, cfg()).unwrap();
        assert!((r.receipt.segments[0].duration_ms - 0.029).abs() < 1e-12);
        assert!((r.receipt.segments[0].energy_uj - 0.0957).abs() < 1e-12);
    }

    #[test]
    fn segments_are_integrated_and_averaged_separately() {
        let input = concat!("segment_id,time_ms,current_uA\n","0,0.00,1000\n","0,0.01,1000\n","1,0.00,2000\n","1,0.01,2000\n");
        let r = audit_ppk2_trace_csv(input, cfg()).unwrap();
        assert_eq!(r.receipt.segment_count, 2);
        assert!((r.receipt.mean_segment_energy_uj - 0.0495).abs() < 1e-12);
    }

    #[test]
    fn rejects_duplicate_header_and_missing_columns() {
        let duplicate = "segment_id,time_ms,time_ms,current_uA\n0,0.0,0.0,10\n0,0.01,0.01,10\n";
        assert!(audit_ppk2_trace_csv(duplicate, cfg()).is_err());
        let missing = "segment_id,time_ms\n0,0.0\n0,0.01\n";
        assert!(audit_ppk2_trace_csv(missing, cfg()).is_err());
    }

    #[test]
    fn rejects_duplicate_or_reversed_timestamps() {
        let duplicate = concat!("segment_id,time_ms,current_uA\n","0,0.00,10\n","0,0.01,10\n","0,0.01,10\n");
        assert!(audit_ppk2_trace_csv(duplicate, cfg()).is_err());
        let reversed = concat!("segment_id,time_ms,current_uA\n","0,0.01,10\n","0,0.00,10\n","0,0.02,10\n");
        assert!(audit_ppk2_trace_csv(reversed, cfg()).is_err());
    }

    #[test]
    fn rejects_non_finite_and_negative_current() {
        let non_finite = concat!("segment_id,time_ms,current_uA\n","0,0.00,NaN\n","0,0.01,10\n");
        assert!(audit_ppk2_trace_csv(non_finite, cfg()).is_err());
        let negative = concat!("segment_id,time_ms,current_uA\n","0,0.00,-1\n","0,0.01,10\n");
        assert!(audit_ppk2_trace_csv(negative, cfg()).is_err());
    }

    #[test]
    fn rejects_short_trace_large_gap_and_rate_mismatch() {
        let short = "segment_id,time_ms,current_uA\n0,0.00,10\n";
        assert!(audit_ppk2_trace_csv(short, cfg()).is_err());
        let gap = concat!("segment_id,time_ms,current_uA\n","0,0.00,10\n","0,0.01,10\n","0,0.10,10\n");
        assert!(audit_ppk2_trace_csv(gap, cfg()).is_err());
        let bad_rate = concat!("segment_id,time_ms,current_uA\n","0,0.00,10\n","0,0.02,10\n","0,0.04,10\n");
        assert!(audit_ppk2_trace_csv(bad_rate, cfg()).is_err());
    }

    #[test]
    fn receipt_binds_source_bytes_and_voltage() {
        let input = concat!("segment_id,time_ms,current_uA\n","0,0.00,1000\n","0,0.01,1000\n");
        let a = audit_ppk2_trace_csv(input, cfg()).unwrap();
        let b = audit_ppk2_trace_csv(input, cfg()).unwrap();
        assert_eq!(a.receipt.receipt_sha256, b.receipt.receipt_sha256);
        let mut changed = cfg();
        changed.voltage_v = 3.0;
        let c = audit_ppk2_trace_csv(input, changed).unwrap();
        assert_ne!(a.receipt.receipt_sha256, c.receipt.receipt_sha256);
    }
}
