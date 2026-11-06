//! CLI-facing consumer for the public CEB API-migration development fixture.

use ceb_api_migration_record_parser::parse_record_legacy;

/// Current CLI behavior collapses parse failures into one generic error.
///
/// The candidate task would migrate this consumer to the typed public API while
/// preserving the result formatting for valid records.
pub fn summarize_record(input: &str) -> Result<String, &'static str> {
    let record = parse_record_legacy(input).ok_or("invalid record")?;
    Ok(format!("{}:{}", record.id, record.payload))
}
