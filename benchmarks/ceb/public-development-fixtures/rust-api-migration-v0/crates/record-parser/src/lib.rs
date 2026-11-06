//! Parser crate for the public CEB API-migration development fixture.

use ceb_api_migration_domain_types::{ParseError, Record};

/// Internal typed parser used by the public legacy adapter.
///
/// The candidate task is to promote the typed operation as the canonical public
/// API without removing or changing the documented legacy adapter.
pub(crate) fn parse_record_typed(input: &str) -> Result<Record, ParseError> {
    let (id, payload) = input
        .split_once('|')
        .ok_or(ParseError::MissingDelimiter)?;

    if id.is_empty() {
        return Err(ParseError::EmptyId);
    }

    if payload.is_empty() {
        return Err(ParseError::EmptyPayload);
    }

    Ok(Record {
        id: id.to_owned(),
        payload: payload.to_owned(),
    })
}

/// Legacy parser surface retained for existing callers.
///
/// This adapter intentionally preserves the historical `Option` behavior.
pub fn parse_record_legacy(input: &str) -> Option<Record> {
    parse_record_typed(input).ok()
}
