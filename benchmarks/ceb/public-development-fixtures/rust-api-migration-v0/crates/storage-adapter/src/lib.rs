//! Storage-facing compatibility consumer for the public CEB development fixture.

use ceb_api_migration_domain_types::Record;
use ceb_api_migration_record_parser::parse_record_legacy;

/// Existing storage ingestion path. The candidate task must keep this legacy
/// adapter callable and behaviorally compatible while promoting a typed API.
pub fn materialize_legacy_record(input: &str) -> Option<Record> {
    parse_record_legacy(input)
}
