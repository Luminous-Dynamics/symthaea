//! Shared domain types for the public CEB API-migration development fixture.

/// A parsed record used by the parser, storage adapter and CLI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub id: String,
    pub payload: String,
}

/// Typed reasons why a record cannot be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    MissingDelimiter,
    EmptyId,
    EmptyPayload,
}
