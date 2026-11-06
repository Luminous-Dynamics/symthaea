//! Ranking types for the public CEB streaming top-k development fixture.

/// The total order for equal scores is deterministic: score first, then key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RankedItem {
    pub score: i64,
    pub key: String,
}
