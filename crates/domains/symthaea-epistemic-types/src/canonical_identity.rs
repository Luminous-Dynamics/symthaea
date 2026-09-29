//! Deterministic, language-neutral identity encoding for epistemic memory projections.
//!
//! This module deliberately avoids JSON as a cryptographic identity format. The wire
//! representation may remain JSON, but identity digests use explicit byte encoding so
//! independent implementations can reproduce the same bytes and SHA-256 digest.

use crate::{sha256_hex, MemoryKind};

pub const MEMORY_CANONICAL_ENCODING_DOMAIN: &[u8] = b"memory-canonical:v1\\0";
pub const MEMORY_PROJECTION_ENCODING_DOMAIN: &[u8] = b"memory-projection:v1\\0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalEncodingError {
    StringTooLong,
}

fn put_len_prefixed(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), CanonicalEncodingError> {
    let len = u32::try_from(bytes.len()).map_err(|_| CanonicalEncodingError::StringTooLong)?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn put_string(out: &mut Vec<u8>, value: &str) -> Result<(), CanonicalEncodingError> {
    put_len_prefixed(out, value.as_bytes())
}

fn memory_kind_tag(kind: MemoryKind) -> u8 {
    match kind {
        MemoryKind::Working => 0,
        MemoryKind::Episodic => 1,
        MemoryKind::Semantic => 2,
        MemoryKind::Procedural => 3,
        MemoryKind::KnowledgeGraph => 4,
        MemoryKind::Vector => 5,
        MemoryKind::Hdc => 6,
    }
}

pub fn canonical_identity_bytes(identity: &str) -> Result<Vec<u8>, CanonicalEncodingError> {
    let mut out = Vec::with_capacity(MEMORY_CANONICAL_ENCODING_DOMAIN.len() + 4 + identity.len());
    out.extend_from_slice(MEMORY_CANONICAL_ENCODING_DOMAIN);
    put_string(&mut out, identity)?;
    Ok(out)
}

pub fn canonical_identity_digest(identity: &str) -> Result<String, CanonicalEncodingError> {
    Ok(sha256_hex(&canonical_identity_bytes(identity)?))
}

pub fn projection_identity_bytes(
    schema_version: u16,
    canonical_identity: &str,
    memory_kind: MemoryKind,
    projection_profile: &str,
    source_frontier: Option<&str>,
) -> Result<Vec<u8>, CanonicalEncodingError> {
    let mut out = Vec::new();
    out.extend_from_slice(MEMORY_PROJECTION_ENCODING_DOMAIN);
    out.extend_from_slice(&schema_version.to_be_bytes());
    out.push(memory_kind_tag(memory_kind));
    put_string(&mut out, canonical_identity)?;
    put_string(&mut out, projection_profile)?;
    match source_frontier {
        Some(frontier) => {
            out.push(1);
            put_string(&mut out, frontier)?;
        }
        None => out.push(0),
    }
    Ok(out)
}

pub fn projection_identity_digest(
    schema_version: u16,
    canonical_identity: &str,
    memory_kind: MemoryKind,
    projection_profile: &str,
    source_frontier: Option<&str>,
) -> Result<String, CanonicalEncodingError> {
    Ok(sha256_hex(&projection_identity_bytes(
        schema_version,
        canonical_identity,
        memory_kind,
        projection_profile,
        source_frontier,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_vector_is_stable() {
        assert_eq!(
            canonical_identity_digest("claim:abc").unwrap(),
            "223b32c636f5eff3dff7b27cc4087657e15a4b48a5cfb5787f9d01310a09bb8c"
        );
    }

    #[test]
    fn projection_vector_is_stable() {
        assert_eq!(
            projection_identity_digest(1, "claim:abc", MemoryKind::Semantic, "semantic-v1", None).unwrap(),
            "f42c445e8f9c7a1781986b9b4df9a0ef9d45deaf49c63d11f4bf7e8a486c1791"
        );
    }

    #[test]
    fn unicode_vector_is_stable() {
        assert_eq!(
            canonical_identity_digest("π/材料").unwrap(),
            "b93856edcb98cdeb82505efba801b4231ce1541cda75d021658be5534be6ad02"
        );
        assert_eq!(
            projection_identity_digest(1, "π/材料", MemoryKind::Hdc, "hdc-v1", Some("frontier:7")).unwrap(),
            "9a4206df0a45ff92e48ecb6499946b8d79052e6e7324e718cf91eba9c03f9fb0"
        );
    }

    #[test]
    fn optional_frontier_is_structurally_tagged() {
        let without = projection_identity_bytes(1, "claim:abc", MemoryKind::Semantic, "semantic-v1", None).unwrap();
        let with_empty = projection_identity_bytes(1, "claim:abc", MemoryKind::Semantic, "semantic-v1", Some("")).unwrap();
        assert_ne!(without, with_empty);
    }
}
