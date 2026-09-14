use std::collections::BTreeSet;

pub(crate) fn valid_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && !digest.bytes().any(|byte| byte.is_ascii_uppercase())
    })
}

pub(crate) fn canonical_text(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

pub(crate) fn nonempty_refs(values: &[String]) -> bool {
    !values.is_empty() && values.iter().all(|value| canonical_text(value))
}

pub(crate) fn all_unique_valid_digests(values: &[String]) -> bool {
    let mut unique = BTreeSet::new();
    values
        .iter()
        .all(|value| valid_digest(value) && unique.insert(value.as_str()))
}

pub(crate) fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}
