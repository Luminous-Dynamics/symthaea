//! Protocol surface for the public CEB feature/target development fixture.

#[cfg(feature = "json")]
pub fn encode_json(payload: &str) -> String {
    format!("json:{payload}")
}

#[cfg(feature = "binary")]
pub fn encode_binary(payload: &[u8]) -> Vec<u8> {
    let mut encoded = b"binary:".to_vec();
    encoded.extend_from_slice(payload);
    encoded
}
