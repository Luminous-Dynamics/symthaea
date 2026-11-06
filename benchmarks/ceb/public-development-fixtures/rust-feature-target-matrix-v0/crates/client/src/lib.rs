//! Feature-selectable client for the public CEB feature/target fixture.

#[cfg(feature = "json")]
pub fn encode_text(payload: &str) -> String {
    ceb_feature_matrix_protocol::encode_json(payload)
}

#[cfg(feature = "binary")]
pub fn encode_bytes(payload: &[u8]) -> Vec<u8> {
    // The client feature currently fails to activate protocol/binary.
    ceb_feature_matrix_protocol::encode_binary(payload)
}
