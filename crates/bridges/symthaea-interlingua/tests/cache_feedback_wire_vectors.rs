use symthaea_interlingua::{
    CACHE_FEEDBACK_WIRE_LEN, SemanticCacheAck, SemanticCacheFeedback, SemanticCacheMiss,
    SemanticCacheMissKind, SemanticCacheRevoke,
};

fn hash(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn cache_feedback_ack_binary_vector_is_stable() {
    let feedback = SemanticCacheFeedback::Ack(SemanticCacheAck::new(hash('a')).unwrap());
    let bytes = feedback.wire_bytes().unwrap();
    assert_eq!(bytes.len(), CACHE_FEEDBACK_WIRE_LEN);
    assert_eq!(
        hex(&bytes),
        "534346010100aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
    assert_eq!(
        SemanticCacheFeedback::from_wire_bytes(&bytes).unwrap(),
        feedback
    );
}

#[test]
fn cache_feedback_reference_miss_binary_vector_is_stable() {
    let feedback = SemanticCacheFeedback::Miss(
        SemanticCacheMiss::new(hash('b'), SemanticCacheMissKind::SemanticReferenceTarget).unwrap(),
    );
    let bytes = feedback.wire_bytes().unwrap();
    assert_eq!(
        hex(&bytes),
        "534346010201bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    );
    assert_eq!(
        SemanticCacheFeedback::from_wire_bytes(&bytes).unwrap(),
        feedback
    );
}

#[test]
fn cache_feedback_delta_base_miss_binary_vector_is_stable() {
    let feedback = SemanticCacheFeedback::Miss(
        SemanticCacheMiss::new(hash('c'), SemanticCacheMissKind::GraphDeltaBase).unwrap(),
    );
    let bytes = feedback.wire_bytes().unwrap();
    assert_eq!(
        hex(&bytes),
        "534346010202cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    );
    assert_eq!(
        SemanticCacheFeedback::from_wire_bytes(&bytes).unwrap(),
        feedback
    );
}

#[test]
fn cache_feedback_revoke_binary_vector_is_stable() {
    let feedback = SemanticCacheFeedback::Revoke(SemanticCacheRevoke::new(hash('d')).unwrap());
    let bytes = feedback.wire_bytes().unwrap();
    assert_eq!(
        hex(&bytes),
        "534346010300dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
    );
    assert_eq!(
        SemanticCacheFeedback::from_wire_bytes(&bytes).unwrap(),
        feedback
    );
}
