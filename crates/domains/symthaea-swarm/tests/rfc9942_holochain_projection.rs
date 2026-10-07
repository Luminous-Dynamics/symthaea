use symthaea_swarm::holochain_projection::{
    EvidenceAnchorKind, HolochainActionHash, HolochainEvidenceAnchor,
    HolochainProjectionError, ReceiptSelectionContext, HOLOCHAIN_ACTION_HASH_BYTES,
    HOLOCHAIN_ACTION_HASH_PREFIX,
    MAX_RECEIPT_SELECTION_CANDIDATES,
};
use uuid::Uuid;

fn valid_action_hash(fill: u8) -> HolochainActionHash {
    let mut bytes = [fill; HOLOCHAIN_ACTION_HASH_BYTES];
    bytes[..3].copy_from_slice(&HOLOCHAIN_ACTION_HASH_PREFIX);
    HolochainActionHash::from_raw(bytes).unwrap()
}

fn anchor() -> HolochainEvidenceAnchor {
    HolochainEvidenceAnchor {
        anchor_id: Uuid::from_u128(1),
        origin_node_id: Uuid::from_u128(2),
        kind: EvidenceAnchorKind::Attestation,
        evidence_digest: [1; 32],
        parent_evidence_digest: Some([2; 32]),
        parent_action_hash: Some(valid_action_hash(7)),
        vds_root: Some([3; 32]),
        receipt_selection: Some(ReceiptSelectionContext {
            collection_sha256: [4; 32],
            collection_len: 2,
            selected_index: 1,
            selected_receipt_sha256: [5; 32],
            selection_decision_sha256: [6; 32],
            selection_policy: "rfc9942/priority-first-valid-v1".into(),
            selection_policy_version: 1,
        }),
        selection_decision_action_hash: Some(
            valid_action_hash(8),
        ),
        context: b"qualification".to_vec(),
    }
}

#[test]
fn projection_is_deterministic() {
    let first = anchor().canonical_bytes().unwrap();
    let second = anchor().canonical_bytes().unwrap();
    assert_eq!(first, second);
}

#[test]
fn native_action_hashes_are_opaque_but_exactly_address_sized() {
    let hash = HolochainActionHash::from_raw([9; HOLOCHAIN_ACTION_HASH_BYTES]).unwrap();
    assert_eq!(hash.as_bytes().len(), HOLOCHAIN_ACTION_HASH_BYTES);
    assert!(HolochainActionHash::from_raw([0; HOLOCHAIN_ACTION_HASH_BYTES]).is_err());
    assert_eq!(
        HolochainActionHash::from_raw([9; HOLOCHAIN_ACTION_HASH_BYTES]),
        Err(HolochainProjectionError::InvalidActionHashType)
    );
}

#[test]
fn durable_dependency_claims_fail_closed_without_native_addresses() {
    let mut missing_parent = anchor();
    missing_parent.parent_action_hash = None;
    assert_eq!(
        missing_parent.validate(),
        Err(HolochainProjectionError::UnaddressableDependency("parent_evidence"))
    );

    let mut missing_selection = anchor();
    missing_selection.selection_decision_action_hash = None;
    assert_eq!(
        missing_selection.validate(),
        Err(HolochainProjectionError::UnaddressableDependency("selection_decision"))
    );
}

#[test]
fn selection_context_changes_projection_identity() {
    let mut first = anchor();
    let first_bytes = first.canonical_bytes().unwrap();
    first.receipt_selection.as_mut().unwrap().selected_index = 0;
    let second_bytes = first.canonical_bytes().unwrap();
    assert_ne!(first_bytes, second_bytes);
}


#[cfg(feature = "semantic-receipts")]
#[test]
fn bound_selection_projection_requires_exact_source_collection() {
    use symthaea_swarm::rfc9942_selection::evaluate_priority_first_valid;
    use symthaea_swarm::{
        Rfc9942ProofKind, Rfc9942ReceiptCollection, Rfc9942ReceiptEnvelope,
        Rfc9942ReceiptPayload, Rfc9942Vdp, Rfc9162InclusionProof,
        COSE_ES256_ALGORITHM_ID,
    };

    let proof = Rfc9162InclusionProof::new(2, 0, vec![[0x11; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
    let receipts = Rfc9942ReceiptCollection::new(vec![
        Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            vdp.clone(),
            Rfc9942ReceiptPayload::Attached([1; 32]),
            vec![1; 64],
        )
        .unwrap(),
        Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            vdp,
            Rfc9942ReceiptPayload::Attached([2; 32]),
            vec![2; 64],
        )
        .unwrap(),
    ])
    .unwrap();

    let decision = evaluate_priority_first_valid(&receipts, |index, _| {
        if index == 1 {
            Ok(())
        } else {
            Err(symthaea_swarm::Rfc9942VdpError::NoMatchingProof)
        }
    });

    let context =
        ReceiptSelectionContext::from_bound_decision(&decision, &receipts).unwrap();
    assert_eq!(context.collection_len, 2);
    assert_eq!(context.selected_index, 1);
    assert_eq!(
        context.selection_decision_sha256,
        decision.validated_digest().unwrap()
    );

    let reversed = Rfc9942ReceiptCollection::new(
        receipts.receipts().iter().cloned().rev().collect(),
    )
    .unwrap();
    assert_eq!(
        ReceiptSelectionContext::from_bound_decision(&decision, &reversed),
        Err(HolochainProjectionError::InvalidReceiptSelection)
    );

    let mut forged = decision.clone();
    forged.candidates[0].receipt_sha256[0] ^= 1;
    assert_eq!(
        ReceiptSelectionContext::from_bound_decision(&forged, &receipts),
        Err(HolochainProjectionError::InvalidReceiptSelection)
    );

    assert!(MAX_RECEIPT_SELECTION_CANDIDATES >= receipts.len() as u32);
}
