use symthaea_swarm::rfc9942_selection::{
    evaluate_priority_first_valid, ReceiptSelectionCandidateStatus, ReceiptSelectionRejection,
};
use symthaea_swarm::{
    Rfc9942ProofKind, Rfc9942ReceiptCollection, Rfc9942ReceiptEnvelope, Rfc9942ReceiptPayload,
    Rfc9942Vdp, Rfc9162InclusionProof, COSE_ES256_ALGORITHM_ID,
};

fn cbor_bstr(bytes: &[u8]) -> Vec<u8> {
    let len = bytes.len();
    let mut out = Vec::new();
    match len {
        0..=23 => out.push(0x40 | len as u8),
        24..=255 => {
            out.push(0x58);
            out.push(len as u8);
        }
        256..=65_535 => {
            out.push(0x59);
            out.extend_from_slice(&(len as u16).to_be_bytes());
        }
        _ => panic!("test fixture is unexpectedly large"),
    }
    out.extend_from_slice(bytes);
    out
}

fn collection() -> Rfc9942ReceiptCollection {
    let proof = Rfc9162InclusionProof::new(2, 0, vec![[0x11; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
    Rfc9942ReceiptCollection::new(vec![
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
    .unwrap()
}

#[test]
fn priority_first_valid_is_auditable() {
    let collection = collection();
    let collection_bytes = collection.to_cbor();

    let decision = evaluate_priority_first_valid(
        &collection,
        &collection_bytes,
        |index, _| {
            if index == 0 {
                Err(symthaea_swarm::Rfc9942VdpError::InvalidEs256Signature)
            } else {
                Ok(())
            }
        },
    );

    assert_eq!(decision.selected_index, Some(1));
    assert_eq!(
        decision.candidates[0].status,
        ReceiptSelectionCandidateStatus::Rejected(
            ReceiptSelectionRejection::InvalidSignature
        )
    );
    assert_eq!(
        decision.candidates[1].status,
        ReceiptSelectionCandidateStatus::Selected
    );
}

#[test]
fn short_circuit_is_explicit_and_deterministic() {
    let collection = collection();
    let bytes = collection.to_cbor();
    let first = evaluate_priority_first_valid(&collection, &bytes, |_index, _| Ok(()));
    let second = evaluate_priority_first_valid(&collection, &bytes, |_index, _| Ok(()));

    assert_eq!(first, second);
    assert_eq!(first.selected_index, Some(0));
    assert_eq!(
        first.candidates[1].status,
        ReceiptSelectionCandidateStatus::NotEvaluatedAfterSelection
    );
    assert_eq!(first.digest(), second.digest());
}

#[test]
fn changing_the_collection_order_changes_the_decision_identity() {
    let collection = collection();
    let reversed = Rfc9942ReceiptCollection::new(
        collection.receipts().iter().cloned().rev().collect(),
    )
    .unwrap();

    let first = evaluate_priority_first_valid(&collection, &collection.to_cbor(), |_index, _| {
        Err(symthaea_swarm::Rfc9942VdpError::NoMatchingProof)
    });
    let second = evaluate_priority_first_valid(&reversed, &reversed.to_cbor(), |_index, _| {
        Err(symthaea_swarm::Rfc9942VdpError::NoMatchingProof)
    });

    assert_ne!(first.collection_sha256, second.collection_sha256);
    assert_ne!(first.digest(), second.digest());
}

#[test]
fn parsed_noncanonical_receipt_wire_bytes_are_preserved() {
    let collection = collection();
    let canonical_receipt = collection.serialized_receipt_bytes(0).unwrap();

    // COSE_Sign1 permits an indefinite-length top-level array. The protected
    // bytes, payload, and signature remain identical, so this changes only the
    // receipt's source encoding and is useful for testing provenance.
    assert_eq!(canonical_receipt.get(1), Some(&0x84));
    let mut noncanonical_receipt = canonical_receipt.to_vec();
    noncanonical_receipt[1] = 0x9f;
    noncanonical_receipt.push(0xff);

    let mut wire = vec![0x81];
    wire.extend_from_slice(&cbor_bstr(&noncanonical_receipt));


    let parsed = Rfc9942ReceiptCollection::from_cbor(&wire).unwrap();
    assert_eq!(parsed.serialized_bytes(), Some(wire.as_slice()));
    assert_eq!(
        parsed.serialized_receipt_bytes(0),
        Some(noncanonical_receipt.as_slice())
    );
    assert_eq!(parsed.to_cbor(), wire);
}
