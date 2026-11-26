use physical_storage_mds_reference::{
    mds_encode, mds_recover_errors_and_erasures, reference_payload, MediaImage, PhysicalLayout,
    StorageError, BLOCK_BYTES, DATA_SHARDS, SHARDS_PER_STRIPE,
};

fn sample_data() -> [[u8; BLOCK_BYTES]; DATA_SHARDS] {
    std::array::from_fn(|shard| {
        std::array::from_fn(|offset| (shard * 41 + offset * 13 + 7) as u8)
    })
}

#[test]
fn all_single_symbol_error_positions_and_byte_lanes_are_corrected() {
    let data = sample_data();
    let encoded = mds_encode(data);

    for shard in 0..SHARDS_PER_STRIPE {
        for byte_offset in 0..BLOCK_BYTES {
            let mut damaged = encoded.map(Some);
            damaged[shard].as_mut().unwrap()[byte_offset] ^= 0xA7;

            let (recovered, corrected) =
                mds_recover_errors_and_erasures(damaged).unwrap();

            assert_eq!(&recovered[..DATA_SHARDS], &data[..]);
            assert_eq!(corrected, 1, "shard={shard}, byte_offset={byte_offset}");
        }
    }
}

#[test]
fn every_two_known_erasure_pattern_is_reconstructed() {
    let data = sample_data();
    let encoded = mds_encode(data);

    for first in 0..SHARDS_PER_STRIPE {
        for second in first + 1..SHARDS_PER_STRIPE {
            let mut damaged = encoded.map(Some);
            damaged[first] = None;
            damaged[second] = None;

            let (recovered, corrected) =
                mds_recover_errors_and_erasures(damaged).unwrap();

            assert_eq!(recovered, encoded, "erasures={first},{second}");
            assert_eq!(corrected, 0);
        }
    }
}

#[test]
fn one_complete_tile_substitution_is_eight_lane_errors_and_is_corrected() {
    let data = sample_data();
    let encoded = mds_encode(data);
    let mut damaged = encoded.map(Some);

    for byte_offset in 0..BLOCK_BYTES {
        damaged[3].as_mut().unwrap()[byte_offset] ^= (byte_offset as u8) + 1;
    }

    let (recovered, corrected) = mds_recover_errors_and_erasures(damaged).unwrap();

    assert_eq!(recovered, encoded);
    assert_eq!(corrected, BLOCK_BYTES);
}

#[test]
fn one_unknown_error_plus_one_erasure_is_outside_the_guaranteed_radius() {
    let data = sample_data();
    let encoded = mds_encode(data);
    let mut damaged = encoded.map(Some);
    damaged[0] = None;
    damaged[1].as_mut().unwrap()[2] ^= 0x5A;

    assert!(matches!(
        mds_recover_errors_and_erasures(damaged),
        Err(StorageError::UncorrectableErrorsOrErasures { erasures: 1 })
    ));
}

#[test]
fn over_budget_two_error_miscorrection_is_rejected_by_external_payload_digest() {
    let layout = PhysicalLayout::new();
    let payload = reference_payload();
    let mut image = MediaImage::write_reference(&payload, &layout).unwrap();

    // In one byte lane, this two-symbol perturbation moves the received vector
    // one symbol away from a different valid codeword. A bounded decoder may
    // therefore return that nearby codeword. The independent payload digest
    // must reject it: codeword membership is not proof of original content.
    let first = layout.shard_tile(0, 0).unwrap();
    let second = layout.shard_tile(0, 1).unwrap();
    image.corrupt_byte(first, 2, 0x01).unwrap();
    image.corrupt_byte(second, 2, 0x01).unwrap();

    let receipt = image.read_and_recover(&layout);
    assert!(!receipt.digest_matches, "{}", receipt.to_json());
    assert!(!receipt.exact_recovery, "{}", receipt.to_json());
    assert_eq!(receipt.corrected_symbols, 1);
}
