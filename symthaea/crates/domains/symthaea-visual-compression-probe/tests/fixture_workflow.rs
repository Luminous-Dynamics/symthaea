use symthaea_visual_compression_probe::{
    GrayImage, VisualMemoryPacket, benchmark_image, edge_energy, image_hash64, packet_similarity,
};

#[test]
fn fixture_packet_workflow_is_stable() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/tiny_pump_scan.pgm");
    let image = GrayImage::read_pgm(path).expect("fixture should parse");
    assert_ne!(image_hash64(&image), 0);
    assert!(edge_energy(&image) >= 0.0);

    let packet = VisualMemoryPacket::encode(&image, 8, 10).expect("fixture should encode");
    packet.validate().expect("packet should validate");
    assert_ne!(packet.stable_hash64(), 0);

    let text = packet.to_text();
    let roundtrip = VisualMemoryPacket::from_text(&text).expect("packet text should roundtrip");
    roundtrip
        .validate()
        .expect("roundtrip packet should validate");
    assert_eq!(packet.stable_hash64(), roundtrip.stable_hash64());

    let report = benchmark_image(&image, 8, 10).expect("benchmark should run");
    assert!(report.metrics.stored_coefficients > 0);
}

#[test]
fn related_fixture_similarity_is_measurable() {
    let before = GrayImage::read_pgm(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/tiny_pump_scan.pgm"
    ))
    .unwrap();
    let after = GrayImage::read_pgm(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/tiny_pump_scan_after.pgm"
    ))
    .unwrap();
    let crack = GrayImage::read_pgm(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/tiny_crack_scan.pgm"
    ))
    .unwrap();

    let p_before = VisualMemoryPacket::encode(&before, 8, 10).unwrap();
    let p_after = VisualMemoryPacket::encode(&after, 8, 10).unwrap();
    let p_crack = VisualMemoryPacket::encode(&crack, 8, 10).unwrap();

    let related = packet_similarity(&p_before, &p_after).combined_similarity;
    let unrelated = packet_similarity(&p_before, &p_crack).combined_similarity;

    assert!(related >= 0.0 && related <= 1.0);
    assert!(unrelated >= 0.0 && unrelated <= 1.0);
}
