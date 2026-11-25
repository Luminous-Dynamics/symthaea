use physical_storage_rs_reference::{
    reference_payload, MediaImage, PhysicalLayout, STRIPE_COUNT, TILE_COUNT,
};

fn print_case(case_id: &str, mut image: MediaImage, layout: &PhysicalLayout, erase: &[usize]) {
    if let Err(error) = image.erase_tiles(erase) {
        eprintln!("{case_id}: {error}");
        std::process::exit(2);
    }
    let receipt = image.read_and_recover(layout);
    println!(
        "{{\"case_id\":\"{case_id}\",\"erased_tiles\":{},\"receipt\":{}}}",
        erase.len(),
        receipt.to_json()
    );
}

fn main() {
    let layout = PhysicalLayout::new();
    let payload = reference_payload();
    let image = MediaImage::write_reference(&payload, &layout)
        .expect("reference payload and physical layout are fixed");

    println!(
        "{{\"fixture\":\"synthetic-rs6-4-v0.1\",\"payload_bytes\":{},\"physical_tiles\":{},\"stripes\":{}}}",
        payload.len(),
        TILE_COUNT,
        STRIPE_COUNT
    );

    print_case("pristine", image.clone(), &layout, &[]);

    for z in 0..3 {
        let layer: Vec<usize> = (z * 16..(z + 1) * 16).collect();
        print_case(&format!("whole_layer_{z}_dropout"), image.clone(), &layout, &layer);
    }

    print_case(
        "localized_2x2_cluster",
        image.clone(),
        &layout,
        &[5, 6, 9, 10],
    );

    let mut corrupted = image;
    let tile = layout.shard_tile(0, 0).expect("valid fixed shard");
    corrupted
        .corrupt_byte(tile, 3, 0x80)
        .expect("fixed test tile is present");
    print_case("unflagged_byte_substitution", corrupted, &layout, &[]);
}
