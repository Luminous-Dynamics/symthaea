// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[cfg(not(feature = "simd"))]
fn main() {
    eprintln!("re-run with: cargo run -p symthaea-hdc-ltc --example simd_continuous_bench --release --features simd");
}

#[cfg(feature = "simd")]
fn main() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};
    use symthaea_hdc_ltc::{ContinuousHV, ContinuousHvSimdExt, simd_backend};

    println!("symthaea-hdc-ltc continuous SIMD probe");
    println!("backend: {:?}", simd_backend());
    println!("NOTE: this is an engineering microbenchmark, not scientific evidence.\n");

    for dim in [256usize, 512, 1_024, 2_048, 4_096, 8_192, 16_384] {
        let a = ContinuousHV::new_random(dim, 0xA11CE);
        let b = ContinuousHV::new_random(dim, 0xB0B);
        let iterations = (2_000_000usize / dim).clamp(128, 8_192);

        let scalar_similarity = bench(iterations, || black_box(a.similarity(black_box(&b))));
        let simd_similarity = bench(iterations, || black_box(a.similarity_simd(black_box(&b))));

        let scalar_bind = bench(iterations, || black_box(a.bind(black_box(&b))));
        let simd_bind = bench(iterations, || black_box(a.bind_simd(black_box(&b))));

        let scalar_update = bench(iterations, || {
            let mut state = a.clone();
            state.lerp_in_place(black_box(&b), 0.125);
            black_box(state)
        });
        let simd_update = bench(iterations, || {
            let mut state = a.clone();
            state.lerp_in_place_simd(black_box(&b), 0.125);
            black_box(state)
        });

        println!(
            "D={dim:>5}  similarity {:>6.2}x  bind {:>6.2}x  lerp {:>6.2}x  [scalar/simd]",
            ratio(scalar_similarity, simd_similarity),
            ratio(scalar_bind, simd_bind),
            ratio(scalar_update, simd_update),
        );
    }
}

#[cfg(feature = "simd")]
fn bench<T>(iterations: usize, mut f: impl FnMut() -> T) -> Duration {
    for _ in 0..16 {
        std::hint::black_box(f());
    }
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(f());
    }
    start.elapsed()
}

#[cfg(feature = "simd")]
fn ratio(scalar: Duration, accelerated: Duration) -> f64 {
    scalar.as_secs_f64() / accelerated.as_secs_f64().max(f64::MIN_POSITIVE)
}
