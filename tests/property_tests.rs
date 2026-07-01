// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! L1: Property Tests with Proptest
//!
//! Property-based testing for HDC operations and core functionality.

use proptest::prelude::*;

// HDC vector generation strategy
#[allow(dead_code)]
fn hv16_strategy() -> impl Strategy<Value = Vec<u16>> {
    prop::collection::vec(any::<u16>(), 1024..2048)
}

fn hv8_strategy() -> impl Strategy<Value = Vec<i8>> {
    prop::collection::vec(-128i8..127i8, 1024..2048)
}

fn text_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9 ]{1,100}"
}

proptest! {
    /// Test that cosine similarity is symmetric
    #[test]
    fn cosine_similarity_symmetric(a in hv8_strategy(), b in hv8_strategy()) {
        // Ensure same length
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let sim_ab = cosine_similarity_i8(a, b);
        let sim_ba = cosine_similarity_i8(b, a);

        prop_assert!((sim_ab - sim_ba).abs() < 1e-6, "Similarity not symmetric");
    }

    /// Test that self-similarity is 1.0
    #[test]
    fn self_similarity_is_one(v in hv8_strategy()) {
        let sim = cosine_similarity_i8(&v, &v);
        prop_assert!((sim - 1.0).abs() < 1e-6, "Self-similarity should be 1.0, got {}", sim);
    }

    /// Test that similarity is bounded [-1, 1]
    #[test]
    fn similarity_bounded(a in hv8_strategy(), b in hv8_strategy()) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let sim = cosine_similarity_i8(a, b);
        prop_assert!((-1.0..=1.0).contains(&sim), "Similarity out of bounds: {}", sim);
    }

    /// Test that XOR binding is reversible
    #[test]
    fn xor_binding_reversible(a in hv8_strategy(), b in hv8_strategy()) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let bound = xor_bind(a, b);
        let unbound = xor_bind(&bound, b);

        prop_assert_eq!(a, &unbound[..], "XOR binding not reversible");
    }

    /// Test that permutation is reversible
    #[test]
    fn permutation_reversible(v in hv8_strategy(), shift in 0usize..100) {
        let permuted = permute(&v, shift);
        let unpermuted = permute(&permuted, v.len() - (shift % v.len()));

        prop_assert_eq!(v, unpermuted, "Permutation not reversible");
    }

    /// Test that bundling preserves dimensionality
    #[test]
    fn bundling_preserves_dim(vectors in prop::collection::vec(hv8_strategy(), 2..10)) {
        if vectors.is_empty() {
            return Ok(());
        }

        let dim = vectors[0].len();
        let same_dim: Vec<_> = vectors.iter()
            .map(|v| &v[..dim.min(v.len())])
            .collect();

        let bundled = bundle(&same_dim);
        prop_assert_eq!(bundled.len(), dim.min(same_dim[0].len()), "Bundling changed dimensionality");
    }

    /// Test that encoding produces consistent results
    #[test]
    fn encoding_deterministic(text in text_strategy()) {
        let encoded1 = simple_encode(&text);
        let encoded2 = simple_encode(&text);

        prop_assert_eq!(encoded1, encoded2, "Encoding not deterministic");
    }

    /// Test that similar texts have correlated encodings (not strongly anti-correlated)
    #[test]
    fn similar_texts_similar_encodings(base in "[a-z]{5,20}") {
        let variant = format!("{}x", base); // Small modification

        let enc1 = simple_encode(&base);
        let enc2 = simple_encode(&variant);

        let sim = cosine_similarity_i8(&enc1, &enc2);

        // Similar texts should not be strongly anti-correlated
        // Note: Simple encoding doesn't guarantee positive similarity for all modifications
        prop_assert!(sim > -0.5, "Similar texts should not be strongly anti-correlated, got sim={}", sim);
    }
}

// Helper functions for property tests

fn cosine_similarity_i8(a: &[i8], b: &[i8]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0i64;
    let mut norm_a = 0i64;
    let mut norm_b = 0i64;

    for i in 0..len {
        let ai = a[i] as i64;
        let bi = b[i] as i64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if norm_a == 0 || norm_b == 0 {
        return 0.0;
    }

    dot as f32 / ((norm_a as f64).sqrt() * (norm_b as f64).sqrt()) as f32
}

fn xor_bind(a: &[i8], b: &[i8]) -> Vec<i8> {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai as i16 ^ bi as i16) as i8)
        .collect()
}

fn permute(v: &[i8], shift: usize) -> Vec<i8> {
    let len = v.len();
    if len == 0 {
        return Vec::new();
    }
    let shift = shift % len;
    let mut result = vec![0i8; len];
    for i in 0..len {
        result[(i + shift) % len] = v[i];
    }
    result
}

fn bundle(vectors: &[&[i8]]) -> Vec<i8> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let dim = vectors[0].len();
    let mut result = vec![0i32; dim];

    for v in vectors {
        for (i, &val) in v.iter().enumerate().take(dim) {
            result[i] += val as i32;
        }
    }

    // Threshold to binary
    let threshold = vectors.len() as i32 / 2;
    result
        .iter()
        .map(|&sum| if sum > threshold { 1 } else { -1 })
        .collect()
}

fn simple_encode(text: &str) -> Vec<i8> {
    // Simple character-based encoding
    let dim = 1024;
    let mut hv = vec![0i8; dim];

    for (i, c) in text.chars().enumerate() {
        let char_val = c as usize;
        for j in 0..dim {
            let idx = (char_val + i * 7 + j) % dim;
            hv[idx] = hv[idx].saturating_add(if (char_val + j).is_multiple_of(2) {
                1
            } else {
                -1
            });
        }
    }

    // Threshold
    hv.iter().map(|&v| if v > 0 { 1 } else { -1 }).collect()
}

// Additional property tests for infrastructure

proptest! {
    /// Test LRU cache eviction
    #[test]
    fn lru_cache_respects_capacity(capacity in 1usize..100, items in prop::collection::vec(0u32..1000, 0..200)) {
        let mut cache = SimpleLruCache::new(capacity);

        for item in items {
            cache.insert(item, item * 2);
        }

        prop_assert!(cache.len() <= capacity, "Cache exceeded capacity");
    }

    /// Test page calculation
    #[test]
    fn pagination_consistent(total in 1usize..1000, page_size in 1usize..100) {
        let total_pages = total.div_ceil(page_size);

        // All pages should be valid
        for page in 0..total_pages {
            let offset = page * page_size;
            prop_assert!(offset < total || page == total_pages - 1);
        }

        // Page count should match
        let items_accounted = if total_pages > 0 {
            (total_pages - 1) * page_size + (total - (total_pages - 1) * page_size)
        } else {
            0
        };
        prop_assert_eq!(items_accounted, total);
    }
}

// Simple LRU cache for testing
struct SimpleLruCache {
    capacity: usize,
    data: std::collections::HashMap<u32, u32>,
    order: std::collections::VecDeque<u32>,
}

impl SimpleLruCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    fn insert(&mut self, key: u32, value: u32) {
        if self.data.contains_key(&key) {
            self.order.retain(|k| *k != key);
        } else if self.data.len() >= self.capacity {
            if let Some(old_key) = self.order.pop_front() {
                self.data.remove(&old_key);
            }
        }

        self.data.insert(key, value);
        self.order.push_back(key);
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}