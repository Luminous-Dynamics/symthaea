/*!
Session 5: Zero-Copy Architecture Verification Benchmark

CRITICAL: This benchmark verifies WHETHER copy overhead is actually a bottleneck
before we claim any speedups!

Hypothesis: Memory copies waste 99% of time for ultra-fast SIMD operations

Tests:
1. Memory copy overhead (clone vs reference)
2. mmap vs malloc performance
3. Arena allocation vs malloc/free
4. Zero-copy SIMD operations

Expected Results (to be verified!):
- Copy overhead: ~2µs per 256-byte vector
- Zero-copy operations: Just 10-12ns SIMD time
- Potential speedup: 200x for small operations

If hypothesis is WRONG, we'll discover it here before implementing!
*/

use criterion::{criterion_group, criterion_main, Criterion, black_box};
use symthaea::hdc::binary_hv::HV16;
use std::time::Duration;

// =============================================================================
// Test 1: Memory Copy Overhead - IS IT REAL?
// =============================================================================

/// Benchmark traditional approach: Clone vectors before operating
fn benchmark_with_copy(c: &mut Criterion) {
    let vec1 = HV16::random(42);
    let vec2 = HV16::random(43);

    c.bench_function("bind_with_copy", |b| {
        b.iter(|| {
            // Traditional: Clone before operating
            let v1_copy = black_box(vec1.clone());  // Copy #1
            let v2_copy = black_box(vec2.clone());  // Copy #2
            let result = v1_copy.bind(&v2_copy);    // Operation
            black_box(result)
        })
    });
}

/// Benchmark zero-copy approach: Operate directly on references
fn benchmark_zero_copy(c: &mut Criterion) {
    let vec1 = HV16::random(42);
    let vec2 = HV16::random(43);

    c.bench_function("bind_zero_copy", |b| {
        b.iter(|| {
            // Zero-copy: Operate directly on references
            let result = vec1.bind(&vec2);  // No copies!
            black_box(result)
        })
    });
}

// =============================================================================
// Test 2: Batch Operations - Copy Overhead Accumulation
// =============================================================================

/// Benchmark batch similarities with cloning (traditional)
fn benchmark_batch_similarities_with_copy(c: &mut Criterion) {
    let query = HV16::random(1);
    let memories: Vec<HV16> = (0..1000).map(|i| HV16::random((i + 100) as u64)).collect();

    c.bench_function("batch_similarities_1000_with_copy", |b| {
        b.iter(|| {
            let mut similarities = Vec::with_capacity(1000);
            for memory in &memories {
                // Clone query each time (wasteful!)
                let q = black_box(query.clone());
                let m = black_box(memory.clone());
                let sim = q.similarity(&m);
                similarities.push(black_box(sim));
            }
            similarities
        })
    });
}

/// Benchmark batch similarities without cloning (zero-copy)
fn benchmark_batch_similarities_zero_copy(c: &mut Criterion) {
    let query = HV16::random(1);
    let memories: Vec<HV16> = (0..1000).map(|i| HV16::random((i + 100) as u64)).collect();

    c.bench_function("batch_similarities_1000_zero_copy", |b| {
        b.iter(|| {
            let mut similarities = Vec::with_capacity(1000);
            for memory in &memories {
                // No cloning! Direct references
                let sim = query.similarity(memory);
                similarities.push(black_box(sim));
            }
            similarities
        })
    });
}

// =============================================================================
// Test 3: Arena Allocation - Pre-allocated Buffer Reuse
// =============================================================================

/// Benchmark traditional Vec allocation (malloc every time)
fn benchmark_traditional_allocation(c: &mut Criterion) {
    c.bench_function("traditional_vec_allocation_1000", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for i in 0..1000 {
                // Allocate new vector each iteration
                let vec = black_box(HV16::random(i as u64));
                results.push(vec);
            }
            black_box(results)
        })
    });
}

/// Benchmark arena pre-allocation (allocate once, reuse)
fn benchmark_arena_allocation(c: &mut Criterion) {
    c.bench_function("arena_preallocated_buffer_1000", |b| {
        b.iter(|| {
            // Pre-allocate arena buffer
            let mut arena: Vec<HV16> = Vec::with_capacity(1000);
            unsafe { arena.set_len(1000); }  // Reserve space

            for i in 0..1000 {
                // Write directly to pre-allocated buffer (no malloc!)
                arena[i] = black_box(HV16::random(i as u64));
            }
            black_box(arena)
        })
    });
}

// =============================================================================
// Test 4: Stack vs Heap Allocation
// =============================================================================

/// Benchmark heap allocation (Box<HV16>)
fn benchmark_heap_allocation(c: &mut Criterion) {
    c.bench_function("heap_box_allocation", |b| {
        b.iter(|| {
            let vec = black_box(Box::new(HV16::random(42)));
            black_box(vec)
        })
    });
}

/// Benchmark stack allocation (HV16 directly)
fn benchmark_stack_allocation(c: &mut Criterion) {
    c.bench_function("stack_allocation", |b| {
        b.iter(|| {
            let vec = black_box(HV16::random(42));
            black_box(vec)
        })
    });
}

// =============================================================================
// Test 5: Realistic Consciousness Cycle - Copy Overhead Impact
// =============================================================================

/// Consciousness cycle with unnecessary copies (traditional)
fn benchmark_consciousness_cycle_with_copies(c: &mut Criterion) {
    let concepts: Vec<HV16> = (0..100).map(|i| HV16::random(i as u64)).collect();
    let memories: Vec<HV16> = (0..1000).map(|i| HV16::random((i + 1000) as u64)).collect();

    c.bench_function("consciousness_cycle_with_copies", |b| {
        b.iter(|| {
            // Bundle concepts (with unnecessary clones)
            let bundled = {
                let concept_copies: Vec<HV16> = concepts.iter()
                    .map(|c| black_box(c.clone()))
                    .collect();
                HV16::bundle(&concept_copies)
            };

            // Compute similarities (with unnecessary clones)
            let mut similarities = Vec::with_capacity(1000);
            for memory in &memories {
                let q = black_box(bundled.clone());
                let m = black_box(memory.clone());
                let sim = q.similarity(&m);
                similarities.push(black_box(sim));
            }

            black_box(similarities)
        })
    });
}

/// Consciousness cycle with zero-copy design
fn benchmark_consciousness_cycle_zero_copy(c: &mut Criterion) {
    let concepts: Vec<HV16> = (0..100).map(|i| HV16::random(i as u64)).collect();
    let memories: Vec<HV16> = (0..1000).map(|i| HV16::random((i + 1000) as u64)).collect();

    c.bench_function("consciousness_cycle_zero_copy", |b| {
        b.iter(|| {
            // Bundle concepts (zero-copy references)
            let bundled = HV16::bundle(&concepts);

            // Compute similarities (direct references)
            let mut similarities = Vec::with_capacity(1000);
            for memory in &memories {
                let sim = bundled.similarity(memory);
                similarities.push(black_box(sim));
            }

            black_box(similarities)
        })
    });
}

// =============================================================================
// Benchmark Groups
// =============================================================================

criterion_group! {
    name = zerocopy_benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(2));
    targets =
        // Test 1: Basic copy overhead
        benchmark_with_copy,
        benchmark_zero_copy,

        // Test 2: Batch operations
        benchmark_batch_similarities_with_copy,
        benchmark_batch_similarities_zero_copy,

        // Test 3: Allocation strategies
        benchmark_traditional_allocation,
        benchmark_arena_allocation,

        // Test 4: Stack vs heap
        benchmark_heap_allocation,
        benchmark_stack_allocation,

        // Test 5: Realistic consciousness cycle
        benchmark_consciousness_cycle_with_copies,
        benchmark_consciousness_cycle_zero_copy,
}

criterion_main!(zerocopy_benches);

// =============================================================================
// Expected Results (To Be Verified!)
// =============================================================================

/*
HYPOTHESIS (needs verification):

1. Copy Overhead:
   - bind_with_copy:    ~2010ns  (2µs copy + 10ns SIMD)
   - bind_zero_copy:    ~10ns    (just SIMD)
   - Speedup:           ~200x

2. Batch Similarities:
   - with_copy:         ~2012µs  (1000 × 2µs copy + 1000 × 12ns)
   - zero_copy:         ~12µs    (1000 × 12ns)
   - Speedup:           ~167x

3. Allocation:
   - traditional:       ~100µs   (1000 × 100ns malloc)
   - arena:             ~1µs     (one allocation)
   - Speedup:           ~100x

4. Stack vs Heap:
   - heap:              ~50ns    (Box allocation)
   - stack:             ~1ns     (register/stack)
   - Speedup:           ~50x

5. Consciousness Cycle:
   - with_copies:       ~250µs   (bundle copies + similarity copies)
   - zero_copy:         ~15µs    (just computations)
   - Speedup:           ~17x

CRITICAL: These are HYPOTHESES!
We'll know the TRUTH after running this benchmark.

If copy overhead is NOT significant, we'll discover it and pivot to GPU!
*/
