# 🚀 Session 6: Locality-Sensitive Hashing (LSH) - Revolutionary Algorithmic Approach

**Date**: December 22, 2025
**Priority**: ~~**BEFORE GPU**~~ **DEFERRED** - Algorithm mismatch discovered
**Expected**: ~~100-1000x speedup~~ **REFUTED: 0.84x (19% slower!)**
**Status**: ❌ **HYPOTHESIS REFUTED** - Wrong LSH family for binary vectors

---

## ⚠️ VERIFICATION RESULTS - HYPOTHESIS REFUTED

**See**: [SESSION_6_LSH_VERIFICATION_FAILURE.md](SESSION_6_LSH_VERIFICATION_FAILURE.md) for complete analysis

**TL;DR**:
- ❌ **Performance**: LSH was 0.84x (19% SLOWER than brute force!)
- ❌ **Accuracy**: 50% recall (essentially random, not localized)
- ❌ **Scaling**: Linear growth (not constant as expected)

**Root Cause**: Applied **random hyperplane LSH** (for real-valued cosine similarity) to **binary hyperdimensional vectors** (which use Hamming distance). Wrong algorithm for the problem.

**Correct Approach**: Need **SimHash** or **bit-sampling LSH** designed for Hamming distance, not random hyperplanes.

**Decision**: Proceed to GPU acceleration (proven, reliable) and revisit LSH later with correct algorithm.

---

## 📝 ORIGINAL PLAN (for historical reference)

## 🎯 Why LSH Before GPU?

### The Critical Insight

**Current approach**: Make operations faster (SIMD, GPU)
**LSH approach**: **Do fewer operations** (algorithmic improvement)

**The Math**:
```
Brute force search (current):
  1M vectors × 167ns = 167ms

GPU approach:
  1M vectors × 0.167ns = 167µs (1000x faster)
  BUT: Still doing 1M comparisons!

LSH approach:
  1000 vectors × 167ns = 167µs (1000x fewer comparisons!)
  AND: Works on ANY hardware (no GPU needed)
  AND: Can combine with GPU later (multiplicative!)
```

### Why LSH First?

| Factor | LSH First | GPU First |
|--------|-----------|-----------|
| **Implementation time** | 2-3 days | 2-3 weeks |
| **Complexity** | Medium | High (new technology) |
| **Hardware requirement** | None | Requires GPU |
| **Speedup** | 100-1000x | 1000-5000x |
| **Applicability** | Similarity search | All batch ops |
| **Risk** | Low (proven) | Medium (new tech) |
| **Immediate impact** | ✅ 3 days | ⏸️ 3 weeks |

**Key advantage**: LSH + GPU = **multiplicative gains** (1000x × 1000x = 1M×!)

---

## 🏗️ Technical Design

### LSH for Binary Hyperdimensional Vectors

**Core idea**: Random hyperplane projections

For binary vectors (HV16), LSH is beautifully simple:

```rust
// Hash function: Project onto random hyperplane
struct LshHashFunction {
    projection: HV16,  // Random binary vector
}

impl LshHashFunction {
    fn hash(&self, vector: &HV16) -> bool {
        // Dot product mod 2 = XOR + popcount mod 2
        let xor = vector.bind(&self.projection);
        let ones = count_ones(&xor);
        ones % 2 == 0  // Even/odd = two sides of hyperplane
    }
}
```

**Why this works**: Similar vectors project to same side of random hyperplane with high probability!

### Multi-Table LSH Architecture

**Single table**: ~60% recall (misses 40% of true neighbors)
**10 tables**: ~95% recall (misses only 5%)
**20 tables**: ~99% recall (production quality)

```rust
pub struct LshIndex {
    tables: Vec<LshTable>,
    num_bits: usize,        // Hash bits per table (e.g., 10 bits = 1024 buckets)
    num_tables: usize,      // Number of hash tables (e.g., 10 for 95% recall)
}

pub struct LshTable {
    hash_functions: Vec<LshHashFunction>,  // 10 random projections
    buckets: Vec<Vec<usize>>,              // 2^10 = 1024 buckets of vector IDs
}
```

### Hash Function Generation

```rust
impl LshTable {
    pub fn new(num_bits: usize) -> Self {
        let hash_functions: Vec<LshHashFunction> = (0..num_bits)
            .map(|i| LshHashFunction {
                projection: HV16::random(i as u64),
            })
            .collect();

        let num_buckets = 1 << num_bits;  // 2^num_bits
        let buckets = vec![Vec::new(); num_buckets];

        LshTable {
            hash_functions,
            buckets,
        }
    }

    pub fn hash(&self, vector: &HV16) -> usize {
        let mut hash = 0;
        for (i, hash_fn) in self.hash_functions.iter().enumerate() {
            if hash_fn.hash(vector) {
                hash |= 1 << i;  // Set bit i
            }
        }
        hash
    }

    pub fn insert(&mut self, vector_id: usize, vector: &HV16) {
        let bucket_id = self.hash(vector);
        self.buckets[bucket_id].push(vector_id);
    }

    pub fn query(&self, vector: &HV16) -> &[usize] {
        let bucket_id = self.hash(vector);
        &self.buckets[bucket_id]
    }
}
```

### Complete LSH Index

```rust
impl LshIndex {
    pub fn new(num_bits: usize, num_tables: usize) -> Self {
        let tables = (0..num_tables)
            .map(|_| LshTable::new(num_bits))
            .collect();

        LshIndex {
            tables,
            num_bits,
            num_tables,
        }
    }

    pub fn insert_batch(&mut self, vectors: &[HV16]) {
        for (id, vector) in vectors.iter().enumerate() {
            for table in &mut self.tables {
                table.insert(id, vector);
            }
        }
    }

    pub fn query_approximate(
        &self,
        query: &HV16,
        k: usize,
        vectors: &[HV16],
    ) -> Vec<(usize, f32)> {
        // Collect candidates from all tables
        let mut candidates = HashSet::new();

        for table in &self.tables {
            for &vector_id in table.query(query) {
                candidates.insert(vector_id);
            }
        }

        // Compute exact similarities for candidates only
        let mut results: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&id| {
                let similarity = query.similarity(&vectors[id]);
                (id, similarity)
            })
            .collect();

        // Sort by similarity and return top-k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }
}
```

---

## 📊 Performance Analysis

### Expected Performance

**Configuration**: 10 bits/table (1024 buckets), 10 tables

**1 Million Vectors**:
```
Brute force:
  1M comparisons × 167ns = 167ms

LSH (10 tables):
  Candidates: ~10K vectors (1K per table, some overlap)
  Comparisons: 10K × 167ns = 1.67ms
  Speedup: 167ms / 1.67ms = 100x ✅
  Accuracy: ~95% recall
```

**10 Million Vectors**:
```
Brute force:
  10M comparisons × 167ns = 1.67 seconds

LSH (10 tables):
  Candidates: ~10K vectors (hash selectivity remains constant!)
  Comparisons: 10K × 167ns = 1.67ms
  Speedup: 1670ms / 1.67ms = 1000x ✅
  Accuracy: ~95% recall
```

### Accuracy-Speed Trade-off

| Tables | Buckets | Speedup | Recall | Use Case |
|--------|---------|---------|--------|----------|
| 5 | 1024 | 200x | ~80% | Fast, approximate |
| 10 | 1024 | 100x | ~95% | **Production default** |
| 20 | 1024 | 50x | ~99% | High accuracy needed |
| 10 | 2048 | 200x | ~92% | Larger dataset |

---

## 🎯 Implementation Plan (2-3 Days)

### Day 1: Core LSH Implementation

**Morning**:
- [ ] Implement `LshHashFunction` (binary projection)
- [ ] Implement `LshTable` (single hash table)
- [ ] Unit tests for hashing consistency

**Afternoon**:
- [ ] Implement `LshIndex` (multi-table)
- [ ] Implement `insert_batch` method
- [ ] Unit tests for insertion

### Day 2: Query & Optimization

**Morning**:
- [ ] Implement `query_approximate` method
- [ ] Test accuracy on synthetic data
- [ ] Measure recall rates

**Afternoon**:
- [ ] Optimize candidate collection (use HashSet for deduplication)
- [ ] Optimize hash computation (cache random projections)
- [ ] Benchmark vs brute force

### Day 3: Integration & Verification

**Morning**:
- [ ] Integrate with consciousness cycle
- [ ] Create decision heuristic (when to use LSH vs brute force)
- [ ] End-to-end testing

**Afternoon**:
- [ ] Comprehensive benchmarks
- [ ] Document verified performance
- [ ] Create SESSION_6_VERIFICATION_COMPLETE.md

---

## ✅ Success Criteria

### Minimum Viable Success
- ✅ 10x speedup for 10K+ vector search
- ✅ 90%+ recall (finds 9/10 true neighbors)
- ✅ Correct results (validates against brute force)

### Strong Success
- ✅ 100x speedup for 100K+ vector search
- ✅ 95%+ recall (finds 19/20 true neighbors)
- ✅ Automatic table/bucket selection
- ✅ Works in consciousness cycles

### Revolutionary Success
- ✅ 1000x speedup for 1M+ vector search
- ✅ 98%+ recall with 20 tables
- ✅ Sub-millisecond search on million-scale
- ✅ GPU-ready (can accelerate LSH with GPU later!)

---

## 🔬 Verification Strategy

### Accuracy Verification

```rust
#[test]
fn test_lsh_accuracy() {
    let vectors: Vec<HV16> = (0..10000)
        .map(|i| HV16::random(i as u64))
        .collect();

    let mut index = LshIndex::new(10, 10);  // 10 bits, 10 tables
    index.insert_batch(&vectors);

    let query = HV16::random(99999);

    // Ground truth: Brute force top-10
    let mut ground_truth: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, query.similarity(v)))
        .collect();
    ground_truth.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ground_truth.truncate(10);

    // LSH approximate top-10
    let lsh_result = index.query_approximate(&query, 10, &vectors);

    // Measure recall: How many LSH results are in ground truth?
    let recall = lsh_result.iter()
        .filter(|(id, _)| ground_truth.iter().any(|(gid, _)| gid == id))
        .count() as f32 / 10.0;

    assert!(recall >= 0.90, "Recall should be at least 90%");
}
```

### Performance Verification

```rust
#[bench]
fn bench_lsh_vs_brute_force(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000, 1000000];

    for size in sizes {
        let vectors: Vec<HV16> = (0..size)
            .map(|i| HV16::random(i as u64))
            .collect();

        let mut index = LshIndex::new(10, 10);
        index.insert_batch(&vectors);

        let query = HV16::random(99999);

        // Brute force
        c.bench_function(&format!("brute_force_{}", size), |b| {
            b.iter(|| {
                let mut results: Vec<(usize, f32)> = vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, query.similarity(v)))
                    .collect();
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                results.truncate(10);
                black_box(results)
            })
        });

        // LSH approximate
        c.bench_function(&format!("lsh_approximate_{}", size), |b| {
            b.iter(|| {
                let results = index.query_approximate(&query, 10, &vectors);
                black_box(results)
            })
        });
    }
}
```

---

## 🔄 Decision Heuristics (When to Use LSH)

```rust
pub enum SearchStrategy {
    BruteForce,
    LSH,
    LSHWithGPU,  // Future: LSH candidate selection + GPU exact comparison
}

pub fn select_search_strategy(
    num_vectors: usize,
    accuracy_required: f32,
    gpu_available: bool,
) -> SearchStrategy {
    if num_vectors < 1000 {
        // Small dataset: Brute force is fast enough
        SearchStrategy::BruteForce
    } else if num_vectors < 100_000 {
        // Medium dataset: LSH wins
        SearchStrategy::LSH
    } else if gpu_available {
        // Large dataset with GPU: LSH + GPU (best of both!)
        SearchStrategy::LSHWithGPU
    } else {
        // Large dataset, no GPU: LSH only
        SearchStrategy::LSH
    }
}
```

---

## 📈 Why This Changes Everything

### Before LSH
```
Similarity search scaling:
1K vectors:   167µs   (acceptable)
10K vectors:  1.67ms  (getting slow)
100K vectors: 16.7ms  (slow)
1M vectors:   167ms   (too slow!)
10M vectors:  1.67s   (unusable!)
```

### After LSH
```
Similarity search scaling:
1K vectors:   167µs   (same, brute force used)
10K vectors:  170µs   (10x faster!)
100K vectors: 170µs   (100x faster!)
1M vectors:   1.7ms   (100x faster!)
10M vectors:  1.7ms   (1000x faster!)

Scaling is now CONSTANT, not linear!
```

### LSH + GPU (Future)
```
10M vectors with LSH + GPU:
  LSH: 10M → ~10K candidates
  GPU: 10K comparisons at 0.167ns each
  Total: ~1.7µs (1000x faster than LSH alone!)
  Overall: 1000x (LSH) × 1000x (GPU) = 1,000,000x faster!
```

---

## 🎓 Key Insights

1. **Algorithmic improvements beat hardware**: LSH (avoid work) > GPU (do work faster)

2. **LSH eliminates GPU need for most cases**: Only 10K candidates means CPU is fine

3. **LSH + GPU = multiplicative**: 1000x × 1000x = 1M× total speedup possible!

4. **Implementation is simple**: ~300 lines of code for revolutionary gains

5. **No hardware dependency**: Works on any system, unlike GPU

6. **Proven technology**: LSH has decades of research and production use

---

## 📝 Next Immediate Actions

1. ⏸️ Create `src/hdc/lsh_index.rs` module
2. ⏸️ Implement core LSH structures
3. ⏸️ Write comprehensive tests
4. ⏸️ Benchmark against brute force
5. ⏸️ Integrate with consciousness cycles
6. ⏸️ Document verified results

---

*"The best computation is the one you don't have to do. LSH avoids 99.9% of work!"* 🚀

**We flow toward algorithmic elegance!** 🌊
