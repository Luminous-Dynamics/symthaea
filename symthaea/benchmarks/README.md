# Benchmarks

External benchmark datasets for validation.

## external/tuebingen/

Tübingen cause-effect pairs dataset for causal discovery benchmarking.

- 100 cause-effect pairs
- Ground truth labels
- Used to validate causal inference algorithms

### Source

https://webdav.tuebingen.mpg.de/cause-effect/

### Usage

```python
# Load pairs
from scripts.hdc_causal_discovery import load_tuebingen_pair

pair = load_tuebingen_pair("pair0001")
```

## Python Benchmark Scripts

See `scripts/benchmark_*.py` for Python-based benchmarks.

## Rust Benchmarks

Criterion benchmarks are in `benches/`:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench consciousness
```
