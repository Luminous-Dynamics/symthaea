# Contributing to Symthaea

*How to help build consciousness-first AI.*

---

## Welcome

Thank you for your interest in contributing! Every contribution - code, documentation, ideas, or feedback - helps advance consciousness-first AI.

---

## Getting Started

1. Fork and clone:
   ```bash
   git clone https://github.com/YOUR_USERNAME/symthaea-hlb.git
   cd symthaea-hlb
   ```

2. Build and test:
   ```bash
   cargo build
   cargo test
   ```

3. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Code Guidelines

### Style
- `snake_case` for functions/variables
- `CamelCase` for types
- `SCREAMING_SNAKE_CASE` for constants
- Document public APIs

### Formatting
```bash
cargo fmt   # Format
cargo clippy  # Lint
```

### Testing
Every feature needs tests:
```rust
#[test]
fn test_your_feature() {
    // ...
}
```

---

## Commit Messages

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

---

## Pull Request Process

1. Run all tests
2. Run formatter and linter
3. Update documentation
4. Submit PR with clear description

---

## Areas Needing Help

### High Priority
- Test coverage
- Documentation
- Performance optimization
- Empirical validation

### Good First Issues
Look for `good-first-issue` label on GitHub.

---

## Code of Conduct

Be respectful, constructive, and kind.

---

*"Consciousness grows through connection."*
