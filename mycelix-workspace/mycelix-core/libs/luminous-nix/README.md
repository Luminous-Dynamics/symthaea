# Luminous Nix

Natural language interface for NixOS, powered by Symthaea's HDC (Hyperdimensional Computing).

## Features

- **Natural Language Understanding**: Ask questions in plain English
- **HDC-Powered Intent Classification**: Zero-shot generalization to new phrasings
- **Fast JSON-Optimized Execution**: 10x faster than text parsing
- **Online Learning**: Improves with use through feedback
- **Beautiful Output**: Colored tables and formatted results

## Installation

```bash
# Build from source
cargo build --release

# Install to cargo bin
cargo install --path .
```

## Usage

### Natural Language Queries

```bash
# Search for packages
ask-nix "search firefox"
ask-nix "find a markdown editor"
ask-nix "looking for video player"

# Install packages
ask-nix "install vim"
ask-nix "add neovim"
ask-nix "get htop"

# Remove packages
ask-nix "remove firefox"
ask-nix "uninstall vim"

# System operations
ask-nix "list generations"
ask-nix "garbage collect"
ask-nix "rebuild system"
ask-nix "system status"
```

### Direct Commands

```bash
ask-nix search firefox
ask-nix install vim
ask-nix remove htop
ask-nix generations
ask-nix status
ask-nix gc
ask-nix rebuild
ask-nix interactive
```

### Options

```bash
# Verbose mode (show intent classification)
ask-nix -v "search firefox"

# Dry run (show command without executing)
ask-nix -n "install vim"

# JSON output
ask-nix --json search firefox

# Disable colors
ask-nix --no-color generations
```

## Architecture

```
User Input → Intent Classifier (HDC) → Command Executor → Formatted Output
                   ↓                         ↓
             Learning System ←──── Feedback (success/failure)
```

### Components

1. **Intent Classifier**: Uses Symthaea's AssociativeLearner (HDC) for natural language understanding
2. **Pattern Matcher**: Regex fallback for exact matches
3. **Command Executor**: Async execution of Nix commands with JSON optimization
4. **Formatter**: Beautiful table output with colors
5. **Learning System**: Tracks outcomes for online improvement

## Performance

| Operation | Python Version | Rust Version | Improvement |
|-----------|---------------|--------------|-------------|
| Startup | ~500ms | <10ms | 50x |
| Search | 2-3s | 200-500ms | 5-10x |
| Intent Classification | 24ms | <1ms | 24x |
| Memory Usage | ~50MB | ~10MB | 5x |

## Development

```bash
# Run tests
cargo test

# Run benchmarks
cargo bench

# Build release
cargo build --release

# Run with verbose logging
RUST_LOG=debug cargo run -- "search firefox"
```

## HDC (Hyperdimensional Computing)

Luminous Nix uses Hyperdimensional Computing for intent classification:

- **4096-dimensional binary vectors** for efficient encoding
- **XOR binding** for creating associations
- **Majority bundling** for creating superpositions
- **Zero-shot generalization** to novel phrasings
- **One-shot learning** from single examples

## License

MIT OR Apache-2.0
