# Shell Completions

Tab completion scripts for various shells.

## Supported Shells

- Bash
- Zsh
- Fish
- PowerShell

## Installation

### Bash
```bash
source completions/symthaea.bash
# Or add to ~/.bashrc
```

### Zsh
```bash
source completions/symthaea.zsh
# Or copy to fpath
```

### Fish
```bash
cp completions/symthaea.fish ~/.config/fish/completions/
```

## Generation

Completions are generated from CLI definitions:

```bash
cargo run -- --generate-completions bash > completions/symthaea.bash
cargo run -- --generate-completions zsh > completions/symthaea.zsh
cargo run -- --generate-completions fish > completions/symthaea.fish
```
