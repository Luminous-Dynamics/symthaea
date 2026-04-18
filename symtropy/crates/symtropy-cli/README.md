# symtropy-cli

Project scaffolding for Symtropy.

## Install

```bash
cargo install symtropy-cli
```

This installs the `symtropy` binary.

## Usage

```bash
symtropy new my-game                          # default template: 3d-research
symtropy new my-puzzle --template 4d-research
symtropy new my-platformer --template 2d-game
symtropy templates                            # list templates
symtropy --help
```

After running:

```bash
cd my-game
cargo run --release
```

Window opens with the demo scene. Press **F1** for the dev console (Φ
Inspector + Scene controls).

## Templates

| Name | What |
|---|---|
| `3d-research` | 3D scene, one swinging pendulum, PBR rendering, dev console |
| `4d-research` | 4D scene, three bobs at W = -1/0/+1, hyperplane slicing with `[` / `]` keys |
| `2d-game` | 2D scene, one bouncing sprite, click-to-kick |

Templates are embedded in the binary — no network access needed at scaffold
time. They use crates.io for `symtropy-bevy` and git URLs for the still-
unpublished `symtropy-bevy-scene` + `symtropy-devconsole` (the templates'
generated `Cargo.toml` will need a one-line edit to switch to crates.io once
those crates land).

## License

Apache-2.0 OR MIT.
