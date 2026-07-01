# Beta Transition

Alpha.10 is not beta.

It is a preparation release that makes the release surface easier to inspect before a future beta.

Beta should require:

- local `cargo test --all-features` in a real Rust environment
- an API freeze review
- removal or marking of unstable surfaces
- independent method review
- documented external backend adapter boundaries
- stable report fixtures and replay expectations

Command:

`cargo run --bin symthaea-quantum-comp -- beta`
