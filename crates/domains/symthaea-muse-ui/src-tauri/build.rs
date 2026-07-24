use std::path::PathBuf;
use std::process::Command;

/// Rebuild the sibling `symthaea-muse-ui` frontend into `../dist` before every
/// `cargo build`/`cargo run` of this crate — `frontendDist` is what a bare
/// `cargo run` (no `tauri` CLI) actually loads, `devUrl` only applies under
/// `cargo tauri dev`. `dist/` is gitignored, so a fresh checkout has nothing
/// there until this runs; a stale `dist/` (built before a later CSS/JS
/// change) silently loads mismatched files and renders an unstyled, squished
/// layout instead of erroring — this makes that bug class impossible.
fn rebuild_frontend() {
    let ui_crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .expect("symthaea-muse-ui crate root must exist alongside src-tauri");

    println!("cargo:warning=rebuilding Muse frontend (trunk build --release)...");
    let status = Command::new("trunk")
        .args(["build", "--release"])
        .current_dir(&ui_crate_dir)
        // Cargo injects `CARGO_ENCODED_RUSTFLAGS` into every build script's
        // environment, set to *this crate's own* resolved flags — for
        // src-tauri (a native target) that's .cargo/config.toml's
        // `-Ctarget-cpu=native -Clink-arg=-fuse-ld=mold`. That env var sits
        // at Cargo's top precedence tier and applies unconditionally to
        // *any* target a cargo invocation builds, so it leaks straight into
        // trunk's own nested `cargo build --target wasm32-unknown-unknown`
        // and rust-lld rejects the native-only `-fuse-ld=mold` outright
        // ("lld: error: unknown argument"). Confirmed by dumping the build
        // script's actual env (not assumed): CARGO_ENCODED_RUSTFLAGS was
        // present verbatim. Neither `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS`
        // nor plain `RUSTFLAGS` set via `.env()` help on their own —
        // `CARGO_ENCODED_RUSTFLAGS` outranks both. Removing it here lets
        // trunk's inner cargo fall through to `RUSTFLAGS` instead.
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env("RUSTFLAGS", "-C debuginfo=2")
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!(
            "trunk build --release failed (exit {s}) in {}; the Muse desktop shell needs a \
             working frontend build to run, not just a build-time nicety. Run `nix develop \
             .#tauri` for the toolchain and retry.",
            ui_crate_dir.display()
        ),
        Err(e) => panic!("could not run `trunk` (is it on PATH? use `nix develop .#tauri`): {e}"),
    }
}

fn main() {
    rebuild_frontend();
    tauri_build::build()
}
