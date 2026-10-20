#[path = "../../../../crates/bridges/symthaea-physics-bridge/examples/verify_invariants_formal.rs"]
mod formal_verifier;

fn main() -> std::process::ExitCode {
    formal_verifier::run_verifier()
}
