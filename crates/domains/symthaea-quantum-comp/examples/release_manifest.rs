//! Print the alpha.10 release manifest and blocked claims.

use symthaea_quantum_comp::current_release_manifest;

fn main() {
    let manifest = current_release_manifest();
    println!("{}", manifest.to_markdown());
}
