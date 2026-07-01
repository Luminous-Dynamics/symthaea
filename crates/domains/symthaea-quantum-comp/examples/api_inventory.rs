//! Print the alpha.10 API inventory and stability catalog.

use symthaea_quantum_comp::current_api_inventory;

fn main() {
    let inventory = current_api_inventory();
    println!("{}", inventory.to_markdown());
}
