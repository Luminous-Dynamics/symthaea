use symthaea_quantum_comp::alpha9_to_alpha10_migration;

fn main() {
    let guide = alpha9_to_alpha10_migration();
    println!("{}", guide.to_markdown());
}
