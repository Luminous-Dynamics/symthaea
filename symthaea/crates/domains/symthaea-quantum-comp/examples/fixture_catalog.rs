use symthaea_quantum_comp::fixture_catalog;

fn main() {
    for fixture in fixture_catalog() {
        println!("{}", fixture.to_text());
    }
}
