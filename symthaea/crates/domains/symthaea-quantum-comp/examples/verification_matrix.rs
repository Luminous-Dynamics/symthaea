use symthaea_quantum_comp::current_verification_matrix;

fn main() {
    let matrix = current_verification_matrix();
    println!("{}", matrix.to_markdown());
}
