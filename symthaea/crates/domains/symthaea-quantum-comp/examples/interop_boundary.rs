use symthaea_quantum_comp::IntegrationDeclaration;

fn main() {
    println!("{}", IntegrationDeclaration::local_lab().to_text());
    println!(
        "{}",
        IntegrationDeclaration::mycelix_receipt_request().to_text()
    );
    println!(
        "{}",
        IntegrationDeclaration::external_backend_observation("qasm-export-placeholder").to_text()
    );
}
