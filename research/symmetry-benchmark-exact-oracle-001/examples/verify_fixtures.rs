use symmetry_benchmark_exact_oracle::run_frozen_oracle;

fn main() {
    match run_frozen_oracle() {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
        Err(error) => {
            eprintln!("symmetry benchmark oracle rejected fixture: {error}");
            std::process::exit(2);
        }
    }
}
