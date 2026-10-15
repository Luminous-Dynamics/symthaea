use symthaea_crime_economics_sim::{TSV_HEADER, run_matrix};

fn main() {
    println!("{TSV_HEADER}");
    for row in run_matrix() {
        println!("{}", row.to_tsv());
    }
}
