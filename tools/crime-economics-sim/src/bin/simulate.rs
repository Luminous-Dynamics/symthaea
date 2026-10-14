use symthaea_crime_economics_sim::{run_matrix, TSV_HEADER};

fn main() {
    println!("{TSV_HEADER}");
    for row in run_matrix() {
        println!("{row}");
    }
}
