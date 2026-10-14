use symthaea_crime_economics_sim::viability::{rows, TSV_HEADER};

fn main() {
    println!("{TSV_HEADER}");
    for row in rows() {
        println!("{}", row.to_tsv());
    }
}
