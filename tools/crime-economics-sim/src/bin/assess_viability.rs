use symthaea_crime_economics_sim::viability::{TSV_HEADER, rows};

fn main() {
    println!("{TSV_HEADER}");
    for row in rows() {
        println!("{}", row.to_tsv());
    }
}
