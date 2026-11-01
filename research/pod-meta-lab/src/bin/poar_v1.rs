use pod_meta_lab::poar::run_poar_v1;

fn main() {
    let report = run_poar_v1();

    for case in &report.case_results {
        let status = if case.passed { "PASS" } else { "FAIL" };
        println!("{status}\t{}\t{}", case.id, case.observation);
    }

    println!(
        "POAR-v1 conformance: {}/{} passed; {} failed",
        report.passed_count(),
        report.case_count(),
        report.failed_count()
    );

    if report.failed_count() != 0 {
        std::process::exit(1);
    }
}
