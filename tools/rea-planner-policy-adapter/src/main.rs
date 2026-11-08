use rea_planner_policy_adapter::run_embedded_pilot;

fn main() {
    match run_embedded_pilot() {
        Ok(result) => match serde_json::to_string_pretty(&result) {
            Ok(output) => println!("{output}"),
            Err(error) => {
                eprintln!("rea-planner-policy-adapter: serialization failed: {error}");
                std::process::exit(2);
            }
        },
        Err(error) => {
            eprintln!("rea-planner-policy-adapter: {error}");
            std::process::exit(1);
        }
    }
}
