use home_service_metrology_study::validate_tracked;
use std::process::ExitCode;
fn main()->ExitCode{match validate_tracked(){Ok(n)=>{println!("suite=home_service_metrology_study_v0_1");println!("cases={n}");println!("result=all_known_answers_matched");println!("claim_ceiling=study_design_contract_and_synthetic_gate_only");ExitCode::SUCCESS},Err(e)=>{eprintln!("result=invalid");eprintln!("{e}");ExitCode::FAILURE}}}
