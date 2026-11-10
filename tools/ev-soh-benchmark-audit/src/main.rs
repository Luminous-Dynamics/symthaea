use ev_soh_benchmark_audit::audit_corpus;
use std::env;
use std::fs;
use std::process::ExitCode;
fn main()->ExitCode{
 let mut schema_path=None;let mut corpus_path=None;let mut args=env::args().skip(1);
 while let Some(arg)=args.next(){match arg.as_str(){
  "--schema"=>schema_path=args.next(),
  "--corpus"=>corpus_path=args.next(),
  _=>{eprintln!("unknown argument: {arg}");return ExitCode::FAILURE;}
 }}
 let (Some(schema_path),Some(corpus_path))=(schema_path,corpus_path) else {
  eprintln!("usage: ev-soh-benchmark-audit --schema PATH --corpus PATH");return ExitCode::FAILURE;
 };
 let schema=match fs::read(&schema_path){Ok(v)=>v,Err(e)=>{eprintln!("cannot read schema {schema_path}: {e}");return ExitCode::FAILURE;}};
 let corpus=match fs::read(&corpus_path){Ok(v)=>v,Err(e)=>{eprintln!("cannot read corpus {corpus_path}: {e}");return ExitCode::FAILURE;}};
 match audit_corpus(&schema,&corpus){
  Ok(report)=>match serde_json::to_string_pretty(&report){Ok(s)=>{println!("{s}");ExitCode::SUCCESS},Err(e)=>{eprintln!("cannot serialize report: {e}");ExitCode::FAILURE}},
  Err(e)=>{eprintln!("SOH benchmark manifest audit failed: {e}");ExitCode::FAILURE}
 }
}
