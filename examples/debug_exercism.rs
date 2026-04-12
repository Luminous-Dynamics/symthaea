use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::code_generator::{CodeGenerator, CodeContext};
use symthaea::language::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use symthaea::language::code_parser::EntityKind;
use symthaea::mind::structured_thought::EpistemicStatus;
use std::collections::HashMap;

fn gen(generator: &CodeGenerator, name: &str, purpose: &str, sig: &str) {
    let spec = CodeSpec {
        language: "rust".into(), name: name.into(), purpose: purpose.into(),
        purpose_hv: None, signature: Some(sig.into()),
        constraints: vec![], examples: vec![],
        epistemic_status: EpistemicStatus::Certain, metadata: HashMap::new(),
    };
    let intent = CodeIntent::Create {
        target: CodeTarget { kind: EntityKind::Function, name: name.into(), path: None, language: Some("rust".into()), hv: None },
        spec,
    };
    let result = generator.generate(&intent, &CodeContext::default());
    let src = if let Some(pos) = result.source.find("#[cfg(test)]") { &result.source[..pos] } else { &result.source };
    println!("=== {} ===\n{}\n", name, src.trim());
}

fn main() {
    let g = CodeGenerator::new(CodeHDEncoder::new(512));
    gen(&g, "collatz", "Given a positive integer, return the number of steps it takes to reach 1 according to the rules of the Collatz Conjecture", "pub fn collatz(n: u64) -> Option<u64>");
    gen(&g, "reverse", "Reverse a given string", "pub fn reverse(input: &str) -> String");
    gen(&g, "square_of_sum", "Find the square of the sum of the first N natural numbers", "pub fn square_of_sum(n: u32) -> u32");
    gen(&g, "sum_of_squares", "Find the sum of the squares of the first N natural numbers", "pub fn sum_of_squares(n: u32) -> u32");
    gen(&g, "difference", "Find the difference between the square of the sum and the sum of the squares", "pub fn difference(n: u32) -> u32");
    gen(&g, "total", "Calculate the total number of grains on the chessboard", "pub fn total() -> u64");
    gen(&g, "square", "Calculate grains of wheat on a chessboard square", "pub fn square(s: u32) -> u64");
    gen(&g, "nth", "Given a number n, determine what the nth prime is", "pub fn nth(n: u32) -> u32");
}
