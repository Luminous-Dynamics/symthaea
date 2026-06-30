pub struct LeanBridge;

impl LeanBridge {
    pub fn emit_admissibility_stub(tuple: &[u64]) -> String {
        format!(
            "def candidate_tuple : AdmissibleTuple {} := {{ tuple := {:?}, admissible := by sorry }}",
            tuple.len(),
            tuple.iter().cloned().collect::<Vec<_>>()
        )
    }
}
