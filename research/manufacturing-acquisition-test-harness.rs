//! Standalone test harness for the dependency-free EIG selector module.
//! This does not replace the full symthaea-core Cargo build; it lets the exact
//! selector source and its unit tests execute while workspace resolution is repaired.
pub mod hdc {
    pub mod conjecture_engine {
        #[derive(Clone, Debug)]
        pub enum BinOp {
            Pow,
        }

        #[derive(Clone, Debug)]
        pub enum Expr {
            Var(String),
            Const(f64),
            BinOp(BinOp, Box<Expr>, Box<Expr>),
        }

        impl Expr {
            pub fn eval(&self, variables: &[(&str, f64)]) -> f64 {
                match self {
                    Self::Var(name) => variables
                        .iter()
                        .find(|(candidate, _)| *candidate == name)
                        .map(|(_, value)| *value)
                        .unwrap_or(f64::NAN),
                    Self::Const(value) => *value,
                    Self::BinOp(BinOp::Pow, left, right) => {
                        left.eval(variables).powf(right.eval(variables))
                    }
                }
            }
        }
    }
}

#[path = "../crates/core/symthaea-core/src/hdc/conjecture_engine/experiment_selection.rs"]
mod experiment_selection;
