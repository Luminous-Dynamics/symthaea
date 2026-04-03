// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! IB International Baccalaureate content.
//! Used in 160 countries. Focus: inquiry-based, Theory of Knowledge, global perspectives.

use crate::caps_content::TopicContent;

/// IB Mathematics: Analysis & Approaches SL — Functions
pub struct IbMathFunctions;
impl TopicContent for IbMathFunctions {
    fn explanation(&self) -> String {
        "A function maps each input to exactly one output: f(x) = 2x + 3. \
         Domain: all valid inputs. Range: all possible outputs. \
         Key types: linear (f(x)=mx+c), quadratic (f(x)=ax²+bx+c), \
         exponential (f(x)=a·bˣ). \
         Transformations: f(x)+k shifts up, f(x-h) shifts right, -f(x) reflects.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "f(x) = x² - 4x + 3. Find the vertex.\nComplete the square: f(x) = (x-2)² - 4 + 3 = (x-2)² - 1\nVertex: (2, -1)\nAxis of symmetry: x = 2".to_string(),
            1 => "Find the inverse of f(x) = 3x - 7.\nLet y = 3x - 7\nSwap x,y: x = 3y - 7\nSolve: y = (x+7)/3\nf⁻¹(x) = (x+7)/3".to_string(),
            _ => "Describe the transformation: g(x) = f(x-3) + 2\nf(x-3): shift RIGHT by 3 (replace x with x-3)\n+2: shift UP by 2\nThe graph moves 3 right and 2 up.".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "If f(x) = 2x + 1, find f(3).\n7\nf(3) = 2(3)+1 = 7\n5, 9, 6\nSubstitute x=3 into the formula.\n2×3+1=?".to_string(),
            301..=600 => "Find the zeros of f(x) = x² - 5x + 6.\nx=2, x=3\nFactor: (x-2)(x-3)=0 → x=2 or x=3.\n-2,-3 and 1,6\nFactor the quadratic.\nProduct=6, sum=5: 2 and 3.".to_string(),
            _ => "Population grows exponentially: P(t) = 500·1.03ᵗ. When does it double?\nt ≈ 23.4 years\n1000 = 500·1.03ᵗ → 2 = 1.03ᵗ → t = ln2/ln1.03 ≈ 23.4\n33, 50, 10\nSet P(t) = 1000, solve for t.\nln(2)/ln(1.03).".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "f(a) means replace every x with a.".to_string(), 2 => "Zeros: set f(x)=0 and solve.".to_string(), _ => "For exponential doubling: t = ln(2)/ln(growth rate).".to_string() } }
    fn misconception(&self) -> String { "WRONG: f(x+3) shifts the graph RIGHT by 3.\nRIGHT: f(x+3) shifts LEFT by 3. f(x-3) shifts RIGHT.\nWHY: Think: f(x+3)=0 when x=-3 (the zero moved LEFT). Counterintuitive but consistent!".to_string() }
    fn vocabulary(&self) -> String { "domain: Set of all valid inputs | f(x)=1/x has domain x≠0.\nrange: Set of all outputs | f(x)=x² has range [0,∞).\ninverse function: Reverses the original | If f(2)=5, then f⁻¹(5)=2.".to_string() }
    fn flashcard(&self) -> String { "Domain of f(x)=√x? | x ≥ 0 (can't square-root a negative)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Sketch f(x)=|x-2|-1 and find its vertex and zeros.\nVertex: (2,-1). Zeros: x=1 and x=3.\n3\n|x-2|=0 at x=2, shifted down 1. Zeros: |x-2|=1 → x=1,3.".to_string() }
}
