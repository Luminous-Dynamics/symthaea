// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cambridge IGCSE content.
//! Dominant in Africa, Asia, Middle East. Ages 14-16 (equivalent to Gr9-10).

use crate::caps_content::TopicContent;

/// Cambridge IGCSE Mathematics — Number & Algebra
pub struct IgcseMathAlgebra;
impl TopicContent for IgcseMathAlgebra {
    fn explanation(&self) -> String {
        "IGCSE algebra builds on equation solving to include simultaneous equations, \
         quadratics, and inequalities. Simultaneous equations have TWO unknowns (x and y) \
         and need TWO equations to solve. Methods: substitution or elimination. \
         Quadratic formula: x = (-b ± √(b²-4ac)) / 2a for ax²+bx+c=0.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Solve simultaneously: 2x + y = 7 and x - y = 2\nAdd equations: 3x = 9 → x = 3\nSubstitute: 2(3)+y=7 → y=1\nSolution: x=3, y=1".to_string(),
            1 => "Solve: x² + 3x - 10 = 0\nFactor: (x+5)(x-2) = 0\nx = -5 or x = 2".to_string(),
            _ => "Solve using the quadratic formula: 2x² - 5x + 1 = 0\na=2, b=-5, c=1\nx = (5 ± √(25-8))/4 = (5 ± √17)/4\nx ≈ 2.28 or x ≈ 0.22".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Solve: 3x + 4 = 19\nx = 5\n3x=15, x=5.\n3, 15, 23\nSubtract 4, then divide by 3.\n19-4=15, 15÷3=5.".to_string(),
            301..=600 => "Solve simultaneously: x+y=8 and 2x-y=1\nx=3, y=5\nAdd: 3x=9→x=3. Substitute: y=8-3=5.\n(1,7), (4,4), (5,3)\nElimination: add to remove y.\nx+y=8 and 2x-y=1 → 3x=9.".to_string(),
            _ => "The sum of two numbers is 20. Their product is 96. Find them.\n8 and 12\nLet x+y=20, xy=96. y=20-x. x(20-x)=96 → x²-20x+96=0 → (x-8)(x-12)=0.\n6,14 and 4,16\nForm a quadratic from the conditions.\nx+y=20 and xy=96 → x²-20x+96=0.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "For simultaneous: try adding or subtracting to eliminate one variable.".to_string(), 2 => "Quadratic: try factoring first. If not factorable, use the formula.".to_string(), _ => "Discriminant b²-4ac: positive=2 roots, zero=1 root, negative=no real roots.".to_string() } }
    fn misconception(&self) -> String { "WRONG: x² = 9 means x = 3.\nRIGHT: x² = 9 means x = 3 OR x = -3. Don't forget the negative root!\nWHY: Both 3² and (-3)² equal 9. Square roots always have two solutions.".to_string() }
    fn vocabulary(&self) -> String { "simultaneous equations: Two equations with two unknowns solved together.\nquadratic formula: x = (-b±√(b²-4ac))/2a for ax²+bx+c=0.\ndiscriminant: b²-4ac — determines number of real roots.".to_string() }
    fn flashcard(&self) -> String { "Discriminant of 2x²+3x+5=0? | 9-40=-31 (negative → no real roots)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Prove that x²+4x+5 > 0 for all real x.\nComplete square: (x+2)²+1. Since (x+2)²≥0, then (x+2)²+1≥1>0.\n3\nComplete the square to show minimum value is positive.".to_string() }
}
