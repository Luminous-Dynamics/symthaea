// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 7 Mathematics — CAPS Senior Phase.
//! Critical transition year: algebra introduction, exponents, geometry.

use super::TopicContent;

pub(crate) struct Gr7AlgebraIntro;
impl TopicContent for Gr7AlgebraIntro {
    fn explanation(&self) -> String {
        "Algebra uses letters (variables) to represent unknown numbers. \
         Instead of saying 'a number plus 5 equals 12', we write x + 5 = 12. \
         To find x, we do the opposite operation: x = 12 - 5 = 7. \
         This is called 'solving the equation'. The letter x can be any number \
         — our job is to find which number makes the equation true.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Solve: x + 8 = 15\nTo isolate x, subtract 8 from both sides.\nx = 15 - 8\nx = 7\nCheck: 7 + 8 = 15 ✓".to_string(),
            1 => "Solve: 3x = 21\nTo isolate x, divide both sides by 3.\nx = 21 ÷ 3\nx = 7\nCheck: 3(7) = 21 ✓".to_string(),
            _ => "Simplify: 4a + 3b - 2a + b\nGroup like terms: (4a - 2a) + (3b + b)\n= 2a + 4b".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Solve: x - 4 = 9\nx = 13\nAdd 4 to both sides: x = 9 + 4 = 13.\n5, -13, 36\nTo undo subtraction, add.\n9 + 4 = ?".to_string(),
            301..=600 => "Solve: 5x + 3 = 28\nx = 5\nSubtract 3: 5x = 25. Divide by 5: x = 5.\n6, 25, 3\nFirst undo +3, then undo ×5.\n28 - 3 = 25, then 25 ÷ 5 = ?".to_string(),
            _ => "The sum of three consecutive numbers is 54. Find them.\n17, 18, 19\nLet first = n. Then n + (n+1) + (n+2) = 54. 3n + 3 = 54. 3n = 51. n = 17.\n16, 17, 18 and 18, 19, 20\nConsecutive means one after another.\nLet first = n, second = n+1, third = n+2.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "To solve, do the opposite operation on both sides.".to_string(), 2 => "Addition ↔ Subtraction, Multiplication ↔ Division.".to_string(), _ => "x + 5 = 12: subtract 5 from both sides → x = 7.".to_string() }
    }
    fn misconception(&self) -> String { "WRONG: In 3x, the 3 and x are separate — 3x means 3 + x.\nRIGHT: 3x means 3 × x (3 times x). If x = 4, then 3x = 12, not 7.\nWHY: In algebra, a number next to a letter means multiply. The × sign is dropped to avoid confusion with the variable x.".to_string() }
    fn vocabulary(&self) -> String { "variable: A letter representing an unknown number | In x + 5 = 12, x is the variable.\nequation: A statement that two expressions are equal | 3x + 2 = 14 is an equation.\nlike terms: Terms with the same variable | 4a and 2a are like terms; 4a and 3b are not.".to_string() }
    fn flashcard(&self) -> String { "Solve: 2x = 18 | x = 9 (divide both sides by 2)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "A rectangle has perimeter 38 cm. The length is x cm and width is (x-5) cm. Find x.\nx = 12\n3\n2(x + x-5) = 38 → 2(2x-5) = 38 → 4x-10 = 38 → 4x = 48 → x = 12.".to_string() }
}

pub(crate) struct Gr7Exponents;
impl TopicContent for Gr7Exponents {
    fn explanation(&self) -> String {
        "An exponent tells you how many times to multiply a number by itself. \
         2³ = 2 × 2 × 2 = 8. The small number (3) is the exponent. The big number (2) is the base. \
         Special cases: any number to the power 0 is 1 (e.g., 5⁰ = 1). \
         Any number to the power 1 is itself (e.g., 7¹ = 7).".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Calculate: 3⁴\n3 × 3 = 9\n9 × 3 = 27\n27 × 3 = 81\n3⁴ = 81".to_string(),
            1 => "Write 32 as a power of 2.\n2¹=2, 2²=4, 2³=8, 2⁴=16, 2⁵=32\n32 = 2⁵".to_string(),
            _ => "Simplify: 2³ × 2²\nSame base → add exponents\n2³⁺² = 2⁵ = 32".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Calculate: 5²\n25\n5 × 5 = 25\n10, 52, 125\n5² means 5 multiplied by itself.\n5 × 5 = ?".to_string(),
            301..=600 => "Which is larger: 2⁵ or 5²?\n2⁵ = 32\n2⁵ = 32, 5² = 25, so 2⁵ is larger.\n5², Equal, Cannot compare\nCalculate both.\n2×2×2×2×2 vs 5×5".to_string(),
            _ => "Simplify: 3² × 3³ ÷ 3⁴\n3\nAdd then subtract exponents: 3^(2+3-4) = 3¹ = 3.\n9, 27, 81\nSame base: multiply→add exponents, divide→subtract.\n2+3=5, 5-4=1, so 3¹=3.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "The exponent says how many times to multiply.".to_string(), 2 => "Same base: multiply→add exponents, divide→subtract.".to_string(), _ => "2³ means 2×2×2 (three 2s multiplied together).".to_string() }
    }
    fn misconception(&self) -> String { "WRONG: 2³ = 2 × 3 = 6.\nRIGHT: 2³ = 2 × 2 × 2 = 8. The exponent means repeated multiplication, not regular multiplication.\nWHY: The tiny number tells you HOW MANY TIMES to multiply, not what to multiply by.".to_string() }
    fn vocabulary(&self) -> String { "base: The number being multiplied | In 2³, the base is 2.\nexponent/power: How many times the base is multiplied | In 2³, the exponent is 3.\nsquared: Raised to the power 2 | 5² = 25 (five squared).".to_string() }
    fn flashcard(&self) -> String { "What is 10³? | 1 000 (10×10×10)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Write 81 as a power of 3.\n3⁴\n2\n3¹=3, 3²=9, 3³=27, 3⁴=81.".to_string() }
}

pub(crate) struct Gr7MathFallbackTopic;
impl TopicContent for Gr7MathFallbackTopic {
    fn explanation(&self) -> String { "Grade 7 introduces algebra, exponents, and formal geometry. These are the foundations for high school maths.".to_string() }
    fn worked_example(&self, _i: usize) -> String { "Find the area of a triangle: base=10cm, height=6cm.\nA = ½ × b × h = ½ × 10 × 6 = 30 cm²".to_string() }
    fn practice_problem(&self, _d: u16) -> String { "Triangle: base 14cm, height 8cm. Area?\n56 cm²\nA=½×14×8=56\n112, 22, 28\nA=½×base×height.\n½×14=7, 7×8=56".to_string() }
    fn hint(&self, l: u8) -> String { match l { 1 => "Triangle area = half of rectangle area.".to_string(), _ => "A = ½ × base × height".to_string() } }
    fn misconception(&self) -> String { "WRONG: Triangle area = base × height.\nRIGHT: Triangle area = ½ × base × height.\nWHY: A triangle is half of a rectangle with the same base and height.".to_string() }
    fn vocabulary(&self) -> String { "base: The bottom side of a triangle | The side the triangle 'sits' on.\nheight: Perpendicular distance from base to opposite vertex | Must be at 90° to the base.".to_string() }
    fn flashcard(&self) -> String { "Area of triangle: base=20, height=7? | 70 cm² (½×20×7)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Prove 2⁴ + 2⁴ = 2⁵.\n2⁴+2⁴ = 16+16 = 32 = 2⁵\n3\nCalculate both sides separately.".to_string() }
}
