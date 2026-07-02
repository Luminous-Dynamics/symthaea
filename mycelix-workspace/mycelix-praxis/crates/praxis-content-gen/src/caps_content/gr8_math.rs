// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 8 Mathematics — CAPS Senior Phase.
//! Topics: algebraic expressions, Pythagorean theorem, linear equations & graphs.

use super::TopicContent;

pub(crate) struct Gr8Algebra;
impl TopicContent for Gr8Algebra {
    fn explanation(&self) -> String {
        "In Grade 8, algebra gets more powerful. You learn to multiply expressions: \
         (x + 3)(x + 2) = x² + 2x + 3x + 6 = x² + 5x + 6. \
         This is called FOIL: First, Outer, Inner, Last. \
         You also learn to factorise: x² + 5x + 6 = (x + 3)(x + 2). \
         Factorising is the reverse of expanding — it finds the brackets.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Expand: (x + 4)(x + 3)\nFirst: x × x = x²\nOuter: x × 3 = 3x\nInner: 4 × x = 4x\nLast: 4 × 3 = 12\n= x² + 3x + 4x + 12 = x² + 7x + 12".to_string(),
            1 => "Factorise: x² + 8x + 15\nFind two numbers that multiply to 15 and add to 8.\n3 × 5 = 15 and 3 + 5 = 8 ✓\n= (x + 3)(x + 5)".to_string(),
            _ => "Solve: 2(x - 3) = 14\nExpand: 2x - 6 = 14\nAdd 6: 2x = 20\nDivide by 2: x = 10".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Expand: (x + 2)(x + 5)\nx² + 7x + 10\nFOIL: x²+5x+2x+10 = x²+7x+10\nx²+7x, x²+10, 2x²+7x+10\nUse FOIL.\nFirst×First, then Outer, Inner, Last.".to_string(),
            301..=600 => "Factorise: x² + 9x + 20\n(x + 4)(x + 5)\nFactors of 20 that add to 9: 4+5=9 ✓\n(x+2)(x+10), (x+1)(x+20), (x+9)(x+1)\nFind two numbers: multiply to 20, add to 9.\n4 × 5 = 20 and 4 + 5 = 9".to_string(),
            _ => "Solve: 3(2x + 1) - 2(x - 4) = 23\nx = 3\nExpand: 6x+3-2x+8=23 → 4x+11=23 → 4x=12 → x=3.\n5, 2, 7\nExpand brackets first, collect like terms.\n6x+3-2x+8 = 4x+11.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "FOIL: First, Outer, Inner, Last.".to_string(), 2 => "To factorise x²+bx+c, find two numbers that multiply to c and add to b.".to_string(), _ => "Example: x²+7x+12. Factors of 12 that add to 7: 3 and 4. So (x+3)(x+4).".to_string() }
    }
    fn misconception(&self) -> String { "WRONG: (x+3)² = x² + 9.\nRIGHT: (x+3)² = (x+3)(x+3) = x² + 6x + 9. Don't forget the middle term!\nWHY: Squaring a bracket means multiplying it by itself, not squaring each term separately.".to_string() }
    fn vocabulary(&self) -> String { "expand: Multiply out brackets | (x+3)(x+2) = x²+5x+6.\nfactorise: Write as a product of brackets | x²+5x+6 = (x+3)(x+2).\nFOIL: Method for expanding two brackets — First, Outer, Inner, Last.".to_string() }
    fn flashcard(&self) -> String { "Factorise: x² + 10x + 24 | (x+4)(x+6) because 4×6=24 and 4+6=10".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Prove that (n+1)² - n² = 2n + 1 for any integer n.\nExpand: n²+2n+1-n² = 2n+1 ✓\n3\nExpand (n+1)² first, then subtract n².".to_string() }
}

pub(crate) struct Gr8Pythagoras;
impl TopicContent for Gr8Pythagoras {
    fn explanation(&self) -> String {
        "The Pythagorean theorem says: in a right-angled triangle, \
         the square of the hypotenuse (longest side) equals the sum of squares of the other two sides. \
         Formula: a² + b² = c², where c is the hypotenuse. \
         Example: if the two shorter sides are 3cm and 4cm, then c² = 9 + 16 = 25, so c = 5cm. \
         This only works for right-angled triangles!".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Find the hypotenuse: sides 6cm and 8cm.\nc² = 6² + 8² = 36 + 64 = 100\nc = √100 = 10cm".to_string(),
            1 => "Find the missing side: hypotenuse 13cm, one side 5cm.\na² + 5² = 13²\na² = 169 - 25 = 144\na = √144 = 12cm".to_string(),
            _ => "Is a triangle with sides 7, 24, 25 right-angled?\nCheck: 7² + 24² = 49 + 576 = 625\n25² = 625\n625 = 625 ✓ Yes, it's right-angled!".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Find the hypotenuse: sides 3cm and 4cm.\n5cm\nc²=9+16=25, c=5.\n7, 12, 25\nSquare both sides, add, take square root.\n3²=9, 4²=16, 9+16=25, √25=5.".to_string(),
            301..=600 => "A ladder 10m long leans against a wall. The foot is 6m from the wall. How high does it reach?\n8m\n6²+h²=10² → h²=100-36=64 → h=8m.\n4, 16, 14\nThe ladder is the hypotenuse.\n10²-6²=100-36=64, √64=8.".to_string(),
            _ => "A rectangular field is 40m by 30m. How long is the diagonal?\n50m\n40²+30²=1600+900=2500, √2500=50.\n35, 70, 10\nThe diagonal is the hypotenuse of a right triangle.\n40²=1600, 30²=900, 1600+900=?.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "The hypotenuse is ALWAYS the longest side, opposite the right angle.".to_string(), 2 => "a² + b² = c², where c is the hypotenuse.".to_string(), _ => "To find a shorter side: rearrange to a² = c² - b².".to_string() }
    }
    fn misconception(&self) -> String { "WRONG: a + b = c (just add the sides).\nRIGHT: a² + b² = c² (add the SQUARES). 3+4=7, but 3²+4²=25, and √25=5.\nWHY: The theorem is about areas of squares on each side, not the sides themselves.".to_string() }
    fn vocabulary(&self) -> String { "hypotenuse: The longest side of a right triangle, opposite the right angle | In a 3-4-5 triangle, 5 is the hypotenuse.\nPythagorean theorem: a² + b² = c² for right-angled triangles.\nright angle: An angle of exactly 90° | Marked with a small square.".to_string() }
    fn flashcard(&self) -> String { "Sides 5 and 12 — hypotenuse? | 13 (5²+12²=25+144=169, √169=13)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "A TV screen is 80cm wide and 60cm tall. What is the diagonal (screen size)?\n100cm\n2\n80²+60²=6400+3600=10000, √10000=100.".to_string() }
}

pub(crate) struct Gr8MathFallbackTopic;
impl TopicContent for Gr8MathFallbackTopic {
    fn explanation(&self) -> String { "Grade 8 deepens algebra, introduces the Pythagorean theorem, and prepares for Grade 9 functions.".to_string() }
    fn worked_example(&self, _i: usize) -> String { "Plot y = 2x + 1 for x = -2, -1, 0, 1, 2.\nx=-2: y=-3, x=-1: y=-1, x=0: y=1, x=1: y=3, x=2: y=5\nPlot points and draw a straight line.".to_string() }
    fn practice_problem(&self, _d: u16) -> String { "Does (3,7) lie on y=2x+1?\nYes\n2(3)+1=7 ✓\nNo, Cannot tell, Need more info\nSubstitute x=3 into y=2x+1.\n2×3+1=7=y ✓".to_string() }
    fn hint(&self, l: u8) -> String { match l { 1 => "Substitute the x-value into the equation.".to_string(), _ => "If y = 2x+1 and x=3, then y = 2(3)+1 = 7.".to_string() } }
    fn misconception(&self) -> String { "WRONG: y = 2x means y is always twice as big as x.\nRIGHT: y = 2x means y equals 2 times x. When x is negative, y is also negative!\nWHY: The equation works for ALL values of x, including negatives.".to_string() }
    fn vocabulary(&self) -> String { "gradient/slope: How steep a line is | In y=2x+1, the gradient is 2.\ny-intercept: Where the line crosses the y-axis | In y=2x+1, the y-intercept is 1.".to_string() }
    fn flashcard(&self) -> String { "What is the gradient of y = 3x - 5? | 3 (the number in front of x)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Draw y=x+2 and y=-x+4 on the same axes. Where do they cross?\n(1, 3)\n3\nSet equal: x+2=-x+4 → 2x=2 → x=1, y=3.".to_string() }
}
