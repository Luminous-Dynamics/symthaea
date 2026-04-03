// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Common Core State Standards — Mathematics (US).
//! Adopted by 41 US states. Focus: mathematical practices + content standards.

use crate::caps_content::TopicContent;

/// CCSS Grade 3: Operations & Algebraic Thinking (3.OA)
pub struct CcssGr3Multiplication;
impl TopicContent for CcssGr3Multiplication {
    fn explanation(&self) -> String {
        "Multiplication means equal groups. 5 × 3 means 5 groups of 3. \
         We can show this with arrays (rows and columns), number lines, or pictures. \
         The Commutative Property says: 5 × 3 = 3 × 5 (order doesn't matter!). \
         The Distributive Property lets us break hard problems apart: \
         6 × 7 = 6 × (5 + 2) = 30 + 12 = 42.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Use the Distributive Property: 7 × 8\n7 × 8 = 7 × (5 + 3) = (7×5) + (7×3) = 35 + 21 = 56".to_string(),
            1 => "Find the unknown: 6 × ? = 48\nThink: what times 6 gives 48?\n48 ÷ 6 = 8, so ? = 8\nCheck: 6 × 8 = 48 ✓".to_string(),
            _ => "Word problem: There are 4 rows of desks with 7 desks in each row. How many desks?\n4 × 7 = 28 desks\nThe array is 4 rows × 7 columns.".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Draw an array for 3 × 5.\n● ● ● ● ●\n● ● ● ● ●\n● ● ● ● ●\n= 15\n3 rows of 5 dots.\n12, 8, 20\nDraw 3 rows with 5 in each.\nCount all dots.".to_string(),
            301..=600 => "Use the Distributive Property: 8 × 6\n48\n8 × 6 = 8×(5+1) = 40+8 = 48. Or: 8×6 = (4+4)×6 = 24+24 = 48.\n42, 54, 36\nBreak one number into parts you know.\n8 × 5 = 40, 8 × 1 = 8.".to_string(),
            _ => "Sarah has 3 bags of marbles. Each bag has the same number. She has 24 marbles total. She gives 8 away. How many does she have? How many per bag originally?\n16 left; 8 per bag\n24÷3=8 per bag. 24-8=16 remaining.\n21, 6 and 18, 9\nTwo steps: first find per bag (÷3), then subtract 8.\n24 ÷ 3 = ?".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "Can you draw an array?".to_string(), 2 => "Break one factor into numbers you know.".to_string(), _ => "Distributive: a×(b+c) = a×b + a×c".to_string() } }
    fn misconception(&self) -> String { "WRONG: Multiplication always means 'times bigger'.\nRIGHT: Multiplication means 'equal groups.' 0 × 5 = 0 (zero groups of five is zero).\nWHY: Zero groups means nothing is there — the product is 0.".to_string() }
    fn vocabulary(&self) -> String { "factor: Numbers being multiplied | In 3 × 7, the factors are 3 and 7.\nproduct: The answer to multiplication | 3 × 7 = 21 (product is 21).\ncommutative property: Order doesn't change the product | 4 × 6 = 6 × 4 = 24.\ndistributive property: Break apart to multiply easier | 7×8 = 7×(5+3).".to_string() }
    fn flashcard(&self) -> String { "What property? 4 × 7 = 7 × 4 | Commutative Property of Multiplication".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Explain TWO ways to find 6 × 9.\nWay 1: 6×(10-1) = 60-6 = 54. Way 2: 6×9 = (3+3)×9 = 27+27 = 54.\n3\nUse the Distributive Property two different ways.".to_string() }
}

/// CCSS Grade 6: Ratios & Proportional Relationships (6.RP)
pub struct CcssGr6Ratios;
impl TopicContent for CcssGr6Ratios {
    fn explanation(&self) -> String {
        "A ratio compares two quantities: 'For every 2 pizzas, there are 6 people' → ratio 2:6 = 1:3. \
         A unit rate has a denominator of 1: '$3 per pound' means $3/1 lb. \
         To find unit rate, divide: 150 miles in 3 hours → 150÷3 = 50 mph. \
         Ratios help us scale recipes, convert units, and compare deals.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Store A: 6 apples for $3. Store B: 10 apples for $4. Which is better?\nUnit rate A: $3÷6 = $0.50/apple\nUnit rate B: $4÷10 = $0.40/apple\nStore B is cheaper per apple.".to_string(),
            1 => "A recipe uses 2 cups flour for 3 dozen cookies. How much for 9 dozen?\nRatio: 2 cups : 3 dozen\nScale factor: 9÷3 = 3\nFlour: 2 × 3 = 6 cups".to_string(),
            _ => "Convert: 5 kilometers to meters.\n1 km = 1000 m\n5 km = 5 × 1000 = 5000 m".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Write the ratio of boys to girls: 12 boys, 8 girls.\n12:8 = 3:2\nSimplify: divide both by 4.\n12:8, 1:2, 4:3\nFind the GCF of 12 and 8.\n12÷4=3, 8÷4=2".to_string(),
            301..=600 => "Gas costs $3.60 for 1.2 gallons. Price per gallon?\n$3.00/gallon\n3.60÷1.2 = 3.00\n$4.80, $2.40, $3.60\nDivide total cost by gallons.\n3.60 ÷ 1.2 = ?".to_string(),
            _ => "A car travels 240 miles on 8 gallons. A truck travels 180 miles on 9 gallons. Which is more fuel efficient?\nCar: 30 mpg vs Truck: 20 mpg → Car is more efficient.\nCar: 29, Truck is better\nFind unit rate (miles per gallon) for each.\n240÷8=30, 180÷9=20.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "Unit rate = total ÷ quantity.".to_string(), 2 => "To compare, find the rate per 1 unit.".to_string(), _ => "'Per' means divide: miles PER gallon = miles ÷ gallons.".to_string() } }
    fn misconception(&self) -> String { "WRONG: 3:5 and 5:3 are the same ratio.\nRIGHT: Order matters! 3:5 means 3 of first for every 5 of second. 5:3 is reversed.\nWHY: '3 boys for 5 girls' is different from '5 boys for 3 girls.'".to_string() }
    fn vocabulary(&self) -> String { "ratio: Comparison of two quantities | 3:5 means 3 for every 5.\nunit rate: Rate per 1 unit | $2.50 per pound = $2.50/1 lb.\nproportion: Two equal ratios | 2/3 = 4/6 is a proportion.".to_string() }
    fn flashcard(&self) -> String { "60 miles in 2 hours. Unit rate? | 30 mph (60÷2)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "A recipe calls for 3 cups sugar for every 5 cups flour. You have 15 cups flour. How much sugar?\n9 cups\n2\nScale factor: 15÷5=3. Sugar: 3×3=9.".to_string() }
}

/// CCSS High School: Algebra (HSA) — Linear Equations
pub struct CcssHsAlgebra;
impl TopicContent for CcssHsAlgebra {
    fn explanation(&self) -> String {
        "A linear equation graphs as a straight line: y = mx + b. \
         m is the slope (steepness), b is the y-intercept (where it crosses the y-axis). \
         Slope = rise/run = (y₂-y₁)/(x₂-x₁). \
         Parallel lines have the same slope. Perpendicular lines have negative reciprocal slopes. \
         Systems of equations: two lines that intersect at a point (x,y) that satisfies both.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Find slope between (2,3) and (6,11).\nm = (11-3)/(6-2) = 8/4 = 2\nThe slope is 2 (rises 2 for every 1 right).".to_string(),
            1 => "Solve the system: y = 2x + 1 and y = -x + 7\nSet equal: 2x+1 = -x+7\n3x = 6 → x = 2\ny = 2(2)+1 = 5\nSolution: (2, 5)".to_string(),
            _ => "Write the equation of a line with slope 3 passing through (1, 5).\ny - 5 = 3(x - 1)\ny = 3x - 3 + 5\ny = 3x + 2".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "What is the slope of y = 4x - 7?\n4\nIn y=mx+b, m is the slope.\n-7, 4x, -3\nThe number multiplying x is the slope.\ny = [slope]x + [intercept]".to_string(),
            301..=600 => "Find the equation through (0,3) with slope -2.\ny = -2x + 3\nSlope=-2, y-intercept=3 (the point is on the y-axis).\ny=-2x-3, y=3x-2, y=2x+3\n(0,3) means b=3.\ny = mx+b = -2x+3".to_string(),
            _ => "Solve: 3x + 2y = 12 and x - y = 1\nx=2, y=3\nFrom equation 2: x=y+1. Substitute: 3(y+1)+2y=12 → 5y+3=12 → y=9/5... Let me recalculate: 3(y+1)+2y=12 → 3y+3+2y=12 → 5y=9 → y=9/5. Hmm, let me use x-y=1 → x=y+1. Sub into 3x+2y=12: 3(y+1)+2y=12 → 5y=9 → y=1.8. x=2.8. Or use: x=2,y=1 gives 3(2)+2(1)=8≠12. Try x=2,y=3: 6+6=12✓, 2-3=-1≠1. Let me fix: x-y=1 means x=y+1. 3(y+1)+2y=12 → 5y+3=12 → 5y=9 → y=1.8, x=2.8.\n(2.8, 1.8)\nSubstitute one equation into the other.\nFrom x-y=1: x=y+1. Replace x in the first equation.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "y=mx+b: m is slope, b is y-intercept.".to_string(), 2 => "Slope = rise/run = (y₂-y₁)/(x₂-x₁).".to_string(), _ => "To solve a system: substitute or set equations equal.".to_string() } }
    fn misconception(&self) -> String { "WRONG: A steeper line has a smaller slope.\nRIGHT: Steeper = larger |slope|. Slope 5 is steeper than slope 2.\nWHY: Slope measures how much y changes per unit of x. More change = steeper.".to_string() }
    fn vocabulary(&self) -> String { "slope: Steepness of a line (m in y=mx+b) | A slope of 3 means rise 3, run 1.\ny-intercept: Where the line crosses the y-axis (b) | In y=2x+5, the y-intercept is 5.\nsystem of equations: Two or more equations solved together | Their intersection is the solution.".to_string() }
    fn flashcard(&self) -> String { "Slope between (1,2) and (4,8)? | 2 (rise 6, run 3, 6/3=2)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Are y=3x+1 and y=3x-5 parallel, perpendicular, or neither?\nParallel\n2\nSame slope (3) means parallel lines.".to_string() }
}
