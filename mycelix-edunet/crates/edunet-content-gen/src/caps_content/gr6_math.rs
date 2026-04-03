// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 6 Mathematics — CAPS Senior Phase entry.
//! Topics: integers, ratio & rate, decimal operations, area & perimeter.

use super::TopicContent;

pub(crate) struct Gr6Integers;
impl TopicContent for Gr6Integers {
    fn explanation(&self) -> String {
        "Integers include positive numbers, negative numbers, and zero. \
         On a number line, numbers get smaller as you go left: ... -3, -2, -1, 0, 1, 2, 3 ... \
         Negative numbers are used for temperatures below zero, debts, and depths below sea level. \
         In Johannesburg, winter mornings can be -2°C!".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Order from smallest to largest: 3, -5, 0, -1, 4\nOn a number line, further left = smaller.\n-5, -1, 0, 3, 4".to_string(),
            1 => "The temperature is 4°C and drops 7 degrees. What is the new temperature?\n4 - 7 = -3°C\nStart at 4, count 7 steps left past zero.".to_string(),
            _ => "Calculate: -3 + 8\nStart at -3 on the number line. Move 8 steps right.\n-3 + 8 = 5".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Which is smaller: -2 or 3?\n-2\nNegative numbers are always less than positive numbers.\n3, They're equal, Cannot tell\nThink of a number line.\nNegative is to the left of zero.".to_string(),
            301..=600 => "Calculate: 5 + (-8)\n-3\n5 - 8 = -3 (adding a negative is like subtracting).\n3, 13, -13\nAdding a negative = subtracting.\nStart at 5, go 8 left.".to_string(),
            _ => "At 6am it's -4°C. By noon it's 13°C warmer. By midnight it drops 9°C. What's the midnight temperature?\n0°C\n-4 + 13 = 9°C at noon. 9 - 9 = 0°C at midnight.\n4, -4, 9\nDo each step in order.\n-4 + 13 = ?".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "Draw a number line with zero in the middle.".to_string(), 2 => "Moving right = adding, moving left = subtracting.".to_string(), _ => "Adding a negative number is the same as subtracting.".to_string() }
    }
    fn misconception(&self) -> String { "WRONG: -5 is bigger than -2 because 5 is bigger than 2.\nRIGHT: -5 < -2. Further from zero in the negative direction = smaller.\nWHY: Think of temperature: -5°C is colder (smaller) than -2°C.".to_string() }
    fn vocabulary(&self) -> String { "integer: A whole number that can be positive, negative, or zero | -3, 0, 7 are integers.\nnegative number: A number less than zero | -5 means 5 less than zero.".to_string() }
    fn flashcard(&self) -> String { "Is -7 greater or less than -3? | Less than (-7 < -3, further left on number line)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "A submarine is at -120 m. It rises 45 m. New depth?\n-75 m\n2\n-120 + 45 = -75".to_string() }
}

pub(crate) struct Gr6RatioRate;
impl TopicContent for Gr6RatioRate {
    fn explanation(&self) -> String {
        "A ratio compares two quantities: 'For every 2 boys there are 3 girls' means the ratio is 2:3. \
         A rate compares quantities with different units: '60 km in 2 hours' is a rate of 30 km/h. \
         To simplify a ratio, divide both numbers by the same amount: 6:9 = 2:3 (÷3).".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Simplify the ratio 12:8\nFind the HCF: HCF of 12 and 8 = 4\n12÷4 : 8÷4 = 3:2".to_string(),
            1 => "Share R150 in the ratio 2:3\nTotal parts: 2+3=5\nOne part: 150÷5=R30\nShare 1: 2×30=R60, Share 2: 3×30=R90".to_string(),
            _ => "A car travels 240 km in 3 hours. What is the speed?\nSpeed = distance ÷ time\n240 ÷ 3 = 80 km/h".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Simplify 6:10\n3:5\nHCF=2, divide both by 2.\n3:10, 6:5, 1:2\nFind a number that divides both.\n6÷2=3, 10÷2=5".to_string(),
            301..=600 => "Share 48 sweets between Thabo and Nathi in the ratio 3:5. How many does each get?\nThabo: 18, Nathi: 30\n3+5=8 parts, 48÷8=6 per part. 3×6=18, 5×6=30.\n24 each, 16 and 32, 20 and 28\nTotal parts = 3+5. Value per part = 48÷8.\n48 ÷ 8 = 6".to_string(),
            _ => "A recipe for 4 people uses 300g flour. How much flour for 10 people?\n750g\nRatio 4:10 simplifies to 2:5. Flour: 300 × (10/4) = 300 × 2.5 = 750g.\n600g, 1200g, 500g\nFind how many times bigger the group is.\n10 ÷ 4 = 2.5, so multiply flour by 2.5.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "A ratio compares two amounts of the same kind.".to_string(), 2 => "To simplify, divide both parts by the HCF.".to_string(), _ => "To share in a ratio: find total parts, value per part, then multiply.".to_string() }
    }
    fn misconception(&self) -> String { "WRONG: The ratio 2:3 means there are 2 things and 3 things (5 total).\nRIGHT: 2:3 means for every 2 of one, there are 3 of the other. The actual amounts depend on the total.\nWHY: If the total is 50, then 2:3 means 20 and 30, not 2 and 3.".to_string() }
    fn vocabulary(&self) -> String { "ratio: A comparison of two quantities using ':' | 2:3 means '2 for every 3'.\nrate: A comparison with different units | 60 km/h is a rate (distance per time).\nHCF: Highest Common Factor — the largest number that divides both | HCF of 12 and 8 is 4.".to_string() }
    fn flashcard(&self) -> String { "Simplify 15:20 | 3:4 (divide both by 5)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Mix paint in ratio 2:5 (red:white). Need 35 litres total. How much red?\n10 litres\n2\n2+5=7 parts, 35÷7=5 per part, red=2×5=10.".to_string() }
}

pub(crate) struct Gr6MathFallbackTopic;
impl TopicContent for Gr6MathFallbackTopic {
    fn explanation(&self) -> String { "Grade 6 introduces integers, ratio, rate, and builds decimal skills.".to_string() }
    fn worked_example(&self, _i: usize) -> String { "Find the perimeter of a rectangle: length 12cm, width 8cm.\nP = 2(l+w) = 2(12+8) = 2×20 = 40cm".to_string() }
    fn practice_problem(&self, _d: u16) -> String { "Area of rectangle 15m by 9m?\n135 m²\nA=l×w=15×9=135\n24, 48, 150\nArea = length × width.\n15 × 9 = ?".to_string() }
    fn hint(&self, l: u8) -> String { match l { 1 => "What formula do you need?".to_string(), _ => "Perimeter = around the edge. Area = space inside.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Perimeter and area are the same thing.\nRIGHT: Perimeter is the distance around (cm, m). Area is the space inside (cm², m²).\nWHY: Perimeter is a length. Area is a surface. Different units!".to_string() }
    fn vocabulary(&self) -> String { "perimeter: Distance around a shape | Rectangle: P = 2(l+w).\narea: Space inside a shape | Rectangle: A = l × w.".to_string() }
    fn flashcard(&self) -> String { "Area of 7cm × 4cm rectangle? | 28 cm²".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Garden is 20m by 15m. Fence costs R45/m. Total fence cost?\nR3 150\n3\nP=2(20+15)=70m, Cost=70×45=R3150.".to_string() }
}
