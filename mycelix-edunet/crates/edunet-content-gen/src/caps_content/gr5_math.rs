// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 5 Mathematics — CAPS Intermediate Phase.
//! Topics: decimals (tenths/hundredths), long multiplication, percentages.

use super::TopicContent;

pub(crate) struct Gr5Decimals;
impl TopicContent for Gr5Decimals {
    fn explanation(&self) -> String {
        "Decimals are another way to write fractions. The decimal point separates whole numbers from parts. \
         In 3.45, the 3 is whole, the 4 is four tenths (4/10), and the 5 is five hundredths (5/100). \
         Think of money: R3.45 means 3 rand, 4 ten-cent coins, and 5 one-cent coins.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Write 0.7 as a fraction.\n0.7 means 7 tenths -> 7/10\nCheck: 7 ÷ 10 = 0.7 ✓\nSo 0.7 = 7/10".to_string(),
            1 => "Order from smallest to largest: 0.35, 0.4, 0.29\nWrite all with 2 decimal places: 0.35, 0.40, 0.29\nCompare: 29 < 35 < 40\nOrder: 0.29, 0.35, 0.4".to_string(),
            _ => "Add: 2.3 + 1.45\nLine up decimals: 2.30 + 1.45\nHundredths: 0+5=5, Tenths: 3+4=7, Ones: 2+1=3\nAnswer: 3.75".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Write 3/10 as a decimal.\n0.3\n3 tenths = 0.3\n0.03, 3.0, 0.33\nTenths go first after the decimal.\n3/10 means 3 out of 10.".to_string(),
            301..=600 => "Which is larger: 0.6 or 0.52?\n0.6\n0.6 = 0.60 = 60 hundredths > 52 hundredths.\n0.52, Equal, Cannot compare\nWrite both with same decimal places.\n0.6 = 0.60".to_string(),
            _ => "A rope is 4.8 m long. Sipho cuts off 1.35 m. How much is left?\n3.45 m\n4.80 − 1.35 = 3.45\n3.55, 2.45, 5.15\nLine up decimal points.\nWrite 4.8 as 4.80 first.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "Think of decimals like money.".to_string(), 2 => "Make same number of decimal places to compare.".to_string(), _ => "Line up decimal points before adding/subtracting.".to_string() }
    }
    fn misconception(&self) -> String {
        "WRONG: 0.52 > 0.6 because 52 > 6.\nRIGHT: 0.6 = 0.60 = 60 hundredths > 52 hundredths.\nWHY: Compare using same decimal places!".to_string()
    }
    fn vocabulary(&self) -> String { "tenths: First place after decimal — each is 1/10 | In 0.7, there are 7 tenths.\nhundredths: Second place after decimal — each is 1/100 | In 0.35, there are 5 hundredths.".to_string() }
    fn flashcard(&self) -> String { "Write 0.25 as a fraction. | 25/100 = 1/4".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Lerato buys a pen for R4.75 and a book for R12.50. She pays R20. Change?\nR2.75\n2\nAdd prices, subtract from R20.".to_string() }
}

pub(crate) struct Gr5Multiplication;
impl TopicContent for Gr5Multiplication {
    fn explanation(&self) -> String {
        "In Grade 5, we multiply 2-digit × 2-digit numbers using long multiplication. \
         Break the problem: 34 × 26 = (34 × 6) + (34 × 20) = 204 + 680 = 884.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "23 × 15\n23 × 5 = 115\n23 × 10 = 230\n115 + 230 = 345".to_string(),
            1 => "45 rows × 32 trees per row\n45 × 2 = 90\n45 × 30 = 1 350\n90 + 1 350 = 1 440 trees".to_string(),
            _ => "67 × 48\n67 × 8 = 536\n67 × 40 = 2 680\n536 + 2 680 = 3 216".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Calculate: 14 × 12\n168\n14×2=28, 14×10=140, 28+140=168\n156, 182, 126\nBreak into ones and tens.\n14 × 2 = ?".to_string(),
            301..=600 => "36 classrooms × 28 desks each. Total desks?\n1 008\n36×8=288, 36×20=720, 288+720=1008\n864, 1080, 928\nMultiply by ones, then tens.\n36 × 8 = ?".to_string(),
            _ => "57 boxes/day for 43 days. Total?\n2 451\n57×3=171, 57×40=2280, 171+2280=2451\n2341, 2551, 2100\nLong multiplication.\n57 × 3 = ?".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "Break the second number into ones and tens.".to_string(), 2 => "Multiply by ones first, then tens.".to_string(), _ => "23 × 15 = (23×5) + (23×10) = 115 + 230 = 345".to_string() }
    }
    fn misconception(&self) -> String {
        "WRONG: In 23 × 15, multiply 23 × 1 = 23.\nRIGHT: The 1 in 15 means 10! So 23 × 10 = 230.\nWHY: Place value — the 1 is in the tens place.".to_string()
    }
    fn vocabulary(&self) -> String { "partial product: Result of multiplying by one part | In 23×15: 115 (×5) and 230 (×10) are partial products.\nlong multiplication: Breaking large multiplications into parts | Both numbers have 2+ digits.".to_string() }
    fn flashcard(&self) -> String { "25 × 16 = ? | 400 (25×6=150, 25×10=250, 150+250=400)".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Concert hall: 48 rows × 35 seats. Capacity?\n1 680\n2\nUse long multiplication.".to_string() }
}

pub(crate) struct Gr5MathFallbackTopic;
impl TopicContent for Gr5MathFallbackTopic {
    fn explanation(&self) -> String { "Grade 5 extends to 100 000, introduces decimals and percentages.".to_string() }
    fn worked_example(&self, _i: usize) -> String { "25% of 80 = ¼ of 80 = 80 ÷ 4 = 20".to_string() }
    fn practice_problem(&self, _d: u16) -> String { "50% of 120?\n60\n50% = ½, 120÷2=60\n30, 100, 240\n50% means half.\nDivide by 2.".to_string() }
    fn hint(&self, l: u8) -> String { match l { 1 => "What fraction is the percentage?".to_string(), _ => "50%=½, 25%=¼, 10%=1/10".to_string() } }
    fn misconception(&self) -> String { "WRONG: 50% is always 50.\nRIGHT: 50% means half — depends on the number.\nWHY: Percentage means 'per hundred'.".to_string() }
    fn vocabulary(&self) -> String { "percentage: A number out of 100 | 25% = 25 out of 100.".to_string() }
    fn flashcard(&self) -> String { "10% of 350? | 35 (÷10)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Convert ¾ to %.\n75%\n2\n¾ = 0.75 = 75%".to_string() }
}
