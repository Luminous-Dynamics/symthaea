// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 3 Mathematics — CAPS Foundation Phase exit.
//! Ages 8-9. Numbers to 1000, multiplication introduction, time, money.

use super::TopicContent;

pub(crate) struct Gr3Multiplication;
impl TopicContent for Gr3Multiplication {
    fn explanation(&self) -> String {
        "Multiplication is a fast way to add equal groups. \
         Instead of adding 3+3+3+3 (four groups of 3), we write 4 × 3 = 12. \
         The × sign means 'groups of' or 'times'. \
         Think of it as rows: 4 rows with 3 in each row = 12 total. \
         This is called an array!".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "3 × 4 = ?\nDraw 3 rows of 4 dots:\n● ● ● ●\n● ● ● ●\n● ● ● ●\nCount all: 12! So 3 × 4 = 12.".to_string(),
            1 => "5 bags with 2 sweets each. How many sweets?\n5 groups of 2 = 5 × 2\nSkip count by 2: 2, 4, 6, 8, 10\n5 × 2 = 10 sweets.".to_string(),
            _ => "What is 2 × 10?\nSkip count by 10 twice: 10, 20\nOr: 2 groups of 10 = 20.\n2 × 10 = 20".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "3 × 2 = ?\n6\n3 groups of 2: 2+2+2 = 6\n5, 32, 1\nDraw 3 groups with 2 in each.\nCount all: 1,2,3,4,5,6!".to_string(),
            301..=600 => "There are 4 tables. Each table has 5 chairs. How many chairs?\n20\n4 × 5 = 20. Skip count by 5: 5, 10, 15, 20.\n9, 45, 15\nHow many groups? What's in each group?\n4 groups of 5: count by 5s.".to_string(),
            _ => "Eggs come in boxes of 6. Mama buys 4 boxes. How many eggs? She uses 8 for breakfast. How many left?\n24 eggs, 16 left\n4 × 6 = 24 eggs. 24 - 8 = 16 left.\n10, 32 and 28, 18\nFirst multiply, then subtract.\n4 × 6 = ? Then ? - 8 = ?.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "Draw groups to help you see the multiplication.".to_string(), 2 => "Skip counting works: 3 × 5 means count by 5s three times: 5, 10, 15.".to_string(), _ => "An array helps: 3 rows of 4 = 3 × 4. Count all the dots!".to_string() }
    }
    fn misconception(&self) -> String { "WRONG: 3 × 4 and 4 × 3 are different because the groups are different.\nRIGHT: Both give 12! 3 groups of 4 = 4 groups of 3 = 12. The order doesn't change the answer.\nWHY: Draw both arrays — they have the same number of dots, just arranged differently.".to_string() }
    fn vocabulary(&self) -> String { "multiply/times: Find the total in equal groups | 3 × 4 means 3 groups of 4.\narray: Objects in rows and columns | 3 rows of 4 is a 3×4 array.\nproduct: The answer to multiplication | The product of 3 and 4 is 12.".to_string() }
    fn flashcard(&self) -> String { "5 × 3 = ? | 15 (five groups of three)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Draw an array to show 4 × 3 and write the answer.\n● ● ●\n● ● ●\n● ● ●\n● ● ●\n= 12\n1\n4 rows with 3 in each row.".to_string() }
}

pub(crate) struct Gr3MathFallbackTopic;
impl TopicContent for Gr3MathFallbackTopic {
    fn explanation(&self) -> String { "Grade 3 is the last year of Foundation Phase. You work with numbers to 1000, learn multiplication and division, and start telling time on clocks!".to_string() }
    fn worked_example(&self, _i: usize) -> String { "What time is it when the short hand points to 3 and the long hand points to 12?\nThe short hand shows hours: 3\nThe long hand on 12 means o'clock (exactly)\nIt's 3 o'clock!".to_string() }
    fn practice_problem(&self, _d: u16) -> String { "How many R5 coins make R30?\n6\n30 ÷ 5 = 6 coins. Or count by 5s: 5, 10, 15, 20, 25, 30 = 6 jumps.\n5, 25, 35\nSkip count by 5 until you reach 30.\n5, 10, 15, 20, 25, 30 — how many jumps?".to_string() }
    fn hint(&self, l: u8) -> String { match l { 1 => "Use coins or draw them to help you count.".to_string(), _ => "Dividing is the opposite of multiplying.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Division always makes numbers smaller.\nRIGHT: Division means sharing equally — if you share 12 between 1 person, they get 12!\nWHY: 12 ÷ 1 = 12. Only sharing between MORE than 1 makes each share smaller.".to_string() }
    fn vocabulary(&self) -> String { "divide/share: Split into equal groups | 12 ÷ 3 means share 12 into 3 equal groups.\nremainder: What's left over after sharing equally | 13 ÷ 5 = 2 remainder 3.".to_string() }
    fn flashcard(&self) -> String { "12 ÷ 4 = ? | 3 (share 12 into 4 equal groups)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Share 15 sweets equally among 3 children. How many each?\n5\n1\n15 ÷ 3 = 5.".to_string() }
}
