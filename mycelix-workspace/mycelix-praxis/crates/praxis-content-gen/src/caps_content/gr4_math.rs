// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 4 Mathematics — CAPS-aligned content.
//!
//! Key topics (CAPS Intermediate Phase, Gr4):
//! - Whole numbers: counting, place value, ordering to 10,000
//! - Addition & subtraction: 3-digit numbers, word problems
//! - Multiplication & division: basic facts (2,3,4,5,10), word problems
//! - Common fractions: halves, quarters, thirds, fifths, eighths
//! - Measurement: length (mm, cm, m, km), mass (g, kg), time
//! - Data handling: tally tables, bar graphs
//! - 2D shapes: properties, symmetry

use super::TopicContent;

pub(crate) struct Gr4WholeNumbers;
impl TopicContent for Gr4WholeNumbers {
    fn explanation(&self) -> String {
        "In Grade 4, we work with numbers up to 10 000. \
         Place value tells us what each digit is worth: \
         In 3 456, the 3 means 3 thousands (3 000), the 4 means 4 hundreds (400), \
         the 5 means 5 tens (50), and the 6 means 6 ones (6). \
         We can write this as 3 000 + 400 + 50 + 6 = 3 456. \
         This is called expanded notation."
            .to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Write 5 238 in expanded notation.\n\
                  Thousands: 5 × 1 000 = 5 000 -> 5 thousands\n\
                  Hundreds: 2 × 100 = 200 -> 2 hundreds\n\
                  Tens: 3 × 10 = 30 -> 3 tens\n\
                  Ones: 8 × 1 = 8 -> 8 ones\n\
                  5 238 = 5 000 + 200 + 30 + 8".to_string(),
            1 => "Arrange in order from smallest to largest: 4 501, 4 150, 4 510\n\
                  Compare thousands: all have 4 000 -> same\n\
                  Compare hundreds: 1 vs 5 -> 4 150 is smallest\n\
                  Compare 4 501 and 4 510: tens digit 0 < 1 -> 4 501 < 4 510\n\
                  Order: 4 150, 4 501, 4 510".to_string(),
            _ => "Round 3 672 to the nearest hundred.\n\
                  Look at the tens digit: 7 -> 7 ≥ 5, so round up\n\
                  3 672 → 3 700\n\
                  The hundreds digit goes from 6 to 7.".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "What is the value of the 6 in 2 643?\n600\n\
                        The 6 is in the hundreds place, so it means 600.\n\
                        6, 60, 6000\nWhich place is the 6 in?\nCount from the right: ones, tens, hundreds.".to_string(),
            301..=600 => "Write seven thousand and thirty-nine as a number.\n7 039\n\
                          7 thousands + 0 hundreds + 3 tens + 9 ones = 7 039.\n\
                          7 390, 739, 7 309\nHow many thousands? How many hundreds?\nThousands = 7 000, tens = 30, ones = 9.".to_string(),
            _ => "Round 8 465 to the nearest thousand. Then find the difference between the rounded number and the original.\n\
                  8 000; difference = 465\n8 465 rounds down to 8 000 (because 4 < 5). The difference is 8 465 − 8 000 = 465.\n\
                  9 000, 8 500, 8 400\nLook at the hundreds digit to decide: round up or down?\n\
                  Hundreds digit is 4, which is less than 5, so round down.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level {
            1 => "Think about what each digit is worth based on its position.".to_string(),
            2 => "Thousands, Hundreds, Tens, Ones — which place is the digit in?".to_string(),
            _ => "Write the number in expanded form: thousands + hundreds + tens + ones.".to_string(),
        }
    }
    fn misconception(&self) -> String {
        "WRONG: In 4 073, the 0 means there is nothing.\n\
         RIGHT: The 0 is a placeholder. It means there are 0 hundreds, but we still need it to show 4 thousands and 7 tens.\n\
         WHY: Without the 0, 4 073 would look like 473, which is a completely different number!"
            .to_string()
    }
    fn vocabulary(&self) -> String {
        "place value: The value of a digit based on its position in a number | In 3 456, the 4 has a place value of 400.\n\
         expanded notation: Writing a number as the sum of each digit's value | 3 456 = 3 000 + 400 + 50 + 6.\n\
         rounding: Replacing a number with a nearby 'neat' number | 3 672 rounded to the nearest 100 is 3 700."
            .to_string()
    }
    fn flashcard(&self) -> String {
        "What is the place value of 5 in 5 280? | 5 000 (five thousands)".to_string()
    }
    fn assessment_item(&self, context: &str) -> String {
        if context.contains("Apply") {
            "A school has 2 375 books and receives 1 486 more. How many books does the school now have?\n\
             3 861\nWhat operation do you need? Add or subtract?"
                .to_string()
        } else {
            "Write 6 048 in expanded notation.\n\
             6 000 + 0 + 40 + 8\n1\nRemember to include a zero for any empty place."
                .to_string()
        }
    }
}

pub(crate) struct Gr4Fractions;
impl TopicContent for Gr4Fractions {
    fn explanation(&self) -> String {
        "A fraction shows parts of a whole. The bottom number (denominator) tells us \
         how many equal parts the whole is divided into. The top number (numerator) \
         tells us how many parts we are talking about. \
         In Grade 4, we work with halves (½), quarters (¼), thirds (⅓), \
         fifths (⅕), and eighths (⅛). \
         Example: If a pizza is cut into 8 equal slices and you eat 3, \
         you ate ⅜ of the pizza."
            .to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Shade ¾ of this rectangle.\n\
                  Divide the rectangle into 4 equal parts -> 4 equal sections\n\
                  Shade 3 of the 4 parts -> 3 sections coloured\n\
                  ¾ means 3 out of 4 equal parts are shaded.".to_string(),
            1 => "Which is bigger: ⅓ or ⅕?\n\
                  Draw two equal bars -> same size\n\
                  Divide one into 3 parts, the other into 5 parts -> compare\n\
                  ⅓ is bigger because fewer parts means each part is larger.\n\
                  ⅓ > ⅕".to_string(),
            _ => "There are 20 sweets. ¼ are red. How many red sweets?\n\
                  ¼ of 20 means divide 20 into 4 groups -> 20 ÷ 4 = 5\n\
                  5 sweets are red.".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Colour ½ of a circle.\nColour one of two equal halves.\n\
                        ½ means 1 out of 2 equal parts.\n¼, ⅓, ⅔\n\
                        How many equal parts should the circle have?\nDivide the circle into 2 equal parts. Colour 1.".to_string(),
            301..=600 => "There are 15 marbles. ⅓ are blue. How many are blue?\n5\n\
                          ⅓ of 15 = 15 ÷ 3 = 5 blue marbles.\n3, 10, 45\n\
                          To find ⅓, divide by the denominator.\n15 ÷ 3 = ?".to_string(),
            _ => "Thandi has 24 stickers. She gives ⅜ to her friend and ¼ to her sister. \
                  How many does she keep?\n9\n\
                  ⅜ of 24 = 9 (to friend). ¼ of 24 = 6 (to sister). 24 − 9 − 6 = 9 kept.\n\
                  15, 12, 3\nFind each fraction of 24 separately, then subtract both from 24.\n\
                  ⅜ of 24: 24 ÷ 8 = 3, then 3 × 3 = 9.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level {
            1 => "Remember: the bottom number tells you how many equal parts.".to_string(),
            2 => "To find a fraction of a number, divide by the denominator, then multiply by the numerator.".to_string(),
            _ => "Example: ¾ of 20 → 20 ÷ 4 = 5, then 5 × 3 = 15.".to_string(),
        }
    }
    fn misconception(&self) -> String {
        "WRONG: ⅛ is bigger than ¼ because 8 is bigger than 4.\n\
         RIGHT: ⅛ is SMALLER than ¼. When you cut something into more pieces, each piece is smaller.\n\
         WHY: Imagine sharing a chocolate bar between 8 people vs 4 people. With 8 people, each person gets less!"
            .to_string()
    }
    fn vocabulary(&self) -> String {
        "numerator: The top number of a fraction — how many parts we have | In ¾, the numerator is 3.\n\
         denominator: The bottom number — how many equal parts the whole is divided into | In ¾, the denominator is 4.\n\
         unit fraction: A fraction where the numerator is 1 | ½, ⅓, ¼, ⅕ are all unit fractions."
            .to_string()
    }
    fn flashcard(&self) -> String {
        "What is ¼ of 28? | 7 (divide 28 by 4)".to_string()
    }
    fn assessment_item(&self, _context: &str) -> String {
        "A farmer has 40 chickens. ⅕ are brown and ¼ are white. The rest are black. How many are black?\n\
         22\n2\nFirst find ⅕ of 40 and ¼ of 40, then subtract both from 40.".to_string()
    }
}

// Fallback for topics not yet hand-authored
pub(crate) struct Gr4MathFallbackTopic;
impl TopicContent for Gr4MathFallbackTopic {
    fn explanation(&self) -> String { "Grade 4 mathematics builds on what you learned in Grade 3. You'll work with bigger numbers (up to 10 000), learn about fractions, and solve real-world problems.".to_string() }
    fn worked_example(&self, _index: usize) -> String { "Solve: 2 345 + 1 678\nAdd ones: 5 + 8 = 13 (write 3, carry 1) -> 3\nAdd tens: 4 + 7 + 1 = 12 (write 2, carry 1) -> 2\nAdd hundreds: 3 + 6 + 1 = 10 (write 0, carry 1) -> 0\nAdd thousands: 2 + 1 + 1 = 4 -> 4\nAnswer: 4 023".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "Calculate: 3 456 + 2 789\n6 245\nUse column addition, working from right to left.\n5 245, 6 345, 7 245\nRemember to carry when a column adds up to more than 9.\nStart with ones: 6 + 9 = 15 (write 5, carry 1).".to_string() }
    fn hint(&self, level: u8) -> String { match level { 1 => "What operation does the problem need?".to_string(), 2 => "Line up the digits by place value.".to_string(), _ => "Work from right to left: ones, tens, hundreds, thousands.".to_string() } }
    fn misconception(&self) -> String { "WRONG: When adding, you don't need to carry.\nRIGHT: When a column adds to 10 or more, write the ones digit and carry the tens digit to the next column.\nWHY: Each column can only hold one digit (0-9).".to_string() }
    fn vocabulary(&self) -> String { "carry: Moving a tens digit to the next column when a sum is 10 or more | 7 + 8 = 15, write 5 and carry 1.".to_string() }
    fn flashcard(&self) -> String { "What is 4 567 rounded to the nearest thousand? | 5 000 (because 5 in the hundreds ≥ 5, round up)".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "A shop has 3 458 items. They sell 1 279. How many are left?\n2 179\n2\nSubtract using column method.".to_string() }
}
