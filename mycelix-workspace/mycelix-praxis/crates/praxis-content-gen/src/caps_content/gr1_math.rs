// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 1 Mathematics — CAPS Foundation Phase.
//! Ages 6-7. Concrete, visual, playful. Numbers to 20, basic addition/subtraction.

use super::TopicContent;

pub(crate) struct Gr1Counting;
impl TopicContent for Gr1Counting {
    fn explanation(&self) -> String {
        "Counting means saying numbers in order while pointing to things. \
         We count forwards: 1, 2, 3, 4, 5... and backwards: 5, 4, 3, 2, 1! \
         Each number is ONE MORE than the number before it. \
         We can count fingers, apples, blocks — anything! \
         The last number you say tells you HOW MANY there are.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Count the stars: ★ ★ ★ ★ ★\nPoint to each one: 1... 2... 3... 4... 5!\nThere are 5 stars.".to_string(),
            1 => "What comes after 7?\nCount: 1, 2, 3, 4, 5, 6, 7... 8!\nAfter 7 comes 8.".to_string(),
            _ => "Count backwards from 10:\n10, 9, 8, 7, 6, 5, 4, 3, 2, 1... Blast off!".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "How many fingers are on one hand?\n5\nCount them: 1, 2, 3, 4, 5!\n3, 10, 4\nHold up your hand and count.\nTouch each finger as you count.".to_string(),
            301..=600 => "What number comes before 12?\n11\nCount: ...10, 11, 12. Before 12 is 11.\n10, 13, 2\nCount up to 12. What did you say just before?\n...10, 11, ?".to_string(),
            _ => "Fill in: 6, __, 8, 9, __\n7, 10\n6, 7, 8, 9, 10 — counting by ones.\n5, 11 and 8, 12\nCount forwards from 6.\nEach number is one more.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "Use your fingers to help you count!".to_string(), 2 => "Say each number slowly and point to something.".to_string(), _ => "After 7 comes 8. After 8 comes 9. After 9 comes 10!".to_string() }
    }
    fn misconception(&self) -> String { "WRONG: Counting faster means you counted more.\nRIGHT: You must point to ONE thing for each number. Going fast might make you skip or count twice.\nWHY: Each number matches exactly one object!".to_string() }
    fn vocabulary(&self) -> String { "count: Say numbers in order while matching to things | Count the apples: 1, 2, 3.\nmore: A bigger number | 5 is more than 3.\nless: A smaller number | 2 is less than 7.".to_string() }
    fn flashcard(&self) -> String { "What number comes after 9? | 10".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Draw 7 circles.\n○ ○ ○ ○ ○ ○ ○\n1\nCount as you draw each one.".to_string() }
}

pub(crate) struct Gr1Addition;
impl TopicContent for Gr1Addition {
    fn explanation(&self) -> String {
        "Adding means putting things together to find out how many in total. \
         If you have 3 apples and get 2 more, you have 3 + 2 = 5 apples. \
         The + sign means 'and' or 'plus'. The = sign means 'equals' or 'makes'. \
         You can use your fingers, draw pictures, or count on from the bigger number.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "3 + 2 = ?\nHold up 3 fingers. Then hold up 2 more.\nCount all fingers: 1, 2, 3, 4, 5!\n3 + 2 = 5".to_string(),
            1 => "There are 4 birds in a tree. 3 more fly in. How many now?\n4 birds + 3 birds = ?\nStart at 4, count on: 5, 6, 7\n4 + 3 = 7 birds".to_string(),
            _ => "5 + 5 = ?\nHold up all 10 fingers!\n5 + 5 = 10. This is called a 'double'!".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "2 + 1 = ?\n3\nYou have 2, add 1 more: 3!\n1, 4, 5\nStart at 2, count one more.\n2... 3!".to_string(),
            301..=600 => "Mpho has 6 sweets. Thandi gives her 4 more. How many?\n10\n6 + 4 = 10\n2, 24, 8\nStart at 6 and count 4 more.\n6... 7, 8, 9, 10!".to_string(),
            _ => "What two numbers add to make 8? Find 3 different ways.\n1+7, 2+6, 3+5\nAlso: 4+4, 5+3, 6+2, 7+1, 0+8.\n9+1, 10+2\nStart with 1: what do you add to get 8?\n1+?=8 means ?=7.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "Use your fingers or draw dots to help!".to_string(), 2 => "Start with the bigger number and count on.".to_string(), _ => "For 3 + 5: start at 5, count on 3 more: 6, 7, 8!".to_string() }
    }
    fn misconception(&self) -> String { "WRONG: The answer to adding is always a bigger number than what you started with.\nRIGHT: That's true! But be careful — adding 0 gives the same number. 5 + 0 = 5.\nWHY: Adding zero means 'no more', so nothing changes.".to_string() }
    fn vocabulary(&self) -> String { "add/plus: Put together to find the total | 3 + 2 means 3 and 2 together.\nequals: The answer, what it makes | 3 + 2 = 5 means 3 plus 2 makes 5.\nsum/total: The answer when you add | The sum of 3 and 4 is 7.".to_string() }
    fn flashcard(&self) -> String { "4 + 3 = ? | 7".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Draw a picture to show 3 + 4 = 7.\n●●● + ●●●● = ●●●●●●●\n1\nDraw 3 dots, then 4 more dots, count them all.".to_string() }
}

pub(crate) struct Gr1MathFallbackTopic;
impl TopicContent for Gr1MathFallbackTopic {
    fn explanation(&self) -> String { "In Grade 1, you learn to count, add, subtract, and recognise shapes. Use your fingers, draw pictures, and play with blocks to help you understand!".to_string() }
    fn worked_example(&self, _i: usize) -> String { "5 - 2 = ?\nHold up 5 fingers. Put down 2.\nHow many are still up? 3!\n5 - 2 = 3".to_string() }
    fn practice_problem(&self, _d: u16) -> String { "7 - 3 = ?\n4\n7 take away 3: 6, 5, 4.\n3, 10, 5\nStart at 7, count back 3.\n7... 6, 5, 4!".to_string() }
    fn hint(&self, l: u8) -> String { match l { 1 => "Use your fingers!".to_string(), _ => "Take away means count backwards.".to_string() } }
    fn misconception(&self) -> String { "WRONG: 5 - 3 and 3 - 5 are the same.\nRIGHT: 5 - 3 = 2, but 3 - 5 is tricky (you can't take 5 from 3 yet!).\nWHY: The order matters in subtraction.".to_string() }
    fn vocabulary(&self) -> String { "subtract/take away: Remove some to find what's left | 5 - 2 means start with 5, take away 2.".to_string() }
    fn flashcard(&self) -> String { "8 - 3 = ? | 5".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Use pictures to show 6 - 2 = 4.\n●●●●●● cross out ●● = ●●●●\n1\nDraw 6 dots. Cross out 2. Count what's left.".to_string() }
}
