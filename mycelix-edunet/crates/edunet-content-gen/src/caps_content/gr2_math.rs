// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 2 Mathematics — CAPS Foundation Phase.
//! Ages 7-8. Numbers to 100, number bonds, doubling, halving.

use super::TopicContent;

pub(crate) struct Gr2NumberBonds;
impl TopicContent for Gr2NumberBonds {
    fn explanation(&self) -> String {
        "Number bonds are pairs of numbers that add up to a total. \
         The number bonds of 10 are: 1+9, 2+8, 3+7, 4+6, 5+5. \
         Knowing these by heart makes adding MUCH faster! \
         If you know 7+3=10, then you also know 17+3=20 and 27+3=30.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Find: 8 + ? = 10\nWhat goes with 8 to make 10?\nCount on from 8: 9, 10 — that's 2 more!\n8 + 2 = 10".to_string(),
            1 => "Use number bonds to add: 7 + 5\nBreak 5 into 3+2 (because 7+3=10)\n7 + 3 = 10, then 10 + 2 = 12\n7 + 5 = 12!".to_string(),
            _ => "Double 8.\nDoubling means adding a number to itself.\n8 + 8 = 16\nTip: double 8 is the same as 2 × 8.".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "6 + ? = 10\n4\n6 + 4 = 10 (number bond of 10).\n6, 3, 5\nWhat goes with 6 to make 10?\nCount from 6 to 10: 7, 8, 9, 10 — four steps.".to_string(),
            301..=600 => "Use number bonds: 9 + 6\n15\nBreak 6 into 1+5. 9+1=10, 10+5=15.\n13, 16, 14\nMake 10 first: 9 + ? = 10.\n9+1=10, you still have 5 left: 10+5=15.".to_string(),
            _ => "What is double 15?\n30\nDouble means add to itself: 15+15. 10+10=20, 5+5=10, 20+10=30.\n25, 35, 20\nBreak into tens and ones.\nDouble 10=20, double 5=10, 20+10=30.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String {
        match level { 1 => "Learn your bonds of 10: 1+9, 2+8, 3+7, 4+6, 5+5.".to_string(), 2 => "To add past 10: first make 10, then add what's left.".to_string(), _ => "7+5: break 5 into 3+2 → 7+3=10 → 10+2=12.".to_string() }
    }
    fn misconception(&self) -> String { "WRONG: You have to count on your fingers every time.\nRIGHT: If you KNOW 6+4=10, you don't need to count! Learning bonds makes you faster.\nWHY: Your brain can remember pairs, like remembering your friend's name.".to_string() }
    fn vocabulary(&self) -> String { "number bond: Two numbers that add to a total | 3 and 7 are a bond of 10.\ndouble: Add a number to itself | Double 6 = 6+6 = 12.\nhalve: Split into two equal parts | Half of 12 = 6.".to_string() }
    fn flashcard(&self) -> String { "What is the bond partner of 7 (to make 10)? | 3".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Fill in ALL the number bonds of 10.\n1+9, 2+8, 3+7, 4+6, 5+5\n2\nStart with 1: 1+?=10, then 2+?=10...".to_string() }
}

pub(crate) struct Gr2MathFallbackTopic;
impl TopicContent for Gr2MathFallbackTopic {
    fn explanation(&self) -> String { "Grade 2 builds on counting to work with numbers up to 100. You learn number bonds, doubling, halving, and start using coins!".to_string() }
    fn worked_example(&self, _i: usize) -> String { "Count in 2s to 20:\n2, 4, 6, 8, 10, 12, 14, 16, 18, 20\nThis is called skip counting by 2s.".to_string() }
    fn practice_problem(&self, _d: u16) -> String { "Count in 5s to 50.\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50\nEach number is 5 more than the last.\n5, 10, 20, 30, 50 and 5, 15, 25, 35, 45\nStart at 5 and keep adding 5.\n5...10...15...".to_string() }
    fn hint(&self, l: u8) -> String { match l { 1 => "Skip counting is like taking big steps!".to_string(), _ => "Count in 5s: look at a clock face!".to_string() } }
    fn misconception(&self) -> String { "WRONG: When counting in 2s, every answer is even.\nRIGHT: If you START at an even number (2), you get evens. Start at 1: 1, 3, 5, 7... those are odd!\nWHY: The pattern depends on where you start.".to_string() }
    fn vocabulary(&self) -> String { "skip counting: Counting by jumping over numbers | Count in 2s: 2, 4, 6, 8...\neven: A number you reach counting by 2s from 0 | 2, 4, 6, 8, 10.\nodd: A number NOT reached by counting in 2s from 0 | 1, 3, 5, 7, 9.".to_string() }
    fn flashcard(&self) -> String { "Is 15 odd or even? | Odd (ends in 5)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Write the first 5 even numbers.\n2, 4, 6, 8, 10\n1\nCount in 2s starting from 2.".to_string() }
}
