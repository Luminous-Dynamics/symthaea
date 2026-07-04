// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rule-based syllable counting for poetic form constraints.
//!
//! Simple English syllable estimator based on vowel groups and common patterns.
//! Used by the creative mode to enforce haiku (5-7-5), tanka, etc.

/// Count syllables in a word using vowel-group heuristics.
///
/// Algorithm:
/// 1. Count consecutive vowel groups (a, e, i, o, u, y)
/// 2. Subtract 1 for silent-e at end (if word > 2 chars and ends with 'e')
/// 3. Add 1 for common suffixes that add a syllable (-ed after consonant cluster)
/// 4. Minimum 1 syllable per word
///
/// This is an approximation — English syllabification is notoriously irregular.
/// Accuracy is ~85% for common words, sufficient for poetic form constraints.
pub fn count_syllables(word: &str) -> usize {
    let word = word.to_lowercase();
    let chars: Vec<char> = word.chars().filter(|c| c.is_alphabetic()).collect();

    if chars.is_empty() {
        return 0;
    }

    if chars.len() <= 2 {
        return 1;
    }

    let mut count = 0;
    let mut prev_vowel = false;

    for &ch in &chars {
        let is_vowel = matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u' | 'y');
        if is_vowel && !prev_vowel {
            count += 1;
        }
        prev_vowel = is_vowel;
    }

    // Silent-e: subtract 1 if word ends with 'e' (but not "le" pattern like "table")
    if chars.len() > 2 && chars[chars.len() - 1] == 'e' {
        let penult = chars[chars.len() - 2];
        if !matches!(penult, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'l') && count > 1 {
            count -= 1;
        }
    }

    // "-ed" suffix: doesn't add syllable unless preceded by t or d
    if chars.len() > 3 && chars[chars.len() - 2] == 'e' && chars[chars.len() - 1] == 'd' {
        let before_ed = chars[chars.len() - 3];
        if !matches!(before_ed, 't' | 'd') && count > 1 {
            count -= 1;
        }
    }

    count.max(1)
}

/// Count total syllables in a line of text.
pub fn count_line_syllables(line: &str) -> usize {
    line.split_whitespace().map(count_syllables).sum()
}

/// Check if adding a word would exceed the syllable limit for the current line.
pub fn would_exceed_limit(current_count: usize, word: &str, limit: u8) -> bool {
    let word_syllables = count_syllables(word);
    current_count + word_syllables > limit as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_syllable_words() {
        assert_eq!(count_syllables("the"), 1);
        assert_eq!(count_syllables("cat"), 1);
        assert_eq!(count_syllables("tree"), 1);
        assert_eq!(count_syllables("thought"), 1);
    }

    #[test]
    fn two_syllable_words() {
        assert_eq!(count_syllables("water"), 2);
        assert_eq!(count_syllables("happy"), 2);
        assert_eq!(count_syllables("gentle"), 2);
        assert_eq!(count_syllables("mountain"), 2);
    }

    #[test]
    fn three_syllable_words() {
        assert_eq!(count_syllables("beautiful"), 3);
        assert_eq!(count_syllables("consciousness"), 3);
        assert_eq!(count_syllables("important"), 3);
    }

    #[test]
    fn empty_and_short() {
        assert_eq!(count_syllables(""), 0);
        assert_eq!(count_syllables("a"), 1);
        assert_eq!(count_syllables("I"), 1);
    }

    #[test]
    fn haiku_line_count() {
        // "An old silent pond" = 5 syllables
        assert_eq!(count_line_syllables("An old silent pond"), 5);
    }

    #[test]
    fn line_syllable_count() {
        // Simple cases
        let count = count_line_syllables("the cat sat on the mat");
        assert!(count >= 5 && count <= 7, "count = {count}");
    }

    #[test]
    fn would_exceed() {
        assert!(would_exceed_limit(4, "beautiful", 5)); // 4 + 3 = 7 > 5
        assert!(!would_exceed_limit(3, "cat", 5)); // 3 + 1 = 4 <= 5
    }

    #[test]
    fn silent_e() {
        assert_eq!(count_syllables("make"), 1);
        assert_eq!(count_syllables("time"), 1);
        assert_eq!(count_syllables("hope"), 1);
    }

    #[test]
    fn minimum_one_syllable() {
        // Even odd words should have at least 1
        assert!(count_syllables("hmm") >= 1);
        assert!(count_syllables("shhh") >= 1);
    }
}
