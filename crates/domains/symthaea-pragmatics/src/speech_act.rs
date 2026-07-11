// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Speech-act classification (Searle's five illocutionary types), rule-based.

/// Searle's taxonomy of illocutionary acts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechAct {
    /// Commits the speaker to the truth of a proposition ("The sky is blue").
    Assertive,
    /// Gets the hearer to do something — commands, requests, questions.
    Directive,
    /// Commits the speaker to a future action ("I promise to help").
    Commissive,
    /// Expresses a psychological state ("Thank you", "I'm sorry").
    Expressive,
    /// Changes reality by being said ("I hereby resign").
    Declarative,
}

const EXPRESSIVE: &[&str] = &[
    "thank",
    "thanks",
    "sorry",
    "apologize",
    "apologise",
    "congratulations",
    "congratulate",
    "welcome",
    "sympathies",
    "condolences",
];
const DECLARATIVE: &[&str] = &[
    "hereby",
    "i declare",
    "i pronounce",
    "i resign",
    "i name",
    "i christen",
    "i now pronounce",
    "you're fired",
    "i sentence",
];
const COMMISSIVE: &[&str] = &[
    "i promise",
    "i swear",
    "i guarantee",
    "i vow",
    "i pledge",
    "i will ",
    "i shall ",
    "we promise",
    "i commit",
];
const IMPERATIVE_STARTS: &[&str] = &[
    "please", "close", "open", "give", "stop", "go ", "come ", "bring", "take", "put", "let ",
    "don't", "do not", "wait", "listen", "look", "help",
];

/// Classify an utterance into a Searle speech act (rule-based; deterministic).
pub fn classify(utterance: &str) -> SpeechAct {
    let u = utterance.trim().to_lowercase();

    if DECLARATIVE.iter().any(|k| u.contains(k)) {
        return SpeechAct::Declarative;
    }
    if EXPRESSIVE.iter().any(|k| u.starts_with(k) || u.contains(k)) {
        return SpeechAct::Expressive;
    }
    if COMMISSIVE.iter().any(|k| u.contains(k)) {
        return SpeechAct::Commissive;
    }
    // Questions and imperatives are directives.
    if u.ends_with('?') || IMPERATIVE_STARTS.iter().any(|k| u.starts_with(k)) {
        return SpeechAct::Directive;
    }
    SpeechAct::Assertive
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_five_types() {
        assert_eq!(classify("The sky is blue."), SpeechAct::Assertive);
        assert_eq!(classify("Close the door."), SpeechAct::Directive);
        assert_eq!(classify("Where are you going?"), SpeechAct::Directive);
        assert_eq!(classify("I promise to help you."), SpeechAct::Commissive);
        assert_eq!(classify("Thank you so much."), SpeechAct::Expressive);
        assert_eq!(classify("I hereby resign."), SpeechAct::Declarative);
    }

    #[test]
    fn declarative_beats_commissive_ordering() {
        // "I now pronounce you..." is declarative even though it's a commitment-ish.
        assert_eq!(
            classify("I now pronounce you married."),
            SpeechAct::Declarative
        );
    }
}
