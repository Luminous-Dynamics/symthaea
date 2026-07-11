// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deixis resolution: interpret context-dependent expressions (I/you/here/now)
//! against an utterance context.

/// The deictic context of an utterance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Context {
    pub speaker: String,
    pub addressee: String,
    pub place: String,
    pub time: String,
}

/// Resolve a deictic term against a context. Returns the referent, or `None` if
/// the term is not a recognized deictic.
pub fn resolve(term: &str, ctx: &Context) -> Option<String> {
    match term.trim().to_lowercase().as_str() {
        "i" | "me" | "my" | "mine" | "myself" => Some(ctx.speaker.clone()),
        "you" | "your" | "yours" | "yourself" => Some(ctx.addressee.clone()),
        "here" => Some(ctx.place.clone()),
        "now" | "today" => Some(ctx.time.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Context {
        Context {
            speaker: "Alice".into(),
            addressee: "Bob".into(),
            place: "Paris".into(),
            time: "2026".into(),
        }
    }

    #[test]
    fn person_deixis() {
        assert_eq!(resolve("I", &ctx()).as_deref(), Some("Alice"));
        assert_eq!(resolve("you", &ctx()).as_deref(), Some("Bob"));
        assert_eq!(resolve("my", &ctx()).as_deref(), Some("Alice"));
    }

    #[test]
    fn place_and_time_deixis() {
        assert_eq!(resolve("here", &ctx()).as_deref(), Some("Paris"));
        assert_eq!(resolve("now", &ctx()).as_deref(), Some("2026"));
    }

    #[test]
    fn non_deictic_is_none() {
        assert_eq!(resolve("cat", &ctx()), None);
    }
}
