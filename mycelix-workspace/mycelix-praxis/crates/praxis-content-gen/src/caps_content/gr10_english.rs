// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CAPS English First Additional Language (FAL) Grade 10.
//! Reading comprehension, essay writing, visual literacy.

use super::TopicContent;

pub(crate) struct Gr10ReadingComprehension;
impl TopicContent for Gr10ReadingComprehension {
    fn explanation(&self) -> String {
        "Reading comprehension means understanding what you read — not just the words, but the meaning. \
         Good readers: 1) Preview (look at title, headings, pictures), 2) Question (what is this about?), \
         3) Read actively (underline key points), 4) Summarise (main idea in your own words), \
         5) Evaluate (do I agree? Is this fact or opinion?).".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Identify the main idea: 'South Africa has 11 official languages, more than most countries. This linguistic diversity reflects the nation's rich cultural heritage, but also creates challenges in education and government services.'\nMain idea: SA's many languages are both a strength and a challenge.\nSupporting details: 11 languages, cultural heritage, education challenges.".to_string(),
            1 => "Fact or opinion? 'Table Mountain is the most beautiful mountain in the world.'\nOpinion — 'most beautiful' is subjective. Different people have different views.\nFact would be: 'Table Mountain is 1,085 metres high.' (verifiable)".to_string(),
            _ => "What does 'reading between the lines' mean?\nIt means understanding the IMPLIED message — what the author suggests without directly stating.\nExample: 'She looked at the clock for the fifth time.' Implies: she is bored or impatient.\nThe author doesn't SAY she's bored — the reader infers it.".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Is this fact or opinion? 'The Kruger National Park covers about 20,000 km².'\nFact\nIt can be verified/measured. It's either true or false.\nOpinion, Both, Neither\nCan you check this with a measurement?\nIf it can be verified, it's a fact.".to_string(),
            301..=600 => "Read: 'Despite years of progress, access to quality education remains unequal in South Africa.' What is the author's tone?\nConcerned/critical\nWords like 'despite' and 'remains unequal' show dissatisfaction with the current situation.\nHappy, Neutral, Angry\nLook at word choices. Do they express positive or negative feelings?\n'Despite progress' = acknowledging good, but 'remains unequal' = still a problem.".to_string(),
            _ => "A newspaper article uses the headline: 'Youth DEMAND change!' vs 'Youth request reforms.' How does word choice affect the message?\n'DEMAND' is strong, urgent, possibly aggressive. 'Request' is polite, gentle. Same event, different impressions.\nSame meaning, Unimportant\nHow does each word make you FEEL about the youth?\nConnotation: the emotional associations of a word beyond its literal meaning.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "Ask: What is the MAIN POINT the author is making?".to_string(), 2 => "Look for signal words: however, therefore, despite, although.".to_string(), _ => "Fact = can be proven. Opinion = someone's belief or judgment.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Comprehension means you understand every word.\nRIGHT: You can understand the overall meaning even if some words are unfamiliar. Use context clues!\nWHY: Context = the words around the unknown word. They often give hints about meaning.".to_string() }
    fn vocabulary(&self) -> String { "inference: A conclusion drawn from evidence in the text | 'He shivered' — we infer he is cold.\nconnotation: The feeling or association a word carries | 'Home' feels warm; 'house' feels neutral.\ntone: The author's attitude toward the subject | Angry, hopeful, sarcastic, objective.".to_string() }
    fn flashcard(&self) -> String { "What is an inference? | A conclusion based on evidence + reasoning (not directly stated)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Read a passage about water shortages in Cape Town and identify: main idea, 2 supporting facts, the author's purpose, and one inference you can make.\nStructured response using all 4 skills.\n4\nMain idea = one sentence summary. Facts = verifiable. Purpose = why written. Inference = read between lines.".to_string() }
}
