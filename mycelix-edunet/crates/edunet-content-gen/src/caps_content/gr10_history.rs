// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CAPS History Grade 10 — The world in 1600s-1800s, colonialism, SA history.

use super::TopicContent;

pub(crate) struct Gr10Colonialism;
impl TopicContent for Gr10Colonialism {
    fn explanation(&self) -> String {
        "Colonialism is when one country takes control of another's land, people, and resources. \
         European countries colonised much of Africa, Asia, and the Americas from the 1500s-1900s. \
         In South Africa, the Dutch (1652) and British (1806) established colonies. \
         The effects included: loss of land and freedom for indigenous peoples, \
         forced labour, cultural destruction, but also new technologies and trade routes. \
         Understanding colonialism helps us understand today's inequalities.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Why did Europeans colonise Africa?\nPush factors: competition between European nations, need for raw materials.\nPull factors: Africa's resources (gold, diamonds, rubber, ivory).\nEnabling factors: superior military technology (guns vs spears), diseases that weakened African populations.".to_string(),
            1 => "How did colonialism change South African society?\nBefore: diverse African kingdoms (Zulu, Xhosa, Sotho) with own governance.\nDuring: land dispossession, pass laws, racial segregation.\nAfter: lasting economic inequality, cultural trauma, but also united resistance movements.".to_string(),
            _ => "Source analysis: A 19th-century British cartoon shows Africa as 'dark' and Europe bringing 'light'.\nBias: The source promotes the 'civilising mission' — the idea that Europeans were helping Africans.\nReality: Colonialism was primarily about economic exploitation, not helping.\nAlways ask: Who made this? Why? What's missing?".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "When did the Dutch arrive at the Cape?\n1652\nJan van Riebeeck established a refreshment station for the VOC.\n1806, 1994, 1488\nThe VOC needed supplies for ships sailing to the East.\nDutch East India Company (VOC) sent Van Riebeeck.".to_string(),
            301..=600 => "Explain TWO effects of colonialism on African societies.\n1. Loss of land and political independence. 2. Introduction of European education and Christianity, which undermined traditional cultures.\nOther valid: forced labour, racial discrimination, new diseases, cash crop economies.\nNo effects, Only positive effects\nThink about land, culture, economy, and freedom.\nWho gained? Who lost? What changed?".to_string(),
            _ => "Was colonialism 'necessary' for Africa's development? Discuss BOTH sides.\nFor: introduced infrastructure, modern medicine, education, trade connections.\nAgainst: exploitation, genocide, cultural destruction, created dependency. Africa had thriving civilisations before colonialism (Mali, Great Zimbabwe, Axum).\nYes definitely, No never\nConsider multiple perspectives.\nWhat did Africa have BEFORE colonialism? What was TAKEN?".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "Think about WHY colonisers came and WHAT they wanted.".to_string(), 2 => "Consider economic, political, social, and cultural effects.".to_string(), _ => "Always evaluate sources: Who wrote it? When? Why? What bias?".to_string() } }
    fn misconception(&self) -> String { "WRONG: Africa had no history or civilisation before Europeans arrived.\nRIGHT: Africa had powerful kingdoms (Mali, Songhai, Great Zimbabwe, Axum), universities (Timbuktu), and trade networks spanning continents.\nWHY: Colonial narratives deliberately erased African achievements to justify domination.".to_string() }
    fn vocabulary(&self) -> String { "colonialism: A country taking control of another's land and people | Britain colonised India, Nigeria, South Africa.\nimperialism: Building an empire by controlling other territories | The 'Scramble for Africa' (1881-1914).\nsource analysis: Examining evidence critically — who, when, why, bias | Essential history skill.".to_string() }
    fn flashcard(&self) -> String { "What was the 'Scramble for Africa'? | European nations dividing Africa among themselves (1881-1914, Berlin Conference 1884)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Compare the Zulu Kingdom under Shaka with British colonial rule. Which was more democratic?\nShaka: centralised military monarchy (strong but autocratic). British: parliamentary system at home but authoritarian colonial rule in SA.\nNeither was fully democratic — both concentrated power.\n3\nDefine 'democratic' first. Then evaluate BOTH systems.".to_string() }
}

pub(crate) struct Gr10HistoryFallback;
impl TopicContent for Gr10HistoryFallback {
    fn explanation(&self) -> String { "Grade 10 History covers the world from 1600-1900: colonialism, the French Revolution, industrialisation, and the origins of South African conflicts.".to_string() }
    fn worked_example(&self, _i: usize) -> String { "Analyse a political cartoon from 1884 showing European leaders cutting a cake shaped like Africa.\nThe 'cake' = Africa's resources being divided.\nThe leaders' expressions = casual attitude toward African peoples.\nConclusion: Europeans treated Africa as property, not as home to millions of people.".to_string() }
    fn practice_problem(&self, _d: u16) -> String { "Name 3 African kingdoms that existed before European colonisation.\nExamples: Zulu Kingdom, Kingdom of Kongo, Ashanti Empire, Ethiopian Empire, Mali Empire, Great Zimbabwe.\nThere were none\nThink about different regions of Africa.\nWest Africa: Mali, Songhai. Southern Africa: Zulu, Zimbabwe.".to_string() }
    fn hint(&self, l: u8) -> String { match l { 1 => "History requires evidence — always refer to sources.".to_string(), _ => "Every source has a perspective. Ask: whose voice is missing?".to_string() } }
    fn misconception(&self) -> String { "WRONG: History is just memorising dates.\nRIGHT: History is about understanding WHY things happened and HOW they connect to today.\nWHY: Dates are facts, but understanding causes and effects is real historical thinking.".to_string() }
    fn vocabulary(&self) -> String { "primary source: Evidence from the time period studied | A diary from 1652.\nsecondary source: Written later, analysing primary sources | A 2024 textbook about colonialism.".to_string() }
    fn flashcard(&self) -> String { "Primary or secondary: a historian's book about the Anglo-Boer War? | Secondary (written after the event)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Why is it important to study multiple perspectives of historical events?\nDifferent groups experienced events differently. A single perspective gives an incomplete picture. Multiple sources reveal bias and help us approach truth.\n3\nThink about who writes history and whose stories are often left out.".to_string() }
}
