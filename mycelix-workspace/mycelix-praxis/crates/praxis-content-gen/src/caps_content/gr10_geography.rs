// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CAPS Geography Grade 10 — Climate, Geomorphology, Population.

use super::TopicContent;

pub(crate) struct Gr10Climate;
impl TopicContent for Gr10Climate {
    fn explanation(&self) -> String {
        "Climate is the average weather over 30+ years. South Africa has diverse climates: \
         Mediterranean (Cape Town: wet winters, dry summers), Subtropical (Durban: warm, humid), \
         Semi-arid (Karoo: dry), and Highland (Johannesburg: warm days, cold nights). \
         Global climate is driven by: solar radiation, Earth's tilt (seasons), ocean currents, \
         and atmospheric circulation (Hadley, Ferrel, Polar cells).".to_string()
    }
    fn worked_example(&self, i: usize) -> String {
        match i {
            0 => "Why does Cape Town get rain in winter but not summer?\nThe westerly wind belt shifts south in summer → misses the Cape.\nIn winter it shifts north → brings rain to the Western Cape.\nThis is a Mediterranean climate pattern.".to_string(),
            1 => "Explain the rain shadow effect on the Drakensberg.\nMoist air from Indian Ocean rises over mountains → cools → rain on eastern side.\nAir descends on the western side → warms → dry conditions.\nThis creates the semi-arid Karoo.".to_string(),
            _ => "How does the Benguela Current affect Namibia's coast?\nCold current → cools air above → fog but little rain.\nCold water = less evaporation = desert coast (Namib Desert).\nContrast: warm Agulhas Current → moist air → more rain in KwaZulu-Natal.".to_string(),
        }
    }
    fn practice_problem(&self, d: u16) -> String {
        match d {
            0..=300 => "Name SA's two main ocean currents.\nBenguela (cold, west) and Agulhas (warm, east)\nThey dramatically affect coastal climates.\nGulf Stream, Kuroshio\nOne cold (Atlantic side), one warm (Indian side).\nBenguela = cold. Agulhas = warm.".to_string(),
            301..=600 => "Explain why Johannesburg is cooler than Durban despite being further from the equator.\nAltitude: Joburg is ~1,750m above sea level vs Durban at sea level.\nTemperature drops ~6.5°C per 1000m altitude (lapse rate).\nLatitude only, Ocean only\nWhat other factor besides latitude affects temperature?\nAltitude: higher = cooler.".to_string(),
            _ => "Predict how climate change will affect SA's water supply.\nHigher temperatures → more evaporation → drier interior. Changing rainfall patterns: some areas wetter, others drier. Cape Town water crisis (2018) is a preview. Agriculture in Free State threatened.\nNo effect, Only positive\nThink about temperature, rainfall, and who depends on water.\nFarmers, cities, ecosystems all need reliable water.".to_string(),
        }
    }
    fn hint(&self, l: u8) -> String { match l { 1 => "Climate depends on: latitude, altitude, ocean currents, distance from sea.".to_string(), 2 => "SA climate zones: Mediterranean, Subtropical, Semi-arid, Highland.".to_string(), _ => "Warm currents bring rain. Cold currents bring fog and desert.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Climate and weather are the same thing.\nRIGHT: Weather = day-to-day conditions. Climate = average patterns over 30+ years.\nWHY: A cold day in Durban doesn't mean it's not subtropical. Climate is the long-term pattern.".to_string() }
    fn vocabulary(&self) -> String { "climate: Average weather over 30+ years | SA has 4 main climate zones.\nlapse rate: Temperature drop with altitude (~6.5°C/1000m) | Why Joburg is cooler than Durban.\nrain shadow: Dry area on the leeward side of mountains | The Karoo is in the Drakensberg's rain shadow.".to_string() }
    fn flashcard(&self) -> String { "What causes the Namib Desert to exist right next to the ocean? | The cold Benguela Current cools air, reducing evaporation → no rain".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Draw a cross-section of SA from west to east coast showing how altitude and ocean currents affect climate.\nWest: cold Benguela → Namib/Karoo dry. Centre: Highveld plateau (altitude=cool). East: Drakensberg rain, then warm Agulhas → KZN subtropical.\n4\nInclude: ocean currents, altitude profile, rainfall distribution.".to_string() }
}
