// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Natural language time parser for calendar event creation.
//!
//! Parses inputs like:
//! - "Meeting tomorrow at 2pm"
//! - "Lunch with Alice next Monday"
//! - "Team standup every weekday at 9:30"
//! - "Conference April 15 10am-4pm"

use mail_leptos_types::*;

pub struct ParsedEvent {
    pub title: String,
    pub start_offset_hours: f64,
    pub duration_hours: f64,
    pub recurrence: Recurrence,
}

/// Parse natural language into event components.
pub fn parse_natural_event(input: &str) -> Option<ParsedEvent> {
    let lower = input.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    if words.is_empty() { return None; }

    // Handle relative time: "in X hours", "in X minutes", "in X mins"
    if lower.starts_with("in ") {
        // Try to parse "in <number> <unit>" pattern
        for (i, word) in words.iter().enumerate() {
            if *word == "in" {
                if let Some(num_str) = words.get(i + 1) {
                    if let Ok(num) = num_str.parse::<f64>() {
                        if let Some(unit) = words.get(i + 2) {
                            let offset_hours = if unit.starts_with("hour") || *unit == "h" || *unit == "hrs" {
                                num
                            } else if unit.starts_with("min") || *unit == "m" || *unit == "mins" {
                                num / 60.0
                            } else if unit.starts_with("day") {
                                num * 24.0
                            } else {
                                continue;
                            };
                            // Title is everything after "in N unit"
                            let title_words: Vec<&str> = words[i+3..].to_vec();
                            let title = if title_words.is_empty() {
                                format!("Event in {} {}", num_str, unit)
                            } else {
                                let raw = title_words.join(" ");
                                let mut chars = raw.chars();
                                match chars.next() {
                                    Some(c) => c.to_uppercase().to_string() + chars.as_str(),
                                    None => raw,
                                }
                            };
                            return Some(ParsedEvent {
                                title,
                                start_offset_hours: offset_hours,
                                duration_hours: 1.0,
                                recurrence: Recurrence::None,
                            });
                        }
                    }
                }
            }
        }
    }

    // Extract time (Xam/Xpm/X:XX)
    let mut hour: Option<u32> = None;
    let mut minute: u32 = 0;
    for word in &words {
        if let Some(h) = parse_time_word(word) {
            hour = Some(h.0);
            minute = h.1;
        }
    }

    // Extract day offset
    let mut day_offset: i32 = 0;
    if lower.contains("tomorrow") { day_offset = 1; }
    else if lower.contains("today") { day_offset = 0; }
    else if lower.contains("next monday") || lower.contains("monday") { day_offset = days_until_weekday(1); }
    else if lower.contains("next tuesday") || lower.contains("tuesday") { day_offset = days_until_weekday(2); }
    else if lower.contains("next wednesday") || lower.contains("wednesday") { day_offset = days_until_weekday(3); }
    else if lower.contains("next thursday") || lower.contains("thursday") { day_offset = days_until_weekday(4); }
    else if lower.contains("next friday") || lower.contains("friday") { day_offset = days_until_weekday(5); }
    else if lower.contains("next saturday") || lower.contains("saturday") { day_offset = days_until_weekday(6); }
    else if lower.contains("next sunday") || lower.contains("sunday") { day_offset = days_until_weekday(0); }
    else if lower.contains("next week") { day_offset = 7; }
    else if lower.contains("next month") { day_offset = 30; }

    // Check for month names: "April 15", "Jan 3", etc.
    let months = [("january",1),("february",2),("march",3),("april",4),("may",5),("june",6),
                  ("july",7),("august",8),("september",9),("october",10),("november",11),("december",12),
                  ("jan",1),("feb",2),("mar",3),("apr",4),("jun",6),("jul",7),("aug",8),("sep",9),("oct",10),("nov",11),("dec",12)];
    for (name, month_num) in &months {
        if lower.contains(name) {
            // Look for a day number after the month name
            if let Some(pos) = lower.find(name) {
                let after = &lower[pos + name.len()..].trim_start();
                if let Some(day_str) = after.split_whitespace().next() {
                    if let Ok(day) = day_str.trim_matches(|c: char| !c.is_numeric()).parse::<u32>() {
                        let now = js_sys::Date::new_0();
                        let target = js_sys::Date::new_0();
                        target.set_full_year(now.get_full_year());
                        target.set_month(*month_num - 1);
                        target.set_date(day);
                        let diff_ms = target.get_time() - now.get_time();
                        day_offset = (diff_ms / 86400000.0) as i32;
                        if day_offset < 0 { day_offset += 365; } // next year
                    }
                }
            }
            break;
        }
    }

    // Extract recurrence
    let recurrence = if lower.contains("every day") || lower.contains("daily") {
        Recurrence::Daily
    } else if lower.contains("every week") || lower.contains("weekly") || lower.contains("every weekday") {
        Recurrence::Weekly
    } else if lower.contains("every month") || lower.contains("monthly") {
        Recurrence::Monthly
    } else {
        Recurrence::None
    };

    // Extract duration — check for time ranges first ("10am-4pm")
    let mut duration = 1.0f64;
    // Look for "Xam-Ypm" or "X:XX-Y:XX" pattern
    for word in &words {
        if word.contains('-') {
            let parts: Vec<&str> = word.split('-').collect();
            if parts.len() == 2 {
                if let (Some(start), Some(end)) = (parse_time_word(parts[0]), parse_time_word(parts[1])) {
                    let start_h = start.0 as f64 + start.1 as f64 / 60.0;
                    let end_h = end.0 as f64 + end.1 as f64 / 60.0;
                    let d = end_h - start_h;
                    if d > 0.0 {
                        duration = d;
                        if hour.is_none() { hour = Some(start.0); minute = start.1; }
                    }
                }
            }
        }
    }
    // Fallback to explicit duration keywords
    if duration == 1.0 {
        if lower.contains("all day") { duration = 24.0; }
        else if lower.contains("2 hour") || lower.contains("2h") { duration = 2.0; }
        else if lower.contains("30 min") || lower.contains("30m") { duration = 0.5; }
        else if lower.contains("3 hour") || lower.contains("3h") { duration = 3.0; }
    }

    // Extract title (everything that's not a time/day keyword)
    let skip_words = ["at", "on", "next", "every", "tomorrow", "today", "am", "pm",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "daily", "weekly", "monthly", "hour", "hours", "min", "minutes", "all", "day",
        "1h", "2h", "30m", "for", "the", "a", "an"];
    let title: String = words.iter()
        .filter(|w| {
            !skip_words.contains(w) && parse_time_word(w).is_none()
        })
        .copied()
        .collect::<Vec<_>>()
        .join(" ");

    let title = if title.is_empty() { input.to_string() } else {
        // Capitalize first letter
        let mut chars = title.chars();
        match chars.next() {
            Some(c) => c.to_uppercase().to_string() + chars.as_str(),
            None => title,
        }
    };

    // Calculate start as hours from now
    let now = js_sys::Date::new_0();
    let now_hour = now.get_hours();
    let target_hour = hour.unwrap_or(now_hour + 1);
    let start_offset = (day_offset as f64 * 24.0) + (target_hour as f64 - now_hour as f64) + (minute as f64 / 60.0);

    Some(ParsedEvent {
        title,
        start_offset_hours: start_offset,
        duration_hours: duration,
        recurrence,
    })
}

fn parse_time_word(word: &str) -> Option<(u32, u32)> {
    let w = word.trim_end_matches(|c: char| c == ',' || c == '.');

    // "2pm", "3am", "10pm"
    if let Some(h) = w.strip_suffix("pm") {
        if let Ok(mut hour) = h.parse::<u32>() {
            if hour != 12 { hour += 12; }
            return Some((hour, 0));
        }
    }
    if let Some(h) = w.strip_suffix("am") {
        if let Ok(mut hour) = h.parse::<u32>() {
            if hour == 12 { hour = 0; }
            return Some((hour, 0));
        }
    }

    // "14:30", "9:00"
    if w.contains(':') {
        let parts: Vec<&str> = w.split(':').collect();
        if parts.len() == 2 {
            if let (Ok(h), Ok(m)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                return Some((h, m));
            }
        }
    }

    None
}

fn days_until_weekday(target: u32) -> i32 {
    let now = js_sys::Date::new_0();
    let current = now.get_day(); // 0=Sun
    let mut diff = target as i32 - current as i32;
    if diff <= 0 { diff += 7; }
    diff
}
