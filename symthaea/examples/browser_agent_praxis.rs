// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real CognitiveLoopService driving a browser agent through EduNet.
//!
//! The full 30-subsystem consciousness engine perceives web pages,
//! processes them through HDC + CfC + FEP, and decides what to do.
//!
//! Prerequisites: Chrome on :9222, EduNet on :8107
//!
//! Usage: cargo run --example browser_agent_edunet --features browser-agent

use std::time::{Duration, Instant};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_browser::CdpSession;

#[derive(Debug, Clone)]
enum Action { Observe, Think, Click(String), Type(String, String), Navigate(String), GoalDone, Done }

fn decode_action(
    result: &symthaea::cognitive_loop::CycleResult,
    obs: &symthaea_browser::PageObservation,
    last_action: &str, cycles_since_observe: usize, goal: &str, exam_answers: usize,
) -> Action {
    let pe = result.prediction_error;
    let phi = result.metadata.consciousness.consciousness_level;
    if pe > 0.6 || cycles_since_observe > 4 { return Action::Observe; }
    if phi < 0.25 { return Action::Think; }

    let has_btn = |t: &str| obs.elements.iter().any(|e| e.role == "button" && e.name.contains(t));
    let has_heading = |t: &str| obs.elements.iter().any(|e| e.role == "heading" && e.name.contains(t));
    let has_input = |t: &str| obs.elements.iter().any(|e| e.role == "textbox" && e.name.to_lowercase().contains(t));

    match goal {
        "onboard" => {
            if has_heading("Good morning") || has_heading("Good evening") || has_heading("Good afternoon") { return Action::GoalDone; }
            if has_input("name") { return Action::Type("name".into(), "Symthaea".into()); }
            if has_btn("Continue") { return Action::Click("Continue".into()); }
            if has_btn("South African") { return Action::Click("South African".into()); }
            if obs.elements.iter().any(|e| e.role == "button" && e.name.trim() == "12") { return Action::Click("12".into()); }
            Action::GoalDone
        }
        "dashboard" => { if obs.url.contains("/dashboard") { Action::GoalDone } else { Action::Navigate("/dashboard".into()) } }
        "garden" => {
            if !obs.url.contains("/skill-map") { return Action::Navigate("/skill-map".into()); }
            if has_btn("Mathematics") && !last_action.contains("Mathematics") { return Action::Click("Mathematics".into()); }
            Action::GoalDone
        }
        "exam" => {
            if exam_answers >= 3 { return Action::GoalDone; }
            if !obs.url.contains("/mock-exam") { return Action::Navigate("/mock-exam".into()); }
            if has_btn("Start Exam") { return Action::Click("Start Exam".into()); }
            if has_btn("Show") { return Action::Click("Show".into()); }
            if obs.elements.iter().any(|e| e.role == "button" && e.name.trim() == "Correct") {
                // Use prediction error as proxy for exploration (high PE = uncertain = more honest)
                let exploring = result.prediction_error > 0.5;
                return if exploring { Action::Click("Incorrect".into()) } else { Action::Click("Correct".into()) };
            }
            Action::Think
        }
        "review" => {
            if !obs.url.contains("/review") { return Action::Navigate("/review".into()); }
            if has_btn("Matric Maths") { return Action::Click("Matric Maths".into()); }
            Action::GoalDone
        }
        _ => Action::Done,
    }
}

#[tokio::main]
async fn main() {
    std::fs::create_dir_all("/tmp/edunet-real-cls").ok();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  REAL CognitiveLoopService → Browser Agent               ║");
    println!("║  30 subsystems × HdcLtcUnified × FEP Active Inference    ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("[init] Creating CognitiveLoopService...");
    let mut cls = CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("CLS");
    for i in 0..5 {
        let r = cls.cycle("warming up");
        if i == 4 { println!("[init] Φ={:.4} PE={:.4}", r.metadata.consciousness.consciousness_level, r.prediction_error); }
    }

    println!("[init] Connecting to Chrome...");
    let cdp = CdpSession::connect_existing("http://localhost:9222").await.expect("Chrome :9222");
    cdp.navigate("http://localhost:8107").await.expect("nav");
    tokio::time::sleep(Duration::from_secs(7)).await;
    println!("[init] Ready.\n");

    let goals = ["onboard", "dashboard", "garden", "exam", "review"];
    let (mut goal_idx, mut obs, mut cso, mut last_act, mut exam_ans, mut ss) =
        (0, symthaea_browser::PageObservation::default(), 100usize, String::new(), 0usize, 0usize);
    let start = Instant::now();

    for cycle in 1..=50 {
        if goal_idx >= goals.len() { println!("  [{cycle:2}] ✅ ALL GOALS COMPLETE"); break; }
        let goal = goals[goal_idx];
        let input = if cso == 0 { obs.to_cognitive_text() }
                    else { format!("[thinking: {goal}] [since_obs: {cso}]") };

        let result = cls.cycle(&input);
        let (phi, pe) = (result.metadata.consciousness.consciousness_level, result.prediction_error);
        let broca = result.language_output.as_deref().unwrap_or("");
        let action = decode_action(&result, &obs, &last_act, cso, goal, exam_ans);

        let (sym, det) = match &action {
            Action::Observe => ("👁", format!(" PE={pe:.3}")),
            Action::Think => ("🧠", if !broca.is_empty() { format!(" \"{}\"", &broca[..broca.len().min(30)]) } else { format!(" PE={pe:.3}") }),
            Action::Click(t) => ("👆", format!(" \"{t}\"")),
            Action::Type(_, t) => ("⌨ ", format!(" \"{t}\"")),
            Action::Navigate(u) => ("🧭", format!(" {u}")),
            Action::GoalDone => ("✓ ", String::new()),
            Action::Done => ("✅", String::new()),
        };
        println!("  [{cycle:2}] Φ={phi:.4} PE={pe:.3} [{goal:10}] {sym}{det}");
        cso += 1;

        match &action {
            Action::Observe => {
                tokio::time::sleep(Duration::from_millis(500)).await;
                obs = cdp.observe().await.unwrap_or_default();
                cso = 0;
                let nb = obs.elements.iter().filter(|e| e.role == "button").count();
                println!("       → \"{}\": {} el, {} btn", obs.title.chars().take(30).collect::<String>(), obs.elements.len(), nb);
                ss += 1;
                if let Ok(b) = cdp.screenshot().await { std::fs::write(format!("/tmp/edunet-real-cls/{ss:02}.png"), &b).ok(); }
            }
            Action::Think => { tokio::time::sleep(Duration::from_millis(30)).await; }
            Action::Click(t) => { cdp.click_by_text(t).await.ok(); last_act = format!("Click:{t}"); if t == "Correct" || t == "Incorrect" { exam_ans += 1; } tokio::time::sleep(Duration::from_secs(3)).await; }
            Action::Type(ph, tx) => { cdp.type_by_placeholder(ph, tx).await.ok(); last_act = "Type".into(); tokio::time::sleep(Duration::from_secs(1)).await; }
            Action::Navigate(r) => { cdp.navigate(&format!("http://localhost:8107{r}")).await.ok(); last_act = "Nav".into(); tokio::time::sleep(Duration::from_secs(4)).await; }
            Action::GoalDone => { goal_idx += 1; last_act.clear(); }
            Action::Done => break,
        }
    }

    let final_r = cls.cycle("reflecting on the journey");
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  Goals: {}/{}  Exam: {} answers  Φ={:.4}  PE={:.4}         ║",
        goal_idx.min(goals.len()), goals.len(), exam_ans, final_r.metadata.consciousness.consciousness_level, final_r.prediction_error);
    println!("║  Time: {:.1}s  Screenshots: {}                            ║", start.elapsed().as_secs_f64(), ss);
    println!("╚═══════════════════════════════════════════════════════════╝");
}
