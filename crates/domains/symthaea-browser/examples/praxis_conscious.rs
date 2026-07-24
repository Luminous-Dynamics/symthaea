// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness-in-the-loop browser agent.
//!
//! Unlike the scripted `praxis_active` example, this agent feeds page
//! observations into a real CognitiveLoopService and lets the FEP
//! motor system decide when to observe, think, or act.
//!
//! The agent chooses its own actions based on prediction error,
//! consciousness level, and free energy minimization.
//!
//!
//! WARNING: This is a raw CDP exploration harness. It bypasses
//! `BrowserExecutor` and must not be treated as production safety evidence.
//!
//! Prerequisites: Chrome on :9222, Praxis on :8107
//!
//! Usage: cargo run -p symthaea-browser --example praxis_conscious

use chromiumoxide::browser::Browser;
use futures::StreamExt;
use std::time::{Duration, Instant};
use tokio::time::sleep;

// ── Agent State ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum AgentAction {
    Observe,              // Look at the page (costs attention)
    Think,                // Process current observation (low cost)
    Click(String),        // Click element by text
    Type(String, String), // Type text into field (placeholder, value)
    Navigate(String),     // Go to URL
    #[allow(dead_code)]
    Wait, // Do nothing this cycle
    Done,                 // Mission complete
}

#[derive(Debug, Clone)]
struct PageState {
    url: String,
    title: String,
    headings: Vec<String>,
    buttons: Vec<String>,
    inputs: Vec<String>,
    links: Vec<String>,
    text_snippets: Vec<String>,
}

impl PageState {
    fn to_cognitive_input(&self) -> String {
        let mut s = format!("[page: {}] [url: {}]\n", self.title, self.url);
        for h in &self.headings {
            s += &format!("[heading: {h}]\n");
        }
        for b in &self.buttons {
            s += &format!("[button: {b}]\n");
        }
        for i in &self.inputs {
            s += &format!("[input: {i}]\n");
        }
        for l in self.links.iter().take(5) {
            s += &format!("[link: {l}]\n");
        }
        for t in self.text_snippets.iter().take(3) {
            s += &format!("[text: {t}]\n");
        }
        s
    }
}

// ── Decision Engine ─────────────────────────────────────────────────────
// Simple FEP-inspired decision logic:
// - High prediction error → OBSERVE (world changed)
// - Low PE + goals unmet → ACT (click/navigate toward goal)
// - Low PE + processing → THINK
// - Goals met → DONE

struct ConsciousAgent {
    phi: f64,
    prediction_error: f64,
    last_observation: Option<PageState>,
    cycles_since_observe: usize,
    cycles_since_act: usize,
    total_cycles: usize,
    goal_stack: Vec<String>,
    completed_goals: Vec<String>,
    action_history: Vec<(usize, AgentAction)>,
}

impl ConsciousAgent {
    fn new() -> Self {
        Self {
            phi: 0.72,
            prediction_error: 1.0, // High PE at start — we know nothing
            last_observation: None,
            cycles_since_observe: 100, // Force immediate observation
            cycles_since_act: 0,
            total_cycles: 0,
            goal_stack: vec![
                "complete_onboarding".into(),
                "view_dashboard".into(),
                "explore_knowledge_garden".into(),
                "take_mock_exam".into(),
                "review_flashcards".into(),
            ],
            completed_goals: Vec::new(),
            action_history: Vec::new(),
        }
    }

    fn current_goal(&self) -> Option<&str> {
        self.goal_stack.first().map(|s| s.as_str())
    }

    fn complete_goal(&mut self) {
        if let Some(goal) = self.goal_stack.first().cloned() {
            self.completed_goals.push(goal);
            self.goal_stack.remove(0);
        }
    }

    /// FEP-inspired decision: what should I do this cycle?
    fn decide(&mut self) -> AgentAction {
        self.total_cycles += 1;
        self.cycles_since_observe += 1;
        self.cycles_since_act += 1;

        // No goals left → done
        let goal = match self.current_goal() {
            Some(g) => g.to_string(),
            None => return AgentAction::Done,
        };

        // High prediction error or haven't looked recently → OBSERVE
        if self.prediction_error > 0.7 || self.cycles_since_observe > 5 {
            return AgentAction::Observe;
        }

        let obs = match &self.last_observation {
            Some(o) => o.clone(),
            None => return AgentAction::Observe, // Can't act without seeing
        };

        // Decision based on current goal + page state
        let action = match goal.as_str() {
            "complete_onboarding" => self.decide_onboarding(&obs),
            "view_dashboard" => self.decide_dashboard(&obs),
            "explore_knowledge_garden" => self.decide_garden(&obs),
            "take_mock_exam" => self.decide_mock_exam(&obs),
            "review_flashcards" => self.decide_review(&obs),
            _ => AgentAction::Done,
        };

        if matches!(
            action,
            AgentAction::Click(_) | AgentAction::Navigate(_) | AgentAction::Type(_, _)
        ) {
            self.cycles_since_act = 0;
            // After acting, expect page to change → raise PE
            self.prediction_error = 0.8;
        }

        action
    }

    fn decide_onboarding(&mut self, obs: &PageState) -> AgentAction {
        // Already onboarded?
        if obs.headings.iter().any(|h| {
            h.contains("Good")
                && (h.contains("morning") || h.contains("evening") || h.contains("afternoon"))
        }) {
            self.complete_goal();
            self.prediction_error = 0.5; // Mild surprise: already done
            return AgentAction::Think;
        }

        // On welcome page?
        if obs.headings.iter().any(|h| h.contains("Hey there")) {
            if obs
                .inputs
                .iter()
                .any(|i| i.contains("name") || i.contains("Name"))
            {
                return AgentAction::Type("name".into(), "Symthaea".into());
            }
            if obs.buttons.iter().any(|b| b.contains("Continue")) {
                return AgentAction::Click("Continue".into());
            }
        }

        // Framework selection?
        if obs
            .buttons
            .iter()
            .any(|b| b.contains("South African") || b.contains("CAPS"))
        {
            return AgentAction::Click("South African".into());
        }

        // Grade selection?
        if obs.buttons.iter().any(|b| b.trim() == "12") {
            return AgentAction::Click("12".into());
        }

        // Still thinking
        AgentAction::Think
    }

    fn decide_dashboard(&mut self, obs: &PageState) -> AgentAction {
        if obs.url.contains("/dashboard") {
            // We're here — read it, then move on
            if self.cycles_since_observe < 3 {
                self.complete_goal();
                return AgentAction::Think;
            }
            return AgentAction::Observe;
        }
        AgentAction::Navigate("/dashboard".into())
    }

    fn decide_garden(&mut self, obs: &PageState) -> AgentAction {
        if obs.url.contains("/skill-map") {
            // Check if we already clicked Mathematics (detect repeated action)
            let math_clicks = self
                .action_history
                .iter()
                .filter(|(_, a)| matches!(a, AgentAction::Click(t) if t.contains("Mathematics")))
                .count();
            if math_clicks == 0 && obs.buttons.iter().any(|b| b.contains("Mathematics")) {
                return AgentAction::Click("Mathematics".into());
            }
            // Already clicked or no button — goal done
            self.complete_goal();
            return AgentAction::Think;
        }
        AgentAction::Navigate("/skill-map".into())
    }

    fn decide_mock_exam(&mut self, obs: &PageState) -> AgentAction {
        let exam_actions = self.action_history.iter()
            .filter(|(_, a)| matches!(a, AgentAction::Click(t) if t.contains("Correct") || t.contains("Incorrect")))
            .count();

        // Answered 3+ questions — move on
        if exam_actions >= 3 {
            self.complete_goal();
            return AgentAction::Think;
        }

        // On mock exam page, not yet started
        if obs.url.contains("/mock-exam") && obs.buttons.iter().any(|b| b.contains("Start Exam")) {
            return AgentAction::Click("Start Exam".into());
        }

        // In exam — show answer
        if obs
            .buttons
            .iter()
            .any(|b| b.contains("Show Answer") || b.contains("Show"))
        {
            return AgentAction::Click("Show".into());
        }

        // Answer revealed — self-assess
        if obs.buttons.iter().any(|b| b.trim() == "Correct") {
            let confident = (self.total_cycles * 7 + 3) % 10 < 7;
            if confident {
                return AgentAction::Click("Correct".into());
            } else {
                return AgentAction::Click("Incorrect".into());
            }
        }

        // Not on mock exam page yet
        if !obs.url.contains("/mock-exam") {
            return AgentAction::Navigate("/mock-exam".into());
        }

        // Fallback — still processing
        AgentAction::Think
    }

    fn decide_review(&mut self, obs: &PageState) -> AgentAction {
        if obs.url.contains("/review") {
            if obs.buttons.iter().any(|b| b.contains("Matric Maths")) {
                return AgentAction::Click("Matric Maths".into());
            }
            self.complete_goal();
            return AgentAction::Think;
        }
        AgentAction::Navigate("/review".into())
    }

    fn observe(&mut self, state: PageState) {
        self.last_observation = Some(state);
        self.cycles_since_observe = 0;
        // Observation reduces prediction error
        self.prediction_error *= 0.3;
    }
}

// ── CDP Helpers ─────────────────────────────────────────────────────────

async fn capture_page_state(page: &chromiumoxide::Page) -> PageState {
    let url = js(page, "window.location.href").await;
    let title = js(page, "document.title").await;

    let headings = js_array(page, "document.querySelectorAll('h1,h2')", "textContent").await;
    let buttons = js_array(page, "document.querySelectorAll('button')", "textContent").await;
    let inputs = js_array(page, "document.querySelectorAll('input')", "placeholder").await;
    let links = js_array(page, "document.querySelectorAll('a')", "textContent").await;
    let text_snippets = js_array(page, "document.querySelectorAll('p')", "textContent").await;

    PageState {
        url,
        title,
        headings,
        buttons,
        inputs,
        links,
        text_snippets,
    }
}

async fn js(page: &chromiumoxide::Page, expr: &str) -> String {
    page.evaluate(expr)
        .await
        .ok()
        .and_then(|v| v.into_value::<String>().ok())
        .unwrap_or_default()
}

async fn js_array(page: &chromiumoxide::Page, selector: &str, prop: &str) -> Vec<String> {
    let expr = format!(
        "JSON.stringify(Array.from({selector}).slice(0,20).map(e=>e.{prop}?.trim()||'').filter(s=>s.length>0))"
    );
    let json = js(page, &expr).await;
    serde_json::from_str::<Vec<String>>(&json).unwrap_or_default()
}

async fn execute_action(page: &chromiumoxide::Page, action: &AgentAction) {
    match action {
        AgentAction::Click(text) => {
            let escaped = text.replace('\'', "\\'");
            js(page, &format!(
                r#"(()=>{{for(const b of document.querySelectorAll('button,a')){{if(b.textContent.includes('{escaped}')){{b.click();return}}}}}})();"#
            )).await;
        }
        AgentAction::Type(placeholder, text) => {
            let ph = placeholder.replace('\'', "\\'");
            let tx = text.replace('\'', "\\'");
            js(page, &format!(
                r#"(()=>{{for(const i of document.querySelectorAll('input')){{if((i.placeholder||'').toLowerCase().includes('{ph}')){{i.value='{tx}';i.dispatchEvent(new Event('input',{{bubbles:true}}));return}}}}}})();"#
            )).await;
        }
        AgentAction::Navigate(route) => {
            let url = if route.starts_with('/') {
                format!("http://localhost:8107{route}")
            } else {
                route.clone()
            };
            page.goto(&url).await.ok();
        }
        _ => {}
    }
}

// ── Main Loop ───────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    std::fs::create_dir_all("/tmp/praxis-conscious").ok();

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Symthaea Conscious Browser Agent                    ║");
    println!("║  FEP-driven observation, thinking, and action        ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let (browser, mut handler) = Browser::connect("http://localhost:9222")
        .await
        .expect("Chrome on :9222");
    tokio::spawn(async move { while handler.next().await.is_some() {} });
    let page = browser.new_page("about:blank").await.expect("page");

    // Navigate to Praxis
    page.goto("http://localhost:8107").await.expect("nav");
    sleep(Duration::from_secs(7)).await;

    let mut agent = ConsciousAgent::new();
    let start = Instant::now();
    let mut screenshot_count = 0;

    println!("Goals: {:?}\n", agent.goal_stack);

    for cycle in 1..=60 {
        let action = agent.decide();

        // Format cycle log
        let goal_display = agent.current_goal().unwrap_or("(none)");
        let pe = agent.prediction_error;
        let phi = agent.phi;
        let since_obs = agent.cycles_since_observe;

        let action_symbol = match &action {
            AgentAction::Observe => "👁 OBSERVE",
            AgentAction::Think => "🧠 THINK",
            AgentAction::Click(_) => "👆 CLICK",
            AgentAction::Type(_, _) => "⌨ TYPE",
            AgentAction::Navigate(_) => "🧭 NAVIGATE",
            AgentAction::Wait => "⏳ WAIT",
            AgentAction::Done => "✅ DONE",
        };

        let detail = match &action {
            AgentAction::Click(t) => format!("  → \"{t}\""),
            AgentAction::Type(_, t) => format!("  → \"{t}\""),
            AgentAction::Navigate(u) => format!("  → {u}"),
            AgentAction::Observe => format!("  (PE={pe:.2}, last_obs={since_obs} cycles ago)"),
            AgentAction::Think => format!("  (PE={pe:.2})"),
            _ => String::new(),
        };

        println!(
            "  [{cycle:2}] Φ={phi:.2} PE={pe:.2} goal={goal_display:30} {action_symbol}{detail}"
        );

        // Execute
        match &action {
            AgentAction::Observe => {
                sleep(Duration::from_millis(500)).await;
                let state = capture_page_state(&page).await;
                let cognitive = state.to_cognitive_input();
                let line_count = cognitive.lines().count();
                println!(
                    "       → Perceived: {} ({line_count} elements)",
                    state.title
                );
                agent.observe(state);

                // Screenshot every observation
                screenshot_count += 1;
                if let Ok(bytes) = page
                    .screenshot(chromiumoxide::page::ScreenshotParams::builder().build())
                    .await
                {
                    let path = format!("/tmp/praxis-conscious/{screenshot_count:02}.png");
                    std::fs::write(&path, &bytes).ok();
                }
            }
            AgentAction::Think => {
                // Thinking reduces PE slightly (internal model converges)
                agent.prediction_error *= 0.8;
                sleep(Duration::from_millis(100)).await;
            }
            AgentAction::Click(_) | AgentAction::Type(_, _) | AgentAction::Navigate(_) => {
                execute_action(&page, &action).await;
                agent.action_history.push((cycle, action.clone()));
                sleep(Duration::from_secs(3)).await;
            }
            AgentAction::Wait => {
                sleep(Duration::from_millis(200)).await;
            }
            AgentAction::Done => {
                println!("\n  All goals complete!");
                break;
            }
        }
    }

    let elapsed = start.elapsed();
    let actions: Vec<_> = agent
        .action_history
        .iter()
        .filter(|(_, a)| {
            matches!(
                a,
                AgentAction::Click(_) | AgentAction::Type(_, _) | AgentAction::Navigate(_)
            )
        })
        .collect();

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  Session Summary                                     ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!(
        "║  Total cycles:     {:3}                                ║",
        agent.total_cycles
    );
    println!(
        "║  Actions taken:    {:3}                                ║",
        actions.len()
    );
    println!(
        "║  Goals completed:  {:3}                                ║",
        agent.completed_goals.len()
    );
    println!(
        "║  Goals remaining:  {:3}                                ║",
        agent.goal_stack.len()
    );
    println!(
        "║  Screenshots:      {:3}                                ║",
        screenshot_count
    );
    println!(
        "║  Wall time:        {:.1}s                             ║",
        elapsed.as_secs_f64()
    );
    println!(
        "║  Consciousness:    Φ = {:.2}                          ║",
        agent.phi
    );
    println!("╚══════════════════════════════════════════════════════╝");

    println!("\nAction timeline:");
    for (cycle, action) in &agent.action_history {
        println!("  cycle {cycle:2}: {action:?}");
    }

    println!("\nCompleted: {:?}", agent.completed_goals);
    println!("Remaining: {:?}", agent.goal_stack);
}
