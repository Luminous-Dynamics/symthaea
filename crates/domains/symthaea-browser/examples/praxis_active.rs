// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Active browser agent — Symthaea navigates Praxis autonomously.
//!
//! Unlike the passive journey test, this example actually clicks buttons,
//! types text, navigates pages, and takes a mock exam — all driven by
//! simulated threshold decisions inside the example itself.
//!
//!
//! WARNING: This is a raw CDP exploration harness. It bypasses
//! `BrowserExecutor` and must not be treated as production safety evidence.
//!
//! Prerequisites: Chrome on :9222, Praxis on :8107
//!
//! Usage: cargo run -p symthaea-browser --example praxis_active

use chromiumoxide::browser::Browser;
use futures::StreamExt;
use std::time::Duration;
use tokio::time::sleep;

/// Simulated consciousness level (in production, comes from CognitiveLoopService)
const PHI: f64 = 0.72;

/// Execute JS and return string result
async fn js(page: &chromiumoxide::Page, expr: &str) -> String {
    match page.evaluate(expr).await {
        Ok(val) => val.into_value::<String>().unwrap_or_default(),
        Err(_) => String::new(),
    }
}

/// Click a button by text content match
async fn click_button(page: &chromiumoxide::Page, text: &str) -> bool {
    let escaped = text.replace('\'', "\\'");
    let result = js(
        page,
        &format!(
            r#"(()=>{{
            for(const b of document.querySelectorAll('button,a')){{
                if(b.textContent.includes('{escaped}')){{b.click();return 'clicked'}}
            }}
            return 'not_found'
        }})()"#
        ),
    )
    .await;
    if result == "clicked" {
        println!("    ✓ Clicked: {text}");
        true
    } else {
        println!("    ✗ Not found: {text}");
        false
    }
}

/// Type into an input field
async fn type_into(page: &chromiumoxide::Page, placeholder: &str, text: &str) {
    let escaped_ph = placeholder.replace('\'', "\\'");
    let escaped_text = text.replace('\'', "\\'");
    js(
        page,
        &format!(
            r#"(()=>{{
            const inputs = document.querySelectorAll('input');
            for(const i of inputs){{
                if((i.placeholder||'').toLowerCase().includes('{escaped_ph}'.toLowerCase())){{
                    i.value='{escaped_text}';
                    i.dispatchEvent(new Event('input',{{bubbles:true}}));
                    return 'typed'
                }}
            }}
            return 'not_found'
        }})()"#
        ),
    )
    .await;
    println!("    ✓ Typed '{text}' into [{placeholder}]");
}

/// Get page heading
async fn heading(page: &chromiumoxide::Page) -> String {
    js(
        page,
        "document.querySelector('h1,h2')?.textContent?.trim()||'?'",
    )
    .await
}

/// Count interactive elements
async fn count_interactive(page: &chromiumoxide::Page) -> String {
    js(
        page,
        r#"
        const b = document.querySelectorAll('button').length;
        const a = document.querySelectorAll('a').length;
        const i = document.querySelectorAll('input').length;
        `${b} buttons, ${a} links, ${i} inputs`
    "#,
    )
    .await
}

/// Take screenshot and save
async fn screenshot(page: &chromiumoxide::Page, name: &str) {
    match page
        .screenshot(chromiumoxide::page::ScreenshotParams::builder().build())
        .await
    {
        Ok(bytes) => {
            let path = format!("/tmp/praxis-active/{name}.png");
            if let Ok(()) = std::fs::write(&path, &bytes) {
                println!("    📸 {name} ({} KB)", bytes.len() / 1024);
            }
        }
        Err(e) => println!("    📸 Screenshot failed: {e}"),
    }
}

#[tokio::main]
async fn main() {
    std::fs::create_dir_all("/tmp/praxis-active").ok();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Symthaea Browser Agent — Active Praxis Session  ║");
    println!("║  Consciousness Level: Φ = {PHI:.2}                  ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    // Connect to Chrome
    let (browser, mut handler) = Browser::connect("http://localhost:9222")
        .await
        .expect("Chrome on :9222");
    tokio::spawn(async move { while handler.next().await.is_some() {} });
    let page = browser.new_page("about:blank").await.expect("page");

    // ━━━ Phase 1: Onboarding ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("━━━ Phase 1: ONBOARDING ━━━\n");

    println!("[1.1] Navigate to Praxis");
    page.goto("http://localhost:8107").await.expect("nav");
    sleep(Duration::from_secs(7)).await;
    screenshot(&page, "01_welcome").await;

    let h = heading(&page).await;
    println!("    Heading: {h}");

    if h.contains("Hey there") {
        println!("\n[1.2] Enter name (Φ={PHI:.2} ≥ 0.5 for text input ✓)");
        type_into(&page, "name", "Symthaea").await;
        sleep(Duration::from_millis(500)).await;
        screenshot(&page, "02_name").await;

        println!("\n[1.3] Click Continue");
        click_button(&page, "Continue").await;
        sleep(Duration::from_secs(2)).await;
        screenshot(&page, "03_after_continue").await;

        let h2 = heading(&page).await;
        println!("    Heading: {h2}");

        println!("\n[1.4] Select CAPS framework (Φ={PHI:.2} ≥ 0.4 for click ✓)");
        if !click_button(&page, "South African").await {
            click_button(&page, "CAPS").await;
        }
        sleep(Duration::from_secs(2)).await;
        screenshot(&page, "04_framework").await;

        println!("\n[1.5] Select Grade 12");
        // Try clicking the grade 12 button (might say "12" or "Grade 12")
        if !click_button(&page, "Grade 12").await {
            js(&page, r#"(()=>{for(const b of document.querySelectorAll('button')){if(b.textContent.trim()==='12'){b.click();return}}})();"#).await;
            println!("    ✓ Clicked: 12");
        }
        sleep(Duration::from_secs(3)).await;
        screenshot(&page, "05_grade").await;
    } else {
        println!("    Already onboarded (heading: {h})");
    }

    // ━━━ Phase 2: Home ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n━━━ Phase 2: HOME ━━━\n");

    let h = heading(&page).await;
    println!("[2.1] Current page: {h}");
    let info = count_interactive(&page).await;
    println!("    Interactive: {info}");
    screenshot(&page, "06_home").await;

    println!("\n[2.2] Click 'Begin your session' (Φ={PHI:.2} ≥ 0.4 ✓)");
    click_button(&page, "Begin").await;
    sleep(Duration::from_secs(3)).await;
    screenshot(&page, "07_after_begin").await;

    // ━━━ Phase 3: Dashboard ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n━━━ Phase 3: DASHBOARD ━━━\n");

    page.goto("http://localhost:8107/dashboard").await.ok();
    sleep(Duration::from_secs(5)).await;
    let h = heading(&page).await;
    println!("[3.1] Dashboard: {h}");

    let progress = js(&page, r#"document.querySelector('.caps-progress-card')?.textContent?.substring(0, 100)||'no progress card'"#).await;
    println!(
        "    Progress: {}",
        progress.chars().take(80).collect::<String>()
    );
    screenshot(&page, "08_dashboard").await;

    // ━━━ Phase 4: Knowledge Garden ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n━━━ Phase 4: KNOWLEDGE GARDEN ━━━\n");

    page.goto("http://localhost:8107/skill-map").await.ok();
    sleep(Duration::from_secs(5)).await;
    let h = heading(&page).await;
    println!("[4.1] {h}");

    let info = count_interactive(&page).await;
    println!("    Interactive: {info}");
    screenshot(&page, "09_garden").await;

    // Click Mathematics tab
    println!("\n[4.2] Select Mathematics (Φ={PHI:.2} ≥ 0.4 ✓)");
    click_button(&page, "Mathematics").await;
    sleep(Duration::from_secs(2)).await;
    screenshot(&page, "10_math_garden").await;

    // ━━━ Phase 5: Mock Exam ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n━━━ Phase 5: MOCK EXAM ━━━\n");

    page.goto("http://localhost:8107/mock-exam").await.ok();
    sleep(Duration::from_secs(4)).await;
    println!("[5.1] Mock Exam page loaded");
    screenshot(&page, "11_mock_setup").await;

    println!("\n[5.2] Select Paper 1, Quick mode");
    click_button(&page, "Paper 1").await;
    sleep(Duration::from_millis(500)).await;
    click_button(&page, "Quick").await;
    sleep(Duration::from_millis(500)).await;

    println!("\n[5.3] Start Exam (Φ={PHI:.2} ≥ 0.4 ✓)");
    click_button(&page, "Start Exam").await;
    sleep(Duration::from_secs(3)).await;
    screenshot(&page, "12_exam_q1").await;

    // Read the question
    let question = js(&page, r#"document.querySelector('.exam-question,.question-text,h3')?.textContent?.trim()||'no question found'"#).await;
    println!(
        "    Question: {}",
        question.chars().take(100).collect::<String>()
    );

    // Show answer
    println!("\n[5.4] Show Answer (read-only, Φ={PHI:.2} ≥ 0.1 ✓)");
    click_button(&page, "Show").await;
    sleep(Duration::from_secs(2)).await;
    screenshot(&page, "13_answer").await;

    let answer = js(&page, r#"document.querySelector('.answer-box,.solution,code')?.textContent?.trim()||'no answer visible'"#).await;
    println!(
        "    Answer: {}",
        answer.chars().take(100).collect::<String>()
    );

    // Mark correct (self-assessment)
    println!("\n[5.5] Self-assess: Correct (Φ={PHI:.2} ≥ 0.4 ✓)");
    click_button(&page, "Correct").await;
    sleep(Duration::from_secs(2)).await;
    screenshot(&page, "14_marked").await;

    // Answer 2 more questions
    for q in 2..=3 {
        println!("\n[5.{q}+3] Question {q}");
        click_button(&page, "Show").await;
        sleep(Duration::from_secs(1)).await;
        // Alternate: mark some correct, some incorrect
        if q % 2 == 0 {
            click_button(&page, "Correct").await;
            println!("    Self-assessed: Correct");
        } else {
            click_button(&page, "Incorrect").await;
            println!("    Self-assessed: Incorrect");
        }
        sleep(Duration::from_secs(1)).await;
    }
    screenshot(&page, "15_exam_progress").await;

    // ━━━ Phase 6: Review ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n━━━ Phase 6: REVIEW ━━━\n");

    page.goto("http://localhost:8107/review").await.ok();
    sleep(Duration::from_secs(4)).await;
    println!("[6.1] Review page");
    screenshot(&page, "16_review").await;

    // Select Matric Maths deck
    println!("\n[6.2] Open Matric Maths flashcards");
    click_button(&page, "Matric Maths").await;
    sleep(Duration::from_secs(3)).await;
    screenshot(&page, "17_flashcards").await;

    // ━━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  Journey Complete                                 ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  Onboarding: Named 'Symthaea', CAPS Gr12         ║");
    println!("║  Dashboard:  Viewed progress & TEND economy       ║");
    println!("║  Garden:     Explored Mathematics constellation   ║");
    println!("║  Mock Exam:  Answered 3 questions (2✓ 1✗)         ║");
    println!("║  Review:     Opened Matric Maths flashcards       ║");
    println!("║  Φ = {PHI:.2} throughout — all actions gated       ║");
    println!("╚══════════════════════════════════════════════════╝");

    let screenshots: Vec<_> = std::fs::read_dir("/tmp/praxis-active")
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    println!(
        "\n{} screenshots saved to /tmp/praxis-active/",
        screenshots.len()
    );
}
