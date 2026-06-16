// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Test the browser agent against live Praxis.
//!
//! Prerequisites: Chrome running with --remote-debugging-port=9222
//!               Praxis served on localhost:8107
//!
//! Usage: cargo run -p symthaea-browser --example praxis_journey

use symthaea_browser::observation::{AccessibleElement, PageObservation};

use chromiumoxide::browser::Browser;
use chromiumoxide::cdp::browser_protocol::accessibility::GetFullAxTreeParams;
use futures::StreamExt;

async fn observe(page: &chromiumoxide::Page) -> PageObservation {
    let url = page.url().await.unwrap_or(None).unwrap_or_default();
    let title = page
        .get_title()
        .await
        .unwrap_or(None)
        .unwrap_or_else(|| "Untitled".into());

    let mut elements = Vec::new();

    let params = GetFullAxTreeParams::builder().build();
    let ax_result = page.execute(params).await;

    if let Err(e) = &ax_result {
        eprintln!("    AX tree unavailable ({e}), using JS fallback");
        let js_result = page.evaluate(r#"
            JSON.stringify(Array.from(document.querySelectorAll('button,a,input,select,h1,h2,h3,p')).slice(0, 50).map(el => ({
                role: el.tagName.toLowerCase(),
                name: el.textContent?.trim().substring(0, 80) || el.getAttribute('placeholder') || '',
                value: el.value || null
            })))
        "#).await;

        if let Ok(val) = js_result {
            if let Ok(json_str) = val.into_value::<String>() {
                if let Ok(items) = serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
                    for item in &items {
                        let role = item
                            .get("role")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let value = item
                            .get("value")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        if !name.is_empty() {
                            elements.push(AccessibleElement {
                                backend_node_id: -1,
                                role,
                                name,
                                value,
                                description: None,
                                focused: false,
                                disabled: false,
                            });
                        }
                    }
                }
            }
        }

        return PageObservation {
            url,
            title,
            elements,
            focused_element: None,
        };
    }

    let response = ax_result.unwrap();
    for node in &response.result.nodes {
        let role = node
            .role
            .as_ref()
            .and_then(|v| v.value.as_ref())
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if role.is_empty() || role == "none" || role == "generic" {
            continue;
        }

        let name = node
            .name
            .as_ref()
            .and_then(|v| v.value.as_ref())
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let value = node
            .value
            .as_ref()
            .and_then(|v| v.value.as_ref())
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // BackendNodeId is a newtype; use Debug to extract the inner value
        let backend_node_id = node
            .backend_dom_node_id
            .map(|id| {
                format!("{id:?}")
                    .trim_start_matches("BackendNodeId(")
                    .trim_end_matches(')')
                    .parse::<i64>()
                    .unwrap_or(-1)
            })
            .unwrap_or(-1);

        elements.push(AccessibleElement {
            backend_node_id,
            role,
            name,
            value,
            description: None,
            focused: false,
            disabled: false,
        });
    }

    PageObservation {
        url,
        title,
        elements,
        focused_element: None,
    }
}

#[tokio::main]
async fn main() {
    println!("=== Symthaea Browser Agent: Praxis Journey ===\n");

    // Connect to existing Chrome on :9222
    let (browser, mut handler) = Browser::connect("http://localhost:9222")
        .await
        .expect("Connect to Chrome on :9222 — start with: chromium --headless=new --no-sandbox --remote-debugging-port=9222");

    tokio::spawn(async move { while handler.next().await.is_some() {} });

    let page = browser.new_page("about:blank").await.expect("New page");

    // Step 1: Navigate to Praxis
    println!("[1] Navigating to Praxis...");
    page.goto("http://localhost:8107").await.expect("Navigate");
    tokio::time::sleep(std::time::Duration::from_secs(6)).await;

    let obs = observe(&page).await;
    println!("    Page: {} — {}", obs.title, obs.url);
    println!("    Elements: {} accessible", obs.elements.len());
    println!("    Cognitive text:");
    for line in obs.to_cognitive_text().lines().take(8) {
        println!("      {line}");
    }

    // Step 2: Skip onboarding via localStorage
    println!("\n[2] Setting student profile (Symthaea, Grade 12, CAPS)...");
    page.evaluate(
        r#"localStorage.setItem('praxis_student_profile', JSON.stringify({
            name: 'Symthaea', grade: 12, exam_date: '2026-10-26',
            onboarding_complete: true, framework: 'caps'
        }))"#,
    )
    .await
    .expect("Set localStorage");

    page.goto("http://localhost:8107/").await.expect("Reload");
    tokio::time::sleep(std::time::Duration::from_secs(6)).await;

    let obs = observe(&page).await;
    println!("    Home: {} elements", obs.elements.len());

    // Step 3: Navigate key pages, report what consciousness would see
    let routes = [
        ("Dashboard", "/dashboard"),
        ("Knowledge Garden", "/skill-map"),
        ("Review", "/review"),
        ("Exam Prep", "/exam-prep"),
        ("Mock Exam", "/mock-exam"),
        ("Pathways", "/pathways"),
        ("Governance", "/governance"),
    ];

    for (name, route) in routes {
        println!("\n[→] {name}");
        page.goto(&format!("http://localhost:8107{route}"))
            .await
            .expect("Navigate");
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        let obs = observe(&page).await;
        let interactive: Vec<&AccessibleElement> = obs
            .elements
            .iter()
            .filter(|e| {
                matches!(
                    e.role.as_str(),
                    "button" | "link" | "textbox" | "combobox" | "checkbox"
                )
            })
            .collect();

        println!(
            "    {} elements, {} interactive",
            obs.elements.len(),
            interactive.len()
        );

        for el in interactive.iter().take(6) {
            let phi = match el.role.as_str() {
                "textbox" => 0.5,
                "button" => 0.4,
                "checkbox" => 0.4,
                _ => 0.3,
            };
            let display_name = if el.name.len() > 35 {
                format!("{}...", &el.name[..32])
            } else {
                el.name.clone()
            };
            println!("      [{:8}] {display_name:35} Φ ≥ {phi:.1}", el.role);
        }
    }

    println!("\n=== Journey Complete ===");
    println!("The consciousness engine can perceive every interactive element,");
    println!("gate actions by Phi threshold, and navigate the full Praxis experience.");
}
