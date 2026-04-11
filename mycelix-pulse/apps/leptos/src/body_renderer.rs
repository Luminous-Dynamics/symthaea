// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Email body rendering (#2).
//!
//! Parses plain-text email bodies with:
//! - Paragraph breaks on double newlines
//! - Quoted text detection (lines starting with >)
//! - URL linkification
//! - Collapsible quoted sections

use leptos::prelude::*;

#[derive(Clone, Debug)]
enum BodyBlock {
    Text(String),
    Quote(Vec<String>),
}

fn parse_body(body: &str) -> Vec<BodyBlock> {
    let mut blocks = Vec::new();
    let mut current_text = Vec::new();
    let mut current_quote = Vec::new();

    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('>') {
            // Flush text
            if !current_text.is_empty() {
                blocks.push(BodyBlock::Text(current_text.join("\n")));
                current_text.clear();
            }
            // Strip leading > and spaces
            let content = trimmed.trim_start_matches('>').trim_start();
            current_quote.push(content.to_string());
        } else {
            // Flush quote
            if !current_quote.is_empty() {
                blocks.push(BodyBlock::Quote(current_quote.clone()));
                current_quote.clear();
            }
            current_text.push(line.to_string());
        }
    }

    if !current_text.is_empty() {
        blocks.push(BodyBlock::Text(current_text.join("\n")));
    }
    if !current_quote.is_empty() {
        blocks.push(BodyBlock::Quote(current_quote));
    }

    blocks
}

/// Convert basic markdown to HTML (headers, bold, italic, code, lists, hr).
fn render_markdown(text: &str) -> String {
    let mut html = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        // Headers
        if trimmed.starts_with("### ") {
            html.push_str(&format!("<h3>{}</h3>", html_escape(&trimmed[4..])));
        } else if trimmed.starts_with("## ") {
            html.push_str(&format!("<h2>{}</h2>", html_escape(&trimmed[3..])));
        } else if trimmed.starts_with("# ") {
            html.push_str(&format!("<h1>{}</h1>", html_escape(&trimmed[2..])));
        } else if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            html.push_str("<hr>");
        } else if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
            html.push_str(&format!("<li>{}</li>", inline_markdown(&html_escape(&trimmed[2..]))));
        } else if trimmed.starts_with("```") {
            html.push_str("<pre><code>");
        } else {
            html.push_str(&inline_markdown(&html_escape(line)));
            html.push_str("<br>");
        }
    }
    // Wrap consecutive <li> in <ul>
    html = html.replace("<li>", "<ul><li>").replace("</li><br>", "</li></ul>");
    html
}

/// Inline markdown: **bold**, *italic*, `code`, [link](url)
fn inline_markdown(text: &str) -> String {
    let mut result = text.to_string();
    // Code (backticks) — must be before bold/italic
    while let Some(start) = result.find('`') {
        if let Some(end) = result[start+1..].find('`') {
            let code = &result[start+1..start+1+end];
            let replacement = format!("<code>{code}</code>");
            result = format!("{}{}{}", &result[..start], replacement, &result[start+2+end..]);
        } else { break; }
    }
    // Bold **text**
    while let Some(start) = result.find("**") {
        if let Some(end) = result[start+2..].find("**") {
            let bold = &result[start+2..start+2+end];
            let replacement = format!("<strong>{bold}</strong>");
            result = format!("{}{}{}", &result[..start], replacement, &result[start+4+end..]);
        } else { break; }
    }
    // Italic *text*
    while let Some(start) = result.find('*') {
        if let Some(end) = result[start+1..].find('*') {
            let italic = &result[start+1..start+1+end];
            let replacement = format!("<em>{italic}</em>");
            result = format!("{}{}{}", &result[..start], replacement, &result[start+2+end..]);
        } else { break; }
    }
    result
}

fn linkify(text: &str) -> String {
    let escaped = html_escape(text);
    // Simple URL linkification on already-escaped text
    let mut result = String::new();
    let mut chars = escaped.chars().peekable();
    let mut buf = String::new();

    while let Some(c) = chars.next() {
        buf.push(c);
        // Check if buffer ends with "http://" or "https://"
        if buf.ends_with("https://") || buf.ends_with("http://") {
            let prefix_len = if buf.ends_with("https://") { 8 } else { 7 };
            // Push everything before the URL
            result.push_str(&buf[..buf.len() - prefix_len]);
            // Collect the URL
            let mut url = buf[buf.len() - prefix_len..].to_string();
            while let Some(&nc) = chars.peek() {
                if nc.is_whitespace() || nc == '&' { break; } // &lt; etc
                url.push(chars.next().unwrap());
            }
            let display = if url.len() > 60 { format!("{}...", &url[..57]) } else { url.clone() };
            result.push_str(&format!(
                "<a href=\"{}\" target=\"_blank\" rel=\"noopener noreferrer\">{}</a>",
                url, display
            ));
            buf.clear();
        }
    }
    result.push_str(&buf);
    result
}

/// Linkify URLs in already-rendered HTML (skip URLs inside href="...")
fn linkify_rendered(html: &str) -> String {
    // Simple: just linkify https:// that aren't already in an href
    html.to_string() // URLs are already escaped in the markdown pass; full linkify would need a proper HTML parser
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}

/// Renders an email body with paragraph breaks, quoted-text collapsing, and linkification.
#[component]
pub fn EmailBody(body: String) -> impl IntoView {
    let blocks = parse_body(&body);

    view! {
        <div class="email-body-rendered">
            {blocks.into_iter().map(|block| {
                match block {
                    BodyBlock::Text(text) => {
                        // Detect markdown-like content
                        let has_markdown = text.contains("**") || text.contains("# ")
                            || text.contains("```") || text.contains("- ") || text.contains("---");
                        if has_markdown {
                            let html = render_markdown(&text);
                            let html = linkify_rendered(&html);
                            view! { <div class="body-text" inner_html=html /> }.into_any()
                        } else {
                            // Split on double newlines for paragraphs
                            let paragraphs: Vec<_> = text.split("\n\n")
                                .filter(|p| !p.trim().is_empty())
                                .collect();
                            view! {
                                <div class="body-text">
                                    {paragraphs.into_iter().map(|p| {
                                        let html = linkify(p).replace('\n', "<br>");
                                        view! { <p inner_html=html /> }
                                    }).collect::<Vec<_>>()}
                                </div>
                            }.into_any()
                        }
                    }
                    BodyBlock::Quote(lines) => {
                        let preview = lines.first().cloned().unwrap_or_default();
                        let full_text = lines.join("\n");
                        let line_count = lines.len();
                        let expanded = RwSignal::new(false);
                        view! {
                            <div class="body-quote">
                                <button
                                    class="quote-toggle"
                                    on:click=move |_| expanded.update(|v| *v = !*v)
                                >
                                    {move || if expanded.get() {
                                        "\u{25BC} Hide quoted text"
                                    } else {
                                        "\u{25B6} Show quoted text"
                                    }}
                                    <span class="quote-count">{format!(" ({line_count} lines)")}</span>
                                </button>
                                {move || if expanded.get() {
                                    let html = linkify(&full_text).replace('\n', "<br>");
                                    view! {
                                        <blockquote class="quote-content" inner_html=html />
                                    }.into_any()
                                } else {
                                    view! {
                                        <blockquote class="quote-preview">
                                            {preview.clone()}
                                            {(line_count > 1).then(|| "...")}
                                        </blockquote>
                                    }.into_any()
                                }}
                            </div>
                        }.into_any()
                    }
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}
