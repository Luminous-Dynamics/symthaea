// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Chrome DevTools Protocol session wrapper.
//!
//! Provides an async interface to a headless Chrome/Chromium instance via
//! `chromiumoxide`. All CDP calls are exposed as typed methods that return
//! domain objects (`PageObservation`, `AccessibleElement`).
//!
//! This module requires a running Chrome instance with `--remote-debugging-port`.
//! Use `BrowserAgentConfig` to specify the connection URL.

use anyhow::{Context, Result};
use chromiumoxide::Page;
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::accessibility::GetFullAxTreeParams;
use chromiumoxide::handler::viewport::Viewport;
use futures::StreamExt;
use tokio::sync::Mutex;
use tracing::debug;

use crate::actions::ElementSelector;
use crate::config::BrowserAgentConfig;
use crate::observation::{AccessibleElement, PageObservation};

/// CDP session wrapping a `chromiumoxide::Browser` and its active page.
pub struct CdpSession {
    _browser: Browser,
    page: Mutex<Page>,
}

impl CdpSession {
    /// Connect to a Chrome instance using the provided config.
    ///
    /// Launches a new browser process. If `config.cdp_url` is set,
    /// a remote-debugging-port argument is added for the specified port.
    pub async fn connect(config: &BrowserAgentConfig) -> Result<Self> {
        let mut builder = BrowserConfig::builder();
        if config.headless {
            builder = builder.arg("--headless=new");
        }
        if let Some(ref url) = config.cdp_url {
            builder = builder.arg(format!("--remote-debugging-port={}", extract_port(url)));
        }
        builder = builder.viewport(Viewport {
            width: config.viewport_width,
            height: config.viewport_height,
            ..Default::default()
        });

        let (browser, mut handler) =
            Browser::launch(builder.build().map_err(|e| anyhow::anyhow!("{}", e))?)
                .await
                .context("Failed to launch browser")?;

        // Spawn the CDP event handler (Stream-based)
        tokio::spawn(async move {
            while let Some(_event) = handler.next().await {
                // Events consumed to keep the CDP connection alive
            }
        });

        let page = browser
            .new_page("about:blank")
            .await
            .context("Failed to create new page")?;

        Ok(Self {
            _browser: browser,
            page: Mutex::new(page),
        })
    }

    /// Navigate to a URL and wait for the page to load.
    pub async fn navigate(&self, url: &str) -> Result<()> {
        let page = self.page.lock().await;
        page.goto(url).await.context("Navigation failed")?;
        debug!(url, "Navigated");
        Ok(())
    }

    /// Query the full accessibility tree and return a `PageObservation`.
    pub async fn get_accessibility_tree(&self) -> Result<PageObservation> {
        let page = self.page.lock().await;

        let url = page
            .url()
            .await
            .unwrap_or(None)
            .unwrap_or_else(|| "unknown".to_string());

        let title = page
            .get_title()
            .await
            .unwrap_or(None)
            .unwrap_or_else(|| "Untitled".to_string());

        // Fetch the full accessibility tree via CDP
        let params = GetFullAxTreeParams::builder().build();
        let response = page
            .execute(params)
            .await
            .context("Failed to get accessibility tree")?;

        let mut elements = Vec::new();
        let mut focused_idx = None;

        for node in &response.result.nodes {
            let role = node
                .role
                .as_ref()
                .and_then(|v| ax_value_to_string(&v.value))
                .unwrap_or_default();

            // Skip internal/structural roles
            if role.is_empty() || role == "none" || role == "generic" {
                continue;
            }

            let name = node
                .name
                .as_ref()
                .and_then(|v| ax_value_to_string(&v.value))
                .unwrap_or_default();

            let value = node
                .value
                .as_ref()
                .and_then(|v| ax_value_to_string(&v.value));

            let description = node
                .description
                .as_ref()
                .and_then(|v| ax_value_to_string(&v.value));

            let backend_node_id = node
                .backend_dom_node_id
                .as_ref()
                .map(|id| *id.inner())
                .unwrap_or(0);

            let focused = false; // Accessibility tree doesn't directly expose focus
            let disabled = node.ignored;

            elements.push(AccessibleElement {
                backend_node_id,
                role,
                name,
                value,
                description,
                focused,
                disabled,
            });
        }

        // Detect focused element via separate check
        if let Ok(focus_result) = page
            .evaluate("document.activeElement?.getAttribute('role') || ''")
            .await
        {
            if let Some(focus_role) = focus_result.value().and_then(|v| v.as_str()) {
                if !focus_role.is_empty() {
                    for (i, elem) in elements.iter().enumerate() {
                        if elem.role == focus_role {
                            focused_idx = Some(i);
                            break;
                        }
                    }
                }
            }
        }

        Ok(PageObservation {
            url,
            title,
            elements,
            focused_element: focused_idx,
        })
    }

    /// Click an element identified by selector.
    pub async fn click(&self, selector: &ElementSelector) -> Result<()> {
        let page = self.page.lock().await;
        match selector {
            ElementSelector::Css(css) => {
                page.find_element(css)
                    .await
                    .context("Element not found")?
                    .click()
                    .await
                    .context("Click failed")?;
            }
            ElementSelector::BackendNodeId(id) => {
                let js = format!(
                    "document.querySelector('[data-backend-node-id=\"{}\"]')?.click()",
                    id
                );
                page.evaluate(js)
                    .await
                    .context("Click via backend node ID failed")?;
            }
            ElementSelector::Accessible { role, name } => {
                let js = format!(
                    r#"
                    (function() {{
                        const els = document.querySelectorAll('[role="{}"]');
                        for (const el of els) {{
                            if (el.getAttribute('aria-label') === '{}' ||
                                el.textContent.trim() === '{}') {{
                                el.click();
                                return true;
                            }}
                        }}
                        return false;
                    }})()
                    "#,
                    role, name, name
                );
                page.evaluate(js)
                    .await
                    .context("Click via accessible selector failed")?;
            }
        }
        debug!(?selector, "Clicked");
        Ok(())
    }

    /// Type text into an element.
    pub async fn type_text(&self, selector: &ElementSelector, text: &str) -> Result<()> {
        let page = self.page.lock().await;
        match selector {
            ElementSelector::Css(css) => {
                // Focus the element first
                let elem = page.find_element(css).await.context("Element not found")?;
                elem.click().await.context("Focus failed")?;
                // Type via JS since chromiumoxide 0.7 lacks page-level type_str
                let escaped = text.replace('\\', "\\\\").replace('\'', "\\'");
                let js = format!(
                    "document.activeElement.value = '{}'; \
                     document.activeElement.dispatchEvent(new Event('input', {{bubbles: true}}))",
                    escaped
                );
                page.evaluate(js).await.context("Type failed")?;
            }
            _ => {
                // For non-CSS selectors, click first then type
                drop(page);
                self.click(selector).await?;
                let page = self.page.lock().await;
                let escaped = text.replace('\\', "\\\\").replace('\'', "\\'");
                let js = format!(
                    "document.activeElement.value = '{}'; \
                     document.activeElement.dispatchEvent(new Event('input', {{bubbles: true}}))",
                    escaped
                );
                page.evaluate(js).await.context("Type after focus failed")?;
            }
        }
        debug!(text_len = text.len(), "Typed text");
        Ok(())
    }

    /// Capture a PNG screenshot of the current page.
    pub async fn screenshot(&self) -> Result<Vec<u8>> {
        let page = self.page.lock().await;
        let bytes = page
            .screenshot(
                chromiumoxide::page::ScreenshotParams::builder()
                    .full_page(true)
                    .build(),
            )
            .await
            .context("Screenshot failed")?;
        debug!(bytes = bytes.len(), "Screenshot captured");
        Ok(bytes)
    }

    /// Connect to an already-running Chrome instance by its debug URL.
    pub async fn connect_existing(debug_url: &str) -> Result<Self> {
        let (browser, mut handler) = Browser::connect(debug_url)
            .await
            .context("Failed to connect to existing Chrome")?;

        tokio::spawn(async move { while let Some(_event) = handler.next().await {} });

        let page = browser
            .new_page("about:blank")
            .await
            .context("Failed to create new page")?;

        Ok(Self {
            _browser: browser,
            page: Mutex::new(page),
        })
    }

    /// Get page state via JS (fallback when AX tree fails on WASM-heavy pages).
    pub async fn get_page_state_js(&self) -> Result<PageObservation> {
        let page = self.page.lock().await;
        let url = page.url().await.unwrap_or(None).unwrap_or_default();
        let title = page
            .get_title()
            .await
            .unwrap_or(None)
            .unwrap_or_else(|| "Untitled".into());

        let json = page.evaluate(r#"JSON.stringify({
            headings: Array.from(document.querySelectorAll('h1,h2')).slice(0,10).map(e=>e.textContent?.trim()||''),
            buttons: Array.from(document.querySelectorAll('button')).slice(0,20).map(e=>e.textContent?.trim()||''),
            inputs: Array.from(document.querySelectorAll('input')).slice(0,10).map(e=>e.placeholder||''),
            links: Array.from(document.querySelectorAll('a')).slice(0,10).map(e=>e.textContent?.trim()||''),
            texts: Array.from(document.querySelectorAll('p')).slice(0,5).map(e=>e.textContent?.trim().substring(0,80)||'')
        })"#).await.context("JS eval failed")?;

        let data: serde_json::Value =
            serde_json::from_str(&json.into_value::<String>().unwrap_or_default())
                .unwrap_or_default();

        let mut elements = Vec::new();
        let extract = |key: &str, role: &str| -> Vec<AccessibleElement> {
            data.get(key)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| {
                            let name = v.as_str().unwrap_or("").to_string();
                            if name.is_empty() {
                                return None;
                            }
                            Some(AccessibleElement {
                                backend_node_id: -1,
                                role: role.to_string(),
                                name,
                                value: None,
                                description: None,
                                focused: false,
                                disabled: false,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        elements.extend(extract("headings", "heading"));
        elements.extend(extract("buttons", "button"));
        elements.extend(extract("inputs", "textbox"));
        elements.extend(extract("links", "link"));
        elements.extend(extract("texts", "text"));

        Ok(PageObservation {
            url,
            title,
            elements,
            focused_element: None,
        })
    }

    /// Try AX tree first, fall back to JS extraction.
    pub async fn observe(&self) -> Result<PageObservation> {
        match self.get_accessibility_tree().await {
            Ok(obs) => Ok(obs),
            Err(_) => self.get_page_state_js().await,
        }
    }

    /// Click a button/link by its text content.
    pub async fn click_by_text(&self, text: &str) -> Result<()> {
        let page = self.page.lock().await;
        let escaped = text.replace('\\', "\\\\").replace('\'', "\\'");
        page.evaluate(format!(
            r#"(()=>{{for(const b of document.querySelectorAll('button,a')){{if(b.textContent.includes('{escaped}')){{b.click();return}}}}}})();"#
        )).await.context("click_by_text failed")?;
        Ok(())
    }

    /// Type into an input by placeholder text match.
    pub async fn type_by_placeholder(&self, placeholder: &str, text: &str) -> Result<()> {
        let page = self.page.lock().await;
        let ph = placeholder.replace('\\', "\\\\").replace('\'', "\\'");
        let tx = text.replace('\\', "\\\\").replace('\'', "\\'");
        page.evaluate(format!(
            r#"(()=>{{for(const i of document.querySelectorAll('input')){{if((i.placeholder||'').toLowerCase().includes('{ph}')){{i.value='{tx}';i.dispatchEvent(new Event('input',{{bubbles:true}}));return}}}}}})();"#
        )).await.context("type_by_placeholder failed")?;
        Ok(())
    }

    /// Evaluate arbitrary JS and return string result.
    pub async fn eval_js(&self, expr: &str) -> Result<String> {
        let page = self.page.lock().await;
        let result = page.evaluate(expr).await.context("JS eval failed")?;
        Ok(result.into_value::<String>().unwrap_or_default())
    }
}

/// Extract a string from a `serde_json::Value` (the AxValue payload).
fn ax_value_to_string(value: &Option<serde_json::Value>) -> Option<String> {
    match value {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(v) => Some(v.to_string()),
        None => None,
    }
}

/// Extract port number from a WebSocket URL like "ws://127.0.0.1:9222/...".
fn extract_port(url: &str) -> u16 {
    url.split(':')
        .nth(2)
        .and_then(|s| s.split('/').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(9222)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_port() {
        assert_eq!(extract_port("ws://127.0.0.1:9222/devtools"), 9222);
        assert_eq!(extract_port("ws://localhost:9333"), 9333);
        assert_eq!(extract_port("invalid"), 9222); // default
    }

    #[test]
    fn test_ax_value_to_string() {
        assert_eq!(
            ax_value_to_string(&Some(serde_json::Value::String("button".into()))),
            Some("button".to_string())
        );
        assert_eq!(ax_value_to_string(&None), None);
        assert_eq!(
            ax_value_to_string(&Some(serde_json::Value::Number(42.into()))),
            Some("42".to_string())
        );
    }
}
