// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Chrome DevTools Protocol session wrapper.
//!
//! Observation methods are public. Mutating primitives are crate-private and
//! are dispatched through [`crate::executor::BrowserExecutor`], which records a
//! policy decision and action receipt for every attempt.

use std::time::Duration;

use anyhow::{Context, Result, bail};
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
    max_elements: usize,
    navigation_timeout: Duration,
}

impl CdpSession {
    /// Launch a new Chrome instance using the provided configuration.
    ///
    /// To attach to an already-running browser, use
    /// [`Self::connect_existing_with_config`].
    pub async fn connect(config: &BrowserAgentConfig) -> Result<Self> {
        if config.cdp_url.is_some() {
            return Self::connect_existing_with_config(config).await;
        }

        let mut builder = BrowserConfig::builder();
        if config.headless {
            builder = builder.arg("--headless=new");
        }
        builder = builder.viewport(Viewport {
            width: config.viewport_width,
            height: config.viewport_height,
            ..Default::default()
        });

        let (browser, mut handler) = Browser::launch(
            builder
                .build()
                .map_err(|error| anyhow::anyhow!("{error}"))?,
        )
        .await
        .context("Failed to launch browser")?;

        tokio::spawn(async move {
            while let Some(_event) = handler.next().await {
                // Events must be consumed to keep the CDP connection alive.
            }
        });

        let page = browser
            .new_page("about:blank")
            .await
            .context("Failed to create new page")?;

        Ok(Self {
            _browser: browser,
            page: Mutex::new(page),
            max_elements: config.max_elements,
            navigation_timeout: Duration::from_millis(config.navigation_timeout_ms),
        })
    }

    /// Connect to an already-running Chrome instance by its debug URL.
    pub async fn connect_existing(debug_url: &str) -> Result<Self> {
        let config = BrowserAgentConfig {
            cdp_url: Some(debug_url.to_string()),
            ..BrowserAgentConfig::default()
        };
        Self::connect_existing_with_config(&config).await
    }

    /// Connect to an already-running Chrome instance while applying observation
    /// and timeout limits from the full browser configuration.
    pub async fn connect_existing_with_config(config: &BrowserAgentConfig) -> Result<Self> {
        let debug_url = config
            .cdp_url
            .as_deref()
            .context("cdp_url is required when connecting to an existing browser")?;
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
            max_elements: config.max_elements,
            navigation_timeout: Duration::from_millis(config.navigation_timeout_ms),
        })
    }

    /// Navigate to a URL and wait for the page to load.
    pub(crate) async fn navigate(&self, url: &str) -> Result<()> {
        let page = self.page.lock().await;
        tokio::time::timeout(self.navigation_timeout, page.goto(url))
            .await
            .context("Navigation timed out")?
            .context("Navigation failed")?;
        debug!(url, "Navigated");
        Ok(())
    }

    /// Query the full accessibility tree and return a bounded observation.
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

        let params = GetFullAxTreeParams::builder().build();
        let response = page
            .execute(params)
            .await
            .context("Failed to get accessibility tree")?;

        let mut elements = Vec::new();
        for node in &response.result.nodes {
            let role = node
                .role
                .as_ref()
                .and_then(|value| ax_value_to_string(&value.value))
                .unwrap_or_default();
            if role.is_empty() || role == "none" || role == "generic" {
                continue;
            }

            let name = node
                .name
                .as_ref()
                .and_then(|value| ax_value_to_string(&value.value))
                .unwrap_or_default();
            let value = node
                .value
                .as_ref()
                .and_then(|value| ax_value_to_string(&value.value));
            let description = node
                .description
                .as_ref()
                .and_then(|value| ax_value_to_string(&value.value));
            let backend_node_id = node
                .backend_dom_node_id
                .as_ref()
                .map(|id| *id.inner())
                .unwrap_or(0);

            elements.push(AccessibleElement {
                backend_node_id,
                role,
                name,
                value,
                description,
                focused: false,
                // AX `ignored` means omitted from the accessibility projection;
                // it is not the HTML/ARIA disabled state.
                disabled: false,
            });
        }

        let focused_element = detect_focused_element(&page, &elements).await;
        let mut observation = PageObservation {
            url,
            title,
            elements,
            focused_element,
        };
        observation.sanitize(self.max_elements);
        Ok(observation)
    }

    /// Click an element identified by a supported selector.
    pub(crate) async fn click(&self, selector: &ElementSelector) -> Result<()> {
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
                bail!(
                    "backend node targeting is not implemented safely for node {id}; use a CSS or accessible selector"
                );
            }
            ElementSelector::Accessible { role, name } => {
                let role = serde_json::to_string(role).context("Serialize accessible role")?;
                let name = serde_json::to_string(name).context("Serialize accessible name")?;
                let result = page
                    .evaluate(format!(
                        r#"(() => {{
                            const wantedRole = {role};
                            const wantedName = {name};
                            const inferredRole = (el) => {{
                                const explicit = el.getAttribute('role');
                                if (explicit) return explicit;
                                const tag = el.tagName.toLowerCase();
                                if (tag === 'button') return 'button';
                                if (tag === 'a' && el.hasAttribute('href')) return 'link';
                                if (tag === 'textarea') return 'textbox';
                                if (tag === 'select') return 'combobox';
                                if (tag === 'input') {{
                                    const type = (el.getAttribute('type') || 'text').toLowerCase();
                                    if (type === 'checkbox') return 'checkbox';
                                    if (type === 'radio') return 'radio';
                                    return 'textbox';
                                }}
                                return '';
                            }};
                            const accessibleName = (el) =>
                                (el.getAttribute('aria-label') ||
                                 el.getAttribute('title') ||
                                 el.getAttribute('placeholder') ||
                                 el.textContent || '').trim();
                            const matches = Array.from(document.querySelectorAll('*')).filter(
                                el => inferredRole(el) === wantedRole && accessibleName(el) === wantedName
                            );
                            if (matches.length === 0) return 'not_found';
                            if (matches.length > 1) return 'ambiguous';
                            matches[0].click();
                            return 'clicked';
                        }})()"#
                    ))
                    .await
                    .context("Click via accessible selector failed")?
                    .into_value::<String>()
                    .unwrap_or_default();
                match result.as_str() {
                    "clicked" => {}
                    "ambiguous" => bail!("accessible selector matched multiple elements"),
                    _ => bail!("accessible selector did not match an element"),
                }
            }
        }
        debug!(?selector, "Clicked");
        Ok(())
    }

    /// Type text into an element after focusing it.
    pub(crate) async fn type_text(&self, selector: &ElementSelector, text: &str) -> Result<()> {
        match selector {
            ElementSelector::Css(css) => {
                let page = self.page.lock().await;
                let element = page.find_element(css).await.context("Element not found")?;
                element.click().await.context("Focus failed")?;
                set_active_value(&page, text).await.context("Type failed")?;
            }
            _ => {
                self.click(selector).await?;
                let page = self.page.lock().await;
                set_active_value(&page, text)
                    .await
                    .context("Type after focus failed")?;
            }
        }
        debug!(text_len = text.len(), "Typed text");
        Ok(())
    }

    /// Scroll a CSS-selected element into the viewport.
    pub(crate) async fn scroll_to(&self, selector: &ElementSelector) -> Result<()> {
        let ElementSelector::Css(css) = selector else {
            bail!("scroll currently requires a CSS selector");
        };
        let page = self.page.lock().await;
        let css = serde_json::to_string(css).context("Serialize CSS selector")?;
        let result = page
            .evaluate(format!(
                "(() => {{ const el = document.querySelector({css}); if (!el) return 'not_found'; el.scrollIntoView({{block:'center', inline:'nearest'}}); return 'scrolled'; }})()"
            ))
            .await
            .context("Scroll failed")?
            .into_value::<String>()
            .unwrap_or_default();
        if result != "scrolled" {
            bail!("scroll target not found");
        }
        Ok(())
    }

    pub(crate) async fn go_back(&self) -> Result<()> {
        let page = self.page.lock().await;
        page.evaluate("history.back(); true")
            .await
            .context("History back failed")?;
        Ok(())
    }

    pub(crate) async fn go_forward(&self) -> Result<()> {
        let page = self.page.lock().await;
        page.evaluate("history.forward(); true")
            .await
            .context("History forward failed")?;
        Ok(())
    }

    pub(crate) async fn extract_text(&self, selector: Option<&str>) -> Result<String> {
        let page = self.page.lock().await;
        let selector = serde_json::to_string(&selector).context("Serialize text selector")?;
        let raw = page
            .evaluate(format!(
                r#"(() => {{
                    const selector = {selector};
                    const element = selector === null ? document.body : document.querySelector(selector);
                    if (!element) return JSON.stringify({{status:'not_found', text:''}});
                    return JSON.stringify({{status:'ok', text:(element.innerText || element.textContent || '')}});
                }})()"#
            ))
            .await
            .context("Text extraction failed")?
            .into_value::<String>()
            .unwrap_or_default();
        let parsed: serde_json::Value =
            serde_json::from_str(&raw).context("Invalid extraction result")?;
        if parsed.get("status").and_then(|value| value.as_str()) != Some("ok") {
            bail!("text extraction target not found");
        }
        Ok(parsed
            .get("text")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string())
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

    /// Get page state via JS when the AX tree is unavailable.
    pub async fn get_page_state_js(&self) -> Result<PageObservation> {
        let page = self.page.lock().await;
        let url = page.url().await.unwrap_or(None).unwrap_or_default();
        let title = page
            .get_title()
            .await
            .unwrap_or(None)
            .unwrap_or_else(|| "Untitled".into());

        let json = page
            .evaluate(
                r#"JSON.stringify({
                    headings: Array.from(document.querySelectorAll('h1,h2')).slice(0,10).map(e=>e.textContent?.trim()||''),
                    buttons: Array.from(document.querySelectorAll('button')).slice(0,20).map(e=>e.textContent?.trim()||''),
                    inputs: Array.from(document.querySelectorAll('input')).slice(0,10).map(e=>e.placeholder||''),
                    links: Array.from(document.querySelectorAll('a')).slice(0,10).map(e=>e.textContent?.trim()||''),
                    texts: Array.from(document.querySelectorAll('p')).slice(0,5).map(e=>e.textContent?.trim().substring(0,80)||'')
                })"#,
            )
            .await
            .context("JS eval failed")?;

        let data: serde_json::Value =
            serde_json::from_str(&json.into_value::<String>().unwrap_or_default())
                .unwrap_or_default();

        let mut elements = Vec::new();
        let extract = |key: &str, role: &str| -> Vec<AccessibleElement> {
            data.get(key)
                .and_then(|value| value.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|value| {
                            let name = value.as_str().unwrap_or("").to_string();
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

        let mut observation = PageObservation {
            url,
            title,
            elements,
            focused_element: None,
        };
        observation.sanitize(self.max_elements);
        Ok(observation)
    }

    /// Try AX tree first, then fall back to bounded JS extraction.
    pub async fn observe(&self) -> Result<PageObservation> {
        match self.get_accessibility_tree().await {
            Ok(observation) => Ok(observation),
            Err(_) => self.get_page_state_js().await,
        }
    }

    /// Low-level arbitrary JavaScript evaluation for internal diagnostics only.
    pub(crate) async fn eval_js(&self, expression: &str) -> Result<String> {
        let page = self.page.lock().await;
        let result = page
            .evaluate(expression)
            .await
            .context("JS evaluation failed")?;
        Ok(result.into_value::<String>().unwrap_or_default())
    }
}

async fn set_active_value(page: &Page, text: &str) -> Result<()> {
    let text = serde_json::to_string(text).context("Serialize input text")?;
    let result = page
        .evaluate(format!(
            "(() => {{ const el = document.activeElement; const value = {text}; if (!el || !('value' in el)) return 'not_editable'; el.value = value; el.dispatchEvent(new Event('input', {{bubbles:true}})); el.dispatchEvent(new Event('change', {{bubbles:true}})); return 'typed'; }})()"
        ))
        .await
        .context("Set active value failed")?
        .into_value::<String>()
        .unwrap_or_default();
    if result != "typed" {
        bail!("focused element is not editable");
    }
    Ok(())
}

async fn detect_focused_element(page: &Page, elements: &[AccessibleElement]) -> Option<usize> {
    let result = page
        .evaluate(
            r#"(() => {
                const el = document.activeElement;
                if (!el || el === document.body) return '';
                const role = el.getAttribute('role') ||
                    (el.tagName.toLowerCase() === 'button' ? 'button' :
                    (el.tagName.toLowerCase() === 'a' ? 'link' :
                    (['input','textarea'].includes(el.tagName.toLowerCase()) ? 'textbox' : '')));
                const name = (el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.textContent || '').trim();
                return JSON.stringify({role, name});
            })()"#,
        )
        .await
        .ok()?
        .into_value::<String>()
        .ok()?;
    let focused: serde_json::Value = serde_json::from_str(&result).ok()?;
    let role = focused.get("role")?.as_str()?;
    let name = focused.get("name")?.as_str()?;
    elements
        .iter()
        .position(|element| element.role == role && element.name == name)
}

/// Extract a string from a `serde_json::Value` (the AxValue payload).
fn ax_value_to_string(value: &Option<serde_json::Value>) -> Option<String> {
    match value {
        Some(serde_json::Value::String(string)) => Some(string.clone()),
        Some(value) => Some(value.to_string()),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
