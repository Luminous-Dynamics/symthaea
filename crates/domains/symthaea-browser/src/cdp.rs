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
use chromiumoxide::cdp::browser_protocol::dom::{
    BackendNodeId, GetContentQuadsParams, ScrollIntoViewIfNeededParams,
};
use chromiumoxide::cdp::browser_protocol::input::InsertTextParams;
use chromiumoxide::cdp::browser_protocol::page::{
    GetNavigationHistoryParams, NavigateToHistoryEntryParams,
};
use chromiumoxide::handler::viewport::Viewport;
use chromiumoxide::layout::Point;
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
        let mut focused_element = None;
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
            let focused = node.properties.as_ref().is_some_and(|properties| {
                properties.iter().any(|property| {
                    let property_name = serde_json::to_value(&property.name).ok();
                    property_name.as_ref().and_then(serde_json::Value::as_str)
                        == Some("focused")
                        && property.value.value.as_ref()
                            .and_then(serde_json::Value::as_bool) == Some(true)
                })
            });
            if focused {
                focused_element.get_or_insert(elements.len());
            }

            elements.push(AccessibleElement {
                backend_node_id,
                role,
                name,
                value,
                description,
                focused,
                // AX `ignored` means omitted from the accessibility projection;
                // it is not the HTML/ARIA disabled state.
                disabled: false,
            });
        }

        let mut observation = PageObservation {
            url,
            title,
            elements,
            focused_element,
        };
        observation.sanitize(self.max_elements);
        Ok(observation)
    }

    /// Click a CSS or accessibility target using native DOM and Input CDP commands.
    ///
    /// This avoids Chromiumoxide's convenience click path, which runs a JavaScript
    /// scroll helper before dispatching input.
    pub(crate) async fn click(&self, selector: &ElementSelector) -> Result<()> {
        let page = self.page.lock().await;
        let backend_node_id = match selector {
            ElementSelector::Css(css) => page
                .find_element(css)
                .await
                .context("Element not found")?
                .backend_node_id
                .clone(),
            ElementSelector::BackendNodeId(id) => {
                bail!("raw integer backend node selector {id} is unsupported; use CSS or accessibility targeting");
            }
            ElementSelector::Accessible { role, name } => {
                resolve_accessible_backend_node(&page, role, name).await?
            }
        };
        click_backend_node(&page, &backend_node_id)
            .await
            .context("Native CDP click failed")?;
        debug!(?selector, "Clicked with native CDP input");
        Ok(())
    }

    /// Type text using native CDP input and verify the resulting AX value.
    ///
    /// A successful CDP command is not enough: the accessibility tree must
    /// confirm the inserted text, so rejected or reverted framework inputs fail closed.
    pub(crate) async fn type_text(&self, selector: &ElementSelector, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let page = self.page.lock().await;
        let backend_node_id = match selector {
            ElementSelector::Css(css) => page
                .find_element(css)
                .await
                .context("Element not found")?
                .backend_node_id
                .clone(),
            ElementSelector::BackendNodeId(id) => {
                bail!("raw integer backend node selector {id} is unsupported; use CSS or accessibility targeting");
            }
            ElementSelector::Accessible { role, name } => {
                resolve_accessible_backend_node(&page, role, name).await?
            }
        };

        validate_text_entry_target(&page, &backend_node_id)
            .await
            .context("Text target is not a supported editable control")?;
        click_backend_node(&page, &backend_node_id)
            .await
            .context("Focus text target with native CDP input")?;
        page.execute(InsertTextParams::new(text.to_owned()))
            .await
            .context("Input.insertText failed")?;
        if !accessibility_value_contains(&page, &backend_node_id, text).await? {
            bail!("Input.insertText was dispatched, but the AX tree did not confirm the requested text");
        }

        debug!(text_len = text.len(), "Typed text with native CDP input");
        Ok(())
    }

    /// Scroll a CSS-selected element into the viewport using the DOM CDP domain.
    pub(crate) async fn scroll_to(&self, selector: &ElementSelector) -> Result<()> {
        let ElementSelector::Css(css) = selector else {
            bail!("scroll currently requires a CSS selector");
        };
        let page = self.page.lock().await;
        let element = page
            .find_element(css)
            .await
            .context("Scroll target not found")?;
        scroll_backend_node_into_view(&page, &element.backend_node_id)
            .await
            .context("Native CDP scroll failed")?;
        Ok(())
    }

    pub(crate) async fn go_back(&self) -> Result<()> {
        let page = self.page.lock().await;
        navigate_history(&page, -1).await.context("History back failed")
    }

    pub(crate) async fn go_forward(&self) -> Result<()> {
        let page = self.page.lock().await;
        navigate_history(&page, 1).await.context("History forward failed")
    }

    /// Extract page text for the explicit ExtractText action.
    ///
    /// This is the remaining fixed-expression JavaScript helper. Migration to a
    /// DOMSnapshot/text-node implementation remains open; the CSS selector is
    /// JSON-serialized before it is included in the expression.
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

    /// Compatibility shim. The JavaScript observation fallback has been removed.
    ///
    /// Prefer the native accessibility-tree API. This shim is kept for source
    /// compatibility and no longer evaluates page JavaScript.
    #[deprecated(note = "JavaScript observation fallback removed; use get_accessibility_tree")]
    pub async fn get_page_state_js(&self) -> Result<PageObservation> {
        self.get_accessibility_tree().await
    }

    /// Observe through the native accessibility tree only.
    ///
    /// An unavailable tree is returned as an error; there is no silent JavaScript
    /// fallback that could hide a degraded observation path.
    pub async fn observe(&self) -> Result<PageObservation> {
        self.get_accessibility_tree()
            .await
            .context("Accessibility observation unavailable; JavaScript fallback is disabled")
    }

async fn navigate_history(page: &Page, offset: i64) -> Result<()> {
    let history = page
        .execute(GetNavigationHistoryParams::builder().build())
        .await
        .context("Read navigation history")?
        .result;
    let target_index = history
        .current_index
        .checked_add(offset)
        .context("Navigation history index overflow")?;
    if target_index < 0 {
        bail!("no earlier navigation history entry");
    }
    let target_index = usize::try_from(target_index).context("Invalid navigation history index")?;
    let entry = history
        .entries
        .get(target_index)
        .context("no navigation history entry in that direction")?;
    page.execute(
        NavigateToHistoryEntryParams::builder()
            .entry_id(entry.id)
            .build()
            .unwrap(),
    )
    .await
    .context("Navigate to history entry")?;
    Ok(())
}

async fn resolve_accessible_backend_node(
    page: &Page,
    wanted_role: &str,
    wanted_name: &str,
) -> Result<BackendNodeId> {
    let tree = page
        .execute(GetFullAxTreeParams::builder().build())
        .await
        .context("Read accessibility tree for target resolution")?;
    let matches: Vec<_> = tree
        .result
        .nodes
        .iter()
        .filter(|node| {
            let role = node.role.as_ref().and_then(|value| ax_value_to_string(&value.value));
            let name = node.name.as_ref().and_then(|value| ax_value_to_string(&value.value));
            role.as_deref() == Some(wanted_role) && name.as_deref() == Some(wanted_name)
        })
        .collect();

    if matches.is_empty() {
        bail!("accessible selector did not match an element");
    }
    if matches.len() > 1 {
        bail!("accessible selector matched multiple elements");
    }
    matches[0]
        .backend_dom_node_id
        .as_ref()
        .cloned()
        .context("accessible selector has no associated DOM backend node")
}

async fn validate_text_entry_target(page: &Page, backend_node_id: &BackendNodeId) -> Result<()> {
    let tree = page
        .execute(GetFullAxTreeParams::builder().build())
        .await
        .context("Read accessibility tree before text entry")?;
    let node = tree
        .result
        .nodes
        .iter()
        .find(|node| {
            node.backend_dom_node_id
                .as_ref()
                .is_some_and(|id| *id.inner() == *backend_node_id.inner())
        })
        .context("text target is not present in the accessibility tree")?;
    let role = node.role.as_ref().and_then(|value| ax_value_to_string(&value.value));
    if !matches!(role.as_deref(), Some("textbox" | "searchbox" | "spinbutton")) {
        bail!("text target has unsupported accessibility role: {:?}", role);
    }

    let disabled = node.properties.as_ref().is_some_and(|properties| {
        properties.iter().any(|property| {
            let property_name = serde_json::to_value(&property.name).ok();
            property_name.as_ref().and_then(serde_json::Value::as_str) == Some("disabled")
                && property.value.value.as_ref()
                    .and_then(serde_json::Value::as_bool) == Some(true)
        })
    });
    if disabled {
        bail!("text target is disabled");
    }
    Ok(())
}

async fn accessibility_value_contains(
    page: &Page,
    backend_node_id: &BackendNodeId,
    expected: &str,
) -> Result<bool> {
    let tree = page
        .execute(GetFullAxTreeParams::builder().build())
        .await
        .context("Read accessibility tree after text entry")?;
    let node = tree.result.nodes.iter().find(|node| {
        node.backend_dom_node_id
            .as_ref()
            .is_some_and(|id| *id.inner() == *backend_node_id.inner())
    });
    let Some(node) = node else {
        return Ok(false);
    };
    let value = node.value.as_ref().and_then(|value| ax_value_to_string(&value.value));
    Ok(value.is_some_and(|value| value.contains(expected)))
}

async fn scroll_backend_node_into_view(
    page: &Page,
    backend_node_id: &BackendNodeId,
) -> Result<()> {
    page.execute(
        ScrollIntoViewIfNeededParams::builder()
            .backend_node_id(backend_node_id.clone())
            .build(),
    )
    .await
    .context("DOM.scrollIntoViewIfNeeded")?;
    Ok(())
}

async fn click_backend_node(page: &Page, backend_node_id: &BackendNodeId) -> Result<()> {
    scroll_backend_node_into_view(page, backend_node_id).await?;
    let quads = page
        .execute(
            GetContentQuadsParams::builder()
                .backend_node_id(backend_node_id.clone())
                .build(),
        )
        .await
        .context("DOM.getContentQuads")?
        .result;
    let point = quads
        .quads
        .iter()
        .find_map(|quad| quad_center(quad.inner()))
        .context("Click target has no visible content quad")?;
    page.click(point).await.context("Input dispatch click")?;
    Ok(())
}

fn quad_center(coordinates: &[f64]) -> Option<Point> {
    if coordinates.len() != 8 || coordinates.iter().any(|coordinate| !coordinate.is_finite()) {
        return None;
    }
    let mut twice_area = 0.0;
    for index in 0..4 {
        let next = (index + 1) % 4;
        twice_area += coordinates[index * 2] * coordinates[next * 2 + 1]
            - coordinates[next * 2] * coordinates[index * 2 + 1];
    }
    if twice_area.abs() < 2.0 {
        return None;
    }
    let x = (coordinates[0] + coordinates[2] + coordinates[4] + coordinates[6]) / 4.0;
    let y = (coordinates[1] + coordinates[3] + coordinates[5] + coordinates[7]) / 4.0;
    Some(Point::new(x, y))
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

#[cfg(test)]
mod native_cdp_tests {
    use super::*;

    #[test]
    fn quad_center_returns_center_of_visible_quad() {
        let point = quad_center(&[0.0, 0.0, 10.0, 0.0, 10.0, 20.0, 0.0, 20.0])
            .expect("valid quad should have a center");
        assert!((point.x - 5.0).abs() < f64::EPSILON);
        assert!((point.y - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn quad_center_rejects_degenerate_and_malformed_quads() {
        assert!(quad_center(&[0.0, 0.0, 1.0, 1.0]).is_none());
        assert!(quad_center(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).is_none());
        assert!(quad_center(&[0.0, 0.0, f64::NAN, 0.0, 10.0, 10.0, 0.0, 10.0]).is_none());
    }

    #[test]
    fn core_action_paths_do_not_call_page_evaluate() {
        let source = include_str!("cdp.rs");
        let sections = [
            ("pub async fn get_accessibility_tree", "    /// Click an element"),
            ("pub(crate) async fn click", "    /// Type text"),
            ("pub(crate) async fn type_text", "    /// Scroll a CSS-selected"),
            ("pub(crate) async fn scroll_to", "    pub(crate) async fn extract_text"),
            ("pub(crate) async fn go_back", "    pub(crate) async fn go_forward"),
            ("pub(crate) async fn go_forward", "    /// Extract page text"),
            ("pub async fn observe", "/// Extract a string from"),
        ];
        for (start, end) in sections {
            let start_at = source.find(start).expect("source section start exists");
            let end_at = source[start_at..]
                .find(end)
                .map(|offset| start_at + offset)
                .expect("source section end exists");
            assert!(
                !source[start_at..end_at].contains(".evaluate("),
                "JavaScript evaluation remains in action path starting at {start}"
            );
        }
    }
}

