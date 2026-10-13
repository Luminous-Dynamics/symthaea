// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Chrome DevTools Protocol session wrapper.
//!
//! The low-level page-observation and mutation methods are crate-private. Public
//! page reads and actions should go through [`crate::executor::BrowserExecutor`],
//! which applies capability, Phi and URL checks before exposing page state.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use chromiumoxide::Page;
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::accessibility::GetFullAxTreeParams;
use chromiumoxide::cdp::browser_protocol::dom::{
    BackendNodeId, GetContentQuadsParams, GetOuterHtmlParams, ScrollIntoViewIfNeededParams,
};
use chromiumoxide::cdp::browser_protocol::input::InsertTextParams;
use chromiumoxide::cdp::browser_protocol::page::{
    GetNavigationHistoryParams, NavigateToHistoryEntryParams,
};
use chromiumoxide::handler::viewport::Viewport;
use chromiumoxide::layout::Point;
use futures::StreamExt;
use scraper::{Html, Selector};
use tokio::sync::Mutex;
use tracing::debug;

use crate::actions::ElementSelector;
use crate::config::BrowserAgentConfig;
use crate::observation::{AccessibleElement, PageObservation};
use crate::safety::PolicyDecision;

/// Refuse to parse unusually large page fragments after CDP has returned them.
const MAX_TEXT_EXTRACTION_HTML_BYTES: usize = 2 * 1024 * 1024;
/// Bound text delivered to downstream cognition / receipts.
const MAX_TEXT_EXTRACTION_CHARS: usize = 32 * 1024;

/// CDP session wrapping a `chromiumoxide::Browser` and its active page.
pub struct CdpSession {
    _browser: Browser,
    /// Serializes complete executor actions across URL preflight and CDP dispatch.
    action_gate: Mutex<()>,
    page: Mutex<Page>,
    max_elements: usize,
    navigation_timeout: Duration,
}

impl CdpSession {
    /// Acquire the per-session action gate for the full policy/dispatch transaction.
    pub(crate) async fn begin_action(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.action_gate.lock().await
    }

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
            action_gate: Mutex::new(()),
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
            action_gate: Mutex::new(()),
            page: Mutex::new(page),
            max_elements: config.max_elements,
            navigation_timeout: Duration::from_millis(config.navigation_timeout_ms),
        })
    }

    /// Read the URL of the current top-level page using Chromiumoxide's CDP URL path.
    pub(crate) async fn current_url(&self) -> Result<String> {
        let page = self.page.lock().await;
        page.url()
            .await
            .context("Read current page URL")?
            .context("Current page URL is unavailable")
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
    pub(crate) async fn get_accessibility_tree(&self) -> Result<PageObservation> {
        let page = self.page.lock().await;

        let url = page
            .url()
            .await
            .unwrap_or(None)
            .unwrap_or_else(|| "unknown".to_string());
        let title = native_document_title(&page)
            .await
            .unwrap_or_else(|_| "Untitled".to_string());

        let params = GetFullAxTreeParams::builder().build();
        let response = page
            .execute(params)
            .await
            .context("Failed to get accessibility tree")?;

        let mut elements = Vec::new();
        let mut focused_element = None;
        for node in &response.result.nodes {
            // Ignored AX nodes are not actionable user-facing controls. Do not
            // leak hidden/presentational nodes into the observation projection.
            if node.ignored {
                continue;
            }

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
            let ax_property_is_true = |wanted: &str| {
                node.properties.as_ref().is_some_and(|properties| {
                    properties.iter().any(|property| {
                        let property_name = serde_json::to_value(&property.name).ok();
                        property_name.as_ref().and_then(serde_json::Value::as_str)
                            == Some(wanted)
                            && property.value.value.as_ref()
                                .and_then(serde_json::Value::as_bool) == Some(true)
                    })
                })
            };
            let focused = ax_property_is_true("focused");
            let disabled = ax_property_is_true("disabled");
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
                disabled,
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
            ElementSelector::Css(css) => {
                resolve_unique_css_backend_node(&page, css)
                    .await
                    .context("CSS click target must match exactly one element")?
            },
            ElementSelector::BackendNodeId(id) => {
                bail!("unsupported raw backend node selector: {id}");
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
            ElementSelector::Css(css) => {
                resolve_unique_css_backend_node(&page, css)
                    .await
                    .context("CSS text target must match exactly one element")?
            },
            ElementSelector::BackendNodeId(id) => {
                bail!("unsupported raw backend node selector: {id}");
            }
            ElementSelector::Accessible { role, name } => {
                resolve_accessible_backend_node(&page, role, name).await?
            }
        };

        validate_text_entry_target(&page, &backend_node_id)
            .await
            .context("Text target is not a supported editable control")?;
        let previous_value = accessibility_value(&page, &backend_node_id).await?;
        click_backend_node(&page, &backend_node_id)
            .await
            .context("Focus text target with native CDP input")?;
        page.execute(InsertTextParams::new(text.to_owned()))
            .await
            .context("Input.insertText failed")?;
        let current_value = accessibility_value(&page, &backend_node_id).await?;
        let value_changed = current_value != previous_value;
        let contains_inserted_text = current_value
            .as_deref()
            .is_some_and(|value| value.contains(text));
        if !value_changed || !contains_inserted_text {
            bail!("AX tree did not confirm the text insertion");
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
        let backend_node_id = resolve_unique_css_backend_node(&page, css)
            .await
            .context("CSS scroll target must match exactly one element")?;
        scroll_backend_node_into_view(&page, &backend_node_id)
            .await
            .context("Native CDP scroll failed")?;
        Ok(())
    }

    /// Navigate back only after checking the history entry URL under policy.
    ///
    /// The target URL is checked while the page mutex is held and immediately
    /// before the exact history entry ID is dispatched.
    pub(crate) async fn go_back_with_url_policy<F>(&self, check_url: F) -> Result<PolicyDecision>
    where
        F: Fn(&str) -> PolicyDecision,
    {
        let page = self.page.lock().await;
        navigate_history(&page, -1, check_url)
            .await
            .context("History back failed")
    }

    /// Navigate forward only after checking the history entry URL under policy.
    pub(crate) async fn go_forward_with_url_policy<F>(&self, check_url: F) -> Result<PolicyDecision>
    where
        F: Fn(&str) -> PolicyDecision,
    {
        let page = self.page.lock().await;
        navigate_history(&page, 1, check_url)
            .await
            .context("History forward failed")
    }

    /// Extract current DOM text through typed CDP and Rust HTML parsing.
    ///
    /// This is a DOM-text extraction contract, not a promise of pixel-perfect
    /// browser text-layout semantics. The parser drops executable/non-content
    /// elements and explicit hidden-state attributes; computed CSS visibility,
    /// shadow roots, and subframe flattening are not synthesized by this path.
    pub(crate) async fn extract_text(&self, selector: Option<&str>) -> Result<String> {
        let page = self.page.lock().await;
        let target_selector = selector.unwrap_or("body");
        let outer_html = read_outer_html(&page, target_selector)
            .await
            .context("Read text target with DOM.getOuterHTML")?;
        if outer_html.len() > MAX_TEXT_EXTRACTION_HTML_BYTES {
            bail!(
                "DOM text target exceeded the {} byte HTML extraction limit",
                MAX_TEXT_EXTRACTION_HTML_BYTES
            );
        }
        let text = extract_html_text(&outer_html)
            .context("Parse and extract native DOM text")?;
        if text.chars().count() > MAX_TEXT_EXTRACTION_CHARS {
            bail!(
                "DOM text target exceeded the {} character extraction limit",
                MAX_TEXT_EXTRACTION_CHARS
            );
        }
        Ok(text)
    }

    /// Capture a PNG screenshot of the current page.
    pub(crate) async fn screenshot(&self) -> Result<Vec<u8>> {
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

    /// Observe through the native accessibility tree only.
    ///
    /// An unavailable tree is returned as an error; there is no silent JavaScript
    /// fallback that could hide a degraded observation path.
    pub(crate) async fn observe(&self) -> Result<PageObservation> {
        self.get_accessibility_tree()
            .await
            .context("Accessibility observation unavailable; JavaScript fallback is disabled")
    }
}

async fn resolve_unique_css_backend_node(page: &Page, selector: &str) -> Result<BackendNodeId> {
    let matches = page
        .find_elements(selector)
        .await
        .with_context(|| format!("DOM query failed for CSS selector: {selector}"))?;
    require_unique_css_match_count(selector, matches.len())?;
    matches
        .first()
        .map(|element| element.backend_node_id.clone())
        .context("CSS selector match disappeared before target resolution")
}

fn require_unique_css_match_count(selector: &str, count: usize) -> Result<()> {
    match count {
        0 => bail!("CSS selector matched no elements: {selector}"),
        1 => Ok(()),
        multiple => bail!("ambiguous CSS selector {selector}: {multiple} matches"),
    }
}

async fn read_outer_html(page: &Page, selector: &str) -> Result<String> {
    let backend_node_id = resolve_unique_css_backend_node(page, selector)
        .await
        .context("DOM text target must match exactly one element")?;
    let response = page
        .execute(
            GetOuterHtmlParams::builder()
                .backend_node_id(backend_node_id)
                .build(),
        )
        .await
        .context("DOM.getOuterHTML failed")?;
    Ok(response.result.outer_html)
}

async fn native_document_title(page: &Page) -> Result<String> {
    let title_html = read_outer_html(page, "title").await?;
    let title = extract_html_text(&title_html)?;
    if title.is_empty() {
        bail!("document title is empty");
    }
    Ok(title)
}

fn extract_html_text(html: &str) -> Result<String> {
    let mut document = Html::parse_fragment(html);
    let excluded = Selector::parse(
        "script, style, noscript, template, [hidden], [aria-hidden='true']",
    )
    .context("Invalid internal exclusion selector")?;
    let excluded_nodes: Vec<_> = document
        .select(&excluded)
        .map(|element| element.id())
        .collect();

    // Reverse document order so descendants are detached before an ancestor.
    for id in excluded_nodes.into_iter().rev() {
        // `HtmlTreeSink` implements the html5ever tree-sink adapter, but
        // the parsed tree itself is an ego_tree::Tree<Node>. Mutate that tree
        // directly so excluded nodes are detached before their ancestors.
        document
            .tree
            .get_mut(id)
            .context("Excluded HTML node disappeared during extraction")?
            .detach();
    }

    let text = document
        .root_element()
        .text()
        .collect::<Vec<_>>()
        .join(" ");
    Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

async fn navigate_history<F>(
    page: &Page,
    offset: i64,
    check_url: F,
) -> Result<PolicyDecision>
where
    F: Fn(&str) -> PolicyDecision,
{
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
    let decision = check_url(&entry.url);
    if !decision.is_allowed() {
        return Ok(decision);
    }

    let entry_id = entry.id;
    page.execute(
        NavigateToHistoryEntryParams::builder()
            .entry_id(entry_id)
            .build()
            .map_err(|error| anyhow::anyhow!("Build history navigation request: {error}"))?,
    )
    .await
    .context("Navigate to history entry")?;
    Ok(decision)
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
            if node.ignored {
                return false;
            }
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

async fn validate_click_target_enabled(page: &Page, backend_node_id: &BackendNodeId) -> Result<()> {
    let tree = page
        .execute(GetFullAxTreeParams::builder().build())
        .await
        .context("Read accessibility tree before click")?;
    let disabled = tree
        .result
        .nodes
        .iter()
        .find(|node| {
            !node.ignored
                && node
                    .backend_dom_node_id
                    .as_ref()
                    .is_some_and(|id| *id.inner() == *backend_node_id.inner())
        })
        .map(|node| {
        node.properties.as_ref().is_some_and(|properties| {
            properties.iter().any(|property| {
                let property_name = serde_json::to_value(&property.name).ok();
                property_name.as_ref().and_then(serde_json::Value::as_str) == Some("disabled")
                    && property.value.value.as_ref()
                        .and_then(serde_json::Value::as_bool) == Some(true)
            })
        })
    });
    validate_click_target_state(disabled)
}

/// A CSS-resolved node that cannot be confirmed in the accessibility tree is
/// not eligible for an action. Missing evidence must not imply "enabled".
fn validate_click_target_state(disabled: Option<bool>) -> Result<()> {
    match disabled {
        None => bail!("click target is not present in the accessibility tree"),
        Some(true) => bail!("click target is disabled in the accessibility tree"),
        Some(false) => Ok(()),
    }
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
            !node.ignored
                && node
                    .backend_dom_node_id
                    .as_ref()
                    .is_some_and(|id| *id.inner() == *backend_node_id.inner())
        })
        .context("text target is not exposed in the accessibility tree")?;
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
    let read_only = node.properties.as_ref().is_some_and(|properties| {
        properties.iter().any(|property| {
            let property_name = serde_json::to_value(&property.name).ok();
            property_name.as_ref().and_then(serde_json::Value::as_str) == Some("readonly")
                && property.value.value.as_ref()
                    .and_then(serde_json::Value::as_bool) == Some(true)
        })
    });
    if read_only {
        bail!("text target is read-only");
    }
    Ok(())
}

async fn accessibility_value(
    page: &Page,
    backend_node_id: &BackendNodeId,
) -> Result<Option<String>> {
    let tree = page
        .execute(GetFullAxTreeParams::builder().build())
        .await
        .context("Read accessibility tree for text value verification")?;
    let node = tree.result.nodes.iter().find(|node| {
        node.backend_dom_node_id
            .as_ref()
            .is_some_and(|id| *id.inner() == *backend_node_id.inner())
    });
    let Some(node) = node else {
        return Ok(None);
    };
    Ok(node.value.as_ref().and_then(|value| ax_value_to_string(&value.value)))
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
    validate_click_target_enabled(page, backend_node_id)
        .await
        .context("Click target enablement check failed")?;
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
    fn css_target_cardinality_fails_closed() {
        assert!(require_unique_css_match_count("button.primary", 1).is_ok());
        assert!(require_unique_css_match_count("button.primary", 0).is_err());
        assert!(require_unique_css_match_count("button.primary", 2).is_err());
    }

    #[test]
    fn click_target_validation_fails_closed_without_ax_evidence() {
        assert!(
            validate_click_target_state(None).is_err(),
            "a DOM target missing from the AX tree must not be treated as enabled"
        );
        assert!(
            validate_click_target_state(Some(true)).is_err(),
            "an explicitly disabled target must be rejected"
        );
        assert!(
            validate_click_target_state(Some(false)).is_ok(),
            "an observed enabled target may proceed"
        );
    }

    #[test]
    fn native_html_extraction_excludes_non_content_and_hidden_subtrees() {
        let html = r#"
            <main>
                <p>Alpha <em>beta</em></p>
                <script>secret_script()</script>
                <style>.concealed { display: none }</style>
                <noscript>no script</noscript>
                <template>template content</template>
                <p hidden>hidden content</p>
                <p aria-hidden="true">screen-reader-hidden content</p>
                <p>Gamma</p>
            </main>
        "#;
        let text = extract_html_text(html).expect("HTML extraction should succeed");
        assert!(text.contains("Alpha beta"));
        assert!(text.contains("Gamma"));
        assert!(!text.contains("secret_script"));
        assert!(!text.contains("concealed"));
        assert!(!text.contains("template content"));
        assert!(!text.contains("hidden content"));
        assert!(!text.contains("screen-reader-hidden content"));
    }

    #[test]
    fn maintained_browser_control_surface_has_no_injected_js_helpers() {
        let cdp_source = include_str!("cdp.rs");
        // Exclude this file's test modules so the forbidden-token list below
        // cannot satisfy itself by containing the strings it is checking.
        let cdp_production = cdp_source
            .split("#[cfg(test)]")
            .next()
            .expect("production source prefix exists");
        for raw_public_api in [
            "pub async fn get_accessibility_tree(",
            "pub async fn get_page_state_js(",
            "pub async fn observe(",
            "pub async fn screenshot(",
        ] {
            assert!(
                !cdp_production.contains(raw_public_api),
                "low-level page observation API leaked as public: {raw_public_api}"
            );
        }

        let sources = [
            ("cdp.rs", cdp_production),
            ("actions.rs", include_str!("actions.rs")),
            ("config.rs", include_str!("config.rs")),
            ("embodiment.rs", include_str!("embodiment.rs")),
            ("encoder.rs", include_str!("encoder.rs")),
            ("executor.rs", include_str!("executor.rs")),
            ("lib.rs", include_str!("lib.rs")),
            ("observation.rs", include_str!("observation.rs")),
            ("safety.rs", include_str!("safety.rs")),
            ("web_agent.rs", include_str!("web_agent.rs")),
            (
                "examples/praxis_journey.rs",
                include_str!("../examples/praxis_journey.rs"),
            ),
        ];
        let forbidden = [
            ".evaluate(",
            ".get_title(",
            ".content(",
            ".inner_text(",
            ".inner_html(",
            ".outer_html(",
            ".attribute(",
            ".property(",
            ".string_property(",
            ".click()",
            ".scroll_into_view(",
            ".hover()",
            "eval_js(",
            "call_js_fn(",
            "Runtime.evaluate",
        ];

        for (path, source) in sources {
            for token in forbidden {
                assert!(
                    !source.contains(token),
                    "forbidden in-page evaluation helper {token:?} remains in maintained {path}"
                );
            }
        }

        // Keep direct CdpSession access confined to the executor module itself.
        // These maintained perception clients should only use BrowserExecutor.
        let raw_api_callers = [
            ("web_agent.rs", include_str!("web_agent.rs")),
            ("embodiment.rs", include_str!("embodiment.rs")),
            (
                "examples/praxis_journey.rs",
                include_str!("../examples/praxis_journey.rs"),
            ),
        ];
        let forbidden_session_calls = [
            "session.observe()",
            "session.get_accessibility_tree()",
            "session.get_page_state_js()",
            "session.screenshot()",
            "session.navigate(",
            "session.click(",
            "session.type_text(",
            "session.scroll_to(",
            "session.extract_text(",
            "session.go_back_with_url_policy(",
            "session.go_forward_with_url_policy(",
        ];
        for (path, source) in raw_api_callers {
            for token in forbidden_session_calls {
                assert!(
                    !source.contains(token),
                    "maintained caller {path} bypasses BrowserExecutor with {token}"
                );
            }
        }
    }
}

