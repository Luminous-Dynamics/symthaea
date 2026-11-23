//! SFEB-L2-005 Leptos CSR reference candidate.
use leptos::{mount::mount_to_body, prelude::*};
use std::collections::BTreeSet;
use wasm_bindgen::JsValue;

#[path = "../../../contract-check/src/search_filter.rs"]
mod search_filter;
use search_filter::{Asset, PageResult, QueryState};

const DATASET: &str = include_str!("../../../fixtures/search-filter-explorer/dataset.json");

#[derive(Clone)]
struct PageView {
    result: PageResult,
    rows: Vec<Asset>,
}

fn initial_query() -> Result<QueryState, String> {
    let Some(window) = web_sys::window() else { return Ok(QueryState::default()); };
    let search = window.location().search()
        .map_err(|_| "browser could not read the initial URL query".to_string())?;
    if search.is_empty() { return Ok(QueryState::default()); }
    search_filter::decode_state_url(&search)
}

fn cents_to_decimal(cents: u64) -> String {
    format!("{}.{:02}", cents / 100, cents % 100)
}

/// Price parsing is integer-only. None means invalid input; Some(None) clears a bound.
fn parse_price_cents(raw: &str) -> Option<Option<u64>> {
    let value = raw.trim();
    if value.is_empty() { return Some(None); }
    let mut parts = value.split('.');
    let whole_text = parts.next()?;
    let fraction_text = parts.next().unwrap_or("0");
    if parts.next().is_some()
        || fraction_text.len() > 2
        || (!whole_text.is_empty() && !whole_text.bytes().all(|b| b.is_ascii_digit()))
        || (!fraction_text.is_empty() && !fraction_text.bytes().all(|b| b.is_ascii_digit()))
        || (whole_text.is_empty() && fraction_text.is_empty()) {
        return None;
    }
    let whole = if whole_text.is_empty() { 0 } else { whole_text.parse::<u64>().ok()? };
    let fraction = match fraction_text.len() {
        0 => 0,
        1 => fraction_text.parse::<u64>().ok()? * 10,
        2 => fraction_text.parse::<u64>().ok()?,
        _ => return None,
    };
    let cents = whole.checked_mul(100)?.checked_add(fraction)?;
    if cents > 9_007_199_254_740_991 { return None; }
    Some(Some(cents))
}

fn compute_page(assets: &[Asset], query: &QueryState) -> Result<PageView, String> {
    let result = search_filter::evaluate(assets, query)?;
    let rows = result.asset_ids.iter()
        .filter_map(|id| assets.iter().find(|asset| &asset.id == id).cloned())
        .collect();
    Ok(PageView { result, rows })
}

fn persist_url(query: QueryState) {
    let Ok(search) = search_filter::encode_state_url(&query) else { return; };
    let Some(window) = web_sys::window() else { return; };
    let location = window.location();
    let path = location.pathname().unwrap_or_else(|_| "/".to_string());
    let hash = location.hash().unwrap_or_default();
    let url = format!("{path}{search}{hash}");
    if let Ok(history) = window.history() {
        let _ = history.replace_state_with_url(&JsValue::NULL, "", Some(&url));
    }
}

fn App() -> impl IntoView {
    let (dataset_id, assets) = search_filter::parse_dataset(DATASET)
        .expect("bundled public-pilot dataset must be valid");
    let categories: Vec<String> = assets.iter().map(|asset| asset.category.clone())
        .collect::<BTreeSet<_>>().into_iter().collect();
    let (initial_state, initial_url_error) = match initial_query() {
        Ok(state) => (state, None),
        Err(error) => (QueryState::default(), Some(error)),
    };
    let query = RwSignal::new(initial_state);
    let initial_url_error = RwSignal::new(initial_url_error);
    let input_error = RwSignal::new(None::<String>);
    let page = Memo::new(move |_| compute_page(&assets, &query.get()));

    // Do not overwrite an invalid deep link until the user deliberately changes state.
    Effect::new(move |_| {
        if initial_url_error.get().is_none() { persist_url(query.get()); }
    });

    view! {
        <div class="app-shell" data-sfeb-id="inventory.shell">
            <a class="skip-link" href="#main-content">"Skip to main content"</a>
            <header class="topbar" data-sfeb-id="inventory.header">
                <a class="brand" href="#main-content">
                    <span class="brand-mark" aria-hidden="true">"✳"</span>
                    <span><strong>"Luminous Fieldwork"</strong><small>"OPERATIONS / STORES"</small></span>
                </a>
                <nav aria-label="Primary navigation">
                    <a href="#main-content" aria-current="page">"Inventory"</a>
                    <a href="#asset-table">"Assets"</a>
                </nav>
            </header>
            <main id="main-content" data-sfeb-id="inventory.main">
                <section class="page-heading" data-sfeb-id="inventory.heading" aria-labelledby="inventory-title">
                    <div>
                        <p class="eyebrow">"FIELD OPERATIONS / INVENTORY CONTROL"</p>
                        <h1 id="inventory-title">"Asset inventory"</h1>
                        <p class="intro">"Find equipment by name, SKU or description. Share a reproducible search link."</p>
                    </div>
                    <div class="dataset-stamp">
                        <span class="stamp-dot" aria-hidden="true"></span><span>"LOCAL FIXTURE"</span><code>{dataset_id}</code>
                    </div>
                </section>

                <section class="filter-panel panel" data-sfeb-id="inventory.filters" aria-labelledby="filter-title">
                    <div class="section-heading">
                        <div><p class="eyebrow">"REFINE RESULTS"</p><h2 id="filter-title">"Search and filters"</h2></div>
                        <button class="quiet-button" type="button" on:click=move |_| {
                            initial_url_error.set(None);
                            query.set(QueryState::default());
                            input_error.set(None);
                        }>"Clear all"</button>
                    </div>
                    <div class="filter-grid">
                        <div class="field search-field">
                            <label for="asset-search">"Search assets"</label>
                            <input id="asset-search" type="search" autocomplete="off"
                                placeholder="Name, SKU or description"
                                prop:value=move || query.get().q
                                on:input=move |event| {
                                    let value = event_target_value(&event);
                                    initial_url_error.set(None);
                                    query.update(|state| { state.q = value; state.page = 1; });
                                } />
                        </div>
                        <div class="field">
                            <label for="asset-category">"Category"</label>
                            <select id="asset-category" prop:value=move || query.get().category.unwrap_or_default()
                                on:change=move |event| {
                                    let value = event_target_value(&event);
                                    initial_url_error.set(None);
                                    query.update(|state| {
                                        state.category = if value.is_empty() { None } else { Some(value) };
                                        state.page = 1;
                                    });
                                }>
                                <option value="">"All categories"</option>
                                <For each=move || categories.clone()
                                    key=|category: &String| category.clone()
                                    children=move |category: String| view! { <option value=category.clone()>{category}</option> } />
                            </select>
                        </div>
                        <div class="field">
                            <label for="min-price">"Minimum price (USD)"</label>
                            <input id="min-price" type="number" min="0" step="0.01" inputmode="decimal"
                                prop:value=move || query.get().min_price_cents.map(cents_to_decimal).unwrap_or_default()
                                on:input=move |event| {
                                    match parse_price_cents(&event_target_value(&event)) {
                                        Some(value) => {
                                            initial_url_error.set(None);
                                            input_error.set(None);
                                            query.update(|state| { state.min_price_cents = value; state.page = 1; });
                                        }
                                        None => input_error.set(Some("Use a non-negative price with at most two decimal places within the supported safe-integer range.".to_string())),
                                    }
                                } />
                        </div>
                        <div class="field">
                            <label for="max-price">"Maximum price (USD)"</label>
                            <input id="max-price" type="number" min="0" step="0.01" inputmode="decimal"
                                prop:value=move || query.get().max_price_cents.map(cents_to_decimal).unwrap_or_default()
                                on:input=move |event| {
                                    match parse_price_cents(&event_target_value(&event)) {
                                        Some(value) => {
                                            initial_url_error.set(None);
                                            input_error.set(None);
                                            query.update(|state| { state.max_price_cents = value; state.page = 1; });
                                        }
                                        None => input_error.set(Some("Use a non-negative price with at most two decimal places within the supported safe-integer range.".to_string())),
                                    }
                                } />
                        </div>
                        <label class="check-field" for="available-only">
                            <input id="available-only" type="checkbox" prop:checked=move || query.get().available_only
                                on:change=move |event| {
                                    let checked = event_target_checked(&event);
                                    initial_url_error.set(None);
                                    query.update(|state| { state.available_only = checked; state.page = 1; });
                                } />
                            <span>"In stock only"</span>
                        </label>
                    </div>
                    <div class="filter-footnote">
                        <span>"Filters combine together; price bounds are inclusive."</span>
                        <span>"Money is evaluated in integer cents."</span>
                    </div>
                    {move || initial_url_error.get().map(|message| view! { <p class="field-error" role="alert">{format!("Invalid shared URL: {message}. Change a filter or clear all to create a new valid state URL.")}</p> })}
                    {move || input_error.get().map(|message| view! { <p class="field-error" role="alert">{message}</p> })}
                </section>

                <section class="results-panel panel" data-sfeb-id="inventory.results" aria-labelledby="results-title">
                    <div class="results-heading">
                        <div>
                            <p class="eyebrow">"MATCHING EQUIPMENT"</p><h2 id="results-title">"All assets"</h2>
                            <p class="result-count" role="status" aria-live="polite">
                                {move || match page.get() {
                                    Ok(data) => format!("{} assets · page {} of {}", data.result.total_matches, data.result.page, data.result.page_count),
                                    Err(error) => format!("Query rejected: {error}"),
                                }}
                            </p>
                        </div>
                        <div class="sort-field">
                            <label for="sort-order">"Sort by"</label>
                            <select id="sort-order" aria-label="Sort assets" prop:value=move || query.get().sort
                                on:change=move |event| {
                                    let value = event_target_value(&event);
                                    initial_url_error.set(None);
                                    query.update(|state| { state.sort = value; state.page = 1; });
                                }>
                                <option value="name_asc">"Name A to Z"</option>
                                <option value="price_asc">"Price: low to high"</option>
                                <option value="price_desc">"Price: high to low"</option>
                                <option value="stock_asc">"Stock: low to high"</option>
                            </select>
                        </div>
                    </div>

                    <div class="table-wrap">
                        <table id="asset-table" data-sfeb-id="inventory.table">
                            <caption class="visually-hidden">"Synthetic field equipment inventory; prices in US dollars."</caption>
                            <thead><tr>
                                <th scope="col">"Asset"</th><th scope="col">"SKU"</th><th scope="col">"Category"</th>
                                <th scope="col" class="numeric">"Unit price"</th><th scope="col" class="numeric">"Available"</th>
                            </tr></thead>
                            <tbody>
                                <For each=move || page.get().map(|data| data.rows).unwrap_or_default()
                                    key=|asset: &Asset| asset.id.clone()
                                    children=move |asset: Asset| view! {
                                        <tr data-asset-id=asset.id.clone()>
                                            <th scope="row"><span class="asset-name">{asset.name}</span><span class="asset-description">{asset.description}</span></th>
                                            <td><code>{asset.sku}</code></td>
                                            <td><span class="category-tag">{asset.category}</span></td>
                                            <td class="numeric price-cell">{format!("{}{}", "$", cents_to_decimal(asset.price_cents))}</td>
                                            <td class="numeric">
                                                <span class=if asset.stock_count == 0 { "stock-value out-of-stock" } else { "stock-value" }>{asset.stock_count}</span>
                                            </td>
                                        </tr>
                                    } />
                            </tbody>
                        </table>
                    </div>

                    {move || match page.get() {
                        Ok(data) if data.result.total_matches == 0 => view! {
                            <div class="empty-state"><strong>"No assets match these filters"</strong><p>"Try another search or clear one of the filters."</p></div>
                        }.into_any(),
                        Ok(data) if data.rows.is_empty() => view! {
                            <div class="empty-state"><strong>"No results on this page"</strong><p>"Choose another page or refine the query."</p></div>
                        }.into_any(),
                        Ok(_) => view! { <span class="visually-hidden"></span> }.into_any(),
                        Err(_) => view! {
                            <div class="empty-state error-state"><strong>"The current query is invalid"</strong><p>"The minimum price cannot exceed the maximum price."</p></div>
                        }.into_any(),
                    }}

                    <nav class="pagination" data-sfeb-id="inventory.pagination" aria-label="Asset result pages">
                        <div class="page-summary">{move || match page.get() {
                            Ok(data) => {
                                let start = if data.result.total_matches == 0 { 0 } else {
                                    ((data.result.page - 1) as usize * data.result.page_size as usize) + 1
                                };
                                let end = start.saturating_add(data.rows.len().saturating_sub(1));
                                format!("Showing {start}–{end} of {} assets", data.result.total_matches)
                            }
                            Err(_) => "Pagination unavailable for an invalid query".to_string(),
                        }}</div>
                        <div class="page-controls">
                            <label class="page-size-field" for="page-size"><span>"Rows"</span>
                                <input id="page-size" type="number" min="1" max="50" step="1"
                                    prop:value=move || query.get().page_size.to_string()
                                    on:change=move |event| {
                                        if let Ok(size) = event_target_value(&event).parse::<u32>() {
                                            initial_url_error.set(None);
                                            query.update(|state| { state.page_size = size.clamp(1, 50); state.page = 1; });
                                        }
                                    } />
                            </label>
                            <button type="button" class="page-button" disabled=move || query.get().page <= 1
                                on:click=move |_| {
                                    initial_url_error.set(None);
                                    query.update(|state| state.page = state.page.saturating_sub(1).max(1));
                                }>"Previous"</button>
                            <span class="page-number" aria-live="polite">{move || match page.get() {
                                Ok(data) => format!("{} / {}", data.result.page, data.result.page_count.max(1)),
                                Err(_) => "—".to_string(),
                            }}</span>
                            <button type="button" class="page-button"
                                disabled=move || match page.get() {
                                    Ok(data) => data.result.page_count == 0 || data.result.page >= data.result.page_count as u32,
                                    Err(_) => true,
                                }
                                on:click=move |_| if let Ok(data) = page.get() {
                                    let last = data.result.page_count.max(1) as u32;
                                    initial_url_error.set(None);
                                    query.update(|state| state.page = state.page.saturating_add(1).min(last));
                                }>"Next"</button>
                        </div>
                    </nav>
                </section>

                <footer class="page-footer">
                    <span>"SFEB-L2-005 · PUBLIC PILOT REFERENCE"</span>
                    <span>"Local synthetic data · no network services"</span>
                </footer>
            </main>
        </div>
    }
}

fn main() { mount_to_body(App); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_price_parser_is_exact() {
        assert_eq!(parse_price_cents("12.3"), Some(Some(1230)));
        assert_eq!(parse_price_cents("0.09"), Some(Some(9)));
        assert_eq!(parse_price_cents("120"), Some(Some(12000)));
        assert_eq!(parse_price_cents(""), Some(None));
    }

    #[test]
    fn integer_price_parser_rejects_ambiguous_values() {
        for invalid in ["-1", "+1", "1.234", "1.2.3", "abc"] {
            assert_eq!(parse_price_cents(invalid), None, "{invalid}");
        }
    }

    #[test]
    fn invalid_initial_url_is_rejected_without_defaulting() {
        assert!(search_filter::decode_state_url("?state=%").is_err());
    }

    #[test]
    fn integer_price_parser_enforces_javascript_safe_integer_ceiling() {
        assert_eq!(
            parse_price_cents("90071992547409.91"),
            Some(Some(9_007_199_254_740_991))
        );
        assert_eq!(parse_price_cents("90071992547409.92"), None);
    }

    #[test]
    fn cents_format_round_trips() {
        for cents in [0, 1, 9, 10, 99, 100, 101, 12_345, u64::MAX / 100] {
            assert_eq!(parse_price_cents(&cents_to_decimal(cents)), Some(Some(cents)));
        }
    }
}
