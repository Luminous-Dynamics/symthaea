// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generic data table and pagination components.

use leptos::prelude::*;
use std::sync::Arc;

/// Column definition for a data table.
pub struct Column<T: 'static> {
    pub header: String,
    pub render: Arc<dyn Fn(&T) -> AnyView + Send + Sync>,
    pub width: Option<String>,
}

/// Generic data table.
#[component]
pub fn DataTable<T: Clone + Send + Sync + 'static>(
    items: Signal<Vec<T>>,
    #[prop(into)] columns: Vec<Column<T>>,
    #[prop(optional)] empty_message: &'static str,
) -> impl IntoView {
    let empty_msg = if empty_message.is_empty() {
        "No data"
    } else {
        empty_message
    };
    let columns = Arc::new(columns);
    let cols_header = Arc::clone(&columns);
    let cols_body = Arc::clone(&columns);

    view! {
        <div class="data-table-wrapper">
            <table class="data-table" role="grid">
                <thead>
                    <tr>
                        {cols_header.iter().map(|col| {
                            let style = col.width.as_ref()
                                .map(|w| format!("width: {w}"))
                                .unwrap_or_default();
                            let header = col.header.clone();
                            view! { <th scope="col" style=style>{header}</th> }
                        }).collect_view()}
                    </tr>
                </thead>
                <tbody>
                    {move || {
                        let data = items.get();
                        if data.is_empty() {
                            view! {
                                <tr>
                                    <td colspan=cols_body.len().to_string() class="data-table-empty">
                                        {empty_msg}
                                    </td>
                                </tr>
                            }.into_any()
                        } else {
                            data.iter().map(|item| {
                                let cells = cols_body.iter().map(|col| {
                                    let cell = (col.render)(item);
                                    view! { <td>{cell}</td> }
                                }).collect_view();
                                view! { <tr>{cells}</tr> }
                            }).collect_view().into_any()
                        }
                    }}
                </tbody>
            </table>
        </div>
    }
}

/// Simple pagination control.
#[component]
pub fn Pagination(
    total: Signal<usize>,
    page: ReadSignal<usize>,
    #[prop(optional, default = 20)] per_page: usize,
    on_page_change: Callback<usize>,
) -> impl IntoView {
    let total_pages = move || {
        let t = total.get();
        if t == 0 {
            1
        } else {
            (t + per_page - 1) / per_page
        }
    };

    view! {
        <div class="pagination">
            <button
                class="btn btn-ghost pagination-btn"
                disabled=move || page.get() == 0
                on:click=move |_| {
                    let p = page.get();
                    if p > 0 { on_page_change.run(p - 1); }
                }
            >
                "\u{2190} Prev"
            </button>
            <span class="pagination-info">
                {move || format!("{} / {}", page.get() + 1, total_pages())}
            </span>
            <button
                class="btn btn-ghost pagination-btn"
                disabled=move || page.get() + 1 >= total_pages()
                on:click=move |_| {
                    let p = page.get();
                    if p + 1 < total_pages() { on_page_change.run(p + 1); }
                }
            >
                "Next \u{2192}"
            </button>
        </div>
    }
}
