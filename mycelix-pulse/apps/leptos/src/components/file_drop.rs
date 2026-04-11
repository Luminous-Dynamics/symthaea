// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Drag-and-drop file attachment zone for compose page.

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Attachment {
    pub name: String,
    pub mime_type: String,
    pub size: u64,
    pub data_b64: String,
}

impl Attachment {
    pub fn size_display(&self) -> String {
        if self.size < 1024 { format!("{} B", self.size) }
        else if self.size < 1024 * 1024 { format!("{:.1} KB", self.size as f64 / 1024.0) }
        else { format!("{:.1} MB", self.size as f64 / (1024.0 * 1024.0)) }
    }
}

#[component]
pub fn FileDropZone(attachments: RwSignal<Vec<Attachment>>) -> impl IntoView {
    let is_drag_over = RwSignal::new(false);
    let error = RwSignal::new(Option::<String>::None);

    let on_dragover = move |ev: web_sys::DragEvent| {
        ev.prevent_default();
        is_drag_over.set(true);
    };
    let on_dragleave = move |_: web_sys::DragEvent| { is_drag_over.set(false); };

    let on_drop = move |ev: web_sys::DragEvent| {
        ev.prevent_default();
        is_drag_over.set(false);
        error.set(None);
        if let Some(dt) = ev.data_transfer() {
            if let Some(files) = dt.files() {
                for i in 0..files.length() {
                    if let Some(file) = files.get(i) {
                        read_file(file, attachments, error);
                    }
                }
            }
        }
    };

    let on_click = move |_| {
        // Trigger hidden file input
        let _ = js_sys::eval("document.getElementById('file-upload-hidden')?.click()");
    };

    let on_file_input = move |ev: leptos::ev::Event| {
        let input = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok());
        if let Some(input) = input {
            if let Some(files) = input.files() {
                for i in 0..files.length() {
                    if let Some(file) = files.get(i) {
                        read_file(file, attachments, error);
                    }
                }
            }
            input.set_value(""); // allow re-selecting same file
        }
    };

    view! {
        <div
            class=move || if is_drag_over.get() { "file-drop-zone drag-over" } else { "file-drop-zone" }
            on:dragover=on_dragover
            on:dragleave=on_dragleave
            on:drop=on_drop
            on:click=on_click
        >
            <input id="file-upload-hidden" type="file" multiple=true style="display:none"
                on:change=on_file_input />
            <span class="drop-icon">"\u{1F4CE}"</span>
            <span class="drop-text">"Drop files here or click to attach"</span>
        </div>

        // Error
        {move || error.get().map(|e| view! { <div class="drop-error">{e}</div> })}

        // Attachment list
        <div class="attachment-preview-list">
            {move || attachments.get().iter().enumerate().map(|(i, att)| {
                let name = att.name.clone();
                let size = att.size_display();
                view! {
                    <div class="attachment-preview">
                        <span class="att-icon">"\u{1F4C4}"</span>
                        <span class="att-name">{name.clone()}</span>
                        <span class="att-size">{size}</span>
                        <button class="att-remove" on:click=move |_| {
                            attachments.update(|a| { a.remove(i); });
                        }>"\u{2715}"</button>
                    </div>
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}

fn read_file(
    file: web_sys::File,
    attachments: RwSignal<Vec<Attachment>>,
    error: RwSignal<Option<String>>,
) {
    let name = file.name();
    let mime = file.type_();
    let size = file.size() as u64;

    if size > 25 * 1024 * 1024 {
        error.set(Some(format!("{name} exceeds 25MB limit")));
        return;
    }

    let reader = web_sys::FileReader::new().unwrap();
    let reader_clone = reader.clone();
    let name_clone = name.clone();
    let mime_clone = mime.clone();

    let onload = Closure::<dyn Fn()>::new(move || {
        if let Ok(result) = reader_clone.result() {
            let b64 = result.as_string().unwrap_or_default();
            // data:mime;base64,DATA — extract just the DATA part
            let data = b64.split(',').nth(1).unwrap_or(&b64).to_string();
            attachments.update(|a| {
                a.push(Attachment {
                    name: name_clone.clone(),
                    mime_type: mime_clone.clone(),
                    size,
                    data_b64: data,
                });
            });
        }
    });

    reader.set_onload(Some(onload.as_ref().unchecked_ref()));
    onload.forget();
    let _ = reader.read_as_data_url(&file);
}
