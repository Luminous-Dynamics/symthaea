// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rich text editor component using contentEditable.
//!
//! Provides a formatting toolbar (bold, italic, underline, lists, links, code)
//! and extracts HTML content for email composition.

use leptos::prelude::*;

#[component]
pub fn RichEditor(
    #[prop(into)] value: RwSignal<String>,
    #[prop(default = "Write your message...")] placeholder: &'static str,
) -> impl IntoView {
    let editor_ref = NodeRef::<leptos::html::Div>::new();

    // Execute formatting commands via execCommand (using eval for HtmlDocument access)
    let exec = |cmd: &str, arg: &str| {
        let js = format!("document.execCommand('{}',false,'{}')", cmd, arg);
        let _ = js_sys::eval(&js);
    };

    let on_bold = move |_| exec("bold", "");
    let on_italic = move |_| exec("italic", "");
    let on_underline = move |_| exec("underline", "");
    let on_strikethrough = move |_| exec("strikethrough", "");
    let on_ul = move |_| exec("insertUnorderedList", "");
    let on_ol = move |_| exec("insertOrderedList", "");
    let on_code = move |_| {
        let _ = js_sys::eval(
            "var s=window.getSelection();if(s.rangeCount){var r=s.getRangeAt(0);var c=document.createElement('code');c.className='inline-code';r.surroundContents(c)}"
        );
    };
    let on_blockquote = move |_| exec("formatBlock", "blockquote");
    let on_link = move |_| {
        let _ = js_sys::eval(
            "var u=prompt('Enter URL:');if(u)document.execCommand('createLink',false,u)"
        );
    };
    let on_clear = move |_| exec("removeFormat", "");

    // Sync contentEditable HTML → signal
    let on_input = move |_| {
        if let Some(el) = editor_ref.get() {
            let html = el.inner_html();
            value.set(html);
        }
    };

    // Set initial content
    Effect::new(move |prev: Option<bool>| {
        if prev.is_none() {
            if let Some(el) = editor_ref.get() {
                let initial = value.get_untracked();
                if !initial.is_empty() {
                    el.set_inner_html(&initial);
                }
            }
        }
        true
    });

    view! {
        <div class="rich-editor">
            <div class="editor-toolbar" role="toolbar" aria-label="Formatting">
                <button class="toolbar-btn" on:click=on_bold title="Bold (Ctrl+B)" aria-label="Bold">
                    <strong>"B"</strong>
                </button>
                <button class="toolbar-btn" on:click=on_italic title="Italic (Ctrl+I)" aria-label="Italic">
                    <em>"I"</em>
                </button>
                <button class="toolbar-btn" on:click=on_underline title="Underline (Ctrl+U)" aria-label="Underline">
                    <u>"U"</u>
                </button>
                <button class="toolbar-btn" on:click=on_strikethrough title="Strikethrough" aria-label="Strikethrough">
                    <s>"S"</s>
                </button>
                <span class="toolbar-divider" />
                <button class="toolbar-btn" on:click=on_ul title="Bullet list" aria-label="Bullet list">
                    "\u{2022}"
                </button>
                <button class="toolbar-btn" on:click=on_ol title="Numbered list" aria-label="Numbered list">
                    "1."
                </button>
                <span class="toolbar-divider" />
                <button class="toolbar-btn" on:click=on_code title="Inline code" aria-label="Code">
                    "<>"
                </button>
                <button class="toolbar-btn" on:click=on_blockquote title="Quote" aria-label="Block quote">
                    "\u{201C}"
                </button>
                <button class="toolbar-btn" on:click=on_link title="Insert link" aria-label="Insert link">
                    "\u{1F517}"
                </button>
                <span class="toolbar-divider" />
                <button class="toolbar-btn" on:click=on_clear title="Clear formatting" aria-label="Clear formatting">
                    "T\u{0338}"
                </button>
            </div>
            <div
                class="editor-content"
                contenteditable="true"
                node_ref=editor_ref
                on:input=on_input
                role="textbox"
                aria-multiline="true"
                aria-label="Email body"
                data-placeholder=placeholder
            />
        </div>
    }
}
