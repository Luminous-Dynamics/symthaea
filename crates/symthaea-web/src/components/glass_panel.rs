use leptos::prelude::*;

/// Reusable glassmorphism container with an optional title.
#[component]
pub fn GlassPanel(
    #[prop(optional)] title: Option<&'static str>,
    children: Children,
) -> impl IntoView {
    view! {
        <div class="glass-panel">
            {title.map(|t| view! { <div class="panel-title">{t}</div> })}
            {children()}
        </div>
    }
}
