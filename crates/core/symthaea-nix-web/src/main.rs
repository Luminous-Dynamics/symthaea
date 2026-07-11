use symthaea_nix_web::NixMindDashboard;

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount_to_body(|| leptos::view! { <NixMindDashboard /> });
}
