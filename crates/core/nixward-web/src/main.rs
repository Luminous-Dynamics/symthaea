use nixward_web::NixwardDashboard;

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount_to_body(|| leptos::view! { <NixwardDashboard /> });
}
