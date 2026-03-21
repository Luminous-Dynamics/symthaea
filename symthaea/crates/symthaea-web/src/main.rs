use leptos::prelude::*;

mod app;
mod components;
mod pages;
mod service_worker;
mod state;
mod worker;

fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).expect("logger");

    service_worker::register();

    mount_to_body(app::App);
}
