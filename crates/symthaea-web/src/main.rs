use leptos::prelude::*;

mod app;
mod components;
mod pages;
mod state;
mod worker;

fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).expect("logger");

    mount_to_body(app::App);
}
