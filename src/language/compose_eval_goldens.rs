// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Golden-reference Docker Compose snippets for distillation.

pub fn compose_golden_for(prompt: &str) -> Option<&'static str> {
    match prompt.to_lowercase().as_str() {
        "basic nginx service" => Some(NGINX_ONLY),
        "postgres with web app stack" => Some(POSTGRES_WEB_STACK),
        _ => None,
    }
}

pub const COMPOSE_HARVEST_PROMPTS: &[&str] =
    &["basic nginx service", "postgres with web app stack"];

const NGINX_ONLY: &str = r#"services:
  nginx:
    image: nginx:latest
    ports:
    - 80:80
"#;

const POSTGRES_WEB_STACK: &str = r#"services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: example_password
  web:
    image: my-app:latest
    depends_on:
    - db
    ports:
    - 8080:8080
"#;
