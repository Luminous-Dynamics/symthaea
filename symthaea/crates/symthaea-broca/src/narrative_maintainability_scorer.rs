// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Narrative Maintainability Scorer (N-axis)
//! Uses syn to analyze Rust code for maintainability.

#[cfg(feature = "code-sheaf-eval")]
use syn::{
    visit::{self, Visit},
    File, ItemEnum, ItemFn, ItemStruct, ItemTrait,
};

#[cfg(feature = "code-sheaf-eval")]
struct MaintainabilityVisitor {
    function_count: usize,
    documented_count: usize,
    complex_nodes: usize,
    modularity: usize,
}

#[cfg(feature = "code-sheaf-eval")]
impl<'ast> Visit<'ast> for MaintainabilityVisitor {
    fn visit_item_fn(&mut self, i: &'ast ItemFn) {
        self.function_count += 1;
        if i.attrs.iter().any(|attr| attr.path().is_ident("doc")) {
            self.documented_count += 1;
        }
        if matches!(i.vis, syn::Visibility::Public(_)) {
            self.modularity += 1;
        }
        visit::visit_item_fn(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast ItemStruct) {
        if i.attrs.iter().any(|attr| attr.path().is_ident("doc")) {
            self.documented_count += 1;
        }
        if matches!(i.vis, syn::Visibility::Public(_)) {
            self.modularity += 1;
        }
        visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast ItemEnum) {
        if i.attrs.iter().any(|attr| attr.path().is_ident("doc")) {
            self.documented_count += 1;
        }
        if matches!(i.vis, syn::Visibility::Public(_)) {
            self.modularity += 1;
        }
        visit::visit_item_enum(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast ItemTrait) {
        if i.attrs.iter().any(|attr| attr.path().is_ident("doc")) {
            self.documented_count += 1;
        }
        if matches!(i.vis, syn::Visibility::Public(_)) {
            self.modularity += 1;
        }
        visit::visit_item_trait(self, i);
    }

    fn visit_expr_if(&mut self, i: &'ast syn::ExprIf) {
        self.complex_nodes += 1;
        visit::visit_expr_if(self, i);
    }

    fn visit_expr_loop(&mut self, i: &'ast syn::ExprLoop) {
        self.complex_nodes += 1;
        visit::visit_expr_loop(self, i);
    }

    fn visit_expr_match(&mut self, i: &'ast syn::ExprMatch) {
        self.complex_nodes += 1;
        visit::visit_expr_match(self, i);
    }
}

pub fn compute_narrative_maintainability(code: &str) -> f32 {
    #[cfg(feature = "code-sheaf-eval")]
    {
        if let Ok(syntax_tree) = syn::parse_file(code) {
            let mut visitor = MaintainabilityVisitor {
                function_count: 0,
                documented_count: 0,
                complex_nodes: 0,
                modularity: 0,
            };
            visitor.visit_file(&syntax_tree);

            let doc_ratio = if visitor.function_count > 0 {
                visitor.documented_count as f32 / visitor.function_count as f32
            } else {
                1.0
            };

            let complexity_norm = (visitor.complex_nodes as f32 * 0.05).min(1.0);
            let modularity_norm = (visitor.modularity as f32 * 0.1).min(1.0);

            return (0.40 * doc_ratio + 0.30 * (1.0 - complexity_norm) + 0.30 * modularity_norm)
                .clamp(0.0, 1.0);
        }
    }

    // Fallback for non-Rust, malformed code, or when feature is disabled
    let doc_lines = code
        .lines()
        .filter(|l| l.trim().starts_with("//") || l.trim().starts_with("#"))
        .count();
    let total_lines = code.lines().count();
    if total_lines > 0 {
        (doc_lines as f32 / total_lines as f32 + 0.3).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maintainability() {
        let good = "/// hello\npub fn main() {}";
        let bad = "fn main() { if a { if b { if c { d } } } }";
        assert!(compute_narrative_maintainability(good) > compute_narrative_maintainability(bad));
    }
}
