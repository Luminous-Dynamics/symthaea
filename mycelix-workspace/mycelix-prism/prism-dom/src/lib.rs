// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DOM tree with HTML parsing for Prism.
//!
//! Parses HTML via `html5ever` into a tree of `Node` structs that own their
//! content. Provides text extraction for the reflex arc immune scanner and
//! structural queries for layout.

pub mod node;
pub mod parser;
pub mod tree;

pub use node::{ElementData, Node, NodeData, NodeId};
pub use parser::parse_html;
pub use tree::DomTree;
