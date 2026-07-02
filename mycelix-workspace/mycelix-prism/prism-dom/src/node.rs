// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DOM node types.

use std::collections::HashMap;

use html5ever::QualName;

/// Opaque handle into the DOM tree's node arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) usize);

/// Data specific to an HTML element node.
#[derive(Debug, Clone)]
pub struct ElementData {
    /// Tag name, lowercased (e.g. "div", "p", "a").
    pub tag: String,
    /// Qualified name (namespace + local) — kept for html5ever TreeSink compatibility.
    pub qual_name: QualName,
    /// Attribute map (e.g. "class" → "main", "href" → "https://...").
    pub attributes: HashMap<String, String>,
}

impl ElementData {
    /// Returns the value of the `id` attribute, if present.
    pub fn id(&self) -> Option<&str> {
        self.attributes.get("id").map(|s| s.as_str())
    }

    /// Returns the value of the `class` attribute, if present.
    pub fn classes(&self) -> Vec<&str> {
        self.attributes
            .get("class")
            .map(|c| c.split_whitespace().collect())
            .unwrap_or_default()
    }
}

/// The payload of a DOM node.
#[derive(Debug, Clone)]
pub enum NodeData {
    /// The document root.
    Document,
    /// An HTML element (`<div>`, `<p>`, `<a>`, etc.).
    Element(ElementData),
    /// A text node (visible content).
    Text(String),
    /// A comment node (`<!-- ... -->`).
    Comment(String),
    /// A processing instruction.
    ProcessingInstruction { target: String, data: String },
    /// DOCTYPE declaration.
    Doctype {
        name: String,
        public_id: String,
        system_id: String,
    },
}

/// A node in the DOM tree. Nodes live in an arena (`DomTree`); relationships
/// are expressed via `NodeId` indices.
#[derive(Debug, Clone)]
pub struct Node {
    /// What kind of node this is.
    pub data: NodeData,
    /// Parent node, if any (the document root has `None`).
    pub parent: Option<NodeId>,
    /// Ordered list of child node IDs.
    pub children: Vec<NodeId>,
}

impl Node {
    pub fn new(data: NodeData) -> Self {
        Self {
            data,
            parent: None,
            children: Vec::new(),
        }
    }

    /// Returns `true` if this is a text node.
    pub fn is_text(&self) -> bool {
        matches!(self.data, NodeData::Text(_))
    }

    /// Returns `true` if this is an element node.
    pub fn is_element(&self) -> bool {
        matches!(self.data, NodeData::Element(_))
    }

    /// Returns the element data if this is an element node.
    pub fn as_element(&self) -> Option<&ElementData> {
        match &self.data {
            NodeData::Element(e) => Some(e),
            _ => None,
        }
    }

    /// Returns the text content if this is a text node.
    pub fn as_text(&self) -> Option<&str> {
        match &self.data {
            NodeData::Text(t) => Some(t),
            _ => None,
        }
    }
}
