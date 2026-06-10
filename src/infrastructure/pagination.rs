// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! F1: Lazy Loading with Pagination
//!
//! Provides pagination infrastructure for browsing large datasets
//! like NixOS modules, packages, and options without loading everything.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::time::Instant;

/// Page request parameters
#[derive(Debug, Clone)]
pub struct PageRequest {
    /// Zero-based page index
    pub page: usize,
    /// Items per page
    pub page_size: usize,
    /// Optional filter query
    pub filter: Option<String>,
    /// Sort field
    pub sort_by: Option<String>,
    /// Sort direction
    pub ascending: bool,
}

impl Default for PageRequest {
    fn default() -> Self {
        Self {
            page: 0,
            page_size: 50,
            filter: None,
            sort_by: None,
            ascending: true,
        }
    }
}

impl PageRequest {
    pub fn new(page: usize, page_size: usize) -> Self {
        Self {
            page,
            page_size,
            ..Default::default()
        }
    }

    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    pub fn with_sort(mut self, field: impl Into<String>, ascending: bool) -> Self {
        self.sort_by = Some(field.into());
        self.ascending = ascending;
        self
    }

    /// Calculate offset for database/slice operations
    pub fn offset(&self) -> usize {
        self.page * self.page_size
    }
}

/// A page of results
#[derive(Debug, Clone)]
pub struct Page<T> {
    /// Items in this page
    pub items: Vec<T>,
    /// Current page index (0-based)
    pub page: usize,
    /// Items per page
    pub page_size: usize,
    /// Total number of items across all pages
    pub total_items: usize,
    /// Total number of pages
    pub total_pages: usize,
    /// Whether there's a next page
    pub has_next: bool,
    /// Whether there's a previous page
    pub has_prev: bool,
    /// Time taken to fetch this page
    pub fetch_time_ms: u64,
}

impl<T> Page<T> {
    /// Create an empty page
    pub fn empty() -> Self {
        Self {
            items: Vec::new(),
            page: 0,
            page_size: 0,
            total_items: 0,
            total_pages: 0,
            has_next: false,
            has_prev: false,
            fetch_time_ms: 0,
        }
    }

    /// Create a page from a full dataset (for smaller datasets)
    pub fn from_slice(data: &[T], request: &PageRequest) -> Self
    where
        T: Clone,
    {
        let start = Instant::now();
        let total_items = data.len();
        let total_pages = (total_items + request.page_size - 1) / request.page_size.max(1);

        let offset = request.offset();
        let items: Vec<T> = data
            .iter()
            .skip(offset)
            .take(request.page_size)
            .cloned()
            .collect();

        Self {
            items,
            page: request.page,
            page_size: request.page_size,
            total_items,
            total_pages,
            has_next: request.page + 1 < total_pages,
            has_prev: request.page > 0,
            fetch_time_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Map items to a different type
    pub fn map<U, F: FnMut(T) -> U>(self, f: F) -> Page<U> {
        Page {
            items: self.items.into_iter().map(f).collect(),
            page: self.page,
            page_size: self.page_size,
            total_items: self.total_items,
            total_pages: self.total_pages,
            has_next: self.has_next,
            has_prev: self.has_prev,
            fetch_time_ms: self.fetch_time_ms,
        }
    }
}

/// Paginator for lazy loading datasets
pub struct Paginator<K, T> {
    /// Cached pages
    page_cache: HashMap<K, CachedPage<T>>,
    /// Default page size
    default_page_size: usize,
    /// Maximum cached pages
    max_cached_pages: usize,
    /// Cache access order for LRU eviction
    access_order: VecDeque<K>,
}

#[derive(Clone)]
struct CachedPage<T> {
    page: Page<T>,
    #[allow(dead_code)] // RESERVED(future): cache TTL timestamp
    cached_at: Instant,
}

impl<K: Eq + Hash + Clone, T: Clone> Paginator<K, T> {
    /// Create a new paginator
    pub fn new(default_page_size: usize) -> Self {
        Self {
            page_cache: HashMap::new(),
            default_page_size,
            max_cached_pages: 10,
            access_order: VecDeque::new(),
        }
    }

    /// Set maximum cached pages
    pub fn with_max_cached(mut self, max: usize) -> Self {
        self.max_cached_pages = max;
        self
    }

    /// Get a cached page if available
    pub fn get_cached(&mut self, key: &K) -> Option<Page<T>> {
        if let Some(cached) = self.page_cache.get(key) {
            // Update access order
            self.access_order.retain(|k| k != key);
            self.access_order.push_back(key.clone());
            Some(cached.page.clone())
        } else {
            None
        }
    }

    /// Cache a page
    pub fn cache_page(&mut self, key: K, page: Page<T>) {
        // Evict if at capacity
        while self.page_cache.len() >= self.max_cached_pages {
            match self.access_order.front().cloned() {
                Some(old_key) => {
                    self.page_cache.remove(&old_key);
                    self.access_order.pop_front();
                }
                _ => {
                    break;
                }
            }
        }

        self.access_order.retain(|k| k != &key);
        self.access_order.push_back(key.clone());
        self.page_cache.insert(
            key,
            CachedPage {
                page,
                cached_at: Instant::now(),
            },
        );
    }

    /// Invalidate all cached pages
    pub fn invalidate_all(&mut self) {
        self.page_cache.clear();
        self.access_order.clear();
    }

    /// Invalidate a specific cached page
    pub fn invalidate(&mut self, key: &K) {
        self.page_cache.remove(key);
        self.access_order.retain(|k| k != key);
    }

    /// Get default page size
    pub fn default_page_size(&self) -> usize {
        self.default_page_size
    }
}

/// Lazy iterator that fetches pages on demand
pub struct LazyPageIterator<T, F> {
    fetcher: F,
    current_page: usize,
    page_size: usize,
    total_pages: Option<usize>,
    buffer: Vec<T>,
    buffer_index: usize,
    exhausted: bool,
}

impl<T, F> LazyPageIterator<T, F>
where
    F: FnMut(PageRequest) -> Option<Page<T>>,
{
    /// Create a new lazy iterator
    pub fn new(fetcher: F, page_size: usize) -> Self {
        Self {
            fetcher,
            current_page: 0,
            page_size,
            total_pages: None,
            buffer: Vec::new(),
            buffer_index: 0,
            exhausted: false,
        }
    }

    /// Fetch next page into buffer
    fn fetch_next_page(&mut self) -> bool {
        if self.exhausted {
            return false;
        }

        let request = PageRequest::new(self.current_page, self.page_size);
        match (self.fetcher)(request) {
            Some(page) => {
                self.total_pages = Some(page.total_pages);
                self.buffer = page.items;
                self.buffer_index = 0;
                self.current_page += 1;

                if self.buffer.is_empty() || !page.has_next {
                    self.exhausted = self.buffer.is_empty();
                }

                !self.buffer.is_empty()
            }
            _ => {
                self.exhausted = true;
                false
            }
        }
    }
}

impl<T, F> Iterator for LazyPageIterator<T, F>
where
    F: FnMut(PageRequest) -> Option<Page<T>>,
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // If buffer is exhausted, fetch next page
        if self.buffer_index >= self.buffer.len() && !self.fetch_next_page() {
            return None;
        }

        if self.buffer_index < self.buffer.len() {
            let item = self.buffer[self.buffer_index].clone();
            self.buffer_index += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(total_pages) = self.total_pages {
            let remaining_pages = total_pages.saturating_sub(self.current_page);
            let remaining_in_buffer = self.buffer.len().saturating_sub(self.buffer_index);
            let estimated = remaining_in_buffer + remaining_pages * self.page_size;
            (0, Some(estimated))
        } else {
            (0, None)
        }
    }
}

/// Module browser with lazy loading support
pub struct ModuleBrowser {
    /// Paginator for module entries
    paginator: Paginator<String, ModuleEntry>,
    /// Total module count (if known)
    total_modules: Option<usize>,
    /// Current search filter
    current_filter: Option<String>,
}

/// A NixOS module entry for browsing
#[derive(Debug, Clone)]
pub struct ModuleEntry {
    /// Module path (e.g., "services.nginx")
    pub path: String,
    /// Short description
    pub description: String,
    /// Module type (service, program, etc.)
    pub module_type: ModuleType,
    /// Whether it's enabled in current config
    pub is_enabled: bool,
    /// Option count in this module
    pub option_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleType {
    Service,
    Program,
    Hardware,
    Networking,
    Security,
    System,
    User,
    Other,
}

impl ModuleBrowser {
    /// Create a new module browser
    pub fn new() -> Self {
        Self {
            paginator: Paginator::new(50),
            total_modules: None,
            current_filter: None,
        }
    }

    /// Set filter and invalidate cache
    pub fn set_filter(&mut self, filter: Option<String>) {
        if self.current_filter != filter {
            self.current_filter = filter;
            self.paginator.invalidate_all();
        }
    }

    /// Get current filter
    pub fn filter(&self) -> Option<&str> {
        self.current_filter.as_deref()
    }

    /// Request a page of modules
    pub fn request_page(&mut self, page: usize) -> PageRequest {
        let mut request = PageRequest::new(page, self.paginator.default_page_size());
        if let Some(filter) = &self.current_filter {
            request = request.with_filter(filter.clone());
        }
        request
    }

    /// Get cached page or None
    pub fn get_cached_page(&mut self, page: usize) -> Option<Page<ModuleEntry>> {
        let key = self.cache_key(page);
        self.paginator.get_cached(&key)
    }

    /// Cache a fetched page
    pub fn cache_page(&mut self, page: usize, data: Page<ModuleEntry>) {
        self.total_modules = Some(data.total_items);
        let key = self.cache_key(page);
        self.paginator.cache_page(key, data);
    }

    fn cache_key(&self, page: usize) -> String {
        format!("{}:{}", self.current_filter.as_deref().unwrap_or(""), page)
    }

    /// Get total modules (if known from a previous fetch)
    pub fn total_modules(&self) -> Option<usize> {
        self.total_modules
    }
}

impl Default for ModuleBrowser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_request() {
        let req = PageRequest::new(2, 25);
        assert_eq!(req.offset(), 50);
        assert_eq!(req.page_size, 25);
    }

    #[test]
    fn test_page_from_slice() {
        let data: Vec<i32> = (0..100).collect();
        let req = PageRequest::new(1, 20);
        let page = Page::from_slice(&data, &req);

        assert_eq!(page.items.len(), 20);
        assert_eq!(page.items[0], 20);
        assert_eq!(page.total_items, 100);
        assert_eq!(page.total_pages, 5);
        assert!(page.has_next);
        assert!(page.has_prev);
    }

    #[test]
    fn test_page_first() {
        let data: Vec<i32> = (0..100).collect();
        let req = PageRequest::new(0, 20);
        let page = Page::from_slice(&data, &req);

        assert!(!page.has_prev);
        assert!(page.has_next);
    }

    #[test]
    fn test_page_last() {
        let data: Vec<i32> = (0..100).collect();
        let req = PageRequest::new(4, 20);
        let page = Page::from_slice(&data, &req);

        assert!(page.has_prev);
        assert!(!page.has_next);
    }

    #[test]
    fn test_paginator_caching() {
        let mut paginator: Paginator<String, i32> = Paginator::new(10);

        let page = Page {
            items: vec![1, 2, 3],
            page: 0,
            page_size: 10,
            total_items: 3,
            total_pages: 1,
            has_next: false,
            has_prev: false,
            fetch_time_ms: 0,
        };

        paginator.cache_page("test".to_string(), page);
        let cached = paginator.get_cached(&"test".to_string());
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().items, vec![1, 2, 3]);
    }

    #[test]
    fn test_lazy_iterator() {
        let data: Vec<i32> = (0..100).collect();

        let mut iter = LazyPageIterator::new(|req| Some(Page::from_slice(&data, &req)), 10);

        let collected: Vec<i32> = iter.by_ref().take(25).collect();
        assert_eq!(collected.len(), 25);
        assert_eq!(collected[0], 0);
        assert_eq!(collected[24], 24);
    }

    #[test]
    fn test_module_browser() {
        let mut browser = ModuleBrowser::new();

        browser.set_filter(Some("nginx".to_string()));
        assert_eq!(browser.filter(), Some("nginx"));

        let req = browser.request_page(0);
        assert_eq!(req.filter, Some("nginx".to_string()));
    }
}
