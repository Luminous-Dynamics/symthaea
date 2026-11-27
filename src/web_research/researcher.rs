// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Web Research Orchestrator
//!
//! Coordinates the research pipeline: query planning, fetching,
//! extraction, verification, and integration. This is the main
//! entry point for autonomous web research.

use anyhow::{Context, Result};
use futures::{StreamExt, stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::future::Future;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::extractor::{ContentExtractor, ContentType, ExtractedContent};
use super::types::{
    AcquisitionOutcome, AcquisitionReceipt, AcquisitionStage, EpistemicStatus, ResearchSource,
    ResearchStatus, VerifiedClaim, WebResearchConfig, WebResearchReport, WebResearchResult,
};
use super::ssrf::{
    GuardedDnsResolver, is_dns_policy_rejection, is_dns_policy_rejection_anyhow,
    is_redirect_policy_rejection_anyhow, is_url_policy_rejection_anyhow,
    reject_redirect_response, validate_request_url,
};
use super::verifier::{EpistemicVerifier, SourceEvidence, VerificationContext};

const EXTRACTOR_VERSION: &str = "web-research-content-extractor-v1";

#[derive(Debug)]
struct SearchQueryAcquisition {
    query: String,
    urls: Vec<String>,
    receipts: Vec<AcquisitionReceipt>,
    had_failure: bool,
}

#[derive(Debug)]
struct SourcePlanningResult {
    candidates: Vec<PageCandidate>,
    receipts: Vec<AcquisitionReceipt>,
    acquisition_incomplete: bool,
}

#[derive(Debug, Clone)]
struct PageCandidate {
    query_rank: usize,
    source_rank: usize,
    query: String,
    url: String,
}

#[derive(Debug)]
struct PageAcquisition {
    candidate: PageCandidate,
    receipt: AcquisitionReceipt,
    extraction: Option<ExtractedContent>,
}

/// Research orchestrator for autonomous web research
#[derive(Debug)]
pub struct WebResearcher {
    /// Configuration
    config: WebResearchConfig,
    /// HTTP client for fetching pages
    client: reqwest::Client,
    /// Content extractor
    extractor: ContentExtractor,
    /// Epistemic verifier
    verifier: EpistemicVerifier,
}

impl WebResearcher {
    /// Create a new web researcher with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(WebResearchConfig::default())
    }

    /// Create a new web researcher with custom configuration
    pub fn with_config(config: WebResearchConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.request_timeout_ms))
            .user_agent(&config.user_agent)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .dns_resolver(GuardedDnsResolver::system())
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            config,
            client,
            extractor: ContentExtractor::new(),
            verifier: EpistemicVerifier::new(),
        })
    }

    /// Research and verify a query
    ///
    /// This is the main entry point for epistemic research.
    /// Returns a research result with verified claims.
    pub async fn research_and_verify(&self, query: &str) -> Result<WebResearchResult> {
        let report = self.research_and_verify_with_receipts(query).await?;
        Ok(report.result)
    }

    /// Research with a complete acquisition report.
    ///
    /// Search and page requests are bounded by one configured in-flight limit
    /// per phase. Completion order is discarded: results and receipts are
    /// emitted in planned query/source order. Retrieval failures remain
    /// visible and can never be mistaken for a successful zero-result search.
    pub async fn research_and_verify_with_receipts(
        &self,
        query: &str,
    ) -> Result<WebResearchReport> {
        let search_queries = self.plan_queries(query);
        let concurrency = self.config.max_concurrent_requests;
        let mut acquisition_receipts = Vec::new();
        let deadline = tokio::time::Instant::now()
            + Duration::from_millis(self.config.max_research_duration_ms);

        let query_budget = self.config.max_search_queries_per_research;
        let mut indexed_queries = Vec::new();
        for (query_rank, search_query) in search_queries.into_iter().enumerate() {
            if query_rank < query_budget {
                indexed_queries.push((query_rank, search_query));
            } else {
                let mut receipt = make_receipt(
                    query_rank,
                    None,
                    AcquisitionStage::SearchQuery,
                    &search_query,
                    AcquisitionOutcome::BudgetExceeded,
                );
                receipt.reason_code = Some("MAX_SEARCH_QUERIES_PER_RESEARCH".to_string());
                receipt.reason = Some(format!(
                    "planned query budget of {query_budget} was reached before this query ran"
                ));
                acquisition_receipts.push(receipt);
            }
        }

        if concurrency == 0 || self.config.max_research_duration_ms == 0 {
            let reason_code = if concurrency == 0 {
                "ZERO_CONCURRENCY_BUDGET"
            } else {
                "ZERO_RESEARCH_DURATION_BUDGET"
            };
            for (query_rank, search_query) in indexed_queries {
                let mut receipt = make_receipt(
                    query_rank,
                    None,
                    AcquisitionStage::SearchQuery,
                    &search_query,
                    AcquisitionOutcome::BudgetExceeded,
                );
                receipt.reason_code = Some(reason_code.to_string());
                receipt.reason = Some(
                    "acquisition budget prevented the planned query from starting".to_string(),
                );
                acquisition_receipts.push(receipt);
            }
            sort_acquisition_receipts(&mut acquisition_receipts);
            return Ok(WebResearchReport {
                result: WebResearchResult::with_status(ResearchStatus::RetrievalIncomplete),
                acquisition_receipts,
            });
        }

        if indexed_queries.is_empty() {
            return Ok(WebResearchReport {
                result: WebResearchResult::with_status(ResearchStatus::RetrievalIncomplete),
                acquisition_receipts,
            });
        }

        // Search queries are concurrent, but their results are restored to
        // the planned query order before page candidates are constructed.
        let mut search_results =
            bounded_map_ordered(indexed_queries, concurrency, |query_rank, search_query| {
                let deadline = deadline.clone();
                async move {
                    let attempt_started = Instant::now();
                    match tokio::time::timeout_at(
                        deadline,
                        self.fetch_search_results_with_receipts(&search_query, query_rank),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => {
                            let mut receipt = make_receipt(
                                query_rank,
                                None,
                                AcquisitionStage::SearchQuery,
                                &search_query,
                                AcquisitionOutcome::Timeout,
                            );
                            receipt.reason_code =
                                Some("TOTAL_RESEARCH_DEADLINE_EXCEEDED".to_string());
                            receipt.elapsed_ms = elapsed_ms(attempt_started);
                            receipt.reason = Some(
                                "the end-to-end acquisition deadline expired during query search"
                                    .to_string(),
                            );
                            SearchQueryAcquisition {
                                query: search_query,
                                urls: Vec::new(),
                                receipts: vec![receipt],
                                had_failure: true,
                            }
                        }
                    }
                }
            })
            .await;
        search_results.sort_by_key(|(query_rank, _)| *query_rank);

        let prior_acquisition_incomplete = acquisition_receipts
            .iter()
            .any(|receipt: &AcquisitionReceipt| receipt.outcome.is_incomplete());
        let source_plan =
            plan_page_candidates(search_results, self.config.max_sources_per_research);
        acquisition_receipts.extend(source_plan.receipts);
        let mut acquisition_incomplete =
            prior_acquisition_incomplete || source_plan.acquisition_incomplete;
        let source_candidates = source_plan.candidates;

        // Page retrieval is globally bounded for this phase. The mapper
        // restores query/source rank after unordered completion.
        let mut page_results = bounded_map_ordered(
            source_candidates
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>(),
            concurrency,
            |_ordinal, candidate| {
                let deadline = deadline.clone();
                let candidate_for_timeout = candidate.clone();
                async move {
                    let attempt_started = Instant::now();
                    match tokio::time::timeout_at(
                        deadline,
                        self.fetch_and_extract_with_receipt(candidate),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => {
                            let mut receipt = make_receipt(
                                candidate_for_timeout.query_rank,
                                Some(candidate_for_timeout.source_rank),
                                AcquisitionStage::PageFetch,
                                &candidate_for_timeout.query,
                                AcquisitionOutcome::Timeout,
                            );
                            receipt.requested_url = Some(candidate_for_timeout.url.clone());
                            receipt.reason_code =
                                Some("TOTAL_RESEARCH_DEADLINE_EXCEEDED".to_string());
                            receipt.elapsed_ms = elapsed_ms(attempt_started);
                            receipt.reason = Some(
                                "the end-to-end acquisition deadline expired during page fetch"
                                    .to_string(),
                            );
                            PageAcquisition {
                                candidate: candidate_for_timeout,
                                receipt,
                                extraction: None,
                            }
                        }
                    }
                }
            },
        )
        .await;
        page_results
            .sort_by_key(|(_, page)| (page.candidate.query_rank, page.candidate.source_rank));

        let mut all_sources: Vec<SourceEvidence> = Vec::new();
        let mut all_claims: Vec<String> = Vec::new();
        let mut primary_content = String::new();
        let mut primary_title = String::new();

        for (_, page) in page_results {
            acquisition_incomplete |= page.receipt.outcome.is_incomplete();
            acquisition_receipts.push(page.receipt.clone());

            let Some(extraction) = page.extraction else {
                continue;
            };

            let source_url = page
                .receipt
                .final_url
                .clone()
                .unwrap_or_else(|| page.candidate.url.clone());

            println!(
                "\x1b[32m    [🧠 EXTRACT] Harvested {} raw claims (Structural Quality: {:.2}) from {}\x1b[0m",
                extraction.claims.len(),
                extraction.quality,
                source_url
            );

            let source = SourceEvidence {
                url: source_url.clone(),
                domain: self.extract_domain(&source_url),
                source_type: self.classify_source(&source_url, extraction.metadata.content_type),
                relevant_text: extraction.summary.clone(),
                // Retrieval relevance is not claim-level entailment. Only a
                // qualified relation adjudicator may set supports/refutes.
                supports: None,
                similarity: self.calculate_similarity(&extraction.body_text, query),
                credibility: extraction.quality,
            };

            all_sources.push(source);
            all_claims.extend(extraction.claims);

            if primary_content.is_empty() && extraction.quality > 0.5 {
                primary_content = extraction.body_text;
                primary_title = extraction.title;
            }
        }

        if all_sources.is_empty() {
            let status = status_without_usable_sources(&acquisition_receipts);
            sort_acquisition_receipts(&mut acquisition_receipts);
            return Ok(WebResearchReport {
                result: WebResearchResult::with_status(status),
                acquisition_receipts,
            });
        }

        // Verify only claims from successfully captured and extracted pages.
        let mut verified_claims: Vec<VerifiedClaim> = Vec::new();
        for claim in all_claims.iter().take(10) {
            let claim_lower = claim.to_lowercase();
            if claim_lower.contains("@type")
                || claim_lower.contains("breadcrumb")
                || claim_lower.contains("itemlist")
                || claim_lower.contains(".push([")
                || claim_lower.contains("window.__")
                || claim_lower.contains("select citation style")
                || claim_lower.contains("table of contents")
            {
                continue;
            }

            let context = VerificationContext {
                sources: all_sources.clone(),
                query: query.to_string(),
                domain: self.detect_domain(query),
            };
            let verification = self.verifier.verify_claim(claim, &context);
            println!(
                "    \x1b[35m[⚖️ VERIFY] Status: {:?} (Confidence: {:.2}) -> \"{}\"\x1b[0m",
                verification.status, verification.confidence, claim
            );
            verified_claims.push(VerifiedClaim {
                text: verification.claim,
                status: verification.status,
                confidence: verification.confidence,
                supporting_sources: verification.supporting_sources,
                contradicting_sources: verification.contradicting_sources,
                hedge: verification.hedge,
            });
        }

        let overall_confidence = if verified_claims.is_empty() {
            0.3
        } else {
            verified_claims
                .iter()
                .map(|claim| claim.confidence)
                .sum::<f32>()
                / verified_claims.len() as f32
        };
        let epistemic_status = self.aggregate_epistemic_status(&verified_claims);
        let summary = self.generate_summary(&primary_content, query);

        let result = WebResearchResult {
            title: primary_title,
            url: all_sources
                .first()
                .map_or(String::new(), |source| source.url.clone()),
            content: primary_content,
            summary,
            source_type: all_sources
                .first()
                .map_or(ResearchSource::Web, |source| source.source_type),
            relevance: self.calculate_average_relevance(&all_sources),
            confidence: overall_confidence,
            epistemic_status,
            status: if acquisition_incomplete {
                ResearchStatus::PartialSuccess
            } else {
                ResearchStatus::Success
            },
            claims: verified_claims,
        };

        sort_acquisition_receipts(&mut acquisition_receipts);
        Ok(WebResearchReport {
            result,
            acquisition_receipts,
        })
    }

    /// Plan search queries for a user query
    fn plan_queries(&self, query: &str) -> Vec<String> {
        let mut queries = vec![query.to_string()];

        // Add domain-specific query variants
        let lower = query.to_lowercase();

        if lower.contains("rust") || lower.contains("programming") {
            queries.push(format!("{} site:rust-lang.org", query));
            queries.push(format!("{} documentation", query));
        }

        if lower.contains("nix") || lower.contains("nixos") {
            queries.push(format!("{} site:nixos.org", query));
        }

        if lower.contains("python") {
            queries.push(format!("{} site:docs.python.org", query));
        }

        if lower.contains("state space")
            || lower.contains("ssm")
            || lower.contains("liquid time")
            || lower.contains("mamba")
            || lower.contains("selective scan")
        {
            queries.push(format!("{} arxiv", query));
            queries.push(format!("{} research paper", query));
            queries.push(format!("{} openreview", query));
        }

        // Relational Manifold Intersection Vectoring
        if lower.contains("luminous dynamics")
            || lower.contains("tristan stoltz")
            || lower.contains("evolving resonant")
            || lower.contains("mycelix")
        {
            queries.push("Luminous Dynamics Tristan Stoltz".to_string());
            queries.push("luminous-nix natural language interface NixOS".to_string());
            queries.push("Evolving Resonant Co-creationism framework".to_string());
        }

        queries
    }

    /// Search one planned query and retain a receipt for each attempted provider.
    async fn fetch_search_results_with_receipts(
        &self,
        query: &str,
        query_rank: usize,
    ) -> SearchQueryAcquisition {
        let mut urls = Vec::new();
        let mut receipts = Vec::new();
        let query_lower = query.to_lowercase();
        let encoded_query = urlencoding::encode(query);

        // External search provider.
        if let Some(provider) = self.config.search_provider.as_deref() {
            let started = Instant::now();
            let requested_url = self.search_provider_endpoint();
            match self.search_with_provider(query).await {
                Ok(found) => {
                    let normalized_provider = provider.trim().to_lowercase();
                    let supported = matches!(
                        normalized_provider.as_str(),
                        "serper" | "brave" | "searxng" | "searx"
                    );
                    let missing_api_key =
                        matches!(normalized_provider.as_str(), "serper" | "brave")
                            && self
                                .config
                                .search_api_key
                                .as_deref()
                                .is_none_or(|key| key.trim().is_empty());
                    let outcome = if !supported || missing_api_key {
                        AcquisitionOutcome::ProviderFailed
                    } else if found.is_empty() {
                        AcquisitionOutcome::NoResults
                    } else {
                        AcquisitionOutcome::ResultsFound
                    };
                    let reason = if !supported {
                        Some(format!(
                            "unsupported configured search provider: {provider}"
                        ))
                    } else if missing_api_key {
                        Some(format!(
                            "configured provider {provider} requires an API key"
                        ))
                    } else {
                        None
                    };
                    let payload = serde_json::to_vec(&found).unwrap_or_default();
                    let mut receipt = make_receipt(
                        query_rank,
                        None,
                        AcquisitionStage::SearchProvider,
                        query,
                        outcome,
                    );
                    receipt.provider = Some(provider.to_string());
                    receipt.requested_url = requested_url;
                    receipt.captured_bytes = payload.len();
                    receipt.content_sha256 = Some(sha256_hex(&payload));
                    receipt.reason_code = if !supported {
                        Some("UNSUPPORTED_SEARCH_PROVIDER".to_string())
                    } else if missing_api_key {
                        Some("SEARCH_PROVIDER_API_KEY_MISSING".to_string())
                    } else {
                        None
                    };
                    receipt.reason = reason;
                    receipt.elapsed_ms = elapsed_ms(started);
                    if !supported {
                        receipt.outcome = AcquisitionOutcome::ProviderFailed;
                    }
                    urls.extend(found);
                    receipts.push(receipt);
                }
                Err(error) => {
                    let http_status = http_status_from_error(&error);
                    let url_rejected = is_url_policy_rejection_anyhow(&error);
                    let dns_rejected = is_dns_policy_rejection_anyhow(&error);
                    let redirect_rejected = is_redirect_policy_rejection_anyhow(&error);
                    let outcome = if http_status == Some(429) {
                        AcquisitionOutcome::RateLimited
                    } else if redirect_rejected {
                        AcquisitionOutcome::RedirectRejected
                    } else if url_rejected {
                        AcquisitionOutcome::UrlPolicyRejected
                    } else if dns_rejected {
                        AcquisitionOutcome::DnsPolicyRejected
                    } else {
                        AcquisitionOutcome::ProviderFailed
                    };
                    let mut receipt = make_receipt(
                        query_rank,
                        None,
                        AcquisitionStage::SearchProvider,
                        query,
                        outcome,
                    );
                    receipt.provider = Some(provider.to_string());
                    receipt.requested_url = requested_url;
                    receipt.http_status = http_status;
                    receipt.reason_code = Some(if redirect_rejected {
                        "REDIRECT_REJECTED".to_string()
                    } else if url_rejected {
                        "URL_POLICY_REJECTED".to_string()
                    } else if dns_rejected {
                        "DNS_POLICY_REJECTED".to_string()
                    } else if http_status == Some(429) {
                        "SEARCH_RATE_LIMITED".to_string()
                    } else {
                        "SEARCH_PROVIDER_FAILED".to_string()
                    });
                    receipt.reason = Some(format!("{error:#}"));
                    receipt.elapsed_ms = elapsed_ms(started);
                    receipts.push(receipt);
                }
            }
        }

        // arXiv API search.
        if self.config.enable_arxiv_api
            && (query_lower.contains("arxiv")
                || query_lower.contains("paper")
                || query_lower.contains("research")
                || query_lower.contains("state space")
                || query_lower.contains("ssm")
                || query_lower.contains("liquid time")
                || query_lower.contains("mamba")
                || query_lower.contains("selective scan"))
        {
            let started = Instant::now();
            match self.search_arxiv_api(query).await {
                Ok(found) => {
                    let payload = serde_json::to_vec(&found).unwrap_or_default();
                    let mut receipt = make_receipt(
                        query_rank,
                        None,
                        AcquisitionStage::ArxivApi,
                        query,
                        if found.is_empty() {
                            AcquisitionOutcome::NoResults
                        } else {
                            AcquisitionOutcome::ResultsFound
                        },
                    );
                    receipt.provider = Some("arxiv".to_string());
                    receipt.requested_url = Some(self.config.arxiv_endpoint.clone());
                    receipt.captured_bytes = payload.len();
                    receipt.content_sha256 = Some(sha256_hex(&payload));
                    receipt.elapsed_ms = elapsed_ms(started);
                    urls.extend(found);
                    receipts.push(receipt);
                }
                Err(error) => {
                    let http_status = http_status_from_error(&error);
                    let redirect_rejected = is_redirect_policy_rejection_anyhow(&error);
                    let url_rejected = is_url_policy_rejection_anyhow(&error);
                    let dns_rejected = is_dns_policy_rejection_anyhow(&error);
                    let outcome = if http_status == Some(429) {
                        AcquisitionOutcome::RateLimited
                    } else if redirect_rejected {
                        AcquisitionOutcome::RedirectRejected
                    } else if url_rejected {
                        AcquisitionOutcome::UrlPolicyRejected
                    } else if dns_rejected {
                        AcquisitionOutcome::DnsPolicyRejected
                    } else {
                        AcquisitionOutcome::ProviderFailed
                    };
                    let mut receipt = make_receipt(
                        query_rank,
                        None,
                        AcquisitionStage::ArxivApi,
                        query,
                        outcome,
                    );
                    receipt.provider = Some("arxiv".to_string());
                    receipt.requested_url = Some(self.config.arxiv_endpoint.clone());
                    receipt.http_status = http_status;
                    receipt.reason_code = Some(if redirect_rejected {
                        "REDIRECT_REJECTED".to_string()
                    } else if url_rejected {
                        "URL_POLICY_REJECTED".to_string()
                    } else if dns_rejected {
                        "DNS_POLICY_REJECTED".to_string()
                    } else {
                        "ARXIV_SEARCH_FAILED".to_string()
                    });
                    receipt.reason = Some(format!("{error:#}"));
                    receipt.elapsed_ms = elapsed_ms(started);
                    receipts.push(receipt);
                }
            }
        }

        // OpenReview API search.
        if self.config.enable_openreview_api
            && (query_lower.contains("openreview")
                || query_lower.contains("paper")
                || query_lower.contains("research")
                || query_lower.contains("state space")
                || query_lower.contains("ssm")
                || query_lower.contains("liquid time")
                || query_lower.contains("mamba")
                || query_lower.contains("selective scan"))
        {
            let started = Instant::now();
            match self.search_openreview_api(query).await {
                Ok(found) => {
                    let payload = serde_json::to_vec(&found).unwrap_or_default();
                    let mut receipt = make_receipt(
                        query_rank,
                        None,
                        AcquisitionStage::OpenReviewApi,
                        query,
                        if found.is_empty() {
                            AcquisitionOutcome::NoResults
                        } else {
                            AcquisitionOutcome::ResultsFound
                        },
                    );
                    receipt.provider = Some("openreview".to_string());
                    receipt.requested_url = Some(self.config.openreview_endpoint.clone());
                    receipt.captured_bytes = payload.len();
                    receipt.content_sha256 = Some(sha256_hex(&payload));
                    receipt.elapsed_ms = elapsed_ms(started);
                    urls.extend(found);
                    receipts.push(receipt);
                }
                Err(error) => {
                    let http_status = http_status_from_error(&error);
                    let redirect_rejected = is_redirect_policy_rejection_anyhow(&error);
                    let url_rejected = is_url_policy_rejection_anyhow(&error);
                    let dns_rejected = is_dns_policy_rejection_anyhow(&error);
                    let outcome = if http_status == Some(429) {
                        AcquisitionOutcome::RateLimited
                    } else if redirect_rejected {
                        AcquisitionOutcome::RedirectRejected
                    } else if url_rejected {
                        AcquisitionOutcome::UrlPolicyRejected
                    } else if dns_rejected {
                        AcquisitionOutcome::DnsPolicyRejected
                    } else {
                        AcquisitionOutcome::ProviderFailed
                    };
                    let mut receipt = make_receipt(
                        query_rank,
                        None,
                        AcquisitionStage::OpenReviewApi,
                        query,
                        outcome,
                    );
                    receipt.provider = Some("openreview".to_string());
                    receipt.requested_url = Some(self.config.openreview_endpoint.clone());
                    receipt.http_status = http_status;
                    receipt.reason_code = Some(if redirect_rejected {
                        "REDIRECT_REJECTED".to_string()
                    } else if url_rejected {
                        "URL_POLICY_REJECTED".to_string()
                    } else if dns_rejected {
                        "DNS_POLICY_REJECTED".to_string()
                    } else {
                        "OPENREVIEW_SEARCH_FAILED".to_string()
                    });
                    receipt.reason = Some(format!("{error:#}"));
                    receipt.elapsed_ms = elapsed_ms(started);
                    receipts.push(receipt);
                }
            }
        }

        // Domain-specific, deterministic fallback targets. These are explicitly
        // recorded as local heuristic offers, not remote search results.
        let mut seeded_urls = Vec::new();
        if query_lower.contains("rust") {
            seeded_urls.push("https://doc.rust-lang.org/book/".to_string());
            seeded_urls.push("https://docs.rs/".to_string());
            seeded_urls.push(format!(
                "https://github.com/search?q={}+language%3ARust",
                encoded_query
            ));
        }
        if query_lower.contains("nix") || query_lower.contains("nixos") {
            seeded_urls.push("https://nixos.org/manual/nixos/stable/".to_string());
            seeded_urls.push("https://nixos.wiki/".to_string());
            seeded_urls.push("https://search.nixos.org/packages".to_string());
        }
        if query_lower.contains("hdc") || query_lower.contains("hypervector") {
            seeded_urls
                .push("https://en.wikipedia.org/wiki/Hyperdimensional_computing".to_string());
            seeded_urls.push("https://arxiv.org/abs/2112.15425".to_string());
        }
        if query_lower.contains("python") {
            seeded_urls.push("https://docs.python.org/3/".to_string());
            seeded_urls.push("https://pypi.org/".to_string());
        }
        if query_lower.contains("science")
            || query_lower.contains("research")
            || query_lower.contains("paper")
        {
            seeded_urls.push("https://arxiv.org/search/".to_string());
            seeded_urls.push("https://scholar.google.com/".to_string());
        }
        if query_lower.contains("kernel") || query_lower.contains("linux") {
            seeded_urls.push("https://www.kernel.org/doc/html/latest/".to_string());
        }
        if query_lower.contains("state space")
            || query_lower.contains("ssm")
            || query_lower.contains("liquid time")
            || query_lower.contains("mamba")
            || query_lower.contains("selective scan")
        {
            seeded_urls.push(format!(
                "https://arxiv.org/search/?query={}&searchtype=all&source=header",
                encoded_query
            ));
            seeded_urls.push(format!(
                "https://openreview.net/search?term={}",
                encoded_query
            ));
        }
        if !seeded_urls.is_empty() {
            let payload = serde_json::to_vec(&seeded_urls).unwrap_or_default();
            let mut receipt = make_receipt(
                query_rank,
                None,
                AcquisitionStage::HeuristicSeeds,
                query,
                AcquisitionOutcome::ResultsFound,
            );
            receipt.provider = Some("domain-seeded-targets-v1".to_string());
            receipt.captured_bytes = payload.len();
            receipt.content_sha256 = Some(sha256_hex(&payload));
            receipts.push(receipt);
            urls.extend(seeded_urls);
        }

        // DuckDuckGo HTML fallback. Fixed provider URLs still pass the common URL gate.
        let ddg_url = format!("https://html.duckduckgo.com/html/?q={}", encoded_query);
        if let Err(error) = validate_request_url(&ddg_url) {
            let mut receipt = make_receipt(
                query_rank,
                None,
                AcquisitionStage::DuckDuckGo,
                query,
                AcquisitionOutcome::UrlPolicyRejected,
            );
            receipt.provider = Some("duckduckgo-html".to_string());
            receipt.requested_url = Some(ddg_url);
            receipt.reason_code = Some("URL_POLICY_REJECTED".to_string());
            receipt.reason = Some(error.to_string());
            receipts.push(receipt);
        } else {
            let started = Instant::now();
            match self.client.get(&ddg_url).send().await {
            Err(error) => {
                let dns_rejected = is_dns_policy_rejection(&error);
                let mut receipt = make_receipt(
                    query_rank,
                    None,
                    AcquisitionStage::DuckDuckGo,
                    query,
                    if error.is_timeout() {
                        AcquisitionOutcome::Timeout
                    } else if dns_rejected {
                        AcquisitionOutcome::DnsPolicyRejected
                    } else {
                        AcquisitionOutcome::ProviderFailed
                    },
                );
                receipt.provider = Some("duckduckgo-html".to_string());
                receipt.requested_url = Some(ddg_url);
                receipt.reason_code = Some(if error.is_timeout() {
                    "SEARCH_TIMEOUT".to_string()
                } else if dns_rejected {
                    "DNS_POLICY_REJECTED".to_string()
                } else {
                    "SEARCH_REQUEST_FAILED".to_string()
                });
                receipt.reason = Some(error.to_string());
                receipt.elapsed_ms = elapsed_ms(started);
                receipts.push(receipt);
            }
            Ok(mut response) => {
                let status = response.status();
                let final_url = response.url().to_string();
                let content_type = response
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string);
                if !status.is_success() {
                    let mut receipt = make_receipt(
                        query_rank,
                        None,
                        AcquisitionStage::DuckDuckGo,
                        query,
                        if status.as_u16() == 429 {
                            AcquisitionOutcome::RateLimited
                        } else if status.is_redirection() {
                            AcquisitionOutcome::RedirectRejected
                        } else {
                            AcquisitionOutcome::HttpError
                        },
                    );
                    receipt.provider = Some("duckduckgo-html".to_string());
                    receipt.requested_url = Some(ddg_url);
                    receipt.final_url = Some(final_url);
                    receipt.http_status = Some(status.as_u16());
                    receipt.content_type = content_type;
                    receipt.reason_code = Some(if status.as_u16() == 429 {
                        "SEARCH_RATE_LIMITED".to_string()
                    } else if status.is_redirection() {
                        "REDIRECT_REJECTED".to_string()
                    } else {
                        "SEARCH_HTTP_ERROR".to_string()
                    });
                    receipt.reason = Some(if status.is_redirection() {
                        format!("automatic redirects are disabled; refused HTTP {status}")
                    } else {
                        format!("DuckDuckGo returned HTTP {status}")
                    });
                    receipt.elapsed_ms = elapsed_ms(started);
                    receipts.push(receipt);
                } else {
                    let mut body = Vec::new();
                    let mut truncated = false;
                    let mut read_error = None;
                    loop {
                        match response.chunk().await {
                            Ok(Some(chunk)) => {
                                let remaining =
                                    self.config.max_content_length.saturating_sub(body.len());
                                let take = remaining.min(chunk.len());
                                body.extend_from_slice(&chunk[..take]);
                                if take < chunk.len() {
                                    truncated = true;
                                    break;
                                }
                            }
                            Ok(None) => break,
                            Err(error) => {
                                read_error = Some(error.to_string());
                                break;
                            }
                        }
                    }

                    if let Some(reason) = read_error {
                        let mut receipt = make_receipt(
                            query_rank,
                            None,
                            AcquisitionStage::DuckDuckGo,
                            query,
                            AcquisitionOutcome::ReadFailed,
                        );
                        receipt.provider = Some("duckduckgo-html".to_string());
                        receipt.requested_url = Some(ddg_url);
                        receipt.final_url = Some(final_url);
                        receipt.http_status = Some(status.as_u16());
                        receipt.content_type = content_type;
                        receipt.captured_bytes = body.len();
                        receipt.content_sha256 = Some(sha256_hex(&body));
                        receipt.reason_code = Some("SEARCH_BODY_READ_FAILED".to_string());
                        receipt.reason = Some(reason);
                        receipt.elapsed_ms = elapsed_ms(started);
                        receipts.push(receipt);
                    } else {
                        let html_text = String::from_utf8_lossy(&body).into_owned();
                        match scraper::Selector::parse("a.result__url") {
                            Err(error) => {
                                let mut receipt = make_receipt(
                                    query_rank,
                                    None,
                                    AcquisitionStage::DuckDuckGo,
                                    query,
                                    AcquisitionOutcome::ExtractionFailed,
                                );
                                receipt.provider = Some("duckduckgo-html".to_string());
                                receipt.requested_url = Some(ddg_url);
                                receipt.final_url = Some(final_url);
                                receipt.http_status = Some(status.as_u16());
                                receipt.content_type = content_type;
                                receipt.captured_bytes = body.len();
                                receipt.content_sha256 = Some(sha256_hex(&body));
                                receipt.reason_code =
                                    Some("SEARCH_RESULT_EXTRACTION_FAILED".to_string());
                                receipt.reason = Some(error.to_string());
                                receipt.elapsed_ms = elapsed_ms(started);
                                receipts.push(receipt);
                            }
                            Ok(selector) => {
                                let document = scraper::Html::parse_document(&html_text);
                                let mut found = Vec::new();
                                for element in document.select(&selector) {
                                    if let Some(href) = element.value().attr("href") {
                                        if !href.contains("duckduckgo.com")
                                            && href.starts_with("http")
                                        {
                                            found.push(href.to_string());
                                        }
                                    }
                                }
                                let mut receipt = make_receipt(
                                    query_rank,
                                    None,
                                    AcquisitionStage::DuckDuckGo,
                                    query,
                                    if truncated {
                                        AcquisitionOutcome::FetchedTruncated
                                    } else if found.is_empty() {
                                        AcquisitionOutcome::NoResults
                                    } else {
                                        AcquisitionOutcome::ResultsFound
                                    },
                                );
                                receipt.provider = Some("duckduckgo-html".to_string());
                                receipt.requested_url = Some(ddg_url);
                                receipt.final_url = Some(final_url);
                                receipt.http_status = Some(status.as_u16());
                                receipt.content_type = content_type;
                                receipt.captured_bytes = body.len();
                                receipt.content_sha256 = Some(sha256_hex(&body));
                                receipt.reason_code = if truncated {
                                    Some("SEARCH_RESPONSE_TRUNCATED_AT_BUDGET".to_string())
                                } else {
                                    None
                                };
                                receipt.reason = if truncated {
                                    Some(format!(
                                        "search HTML capture stopped at the {} byte budget",
                                        self.config.max_content_length
                                    ))
                                } else if found.is_empty() {
                                    Some(
                                        "search completed successfully with zero organic links"
                                            .to_string(),
                                    )
                                } else {
                                    None
                                };
                                receipt.elapsed_ms = elapsed_ms(started);
                                receipts.push(receipt);
                                urls.extend(found);
                            }
                        }
                    }
                }
            }
        }

        }

        // Preserve duplicate offers here so the global planner can emit one
        // DuplicateSuppressed receipt for each repeated URL in deterministic
        // query/provider order.
        let had_failure = receipts
            .iter()
            .any(|receipt| receipt.outcome.is_incomplete());

        SearchQueryAcquisition {
            query: query.to_string(),
            urls,
            receipts,
            had_failure,
        }
    }

    fn search_provider_endpoint(&self) -> Option<String> {
        let provider = self
            .config
            .search_provider
            .as_deref()?
            .trim()
            .to_lowercase();
        match provider.as_str() {
            "serper" => Some(
                self.config
                    .search_endpoint
                    .clone()
                    .unwrap_or_else(|| "https://google.serper.dev/search".to_string()),
            ),
            "brave" => {
                Some(self.config.search_endpoint.clone().unwrap_or_else(|| {
                    "https://api.search.brave.com/res/v1/web/search".to_string()
                }))
            }
            "searxng" | "searx" => Some(
                self.config
                    .search_endpoint
                    .clone()
                    .unwrap_or_else(|| "https://searx.be".to_string()),
            ),
            _ => None,
        }
    }

    async fn search_with_provider(&self, query: &str) -> Result<Vec<String>> {
        let provider = match self.config.search_provider.as_deref() {
            Some(provider) => provider.trim().to_lowercase(),
            None => return Ok(Vec::new()),
        };

        match provider.as_str() {
            "serper" => self.search_serper(query).await,
            "brave" => self.search_brave(query).await,
            "searxng" | "searx" => self.search_searxng(query).await,
            _ => Ok(Vec::new()),
        }
    }

    async fn search_serper(&self, query: &str) -> Result<Vec<String>> {
        let api_key = match self.config.search_api_key.as_deref() {
            Some(key) if !key.is_empty() => key,
            _ => return Ok(Vec::new()),
        };
        let endpoint = self
            .config
            .search_endpoint
            .as_deref()
            .unwrap_or("https://google.serper.dev/search");
        validate_request_url(endpoint)?;

        let body = serde_json::json!({
            "q": query,
            "num": 10
        });

        let response = self
            .client
            .post(endpoint)
            .header("X-API-KEY", api_key)
            .header("Accept", "application/json")
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .json(&body)
            .send()
            .await
            .context("Serper search request failed")?;
        let response = reject_redirect_response(response)?;
        let response = response
            .error_for_status()
            .context("Serper search returned a non-success HTTP status")?;

        let json: Value = response.json().await.context("Serper JSON parse failed")?;
        Ok(extract_links(&json, &["organic"], "link"))
    }

    async fn search_brave(&self, query: &str) -> Result<Vec<String>> {
        let api_key = match self.config.search_api_key.as_deref() {
            Some(key) if !key.is_empty() => key,
            _ => return Ok(Vec::new()),
        };
        let endpoint = self
            .config
            .search_endpoint
            .as_deref()
            .unwrap_or("https://api.search.brave.com/res/v1/web/search");
        let url = format!("{endpoint}?q={}", urlencoding::encode(query));
        validate_request_url(&url)?;

        let response = self
            .client
            .get(url)
            .header("X-Subscription-Token", api_key)
            .header("Accept", "application/json")
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .send()
            .await
            .context("Brave search request failed")?;
        let response = reject_redirect_response(response)?;
        let response = response
            .error_for_status()
            .context("Brave search returned a non-success HTTP status")?;

        let json: Value = response.json().await.context("Brave JSON parse failed")?;
        Ok(extract_links(&json, &["web", "results"], "url"))
    }

    async fn search_searxng(&self, query: &str) -> Result<Vec<String>> {
        let endpoint = self
            .config
            .search_endpoint
            .as_deref()
            .unwrap_or("https://searx.be");
        let endpoint = endpoint.trim_end_matches('/');
        let url = format!(
            "{}/search?q={}&format=json",
            endpoint,
            urlencoding::encode(query)
        );
        validate_request_url(&url)?;

        let response = self
            .client
            .get(url)
            .header("Accept", "application/json")
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .send()
            .await
            .context("SearXNG search request failed")?;
        let response = reject_redirect_response(response)?;
        let response = response
            .error_for_status()
            .context("SearXNG search returned a non-success HTTP status")?;

        let json: Value = response.json().await.context("SearXNG JSON parse failed")?;
        Ok(extract_links(&json, &["results"], "url"))
    }

    async fn search_arxiv_api(&self, query: &str) -> Result<Vec<String>> {
        // Distill query to remove token pollution from academic indexing
        let cleaned_query = query
            .to_lowercase()
            .replace("arxiv", "")
            .replace("papers", "")
            .replace("paper", "")
            .replace("search", "")
            .replace("latest", "")
            .replace("status", "")
            .replace("architecture", "")
            .trim()
            .to_string();

        // Target both titles and abstracts specifically for high-density accuracy
        let search_query = format!("ti:\"{0}\" OR abs:\"{0}\"", cleaned_query);
        let encoded_query = urlencoding::encode(&search_query);

        // Force chronological sorting to capture state-of-the-art results (2024-2026)
        let url = format!(
            "{}?search_query={}&id_list=&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending",
            self.config.arxiv_endpoint, encoded_query
        );
        validate_request_url(&url)?;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .context("arXiv API request failed")?;
        let response = reject_redirect_response(response)?;
        let response = response
            .error_for_status()
            .context("arXiv API returned a non-success HTTP status")?;

        let body = response.text().await.context("arXiv API read failed")?;
        Ok(extract_arxiv_ids(&body))
    }

    async fn search_openreview_api(&self, query: &str) -> Result<Vec<String>> {
        let endpoint = self.config.openreview_endpoint.trim_end_matches('/');
        let encoded_query = urlencoding::encode(query);
        let url = format!("{endpoint}?term={encoded_query}");
        validate_request_url(&url)?;

        let mut request = self.client.get(url).header("Accept", "application/json")
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
        if let Some(token) = self.config.openreview_token.as_deref() {
            request = request.header("Cookie", format!("openreview.accessToken={token}"));
        }

        let response = request
            .send()
            .await
            .context("OpenReview search failed")?;
        let response = reject_redirect_response(response)?;
        let response = response
            .error_for_status()
            .context("OpenReview search returned a non-success HTTP status")?;
        let json: Value = response
            .json()
            .await
            .context("OpenReview JSON parse failed")?;

        Ok(extract_openreview_ids(&json))
    }

    /// Fetch, capture and extract a page while preserving a complete attempt receipt.
    async fn fetch_and_extract_with_receipt(&self, candidate: PageCandidate) -> PageAcquisition {
        let started = Instant::now();
        if let Err(error) = validate_request_url(&candidate.url) {
            let mut receipt = make_receipt(
                candidate.query_rank,
                Some(candidate.source_rank),
                AcquisitionStage::PageFetch,
                &candidate.query,
                AcquisitionOutcome::UrlPolicyRejected,
            );
            receipt.requested_url = Some(candidate.url.clone());
            receipt.reason_code = Some("URL_POLICY_REJECTED".to_string());
            receipt.reason = Some(error.to_string());
            receipt.elapsed_ms = elapsed_ms(started);
            return PageAcquisition {
                candidate,
                receipt,
                extraction: None,
            };
        }
        let response = self.client.get(&candidate.url).send().await;
        let mut response = match response {
            Ok(response) => response,
            Err(error) => {
                let dns_rejected = is_dns_policy_rejection(&error);
                let outcome = if error.is_timeout() {
                    AcquisitionOutcome::Timeout
                } else if dns_rejected {
                    AcquisitionOutcome::DnsPolicyRejected
                } else {
                    AcquisitionOutcome::ReadFailed
                };
                let mut receipt = make_receipt(
                    candidate.query_rank,
                    Some(candidate.source_rank),
                    AcquisitionStage::PageFetch,
                    &candidate.query,
                    outcome,
                );
                receipt.requested_url = Some(candidate.url.clone());
                receipt.reason_code = Some(if error.is_timeout() {
                    "PAGE_FETCH_TIMEOUT".to_string()
                } else if dns_rejected {
                    "DNS_POLICY_REJECTED".to_string()
                } else {
                    "PAGE_FETCH_REQUEST_FAILED".to_string()
                });
                receipt.reason = Some(error.to_string());
                receipt.elapsed_ms = elapsed_ms(started);
                return PageAcquisition {
                    candidate,
                    receipt,
                    extraction: None,
                };
            }
        };

        let status = response.status();
        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);

        if !status.is_success() {
            let mut receipt = make_receipt(
                candidate.query_rank,
                Some(candidate.source_rank),
                AcquisitionStage::PageFetch,
                &candidate.query,
                if status.as_u16() == 429 {
                    AcquisitionOutcome::RateLimited
                } else if status.is_redirection() {
                    AcquisitionOutcome::RedirectRejected
                } else {
                    AcquisitionOutcome::HttpError
                },
            );
            receipt.requested_url = Some(candidate.url.clone());
            receipt.final_url = Some(final_url);
            receipt.http_status = Some(status.as_u16());
            receipt.content_type = content_type;
            receipt.reason_code = Some(if status.as_u16() == 429 {
                "PAGE_RATE_LIMITED".to_string()
            } else if status.is_redirection() {
                "REDIRECT_REJECTED".to_string()
            } else {
                "PAGE_HTTP_ERROR".to_string()
            });
            receipt.reason = Some(if status.is_redirection() {
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or("<missing or invalid Location>");
                format!("automatic redirects are disabled; refused next hop {location}")
            } else {
                format!("page fetch returned HTTP {status}")
            });
            receipt.elapsed_ms = elapsed_ms(started);
            return PageAcquisition {
                candidate,
                receipt,
                extraction: None,
            };
        }

        if let Some(content_type_value) = content_type.as_deref() {
            let lower = content_type_value.to_lowercase();
            if lower.contains("pdf") || lower.contains("octet-stream") || lower.contains("zip") {
                let mut receipt = make_receipt(
                    candidate.query_rank,
                    Some(candidate.source_rank),
                    AcquisitionStage::PageFetch,
                    &candidate.query,
                    AcquisitionOutcome::UnsupportedMedia,
                );
                receipt.requested_url = Some(candidate.url.clone());
                receipt.final_url = Some(final_url);
                receipt.http_status = Some(status.as_u16());
                receipt.content_type = content_type.clone();
                receipt.reason_code = Some("UNSUPPORTED_MEDIA_TYPE".to_string());
                receipt.reason = Some(format!(
                    "binary/non-text media is not admitted by the HTML extractor: {content_type_value}"
                ));
                receipt.elapsed_ms = elapsed_ms(started);
                return PageAcquisition {
                    candidate,
                    receipt,
                    extraction: None,
                };
            }
        }

        // Stream only up to the configured per-page byte budget. The total
        // captured HTML is bounded by max_sources_per_research × max_content_length.
        let mut body = Vec::new();
        let mut truncated = false;
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    let remaining = self.config.max_content_length.saturating_sub(body.len());
                    let take = remaining.min(chunk.len());
                    body.extend_from_slice(&chunk[..take]);
                    if take < chunk.len() {
                        truncated = true;
                        break;
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    let mut receipt = make_receipt(
                        candidate.query_rank,
                        Some(candidate.source_rank),
                        AcquisitionStage::PageFetch,
                        &candidate.query,
                        AcquisitionOutcome::ReadFailed,
                    );
                    receipt.requested_url = Some(candidate.url.clone());
                    receipt.final_url = Some(final_url);
                    receipt.http_status = Some(status.as_u16());
                    receipt.content_type = content_type;
                    receipt.captured_bytes = body.len();
                    receipt.content_sha256 = Some(sha256_hex(&body));
                    receipt.reason_code = Some("PAGE_BODY_READ_FAILED".to_string());
                    receipt.reason = Some(error.to_string());
                    receipt.elapsed_ms = elapsed_ms(started);
                    return PageAcquisition {
                        candidate,
                        receipt,
                        extraction: None,
                    };
                }
            }
        }

        let content_sha256 = sha256_hex(&body);
        if truncated {
            let mut receipt = make_receipt(
                candidate.query_rank,
                Some(candidate.source_rank),
                AcquisitionStage::PageFetch,
                &candidate.query,
                AcquisitionOutcome::FetchedTruncated,
            );
            receipt.requested_url = Some(candidate.url.clone());
            receipt.final_url = Some(final_url);
            receipt.http_status = Some(status.as_u16());
            receipt.content_type = content_type;
            receipt.captured_bytes = body.len();
            receipt.content_sha256 = Some(content_sha256);
            receipt.extractor_version = Some(EXTRACTOR_VERSION.to_string());
            receipt.reason_code = Some("PAGE_BODY_TRUNCATED_AT_BUDGET".to_string());
            receipt.reason = Some(format!(
                "captured the first {} bytes; truncated content is not passed to claim verification",
                self.config.max_content_length
            ));
            receipt.elapsed_ms = elapsed_ms(started);
            return PageAcquisition {
                candidate,
                receipt,
                extraction: None,
            };
        }

        let html = String::from_utf8_lossy(&body).into_owned();
        match self.extractor.extract(&html, &final_url) {
            Ok(extraction) if extraction.body_text.trim().is_empty() => {
                let mut receipt = make_receipt(
                    candidate.query_rank,
                    Some(candidate.source_rank),
                    AcquisitionStage::PageFetch,
                    &candidate.query,
                    AcquisitionOutcome::ExtractionFailed,
                );
                receipt.requested_url = Some(candidate.url.clone());
                receipt.final_url = Some(final_url);
                receipt.http_status = Some(status.as_u16());
                receipt.content_type = content_type;
                receipt.captured_bytes = body.len();
                receipt.content_sha256 = Some(content_sha256);
                receipt.extractor_version = Some(EXTRACTOR_VERSION.to_string());
                receipt.reason_code = Some("EMPTY_EXTRACTED_CONTENT".to_string());
                receipt.reason = Some(
                    "capture completed but the extractor produced no usable body text".to_string(),
                );
                receipt.elapsed_ms = elapsed_ms(started);
                PageAcquisition {
                    candidate,
                    receipt,
                    extraction: None,
                }
            }
            Ok(extraction) => {
                let mut receipt = make_receipt(
                    candidate.query_rank,
                    Some(candidate.source_rank),
                    AcquisitionStage::PageFetch,
                    &candidate.query,
                    AcquisitionOutcome::Fetched,
                );
                receipt.requested_url = Some(candidate.url.clone());
                receipt.final_url = Some(final_url);
                receipt.http_status = Some(status.as_u16());
                receipt.content_type = content_type;
                receipt.captured_bytes = body.len();
                receipt.content_sha256 = Some(content_sha256);
                receipt.extractor_version = Some(EXTRACTOR_VERSION.to_string());
                receipt.elapsed_ms = elapsed_ms(started);
                PageAcquisition {
                    candidate,
                    receipt,
                    extraction: Some(extraction),
                }
            }
            Err(error) => {
                let mut receipt = make_receipt(
                    candidate.query_rank,
                    Some(candidate.source_rank),
                    AcquisitionStage::PageFetch,
                    &candidate.query,
                    AcquisitionOutcome::ExtractionFailed,
                );
                receipt.requested_url = Some(candidate.url.clone());
                receipt.final_url = Some(final_url);
                receipt.http_status = Some(status.as_u16());
                receipt.content_type = content_type;
                receipt.captured_bytes = body.len();
                receipt.content_sha256 = Some(content_sha256);
                receipt.extractor_version = Some(EXTRACTOR_VERSION.to_string());
                receipt.reason_code = Some("HTML_EXTRACTION_FAILED".to_string());
                receipt.reason = Some(format!("{error:#}"));
                receipt.elapsed_ms = elapsed_ms(started);
                PageAcquisition {
                    candidate,
                    receipt,
                    extraction: None,
                }
            }
        }
    }

    /// Extract domain from URL
    fn extract_domain(&self, url: &str) -> String {
        url.split("://")
            .nth(1)
            .unwrap_or("")
            .split('/')
            .next()
            .unwrap_or("")
            .to_string()
    }

    /// Classify source type based on URL and content type
    fn classify_source(&self, url: &str, content_type: ContentType) -> ResearchSource {
        let domain = self.extract_domain(url).to_lowercase();

        if domain.contains("arxiv.org") || domain.contains("doi.org") || domain.contains("pubmed") {
            return ResearchSource::Academic;
        }

        if domain.contains("docs.")
            || domain.contains("doc.")
            || content_type == ContentType::Documentation
        {
            return ResearchSource::Documentation;
        }

        ResearchSource::Web
    }

    /// Calculate text similarity (simple word overlap)
    fn calculate_similarity(&self, text: &str, query: &str) -> f32 {
        let text_lower = text.to_lowercase();
        let text_words: std::collections::HashSet<&str> = text_lower.split_whitespace().collect();

        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        if query_words.is_empty() {
            return 0.0;
        }

        let matches = query_words
            .iter()
            .filter(|w| text_words.contains(*w))
            .count();

        matches as f32 / query_words.len() as f32
    }

    /// Detect the domain of a query
    fn detect_domain(&self, query: &str) -> String {
        let lower = query.to_lowercase();

        if lower.contains("programming")
            || lower.contains("code")
            || lower.contains("rust")
            || lower.contains("python")
        {
            return "programming".to_string();
        }

        if lower.contains("nixos") || lower.contains("nix") || lower.contains("linux") {
            return "systems".to_string();
        }

        if lower.contains("science") || lower.contains("research") || lower.contains("study") {
            return "science".to_string();
        }

        "general".to_string()
    }

    /// Calculate average relevance across sources
    fn calculate_average_relevance(&self, sources: &[SourceEvidence]) -> f32 {
        if sources.is_empty() {
            return 0.0;
        }
        sources.iter().map(|s| s.similarity).sum::<f32>() / sources.len() as f32
    }

    /// Aggregate epistemic status from verified claims
    fn aggregate_epistemic_status(&self, claims: &[VerifiedClaim]) -> EpistemicStatus {
        if claims.is_empty() {
            return EpistemicStatus::InsufficientEvidence;
        }

        let high_conf = claims
            .iter()
            .filter(|c| c.status == EpistemicStatus::HighConfidence)
            .count();
        let contradicted = claims
            .iter()
            .filter(|c| {
                c.status == EpistemicStatus::Contradicted || c.status == EpistemicStatus::False
            })
            .count();

        if contradicted > claims.len() / 3 {
            return EpistemicStatus::Contradicted;
        }

        if high_conf > claims.len() / 2 {
            return EpistemicStatus::HighConfidence;
        }

        let avg_confidence: f32 =
            claims.iter().map(|c| c.confidence).sum::<f32>() / claims.len() as f32;

        if avg_confidence > 0.7 {
            EpistemicStatus::ModerateConfidence
        } else if avg_confidence > 0.4 {
            EpistemicStatus::LowConfidence
        } else {
            EpistemicStatus::InsufficientEvidence
        }
    }

    /// Generate a summary from content
    fn generate_summary(&self, content: &str, query: &str) -> String {
        if content.is_empty() {
            return format!("No substantial content found for: {}", query);
        }

        // Take first 500 chars as summary
        let summary = if content.len() > 500 {
            format!("{}...", &content[..497])
        } else {
            content.to_string()
        };

        // Clean up
        summary.lines().take(5).collect::<Vec<_>>().join(" ")
    }

    /// Get a reference to the verifier for direct access
    pub fn verifier(&self) -> &EpistemicVerifier {
        &self.verifier
    }

    /// Get a mutable reference to the verifier
    pub fn verifier_mut(&mut self) -> &mut EpistemicVerifier {
        &mut self.verifier
    }
}

static NEXT_ACQUISITION_ATTEMPT_ID: Mutex<u64> = Mutex::new(1);

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn plan_page_candidates(
    search_results: Vec<(usize, SearchQueryAcquisition)>,
    max_sources: usize,
) -> SourcePlanningResult {
    let mut candidates = Vec::new();
    let mut receipts = Vec::new();
    let mut seen_urls = HashSet::new();
    let mut acquisition_incomplete = false;

    for (query_rank, search_result) in search_results {
        acquisition_incomplete |= search_result.had_failure;
        receipts.extend(search_result.receipts);

        for (source_rank, url) in search_result.urls.into_iter().enumerate() {
            if !seen_urls.insert(url.clone()) {
                let mut receipt = make_receipt(
                    query_rank,
                    Some(source_rank),
                    AcquisitionStage::PageFetch,
                    &search_result.query,
                    AcquisitionOutcome::DuplicateSuppressed,
                );
                receipt.requested_url = Some(url);
                receipt.reason_code = Some("DUPLICATE_SOURCE_URL".to_string());
                receipt.reason = Some(
                    "the same URL was already scheduled by an earlier provider or query"
                        .to_string(),
                );
                receipts.push(receipt);
                continue;
            }

            if candidates.len() >= max_sources {
                let mut receipt = make_receipt(
                    query_rank,
                    Some(source_rank),
                    AcquisitionStage::PageFetch,
                    &search_result.query,
                    AcquisitionOutcome::BudgetExceeded,
                );
                receipt.requested_url = Some(url);
                receipt.reason_code = Some("MAX_SOURCES_PER_RESEARCH".to_string());
                receipt.reason = Some(format!(
                    "source cap of {max_sources} was reached before this URL was fetched"
                ));
                receipts.push(receipt);
                acquisition_incomplete = true;
                continue;
            }

            candidates.push(PageCandidate {
                query_rank,
                source_rank,
                query: search_result.query.clone(),
                url,
            });
        }
    }

    SourcePlanningResult {
        candidates,
        receipts,
        acquisition_incomplete,
    }
}

fn http_status_from_error(error: &anyhow::Error) -> Option<u16> {
    error.chain().find_map(|cause| {
        cause
            .downcast_ref::<reqwest::Error>()
            .and_then(reqwest::Error::status)
            .map(|status| status.as_u16())
    })
}

fn acquisition_stage_order(stage: AcquisitionStage) -> u8 {
    match stage {
        AcquisitionStage::SearchQuery => 0,
        AcquisitionStage::SearchProvider => 1,
        AcquisitionStage::ArxivApi => 2,
        AcquisitionStage::OpenReviewApi => 3,
        AcquisitionStage::DuckDuckGo => 4,
        AcquisitionStage::HeuristicSeeds => 5,
        AcquisitionStage::PageFetch => 6,
    }
}

fn sort_acquisition_receipts(receipts: &mut [AcquisitionReceipt]) {
    receipts.sort_by_key(|receipt| {
        (
            receipt.query_rank,
            acquisition_stage_order(receipt.stage),
            receipt.source_rank.unwrap_or(usize::MAX),
        )
    });
}

fn status_without_usable_sources(receipts: &[AcquisitionReceipt]) -> ResearchStatus {
    let incomplete: Vec<AcquisitionOutcome> = receipts
        .iter()
        .map(|receipt| receipt.outcome)
        .filter(AcquisitionOutcome::is_incomplete)
        .collect();

    if incomplete.is_empty() {
        ResearchStatus::NoResults
    } else if incomplete
        .iter()
        .all(|outcome| *outcome == AcquisitionOutcome::RateLimited)
    {
        ResearchStatus::RateLimited
    } else {
        ResearchStatus::RetrievalIncomplete
    }
}

fn make_receipt(
    query_rank: usize,
    source_rank: Option<usize>,
    stage: AcquisitionStage,
    query: &str,
    outcome: AcquisitionOutcome,
) -> AcquisitionReceipt {
    let stage_id = match stage {
        AcquisitionStage::SearchQuery => "search-query",
        AcquisitionStage::SearchProvider => "search-provider",
        AcquisitionStage::ArxivApi => "arxiv",
        AcquisitionStage::OpenReviewApi => "openreview",
        AcquisitionStage::DuckDuckGo => "duckduckgo",
        AcquisitionStage::HeuristicSeeds => "heuristic-seeds",
        AcquisitionStage::PageFetch => "page-fetch",
    };
    let rank_id = source_rank
        .map(|rank| rank.to_string())
        .unwrap_or_else(|| "query".to_string());
    let captured_at_unix_ms = now_unix_ms();
    let process_id = std::process::id();
    let sequence = {
        let mut next = NEXT_ACQUISITION_ATTEMPT_ID
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let sequence = *next;
        *next = (*next).saturating_add(1);
        sequence
    };
    AcquisitionReceipt {
        attempt_id: format!(
            "a{captured_at_unix_ms}-p{process_id}-{sequence}-q{query_rank}-{stage_id}-{rank_id}"
        ),
        query_rank,
        source_rank,
        stage,
        provider: None,
        query: query.to_string(),
        requested_url: None,
        final_url: None,
        http_status: None,
        content_type: None,
        captured_bytes: 0,
        content_sha256: None,
        outcome,
        reason_code: None,
        reason: None,
        elapsed_ms: 0,
        captured_at_unix_ms,
        extractor_version: None,
    }
}

/// Run indexed futures with a hard in-flight limit, then restore input order.
async fn bounded_map_ordered<T, U, F, Fut>(
    items: Vec<(usize, T)>,
    concurrency: usize,
    map: F,
) -> Vec<(usize, U)>
where
    F: Fn(usize, T) -> Fut,
    Fut: Future<Output = U>,
{
    let limit = concurrency.max(1);
    let mut results = stream::iter(items)
        .map(|(rank, item)| {
            let future = map(rank, item);
            async move { (rank, future.await) }
        })
        .buffer_unordered(limit)
        .collect::<Vec<_>>()
        .await;
    results.sort_by_key(|(rank, _)| *rank);
    results
}

fn extract_links(json: &Value, path: &[&str], key: &str) -> Vec<String> {
    let mut cursor = json;
    for segment in path {
        if let Some(next) = cursor.get(*segment) {
            cursor = next;
        } else {
            return Vec::new();
        }
    }

    let mut urls = Vec::new();
    if let Some(entries) = cursor.as_array() {
        for entry in entries {
            if let Some(url) = entry.get(key).and_then(|v| v.as_str()) {
                urls.push(url.to_string());
            }
        }
    }

    urls
}

fn extract_arxiv_ids(feed: &str) -> Vec<String> {
    let mut urls = Vec::new();
    for chunk in feed.split("<entry>").skip(1) {
        if let Some(start) = chunk.find("<id>") {
            if let Some(end) = chunk[start + 4..].find("</id>") {
                let raw = &chunk[start + 4..start + 4 + end];
                let trimmed = raw.trim();
                if trimmed.contains("arxiv.org/abs/") {
                    let url = trimmed.replace("http://", "https://");
                    urls.push(url);
                }
            }
        }
    }
    urls
}

fn extract_openreview_ids(json: &Value) -> Vec<String> {
    let mut urls = Vec::new();

    let candidates = if let Some(notes) = json.get("notes").and_then(|v| v.as_array()) {
        notes
    } else if let Some(data) = json.get("data").and_then(|v| v.as_array()) {
        data
    } else if let Some(arr) = json.as_array() {
        arr
    } else {
        return urls;
    };

    for entry in candidates {
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .or_else(|| {
                entry
                    .get("note")
                    .and_then(|v| v.get("id"))
                    .and_then(|v| v.as_str())
            })
            .or_else(|| entry.get("forum").and_then(|v| v.as_str()));
        if let Some(id) = id {
            urls.push(format!("https://openreview.net/forum?id={id}"));
        }
    }

    urls
}

/// Error returned when WebResearcher creation fails
#[derive(Debug)]
pub struct WebResearcherCreationError(anyhow::Error);

impl std::fmt::Display for WebResearcherCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to create WebResearcher: {}", self.0)
    }
}

impl std::error::Error for WebResearcherCreationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

impl WebResearcher {
    /// Try to create a WebResearcher with default configuration.
    ///
    /// Returns `None` if the HTTP client cannot be created (e.g., TLS initialization failure).
    /// This is a safer alternative to `Default::default()` which can panic.
    pub fn try_default() -> Option<Self> {
        Self::new().ok()
    }
}

impl Default for WebResearcher {
    /// Creates a WebResearcher with default configuration.
    ///
    /// # Panics
    ///
    /// Panics if the HTTP client cannot be created. For a non-panicking alternative,
    /// use `WebResearcher::new()` or `WebResearcher::try_default()`.
    fn default() -> Self {
        Self::new()
            .expect("Failed to create default WebResearcher: HTTP client initialization failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_planning() {
        let researcher = WebResearcher::new().unwrap();

        let queries = researcher.plan_queries("rust memory safety");
        assert!(queries.len() >= 1);
        assert!(queries[0].contains("rust"));
    }

    #[test]
    fn test_domain_detection() {
        let researcher = WebResearcher::new().unwrap();

        assert_eq!(
            researcher.detect_domain("rust programming language"),
            "programming"
        );
        assert_eq!(researcher.detect_domain("nixos configuration"), "systems");
        assert_eq!(researcher.detect_domain("what is the weather"), "general");
    }

    #[test]
    fn test_similarity_calculation() {
        let researcher = WebResearcher::new().unwrap();

        let sim = researcher.calculate_similarity(
            "Rust is a systems programming language focused on safety",
            "rust programming safety",
        );
        assert!(sim > 0.5);

        let sim_low =
            researcher.calculate_similarity("The weather is nice today", "rust programming");
        assert!(sim_low < 0.5);
    }

    #[test]
    fn acquisition_outcomes_distinguish_zero_results_from_failures() {
        assert!(!AcquisitionOutcome::NoResults.is_incomplete());
        assert!(!AcquisitionOutcome::ResultsFound.is_incomplete());
        assert!(!AcquisitionOutcome::Fetched.is_incomplete());
        assert!(AcquisitionOutcome::Timeout.is_incomplete());
        assert!(AcquisitionOutcome::HttpError.is_incomplete());
        assert!(AcquisitionOutcome::FetchedTruncated.is_incomplete());
        assert!(AcquisitionOutcome::BudgetExceeded.is_incomplete());
    }

    #[test]
    fn default_source_budget_is_explicit_and_bounded() {
        let config = WebResearchConfig::default();
        assert_eq!(config.max_concurrent_requests, 5);
        assert_eq!(config.max_sources_per_research, 20);
        assert_eq!(config.max_search_queries_per_research, 8);
        assert_eq!(config.max_research_duration_ms, 60_000);
        assert_eq!(config.max_content_length, 100_000);
    }

    #[test]
    fn research_status_distinguishes_partial_and_incomplete_acquisition() {
        assert!(ResearchStatus::Success.is_success());
        assert!(ResearchStatus::PartialSuccess.is_success());
        assert!(!ResearchStatus::NoResults.is_success());
        assert!(!ResearchStatus::RetrievalIncomplete.is_success());
        assert!(ResearchStatus::PartialSuccess.is_retriable());
        assert!(ResearchStatus::RateLimited.is_retriable());
        assert!(ResearchStatus::RetrievalIncomplete.is_retriable());
        assert!(!ResearchStatus::NoResults.is_retriable());
    }

    #[tokio::test]
    async fn bounded_map_respects_limit_and_restores_planned_order() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let jobs = (0..12).map(|rank| (rank, rank)).collect::<Vec<_>>();
        let active_for_map = Arc::clone(&active);
        let peak_for_map = Arc::clone(&peak);

        let completed = bounded_map_ordered(jobs, 3, move |rank, value| {
            let active = Arc::clone(&active_for_map);
            let peak = Arc::clone(&peak_for_map);
            async move {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis((12 - rank) as u64)).await;
                active.fetch_sub(1, Ordering::SeqCst);
                (rank, value)
            }
        })
        .await;

        assert_eq!(
            completed,
            (0..12).map(|rank| (rank, (rank, rank))).collect::<Vec<_>>()
        );
        assert!(peak.load(Ordering::SeqCst) <= 3);
        assert!(peak.load(Ordering::SeqCst) > 1);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn no_source_status_distinguishes_empty_search_rate_limit_and_failure() {
        let no_results = make_receipt(
            0,
            None,
            AcquisitionStage::DuckDuckGo,
            "test query",
            AcquisitionOutcome::NoResults,
        );
        let rate_limited = make_receipt(
            0,
            None,
            AcquisitionStage::SearchProvider,
            "test query",
            AcquisitionOutcome::RateLimited,
        );
        let failed = make_receipt(
            0,
            None,
            AcquisitionStage::PageFetch,
            "test query",
            AcquisitionOutcome::Timeout,
        );

        assert_eq!(
            status_without_usable_sources(&[no_results.clone()]),
            ResearchStatus::NoResults
        );
        assert_eq!(
            status_without_usable_sources(&[rate_limited]),
            ResearchStatus::RateLimited
        );
        assert_eq!(
            status_without_usable_sources(&[no_results, failed]),
            ResearchStatus::RetrievalIncomplete
        );
    }

    #[test]
    fn source_planner_records_within_query_and_cross_query_duplicates() {
        let results = vec![
            (
                0,
                SearchQueryAcquisition {
                    query: "first query".to_string(),
                    urls: vec![
                        "https://source.example/a".to_string(),
                        "https://source.example/a".to_string(),
                        "https://source.example/b".to_string(),
                    ],
                    receipts: Vec::new(),
                    had_failure: false,
                },
            ),
            (
                1,
                SearchQueryAcquisition {
                    query: "second query".to_string(),
                    urls: vec![
                        "https://source.example/b".to_string(),
                        "https://source.example/c".to_string(),
                    ],
                    receipts: Vec::new(),
                    had_failure: false,
                },
            ),
        ];

        let plan = plan_page_candidates(results, 5);
        assert_eq!(
            plan.candidates
                .iter()
                .map(|candidate| candidate.url.as_str())
                .collect::<Vec<_>>(),
            vec![
                "https://source.example/a",
                "https://source.example/b",
                "https://source.example/c"
            ]
        );
        assert!(!plan.acquisition_incomplete);
        assert_eq!(
            plan.receipts
                .iter()
                .filter(|receipt| receipt.outcome == AcquisitionOutcome::DuplicateSuppressed)
                .count(),
            2
        );
        assert!(
            plan.receipts
                .iter()
                .all(|receipt| { receipt.outcome != AcquisitionOutcome::BudgetExceeded })
        );
        assert_eq!(plan.receipts[0].query_rank, 0);
        assert_eq!(plan.receipts[0].source_rank, Some(1));
        assert_eq!(plan.receipts[1].query_rank, 1);
        assert_eq!(plan.receipts[1].source_rank, Some(0));
    }

    #[test]
    fn source_planner_marks_global_cap_rejections_incomplete() {
        let results = vec![(
            0,
            SearchQueryAcquisition {
                query: "single query".to_string(),
                urls: vec![
                    "https://source.example/a".to_string(),
                    "https://source.example/b".to_string(),
                    "https://source.example/c".to_string(),
                ],
                receipts: Vec::new(),
                had_failure: false,
            },
        )];

        let plan = plan_page_candidates(results, 2);
        assert_eq!(plan.candidates.len(), 2);
        assert!(plan.acquisition_incomplete);
        assert_eq!(plan.receipts.len(), 1);
        assert_eq!(plan.receipts[0].outcome, AcquisitionOutcome::BudgetExceeded);
        assert_eq!(
            plan.receipts[0].reason_code.as_deref(),
            Some("MAX_SOURCES_PER_RESEARCH")
        );
    }

    #[test]
    fn acquisition_receipt_ids_bind_query_stage_and_source_rank() {
        let search = make_receipt(
            2,
            None,
            AcquisitionStage::DuckDuckGo,
            "example query",
            AcquisitionOutcome::NoResults,
        );
        let page = make_receipt(
            2,
            Some(4),
            AcquisitionStage::PageFetch,
            "example query",
            AcquisitionOutcome::Fetched,
        );

        assert!(search.attempt_id.contains("q2-duckduckgo-query"));
        assert!(page.attempt_id.contains("q2-page-fetch-4"));
        assert_ne!(search.attempt_id, page.attempt_id);
        assert!(search.captured_at_unix_ms > 0);
    }
}
