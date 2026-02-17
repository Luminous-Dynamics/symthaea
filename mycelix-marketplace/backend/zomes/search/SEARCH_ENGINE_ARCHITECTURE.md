# 🔍 Advanced Search Engine - Architecture

**Purpose**: Revolutionary full-text search across all marketplace data with intelligent ranking and natural language understanding.

---

## 🎯 Vision

The Mycelix-Marketplace search engine goes beyond simple keyword matching to provide:
- **Natural Language Understanding**: "Show me electronics under $50 from trusted sellers"
- **Multi-Entity Search**: Search across listings, transactions, conversations, reviews
- **Intelligent Ranking**: MATL-weighted results, relevance scoring, recency bias
- **Advanced Filters**: Price ranges, categories, trust scores, location, time
- **Real-Time Indexing**: Instant search updates as new listings are created

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Search Coordinator                    │
│  (Business Logic - Orchestrates searches)               │
└─────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │  Index  │    │  Query  │    │  Rank   │
    │ Builder │    │ Parser  │    │ Engine  │
    └─────────┘    └─────────┘    └─────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │       Search Index            │
         │  (Inverted index + metadata)  │
         └───────────────────────────────┘
```

### Key Components

1. **Index Builder**
   - Builds inverted index from all marketplace data
   - Tokenizes text (title, description, tags)
   - Stores term frequencies (TF-IDF)
   - Updates incrementally on new entries

2. **Query Parser**
   - Parses natural language queries
   - Extracts filters (price, category, etc.)
   - Handles operators (AND, OR, NOT, quotes)
   - Synonym expansion

3. **Rank Engine**
   - Scores results by relevance (TF-IDF)
   - Boosts by MATL score (trust weighting)
   - Applies recency decay
   - Personalizes based on user history

4. **Search Coordinator**
   - Orchestrates search workflow
   - Manages multiple indices
   - Caches frequent queries
   - Rate limits searches

---

## 📊 Data Model

### SearchIndex Entry

```rust
pub struct SearchIndex {
    /// Unique index ID
    pub index_id: String,

    /// Entity type being indexed
    pub entity_type: EntityType,  // Listing, Transaction, Conversation, etc.

    /// Original entry hash
    pub entity_hash: ActionHash,

    /// Searchable fields
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub category: String,

    /// Metadata for ranking
    pub creator: AgentPubKey,
    pub creator_matl_score: f64,
    pub created_at: u64,
    pub updated_at: u64,
    pub view_count: u32,
    pub engagement_score: f64,

    /// Filters
    pub price_cents: Option<u64>,
    pub location: Option<String>,
    pub status: String,

    /// Inverted index data (stored separately)
    pub term_frequencies: Vec<(String, u32)>,  // (term, frequency)
}

pub enum EntityType {
    Listing,
    Transaction,
    Conversation,
    Review,
    Agent,
}
```

### SearchQuery

```rust
pub struct SearchQuery {
    /// Raw query string
    pub query: String,

    /// Parsed query terms
    pub terms: Vec<String>,
    pub required_terms: Vec<String>,  // Must match
    pub excluded_terms: Vec<String>,  // Must not match
    pub phrase_terms: Vec<String>,    // Exact phrase match

    /// Filters
    pub filters: SearchFilters,

    /// Pagination
    pub offset: u32,
    pub limit: u32,

    /// Sorting
    pub sort_by: SortOption,  // Relevance, Price, Date, Trust
    pub sort_order: SortOrder,  // Asc, Desc
}

pub struct SearchFilters {
    pub entity_types: Vec<EntityType>,
    pub price_min: Option<u64>,
    pub price_max: Option<u64>,
    pub categories: Vec<String>,
    pub min_matl_score: Option<f64>,
    pub created_after: Option<u64>,
    pub created_before: Option<u64>,
    pub location: Option<String>,
    pub status: Option<String>,
}
```

### SearchResult

```rust
pub struct SearchResult {
    /// Matched entity
    pub entity_hash: ActionHash,
    pub entity_type: EntityType,

    /// Score breakdown
    pub total_score: f64,
    pub relevance_score: f64,  // TF-IDF
    pub trust_score: f64,      // MATL contribution
    pub recency_score: f64,    // Time decay
    pub engagement_score: f64, // Views, favorites

    /// Matched terms (for highlighting)
    pub matched_terms: Vec<String>,
    pub matched_fields: Vec<String>,  // title, description, etc.

    /// Preview data
    pub preview_title: String,
    pub preview_snippet: String,  // Highlighted excerpt

    /// Metadata
    pub creator: AgentPubKey,
    pub created_at: u64,
}

pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total_count: u32,
    pub query_time_ms: u32,
    pub suggested_queries: Vec<String>,  // "Did you mean...?"
}
```

---

## 🔍 Search Algorithms

### 1. Inverted Index Construction

**Algorithm**:
```
For each entity (listing, transaction, etc.):
  1. Tokenize text fields (title, description, tags)
  2. Remove stop words ("the", "and", "or", etc.)
  3. Stem terms ("running" → "run")
  4. Calculate term frequency (TF)
  5. Store term → entity mappings

Global stats:
  - Total documents (N)
  - Document frequency per term (DF)
  - Calculate IDF = log(N / DF)
```

**Data Structure**:
```
InvertedIndex {
  "laptop": [
    (listing_hash_1, TF=3),
    (listing_hash_2, TF=1),
    ...
  ],
  "macbook": [
    (listing_hash_3, TF=2),
    ...
  ],
  ...
}
```

### 2. TF-IDF Ranking

**Formula**:
```
TF-IDF(term, doc) = TF(term, doc) * IDF(term)

where:
  TF(term, doc) = (frequency of term in doc) / (total terms in doc)
  IDF(term) = log(total documents / documents containing term)
```

**Example**:
```
Query: "macbook laptop"
Document: "MacBook Pro 16-inch laptop for sale"

TF-IDF("macbook") = (1/7) * log(1000/50) = 0.143 * 3.0 = 0.429
TF-IDF("laptop") = (1/7) * log(1000/200) = 0.143 * 1.7 = 0.243

Total score = 0.429 + 0.243 = 0.672
```

### 3. MATL-Weighted Ranking

**Formula**:
```
Final Score = (0.6 * TF-IDF) + (0.3 * MATL) + (0.1 * Recency)

where:
  TF-IDF ∈ [0, 1] (normalized)
  MATL ∈ [0, 1] (seller's composite MATL score)
  Recency = e^(-days_old / 30) ∈ [0, 1]
```

**Rationale**:
- 60% relevance (content match)
- 30% trust (seller reputation)
- 10% freshness (recent listings)

**Example**:
```
Listing A:
  TF-IDF = 0.85
  MATL = 0.92 (highly trusted seller)
  Recency = 0.95 (3 days old)
  Final = 0.6*0.85 + 0.3*0.92 + 0.1*0.95 = 0.881

Listing B:
  TF-IDF = 0.90
  MATL = 0.35 (new seller)
  Recency = 0.20 (60 days old)
  Final = 0.6*0.90 + 0.3*0.35 + 0.1*0.20 = 0.665

Result: Listing A ranks higher despite lower TF-IDF!
```

### 4. Query Parsing

**Natural Language Examples**:
```
"laptop under $500"
→ terms: ["laptop"]
→ filters: { price_max: 50000 }

"macbook from trusted sellers"
→ terms: ["macbook"]
→ filters: { min_matl_score: 0.7 }

"electronics in texas created this week"
→ terms: []
→ filters: { category: "Electronics", location: "Texas", created_after: <7_days_ago> }
```

**Operator Parsing**:
```
"laptop AND macbook"    → Both terms required
"laptop OR macbook"     → Either term matches
"-refurbished"          → Exclude this term
"\"macbook pro\""       → Exact phrase match
```

---

## 🚀 API Endpoints

### 1. Advanced Search

```rust
#[hdk_extern]
pub fn search(query: SearchQuery) -> ExternResult<SearchResponse>
```

**Input**:
```javascript
{
  query: "macbook laptop under $1000",
  filters: {
    entity_types: ["Listing"],
    price_max: 100000,
    min_matl_score: 0.5
  },
  offset: 0,
  limit: 20,
  sort_by: "Relevance"
}
```

**Output**:
```javascript
{
  results: [
    {
      entity_hash: "uhCkk...",
      entity_type: "Listing",
      total_score: 0.881,
      relevance_score: 0.85,
      trust_score: 0.92,
      recency_score: 0.95,
      matched_terms: ["macbook", "laptop"],
      preview_title: "MacBook Pro 16\" 2021",
      preview_snippet: "...perfect <b>laptop</b> for developers...",
      creator: "uhCAk...",
      created_at: 1703980800000
    },
    ...
  ],
  total_count: 47,
  query_time_ms: 12,
  suggested_queries: ["macbook pro", "macbook air", "laptop deals"]
}
```

### 2. Build Index

```rust
#[hdk_extern]
pub fn build_search_index(entity: IndexableEntity) -> ExternResult<ActionHash>
```

**Purpose**: Create/update search index for an entity

**Called by**: Other zomes when creating/updating entries

### 3. Autocomplete

```rust
#[hdk_extern]
pub fn autocomplete(prefix: String, limit: u32) -> ExternResult<Vec<String>>
```

**Example**:
```
Input: "mac"
Output: ["macbook", "macbook pro", "macbook air", "mac mini", "macos"]
```

### 4. Trending Searches

```rust
#[hdk_extern]
pub fn get_trending_searches(limit: u32) -> ExternResult<Vec<TrendingQuery>>
```

**Output**:
```javascript
[
  { query: "macbook pro", count: 347, trend: "up" },
  { query: "iphone 15", count: 289, trend: "up" },
  { query: "gaming laptop", count: 156, trend: "stable" }
]
```

---

## 🎨 Frontend Integration

### Search Component

```typescript
// Advanced search with autocomplete
<SearchBox
  onSearch={(query, filters) => search(query, filters)}
  onAutocomplete={(prefix) => autocomplete(prefix)}
  placeholder="Search for anything..."
  filters={<SearchFilters />}
/>

// Search results with highlighting
<SearchResults
  results={searchResults}
  onResultClick={(result) => navigate(`/listing/${result.entity_hash}`)}
  renderResult={(result) => (
    <div>
      <h3>{result.preview_title}</h3>
      <div dangerouslySetInnerHTML={{ __html: result.preview_snippet }} />
      <div>Score: {result.total_score.toFixed(3)}</div>
    </div>
  )}
/>
```

---

## 📈 Performance Optimizations

### 1. Index Caching
- Cache inverted index in memory (SQLite backend)
- Periodic index rebuilds (every 10,000 entries or 24 hours)
- Incremental updates for new entries

### 2. Query Caching
- Cache search results for 5 minutes
- Cache popular queries longer (1 hour)
- Invalidate cache on new listings in category

### 3. Pagination
- Limit results to 100 per page (default 20)
- Use offset-based pagination
- Consider cursor-based for infinite scroll

### 4. Rate Limiting
- 10 searches per minute per user
- 100 autocomplete requests per minute
- Exponential backoff for abusers

---

## 🔐 Security Considerations

### 1. Query Sanitization
- Prevent SQL injection (even though using Rust)
- Limit query length (max 500 characters)
- Block regex exploits

### 2. Privacy
- Don't index private conversations (unless participants)
- Respect blocked users (exclude from results)
- Don't leak MATL scores of users who opted out

### 3. Spam Prevention
- Detect and penalize keyword stuffing
- Downrank listings from low-MATL users
- Flag suspicious search patterns

---

## 🎯 Implementation Phases

### Phase 3A: Core Search (Week 1)
- ✅ Index builder for listings
- ✅ TF-IDF ranking algorithm
- ✅ Basic search endpoint
- ✅ Simple query parser

### Phase 3B: Advanced Features (Week 2)
- ✅ MATL-weighted ranking
- ✅ Natural language parser
- ✅ Autocomplete
- ✅ Search filters

### Phase 3C: Multi-Entity Search (Week 3)
- ✅ Index transactions
- ✅ Index conversations
- ✅ Index reviews
- ✅ Unified search across all entities

### Phase 3D: Intelligence (Week 4)
- ✅ Trending searches
- ✅ Suggested queries
- ✅ Personalized ranking
- ✅ Search analytics

---

## 📊 Success Metrics

**Technical**:
- Query latency < 100ms (p95)
- Index build time < 5 seconds for 10K listings
- Cache hit rate > 60%
- Memory usage < 500MB for 100K listings

**User Experience**:
- First result click rate > 70%
- Zero-result queries < 5%
- Average results per query > 10
- User satisfaction score > 4.5/5

---

## 🔮 Future Enhancements

1. **Semantic Search**: Use embeddings for meaning-based search
2. **Image Search**: Search by product photos (reverse image search)
3. **Voice Search**: Natural language voice queries
4. **Federated Search**: Search across multiple marketplaces
5. **AI Recommendations**: "You might also like..."

---

**Status**: Ready for implementation 🚀
**Dependencies**: None (standalone utility module)
**Priority**: High (critical for user experience)

---

*"A marketplace is only as good as its search. Let's make ours revolutionary."* 🔍
