# 🔍 Advanced Search Engine - Complete Guide

## Overview

The Mycelix-Marketplace search engine provides revolutionary full-text search across all marketplace data with intelligent ranking that combines relevance, trust, and recency.

**Key Features**:
- 🔍 **Full-Text Search** - Search across listings, transactions, conversations, reviews
- 🧠 **Intelligent Ranking** - TF-IDF + MATL trust scores + recency decay
- 💬 **Natural Language** - "macbook laptop under $500 from trusted sellers"
- ⚡ **Fast** - Optimized inverted index with caching
- 🎯 **Advanced Filters** - Price, category, location, time, trust score
- 📊 **Autocomplete** - Smart suggestions as you type
- 🔥 **Trending** - See what others are searching for

---

## Architecture

### How It Works

```
User Query: "macbook laptop under $1000"
     ↓
┌────────────────────────────────────┐
│  1. Query Parser                   │
│  - Extract terms: ["macbook", "laptop"]
│  - Extract filters: {price_max: 100000}
└────────────────────────────────────┘
     ↓
┌────────────────────────────────────┐
│  2. Search Index                   │
│  - Find documents with terms       │
│  - Apply filters                   │
└────────────────────────────────────┘
     ↓
┌────────────────────────────────────┐
│  3. TF-IDF Scoring                 │
│  - Calculate relevance per document│
│  - Higher score = better match     │
└────────────────────────────────────┘
     ↓
┌────────────────────────────────────┐
│  4. MATL-Weighted Ranking          │
│  - Combine: 60% relevance          │
│           + 30% trust (MATL)       │
│           + 10% recency            │
└────────────────────────────────────┘
     ↓
Results: [Listing A (0.881), Listing B (0.745), ...]
```

---

## API Reference

### 1. Advanced Search

**Endpoint**: `search`

**Input**:
```typescript
interface SearchQuery {
  query: string;                // Search query
  filters: SearchFilters;       // Optional filters
  offset: number;               // Pagination offset
  limit: number;                // Results per page
  sort_by: SortOption;          // Relevance | Price | Date | Trust
  sort_order: SortOrder;        // Asc | Desc
}

interface SearchFilters {
  entity_types?: EntityType[];  // Filter by type
  price_min?: number;           // Min price in cents
  price_max?: number;           // Max price in cents
  categories?: string[];        // Filter by category
  min_matl_score?: number;      // Minimum trust score
  created_after?: number;       // Filter by date
  created_before?: number;      // Filter by date
  location?: string;            // Filter by location
  status?: string;              // Filter by status
}
```

**Output**:
```typescript
interface SearchResponse {
  results: SearchResult[];      // Matched results
  total_count: number;          // Total matches (for pagination)
  query_time_ms: number;        // Query execution time
  suggested_queries: string[];  // "Did you mean...?"
}

interface SearchResult {
  entity_hash: ActionHash;      // Hash of matched entity
  entity_type: EntityType;      // Listing | Transaction | etc.

  // Scoring
  total_score: number;          // Final score (0.0-1.0)
  relevance_score: number;      // TF-IDF component
  trust_score: number;          // MATL component
  recency_score: number;        // Time decay component
  engagement_score: number;     // Views, favorites

  // Highlighting
  matched_terms: string[];      // Terms that matched
  matched_fields: string[];     // Where they matched

  // Preview
  preview_title: string;        // Highlighted title
  preview_snippet: string;      // Highlighted excerpt

  // Metadata
  creator: AgentPubKey;
  created_at: number;
}
```

**Example**:
```javascript
const results = await search({
  query: "macbook laptop",
  filters: {
    price_max: 100000,  // $1000
    min_matl_score: 0.5,
    categories: ["Electronics"],
  },
  offset: 0,
  limit: 20,
  sort_by: "Relevance",
  sort_order: "Desc",
});

console.log(`Found ${results.total_count} results in ${results.query_time_ms}ms`);
results.results.forEach(result => {
  console.log(`${result.preview_title} (score: ${result.total_score.toFixed(3)})`);
});
```

---

### 2. Build Search Index

**Endpoint**: `build_search_index`

**Purpose**: Index an entity for search (called by other zomes)

**Input**:
```typescript
interface IndexableEntity {
  entity_type: EntityType;
  entity_hash: ActionHash;
  title: string;
  description: string;
  tags: string[];
  category: string;
  creator: AgentPubKey;
  creator_matl_score: number;
  created_at: number;
  updated_at: number;
  view_count: number;
  engagement_score: number;
  price_cents?: number;
  location?: string;
  status: string;
}
```

**Output**: `ActionHash` of search index entry

**Example** (from listings zome):
```rust
// When creating a listing
let listing_hash = create_entry(&listing)?;

// Index it for search
let index_data = IndexableEntity {
    entity_type: EntityType::Listing,
    entity_hash: listing_hash.clone(),
    title: listing.title.clone(),
    description: listing.description.clone(),
    tags: extract_tags(&listing),
    category: category_to_string(&listing.category),
    creator: agent_info.agent_latest_pubkey,
    creator_matl_score: get_matl_score()?,
    created_at: listing.created_at,
    updated_at: listing.updated_at,
    view_count: 0,
    engagement_score: 0.0,
    price_cents: Some(listing.price_cents),
    location: None,
    status: "Active".to_string(),
};

call_remote(
    None,
    "search".into(),
    "build_search_index".into(),
    None,
    &index_data,
)?;
```

---

### 3. Autocomplete

**Endpoint**: `autocomplete`

**Input**:
```typescript
{
  prefix: string;   // Partial query
  limit: number;    // Max suggestions
}
```

**Output**: `string[]` - Suggested completions

**Example**:
```javascript
const suggestions = await autocomplete("mac", 5);
// Returns: ["macbook", "macbook pro", "macbook air", "mac mini", "macos"]
```

---

## Query Syntax

### Basic Search

```
"laptop"                    → Search for "laptop"
"laptop macbook"            → Search for "laptop" OR "macbook"
```

### Operators

```
"laptop AND macbook"        → Both terms required
"laptop OR desktop"         → Either term (same as space)
"-refurbished"              → Exclude term
"\"macbook pro\""           → Exact phrase match
```

### Natural Language Filters

```
"laptop under $500"         → price_max: 50000
"macbook over $1000"        → price_min: 100000
"from trusted sellers"      → min_matl_score: 0.7
"in electronics"            → category: "Electronics"
"created this week"         → created_after: <7_days_ago>
```

### Complex Queries

```
"macbook pro" under $2000 from trusted sellers -refurbished
```

Parsed as:
- Exact phrase: "macbook pro"
- Price filter: max $2000
- Trust filter: >= 0.7
- Excluded term: "refurbished"

---

## Ranking Algorithm

### Formula

```
Final Score = 0.6 × TF-IDF + 0.3 × MATL + 0.1 × Recency
```

### TF-IDF (Term Frequency-Inverse Document Frequency)

Measures how relevant a document is to the query:

```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)

where:
  TF(term, doc) = frequency of term in doc / total terms in doc
  IDF(term) = log(total documents / documents containing term)
```

**Example**:
```
Query: "macbook laptop"
Doc 1: "MacBook Pro 16-inch laptop for developers"

TF("macbook") = 1/7 = 0.143
IDF("macbook") = log(1000/50) = 3.0
TF-IDF("macbook") = 0.143 × 3.0 = 0.429

TF("laptop") = 1/7 = 0.143
IDF("laptop") = log(1000/200) = 1.7
TF-IDF("laptop") = 0.143 × 1.7 = 0.243

Total TF-IDF = 0.429 + 0.243 = 0.672
```

### MATL Trust Weighting

Boosts results from trusted sellers:

```
Trust Score = Seller's MATL Score (0.0 to 1.0)

Effect on final score:
- 0.3 MATL → +0.09 to final score
- 0.7 MATL → +0.21 to final score
- 0.9 MATL → +0.27 to final score
```

**Example**:
```
Listing A: TF-IDF = 0.85, MATL = 0.92
Listing B: TF-IDF = 0.90, MATL = 0.35

Final A = 0.6×0.85 + 0.3×0.92 + 0.1×recency = ~0.786
Final B = 0.6×0.90 + 0.3×0.35 + 0.1×recency = ~0.645

→ Listing A ranks higher despite lower TF-IDF!
```

### Recency Decay

Newer listings get a boost:

```
Recency Score = e^(-age_days / 30)

Age → Recency Score
  0 days  → 1.00 (100%)
  7 days  → 0.80 (80%)
 30 days  → 0.37 (37%)
 90 days  → 0.05 (5%)
```

---

## Integration Examples

### Example 1: Basic Search

```javascript
import { search } from "@holochain/client";

async function searchListings(query) {
  const results = await search({
    query: query,
    filters: {
      entity_types: ["Listing"],
      status: "Active",
    },
    offset: 0,
    limit: 20,
    sort_by: "Relevance",
    sort_order: "Desc",
  });

  return results;
}

const results = await searchListings("laptop");
console.log(`Found ${results.total_count} laptops`);
```

### Example 2: Advanced Filters

```javascript
async function findTrustedElectronics() {
  const results = await search({
    query: "",  // No text search, just filters
    filters: {
      entity_types: ["Listing"],
      categories: ["Electronics"],
      min_matl_score: 0.7,  // Trusted sellers only
      price_max: 100000,    // Under $1000
      created_after: Date.now() - (7 * 86400 * 1000),  // Last 7 days
    },
    offset: 0,
    limit: 50,
    sort_by: "Date",
    sort_order: "Desc",
  });

  return results;
}
```

### Example 3: Autocomplete Component

```typescript
import React, { useState, useEffect } from 'react';

function SearchBox() {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);

  useEffect(() => {
    if (query.length >= 2) {
      autocomplete(query, 5).then(setSuggestions);
    } else {
      setSuggestions([]);
    }
  }, [query]);

  return (
    <div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search marketplace..."
      />
      {suggestions.length > 0 && (
        <ul className="suggestions">
          {suggestions.map((s, i) => (
            <li key={i} onClick={() => setQuery(s)}>
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

### Example 4: Search Results Display

```typescript
function SearchResults({ results }: { results: SearchResponse }) {
  return (
    <div>
      <div className="stats">
        {results.total_count} results in {results.query_time_ms}ms
      </div>

      {results.results.map((result) => (
        <div key={result.entity_hash} className="result">
          <h3 dangerouslySetInnerHTML={{ __html: result.preview_title }} />

          <div className="snippet"
               dangerouslySetInnerHTML={{ __html: result.preview_snippet }} />

          <div className="metadata">
            <span className="score">
              Relevance: {(result.relevance_score * 100).toFixed(0)}%
            </span>
            <span className="trust">
              Trust: {(result.trust_score * 100).toFixed(0)}%
            </span>
            <span className="date">
              {new Date(result.created_at).toLocaleDateString()}
            </span>
          </div>

          <div className="matched-terms">
            Matched: {result.matched_terms.join(", ")}
          </div>
        </div>
      ))}

      {results.suggested_queries.length > 0 && (
        <div className="suggestions">
          Did you mean: {results.suggested_queries.join(", ")}
        </div>
      )}
    </div>
  );
}
```

---

## Performance Optimization

### Caching

```javascript
// Cache search results for 5 minutes
const cache = new Map();

async function cachedSearch(query) {
  const key = JSON.stringify(query);

  if (cache.has(key)) {
    const { result, timestamp } = cache.get(key);
    if (Date.now() - timestamp < 300000) {  // 5 minutes
      return result;
    }
  }

  const result = await search(query);
  cache.set(key, { result, timestamp: Date.now() });

  return result;
}
```

### Pagination

```javascript
function SearchResultsPaginated() {
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 20;

  const results = await search({
    query: searchQuery,
    filters: filters,
    offset: page * PAGE_SIZE,
    limit: PAGE_SIZE,
    sort_by: "Relevance",
    sort_order: "Desc",
  });

  const totalPages = Math.ceil(results.total_count / PAGE_SIZE);

  return (
    <>
      <SearchResults results={results} />
      <Pagination
        currentPage={page}
        totalPages={totalPages}
        onPageChange={setPage}
      />
    </>
  );
}
```

---

## Text Processing

### Tokenization

Text is processed through these steps:

1. **Normalization**: Lowercase + remove punctuation
   ```
   "Hello, World!" → "hello world"
   ```

2. **Tokenization**: Split into words
   ```
   "hello world" → ["hello", "world"]
   ```

3. **Stop Word Removal**: Remove common words
   ```
   ["the", "quick", "brown", "fox"] → ["quick", "brown", "fox"]
   ```

4. **Stemming**: Reduce to root form
   ```
   ["running", "runs", "ran"] → ["run", "run", "run"]
   ```

### Supported Languages

Currently: English only
Planned: Spanish, French, German, Chinese, Japanese

---

## Best Practices

### For Developers

1. **Index incrementally**: Call `build_search_index` when creating/updating entities
2. **Cache aggressively**: Search results don't change frequently
3. **Use filters**: Narrow down results before text search
4. **Highlight matches**: Show users why results matched
5. **Track analytics**: Monitor popular queries, zero-result queries

### For Users

1. **Be specific**: "macbook pro 16 inch 2021" > "laptop"
2. **Use filters**: Price, category, and trust filters are fast
3. **Try variations**: "laptop" vs "notebooks" vs "computers"
4. **Use quotes**: "macbook pro" for exact phrases
5. **Exclude terms**: "-refurbished" to filter out results

---

## Troubleshooting

### No Results Found

**Problem**: Query returns 0 results

**Solutions**:
- Try fewer terms ("laptop" instead of "laptop macbook pro 16 inch")
- Remove filters one by one
- Check spelling
- Use autocomplete for valid terms

### Irrelevant Results

**Problem**: Top results don't match query well

**Possible causes**:
- Keywords in description but not title
- Low-quality listings have matching terms
- Very common terms (use more specific queries)

**Solutions**:
- Add more specific terms
- Use exact phrases: "\"macbook pro\""
- Increase `min_matl_score` filter
- Sort by Trust instead of Relevance

### Slow Queries

**Problem**: Queries taking > 500ms

**Possible causes**:
- Large result set (thousands of matches)
- Complex query with many terms
- No filters (searching entire index)

**Solutions**:
- Add filters to narrow results
- Use pagination (limit to 20-50 results)
- Enable caching
- Index may need rebuilding

---

## Future Enhancements

### Phase 4: Intelligent Search
- **Semantic search**: Meaning-based matching using embeddings
- **Spell correction**: "macbok" → "macbook"
- **Synonym expansion**: "laptop" → "notebook", "computer"
- **Query understanding**: Extract intent from natural language

### Phase 5: Visual Search
- **Image search**: Upload photo to find similar listings
- **Reverse image search**: Find where a product photo appears
- **Visual similarity**: ML-based image matching

### Phase 6: Voice Search
- **Speech-to-text**: Voice query input
- **Natural language**: "Show me cheap laptops from good sellers"
- **Voice results**: Read results aloud

### Phase 7: Personalization
- **Search history**: Learn from user's searches
- **Preference learning**: Adjust ranking based on clicks
- **Recommendation**: "You might also like..."

---

## Statistics & Metrics

### Performance Targets

- **Query latency**: < 100ms (p95)
- **Index build**: < 5 seconds for 10,000 entries
- **Cache hit rate**: > 60%
- **Memory usage**: < 500MB for 100,000 entries

### User Experience Metrics

- **First result click**: > 70%
- **Zero-result queries**: < 5%
- **Results per query**: > 10 average
- **Satisfaction score**: > 4.5/5

---

## Conclusion

The Mycelix-Marketplace search engine combines cutting-edge text search (TF-IDF) with decentralized trust (MATL) to create a revolutionary search experience that prioritizes both relevance and trustworthiness.

**Key Takeaways**:
- 60% relevance, 30% trust, 10% recency = perfect balance
- Natural language queries make search intuitive
- MATL weighting prevents scammers from gaming rankings
- Full integration with all marketplace entities

**Next Steps**:
1. Integrate search into frontend
2. Add search bar to all pages
3. Monitor query analytics
4. Iterate based on user feedback

---

**Questions?** See [SEARCH_ENGINE_ARCHITECTURE.md](./SEARCH_ENGINE_ARCHITECTURE.md) for technical details.

---

*Last Updated*: December 30, 2025
*Version*: v1.0.0
*Status*: Production Ready ✅
