# St. Lucia Sentinel Pilot v1 — catalogue integrity amendment

Status: **pre-product-selection amendment**

This document strengthens `ST_LUCIA_SENTINEL_PILOT_V1.md` before any individual St. Lucia Sentinel item is selected. It changes neither the site nor the scientific success criteria.

## Exhaustive discovery

The candidate set must not depend on a server's first page or default ordering.

For each frozen Sentinel-2 and Sentinel-1 search:

1. request the maximum practical documented page size for the collection;
2. for `sentinel-2-l2a`, do not exceed `limit=200` unless a documented Fields-Extension query explicitly supports the larger response;
3. follow the response's official STAC `next` link/token until exhaustion;
4. retain every response page or a canonical concatenated snapshot plus a digest for every page;
5. deduplicate only by exact STAC item ID after complete retrieval;
6. if one item ID appears with non-identical canonical metadata, fail the discovery run;
7. construct the complete eligible candidate set;
8. sort that set locally according to the already-preregistered deterministic selection rules;
9. only then select the item.

This deliberately prevents API pagination or default server ordering from becoming an undocumented selection rule.

## Null discovery semantics

`no-eligible-item` is too broad for an evidence record. The canonical outcome vocabulary is:

```text
selected(item_id)
no-eligible-item-visible
catalogue-query-failed
catalogue-pagination-inconsistent
duplicate-id-metadata-conflict
catalogue-availability-uncertain
```

`no-eligible-item-visible` means only that no qualifying item was visible from the frozen catalogue endpoint under the frozen query at the recorded retrieval time.

It does **not** prove that the satellite made no acquisition.

Recent Copernicus Data Space Ecosystem community reports have documented temporary Sentinel-2 L2A STAC discoverability/ingestion gaps. The pilot therefore preserves catalogue availability as part of provenance rather than conflating catalogue state with physical acquisition history.

## Alternate-catalogue diagnostics

A second official catalogue may be queried to diagnose availability, but the result must be labelled separately from the locked product-selection lineage.

An alternate catalogue may answer:

```text
Is the primary catalogue potentially incomplete?
```

It may not silently answer:

```text
Which different item should enter the experiment instead?
```

Changing the source catalogue or substituting a different product requires a prospective protocol amendment.

## Evidence retained per page

Retain, where exposed:

- request URL / POST body;
- endpoint identity;
- retrieval timestamp;
- HTTP/service status;
- response body or canonical response snapshot;
- response digest;
- page number or continuation token;
- previous/next link identity;
- item IDs on the page;
- server/catalogue version metadata.

A failed query is itself retained evidence; it is not replaced by a successful retry without recording the failed attempt.

## References

Current CDSE STAC documentation:

- https://documentation.dataspace.copernicus.eu/APIs/STAC.html

CDSE support has documented collection-specific page-size behaviour for Sentinel-2 L2A and recent catalogue availability incidents. Those operational facts motivate this evidence distinction but do not change the frozen scientific target.
