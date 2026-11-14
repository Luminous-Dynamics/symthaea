# Atelier wardrobe exact-search oracle v0.1

## Purpose

Provide an independently testable exact small-instance oracle for the repair-first wardrobe planner. It is a reference for correctness testing and objective trade-off inspection, not the production solver for arbitrarily large wardrobes.

## Domain

The caller supplies:

- user-declared needs (activity, occasion, weather/coverage requirements, or other explicitly chosen dimensions);
- available garments and alternative actions;
- explicit compatibility between each garment/action and each need;
- protected garments;
- user cost in integer minor units, an explicit planning currency and effort estimates;
- acquisition policy and made-to-order policy.

The optimizer does not infer fit, hygiene, safety, weatherproofness, cultural meaning, style quality, sentimental value, or lifecycle benefit.

## Choice origins

- Owned
- Repair
- Alteration
- Borrowed
- Rented
- Second-hand
- Purchase
- Made-to-order

Owned, Repair and Alteration choices are always eligible. Borrowed, Rented, Second-hand, Purchase and Made-to-order choices each have independent policy flags. A no-purchase policy can still allow borrowing, rental or second-hand access. A strict no-acquisition policy disables all five acquisition routes. Made-to-order has its own flag because it implies new production.

A protected garment is retained in every plan, even if currently unusable. An unusable garment never satisfies a need. A conflict between the protected set and an acquisition policy is an explicit error, not a silent deletion of the user's protected item.

## Exact search

The current reference implementation enumerates every subset of up to 18 optional choices. It returns an error above that limit; no partial result is called exact.

The Pareto dimensions are all minimized and remain separate. Every cost is an integer minor-unit amount in one explicitly declared currency. All choices in a search must use the same three-letter uppercase currency code as the policy; mixed currencies are rejected. The crate checks code shape, not authoritative ISO registry membership or FX conversion.

1. newly acquired items;
2. made-to-order items;
3. estimated user cost;
4. user effort;
5. number of selected choices.

A plan dominates another only when it is no worse on every dimension and strictly better on at least one. Equal metric vectors are not automatically collapsed by arbitrary style preference.

## Required correctness tests

- An existing owned garment covers declared needs with zero acquisition.
- A strict no-acquisition policy cannot return a purchase, rental, borrowing, second-hand or made-to-order recommendation.
- A no-purchase policy can still select a user-permitted borrowing or rental path.
- Repair and purchase may both remain non-dominated where one saves money and the other saves effort.
- Protected archive/ceremonial garments remain present but cannot satisfy needs when unusable.
- Unknown need references are rejected.
- Exceeding the exact search bound returns an explicit error.
- Protected acquired items conflict explicitly with a no-acquisition policy.
- Made-to-order can be disabled independently of other acquisition types.
- Uncovered requirements are listed.
- Empty required-need sets still preserve protected items.
- Mixed currency inputs are rejected rather than compared.
- Malformed currency-code shape is rejected.

## Benchmark extensions

Compare with a simple greedy set-cover baseline, a weighted-sum baseline, a user-authored spreadsheet, and a shopping-oriented recommender on the same synthetic and adjudicated user scenarios.

Report:

- exact-vs-heuristic objective gaps on small cases;
- coverage and infeasibility detection;
- false acquisition recommendations under no-addition policy;
- Pareto-front preservation and decision regret;
- runtime growth and memory with choice count;
- user-selected-plan satisfaction and agency;
- performance by wardrobe size and task type.

Use a separate approximate solver for larger cases only after the exact oracle provides a correctness reference. Never silently switch from exact to approximate.

## Claim ceiling

This oracle qualifies only combinatorial coverage and objective arithmetic under caller-supplied compatibility data. It does not establish that a garment is clean, wearable, safe, comfortable, in good repair, suitable for weather, or environmentally preferable. Those facts require separate observations and evidence.
