# Market replay and price-impact evaluation protocol v1

**Status:** research protocol; no market simulator or external vendor has been executed through the reference evaluator.  
**Parent:** SAWEM BESS underwriting profile v1.  
**Purpose:** separate market-model fidelity from battery-policy quality, and distinguish price-taking from price-making revenue.

## 1. Why the distinction matters

A price-taking dispatch model treats the market price path as exogenous. A price-making model allows the battery's submitted offer or net injection to change the clearing price. They produce different counterfactuals and must not be collapsed into one revenue estimate.

The evaluation has four distinct outputs:

1. **Forecast-following implementable baseline:** only price and availability information available at the historical offer/dispatch decision.
2. **Price-taking perfect-information bound:** the best action under realized exogenous prices. This is a non-implementable mathematical bound.
3. **Price-impact-adjusted perfect-information bound:** the same policy evaluated against clearing prices that include its own market impact, with market rules, offers, demand and network assumptions frozen.
4. **Paired implementable policy comparison:** candidate and comparator evaluated through the same market-impact model build, on the same scenarios, price references, horizons and information frontier.

A perfect-information result is not realized revenue. An impact-adjusted oracle remains an oracle, not an implementable policy.

## 2. External reference studies

Supagrid's September 2026 NEM research provides a useful empirical reference. Its bidding-family analysis examined 9.7 million battery rebids and one year of outcomes for 45 batteries. It reports that 34 of 45 beat a simple AEMO-forecast-following baseline. It also reports that including self-price impact changed the median score against a naive hindsight ceiling from 69% to 77%.

Supagrid separately reports a held-out market replay across nine unseen September 2026 days: 12,960 five-minute regional prices, 98.2% within US$1/MWh of AEMO, 99.8% of unit outputs within 1 MW of AEMO targets and 99.9% agreement on binding network limits. These are vendor-published NEM results, not Symthaea measurements and not SAWEM calibration constants. The replay did not include prices above US$1,000/MWh. Its price-impact reconstruction did not include within-region network effects; the bidding study also lacked visibility of private contracts and hedges.

A peer-reviewed 2025 price-maker bidding paper formulates storage participation as a bi-level optimization problem with decision-dependent price uncertainty. This supports treating price-making as a distinct modelling class from price-taking arbitrage.

Sources:
- https://www.supagrid.com.au/research/machines-that-bid-our-batteries
- https://www.supagrid.com.au/research/dispatch-held-out-september-2026
- https://doi.org/10.1016/j.epsr.2025.111461
- https://doi.org/10.1016/j.apenergy.2025.125824

## 3. Market replay gate before policy comparison

The market simulator must reproduce the baseline market before its counterfactual battery results are trusted.

For a rights-cleared held-out replay, freeze and report:
- the complete five-minute offer, demand, interconnector, network and constraint inputs;
- price formation and regional settlement definitions;
- exact market simulator build, configuration and run receipt;
- input and output content digests;
- price MAE/median absolute error and proportion within preregistered absolute currency/MWh tolerances;
- unit dispatch-target error and binding-constraint agreement;
- network and economic curtailment error;
- high-price, negative-price and network-stress event recall;
- missing intervals and exclusions.

Do not score price errors in percentage terms across negative and near-zero prices. Use absolute currency/MWh error, signed bias, quantiles and event recall.

A same-market reproduction target may use Supagrid's nine-day NEM replay as a reference, but only when its exact data and definitions can be frozen. A SAWEM market replay is a separate cohort requiring authorized data and SAWEM-specific rules. NEM results must not be transferred to South Africa.

## 4. Matched price-impact policy comparison

For each Trading Period, provide:
- frozen exogenous reference price and source availability timestamp;
- candidate net grid energy and source receipt;
- candidate impact-adjusted clearing price and market model receipt;
- comparator net grid energy and impact-adjusted price;
- the same market-impact model identity and exact build digest;
- the same scenario, horizon, interval census and exogenous reference path.

Report exact price-taking revenue, impact-adjusted revenue, the self-impact adjustment and the candidate-minus-comparator difference. Positive signed grid energy is export; negative signed grid energy is import. Negative-price charging must retain signed cashflow.

The Rust module `market_impact_attribution.rs` implements this revenue attribution over externally supplied reference and impact-adjusted prices. It does not clear the market or estimate elasticity. The comparison rejects mismatched reference-price records or availability vintages, market-model builds, horizons and interval census.

## 5. Required stratification

Report outcomes separately by:
- market region and congestion regime;
- battery size and net injection scale;
- market depth / concentration;
- forecast and dispatch horizon;
- negative, ordinary, high and extreme price regimes;
- energy-only versus reserve/ancillary-service obligations;
- contract, hedge and network-service restrictions where the information is legally available.

Do not attribute a difference to the bidding engine if it is explained by region, network topology, market depth, operating contract, reserve enablement or different input information.

## 6. Claim ceiling and acceptance sequence

The current evaluator is a synthetic arithmetic and matching contract. It is not an external engine adapter or empirical market-impact estimate.

Acceptance sequence:
1. reproduce the selected public market replay baseline on a sealed held-out set;
2. quantify price and dispatch errors, including tail regimes;
3. validate price impact against observed or independently replayed clearing-price response;
4. compare implementable policies on the same market model and source frontier;
5. report policy revenue, downside, market-impact effects, forecast error, execution deviation and settlement separately.

No claim of beating PLEXOS, Chronos, HOMER, Supagrid or any other vendor is justified until exact versions, feature entitlements, data, model boundaries and outputs are frozen and independently reproduced.
