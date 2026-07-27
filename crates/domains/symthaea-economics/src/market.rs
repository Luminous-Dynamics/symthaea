// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validated linear supply-and-demand models, welfare, and interventions.

use crate::error::{EconomicsError, Result, ensure_finite};

/// Linear demand curve `Qd = intercept − slope·P`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Demand {
    intercept: f64,
    slope: f64,
}

/// Linear supply curve `Qs = intercept + slope·P`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Supply {
    intercept: f64,
    slope: f64,
}

/// A non-negative market-clearing point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equilibrium {
    pub price: f64,
    pub quantity: f64,
}

/// Welfare under an untruncated linear supply-and-demand model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketSurplus {
    pub consumer_surplus: f64,
    pub producer_surplus: f64,
    pub total_surplus: f64,
}

/// Market quantities at an administered price.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriceOutcome {
    pub price: f64,
    pub quantity_demanded: f64,
    pub quantity_supplied: f64,
    /// Quantity that can actually trade without rationing assumptions.
    pub traded_quantity: f64,
    pub shortage: f64,
    pub surplus: f64,
}

/// Equilibrium and welfare under a per-unit tax paid as a buyer/seller wedge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaxedEquilibrium {
    pub tax_per_unit: f64,
    pub buyer_price: f64,
    pub seller_price: f64,
    pub quantity: f64,
    pub consumer_tax_incidence: f64,
    pub producer_tax_incidence: f64,
    pub consumer_surplus: f64,
    pub producer_surplus: f64,
    pub tax_revenue: f64,
    pub total_surplus: f64,
    pub deadweight_loss: f64,
}

impl Demand {
    /// Construct a demand curve with non-negative intercept and positive slope.
    pub fn new(intercept: f64, slope: f64) -> Result<Self> {
        ensure_finite(intercept, "demand intercept")?;
        ensure_finite(slope, "demand slope")?;
        if intercept < 0.0 || slope <= 0.0 {
            return Err(EconomicsError::InvalidParameter {
                context: "demand requires intercept >= 0 and slope > 0",
            });
        }
        Ok(Self { intercept, slope })
    }

    pub fn intercept(self) -> f64 {
        self.intercept
    }

    pub fn slope(self) -> f64 {
        self.slope
    }

    /// Price at which the raw linear demand curve reaches zero quantity.
    pub fn choke_price(self) -> f64 {
        self.intercept / self.slope
    }

    pub fn quantity_at(self, price: f64) -> Result<f64> {
        ensure_finite(price, "demand price")?;
        let quantity = self.intercept - self.slope * price;
        if quantity.is_finite() {
            Ok(quantity)
        } else {
            Err(EconomicsError::NumericalFailure {
                context: "demand quantity overflowed",
            })
        }
    }

    /// Inverse-demand price for a quantity on the raw linear curve.
    pub fn price_at_quantity(self, quantity: f64) -> Result<f64> {
        ensure_finite(quantity, "demand quantity")?;
        if quantity < 0.0 {
            return Err(EconomicsError::InvalidParameter {
                context: "demand quantity must be non-negative",
            });
        }
        Ok((self.intercept - quantity) / self.slope)
    }
}

impl Supply {
    /// Construct a supply curve with finite intercept and positive slope.
    pub fn new(intercept: f64, slope: f64) -> Result<Self> {
        ensure_finite(intercept, "supply intercept")?;
        ensure_finite(slope, "supply slope")?;
        if slope <= 0.0 {
            return Err(EconomicsError::InvalidParameter {
                context: "supply requires slope > 0",
            });
        }
        Ok(Self { intercept, slope })
    }

    pub fn intercept(self) -> f64 {
        self.intercept
    }

    pub fn slope(self) -> f64 {
        self.slope
    }

    /// Inverse-supply price at zero quantity on the raw linear curve.
    pub fn reservation_price(self) -> f64 {
        -self.intercept / self.slope
    }

    pub fn quantity_at(self, price: f64) -> Result<f64> {
        ensure_finite(price, "supply price")?;
        let quantity = self.intercept + self.slope * price;
        if quantity.is_finite() {
            Ok(quantity)
        } else {
            Err(EconomicsError::NumericalFailure {
                context: "supply quantity overflowed",
            })
        }
    }

    /// Inverse-supply price for a quantity on the raw linear curve.
    pub fn price_at_quantity(self, quantity: f64) -> Result<f64> {
        ensure_finite(quantity, "supply quantity")?;
        if quantity < 0.0 {
            return Err(EconomicsError::InvalidParameter {
                context: "supply quantity must be non-negative",
            });
        }
        Ok((quantity - self.intercept) / self.slope)
    }
}

/// Market-clearing point where `Qd = Qs`.
pub fn equilibrium(demand: &Demand, supply: &Supply) -> Result<Equilibrium> {
    let denominator = demand.slope + supply.slope;
    let price = (demand.intercept - supply.intercept) / denominator;
    let quantity = demand.quantity_at(price)?;
    if !price.is_finite() || !quantity.is_finite() {
        return Err(EconomicsError::NumericalFailure {
            context: "market equilibrium overflowed",
        });
    }
    if price < 0.0 || quantity < 0.0 {
        return Err(EconomicsError::NoSolution {
            context: "linear curves have no non-negative interior equilibrium",
        });
    }
    Ok(Equilibrium { price, quantity })
}

/// Consumer, producer, and total surplus at a supplied equilibrium.
///
/// The calculation integrates the raw linear curves. If supply has a positive
/// quantity intercept, its inverse curve extends below zero price; callers who
/// truncate prices or quantities should model that boundary explicitly.
pub fn market_surplus(
    demand: &Demand,
    supply: &Supply,
    point: Equilibrium,
) -> Result<MarketSurplus> {
    ensure_finite(point.price, "surplus equilibrium price")?;
    ensure_finite(point.quantity, "surplus equilibrium quantity")?;
    if point.price < 0.0 || point.quantity < 0.0 {
        return Err(EconomicsError::InvalidParameter {
            context: "surplus requires a non-negative equilibrium",
        });
    }
    let demand_quantity = demand.quantity_at(point.price)?;
    let supply_quantity = supply.quantity_at(point.price)?;
    let tolerance = 1e-9 * point.quantity.abs().max(1.0);
    if (demand_quantity - point.quantity).abs() > tolerance
        || (supply_quantity - point.quantity).abs() > tolerance
    {
        return Err(EconomicsError::InvalidParameter {
            context: "surplus point does not lie on both market curves",
        });
    }

    let consumer_surplus = 0.5 * (demand.choke_price() - point.price) * point.quantity;
    let producer_surplus = 0.5 * (point.price - supply.reservation_price()) * point.quantity;
    let total_surplus = consumer_surplus + producer_surplus;
    if consumer_surplus < -tolerance || producer_surplus < -tolerance || !total_surplus.is_finite()
    {
        return Err(EconomicsError::NumericalFailure {
            context: "market surplus is outside the supported linear domain",
        });
    }
    Ok(MarketSurplus {
        consumer_surplus: consumer_surplus.max(0.0),
        producer_surplus: producer_surplus.max(0.0),
        total_surplus: total_surplus.max(0.0),
    })
}

/// Quantities and imbalance at a non-negative administered price.
pub fn market_at_price(demand: &Demand, supply: &Supply, price: f64) -> Result<PriceOutcome> {
    ensure_finite(price, "administered market price")?;
    if price < 0.0 {
        return Err(EconomicsError::InvalidParameter {
            context: "administered market price must be non-negative",
        });
    }
    let quantity_demanded = demand.quantity_at(price)?.max(0.0);
    let quantity_supplied = supply.quantity_at(price)?.max(0.0);
    Ok(PriceOutcome {
        price,
        quantity_demanded,
        quantity_supplied,
        traded_quantity: quantity_demanded.min(quantity_supplied),
        shortage: (quantity_demanded - quantity_supplied).max(0.0),
        surplus: (quantity_supplied - quantity_demanded).max(0.0),
    })
}

/// Equilibrium under a non-negative per-unit tax.
pub fn equilibrium_with_tax(
    demand: &Demand,
    supply: &Supply,
    tax_per_unit: f64,
) -> Result<TaxedEquilibrium> {
    ensure_finite(tax_per_unit, "per-unit tax")?;
    if tax_per_unit < 0.0 {
        return Err(EconomicsError::InvalidParameter {
            context: "tax must be non-negative; subsidies require a separate policy model",
        });
    }

    let baseline = equilibrium(demand, supply)?;
    let baseline_surplus = market_surplus(demand, supply, baseline)?;
    let denominator = demand.slope + supply.slope;
    let seller_price =
        (demand.intercept - supply.intercept - demand.slope * tax_per_unit) / denominator;
    let buyer_price = seller_price + tax_per_unit;
    let quantity = demand.quantity_at(buyer_price)?;
    if seller_price < 0.0 || buyer_price < 0.0 || quantity < 0.0 {
        return Err(EconomicsError::NoSolution {
            context: "tax eliminates the supported non-negative interior market",
        });
    }

    let consumer_surplus = 0.5 * (demand.choke_price() - buyer_price) * quantity;
    let producer_surplus = 0.5 * (seller_price - supply.reservation_price()) * quantity;
    let tax_revenue = tax_per_unit * quantity;
    let total_surplus = consumer_surplus + producer_surplus + tax_revenue;
    let deadweight_loss = (baseline_surplus.total_surplus - total_surplus).max(0.0);
    if !consumer_surplus.is_finite()
        || !producer_surplus.is_finite()
        || !tax_revenue.is_finite()
        || !total_surplus.is_finite()
        || !deadweight_loss.is_finite()
    {
        return Err(EconomicsError::NumericalFailure {
            context: "taxed market calculation overflowed",
        });
    }

    Ok(TaxedEquilibrium {
        tax_per_unit,
        buyer_price,
        seller_price,
        quantity,
        consumer_tax_incidence: buyer_price - baseline.price,
        producer_tax_incidence: baseline.price - seller_price,
        consumer_surplus: consumer_surplus.max(0.0),
        producer_surplus: producer_surplus.max(0.0),
        tax_revenue,
        total_surplus,
        deadweight_loss,
    })
}

/// Point price elasticity of demand, reported as a non-negative magnitude.
pub fn price_elasticity_of_demand(demand: &Demand, price: f64) -> Result<f64> {
    ensure_finite(price, "elasticity price")?;
    if price < 0.0 {
        return Err(EconomicsError::InvalidParameter {
            context: "elasticity price must be non-negative",
        });
    }
    let quantity = demand.quantity_at(price)?;
    if quantity < 0.0 {
        return Err(EconomicsError::NoSolution {
            context: "price lies beyond the non-negative demand domain",
        });
    }
    if quantity == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok((demand.slope * price / quantity).abs())
}

/// Midpoint (arc) price elasticity between two non-negative prices.
pub fn arc_price_elasticity_of_demand(
    demand: &Demand,
    first_price: f64,
    second_price: f64,
) -> Result<f64> {
    ensure_finite(first_price, "first arc-elasticity price")?;
    ensure_finite(second_price, "second arc-elasticity price")?;
    if first_price < 0.0 || second_price < 0.0 || first_price == second_price {
        return Err(EconomicsError::InvalidParameter {
            context: "arc elasticity requires distinct non-negative prices",
        });
    }
    let first_quantity = demand.quantity_at(first_price)?;
    let second_quantity = demand.quantity_at(second_price)?;
    if first_quantity < 0.0 || second_quantity < 0.0 {
        return Err(EconomicsError::NoSolution {
            context: "arc elasticity prices must remain in the demand domain",
        });
    }
    let average_quantity = 0.5 * (first_quantity + second_quantity);
    let average_price = 0.5 * (first_price + second_price);
    if average_quantity == 0.0 || average_price == 0.0 {
        return Err(EconomicsError::NoSolution {
            context: "arc elasticity is undefined at a zero midpoint",
        });
    }
    Ok(((second_quantity - first_quantity)
        / average_quantity
        / ((second_price - first_price) / average_price))
        .abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textbook_market() -> (Demand, Supply) {
        (
            Demand::new(100.0, 2.0).unwrap(),
            Supply::new(20.0, 2.0).unwrap(),
        )
    }

    #[test]
    fn equilibrium_known() {
        let (demand, supply) = textbook_market();
        let point = equilibrium(&demand, &supply).unwrap();
        assert!((point.price - 20.0).abs() < 1e-9);
        assert!((point.quantity - 60.0).abs() < 1e-9);
    }

    #[test]
    fn surplus_is_complete() {
        let (demand, supply) = textbook_market();
        let point = equilibrium(&demand, &supply).unwrap();
        let welfare = market_surplus(&demand, &supply, point).unwrap();
        assert!((welfare.consumer_surplus - 900.0).abs() < 1e-9);
        assert!((welfare.producer_surplus - 900.0).abs() < 1e-9);
        assert!((welfare.total_surplus - 1800.0).abs() < 1e-9);
    }

    #[test]
    fn price_controls_report_imbalance() {
        let (demand, supply) = textbook_market();
        let ceiling = market_at_price(&demand, &supply, 10.0).unwrap();
        assert_eq!(ceiling.quantity_demanded, 80.0);
        assert_eq!(ceiling.quantity_supplied, 40.0);
        assert_eq!(ceiling.shortage, 40.0);
        assert_eq!(ceiling.traded_quantity, 40.0);
    }

    #[test]
    fn tax_incidence_and_welfare_reconcile() {
        let (demand, supply) = textbook_market();
        let taxed = equilibrium_with_tax(&demand, &supply, 10.0).unwrap();
        assert!((taxed.buyer_price - 25.0).abs() < 1e-9);
        assert!((taxed.seller_price - 15.0).abs() < 1e-9);
        assert!((taxed.quantity - 50.0).abs() < 1e-9);
        assert!((taxed.consumer_tax_incidence + taxed.producer_tax_incidence - 10.0).abs() < 1e-9);
        assert!((taxed.deadweight_loss - 50.0).abs() < 1e-9);
    }

    #[test]
    fn elasticities_are_consistent() {
        let demand = Demand::new(100.0, 2.0).unwrap();
        assert!((price_elasticity_of_demand(&demand, 25.0).unwrap() - 1.0).abs() < 1e-9);
        let arc = arc_price_elasticity_of_demand(&demand, 20.0, 30.0).unwrap();
        assert!((arc - 1.0).abs() < 1e-9);
    }
}
