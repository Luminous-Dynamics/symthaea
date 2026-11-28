#![forbid(unsafe_code)]

//! Exact dyadic-epoch prior for unknown start times.
//!
//! Epoch j contains starts [2^(j-1), 2^j - 1]. Its mass is 1/[j(j+1)],
//! and that mass is split uniformly over its 2^(j-1) start times. The
//! epoch masses telescope to one. No floating-point value participates in
//! the prior arithmetic.

use cmaq_053_compositional_risk_budget::{Rational, RiskBudgetError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StartPriorError {
    ZeroStartTime,
    ZeroEpoch,
    ArithmeticOverflow,
    InvalidMass,
    RiskBudget(RiskBudgetError),
}

impl From<RiskBudgetError> for StartPriorError {
    fn from(value: RiskBudgetError) -> Self {
        Self::RiskBudget(value)
    }
}

impl std::fmt::Display for StartPriorError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StartPriorError {}

/// Identifies the one-based dyadic epoch containing a positive start time.
pub fn epoch_for_start(start_time: u64) -> Result<u32, StartPriorError> {
    if start_time == 0 {
        return Err(StartPriorError::ZeroStartTime);
    }
    Ok(64 - start_time.leading_zeros())
}

/// Prior mass assigned to one start time.
///
/// For epoch j, nu_s = 1/[2^(j-1) * j * (j+1)].
pub fn weight_for_start(start_time: u64) -> Result<Rational, StartPriorError> {
    let epoch = u128::from(epoch_for_start(start_time)?);
    let epoch_size = 1u128
        .checked_shl(u32::try_from(epoch - 1).map_err(|_| StartPriorError::ArithmeticOverflow)?)
        .ok_or(StartPriorError::ArithmeticOverflow)?;
    let next_epoch = epoch
        .checked_add(1)
        .ok_or(StartPriorError::ArithmeticOverflow)?;
    let denominator = epoch_size
        .checked_mul(epoch)
        .and_then(|value| value.checked_mul(next_epoch))
        .ok_or(StartPriorError::ArithmeticOverflow)?;
    Ok(Rational::new(1, denominator)?)
}

/// Total prior mass assigned to one dyadic epoch.
pub fn epoch_mass(epoch: u32) -> Result<Rational, StartPriorError> {
    if epoch == 0 {
        return Err(StartPriorError::ZeroEpoch);
    }
    let epoch = u128::from(epoch);
    let next_epoch = epoch
        .checked_add(1)
        .ok_or(StartPriorError::ArithmeticOverflow)?;
    let denominator = epoch
        .checked_mul(next_epoch)
        .ok_or(StartPriorError::ArithmeticOverflow)?;
    Ok(Rational::new(1, denominator)?)
}

/// Residual mass of all start-time components whose start is strictly after
/// the number of observations already seen.
///
/// Components that have not started have wealth exactly one. Including this
/// residual mass is necessary to keep the all-start mixture's initial wealth
/// at one at every finite time. For zero observations, the residual is one.
pub fn future_mass_after(observations_seen: u64) -> Result<Rational, StartPriorError> {
    if observations_seen == 0 {
        return Ok(Rational::one());
    }

    let epoch = u128::from(epoch_for_start(observations_seen)?);
    let epoch_size = 1u128
        .checked_shl(u32::try_from(epoch - 1).map_err(|_| StartPriorError::ArithmeticOverflow)?)
        .ok_or(StartPriorError::ArithmeticOverflow)?;
    let epoch_end_exclusive = 1u128
        .checked_shl(u32::try_from(epoch).map_err(|_| StartPriorError::ArithmeticOverflow)?)
        .ok_or(StartPriorError::ArithmeticOverflow)?;
    let epoch_end = epoch_end_exclusive
        .checked_sub(1)
        .ok_or(StartPriorError::ArithmeticOverflow)?;
    let remaining_starts = epoch_end
        .checked_sub(u128::from(observations_seen))
        .ok_or(StartPriorError::ArithmeticOverflow)?;

    // All epochs after j have total mass sum_(q=j+1)^infinity 1/[q(q+1)]
    // = 1/(j+1).
    let next_epoch = epoch
        .checked_add(1)
        .ok_or(StartPriorError::ArithmeticOverflow)?;
    let later_epochs = Rational::new(1, next_epoch)?;

    if remaining_starts == 0 {
        return Ok(later_epochs);
    }

    let within_epoch_denominator = epoch_size
        .checked_mul(epoch)
        .and_then(|value| value.checked_mul(next_epoch))
        .ok_or(StartPriorError::ArithmeticOverflow)?;
    let remaining_mass = Rational::new(remaining_starts, within_epoch_denominator)?;
    Ok(remaining_mass.checked_add(later_epochs)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum_weights(start: u64, end: u64) -> Rational {
        assert!(start <= end);
        (start..=end)
            .map(|value| weight_for_start(value).unwrap())
            .reduce(|left, right| left.checked_add(right).unwrap())
            .unwrap()
    }

    #[test]
    fn epoch_mapping_is_correct_at_dyadic_boundaries() {
        assert_eq!(epoch_for_start(1), Ok(1));
        assert_eq!(epoch_for_start(2), Ok(2));
        assert_eq!(epoch_for_start(3), Ok(2));
        assert_eq!(epoch_for_start(4), Ok(3));
        assert_eq!(epoch_for_start(7), Ok(3));
        assert_eq!(epoch_for_start(8), Ok(4));
        assert_eq!(epoch_for_start(u64::MAX), Ok(64));
        assert_eq!(epoch_for_start(0), Err(StartPriorError::ZeroStartTime));
    }

    #[test]
    fn each_start_receives_equal_mass_inside_its_epoch() {
        assert_eq!(weight_for_start(1).unwrap(), Rational::new(1, 2).unwrap());
        assert_eq!(weight_for_start(2).unwrap(), Rational::new(1, 12).unwrap());
        assert_eq!(weight_for_start(3).unwrap(), Rational::new(1, 12).unwrap());
        assert_eq!(weight_for_start(4).unwrap(), Rational::new(1, 48).unwrap());
        assert_eq!(weight_for_start(7).unwrap(), Rational::new(1, 48).unwrap());
        assert_eq!(weight_for_start(8).unwrap(), Rational::new(1, 160).unwrap());
    }

    #[test]
    fn complete_epoch_masses_telescope_exactly() {
        for epoch in 1..=12u32 {
            let start = 1u64 << (epoch - 1);
            let end = (1u64 << epoch) - 1;
            assert_eq!(sum_weights(start, end), epoch_mass(epoch).unwrap());
        }

        let mut total = epoch_mass(1).unwrap();
        for epoch in 2..=12u32 {
            let start = 1u64 << (epoch - 1);
            let end = (1u64 << epoch) - 1;
            total = total.checked_add(sum_weights(start, end)).unwrap();
        }
        assert_eq!(total, Rational::new(12, 13).unwrap());
    }

    #[test]
    fn observed_start_mass_plus_future_residual_is_exactly_one() {
        assert_eq!(future_mass_after(0).unwrap(), Rational::one());

        for horizon in [
            1u64, 2, 3, 4, 7, 8, 9, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1_000,
        ] {
            let observed_mass = (1..=horizon)
                .map(weight_for_start)
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
                .into_iter()
                .reduce(|left, right| left.checked_add(right).unwrap())
                .unwrap();
            assert_eq!(
                observed_mass.checked_add(future_mass_after(horizon).unwrap()).unwrap(),
                Rational::one()
            );
        }
    }

    #[test]
    fn future_residual_is_correct_at_epoch_edges() {
        assert_eq!(future_mass_after(1).unwrap(), Rational::new(1, 2).unwrap());
        assert_eq!(future_mass_after(2).unwrap(), Rational::new(5, 12).unwrap());
        assert_eq!(future_mass_after(3).unwrap(), Rational::new(1, 3).unwrap());
        assert_eq!(future_mass_after(4).unwrap(), Rational::new(5, 16).unwrap());
        assert_eq!(future_mass_after(7).unwrap(), Rational::new(1, 4).unwrap());
        assert_eq!(future_mass_after(8).unwrap(), Rational::new(39, 160).unwrap());
    }

    #[test]
    fn maximum_u64_start_time_is_representable_and_has_future_tail() {
        let last_start = weight_for_start(u64::MAX).unwrap();
        assert_eq!(last_start, Rational::new(1, 64 * 65 * (1u128 << 63)).unwrap());
        assert_eq!(
            future_mass_after(u64::MAX).unwrap(),
            Rational::new(1, 65).unwrap()
        );
    }

    #[test]
    fn invalid_epoch_is_rejected() {
        assert_eq!(epoch_mass(0), Err(StartPriorError::ZeroEpoch));
    }
}

mod restart_portfolio;
pub use restart_portfolio::{
    KernelStartPosterior, RestartComponentUpdate, RestartPortfolioError,
    RestartPortfolioUpdate, RestartableKernelFamilyPortfolio,
};
