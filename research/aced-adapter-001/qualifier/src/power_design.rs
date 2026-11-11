const MIN_ALPHA: f64 = 1e-6;
const MAX_ALPHA: f64 = 0.2;
const MIN_POWER: f64 = 0.5;
const MAX_POWER: f64 = 0.999_999;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoSidedMeanPowerDesign {
    /// Standardized mean difference (Cohen's d); must be positive.
    pub standardized_effect_size: f64,
    /// Two-sided type-I error probability.
    pub alpha: f64,
    /// Target statistical power.
    pub target_power: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EqualAllocationSampleSize {
    pub units_per_arm: u64,
    pub total_units: u64,
    pub achieved_power_at_planned_n: f64,
}

fn normal_cdf(value: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26; adequate for a planning approximation, not an exact tail probability.
    let absolute = value.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * absolute);
    let density = (-0.5 * absolute * absolute).exp()
        / (2.0 * std::f64::consts::PI).sqrt();
    let polynomial = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937
                    + t * (-1.821_255_978 + t * 1.330_274_429))));
    let upper_tail = density * polynomial;
    if value >= 0.0 {
        1.0 - upper_tail
    } else {
        upper_tail
    }
}

fn inverse_normal_cdf(probability: f64) -> Result<f64, String> {
    if !probability.is_finite() || probability <= 0.0 || probability >= 1.0 {
        return Err("normal quantile probability must be finite and strictly between zero and one".into());
    }

    // Acklam's rational approximation.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.024_25;
    const HIGH: f64 = 1.0 - LOW;

    let quantile = if probability < LOW {
        let q = (-2.0 * probability.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if probability <= HIGH {
        let q = probability - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - probability).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    if quantile.is_finite() {
        Ok(quantile)
    } else {
        Err("normal quantile calculation was non-finite".into())
    }
}

impl TwoSidedMeanPowerDesign {
    pub fn validate(self) -> Result<(), String> {
        if !self.standardized_effect_size.is_finite() || self.standardized_effect_size <= 0.0 {
            return Err("standardized effect size must be finite and positive".into());
        }
        if !self.alpha.is_finite() || !(MIN_ALPHA..=MAX_ALPHA).contains(&self.alpha) {
            return Err(format!("alpha must be in [{MIN_ALPHA}, {MAX_ALPHA}]"));
        }
        if !self.target_power.is_finite()
            || !(MIN_POWER..=MAX_POWER).contains(&self.target_power)
        {
            return Err(format!("target power must be in [{MIN_POWER}, {MAX_POWER}]"));
        }
        Ok(())
    }

    pub fn required_equal_allocation_sample_size(
        self,
    ) -> Result<EqualAllocationSampleSize, String> {
        self.validate()?;
        let critical = inverse_normal_cdf(1.0 - self.alpha / 2.0)?;
        let power_quantile = inverse_normal_cdf(self.target_power)?;
        let units = 2.0
            * (critical + power_quantile).powi(2)
            / self.standardized_effect_size.powi(2);
        if !units.is_finite() || units > (u64::MAX / 2) as f64 {
            return Err("required sample size is non-finite or exceeds supported range".into());
        }
        let units_per_arm = units.ceil().max(2.0) as u64;
        let total_units = units_per_arm
            .checked_mul(2)
            .ok_or_else(|| "total planned sample size overflowed".to_string())?;
        let achieved = self.achieved_power(units_per_arm)?;
        Ok(EqualAllocationSampleSize {
            units_per_arm,
            total_units,
            achieved_power_at_planned_n: achieved,
        })
    }

    pub fn achieved_power(self, units_per_arm: u64) -> Result<f64, String> {
        self.validate()?;
        if units_per_arm < 2 {
            return Err("at least two independent units per arm are required".into());
        }
        let critical = inverse_normal_cdf(1.0 - self.alpha / 2.0)?;
        let noncentrality =
            self.standardized_effect_size * (units_per_arm as f64 / 2.0).sqrt();
        let power = normal_cdf(-critical - noncentrality)
            + (1.0 - normal_cdf(critical - noncentrality));
        if !power.is_finite() {
            return Err("achieved power calculation was non-finite".into());
        }
        Ok(power.clamp(0.0, 1.0))
    }

    pub fn qualify_planned_total(self, planned_units: u64) -> Result<EqualAllocationSampleSize, String> {
        if planned_units % 2 != 0 {
            return Err("equal-allocation planned_units must be even".into());
        }
        let required = self.required_equal_allocation_sample_size()?;
        if planned_units < required.total_units {
            return Err(format!(
                "planned_units {planned_units} is below required total {} for target power {}",
                required.total_units, self.target_power
            ));
        }
        let achieved = self.achieved_power(planned_units / 2)?;
        Ok(EqualAllocationSampleSize {
            units_per_arm: planned_units / 2,
            total_units: planned_units,
            achieved_power_at_planned_n: achieved,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard() -> TwoSidedMeanPowerDesign {
        TwoSidedMeanPowerDesign {
            standardized_effect_size: 0.5,
            alpha: 0.05,
            target_power: 0.8,
        }
    }

    #[test]
    fn classic_half_standard_deviation_design_requires_about_126_units() {
        let result = standard().required_equal_allocation_sample_size().unwrap();
        assert_eq!(result.total_units, 126);
        assert_eq!(result.units_per_arm, 63);
        assert!(result.achieved_power_at_planned_n >= 0.8);
    }

    #[test]
    fn underpowered_plan_fails_closed_and_adequate_plan_qualifies() {
        assert!(standard().qualify_planned_total(124).is_err());
        assert!(standard().qualify_planned_total(126).is_ok());
    }

    #[test]
    fn odd_equal_allocation_sample_is_rejected() {
        assert!(standard().qualify_planned_total(127).is_err());
    }

    #[test]
    fn larger_effect_needs_fewer_units() {
        let small = standard().required_equal_allocation_sample_size().unwrap();
        let large = TwoSidedMeanPowerDesign {
            standardized_effect_size: 0.8,
            ..standard()
        }
        .required_equal_allocation_sample_size()
        .unwrap();
        assert!(large.total_units < small.total_units);
    }

    #[test]
    fn higher_power_target_needs_more_units() {
        let ordinary = standard().required_equal_allocation_sample_size().unwrap();
        let high = TwoSidedMeanPowerDesign {
            target_power: 0.9,
            ..standard()
        }
        .required_equal_allocation_sample_size()
        .unwrap();
        assert!(high.total_units > ordinary.total_units);
    }

    #[test]
    fn invalid_effect_or_probability_fails_closed() {
        assert!(TwoSidedMeanPowerDesign {
            standardized_effect_size: 0.0,
            ..standard()
        }
        .required_equal_allocation_sample_size()
        .is_err());
        assert!(TwoSidedMeanPowerDesign {
            alpha: 1.0,
            ..standard()
        }
        .required_equal_allocation_sample_size()
        .is_err());
        assert!(TwoSidedMeanPowerDesign {
            target_power: 1.0,
            ..standard()
        }
        .required_equal_allocation_sample_size()
        .is_err());
    }
}
