//! Resource admissibility for extensible HDC resolutions.
//!
//! Mathematical resolution validity and empirical evidence are intentionally
//! separate from runtime resource policy. This research-only budget prevents an
//! open resolution space from silently becoming an open-ended allocation policy.

use super::{HdcResolution, ResolutionError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionBudget {
    /// Maximum bytes permitted for one resident vector.
    pub max_vector_bytes: usize,
    /// Maximum bytes permitted for all resident vectors represented by this
    /// budget.
    pub max_resident_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetError {
    Resolution(ResolutionError),
    VectorTooLarge { required: usize, limit: usize },
    ResidentWorkingSetTooLarge { required: usize, limit: usize },
    WorkingSetOverflow { vector_bytes: usize, resident_vectors: usize },
}

impl ResolutionBudget {
    pub const fn new(max_vector_bytes: usize, max_resident_bytes: usize) -> Self {
        Self { max_vector_bytes, max_resident_bytes }
    }

    pub const fn check_vector(
        self,
        resolution: HdcResolution,
        element_size: usize,
    ) -> Result<usize, BudgetError> {
        let required = match resolution.checked_bytes(element_size) {
            Ok(bytes) => bytes,
            Err(error) => return Err(BudgetError::Resolution(error)),
        };
        if required > self.max_vector_bytes {
            return Err(BudgetError::VectorTooLarge {
                required,
                limit: self.max_vector_bytes,
            });
        }
        Ok(required)
    }

    pub const fn check_resident(
        self,
        resolution: HdcResolution,
        element_size: usize,
        resident_vectors: usize,
    ) -> Result<usize, BudgetError> {
        let vector_bytes = match resolution.checked_bytes(element_size) {
            Ok(bytes) => bytes,
            Err(error) => return Err(BudgetError::Resolution(error)),
        };
        let required = match vector_bytes.checked_mul(resident_vectors) {
            Some(bytes) => bytes,
            None => {
                return Err(BudgetError::WorkingSetOverflow {
                    vector_bytes,
                    resident_vectors,
                })
            }
        };
        if required > self.max_resident_bytes {
            return Err(BudgetError::ResidentWorkingSetTooLarge {
                required,
                limit: self.max_resident_bytes,
            });
        }
        Ok(required)
    }

    pub const fn check_f32(
        self,
        resolution: HdcResolution,
    ) -> Result<usize, BudgetError> {
        self.check_vector(resolution, std::mem::size_of::<f32>())
    }

    pub const fn check_binary(
        self,
        resolution: HdcResolution,
    ) -> Result<usize, BudgetError> {
        match resolution.binary_bytes() {
            Ok(bytes) => {
                if bytes > self.max_vector_bytes {
                    Err(BudgetError::VectorTooLarge {
                        required: bytes,
                        limit: self.max_vector_bytes,
                    })
                } else {
                    Ok(bytes)
                }
            }
            Err(error) => Err(BudgetError::Resolution(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_vector_and_working_set_limits() {
        let budget = ResolutionBudget::new(512 * 1024, 2 * 1024 * 1024);
        let d128 = HdcResolution::new(131_072).unwrap();

        assert_eq!(budget.check_f32(d128).unwrap(), 512 * 1024);
        assert_eq!(
            budget.check_resident(d128, 4, 4).unwrap(),
            2 * 1024 * 1024
        );
        assert!(matches!(
            budget.check_resident(d128, 4, 5),
            Err(BudgetError::ResidentWorkingSetTooLarge { .. })
        ));
    }

    #[test]
    fn rejects_vector_above_budget() {
        let budget = ResolutionBudget::new(512 * 1024, 8 * 1024 * 1024);
        let d256 = HdcResolution::new(262_144).unwrap();

        assert!(matches!(
            budget.check_f32(d256),
            Err(BudgetError::VectorTooLarge {
                required: 1024 * 1024,
                limit: 512 * 1024
            })
        ));
    }

    #[test]
    fn catches_working_set_overflow() {
        let budget = ResolutionBudget::new(usize::MAX, usize::MAX);
        let d1 = HdcResolution::new(1).unwrap();

        assert!(matches!(
            budget.check_resident(d1, usize::MAX, 2),
            Err(BudgetError::WorkingSetOverflow { .. })
        ));
    }

    #[test]
    fn binary_budget_uses_representation_size() {
        let budget = ResolutionBudget::new(32 * 1024, 64 * 1024);
        let d256 = HdcResolution::new(262_144).unwrap();

        assert_eq!(budget.check_binary(d256).unwrap(), 32 * 1024);
    }
}
