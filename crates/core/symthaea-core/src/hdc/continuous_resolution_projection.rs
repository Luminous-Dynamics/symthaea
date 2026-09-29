//! Research-only continuous-HV resolution projections.
//!
//! This module intentionally does not participate in production adaptive resizing.
//! It provides a deterministic orthogonal-basis down-projection candidate so the
//! trajectory matrix can distinguish properties of legacy dilation from properties
//! of dimensional reduction itself.

use super::ContinuousHV;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuousProjectionError {
    SameDimension,
    ExpansionUnsupported,
    NonPowerOfTwo,
    InvalidTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuousProjectionFamily {
    /// Normalized Walsh-Hadamard transform followed by deterministic low-index
    /// coefficient selection. This is an orthogonal change of basis followed by
    /// truncation; it makes no claim that HDC binding/bundling are preserved.
    HadamardTruncateV1,
}

fn is_power_of_two(n: usize) -> bool {
    n != 0 && n.is_power_of_two()
}

fn fwht_normalized(values: &mut [f32]) {
    let n = values.len();
    debug_assert!(is_power_of_two(n));
    let mut width = 1;
    while width < n {
        let step = width * 2;
        let mut base = 0;
        while base < n {
            for i in 0..width {
                let a = values[base + i];
                let b = values[base + width + i];
                values[base + i] = a + b;
                values[base + width + i] = a - b;
            }
            base += step;
        }
        width = step;
    }

    let scale = 1.0 / (n as f32).sqrt();
    for value in values.iter_mut() {
        *value *= scale;
    }
}

/// Deterministically project a ContinuousHV to a lower power-of-two dimension.
///
/// The transform is orthogonal before coefficient truncation. Because truncation
/// discards information, callers must treat the result as a candidate
/// representation rather than a semantics-preserving conversion.
pub fn project(
    source: &ContinuousHV,
    target_dim: usize,
    family: ContinuousProjectionFamily,
) -> Result<ContinuousHV, ContinuousProjectionError> {
    let source_dim = source.dim();

    if source_dim == target_dim {
        return Err(ContinuousProjectionError::SameDimension);
    }
    if target_dim == 0 || target_dim > source_dim {
        return Err(ContinuousProjectionError::ExpansionUnsupported);
    }
    if !is_power_of_two(source_dim) || !is_power_of_two(target_dim) {
        return Err(ContinuousProjectionError::NonPowerOfTwo);
    }
    if target_dim > source_dim {
        return Err(ContinuousProjectionError::InvalidTarget);
    }

    match family {
        ContinuousProjectionFamily::HadamardTruncateV1 => {
            let mut transformed = source.as_slice().to_vec();
            fwht_normalized(&mut transformed);
            Ok(ContinuousHV::from_slice(&transformed[..target_dim]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_projection_is_deterministic() {
        let source = ContinuousHV::random(4096, 0x4841442d5631);
        let a = project(&source, 1024, ContinuousProjectionFamily::HadamardTruncateV1)
            .expect("valid projection");
        let b = project(&source, 1024, ContinuousProjectionFamily::HadamardTruncateV1)
            .expect("valid projection");
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn hadamard_transform_preserves_energy_before_truncation() {
        let mut values = ContinuousHV::random(4096, 0x4841442d5632).values;
        let before: f32 = values.iter().map(|v| v * v).sum();
        fwht_normalized(&mut values);
        let after: f32 = values.iter().map(|v| v * v).sum();
        let relative = (before - after).abs() / before.max(1e-12);
        assert!(relative < 1e-5, "relative energy error={relative}");
    }

    #[test]
    fn projection_rejects_expansion() {
        let source = ContinuousHV::random(1024, 7);
        assert_eq!(
            project(&source, 2048, ContinuousProjectionFamily::HadamardTruncateV1),
            Err(ContinuousProjectionError::ExpansionUnsupported)
        );
    }

    #[test]
    fn projection_rejects_non_power_of_two_dimensions() {
        let source = ContinuousHV::random(3000, 7);
        assert_eq!(
            project(&source, 1000, ContinuousProjectionFamily::HadamardTruncateV1),
            Err(ContinuousProjectionError::NonPowerOfTwo)
        );
    }
}
