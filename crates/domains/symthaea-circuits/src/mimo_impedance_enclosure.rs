//! Conservative hard-bound enclosures for MIMO impedance matrix elements.
//!
//! Each voltage/current phasor is treated as a closed complex disc. V0.1
//! encloses each quotient with axis-aligned numerator rectangles and an
//! independent radial denominator interval. The resulting rectangle is
//! conservative but may be wider than a disk/sector or affine enclosure.
//! Statistical uncertainty is never accepted as a deterministic hard bound.

#![forbid(unsafe_code)]

use crate::mimo_impedance_passport::{
    ComplexValue, EvidenceClass, ImpedanceMatrixPointEstimate, MimoImpedanceSweep,
    PortMatrixError, matrix_at_exact_frequency, validate_mimo_impedance_sweep,
};

/// Version of the conservative complex quotient enclosure algorithm.
pub const ENCLOSURE_ALGORITHM_VERSION: &str =
    "mimo-complex-disc-rectangular-quotient-v0.1";

/// Whether supplied phasor error radii are deterministic hard bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorBoundSemantics {
    /// The method reference must identify the receipt/procedure that justifies
    /// each radius as a deterministic outer bound.
    DeterministicHardBound { method_reference: String },
    /// Standard uncertainties, confidence intervals and statistical bounds
    /// are not promoted to deterministic enclosures by this API.
    StatisticalOnly { method_reference: String },
}

/// Dependence assumption used for numerator and denominator uncertainty sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UncertaintyDependenceAssumption {
    /// Voltage and current discs are treated as independent sets. The output
    /// rectangle is an outer enclosure; correlation may make it conservative.
    IndependentDiscsOuterEnclosure,
}

/// A finite closed interval. Endpoints are expanded outwards by one binary64
/// representable number after each arithmetic operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericInterval {
    pub lower: f64,
    pub upper: f64,
}

impl NumericInterval {
    fn new(lower: f64, upper: f64) -> Result<Self, MatrixEnclosureError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(MatrixEnclosureError::NonFiniteOrInvalidInterval);
        }
        Ok(Self { lower, upper })
    }

    fn point(value: f64) -> Result<Self, MatrixEnclosureError> {
        Self::new(value, value)
    }

    fn around(center: f64, radius: f64) -> Result<Self, MatrixEnclosureError> {
        if !center.is_finite() || !radius.is_finite() || radius < 0.0 {
            return Err(MatrixEnclosureError::NonFiniteOrInvalidInterval);
        }
        let lower = outward_down(center - radius)?;
        let upper = outward_up(center + radius)?;
        Self::new(lower, upper)
    }

    fn add(self, rhs: Self) -> Result<Self, MatrixEnclosureError> {
        Self::new(
            outward_down(self.lower + rhs.lower)?,
            outward_up(self.upper + rhs.upper)?,
        )
    }

    fn subtract(self, rhs: Self) -> Result<Self, MatrixEnclosureError> {
        Self::new(
            outward_down(self.lower - rhs.upper)?,
            outward_up(self.upper - rhs.lower)?,
        )
    }

    fn multiply(self, rhs: Self) -> Result<Self, MatrixEnclosureError> {
        let products = [
            self.lower * rhs.lower,
            self.lower * rhs.upper,
            self.upper * rhs.lower,
            self.upper * rhs.upper,
        ];
        if products.iter().any(|value| !value.is_finite()) {
            return Err(MatrixEnclosureError::NonFiniteOrInvalidInterval);
        }
        let lower = products.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = products
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Self::new(outward_down(lower)?, outward_up(upper)?)
    }

    fn divide_by_positive(self, denominator: Self) -> Result<Self, MatrixEnclosureError> {
        if denominator.lower <= 0.0 {
            return Err(MatrixEnclosureError::DenominatorIntervalIncludesZero);
        }
        let reciprocal = Self::new(
            outward_down(1.0 / denominator.upper)?,
            outward_up(1.0 / denominator.lower)?,
        )?;
        self.multiply(reciprocal)
    }
}

/// Complex impedance enclosure for one observed-port / injected-port pair.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexImpedanceEnclosure {
    pub observed_port_id: String,
    pub driven_port_id: String,
    /// Nominal V/I point estimate; kept separate from its enclosure.
    pub nominal_ohm: ComplexValue,
    pub real_ohm: NumericInterval,
    pub imaginary_ohm: NumericInterval,
    pub voltage_error_radius_v: f64,
    pub current_error_radius_a: f64,
    /// Conservative lower/upper bounds on |I| over the current disc.
    pub current_magnitude_a: NumericInterval,
    /// Conservative lower/upper bounds on |I|² over the current disc.
    pub current_magnitude_squared_a2: NumericInterval,
}

/// Complete matrix enclosure at one exact frequency knot.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpedanceMatrixEnclosure {
    pub angular_frequency_rad_s: f64,
    pub port_ids: Vec<String>,
    /// Row-major [observed voltage port][injected current port].
    pub impedance_ohm: Vec<Vec<ComplexImpedanceEnclosure>>,
    pub configuration_digest: String,
    pub operating_point_digest: String,
    pub sweep_digest: String,
    pub evidence_class: EvidenceClass,
    pub hard_bound_method_reference: String,
    pub dependence_assumption: UncertaintyDependenceAssumption,
    pub algorithm_version: String,
}

/// All exact measured frequency-knot enclosures for one passport.
#[derive(Debug, Clone, PartialEq)]
pub struct MimoImpedanceSweepEnclosure {
    pub configuration_digest: String,
    pub operating_point_digest: String,
    pub sweep_digest: String,
    pub evidence_class: EvidenceClass,
    pub hard_bound_method_reference: String,
    pub dependence_assumption: UncertaintyDependenceAssumption,
    pub algorithm_version: String,
    pub frequency_matrices: Vec<ImpedanceMatrixEnclosure>,
}

/// Why a matrix enclosure could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixEnclosureError {
    Passport(PortMatrixError),
    StatisticalOnlyBounds { method_reference: String },
    MissingHardBoundMethodReference,
    UnsupportedFrequencyKnot,
    CurrentUncertaintyIncludesZero {
        frequency_index: usize,
        driven_port_id: String,
    },
    DenominatorIntervalIncludesZero,
    NonFiniteOrInvalidInterval,
    ElementArithmeticFailure {
        frequency_index: usize,
        observed_port_id: String,
        driven_port_id: String,
    },
}

impl From<PortMatrixError> for MatrixEnclosureError {
    fn from(value: PortMatrixError) -> Self {
        Self::Passport(value)
    }
}

/// Compute a complete conservative impedance matrix enclosure at one exact
/// recorded frequency. Missing matrix terms, off-knot queries and
/// non-deterministic uncertainty semantics fail closed.
pub fn enclose_mimo_impedance_matrix_at_exact_frequency(
    sweep: &MimoImpedanceSweep,
    angular_frequency_rad_s: f64,
    semantics: &ErrorBoundSemantics,
) -> Result<ImpedanceMatrixEnclosure, MatrixEnclosureError> {
    let method_reference = validate_semantics(semantics)?;
    validate_mimo_impedance_sweep(sweep)?;

    let frequency_index = sweep
        .frequency_slices
        .iter()
        .position(|slice| slice.angular_frequency_rad_s == angular_frequency_rad_s)
        .ok_or(MatrixEnclosureError::UnsupportedFrequencyKnot)?;
    let slice = &sweep.frequency_slices[frequency_index];
    let nominal =
        matrix_at_exact_frequency(sweep, angular_frequency_rad_s).map_err(MatrixEnclosureError::Passport)?;

    build_matrix_enclosure(
        sweep,
        frequency_index,
        slice,
        &nominal,
        method_reference,
    )
}

/// Enclose every recorded frequency knot. No interpolation or extrapolation
/// is performed; one invalid element blocks the entire sweep enclosure.
pub fn enclose_mimo_impedance_sweep(
    sweep: &MimoImpedanceSweep,
    semantics: &ErrorBoundSemantics,
) -> Result<MimoImpedanceSweepEnclosure, MatrixEnclosureError> {
    let method_reference = validate_semantics(semantics)?;
    validate_mimo_impedance_sweep(sweep)?;
    let evidence_class =
        validate_mimo_impedance_sweep(sweep)?.evidence_class;

    let mut frequency_matrices = Vec::with_capacity(sweep.frequency_slices.len());
    for (frequency_index, slice) in sweep.frequency_slices.iter().enumerate() {
        let nominal = matrix_at_exact_frequency(sweep, slice.angular_frequency_rad_s)
            .map_err(MatrixEnclosureError::Passport)?;
        frequency_matrices.push(build_matrix_enclosure(
            sweep,
            frequency_index,
            slice,
            &nominal,
            &method_reference,
        )?);
    }

    Ok(MimoImpedanceSweepEnclosure {
        configuration_digest: sweep.configuration_digest.clone(),
        operating_point_digest: sweep.operating_point_digest.clone(),
        sweep_digest: sweep.sweep_digest.clone(),
        evidence_class,
        hard_bound_method_reference: method_reference,
        dependence_assumption:
            UncertaintyDependenceAssumption::IndependentDiscsOuterEnclosure,
        algorithm_version: ENCLOSURE_ALGORITHM_VERSION.to_string(),
        frequency_matrices,
    })
}

fn validate_semantics(
    semantics: &ErrorBoundSemantics,
) -> Result<String, MatrixEnclosureError> {
    match semantics {
        ErrorBoundSemantics::StatisticalOnly { method_reference } => {
            Err(MatrixEnclosureError::StatisticalOnlyBounds {
                method_reference: method_reference.clone(),
            })
        }
        ErrorBoundSemantics::DeterministicHardBound { method_reference } => {
            if method_reference.trim().is_empty() {
                Err(MatrixEnclosureError::MissingHardBoundMethodReference)
            } else {
                Ok(method_reference.clone())
            }
        }
    }
}

fn build_matrix_enclosure(
    sweep: &MimoImpedanceSweep,
    frequency_index: usize,
    slice: &crate::mimo_impedance_passport::FrequencySlice,
    nominal: &ImpedanceMatrixPointEstimate,
    method_reference: &str,
) -> Result<ImpedanceMatrixEnclosure, MatrixEnclosureError> {
    let port_ids = &nominal.port_ids;
    let n = port_ids.len();
    let mut matrix: Vec<Vec<Option<ComplexImpedanceEnclosure>>> =
        (0..n).map(|_| (0..n).map(|_| None).collect()).collect();

    for column in &slice.columns {
        let column_index = port_ids
            .iter()
            .position(|port_id| port_id == &column.driven_port_id)
            .ok_or_else(|| {
                MatrixEnclosureError::Passport(PortMatrixError::UnknownDrivenPort {
                    frequency_index,
                    port_id: column.driven_port_id.clone(),
                })
            })?;

        let denominator_squared = denominator_squared_interval(
            column.injected_current_a,
            column.absolute_error_bound_a,
        )
        .map_err(|error| match error {
            MatrixEnclosureError::CurrentUncertaintyIncludesZero {
                frequency_index: _,
                driven_port_id: _,
            } => error,
            _ => MatrixEnclosureError::ElementArithmeticFailure {
                frequency_index,
                observed_port_id: "<denominator>".to_string(),
                driven_port_id: column.driven_port_id.clone(),
            },
        })?;
        let current_magnitude = NumericInterval::new(
            denominator_squared.lower.sqrt().max(0.0),
            outward_up(denominator_squared.upper.sqrt())?,
        )?;

        for observation in &column.voltage_observations {
            let row_index = port_ids
                .iter()
                .position(|port_id| port_id == &observation.observed_port_id)
                .ok_or_else(|| {
                    MatrixEnclosureError::Passport(PortMatrixError::UnknownObservedPort {
                        frequency_index,
                        driven_port_id: column.driven_port_id.clone(),
                        observed_port_id: observation.observed_port_id.clone(),
                    })
                })?;

            let element = enclose_complex_quotient(
                observation.response_voltage_v,
                observation.absolute_error_bound_v,
                column.injected_current_a,
                column.absolute_error_bound_a,
                denominator_squared,
                current_magnitude,
            )
            .map_err(|_| MatrixEnclosureError::ElementArithmeticFailure {
                frequency_index,
                observed_port_id: observation.observed_port_id.clone(),
                driven_port_id: column.driven_port_id.clone(),
            })?;

            matrix[row_index][column_index] = Some(ComplexImpedanceEnclosure {
                observed_port_id: observation.observed_port_id.clone(),
                driven_port_id: column.driven_port_id.clone(),
                nominal_ohm: nominal.impedance_ohm[row_index][column_index],
                real_ohm: element.0,
                imaginary_ohm: element.1,
                voltage_error_radius_v: observation.absolute_error_bound_v,
                current_error_radius_a: column.absolute_error_bound_a,
                current_magnitude_a: current_magnitude,
                current_magnitude_squared_a2: denominator_squared,
            });
        }
    }

    let mut complete = Vec::with_capacity(n);
    for (row_index, row) in matrix.into_iter().enumerate() {
        let mut complete_row = Vec::with_capacity(n);
        for (column_index, element) in row.into_iter().enumerate() {
            complete_row.push(element.ok_or_else(|| {
                MatrixEnclosureError::ElementArithmeticFailure {
                    frequency_index,
                    observed_port_id: port_ids[row_index].clone(),
                    driven_port_id: port_ids[column_index].clone(),
                }
            })?);
        }
        complete.push(complete_row);
    }

    Ok(ImpedanceMatrixEnclosure {
        angular_frequency_rad_s: slice.angular_frequency_rad_s,
        port_ids: port_ids.clone(),
        impedance_ohm: complete,
        configuration_digest: sweep.configuration_digest.clone(),
        operating_point_digest: sweep.operating_point_digest.clone(),
        sweep_digest: sweep.sweep_digest.clone(),
        evidence_class: nominal.evidence_class,
        hard_bound_method_reference: method_reference.to_string(),
        dependence_assumption:
            UncertaintyDependenceAssumption::IndependentDiscsOuterEnclosure,
        algorithm_version: ENCLOSURE_ALGORITHM_VERSION.to_string(),
    })
}

fn enclose_complex_quotient(
    voltage: ComplexValue,
    voltage_radius: f64,
    current: ComplexValue,
    _current_radius: f64,
    denominator_squared: NumericInterval,
    _current_magnitude: NumericInterval,
) -> Result<(NumericInterval, NumericInterval), MatrixEnclosureError> {
    let voltage_real = NumericInterval::around(voltage.real, voltage_radius)?;
    let voltage_imag = NumericInterval::around(voltage.imag, voltage_radius)?;
    let current_real = NumericInterval::around(current.real, _current_radius)?;
    let current_imag = NumericInterval::around(current.imag, _current_radius)?;

    // (V/I) = ((Vr*Ir + Vi*Ii) + j(Vi*Ir - Vr*Ii)) / (Ir² + Ii²).
    // Rectangle supersets intentionally ignore phasor-coordinate dependence.
    let numerator_real = voltage_real
        .multiply(current_real)?
        .add(voltage_imag.multiply(current_imag)?)?;
    let numerator_imag = voltage_imag
        .multiply(current_real)?
        .subtract(voltage_real.multiply(current_imag)?)?;

    Ok((
        numerator_real.divide_by_positive(denominator_squared)?,
        numerator_imag.divide_by_positive(denominator_squared)?,
    ))
}

/// Bound |I|² for every current phasor in the disc D(current, radius).
fn denominator_squared_interval(
    current: ComplexValue,
    radius: f64,
) -> Result<NumericInterval, MatrixEnclosureError> {
    if !current.real.is_finite()
        || !current.imag.is_finite()
        || !radius.is_finite()
        || radius < 0.0
    {
        return Err(MatrixEnclosureError::NonFiniteOrInvalidInterval);
    }

    let real = NumericInterval::point(current.real)?;
    let imag = NumericInterval::point(current.imag)?;
    let norm_squared = real
        .multiply(real)?
        .add(imag.multiply(imag)?)?;
    let norm_lower = outward_down(norm_squared.lower.max(0.0).sqrt())?.max(0.0);
    let norm_upper = outward_up(norm_squared.upper.max(0.0).sqrt())?;

    let magnitude_lower = outward_down(norm_lower - radius)?;
    let magnitude_upper = outward_up(norm_upper + radius)?;
    if magnitude_lower <= 0.0 {
        return Err(MatrixEnclosureError::CurrentUncertaintyIncludesZero {
            frequency_index: usize::MAX,
            driven_port_id: String::new(),
        });
    }

    let squared_lower = outward_down(magnitude_lower * magnitude_lower)?;
    let squared_upper = outward_up(magnitude_upper * magnitude_upper)?;
    if squared_lower <= 0.0 {
        return Err(MatrixEnclosureError::DenominatorIntervalIncludesZero);
    }
    NumericInterval::new(squared_lower, squared_upper)
}

fn outward_down(value: f64) -> Result<f64, MatrixEnclosureError> {
    if !value.is_finite() {
        return Err(MatrixEnclosureError::NonFiniteOrInvalidInterval);
    }
    let rounded = next_down(value);
    if rounded.is_finite() {
        Ok(rounded)
    } else {
        Err(MatrixEnclosureError::NonFiniteOrInvalidInterval)
    }
}

fn outward_up(value: f64) -> Result<f64, MatrixEnclosureError> {
    if !value.is_finite() {
        return Err(MatrixEnclosureError::NonFiniteOrInvalidInterval);
    }
    let rounded = next_up(value);
    if rounded.is_finite() {
        Ok(rounded)
    } else {
        Err(MatrixEnclosureError::NonFiniteOrInvalidInterval)
    }
}

fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits((1_u64 << 63) | 1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mimo_impedance_passport::{
        EvidenceOrigin, ExcitationColumn, FrequencySlice, PortDescriptor, VoltageObservation,
    };

    fn observed(id: &str, real: f64, imag: f64, radius: f64) -> VoltageObservation {
        VoltageObservation {
            observed_port_id: id.to_string(),
            response_voltage_v: ComplexValue { real, imag },
            absolute_error_bound_v: radius,
        }
    }

    fn column(
        driven: &str,
        current_real: f64,
        current_imag: f64,
        current_radius: f64,
        observations: Vec<VoltageObservation>,
    ) -> ExcitationColumn {
        ExcitationColumn {
            driven_port_id: driven.to_string(),
            injected_current_a: ComplexValue {
                real: current_real,
                imag: current_imag,
            },
            absolute_error_bound_a: current_radius,
            excitation_digest: format!("excitation-{driven}"),
            column_payload_digest: format!("payload-{driven}"),
            voltage_observations: observations,
        }
    }

    fn fixture() -> MimoImpedanceSweep {
        MimoImpedanceSweep {
            schema_version: "0.2.0".to_string(),
            configuration_digest: "sha256:uq-config".to_string(),
            operating_point_digest: "sha256:uq-op".to_string(),
            sweep_digest: "sha256:uq-sweep".to_string(),
            ports: vec![
                PortDescriptor {
                    port_id: "bus-a".to_string(),
                    voltage_positive_reference: "bus-a-return".to_string(),
                    current_positive_direction: "into-bus-a".to_string(),
                },
                PortDescriptor {
                    port_id: "bus-b".to_string(),
                    voltage_positive_reference: "bus-b-return".to_string(),
                    current_positive_direction: "into-bus-b".to_string(),
                },
            ],
            origin: EvidenceOrigin::AnalyticalReference {
                derivation_digest: "sha256:uq-fixture".to_string(),
            },
            frequency_slices: vec![FrequencySlice {
                angular_frequency_rad_s: 0.0,
                columns: vec![
                    column(
                        "bus-a",
                        2.0,
                        0.0,
                        0.25,
                        vec![
                            observed("bus-a", 4.0, 2.0, 0.5),
                            observed("bus-b", 1.0, -2.0, 0.25),
                        ],
                    ),
                    column(
                        "bus-b",
                        4.0,
                        0.0,
                        0.5,
                        vec![
                            observed("bus-a", 4.0, 8.0, 0.5),
                            observed("bus-b", 12.0, -4.0, 0.5),
                        ],
                    ),
                ],
            }],
        }
    }

    fn semantics() -> ErrorBoundSemantics {
        ErrorBoundSemantics::DeterministicHardBound {
            method_reference: "sha256:hard-bound-method".to_string(),
        }
    }

    fn assert_contains(actual: NumericInterval, lower: f64, upper: f64) {
        assert!(
            actual.lower <= lower + 1.0e-12,
            "actual lower {} must contain expected lower {}",
            actual.lower,
            lower
        );
        assert!(
            actual.upper >= upper - 1.0e-12,
            "actual upper {} must contain expected upper {}",
            actual.upper,
            upper
        );
    }

    #[test]
    fn rational_known_answer_bounds_are_contained_for_all_four_matrix_elements() {
        let enclosure =
            enclose_mimo_impedance_matrix_at_exact_frequency(&fixture(), 0.0, &semantics())
                .unwrap();
        assert_eq!(enclosure.impedance_ohm.len(), 2);
        assert_eq!(enclosure.impedance_ohm[0].len(), 2);

        // Exact Fraction oracle answers are documented in the companion JSON.
        assert_contains(
            enclosure.impedance_ohm[0][0].real_ohm,
            88.0 / 81.0,
            172.0 / 49.0,
        );
        assert_contains(
            enclosure.impedance_ohm[0][0].imaginary_ohm,
            8.0 / 27.0,
            108.0 / 49.0,
        );
        assert_contains(
            enclosure.impedance_ohm[1][0].real_ohm,
            4.0 / 27.0,
            54.0 / 49.0,
        );
        assert_contains(
            enclosure.impedance_ohm[1][0].imaginary_ohm,
            -86.0 / 49.0,
            -44.0 / 81.0,
        );
        assert_contains(
            enclosure.impedance_ohm[0][1].real_ohm,
            32.0 / 81.0,
            2.0,
        );
        assert_contains(
            enclosure.impedance_ohm[0][1].imaginary_ohm,
            32.0 / 27.0,
            162.0 / 49.0,
        );
        assert_contains(
            enclosure.impedance_ohm[1][1].real_ohm,
            152.0 / 81.0,
            234.0 / 49.0,
        );
        assert_contains(
            enclosure.impedance_ohm[1][1].imaginary_ohm,
            -106.0 / 49.0,
            -8.0 / 27.0,
        );
    }

    #[test]
    fn output_preserves_nominal_and_evidence_identity() {
        let enclosure =
            enclose_mimo_impedance_matrix_at_exact_frequency(&fixture(), 0.0, &semantics())
                .unwrap();
        assert_eq!(
            enclosure.impedance_ohm[0][0].nominal_ohm,
            ComplexValue {
                real: 2.0,
                imag: 1.0
            }
        );
        assert_eq!(enclosure.configuration_digest, "sha256:uq-config");
        assert_eq!(enclosure.operating_point_digest, "sha256:uq-op");
        assert_eq!(enclosure.sweep_digest, "sha256:uq-sweep");
        assert_eq!(
            enclosure.dependence_assumption,
            UncertaintyDependenceAssumption::IndependentDiscsOuterEnclosure
        );
    }

    #[test]
    fn all_frequency_knots_are_enclosed_without_interpolation() {
        let mut sweep = fixture();
        let mut second = sweep.frequency_slices[0].clone();
        second.angular_frequency_rad_s = 10.0;
        sweep.frequency_slices.push(second);
        let result = enclose_mimo_impedance_sweep(&sweep, &semantics()).unwrap();
        assert_eq!(result.frequency_matrices.len(), 2);
        assert_eq!(result.frequency_matrices[1].angular_frequency_rad_s, 10.0);
    }

    #[test]
    fn statistical_uncertainty_is_not_promoted_to_a_hard_enclosure() {
        let semantics = ErrorBoundSemantics::StatisticalOnly {
            method_reference: "one-sigma covariance".to_string(),
        };
        assert!(matches!(
            enclose_mimo_impedance_matrix_at_exact_frequency(&fixture(), 0.0, &semantics),
            Err(MatrixEnclosureError::StatisticalOnlyBounds { .. })
        ));
    }

    #[test]
    fn missing_hard_bound_method_reference_is_rejected() {
        let semantics = ErrorBoundSemantics::DeterministicHardBound {
            method_reference: " ".to_string(),
        };
        assert_eq!(
            enclose_mimo_impedance_matrix_at_exact_frequency(&fixture(), 0.0, &semantics),
            Err(MatrixEnclosureError::MissingHardBoundMethodReference)
        );
    }

    #[test]
    fn current_disc_touching_zero_is_not_evaluable() {
        let mut sweep = fixture();
        sweep.frequency_slices[0].columns[0].injected_current_a =
            ComplexValue { real: 1.0, imag: 0.0 };
        sweep.frequency_slices[0].columns[0].absolute_error_bound_a = 1.0;
        assert!(matches!(
            enclose_mimo_impedance_matrix_at_exact_frequency(&sweep, 0.0, &semantics()),
            Err(MatrixEnclosureError::Passport(
                PortMatrixError::ExcitationUncertaintyCoversZero { .. }
            ))
        ));
    }

    #[test]
    fn off_knot_query_is_rejected() {
        assert_eq!(
            enclose_mimo_impedance_matrix_at_exact_frequency(&fixture(), 1.0, &semantics()),
            Err(MatrixEnclosureError::UnsupportedFrequencyKnot)
        );
    }

    #[test]
    fn tiny_current_that_underflows_squared_denominator_is_not_evaluable() {
        let mut sweep = fixture();
        sweep.frequency_slices[0].columns[0].injected_current_a =
            ComplexValue { real: 1.0e-300, imag: 0.0 };
        sweep.frequency_slices[0].columns[0].absolute_error_bound_a = 0.0;
        assert!(matches!(
            enclose_mimo_impedance_matrix_at_exact_frequency(&sweep, 0.0, &semantics()),
            Err(MatrixEnclosureError::ElementArithmeticFailure { .. })
                | Err(MatrixEnclosureError::DenominatorIntervalIncludesZero)
        ));
    }

    #[test]
    fn deterministic_samples_inside_voltage_and_current_discs_are_contained() {
        let sweep = fixture();
        let enclosure =
            enclose_mimo_impedance_matrix_at_exact_frequency(&sweep, 0.0, &semantics()).unwrap();
        let element = &enclosure.impedance_ohm[0][0];
        let v0 = ComplexValue { real: 4.0, imag: 2.0 };
        let i0 = ComplexValue { real: 2.0, imag: 0.0 };
        let rv = 0.5_f64;
        let ri = 0.25_f64;

        for k in 0..=64 {
            for radial in [0.0_f64, 0.5, 1.0] {
                let angle_v = (k as f64) * std::f64::consts::TAU / 65.0;
                let angle_i = (k as f64 + 0.37) * std::f64::consts::TAU / 65.0;
                let voltage = ComplexValue {
                    real: v0.real + rv * radial * angle_v.cos(),
                    imag: v0.imag + rv * radial * angle_v.sin(),
                };
                let current = ComplexValue {
                    real: i0.real + ri * radial * angle_i.cos(),
                    imag: i0.imag + ri * radial * angle_i.sin(),
                };
                let denominator = current.real * current.real + current.imag * current.imag;
                let ratio = ComplexValue {
                    real: (voltage.real * current.real + voltage.imag * current.imag)
                        / denominator,
                    imag: (voltage.imag * current.real - voltage.real * current.imag)
                        / denominator,
                };
                assert!(ratio.real >= element.real_ohm.lower);
                assert!(ratio.real <= element.real_ohm.upper);
                assert!(ratio.imag >= element.imaginary_ohm.lower);
                assert!(ratio.imag <= element.imaginary_ohm.upper);
            }
        }
    }
}
