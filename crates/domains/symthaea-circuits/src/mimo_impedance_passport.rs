//! Evidence passport and fail-closed validator for measured/modelled MIMO
//! impedance sweeps.
//!
//! A complete impedance matrix is assembled column-by-column: each column is
//! one injected current port and every output voltage port is observed at the
//! same declared operating point and exact frequency knot. Missing entries
//! are never synthesized as zero. V0.1 performs exact-knot lookup only and
//! does not authorize interpolation or extrapolation.

#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexValue {
    pub real: f64,
    pub imag: f64,
}

impl ComplexValue {
    const ZERO: Self = Self {
        real: 0.0,
        imag: 0.0,
    };

    fn divide(self, rhs: Self) -> Result<Self, PortMatrixError> {
        let denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(PortMatrixError::ZeroOrUnrepresentableExcitation);
        }
        let result = Self {
            real: (self.real * rhs.real + self.imag * rhs.imag) / denominator,
            imag: (self.imag * rhs.real - self.real * rhs.imag) / denominator,
        };
        if !result.real.is_finite() || !result.imag.is_finite() {
            return Err(PortMatrixError::NonFiniteDerivedImpedance);
        }
        Ok(result)
    }

    fn is_finite(self) -> bool {
        self.real.is_finite() && self.imag.is_finite()
    }
}

/// Stable physical port identity and sign conventions for Z = V/I.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortDescriptor {
    pub port_id: String,
    /// Node/reference defining positive small-signal port voltage.
    pub voltage_positive_reference: String,
    /// Explicit physical direction defining positive injected current.
    pub current_positive_direction: String,
}

/// Evidence origin is explicit. Synthetic or simulated values cannot be
/// silently relabelled as measurements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceOrigin {
    Measured {
        instrument_id: String,
        calibration_certificate_digest: String,
        channel_map_digest: String,
        synchronization_digest: String,
    },
    Simulated {
        simulator_id: String,
        simulator_version: String,
        model_digest: String,
    },
    AnalyticalReference {
        derivation_digest: String,
    },
}

/// A voltage response measured at one output port during one driven-port
/// excitation. Error bounds are retained as supplied; this module does not
/// promote them to a propagated complex-impedance enclosure.
#[derive(Debug, Clone, PartialEq)]
pub struct VoltageObservation {
    pub observed_port_id: String,
    pub response_voltage_v: ComplexValue,
    pub absolute_error_bound_v: f64,
}

/// One matrix column: inject current at driven_port_id, observe every output
/// port voltage, then form Z_ij = V_i / I_j.
#[derive(Debug, Clone, PartialEq)]
pub struct ExcitationColumn {
    pub driven_port_id: String,
    pub injected_current_a: ComplexValue,
    pub absolute_error_bound_a: f64,
    /// Digest of the injected waveform and its amplitude/phase setup.
    pub excitation_digest: String,
    /// For measured origin, identifies the raw acquisition bundle. For other
    /// origins, identifies that column's result payload.
    pub column_payload_digest: String,
    pub voltage_observations: Vec<VoltageObservation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrequencySlice {
    pub angular_frequency_rad_s: f64,
    pub columns: Vec<ExcitationColumn>,
}

/// Full sweep passport. All columns and frequency knots share this exact
/// configuration and operating-point identity.
#[derive(Debug, Clone, PartialEq)]
pub struct MimoImpedanceSweep {
    pub schema_version: String,
    pub configuration_digest: String,
    pub operating_point_digest: String,
    pub sweep_digest: String,
    pub ports: Vec<PortDescriptor>,
    pub origin: EvidenceOrigin,
    pub frequency_slices: Vec<FrequencySlice>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortMatrixError {
    EmptyIdentity(&'static str),
    NoPorts,
    DuplicatePortId(String),
    InvalidPortDescriptor(String),
    NoFrequencySlices,
    InvalidFrequency(usize),
    FrequenciesNotStrictlyIncreasing,
    MissingColumns {
        frequency_index: usize,
    },
    DuplicateDrivenPort {
        frequency_index: usize,
        port_id: String,
    },
    UnknownDrivenPort {
        frequency_index: usize,
        port_id: String,
    },
    MissingOutputObservation {
        frequency_index: usize,
        driven_port_id: String,
        observed_port_id: String,
    },
    DuplicateOutputObservation {
        frequency_index: usize,
        driven_port_id: String,
        observed_port_id: String,
    },
    UnknownObservedPort {
        frequency_index: usize,
        driven_port_id: String,
        observed_port_id: String,
    },
    InvalidExcitation {
        frequency_index: usize,
        driven_port_id: String,
    },
    ExcitationUncertaintyCoversZero {
        frequency_index: usize,
        driven_port_id: String,
    },
    InvalidUncertaintyBound {
        frequency_index: usize,
        driven_port_id: String,
        observed_port_id: Option<String>,
    },
    InvalidComplexObservation {
        frequency_index: usize,
        driven_port_id: String,
        observed_port_id: String,
    },
    EmptyColumnDigest {
        frequency_index: usize,
        driven_port_id: String,
    },
    MissingMeasuredCalibration(&'static str),
    MissingSimulationIdentity(&'static str),
    MissingAnalyticalIdentity,
    UnsupportedFrequencyKnot,
    InvalidQueryFrequency,
    ZeroOrUnrepresentableExcitation,
    NonFiniteDerivedImpedance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceClass {
    Measured,
    Simulated,
    AnalyticalReference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixValidationReport {
    pub port_count: usize,
    pub frequency_knot_count: usize,
    pub validated_matrix_element_count: usize,
    pub evidence_class: EvidenceClass,
}

/// Point estimates for one exact frequency knot. impedance_ohm entries are
/// derived V/I values; source voltage/current error bounds remain in the
/// sweep and are not a formally propagated complex-impedance enclosure.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpedanceMatrixPointEstimate {
    pub angular_frequency_rad_s: f64,
    pub port_ids: Vec<String>,
    /// Row-major [observed voltage port][injected current port].
    pub impedance_ohm: Vec<Vec<ComplexValue>>,
    pub configuration_digest: String,
    pub operating_point_digest: String,
    pub sweep_digest: String,
    pub evidence_class: EvidenceClass,
}

/// Validate the whole sweep: complete square matrix on every knot, stable
/// port identities, explicit provenance, and finite excitation/response data.
pub fn validate_mimo_impedance_sweep(
    sweep: &MimoImpedanceSweep,
) -> Result<MatrixValidationReport, PortMatrixError> {
    if sweep.schema_version.trim().is_empty() {
        return Err(PortMatrixError::EmptyIdentity("schema_version"));
    }
    if sweep.configuration_digest.trim().is_empty() {
        return Err(PortMatrixError::EmptyIdentity("configuration_digest"));
    }
    if sweep.operating_point_digest.trim().is_empty() {
        return Err(PortMatrixError::EmptyIdentity("operating_point_digest"));
    }
    if sweep.sweep_digest.trim().is_empty() {
        return Err(PortMatrixError::EmptyIdentity("sweep_digest"));
    }
    if sweep.ports.is_empty() {
        return Err(PortMatrixError::NoPorts);
    }

    let mut port_ids = Vec::with_capacity(sweep.ports.len());
    for port in &sweep.ports {
        if port.port_id.trim().is_empty()
            || port.voltage_positive_reference.trim().is_empty()
            || port.current_positive_direction.trim().is_empty()
        {
            return Err(PortMatrixError::InvalidPortDescriptor(port.port_id.clone()));
        }
        if port_ids.iter().any(|existing| existing == &port.port_id) {
            return Err(PortMatrixError::DuplicatePortId(port.port_id.clone()));
        }
        port_ids.push(port.port_id.clone());
    }

    let evidence_class = validate_origin(&sweep.origin)?;
    if sweep.frequency_slices.is_empty() {
        return Err(PortMatrixError::NoFrequencySlices);
    }

    let mut previous_frequency = None;
    for (frequency_index, slice) in sweep.frequency_slices.iter().enumerate() {
        let frequency = slice.angular_frequency_rad_s;
        if !frequency.is_finite() || frequency < 0.0 {
            return Err(PortMatrixError::InvalidFrequency(frequency_index));
        }
        if let Some(previous) = previous_frequency {
            if frequency <= previous {
                return Err(PortMatrixError::FrequenciesNotStrictlyIncreasing);
            }
        }
        previous_frequency = Some(frequency);

        if slice.columns.len() != port_ids.len() {
            return Err(PortMatrixError::MissingColumns { frequency_index });
        }

        let mut driven_ports: Vec<&str> = Vec::with_capacity(port_ids.len());
        for column in &slice.columns {
            let driven_id = column.driven_port_id.as_str();
            if !port_ids.iter().any(|known| known == driven_id) {
                return Err(PortMatrixError::UnknownDrivenPort {
                    frequency_index,
                    port_id: column.driven_port_id.clone(),
                });
            }
            if driven_ports.contains(&driven_id) {
                return Err(PortMatrixError::DuplicateDrivenPort {
                    frequency_index,
                    port_id: column.driven_port_id.clone(),
                });
            }
            driven_ports.push(driven_id);

            if column.excitation_digest.trim().is_empty()
                || column.column_payload_digest.trim().is_empty()
            {
                return Err(PortMatrixError::EmptyColumnDigest {
                    frequency_index,
                    driven_port_id: column.driven_port_id.clone(),
                });
            }
            if !column.injected_current_a.is_finite()
                || (column.injected_current_a.real == 0.0 && column.injected_current_a.imag == 0.0)
            {
                return Err(PortMatrixError::InvalidExcitation {
                    frequency_index,
                    driven_port_id: column.driven_port_id.clone(),
                });
            }
            if !column.absolute_error_bound_a.is_finite() || column.absolute_error_bound_a < 0.0 {
                return Err(PortMatrixError::InvalidUncertaintyBound {
                    frequency_index,
                    driven_port_id: column.driven_port_id.clone(),
                    observed_port_id: None,
                });
            }
            let injected_current_magnitude = column
                .injected_current_a
                .real
                .hypot(column.injected_current_a.imag);
            if !injected_current_magnitude.is_finite() {
                return Err(PortMatrixError::InvalidExcitation {
                    frequency_index,
                    driven_port_id: column.driven_port_id.clone(),
                });
            }
            // A circular absolute-error disk with radius >= |I| can include
            // zero current, making any hard-bound V/I impedance unbounded.
            if column.absolute_error_bound_a >= injected_current_magnitude {
                return Err(PortMatrixError::ExcitationUncertaintyCoversZero {
                    frequency_index,
                    driven_port_id: column.driven_port_id.clone(),
                });
            }

            if column.voltage_observations.len() != port_ids.len() {
                return Err(PortMatrixError::MissingOutputObservation {
                    frequency_index,
                    driven_port_id: column.driven_port_id.clone(),
                    observed_port_id: "<matrix-row-missing>".to_string(),
                });
            }

            let mut observed_ports: Vec<&str> = Vec::with_capacity(port_ids.len());
            for observation in &column.voltage_observations {
                let observed_id = observation.observed_port_id.as_str();
                if !port_ids.iter().any(|known| known == observed_id) {
                    return Err(PortMatrixError::UnknownObservedPort {
                        frequency_index,
                        driven_port_id: column.driven_port_id.clone(),
                        observed_port_id: observation.observed_port_id.clone(),
                    });
                }
                if observed_ports.contains(&observed_id) {
                    return Err(PortMatrixError::DuplicateOutputObservation {
                        frequency_index,
                        driven_port_id: column.driven_port_id.clone(),
                        observed_port_id: observation.observed_port_id.clone(),
                    });
                }
                observed_ports.push(observed_id);

                if !observation.response_voltage_v.is_finite() {
                    return Err(PortMatrixError::InvalidComplexObservation {
                        frequency_index,
                        driven_port_id: column.driven_port_id.clone(),
                        observed_port_id: observation.observed_port_id.clone(),
                    });
                }
                if !observation.absolute_error_bound_v.is_finite()
                    || observation.absolute_error_bound_v < 0.0
                {
                    return Err(PortMatrixError::InvalidUncertaintyBound {
                        frequency_index,
                        driven_port_id: column.driven_port_id.clone(),
                        observed_port_id: Some(observation.observed_port_id.clone()),
                    });
                }
            }

            for expected_id in &port_ids {
                if !observed_ports.contains(&expected_id.as_str()) {
                    return Err(PortMatrixError::MissingOutputObservation {
                        frequency_index,
                        driven_port_id: column.driven_port_id.clone(),
                        observed_port_id: expected_id.clone(),
                    });
                }
            }
        }

        for expected_id in &port_ids {
            if !driven_ports.contains(&expected_id.as_str()) {
                return Err(PortMatrixError::MissingColumns { frequency_index });
            }
        }
    }

    Ok(MatrixValidationReport {
        port_count: port_ids.len(),
        frequency_knot_count: sweep.frequency_slices.len(),
        validated_matrix_element_count: port_ids.len()
            * port_ids.len()
            * sweep.frequency_slices.len(),
        evidence_class,
    })
}

/// Return a V/I matrix only at an exact recorded frequency knot. No
/// interpolation, extrapolation, nearest-neighbour substitution, or gap fill.
pub fn matrix_at_exact_frequency(
    sweep: &MimoImpedanceSweep,
    angular_frequency_rad_s: f64,
) -> Result<ImpedanceMatrixPointEstimate, PortMatrixError> {
    validate_mimo_impedance_sweep(sweep)?;
    if !angular_frequency_rad_s.is_finite() || angular_frequency_rad_s < 0.0 {
        return Err(PortMatrixError::InvalidQueryFrequency);
    }
    let slice = sweep
        .frequency_slices
        .iter()
        .find(|candidate| candidate.angular_frequency_rad_s == angular_frequency_rad_s)
        .ok_or(PortMatrixError::UnsupportedFrequencyKnot)?;

    let port_ids: Vec<String> = sweep
        .ports
        .iter()
        .map(|port| port.port_id.clone())
        .collect();
    let n = port_ids.len();
    let mut matrix = vec![vec![ComplexValue::ZERO; n]; n];

    for column in &slice.columns {
        let column_index = port_ids
            .iter()
            .position(|id| id == &column.driven_port_id)
            .ok_or(PortMatrixError::UnknownDrivenPort {
                frequency_index: 0,
                port_id: column.driven_port_id.clone(),
            })?;

        for observation in &column.voltage_observations {
            let row_index = port_ids
                .iter()
                .position(|id| id == &observation.observed_port_id)
                .ok_or(PortMatrixError::UnknownObservedPort {
                    frequency_index: 0,
                    driven_port_id: column.driven_port_id.clone(),
                    observed_port_id: observation.observed_port_id.clone(),
                })?;
            matrix[row_index][column_index] = observation
                .response_voltage_v
                .divide(column.injected_current_a)?;
        }
    }

    let evidence_class = validate_origin(&sweep.origin)?;
    Ok(ImpedanceMatrixPointEstimate {
        angular_frequency_rad_s,
        port_ids,
        impedance_ohm: matrix,
        configuration_digest: sweep.configuration_digest.clone(),
        operating_point_digest: sweep.operating_point_digest.clone(),
        sweep_digest: sweep.sweep_digest.clone(),
        evidence_class,
    })
}

fn validate_origin(origin: &EvidenceOrigin) -> Result<EvidenceClass, PortMatrixError> {
    match origin {
        EvidenceOrigin::Measured {
            instrument_id,
            calibration_certificate_digest,
            channel_map_digest,
            synchronization_digest,
        } => {
            if instrument_id.trim().is_empty() {
                return Err(PortMatrixError::MissingMeasuredCalibration("instrument_id"));
            }
            if calibration_certificate_digest.trim().is_empty() {
                return Err(PortMatrixError::MissingMeasuredCalibration(
                    "calibration_certificate_digest",
                ));
            }
            if channel_map_digest.trim().is_empty() {
                return Err(PortMatrixError::MissingMeasuredCalibration(
                    "channel_map_digest",
                ));
            }
            if synchronization_digest.trim().is_empty() {
                return Err(PortMatrixError::MissingMeasuredCalibration(
                    "synchronization_digest",
                ));
            }
            Ok(EvidenceClass::Measured)
        }
        EvidenceOrigin::Simulated {
            simulator_id,
            simulator_version,
            model_digest,
        } => {
            if simulator_id.trim().is_empty() {
                return Err(PortMatrixError::MissingSimulationIdentity("simulator_id"));
            }
            if simulator_version.trim().is_empty() {
                return Err(PortMatrixError::MissingSimulationIdentity(
                    "simulator_version",
                ));
            }
            if model_digest.trim().is_empty() {
                return Err(PortMatrixError::MissingSimulationIdentity("model_digest"));
            }
            Ok(EvidenceClass::Simulated)
        }
        EvidenceOrigin::AnalyticalReference { derivation_digest } => {
            if derivation_digest.trim().is_empty() {
                return Err(PortMatrixError::MissingAnalyticalIdentity);
            }
            Ok(EvidenceClass::AnalyticalReference)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed(id: &str, real: f64, imag: f64) -> VoltageObservation {
        VoltageObservation {
            observed_port_id: id.to_string(),
            response_voltage_v: ComplexValue { real, imag },
            absolute_error_bound_v: 0.01,
        }
    }

    fn column(
        driven: &str,
        current_real: f64,
        current_imag: f64,
        observations: Vec<VoltageObservation>,
    ) -> ExcitationColumn {
        ExcitationColumn {
            driven_port_id: driven.to_string(),
            injected_current_a: ComplexValue {
                real: current_real,
                imag: current_imag,
            },
            absolute_error_bound_a: 0.005,
            excitation_digest: format!("excitation-{driven}"),
            column_payload_digest: format!("column-{driven}"),
            voltage_observations: observations,
        }
    }

    fn fixture() -> MimoImpedanceSweep {
        MimoImpedanceSweep {
            schema_version: "0.1.0".to_string(),
            configuration_digest: "sha256:aircraft-config".to_string(),
            operating_point_digest: "sha256:op-42".to_string(),
            sweep_digest: "sha256:sweep-7".to_string(),
            ports: vec![
                PortDescriptor {
                    port_id: "bus-a".to_string(),
                    voltage_positive_reference: "bus-a-to-return".to_string(),
                    current_positive_direction: "injected-into-bus-a".to_string(),
                },
                PortDescriptor {
                    port_id: "bus-b".to_string(),
                    voltage_positive_reference: "bus-b-to-return".to_string(),
                    current_positive_direction: "injected-into-bus-b".to_string(),
                },
            ],
            origin: EvidenceOrigin::AnalyticalReference {
                derivation_digest: "sha256:analytical-matrix".to_string(),
            },
            frequency_slices: vec![
                FrequencySlice {
                    angular_frequency_rad_s: 0.0,
                    columns: vec![
                        column(
                            "bus-a",
                            2.0,
                            0.0,
                            vec![observed("bus-a", 4.0, 2.0), observed("bus-b", 1.0, -2.0)],
                        ),
                        column(
                            "bus-b",
                            4.0,
                            0.0,
                            vec![observed("bus-a", 4.0, 8.0), observed("bus-b", 12.0, -4.0)],
                        ),
                    ],
                },
                FrequencySlice {
                    angular_frequency_rad_s: 10.0,
                    columns: vec![
                        column(
                            "bus-a",
                            2.0,
                            0.0,
                            vec![observed("bus-a", 4.0, 2.0), observed("bus-b", 1.0, -2.0)],
                        ),
                        column(
                            "bus-b",
                            4.0,
                            0.0,
                            vec![observed("bus-a", 4.0, 8.0), observed("bus-b", 12.0, -4.0)],
                        ),
                    ],
                },
            ],
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn complete_two_by_two_matrix_validates_and_derives_all_four_terms() {
        let sweep = fixture();
        let report = validate_mimo_impedance_sweep(&sweep).unwrap();
        assert_eq!(report.port_count, 2);
        assert_eq!(report.frequency_knot_count, 2);
        assert_eq!(report.validated_matrix_element_count, 8);
        assert_eq!(report.evidence_class, EvidenceClass::AnalyticalReference);

        let matrix = matrix_at_exact_frequency(&sweep, 0.0).unwrap();
        close(matrix.impedance_ohm[0][0].real, 2.0, 1e-12);
        close(matrix.impedance_ohm[0][0].imag, 1.0, 1e-12);
        close(matrix.impedance_ohm[1][0].real, 0.5, 1e-12);
        close(matrix.impedance_ohm[1][0].imag, -1.0, 1e-12);
        close(matrix.impedance_ohm[0][1].real, 1.0, 1e-12);
        close(matrix.impedance_ohm[0][1].imag, 2.0, 1e-12);
        close(matrix.impedance_ohm[1][1].real, 3.0, 1e-12);
        close(matrix.impedance_ohm[1][1].imag, -1.0, 1e-12);
    }

    #[test]
    fn exact_knot_lookup_rejects_interpolation_and_extrapolation() {
        let sweep = fixture();
        assert!(matrix_at_exact_frequency(&sweep, 10.0).is_ok());
        assert_eq!(
            matrix_at_exact_frequency(&sweep, 5.0),
            Err(PortMatrixError::UnsupportedFrequencyKnot)
        );
        assert_eq!(
            matrix_at_exact_frequency(&sweep, 11.0),
            Err(PortMatrixError::UnsupportedFrequencyKnot)
        );
    }

    #[test]
    fn missing_cross_port_response_is_not_promoted_to_zero() {
        let mut sweep = fixture();
        sweep.frequency_slices[0].columns[0]
            .voltage_observations
            .pop();
        assert!(matches!(
            validate_mimo_impedance_sweep(&sweep),
            Err(PortMatrixError::MissingOutputObservation { .. })
        ));
    }

    #[test]
    fn zero_measured_voltage_is_a_real_observation_not_a_missing_entry() {
        let mut sweep = fixture();
        sweep.frequency_slices[0].columns[0].voltage_observations[1].response_voltage_v =
            ComplexValue::ZERO;
        let matrix = matrix_at_exact_frequency(&sweep, 0.0).unwrap();
        assert_eq!(matrix.impedance_ohm[1][0], ComplexValue::ZERO);
    }

    #[test]
    fn measured_sweep_requires_calibration_channel_and_sync_identity() {
        let mut sweep = fixture();
        sweep.origin = EvidenceOrigin::Measured {
            instrument_id: "scope-12".to_string(),
            calibration_certificate_digest: String::new(),
            channel_map_digest: "sha256:channel-map".to_string(),
            synchronization_digest: "sha256:sync".to_string(),
        };
        assert_eq!(
            validate_mimo_impedance_sweep(&sweep),
            Err(PortMatrixError::MissingMeasuredCalibration(
                "calibration_certificate_digest"
            ))
        );
    }

    #[test]
    fn duplicate_driven_port_is_rejected() {
        let mut sweep = fixture();
        sweep.frequency_slices[0].columns[1].driven_port_id = "bus-a".to_string();
        assert!(matches!(
            validate_mimo_impedance_sweep(&sweep),
            Err(PortMatrixError::DuplicateDrivenPort { .. })
        ));
    }

    #[test]
    fn mismatched_or_empty_configuration_identity_is_rejected() {
        let mut sweep = fixture();
        sweep.operating_point_digest.clear();
        assert_eq!(
            validate_mimo_impedance_sweep(&sweep),
            Err(PortMatrixError::EmptyIdentity("operating_point_digest"))
        );
    }

    #[test]
    fn current_uncertainty_must_not_include_zero() {
        let mut sweep = fixture();
        sweep.frequency_slices[0].columns[0].absolute_error_bound_a = 2.0;
        assert!(matches!(
            validate_mimo_impedance_sweep(&sweep),
            Err(PortMatrixError::ExcitationUncertaintyCoversZero { .. })
        ));
    }

    #[test]
    fn zero_excitation_and_non_finite_response_are_rejected() {
        let mut sweep = fixture();
        sweep.frequency_slices[0].columns[0].injected_current_a = ComplexValue::ZERO;
        assert!(matches!(
            validate_mimo_impedance_sweep(&sweep),
            Err(PortMatrixError::InvalidExcitation { .. })
        ));

        let mut sweep = fixture();
        sweep.frequency_slices[0].columns[0].voltage_observations[0]
            .response_voltage_v
            .real = f64::NAN;
        assert!(matches!(
            validate_mimo_impedance_sweep(&sweep),
            Err(PortMatrixError::InvalidComplexObservation { .. })
        ));
    }
}
