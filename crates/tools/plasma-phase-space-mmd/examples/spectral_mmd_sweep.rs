use plasma_phase_space_mmd::{
    MmdConfig, PhaseSpaceParticle1d, SpectralMmdConfig, compare_phase_space_measures_mmd,
    compare_phase_space_measures_spectral,
};
use std::error::Error;
use std::time::Instant;

struct SweepCase {
    name: &'static str,
    kernel: MmdConfig,
    candidate: Vec<PhaseSpaceParticle1d>,
    reference: Vec<PhaseSpaceParticle1d>,
}

fn particle(position: f64, velocity: f64, weight: f64) -> PhaseSpaceParticle1d {
    PhaseSpaceParticle1d {
        position,
        velocity,
        weight,
    }
}

fn cases() -> Vec<SweepCase> {
    let standard = MmdConfig {
        domain_length: 8.0,
        spatial_concentration: 2.0,
        velocity_length_scale: 1.0,
    };
    let high_spatial_frequency = MmdConfig {
        domain_length: 8.0,
        spatial_concentration: 30.0,
        velocity_length_scale: 0.5,
    };

    vec![
        SweepCase {
            name: "finite_fourier_alias",
            kernel: standard,
            candidate: (0..4)
                .map(|index| particle(f64::from(index) * 2.0, 0.0, 0.25))
                .collect(),
            reference: (0..4)
                .map(|index| particle(1.0 + f64::from(index) * 2.0, 0.0, 0.25))
                .collect(),
        },
        SweepCase {
            name: "joint_position_velocity_correlation",
            kernel: standard,
            candidate: vec![particle(0.0, 0.0, 0.5), particle(4.0, 1.0, 0.5)],
            reference: vec![particle(0.0, 1.0, 0.5), particle(4.0, 0.0, 0.5)],
        },
        SweepCase {
            name: "signed_colocated_cancellation",
            kernel: standard,
            candidate: vec![particle(1.0, 0.0, 0.25)],
            reference: vec![particle(1.0, 0.0, 0.50), particle(1.0, 0.0, -0.25)],
        },
        SweepCase {
            name: "high_spatial_concentration",
            kernel: high_spatial_frequency,
            candidate: vec![
                particle(0.5, -0.5, 0.25),
                particle(2.0, 0.25, 0.25),
                particle(4.5, 0.75, 0.25),
                particle(7.0, -0.25, 0.25),
            ],
            reference: vec![
                particle(0.75, -0.45, 0.25),
                particle(2.25, 0.20, 0.25),
                particle(4.75, 0.80, 0.25),
                particle(7.25, -0.20, 0.25),
            ],
        },
    ]
}

fn main() -> Result<(), Box<dyn Error>> {
    const FEATURE_COUNTS: [usize; 9] = [64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384];
    const SEEDS: [u64; 10] = [7, 17, 29, 43, 59, 71, 89, 101, 2_026, 20_261_121];

    println!(
        "case,spatial_concentration,velocity_length_scale,feature_count,seed,exact_mmd_squared,approx_mmd_squared,absolute_error,relative_error,feature_standard_error,hoeffding_95_half_width,particle_feature_evaluations,runtime_us,target_arch,target_os"
    );

    for case in cases() {
        let exact =
            compare_phase_space_measures_mmd(&case.candidate, &case.reference, case.kernel)?;

        for feature_count in FEATURE_COUNTS {
            for seed in SEEDS {
                let start = Instant::now();
                let approximate = compare_phase_space_measures_spectral(
                    &case.candidate,
                    &case.reference,
                    case.kernel,
                    SpectralMmdConfig {
                        feature_count,
                        seed,
                    },
                )?;
                let runtime_micros = start.elapsed().as_micros();
                let absolute_error =
                    (approximate.normalized_mmd_squared - exact.normalized_mmd_squared).abs();
                let relative_error = if exact.normalized_mmd_squared > 1.0e-12 {
                    format!("{:.17e}", absolute_error / exact.normalized_mmd_squared)
                } else {
                    String::new()
                };

                println!(
                    "{},{:.8},{:.8},{},{},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{},{},{},{}",
                    case.name,
                    case.kernel.spatial_concentration,
                    case.kernel.velocity_length_scale,
                    feature_count,
                    seed,
                    exact.normalized_mmd_squared,
                    approximate.normalized_mmd_squared,
                    absolute_error,
                    relative_error,
                    approximate.feature_standard_error,
                    approximate.hoeffding_95_half_width,
                    approximate.particle_feature_evaluations,
                    runtime_micros,
                    std::env::consts::ARCH,
                    std::env::consts::OS,
                );
            }
        }
    }

    Ok(())
}
