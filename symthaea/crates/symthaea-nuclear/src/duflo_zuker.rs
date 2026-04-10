// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Duflo-Zuker 10-parameter mass formula
//!
//! Faithful Rust implementation of the DZ10 mass formula from
//! Duflo & Zuker, Phys. Rev. C 52, R23 (1995).
//!
//! The algorithm fills harmonic-oscillator sub-shells (alternating j and r
//! sub-shells) for neutrons and protons independently, then computes 10
//! physics terms:
//!
//!   1. Coulomb energy
//!   2. Master (Fock-Migdal) volume term
//!   3. Master surface term
//!   4. Isospin volume
//!   5. Isospin surface + Wigner term
//!   6. S3 volume (cubic filling)
//!   7. S3 surface
//!   8. QQ spherical
//!   9. QQ deformed
//!  10. Pairing
//!
//! All 10 terms are divided by `ra` (asymmetry-corrected radius) before
//! summation. The formula considers both spherical and deformed shapes and
//! picks the lower energy (deformed wins unless Z ≤ 50 or deformation
//! energy is non-negative).
//!
//! **RMS accuracy**: ~506 keV on AME1995, ~600-700 keV on AME2020.
//!
//! ## Reference implementation
//!
//! Translated from the TALYS Fortran code (`duflo.f90`), which implements
//! the original Duflo-Zuker février 1996 algorithm.
//!
//! ## References
//!
//! - Duflo, J. & Zuker, A. P. (1995). Phys. Rev. C 52, R23.
//! - TALYS nuclear reaction code, `source/duflo.f90`.

use serde::{Deserialize, Serialize};

/// The 10 Duflo-Zuker coefficients (published values).
const B: [f64; 10] = [
    0.7043, 17.7418, 16.2562, 37.5562, 53.9017, 0.4711, 2.1307, 0.0210, 40.5356, 6.0632,
];

/// Duflo-Zuker mass prediction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DZPrediction {
    pub binding_energy: f64,
    pub binding_energy_per_nucleon: f64,
}

/// Compute the DZ10 binding energy (MeV) for a nucleus with Z protons and N neutrons.
///
/// Returns the total binding energy (positive = bound).
/// Internally calls `mass10` which implements the full subshell filling algorithm.
pub fn dz_binding_energy(z: u16, n: u16) -> f64 {
    if z < 1 || n < 1 {
        return 0.0;
    }
    let a = (z as f64) + (n as f64);
    if a < 3.0 {
        if z == 1 && n == 1 {
            return 2.225;
        }
        return 0.0;
    }

    let e = mass10(n as i32, z as i32);
    // mass10 returns the binding energy directly (positive = bound).
    // The TALYS wrapper computes mass excess as Z*M_H + N*M_n - E,
    // but we want binding energy, which is E itself.
    e
}

/// DZ prediction with per-nucleon value.
pub fn dz_predict(z: u16, n: u16) -> DZPrediction {
    let be = dz_binding_energy(z, n);
    let a = (z + n) as f64;
    DZPrediction {
        binding_energy: be,
        binding_energy_per_nucleon: if a > 0.0 { be / a } else { 0.0 },
    }
}

/// Core DZ10 algorithm — faithful translation of TALYS `mass10` subroutine.
///
/// Arguments: `nx` = neutron number, `nz` = proton number.
/// Returns: binding energy in MeV (positive = bound).
fn mass10(nx: i32, nz: i32) -> f64 {
    // nn[0] = neutrons, nn[1] = protons
    let nn = [nx, nz];
    let a = (nx + nz) as f64;
    let t = (nx - nz).unsigned_abs() as f64;
    let r = a.powf(1.0 / 3.0);
    let rc = r * (1.0 - 0.25 * (t / a).powi(2)); // Charge radius
    let ra = (rc * rc) / r;

    // Coulomb energy (term 1) — computed once, outside the ndef loop
    let z2 = (nz * (nz - 1)) as f64;
    let dyda1 = (-z2 + 0.76 * z2.powf(2.0 / 3.0)) / rc;

    let mut y = [0.0_f64; 2]; // y[0] = spherical, y[1] = deformed

    // Main loop: ndef=0 → spherical, ndef=1 → deformed
    for ndef in 0..2_usize {
        let ju: i32 = if ndef == 1 { 4 } else { 0 }; // nucleons associated to deformation

        let mut dyda = [0.0_f64; 10];
        dyda[0] = dyda1; // Coulomb (same for both shapes)

        let mut n2 = [0_i32; 2];
        let mut oei = [0.0_f64; 2];
        let mut dei = [0.0_f64; 2];
        let mut qx = [0.0_f64; 2];
        let mut dx = [0.0_f64; 2];
        let mut pp_arr = [0.0_f64; 2]; // HO quantum number of last subshell
        let mut op = [0.0_f64; 2]; // FM (Fock-Migdal) term
        let mut os = [0.0_f64; 2]; // SO (spin-orbit) term

        // Loop over neutrons (j=0) and protons (j=1)
        for j in 0..2_usize {
            let mut noc = [0_i32; 29]; // noc[1..28], occupancies per subshell
            let mut onp = [[0.0_f64; 21]; 2]; // onp[ip=0..20][0=FM, 1=SO]

            n2[j] = 2 * (nn[j] / 2); // For pairing: largest even number ≤ nn[j]

            let mut ncum: i32 = 0;
            let mut i: i32 = 0;
            let mut imax: i32;
            let mut ip: i32;
            let mut ipm: i32;

            // Sub-shell filling loop
            loop {
                i += 1;
                let i2 = (i / 2) * 2; // i rounded down to even

                // Sub-shell degeneracy: j-subshell or r-subshell
                let id = if i2 != i {
                    // Odd i → j sub-shell
                    i + 1
                } else {
                    // Even i → r sub-shell
                    i * (i - 2) / 4
                };

                ncum += id;
                if ncum < nn[j] {
                    noc[i as usize] = id;
                    continue; // goto 20 equivalent
                }

                // We've reached or passed nn[j]
                imax = i + 1; // last subshell number
                ip = (i - 1) / 2; // HO quantum number (p)
                ipm = i / 2;
                pp_arr[j] = ip as f64;

                let moc = nn[j] - ncum + id; // nucleons in last subshell (before deformation)
                noc[i as usize] = moc - ju;
                if (i + 1) < 29 {
                    noc[(i + 1) as usize] = ju;
                }

                let i2 = (i / 2) * 2;
                if i2 != i {
                    // j sub-shell
                    oei[j] = (moc + ip * (ip - 1)) as f64;
                    dei[j] = (ip * (ip + 1) + 2) as f64;
                } else {
                    // r sub-shell
                    oei[j] = (moc - ju) as f64;
                    dei[j] = ((ip + 1) * (ip + 2) + 2) as f64;
                }

                // n*(D-n)/D  (S3 term)
                qx[j] = oei[j] * (dei[j] - oei[j] - ju as f64) / dei[j];
                // n*(D-n)*(2n-D)/D  (Q term)
                dx[j] = qx[j] * (2.0 * oei[j] - dei[j]);

                if ndef == 1 {
                    // Scaling for deformed
                    qx[j] /= dei[j].sqrt();
                }

                // Compute amplitudes for FM and SO terms
                for ii in 1..=(imax as usize) {
                    if ii >= 29 {
                        break;
                    }
                    let ip_ii = ((ii as i32 - 1) / 2) as usize;
                    let fact = ((ip_ii as f64 + 1.0) * (ip_ii as f64 + 2.0)).sqrt();

                    if ip_ii < 21 {
                        // FM amplitude
                        onp[0][ip_ii] += noc[ii] as f64 / fact;

                        // SO amplitude
                        let vm = if (2 * (ii / 2)) != ii {
                            0.5 * ip_ii as f64 // odd → j subshell
                        } else {
                            -1.0 // even → r subshell
                        };
                        onp[1][ip_ii] += noc[ii] as f64 * vm;
                    }
                }

                // Accumulate FM and SO sums
                op[j] = 0.0;
                os[j] = 0.0;
                for ip_idx in 0..=(ipm as usize) {
                    if ip_idx >= 21 {
                        break;
                    }
                    let pi_val = ip_idx as f64;
                    let den = ((pi_val + 1.0) * (pi_val + 2.0)).powf(1.5);

                    op[j] += onp[0][ip_idx]; // FM
                    os[j] += onp[1][ip_idx] * (1.0 + onp[0][ip_idx]) * (pi_val * pi_val / den)
                        + onp[1][ip_idx] * (1.0 - onp[0][ip_idx]) * ((4.0 * pi_val - 5.0) / den);
                    // SO
                }
                op[j] *= op[j]; // Square the FM term

                break; // Done filling this nucleon type
            }
        }

        // Assemble the 10 physics terms (indices 1-9 in dyda, 0 is Coulomb)
        dyda[1] = op[0] + op[1]; // Master term (FM): volume
        dyda[2] = -dyda[1] / ra; // Master surface
        dyda[1] += os[0] + os[1]; // FM + SO
        dyda[3] = -t * (t + 2.0) / (r * r); // Isospin volume
        dyda[4] = -dyda[3] / ra; // Isospin surface

        if ndef == 0 {
            // Spherical
            dyda[5] = dx[0] + dx[1]; // S3 volume
            dyda[6] = -dyda[5] / ra; // S3 surface
            let px = pp_arr[0].sqrt() + pp_arr[1].sqrt();
            dyda[7] = qx[0] * qx[1] * 2.0_f64.powf(px); // QQ spherical
        } else {
            // Deformed
            dyda[8] = qx[0] * qx[1]; // QQ deformed
        }

        // Wigner term (added to isospin surface)
        dyda[4] = t * (1.0 - t) / (a * ra.powi(3)) + dyda[4];

        // Pairing (term 10)
        dyda[9] = 0.0;
        if n2[0] != nn[0] && n2[1] != nn[1] {
            // Both odd
            dyda[9] = t / a;
        }
        if nx > nz {
            if n2[0] == nn[0] && n2[1] != nn[1] {
                dyda[9] = 1.0 - t / a;
            }
            if n2[0] != nn[0] && n2[1] == nn[1] {
                dyda[9] = 1.0;
            }
        } else {
            if n2[0] == nn[0] && n2[1] != nn[1] {
                dyda[9] = 1.0;
            }
            if n2[0] != nn[0] && n2[1] == nn[1] {
                dyda[9] = 1.0 - t / a;
            }
        }
        if n2[1] == nn[1] && n2[0] == nn[0] {
            // Both even
            dyda[9] = 2.0 - t / a;
        }

        // Divide all terms (except Coulomb is also divided) by ra
        // Fortran: do mss=2,10: dyda(mss)=dyda(mss)/ra
        // Our indices 1..9 correspond to Fortran 2..10
        for k in 1..10 {
            dyda[k] /= ra;
        }

        // Sum: y(ndef) = sum(dyda * b)
        y[ndef] = 0.0;
        for k in 0..10 {
            y[ndef] += dyda[k] * B[k];
        }
    }

    // Choose between spherical and deformed
    let de = y[1] - y[0];
    let mut e = y[1]; // Deformed by default
    if de <= 0.0 || nz <= 50 {
        e = y[0]; // Spherical
    }

    e.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fe56_peak() {
        let pred = dz_predict(26, 30);
        eprintln!(
            "Fe-56: BE = {:.3} MeV, B/A = {:.4} MeV",
            pred.binding_energy, pred.binding_energy_per_nucleon
        );
        assert!(
            (pred.binding_energy_per_nucleon - 8.79).abs() < 0.5,
            "Fe-56 B/A = {:.4}, expected ~8.79",
            pred.binding_energy_per_nucleon
        );
    }

    #[test]
    fn test_pb208() {
        let pred = dz_predict(82, 126);
        eprintln!("Pb-208: BE = {:.3} MeV", pred.binding_energy);
        assert!(
            (pred.binding_energy - 1636.0).abs() < 10.0,
            "Pb-208 BE = {:.3}, expected ~1636 MeV (±10 for DZ10)",
            pred.binding_energy
        );
    }

    #[test]
    fn test_he4() {
        let pred = dz_predict(2, 2);
        eprintln!("He-4: BE = {:.3} MeV", pred.binding_energy);
        assert!(
            pred.binding_energy > 20.0,
            "He-4 BE = {:.3} should be > 20",
            pred.binding_energy
        );
    }

    #[test]
    fn test_deuteron() {
        let pred = dz_predict(1, 1);
        assert!(
            (pred.binding_energy - 2.225).abs() < 0.01,
            "Deuteron BE = {:.3}, expected 2.225",
            pred.binding_energy
        );
    }

    #[test]
    fn test_o16() {
        // O-16: doubly magic, BE ~ 127.6 MeV
        let pred = dz_predict(8, 8);
        eprintln!("O-16: BE = {:.3} MeV", pred.binding_energy);
        assert!(
            (pred.binding_energy - 127.6).abs() < 5.0,
            "O-16 BE = {:.3}, expected ~127.6 MeV",
            pred.binding_energy
        );
    }

    #[test]
    fn test_ca48() {
        // Ca-48: doubly magic, BE ~ 416.0 MeV
        let pred = dz_predict(20, 28);
        eprintln!("Ca-48: BE = {:.3} MeV", pred.binding_energy);
        assert!(
            (pred.binding_energy - 416.0).abs() < 5.0,
            "Ca-48 BE = {:.3}, expected ~416.0 MeV",
            pred.binding_energy
        );
    }

    #[test]
    fn test_positive_for_stable() {
        for &(z, n) in &[(6, 6), (20, 20), (26, 30), (50, 82), (82, 126)] {
            let be = dz_binding_energy(z, n);
            assert!(be > 0.0, "Z={} N={}: BE={:.3}", z, n, be);
        }
    }

    #[test]
    fn test_rms_on_ame2020() {
        use crate::ame2020::ame2020_reference_nuclei;

        let nuclei = ame2020_reference_nuclei();
        let mut errors = Vec::new();
        let mut worst_z = 0;
        let mut worst_n = 0;
        let mut worst_err = 0.0_f64;

        for nuc in &nuclei {
            if !nuc.is_measured || nuc.z < 3 {
                continue;
            }
            let dz_be = dz_binding_energy(nuc.z, nuc.n);
            let error = dz_be - nuc.binding_energy_mev;
            errors.push(error);
            if error.abs() > worst_err.abs() {
                worst_err = error;
                worst_z = nuc.z;
                worst_n = nuc.n;
            }
        }

        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        let rms = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();

        eprintln!(
            "DZ10 (TALYS): RMS = {:.3} MeV, mean = {:.3} MeV on {} nuclei",
            rms,
            mean,
            errors.len()
        );
        eprintln!(
            "  Worst: Z={} N={} error = {:.3} MeV",
            worst_z, worst_n, worst_err
        );

        // DZ10 published RMS is ~506 keV on AME1995
        // On AME2020 (3,555 nuclei including extrapolated) expect ~600-900 keV
        assert!(rms < 5.0, "DZ10 RMS = {:.3} MeV should be < 5.0 MeV", rms);
    }
}
