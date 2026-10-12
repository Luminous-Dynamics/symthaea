// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{PrimitiveSystem, seed_from_name};

/// Prime factorization of a number with HDC encoding
#[derive(Debug, Clone)]
pub struct PrimeFactorization {
    pub value: u64,
    pub factors: Vec<(u64, u32)>,
    pub encoding: BinaryHV,
    pub phi: f64,
}

/// Number theory engine for HDC-based arithmetic
pub struct NumberTheoryEngine {
    multiplication: BinaryHV,
    addition: BinaryHV,
    zero: BinaryHV,
    successor: BinaryHV,
}

impl NumberTheoryEngine {
    pub fn new() -> Self {
        let prims = PrimitiveSystem::global();
        Self {
            multiplication: prims
                .get("MULTIPLICATION")
                .expect("MULTIPLICATION primitive must exist")
                .encoding,
            addition: prims
                .get("ADDITION")
                .expect("ADDITION primitive must exist")
                .encoding,
            zero: prims
                .get("ZERO")
                .expect("ZERO primitive must exist")
                .encoding,
            successor: prims
                .get("SUCCESSOR")
                .expect("SUCCESSOR primitive must exist")
                .encoding,
        }
    }

    /// Bounded deterministic-seed Pollard-Brent attempt to find a nontrivial factor.
    ///
    /// Returns `None` if the fixed work budget is exhausted; callers retain a
    /// trial-division fallback so the factorization API remains complete.
    fn pollard_brent_factor(&self, n: u64) -> Option<u64> {
        if n.is_multiple_of(2) {
            return Some(2);
        }
        if n.is_multiple_of(3) {
            return Some(3);
        }
        if n < 5 {
            return None;
        }

        const ATTEMPTS: u64 = 16;
        const ITERATION_BUDGET: u64 = 250_000;
        const BLOCK_SIZE: u64 = 128;
        let ring = ModularRing::new(n);

        for attempt in 0..ATTEMPTS {
            // Fixed seeds make behavior reproducible across runs and CI machines.
            let mut y = 2 + attempt % (n - 3);
            let c = 1 + 2 * attempt;
            let mut g = 1u64;
            let mut r = 1u64;
            let mut q = 1u64;
            let mut x = 0u64;
            let mut ys = 0u64;
            let mut iterations = 0u64;

            while g == 1 && iterations < ITERATION_BUDGET {
                x = y;

                // Keep each doubling phase inside the per-attempt work budget.
                if r > ITERATION_BUDGET - iterations {
                    break;
                }
                for _ in 0..r {
                    y = ring.add(ring.multiply(y, y), c);
                }
                iterations += r;

                let mut k = 0u64;
                while k < r && g == 1 {
                    ys = y;
                    let limit = BLOCK_SIZE.min(r - k).min(ITERATION_BUDGET - iterations);
                    if limit == 0 {
                        break;
                    }
                    for _ in 0..limit {
                        y = ring.add(ring.multiply(y, y), c);
                        q = ring.multiply(q, x.abs_diff(y));
                    }
                    iterations += limit;
                    g = self.gcd(q, n);
                    k += limit;
                }

                if g != 1 || iterations >= ITERATION_BUDGET || r > ITERATION_BUDGET / 2 {
                    break;
                }
                r *= 2;
            }

            // A batch can make q divisible by n. Recover by checking each step
            // from the beginning of the final batch, still under the same budget.
            if g == n {
                while iterations < ITERATION_BUDGET {
                    ys = ring.add(ring.multiply(ys, ys), c);
                    iterations += 1;
                    g = self.gcd(x.abs_diff(ys), n);
                    if g > 1 {
                        break;
                    }
                }
            }

            if g > 1 && g < n {
                return Some(g);
            }
        }

        None
    }

    /// Recursively collect prime factors. Miller-Rabin handles prime leaves;
    /// Pollard-Brent handles hard composites, with exact trial division as fallback.
    fn collect_prime_factors(&self, n: u64, out: &mut Vec<u64>) {
        if n <= 1 {
            return;
        }
        if self.miller_rabin(n) {
            out.push(n);
            return;
        }

        if let Some(factor) = self.pollard_brent_factor(n) {
            self.collect_prime_factors(factor, out);
            self.collect_prime_factors(n / factor, out);
            return;
        }

        // Completeness fallback: this may be slower for an unusually difficult
        // input, but it preserves exactness if the bounded rho attempts fail.
        let mut remaining = n;
        let mut divisor = 2u64;
        while divisor <= remaining / divisor {
            while remaining.is_multiple_of(divisor) {
                out.push(divisor);
                remaining /= divisor;
            }
            divisor += if divisor == 2 { 1 } else { 2 };
        }
        if remaining > 1 {
            out.push(remaining);
        }
    }

    /// Prime factorization using deterministic primality testing and Pollard-Brent,
    /// returning an HDC encoding of the factorization.
    pub fn factorize(&self, n: u64) -> PrimeFactorization {
        if n == 0 {
            return PrimeFactorization {
                value: 0,
                factors: vec![],
                encoding: self.zero,
                phi: 0.0,
            };
        }

        if n == 1 {
            return PrimeFactorization {
                value: 1,
                factors: vec![],
                encoding: self.successor.bind(&self.zero),
                phi: 1.0,
            };
        }

        let mut prime_factors = Vec::new();
        self.collect_prime_factors(n, &mut prime_factors);
        prime_factors.sort_unstable();

        let mut factors: Vec<(u64, u32)> = Vec::new();
        for prime in prime_factors {
            if let Some((last_prime, exponent)) = factors.last_mut() {
                if *last_prime == prime {
                    *exponent += 1;
                    continue;
                }
            }
            factors.push((prime, 1));
        }

        // Encode as bundle of MULTIPLICATION.bind(prime_enc).bind(exp_enc)
        let mut factor_encodings = Vec::new();
        for (prime, exponent) in &factors {
            let prime_enc = BinaryHV::random(seed_from_name(&format!("prime_{prime}")));
            let exp_enc = BinaryHV::random(seed_from_name(&format!("exp_{exponent}")));
            let factor_enc = self.multiplication.bind(&prime_enc).bind(&exp_enc);
            factor_encodings.push(factor_enc);
        }

        let encoding = if factor_encodings.is_empty() {
            self.successor.bind(&self.zero)
        } else {
            BinaryHV::bundle(&factor_encodings)
        };

        // Calculate Euler's totient function: φ(n) = n * ∏(1 - 1/p)
        let mut phi = n as f64;
        for (prime, _) in &factors {
            phi *= 1.0 - 1.0 / (*prime as f64);
        }

        PrimeFactorization {
            value: n,
            factors,
            encoding,
            phi,
        }
    }

    /// Widened extended Euclidean algorithm over the full i64 input domain.
    ///
    /// Returns (g, x, y) with non-negative g and a*x + b*y = g exactly.
    /// A u64 is required for g = abs(i64::MIN); i128 coefficients avoid
    /// intermediate overflow in the Euclidean recurrence.
    pub fn extended_gcd_wide(&self, a: i64, b: i64) -> (u64, i128, i128) {
        let mut old_r = i128::from(a);
        let mut r = i128::from(b);
        let mut old_s = 1i128;
        let mut s = 0i128;
        let mut old_t = 0i128;
        let mut t = 1i128;

        while r != 0 {
            let quotient = old_r / r;
            let next_r = old_r - quotient * r;
            old_r = r;
            r = next_r;

            let next_s = old_s - quotient * s;
            old_s = s;
            s = next_s;

            let next_t = old_t - quotient * t;
            old_t = t;
            t = next_t;
        }

        if old_r < 0 {
            old_r = -old_r;
            old_s = -old_s;
            old_t = -old_t;
        }

        // The absolute GCD of two i64 values is at most 2^63, which fits u64.
        let gcd = u64::try_from(old_r).expect("absolute GCD of i64 values fits u64");
        (gcd, old_s, old_t)
    }

    /// Extended Euclidean algorithm: returns (gcd, x, y) where ax + by = gcd.
    ///
    /// This compatibility signature uses i64 for all outputs and therefore
    /// panics explicitly when the exact result cannot be represented. Use
    /// `extended_gcd_wide` to support the entire i64 input domain.
    pub fn extended_gcd(&self, a: i64, b: i64) -> (i64, i64, i64) {
        let (gcd, x, y) = self.extended_gcd_wide(a, b);
        (
            i64::try_from(gcd)
                .expect("GCD exceeds i64::MAX; use extended_gcd_wide to preserve the exact result"),
            i64::try_from(x)
                .expect("extended GCD coefficient exceeds i64 range; use extended_gcd_wide"),
            i64::try_from(y)
                .expect("extended GCD coefficient exceeds i64 range; use extended_gcd_wide"),
        )
    }

    /// Greatest common divisor using Euclidean algorithm
    pub fn gcd(&self, a: u64, b: u64) -> u64 {
        let mut a = a;
        let mut b = b;
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    /// Checked least common multiple.
    ///
    /// Returns `None` when the exact result is greater than `u64::MAX`.
    /// Dividing before multiplying minimizes the intermediate and avoids wraparound.
    pub fn checked_lcm(&self, a: u64, b: u64) -> Option<u64> {
        if a == 0 || b == 0 {
            return Some(0);
        }
        (a / self.gcd(a, b)).checked_mul(b)
    }

    /// Least common multiple, panicking rather than silently wrapping on overflow.
    ///
    /// Use `checked_lcm` when the inputs may be outside a known-safe range.
    pub fn lcm(&self, a: u64, b: u64) -> u64 {
        self.checked_lcm(a, b)
            .expect("least common multiple exceeds u64::MAX; use checked_lcm to handle overflow")
    }

    /// Sieve of Eratosthenes
    pub fn primes_up_to(&self, n: u64) -> Vec<(u64, BinaryHV)> {
        if n < 2 {
            return vec![];
        }

        let n_size = n as usize + 1;
        let mut is_prime = vec![true; n_size];
        is_prime[0] = false;
        is_prime[1] = false;

        let limit = (n as f64).sqrt() as usize;
        for i in 2..=limit {
            if is_prime[i] {
                let mut j = i * i;
                while j < n_size {
                    is_prime[j] = false;
                    j += i;
                }
            }
        }

        is_prime
            .iter()
            .enumerate()
            .filter(|&(_, &prime)| prime)
            .map(|(i, _)| {
                let p = i as u64;
                let encoding = BinaryHV::random(seed_from_name(&format!("prime_{p}")));
                (p, encoding)
            })
            .collect()
    }

    /// Deterministic primality test for the entire u64 domain.
    ///
    /// Uses Miller–Rabin with witness sets selected by proven numeric bounds,
    /// avoiding floating-point square-root rounding and unbounded O(sqrt(n))
    /// trial division for large inputs.
    pub fn is_prime(&self, n: u64) -> bool {
        self.miller_rabin(n)
    }

    /// Fermat primality test: for each witness a, check a^(n-1) mod n == 1
    pub fn fermat_test(&self, n: u64, witnesses: &[u64]) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n.is_multiple_of(2) {
            return false;
        }

        let ring = ModularRing::new(n);
        for &a in witnesses {
            if a >= n {
                continue;
            }
            if self.gcd(a, n) != 1 {
                return false;
            }
            if ring.power(a, n - 1) != 1 {
                return false;
            }
        }
        true
    }

    /// Miller-Rabin primality test — deterministic for n < 3.3×10^24.
    ///
    /// Unlike Fermat, this correctly rejects Carmichael numbers (561, 1105, 1729).
    /// Uses known witness sets that make the test deterministic (not probabilistic)
    /// for numbers up to specific bounds:
    /// - n < 2,047: witnesses {2}
    /// - n < 1,373,653: witnesses {2, 3}
    /// - n < 3,215,031,751: witnesses {2, 3, 5, 7}
    /// - n < 3,317,044,064,679,887,385,961,981: witnesses {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
    pub fn miller_rabin(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        // Write n-1 = 2^r * d where d is odd
        let mut d = n - 1;
        let mut r = 0u32;
        while d % 2 == 0 {
            d /= 2;
            r += 1;
        }

        // Deterministic witness sets by range
        let witnesses: &[u64] = if n < 2_047 {
            &[2]
        } else if n < 1_373_653 {
            &[2, 3]
        } else if n < 3_215_031_751 {
            &[2, 3, 5, 7]
        } else {
            // Covers all u64 values (deterministic for n < 3.3×10^24)
            &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        };

        let ring = ModularRing::new(n);

        'witness: for &a in witnesses {
            if a >= n {
                continue;
            }

            // Compute a^d mod n
            let mut x = ring.power(a, d);

            if x == 1 || x == n - 1 {
                continue 'witness;
            }

            // Square r-1 times
            for _ in 0..r - 1 {
                x = ring.multiply(x, x);
                if x == n - 1 {
                    continue 'witness;
                }
            }

            // If we never hit n-1, n is composite
            return false;
        }

        true
    }
}

impl Default for NumberTheoryEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Phase 1: IMO-level number theory primitives ─────────────────────────────

impl NumberTheoryEngine {
    /// Chinese Remainder Theorem. Given residues `[(a_i, m_i)]` representing
    /// `x ≡ a_i (mod m_i)`, returns `Some((x, M))` with `M = lcm(m_i)` and
    /// `0 ≤ x < M`, or `None` if the system is inconsistent.
    ///
    /// Moduli do NOT need to be pairwise coprime — a consistent overlap is
    /// merged via extended gcd.
    pub fn crt(&self, residues: &[(i64, i64)]) -> Option<(i64, i64)> {
        if residues.is_empty() {
            return Some((0, 1));
        }
        if residues[0].1 <= 0 {
            return None;
        }
        let mut x = residues[0].0.rem_euclid(residues[0].1);
        let mut m = residues[0].1;
        for &(a_i, m_i) in &residues[1..] {
            if m_i <= 0 {
                return None;
            }
            let a_i = a_i.rem_euclid(m_i);
            let (g, p, _q) = self.extended_gcd(m, m_i);
            let diff = a_i - x;
            if diff % g != 0 {
                return None;
            }
            let lcm = m / g * m_i;
            let step = m_i / g;
            let k = ((diff / g) % step * (p % step)).rem_euclid(step);
            x = (x + k * m).rem_euclid(lcm);
            m = lcm;
        }
        Some((x, m))
    }

    /// Legendre symbol (a/p) for odd prime p. Returns -1, 0, or 1.
    pub fn legendre_symbol(&self, a: i64, p: i64) -> i32 {
        assert!(p > 2 && p % 2 == 1, "legendre_symbol requires odd prime p");
        let a_mod = a.rem_euclid(p);
        if a_mod == 0 {
            return 0;
        }
        let ring = ModularRing::new(p as u64);
        let v = ring.power(a_mod as u64, ((p - 1) / 2) as u64);
        if v == 1 {
            1
        } else if v == (p - 1) as u64 {
            -1
        } else {
            0
        }
    }

    /// Jacobi symbol (a/n) for odd positive n ≥ 1. Returns -1, 0, or 1.
    /// Generalizes Legendre to composite n via quadratic reciprocity.
    pub fn jacobi_symbol(&self, a: i64, n: i64) -> i32 {
        assert!(n > 0 && n % 2 == 1, "jacobi_symbol requires odd positive n");
        let mut a = a.rem_euclid(n);
        let mut n = n;
        let mut result: i32 = 1;
        while a != 0 {
            while a % 2 == 0 {
                a /= 2;
                let r = n % 8;
                if r == 3 || r == 5 {
                    result = -result;
                }
            }
            std::mem::swap(&mut a, &mut n);
            if a % 4 == 3 && n % 4 == 3 {
                result = -result;
            }
            a %= n;
        }
        if n == 1 { result } else { 0 }
    }

    /// Tonelli–Shanks: modular square root of `a` mod prime `p`.
    /// Returns `Some(r)` with r² ≡ a (mod p), or `None` if a is a non-residue.
    pub fn tonelli_shanks(&self, a: u64, p: u64) -> Option<u64> {
        if p == 2 {
            return Some(a % 2);
        }
        let a = a % p;
        if a == 0 {
            return Some(0);
        }
        if self.legendre_symbol(a as i64, p as i64) != 1 {
            return None;
        }
        let ring = ModularRing::new(p);
        if p % 4 == 3 {
            return Some(ring.power(a, (p + 1) / 4));
        }
        let mut q = p - 1;
        let mut s = 0u64;
        while q % 2 == 0 {
            q /= 2;
            s += 1;
        }
        let mut z = 2u64;
        while self.legendre_symbol(z as i64, p as i64) != -1 {
            z += 1;
        }
        let mut m = s;
        let mut c = ring.power(z, q);
        let mut t = ring.power(a, q);
        let mut r = ring.power(a, q.div_ceil(2));
        loop {
            if t == 1 {
                return Some(r);
            }
            let mut i = 0u64;
            let mut temp = t;
            while temp != 1 && i < m {
                temp = ring.multiply(temp, temp);
                i += 1;
            }
            if i == m {
                return None;
            }
            let mut b = c;
            for _ in 0..(m - i - 1) {
                b = ring.multiply(b, b);
            }
            m = i;
            c = ring.multiply(b, b);
            t = ring.multiply(t, c);
            r = ring.multiply(r, b);
        }
    }

    /// p-adic valuation v_p(n): the largest k such that p^k divides n.
    /// For n = 0 returns u32::MAX (conventionally ∞).
    pub fn p_adic_valuation(&self, n: i64, p: i64) -> u32 {
        if n == 0 {
            return u32::MAX;
        }
        let mut k = 0u32;
        let mut m = n.abs();
        while m % p == 0 {
            m /= p;
            k += 1;
        }
        k
    }

    /// Lifting the Exponent (LTE): computes v_p(a^n − b^n) under LTE hypotheses.
    /// Returns `Some(v)` if the lemma applies, `None` otherwise.
    ///
    /// p odd, p ∤ a, p ∤ b, p | (a−b):
    ///   v_p(a^n − b^n) = v_p(a−b) + v_p(n)
    /// p = 2, n odd, 2 | (a−b):
    ///   v_2(a^n − b^n) = v_2(a−b)
    /// p = 2, n even, 4 | (a−b):
    ///   v_2(a^n − b^n) = v_2(a−b) + v_2(n)
    /// p = 2, n even, 2 | (a−b) but 4 ∤ (a−b):
    ///   v_2(a^n − b^n) = v_2(a−b) + v_2(a+b) + v_2(n) − 1
    pub fn lifting_the_exponent(&self, p: i64, a: i64, b: i64, n: u32) -> Option<u32> {
        if p < 2 || n == 0 {
            return None;
        }
        if a % p == 0 || b % p == 0 {
            return None;
        }
        if (a - b) % p != 0 {
            return None;
        }
        let v_diff = self.p_adic_valuation(a - b, p);
        let v_n = self.p_adic_valuation(n as i64, p);
        if p % 2 == 1 {
            return Some(v_diff + v_n);
        }
        if n % 2 == 1 {
            return Some(v_diff);
        }
        if (a - b) % 4 == 0 {
            return Some(v_diff + v_n);
        }
        let v_sum = self.p_adic_valuation(a + b, 2);
        Some(v_diff + v_sum + v_n - 1)
    }

    /// Linear Diophantine equation ax + by = c.
    ///
    /// Returns `Some((x0, y0, dx, dy))`: particular solution `(x0, y0)` plus
    /// increment `(dx, dy)` so the general solution is
    /// `(x0 + k·dx, y0 − k·dy)` for k ∈ ℤ. Returns `None` when gcd(a,b) ∤ c
    /// or when a = b = 0 and c ≠ 0.
    pub fn linear_diophantine(&self, a: i64, b: i64, c: i64) -> Option<(i64, i64, i64, i64)> {
        if a == 0 && b == 0 {
            return if c == 0 { Some((0, 0, 1, 0)) } else { None };
        }
        let (g, p, q) = self.extended_gcd(a, b);
        if c % g != 0 {
            return None;
        }
        let mult = c / g;
        let x0 = p * mult;
        let y0 = q * mult;
        let dx = b / g;
        let dy = a / g;
        Some((x0, y0, dx, dy))
    }
}

/// Modular arithmetic ring Z/nZ
pub struct ModularRing {
    pub modulus: u64,
    pub modulus_encoding: BinaryHV,
}

impl ModularRing {
    /// Construct a modular ring only when its modulus is positive.
    pub fn try_new(modulus: u64) -> Option<Self> {
        if modulus == 0 {
            return None;
        }
        let modulus_encoding = BinaryHV::random(seed_from_name(&format!("mod_{modulus}")));
        Some(Self {
            modulus,
            modulus_encoding,
        })
    }

    /// Construct a modular ring, failing explicitly for the invalid zero modulus.
    pub fn new(modulus: u64) -> Self {
        Self::try_new(modulus).expect("modular arithmetic requires a non-zero modulus")
    }

    // The fields remain public for compatibility, so validate the invariant at
    // operation boundaries as well as at construction.
    fn assert_valid_modulus(&self) {
        assert!(self.modulus > 0, "modular arithmetic requires a non-zero modulus");
    }

    /// Modular addition without u64 intermediate overflow.
    pub fn add(&self, a: u64, b: u64) -> u64 {
        self.assert_valid_modulus();
        let modulus = u128::from(self.modulus);
        let a = u128::from(a % self.modulus);
        let b = u128::from(b % self.modulus);
        ((a + b) % modulus) as u64
    }

    /// Modular multiplication
    pub fn multiply(&self, a: u64, b: u64) -> u64 {
        self.assert_valid_modulus();
        let a = a % self.modulus;
        let b = b % self.modulus;
        ((a as u128 * b as u128) % self.modulus as u128) as u64
    }

    /// Modular exponentiation by squaring
    pub fn power(&self, base: u64, exp: u64) -> u64 {
        self.assert_valid_modulus();
        if exp == 0 {
            return 1 % self.modulus;
        }

        let mut result = 1u64;
        let mut base = base % self.modulus;
        let mut exp = exp;

        while exp > 0 {
            if exp % 2 == 1 {
                result = self.multiply(result, base);
            }
            base = self.multiply(base, base);
            exp /= 2;
        }

        result
    }

    /// Modular multiplicative inverse for the full u64 modulus domain.
    ///
    /// Inputs are reduced modulo the ring modulus before the Euclidean algorithm;
    /// no u64-to-i64 casts or signed remainder corrections are used.
    pub fn inverse(&self, a: u64) -> Option<u64> {
        self.assert_valid_modulus();
        let modulus = i128::from(self.modulus);
        let mut old_r = i128::from(a % self.modulus);
        let mut r = modulus;
        let mut old_s = 1i128;
        let mut s = 0i128;

        while r != 0 {
            let quotient = old_r / r;
            let next_r = old_r - quotient * r;
            old_r = r;
            r = next_r;

            let next_s = old_s - quotient * s;
            old_s = s;
            s = next_s;
        }

        if old_r != 1 {
            return None;
        }

        Some(old_s.rem_euclid(modulus) as u64)
    }

    /// Check if element is a unit (has multiplicative inverse)
    pub fn is_unit(&self, a: u64) -> bool {
        self.assert_valid_modulus();
        let engine = NumberTheoryEngine::new();
        engine.gcd(a, self.modulus) == 1
    }

    /// Euler's totient function, computed from the distinct prime factors of the modulus.
    ///
    /// Uses phi(n) = n * product(1 - 1/p) over distinct prime factors p of n,
    /// avoiding the previous linear scan over every residue. Prime moduli therefore
    /// benefit from the deterministic Miller-Rabin fast path in factorize().
    /// By convention, phi(1) = 1.
    pub fn euler_totient(&self) -> u64 {
        self.assert_valid_modulus();

        let factorization = NumberTheoryEngine::new().factorize(self.modulus);
        let mut totient = self.modulus;
        for (prime, _) in factorization.factors {
            // Divide first: this stays exact and avoids an unnecessary product
            // that could overflow even if the final totient is representable.
            totient = (totient / prime) * (prime - 1);
        }
        totient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorize_60() {
        let engine = NumberTheoryEngine::new();
        let f = engine.factorize(60);
        assert_eq!(f.factors, vec![(2, 2), (3, 1), (5, 1)]);
    }

    #[test]
    fn test_factorize_1() {
        let engine = NumberTheoryEngine::new();
        let f = engine.factorize(1);
        assert!(f.factors.is_empty());
    }

    #[test]
    fn test_factorize_prime() {
        let engine = NumberTheoryEngine::new();
        let f = engine.factorize(17);
        assert_eq!(f.factors, vec![(17, 1)]);
    }

    #[test]
    fn test_factorize_large_semiprime_uses_pollard_brent() {
        let engine = NumberTheoryEngine::new();
        let p = 1_000_000_007u64;
        let q = 1_000_000_009u64;
        let semiprime = p * q;
        let factorization = engine.factorize(semiprime);
        assert_eq!(factorization.value, semiprime);
        assert_eq!(factorization.factors, vec![(p, 1), (q, 1)]);

        let reconstructed = factorization.factors.iter().fold(1u128, |product, &(prime, exponent)| {
            product * u128::from(prime).pow(exponent)
        });
        assert_eq!(reconstructed, u128::from(semiprime));
    }

    #[test]
    fn test_factorize_near_u64_limit_semiprime() {
        let engine = NumberTheoryEngine::new();
        let p = 4_294_967_291u64;
        let q = 4_294_967_279u64;
        assert!(engine.miller_rabin(p));
        assert!(engine.miller_rabin(q));

        let semiprime = p * q;
        assert!(semiprime > u64::MAX - 100_000_000_000);
        let factorization = engine.factorize(semiprime);
        assert_eq!(factorization.factors, vec![(q, 1), (p, 1)]);

        let reconstructed = factorization.factors.iter().fold(1u128, |product, &(prime, exponent)| {
            product * u128::from(prime).pow(exponent)
        });
        assert_eq!(reconstructed, u128::from(semiprime));
    }

    #[test]
    fn test_factorize_large_prime_powers_and_mixed_composite() {
        let engine = NumberTheoryEngine::new();
        for (n, expected) in [
            (2u64.pow(40), vec![(2, 40)]),
            (3u64.pow(20), vec![(3, 20)]),
            (3_215_031_751u64, vec![(151, 1), (751, 1), (28_351, 1)]),
        ] {
            let result = engine.factorize(n);
            assert_eq!(result.factors, expected, "n={n}");
        }
    }

    #[test]
    fn test_extended_gcd() {
        let engine = NumberTheoryEngine::new();
        let (g, x, y) = engine.extended_gcd(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, 5);
    }

    #[test]
    fn test_extended_gcd_wide_satisfies_bezout_identity_on_signed_grid() {
        let engine = NumberTheoryEngine::new();
        for a in -40i64..=40 {
            for b in -40i64..=40 {
                let (gcd, x, y) = engine.extended_gcd_wide(a, b);
                assert_eq!(i128::from(a) * x + i128::from(b) * y, i128::from(gcd), "a={a}, b={b}");
                assert_eq!(gcd, engine.gcd(a.unsigned_abs(), b.unsigned_abs()));
            }
        }
    }

    #[test]
    fn test_extended_gcd_wide_handles_signed_integer_boundaries() {
        let engine = NumberTheoryEngine::new();
        for (a, b) in [
            (i64::MIN, 0),
            (0, i64::MIN),
            (i64::MIN, i64::MAX),
            (i64::MAX, i64::MIN),
            (i64::MIN, i64::MIN),
            (i64::MAX, i64::MAX),
            (0, 0),
        ] {
            let (gcd, x, y) = engine.extended_gcd_wide(a, b);
            assert_eq!(i128::from(a) * x + i128::from(b) * y, i128::from(gcd), "a={a}, b={b}");
            assert_eq!(gcd, engine.gcd(a.unsigned_abs(), b.unsigned_abs()));
        }
        assert_eq!(engine.extended_gcd_wide(i64::MIN, 0), (1u64 << 63, -1, 0));
    }

    #[test]
    #[should_panic(expected = "GCD exceeds i64::MAX; use extended_gcd_wide")]
    fn test_legacy_extended_gcd_rejects_unrepresentable_gcd_explicitly() {
        let engine = NumberTheoryEngine::new();
        let _ = engine.extended_gcd(i64::MIN, 0);
    }

    #[test]
    fn test_modular_inverse() {
        let ring = ModularRing::new(7);
        let inv = ring.inverse(3).unwrap();
        assert_eq!(inv, 5); // 3 * 5 = 15 ≡ 1 (mod 7)
        assert_eq!(ring.multiply(3, inv), 1);
    }

    #[test]
    fn test_no_inverse() {
        let ring = ModularRing::new(6);
        assert!(ring.inverse(2).is_none()); // gcd(2, 6) = 2 != 1
    }

    #[test]
    fn test_modular_ring_rejects_zero_modulus_explicitly() {
        assert!(ModularRing::try_new(0).is_none());
        let ring = ModularRing::try_new(1).unwrap();
        assert_eq!(ring.add(u64::MAX, u64::MAX), 0);
        assert_eq!(ring.multiply(u64::MAX, u64::MAX), 0);
        assert_eq!(ring.power(u64::MAX, 0), 0);
    }

    #[test]
    #[should_panic(expected = "modular arithmetic requires a non-zero modulus")]
    fn test_modular_ring_legacy_constructor_fails_on_zero_modulus() {
        let _ = ModularRing::new(0);
    }

    #[test]
    #[should_panic(expected = "modular arithmetic requires a non-zero modulus")]
    fn test_modular_ring_operations_revalidate_public_mutable_modulus() {
        let mut ring = ModularRing::new(7);
        ring.modulus = 0;
        let _ = ring.add(1, 2);
    }

    #[test]
    fn test_modular_addition_does_not_overflow_near_u64_max() {
        let ring = ModularRing::new(u64::MAX);
        assert_eq!(ring.add(u64::MAX - 1, u64::MAX - 1), u64::MAX - 2);
        assert_eq!(ring.add(u64::MAX, 1), 1);
        assert_eq!(ring.add(u64::MAX - 1, 2), 1);
    }

    #[test]
    fn test_modular_inverse_supports_moduli_above_i64_max() {
        let modulus = u64::MAX - 1;
        let ring = ModularRing::new(modulus);
        let inverse = ring.inverse(3).expect("gcd(3, modulus) = 1");
        assert_eq!(ring.multiply(3, inverse), 1);

        let large_modulus = (1u64 << 63) + 29;
        let prime_ring = ModularRing::new(large_modulus);
        let inv = prime_ring.inverse(prime_ring.modulus - 1).unwrap();
        assert_eq!(inv, prime_ring.modulus - 1);
        assert_eq!(prime_ring.multiply(prime_ring.modulus - 1, inv), 1);
    }

    #[test]
    fn test_modular_power() {
        let ring = ModularRing::new(13);
        assert_eq!(ring.power(2, 10), 10); // 2^10 = 1024, 1024 mod 13 = 10
    }

    #[test]
    fn test_euler_totient() {
        let ring = ModularRing::new(12);
        assert_eq!(ring.euler_totient(), 4); // {1, 5, 7, 11}
        assert_eq!(ModularRing::new(1).euler_totient(), 1); // phi(1) = 1 by convention
    }

    #[test]
    fn test_euler_totient_matches_coprime_count_for_small_moduli() {
        let engine = NumberTheoryEngine::new();
        for modulus in 1u64..=128 {
            let expected = (1..=modulus)
                .filter(|&a| engine.gcd(a, modulus) == 1)
                .count() as u64;
            assert_eq!(
                ModularRing::new(modulus).euler_totient(),
                expected,
                "modulus={modulus}"
            );
        }
    }

    #[test]
    fn test_euler_totient_handles_large_u64_prime_without_residue_scan() {
        let prime = 18_446_744_073_709_551_557u64;
        assert_eq!(ModularRing::new(prime).euler_totient(), prime - 1);
    }

    #[test]
    fn test_primes_up_to() {
        let engine = NumberTheoryEngine::new();
        let primes = engine.primes_up_to(20);
        let prime_values: Vec<u64> = primes.iter().map(|(p, _)| *p).collect();
        assert_eq!(prime_values, vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_is_prime() {
        let engine = NumberTheoryEngine::new();
        assert!(engine.is_prime(17));
        assert!(!engine.is_prime(15));
        assert!(!engine.is_prime(1));
        assert!(engine.is_prime(2));
    }

    #[test]
    fn test_is_prime_is_deterministic_near_u64_limits() {
        let engine = NumberTheoryEngine::new();
        // Largest known prime below 2^64.
        assert!(engine.is_prime(18_446_744_073_709_551_557));
        assert!(!engine.is_prime(u64::MAX));
        assert!(!engine.is_prime(3_215_031_751)); // strong pseudoprime to small bases
        assert!(!engine.is_prime(561)); // Carmichael number
    }

    #[test]
    fn test_factorize_large_u64_prime_uses_fast_primality_path() {
        let engine = NumberTheoryEngine::new();
        let prime = 18_446_744_073_709_551_557;
        let factorization = engine.factorize(prime);
        assert_eq!(factorization.value, prime);
        assert_eq!(factorization.factors, vec![(prime, 1)]);
    }

    #[test]
    fn test_fermat_test() {
        let engine = NumberTheoryEngine::new();
        assert!(engine.fermat_test(17, &[2, 3, 5]));
        assert!(!engine.fermat_test(15, &[2]));
    }

    #[test]
    fn test_lcm() {
        let engine = NumberTheoryEngine::new();
        assert_eq!(engine.lcm(12, 18), 36);
        assert_eq!(engine.checked_lcm(12, 18), Some(36));
        assert_eq!(engine.checked_lcm(0, u64::MAX), Some(0));
        assert_eq!(engine.checked_lcm(u64::MAX, 1), Some(u64::MAX));
        assert_eq!(engine.checked_lcm(u64::MAX, 2), None);
    }

    #[test]
    #[should_panic(expected = "least common multiple exceeds u64::MAX")]
    fn test_lcm_fails_explicitly_instead_of_wrapping_on_overflow() {
        let engine = NumberTheoryEngine::new();
        let _ = engine.lcm(u64::MAX, 2);
    }

    // ── Miller-Rabin tests ──────────────────────────────────────────────

    #[test]
    fn test_is_prime_matches_independent_trial_division_through_10000() {
        let engine = NumberTheoryEngine::new();
        for n in 0u64..=10_000 {
            let expected = if n < 2 {
                false
            } else {
                let mut divisor = 2u64;
                let mut prime = true;
                while divisor <= n / divisor {
                    if n.is_multiple_of(divisor) {
                        prime = false;
                        break;
                    }
                    divisor += if divisor == 2 { 1 } else { 2 };
                }
                prime
            };
            assert_eq!(engine.is_prime(n), expected, "n={n}");
        }
    }

    #[test]
    fn test_factorization_reconstructs_small_integers_and_uses_prime_factors() {
        let engine = NumberTheoryEngine::new();
        for n in 2u64..=2_000 {
            let result = engine.factorize(n);
            let mut product = 1u128;
            for &(prime, exponent) in &result.factors {
                assert!(engine.miller_rabin(prime), "factor {prime} of {n} must be prime");
                assert!(exponent > 0);
                for _ in 0..exponent {
                    product *= u128::from(prime);
                }
            }
            assert_eq!(product, u128::from(n), "factorization of {n}: {:?}", result.factors);
        }
    }

    #[test]
    fn test_miller_rabin_small_primes() {
        let engine = NumberTheoryEngine::new();
        let primes = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97,
        ];
        for &p in &primes {
            assert!(engine.miller_rabin(p), "{} should be prime", p);
        }
    }

    #[test]
    fn test_miller_rabin_composites() {
        let engine = NumberTheoryEngine::new();
        let composites = [4, 6, 8, 9, 10, 12, 15, 21, 25, 49, 100, 121, 1000];
        for &c in &composites {
            assert!(!engine.miller_rabin(c), "{} should be composite", c);
        }
    }

    /// Carmichael numbers fool the Fermat test but NOT Miller-Rabin.
    /// This is the critical advantage of Miller-Rabin.
    #[test]
    fn test_miller_rabin_rejects_carmichael_numbers() {
        let engine = NumberTheoryEngine::new();
        // First 10 Carmichael numbers
        let carmichael = [561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841, 29341];
        for &c in &carmichael {
            assert!(
                !engine.miller_rabin(c),
                "Carmichael number {} should be rejected by Miller-Rabin",
                c
            );
        }

        // Verify that Fermat INCORRECTLY accepts some Carmichael numbers
        // (561 = 3×11×17 passes Fermat for witness 2)
        let fermat_561 = engine.fermat_test(561, &[2]);
        // Note: 561 actually DOES pass Fermat with witness 2
        // because 2^560 mod 561 = 1 (this is why Carmichael numbers are dangerous)
        eprintln!(
            "Fermat(561, [2]) = {} (Carmichael — Fermat may be fooled)",
            fermat_561
        );
        eprintln!(
            "Miller-Rabin(561) = {} (correctly rejects)",
            engine.miller_rabin(561)
        );
    }

    #[test]
    fn test_miller_rabin_large_primes() {
        let engine = NumberTheoryEngine::new();
        // Large known primes
        assert!(engine.miller_rabin(104729)); // 10000th prime
        assert!(engine.miller_rabin(1_000_003)); // prime just above 1M
        assert!(engine.miller_rabin(15_485_863)); // 1 millionth prime
        assert!(!engine.miller_rabin(15_485_864)); // not prime
    }

    #[test]
    fn test_miller_rabin_edge_cases() {
        let engine = NumberTheoryEngine::new();
        assert!(!engine.miller_rabin(0));
        assert!(!engine.miller_rabin(1));
        assert!(engine.miller_rabin(2));
        assert!(engine.miller_rabin(3));
    }

    // ── Phase 1 primitives ──────────────────────────────────────────────

    #[test]
    fn test_crt_coprime() {
        let e = NumberTheoryEngine::new();
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → x = 23 mod 105
        let (x, m) = e.crt(&[(2, 3), (3, 5), (2, 7)]).unwrap();
        assert_eq!(m, 105);
        assert_eq!(x, 23);
    }

    #[test]
    fn test_crt_non_coprime_consistent() {
        let e = NumberTheoryEngine::new();
        // x ≡ 2 (mod 6), x ≡ 8 (mod 10) → agree mod gcd(6,10)=2; lcm=30, x=8
        let (x, m) = e.crt(&[(2, 6), (8, 10)]).unwrap();
        assert_eq!(m, 30);
        assert_eq!(x.rem_euclid(6), 2);
        assert_eq!(x.rem_euclid(10), 8);
    }

    #[test]
    fn test_crt_inconsistent() {
        let e = NumberTheoryEngine::new();
        // x ≡ 1 (mod 4), x ≡ 2 (mod 6): disagree mod gcd=2
        assert!(e.crt(&[(1, 4), (2, 6)]).is_none());
    }

    #[test]
    fn test_legendre_symbol() {
        let e = NumberTheoryEngine::new();
        // QRs mod 7: {1, 2, 4}; NQRs: {3, 5, 6}
        assert_eq!(e.legendre_symbol(1, 7), 1);
        assert_eq!(e.legendre_symbol(2, 7), 1);
        assert_eq!(e.legendre_symbol(3, 7), -1);
        assert_eq!(e.legendre_symbol(4, 7), 1);
        assert_eq!(e.legendre_symbol(5, 7), -1);
        assert_eq!(e.legendre_symbol(6, 7), -1);
        assert_eq!(e.legendre_symbol(7, 7), 0);
    }

    #[test]
    fn test_jacobi_symbol() {
        let e = NumberTheoryEngine::new();
        // Jacobi = Legendre when n is prime
        for a in 1..11 {
            assert_eq!(e.jacobi_symbol(a, 11), e.legendre_symbol(a, 11));
        }
        // (2/15) = (2/3)(2/5) = (-1)(-1) = 1
        assert_eq!(e.jacobi_symbol(2, 15), 1);
        // (7/15) = (7/3)(7/5) = (1)(-1) = -1
        assert_eq!(e.jacobi_symbol(7, 15), -1);
    }

    #[test]
    fn test_tonelli_shanks_p3_mod_4() {
        let e = NumberTheoryEngine::new();
        // sqrt(2) mod 7: 7 ≡ 3 (mod 4). 2^((7+1)/4) = 2^2 = 4. 4² = 16 ≡ 2. ✓
        let r = e.tonelli_shanks(2, 7).unwrap();
        assert_eq!((r * r) % 7, 2);
    }

    #[test]
    fn test_tonelli_shanks_p1_mod_4() {
        let e = NumberTheoryEngine::new();
        // sqrt(10) mod 13: 13 ≡ 1 (mod 4), full algorithm path
        let r = e.tonelli_shanks(10, 13).unwrap();
        assert_eq!((r * r) % 13, 10);
    }

    #[test]
    fn test_tonelli_shanks_larger() {
        let e = NumberTheoryEngine::new();
        // sqrt(5) mod 41 exists since (5/41)=1
        let r = e.tonelli_shanks(5, 41).unwrap();
        assert_eq!((r * r) % 41, 5);
        // sqrt(3) mod 41: (3/41)=?  check both exist or not
        if e.legendre_symbol(3, 41) == 1 {
            let r = e.tonelli_shanks(3, 41).unwrap();
            assert_eq!((r * r) % 41, 3);
        } else {
            assert!(e.tonelli_shanks(3, 41).is_none());
        }
    }

    #[test]
    fn test_tonelli_shanks_non_residue() {
        let e = NumberTheoryEngine::new();
        // 3 is non-residue mod 7: (3/7) = -1
        assert!(e.tonelli_shanks(3, 7).is_none());
    }

    #[test]
    fn test_p_adic_valuation() {
        let e = NumberTheoryEngine::new();
        assert_eq!(e.p_adic_valuation(12, 2), 2); // 12 = 4·3
        assert_eq!(e.p_adic_valuation(12, 3), 1);
        assert_eq!(e.p_adic_valuation(12, 5), 0);
        assert_eq!(e.p_adic_valuation(-48, 2), 4);
        assert_eq!(e.p_adic_valuation(81, 3), 4);
    }

    #[test]
    fn test_lte_odd_prime() {
        let e = NumberTheoryEngine::new();
        // v_3(5^6 − 2^6) = v_3(5−2) + v_3(6) = 1 + 1 = 2
        // 5^6 − 2^6 = 15625 − 64 = 15561 = 3² · 7 · 13 · 19
        assert_eq!(e.lifting_the_exponent(3, 5, 2, 6), Some(2));
        // v_5(7^10 − 2^10) = v_5(5) + v_5(10) = 1 + 1 = 2
        assert_eq!(e.lifting_the_exponent(5, 7, 2, 10), Some(2));
    }

    #[test]
    fn test_lte_precondition_violation() {
        let e = NumberTheoryEngine::new();
        // p | a: fails precondition
        assert!(e.lifting_the_exponent(3, 6, 2, 4).is_none());
        // p ∤ (a−b): fails
        assert!(e.lifting_the_exponent(5, 7, 3, 4).is_none());
    }

    #[test]
    fn test_lte_p2_n_even_4_divides() {
        let e = NumberTheoryEngine::new();
        // a=5, b=1, a−b=4 (4 divides), n=2
        // v_2(5^2 − 1^2) = v_2(24) = 3
        // Formula: v_2(a−b) + v_2(n) = 2 + 1 = 3 ✓
        assert_eq!(e.lifting_the_exponent(2, 5, 1, 2), Some(3));
    }

    #[test]
    fn test_lte_p2_n_even_only_2_divides() {
        let e = NumberTheoryEngine::new();
        // a=3, b=1, a−b=2, a+b=4, n=2
        // v_2(3^2 − 1^2) = v_2(8) = 3
        // Formula: v_2(a−b) + v_2(a+b) + v_2(n) − 1 = 1 + 2 + 1 − 1 = 3 ✓
        assert_eq!(e.lifting_the_exponent(2, 3, 1, 2), Some(3));
    }

    #[test]
    fn test_linear_diophantine_solvable() {
        let e = NumberTheoryEngine::new();
        // 12x + 8y = 20 → gcd=4 | 20. Particular + increments.
        let (x0, y0, dx, dy) = e.linear_diophantine(12, 8, 20).unwrap();
        assert_eq!(12 * x0 + 8 * y0, 20);
        // k=1 shift
        assert_eq!(12 * (x0 + dx) + 8 * (y0 - dy), 20);
        // dx = b/g = 2, dy = a/g = 3
        assert_eq!(dx, 2);
        assert_eq!(dy, 3);
    }

    #[test]
    fn test_linear_diophantine_unsolvable() {
        let e = NumberTheoryEngine::new();
        // 6x + 9y = 5: gcd=3 ∤ 5
        assert!(e.linear_diophantine(6, 9, 5).is_none());
    }

    #[test]
    fn test_linear_diophantine_trivial() {
        let e = NumberTheoryEngine::new();
        assert!(e.linear_diophantine(0, 0, 5).is_none());
        assert!(e.linear_diophantine(0, 0, 0).is_some());
    }
}
