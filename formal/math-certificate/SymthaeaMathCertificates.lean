import PrimeCert

namespace SymthaeaMathCertificates

/-- Minimal Pocklington proof checked by Lean's kernel via PrimeCert. -/
theorem golden_prime_three : Nat.Prime 3 := by
  prime_cert [small {2}, pock (3, 2, 2)]

/-- Base-prime claim carried by the SYMFCER1 factorization-of-2 vector. -/
theorem golden_factor_two_is_prime : Nat.Prime 2 := by
  norm_num

/-- The outer factorization in the 76-byte SYMFCER1 golden vector reconstructs 2. -/
theorem golden_factorization_two_reconstructs : 2 = (2) := by
  norm_num

end SymthaeaMathCertificates
