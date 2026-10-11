-- Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
-- SPDX-License-Identifier: AGPL-3.0-or-later
--
-- SYM-FV-XENIA-002
-- Abstract non-wrapping allocator theorem for XEN-WIRE-V2-001.
-- Mathematical Nat is deliberate; Rust u64 refinement is a separate obligation.

namespace Symthaea.Formal.XeniaV2Allocator

/-- Number of distinct values representable by one unsigned 64-bit sequence. -/
def SequenceLimit : Nat := 2 ^ 64

/-- Abstract sender allocator state. -/
structure AllocatorState where
  next : Nat

/-- Allocate exactly the current sequence and advance by one while the 64-bit
    sequence space remains available. Refuse at and beyond the limit. -/
def allocate (s : AllocatorState) : Option (Nat × AllocatorState) :=
  if s.next < SequenceLimit then
    some (s.next, { next := s.next + 1 })
  else
    none

/-- Exact success shape: every successful allocation returns the current value,
    advances state exactly once, and returns an in-range sequence. -/
theorem allocate_success_shape
    (s s' : AllocatorState) (seq : Nat)
    (halloc : allocate s = some (seq, s')) :
    seq = s.next ∧ s'.next = s.next + 1 ∧ seq < SequenceLimit := by
  unfold allocate at halloc
  by_cases hlt : s.next < SequenceLimit
  · rw [if_pos hlt] at halloc
    have hpair : (s.next, { next := s.next + 1 }) = (seq, s') :=
      Option.some.inj halloc
    cases hpair
    exact ⟨rfl, rfl, hlt⟩
  · rw [if_neg hlt] at halloc
    contradiction

/-- XEN-V2-ALLOC-001: every successfully allocated sequence is below 2^64. -/
theorem allocate_success_in_range
    (s s' : AllocatorState) (seq : Nat)
    (halloc : allocate s = some (seq, s')) :
    seq < SequenceLimit :=
  (allocate_success_shape s s' seq halloc).2.2

/-- XEN-V2-ALLOC-002a: success returns exactly the pre-state sequence. -/
theorem allocate_success_returns_current
    (s s' : AllocatorState) (seq : Nat)
    (halloc : allocate s = some (seq, s')) :
    seq = s.next :=
  (allocate_success_shape s s' seq halloc).1

/-- XEN-V2-ALLOC-002b: success advances the allocator exactly once. -/
theorem allocate_success_advances_exactly_once
    (s s' : AllocatorState) (seq : Nat)
    (halloc : allocate s = some (seq, s')) :
    s'.next = s.next + 1 :=
  (allocate_success_shape s s' seq halloc).2.1

/-- XEN-V2-ALLOC-003: allocation refuses at or beyond the sequence limit. -/
theorem allocate_refuses_when_exhausted
    (s : AllocatorState) (hexhausted : SequenceLimit ≤ s.next) :
    allocate s = none := by
  unfold allocate
  have hnot : ¬ s.next < SequenceLimit := Nat.not_lt.mpr hexhausted
  simp [hnot]

/-- Exact boundary regression: `next = 2^64` is refusal, not wrap to zero. -/
theorem allocate_refuses_at_exact_limit :
    allocate { next := SequenceLimit } = none := by
  apply allocate_refuses_when_exhausted
  exact Nat.le_refl SequenceLimit

/-- XEN-V2-ALLOC-005: every successful allocation strictly advances state. -/
theorem allocator_state_strictly_monotone_on_success
    (s s' : AllocatorState) (seq : Nat)
    (halloc : allocate s = some (seq, s')) :
    s.next < s'.next := by
  rw [allocate_success_advances_exactly_once s s' seq halloc]
  exact Nat.lt_succ_self s.next

/-- XEN-V2-ALLOC-004a: two consecutive successful allocations differ by exactly one. -/
theorem consecutive_successors
    (s0 s1 s2 : AllocatorState) (q0 q1 : Nat)
    (h0 : allocate s0 = some (q0, s1))
    (h1 : allocate s1 = some (q1, s2)) :
    q1 = q0 + 1 := by
  have hq0 := allocate_success_returns_current s0 s1 q0 h0
  have hs1 := allocate_success_advances_exactly_once s0 s1 q0 h0
  have hq1 := allocate_success_returns_current s1 s2 q1 h1
  calc
    q1 = s1.next := hq1
    _ = s0.next + 1 := hs1
    _ = q0 + 1 := by rw [hq0]

/-- XEN-V2-ALLOC-004b: consecutive successful allocations cannot repeat a sequence. -/
theorem consecutive_successes_are_distinct
    (s0 s1 s2 : AllocatorState) (q0 q1 : Nat)
    (h0 : allocate s0 = some (q0, s1))
    (h1 : allocate s1 = some (q1, s2)) :
    q0 ≠ q1 := by
  have hstep : q1 = q0 + 1 := consecutive_successors s0 s1 s2 q0 q1 h0 h1
  have hlt : q0 < q1 := by
    rw [hstep]
    exact Nat.lt_succ_self q0
  exact ne_of_lt hlt

#print axioms allocate_success_shape
#print axioms allocate_success_in_range
#print axioms allocate_success_returns_current
#print axioms allocate_success_advances_exactly_once
#print axioms allocate_refuses_when_exhausted
#print axioms allocate_refuses_at_exact_limit
#print axioms allocator_state_strictly_monotone_on_success
#print axioms consecutive_successors
#print axioms consecutive_successes_are_distinct

end Symthaea.Formal.XeniaV2Allocator
