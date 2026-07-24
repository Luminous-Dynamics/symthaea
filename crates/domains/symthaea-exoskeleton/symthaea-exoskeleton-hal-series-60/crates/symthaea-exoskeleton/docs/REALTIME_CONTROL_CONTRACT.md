# Real-Time Control Contract

The body-coupled control path uses monotonic integer timestamps. System wall
clock, network time, and floating-point timestamps are not safety inputs.

## RT0 — sequence integrity

Every accepted tick increments the sequence exactly once. Duplicate, replayed,
or skipped frames are rejected before actuator command generation.

## RT1 — sample freshness

Sensor age is bounded independently of the loop period. A control loop that is
running on time with stale observations still fails closed.

## RT2 — period envelope

The reference software contract is 200 Hz with a 5 ms period and ±0.75 ms
jitter envelope. Hardware targets may tighten this contract but may not loosen
it without a new evidence profile.

## RT3 — execution deadline

The reference deadline is 4 ms. One miss is observable; two consecutive misses
exhaust the software budget and require authority withdrawal. A hardware
watchdog remains mandatory because software cannot supervise its own failure.

## RT4 — allocation discipline

`RealtimeGuard::observe` performs no heap allocation, I/O, logging, locking, or
wall-clock acquisition. Runtime adapters measure time and pass plain values into
the guard.
