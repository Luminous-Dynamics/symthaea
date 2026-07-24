# Qualification Mission Profile

A production qualification policy must be derived from the declared use case,
wearer population, actuator hardware, expected environment, and maintenance
interval. The small thresholds used by repository evidence are smoke tests only.

A release mission profile should define, at minimum:

- total and active operating hours across representative loads and speeds;
- power-cycle, brownout, watchdog-reset, and clean-shutdown counts;
- hot, cold, humidity, vibration, shock, ingress, and EMC exposure;
- actuator reversals, gearbox cycles, cable and cuff fatigue, and energy throughput;
- every fault-injection domain with detection and safe-state deadlines;
- repeated emergency contactor isolation and DC-link discharge tests;
- redundant safety-link loss, disagreement, replay, and common-cause tests;
- calibration, software-update, rollback, and maintenance-session ceremonies;
- wearer-specific range, work, amplification, interface-load, and stop trials;
- independent review, HIL, bench, endurance, human-factors, and regulatory evidence.

No aggregate pass rate may hide a safety escape. Any invariant violation,
uncommanded positive mechanical power, failed isolation, or automatic re-arm is
a release-blocking event requiring root-cause analysis and campaign repetition.
