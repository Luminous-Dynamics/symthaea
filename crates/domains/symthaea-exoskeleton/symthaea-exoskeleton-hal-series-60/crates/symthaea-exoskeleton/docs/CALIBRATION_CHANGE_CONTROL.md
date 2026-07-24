# Calibration Change Control

Calibration is executable safety policy. Series 34 therefore treats an update as a reviewed release, not a mutable settings file.

A live-profile successor must:

- retain the same enrolled identity;
- increment the revision exactly once;
- bind the trusted previous digest and a distinct candidate digest;
- keep sensor and actuator polarity unchanged;
- keep zero, scale, and neutral changes inside bounded deltas;
- narrow or preserve joint travel, never widen it;
- reduce or preserve torque and actuator authority, never increase it;
- carry bench and fit evidence;
- carry wearer acknowledgement;
- have digest-bound approvals from a calibration engineer and an independent safety reviewer with different identities.

The returned permit is short lived and reports the most restrictive candidate actuator authority. Increasing torque, widening travel, changing polarity, changing hardware identity, or performing a large recalibration requires a new enrollment and hardware safety case rather than this update path.

Wearer mass and segment-length changes are bounded as corrections to an existing enrollment. Large anthropometric changes, including a different wearer, require a new enrollment rather than an update. Approval timestamps must fall between candidate creation and permit issuance.
