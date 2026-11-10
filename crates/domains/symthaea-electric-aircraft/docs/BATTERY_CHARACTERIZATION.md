# Battery characterization contract

**Model increment:** electric-fixed-wing-energy-v0.3.0  
**Evidence class:** model-interface specification. Unit-test surfaces are synthetic only.

## Purpose

The scalar pack open-circuit voltage and DC resistance used in v0.1 are not representative of a full operating envelope. v0.3 adds optional pack-level maps for open-circuit voltage and internal resistance over state of charge (SOC) and pack temperature.

The two surfaces must share axes:
- x = state_of_charge, unit 1;
- y = temperature, unit K.

Values are stored on a rectilinear grid with strictly increasing axes. Bilinear interpolation is used only inside the measured grid domain. Extrapolation is rejected.

## Conservative mission use

At the start of each mission segment, the model samples the battery maps at the segment's initial SOC and current lumped pack temperature.

- OCV used by the mission solver = interpolated nominal OCV minus its absolute uncertainty bound.
- DC resistance used by the mission solver = interpolated nominal resistance plus its absolute uncertainty bound.

The resulting OCV and resistance drive the equivalent-circuit current solution, terminal voltage, chemical-energy estimate, I²R heat and next segment's thermal/SOC state.

The report records the conservative OCV and resistance actually used for each segment and lists the evidence dataset IDs in a separate battery-characterization field.

The uncertainty values are absolute bounds supplied by the data provider. They are not standard deviations or statistical confidence intervals.

## Required real-world data

For every pack configuration and intended operating envelope, collect:

- pack OCV versus SOC and temperature after a documented rest/relaxation protocol;
- pack DC resistance versus SOC and temperature, including the exact pulse/measurement duration and current direction;
- pack serial/configuration, cell chemistry, series/parallel arrangement, BMS firmware, contactors, fuses, bus bars, cooling plate and enclosure revision;
- sensor calibration certificates, measurement equipment, test procedure, ambient/pack temperature, SOC estimator version, charge/discharge history, SOH and test date;
- repeatability, hysteresis, cell/parallel-string imbalance, instrument uncertainty and data exclusions.

Resistance measured with different pulse durations or different instrument boundaries is not automatically comparable. DC resistance is method-dependent; record the measurement protocol. OCV also depends on rest time, hysteresis and prior charge/discharge history.

## Boundaries and omissions

This is a pack-level reduced-order equivalent circuit, not a cell electrochemistry model or a battery safety case.

Current limitations:
- no cell-level voltage, temperature, impedance or imbalance model;
- no charge/discharge-direction hysteresis state;
- no rate-dependent capacity or resistance map;
- no explicit SOH-dependent OCV/resistance surface; SOH currently scales nominal capacity only;
- no transient RC polarization branches or dynamic voltage recovery;
- no busbar/contact resistance decomposition;
- no fuse, contactor, isolation monitor, short-circuit, arc, venting, propagation or thermal-runaway model;
- the thermal model remains a single lumped pack node;
- no pack-to-aircraft installation, crashworthiness or fire-containment qualification.

Do not use this model to size cell-level protection, establish safe operating limits, justify dispatch or establish airworthiness. Physical pack testing and an independent safety analysis are required.

## Evidence handling

The grid requires a dataset ID, URI, evidence class, description and 64-character SHA-256 digest string. Current code validates digest syntax only. The surrounding evidence pipeline must resolve the URI, hash the exact immutable artifact bytes, verify signatures/authority where applicable, and bind the dataset to the exact battery configuration before it is treated as verified evidence.

Synthetic test surfaces are explicitly labelled Simulation and do not represent physical measurements.
