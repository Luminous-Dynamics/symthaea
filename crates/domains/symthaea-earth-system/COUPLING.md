# Earth–ecology coupling contract

`symthaea-earth-system` intentionally does not depend on `symthaea-ecology`, and
`symthaea-ecology` intentionally does not depend on this crate. A higher-level
bridge should translate climate outputs into explicit ecological drivers.

Recommended boundary record:

```text
ClimateDriverFrame
  elapsed_time_seconds
  temperature_kelvin
  productivity_index
  disturbance_index
  soil_moisture_fraction
  source_model
  parameter_set_id
  uncertainty_or_ensemble_member
  validity_interval
```

The current `EnvironmentalDrivers` type in `symthaea-ecology` is the minimal
numeric payload. Production integrations should retain the surrounding
provenance rather than discarding it during conversion.

For transient coupling, align timestamps and integration intervals explicitly.
A discontinuous climate forcing event and an ecological update should not be
silently treated as simultaneous unless the scheduler defines that ordering.
Uncertainty envelopes should be propagated as ensembles or bounded evaluations,
not collapsed to a central value before the ecological transformation.

This contract is suitable for oracle experiments: an agent simulation can be
run under a controlled driver frame and compared with the corresponding
analytic logistic or predator-prey baseline without making either crate the
owner of the other domain.

## Temperature driver export

`one_box_temperature_drivers`, `two_box_surface_temperature_drivers`, and
`latitude_temperature_drivers` emit a
strictly ordered dependency-neutral record. Global trajectories carry elapsed
SI seconds, absolute surface temperature, and anomaly. Latitudinal exports carry
band identity, latitude, equal-area weight, absolute temperature, and anomaly.
They do not infer ecological productivity or disturbance. Consumers must
convert seconds into the ecological model's time unit explicitly and must not
confuse equal-area weights with habitat fractions.


## Protocol breakpoints

Consumers replaying pulses should prefer event-aligned trajectories and retain
the generated timestamps; assuming a uniform grid after breakpoint insertion is
incorrect. Smooth sinusoidal protocols have no discrete events.

## Carbon model identity

Coupling evidence must record whether atmospheric concentration came from the
two-box exact oracle or the configurable three-reservoir exchange model. They
are distinct structural hypotheses and must not share an unlabeled parameter
record.


## New v7 coupling boundaries

Piecewise emissions stages remain an Earth-system input protocol and should be
exported with their exact time boundaries and GtC/year units. Temperature-
dependent zonal albedo is internal climate feedback evidence; ecology should
receive the resulting temperatures and anomalies, not infer ice state from a
bare albedo scalar. Sea-level anomalies may become habitat drivers only in a
higher-level adapter that declares shoreline geometry, exposure, datum, and
response-component calibration.


## Hydrology and soil-carbon boundaries

`hydrology_drivers` exports storage, relative soil moisture, actual
evapotranspiration, and runoff in explicit day-based units. The matching
`HydroEnvironmentalDrivers` adapter in `symthaea-ecology` consumes only the
soil-moisture fraction and keeps its piecewise response, floor, and exponents
visible. Runoff is not silently converted into disturbance or habitat loss.

`soil_carbon_drivers` exports stocks, respiration rate, and budget residual. A
higher-level carbon-climate bridge may use respiration only after declaring
area conversion, carbon units, atmosphere coupling, temperature/moisture
feedback ordering, and whether litter inputs are externally prescribed.

## Nutrient and productivity boundaries

`nutrient_drivers` exports organic and mineral stocks, mineralization, uptake,
leaching, and the exact cumulative budget residual in the caller's declared
nutrient units. The receiving ecology layer must not assume that the full
mineral pool is biologically accessible, spatially colocated, or measured on the
same support as a population state.

`productivity_driver` exports a finite-interval carbon ledger. It intentionally
does not convert NPP into intrinsic growth, carrying capacity, trophic support,
or population abundance. Such adapters must declare carbon-to-biomass units,
organism or guild identity, turnover, allocation, and the time interval over
which the ledger was accumulated.

The matching `NutrientLogisticEnvironmentCoupling` consumes a non-negative
mineral stock through an explicit half-saturation response. Its floor and growth
and capacity exponents remain public. It does not feed ecological uptake back
into the Earth-system nutrient state; a higher-level coupled integrator must
apply that transaction once, with an explicit ordering convention.
