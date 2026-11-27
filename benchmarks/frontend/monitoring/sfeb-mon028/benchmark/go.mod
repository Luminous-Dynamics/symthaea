module luminousdynamics/mon028-benchmark

go 1.23

require (
	luminousdynamics/mon026-custody v0.0.0
	luminousdynamics/mon029-reader v0.0.0
)

replace luminousdynamics/mon026-custody => ../../sfeb-mon026/custody
replace luminousdynamics/mon029-reader => ../../sfeb-mon029/reader
