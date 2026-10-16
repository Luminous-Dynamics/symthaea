# symthaea-geodesy

Geodesy & navigation on a spherical Earth — great-circle distance and bearing.

Pure std, zero dependencies, no symthaea-core link. Distances use the
declared mean spherical Earth radius; they are not WGS84 ellipsoidal geodesics.

- sphere::haversine_distance — great-circle distance (km); raw float API retained for compatibility.
- sphere::initial_bearing — spherical initial bearing (degrees).
- GeoPoint — validated coordinate value with private fields and a checked constructor.
- GeoPointError — explicit non-finite and out-of-range input failures.
- GeoPoint::distance_to — distance between validated points (km), using the spherical model.

Example:

~~~rust
use symthaea_geodesy::{GeoPoint, GeoPointError};

fn main() -> Result<(), GeoPointError> {
    let a = GeoPoint::new(-33.8688, 151.2093)?;
    let b = GeoPoint::new(35.6762, 139.6503)?;
    println!("distance_km={:.3}", a.distance_to(b));
    Ok(())
}
~~~

Coordinate policy:
- Latitude must be finite and within [-90°, 90°].
- Longitude must be finite and within [-180°, 180°].
- Inputs are rejected, not clamped or silently normalized.
- The point type validates numeric representation only; it does not encode a datum, altitude, uncertainty, or ground-truth location.

Run:

~~~bash
cargo test -p symthaea-geodesy
~~~
