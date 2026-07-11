# symthaea-geodesy

Geodesy & navigation on a spherical Earth — great-circle distance and bearing.

Pure `std`, zero deps, no `symthaea-core` link. Checked vs known distances
(London→Paris ≈ 343 km, quarter-equator ≈ 10007 km).

- `sphere::haversine_distance` — great-circle distance (km).
- `sphere::initial_bearing` — forward azimuth (degrees).

```bash
cargo test -p symthaea-geodesy
```
