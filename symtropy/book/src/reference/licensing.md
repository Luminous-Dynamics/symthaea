# Licensing

Symtropy uses a **dual-track license model** — see `../../LICENSING.md` at the repo root for the authoritative, crate-by-crate breakdown.

## Summary

| Track | Crates | License |
|---|---|---|
| **Core** (generalist adoption) | `symtropy-math`, `symtropy-physics`, `symtropy-render-bridge`, `symtropy-robotics-bridge`, `symtropy-net`, `symtropy-bevy` | **Apache-2.0 OR MIT** |
| **Research** (copyleft) | `symtropy-consciousness-physics`, `symtropy-sim-bridge`, `symtropy-world`, `symtropy-holochain-relay`, `symtropy-lightyear`, `symthaea-bevy-brain` | **AGPL-3.0-or-later** |
| **Game / demo** | Root crate + demo crates | **AGPL-3.0-or-later** |

## Fast Q&A

**Q: Can I ship a closed-source commercial game using Symtropy's physics?**  
Yes — use only the core crates (`symtropy-math`, `symtropy-physics`, `symtropy-bevy`, etc.). They're Apache-2.0 OR MIT.

**Q: What if I want the Φ-coupling?**  
Then you're in AGPL territory. Either open-source your modifications under AGPL, or contact `tristan.stoltz@evolvingresonantcocreationism.com` for a commercial licence. Cooperatives and B-corps may qualify for favourable terms.

**Q: The `0.1.0` crate on crates.io says AGPL. What happened?**  
Initial publications were AGPL-wide. The license split landed after that. Published `0.1.0` versions are immutable; the next release of the now-permissive crates will be `0.2.0+` with the new license. Depend on `>= 0.2.0` if you need permissive.

**Q: Can I implement a custom `PhysicsCallback` in a proprietary project?**  
Yes. The `PhysicsCallback` trait lives in `symtropy-physics` (permissive). You can implement it without any AGPL obligations.

## References

- `../../LICENSING.md` — Crate-by-crate breakdown
- `../../LICENSE-APACHE` — Apache 2.0 text
- `../../LICENSE-MIT` — MIT text
- `/srv/luminous-dynamics/LICENSE` — GNU AGPL v3 text
- `/srv/luminous-dynamics/COMMERCIAL_LICENSE.md` — Commercial licensing terms
- `/srv/luminous-dynamics/CLA.md` — Contributor License Agreement
