# symtropy-robotics-bridge-core

Permissively-licensed (Apache-2.0 OR MIT) core types and traits for symtropy-robotics-bridge.

## Features
- `PlatformType`: Enumeration of Symthaea robotics platforms.
- `RoboticAgent`: Trait for agents that can be ticked by the game engine.
- `MotorPlanner`: Trait for per-joint motor command planning.
- `spawn_robot_body`: Helper for spawning a platform-appropriate body into the physics world.

This crate contains NO AGPL-licensed dependencies and is safe for use in proprietary applications.
