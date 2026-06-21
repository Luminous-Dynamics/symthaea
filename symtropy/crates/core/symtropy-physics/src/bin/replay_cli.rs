// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replay CLI: write/read a versioned tape and print per-tick state hashes.
//!
//! This is a proof-of-concept for cross-machine determinism auditing.
//! It does **not** guarantee bitwise-equal results across different CPUs/OSes,
//! but it makes divergence measurable and debuggable.

use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

use nalgebra::SVector;
use symtropy_math::{Bivector, Point, Sphere};
use symtropy_physics::PhysicsWorld;
use symtropy_physics::body::{BodyHandle, BodyType, NetId, RigidBody};
use symtropy_physics::integrator;
use symtropy_physics::replay::WorldSnapshot;

const MAGIC: [u8; 8] = *b"SYMTAPE\0";
const VERSION: u32 = 1;

fn main() {
    if let Err(err) = try_main() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

fn try_main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        return Err(usage());
    }

    match args[0].as_str() {
        "record" => cmd_record(&args),
        "hash" => cmd_hash(&args),
        "info" => cmd_info(&args),
        _ => Err(usage()),
    }
}

fn cmd_record(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err(usage());
    }
    let path = &args[1];

    let mut dims: u32 = 3;
    let mut ticks: u32 = 192;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--dims" => {
                let v = args.get(i + 1).ok_or("--dims requires a value")?;
                dims = v
                    .parse::<u32>()
                    .map_err(|_| format!("--dims expects 2/3/4, got {v:?}"))?;
                if !matches!(dims, 2..=4) {
                    return Err(format!("--dims must be 2, 3, or 4 (got {dims})"));
                }
                i += 2;
            }
            "--ticks" => {
                let v = args.get(i + 1).ok_or("--ticks requires a value")?;
                ticks = v
                    .parse::<u32>()
                    .map_err(|_| format!("--ticks expects a u32, got {v:?}"))?;
                i += 2;
            }
            other => return Err(format!("unexpected arg {other:?}\n\n{}", usage())),
        }
    }

    match dims {
        2 => {
            let tape = demo_tape::<2>(ticks);
            write_tape_file(path, &tape).map_err(|e| format!("write failed: {e}"))?;
        }
        3 => {
            let tape = demo_tape::<3>(ticks);
            write_tape_file(path, &tape).map_err(|e| format!("write failed: {e}"))?;
        }
        4 => {
            let tape = demo_tape::<4>(ticks);
            write_tape_file(path, &tape).map_err(|e| format!("write failed: {e}"))?;
        }
        _ => return Err(format!("unsupported dims {dims} (only 2/3/4 supported)")),
    }
    println!("wrote {path} (version {VERSION}, dims {dims}, ticks {ticks})");
    Ok(())
}

fn cmd_info(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err(usage());
    }
    let path = &args[1];
    let header = read_header(path).map_err(|e| format!("read header failed: {e}"))?;
    match header.dims {
        2 => {
            let tape = read_tape_file::<2>(path).map_err(|e| format!("read failed: {e}"))?;
            println!(
                "version={} dims=2 bodies={} frames={}",
                header.version,
                tape.world.bodies.len(),
                tape.frames.len()
            );
        }
        3 => {
            let tape = read_tape_file::<3>(path).map_err(|e| format!("read failed: {e}"))?;
            println!(
                "version={} dims=3 bodies={} frames={}",
                header.version,
                tape.world.bodies.len(),
                tape.frames.len()
            );
        }
        4 => {
            let tape = read_tape_file::<4>(path).map_err(|e| format!("read failed: {e}"))?;
            println!(
                "version={} dims=4 bodies={} frames={}",
                header.version,
                tape.world.bodies.len(),
                tape.frames.len()
            );
        }
        d => return Err(format!("unsupported dims {d} (only 2/3/4 supported)")),
    }
    Ok(())
}

fn cmd_hash(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err(usage());
    }
    let path = &args[1];

    let mut every: u32 = 1;
    let mut max_ticks: Option<u32> = None;
    let mut summary_only = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--every" => {
                let v = args.get(i + 1).ok_or("--every requires a value")?;
                every = v
                    .parse::<u32>()
                    .map_err(|_| format!("--every expects a u32, got {v:?}"))?
                    .max(1);
                i += 2;
            }
            "--ticks" => {
                let v = args.get(i + 1).ok_or("--ticks requires a value")?;
                max_ticks = Some(
                    v.parse::<u32>()
                        .map_err(|_| format!("--ticks expects a u32, got {v:?}"))?,
                );
                i += 2;
            }
            "--summary" => {
                summary_only = true;
                i += 1;
            }
            other => return Err(format!("unexpected arg {other:?}\n\n{}", usage())),
        }
    }

    let header = read_header(path).map_err(|e| format!("read header failed: {e}"))?;
    if header.version != VERSION {
        return Err(format!(
            "unsupported tape version {} (expected {VERSION})",
            header.version
        ));
    }

    match header.dims {
        2 => hash_tape::<2>(path, every, max_ticks, summary_only),
        3 => hash_tape::<3>(path, every, max_ticks, summary_only),
        4 => hash_tape::<4>(path, every, max_ticks, summary_only),
        d => Err(format!("unsupported dims {d} (only 2/3/4 supported)")),
    }
}

fn usage() -> String {
    [
        "symtropy-physics replay CLI",
        "",
        "Commands:",
        "  replay_cli record <file.symtape> [--dims 2|3|4] [--ticks N]",
        "  replay_cli hash   <file.symtape> [--every N] [--ticks N] [--summary]",
        "  replay_cli info   <file.symtape>",
        "",
        "Examples:",
        "  cargo run --manifest-path crates/symtropy-physics/Cargo.toml --bin replay_cli -- record /tmp/demo.symtape",
        "  cargo run --manifest-path crates/symtropy-physics/Cargo.toml --bin replay_cli -- hash /tmp/demo.symtape --every 1",
        "",
    ]
    .join("\n")
}

#[derive(Clone, Debug)]
struct Header {
    version: u32,
    dims: u32,
}

fn read_header(path: &str) -> io::Result<Header> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let version = read_u32_le(&mut r)?;
    let dims = read_u32_le(&mut r)?;
    Ok(Header { version, dims })
}

#[derive(Clone, Debug)]
struct TapeFile<const D: usize> {
    world: WorldDef<D>,
    frames: Vec<TapeFrame<D>>,
}

#[derive(Clone, Debug)]
struct WorldDef<const D: usize> {
    gravity_bits: [u64; D],
    solver_iterations: u32,
    sleep_threshold_bits: u64,
    sleep_ticks: u32,
    slop_bits: u64,
    baumgarte_bits: u64,
    bodies: Vec<BodyDef<D>>,
}

#[derive(Clone, Debug)]
struct BodyDef<const D: usize> {
    net_id: u64,
    body_type: BodyTypeDef,
    radius_bits: u64,
    mass_bits: u64,
    restitution_bits: u64,
    friction_bits: u64,
    linear_damping_bits: u64,
    angular_damping_bits: u64,
    translation_bits: [u64; D],
    linear_velocity_bits: [u64; D],
    angular_velocity_bits: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BodyTypeDef {
    Dynamic = 0,
    Static = 1,
    Kinematic = 2,
}

impl BodyTypeDef {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Dynamic),
            1 => Some(Self::Static),
            2 => Some(Self::Kinematic),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct TapeFrame<const D: usize> {
    dt_bits: u64,
    commands: Vec<TapeCommand<D>>,
}

#[derive(Clone, Debug)]
enum TapeCommand<const D: usize> {
    ApplyForce {
        net_id: u64,
        force_bits: [u64; D],
    },
    ApplyImpulse {
        net_id: u64,
        impulse_bits: [u64; D],
    },
    SetLinearVelocity {
        net_id: u64,
        velocity_bits: [u64; D],
    },
    SetAngularVelocity {
        net_id: u64,
        velocity_bits: Vec<u64>,
    },
    Wake {
        net_id: u64,
    },
}

fn demo_tape<const D: usize>(ticks: u32) -> TapeFile<D> {
    // Stable network IDs (sorted):
    // 1 = ground, 2 = a, 3 = b, 4 = c
    let gravity = std::array::from_fn(|i| {
        if i == 1 {
            (-10.0f64).to_bits()
        } else {
            0.0f64.to_bits()
        }
    });

    let bodies = vec![
        BodyDef {
            net_id: 1,
            body_type: BodyTypeDef::Static,
            radius_bits: 64.0f64.to_bits(),
            mass_bits: f64::INFINITY.to_bits(),
            restitution_bits: 0.5f64.to_bits(),
            friction_bits: 0.5f64.to_bits(),
            linear_damping_bits: 0.0f64.to_bits(),
            angular_damping_bits: 0.0f64.to_bits(),
            translation_bits: point_bits::<D>(&Point::new(axis_array::<D>([0.0, -64.0, 0.0]))),
            linear_velocity_bits: vec_bits::<D>(&SVector::zeros()),
            angular_velocity_bits: bivector_bits::<D>(&Bivector::<D>::zero()),
        },
        BodyDef {
            net_id: 2,
            body_type: BodyTypeDef::Dynamic,
            radius_bits: 0.5f64.to_bits(),
            mass_bits: 1.0f64.to_bits(),
            restitution_bits: 0.5f64.to_bits(),
            friction_bits: 0.3f64.to_bits(),
            linear_damping_bits: 0.01f64.to_bits(),
            angular_damping_bits: 0.05f64.to_bits(),
            translation_bits: point_bits::<D>(&Point::new(axis_array::<D>([-1.5, 6.0, 0.0]))),
            linear_velocity_bits: vec_bits::<D>(&axis_vec::<D>([2.0, 0.0, 0.0])),
            angular_velocity_bits: bivector_bits::<D>(&Bivector::<D>::zero()),
        },
        BodyDef {
            net_id: 3,
            body_type: BodyTypeDef::Dynamic,
            radius_bits: 0.5f64.to_bits(),
            mass_bits: 1.0f64.to_bits(),
            restitution_bits: 0.5f64.to_bits(),
            friction_bits: 0.3f64.to_bits(),
            linear_damping_bits: 0.01f64.to_bits(),
            angular_damping_bits: 0.05f64.to_bits(),
            translation_bits: point_bits::<D>(&Point::new(axis_array::<D>([1.5, 6.0, 0.0]))),
            linear_velocity_bits: vec_bits::<D>(&axis_vec::<D>([-2.0, 0.0, 0.0])),
            angular_velocity_bits: bivector_bits::<D>(&Bivector::<D>::zero()),
        },
        BodyDef {
            net_id: 4,
            body_type: BodyTypeDef::Dynamic,
            radius_bits: 0.5f64.to_bits(),
            mass_bits: 2.0f64.to_bits(),
            restitution_bits: 0.5f64.to_bits(),
            friction_bits: 0.3f64.to_bits(),
            linear_damping_bits: 0.01f64.to_bits(),
            angular_damping_bits: 0.05f64.to_bits(),
            translation_bits: point_bits::<D>(&Point::new(axis_array::<D>([0.0, 9.0, 0.0]))),
            linear_velocity_bits: vec_bits::<D>(&SVector::zeros()),
            angular_velocity_bits: bivector_bits::<D>(&Bivector::<D>::zero()),
        },
    ];

    let world = WorldDef {
        gravity_bits: gravity,
        solver_iterations: 6,
        sleep_threshold_bits: 0.25f64.to_bits(),
        sleep_ticks: 32,
        slop_bits: 0.01f64.to_bits(),
        baumgarte_bits: 0.2f64.to_bits(),
        bodies,
    };

    let dt_bits = (1.0f64 / 64.0).to_bits();
    let mut frames = Vec::with_capacity(ticks as usize);
    for tick in 0..ticks {
        let fx = ((tick as i32 % 32) - 16) as f64 / 128.0;
        let fz = ((tick as i32 % 16) - 8) as f64 / 256.0;

        let mut commands = vec![
            TapeCommand::ApplyForce {
                net_id: 2,
                force_bits: vec_bits::<D>(&axis_vec::<D>([fx, 0.0, fz])),
            },
            TapeCommand::ApplyForce {
                net_id: 3,
                force_bits: vec_bits::<D>(&axis_vec::<D>([-fx, 0.0, -fz])),
            },
        ];
        if tick % 24 == 0 {
            commands.push(TapeCommand::ApplyImpulse {
                net_id: 4,
                impulse_bits: vec_bits::<D>(&axis_vec::<D>([0.25, 0.0, 0.0])),
            });
        }

        frames.push(TapeFrame { dt_bits, commands });
    }

    TapeFile { world, frames }
}

fn write_tape_file<const D: usize>(path: &str, tape: &TapeFile<D>) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&MAGIC)?;
    write_u32_le(&mut w, VERSION)?;
    write_u32_le(&mut w, D as u32)?;

    // World scalar fields
    for b in tape.world.gravity_bits {
        write_u64_le(&mut w, b)?;
    }
    write_u32_le(&mut w, tape.world.solver_iterations)?;
    write_u64_le(&mut w, tape.world.sleep_threshold_bits)?;
    write_u32_le(&mut w, tape.world.sleep_ticks)?;
    write_u64_le(&mut w, tape.world.slop_bits)?;
    write_u64_le(&mut w, tape.world.baumgarte_bits)?;

    // Bodies
    write_u32_le(&mut w, tape.world.bodies.len() as u32)?;
    for body in &tape.world.bodies {
        let bivector_components = D * (D - 1) / 2;
        if body.angular_velocity_bits.len() != bivector_components {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "body net_id={} has angular_velocity_bits len {}, expected {bivector_components}",
                    body.net_id,
                    body.angular_velocity_bits.len()
                ),
            ));
        }

        write_u64_le(&mut w, body.net_id)?;
        write_u8(&mut w, body.body_type as u8)?;
        write_u64_le(&mut w, body.radius_bits)?;
        write_u64_le(&mut w, body.mass_bits)?;
        write_u64_le(&mut w, body.restitution_bits)?;
        write_u64_le(&mut w, body.friction_bits)?;
        write_u64_le(&mut w, body.linear_damping_bits)?;
        write_u64_le(&mut w, body.angular_damping_bits)?;
        write_array_u64_le(&mut w, &body.translation_bits)?;
        write_array_u64_le(&mut w, &body.linear_velocity_bits)?;
        for b in &body.angular_velocity_bits {
            write_u64_le(&mut w, *b)?;
        }
    }

    // Frames
    write_u32_le(&mut w, tape.frames.len() as u32)?;
    for frame in &tape.frames {
        write_u64_le(&mut w, frame.dt_bits)?;
        write_u32_le(&mut w, frame.commands.len() as u32)?;
        for cmd in &frame.commands {
            write_command(&mut w, cmd)?;
        }
    }

    w.flush()
}

fn read_tape_file<const D: usize>(path: &str) -> io::Result<TapeFile<D>> {
    let mut r = BufReader::new(File::open(path)?);

    // Header
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let version = read_u32_le(&mut r)?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported version {version} (expected {VERSION})"),
        ));
    }
    let dims = read_u32_le(&mut r)?;
    if dims != D as u32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("dims mismatch: file has {dims}, expected {D}"),
        ));
    }

    // World scalar fields
    let gravity_bits = read_array_u64_le::<D>(&mut r)?;
    let solver_iterations = read_u32_le(&mut r)?;
    let sleep_threshold_bits = read_u64_le(&mut r)?;
    let sleep_ticks = read_u32_le(&mut r)?;
    let slop_bits = read_u64_le(&mut r)?;
    let baumgarte_bits = read_u64_le(&mut r)?;

    // Bodies
    let body_count = read_u32_le(&mut r)? as usize;
    let mut bodies = Vec::with_capacity(body_count);
    for _ in 0..body_count {
        let net_id = read_u64_le(&mut r)?;
        let body_type_u8 = read_u8(&mut r)?;
        let Some(body_type) = BodyTypeDef::from_u8(body_type_u8) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown body type {body_type_u8}"),
            ));
        };
        let radius_bits = read_u64_le(&mut r)?;
        let mass_bits = read_u64_le(&mut r)?;
        let restitution_bits = read_u64_le(&mut r)?;
        let friction_bits = read_u64_le(&mut r)?;
        let linear_damping_bits = read_u64_le(&mut r)?;
        let angular_damping_bits = read_u64_le(&mut r)?;
        let translation_bits = read_array_u64_le::<D>(&mut r)?;
        let linear_velocity_bits = read_array_u64_le::<D>(&mut r)?;
        let bivector_components = D * (D - 1) / 2;
        let mut angular_velocity_bits = Vec::with_capacity(bivector_components);
        for _ in 0..bivector_components {
            angular_velocity_bits.push(read_u64_le(&mut r)?);
        }

        bodies.push(BodyDef {
            net_id,
            body_type,
            radius_bits,
            mass_bits,
            restitution_bits,
            friction_bits,
            linear_damping_bits,
            angular_damping_bits,
            translation_bits,
            linear_velocity_bits,
            angular_velocity_bits,
        });
    }

    // Frames
    let frame_count = read_u32_le(&mut r)? as usize;
    let mut frames = Vec::with_capacity(frame_count);
    for _ in 0..frame_count {
        let dt_bits = read_u64_le(&mut r)?;
        let cmd_count = read_u32_le(&mut r)? as usize;
        let mut commands = Vec::with_capacity(cmd_count);
        for _ in 0..cmd_count {
            commands.push(read_command::<D>(&mut r)?);
        }
        frames.push(TapeFrame { dt_bits, commands });
    }

    Ok(TapeFile {
        world: WorldDef {
            gravity_bits,
            solver_iterations,
            sleep_threshold_bits,
            sleep_ticks,
            slop_bits,
            baumgarte_bits,
            bodies,
        },
        frames,
    })
}

fn hash_tape<const D: usize>(
    path: &str,
    every: u32,
    max_ticks: Option<u32>,
    summary_only: bool,
) -> Result<(), String> {
    let tape = read_tape_file::<D>(path).map_err(|e| format!("read failed: {e}"))?;
    let mut world = build_world_from_tape(&tape.world)?;

    let mut final_hasher = Fnv1a64::new();
    let ticks = max_ticks
        .map(|n| n.min(tape.frames.len() as u32))
        .unwrap_or(tape.frames.len() as u32) as usize;

    for tick in 0..ticks {
        let frame = &tape.frames[tick];
        apply_frame(&mut world, frame)?;
        world.step(f64::from_bits(frame.dt_bits));

        let snapshot = WorldSnapshot::capture(&world);
        let hash = snapshot_hash::<D>(&snapshot);

        final_hasher.update_u64(hash);
        if !summary_only && (tick as u32).is_multiple_of(every) {
            println!("tick {tick:06} {hash:016x}");
        }
    }

    println!("final {:016x}", final_hasher.finish());
    Ok(())
}

fn build_world_from_tape<const D: usize>(def: &WorldDef<D>) -> Result<PhysicsWorld<D>, String> {
    let gravity =
        SVector::<f64, D>::from(std::array::from_fn(|i| f64::from_bits(def.gravity_bits[i])));
    let mut world = PhysicsWorld::<D>::new(gravity);
    world.solver_iterations = def.solver_iterations as usize;
    world.sleep_threshold = f64::from_bits(def.sleep_threshold_bits);
    world.sleep_ticks = def.sleep_ticks;
    world.slop = f64::from_bits(def.slop_bits);
    world.baumgarte = f64::from_bits(def.baumgarte_bits);

    let mut bodies: Vec<(NetId, RigidBody<D>)> = Vec::with_capacity(def.bodies.len());
    for b in def.bodies.clone() {
        let net_id = NetId(b.net_id);
        let radius = f64::from_bits(b.radius_bits);
        let mass = f64::from_bits(b.mass_bits);
        let restitution = f64::from_bits(b.restitution_bits);
        let friction = f64::from_bits(b.friction_bits);
        let linear_damping = f64::from_bits(b.linear_damping_bits);
        let angular_damping = f64::from_bits(b.angular_damping_bits);

        let pos = Point::new(std::array::from_fn(|i| {
            f64::from_bits(b.translation_bits[i])
        }));
        let lin_vel = SVector::<f64, D>::from(std::array::from_fn(|i| {
            f64::from_bits(b.linear_velocity_bits[i])
        }));
        let ang_vel = bivector_from_bits::<D>(&b.angular_velocity_bits);

        let mut body = match b.body_type {
            BodyTypeDef::Static => RigidBody::<D>::static_body(
                BodyHandle(0),
                pos,
                Box::new(Sphere::<D>::new(Point::origin(), radius)),
            ),
            BodyTypeDef::Dynamic | BodyTypeDef::Kinematic => {
                let mut body = RigidBody::<D>::dynamic_sphere(BodyHandle(0), pos, radius, mass);
                if b.body_type == BodyTypeDef::Kinematic {
                    body.body_type = BodyType::Kinematic;
                    // Treat kinematic as infinite mass for impulse computation.
                    body.mass = f64::INFINITY;
                    body.inv_mass = 0.0;
                    body.inertia = SVector::from_element(f64::INFINITY);
                    body.inv_inertia = SVector::zeros();
                }
                body
            }
        };

        body.restitution = restitution;
        body.friction = friction;
        body.linear_damping = linear_damping;
        body.angular_damping = angular_damping;
        body.linear_velocity = lin_vel;
        body.angular_velocity = ang_vel;

        bodies.push((net_id, body));
    }

    world
        .add_bodies_deterministic(bodies)
        .map_err(|e| format!("net id insertion failed: {e:?}"))?;

    Ok(world)
}

fn apply_frame<const D: usize>(
    world: &mut PhysicsWorld<D>,
    frame: &TapeFrame<D>,
) -> Result<(), String> {
    for cmd in &frame.commands {
        apply_command(world, cmd)?;
    }
    Ok(())
}

fn apply_command<const D: usize>(
    world: &mut PhysicsWorld<D>,
    cmd: &TapeCommand<D>,
) -> Result<(), String> {
    let resolve = |net_id: u64| {
        world
            .handle_for_net_id(NetId(net_id))
            .ok_or_else(|| format!("command references unknown net_id {net_id}"))
    };

    match cmd {
        TapeCommand::ApplyForce { net_id, force_bits } => {
            let handle = resolve(*net_id)?;
            let force =
                SVector::<f64, D>::from(std::array::from_fn(|i| f64::from_bits(force_bits[i])));
            let Some(body) = world.body_mut(handle) else {
                return Err(format!("missing body for net_id {net_id}"));
            };
            body.apply_force(force);
        }
        TapeCommand::ApplyImpulse {
            net_id,
            impulse_bits,
        } => {
            let handle = resolve(*net_id)?;
            let impulse =
                SVector::<f64, D>::from(std::array::from_fn(|i| f64::from_bits(impulse_bits[i])));
            let Some(body) = world.body_mut(handle) else {
                return Err(format!("missing body for net_id {net_id}"));
            };
            integrator::apply_impulse(body, &impulse);
        }
        TapeCommand::SetLinearVelocity {
            net_id,
            velocity_bits,
        } => {
            let handle = resolve(*net_id)?;
            let vel =
                SVector::<f64, D>::from(std::array::from_fn(|i| f64::from_bits(velocity_bits[i])));
            let Some(body) = world.body_mut(handle) else {
                return Err(format!("missing body for net_id {net_id}"));
            };
            body.linear_velocity = vel;
        }
        TapeCommand::SetAngularVelocity {
            net_id,
            velocity_bits,
        } => {
            let handle = resolve(*net_id)?;
            let bv = bivector_from_bits::<D>(velocity_bits);
            let Some(body) = world.body_mut(handle) else {
                return Err(format!("missing body for net_id {net_id}"));
            };
            body.angular_velocity = bv;
        }
        TapeCommand::Wake { net_id } => {
            let handle = resolve(*net_id)?;
            let Some(body) = world.body_mut(handle) else {
                return Err(format!("missing body for net_id {net_id}"));
            };
            body.wake();
        }
    }
    Ok(())
}

fn write_command<const D: usize>(w: &mut impl Write, cmd: &TapeCommand<D>) -> io::Result<()> {
    match cmd {
        TapeCommand::ApplyForce { net_id, force_bits } => {
            write_u8(w, 0)?;
            write_u64_le(w, *net_id)?;
            write_array_u64_le(w, force_bits)?;
        }
        TapeCommand::ApplyImpulse {
            net_id,
            impulse_bits,
        } => {
            write_u8(w, 1)?;
            write_u64_le(w, *net_id)?;
            write_array_u64_le(w, impulse_bits)?;
        }
        TapeCommand::SetLinearVelocity {
            net_id,
            velocity_bits,
        } => {
            write_u8(w, 2)?;
            write_u64_le(w, *net_id)?;
            write_array_u64_le(w, velocity_bits)?;
        }
        TapeCommand::SetAngularVelocity {
            net_id,
            velocity_bits,
        } => {
            let bivector_components = D * (D - 1) / 2;
            if velocity_bits.len() != bivector_components {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "SetAngularVelocity for net_id={net_id} has velocity_bits len {}, expected {bivector_components}",
                        velocity_bits.len()
                    ),
                ));
            }
            write_u8(w, 3)?;
            write_u64_le(w, *net_id)?;
            for b in velocity_bits {
                write_u64_le(w, *b)?;
            }
        }
        TapeCommand::Wake { net_id } => {
            write_u8(w, 4)?;
            write_u64_le(w, *net_id)?;
        }
    }
    Ok(())
}

fn read_command<const D: usize>(r: &mut impl Read) -> io::Result<TapeCommand<D>> {
    let kind = read_u8(r)?;
    let net_id = read_u64_le(r)?;
    Ok(match kind {
        0 => TapeCommand::ApplyForce {
            net_id,
            force_bits: read_array_u64_le::<D>(r)?,
        },
        1 => TapeCommand::ApplyImpulse {
            net_id,
            impulse_bits: read_array_u64_le::<D>(r)?,
        },
        2 => TapeCommand::SetLinearVelocity {
            net_id,
            velocity_bits: read_array_u64_le::<D>(r)?,
        },
        3 => TapeCommand::SetAngularVelocity {
            net_id,
            velocity_bits: {
                let bivector_components = D * (D - 1) / 2;
                let mut bits = Vec::with_capacity(bivector_components);
                for _ in 0..bivector_components {
                    bits.push(read_u64_le(r)?);
                }
                bits
            },
        },
        4 => TapeCommand::Wake { net_id },
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown command kind {kind}"),
            ));
        }
    })
}

fn axis_array<const D: usize>(xyz: [f64; 3]) -> [f64; D] {
    std::array::from_fn(|i| if i < 3 { xyz[i] } else { 0.0 })
}

fn axis_vec<const D: usize>(xyz: [f64; 3]) -> SVector<f64, D> {
    SVector::from(axis_array::<D>(xyz))
}

fn point_bits<const D: usize>(p: &Point<D>) -> [u64; D] {
    std::array::from_fn(|i| p.0[i].to_bits())
}

fn vec_bits<const D: usize>(v: &SVector<f64, D>) -> [u64; D] {
    std::array::from_fn(|i| v[i].to_bits())
}

fn bivector_bits<const D: usize>(b: &Bivector<D>) -> Vec<u64> {
    let bivector_components = D * (D - 1) / 2;
    let mut out = Vec::with_capacity(bivector_components);
    for i in 0..D {
        for j in (i + 1)..D {
            out.push(b.get(i, j).to_bits());
        }
    }
    out
}

fn bivector_from_bits<const D: usize>(bits: &[u64]) -> Bivector<D> {
    let mut bv = Bivector::<D>::zero();
    let expected = D * (D - 1) / 2;
    if bits.len() != expected {
        return bv;
    }
    let mut idx = 0;
    for i in 0..D {
        for j in (i + 1)..D {
            bv.set(i, j, f64::from_bits(bits[idx]));
            idx += 1;
        }
    }
    bv
}

struct Fnv1a64 {
    state: u64,
}

impl Fnv1a64 {
    fn new() -> Self {
        Self {
            state: 14695981039346656037,
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.state ^= *b as u64;
            self.state = self.state.wrapping_mul(1099511628211);
        }
    }

    fn update_u8(&mut self, v: u8) {
        self.update(&[v]);
    }

    fn update_u32(&mut self, v: u32) {
        self.update(&v.to_le_bytes());
    }

    fn update_u64(&mut self, v: u64) {
        self.update(&v.to_le_bytes());
    }

    fn finish(self) -> u64 {
        self.state
    }
}

fn snapshot_hash<const D: usize>(snapshot: &WorldSnapshot<D>) -> u64 {
    let mut h = Fnv1a64::new();

    h.update_u32(snapshot.bodies.len() as u32);
    for b in &snapshot.bodies {
        h.update_u64(b.handle.0 as u64);
        h.update_u8(match b.body_type {
            BodyType::Dynamic => 0,
            BodyType::Static => 1,
            BodyType::Kinematic => 2,
        });
        for v in b.translation {
            h.update_u64(v);
        }
        for row in b.rotation {
            for v in row {
                h.update_u64(v);
            }
        }
        for v in b.linear_velocity {
            h.update_u64(v);
        }
        for row in b.angular_velocity {
            for v in row {
                h.update_u64(v);
            }
        }
        h.update_u8(if b.sleeping { 1 } else { 0 });
        h.update_u32(b.sleep_counter);
    }

    h.update_u32(snapshot.collision_events.len() as u32);
    for e in &snapshot.collision_events {
        h.update_u64(e.body_a.0 as u64);
        h.update_u64(e.body_b.0 as u64);
        h.update_u64(e.impulse);
        for v in e.normal {
            h.update_u64(v);
        }
        h.update_u64(e.depth);
    }

    h.finish()
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn write_u8(w: &mut impl Write, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

fn read_u32_le(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn write_u32_le(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u64_le(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_u64_le(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_array_u64_le<const N: usize>(r: &mut impl Read) -> io::Result<[u64; N]> {
    let mut out = [0u64; N];
    for v in &mut out {
        *v = read_u64_le(r)?;
    }
    Ok(out)
}

fn write_array_u64_le<const N: usize>(w: &mut impl Write, v: &[u64; N]) -> io::Result<()> {
    for x in v {
        write_u64_le(w, *x)?;
    }
    Ok(())
}
