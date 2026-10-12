//! Standalone test harness for the production DFT grid source.
//!
//! This compiles the actual grid.rs and its embedded #[cfg(test)] module while
//! supplying only the minimal Atom/Molecule interface needed by those tests.
//! It deliberately does not stand in for crate integration or molecular DFT.
#![allow(dead_code)]

pub mod molecule {
    #[derive(Debug, Clone, Copy)]
    pub struct Atom {
        pub atomic_number: u8,
        pub position: [f64; 3],
    }

    impl Atom {
        pub fn new(atomic_number: u8, x: f64, y: f64, z: f64) -> Self {
            Self { atomic_number, position: [x, y, z] }
        }

        pub fn distance_to(&self, other: &Atom) -> f64 {
            let dx = self.position[0] - other.position[0];
            let dy = self.position[1] - other.position[1];
            let dz = self.position[2] - other.position[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        }
    }

    #[derive(Debug, Clone)]
    pub struct Molecule {
        pub atoms: Vec<Atom>,
    }

    impl Molecule {
        pub fn h2() -> Self {
            Self { atoms: vec![
                Atom::new(1, 0.0, 0.0, -0.7),
                Atom::new(1, 0.0, 0.0, 0.7),
            ] }
        }

        pub fn water() -> Self {
            Self { atoms: vec![
                Atom::new(8, 0.0, 0.0, 0.0),
                Atom::new(1, 1.43, 0.0, 1.10),
                Atom::new(1, -1.43, 0.0, 1.10),
            ] }
        }
    }
}

#[path = "../../crates/domains/symthaea-quantum-chemistry/src/dft/grid.rs"]
mod grid;
