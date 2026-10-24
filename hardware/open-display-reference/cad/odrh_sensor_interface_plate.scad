/*
 * ODRH-001 optical sensor interface plate — geometry prototype v0.1
 * Copyright (C) 2026 Luminous Dynamics
 * SPDX-License-Identifier: CERN-OHL-S-2.0
 *
 * Claim ceiling: parameterized interface concept only. No sensor, screw fit,
 * print process, strength, flatness, optical performance, or metrology is
 * qualified. Measure the actual sensor PCB and bracket before fabrication.
 */

// Screen is assumed to lie in the X-Z plane. Sensor looks toward negative Y.
// The plate is shown in its installed orientation: thickness along Y.

plate_width_mm = 80;
plate_height_mm = 80;
plate_thickness_mm = 5;

sensor_aperture_d_mm = 20;

mount_hole_d_mm = 3.4;       // nominal M3 clearance; verify actual hardware
mount_pitch_x_mm = 25;
mount_pitch_z_mm = 25;
mount_counterbore_d_mm = 6.2;
mount_counterbore_depth_mm = 1.6;

bracket_slot_width_mm = 5.5; // nominal M5 clearance; verify actual hardware
bracket_slot_length_mm = 18;
bracket_slot_x_mm = 28.5;
edge_margin_mm = 6;

render_datum_axes = false;
render_sensor_outline = false;
sensor_outline_width_mm = 42;
sensor_outline_height_mm = 42;
sensor_outline_depth_mm = 1;

// Export only the plate. Optional outlines are visualization aids, not geometry.
$fn = 64;

assert(plate_width_mm > 2 * edge_margin_mm, "Plate width is too small");
assert(plate_height_mm > 2 * edge_margin_mm, "Plate height is too small");
assert(plate_thickness_mm > 0, "Plate thickness must be positive");
assert(sensor_aperture_d_mm > 0, "Aperture must be positive");
assert(sensor_aperture_d_mm + mount_hole_d_mm <
       sqrt(mount_pitch_x_mm * mount_pitch_x_mm +
            mount_pitch_z_mm * mount_pitch_z_mm),
       "Sensor aperture overlaps the diagonal mounting-hole envelope");
assert(mount_pitch_x_mm + mount_hole_d_mm + 2 * edge_margin_mm <= plate_width_mm,
       "Horizontal mounting-hole grid violates edge margin");
assert(mount_pitch_z_mm + mount_hole_d_mm + 2 * edge_margin_mm <= plate_height_mm,
       "Vertical mounting-hole grid violates edge margin");
assert(bracket_slot_x_mm + bracket_slot_width_mm / 2 + edge_margin_mm <=
       plate_width_mm / 2, "Bracket slot violates horizontal edge margin");
assert(bracket_slot_length_mm <= plate_height_mm - 2 * edge_margin_mm,
       "Bracket slot violates vertical edge margin");
assert(mount_counterbore_depth_mm >= 0 &&
       mount_counterbore_depth_mm < plate_thickness_mm,
       "Counterbore depth must be smaller than plate thickness");
assert(mount_counterbore_d_mm >= mount_hole_d_mm,
       "Counterbore must not be smaller than through-hole");

module y_axis_bore(x_mm, z_mm, diameter_mm, depth_mm = plate_thickness_mm + 2) {
    translate([x_mm, 0, z_mm])
        rotate([90, 0, 0])
            cylinder(d = diameter_mm, h = depth_mm, center = true);
}

module vertical_slot(x_mm, z_mm, width_mm, overall_length_mm) {
    offset_mm = (overall_length_mm - width_mm) / 2;
    hull() {
        y_axis_bore(x_mm, z_mm - offset_mm, width_mm);
        y_axis_bore(x_mm, z_mm + offset_mm, width_mm);
    }
}

module sensor_interface_plate() {
    difference() {
        cube([plate_width_mm, plate_thickness_mm, plate_height_mm], center = true);

        // Optical axis reference aperture. This is not a lens or baffle.
        y_axis_bore(0, 0, sensor_aperture_d_mm);

        // Four configurable through-holes. Counterbores are on the +Y face.
        for (sx = [-1, 1], sz = [-1, 1]) {
            x_mm = sx * mount_pitch_x_mm / 2;
            z_mm = sz * mount_pitch_z_mm / 2;
            y_axis_bore(x_mm, z_mm, mount_hole_d_mm);
            translate([x_mm, plate_thickness_mm / 2 -
                       mount_counterbore_depth_mm / 2, z_mm])
                rotate([90, 0, 0])
                    cylinder(d = mount_counterbore_d_mm,
                             h = mount_counterbore_depth_mm + 0.02,
                             center = true);
        }

        // Two vertical adjustment slots for a separate bracket.
        for (sx = [-1, 1]) {
            vertical_slot(sx * bracket_slot_x_mm, 0,
                          bracket_slot_width_mm, bracket_slot_length_mm);
        }
    }
}

if (render_sensor_outline) {
    // Visualization only: outline of a hypothetical sensor PCB behind the plate.
    // It is not exported as part of the physical plate.
    %translate([0, plate_thickness_mm / 2 + sensor_outline_depth_mm / 2, 0])
        cube([sensor_outline_width_mm, sensor_outline_depth_mm,
              sensor_outline_height_mm], center = true);
}

sensor_interface_plate();

if (render_datum_axes) {
    %color("red") cube([plate_width_mm * 0.7, 0.5, 0.5], center = true);
    %color("green") cube([0.5, 0.5, plate_height_mm * 0.7], center = true);
    %color("blue")
        translate([0, plate_thickness_mm * 1.5, 0])
            cube([0.5, plate_thickness_mm * 2, 0.5], center = true);
}
